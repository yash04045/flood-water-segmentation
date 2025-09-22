import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from tqdm import tqdm, trange
from torch.amp import autocast, GradScaler

from datasetLoader import FloodDataset, get_train_transform, get_val_transform, get_sampler

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)
    except:
        pass

# ---------------- Detect classes ----------------
def detect_num_classes(mask_dirs):
    labels = set()
    for d in mask_dirs:
        if not os.path.isdir(d): continue
        for fname in os.listdir(d):
            if fname.startswith('.') or not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                continue
            path = os.path.join(d, fname)
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if m is None: continue
            labels.update(np.unique(m).tolist())
    if not labels:
        raise RuntimeError("No masks found. Check data/masks paths.")
    labels = sorted([int(x) for x in labels])
    return len(labels), labels

_mask_dirs = ["data/masks/train", "data/masks/val", "data/masks/test"]
NUM_CLASSES, DETECTED_LABELS = detect_num_classes(_mask_dirs)
print(f"Detected labels: {DETECTED_LABELS} -> NUM_CLASSES = {NUM_CLASSES}")

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
NUM_WORKERS = 2 if os.name != "nt" else 2

# ---------------- Metrics ----------------
def compute_iou(preds, labels, num_classes):
    ious = []
    preds = preds.view(-1); labels = labels.view(-1)
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        inter = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - inter
        if union == 0:
            ious.append(1.0 if inter == 0 else 0.0)
        else:
            ious.append(inter / union)
    return ious

def compute_dice(preds, labels, num_classes):
    dices = []
    preds = preds.view(-1); labels = labels.view(-1)
    for cls in range(num_classes):
        pred_inds = (preds == cls).float()
        target_inds = (labels == cls).float()
        inter = (pred_inds * target_inds).sum().item()
        denom = pred_inds.sum().item() + target_inds.sum().item()
        dices.append((2.0 * inter) / denom if denom > 0 else float("nan"))
    return dices

# ---------------- Losses ----------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction="none")

    def forward(self, logits, targets):
        logpt = -self.ce(logits, targets)
        pt = torch.exp(logpt)
        focal = ((1 - pt) ** self.gamma) * (-logpt)
        mask = (targets != self.ignore_index).float()
        return (focal * mask).sum() / mask.sum().clamp_min(1.0)

def dice_loss(pred, target, smooth=1e-6):
    pred = F.softmax(pred, dim=1)
    target_1h = F.one_hot(target, num_classes=pred.shape[1]).permute(0,3,1,2).float()
    inter = torch.sum(pred * target_1h, dim=(0,2,3))
    union = torch.sum(pred, dim=(0,2,3)) + torch.sum(target_1h, dim=(0,2,3))
    dice = (2. * inter + smooth) / (union + smooth)
    return 1 - dice.mean()

def get_focal_dice_loss(weights):
    focal = FocalLoss(gamma=2.0, weight=weights, ignore_index=-1)
    def hybrid(outputs, targets):
        return 0.6 * focal(outputs, targets) + 0.4 * dice_loss(outputs, targets)
    return hybrid

# ---------------- Utils ----------------
def compute_pixel_class_counts(dataset):
    counts = {}
    for i in range(len(dataset)):
        _, mask = dataset[i]
        labels, cnt = torch.unique(mask, return_counts=True)
        for l, c in zip(labels.tolist(), cnt.tolist()):
            counts[int(l)] = counts.get(int(l), 0) + int(c)
    return counts

# ---------------- Training ----------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, accumulation_steps=8):
    model.train(); running_loss = 0.0
    optimizer.zero_grad()
    for i, (images, masks) in enumerate(tqdm(loader, desc="Training", leave=False, position=1)):
        images, masks = images.to(DEVICE), masks.to(DEVICE).long()
        with autocast(device_type="cuda", enabled=True):
            outputs = model(images)["out"]
            loss = criterion(outputs, masks) / accumulation_steps
        scaler.scale(loss).backward()
        if (i+1) % accumulation_steps == 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            if (i+1) % (accumulation_steps*10) == 0:
                torch.cuda.empty_cache()
        running_loss += loss.item() * accumulation_steps
    if (i+1) % accumulation_steps != 0:
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
    return running_loss / len(loader)

def validate(model, loader, criterion):
    model.eval(); running_loss=0.0; all_ious=[]; all_dices=[]
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False, position=1):
            images, masks = images.to(DEVICE), masks.to(DEVICE).long()
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_ious.append(compute_iou(preds.cpu(), masks.cpu(), NUM_CLASSES))
            all_dices.append(compute_dice(preds.cpu(), masks.cpu(), NUM_CLASSES))
    return running_loss/len(loader), np.nanmean(all_ious,0), np.nanmean(all_dices,0)

# ---------------- Main ----------------
def main():
    train_dataset = FloodDataset("data/images/train", "data/masks/train", transforms=get_train_transform())
    val_dataset   = FloodDataset("data/images/val", "data/masks/val", transforms=get_val_transform())

    print("Counting pixel frequencies...")
    pixel_counts = compute_pixel_class_counts(train_dataset)
    total = sum(pixel_counts.values())
    weights = torch.ones(NUM_CLASSES, dtype=torch.float)
    for i in range(NUM_CLASSES):
        weights[i] = total / (pixel_counts.get(i,1))
    weights = (weights/weights.sum())*NUM_CLASSES
    weights = weights.to(DEVICE)
    print("Class weights:", weights.cpu().numpy())

    EPOCHS=120; BATCH_SIZE=2
    train_sampler = get_sampler(train_dataset)
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,sampler=train_sampler,
                              num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
    val_loader   = DataLoader(val_dataset,batch_size=1,shuffle=False,
                              num_workers=NUM_WORKERS,pin_memory=True,drop_last=False)

    # model choice
    use_resnet101=False
    if use_resnet101:
        model=models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    else:
        model=models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead
    model.classifier=DeepLabHead(2048,NUM_CLASSES)
    model=model.to(DEVICE)

    for p in model.backbone.parameters(): p.requires_grad=False
    print("Backbone frozen.")

    criterion=get_focal_dice_loss(weights)
    optimizer=optim.AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=4,verbose=True,min_lr=1e-6)

    start_epoch=0; best_miou=-1.0
    if os.path.exists("checkpoint.pth"):
        try:
            ckpt=torch.load("checkpoint.pth",map_location=DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch=ckpt["epoch"]+1; best_miou=ckpt["best_miou"]
            print(f"Resumed from epoch {start_epoch}, best mIoU={best_miou:.4f}")
        except: pass

    scaler=GradScaler(); patience=20; counter=0

    for epoch in trange(start_epoch,EPOCHS,desc="Epochs"):
        if epoch==1: 
            for p in model.backbone.parameters(): p.requires_grad=True
            for g in optimizer.param_groups: g["lr"]*=0.5
            print("Backbone unfrozen.")
        train_loss=train_one_epoch(model,train_loader,criterion,optimizer,scaler)
        val_loss,mean_ious,mean_dices=validate(model,val_loader,criterion)
        miou=float(np.nanmean(mean_ious))
        print(f"Epoch {epoch+1}/{EPOCHS} Train={train_loss:.4f} Val={val_loss:.4f} mIoU={miou:.4f}")
        print("Per-class IoU:", np.round(mean_ious,3))

        scheduler.step(val_loss)
        torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "scheduler_state_dict":scheduler.state_dict(),
                    "best_miou":best_miou}, "checkpoint.pth")

        if miou>best_miou:
            best_miou=miou
            torch.save(model.state_dict(),"best_model.pth")
            print("Saved best_model.pth")
            counter=0
        else:
            counter+=1; print(f"No improvement {counter}/{patience}")
            if counter>=patience: print("Early stopping"); break

if __name__=="__main__":
    main()
