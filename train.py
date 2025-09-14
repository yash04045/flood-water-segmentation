import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm, trange
from torch.amp import autocast, GradScaler
import numpy as np
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

from datasetLoader import FloodDataset, get_train_transform, get_val_transform

NUM_CLASSES = 10  # FloodNet has 10 classes (0–9)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Metrics ----------------
def compute_iou(preds, labels, num_classes):
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    return ious

def compute_dice(preds, labels, num_classes):
    dices = []
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        dice = (2. * intersection) / (pred_inds.sum().item() + target_inds.sum().item() + 1e-6)
        dices.append(dice)

    return dices

# ---------------- Training ----------------
def dice_loss(pred, target, smooth=1e-6):
    """
    pred: (N, C, H, W) raw logits
    target: (N, H, W) with class indices
    """
    pred = F.softmax(pred, dim=1)   # convert to probs
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

    intersection = torch.sum(pred * target_onehot, dim=(0, 2, 3))
    union = torch.sum(pred, dim=(0, 2, 3)) + torch.sum(target_onehot, dim=(0, 2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def get_hybrid_loss():
    # Example weights (increase rare classes: vehicles, pools, background)
    class_weights = torch.tensor(
        [1.0, 1.0, 1.2, 1.2, 1.5, 1.0, 2.0, 2.5, 2.0, 1.0],
        dtype=torch.float
    ).to(DEVICE)

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def hybrid(outputs, targets):
        ce = ce_loss(outputs, targets)
        dsc = dice_loss(outputs, targets)
        return 0.7 * ce + 0.3 * dsc  # adjust ratio if needed

    return hybrid

def train_one_epoch(model, loader, criterion, optimizer, scaler, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()  # Zero gradients at the beginning

    for i, (images, masks) in enumerate(tqdm(loader, desc="Training", leave=False, position=1)):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        masks = masks.long()

        # Mixed precision
        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)["out"]
            loss = criterion(outputs, masks) / accumulation_steps  # Normalize loss

        scaler.scale(loss).backward()

        # Update weights only after accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps  # Re-scale for logging

    return running_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_ious, all_dices = [], []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation", leave=False, position=1):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)["out"]

            loss = criterion(outputs, masks)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            ious = compute_iou(preds, masks, NUM_CLASSES)
            dices = compute_dice(preds, masks, NUM_CLASSES)

            all_ious.append(ious)
            all_dices.append(dices)

    mean_ious = np.nanmean(np.array(all_ious), axis=0)
    mean_dices = np.nanmean(np.array(all_dices), axis=0)

    return running_loss / len(loader), mean_ious, mean_dices

def main():
    # -------- Dataset --------
    train_dataset = FloodDataset("data/images/train", "data/masks/train", transforms=get_train_transform())
    val_dataset   = FloodDataset("data/images/val", "data/masks/val", transforms=get_val_transform())

    DEBUG = False
    if DEBUG:
        train_indices = list(range(0, len(train_dataset), 10))
        val_indices = list(range(0, len(val_dataset), 10))
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        EPOCHS = 5
    else:
        EPOCHS = 100

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, drop_last=False)

    # -------- Model --------
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights=weights)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model = model.to(DEVICE)

    # 🔄 Try to load best model for fine-tuning
    if os.path.exists("best_model.pth"):
        print("Loading best_model.pth for fine-tuning...")
        checkpoint = torch.load("best_model.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint)
    # Freeze backbone initially
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone frozen for first 5 epochs")

    criterion =  get_hybrid_loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    # -------- Checkpoint --------
    start_epoch = 0
    best_miou = 0.0
    checkpoint_path = "checkpoint.pth"

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
        except:
            print("Failed with weights_only=True, retrying...")
            checkpoint = torch.load(checkpoint_path, weights_only=False)

        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_miou = checkpoint['best_miou']
            print(f"Resuming from epoch {start_epoch}")

    # -------- Training --------
    patience = 10
    patience_counter = 0
    scaler = GradScaler(device="cuda")

    epoch_iterator = trange(start_epoch, EPOCHS, desc="Epochs", position=0)

    for epoch in epoch_iterator:
        # Unfreeze after 5 epochs
        if epoch == 5:
            for param in model.backbone.parameters():
                param.requires_grad = True
            epoch_iterator.write("Backbone unfrozen")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, mean_ious, mean_dices = validate(model, val_loader, criterion)

        mean_iou = np.nanmean(mean_ious)
        mean_dice = np.nanmean(mean_dices)

        epoch_iterator.write(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
            f"| mIoU: {mean_iou:.4f} | mDice: {mean_dice:.4f}"
        )

        scheduler.step(val_loss)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_miou': best_miou
        }
        torch.save(checkpoint, "checkpoint.pth")

        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), "best_model.pth")
            epoch_iterator.write("✅ Best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            epoch_iterator.write(f"⏳ No improvement. Early stopping counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            epoch_iterator.write("⛔ Early stopping triggered.")
            break

    epoch_iterator.write("🎉 Training complete!")


if __name__ == "__main__":
    main()
