import torch
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from datasetLoader import FloodDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
import os

# ====================
# CONFIG
# ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
SAVE_VISUALS = True  # set False if you don’t want prediction images
OUTPUT_DIR = "predictions"

# ====================
# DATA TRANSFORM
# ====================
val_transform = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])

# ====================
# DATASET & LOADER
# ====================
val_dataset = FloodDataset("data/images/val", "data/masks/val", transforms=val_transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ====================
# MODEL
# ====================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,   # because we are loading our trained weights
    in_channels=3,
    classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# ====================
# METRICS
# ====================
def compute_iou(pred, target, num_classes=2):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds[target_inds]).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious

def compute_dice(pred, target, num_classes=2, smooth=1e-6):
    dices = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = (pred == cls).float()
        target_inds = (target == cls).float()
        intersection = (pred_inds * target_inds).sum().item()
        dice = (2 * intersection + smooth) / (pred_inds.sum().item() + target_inds.sum().item() + smooth)
        dices.append(dice)
    return dices

# ====================
# EVALUATION LOOP
# ====================
all_ious, all_dices = [], []

if SAVE_VISUALS and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

with torch.no_grad():
    for idx, (image, mask) in enumerate(tqdm(val_loader, desc="Evaluating")):
        image, mask = image.to(DEVICE), mask.to(DEVICE)

        output = model(image)
        pred = torch.argmax(output, dim=1)  # predicted mask

        # metrics
        ious = compute_iou(pred.cpu(), mask.cpu(), num_classes=NUM_CLASSES)
        dices = compute_dice(pred.cpu(), mask.cpu(), num_classes=NUM_CLASSES)

        all_ious.append(ious)
        all_dices.append(dices)

        # visualization
        if SAVE_VISUALS:
            img_np = (image[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pred_mask = pred[0].cpu().numpy().astype(np.uint8) * 255
            gt_mask = mask[0].cpu().numpy().astype(np.uint8) * 255

            # overlay prediction in red, ground truth in green
            overlay = img_np.copy()
            overlay[pred_mask > 0, 0] = 255   # red channel
            overlay[gt_mask > 0, 1] = 255     # green channel

            cv2.imwrite(os.path.join(OUTPUT_DIR, f"sample_{idx}.png"), overlay)

# ====================
# RESULTS
# ====================
mean_ious = np.nanmean(np.array(all_ious), axis=0)
mean_dices = np.nanmean(np.array(all_dices), axis=0)

print(f"\nPer-class IoU: {mean_ious}")
print(f"Mean IoU: {np.nanmean(mean_ious)}")
print(f"Per-class Dice: {mean_dices}")
print(f"Mean Dice: {np.nanmean(mean_dices)}")
