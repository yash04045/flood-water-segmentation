import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from tqdm import tqdm
import torch.nn as nn

from datasetLoader import FloodDataset, get_val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10

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

def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_ious, all_dices = [], []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)["out"]

            loss = criterion(outputs, masks)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            ious = compute_iou(preds, masks, NUM_CLASSES)
            dices = compute_dice(preds, masks, NUM_CLASSES)

            all_ious.append(ious)
            all_dices.append(dices)

    mean_iou = np.nanmean(np.array(all_ious), axis=0)
    mean_dice = np.nanmean(np.array(all_dices), axis=0)

    return running_loss / len(loader), mean_iou, mean_dice


def main():
    # Load dataset
    val_dataset = FloodDataset("data/images/val", "data/masks/val", transforms=get_val_transform())
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

    # Load model
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights=weights)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model = model.to(DEVICE)

    # Load trained weights
    checkpoint_path = "best_model.pth"
    assert os.path.exists(checkpoint_path), "❌ best_model.pth not found!"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f"✅ Loaded best model from {checkpoint_path}")

    criterion = nn.CrossEntropyLoss()

    # Evaluate
    val_loss, mean_iou, mean_dice = evaluate_model(model, val_loader, criterion)

    print("\n📊 Evaluation Results:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Mean IoU: {np.nanmean(mean_iou):.4f}")
    print(f"Mean Dice: {np.nanmean(mean_dice):.4f}")

    for cls, (iou, dice) in enumerate(zip(mean_iou, mean_dice)):
        print(f"Class {cls}: IoU={iou:.4f}, Dice={dice:.4f}")


if __name__ == "__main__":
    main()
