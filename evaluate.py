import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from tqdm import tqdm

# Make sure these are imported correctly from your project
from datasetLoader import FloodDataset, get_val_transform

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
CHECKPOINT_PATH = "best_model.pth"
NUM_CLASSES = 10

# Class names based on your dataset
CLASS_NAMES = {
    0: 'Background', 1: 'Building-flooded', 2: 'Building-non-flooded',
    3: 'Road-flooded', 4: 'Road-non-flooded', 5: 'Water',
    6: 'Tree', 7: 'Vehicle', 8: 'Pool', 9: 'Grass'
}

# --- Metrics ---
def compute_iou(preds, labels, num_classes):
    """Computes Intersection over Union for each class."""
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious

# --- Evaluation Loop ---
def evaluate_model(model, loader, split_name="Val"):
    """Runs the evaluation loop and computes metrics."""
    model.eval()
    all_ious = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"Evaluating {split_name}"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            ious = compute_iou(preds.cpu(), masks.cpu(), NUM_CLASSES)
            all_ious.append(ious)

    mean_ious_per_class = np.nanmean(np.array(all_ious), axis=0)
    mean_iou_overall = np.nanmean(mean_ious_per_class)
    
    return mean_iou_overall, mean_ious_per_class

# --- Main Execution ---
def main():
    """Main function to run the evaluation on train + val + test sets."""
    # Load datasets
    train_dataset = FloodDataset("data/images/train", "data/masks/train", transforms=get_val_transform())
    val_dataset   = FloodDataset("data/images/val", "data/masks/val", transforms=get_val_transform())
    test_dataset  = FloodDataset("data/images/test", "data/masks/test", transforms=get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loaded {len(train_dataset)} training images, "
          f"{len(val_dataset)} validation images, "
          f"{len(test_dataset)} test images.")

    # --- Model definition (matches training) ---
    print("Initializing ResNet-101 model...")
    model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    from torchvision.models.segmentation.deeplabv3 import DeepLabHead
    from torchvision.models.segmentation.fcn import FCNHead
    model.classifier = DeepLabHead(2048, NUM_CLASSES)
    model.aux_classifier = FCNHead(1024, NUM_CLASSES)
    model = model.to(DEVICE)

    # Load weights
    assert os.path.exists(CHECKPOINT_PATH), f"❌ Error: '{CHECKPOINT_PATH}' not found!"
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    print(f"✅ Loaded best model weights from {CHECKPOINT_PATH}")

    # --- Evaluate on Train ---
    train_miou, train_per_class = evaluate_model(model, train_loader, split_name="Train")
    # --- Evaluate on Val ---
    val_miou, val_per_class = evaluate_model(model, val_loader, split_name="Val")
    # --- Evaluate on Test ---
    test_miou, test_per_class = evaluate_model(model, test_loader, split_name="Test")

    # --- Print Results ---
    print("\n\n--- 📊 Final Evaluation Results ---")

    print(f"\nOverall Train mIoU: {train_miou:.4f}")
    print("--- Per-Class Train IoU ---")
    for i, iou in enumerate(train_per_class):
        class_name = CLASS_NAMES.get(i, f"Class {i}")
        print(f"  - {class_name:<22}: {iou:.4f}")

    print(f"\nOverall Val mIoU: {val_miou:.4f}")
    print("--- Per-Class Val IoU ---")
    for i, iou in enumerate(val_per_class):
        class_name = CLASS_NAMES.get(i, f"Class {i}")
        print(f"  - {class_name:<22}: {iou:.4f}")

    print(f"\nOverall Test mIoU: {test_miou:.4f}")
    print("--- Per-Class Test IoU ---")
    for i, iou in enumerate(test_per_class):
        class_name = CLASS_NAMES.get(i, f"Class {i}")
        print(f"  - {class_name:<22}: {iou:.4f}")

    print("---------------------\n")

if __name__ == "__main__":
    main()
