# evaluate.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datasetLoader import FloodDataset, get_val_transform
import segmentation_models_pytorch as smp  # Use the same library as in train.py

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Metrics ----
def iou_score(pred, target, num_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # ignore empty class
        else:
            ious.append(intersection / union)
    return ious

def dice_score(pred, target, num_classes):
    dices = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        dice = (2 * intersection) / (pred_inds.sum().item() + target_inds.sum().item() + 1e-7)
        dices.append(dice)
    return dices

# ---- Evaluation ----
def evaluate(model_path="best_model.pth", data_dir="data/images/test", mask_dir="data/masks/test", num_classes=2, batch_size=4):
    dataset = FloodDataset(
        img_dir=data_dir,
        mask_dir=mask_dir,
        transforms=get_val_transform()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Define the model (same as in train.py)
    model = smp.Unet(
        encoder_name="mobilenet_v2",  # Same encoder used during training
        encoder_weights=None,        # No pre-trained weights, as we are loading our own
        in_channels=3,
        classes=num_classes
    ).to(DEVICE)

    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_ious, all_dices = [], []

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            ious = iou_score(preds, masks, num_classes)
            dices = dice_score(preds, masks, num_classes)

            all_ious.append(ious)
            all_dices.append(dices)

    mean_ious = np.nanmean(np.array(all_ious), axis=0)
    mean_dices = np.nanmean(np.array(all_dices), axis=0)

    print("Per-class IoU:", mean_ious)
    print("Mean IoU:", np.nanmean(mean_ious))
    print("Per-class Dice:", mean_dices)
    print("Mean Dice:", np.nanmean(mean_dices))

if __name__ == "__main__":
    evaluate()
