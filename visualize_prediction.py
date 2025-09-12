import os
import torch
import matplotlib.pyplot as plt
from datasetLoader import FloodDataset, get_val_transform
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

# ===============================
# CONFIG
# ===============================
IMG_DIR = "data/images/val"
MASK_DIR = "data/masks/val"
MODEL_PATH = "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2  # background + flood

# ===============================
# DATASET & LOADER
# ===============================
dataset = FloodDataset(
    img_dir=IMG_DIR,
    mask_dir=MASK_DIR,
    transforms=get_val_transform()
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ===============================
# LOAD MODEL
# ===============================
model = smp.Unet(
    encoder_name="resnet34",  # Ensure this matches train.py
    encoder_weights="imagenet",        # Set to None for inference
    in_channels=3,
    classes=2
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===============================
# VISUALIZATION FUNCTION
# ===============================
def visualize(img, mask, pred):
    plt.figure(figsize=(12, 4))

    # Original Image
    plt.subplot(1, 4, 1)
    plt.imshow(img.permute(1, 2, 0).cpu())
    plt.title("Satellite Image")
    plt.axis("off")

    # Ground Truth Mask
    plt.subplot(1, 4, 2)
    plt.imshow(mask.cpu(), cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # Predicted Mask
    plt.subplot(1, 4, 3)
    plt.imshow(pred.cpu(), cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    # Overlay Prediction on Image
    plt.subplot(1, 4, 4)
    plt.imshow(img.permute(1, 2, 0).cpu())
    plt.imshow(pred.cpu(), cmap="jet", alpha=0.5)  # overlay
    plt.title("Overlay (Flood Highlight)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ===============================
# RUN VISUALIZATION
# ===============================
if __name__ == "__main__":
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            visualize(imgs[0], masks[0], preds[0])
            break  # show 1 batch (remove break to show more)
