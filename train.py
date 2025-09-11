# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from datasetLoader import FloodDataset, get_train_transform, get_val_transform

# ===============================
# CONFIG
# ===============================
IMG_DIR = "data/images"
MASK_DIR = "data/masks"

BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2  # background + flood

# ===============================
# DATASETS & LOADERS
# ===============================
train_dataset = FloodDataset(
    img_dir=os.path.join(IMG_DIR, "train"),
    mask_dir=os.path.join(MASK_DIR, "train"),
    transforms=get_train_transform()  # Changed 'transform' to 'transforms'
)

val_dataset = FloodDataset(
    img_dir=os.path.join(IMG_DIR, "val"),
    mask_dir=os.path.join(MASK_DIR, "val"),
    transforms=get_val_transform()  # Changed 'transform' to 'transforms'
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ===============================
# MODEL
# ===============================
model = smp.Unet(
    encoder_name="mobilenet_v2",  # Changed from mobilenet_v3_large to mobilenet_v2
    encoder_weights="imagenet",  # Transfer learning
    in_channels=3,
    classes=NUM_CLASSES
).to(DEVICE)

# ===============================
# LOSS & OPTIMIZER
# ===============================
loss_fn = smp.losses.DiceLoss(mode="multiclass")  # Dice loss for segmentation
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===============================
# TRAINING LOOP
# ===============================
def train_one_epoch(loader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def validate(loader, model, loss_fn):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.long().to(DEVICE)
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


# ===============================
# MAIN LOOP
# ===============================
if __name__ == '__main__':
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer)
        val_loss = validate(val_loader, model, loss_fn)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Saved new best model!")

    print("🎉 Training finished!")
