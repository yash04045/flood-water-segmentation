import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
from datasetLoader import FloodDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2

# ====================
# CONFIG
# ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 100
PATIENCE = 10   # for early stopping

# ====================
# DATA AUGMENTATION
# ====================
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.2),
    A.Normalize(),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])

# ====================
# DATASET & LOADER
# ====================
train_dataset = FloodDataset(
    "C:/flood_segmentation/data/images/train",
    "C:/flood_segmentation/data/masks/train",
    transforms=train_transform
)

val_dataset = FloodDataset(
    "C:/flood_segmentation/data/images/val",
    "C:/flood_segmentation/data/masks/val",
    transforms=val_transform
)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ====================
# MODEL
# ====================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2
).to(DEVICE)

# ====================
# LOSS FUNCTION
# ====================
class_weights = torch.tensor([0.3, 0.7]).to(DEVICE)  # adjust based on dataset
ce_loss = nn.CrossEntropyLoss(weight=class_weights)

def dice_loss(pred, target, smooth=1):
    pred = torch.softmax(pred, dim=1)[:,1,...]
    target = (target == 1).float()
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def combined_loss(pred, target):
    return ce_loss(pred, target) + dice_loss(pred, target)

# ====================
# OPTIMIZER & SCHEDULER
# ====================
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ====================
# TRAIN & VALIDATION LOOP
# ====================
if __name__ == '__main__':
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        # ---- TRAIN ----
        model.train()
        train_losses = []
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ---- VALIDATION ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ---- CHECKPOINTING ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Saved new best model!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹ Early stopping triggered!")
                break

    print("🎉 Training finished!")

class FloodDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None):
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask

    def __len__(self):
        return len(self.images)
