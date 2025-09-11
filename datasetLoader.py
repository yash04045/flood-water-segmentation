import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    """Data augmentation pipeline for training."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(p=0.3),
        A.Normalize(),  # Normalize images to [0,1]
        ToTensorV2()
    ])

def get_val_transform():
    """Transforms for validation/testing (no heavy augmentation)."""
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

class FloodDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Load image (RGB)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale → binary)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)  # binary: 0 = background, 1 = flood

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, torch.tensor(mask, dtype=torch.long)
