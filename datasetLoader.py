import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -----------------------
# Augmentations
# -----------------------
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=0.1),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

# -----------------------
# Dataset
# -----------------------
class FloodDataset(Dataset):
    """
    FloodNet multiclass dataset
    Classes: 0–9
    """
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

        # Load mask (grayscale, multiclass labels)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # If transforms didn’t already convert to tensor → do it manually
        if isinstance(image, np.ndarray):
            image = np.transpose(image, (2, 0, 1))  # HWC → CHW
            image = torch.tensor(image, dtype=torch.float32)

        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask