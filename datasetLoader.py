import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# FloodNet classes (10 classes)
CLASS_COLORS = {
    0: [0, 0, 0],          # Background
    1: [0, 0, 255],        # Building Flooded
    2: [0, 255, 255],      # Building Non-Flooded
    3: [0, 255, 0],        # Road Flooded
    4: [255, 255, 0],      # Road Non-Flooded
    5: [255, 0, 0],        # Water
    6: [255, 0, 255],      # Tree
    7: [192, 192, 192],    # Vehicle
    8: [128, 128, 128],    # Pool
    9: [128, 0, 0],        # Grass
}

COLOR_TO_CLASS = {tuple(v): k for k, v in CLASS_COLORS.items()}

def mask_to_class(mask):
    """Convert color mask to class ID mask."""
    h, w, _ = mask.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for color, cls_id in COLOR_TO_CLASS.items():
        matches = np.all(mask == np.array(color), axis=-1)
        class_mask[matches] = cls_id
    return class_mask

def get_transforms():
    """Data augmentation pipeline."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(p=0.3),
        A.Normalize(),  # Normalize images to [0,1]
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

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask_to_class(mask)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, torch.tensor(mask, dtype=torch.long)
