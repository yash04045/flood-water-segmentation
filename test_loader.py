# test_loader.py
import torch
from torch.utils.data import DataLoader
from datasetLoader import FloodDataset

if __name__ == "__main__":
    # Paths
    image_dir = "data/images/train"
    mask_dir = "data/masks/train"

    # Create dataset
    dataset = FloodDataset(image_dir, mask_dir)

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Fetch a batch
    images, masks = next(iter(dataloader))

    print("Images:", images.shape)   # [B, 3, H, W]
    print("Masks:", masks.shape)     # [B, H, W]
    print("Unique values in mask batch:", torch.unique(masks))
