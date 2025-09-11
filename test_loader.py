from datasetLoader import FloodDataset, get_transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    train_dataset = FloodDataset(
        "data/images/train",
        "data/masks/train",
        transforms=get_transforms()
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    for imgs, masks in train_loader:
        print(f"Images: {imgs.shape}, Masks: {masks.shape}")
        break

    print("✅ Dataset loading works correctly!")
