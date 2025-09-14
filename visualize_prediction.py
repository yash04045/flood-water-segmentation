import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import torch.nn as nn

from datasetLoader import FloodDataset, get_val_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10

# Class names (FloodNet challenge)
CLASS_NAMES = [
    "Background", 
    "Building-flooded", 
    "Building-non-flooded", 
    "Road-flooded", 
    "Road-non-flooded", 
    "Water", 
    "Tree", 
    "Vehicle", 
    "Pool", 
    "Grass"
]

# Simple color palette for 10 classes
COLORS = np.array([
    [0, 0, 0],         # Background
    [0, 0, 255],       # Building-flooded
    [0, 255, 0],       # Building-non-flooded
    [255, 0, 0],       # Road-flooded
    [255, 255, 0],     # Road-non-flooded
    [0, 255, 255],     # Water
    [255, 0, 255],     # Tree
    [128, 128, 128],   # Vehicle
    [255, 165, 0],     # Pool
    [128, 0, 128],     # Grass
], dtype=np.uint8)

def decode_segmap(mask):
    """Convert class mask to RGB image"""
    return COLORS[mask]

def get_legend_elements():
    """Create legend patches for all classes"""
    legend_elements = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        legend_elements.append(Patch(
            facecolor=np.array(COLORS[class_id])/255.0,
            edgecolor='black',
            label=f"{class_id}: {class_name}"
        ))
    return legend_elements

def visualize_sample(model, dataset, idx=0):
    model.eval()
    image, mask = dataset[idx]
    image_tensor = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)["out"]
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Convert to numpy
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.numpy()
    pred_np = pred

    # Decode colors
    mask_color = decode_segmap(mask_np)
    pred_color = decode_segmap(pred_np)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(image_np)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(mask_color)
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(pred_color)
    axs[2].set_title("Prediction")
    axs[2].axis("off")

    # Add legend below the plots
    legend_elements = get_legend_elements()
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=8)

    # Add version info
    fig.suptitle("FloodNet-Supervised_v1.0", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


def main():
    # Load dataset
    val_dataset = FloodDataset("data/images/val", "data/masks/val", transforms=get_val_transform())

    # Find index of target image
    target_image_name = "7334.jpg"
    target_idx = None
    for i, img_name in enumerate(val_dataset.images):
        if img_name == target_image_name:
            target_idx = i
            break
    if target_idx is None:
        raise ValueError(f"Image {target_image_name} not found in validation dataset")

    # Load model
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights=weights)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model = model.to(DEVICE)

    # Load trained weights
    checkpoint_path = "best_model.pth"
    assert os.path.exists(checkpoint_path), "❌ best_model.pth not found!"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f"✅ Loaded best model from {checkpoint_path}")

    # Visualize the sample
    visualize_sample(model, val_dataset, idx=target_idx)


if __name__ == "__main__":
    main()
