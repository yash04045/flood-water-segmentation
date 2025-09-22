import os
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Patch
from datasetLoader import get_val_transform
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
MODEL_PATH = "best_model.pth"
IMG_PATH = r"data/images/test/6561.jpg"   # update path as needed
MASK_PATH = r"data/masks/test/6561_lab.png"  # update path as needed

# ✅ Class mapping (your dataset)
CLASS_NAMES = [
    "Background",          # 0
    "Building-flooded",    # 1
    "Building-non-flooded",# 2
    "Road-flooded",        # 3
    "Road-non-flooded",    # 4
    "Water",               # 5
    "Tree",                # 6
    "Vehicle",             # 7
    "Pool",                # 8
    "Grass"                # 9
]

# Build model
weights = None
model = models.segmentation.deeplabv3_resnet50(weights=weights)
model.classifier = DeepLabHead(2048, NUM_CLASSES)
model = model.to(DEVICE)

# Load weights
sd = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(sd, strict=False)
model.eval()

transform = get_val_transform()

# Read image & mask
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError(f"Mask not found: {MASK_PATH}")

aug = transform(image=img, mask=mask)
img_t = aug['image'].unsqueeze(0).to(DEVICE)  # (1,C,H,W)

with torch.no_grad():
    out = model(img_t)['out']
    pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

# ✅ Distinct color map for 10 classes
colors = np.array([
    [0, 0, 0],        # Background - black
    [0, 0, 255],      # Building-flooded - blue
    [0, 255, 0],      # Building-non-flooded - green
    [255, 0, 0],      # Road-flooded - red
    [255, 255, 0],    # Road-non-flooded - yellow
    [0, 255, 255],    # Water - cyan
    [0, 128, 0],      # Tree - dark green
    [255, 165, 0],    # Vehicle - orange
    [128, 0, 128],    # Pool - purple
    [192, 192, 192],  # Grass - light gray
], dtype=np.uint8)

# Apply color map
def decode_segmap(mask, num_classes=NUM_CLASSES):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(num_classes):
        color_mask[mask == cls] = colors[cls]
    return color_mask

mask_color = decode_segmap(mask)
pred_color = decode_segmap(pred)

# Plot
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
axes[0].imshow(img); axes[0].set_title("Satellite Image"); axes[0].axis('off')
axes[1].imshow(mask_color); axes[1].set_title("Ground Truth"); axes[1].axis('off')
axes[2].imshow(pred_color); axes[2].set_title("Predicted Mask"); axes[2].axis('off')

# Overlay
pred_resized = cv2.resize(pred.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
overlay = img.copy()
for cls in range(NUM_CLASSES):
    overlay[pred_resized == cls] = colors[cls]
axes[3].imshow(overlay); axes[3].set_title("Overlay"); axes[3].axis('off')

# ✅ Legend with class names
legend_elements = [Patch(facecolor=np.array(colors[i]) / 255.0, edgecolor='black',
                         label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)]
fig.legend(handles=legend_elements, loc='center right', title="Classes", fontsize=10)

plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for legend
plt.show()
