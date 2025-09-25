import os
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Patch
from datasetLoader import get_val_transform
from torchvision import models

# --- Import correct classes for model definition ---
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
MODEL_PATH = "best_model.pth"
IMG_PATH = r"data/images/test/7935.jpg"   # update path as needed
MASK_PATH = r"data/masks/test/7935_lab.png" # update path as needed

# Class mapping (your dataset)
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

# Correctly define the ResNet-101 model architecture
print("Initializing ResNet-101 model...")
model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
model.classifier = DeepLabHead(2048, NUM_CLASSES)
model.aux_classifier = FCNHead(1024, NUM_CLASSES)
model = model.to(DEVICE)

# Load your trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Loaded best model weights from {MODEL_PATH}")

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
img_t = aug['image'].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    out = model(img_t)['out']
    pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

# Distinct color map for 10 classes
colors = np.array([
    [0, 0, 0],       # 0 Background - black
    [0, 0, 255],     # 1 Building-flooded - blue
    [0, 255, 0],     # 2 Building-non-flooded - green
    [255, 0, 0],     # 3 Road-flooded - red
    [255, 255, 0],   # 4 Road-non-flooded - yellow
    [0, 255, 255],   # 5 Water - cyan
    [0, 128, 0],     # 6 Tree - dark green
    [255, 165, 0],   # 7 Vehicle - orange
    [128, 0, 128],   # 8 Pool - purple
    [192, 192, 192], # 9 Grass - light gray
], dtype=np.uint8)

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

# --- CORRECTED OVERLAY SECTION ---
# First, resize the color prediction mask to match the original image size
pred_color_resized = cv2.resize(pred_color, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

# Now, blend the original image with the RESIZED mask
overlay = cv2.addWeighted(img, 0.6, pred_color_resized, 0.4, 0)
axes[3].imshow(overlay); axes[3].set_title("Overlay"); axes[3].axis('off')
# ---------------------------------

legend_elements = [Patch(facecolor=np.array(colors[i]) / 255.0, edgecolor='black',
                         label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)]
fig.legend(handles=legend_elements, loc='center right', title="Classes", fontsize=10)

plt.tight_layout(rect=[0, 0, 0.9, 1])
# --- SAVE THE FIGURE ---

# 1. Define the output folder
output_folder = "output"

# 2. Create a unique filename from the input image path
# This gets the base name of the image (e.g., "10808")
base_name = os.path.splitext(os.path.basename(IMG_PATH))[0]
output_filename = f"{base_name}_prediction.png"

# 3. Create the folder if it doesn't already exist
os.makedirs(output_folder, exist_ok=True)

# 4. Create the full path and save the figure
output_path = os.path.join(output_folder, output_filename)
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"✅ Visualization saved to {output_path}")


plt.show()