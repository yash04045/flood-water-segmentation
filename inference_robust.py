import os
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"

# Class names
CLASS_NAMES = [
    "background",
    "building-flooded",
    "building-nonflooded", 
    "road",
    "water",
    "tree",
    "vehicle",
    "pool",
    "other",
    "unknown"
]

# Define colormap
COLOR_MAP = np.array([
    [0, 0, 0],        # 0 background
    [0, 0, 255],      # 1 building-flooded
    [0, 255, 255],    # 2 building-nonflooded
    [0, 255, 0],      # 3 road
    [255, 0, 0],      # 4 water
    [255, 255, 0],    # 5 tree
    [128, 0, 128],    # 6 vehicle
    [255, 165, 0],    # 7 pool
    [128, 128, 128],  # 8 other
    [255, 105, 180],  # 9 unknown
], dtype=np.uint8)

# -------------------------
# MODEL LOADING
# -------------------------
def load_model():
    num_classes = len(CLASS_NAMES)
    
    # Create model with correct architecture
    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.classifier = DeepLabHead(2048, num_classes)
    model = model.to(DEVICE)
    
    # Load weights with error handling
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Loading with strict=False to ignore mismatched keys...")
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model

# -------------------------
# PREPROCESS
# -------------------------
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((384, 384)),   # Match training size
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# -------------------------
# PREDICT FUNCTION
# -------------------------
def predict(model, image_path, save_mask_path="prediction_mask.png", save_compare_path="comparison.png"):
    # Load original
    orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if orig is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # Preprocess
    augmented = transform(orig)
    tensor = augmented.unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(tensor)["out"]
        pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

    # Colorize mask
    color_mask = COLOR_MAP[pred_mask].astype(np.uint8)

    # Save mask separately
    cv2.imwrite(save_mask_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

    # Prepare side-by-side comparison
    orig_resized = cv2.resize(orig, (384, 384))
    overlay = cv2.addWeighted(orig_resized, 0.6, color_mask, 0.4, 0)

    # Plot side-by-side
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))

    ax[0].imshow(orig_resized)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(color_mask)
    ax[1].set_title("Predicted Segmentation")
    ax[1].axis("off")

    ax[2].imshow(overlay)
    ax[2].set_title("Overlay")
    ax[2].axis("off")

    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=tuple(c/255 for c in COLOR_MAP[i])) for i in range(len(CLASS_NAMES))]
    fig.legend(handles, CLASS_NAMES, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save the side-by-side comparison
    plt.savefig(save_compare_path, bbox_to_anchor="tight")
    plt.show()

    print(f"✅ Prediction mask saved at {save_mask_path}")
    print(f"✅ Comparison image saved at {save_compare_path}")
    
    # Print class statistics
    unique_classes, counts = np.unique(pred_mask, return_counts=True)
    print("\n📊 Class Distribution:")
    for cls, count in zip(unique_classes, counts):
        percentage = (count / pred_mask.size) * 100
        print(f"Class {cls} ({CLASS_NAMES[cls]}): {count} pixels ({percentage:.1f}%)")

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    model = load_model()
    
    # Test with a sample image
    test_image = r"data/images/val/10169.jpg"  # Change this to your test image
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        print("Please update the test_image path in the script")
    else:
        predict(model, test_image, "pred_mask.png", "comparison.png")
