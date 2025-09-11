import os
import cv2
from tqdm import tqdm

# === CONFIG ===
IMG_SIZE = 256  # Change to 256 for faster training
SPLITS = ["train", "val", "test"]
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

def resize_and_save(folder_path, size):
    """Resize all images in a folder and save them back."""
    for file in tqdm(os.listdir(folder_path), desc=f"Resizing {folder_path}"):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"⚠️ Skipped {file} (could not read)")
            continue

        # Resize image/mask
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(img_path, resized)

if __name__ == "__main__":
    for split in SPLITS:
        image_path = os.path.join(IMAGE_DIR, split)
        mask_path = os.path.join(MASK_DIR, split)
        
        # Check if directories exist
        if not os.path.exists(image_path):
            print(f"⚠️ Directory {image_path} does not exist!")
            continue

        if not os.path.exists(mask_path):
            print(f"⚠️ Directory {mask_path} does not exist!")
            continue
            
        print(f"📁 Processing {split} split...")
        resize_and_save(image_path, IMG_SIZE)
        resize_and_save(mask_path, IMG_SIZE)
