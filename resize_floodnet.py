import os
import cv2
from tqdm import tqdm

# === CONFIG ===
IMG_SIZE = 512  # Resize target (can change if needed)
SPLITS = ["train", "val", "test"]
IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

def resize_and_save(folder_path, size, is_mask=False):
    """Resize all images or masks in a folder and overwrite them."""
    for file in tqdm(os.listdir(folder_path), desc=f"Resizing {folder_path}"):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"⚠️ Skipped {file} (could not read)")
            continue

        # Choose interpolation
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized = cv2.resize(img, (size, size), interpolation=interp)

        # Save back
        cv2.imwrite(img_path, resized)

if __name__ == "__main__":
    for split in SPLITS:
        image_path = os.path.join(IMAGE_DIR, split)
        mask_path = os.path.join(MASK_DIR, split)

        # Check dirs exist
        if not os.path.exists(image_path):
            print(f"⚠️ Directory {image_path} does not exist!")
            continue
        if not os.path.exists(mask_path):
            print(f"⚠️ Directory {mask_path} does not exist!")
            continue

        print(f"\n📁 Processing {split} split...")
        resize_and_save(image_path, IMG_SIZE, is_mask=False)  # Images
        resize_and_save(mask_path, IMG_SIZE, is_mask=True)   # Masks

    print("\n✅ Resizing completed for all splits!")

