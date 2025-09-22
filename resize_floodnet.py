import os
import cv2
from tqdm import tqdm

# === CONFIG ===
IMG_SIZE = 512  # Resize target (can change if needed)
SPLITS = ["train", "val", "test"]
SRC_ROOT = "raw"  # set to your raw dataset root
DST_ROOT = "data"

os.makedirs(DST_ROOT, exist_ok=True)
for split in SPLITS:
    src_images = os.path.join(SRC_ROOT, split, "images")
    src_masks  = os.path.join(SRC_ROOT, split, "masks")
    dst_images = os.path.join(DST_ROOT, "images", split)
    dst_masks  = os.path.join(DST_ROOT, "masks", split)
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_masks, exist_ok=True)
    if not os.path.isdir(src_images):
        continue
    files = sorted(os.listdir(src_images))
    for f in tqdm(files, desc=f"Resizing {split} images"):
        img = cv2.imread(os.path.join(src_images, f))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(dst_images, f), img)
    files = sorted(os.listdir(src_masks))
    for f in tqdm(files, desc=f"Resizing {split} masks"):
        m = cv2.imread(os.path.join(src_masks, f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(dst_masks, f), m)

print("\n✅ Resizing completed for all splits!")

