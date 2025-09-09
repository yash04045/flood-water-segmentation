import os
import shutil

# Source directory where FloodNet-Supervised_v1.0 is extracted
SRC_DIR = "data/FloodNet-Supervised_v1.0"
DST_DIR = "data"

splits = ["train", "val", "test"]

for split in splits:
    if split == "train":
        img_src = os.path.join(SRC_DIR, "train", "train-org-img")
        mask_src = os.path.join(SRC_DIR, "train", "train-label-img")
    elif split == "val":
        img_src = os.path.join(SRC_DIR, "val", "val-org-img")
        mask_src = os.path.join(SRC_DIR, "val", "val-label-img")
    else:  # test
        img_src = os.path.join(SRC_DIR, "test", "test-org-img")
        mask_src = os.path.join(SRC_DIR, "test", "test-label-img")
    
    img_dst = os.path.join(DST_DIR, "images", split)
    mask_dst = os.path.join(DST_DIR, "masks", split)
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(mask_dst, exist_ok=True)

    # Copy images
    for file in os.listdir(img_src):
        if file.endswith((".jpg", ".png", ".jpeg")):
            shutil.copy(os.path.join(img_src, file), os.path.join(img_dst, file))

    # Copy masks
    for file in os.listdir(mask_src):
        if file.endswith((".jpg", ".png", ".jpeg")):
            shutil.copy(os.path.join(mask_src, file), os.path.join(mask_dst, file))

print("✅ Dataset organized successfully into images/ and masks/ folders!")
