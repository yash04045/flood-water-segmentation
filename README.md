# Flood Water Segmentation (WIP)

Semantic segmentation of flood-related classes using DeepLabV3-ResNet50 in PyTorch. This repository is under active development; interfaces, scripts, and hyperparameters may change.

## Project status
- Work in progress: training/evaluation pipelines are functional, but still evolving
- Large artifacts (datasets, model weights, predictions) are ignored in git via `.gitignore`

## Environment setup
- Create a virtual environment (Windows PowerShell):
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## Dataset
This project targets a 10-class FloodNet-style dataset.
- Expected on-disk layout after organization:
```
C:/flood_segmentation/
  data/
    images/
      train/ *.jpg|*.png
      val/   *.jpg|*.png
      test/  *.jpg|*.png
    masks/
      train/ *.png (indexed labels)
      val/   *.png (indexed labels)
      test/  *.png (indexed labels)
```

Two helper scripts are provided:
- `organize_dataset.py`: copies from `FloodNet-Supervised_v1.0/` into `data/images|masks/{train,val,test}`
  - Place the extracted FloodNet dataset folder at `C:/flood_segmentation/FloodNet-Supervised_v1.0/`
  - Run:
```powershell
python organize_dataset.py
```
- `resize_floodnet.py`: alternative pipeline if you have `raw/{train,val,test}/{images,masks}` and want to resize into `data/`
  - Adjust `IMG_SIZE` and `SRC_ROOT` as needed

Notes:
- Dataset and the `FloodNet-Supervised_v1.0/` folder are `.gitignore`d.
- Masks should be single-channel with integer class IDs.

## Training
- Main script: `train.py`
- Key details:
  - Auto-detects number of classes from masks under `data/masks/{train,val,test}`
  - Uses DeepLabV3-ResNet50 with a custom `DeepLabHead(NUM_CLASSES)`
  - Mixed precision (torch.amp), gradient accumulation, AdamW, ReduceLROnPlateau
  - Backbone frozen for epoch 0, then unfrozen from epoch 1
  - Class weights derived from pixel frequencies
  - Checkpoints: `checkpoint.pth` (rolling), `best_model.pth` (best mIoU)

Run:
```powershell
python train.py
```
Configuration touchpoints:
- Image size and augmentations: `datasetLoader.py` (`IMG_SIZE`, `get_train_transform`)
- Data roots: `train.py` uses `data/images/*` and `data/masks/*`

## Evaluation
- Script: `evaluate.py`
- Loads `best_model.pth` (make sure it exists from training)
```powershell
python evaluate.py
```
Outputs per-class IoU/Dice and averages.

## Inference
- Simple script: `inference.py`
  - Edit `MODEL_PATH` (defaults to `best_model.pth`) and set an input image path
  - Saves a colorized mask and a side-by-side comparison image
```powershell
python inference.py
```
- Robust script with extra checks: `inference_robust.py`
```powershell
python inference_robust.py
```

## Visualization against ground truth
- Script: `visualize_prediction.py`
  - Set `IMG_PATH` and `MASK_PATH`
```powershell
python visualize_prediction.py
```

## Classes
Class lists are defined in inference/visualization scripts and may vary slightly by file. Training auto-detects classes from masks. Ensure consistency between your dataset labels and the scripts you use for inference/visualization.

## Files of interest
- `datasetLoader.py`: Albumentations pipelines, dataset, sampler
- `train.py`: training loop, losses (Focal + Dice), checkpoints
- `evaluate.py`: metrics calculation and reporting
- `inference.py`, `inference_robust.py`: single-image inference utilities
- `visualize_prediction.py`: side-by-side GT vs prediction plots
- `organize_dataset.py`, `resize_floodnet.py`: dataset preparation helpers

## Notes on artifacts
- `.gitignore` excludes `data/`, `best_model.pth`, `checkpoint.pth`, `predictions/`, and other large/log files
- To share large weights publicly, consider Git LFS, but it is not enabled here

## Roadmap / TODO
- Finalize class naming consistency across all scripts
- Add configurable CLI args (paths, sizes, hyperparameters)
- Improve evaluation: per-split reporting and CSV export
- Add tiling/inference over folders with batching
- Add deterministic seeds and proper experiment logging
- Write full documentation and examples when stable

---
WIP: Contributions and suggestions are welcome while the project evolves.

## Current results (WIP)
- Environment: CUDA available = True; GPU = NVIDIA GeForce RTX 3050 Laptop GPU
- Throughput: ~131–135 s/epoch with batch size 2 (observed)
- Training (validation mIoU reported by `train.py`):
  - Notable checkpoints: epoch 79 mIoU=0.6644, epoch 96 mIoU=0.6669, epoch 111 mIoU=0.6795, epoch 118 mIoU=0.6817 (best)
  - Typical losses around Train≈0.27–0.29, Val≈0.32–0.33 near late epochs
- Evaluation (`evaluate.py` on val split, using `best_model.pth`):
  - Mean IoU: 0.3150
  - Mean Dice: 0.2778
  - Per-class (IoU, Dice):
    - 0: (0.0148, 0.0041)
    - 1: (0.2948, 0.0685)
    - 2: (0.3360, 0.2774)
    - 3: (0.2176, 0.0613)
    - 4: (0.4183, 0.4062)
    - 5: (0.4310, 0.3778)
    - 6: (0.4963, 0.5729)
    - 7: (0.0936, 0.1130)
    - 8: (0.1496, 0.1183)
    - 9: (0.6977, 0.7783)
- Note: There is a discrepancy between the training-reported mIoU and the evaluation script’s mean IoU. Likely causes include differences in metric definitions (mean over classes vs. NaN handling), preprocessing/resize pipelines, or class mappings. This will be reconciled in future updates.
- Misc: Albumentations occasionally logs a harmless version check warning due to network timeouts during training.
