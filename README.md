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
