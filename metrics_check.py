import pandas as pd
import numpy as np
import sys

CSV = "metrics.csv"
if len(sys.argv) > 1:
    CSV = sys.argv[1]

df = pd.read_csv(CSV)
if df.empty:
    raise SystemExit("metrics.csv empty")

# look at last N epochs
N = min(10, len(df))
recent = df.tail(N)
val_loss_mean = recent.val_loss.mean()
val_loss_std = recent.val_loss.std()
miou_mean = recent.mIoU.mean()
miou_std = recent.mIoU.std()

print(f"Last {N} epochs — val_loss mean={val_loss_mean:.4f} std={val_loss_std:.4f} (cv={val_loss_std/val_loss_mean:.3f})")
print(f"Last {N} epochs — mIoU mean={miou_mean:.4f} std={miou_std:.4f} (cv={miou_std/(miou_mean+1e-9):.3f})")

# simple instability rules
unstable = False
if val_loss_std / (val_loss_mean + 1e-9) > 0.08:
    print("WARNING: validation loss is noisy (cv > 0.08).")
    unstable = True
if miou_std / (abs(miou_mean) + 1e-9) > 0.06:
    print("WARNING: mIoU is noisy (cv > 0.06).")
    unstable = True

if not unstable:
    print("Validation losses / mIoU appear reasonably stable in the last", N, "epochs.")
else:
    print("Unstable training detected. See suggestions below.")