"""Check converted volumes + save a picture of random ones to eyeball.
  python check_npy.py --dir outputs/adni_npy --n 12"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()
files = sorted(Path(a.dir).glob("*.npy"))
print(f"{len(files)} volumes in {a.dir}")
shapes, bad = set(), []
for f in files:
    v = np.load(f).astype(np.float32)
    shapes.add(v.shape)
    if not np.isfinite(v).all() or v.std() < 0.2 or (v != 0).mean() < 0.05:
        bad.append(f.name)
print("shapes:", shapes, "(should be a single shape)")
print("suspicious volumes (NaN / flat / mostly empty):", bad if bad else "none")
pick = np.random.default_rng(0).choice(len(files), min(a.n, len(files)), replace=False)
cols = 4
rows = int(np.ceil(len(pick) / cols))
fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
for a_ in np.atleast_1d(ax).ravel():
    a_.axis("off")
for a_, i in zip(np.atleast_1d(ax).ravel(), pick):
    v = np.load(files[i]).astype(np.float32)
    a_.imshow(np.rot90(v[:, :, v.shape[2] // 2]), cmap="gray")
    a_.set_title(files[i].stem[-22:], fontsize=7)
out = Path(a.dir) / "montage.png"
plt.savefig(out, dpi=90, bbox_inches="tight")
print("picture saved:", out)
