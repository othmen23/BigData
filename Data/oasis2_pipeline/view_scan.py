"""Look at one raw OASIS-2 scan (3 views).  Usage: python view_scan.py path/to/mpr-1.nifti.img [out.png]"""
import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

img = nib.load(sys.argv[1])                       # point to the .img file (the .hdr must be next to it)
d = np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
print("shape:", d.shape, "| dtype on disk:", img.get_data_dtype(), "| voxel size:", img.header.get_zooms()[:3])
x, y, z = (s // 2 for s in d.shape)
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, sl, t in zip(ax, (d[x, :, :], d[:, y, :], d[:, :, z]), ("axis 0", "axis 1", "axis 2")):
    a.imshow(np.rot90(sl), cmap="gray"); a.set_title(t); a.axis("off")
out = sys.argv[2] if len(sys.argv) > 2 else "scan_preview.png"
plt.savefig(out, dpi=100, bbox_inches="tight"); print("saved", out)
