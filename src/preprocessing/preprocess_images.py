"""
preprocess_images.py - turn every MRI (ADNI DICOM series, OASIS-2 .nii.gz) into one standard 3D array.

Pipeline per scan
    1. load (DICOM: orientation from ImageOrientationPatient, slices sorted by position)
    2. reorient to RAS (same anatomical axes for ADNI and OASIS-2)
    3. resample to isotropic --voxel mm (default 2 mm)
    4. intensity: clip 0.5-99.5 percentile of the head, z-score on head voxels, background = 0
    5. centre on the head's centre of mass, crop/pad to --size^3 (default 96^3)
    6. save float16 .npy  -> Data_small/preprocessed/{adni,oasis2}/<id>.npy

Also writes preprocessed/index.csv (id, subject, date, npy path, original shape/spacing)
and preprocessed/qc_preprocessed.png (same views for both cohorts -> CHECK ORIENTATION).

NOT done (needs heavier tools): skull stripping and registration to a template (MNI).
This is a deliberate lightweight baseline; see the notes printed at the end.

Usage:
    pip install numpy pandas scipy nibabel pydicom matplotlib
    python preprocess_images.py --limit 6          # quick test on 6 scans per cohort
    python preprocess_images.py                    # everything (resumable: existing .npy are skipped)
Options: --data, --size 96, --voxel 2.0, --only adni|oasis2
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


# ----------------------------------------------------------------- loaders (heavy imports are lazy)
def load_dicom_series(folder):
    """Returns (array [r, c, slice], affine voxel->RAS mm)."""
    import pydicom
    files = sorted(Path(folder).glob("*.dcm"))
    ds = [pydicom.dcmread(str(f)) for f in files]
    iop = np.array(ds[0].ImageOrientationPatient, float)
    x_dir, y_dir = iop[:3], iop[3:]          # x_dir: direction of increasing column, y_dir: of increasing row
    normal = np.cross(x_dir, y_dir)
    pos = np.array([np.array(d.ImagePositionPatient, float) for d in ds])
    order = np.argsort(pos @ normal)
    ds = [ds[i] for i in order]; pos = pos[order]
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, "RescaleSlope", 1))
                    + float(getattr(d, "RescaleIntercept", 0)) for d in ds], axis=-1)
    d_row, d_col = [float(v) for v in ds[0].PixelSpacing]
    n = len(ds)
    step = (pos[-1] - pos[0]) / (n - 1) if n > 1 else normal * float(getattr(ds[0], "SliceThickness", 1.0))
    aff = np.eye(4)
    aff[:3, 0] = y_dir * d_row               # array axis 0 = row index
    aff[:3, 1] = x_dir * d_col               # array axis 1 = column index
    aff[:3, 2] = step
    aff[:3, 3] = pos[0]
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return vol, lps_to_ras @ aff


def load_nifti(path):
    import nibabel as nib
    im = nib.load(str(path))
    return np.asarray(im.dataobj).squeeze().astype(np.float32), im.affine


def to_ras(vol, affine):
    import nibabel as nib
    img = nib.as_closest_canonical(nib.Nifti1Image(vol, affine))
    return np.asarray(img.dataobj).astype(np.float32), nib.affines.voxel_sizes(img.affine)


# ----------------------------------------------------------------- core (pure numpy/scipy)
def resample(vol, zooms, voxel):
    f = [z / voxel for z in zooms]
    return ndi.zoom(vol, f, order=1)


def normalise_and_crop(vol, size):
    p99 = np.percentile(vol, 99)
    mask = vol > 0.1 * p99
    if mask.sum() < 1000:
        mask = vol > vol.mean()
    opened = ndi.binary_opening(mask, iterations=1)
    mask = opened if opened.sum() > 1000 else mask
    lo, hi = np.percentile(vol[mask], [0.5, 99.5])
    v = np.clip(vol, lo, hi)
    mu, sd = v[mask].mean(), v[mask].std() + 1e-6
    v = (v - mu) / sd
    v[~mask] = 0.0
    com = ndi.center_of_mass(mask)
    out = np.zeros((size,) * 3, np.float32)
    src, dst = [], []
    for ax in range(3):
        start = int(round(com[ax])) - size // 2
        s0, s1 = max(start, 0), min(start + size, v.shape[ax])
        d0 = s0 - start
        src.append(slice(s0, s1)); dst.append(slice(d0, d0 + (s1 - s0)))
    out[tuple(dst)] = v[tuple(src)]
    return out


def qc_figure(items, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(items), 3, figsize=(7, 2.3 * len(items)))
    axes = np.atleast_2d(axes)
    for r, (title, arr) in enumerate(items):
        c = arr.shape[0] // 2
        for k, sl in enumerate((arr[c, :, :], arr[:, c, :], arr[:, :, c])):
            axes[r, k].imshow(np.rot90(sl), cmap="gray", vmin=-2, vmax=3); axes[r, k].axis("off")
        axes[r, 0].set_title(title, fontsize=7, loc="left")
    fig.suptitle("RAS views: sagittal | coronal | axial (brains should look alike across cohorts)", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    ap.add_argument("--data", default=PROJECT_ROOT / "Data_small", type=Path)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--voxel", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N scans per cohort (test)")
    ap.add_argument("--only", choices=["adni", "oasis2"])
    a = ap.parse_args()

    D = Path(a.data); P = D / "preprocessed"
    jobs = []
    if a.only in (None, "adni"):
        m = pd.read_csv(D / "manifest_adni.csv")
        if a.limit: m = m.groupby("PTID").head(1).head(a.limit)
        for r in m.itertuples():
            jobs.append(dict(cohort="adni", id=f"{r.PTID}_{r.date}", subject=r.PTID, date=r.date,
                             src=D / r.subset_path, kind="dicom"))
    if a.only in (None, "oasis2"):
        m = pd.read_csv(D / "manifest_oasis2.csv")
        if a.limit: m = m.groupby("Subject ID").head(1).head(a.limit)
        for _, r in m.iterrows():
            jobs.append(dict(cohort="oasis2", id=r["MRI ID"], subject=r["Subject ID"], date=int(r["MR Delay"]),
                             src=D / r["subset_path"], kind="nifti"))

    rows, qc, t0 = [], {"adni": [], "oasis2": []}, time.time()
    for i, j in enumerate(jobs, 1):
        outp = P / j["cohort"] / f"{j['id']}.npy"
        outp.parent.mkdir(parents=True, exist_ok=True)
        try:
            if outp.exists():                       # resume: do not redo finished scans
                arr = np.load(outp).astype(np.float32); shape0, zooms = "", ()
            else:
                vol, aff = load_dicom_series(j["src"]) if j["kind"] == "dicom" else load_nifti(j["src"])
                shape0 = vol.shape
                vol, zooms = to_ras(vol, aff)
                arr = normalise_and_crop(resample(vol, zooms, a.voxel), a.size)
                np.save(outp, arr.astype(np.float16))
            rows.append(dict(cohort=j["cohort"], id=j["id"], subject=j["subject"], date=j["date"],
                             npy=str(outp.relative_to(D)).replace("\\", "/"), orig_shape=shape0,
                             orig_spacing_mm=tuple(round(float(z), 2) for z in zooms) if len(zooms) else "",
                             nonzero_frac=round(float((arr != 0).mean()), 3)))
            if len(qc[j["cohort"]]) < 4:
                qc[j["cohort"]].append((f"{j['cohort']} {j['id']}", arr))
        except Exception as e:
            print(f"  [ERROR] {j['id']}: {e}")
            rows.append(dict(cohort=j["cohort"], id=j["id"], subject=j["subject"], date=j["date"], npy="", orig_shape=f"ERROR {e}"))
        if i % 10 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} scans  ({time.time()-t0:.0f}s)")

    idx = pd.DataFrame(rows); idx.to_csv(P / "index.csv", index=False)
    items = qc["adni"] + qc["oasis2"]
    if items:
        qc_figure(items, P / "qc_preprocessed.png")
    ok = (idx.npy != "").sum()
    size = sum(f.stat().st_size for f in P.rglob("*.npy")) / 1e6
    print(f"\nDone: {ok}/{len(idx)} scans OK, {size:.0f} MB in {P}")
    print("CHECK preprocessed/qc_preprocessed.png : ADNI and OASIS-2 rows must show the same orientation.")
    print("Not done: skull stripping and template registration (use ANTsPy/FSL later if the CNN needs it).")
