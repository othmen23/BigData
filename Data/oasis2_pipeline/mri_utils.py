"""Readers for the two MRI formats you have, plus the shared normalisation."""
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def load_analyze(path) -> np.ndarray:
    """OASIS-2: path to the .img of an .img/.hdr pair (the .hdr must sit next to it)."""
    img = nib.load(str(path))
    try:
        img = nib.as_closest_canonical(img)     # Analyze headers can carry unreliable orientation
    except Exception:
        pass
    return np.squeeze(np.asarray(img.dataobj, dtype=np.float32))


def load_dicom_series(folder) -> np.ndarray:
    """ADNI: a folder holding ONE series = one .dcm file per slice. Returns a 3D array."""
    import pydicom
    files = sorted(Path(folder).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no .dcm files in {folder}")
    ds = [pydicom.dcmread(str(f), force=True) for f in files]
    try:   # sort slices along the slice normal (robust), else by InstanceNumber
        iop = np.array(ds[0].ImageOrientationPatient, float)
        normal = np.cross(iop[:3], iop[3:])
        ds.sort(key=lambda d: float(np.dot(normal, np.array(d.ImagePositionPatient, float))))
    except Exception:
        ds.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
    shapes = {d.pixel_array.shape for d in ds}
    if len(shapes) != 1:
        raise ValueError(f"inconsistent slice shapes {shapes} (mixed series in {folder}?)")
    slices = []
    for d in ds:
        a = d.pixel_array.astype(np.float32)
        slices.append(a * float(getattr(d, "RescaleSlope", 1)) + float(getattr(d, "RescaleIntercept", 0)))
    return np.stack(slices, axis=-1)


def normalize_resize(d: np.ndarray, size: int):
    """Clip, z-score on nonzero voxels, resample to size^3. Returns (float16 volume, nonzero fraction)."""
    if d.ndim != 3:
        raise ValueError(f"expected 3D volume, got {d.shape}")
    d = np.clip(np.nan_to_num(d), 0, np.percentile(d, 99.5))
    mask = d > 0
    d = np.where(mask, (d - d[mask].mean()) / (d[mask].std() + 1e-6), 0).astype(np.float32)
    out = zoom(d, [size / s for s in d.shape], order=1)
    if out.shape != (size,) * 3:                 # guard against rounding
        fixed = np.zeros((size,) * 3, np.float32)
        s = tuple(slice(0, min(a, size)) for a in out.shape)
        fixed[s] = out[s]
        out = fixed
    return out.astype(np.float16), float(mask.mean())
