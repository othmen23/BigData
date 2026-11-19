"""Convert the selected ADNI DICOM series to small .npy volumes (then the raw DICOM can be archived).
  python convert_dicom.py --subset outputs/subset_scans.csv --size 96"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import config
from mri_utils import load_dicom_series, normalize_resize

ADNI_NPY = config.OUT_DIR / "adni_npy"


def work(args):
    name, folder, size = args
    target = ADNI_NPY / f"{name}.npy"
    if target.exists():
        return name, "skipped", None, None
    try:
        d = load_dicom_series(folder)
        vol, frac = normalize_resize(d, size)
        np.save(target, vol)
        return name, "ok", str(d.shape), frac
    except Exception as e:
        return name, f"ERROR {e}", None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(config.OUT_DIR / "subset_scans.csv"))
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    ADNI_NPY.mkdir(parents=True, exist_ok=True)
    sub = pd.read_csv(a.subset)
    jobs = [(f"{r.ptid}_{r.scan_date}_{r.image_id}", r.folder, a.size) for r in sub.itertuples()]
    with ProcessPoolExecutor(a.n_jobs) as ex:
        res = list(ex.map(work, jobs))
    qc = pd.DataFrame(res, columns=["name", "status", "orig_shape", "nonzero_fraction"])
    qc.to_csv(config.OUT_DIR / "adni_convert_qc.csv", index=False)
    print(qc["status"].str.split().str[0].value_counts().to_dict())
    for r in qc[qc["status"].str.startswith("ERROR")].itertuples():
        print(r.name, r.status)
    print("Volumes in", ADNI_NPY)


if __name__ == "__main__":
    main()
