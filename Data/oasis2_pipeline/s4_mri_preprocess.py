"""Step 4 (OASIS-2): first mpr scan of each session -> small .npy volume.
Not done here: skull stripping, registration, averaging the 3 scans, bias-field correction."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import config
from mri_utils import load_analyze, normalize_resize


def work(args):
    mri_id, path, size = args
    target = config.MRI_DIR / f"{mri_id}.npy"
    if target.exists():
        return mri_id, "skipped", None, None
    try:
        d = load_analyze(path)
        vol, frac = normalize_resize(d, size)
        np.save(target, vol)
        return mri_id, "ok", str(d.shape), frac
    except Exception as e:
        return mri_id, f"ERROR {e}", None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=2)
    a = ap.parse_args()
    config.MRI_DIR.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(config.MRI_INDEX_CSV).dropna(subset=["first_scan"])
    jobs = [(r.mri_id, r.first_scan, a.size) for r in idx.itertuples()][: a.limit]
    with ProcessPoolExecutor(a.n_jobs) as ex:
        res = list(ex.map(work, jobs))
    qc = pd.DataFrame(res, columns=["mri_id", "status", "orig_shape", "nonzero_fraction"])
    qc.to_csv(config.OUT_DIR / "mri_preprocess_qc.csv", index=False)
    print(qc["status"].str.split().str[0].value_counts().to_dict())
    for r in qc[qc["status"].str.startswith("ERROR")].itertuples():
        print(r.mri_id, r.status)
    print(f"Volumes in {config.MRI_DIR}")


if __name__ == "__main__":
    main()
