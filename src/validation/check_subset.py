"""
check_subset.py - verify Data_small/ before you delete anything from the original Data.

Checks: files exist, no duplicate IDs, class balance, visits per subject, every subject
has rows in each table, image-to-diagnosis date gap (ADNI), images load (DICOM series /
NIfTI), shapes consistent, not empty. Saves mid-slice PNGs for visual QC.

Usage:
    pip install pandas nibabel pydicom numpy matplotlib
    python check_subset.py                   # uses ../Data_small relative to default root
    python check_subset.py "C:\\...\\Data_small"
Output: Data_small/check_report.txt  and  Data_small/qc_png/*.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT = Path(__file__).resolve().parents[2] / "Data_small"
out = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
qc = out / "qc_png"; qc.mkdir(exist_ok=True)
rep, problems = [], []


def say(*x):
    s = " ".join(str(i) for i in x); print(s); rep.append(s)


def bad(msg):
    problems.append(msg); say("  [PROBLEM]", msg)


def save_png(vol, name):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        v = np.asarray(vol, dtype=float)
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        for i in range(3):
            sl = np.take(v, v.shape[i] // 2, axis=i)
            ax[i].imshow(np.rot90(sl), cmap="gray"); ax[i].axis("off")
        fig.suptitle(name, fontsize=8); fig.savefig(qc / f"{name}.png", dpi=80); plt.close(fig)
    except Exception as e:
        say("  (png skipped:", e, ")")


def vol_stats(v, label):
    v = np.asarray(v, dtype=float)
    fin = np.isfinite(v).mean(); nz = (v != 0).mean()
    say(f"    {label}: shape={v.shape} finite={fin:.3f} nonzero={nz:.3f} mean={np.nanmean(v):.1f}")
    if fin < 1: bad(f"{label}: contains NaN/inf")
    if nz < 0.05: bad(f"{label}: almost empty image")


# ============================================================ ADNI
say("=" * 70, "\nADNI\n" + "=" * 70)
m = pd.read_csv(out / "manifest_adni.csv")
say(f"subjects: {m.PTID.nunique()} | visits(images): {len(m)}")
if m.duplicated(["PTID", "date"]).any(): bad("duplicate (PTID, date) in ADNI manifest")
vis = m.groupby("PTID").date.nunique()
say("visits per subject:", vis.value_counts().sort_index().to_dict())
say("class balance (subjects):", m.drop_duplicates("PTID").stratum.value_counts().to_dict())

# folders / files
for r in m.itertuples():
    d = out / r.subset_path
    n = len(list(d.glob("*.dcm"))) if d.exists() else 0
    if n == 0: bad(f"{r.subset_path}: missing or no .dcm")
    elif n != r.n_dcm: bad(f"{r.subset_path}: {n} dcm vs {r.n_dcm} expected")

# table coverage
say("\ntable coverage (subjects found / selected):")
ids = set(m.PTID)
for f in sorted((out / "ADNI_Tables").glob("*.csv")):
    if "DATADIC" in f.name: continue
    d = pd.read_csv(f, dtype=str, usecols=lambda c: c in ("PTID", "subject_id"))
    col = d.columns[0]
    found = len(ids & set(d[col]))
    say(f"  {f.name[:60]:60s} {found:3d}/{len(ids)}  rows={len(d)}")

# image date vs diagnosis date
dxf = next((out / "ADNI_Tables").glob("*DXSUM*.csv"), None)
if dxf is not None:
    dx = pd.read_csv(dxf, dtype=str)[["PTID", "EXAMDATE"]].dropna()
    dx["EXAMDATE"] = pd.to_datetime(dx.EXAMDATE, errors="coerce")
    gaps = []
    for r in m.itertuples():
        e = dx[dx.PTID == r.PTID].EXAMDATE
        if len(e): gaps.append(int((e - pd.to_datetime(r.date)).abs().dt.days.min()))
    if gaps:
        say(f"\nclosest diagnosis visit to each scan (days): median={np.median(gaps):.0f} max={max(gaps)}")
        g = np.array(gaps)
        say(f"  scans with nearest diagnosis > 90 days: {(g>90).sum()} | > 180 days: {(g>180).sum()} | > 365 days: {(g>365).sum()}")
        if (g > 365).any(): bad("some scans have no diagnosis within 365 days")
        elif (g > 180).any(): say("  [WARNING] a few scans are >180 days from a diagnosis: normal for 6-month ADNI visits, match with a tolerance")

# load some DICOM series
say("\nDICOM load test (up to 6 series):")
try:
    import pydicom
    for r in m.sample(min(6, len(m)), random_state=0).itertuples():
        d = out / r.subset_path
        ds = [pydicom.dcmread(str(f)) for f in sorted(d.glob("*.dcm"))]
        ds.sort(key=lambda x: int(getattr(x, "InstanceNumber", 0)))
        pid = str(ds[0].PatientID)
        if pid != r.PTID: bad(f"{r.subset_path}: PatientID {pid} != folder {r.PTID}")
        if len({(x.Rows, x.Columns) for x in ds}) > 1: bad(f"{r.subset_path}: inconsistent slice size")
        vol = np.stack([x.pixel_array for x in ds], axis=-1)
        vol_stats(vol, f"{r.PTID} {r.date}")
        save_png(vol, f"ADNI_{r.PTID}_{r.date}")
except ImportError:
    say("  pydicom not installed -> skipped")

# ============================================================ OASIS-2
say("\n" + "=" * 70, "\nOASIS-2\n" + "=" * 70)
o = pd.read_csv(out / "manifest_oasis2.csv")
say(f"subjects: {o['Subject ID'].nunique()} | sessions: {len(o)}")
if o["MRI ID"].duplicated().any(): bad("duplicate MRI ID")
say("visits per subject:", o.groupby("Subject ID").size().value_counts().sort_index().to_dict())
say("class balance (subjects):", o.drop_duplicates("Subject ID").Group.value_counts().to_dict())
say("missing values (%):", (o[["MMSE", "CDR", "SES", "eTIV", "nWBV"]].isna().mean() * 100).round(1).to_dict())
shapes = set()
try:
    import nibabel as nib
    for i, r in enumerate(o.itertuples()):
        p = out / r.subset_path
        if not p.exists(): bad(f"missing image {r.subset_path}"); continue
        if i % max(len(o) // 6, 1) == 0 or len(o) <= 8:   # full load only for a sample
            im = nib.load(str(p)); v = np.asarray(im.dataobj).squeeze()
            shapes.add(v.shape); vol_stats(v, r.subset_path); save_png(v, "OAS2_" + Path(r.subset_path).stem)
        else:
            shapes.add(nib.load(str(p)).shape)       # header only
    say("distinct shapes:", shapes)
    if len(shapes) > 1: bad("OASIS-2 images do not share one shape")
except ImportError:
    say("  nibabel not installed -> skipped")

# ============================================================ verdict
size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
say(f"\nTotal size of {out.name}: {size:.0f} MB")
say("VERDICT:", "ALL CHECKS PASSED" if not problems else f"{len(problems)} PROBLEM(S) - see [PROBLEM] lines above")
(out / "check_report.txt").write_text("\n".join(rep), encoding="utf-8")
