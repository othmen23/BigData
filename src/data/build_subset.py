"""
build_subset.py - build a small, clean, reproducible subset of ADNI + OASIS-2.

Never modifies or deletes your original Data folder: it COPIES the selection to Data_small/.
Once check_subset.py passes, you can archive/delete the originals.

Usage (Windows):
    pip install pandas openpyxl nibabel numpy
    python build_subset.py --n 10          # mini set: 10 subjects per cohort (+2 single-visit)
    python build_subset.py --n 0           # everything available (all 64 ADNI + 82 OASIS-2),
                                           # but with 1 image per visit, converted/compressed
Options: --root, --out, --min-visits 2, --extra-single 2, --seed 42, --no-convert
"""
import argparse, random, shutil
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ap.add_argument("--root", default=PROJECT_ROOT / "Data", type=Path)
ap.add_argument("--out", default=None)
ap.add_argument("--n", type=int, default=10, help="subjects per cohort with >= min-visits (0 = all)")
ap.add_argument("--min-visits", type=int, default=2)
ap.add_argument("--extra-single", type=int, default=2, help="extra 1-visit subjects (tests RQ3 missing visit)")
ap.add_argument("--min-slices", type=int, default=100, help="ignore DICOM series with fewer slices (empty/incomplete downloads)")
ap.add_argument("--include-single", action="store_true", help="with --n 0: also keep subjects with a single visit")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--no-convert", action="store_true", help="OASIS-2: copy .img/.hdr instead of .nii.gz")
a = ap.parse_args()

root = Path(a.root)
out = Path(a.out) if a.out else root.parent / "Data_small"
out.mkdir(parents=True, exist_ok=True)
log = []


def say(*x):
    s = " ".join(str(i) for i in x)
    print(s); log.append(s)


def stratified(df, key, n, seed):
    """Round-robin pick across strata so every class is represented."""
    if n <= 0 or len(df) <= n:
        return df
    rng = random.Random(seed)
    pools = {k: g.index.tolist() for k, g in df.groupby(key)}
    for v in pools.values():
        rng.shuffle(v)
    chosen = []
    while len(chosen) < n and any(pools.values()):
        for k in list(pools):
            if pools[k] and len(chosen) < n:
                chosen.append(pools[k].pop())
    return df.loc[chosen]


def pick_cohort(df, key, n, min_visits, extra_single, seed):
    multi = df[df.n_visits >= min_visits]
    single = df[df.n_visits < min_visits]
    if n <= 0:
        return df if a.include_single else multi
    sel = stratified(multi, key, n, seed)
    if extra_single > 0 and n > 0 and len(single):
        sel = pd.concat([sel, single.sample(min(extra_single, len(single)), random_state=seed)])
    return sel


# =============================================================== ADNI
say("=== ADNI ===")
img_root = root / "T1w_Imaging_Cohort_MRI_1" / "ADNI"
rows = []
for subj in sorted(p for p in img_root.iterdir() if p.is_dir()):
    for seq in subj.iterdir():
        for dt in seq.iterdir():
            for uid in dt.iterdir():
                rows.append(dict(PTID=subj.name, seq=seq.name, date=dt.name[:10], path=uid,
                                 n_dcm=len(list(uid.glob("*.dcm")))))
S = pd.DataFrame(rows)
bad_series = S[S.n_dcm < a.min_slices]
for r in bad_series.itertuples():
    say(f"  skipped incomplete series: {r.PTID} {r.date} ({r.n_dcm} dcm files)")
S = S[S.n_dcm >= a.min_slices].copy()
S["is_repeat"] = S.seq.str.contains("repeat|repet|repea", case=False)
# one series per (subject, date): prefer non-repeat, then the one with most slices
S = S.sort_values(["PTID", "date", "is_repeat", "n_dcm"], ascending=[True, True, True, False])
best = S.drop_duplicates(["PTID", "date"]).copy()
say(f"{S.PTID.nunique()} subjects, {len(S)} series -> {len(best)} visits after keeping 1 series/visit")

tabs = root / "ADNI_Tables"
dx = pd.read_csv(next(tabs.glob("*DXSUM*.csv")), usecols=["PTID", "EXAMDATE", "DIAGNOSIS"])
dx = dx[dx.DIAGNOSIS.isin([1, 2, 3])].sort_values("EXAMDATE")   # 1=CN 2=MCI 3=AD (check DATADIC)
dx["dx"] = dx.DIAGNOSIS.map({1: "CN", 2: "MCI", 3: "AD"})
g = dx.groupby("PTID").dx
subj = pd.DataFrame({"baseline_dx": g.first(), "last_dx": g.last(), "n_dx_visits": g.size()})
rank = {"CN": 0, "MCI": 1, "AD": 2}
subj["progressed"] = subj.last_dx.map(rank) > subj.baseline_dx.map(rank)
subj["n_visits"] = best.groupby("PTID").date.nunique()
subj = subj.dropna(subset=["n_visits"])
subj["n_visits"] = subj.n_visits.astype(int)
subj["stratum"] = subj.baseline_dx + np.where(subj.progressed, "_progressor", "")
subj = subj.reset_index()
say("ADNI subjects with images + diagnosis:", len(subj))
say(subj.stratum.value_counts().to_string())

sel = pick_cohort(subj, "stratum", a.n, a.min_visits, a.extra_single, a.seed)
adni_ids = set(sel.PTID)
say(f"-> selected {len(adni_ids)} ADNI subjects")

# copy 1 series per selected visit
man = best[best.PTID.isin(adni_ids)].merge(sel[["PTID", "baseline_dx", "last_dx", "progressed", "stratum"]], on="PTID")
dest_rel = []
for r in man.itertuples():
    dst = out / "ADNI_images" / r.PTID / r.date
    shutil.copytree(r.path, dst, dirs_exist_ok=True)
    dest_rel.append(str(dst.relative_to(out)))
man["subset_path"] = dest_rel
man.drop(columns=["path"]).to_csv(out / "manifest_adni.csv", index=False)

# filter every table to the selected subjects (chunked => low RAM)
(out / "ADNI_Tables").mkdir(exist_ok=True)
tfiles = list(tabs.glob("*.csv")) + list((root / "PET+MRI_Images").glob("*.csv"))
for f in tfiles:
    if "DATADIC" in f.name:
        continue
    kept = []
    for ch in pd.read_csv(f, dtype=str, keep_default_na=False, chunksize=100000):
        col = "PTID" if "PTID" in ch.columns else ("subject_id" if "subject_id" in ch.columns else None)
        if col:
            kept.append(ch[ch[col].isin(adni_ids)])
    if kept:
        d = pd.concat(kept)
        d.to_csv(out / "ADNI_Tables" / f.name, index=False)
        say(f"  {f.name}: {len(d)} rows kept")
# keep one data dictionary (small enough to be useful, needed to decode columns)
dd = next(tabs.glob("DATADIC*.csv"), None)
if dd:
    shutil.copy(dd, out / "ADNI_Tables" / dd.name)

# =============================================================== OASIS-2
say("\n=== OASIS-2 ===")
xl = next(root.glob("oasis_longitudinal_demographics*.xlsx"))
demo = pd.read_excel(xl)
base = root / "OAS2_RAW_PART1" / "OAS2_RAW_PART1"
have = {d.name for d in base.iterdir() if d.is_dir() and (d / "RAW" / "mpr-1.nifti.img").exists()}
demo = demo[demo["MRI ID"].isin(have)]
os_subj = demo.groupby("Subject ID").agg(n_visits=("MRI ID", "nunique"), group=("Group", "first")).reset_index()
say(f"{len(os_subj)} subjects / {len(demo)} sessions available on disk (demographics has 373 sessions)")
say(os_subj.group.value_counts().to_string())
osel = pick_cohort(os_subj, "group", a.n, a.min_visits, a.extra_single, a.seed)
oas_ids = set(osel["Subject ID"])
say(f"-> selected {len(oas_ids)} OASIS-2 subjects")

dsub = demo[demo["Subject ID"].isin(oas_ids)].copy()
(out / "OASIS2").mkdir(exist_ok=True)
paths = []
if not a.no_convert:
    import nibabel as nib
for mri in dsub["MRI ID"]:
    src = base / mri / "RAW" / "mpr-1.nifti.img"
    if a.no_convert:
        for ext in (".img", ".hdr"):
            shutil.copy(src.with_suffix(ext), out / "OASIS2" / f"{mri}_mpr-1{ext}")
        paths.append(f"OASIS2/{mri}_mpr-1.img")
    else:
        im = nib.load(str(src))
        arr = np.asarray(im.dataobj).squeeze().astype(np.float32)
        dst = out / "OASIS2" / f"{mri}_mpr-1.nii.gz"
        nib.save(nib.Nifti1Image(arr, im.affine), str(dst))
        paths.append(str(dst.relative_to(out)).replace("\\", "/"))
dsub["subset_path"] = paths
dsub.to_csv(out / "manifest_oasis2.csv", index=False)

# =============================================================== summary
size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
say(f"\nDONE. Output: {out}  total size: {size/1e6:.0f} MB")
(out / "build_log.txt").write_text("\n".join(log), encoding="utf-8")
