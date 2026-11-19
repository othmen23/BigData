"""Inventory of what you downloaded (nothing is read or changed).
  python inspect_data.py --adni "Data/T1w_Imaging_Cohort_MRI_1" --oasis "Data/OAS2_RAW_PART1"
ADNI layout expected: <root>/ADNI/<PTID>/<sequence>/<scan date>/<Image ID>/*.dcm
"""
import argparse
import os
import re
from pathlib import Path
import pandas as pd
import config

PTID = re.compile(r"^\d{3}_S_\d{4}$")
DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})_")


def adni_inventory(root: Path) -> pd.DataFrame:
    rows = []
    for dirpath, _, files in os.walk(root):
        dcm = [f for f in files if f.lower().endswith(".dcm")]
        if not dcm:
            continue
        parts = Path(dirpath).relative_to(root).parts
        ptid = next((p for p in parts if PTID.match(p)), None)
        date = next((DATE.match(p).group(1) for p in parts if DATE.match(p)), None)
        img = next((p for p in parts if re.fullmatch(r"I\d+", p)), None)
        i = parts.index(ptid) if ptid else 0
        seq = parts[i + 1] if ptid and len(parts) > i + 1 else None
        size = sum(os.path.getsize(os.path.join(dirpath, f)) for f in dcm) / 1e6
        rows.append({"ptid": ptid, "rid": int(ptid.split("_")[-1]) if ptid else None, "sequence": seq,
                     "scan_date": date, "image_id": img, "n_slices": len(dcm),
                     "size_mb": round(size, 1), "folder": dirpath})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adni")
    ap.add_argument("--oasis")
    a = ap.parse_args()
    if a.adni:
        inv = adni_inventory(Path(a.adni))
        out = config.OUT_DIR / "adni_scan_inventory.csv"
        inv.to_csv(out, index=False)
        print(f"ADNI: {inv['ptid'].nunique()} subjects | {len(inv)} scans (series) | "
              f"{inv['size_mb'].sum() / 1000:.1f} GB")
        print("scans per subject:", inv.groupby("ptid").size().value_counts().sort_index().to_dict())
        print("sequences:", inv["sequence"].value_counts().to_dict())
        print("slices per scan (min/median/max):", inv["n_slices"].min(), inv["n_slices"].median(), inv["n_slices"].max())
        print("saved", out)
    if a.oasis:
        tot = mpr1 = 0
        sessions = set()
        for p in Path(a.oasis).rglob("*mpr-*"):
            s = p.stat().st_size
            tot += s
            mpr1 += s if "mpr-1" in p.name else 0
            m = re.search(r"OAS2_\d{4}_MR\d+", str(p))
            if m:
                sessions.add(m.group(0))
        print(f"OASIS-2: {len(sessions)} sessions | {tot / 1e9:.1f} GB | mpr-1 only = {mpr1 / 1e9:.1f} GB "
              f"(mpr-2/3 are {(tot - mpr1) / 1e9:.1f} GB you do not need for the baseline)")


if __name__ == "__main__":
    main()
