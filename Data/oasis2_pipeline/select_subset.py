"""Pick a SMALL subset of ADNI scans to convert.
  python select_subset.py --n-subjects 10  --out subset_dev.csv       # debugging set
  python select_subset.py --n-subjects 150 --labels labels.csv         # experiment set
labels.csv (optional): columns rid,label  -> the subset is balanced between labels.
Rules: only subjects with >= --min-scans different scan dates; one series per date; at most
--max-scans per subject, evenly spaced in time (first ... last)."""
import argparse
import numpy as np
import pandas as pd
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=str(config.OUT_DIR / "adni_scan_inventory.csv"))
    ap.add_argument("--n-subjects", type=int, default=100)
    ap.add_argument("--min-scans", type=int, default=2)
    ap.add_argument("--max-scans", type=int, default=3)
    ap.add_argument("--labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(config.OUT_DIR / "subset_scans.csv"))
    a = ap.parse_args()
    inv = pd.read_csv(a.inventory).dropna(subset=["rid", "scan_date"])
    inv = inv.sort_values(["rid", "scan_date", "n_slices"], ascending=[True, True, False])
    inv = inv.drop_duplicates(["rid", "scan_date"])                 # one series per date
    n = inv.groupby("rid")["scan_date"].transform("count")
    inv = inv[n >= a.min_scans]
    subjects = inv[["rid"]].drop_duplicates()
    rng = np.random.default_rng(a.seed)
    if a.labels:
        lab = pd.read_csv(a.labels)[["rid", "label"]].drop_duplicates("rid")
        subjects = subjects.merge(lab, on="rid")
        per = a.n_subjects // subjects["label"].nunique()
        chosen = pd.concat([g.sample(min(per, len(g)), random_state=a.seed) for _, g in subjects.groupby("label")])
    else:
        chosen = subjects.sample(min(a.n_subjects, len(subjects)), random_state=a.seed)
    out = []
    for rid in chosen["rid"]:
        g = inv[inv["rid"] == rid].reset_index(drop=True)
        k = min(a.max_scans, len(g))
        out.append(g.iloc[np.unique(np.linspace(0, len(g) - 1, k).round().astype(int))])
    sub = pd.concat(out)
    sub.to_csv(a.out, index=False)
    print(f"{sub['rid'].nunique()} subjects | {len(sub)} scans | ~{sub['size_mb'].sum() / 1000:.1f} GB raw to read "
          f"| ~{len(sub) * 96 ** 3 * 2 / 1e6:.0f} MB after conversion (96^3 float16)")
    print("saved", a.out)


if __name__ == "__main__":
    main()
