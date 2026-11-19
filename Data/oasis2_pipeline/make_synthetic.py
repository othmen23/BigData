"""Create FAKE OASIS-2-like data (demographics xlsx + tiny scans) to test the pipeline.
The numbers are random: results on this data mean NOTHING. Delete it when real data is there.
Usage: OASIS2_DATA_DIR=data_synthetic/oasis2 python make_synthetic.py
"""
import numpy as np
import pandas as pd
import nibabel as nib
import config

rng = np.random.default_rng(1)
rows = []
for i in range(1, 61):
    grp = rng.choice(["Nondemented", "Demented", "Converted"], p=[0.5, 0.3, 0.2])
    nv = rng.integers(3, 6)
    age0, mmse, cdr, nwbv = rng.integers(62, 88), 29.0, 0.0, 0.75
    delay = 0
    for v in range(1, nv + 1):
        if grp == "Demented":
            cdr_v = 0.5 if v < 3 else 1.0
        elif grp == "Converted":
            cdr_v = 0.0 if v < 3 else 0.5
        else:
            cdr_v = 0.0 if rng.random() > 0.15 else 0.5
        mmse_v = np.clip(mmse - (0 if cdr_v == 0 else 4 * cdr_v + rng.normal(0, 1)), 8, 30)
        rows.append({"Subject ID": f"OAS2_{i:04d}", "MRI ID": f"OAS2_{i:04d}_MR{v}", "Group": grp,
                     "Visit": v, "MR Delay": delay, "M/F": rng.choice(["M", "F"]), "Hand": "R",
                     "Age": age0 + delay // 365, "EDUC": rng.integers(8, 23),
                     "SES": np.nan if rng.random() < 0.08 else rng.integers(1, 6),
                     "MMSE": np.nan if rng.random() < 0.02 else round(mmse_v),
                     "CDR": cdr_v, "eTIV": rng.integers(1200, 1900),
                     "nWBV": round(nwbv - 0.01 * v - 0.015 * cdr_v + rng.normal(0, 0.01), 3),
                     "ASF": round(rng.uniform(0.9, 1.4), 3)})
        delay += int(rng.integers(300, 800))
df = pd.DataFrame(rows)
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_excel(config.DATA_DIR / "oasis_longitudinal_demographics-synthetic.xlsx", index=False)

for mri_id in df["MRI ID"]:
    folder = config.RAW_DIR / "OAS2_RAW_PART1" / mri_id / "RAW"
    folder.mkdir(parents=True, exist_ok=True)
    for k in (1, 2):
        vol = rng.normal(100, 20, (48, 48, 24, 1)).clip(0).astype(np.float32)
        nib.save(nib.AnalyzeImage(vol, np.eye(4)), str(folder / f"mpr-{k}.nifti.img"))
print(f"Synthetic data: {df['Subject ID'].nunique()} subjects, {len(df)} sessions in {config.DATA_DIR}")
