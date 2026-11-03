
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
FILE = BASE / "results" / "analysis" / "assessment_progression_analysis.csv"

df = pd.read_csv(FILE)

df["assessment_date"] = pd.to_datetime(df["assessment_date"])
df["MRI_date"] = pd.to_datetime(df["MRI_date"])
df["progress"] = pd.to_numeric(df["progress"], errors="coerce")

# Keep assessments with MRI within 180 days
df = df[df["MRI_gap_days"].between(0, 180)].copy()

print("=" * 65)
print("MRI REUSE AND ASSESSMENT SPACING")
print("=" * 65)

# Identify unique MRI scans
df["MRI_key"] = (
    df["PTID"].astype(str)
    + "_"
    + df["MRI_date"].dt.strftime("%Y-%m-%d")
)

reuse = (
    df.groupby("MRI_key")
    .agg(
        subject=("PTID", "first"),
        MRI_date=("MRI_date", "first"),
        assessment_count=("assessment_date", "size"),
        first_assessment=("assessment_date", "min"),
        last_assessment=("assessment_date", "max")
    )
    .reset_index()
)

print("\nMRI REUSE")
print("Matched assessments:", len(df))
print("Unique MRI scans used:", reuse["MRI_key"].nunique())
print("MRI scans used more than once:", (reuse["assessment_count"] > 1).sum())
print("Maximum assessments using one MRI:", reuse["assessment_count"].max())

print("\nASSESSMENTS PER MRI SCAN:")
print(reuse["assessment_count"].value_counts().sort_index())

print("\nMRI-TO-ASSESSMENT GAP (DAYS):")
print(df["MRI_gap_days"].describe())

# Assessment spacing within each subject
df = df.sort_values(["PTID", "assessment_date"])
df["previous_assessment"] = (
    df.groupby("PTID")["assessment_date"].shift(1)
)

df["assessment_spacing_days"] = (
    df["assessment_date"] - df["previous_assessment"]
).dt.days

print("\nASSESSMENT SPACING (DAYS):")
print(df["assessment_spacing_days"].describe())

# Check whether multiple assessments from the same subject
# have overlapping 36-month prediction windows
overlap_subjects = []

for subject, group in df.groupby("PTID"):
    dates = sorted(group["assessment_date"].dropna())

    overlaps = 0

    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days < 36 * 30:
            overlaps += 1

    if overlaps > 0:
        overlap_subjects.append((subject, overlaps))

print("\nSUBJECTS WITH OVERLAPPING 36-MONTH WINDOWS:")
print("Number of subjects:", len(overlap_subjects))
print(overlap_subjects)

# Save MRI reuse report
output = BASE / "mri_reuse_analysis.csv"
reuse.to_csv(output, index=False)

print("\nSaved:", output)
