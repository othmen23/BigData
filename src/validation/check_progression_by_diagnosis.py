
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
FILE = BASE / "results" / "analysis" / "assessment_progression_analysis.csv"

df = pd.read_csv(FILE)

df["progress"] = pd.to_numeric(df["progress"], errors="coerce")
df["DIAGNOSIS"] = pd.to_numeric(df["DIAGNOSIS"], errors="coerce")

# Keep assessments with MRI within 180 days
df = df[df["MRI_gap_days"].between(0, 180)].copy()

diagnoses = {
    1: "CN",
    2: "MCI",
    3: "AD"
}

print("=" * 65)
print("PROGRESSION ANALYSIS BY BASELINE DIAGNOSIS")
print("=" * 65)

for diagnosis, name in diagnoses.items():
    subset = df[df["DIAGNOSIS"] == diagnosis]

    print(f"\n{name}")
    print("-" * 40)

    print("Assessments:", len(subset))
    print("Unique subjects:", subset["PTID"].nunique())

    for label, label_name in [(1, "Positive"), (0, "Negative")]:
        selected = subset[subset["progress"] == label]

        print(
            f"{label_name}: {len(selected)} assessments | "
            f"{selected['PTID'].nunique()} subjects"
        )

    print(
        "Censored:",
        subset["progress"].isna().sum()
    )

    # Number of subjects with repeated positive assessments
    positive = subset[subset["progress"] == 1]

    print(
        "Subjects with multiple positive assessments:",
        (positive.groupby("PTID").size() > 1).sum()
    )

print("\nOVERALL SUBJECT COUNTS BY DIAGNOSIS AND LABEL")

for diagnosis, name in diagnoses.items():
    subset = df[df["DIAGNOSIS"] == diagnosis]

    table = (
        subset.groupby("PTID")["progress"]
        .agg(
            positive=lambda x: (x == 1).any(),
            negative=lambda x: (x == 0).any()
        )
    )

    print(f"\n{name}")
    print("Subjects with positive:", table["positive"].sum())
    print("Subjects with negative:", table["negative"].sum())
    print(
        "Subjects with both:",
        (table["positive"] & table["negative"]).sum()
    )
