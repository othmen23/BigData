
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]

FILE = BASE / "results" / "analysis" / "assessment_progression_analysis.csv"

df = pd.read_csv(FILE)

df["assessment_date"] = pd.to_datetime(df["assessment_date"])
df["progress"] = pd.to_numeric(df["progress"], errors="coerce")

# Keep assessments with MRI within 180 days
df = df[df["MRI_gap_days"].between(0, 180)].copy()

# Separate subjects with positive and negative labels
subject_summary = (
    df.groupby("PTID")
    .agg(
        total_assessments=("PTID", "size"),
        positive_assessments=("progress", lambda x: (x == 1).sum()),
        negative_assessments=("progress", lambda x: (x == 0).sum()),
        censored_assessments=("progress", lambda x: x.isna().sum()),
        first_assessment=("assessment_date", "min"),
        last_assessment=("assessment_date", "max")
    )
    .reset_index()
)

subject_summary["has_positive"] = (
    subject_summary["positive_assessments"] > 0
)

subject_summary["has_negative"] = (
    subject_summary["negative_assessments"] > 0
)

subject_summary["has_both_labels"] = (
    subject_summary["has_positive"]
    & subject_summary["has_negative"]
)

print("=" * 60)
print("SUBJECT-LEVEL LABEL CONSISTENCY")
print("=" * 60)

print("\nNumber of subjects:", len(subject_summary))

print(
    "Subjects with positive labels:",
    subject_summary["has_positive"].sum()
)

print(
    "Subjects with negative labels:",
    subject_summary["has_negative"].sum()
)

print(
    "Subjects with both positive and negative labels:",
    subject_summary["has_both_labels"].sum()
)

print("\nASSESSMENTS PER SUBJECT:")
print(subject_summary["total_assessments"].describe())

print("\nSUBJECTS WITH BOTH LABELS:")
print(
    subject_summary.loc[
        subject_summary["has_both_labels"],
        [
            "PTID",
            "total_assessments",
            "positive_assessments",
            "negative_assessments",
            "censored_assessments"
        ]
    ].to_string(index=False)
)

print("\nLABEL COUNTS BY BASELINE DIAGNOSIS:")
print(
    df.groupby("DIAGNOSIS")["progress"]
    .value_counts(dropna=False)
    .unstack(fill_value=0)
)

output = BASE / "subject_label_consistency.csv"
subject_summary.to_csv(output, index=False)

print("\nSaved:", output)
