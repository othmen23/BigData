
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]

MATCH_FILE = BASE / "results" / "analysis" / "clinical_time_matching_analysis.csv"
DX_FILE = BASE / "Data_small" / "ADNI_Tables" / "All_Subjects_DXSUM_13Oct2026.csv"

HORIZON_DAYS = 36 * 30.4375
MIN_FOLLOWUP_DAYS = HORIZON_DAYS

# Load matched clinical assessments
matched = pd.read_csv(MATCH_FILE)
matched["assessment_date"] = pd.to_datetime(
    matched["assessment_date"], errors="coerce"
)
matched["MRI_date"] = pd.to_datetime(
    matched["MRI_date"], errors="coerce"
)
matched["DIAGNOSIS"] = pd.to_numeric(
    matched["DIAGNOSIS"], errors="coerce"
)

# Keep assessments with MRI within 90 or 180 days
matched = matched[
    matched["MRI_gap_days"].notna()
    & matched["MRI_gap_days"].between(0, 180)
].copy()

# Load all valid diagnosis records
dx = pd.read_csv(DX_FILE)
dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
dx["DIAGNOSIS"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")

dx = dx[
    dx["PTID"].notna()
    & dx["EXAMDATE"].notna()
    & dx["DIAGNOSIS"].isin([1, 2, 3])
].copy()

dx = dx.sort_values(["PTID", "EXAMDATE"])

def label_assessment(row):
    subject = row["PTID"]
    index_date = row["assessment_date"]
    baseline = int(row["DIAGNOSIS"])

    if baseline == 3:
        return pd.Series({
            "progress": None,
            "followup_days": None,
            "first_worsening_date": None
        })

    future = dx[
        (dx["PTID"] == subject)
        & (dx["EXAMDATE"] > index_date)
        & (dx["EXAMDATE"] <= index_date + pd.Timedelta(days=HORIZON_DAYS))
    ]

    baseline_rank = baseline
    worsening = future[future["DIAGNOSIS"] > baseline_rank]

    if not worsening.empty:
        first_date = worsening.iloc[0]["EXAMDATE"]

        return pd.Series({
            "progress": 1,
            "followup_days": (first_date - index_date).days,
            "first_worsening_date": first_date
        })

    all_future = dx[
        (dx["PTID"] == subject)
        & (dx["EXAMDATE"] > index_date)
    ]

    if all_future.empty:
        last_date = None
        followup = 0
    else:
        last_date = all_future["EXAMDATE"].max()
        followup = (last_date - index_date).days

    if followup >= MIN_FOLLOWUP_DAYS:
        label = 0
    else:
        label = None

    return pd.Series({
        "progress": label,
        "followup_days": followup,
        "first_worsening_date": None
    })

labels = matched.apply(label_assessment, axis=1)
result = pd.concat([matched, labels], axis=1)

result["matching_90"] = result["MRI_gap_days"].between(0, 90)
result["matching_180"] = result["MRI_gap_days"].between(0, 180)

print("=" * 60)
print("36-MONTH PROGRESSION LABEL ANALYSIS")
print("=" * 60)

for name, subset in [
    ("MRI WITHIN 90 DAYS", result[result["matching_90"]]),
    ("MRI WITHIN 180 DAYS", result[result["matching_180"]])
]:
    print(f"\n{name}")
    print("Total assessments:", len(subset))
    print("Unique subjects:", subset["PTID"].nunique())
    print("Positive:", (subset["progress"] == 1).sum())
    print("Negative:", (subset["progress"] == 0).sum())
    print("Censored:", subset["progress"].isna().sum())
    print(
        "Subjects with positive labels:",
        subset.loc[subset["progress"] == 1, "PTID"].nunique()
    )
    print(
        "Subjects with negative labels:",
        subset.loc[subset["progress"] == 0, "PTID"].nunique()
    )

print("\nBASELINE DIAGNOSIS DISTRIBUTION — 180 DAYS")
print(
    result["DIAGNOSIS"]
    .map({1: "CN", 2: "MCI", 3: "AD"})
    .value_counts()
)

output = BASE / "assessment_progression_analysis.csv"
result.to_csv(output, index=False)

print("\nSaved:", output)
