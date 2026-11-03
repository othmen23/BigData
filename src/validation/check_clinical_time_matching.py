
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]

MRI_FILE = BASE / "Data_small" / "manifest_adni.csv"
DX_FILE = BASE / "Data_small" / "ADNI_Tables" / "All_Subjects_DXSUM_13Oct2026.csv"

# Load MRI visits
mri = pd.read_csv(MRI_FILE)
mri["scan_date"] = pd.to_datetime(mri["date"], errors="coerce")
mri = mri.dropna(subset=["PTID", "scan_date"])

# Load clinical assessments
dx = pd.read_csv(DX_FILE)
dx["assessment_date"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
dx["DIAGNOSIS"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")

# Keep valid diagnosis assessments
dx = dx[
    dx["PTID"].notna()
    & dx["assessment_date"].notna()
    & dx["DIAGNOSIS"].isin([1, 2, 3])
].copy()

# Sort MRI records by subject and date
mri = mri.sort_values(["PTID", "scan_date"])

# Check duplicate clinical assessments
duplicates = dx.duplicated(
    subset=["PTID", "assessment_date"],
    keep=False
)

print("=" * 65)
print("CLINICAL-ASSESSMENT-TIME MRI MATCHING")
print("=" * 65)

print(f"\nValid clinical assessment records: {len(dx)}")
print(f"Unique subjects: {dx['PTID'].nunique()}")
print(f"Duplicate subject/date records: {duplicates.sum()}")

# Match each assessment to the closest MRI on or before its date
results = []

for _, assessment in dx.iterrows():
    subject = assessment["PTID"]
    assessment_date = assessment["assessment_date"]

    subject_mri = mri[mri["PTID"] == subject]
    eligible = subject_mri[
        subject_mri["scan_date"] <= assessment_date
    ]

    if eligible.empty:
        gap = None
        matched_scan_date = None
    else:
        best = eligible.iloc[-1]
        gap = (assessment_date - best["scan_date"]).days
        matched_scan_date = best["scan_date"]

    results.append({
        "PTID": subject,
        "assessment_date": assessment_date,
        "DIAGNOSIS": int(assessment["DIAGNOSIS"]),
        "MRI_date": matched_scan_date,
        "MRI_gap_days": gap
    })

check = pd.DataFrame(results)

print("\nTOTAL CLINICAL ASSESSMENTS:", len(check))
print("ASSESSMENTS WITH PREVIOUS MRI:", check["MRI_gap_days"].notna().sum())

# Compare matching windows
print("\nMATCHING WINDOWS:")

for days in [30, 60, 90, 180, 270, 365]:
    matched = check["MRI_gap_days"].between(0, days)
    visits = matched.sum()
    subjects = check.loc[matched, "PTID"].nunique()

    print(
        f"{days:>3} days: "
        f"{visits:>3} assessments | "
        f"{subjects:>3} unique subjects"
    )

# Diagnosis distribution
labels = {1: "CN", 2: "MCI", 3: "AD"}

print("\nDIAGNOSIS DISTRIBUTION — ALL ASSESSMENTS:")
print(
    check["DIAGNOSIS"]
    .map(labels)
    .value_counts()
    .to_string()
)

print("\nDIAGNOSIS DISTRIBUTION — ASSESSMENTS WITH MRI WITHIN 180 DAYS:")
matched_180 = check[check["MRI_gap_days"].between(0, 180)]

print(
    matched_180["DIAGNOSIS"]
    .map(labels)
    .value_counts()
    .to_string()
)

# Repeated assessments per subject
print("\nASSESSMENTS PER SUBJECT:")
print(check.groupby("PTID").size().describe())

# Save detailed results
output = BASE / "clinical_time_matching_analysis.csv"
check.to_csv(output, index=False)

print("\nDetailed results saved to:")
print(output)
