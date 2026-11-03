
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parents[2]
MRI_FILE = BASE / "Data_small" / "manifest_adni.csv"
DX_FILE = BASE / "Data_small" / "ADNI_Tables" / "All_Subjects_DXSUM_13Oct2026.csv"

# Load MRI visits
mri = pd.read_csv(MRI_FILE)
mri["scan_date"] = pd.to_datetime(mri["date"], errors="coerce")

# Load clinical diagnoses
dx = pd.read_csv(DX_FILE)
dx["dx_date"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
dx["DIAGNOSIS"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")

# Keep valid diagnosis records
dx = dx[
    dx["PTID"].notna()
    & dx["dx_date"].notna()
    & dx["DIAGNOSIS"].isin([1, 2, 3])
].copy()

dx = dx.sort_values(["PTID", "dx_date"])
mri = mri[mri["PTID"].notna() & mri["scan_date"].notna()].copy()

# Calculate diagnosis gaps for each MRI visit
results = []

for _, scan in mri.iterrows():
    subject = scan["PTID"]
    scan_date = scan["scan_date"]

    events = dx[dx["PTID"] == subject].sort_values("dx_date")

    previous = events[events["dx_date"] <= scan_date]
    future = events[events["dx_date"] > scan_date]

    previous_gap = (
        (scan_date - previous.iloc[-1]["dx_date"]).days
        if not previous.empty else np.nan
    )

    future_gap = (
        (future.iloc[0]["dx_date"] - scan_date).days
        if not future.empty else np.nan
    )

    nearest_gap = min(
        [x for x in [previous_gap, future_gap] if pd.notna(x)],
        default=np.nan
    )

    results.append({
        "PTID": subject,
        "scan_date": scan_date,
        "previous_gap_days": previous_gap,
        "future_gap_days": future_gap,
        "nearest_gap_days": nearest_gap
    })

check = pd.DataFrame(results)

# Compare matching windows
print("\n" + "=" * 60)
print("ADNI DIAGNOSIS MATCHING ANALYSIS")
print("=" * 60)

print(f"\nTOTAL MRI VISITS: {len(check)}")

print("\nMATCHING WINDOWS:")
for days in [90, 180, 270, 365, 540]:
    previous = check["previous_gap_days"].between(0, days).sum()
    nearest = check["nearest_gap_days"].between(0, days).sum()
    future = check["future_gap_days"].between(1, days).sum()

    print(
        f"{days:>3} days | "
        f"Previous: {previous:>3} | "
        f"Nearest: {nearest:>3} | "
        f"Future: {future:>3}"
    )

print("\nPREVIOUS DIAGNOSIS GAP:")
print(check["previous_gap_days"].describe())

print("\nFUTURE DIAGNOSIS GAP:")
print(check["future_gap_days"].describe())

print("\nFIRST 20 UNMATCHED VISITS (180 days):")
unmatched = check[
    ~check["previous_gap_days"].between(0, 180)
]
print(unmatched.head(20).to_string(index=False))

# Save detailed results
output = BASE / "diagnosis_matching_analysis.csv"
check.to_csv(output, index=False)

print(f"\nDetailed results saved to: {output}")
