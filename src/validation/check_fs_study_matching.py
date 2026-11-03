
import pandas as pd

manifest = pd.read_csv("Data_small/manifest_adni.csv")
fs = pd.read_csv(
    "Data_small/ADNI_Tables/All_Subjects_UCSFFSX7_13Oct2026.csv",
    low_memory=False
)
images = pd.read_csv(
    "Data_small/ADNI_Tables/All_Subjects_All_Images_13Oct2026.csv",
    low_memory=False
)

fs["IMAGEUID"] = pd.to_numeric(fs["IMAGEUID"], errors="coerce")
images["image_id"] = pd.to_numeric(images["image_id"], errors="coerce")

fs = fs.merge(
    images[["image_id", "subject_id", "image_description", "study_id"]],
    left_on=["IMAGEUID", "PTID"],
    right_on=["image_id", "subject_id"],
    how="left"
)

fs["EXAMDATE"] = pd.to_datetime(fs["EXAMDATE"], errors="coerce")
manifest["date"] = pd.to_datetime(manifest["date"], errors="coerce")
images["image_date"] = pd.to_datetime(images["image_date"], errors="coerce")

results = []

for row in manifest.itertuples():
    original = images[
        (images["subject_id"] == row.PTID)
        & (images["image_date"] == row.date)
        & (images["image_type"] == "Original")
        & images["image_description"].str.contains(
            "MPR|MP-RAGE", case=False, na=False
        )
    ]

    candidates = fs[
        (fs["PTID"] == row.PTID)
        & (fs["EXAMDATE"] == row.date)
        & fs["image_description"].str.startswith(
            ("MPR;", "MPR-R;"), na=False
        )
    ]

    original_studies = set(original["study_id"].dropna())
    fs_studies = set(candidates["study_id"].dropna())

    results.append({
        "PTID": row.PTID,
        "date": row.date,
        "sequence": row.seq,
        "original_studies": sorted(original_studies),
        "FS_candidates": len(candidates),
        "matching_studies": len(original_studies & fs_studies)
    })

result = pd.DataFrame(results)

print("TOTAL VISITS:", len(result))
print("\nMATCHING STUDY COUNTS:")
print(result["matching_studies"].value_counts().sort_index())

print("\nVISITS WITH NO MATCHING STUDY OR MULTIPLE MATCHES:")
print(
    result[result["matching_studies"] != 1].to_string(index=False)
)

result.to_csv(
    "Data_small/tables/fs_study_matching_check.csv",
    index=False
)
print("\nReport saved: Data_small/tables/fs_study_matching_check.csv")
