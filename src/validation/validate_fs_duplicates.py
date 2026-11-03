
import pandas as pd
import re
from pathlib import Path

d = Path("Data_small")

manifest = pd.read_csv(d / "manifest_adni.csv")
images = pd.read_csv(
    d / "ADNI_Tables/All_Subjects_All_Images_13Oct2026.csv"
)
fs = pd.read_csv(
    d / "ADNI_Tables/All_Subjects_UCSFFSX7_13Oct2026.csv"
)

fs = fs.merge(
    images[["image_id", "subject_id", "study_id"]],
    left_on="IMAGEUID",
    right_on="image_id",
    how="left"
)

results = []

for _, row in manifest.iterrows():
    folder = d / row["subset_path"]

    ids = {
        int(match.group(1))
        for file in folder.glob("*.dcm")
        if (match := re.search(r"_I(\d+)\.dcm$", file.name))
    }

    original = images[
        (images["subject_id"] == row["PTID"]) &
        (images["image_id"].isin(ids))
    ]

    study_ids = original["study_id"].dropna().unique()

    candidates = fs[
        (fs["PTID"] == row["PTID"]) &
        (fs["study_id"].isin(study_ids))
    ]

    results.append({
        "PTID": row["PTID"],
        "date": row["date"],
        "original_studies": len(study_ids),
        "fs_candidates": len(candidates),
        "fs_IMAGEUIDs": ",".join(
            candidates["IMAGEUID"].astype(str)
        )
    })

result = pd.DataFrame(results)

print("Total visits:", len(result))
print("\nFreeSurfer candidate counts:")
print(result["fs_candidates"].value_counts().sort_index())

print("\nVisits with multiple candidates:")
print(result[result["fs_candidates"] > 1].to_string(index=False))

print("\nVisits without candidates:")
print(result[result["fs_candidates"] == 0].to_string(index=False))


matched = result[result["fs_candidates"] == 1].copy()

matched["fs_IMAGEUID"] = matched["fs_IMAGEUIDs"].astype(int)

details = matched.merge(
    fs,
    left_on="fs_IMAGEUID",
    right_on="IMAGEUID",
    how="left"
)

details = details.merge(
    images[["image_id", "image_description"]],
    left_on="IMAGEUID",
    right_on="image_id",
    how="left"
)

print("\nFreeSurfer processing descriptions:")
print(details["image_description"].value_counts(dropna=False).to_string())

print("\nFreeSurfer STATUS:")
print(details["STATUS"].value_counts(dropna=False).to_string())
