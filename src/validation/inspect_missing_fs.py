
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

missing = []

for row in manifest.itertuples():
    candidates = fs[
        (fs["PTID"] == row.PTID)
        & (fs["EXAMDATE"] == row.date)
    ]

    has_mpr = candidates["image_description"].str.startswith(
        ("MPR;", "MPR-R;"), na=False
    ).any()

    if not has_mpr:
        missing.append(row)

for row in missing:
    print("\nVISIT:", row.PTID, row.date.date(), "|", row.seq)

    candidates = fs[
        (fs["PTID"] == row.PTID)
        & (fs["EXAMDATE"] == row.date)
    ]

    print("\nFreeSurfer records:")
    print(
        candidates[
            ["IMAGEUID", "image_description", "study_id"]
        ].to_string(index=False)
    )

    original = images[
        (images["subject_id"] == row.PTID)
        & (images["image_date"] == row.date)
        & (images["image_type"] == "Original")
    ]

    print("\nOriginal acquisitions:")
    print(
        original[
            ["image_id", "image_description", "study_id"]
        ].to_string(index=False)
    )

print("\nTOTAL VISITS WITHOUT MPR:", len(missing))
