
from pathlib import Path
import pandas as pd

D = Path(__file__).resolve().parents[2]
T = D / "Data_small" / "ADNI_Tables"

manifest = pd.read_csv(D / "Data_small" / "manifest_adni.csv")
images = pd.read_csv(
    next(T.glob("*All_Images*.csv")),
    low_memory=False
)
fs = pd.read_csv(
    next(T.glob("*UCSFFSX7*.csv")),
    low_memory=False
)

manifest["date"] = pd.to_datetime(manifest["date"])
images["image_date"] = pd.to_datetime(images["image_date"])
fs["EXAMDATE"] = pd.to_datetime(fs["EXAMDATE"])

images["image_id"] = pd.to_numeric(images["image_id"], errors="coerce")
fs["IMAGEUID"] = pd.to_numeric(fs["IMAGEUID"], errors="coerce")

# Match the selected original acquisition to the image inventory
selected = manifest.merge(
    images,
    left_on=["PTID", "date", "seq"],
    right_on=["subject_id", "image_date", "image_description"],
    how="left"
)

# Match FreeSurfer records to the inventory
fs_images = fs.merge(
    images[["image_id", "study_id", "image_description"]],
    left_on="IMAGEUID",
    right_on="image_id",
    how="left"
)

# Compare selected study IDs with FreeSurfer candidates
candidates = selected.merge(
    fs_images,
    left_on=["PTID", "date"],
    right_on=["PTID", "EXAMDATE"],
    how="left",
    suffixes=("_selected", "_fs")
)

candidates["study_match"] = (
    candidates["study_id_selected"].notna()
    & candidates["study_id_fs"].notna()
    & (candidates["study_id_selected"] == candidates["study_id_fs"])
)

summary = candidates.groupby(
    ["PTID", "date", "seq"],
    dropna=False
).agg(
    selected_study=("study_id_selected", "first"),
    fs_candidates=("IMAGEUID", "count"),
    matching_candidates=("study_match", "sum"),
    matched_imageuid=(
        "IMAGEUID",
        lambda x: ", ".join(x.dropna().astype(int).astype(str))
    )
).reset_index()

print("\nSUMMARY")
print("Manifest visits:", len(manifest))
print("Visits with selected study:", selected["study_id"].notna().sum())
print("Visits with exactly one FS study match:",
      (summary["matching_candidates"] == 1).sum())
print("Visits with multiple FS study matches:",
      (summary["matching_candidates"] > 1).sum())
print("Visits without FS study match:",
      (summary["matching_candidates"] == 0).sum())

print("\nVISITS NEEDING REVIEW")
print(
    summary[summary["matching_candidates"] != 1]
    .to_string(index=False)
)

summary.to_csv(D / "fs_acquisition_mapping_check.csv", index=False)
print("\nSaved: fs_acquisition_mapping_check.csv")
