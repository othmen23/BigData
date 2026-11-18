import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="Data_small/tables_assessment/adni_visits.csv"
    )
    parser.add_argument(
        "--output",
        default="Data_small/tables_assessment/dl_manifest_adni.csv"
    )
    parser.add_argument(
        "--images",
        default="Data_small/preprocessed/adni"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    image_dir = Path(args.images)

    df = pd.read_csv(input_path)
    df["scan_date"] = pd.to_datetime(df["scan_date"])
    df["progress"] = pd.to_numeric(df["progress"], errors="coerce")
    df = df.sort_values(["PTID", "scan_date"])

    if df["PTID"].isna().any() or df["scan_date"].isna().any():
        raise ValueError("Missing subject IDs or scan dates.")

    if df.duplicated(["PTID", "scan_date"]).any():
        raise ValueError("Duplicate subject/date pairs detected.")

    labelled = df[df["progress"].notna()].copy()

    if not labelled["progress"].isin([0, 1]).all():
        raise ValueError("Progress labels must be binary (0 or 1).")

    targets = labelled.groupby("PTID", group_keys=False).tail(1)

    records = []

    for target in targets.itertuples(index=False):
        history = df[
            (df["PTID"] == target.PTID)
            & (df["scan_date"] <= target.scan_date)
        ].sort_values("scan_date")

        for index, visit in enumerate(history.itertuples(index=False), start=1):
            image_path = image_dir / (
                f"{visit.PTID}_{visit.scan_date.strftime('%Y-%m-%d')}.npy"
            )

            if not image_path.is_file():
                raise FileNotFoundError(f"MRI file not found: {image_path}")

            records.append({
                "subject_id": target.PTID,
                "target_date": target.scan_date.strftime("%Y-%m-%d"),
                "label": int(target.progress),
                "sequence_index": index,
                "sequence_length": len(history),
                "scan_date": visit.scan_date.strftime("%Y-%m-%d"),
                "image_path": str(image_path).replace("\\", "/")
            })

    manifest = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    print("Manifest created:", output_path)
    print("Subjects:", manifest["subject_id"].nunique())
    print("MRI visits:", len(manifest))
    print("Class distribution:")
    print(targets["progress"].value_counts().sort_index().to_string())
    print("Sequence lengths:")
    print(
        targets.assign(
            sequence_length=targets["PTID"].map(
                manifest.groupby("subject_id").size()
            )
        )["sequence_length"].value_counts().sort_index().to_string()
    )
    print("Missing MRI files: 0")
    print("\nFirst rows:")
    print(manifest.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
