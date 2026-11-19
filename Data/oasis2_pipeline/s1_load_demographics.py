"""Step 1: load the OASIS-2 demographics file, standardise columns, validate, save a clean CSV."""
import re
import pandas as pd
import config

COLMAP = {"subjectid": "subject_id", "mriid": "mri_id", "group": "group", "visit": "visit",
          "mrdelay": "mr_delay_days", "mf": "sex", "hand": "hand", "age": "age", "educ": "educ",
          "ses": "ses", "mmse": "mmse", "cdr": "cdr", "etiv": "etiv", "nwbv": "nwbv", "asf": "asf"}
REQUIRED = ["subject_id", "mri_id", "group", "visit", "mr_delay_days", "sex", "age",
            "educ", "ses", "mmse", "cdr", "etiv", "nwbv"]
NUMERIC = ["visit", "mr_delay_days", "age", "educ", "ses", "mmse", "cdr", "etiv", "nwbv", "asf"]


def main():
    path = config.find_demographics()
    print(f"Reading {path.name}")
    df = pd.read_excel(path) if path.suffix in (".xlsx", ".xls") else pd.read_csv(path)
    df = df.rename(columns={c: COLMAP[re.sub(r"[^a-z0-9]", "", str(c).lower())]
                            for c in df.columns if re.sub(r"[^a-z0-9]", "", str(c).lower()) in COLMAP})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Columns found: {list(df.columns)}")

    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["sex"] = df["sex"].astype(str).str.strip().str.upper().map({"M": 0, "F": 1})
    df["group"] = df["group"].astype(str).str.strip()
    df = df.sort_values(["subject_id", "visit"]).reset_index(drop=True)
    df["years_since_baseline"] = df["mr_delay_days"] / 365.25

    # ---- validation ----
    assert not df.duplicated(["subject_id", "visit"]).any(), "duplicate (subject, visit) rows"
    bad_cdr = set(df["cdr"].dropna().unique()) - {0.0, 0.5, 1.0, 2.0, 3.0}
    if bad_cdr:
        print("WARNING unexpected CDR values:", bad_cdr)
    nonmono = df.groupby("subject_id")["mr_delay_days"].apply(lambda s: not s.is_monotonic_increasing)
    if nonmono.any():
        print("WARNING visits not ordered by time for:", list(nonmono[nonmono].index[:5]))

    df.to_csv(config.VISITS_CSV, index=False)

    # ---- summary ----
    n_sub = df["subject_id"].nunique()
    print(f"\nSubjects: {n_sub} | sessions: {len(df)}")
    print("Visits per subject:", df.groupby("subject_id").size().value_counts().sort_index().to_dict())
    print("Groups (subjects):", df.groupby("subject_id")["group"].first().value_counts().to_dict())
    print("Missing values:", df[["ses", "mmse", "cdr", "nwbv", "etiv"]].isna().sum().to_dict())
    print(f"\nSaved {config.VISITS_CSV}")


if __name__ == "__main__":
    main()
