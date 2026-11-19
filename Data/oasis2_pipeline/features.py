"""Shared logic: loading clean visits and building the landmark dataset (pandas version)."""
import numpy as np
import pandas as pd

CLINICAL = ["age", "sex", "educ", "ses", "mmse", "cdr"]
IMAGING = ["nwbv", "etiv"]            # MRI-derived volumes provided by OASIS
STATIC_COLS = CLINICAL + IMAGING
TREND_VARS = ["mmse", "nwbv", "cdr"]

# Column order of the landmark table (the Spark version must match it)
LANDMARK_COLS = (["subject_id", "landmark"]
                 + [f"s_{c}" for c in STATIC_COLS]
                 + ["l_years", "l_n_visits"]
                 + [f"b_{c}" for c in TREND_VARS]
                 + [f"l_{c}_{k}" for c in TREND_VARS for k in ("delta", "slope")]
                 + ["future_years", "progressed", "group_converted"])


def load_visits(path) -> pd.DataFrame:
    return pd.read_csv(path).sort_values(["subject_id", "visit"]).reset_index(drop=True)


def build_landmark_pandas(visits: pd.DataFrame, L: int = 2, max_cdr: float = 0.5) -> pd.DataFrame:
    """One row per subject.

    Features use ONLY visits 1..L. The label looks ONLY at visits after L:
      progressed = 1 if the maximum future CDR is higher than the CDR at visit L.
    Cohort: subjects with at least L+1 visits and CDR <= max_cdr at visit L
    (so progression is still possible).
    """
    rows = []
    for sid, g in visits.sort_values(["subject_id", "visit"]).groupby("subject_id"):
        g = g.reset_index(drop=True)
        if len(g) < L + 1:
            continue
        cur, first, fut = g.iloc[L - 1], g.iloc[0], g.iloc[L:]
        if pd.isna(cur["cdr"]) or cur["cdr"] > max_cdr or fut["cdr"].isna().all():
            continue
        years = (cur["mr_delay_days"] - first["mr_delay_days"]) / 365.25
        r = {"subject_id": sid, "landmark": L}
        for c in STATIC_COLS:
            r[f"s_{c}"] = cur[c]
        r["l_years"] = years
        r["l_n_visits"] = L
        for c in TREND_VARS:
            r[f"b_{c}"] = first[c]
            if L >= 2 and years > 0:
                d = cur[c] - first[c]
                r[f"l_{c}_delta"], r[f"l_{c}_slope"] = d, d / years
            else:
                r[f"l_{c}_delta"] = r[f"l_{c}_slope"] = np.nan
        r["future_years"] = (fut["mr_delay_days"].max() - cur["mr_delay_days"]) / 365.25
        r["progressed"] = int(fut["cdr"].max() > cur["cdr"])
        r["group_converted"] = int(first["group"] == "Converted")
        rows.append(r)
    return pd.DataFrame(rows, columns=LANDMARK_COLS)
