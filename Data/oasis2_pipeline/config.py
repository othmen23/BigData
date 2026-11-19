"""Paths and constants. Override with env vars OASIS2_DATA_DIR / OASIS2_OUT_DIR if needed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("OASIS2_DATA_DIR", ROOT / "data" / "oasis2"))
RAW_DIR = DATA_DIR / "raw"            # extract OAS2_RAW_PART1 and PART2 here
OUT_DIR = Path(os.environ.get("OASIS2_OUT_DIR", ROOT / "outputs"))
MRI_DIR = OUT_DIR / "mri_npy"
VISITS_CSV = OUT_DIR / "visits_clean.csv"
MRI_INDEX_CSV = OUT_DIR / "mri_index.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_demographics() -> Path:
    """Locate the demographics file (xlsx or csv) inside DATA_DIR."""
    cands = []
    for pat in ("*demographics*.xlsx", "*demographics*.xls", "*demographics*.csv",
                "oasis_longitudinal*.csv", "oasis_longitudinal*.xlsx"):
        cands += sorted(DATA_DIR.glob(pat))
    if not cands:
        raise FileNotFoundError(f"No demographics file found in {DATA_DIR}. "
                                "Put oasis_longitudinal_demographics*.xlsx there.")
    return cands[0]
