"""Step 3: index the raw MRI scans and match them to the demographics sessions.

Expected layout (verify on your machine, adapt SCAN_PATTERNS if different):
  data/oasis2/raw/**/OAS2_0001_MR1/RAW/mpr-1.nifti.img (+ .hdr)
"""
import re
import pandas as pd
import config

SCAN_PATTERNS = ["*mpr*.img", "*mpr*.nii", "*mpr*.nii.gz"]
SESSION_RE = re.compile(r"(OAS2_\d{4}_MR\d+)")


def main():
    if not config.RAW_DIR.exists():
        raise FileNotFoundError(f"{config.RAW_DIR} not found. Extract OAS2_RAW_PART1/2 there.")
    scans = {}
    for pat in SCAN_PATTERNS:
        for p in config.RAW_DIR.rglob(pat):
            m = SESSION_RE.search(str(p))
            if m:
                scans.setdefault(m.group(1), set()).add(str(p))
    idx = pd.DataFrame([{"mri_id": k, "n_scans": len(v), "first_scan": sorted(v)[0]}
                        for k, v in scans.items()])
    visits = pd.read_csv(config.VISITS_CSV)[["subject_id", "mri_id", "visit", "group"]]
    m = visits.merge(idx, on="mri_id", how="outer", indicator=True)
    m.to_csv(config.MRI_INDEX_CSV, index=False)

    both = (m["_merge"] == "both").sum()
    print(f"Sessions in demographics: {len(visits)}")
    print(f"Sessions with a scan:     {both}")
    print(f"Sessions WITHOUT a scan:  {(m['_merge'] == 'left_only').sum()}  (download the other RAW part?)")
    print(f"Scans without demographics row: {(m['_merge'] == 'right_only').sum()}")
    print(f"Saved {config.MRI_INDEX_CSV}")


if __name__ == "__main__":
    main()
