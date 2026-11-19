"""Free disk space: remove mpr-2 and mpr-3 (and mpr-4) from OASIS-2 raw, keep mpr-1.
DRY RUN by default (only lists). Add --yes to really delete. Deleted files can be re-downloaded."""
import argparse
import re
import config

ap = argparse.ArgumentParser()
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()
victims = [p for p in config.RAW_DIR.rglob("*mpr-*")
           if re.search(r"mpr-([2-9])", p.name) and p.suffix in (".img", ".hdr", ".nifti", "")]
size = sum(p.stat().st_size for p in victims if p.is_file())
print(f"{len(victims)} files, {size / 1e9:.1f} GB would be removed")
if a.yes:
    for p in victims:
        p.unlink()
    print("deleted")
else:
    print("dry run: nothing deleted (use --yes to delete)")
