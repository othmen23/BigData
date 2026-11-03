"""
inventory_data.py  -  Full inventory of the ADNI / OASIS data folder.

Reads ONLY metadata / headers / small previews (never full images), so it is fast
and light on memory even for 100+ GB.

Usage (Windows):
    pip install pandas openpyxl nibabel pydicom
    python inventory_data.py
    python inventory_data.py "C:\\other\\folder"

Outputs (in ./inventory_output):
    1_inventory_all_files.csv   one row per file (path, ext, size, folder...)
    2_summary_by_type.csv       count + size per top folder / extension
    3_folder_tree.txt           folders to depth 3 with file counts + sizes
    4_tables_report.txt         every CSV/XLSX: shape, columns, missing %, preview
    5_imaging_headers.csv       sample of image headers (shape, voxel size...)
    6_archives_report.txt       peek inside zip / tar files without extracting
    7_subjects_report.txt       subject IDs detected from file paths
    REPORT.txt                  everything important in one file -> send me this
"""
import os, sys, csv, re, time, zipfile, tarfile, collections
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "Data"
if len(sys.argv) > 1:
    ROOT = Path(sys.argv[1])

OUT = Path("inventory_output")
OUT.mkdir(exist_ok=True)

TABLE_EXT = {".csv", ".tsv", ".xlsx", ".xls"}
NIFTI_EXT = {".nii", ".nii.gz", ".img", ".hdr", ".mgz", ".mgh", ".mnc"}
DICOM_EXT = {".dcm", ".ima"}
ARCHIVE_EXT = {".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".7z", ".rar"}
IMG_SAMPLES_PER_GROUP = 5
TABLE_FULL_READ_MAX_MB = 200

try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import nibabel as nib
except ImportError:
    nib = None
try:
    import pydicom
except ImportError:
    pydicom = None


def get_ext(name):
    n = name.lower()
    for e in (".nii.gz", ".tar.gz"):
        if n.endswith(e):
            return e
    return os.path.splitext(n)[1] or "(none)"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def lp(p):
    """Long-path safe for Windows."""
    p = os.path.abspath(p)
    return "\\\\?\\" + p if os.name == "nt" and not p.startswith("\\\\?\\") else p


# ------------------------------------------------------------------ 1. walk
t0 = time.time()
root = Path(ROOT)
if not root.exists():
    sys.exit(f"Folder not found: {ROOT}")

rows = []
dir_stats = collections.defaultdict(lambda: [0, 0])  # rel dir -> [files, bytes]
print("Scanning files ...")
for dirpath, dirnames, filenames in os.walk(ROOT, onerror=lambda e: print("  skip:", e)):
    rel_dir = os.path.relpath(dirpath, ROOT)
    parts = [] if rel_dir == "." else rel_dir.split(os.sep)
    for fn in filenames:
        full = os.path.join(dirpath, fn)
        try:
            size = os.path.getsize(lp(full))
        except OSError:
            size = -1
        ext = get_ext(fn)
        top = parts[0] if parts else "(root)"
        rows.append({
            "top_folder": top, "rel_path": os.path.join(rel_dir, fn) if parts else fn,
            "filename": fn, "ext": ext, "size_bytes": size,
            "size_mb": round(size / 1e6, 3) if size >= 0 else -1, "depth": len(parts),
        })
        for d in range(0, min(len(parts), 3) + 1):
            key = os.sep.join(parts[:d]) or "(root)"
            dir_stats[key][0] += 1
            dir_stats[key][1] += max(size, 0)

with open(OUT / "1_inventory_all_files.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"  {len(rows):,} files found in {time.time()-t0:.0f}s")

# ------------------------------------------------------------------ 2. summary
agg = collections.defaultdict(lambda: [0, 0])
for r in rows:
    k = (r["top_folder"], r["ext"])
    agg[k][0] += 1
    agg[k][1] += max(r["size_bytes"], 0)
with open(OUT / "2_summary_by_type.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["top_folder", "extension", "n_files", "total_size", "total_bytes"])
    for (top, ext), (n, s) in sorted(agg.items(), key=lambda x: -x[1][1]):
        w.writerow([top, ext, n, human(s), s])

# ------------------------------------------------------------------ 3. tree
with open(OUT / "3_folder_tree.txt", "w", encoding="utf-8") as f:
    for k in sorted(dir_stats):
        n, s = dir_stats[k]
        depth = 0 if k == "(root)" else k.count(os.sep) + 1
        f.write(f"{'  '*depth}{k.split(os.sep)[-1]}  [{n:,} files, {human(s)}]\n")

# ------------------------------------------------------------------ 4. tables
tables = [r for r in rows if r["ext"] in TABLE_EXT]
lines = []
print(f"Reading {len(tables)} tabular files ...")
for r in tables:
    path = os.path.join(ROOT, r["rel_path"])
    lines.append("=" * 90)
    lines.append(f"{r['rel_path']}  ({human(r['size_bytes'])})")
    if pd is None:
        lines.append("  pandas not installed"); continue
    try:
        big = r["size_bytes"] > TABLE_FULL_READ_MAX_MB * 1e6
        if r["ext"] in (".csv", ".tsv"):
            sep = "\t" if r["ext"] == ".tsv" else None
            df = pd.read_csv(lp(path), sep=sep, engine="python" if sep is None else "c",
                             nrows=200000 if big else None, low_memory=False) if sep else \
                 pd.read_csv(lp(path), nrows=200000 if big else None, low_memory=False)
        else:
            df = pd.read_excel(lp(path))
        lines.append(f"  shape: {df.shape}{'  (truncated to 200k rows)' if big else ''}")
        lines.append(f"  columns ({df.shape[1]}):")
        miss = (df.isna().mean() * 100).round(1)
        for c in df.columns[:150]:
            ex = df[c].dropna().iloc[0] if df[c].notna().any() else ""
            lines.append(f"    - {c} | {df[c].dtype} | missing {miss[c]}% | e.g. {str(ex)[:40]}")
        if df.shape[1] > 150:
            lines.append(f"    ... {df.shape[1]-150} more columns")
        # candidate ID / diagnosis columns
        cand = [c for c in df.columns if re.search(r"(subject|rid|ptid|id$|dx|cdr|group|diag|visit|age)", str(c), re.I)]
        lines.append(f"  likely key columns: {cand[:25]}")
        for c in cand:
            if df[c].nunique() <= 12:
                lines.append(f"    value counts of '{c}': {df[c].value_counts(dropna=False).to_dict()}")
        lines.append("  first 3 rows:")
        lines.append(df.head(3).to_string(max_cols=12, max_colwidth=18))
    except Exception as e:
        lines.append(f"  ERROR reading: {e}")
(OUT / "4_tables_report.txt").write_text("\n".join(lines), encoding="utf-8")

# ------------------------------------------------------------------ 5. imaging
img_rows = []
groups = collections.defaultdict(list)
for r in rows:
    if r["ext"] in NIFTI_EXT or r["ext"] in DICOM_EXT:
        groups[(r["top_folder"], r["ext"])].append(r)
print(f"Reading headers of image samples from {len(groups)} groups ...")
for (top, ext), lst in groups.items():
    if ext == ".hdr":      # .img/.hdr come in pairs, read .img only
        continue
    step = max(len(lst) // IMG_SAMPLES_PER_GROUP, 1)
    for r in lst[::step][:IMG_SAMPLES_PER_GROUP]:
        path = os.path.join(ROOT, r["rel_path"])
        info = {"top_folder": top, "ext": ext, "rel_path": r["rel_path"],
                "size_mb": r["size_mb"], "n_files_in_group": len(lst)}
        try:
            if ext in DICOM_EXT and pydicom:
                ds = pydicom.dcmread(lp(path), stop_before_pixels=True)
                info.update(modality=getattr(ds, "Modality", ""),
                            series=getattr(ds, "SeriesDescription", ""),
                            patient=getattr(ds, "PatientID", ""),
                            date=getattr(ds, "StudyDate", ""),
                            rows=getattr(ds, "Rows", ""), cols=getattr(ds, "Columns", ""))
            elif nib:
                im = nib.load(path)           # header only, data is lazy
                h = im.header
                info.update(shape=im.shape, voxel_mm=tuple(round(float(z), 2) for z in h.get_zooms()),
                            dtype=str(h.get_data_dtype()), orientation="".join(nib.aff2axcodes(im.affine)))
            else:
                info["note"] = "nibabel/pydicom not installed"
        except Exception as e:
            info["error"] = str(e)[:120]
        img_rows.append(info)
if img_rows:
    keys = sorted({k for d in img_rows for k in d})
    with open(OUT / "5_imaging_headers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(img_rows)

# ------------------------------------------------------------------ 6. archives
arch = [r for r in rows if r["ext"] in {".zip", ".tar", ".tar.gz", ".tgz"}]
al = []
print(f"Peeking inside {len(arch)} archives ...")
for r in arch:
    path = os.path.join(ROOT, r["rel_path"])
    al.append("=" * 90); al.append(f"{r['rel_path']} ({human(r['size_bytes'])})")
    try:
        names = []
        if r["ext"] == ".zip":
            with zipfile.ZipFile(lp(path)) as z:
                names = z.namelist()
        else:
            with tarfile.open(lp(path)) as t:   # capped: listing a big .tar.gz is slow
                for i, m in enumerate(t):
                    names.append(m.name)
                    if i >= 5000:
                        names.append("... (listing capped at 5000 entries)"); break
        c = collections.Counter(get_ext(n) for n in names)
        al.append(f"  entries: {len(names):,} | types: {dict(c.most_common(8))}")
        al += [f"    {n}" for n in names[:8]]
    except Exception as e:
        al.append(f"  ERROR: {e}")
(OUT / "6_archives_report.txt").write_text("\n".join(al), encoding="utf-8")

# ------------------------------------------------------------------ 7. subjects
pat = {
    "OASIS-1/2 (OAS1_/OAS2_)": re.compile(r"OAS[12]_\d{4}(?:_MR\d+)?"),
    "OASIS-3 (OAS3xxxx / sub-OAS3)": re.compile(r"OAS3\d{4}(?:_\w+?_d\d{4})?"),
    "ADNI (000_S_0000)": re.compile(r"\d{3}_S_\d{4}"),
}
sl = []
for name, p in pat.items():
    ids = set(); sessions = set()
    for r in rows:
        for m in p.findall(r["rel_path"]):
            ids.add(re.match(r"OAS[123]_?\d{4}|\d{3}_S_\d{4}", m).group(0) if re.match(r"OAS[123]_?\d{4}|\d{3}_S_\d{4}", m) else m)
            sessions.add(m)
    sl.append(f"{name}: {len(ids):,} unique subjects, {len(sessions):,} unique subject/session IDs in paths")
(OUT / "7_subjects_report.txt").write_text("\n".join(sl), encoding="utf-8")

# ------------------------------------------------------------------ REPORT
rep = []
rep.append(f"ROOT: {ROOT}\nTotal files: {len(rows):,} | Total size: {human(sum(max(r['size_bytes'],0) for r in rows))}\n")
rep.append("--- LIBRARIES ---")
rep.append(f"pandas={'OK' if pd else 'MISSING'} nibabel={'OK' if nib else 'MISSING'} pydicom={'OK' if pydicom else 'MISSING'}\n")
rep.append("--- FOLDER TREE (depth 3) ---")
rep.append((OUT / "3_folder_tree.txt").read_text(encoding="utf-8"))
rep.append("--- SIZE BY FOLDER / EXTENSION (top 40) ---")
for (top, ext), (n, s) in sorted(agg.items(), key=lambda x: -x[1][1])[:40]:
    rep.append(f"{top:35s} {ext:9s} {n:>9,} files {human(s):>10s}")
rep.append("\n--- SUBJECTS ---"); rep += sl
rep.append("\n--- ARCHIVES ---"); rep += al[:80]
rep.append("\n--- IMAGE HEADER SAMPLES ---")
for d in img_rows[:40]:
    rep.append(str(d))
rep.append("\n--- TABLES (first 400 lines) ---")
rep += lines[:400]
(OUT / "REPORT.txt").write_text("\n".join(rep), encoding="utf-8")
print(f"\nDone in {time.time()-t0:.0f}s. Open: {OUT.resolve()}\\REPORT.txt")
