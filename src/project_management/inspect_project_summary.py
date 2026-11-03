
from pathlib import Path
from collections import Counter
import csv

ROOT = Path(__file__).resolve().parents[2]

EXCLUDED = {
    "Data",
    "Data_small",
    ".git",
    "__pycache__",
    ".venv",
    "venv"
}

rows = []
top_level = Counter()

for path in ROOT.iterdir():
    if path.is_dir():
        if path.name not in EXCLUDED:
            files = list(path.rglob("*"))
            file_count = sum(p.is_file() for p in files)
            rows.append({
                "type": "folder",
                "name": path.name,
                "files": file_count,
                "size_mb": round(
                    sum(p.stat().st_size for p in files if p.is_file())
                    / (1024 ** 2), 2
                )
            })

    elif path.is_file():
        rows.append({
            "type": "file",
            "name": path.name,
            "files": 1,
            "size_mb": round(path.stat().st_size / (1024 ** 2), 2)
        })

output = ROOT / "project_root_summary.csv"

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f, fieldnames=["type", "name", "files", "size_mb"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nPROJECT ROOT CONTENTS")
print("-" * 60)

for row in sorted(rows, key=lambda x: (x["type"], x["name"].lower())):
    print(
        f'{row["type"]:8} | {row["name"]:45} | '
        f'{row["files"]:6} files | {row["size_mb"]:10} MB'
    )

print(f"\nReport created: {output}")
