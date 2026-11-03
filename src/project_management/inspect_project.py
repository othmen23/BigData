
from pathlib import Path
from collections import Counter
import csv

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "project_inventory.csv"

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv"}

files = []
folders = []

for path in ROOT.rglob("*"):
    if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
        continue

    if path.is_dir():
        folders.append(path)
    elif path.is_file():
        files.append({
            "path": str(path.relative_to(ROOT)),
            "name": path.name,
            "extension": path.suffix.lower() or "[no extension]",
            "size_mb": round(path.stat().st_size / (1024 ** 2), 3)
        })

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["path", "name", "extension", "size_mb"]
    )
    writer.writeheader()
    writer.writerows(files)

print("\nPROJECT INVENTORY")
print("-" * 50)
print(f"Project root: {ROOT}")
print(f"Folders found: {len(folders)}")
print(f"Files found: {len(files)}")

extensions = Counter(item["extension"] for item in files)

print("\nFILES BY EXTENSION")
for ext, count in extensions.most_common():
    print(f"{ext}: {count}")

print("\nLARGEST FILES")
for item in sorted(files, key=lambda x: x["size_mb"], reverse=True)[:20]:
    print(f"{item['size_mb']:>10} MB | {item['path']}")

print(f"\nInventory saved to: {OUTPUT}")
