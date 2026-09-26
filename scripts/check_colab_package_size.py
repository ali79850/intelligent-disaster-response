"""
Phase 5: Check the total size of what needs to be packaged for Colab upload,
before committing to a zip/upload approach.
"""
from pathlib import Path

def dir_size_gb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1e9

def main():
    paths = {
        "images": Path("data/raw/xbd/train/images"),
        "masks": Path("data/processed/masks"),
        "splits": Path("data/interim/splits"),
    }
    total = 0.0
    for name, p in paths.items():
        size = dir_size_gb(p)
        total += size
        print(f"{name:10s}: {size:.2f} GB")
    print(f"{'TOTAL':10s}: {total:.2f} GB")

if __name__ == "__main__":
    main()