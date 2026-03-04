"""Verify MIMIC data files exist and are readable."""
import os
import gzip
from config import FILES, MIMIC_BASE_PATH

def verify_mimic_data():
    """Check all required files exist and can be opened."""
    print(f"MIMIC Base Path: {MIMIC_BASE_PATH}")
    print(f"Path exists: {os.path.exists(MIMIC_BASE_PATH)}")
    print()

    results = {}
    for name, path in FILES.items():
        exists = os.path.exists(path)
        size_mb = 0
        readable = False

        if exists:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            try:
                with gzip.open(path, 'rt') as f:
                    header = f.readline()
                    readable = True
            except Exception as e:
                readable = False

        results[name] = {
            "exists": exists,
            "size_mb": round(size_mb, 2),
            "readable": readable,
            "path": path
        }

        status = "+" if exists and readable else "X"
        print(f"{status} {name}: {size_mb:.1f} MB - {path}")

    return results

if __name__ == "__main__":
    verify_mimic_data()
