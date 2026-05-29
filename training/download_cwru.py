"""
Download CWRU bearing dataset .mat files.

Usage
-----
  python download_cwru.py
"""

import pathlib
import urllib.request

BASE_URL = "https://engineering.case.edu/sites/default/files"

FILES = {
    "97.mat":  "normal baseline (0 HP)",
    "105.mat": "inner-race fault 0.007\" (0 HP)",
    "118.mat": "ball fault 0.007\" (0 HP)",
    "130.mat": "outer-race fault 0.007\" @6 (0 HP)",
}

OUT_DIR = pathlib.Path("data/cwru")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://engineering.case.edu/bearingdatacenter/download-data-file",
}

for fname, desc in FILES.items():
    out = OUT_DIR / fname
    if out.exists():
        print(f"  skip  {fname}  (already exists)")
        continue
    url = f"{BASE_URL}/{fname}"
    print(f"  downloading {fname}  ({desc}) ...")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp, open(out, "wb") as f:
        f.write(resp.read())
    print(f"    saved → {out}  ({out.stat().st_size // 1024} KB)")

print("\nDone. Files in data/cwru/")
