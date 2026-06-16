"""Download the Fukuchi running (2017) and optionally walking (2018) data.

Running (default): pulls every 'RBDS###processed.txt' + 'RBDSinfo.txt' into
data/raw/ (~5 MB total).

Walking (--walking): pulls the Fukuchi 2018 treadmill-walking archive
(WBDSascii.zip, ~586 MB) + WBDSinfo.xlsx, and extracts only the per-trial angle
and kinetics files (walkT *ang.txt / *knt.txt) into data/raw/wbds/. This is the
gait mode that matches Mirai's walking/rehab cohort.

If a download fails, the pipeline still runs on the synthetic source:
pass --source synthetic to any stage script.

Usage:
    python scripts/download_data.py            # running only
    python scripts/download_data.py --walking  # also walking (large)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from achilles.config import DATA_RAW

ARTICLE_API = "https://api.figshare.com/v2/articles/4543435?page_size=1500"
WALKING_API = "https://api.figshare.com/v2/articles/5722711?page_size=2000"


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def main() -> int:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"querying Figshare article 4543435 ...")
    try:
        meta = _fetch_json(ARTICLE_API)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not reach Figshare ({e}).")
        print("Fall back to the synthetic source: add --source synthetic to a stage script.")
        return 1

    files = {f["name"]: f["download_url"] for f in meta["files"]}
    wanted = [n for n in files if n.endswith("processed.txt")] + ["RBDSinfo.txt"]
    print(f"downloading {len(wanted)} files to {DATA_RAW} ...")

    for i, name in enumerate(sorted(wanted), 1):
        dest = DATA_RAW / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            urllib.request.urlretrieve(files[name], dest)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: failed {name}: {e}")
            continue
        if i % 10 == 0:
            print(f"  {i}/{len(wanted)} ...")

    got = list(DATA_RAW.glob("RBDS*processed.txt"))
    print(f"done: {len(got)} processed files present, info file "
          f"{'present' if (DATA_RAW / 'RBDSinfo.txt').exists() else 'MISSING'}.")
    return 0


def download_walking() -> int:
    """Download the Fukuchi 2018 walking archive and extract angle/kinetics files."""
    wbds_dir = DATA_RAW / "wbds"
    wbds_dir.mkdir(parents=True, exist_ok=True)
    print("querying Figshare article 5722711 (walking) ...")
    try:
        meta = _fetch_json(WALKING_API)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not reach Figshare ({e}).")
        return 1
    files = {f["name"]: f for f in meta["files"]}

    info = DATA_RAW / "WBDSinfo.xlsx"
    if not info.exists():
        urllib.request.urlretrieve(files["WBDSinfo.xlsx"]["download_url"], info)

    zip_path = DATA_RAW / "WBDSascii.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 5e8:
        size_mb = files["WBDSascii.zip"]["size"] / 1e6
        print(f"downloading WBDSascii.zip ({size_mb:.0f} MB, this is the large one) ...")
        urllib.request.urlretrieve(files["WBDSascii.zip"]["download_url"], zip_path)

    print("extracting treadmill-walking angle/kinetics files ...")
    with zipfile.ZipFile(zip_path) as z:
        wanted = [n for n in z.namelist()
                  if "walkT" in n and (n.endswith("ang.txt") or n.endswith("knt.txt"))]
        for n in wanted:
            target = wbds_dir / Path(n).name
            if not target.exists():
                with z.open(n) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    n_ang = len(list(wbds_dir.glob("*ang.txt")))
    print(f"done: {n_ang} walking angle files in {wbds_dir}.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--walking", action="store_true",
                    help="also download the Fukuchi 2018 walking data (~586 MB)")
    args = ap.parse_args()
    rc = main()
    if args.walking:
        rc = download_walking() or rc
    sys.exit(rc)
