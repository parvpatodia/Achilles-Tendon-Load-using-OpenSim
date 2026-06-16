"""Download the Fukuchi 2017 processed running data from Figshare.

Pulls every 'RBDS###processed.txt' file plus 'RBDSinfo.txt' (body mass) into
data/raw/. Each processed file is ~130 KB (≈5 MB total).

If this fails (no network / Figshare down), the pipeline still runs on the
synthetic source: pass --source synthetic to any stage script.

Usage:
    python scripts/download_data.py
"""
from __future__ import annotations

import json
import sys
import urllib.request

from achilles.config import DATA_RAW

ARTICLE_API = "https://api.figshare.com/v2/articles/4543435?page_size=1500"


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


if __name__ == "__main__":
    sys.exit(main())
