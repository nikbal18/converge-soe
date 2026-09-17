"""Which meter export is currently in build/timeseries/all_nmis.parquet?

There is only ONE timeseries cache. Preparing a second export overwrites the
first, and run_feeder.py silently falls back to the in-memory reader (which
then dies on a headerless export, or eats 12 GB on a headed one). Check before
every run that flips season or export.

    python tools/which_meter_cached.py
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def fingerprint(*paths):
    h = hashlib.sha256()
    for p in map(Path, paths):
        h.update(p.name.encode())
        if p.exists():
            h.update(p.read_bytes())
    return "sha256:" + h.hexdigest()


def main():
    man = REPO / "build" / "manifest.json"
    if not man.exists():
        print("no build/manifest.json — nothing cached")
        return 1
    entry = json.loads(man.read_text()).get("timeseries")
    if not entry:
        print("manifest has no timeseries entry — nothing cached")
        return 1
    stored = entry["fingerprint"]
    print(f"cached fingerprint : {stored}")
    print(f"stamped            : {entry.get('updated', '?')}")
    hit = None
    for csv in sorted((REPO / "data" / "meter").glob("*.csv")):
        for va in ("kwh_per_interval", "kw", "w"):
            if fingerprint(csv) + f"|{va}" == stored:
                hit = (csv, va)
    print()
    if hit:
        print(f"CACHED FILE : {hit[0].relative_to(REPO)}   (values-are {hit[1]})")
        print("Runs using this --meter will print 'stage (2) ... cached'.")
        print("Any OTHER --meter needs tools/prepare_wide_export.py first.")
    else:
        print("No file in data/meter matches. The cache was built from something")
        print("else, or a file has changed since. Re-prepare before running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
