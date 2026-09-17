"""Find parquet files a killed process left truncated.

A parquet file is written header-first and footer-last. Kill the process
mid-write and you get a file with a valid PAR1 header, real data, and no
footer. It cannot be resumed and it cannot be read, so the analysis stage dies
with "ArrowInvalid: Parquet magic bytes not found in footer" and the run ends
with no metrics_by_substation.csv and no RUN_SUMMARY.md.

Checks the first and last four bytes only, so it is fast and it does not need
pyarrow.

    python tools/check_parquet_footers.py out/GOLDCR_8HB_LEXCEN/pen15_summer
    python tools/check_parquet_footers.py out --quiet
"""
from __future__ import annotations

import argparse
import os
import sys


def check(path):
    size = os.path.getsize(path)
    if size < 12:
        return f"only {size} bytes"
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
            fh.seek(-4, os.SEEK_END)
            tail = fh.read(4)
    except OSError as e:
        return f"unreadable: {e}"
    if head != b"PAR1":
        return f"bad header {head!r}"
    if tail != b"PAR1":
        return f"TRUNCATED, no footer (tail {tail!r})"
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="out",
                    help="directory to walk (default: out)")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the broken files")
    args = ap.parse_args()

    bad, ok = [], 0
    for dirpath, _, files in os.walk(args.root):
        for f in files:
            if not f.endswith(".parquet"):
                continue
            p = os.path.join(dirpath, f)
            why = check(p)
            if why:
                bad.append((p, why))
            else:
                ok += 1

    if not args.quiet:
        print(f"intact: {ok}")
    print(f"broken: {len(bad)}")
    for p, why in bad:
        print(f"  {p}\n      {why}")

    if bad:
        subs = sorted({os.path.basename(os.path.dirname(p)) for p, _ in bad})
        print(f"\nAffected substation(s): {', '.join(subs)}")
        print("Re-solve them from scratch. With run_penetration_sweep.sh:")
        print("  RETRY=1 RESTART=1 LEVELS=\"<level>\" FEEDERS=\"<feeder>\" "
              "bash run_penetration_sweep.sh")
        print("--restart only touches queued tasks, so completed substations "
              "are left alone.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
