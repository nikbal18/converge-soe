#!/usr/bin/env python3
"""
Stage ② standalone: normalise raw meter exports into the tidy long table.

    python scripts/prepare_timeseries.py --values-are kwh_per_interval
    python scripts/prepare_timeseries.py --meter data/meter/lexcen_data.csv \
           --values-are kwh_per_interval --fetch-ambient

Accepts one or many CSVs, wide (EvoEnergy Data_HH_MM) or long, auto-detected.
Units are NEVER guessed silently — pass --values-are for wide exports.
Output: build/timeseries/all_nmis.parquet.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from converge_soe import pipeline as pl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meter", nargs="*", default=None)
    ap.add_argument("--values-are", default=None,
                    choices=["w", "kw", "kwh_per_interval"])
    ap.add_argument("--fetch-ambient", action="store_true",
                    help="fetch + cache Open-Meteo ambient for the data range")
    args = ap.parse_args()

    overrides = {}
    if args.values_are:
        overrides["timeseries"] = {"values_are": args.values_are}
    cfg = pl.load_config(REPO, overrides=overrides)
    df = pl.stage_prepare_timeseries(REPO, cfg, meter_files=args.meter,
                                     log=print)
    print(df.describe(include="all").to_string())
    if args.fetch_ambient:
        pl.fetch_open_meteo(cfg, sorted(df["timestamp"].unique()),
                            REPO / "data" / "ambient", log=print)
    return 0


if __name__ == "__main__":
    sys.exit(main())
