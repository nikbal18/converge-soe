#!/usr/bin/env python3
"""
Translate an EvoEnergy wide meter export into the long-format timeseries the
DOE solver reads (timestamp, load_id, real_power_w, reactive_power_var).

Input (tab-separated), one row per NMI / day / meter channel:
    Nmi  IntervalDay  NmiSuffix  ... Feeder Substation  Data_00_30 ... Data_24_00
    * Data_HH_MM columns are interval-ENDING readings (kWh per interval).
    * NmiSuffix channels:  E* = active import (consumption, incl. controlled load)
                           B* = active export (PV / battery back-feed)
                           Q* = reactive energy (kVArh)   K* = ignored

What this does (fixed from the original E1-only version):
    * nets active power  = sum(E*) - sum(B*)   -> captures PV reverse flow
    * load_id            = "nmi_<NMI>"         -> matches the network JSON ids
    * reactive power     = 0.4 x active (default) OR summed Q* with --reactive-from-q
    * timestamps         = date + interval-ending time (handles 24:00 -> next 00:00)
    * kWh/interval -> average watts (W = kWh * 1000 / interval_hours); use
                      --values-are-kw if your export is already average kW.

Usage:
    python wide_to_long_translator.py                     # defaults below
    python wide_to_long_translator.py -i lexcen_data.csv -o ../forecast_timeseries.csv
    python wide_to_long_translator.py --day 2/06/2023 --reactive-from-q
"""

import argparse
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", default=str(BASE_DIR / "lexcen_data.csv"))
    p.add_argument("-o", "--output", default=str(BASE_DIR.parent / "forecast_timeseries.csv"))
    p.add_argument("--day", default=None,
                   help="keep only this IntervalDay (e.g. 2/06/2023); default = all days")
    p.add_argument("--reactive-from-q", action="store_true",
                   help="use summed Q* channels for reactive power instead of 0.4 x active")
    p.add_argument("--values-are-kw", action="store_true",
                   help="treat Data_ values as average kW instead of kWh per interval")
    p.add_argument("--pf-ratio", type=float, default=0.4,
                   help="reactive = ratio x active when no Q channels used (default 0.4)")
    return p.parse_args()


def interval_minutes(cols):
    """Infer interval length (minutes) from the first two Data_ column labels."""
    def mins(c):
        h, m = c.replace("Data_", "").split("_")
        return int(h) * 60 + int(m)
    times = sorted(mins(c) for c in cols)
    return times[1] - times[0] if len(times) > 1 else 30


def main():
    args = parse_args()
    df = pd.read_csv(args.input, sep="\t")
    print(f"Read {len(df):,} rows from {Path(args.input).name}")

    if args.day is not None:
        df = df[df["IntervalDay"].astype(str) == args.day]
        print(f"  after --day {args.day}: {len(df):,} rows")

    interval_cols = [c for c in df.columns if c.startswith("Data_")]
    dt_min = interval_minutes(interval_cols)
    dt_h = dt_min / 60.0
    print(f"  interval = {dt_min} min")

    # Group each channel into active-import / active-export / reactive.
    df["group"] = df["NmiSuffix"].str[0].map(
        {"E": "imp", "B": "exp", "Q": "rea"}).fillna("other")
    df = df[df["group"] != "other"]

    # Wide -> long over the interval columns.
    long = df.melt(id_vars=["Nmi", "IntervalDay", "group"],
                   value_vars=interval_cols,
                   var_name="interval", value_name="energy")
    long = long.dropna(subset=["energy"])
    long["energy"] = pd.to_numeric(long["energy"], errors="coerce").fillna(0.0)

    # Sum channels within each group, then pivot groups to columns.
    grouped = (long.groupby(["Nmi", "IntervalDay", "interval", "group"])["energy"]
               .sum().unstack("group").fillna(0.0).reset_index())
    for col in ("imp", "exp", "rea"):
        if col not in grouped.columns:
            grouped[col] = 0.0

    # Net active energy per interval; reactive from Q or estimated.
    grouped["net_active"] = grouped["imp"] - grouped["exp"]

    # Interval-ending timestamp: date + HH:MM (24:00 rolls to next day 00:00).
    def to_minutes(c):
        h, m = c.replace("Data_", "").split("_")
        return int(h) * 60 + int(m)
    grouped["timestamp"] = (
        pd.to_datetime(grouped["IntervalDay"], dayfirst=True)
        + pd.to_timedelta(grouped["interval"].map(to_minutes), unit="m")
    )

    grouped["load_id"] = "nmi_" + grouped["Nmi"].astype(str)

    scale = 1000.0 if args.values_are_kw else 1000.0 / dt_h  # -> watts
    grouped["real_power_w"] = (grouped["net_active"] * scale).round(0).astype(int)

    if args.reactive_from_q:
        grouped["reactive_power_var"] = (grouped["rea"] * scale).round(0).astype(int)
    else:
        grouped["reactive_power_var"] = (
            grouped["real_power_w"] * args.pf_ratio).round(0).astype(int)

    out = (grouped[["timestamp", "load_id", "real_power_w", "reactive_power_var"]]
           .sort_values(["timestamp", "load_id"]))
    out.to_csv(args.output, index=False)

    print(f"\nWrote {Path(args.output)}")
    print(f"  {out['load_id'].nunique():,} NMIs x "
          f"{out['timestamp'].nunique()} timesteps = {len(out):,} rows")
    print(f"  active power W: min {out['real_power_w'].min():,} "
          f"(most negative = biggest export)  max {out['real_power_w'].max():,}")
    print(f"  reactive from: {'Q channels' if args.reactive_from_q else f'{args.pf_ratio} x active'}")
    print(f"  units assumed: {'average kW' if args.values_are_kw else 'kWh per interval'}")


if __name__ == "__main__":
    main()
