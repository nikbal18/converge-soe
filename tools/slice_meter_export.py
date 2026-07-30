#!/usr/bin/env python3
"""Add the header to a raw GCMDS wide export and slice it to a few days.

The summer export (data/meter/lexcen_summer_data.csv) arrives with NO header
row, comma-separated, 55 columns: 7 metadata then 48 half-hour channels. The
older lexcen_data.csv was tab-separated WITH a header, so `read_wide` cannot
read the new one as-is — it looks for columns named ``Data_HH_MM``.

It is also 92 days x 413 NMIs. At roughly 30 minutes of solve time per 3 days
per scenario, running the whole thing is days of compute; slice first.

    # 3 consecutive days with the most PV export (where DTR matters most)
    python tools/slice_meter_export.py --days 3

    # or an explicit window
    python tools/slice_meter_export.py --from 2024-01-15 --to 2024-01-17

Writes data/meter/<stem>_<from>_<to>.csv, ready for
    run_feeder.py --meter <that file> --values-are kwh_per_interval
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

META = ["Nmi", "IntervalDay", "NmiSuffix", "QualityMethod",
        "GCMDSExtractTime", "Feeder", "Substation"]


def header_for(n_data_cols, first_min=30, step_min=30):
    """Data_00_30 ... Data_24_00 for an interval-ENDING export."""
    cols = []
    for k in range(n_data_cols):
        t = first_min + k * step_min
        cols.append(f"Data_{t // 60:02d}_{t % 60:02d}")
    return META + cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=REPO / "data/meter/lexcen_summer_data.csv")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--days", type=int, default=None,
                    help="pick this many CONSECUTIVE days with the highest "
                         "total PV export (sum of B* channels)")
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--has-header", action="store_true",
                    help="input already has a header row")
    args = ap.parse_args()

    probe = pd.read_csv(args.input, nrows=1, header=None, dtype=str,
                        encoding="utf-8-sig")
    ncols = probe.shape[1]
    if args.has_header:
        df = pd.read_csv(args.input, dtype={0: str}, encoding="utf-8-sig")
    else:
        names = header_for(ncols - len(META))
        df = pd.read_csv(args.input, header=None, names=names,
                         dtype={"Nmi": str, "IntervalDay": str,
                                "NmiSuffix": str}, encoding="utf-8-sig")
        print(f"added header: {len(META)} metadata + {ncols - len(META)} "
              f"interval columns ({names[len(META)]} .. {names[-1]})")

    df["_day"] = pd.to_datetime(df["IntervalDay"], errors="coerce")
    days = sorted(df["_day"].dropna().unique())
    print(f"{len(df):,} rows | {df['Nmi'].nunique()} NMIs | {len(days)} days "
          f"({pd.Timestamp(days[0]).date()} .. {pd.Timestamp(days[-1]).date()})")

    if args.days:
        data_cols = [c for c in df.columns if c.startswith("Data_")]
        exp = df[df["NmiSuffix"].str.startswith("B", na=False)]
        by_day = (exp.groupby("_day")[data_cols]
                  .apply(lambda g: pd.to_numeric(
                      g.stack(), errors="coerce").sum()))
        # best consecutive window
        roll = by_day.rolling(args.days).sum()
        end = roll.idxmax()
        sel = by_day.loc[:end].index[-args.days:]
        lo, hi = pd.Timestamp(sel[0]), pd.Timestamp(sel[-1])
        print(f"picked the {args.days} consecutive day(s) with peak export: "
              f"{lo.date()} .. {hi.date()} "
              f"({roll.max():,.0f} kWh exported vs {by_day.mean():,.0f} "
              f"kWh/day average)")
    else:
        lo = pd.Timestamp(args.date_from) if args.date_from else pd.Timestamp(days[0])
        hi = pd.Timestamp(args.date_to) if args.date_to else pd.Timestamp(days[-1])

    out_df = df[(df["_day"] >= lo) & (df["_day"] <= hi)].drop(columns="_day")
    if out_df.empty:
        raise SystemExit(f"no rows between {lo.date()} and {hi.date()}")

    out = args.out or (args.input.parent /
                       f"{args.input.stem}_{lo.date()}_{hi.date()}.csv")
    out_df.to_csv(out, index=False)
    print(f"\n{len(out_df):,} rows -> {out}")
    print(f"  NMIs {out_df['Nmi'].nunique()} | days "
          f"{out_df['IntervalDay'].nunique()} | suffixes "
          f"{sorted(out_df['NmiSuffix'].str[0].unique())}")
    print(f"\nnext:\n  python scripts/prepare_timeseries.py --meter {out} "
          f"--values-are kwh_per_interval --fetch-ambient")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
