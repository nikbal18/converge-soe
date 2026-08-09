#!/usr/bin/env python3
"""Add the header to a raw GCMDS wide export and slice it to a few days.

Some exports arrive with NO header row: comma-separated, 55 columns, 7
metadata then 48 half-hour channels. Older ones were tab-separated WITH a
header, so `read_wide` cannot read the headerless ones as-is — it looks for
columns named ``Data_HH_MM``.

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


def daily_totals(df, data_cols, metric):
    """kWh per day for export (ΣB), load (ΣE), or net demand (ΣE − ΣB).

    'Minimum demand' in distribution planning means minimum NET demand — what
    is left after rooftop PV — and that is the condition where reverse flow
    and voltage rise bind. Ranking on the E channels alone would find the
    mildest consumption day, which is a different day and a different
    question.
    """
    vals = df[data_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    grp = df["NmiSuffix"].astype(str).str[0]
    out = pd.DataFrame({"_day": df["_day"], "g": grp, "kwh": vals})
    piv = (out[out["g"].isin(["E", "B"])]
           .groupby(["_day", "g"])["kwh"].sum().unstack("g").fillna(0.0))
    for c in ("E", "B"):
        if c not in piv.columns:
            piv[c] = 0.0
    if metric == "export":
        return piv["B"].sort_index()
    if metric == "load":
        return piv["E"].sort_index()
    return (piv["E"] - piv["B"]).sort_index()


def feeder_nmis(feeder_csv):
    """NMIs that are Load components on the named built feeder(s)."""
    import json
    out = set()
    for name in [f.strip() for f in feeder_csv.split(",") if f.strip()]:
        p = REPO / "build" / "network" / "feeders" / f"{name}_network.json"
        if not p.exists():
            raise SystemExit(f"{p} not found — run scripts/build_network.py, "
                             f"and use the feeder id, not the XML name")
        ej = json.loads(p.read_text(encoding="utf-8"))
        for cid, comp in ej["components"].items():
            if "Load" in comp:
                out.add(cid[4:] if cid.startswith("nmi_") else cid)
    return out


def detect_header(path):
    """(headed, names, n_meta, n_data) — metadata width is NOT always 7.

    Some exports carry seven metadata columns; others carry FIVE — no Feeder
    column and no Substation column, 53 columns rather than 55. Assuming
    seven shifts every channel two places and reads
    Data_00_30 as QualityMethod, with no error and no visible symptom until
    the numbers come out wrong. The width is detected by counting the trailing
    numeric columns; shared with tools/prepare_wide_export.py so the two
    cannot drift apart.
    """
    from prepare_wide_export import detect_layout
    return detect_layout(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="raw or headed wide meter export to slice")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--days", type=int, default=None,
                    help="window length in days")
    ap.add_argument("--metric", choices=["export", "load", "net"],
                    default="export",
                    help="the daily quantity to rank. export = sum of B* "
                         "(PV back-feed). load = sum of E* (gross "
                         "consumption). net = E* - B*, i.e. NET demand as the "
                         "network sees it — the quantity that actually loads "
                         "the transformer, and the one 'minimum demand' means "
                         "in distribution planning.")
    ap.add_argument("--select", choices=["window", "day-max", "day-min"],
                    default="window",
                    help="window (default): the --days CONSECUTIVE days with "
                         "the largest total. day-max / day-min: find the "
                         "single most extreme DAY, then take a --days window "
                         "CENTRED on it. Use day-min with --metric net for "
                         "the minimum-demand case (peak reverse flow, worst "
                         "voltage rise) and day-max for the winter evening "
                         "peak.")
    ap.add_argument("--feeder", default=None,
                    help="comma list of built feeder ids; restrict the daily "
                         "totals to NMIs that are Loads on those feeders. "
                         "Without it the extreme day is chosen from every NMI "
                         "in the export, which may sit on feeders you are not "
                         "solving.")
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--has-header", action="store_true",
                    help="input already has a header row")
    args = ap.parse_args()

    headed, names, n_meta, n_data = detect_header(args.input)
    if args.has_header or headed:
        df = pd.read_csv(args.input, dtype={0: str}, encoding="utf-8-sig")
    else:
        df = pd.read_csv(args.input, header=None, names=names,
                         dtype={"Nmi": str, "IntervalDay": str,
                                "NmiSuffix": str}, encoding="utf-8-sig")
        print(f"added header: {n_meta} metadata + {n_data} "
              f"interval columns ({names[n_meta]} .. {names[-1]})")
        if n_meta == 5:
            print("  (5 metadata columns — this export has no Feeder or "
                  "Substation column; NMIs are mapped to substations by LV "
                  "subtree in stage ③, not from the file)")

    df["_day"] = pd.to_datetime(df["IntervalDay"], errors="coerce")
    days = sorted(df["_day"].dropna().unique())
    print(f"{len(df):,} rows | {df['Nmi'].nunique()} NMIs | {len(days)} days "
          f"({pd.Timestamp(days[0]).date()} .. {pd.Timestamp(days[-1]).date()})")

    if args.days:
        data_cols = [c for c in df.columns if c.startswith("Data_")]

        scope = df
        if args.feeder:
            keep = feeder_nmis(args.feeder)
            scope = df[df["Nmi"].astype(str).isin(keep)]
            print(f"restricted to {scope['Nmi'].nunique()} of "
                  f"{df['Nmi'].nunique()} NMIs on {args.feeder}")
            if scope.empty:
                raise SystemExit(
                    f"no NMI in {args.input.name} is a Load on {args.feeder} "
                    f"— check the feeder id against build/network/feeders/")

        by_day = daily_totals(scope, data_cols, args.metric)
        if by_day.empty:
            raise SystemExit(f"no usable channels for --metric {args.metric}")

        if args.select == "window":
            roll = by_day.rolling(args.days).sum()
            end = roll.idxmax()
            sel = by_day.loc[:end].index[-args.days:]
            lo, hi = pd.Timestamp(sel[0]), pd.Timestamp(sel[-1])
            print(f"picked the {args.days} consecutive day(s) with peak "
                  f"{args.metric}: {lo.date()} .. {hi.date()} "
                  f"({roll.max():,.0f} kWh vs {by_day.mean():,.0f} kWh/day "
                  f"average)")
        else:
            want_max = args.select == "day-max"
            day = by_day.idxmax() if want_max else by_day.idxmin()
            # Centre the window on the extreme day, then slide it back inside
            # the data rather than truncating: a 5-day window around a day
            # that happens to be the second of the export is still a 7-day
            # comparison against the other season, and an unequal number of
            # intervals between scenarios is exactly what analysis.py has to
            # throw away later.
            half = (args.days - 1) // 2
            lo = pd.Timestamp(day) - pd.Timedelta(days=half)
            hi = lo + pd.Timedelta(days=args.days - 1)
            first, last = pd.Timestamp(by_day.index[0]), pd.Timestamp(by_day.index[-1])
            if lo < first:
                lo, hi = first, first + pd.Timedelta(days=args.days - 1)
            if hi > last:
                hi, lo = last, last - pd.Timedelta(days=args.days - 1)
            lo = max(lo, first)

            extreme = "maximum" if want_max else "minimum"
            print(f"\n{extreme} {args.metric} demand day: "
                  f"{pd.Timestamp(day).date()} "
                  f"({by_day.loc[day]:,.0f} kWh, against a "
                  f"{by_day.mean():,.0f} kWh/day average)")
            ranked = by_day.sort_values(ascending=want_max).tail(5)[::-1]
            print(f"  five most extreme days, for a sanity check:")
            for d, v in ranked.items():
                mark = "  <- picked" if d == day else ""
                print(f"    {pd.Timestamp(d).date()}  {v:12,.0f} kWh{mark}")
            print(f"  {args.days}-day window centred on it: "
                  f"{lo.date()} .. {hi.date()}")
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
