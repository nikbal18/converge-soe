#!/usr/bin/env python3
"""Fill data/ambient with a gap-free hourly temperature series.

Why this exists
---------------
`pipeline.load_ambient()` globs every CSV in the ambient cache directory and
concatenates them. That check happens BEFORE the `source == "open_meteo"`
branch, so as soon as one cache file exists the pipeline never fetches again,
even if the cached files cover only a fraction of the run window. Timestamps
with no cached temperature become NaN, and `scenarios.py` then falls back to
`ambient.constant_c` (25 C) for those intervals. A dynamic thermal rating
driven by a flat 25 C is not a dynamic thermal rating.

`preflight` with `strict: true` should catch the gap and stop the run. This
script exists so you never get there: fetch the whole window up front, verify
it is complete, then run.

Usage
-----
Check what the cache currently covers:

    python scripts/fetch_ambient.py --check-only

Fetch an explicit window (writes one file covering the whole range):

    python scripts/fetch_ambient.py --start 2023-12-01 --end 2024-03-01

Take the window from a meter export instead of typing dates:

    python scripts/fetch_ambient.py --from-meter data/meter/<export>.csv

Clear stale partial files first (they are moved to data/ambient/_superseded/):

    python scripts/fetch_ambient.py --start 2023-06-01 --end 2023-09-01 --replace

Notes
-----
* Open-Meteo's ERA5 archive is free, public and needs no API key. It lags
  real time by about five days, which is irrelevant for 2023-24 data.
* Timezone defaults to Australia/Sydney, which handles daylight saving. The
  older cached files were fetched with a fixed UTC+11 offset, correct for
  February but one hour out for a winter window. If you are fetching winter
  data, re-fetch the summer window too so the whole cache uses one convention.
* Meter timestamps are local half-hourly, so ambient must be local as well.
"""

import argparse
import csv
import datetime as dt
import shutil
import sys
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO / "data" / "ambient"

# Canberra, matching config/default.yaml
DEFAULT_LAT = -35.2035
DEFAULT_LON = 149.1548
DEFAULT_TZ = "Australia/Sydney"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def parse_date(text):
    for fmt in DATE_FORMATS:
        try:
            return dt.datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {text!r}")


def meter_date_range(path, date_col=1):
    """Min and max IntervalDay in a wide-format meter export.

    Handles both the tab-separated variant with a header and the
    comma-separated variant without one.
    """
    lo = hi = None
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        delim = "\t" if sample.count("\t") > sample.count(",") else ","
        for row in csv.reader(fh, delimiter=delim):
            if len(row) <= date_col:
                continue
            try:
                d = parse_date(row[date_col])
            except ValueError:
                continue          # header row or junk
            if lo is None or d < lo:
                lo = d
            if hi is None or d > hi:
                hi = d
    if lo is None:
        raise SystemExit(f"no parseable dates in column {date_col} of {path}")
    return lo, hi


def read_cache(cache_dir):
    """Concatenate the cache exactly the way pipeline.load_ambient does."""
    frames = []
    for p in sorted(Path(cache_dir).glob("*.csv")):
        d = pd.read_csv(p)
        cols = {c.lower(): c for c in d.columns}
        tcol = cols.get("timestamp") or d.columns[0]
        vcol = cols.get("temperature_c") or d.columns[1]
        s = pd.Series(
            pd.to_numeric(d[vcol], errors="coerce").values,
            index=pd.to_datetime(d[tcol], errors="coerce"),
            name=p.name,
        )
        frames.append(s.dropna())
    if not frames:
        return pd.Series(dtype=float)
    s = pd.concat(frames).sort_index()
    return s[~s.index.duplicated()]


def coverage_report(series, start, end):
    """Hourly gaps between start and end (inclusive of the end date)."""
    wanted = pd.date_range(start, dt.datetime.combine(end, dt.time(23, 0)),
                           freq="h")
    have = series.reindex(wanted)
    missing = have[have.isna()]
    return wanted, missing


def fetch_chunk(lat, lon, start, end, tz, timeout=60):
    r = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m",
            "start_date": str(start),
            "end_date": str(end),
            "timezone": tz,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()
    if "hourly" not in payload:
        raise SystemExit(f"unexpected Open-Meteo response: {payload}")
    idx = pd.to_datetime(payload["hourly"]["time"])
    return pd.Series(payload["hourly"]["temperature_2m"], index=idx,
                     dtype=float)


def chunked_ranges(start, end, days):
    cur = start
    while cur <= end:
        stop = min(cur + dt.timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + dt.timedelta(days=1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fetch and cache Open-Meteo ambient temperature.")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--from-meter", metavar="CSV",
                    help="take the window from a wide-format meter export")
    ap.add_argument("--pad-days", type=int, default=1,
                    help="extra days either side, for interval alignment "
                         "(default 1)")
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--tz", default=DEFAULT_TZ)
    ap.add_argument("--out-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--chunk-days", type=int, default=365)
    ap.add_argument("--replace", action="store_true",
                    help="move existing cache CSVs to _superseded/ first")
    ap.add_argument("--check-only", action="store_true",
                    help="report cache coverage and exit without fetching")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)

    # ---- work out the window ------------------------------------------
    if args.from_meter:
        lo, hi = meter_date_range(args.from_meter)
        print(f"meter export covers {lo} -> {hi}")
    elif args.start and args.end:
        lo, hi = parse_date(args.start), parse_date(args.end)
    elif args.check_only:
        lo = hi = None
    else:
        ap.error("give --start and --end, or --from-meter, or --check-only")

    if lo is not None:
        lo -= dt.timedelta(days=args.pad_days)
        hi += dt.timedelta(days=args.pad_days)

    # ---- what do we already have --------------------------------------
    cached = read_cache(out_dir)
    if len(cached):
        print(f"cache: {len(cached)} hourly points, "
              f"{cached.index.min()} -> {cached.index.max()}, "
              f"across {len(list(out_dir.glob('*.csv')))} file(s)")
    else:
        print(f"cache: empty ({out_dir})")

    if lo is None:
        return 0

    print(f"target window: {lo} -> {hi} "
          f"({(hi - lo).days + 1} days, {((hi - lo).days + 1) * 24} hours)")

    if len(cached):
        _, missing = coverage_report(cached, lo, hi)
        if missing.empty:
            print("cache already covers the target window. Nothing to do.")
            return 0
        print(f"cache is missing {len(missing)} hours in this window "
              f"(first gap {missing.index[0]}, last {missing.index[-1]})")
        print("Those hours would silently become "
              "ambient.constant_c during a run.")

    if args.check_only:
        return 1

    # ---- clear stale partials -----------------------------------------
    if args.replace:
        keep = out_dir / "_superseded"
        moved = 0
        for p in sorted(out_dir.glob("*.csv")):
            keep.mkdir(exist_ok=True)
            shutil.move(str(p), str(keep / p.name))
            moved += 1
        if moved:
            print(f"moved {moved} existing cache file(s) to {keep}")

    # ---- fetch ---------------------------------------------------------
    if args.dry_run:
        for a, b in chunked_ranges(lo, hi, args.chunk_days):
            print(f"  would fetch {a} -> {b}")
        return 0

    parts = []
    for a, b in chunked_ranges(lo, hi, args.chunk_days):
        print(f"fetching {a} -> {b} ({args.tz}) ...", flush=True)
        parts.append(fetch_chunk(args.lat, args.lon, a, b, args.tz))
    series = pd.concat(parts).sort_index()
    series = series[~series.index.duplicated()]

    # ---- verify before writing ----------------------------------------
    wanted, missing = coverage_report(series, lo, hi)
    if not missing.empty:
        print(f"WARNING: {len(missing)} hours still missing after fetch, "
              f"first {missing.index[0]}", file=sys.stderr)
    if series.isna().any():
        print(f"WARNING: {int(series.isna().sum())} NaN temperatures returned",
              file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"open_meteo_{lo}_{hi}.csv"
    (series.rename("temperature_c")
           .rename_axis("timestamp")
           .to_csv(out))
    print(f"\nwrote {out}  ({len(series)} hourly points)")

    # ---- summary that is actually useful for the paper ------------------
    daily_max = series.resample("D").max()
    print(f"  mean {series.mean():.1f} C, "
          f"min {series.min():.1f} C, max {series.max():.1f} C")
    print(f"  days with max >= 30 C: {(daily_max >= 30).sum()} of {len(daily_max)}")
    print(f"  days with max >= 35 C: {(daily_max >= 35).sum()} of {len(daily_max)}")
    print("\nSet ambient.source: cache in your config and re-run preflight "
          "with strict: true to confirm coverage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
