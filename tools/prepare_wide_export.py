#!/usr/bin/env python3
"""Stream a raw GCMDS wide export into the stage-② long parquet cache.

Why this exists
---------------
``timeseries.read_wide`` does the whole file in memory: ``pd.read_csv`` the
293 MB export, then ``melt`` it to one row per (NMI, day, channel, interval).
On a 92-day export that is 742,740 × 48 = **35.6 million** rows before
the groupby, and the groupby then sorts them. Peak RSS lands somewhere between
6 and 12 GB. On a 16 GB laptop that is a coin flip; on 8 GB it is a
MemoryError at 2 a.m. with nothing to show for the night.

This script does the identical arithmetic in day-sized chunks and never holds
more than one chunk of melted data, then writes the result straight to
``build/timeseries/all_nmis.parquet`` and stamps ``build/manifest.json`` with
the exact fingerprint stage ② computes. The next ``run_feeder.py`` prints
"stage ② prepare timeseries: cached (inputs unchanged)" and moves on.

It also handles the two things ``tools/slice_meter_export.py`` gets wrong on
this export:

* **5 metadata columns, not 7.** ``slice_meter_export.META`` is hardcoded to
  seven (…, Feeder, Substation). Some exports carry only five — no Feeder,
  no Substation column at all — so a 7-column assumption shifts every channel
  two places and silently reads Data_00_30 as QualityMethod. The metadata
  width is detected here from how many trailing columns are numeric.
* **No header row.** ``sniff_format`` needs ``Nmi`` and ``Data_HH_MM`` column
  names, so the raw export is rejected outright as "unknown format".

Arithmetic parity with read_wide
--------------------------------
net_active = Σ(E* channels) − Σ(B* channels); reactive = Σ(Q* channels) when
ANY of them is non-zero across the whole file, else pf_ratio × net_active;
scale = 1000/dt_h for kwh_per_interval; timestamps are interval-ENDING, so
Data_24_00 is 1440 minutes past midnight = 00:00 the following day. The
``--verify`` mode re-derives one day through the real ``read_wide`` and
asserts the two agree.

Usage
-----
    # convert, cache, and stamp the manifest (do this once per season)
    python tools/prepare_wide_export.py \
        --input data/meter/<export>.csv \
        --values-are kwh_per_interval

    # prove it matches read_wide on one day before trusting it
    python tools/prepare_wide_export.py \
        --input data/meter/<export>.csv \
        --values-are kwh_per_interval --verify 2023-12-15

    # just report what the file looks like
    python tools/prepare_wide_export.py --input <csv> --inspect
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from converge_soe import pipeline as pl        # noqa: E402
from converge_soe import timeseries as tsm     # noqa: E402

META_5 = ["Nmi", "IntervalDay", "NmiSuffix", "QualityMethod", "GCMDSExtractTime"]
META_7 = META_5 + ["Feeder", "Substation"]


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------
def data_column_names(n_data_cols, first_min=30, step_min=30):
    """Data_00_30 … Data_24_00 for an interval-ENDING export."""
    out = []
    for k in range(n_data_cols):
        t = first_min + k * step_min
        out.append(f"Data_{t // 60:02d}_{t % 60:02d}")
    return out


def _numeric_frac(series):
    v = pd.to_numeric(series, errors="coerce")
    return float(v.notna().mean()) if len(v) else 0.0


def detect_layout(path, probe_rows=400):
    """Return (headed, names, n_meta, n_data).

    The metadata width is found by walking in from the right and counting how
    many trailing columns are essentially all numeric. Everything before that
    is metadata. 53 columns → 48 data + 5 meta; 55 → 48 data + 7 meta.
    """
    probe = pd.read_csv(path, nrows=probe_rows, header=None, dtype=str,
                        encoding="utf-8-sig")
    ncols = probe.shape[1]
    headed = str(probe.iloc[0, 0]).strip().lower() == "nmi"
    if headed:
        names = [str(c).strip() for c in probe.iloc[0].tolist()]
        n_data = sum(1 for c in names if c.startswith("Data_"))
        return True, names, ncols - n_data, n_data

    body = probe
    n_data = 0
    for j in range(ncols - 1, -1, -1):
        if _numeric_frac(body[j]) > 0.98:
            n_data += 1
        else:
            break
    if n_data == 0:
        raise SystemExit(
            f"{path}: found no trailing numeric columns — this does not look "
            f"like a wide GCMDS export ({ncols} columns).")
    n_meta = ncols - n_data
    if n_meta == 5:
        meta = META_5
    elif n_meta == 7:
        meta = META_7
    else:
        raise SystemExit(
            f"{path}: {ncols} columns = {n_meta} metadata + {n_data} interval "
            f"columns. Only 5- and 7-column metadata layouts are known. Check "
            f"the export before going further — guessing here silently "
            f"mislabels every channel.")
    if n_data not in (48, 96, 24):
        print(f"  ! {n_data} interval columns is unusual (expected 48 for "
              f"half-hourly). Continuing, but check the result.")
    step = 1440 // n_data
    return False, meta + data_column_names(n_data, first_min=step,
                                           step_min=step), n_meta, n_data


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def _to_minutes(col):
    h, m = col.replace("Data_", "").split("_")
    return int(h) * 60 + int(m)


def _day_format(path, headed, names, sample_rows=2_000_000):
    """Pin the IntervalDay format ONCE, over every distinct day in the file.

    read_wide detects per call. Detecting per chunk would be worse than that:
    the contiguity tie-break in detect_column_format looks at the span of the
    values it is shown, so a chunk covering three days and a chunk covering
    ninety could legitimately disagree and splice two different calendars into
    one series. Reading the single date column over the whole file is cheap.
    """
    kw = dict(usecols=[1], dtype=str, encoding="utf-8-sig")
    if headed:
        days = pd.read_csv(path, **kw).iloc[:, 0]
    else:
        days = pd.read_csv(path, header=None, names=["IntervalDay"],
                           **kw)["IntervalDay"]
    uniq = pd.Series(pd.unique(days.dropna().astype(str).str.strip()))
    fmt = tsm.detect_column_format(uniq)
    if fmt is None:
        raise SystemExit(
            f"{path}: could not pin down the IntervalDay format from "
            f"{uniq.head(3).tolist()}")
    parsed = pd.to_datetime(uniq, format=fmt)
    print(f"  IntervalDay format {fmt!r}: {len(uniq)} distinct days, "
          f"{parsed.min().date()} → {parsed.max().date()}")
    return fmt


def convert(path, values_are, reactive_from_q=True, pf_ratio=0.4,
            chunk_rows=150_000, out_parts=None):
    """Stream the export into (timestamp, load_id, real_power_w, _rea) parts.

    Returns (list_of_part_paths, any_nonzero_rea).
    """
    headed, names, n_meta, n_data = detect_layout(path)
    print(f"  layout: {'headed' if headed else 'headerless'}, "
          f"{n_meta} metadata + {n_data} interval columns")

    data_cols = [c for c in names if c.startswith("Data_")]
    minutes = {c: _to_minutes(c) for c in data_cols}
    times = sorted(minutes.values())
    dt_min = times[1] - times[0] if len(times) > 1 else 30
    dt_h = dt_min / 60.0
    scale = {"w": 1.0, "kw": 1000.0,
             "kwh_per_interval": 1000.0 / dt_h}[values_are]

    fmt = _day_format(path, headed, names)

    reader_kw = dict(chunksize=chunk_rows, encoding="utf-8-sig",
                     dtype={c: str for c in ("Nmi", "IntervalDay",
                                             "NmiSuffix")})
    if headed:
        reader = pd.read_csv(path, **reader_kw)
    else:
        reader = pd.read_csv(path, header=None, names=names, **reader_kw)

    parts, carry = [], None
    any_rea = False
    n_in = n_out = 0
    for i, chunk in enumerate(reader):
        if carry is not None and len(carry):
            chunk = pd.concat([carry, chunk], ignore_index=True)
        n_in += len(chunk)

        # A (Nmi, IntervalDay) group is summed across its channel rows, so a
        # group must never be split across chunks. Hold back the trailing
        # group (a handful of rows: E1/B1/Q1/K1…) and prepend it to the next.
        if len(chunk):
            last = (chunk["Nmi"].iat[-1], chunk["IntervalDay"].iat[-1])
            tail = ((chunk["Nmi"] == last[0]) &
                    (chunk["IntervalDay"] == last[1]))
            carry = chunk[tail].copy()
            chunk = chunk[~tail]
        if not len(chunk):
            continue

        part, rea_seen = _convert_chunk(chunk, data_cols, minutes, fmt, scale)
        any_rea = any_rea or rea_seen
        if part is None or not len(part):
            continue
        p = Path(out_parts) / f"part_{i:05d}.parquet"
        part.to_parquet(p, index=False)
        parts.append(p)
        n_out += len(part)
        print(f"    chunk {i}: {n_in:,} rows read → {n_out:,} long rows",
              end="\r", flush=True)

    if carry is not None and len(carry):
        part, rea_seen = _convert_chunk(carry, data_cols, minutes, fmt, scale)
        any_rea = any_rea or rea_seen
        if part is not None and len(part):
            p = Path(out_parts) / "part_final.parquet"
            part.to_parquet(p, index=False)
            parts.append(p)
            n_out += len(part)

    print(f"\n  {n_in:,} export rows → {n_out:,} long rows in {len(parts)} part(s)")
    if not parts:
        raise SystemExit(f"{path}: produced no rows — check --values-are and "
                         f"the NmiSuffix channels (expect E*/B*/Q*).")
    return parts, any_rea


def _convert_chunk(chunk, data_cols, minutes, fmt, scale):
    """One chunk → long rows. Mirrors timeseries.read_wide exactly."""
    grp = chunk["NmiSuffix"].astype(str).str[0].map(
        {"E": "imp", "B": "exp", "Q": "rea"})
    chunk = chunk.assign(group=grp)
    chunk = chunk[chunk["group"].notna()]
    if not len(chunk):
        return None, False

    long = chunk.melt(id_vars=["Nmi", "IntervalDay", "group"],
                      value_vars=data_cols, var_name="interval",
                      value_name="energy")
    long = long.dropna(subset=["energy"])
    long["energy"] = pd.to_numeric(long["energy"], errors="coerce").fillna(0.0)

    g = (long.groupby(["Nmi", "IntervalDay", "interval", "group"],
                      observed=True)["energy"]
         .sum().unstack("group").fillna(0.0).reset_index())
    for col in ("imp", "exp", "rea"):
        if col not in g.columns:
            g[col] = 0.0

    day_ts = pd.to_datetime(g["IntervalDay"].astype(str).str.strip(), format=fmt)
    out = pd.DataFrame({
        "timestamp": day_ts + pd.to_timedelta(
            g["interval"].map(minutes), unit="m"),
        "load_id": "nmi_" + g["Nmi"].astype(str),
        "real_power_w": (g["imp"] - g["exp"]) * scale,
        "_rea": g["rea"] * scale,
    })
    return out, bool((g["rea"] != 0).any())


def finalise(parts, any_rea, reactive_from_q, pf_ratio):
    """Concatenate the parts, apply the reactive rule, sort and dedupe."""
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if reactive_from_q and any_rea:
        df["reactive_power_var"] = df["_rea"]
        print("  reactive: from summed Q* channels")
    else:
        df["reactive_power_var"] = df["real_power_w"] * pf_ratio
        print(f"  reactive: pf_ratio × active ({pf_ratio}) — "
              f"{'no Q* channels present' if not any_rea else 'disabled'}")
    df = df.drop(columns="_rea")[tsm.LONG_COLUMNS]
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp", "load_id"], keep="first")
    if len(df) != before:
        print(f"  dropped {before - len(df):,} duplicate (timestamp, load_id) row(s)")
    df = df.sort_values(["timestamp", "load_id"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Verification against the real read_wide
# ---------------------------------------------------------------------------
def verify_day(path, day, values_are, reactive_from_q, pf_ratio, ours):
    """Re-derive ONE day through timeseries.read_wide and compare.

    read_wide needs a header, so a headed one-day slice is written to a temp
    file first. This is the check that the chunked path is not quietly
    different arithmetic.
    """
    headed, names, n_meta, n_data = detect_layout(path)
    kw = dict(encoding="utf-8-sig",
              dtype={c: str for c in ("Nmi", "IntervalDay", "NmiSuffix")})
    reader = (pd.read_csv(path, chunksize=200_000, **kw) if headed else
              pd.read_csv(path, header=None, names=names, chunksize=200_000, **kw))
    rows = [c[c["IntervalDay"].astype(str).str.strip() == day] for c in reader]
    slab = pd.concat(rows, ignore_index=True)
    if not len(slab):
        raise SystemExit(f"--verify {day}: no rows for that IntervalDay")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "one_day.csv"
        slab.to_csv(tmp, index=False)
        ref = tsm.read_wide(tmp, values_are, reactive_from_q=reactive_from_q,
                            pf_ratio=pf_ratio)

    mine = ours[ours["timestamp"].dt.normalize().isin(
        [pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1)])]
    mine = mine.merge(ref, on=["timestamp", "load_id"], how="inner",
                      suffixes=("_mine", "_ref"))
    if not len(mine):
        raise SystemExit(f"--verify {day}: no overlapping rows to compare — "
                         f"the timestamp derivation differs, which is exactly "
                         f"the bug this check exists to catch.")
    dp = np.abs(mine["real_power_w_mine"] - mine["real_power_w_ref"]).max()
    dq = np.abs(mine["reactive_power_var_mine"] -
                mine["reactive_power_var_ref"]).max()
    n_ref = len(ref)
    print(f"\n  VERIFY {day}: {len(mine):,} of read_wide's {n_ref:,} rows "
          f"matched on (timestamp, load_id)")
    print(f"    max |ΔP| = {dp:.6g} W    max |ΔQ| = {dq:.6g} VAr")
    ok = dp < 1e-6 and dq < 1e-6 and len(mine) == n_ref
    print("    " + ("PASS — identical to read_wide" if ok else
                    "FAIL — do NOT use this cache; report the mismatch"))
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="raw wide export CSV")
    ap.add_argument("--values-are", default="kwh_per_interval",
                    choices=["w", "kw", "kwh_per_interval"])
    ap.add_argument("--pf-ratio", type=float, default=None,
                    help="default: config/default.yaml timeseries.pf_ratio")
    ap.add_argument("--no-reactive-from-q", dest="reactive_from_q",
                    action="store_false", default=None)
    ap.add_argument("--chunk-rows", type=int, default=150_000)
    ap.add_argument("--inspect", action="store_true",
                    help="report the detected layout and exit")
    ap.add_argument("--verify", metavar="YYYY-MM-DD|auto", default=None,
                    help="cross-check one day against timeseries.read_wide. "
                         "'auto' picks a day from the middle of THIS file, "
                         "which is what you want when the same flag is applied "
                         "to several exports covering different windows")
    ap.add_argument("--out", type=Path, default=None,
                    help="parquet path (default: the stage-② cache)")
    ap.add_argument("--no-stamp", action="store_true",
                    help="write the parquet but do not touch build/manifest.json")
    args = ap.parse_args(argv)

    if not args.input.exists():
        raise SystemExit(f"{args.input} not found")

    cfg = pl.load_config(REPO)
    tcfg = cfg.get("timeseries", {}) or {}
    pf_ratio = args.pf_ratio if args.pf_ratio is not None else tcfg.get("pf_ratio", 0.4)
    reactive_from_q = (args.reactive_from_q if args.reactive_from_q is not None
                       else tcfg.get("reactive_from_q", True))

    print(f"input: {args.input}  ({args.input.stat().st_size / 1e6:.0f} MB)")
    if args.inspect:
        headed, names, n_meta, n_data = detect_layout(args.input)
        print(f"  {'headed' if headed else 'headerless'}, {n_meta} metadata "
              f"+ {n_data} interval columns")
        print(f"  metadata: {names[:n_meta]}")
        print(f"  first/last interval: {names[n_meta]} … {names[-1]}")
        _day_format(args.input, headed, names)
        return 0

    out = args.out or (REPO / "build" / "timeseries" / "all_nmis.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    tmpdir = Path(tempfile.mkdtemp(prefix="csoe_wide_"))
    try:
        parts, any_rea = convert(args.input, args.values_are,
                                 reactive_from_q=reactive_from_q,
                                 pf_ratio=pf_ratio,
                                 chunk_rows=args.chunk_rows,
                                 out_parts=tmpdir)
        df = finalise(parts, any_rea, reactive_from_q, pf_ratio)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    warnings = tsm.check_magnitudes(df)
    print(f"\n  {len(df):,} rows | {df['load_id'].nunique():,} NMIs | "
          f"{df['timestamp'].nunique():,} timestamps")
    print(f"  {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  P: min {df['real_power_w'].min() / 1000:.1f} kW, "
          f"max {df['real_power_w'].max() / 1000:.1f} kW")
    for w in warnings:
        print(f"  ! UNITS: {w}")

    ok = True
    if args.verify:
        day = args.verify
        days = sorted(df["timestamp"].dt.normalize().unique())
        if day == "auto":
            # Middle of the file, not the first day: the first day is the one
            # most likely to be a partial delivery.
            day = str(pd.Timestamp(days[len(days) // 2]).date())
            print(f"\n  --verify auto -> {day}")
        elif pd.Timestamp(day) not in set(days):
            raise SystemExit(
                f"--verify {day} is not in this export "
                f"({pd.Timestamp(days[0]).date()} → "
                f"{pd.Timestamp(days[-1]).date()}). Use --verify auto, or a "
                f"date inside the window.")
        ok = verify_day(args.input, day, args.values_are,
                        reactive_from_q, pf_ratio, df)
        if not ok:
            return 2

    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.0f} MB)")

    if not args.no_stamp:
        fp = tsm.file_fingerprint(args.input) + f"|{args.values_are}"
        pl.update_manifest(REPO, "timeseries", fp)
        print(f"stamped build/manifest.json for --meter {args.input.as_posix()} "
              f"--values-are {args.values_are}")
        print("stage ② will now report 'cached (inputs unchanged)' for that "
              "exact --meter/--values-are pair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
