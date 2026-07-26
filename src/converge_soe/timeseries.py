"""Load, normalise, and pre-index NMI interval-meter timeseries.

Two jobs:

1. **Prepare** (pipeline stage ②): read one or many raw meter exports —
   the wide EvoEnergy tab-separated format (``Data_HH_MM`` interval columns,
   ``NmiSuffix`` channels) or an already-long CSV — and normalise them to one
   tidy long table::

       timestamp, load_id, real_power_w, reactive_power_var

   Units are NEVER guessed silently: ``values_are`` must be one of
   ``w | kw | kwh_per_interval`` for wide exports. A loud warning is raised
   if the chosen unit produces implausible magnitudes.

2. **Pre-index** (pipeline stage ⑤): slice the year of data down to one
   substation's NMIs as plain NumPy arrays, once, before any solving::

       timestamps : int64   [T]   epoch ns, sorted, unique
       load_ids   : <U32    [N]   NMIs on this substation, sorted
       P          : float32 [T,N] real power, W, load-positive
       Q          : float32 [T,N] reactive power, VAr
       theta_A    : float32 [T]   ambient °C, aligned to these timestamps
       mask       : bool    [T,N] True = real reading, False = filled
       dt_minutes : float         validated constant interval

   The timestep loop then contains **no pandas at all** — ``P[t, :]`` is a
   contiguous float32 row. float32 is deliberate: meter data has nowhere near
   float64 precision and halving the array doubles cache efficiency.
"""

import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LONG_COLUMNS = ["timestamp", "load_id", "real_power_w", "reactive_power_var"]

# Formats tried, in order, on the first timestamp value; the matching format
# is then passed explicitly to pd.to_datetime for the whole column (TS002 —
# kills the "Could not infer format" warning and the per-element fallback).
_TS_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%Y-%m-%d", "%d/%m/%Y", "%H:%M",
]


def detect_timestamp_format(sample):
    for fmt in _TS_FORMATS:
        try:
            pd.to_datetime(str(sample), format=fmt)
            return fmt
        except (ValueError, TypeError):
            continue
    return None


def parse_timestamps(series):
    """Parse a timestamp column with an explicit format when one fits."""
    fmt = detect_timestamp_format(series.iloc[0]) if len(series) else None
    if fmt:
        try:
            return pd.to_datetime(series, format=fmt), fmt
        except (ValueError, TypeError):
            pass
    logger.warning("timestamp format could not be pinned down from %r; "
                   "falling back to dateutil (slow)", series.iloc[0] if len(series) else None)
    return pd.to_datetime(series, dayfirst=True), None


# ---------------------------------------------------------------------------
# Reading raw exports
# ---------------------------------------------------------------------------
def sniff_format(path):
    """Return 'wide' | 'long' | 'unknown' by inspecting the header row."""
    head = Path(path).open(encoding="utf-8-sig", errors="replace").readline()
    sep = "\t" if "\t" in head else ","
    cols = [c.strip() for c in head.split(sep)]
    if any(c.startswith("Data_") for c in cols) and any(
            c.lower() == "nmi" for c in cols):
        return "wide"
    lowered = {c.lower() for c in cols}
    if {"timestamp", "load_id"} <= lowered:
        return "long"
    return "unknown"


def read_wide(path, values_are, reactive_from_q=True, pf_ratio=0.4, day=None):
    """EvoEnergy wide export → normalised long frame.

    This is the maintained implementation of the logic in
    examples/legacy/wide_to_long_translator.py:
      * nets active power = sum(E*) − sum(B*)  (captures PV reverse flow)
      * load_id = "nmi_<NMI>"
      * reactive from summed Q* channels or pf_ratio × active
      * interval-ENDING timestamps (24:00 rolls to next day 00:00)
    """
    if values_are not in ("w", "kw", "kwh_per_interval"):
        raise ValueError(
            "values_are must be given explicitly for wide exports: "
            "'w', 'kw' or 'kwh_per_interval' (no silent unit guessing)")

    df = pd.read_csv(path, sep="\t")
    if "Nmi" not in df.columns:  # comma-separated variant
        df = pd.read_csv(path)
    if day is not None:
        df = df[df["IntervalDay"].astype(str) == day]

    interval_cols = [c for c in df.columns if c.startswith("Data_")]

    def to_minutes(c):
        h, m = c.replace("Data_", "").split("_")
        return int(h) * 60 + int(m)

    times = sorted(to_minutes(c) for c in interval_cols)
    dt_min = times[1] - times[0] if len(times) > 1 else 30
    dt_h = dt_min / 60.0

    df["group"] = df["NmiSuffix"].str[0].map(
        {"E": "imp", "B": "exp", "Q": "rea"}).fillna("other")
    df = df[df["group"] != "other"]

    long = df.melt(id_vars=["Nmi", "IntervalDay", "group"],
                   value_vars=interval_cols,
                   var_name="interval", value_name="energy")
    long = long.dropna(subset=["energy"])
    long["energy"] = pd.to_numeric(long["energy"], errors="coerce").fillna(0.0)

    grouped = (long.groupby(["Nmi", "IntervalDay", "interval", "group"])["energy"]
               .sum().unstack("group").fillna(0.0).reset_index())
    for col in ("imp", "exp", "rea"):
        if col not in grouped.columns:
            grouped[col] = 0.0

    grouped["net_active"] = grouped["imp"] - grouped["exp"]
    day_ts, _ = parse_timestamps(grouped["IntervalDay"].astype(str))
    grouped["timestamp"] = (
        day_ts + pd.to_timedelta(grouped["interval"].map(to_minutes), unit="m"))
    grouped["load_id"] = "nmi_" + grouped["Nmi"].astype(str)

    scale = {"w": 1.0, "kw": 1000.0, "kwh_per_interval": 1000.0 / dt_h}[values_are]
    grouped["real_power_w"] = grouped["net_active"] * scale
    if reactive_from_q and (grouped["rea"] != 0).any():
        grouped["reactive_power_var"] = grouped["rea"] * scale
    else:
        grouped["reactive_power_var"] = grouped["real_power_w"] * pf_ratio

    return grouped[LONG_COLUMNS].sort_values(["timestamp", "load_id"]).reset_index(drop=True)


def read_long(path, values_are=None):
    """Already-long CSV → normalised long frame (column names normalised)."""
    df = pd.read_csv(path, dtype={"load_id": str})
    df.columns = [c.strip().lower() for c in df.columns]
    ren = {}
    for want, alts in {
        "timestamp": ("timestamp", "datetime", "time", "date"),
        "load_id": ("load_id", "nmi", "loadid"),
        "real_power_w": ("real_power_w", "p_w", "active_power_w", "real_power"),
        "reactive_power_var": ("reactive_power_var", "q_var",
                               "reactive_power", "reactive_var"),
    }.items():
        for a in alts:
            if a in df.columns:
                ren[a] = want
                break
    df = df.rename(columns=ren)
    missing = [c for c in ("timestamp", "load_id", "real_power_w") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: long-format file missing columns {missing}")
    if "reactive_power_var" not in df.columns:
        df["reactive_power_var"] = 0.0
    df["timestamp"], _ = parse_timestamps(df["timestamp"])
    df["load_id"] = df["load_id"].astype(str)
    scale = {None: 1.0, "w": 1.0, "kw": 1000.0}.get(values_are, 1.0)
    df["real_power_w"] = pd.to_numeric(df["real_power_w"], errors="coerce") * scale
    df["reactive_power_var"] = pd.to_numeric(df["reactive_power_var"], errors="coerce") * scale
    return df[LONG_COLUMNS].sort_values(["timestamp", "load_id"]).reset_index(drop=True)


def check_magnitudes(df):
    """Loud plausibility warnings after unit selection (TS006)."""
    warnings = []
    med = df.groupby("load_id")["real_power_w"].apply(lambda s: s.abs().median())
    n_big = int((med > 20000).sum())
    if n_big:
        warnings.append(
            f"{n_big} NMI(s) have median |real_power_w| > 20 kW — values are "
            "probably kW or kWh-per-interval, not W. Check --values-are.")
    if df["real_power_w"].min() >= 0:
        warnings.append(
            "min(real_power_w) >= 0 across all NMIs — no PV export captured; "
            "check the export/B-channel handling or translator flags.")
    both = df[(df["reactive_power_var"].abs() > df["real_power_w"].abs())]
    if len(df) and len(both) / len(df) > 0.05:
        warnings.append(
            f"|Q| > |P| in {100 * len(both) / len(df):.1f} % of rows — "
            "reactive power units look wrong.")
    for w in warnings:
        logger.warning("UNITS: %s", w)
    return warnings


def prepare(paths, values_are=None, reactive_from_q=True, pf_ratio=0.4,
            day=None):
    """Read one or many CSVs (wide or long, auto-detected) → one long frame."""
    frames = []
    for p in map(Path, paths):
        kind = sniff_format(p)
        logger.info("reading %s (%s format)", p.name, kind)
        if kind == "wide":
            frames.append(read_wide(p, values_are, reactive_from_q=reactive_from_q,
                                    pf_ratio=pf_ratio, day=day))
        elif kind == "long":
            frames.append(read_long(p, values_are if values_are in (None, "w", "kw") else None))
        else:
            raise ValueError(f"{p}: cannot determine meter file format "
                             "(neither wide Data_HH_MM nor long timestamp/load_id)")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp", "load_id"], keep="first")
    check_magnitudes(df)
    return df.sort_values(["timestamp", "load_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# NMI id reconciliation (the nmi_ prefix trap)
# ---------------------------------------------------------------------------
def reconcile_load_ids(ts_ids, network_ids):
    """Match timeseries load ids to network Load component ids.

    Detects the ``nmi_`` prefix mismatch in both directions. Returns
    (mapping, report) where mapping maps timeseries id → network id for every
    id that matches (identically or after prefix fixing).
    """
    ts_ids = set(map(str, ts_ids))
    network_ids = set(map(str, network_ids))
    mapping = {i: i for i in ts_ids & network_ids}
    unmatched = ts_ids - network_ids

    fixed = {}
    for i in unmatched:
        if f"nmi_{i}" in network_ids:
            fixed[i] = f"nmi_{i}"
        elif i.startswith("nmi_") and i[4:] in network_ids:
            fixed[i] = i[4:]
    mapping.update(fixed)

    report = {
        "n_ts_ids": len(ts_ids),
        "n_network_ids": len(network_ids),
        "n_exact": len(ts_ids & network_ids),
        "n_prefix_fixed": len(fixed),
        "ts_unmatched": sorted(ts_ids - set(mapping))[:20],
        "network_unmatched": sorted(network_ids - set(mapping.values()))[:20],
    }
    if fixed:
        logger.info("fixed nmi_ prefix on %d load ids (e.g. %s -> %s)",
                    len(fixed), *next(iter(fixed.items())))
    return mapping, report


# ---------------------------------------------------------------------------
# Pre-indexing (stage ⑤)
# ---------------------------------------------------------------------------
def modal_dt_minutes(timestamps):
    ts = pd.Series(pd.to_datetime(np.asarray(timestamps))).sort_values()
    d = ts.diff().dropna().dt.total_seconds() / 60.0
    return float(d.mode().iat[0]) if len(d) else 30.0


def pre_index(df_long, load_ids, ambient=None, out_path=None):
    """Build the per-substation NumPy bundle from the normalised long table.

    df_long   normalised long frame (whole feeder is fine)
    load_ids  the Load component ids of THIS substation (mapping already
              applied — see reconcile_load_ids)
    ambient   optional pd.Series of °C indexed by timestamp; aligned with
              interpolation onto the regular grid. None → filled with NaN
              (caller must supply a constant instead).
    out_path  optional .npz path (np.savez_compressed)

    Returns the dict of arrays (see module docstring).
    """
    load_ids = sorted(map(str, load_ids))
    sub = df_long[df_long["load_id"].isin(load_ids)]

    # One reshape for the whole year (never a per-timestep scan).
    wide_p = sub.pivot(index="timestamp", columns="load_id", values="real_power_w")
    wide_q = sub.pivot(index="timestamp", columns="load_id", values="reactive_power_var")

    # Columns: exactly this substation's loads, even if some have no data.
    wide_p = wide_p.reindex(columns=load_ids)
    wide_q = wide_q.reindex(columns=load_ids)

    # Rows: regular grid at the modal Δt so gaps become explicit.
    dt_min = modal_dt_minutes(wide_p.index)
    full_idx = pd.date_range(wide_p.index.min(), wide_p.index.max(),
                             freq=pd.Timedelta(minutes=dt_min))
    wide_p = wide_p.reindex(full_idx)
    wide_q = wide_q.reindex(full_idx)

    mask = wide_p.notna().values
    P = wide_p.fillna(0.0).values.astype(np.float32)
    Q = wide_q.fillna(0.0).values.astype(np.float32)

    if ambient is not None:
        amb = ambient.copy()
        amb.index = pd.to_datetime(amb.index)
        amb = amb[~amb.index.duplicated()].sort_index()
        theta = (amb.reindex(amb.index.union(full_idx))
                 .interpolate(method="time")
                 .reindex(full_idx)
                 .ffill().bfill()
                 .values.astype(np.float32))
    else:
        theta = np.full(len(full_idx), np.nan, dtype=np.float32)

    bundle = {
        "timestamps": full_idx.values.astype("datetime64[ns]").astype(np.int64),
        "load_ids": np.array(load_ids, dtype="<U32"),
        "P": P,
        "Q": Q,
        "theta_A": theta,
        "mask": mask,
        "dt_minutes": np.float64(dt_min),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **bundle)
    return bundle


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def timestamps_index(bundle):
    """Recover the pandas DatetimeIndex from a bundle."""
    return pd.DatetimeIndex(bundle["timestamps"].view("datetime64[ns]"))


def fingerprint(*byte_chunks):
    """SHA-256 over the given byte chunks (for cache/resume validation)."""
    h = hashlib.sha256()
    for c in byte_chunks:
        h.update(c)
    return "sha256:" + h.hexdigest()


def file_fingerprint(*paths):
    h = hashlib.sha256()
    for p in map(Path, paths):
        h.update(p.name.encode())
        if p.exists():
            h.update(p.read_bytes())
    return "sha256:" + h.hexdigest()
