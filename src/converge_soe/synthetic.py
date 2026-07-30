"""Synthetic profiles for network NMIs that have no interval-meter data.

The meter export only covers NMIs that actually have measurements, which is not
every NMI on the network. Those without data are absent from ``nmi_index``
(stage ③ only iterates *matched* timeseries ids) and therefore are not even
columns in the pre-indexed bundle (stage ⑤). They contribute nothing to the
power flow, so the feeder looks unloaded and the resulting DOEs are far too
generous. On the Gold Creek LV network 72 of 121 Loads (60%) are in this state,
carrying an estimated 61% of the real feeder load.

This module fills that gap. It runs between stage ⑤ and stage ⑥ and **adds
columns** to each substation's bundle.

Method — donor sampling (default)
---------------------------------
Each unmapped NMI is assigned a *real observed profile* drawn from the metered
NMIs on the feeder. Sampling rather than a mean profile, because the intervals
that bind the envelope constraints are the peak and minimum-demand ones, and
averaging smooths exactly those away. Drawing without replacement preserves the
observed distribution instead of over-weighting repeat picks.

By default (``allow_feeder_donors``) the pool is the whole feeder rather than
the target's own substation: consumption does not vary much geographically
within a feeder, so a donor two substations away is as representative as one
next door, and the larger pool avoids the repeated reuse that would understate
coincidence diversity. On Gold Creek that is 376 donors instead of 49, for 72
gaps.

Method — transformer disaggregation (fallback)
----------------------------------------------
When a substation has fewer than ``min_donors`` usable donors, sampling is
meaningless. The substation's measured transformer demand, minus the metered
NMIs already accounted for, is split into profiles that sum exactly to that
target at every timestep (log-normal size diversity, smoothed temporal noise,
column-wise normalisation).

Why not the ``ESTIMATION - METER_SUMMATION`` residual
----------------------------------------------------
It looks like the unmetered load but is not. ``ESTIMATION / METER_SUMMATION``
is a near-constant ~1.03, rising to ~2.3 when ``METER_COUNT`` collapses. It is
a multiplicative gross-up for temporarily *missing meter reads*, not for
permanently unmetered customers, and being multiplicative it flips sign with
the feeder — positive importing, negative exporting. Using it would give
unmetered customers negative load during solar hours, biasing the envelopes in
the unsafe direction. Do not reintroduce it.

Provenance
----------
``mask`` stays False for every synthetic column (it is the record that a value
was measured), and a new per-column boolean array ``synthetic`` marks which
columns were invented. Synthetic NMIs must never receive an envelope: callers
pass ``participant_load_ids`` = the ids with real data, and
``DoeSolver._calculate_bus_loads_kw`` adds their active power as background
load instead.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import timeseries as tsm

logger = logging.getLogger(__name__)

# Diversity flags
OK, WARN, FALLBACK, NONE = "OK", "WARN", "FALLBACK", "NONE"

DEFAULTS = {
    "enabled": True,
    "min_donors": 5,
    "match_pv": True,
    "without_replacement": True,
    "include_placeholder_loads": False,
    "allow_feeder_donors": True,
    "max_donor_share": 0.20,
    "on_missing": "error",          # error | warn_zero
    "seed": 42,
    "transformer_series": {},
    "size_diversity": 0.6,
    "temporal_diversity": 0.15,
    "power_factor": 0.95,
}


def config(cfg):
    """Merge the ``synthetic`` block of a run config over the defaults."""
    out = dict(DEFAULTS)
    out.update((cfg or {}).get("synthetic", {}) or {})
    return out


# ---------------------------------------------------------------------------
# Network population
# ---------------------------------------------------------------------------
def network_loads(sub_ej):
    """{component_id: load_dict} for every Load in a substation ejson."""
    return {cid: cd
            for cid, comp in sub_ej["components"].items()
            for ctype, cd in comp.items() if ctype == "Load"}


def split_placeholders(loads):
    """(real, placeholder) — placeholders are cim_to_json LV dead-end stubs."""
    real, ph = {}, {}
    for cid, cd in loads.items():
        ud = cd.get("user_data") or {}
        (ph if ud.get("placeholder") else real)[cid] = cd
    return real, ph


def _reconcile(ids, pool):
    """Map ids onto pool, fixing the ``nmi_`` prefix in either direction."""
    pool = set(map(str, pool))
    out = {}
    for i in map(str, ids):
        if i in pool:
            out[i] = i
        elif f"nmi_{i}" in pool:
            out[i] = f"nmi_{i}"
        elif i.startswith("nmi_") and i[4:] in pool:
            out[i] = i[4:]
    return out


def find_gaps(sub_ej, bundle, include_placeholders=False):
    """Loads in the network model with no column in the bundle.

    Returns (gaps, loads_considered, n_real, n_placeholder).
    """
    loads = network_loads(sub_ej)
    real, ph = split_placeholders(loads)
    considered = dict(real)
    if include_placeholders:
        considered.update(ph)

    have = set(map(str, bundle["load_ids"]))
    # Reconcile so a prefix mismatch is never mistaken for a missing NMI.
    matched = set(_reconcile(have, considered).values())
    gaps = sorted(set(map(str, considered)) - matched)
    return gaps, considered, len(real), len(ph)


# ---------------------------------------------------------------------------
# Profile synthesis
# ---------------------------------------------------------------------------
def _diverse_profiles(n, total_t, rng, size_diversity, temporal_diversity,
                      window):
    """n profiles summing exactly to ``total_t`` at every timestep."""
    T = len(total_t)
    if n == 0:
        return np.zeros((0, T))
    weights = rng.lognormal(0.0, size_diversity, n)
    raw = np.empty((n, T))
    for c in range(n):
        noise = rng.normal(0.0, temporal_diversity, T)
        smoothed = (pd.Series(noise)
                    .rolling(window=window, center=True, min_periods=1)
                    .mean().values)
        raw[c] = np.maximum(weights[c] * (1.0 + smoothed), 1e-6)
    col = raw.sum(axis=0)
    return np.where(total_t > 0, raw / col * total_t[np.newaxis, :], 0.0)


def _load_transformer_series(spec, index, dt_minutes):
    """Read a transformer timeseries and align it onto ``index`` (kW)."""
    if isinstance(spec, (str, Path)):
        spec = {"path": str(spec)}
    path = Path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"transformer series not found: {path}")

    col = spec.get("column", "METER_SUMMATION (kW)")
    date_col = spec.get("date_column", "date")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if col not in df.columns:
        raise KeyError(f"{path.name}: no column {col!r}; found {list(df.columns)}")
    if "ESTIMATION" in col.upper():
        logger.warning(
            "transformer series column %r is meter-derived: it is a gross-up of "
            "the very meters being subtracted, so the fallback target is "
            "circular. Prefer an independent SCADA measurement.", col)

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=spec.get("dayfirst", True))
    s = (df.set_index(date_col)[col]
         .astype(float)
         .sort_index())
    s = s[~s.index.duplicated()]
    # Align onto the bundle grid: interpolate in time, then fill the edges.
    return (s.reindex(s.index.union(index))
            .interpolate(method="time")
            .reindex(index)
            .ffill().bfill()
            .values)


# ---------------------------------------------------------------------------
# Per-substation entry point
# ---------------------------------------------------------------------------
def synthesise_substation(sub_ej, bundle, scfg, substation="", rng=None,
                          feeder_donors=None):
    """Add columns to ``bundle`` for every Load with no meter data.

    ``feeder_donors`` is an optional ``(ids, P, Q)`` bank of complete profiles
    from elsewhere on the feeder, used only when ``allow_feeder_donors`` is on
    and this substation cannot cover its own gaps. Substation-local donors are
    always preferred.

    Returns (bundle, report). ``bundle`` is a new dict; the input is not
    mutated. ``report`` records counts, method, diversity flag and the full
    donor -> target assignment.
    """
    rng = rng or np.random.default_rng(scfg.get("seed", 42))
    bundle = dict(bundle)

    gaps, considered, n_real, n_ph = find_gaps(
        sub_ej, bundle, scfg.get("include_placeholder_loads", False))

    load_ids = [str(x) for x in bundle["load_ids"]]
    P, Q, mask = bundle["P"], bundle["Q"], bundle["mask"]
    T = P.shape[0]
    index = tsm.timestamps_index(bundle)
    dt_min = float(bundle.get("dt_minutes", 30.0))
    window = max(1, round(60 / max(dt_min, 1e-9)))

    synthetic = bundle.get("synthetic")
    synthetic = (np.zeros(len(load_ids), dtype=bool) if synthetic is None
                 else np.asarray(synthetic, dtype=bool).copy())

    report = {
        "substation": substation,
        "n_network_real_loads": n_real,
        "n_network_placeholder_loads": n_ph,
        "placeholders_included": bool(scfg.get("include_placeholder_loads", False)),
        "n_with_data": int(mask.any(axis=0).sum()) if mask.size else 0,
        "n_gaps": len(gaps),
        "method": None,
        "flag": NONE,
        "n_donors": 0,
        "n_donors_reused": 0,
        "donor_assignment": {},
        "notes": [],
    }

    if not gaps:
        report["method"] = "none"
        report["flag"] = OK
        return bundle, report

    # --- donor pool ------------------------------------------------------
    # Usable donors are existing columns with COMPLETE real coverage; a
    # half-empty donor would inject spurious zeros into the network.
    complete = mask.all(axis=0) if mask.size else np.zeros(len(load_ids), bool)
    donor_idx = [j for j in range(len(load_ids)) if complete[j] and not synthetic[j]]
    d_ids = [load_ids[j] for j in donor_idx]
    d_P = P[:, donor_idx] if donor_idx else np.zeros((T, 0), P.dtype)
    d_Q = Q[:, donor_idx] if donor_idx else np.zeros((T, 0), Q.dtype)
    n_local = len(d_ids)

    # Draw from the whole feeder's metered population, not just this
    # substation's. Customer consumption does not vary much geographically
    # within a feeder, so a donor two substations away is as representative as
    # one next door, and the larger pool avoids reusing the same profile many
    # times (which would understate coincidence diversity). Set
    # allow_feeder_donors: false to restrict to substation-local donors.
    n_widened = 0
    if scfg.get("allow_feeder_donors", True) and feeder_donors:
        f_ids, f_P, f_Q = feeder_donors
        local = set(d_ids)
        keep = [k for k, fid in enumerate(f_ids)
                if fid not in local and f_P.shape[0] == T]
        if keep:
            d_ids = d_ids + [f_ids[k] for k in keep]
            d_P = np.concatenate([d_P, f_P[:, keep]], axis=1)
            d_Q = np.concatenate([d_Q, f_Q[:, keep]], axis=1)
            n_widened = len(keep)

    report["n_donors"] = len(d_ids)
    report["n_donors_local"] = n_local
    report["n_donors_from_feeder"] = n_widened
    if n_widened:
        report["notes"].append(
            f"donor pool drawn feeder-wide: {n_local} local + {n_widened} from "
            f"other substations, for {len(gaps)} gap(s)")

    # Evaluated against the FEEDER-WIDE pool, so this is now a guard against a
    # feeder with almost no metered NMIs at all, not against a thin substation.
    # It should effectively never fire on a real export.
    min_donors = int(scfg.get("min_donors", 5))

    if len(d_ids) < min_donors:
        new_P, new_Q, rep = _fallback(gaps, P, mask, index, dt_min, window,
                                      scfg, substation, rng)
        report.update(rep)
    else:
        new_P, new_Q, rep = _sample(gaps, considered, d_ids, d_P, d_Q,
                                    scfg, rng)
        report.update(rep)

    # --- splice columns in, keeping everything sorted and aligned --------
    all_ids = load_ids + list(gaps)
    P2 = np.concatenate([P, new_P.astype(P.dtype)], axis=1)
    Q2 = np.concatenate([Q, new_Q.astype(Q.dtype)], axis=1)
    mask2 = np.concatenate(
        [mask, np.zeros((T, len(gaps)), dtype=bool)], axis=1)
    syn2 = np.concatenate([synthetic, np.ones(len(gaps), dtype=bool)])

    order = np.argsort(np.array(all_ids, dtype=object), kind="stable")
    bundle["load_ids"] = np.array([all_ids[i] for i in order], dtype="<U32")
    bundle["P"] = P2[:, order]
    bundle["Q"] = Q2[:, order]
    bundle["mask"] = mask2[:, order]
    bundle["synthetic"] = syn2[order]

    # --- diversity flag --------------------------------------------------
    if report["method"] == "transformer_fallback":
        report["flag"] = FALLBACK
    elif report["n_donors"] < len(gaps):
        report["flag"] = WARN
        report["notes"].append(
            f"{report['n_donors']} donors for {len(gaps)} gaps: donors reused, "
            "coincidence diversity understated")
    else:
        report["flag"] = OK

    share = scfg.get("max_donor_share", 0.20)
    # Only meaningful once there are enough gaps for the limit to be
    # satisfiable at all: with 3 gaps every donor covers 33% by definition.
    if report["donor_assignment"] and share and len(gaps) * share >= 1.0:
        counts = pd.Series(list(report["donor_assignment"].values())).value_counts()
        worst = counts.iloc[0] / max(len(gaps), 1)
        if worst > share:
            report["flag"] = WARN if report["flag"] == OK else report["flag"]
            report["notes"].append(
                f"donor {counts.index[0]} covers {100*worst:.0f}% of this "
                f"substation's synthetic load (limit {100*share:.0f}%)")

    return bundle, report


def _sample(gaps, considered, d_ids, d_P, d_Q, scfg, rng):
    """Donor sampling. Returns (new_P, new_Q, report_fragment)."""
    idx = list(range(len(d_ids)))
    pv = [j for j in idx if bool((d_P[:, j] < 0).any())]
    non_pv = [j for j in idx if j not in set(pv)]
    match_pv = bool(scfg.get("match_pv", True))
    without_repl = bool(scfg.get("without_replacement", True))

    used, assignment, picks = set(), {}, []
    n_from_der = 0
    for cid in gaps:
        ud = (considered.get(cid, {}).get("user_data") or {})
        want = bool(ud["der"]) if "der" in ud else None
        n_from_der += "der" in ud

        if match_pv and want is True and pv:
            cands = pv
        elif match_pv and want is False and non_pv:
            cands = non_pv
        else:
            cands = idx

        avail = [c for c in cands if c not in used] if without_repl else list(cands)
        if not avail:
            avail = list(cands)          # pool exhausted — allow reuse
        j = int(rng.choice(avail))
        used.add(j)
        picks.append(j)
        assignment[cid] = d_ids[j]

    new_P = np.stack([d_P[:, j] for j in picks], axis=1)
    new_Q = np.stack([d_Q[:, j] for j in picks], axis=1)
    return new_P, new_Q, {
        "method": "donor_sampling",
        "n_donors_with_pv": len(pv),
        "n_targets_with_der_data": int(n_from_der),
        "n_donors_reused": len(picks) - len(set(picks)),
        "donor_assignment": assignment,
    }


def _fallback(gaps, P, mask, index, dt_min, window, scfg, substation, rng):
    """Transformer disaggregation. Returns (new_P, new_Q, report_fragment)."""
    series = (scfg.get("transformer_series") or {}).get(substation)
    n = len(gaps)
    T = P.shape[0]

    if series is None:
        msg = (f"substation {substation!r}: only "
               f"{int(mask.all(axis=0).sum())} usable donor(s), below "
               f"min_donors={scfg.get('min_donors', 5)}, and no "
               f"synthetic.transformer_series configured for it. Either supply "
               f"a transformer measurement or lower min_donors.")
        if scfg.get("on_missing", "error") == "error":
            raise ValueError(msg)
        logger.warning("%s — filling with zeros (on_missing=warn_zero)", msg)
        return (np.zeros((T, n)), np.zeros((T, n)),
                {"method": "zero_fill", "notes": [msg]})

    tx_kw = _load_transformer_series(series, index, dt_min)

    # Target = transformer demand minus the metered NMIs already accounted
    # for, or those customers get counted twice.
    metered_kw = np.where(mask, P, 0.0).sum(axis=1) / 1000.0
    target_kw = tx_kw - metered_kw

    base_t = np.maximum(target_kw, 0.0)
    solar_t = np.maximum(-target_kw, 0.0)
    sd = float(scfg.get("size_diversity", 0.6))
    td = float(scfg.get("temporal_diversity", 0.15))

    base = _diverse_profiles(n, base_t, rng, sd, td, window)
    n_solar = n if solar_t.any() else 0
    solar = _diverse_profiles(n_solar, solar_t, rng, sd, td, window)
    net_kw = base - solar if n_solar else base

    err = float(np.abs(net_kw.sum(axis=0) - target_kw).max())
    if err > 1e-6:
        raise AssertionError(f"fallback sum mismatch: {err:.2e} kW")

    tan_phi = np.tan(np.arccos(float(scfg.get("power_factor", 0.95))))
    new_P = (net_kw * 1000.0).T
    return new_P, new_P * tan_phi, {
        "method": "transformer_fallback",
        "transformer_series": str(series),
        "mean_target_kw": round(float(target_kw.mean()), 2),
        "sum_check_max_error_kw": err,
        "notes": [f"disaggregated transformer demand across {n} NMIs"],
    }


# ---------------------------------------------------------------------------
# All substations
# ---------------------------------------------------------------------------
def _donor_bank(bundles):
    """(ids, P, Q) of every complete, real profile across the whole feeder."""
    ids, cols_p, cols_q = [], [], []
    for b in bundles.values():
        mask = b["mask"]
        if not mask.size:
            continue
        syn = np.asarray(b.get("synthetic", np.zeros(mask.shape[1], bool)), bool)
        complete = mask.all(axis=0)
        for j, lid in enumerate(map(str, b["load_ids"])):
            if complete[j] and not syn[j] and lid not in set(ids):
                ids.append(lid)
                cols_p.append(b["P"][:, j])
                cols_q.append(b["Q"][:, j])
    if not ids:
        return None
    return ids, np.stack(cols_p, axis=1), np.stack(cols_q, axis=1)


def synthesise_all(substations, bundles, cfg, log=logger.info):
    """Fill gaps in every substation bundle. Returns (bundles, reports)."""
    scfg = config(cfg)
    if not scfg.get("enabled", True):
        log("stage ⑤b synthesise: disabled by config")
        return bundles, []

    feeder_donors = _donor_bank(bundles) if scfg.get("allow_feeder_donors") \
        else None

    out, reports = {}, []
    for safe, bundle in bundles.items():
        sub_ej = substations.get(safe)
        if sub_ej is None:
            out[safe] = bundle
            continue
        # Seed per substation so adding or reordering substations does not
        # change the draw for the others.
        rng = np.random.default_rng(
            [int(scfg.get("seed", 42)), *map(ord, safe[:16])])
        b, rep = synthesise_substation(sub_ej, bundle, scfg, substation=safe,
                                       rng=rng, feeder_donors=feeder_donors)
        out[safe], _ = b, reports.append(rep)

        if rep["n_gaps"]:
            log(f"stage ⑤b synthesise: {safe}: {rep['n_gaps']} NMI(s) without "
                f"data filled by {rep['method']} from {rep['n_donors']} donor(s) "
                f"[{rep['flag']}]")
            for note in rep["notes"]:
                log(f"stage ⑤b synthesise: {safe}: {note}")
        else:
            log(f"stage ⑤b synthesise: {safe}: no gaps")
    return out, reports


def findings(reports):
    """Preflight-style findings so this surfaces in the normal run report."""
    from .preflight import finding, ERROR, WARN as W, INFO

    F = []
    for r in reports:
        scope = f"synthetic:{r['substation']}"
        if not r["n_gaps"]:
            continue
        pct = 100 * r["n_gaps"] / max(r["n_gaps"] + r["n_with_data"], 1)
        F.append(finding(
            "SY001", INFO if r["flag"] == OK else W, scope,
            f"{r['n_gaps']} of {r['n_gaps'] + r['n_with_data']} NMI(s) "
            f"({pct:.0f}%) have no meter data and were synthesised by "
            f"{r['method']}",
            {"donors": r["n_donors"], "reused": r["n_donors_reused"]},
            "these NMIs load the network but never receive an envelope"))
        if r["flag"] == WARN:
            F.append(finding("SY002", W, scope,
                             "; ".join(r["notes"]) or "low donor diversity",
                             {"n_donors": r["n_donors"], "n_gaps": r["n_gaps"]},
                             "widen the donor pool or lower min_donors"))
        if r["method"] == "zero_fill":
            F.append(finding("SY003", ERROR, scope,
                             "NMIs without data were filled with ZERO load",
                             {"n_gaps": r["n_gaps"]},
                             "configure synthetic.transformer_series, or accept "
                             "that the feeder is under-loaded in this run"))
    return F
