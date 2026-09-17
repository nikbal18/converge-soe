"""Where does thermal overtake voltage as the binding constraint?

Reads the penetration ladder (1.0x baseline plus the pen* runs) and answers
SQ1 with a number: the export multiplier at which a substation's peak thermal
utilisation reaches the DTR cap. Below that multiplier the DTR recovers
essentially nothing because voltage rise is what curtails; above it the DTR is
the thing setting the envelope.

Text and CSV by default. No solving. Reads, per run:
    comparison/metrics_by_substation.csv
    scenarios/doe_dtr/<SUB>/thermal.parquet   (K2_actual, K2_max, dtr_status)
    config_resolved.yaml                      (the comparability guard)
    synthetic_report.json                     (coverage, and the donor check)

    python tools/aggregate_penetration.py --feeder lexcen
    python tools/aggregate_penetration.py --feeder lexcen --plot
    python tools/aggregate_penetration.py --feeder saunders --baseline peak7_summer

THE GUARD, AND WHY IT IS NOT OPTIONAL
    The 1.0x rung was run in August, the rest today. If any setting that
    affects comparability drifted in between — series reduction, tap handling,
    feeder-head voltage, envelope cap, synthetic seed, solver soft limits — the
    ladder is measuring that drift as well as penetration, and the crossover is
    meaningless. This script compares every such key across rungs and refuses
    to write anything on a mismatch. --force downgrades the refusal to a
    warning; use it only when you can name the difference and argue it is
    harmless.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import yaml

# feeder alias -> output directory, matching TARGET_FEEDERS in run_all_feeders.py
FEEDER_DIRS = {
    "lexcen":    "GOLDCR_8HB_LEXCEN",
    "saunders":  "GOLDCR_8FB_SAUNDERS",
    "birrigai":  "GOLDCR_8+MB_BIRRIGAI",
    "streeton":  "WODEN_8+NB_STREETON",
    "wanganeen-bunburung": "GOLDCR_8+NB_WANGANEE",
    # wellington-gurrang is deliberately absent: 115 of 116 NMIs on its two
    # ageing-dominant substations are donor-sampled, so the envelope controls
    # nothing there and the soft current limit absorbs the overload. Scaling an
    # uncontrolled substation harder measures the soft limit, not the network.
}
DEFAULT_BASELINE = {"lexcen": "lexcen7_summer"}   # everything else: peak7_summer

THERMAL_BOUND_PCT = 95.0   # peak utilisation at or above this = the DTR cap is binding
MAX_SYNTHETIC_PCT = 90.0   # above this the substation has no customers to control

# Keys that must be identical across every rung. Anything not listed is either
# a scaling key (the point of the exercise) or a scheduling key that cannot
# change a result: jobs, flush_every, resume, csv_mirror.
GUARDED_PREFIXES = (
    "scenarios", "envelope_abs_max", "tx_limit", "thermal.", "network.",
    "synthetic.", "solver.", "timeseries.", "ambient.source",
)


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = tuple(v) if isinstance(v, list) else v
    return out


def guarded(cfg):
    flat = flatten(cfg)
    return {k: v for k, v in flat.items() if k.startswith(GUARDED_PREFIXES)}


def read_config(run_dir):
    p = os.path.join(run_dir, "config_resolved.yaml")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return yaml.safe_load(fh)


def synthetic_share(run_dir):
    """{substation: (n_loads, n_metered, pct_synthetic)}"""
    out = {}
    p = os.path.join(run_dir, "synthetic_report.json")
    if not os.path.exists(p):
        return out
    try:
        for e in json.load(open(p)):
            tot = e.get("n_network_real_loads") or 0
            wd = e.get("n_with_data") or 0
            if tot:
                out[e["substation"]] = (tot, wd, 100.0 * (tot - wd) / tot)
    except Exception as e:                                      # noqa: BLE001
        print(f"WARN: could not read synthetic_report.json in {run_dir}: {e}")
    return out


_THERMAL_WARNED = set()


def thermal_stats(run_dir, sub):
    """Peak and mean DTR utilisation, and what dtr_status says was binding."""
    p = os.path.join(run_dir, "scenarios", "doe_dtr", sub, "thermal.parquet")
    blank = dict(util_max_pct=float("nan"), util_mean_pct=float("nan"),
                 pct_intervals_at_cap=float("nan"),
                 pct_intervals_thermal_binding=float("nan"))
    if not os.path.exists(p):
        return blank
    try:
        t = pd.read_parquet(p, columns=["K2_actual", "K2_max", "dtr_status"])
    except Exception as e:                                      # noqa: BLE001
        # Once per run, not once per substation: the usual cause is a missing
        # parquet engine, which fails identically 18 times and buries the report.
        if run_dir not in _THERMAL_WARNED:
            _THERMAL_WARNED.add(run_dir)
            print(f"WARN: {run_dir}: could not read thermal.parquet "
                  f"(first failure at {sub}) — {str(e).splitlines()[0]}")
        return blank
    u = pd.to_numeric(t["K2_actual"] / t["K2_max"], errors="coerce")
    u = u.replace([float("inf"), float("-inf")], pd.NA).dropna().clip(lower=0)
    st = t["dtr_status"].astype(str)
    if not len(u):
        return blank
    return dict(
        util_max_pct=100 * float(u.max()),
        util_mean_pct=100 * float(u.mean()),
        pct_intervals_at_cap=100 * float((u > 0.95).mean()),
        # already_over_limit counts too: the cap is binding, it is just being
        # breached through the soft limit rather than respected.
        pct_intervals_thermal_binding=100 * float(
            st.isin(["thermal_binding", "already_over_limit"]).mean()),
    )


def collect_run(run_dir, level):
    metrics = os.path.join(run_dir, "comparison", "metrics_by_substation.csv")
    if not os.path.exists(metrics):
        print(f"SKIP level {level}x: no comparison/metrics_by_substation.csv in "
              f"{run_dir} (the analysis stage never completed — the run was "
              f"interrupted, not finished)")
        return None
    m = pd.read_csv(metrics)
    syn = synthetic_share(run_dir)

    def piv(col):
        if col not in m.columns:
            return pd.DataFrame()
        return m.pivot(index="substation", columns="scenario", values=col)

    age, curt = piv("ageing_hours"), piv("E_curt_kwh")
    peak, des = piv("peak_theta_HS_C"), piv("E_des_export_kwh")

    def g(df, sub, sc):
        try:
            return float(df.loc[sub, sc])
        except Exception:                                       # noqa: BLE001
            return float("nan")

    rows = []
    for sub in age.index:
        n_loads, n_metered, pct_syn = syn.get(sub, (None, None, float("nan")))
        c_static, c_dtr = g(curt, sub, "doe_static"), g(curt, sub, "doe_dtr")
        recovered = c_static - c_dtr
        rows.append(dict(
            level=level, substation=sub,
            **thermal_stats(run_dir, sub),
            ageing_h_bau=g(age, sub, "bau"),
            ageing_h_static=g(age, sub, "doe_static"),
            ageing_h_dtr=g(age, sub, "doe_dtr"),
            peak_thetaHS_bau=g(peak, sub, "bau"),
            peak_thetaHS_dtr=g(peak, sub, "doe_dtr"),
            desired_export_kwh=g(des, sub, "bau"),
            curt_static_kwh=c_static, curt_dtr_kwh=c_dtr,
            recovered_kwh=recovered,
            recovery_frac_pct=(100 * recovered / c_static)
            if c_static and c_static == c_static and c_static > 0 else float("nan"),
            n_loads=n_loads, n_metered=n_metered, pct_synthetic=pct_syn,
        ))
    return pd.DataFrame(rows)


def crossover(df, threshold=THERMAL_BOUND_PCT):
    """Per substation: the export multiplier at which peak utilisation hits the cap.

    Linear interpolation between the two rungs that straddle the threshold.
    Utilisation is monotonic in export here (more export, more current, more
    heat), so a straight line between adjacent rungs is a fair reading. It is
    still an interpolation on four points — quote it to one decimal place, and
    never outside the range actually run.
    """
    out = []
    for sub, g in df.groupby("substation"):
        g = g.sort_values("level")
        lv = g["level"].to_numpy(float)
        u = g["util_max_pct"].to_numpy(float)
        if pd.isna(u).all():
            out.append(dict(substation=sub, crossover_level=float("nan"),
                            note="no thermal data"))
            continue
        if u[0] >= threshold:
            out.append(dict(substation=sub, crossover_level=lv[0],
                            note="already thermally bound at the lowest rung"))
            continue
        if u[-1] < threshold:
            out.append(dict(substation=sub, crossover_level=float("nan"),
                            note=f"still voltage-limited at {lv[-1]:g}x "
                                 f"(peak utilisation {u[-1]:.0f}%)"))
            continue
        for i in range(1, len(lv)):
            if u[i] >= threshold > u[i - 1]:
                span = u[i] - u[i - 1]
                frac = (threshold - u[i - 1]) / span if span else 0.0
                out.append(dict(substation=sub,
                                crossover_level=lv[i - 1] + frac * (lv[i] - lv[i - 1]),
                                note="interpolated"))
                break
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feeder", default="lexcen,saunders,birrigai",
                    help="comma list from " + ", ".join(sorted(FEEDER_DIRS)) +
                         ". All three defaults share the same 27 Dec - 3 Jan "
                         "summer week at 1.0x, so they ladder together.")
    ap.add_argument("--out-root", default="out")
    ap.add_argument("--baseline", default=None,
                    help="run-id of the 1.0x rung (default: lexcen7_summer for "
                         "lexcen, peak7_summer otherwise)")
    ap.add_argument("--levels", default="1.5,2.0,3.0",
                    help="the ladder above 1.0x; each maps to run-id pen<level>_summer")
    ap.add_argument("--threshold", type=float, default=THERMAL_BOUND_PCT)
    ap.add_argument("--out-dir", default=None,
                    help="default: out/penetration/<feeder>")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="aggregate even if the comparability guard fails")
    args = ap.parse_args()

    feeders = [f.strip() for f in args.feeder.split(",") if f.strip()]
    unknown = [f for f in feeders if f not in FEEDER_DIRS]
    if unknown:
        print(f"Unknown feeder(s): {', '.join(unknown)}. "
              f"Known: {', '.join(sorted(FEEDER_DIRS))}")
        return 1

    if len(feeders) > 1 and (args.out_dir or args.baseline):
        print("--out-dir and --baseline name one feeder's run, so they only "
              "make sense with a single --feeder. Drop them, or run one feeder "
              "at a time.")
        return 1

    results = {}
    for feeder in feeders:
        print(f"\n{'#' * 78}\n# {feeder.upper()}\n{'#' * 78}")
        r = run_one(args, feeder)
        if r is not None:
            results[feeder] = r

    if not results:
        print("\nNo feeder produced a usable ladder.")
        return 1

    if len(results) > 1:
        compare_feeders(results, args)
    return 0


def compare_feeders(results, args):
    """Does the crossover move with metering coverage?

    The mechanism to test: fewer metered customers means fewer customers the
    envelope can actually curtail, so the transformer sees more uncontrolled
    export and reaches its thermal cap at a lower multiplier. If that shows up
    across three feeders it is a result about where DTR matters, not a
    robustness check. If it does not, the crossover is a property of the
    network rather than the metering, which is also worth knowing.
    """
    rows = []
    for feeder, (df, summary, cross) in results.items():
        base = df[df["level"] == df["level"].min()]
        n_loads, n_metered = base["n_loads"].sum(), base["n_metered"].sum()
        named = cross["crossover_level"].dropna()
        still = cross["note"].str.startswith("still").sum()
        top = summary["level"].max()
        rows.append({
            "feeder": feeder,
            "n_substations": base["substation"].nunique(),
            "metering_coverage_pct": 100 * n_metered / n_loads if n_loads else float("nan"),
            "median_crossover": named.median() if len(named) else float("nan"),
            "n_crossing": len(named),
            f"n_still_voltage_limited_at_{top:g}x": int(still),
            "share_bound_at_top_rung_pct":
                float(summary.loc[summary["level"] == top,
                                  "share_thermally_bound_pct"].iloc[0]),
            "recovery_at_top_rung_pct":
                float(summary.loc[summary["level"] == top,
                                  "recovery_frac_pct"].iloc[0]),
        })
    comp = pd.DataFrame(rows).sort_values("metering_coverage_pct", ascending=False)

    out_dir = os.path.join(args.out_root, "penetration")
    os.makedirs(out_dir, exist_ok=True)
    comp.to_csv(os.path.join(out_dir, "crossover_by_feeder.csv"), index=False)

    print(f"\n{'#' * 78}\n# ACROSS FEEDERS\n{'#' * 78}")
    print(comp.round(2).to_string(index=False))
    print(f"\nWritten to {out_dir}/crossover_by_feeder.csv")

    cov = comp["metering_coverage_pct"]
    xo = comp["median_crossover"]
    ok = cov.notna() & xo.notna()
    if ok.sum() >= 3:
        r = cov[ok].corr(xo[ok])
        print(f"\nCoverage vs crossover, Pearson r = {r:.2f} on {int(ok.sum())} "
              f"feeders. Three points is an observation, not a regression — "
              f"report the three numbers and describe the direction. Do not "
              f"fit a line to it and do not quote this r in the thesis.")


def run_one(args, feeder):
    """Aggregate one feeder's ladder. Returns (df, summary, cross) or None."""

    fdir = os.path.join(args.out_root, FEEDER_DIRS[feeder])
    baseline = args.baseline or DEFAULT_BASELINE.get(feeder, "peak7_summer")

    runs = [(1.0, os.path.join(fdir, baseline))]
    for s in args.levels.split(","):
        s = s.strip()
        if s:
            runs.append((float(s), os.path.join(fdir, f"pen{s.replace('.', '')}_summer")))

    # --- comparability guard -------------------------------------------------
    print("Ladder")
    ref_keys, ref_label, problems = None, None, []
    for level, run_dir in runs:
        cfg = read_config(run_dir)
        if cfg is None:
            problems.append(f"{level}x: no config_resolved.yaml in {run_dir}")
            print(f"  {level:>4g}x  MISSING   {run_dir}")
            continue
        declared = (cfg.get("scaling") or {}).get("export", 1.0)
        imp = (cfg.get("scaling") or {}).get("import", 1.0)
        print(f"  {level:>4g}x  scaling.export={declared:<5g} import={imp:<5g} {run_dir}")
        if abs(float(declared) - level) > 1e-9:
            problems.append(f"{level}x: run says scaling.export={declared}. "
                            f"The run-id does not match what was run.")
        if abs(float(imp) - 1.0) > 1e-9:
            problems.append(f"{level}x: scaling.import={imp}, expected 1.0. "
                            f"Demand was scaled too, so this is not a pure "
                            f"penetration rung.")
        keys = guarded(cfg)
        if ref_keys is None:
            ref_keys, ref_label = keys, f"{level}x"
        else:
            for k in sorted(set(ref_keys) | set(keys)):
                a, b = ref_keys.get(k, "<absent>"), keys.get(k, "<absent>")
                if a != b:
                    problems.append(f"{level}x: {k} = {b!r}, but {ref_label} "
                                    f"has {a!r}")

    if problems:
        print("\nCOMPARABILITY PROBLEMS")
        for p in problems:
            print("  !", p)
        if not args.force:
            print("\nRefusing to aggregate. Every rung must differ ONLY in "
                  "scaling.export, or the crossover is measuring configuration "
                  "drift as well as penetration. Fix the run, or pass --force "
                  "if you can name the difference and argue it is harmless.")
            return None
        print("\n--force given: continuing anyway. State this in the thesis.")
    else:
        print("\nComparability guard: PASS (rungs differ only in scaling.export)")

    # --- collect -------------------------------------------------------------
    frames = [f for f in (collect_run(rd, lv) for lv, rd in runs) if f is not None]
    if len(frames) < 2:
        print("\nNeed at least two completed rungs to say anything. Stopping.")
        return None
    df = pd.concat(frames, ignore_index=True)

    # A substation whose load is almost entirely donor-sampled has almost no
    # customers receiving an envelope, so the DOE structurally cannot constrain
    # it and scaling it harder measures the soft current limit instead.
    excluded = sorted(set(df.loc[df["pct_synthetic"] > MAX_SYNTHETIC_PCT, "substation"]))
    if excluded:
        print(f"\nExcluded (>{MAX_SYNTHETIC_PCT:g}% donor-sampled, no customers "
              f"to control): {', '.join(excluded)}")
        df = df[~df["substation"].isin(excluded)]

    # Scaling happens at stage 2b, BEFORE substation selection and donor
    # sampling, so in principle a rung could draw different donors and the
    # ladder would be measuring the draw as well as the penetration. The seed is
    # pinned at 42, but donor matching also looks at magnitudes (match_pv,
    # size_diversity), so check rather than assume.
    drift = []
    for sub, g in df.groupby("substation"):
        for col in ("n_loads", "n_metered"):
            vals = set(g[col].dropna().tolist())
            if len(vals) > 1:
                drift.append(f"{sub}: {col} varies across rungs {sorted(vals)}")
    if drift:
        print("\n!!! DONOR ASSIGNMENT IS NOT STABLE ACROSS RUNGS")
        for d in drift:
            print("  !", d)
        print("    The rungs have different customer populations, so part of "
              "any difference between them is the draw, not the penetration. "
              "Resolve this before quoting a crossover.")
    else:
        print("Donor assignment: stable across every rung.")

    # Only substations present at EVERY rung can be compared across the ladder.
    n_levels = df["level"].nunique()
    complete = df.groupby("substation")["level"].nunique() == n_levels
    dropped = sorted(complete[~complete].index)
    if dropped:
        print(f"Dropped (missing from at least one rung): {', '.join(dropped)}")
        df = df[df["substation"].isin(complete[complete].index)]

    if df.empty:
        print("\nNothing left to aggregate.")
        return None

    # --- summarise -----------------------------------------------------------
    # Built as an explicit loop rather than groupby().apply(): the apply path
    # needs include_groups= on pandas 2.2+ and warns without it on 2.1, and this
    # has to run on whatever pandas the conda env happens to carry.
    srows = []
    for level, g in df.groupby("level"):
        bound = g["util_max_pct"] >= args.threshold
        age_bau = g["ageing_h_bau"].sum()
        curt_s = g["curt_static_kwh"].sum()
        srows.append({
            "level": level,
            "n_substations": len(g),
            "n_thermally_bound": int(bound.sum()),
            "share_thermally_bound_pct": 100 * float(bound.mean()),
            "median_util_max_pct": g["util_max_pct"].median(),
            "max_util_max_pct": g["util_max_pct"].max(),
            # weighted by ageing, because that is the claim the cost chapter makes
            "share_of_bau_ageing_at_bound_pct":
                100 * g.loc[bound, "ageing_h_bau"].sum() / age_bau
                if age_bau else float("nan"),
            "total_curt_static_kwh": curt_s,
            "total_recovered_kwh": g["recovered_kwh"].sum(),
            "recovery_frac_pct":
                100 * g["recovered_kwh"].sum() / curt_s if curt_s else float("nan"),
            "total_ageing_h_bau": age_bau,
            "total_ageing_h_static": g["ageing_h_static"].sum(),
            "total_ageing_h_dtr": g["ageing_h_dtr"].sum(),
        })
    summary = pd.DataFrame(srows).sort_values("level").reset_index(drop=True)

    cross = crossover(df, args.threshold)

    out_dir = args.out_dir or os.path.join(args.out_root, "penetration", feeder)
    os.makedirs(out_dir, exist_ok=True)
    df.sort_values(["substation", "level"]).to_csv(
        os.path.join(out_dir, "penetration_by_substation.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "penetration_summary.csv"), index=False)
    cross.sort_values("crossover_level").to_csv(
        os.path.join(out_dir, "crossover.csv"), index=False)

    # --- report --------------------------------------------------------------
    pd.set_option("display.width", 200)
    print(f"\n{'=' * 78}\nPER LEVEL — {feeder}\n{'=' * 78}")
    print(summary.round(2).to_string(index=False))

    print(f"\n{'=' * 78}\nCROSSOVER (peak utilisation reaches {args.threshold:g}%)"
          f"\n{'=' * 78}")
    print(cross.sort_values("crossover_level").round(2).to_string(index=False))

    named = cross["crossover_level"].dropna()
    if len(named):
        print(f"\nMedian crossover across {len(named)} substations that reach the "
              f"cap: {named.median():.1f}x")
    never = cross[cross["crossover_level"].isna() & cross["note"].str.startswith("still")]
    if len(never):
        print(f"{len(never)} of {len(cross)} substations are STILL voltage-limited "
              f"at the top of the ladder. That is a result, not a gap: it is the "
              f"quantified version of 'voltage binds first'.")

    print(f"\nWritten to {out_dir}/")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:                                  # noqa: BLE001
            print(f"WARN: no plot ({e})")
            return df, summary, cross
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        for sub, g in df.groupby("substation"):
            g = g.sort_values("level")
            ax1.plot(g["level"], g["util_max_pct"], marker="o", lw=1, alpha=0.7)
            ax2.plot(g["level"], g["recovery_frac_pct"], marker="o", lw=1, alpha=0.7)
        ax1.axhline(args.threshold, ls="--", c="k", lw=1)
        ax1.set_xlabel("DPV export multiplier")
        ax1.set_ylabel("peak thermal utilisation (%)")
        ax1.set_title("When the thermal cap starts to bind")
        ax2.set_xlabel("DPV export multiplier")
        ax2.set_ylabel("curtailment recovered by the DTR (%)")
        ax2.set_title("What the DTR buys, by penetration")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"penetration_crossover.{ext}"), dpi=150)
        print(f"Figure: {out_dir}/penetration_crossover.png")
    return df, summary, cross


if __name__ == "__main__":
    sys.exit(main())
