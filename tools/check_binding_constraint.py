"""What actually binds at the substations that carry the ageing?

Text-only. No matplotlib, no solving. Reads each run's
comparison/metrics_by_substation.csv and scenarios/doe_dtr/<SUB>/thermal.parquet.

The question: the cost chapter only needs the substations that dominate ageing.
If those are thermally bound (peak K2_actual/K2_max near 1) the annual bridge can
use the analytic C57.91 cap and needs no fitted voltage residual. If they are not,
the DTR is not what is driving them and that is a finding in its own right.

    python tools/check_binding_constraint.py
    python tools/check_binding_constraint.py --csv out/cross_feeder/binding_check.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

RUNS = [
    ("Lexcen (whole feeder, Feb 3d)", "GOLDCR_8HB_LEXCEN/summer_fixed"),
    ("Lexcen (max-demand week)",      "GOLDCR_8HB_LEXCEN/lexcen7_summer"),
    ("Lexcen (min-demand week)",      "GOLDCR_8HB_LEXCEN/lexcen7_winter"),
    ("Saunders (max-demand week)",    "GOLDCR_8FB_SAUNDERS/peak7_summer"),
    ("Saunders (min-demand week)",    "GOLDCR_8FB_SAUNDERS/peak7_winter"),
    ("Birrigai (max-demand week)",    "GOLDCR_8+MB_BIRRIGAI/peak7_summer"),
    ("Birrigai (min-demand week)",    "GOLDCR_8+MB_BIRRIGAI/peak7_winter"),
    ("Wellngtn (max-demand week)",    "GOLDCR_8+LB_WELLNGTN/peak7_summer"),
    ("Saunders (top-20)",             "GOLDCR_8FB_SAUNDERS/top20s_SAUNDERS_2"),
    ("Streeton (top-20)",             "WODEN_8+NB_STREETON/top20s_STREETON"),
    ("Magenta (top-20)",              "GOLDCR_8+SB_MAGENTA/top20s_MAGENTA"),
    ("Wanganee (top-20)",             "GOLDCR_8+NB_WANGANEE/top20s_WANGANEE"),
]

THERMAL_BOUND_PCT = 95.0   # peak utilisation above this = thermal cap is doing the work
COVER_PCT         = 90.0   # report the substations covering this much of each run's ageing


def synthetic_share(run_dir):
    """{substation: pct of network loads that had no meter data} from synthetic_report.json."""
    out = {}
    p = os.path.join(run_dir, "synthetic_report.json")
    if not os.path.exists(p):
        return out
    try:
        import json
        for e in json.load(open(p)):
            tot = e.get("n_network_real_loads") or 0
            wd = e.get("n_with_data") or 0
            if tot:
                out[e["substation"]] = (tot, wd, 100.0 * (tot - wd) / tot)
    except Exception:                                       # noqa: BLE001
        pass
    return out


def collect(out_root="out"):
    rows, notes = [], []
    for label, run in RUNS:
        run_dir = os.path.join(out_root, run)
        syn = synthetic_share(run_dir)
        metrics = os.path.join(run_dir, "comparison", "metrics_by_substation.csv")
        if not os.path.exists(metrics):
            notes.append(f"SKIP {label}: no comparison/metrics_by_substation.csv "
                         f"(analysis stage never completed)")
            continue
        m = pd.read_csv(metrics)
        age  = m.pivot(index="substation", columns="scenario", values="ageing_hours")
        curt = m.pivot(index="substation", columns="scenario", values="E_curt_kwh")
        peak = m.pivot(index="substation", columns="scenario", values="peak_theta_HS_C")
        bau_total = age.get("bau", pd.Series(dtype=float)).sum()

        for sub in age.index:
            th = os.path.join(run_dir, "scenarios", "doe_dtr", sub, "thermal.parquet")
            util_max = util_mean = share_at_cap = float("nan")
            status_mix = ""
            if os.path.exists(th):
                try:
                    t = pd.read_parquet(th, columns=["K2_actual", "K2_max", "dtr_status"])
                    u = (t["K2_actual"] / t["K2_max"]).replace([float("inf"), float("-inf")], pd.NA)
                    u = pd.to_numeric(u, errors="coerce").dropna().clip(lower=0)
                    if len(u):
                        util_max = 100 * u.max()
                        util_mean = 100 * u.mean()
                        share_at_cap = 100 * (u > 0.95).mean()
                    vc = t["dtr_status"].value_counts()
                    status_mix = ", ".join(f"{k}:{v}" for k, v in vc.items())
                except Exception as e:                      # noqa: BLE001
                    notes.append(f"WARN {sub} ({label}): could not read thermal.parquet — {e}")
            else:
                notes.append(f"WARN {sub} ({label}): no doe_dtr thermal.parquet")

            def g(df, sc):
                try:    return float(df.loc[sub, sc])
                except Exception:  return float("nan")

            a_bau = g(age, "bau")
            n_loads, n_metered, pct_syn = syn.get(sub, (None, None, float("nan")))
            rows.append(dict(
                run=label, substation=sub,
                bau_ageing_h=a_bau,
                share_of_run_ageing_pct=(100 * a_bau / bau_total) if bau_total else float("nan"),
                util_max_pct=util_max, util_mean_pct=util_mean,
                pct_intervals_at_cap=share_at_cap,
                peak_thetaHS_bau=g(peak, "bau"),
                peak_thetaHS_dtr=g(peak, "doe_dtr"),
                ageing_h_static=g(age, "doe_static"), ageing_h_dtr=g(age, "doe_dtr"),
                curt_static_kwh=g(curt, "doe_static"), curt_dtr_kwh=g(curt, "doe_dtr"),
                n_loads=n_loads, n_metered=n_metered, pct_synthetic=pct_syn,
                dtr_status_mix=status_mix,
            ))
    return pd.DataFrame(rows), notes


# A substation whose load is almost entirely synthetic has almost no customers
# receiving an envelope, so the DOE structurally cannot constrain it. Utilisation
# above ~105% means the solved current exceeded the DTR cap, which the soft current
# limit permits at a penalty — that is a data-coverage failure, not a thermal result.
MAX_SYNTHETIC_PCT = 90.0
UTIL_IMPOSSIBLE    = 105.0


def verdict(r):
    if pd.isna(r.util_max_pct):
        return "no data"
    if pd.notna(r.get("pct_synthetic")) and r.pct_synthetic >= MAX_SYNTHETIC_PCT:
        return "UNUSABLE (mostly synthetic, no envelope)"
    if r.util_max_pct > UTIL_IMPOSSIBLE:
        return "UNUSABLE (envelope cannot bind)"
    if r.util_max_pct >= THERMAL_BOUND_PCT:
        return "THERMAL binds"
    if r.util_max_pct >= 80:
        return "borderline"
    return "NOT thermal (voltage/envelope)"


# One run per substation, and only the high-stress runs: a min-demand week has
# ~1/50th the ageing of a max-demand week and must not carry equal weight.
CANONICAL_RUNS = [
    "Lexcen (max-demand week)", "Saunders (max-demand week)",
    "Birrigai (max-demand week)", "Wellngtn (max-demand week)",
    "Streeton (top-20)", "Magenta (top-20)",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="out")
    ap.add_argument("--csv", default=None, help="also write the full table here")
    a = ap.parse_args()

    df, notes = collect(a.out_root)
    if df.empty:
        print("No runs with a completed analysis stage found under", a.out_root)
        for n in notes: print(" ", n)
        return

    df["recovered_kwh"] = df.curt_static_kwh - df.curt_dtr_kwh
    df["verdict"] = df.apply(verdict, axis=1)

    for run, g in df.groupby("run", sort=False):
        g = g.sort_values("share_of_run_ageing_pct", ascending=False)
        cum = g.share_of_run_ageing_pct.cumsum()
        keep = g[cum.shift(1).fillna(0) < COVER_PCT]
        print(f"\n=== {run}")
        print(f"    substations carrying the first {COVER_PCT:.0f}% of BAU ageing "
              f"({len(keep)} of {len(g)}):")
        print(f"    {'substation':13} {'%ageing':>8} {'utilmax%':>9} {'utilmean%':>10} "
              f"{'%int@cap':>9} {'peakHS bau':>11} {'peakHS dtr':>11} {'recov kWh':>10}  verdict")
        for _, r in keep.iterrows():
            f = lambda x, d=1: ("%.*f" % (d, x)) if pd.notna(x) else "-"
            print(f"    {r.substation:13} {f(r.share_of_run_ageing_pct):>8} {f(r.util_max_pct):>9} "
                  f"{f(r.util_mean_pct):>10} {f(r.pct_intervals_at_cap):>9} "
                  f"{f(r.peak_thetaHS_bau):>11} {f(r.peak_thetaHS_dtr):>11} "
                  f"{f(r.recovered_kwh,0):>10}  {r.verdict}")

    print("\n" + "=" * 78)
    print("VERDICT \u2014 canonical runs only, one row per substation,")
    print("weighted by ABSOLUTE BAU ageing hours (not by within-run share)")
    c = df[df.run.isin(CANONICAL_RUNS)].dropna(subset=["bau_ageing_h"])
    c = c[c.bau_ageing_h > 0].sort_values("bau_ageing_h", ascending=False)
    c = c.drop_duplicates(subset=["substation"], keep="first")
    if c.empty:
        print("  no canonical runs present")
    else:
        tot = c.bau_ageing_h.sum()
        grp = c.groupby("verdict").bau_ageing_h.agg(["sum", "count"]).sort_values("sum", ascending=False)
        for v, row in grp.iterrows():
            print(f"  {v:42} {row['sum']:11.1f} h  {100*row['sum']/tot:5.1f}%  ({int(row['count'])} subs)")
        print(f"  {'TOTAL':42} {tot:11.1f} h")
        unusable = grp[grp.index.str.startswith("UNUSABLE")]["sum"].sum() if len(grp) else 0.0
        usable = tot - unusable
        th = grp.loc["THERMAL binds", "sum"] if "THERMAL binds" in grp.index else 0.0
        if usable > 0:
            print(f"\n  Of USABLE ageing ({usable:.1f} h), {100*th/usable:.1f}% sits at thermally-bound substations.")
            print("  High -> the annual bridge uses the analytic C57.91 cap at those substations;")
            print("          no fitted voltage residual is needed where the ageing actually is.")
            print("  The NOT-thermal group needs no DTR modelling either: the DTR recovers")
            print("  essentially nothing there, so doe_dtr == doe_static for the cost delta.")
    if notes:
        print("\nNotes:")
        for n in notes: print("  ", n)
    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        df.to_csv(a.csv, index=False)
        print("\nFull table written to", a.csv)


if __name__ == "__main__":
    main()
