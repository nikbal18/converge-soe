"""Stage 2: annual transformer ageing per scenario, computed WITHOUT the optimiser.

Why this works
--------------
Ageing depends only on the AGGREGATE loading at the transformer, not on how the
envelope was divided between customers. The two things that set that loading in
each scenario are cheap to compute:

    bau         no cap
    doe_static  K^2 <= 1.0            (nameplate; derive_i_rated makes K=1 at rating)
    doe_dtr     K^2 <= K2_max(t)      (the inverted C57.91 model, a bracketed
                                       root-find in thermal.dtr_k2_max)

Neither needs ipopt. So a full-span BAU pass (vectorised power flow, seconds)
plus this recursion gives annual ageing for all three scenarios with no new
solving.

What it ignores, and where that matters
---------------------------------------
It ignores the voltage constraint. At thermally bound substations (peak
K2_actual/K2_max near 1) the thermal cap is what binds and the reduced form is
exact by construction. At substations where voltage binds first it lets through
more loading than the real DOE did, so it OVERSTATES ageing under the two DOE
scenarios and therefore UNDERSTATES the DOE benefit. That error is conservative
and --validate measures it per substation against the solved runs.

Thermal parameters are applied HERE, not in the solve, so changing them costs
nothing: the BAU K^2 series is parameter-free (BAU has no thermal constraint).

Usage
-----
    # 1. list what is available
    python tools/stage2_annual_ageing.py --bau-run out/<FEEDER>/<ANNUAL_BAU> --list-subs

    # 2. validate the reduced form against a solved window FIRST
    python tools/stage2_annual_ageing.py \
        --bau-run out/GOLDCR_8HB_LEXCEN/lexcen7_summer \
        --validate out/GOLDCR_8HB_LEXCEN/lexcen7_summer \
        --out out/stage2/validate_lexcen

    # 3. then the annual numbers
    python tools/stage2_annual_ageing.py \
        --bau-run out/GOLDCR_8HB_LEXCEN/annual_bau \
        --params evoenergy --out out/stage2/lexcen
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

from converge_soe import thermal as _thermal

# --------------------------------------------------------------------------
# Parameter sets. The repo set is what the existing solves used; the Evoenergy
# set is from "Evo Energy Distribution Transformer Fleet Thermal Modelling
# AS/NZS 60076.7" (Table 1), sourced to AS/NZS 60076.7 Table E1.
# n and m are applied to K^2 here, which is algebraically the same as the
# report's x=0.8 / y=1.6 applied to K.
# --------------------------------------------------------------------------
PARAM_SETS = {
    "repo": dict(tau_TO=60.0, tau_W=7.0, delta_theta_TO_R=55.0, delta_theta_HS_R=23.0,
                 R=6.0, n=0.8, m=0.8, I_rated=1.0, theta_HS_max=120.0, dt=30.0),
    "evoenergy": dict(tau_TO=180.0, tau_W=4.0, delta_theta_TO_R=55.0, delta_theta_HS_R=23.0,
                      R=5.0, n=0.8, m=0.8, I_rated=1.0, theta_HS_max=120.0, dt=30.0),
}

# Hot-spot exposure tiers. SOURCED: AS/NZS 60076.7:2013 Table 4 (p.18),
# "Current and temperature limits applicable to loading beyond nameplate
# rating", DISTRIBUTION TRANSFORMERS column. Verified against the standard.
#
#   Normal cyclic loading        1,5 pu   winding hot-spot 120 C   top-oil 105 C
#   Long-time emergency loading  1,8 pu   winding hot-spot 140 C   top-oil 115 C
#   Short-time emergency loading 2,0 pu   winding hot-spot SEE 7.2.1
#
# READ THE ROW LABELS CAREFULLY. The 120 and 140 figures are for "Winding
# hot-spot temperature AND metallic parts in contact with cellulosic insulation
# material". Table 4 also carries an "Other metallic hot-spot temperature (in
# contact with oil, aramid paper, glass fibre materials)" row at 140/160 - that
# is a DIFFERENT quantity and not what this model computes. This model computes
# winding hot-spot, so only 120 and 140 are limits that apply to it.
#
# 160 C IS NOT A DISTRIBUTION TRANSFORMER LIMIT. Table 4 gives 160 C only for
# MEDIUM and LARGE POWER transformers under short-time emergency loading; the
# distribution column says "See 7.2.1" instead. It is retained in the tier list
# purely as a diagnostic band, because BAU peaks at 206.5 C and one number above
# the ceiling is not enough resolution. Do not present it as a standard limit.
#
# Also note the NOTE under Table 4: the current and temperature limits "are not
# intended to be valid simultaneously" - either may be tightened to satisfy the
# other. Relevant to how the DTR constraint is framed, since K2_max caps current
# while theta_HS_max caps temperature.
DEFAULT_TIERS = (120.0, 140.0, 160.0)
STANDARD_TIERS = (120.0, 140.0)   # the two that AS/NZS 60076.7 actually sets

# Ageing above this is reported as exposure rather than priced as wear.
#
# SOURCED, and the standard states the mechanism directly. AS/NZS 60076.7:2013
# clause 7.2.1, "Specific limitations for distribution transformers":
#
#   "No limit is set for the top-oil and hot-spot temperature under short-time
#    emergency loading for distribution transformers because it is usually
#    impracticable to control the duration of emergency loading in this case.
#    It should be noted that when the hot-spot temperature exceeds 140 C, gas
#    bubbles may develop which could jeopardize the dielectric strength of the
#    transformer (see 5.3)."
#
# So above 140 C the failure mode the standard itself identifies is DIELECTRIC -
# bubble formation collapsing the oil's dielectric strength - not gradual
# thermal ageing of the paper. Summing Arrhenius hours through that region
# prices the wrong mechanism. Hours above are reported unpriced, which makes the
# priced saving conservative: the avoided failure risk is strictly additional.
#
# This supersedes two weaker arguments that were considered and dropped:
#  - "the Arrhenius relation is not credible above 140 C". The 80-140 C range is
#    the span of the IEC relative ageing rate table for NON-thermally-upgraded
#    paper referenced to 98 C. This model uses the IEEE C57.91 relation
#    (theta_ref 110 C, B = 15,000 K, normal life 180,000 h), so that range is
#    the bound on a different curve.
#  - "140 C is the top of the permitted loading envelope". True for long-time
#    emergency, but 7.2.1 sets no hot-spot limit at all for distribution
#    transformers under short-time emergency, so the envelope argument alone
#    does not close it. The gas-bubbling clause does.
#
# METHODOLOGY NOTE: this project uses the IEC/AS-NZS 60076.7 THERMAL model
# (top-oil and hot-spot equations and exponents) with the IEEE C57.91 AGEING
# relation. Common, defensible, and the same combination Evoenergy's own fleet
# study uses - but state it rather than blurring the two, since the midterm
# marker flagged notation drift across variants of the standard's name.
DEFAULT_CEILING = 140.0

SCENARIOS = ("bau", "doe_static", "doe_dtr")


def rated_kva(run_dir, sub, network_root="build/network/substations"):
    """Nameplate kVA for a substation, from the converted LV network JSON."""
    import json
    feeder = os.path.basename(os.path.dirname(os.path.normpath(run_dir)))
    p = os.path.join(network_root, feeder, f"{sub}_lv_network.json")
    if not os.path.exists(p):
        return float("nan")
    try:
        d = json.load(open(p))
    except Exception:                                       # noqa: BLE001
        return float("nan")
    for v in d.get("components", {}).values():
        kind = next(iter(v))
        if kind.startswith("Transformer"):
            sm = v[kind].get("s_max")
            if sm:
                return float(sm) * 1000.0
    return float("nan")


def load_bau_series(run_dir, sub):
    """(k2, theta_A, is_export, timestamps) for one substation's BAU run.

    is_export marks reverse flow at the transformer. The sign convention is read
    from the data rather than assumed: most intervals on a residential LV feeder
    are import, so the sign of the median branch power IS the import direction,
    and export is everything opposite to it.
    """
    p = os.path.join(run_dir, "scenarios", "bau", sub, "thermal.parquet")
    if not os.path.exists(p):
        return None
    t = pd.read_parquet(p, columns=["timestamp", "K2_actual", "theta_A_C",
                                    "transformer_id"])
    t = t.dropna(subset=["K2_actual", "theta_A_C"])
    t["K2_actual"] = t["K2_actual"].clip(lower=0.0)
    t["is_export"] = False

    bp = os.path.join(run_dir, "scenarios", "bau", sub, "branch.parquet")
    if os.path.exists(bp) and len(t):
        try:
            tx = str(t["transformer_id"].iloc[0])
            b = pd.read_parquet(bp, columns=["timestamp", "id", "p_w_oel"])
            b = b[b["id"].astype(str) == tx][["timestamp", "p_w_oel"]]
            if len(b):
                med = float(b["p_w_oel"].median())
                imp_sign = 1.0 if med >= 0 else -1.0
                b["is_export"] = (b["p_w_oel"] * imp_sign) < 0
                t = t.drop(columns=["is_export"]).merge(
                    b[["timestamp", "is_export"]], on="timestamp", how="left")
                t["is_export"] = t["is_export"].fillna(False)
        except Exception:                                   # noqa: BLE001
            pass
    return t


def recurse(k2_bau, theta_A, tp, scenario, k2_emergency=4.0, is_export=None):
    """Forward thermal recursion under one scenario's cap. Returns a DataFrame.

    is_export, when given, restricts the envelope to reverse-flow intervals: an
    export-only DOE, which is the realistic deployment because import is
    inflexible residential load. Import intervals then run uncapped.
    """
    dTO = dHS = 0.0
    theta, k2_used, cap_used, binding = [], [], [], []
    exp = list(is_export) if is_export is not None else None
    for i, (kb, tha) in enumerate(zip(k2_bau, theta_A)):
        if scenario == "bau":
            cap = float("inf")
        elif exp is not None and not exp[i]:
            cap = float("inf")          # export-only envelope: import unconstrained
        elif scenario == "doe_static":
            cap = 1.0
        else:
            cap, _status = _thermal.dtr_k2_max(tp, tha, dTO, dHS, k2_emergency=k2_emergency)
        k2 = kb if kb < cap else cap
        st = _thermal.forward_step(tp, k2, tha, dTO, dHS)
        dTO, dHS = st["delta_theta_TO"], st["delta_theta_HS"]
        theta.append(st["theta_HS"]); k2_used.append(k2); cap_used.append(cap)
        binding.append(bool(kb > cap))
    out = pd.DataFrame(dict(theta_HS_C=theta, k2=k2_used, cap=cap_used, capped=binding))
    out["is_export"] = list(is_export) if is_export is not None else False
    return out


def throughput_kvah(traj, dt_h, kva):
    """Apparent energy passed by the transformer over the span, in kVAh.

    S(t) = K(t) x S_rated, and K = sqrt(k2) because derive_i_rated puts K = 1 at
    nameplate. Comparing this between scenarios gives the energy one envelope
    passed that another did not. It is APPARENT energy: at the power factors
    these feeders run at during PV export it is within a few per cent of kWh,
    but it is not kWh and should be labelled as kVAh.
    """
    if kva != kva:
        return float("nan")
    return float((traj.k2.clip(lower=0) ** 0.5).sum() * kva * dt_h)


def throughput_export_kvah(traj, dt_h, kva):
    """Same, restricted to reverse-flow intervals. This is recovered EXPORT."""
    if kva != kva or "is_export" not in traj:
        return float("nan")
    m = traj.is_export.astype(bool)
    if not m.any():
        return 0.0
    return float((traj.k2[m].clip(lower=0) ** 0.5).sum() * kva * dt_h)


def summarise(traj, dt_h, tiers, ceiling, span_h):
    f = traj.theta_HS_C.map(_thermal.faa)
    ageing = float(f.sum() * dt_h)
    below = traj.theta_HS_C <= ceiling
    out = dict(
        ageing_hours=ageing,
        percent_LOL=_thermal.percent_loss_of_life(ageing),
        ageing_hours_below_ceiling=float(f[below].sum() * dt_h),
        hours_above_ceiling=float((~below).sum() * dt_h),
        peak_theta_HS_C=float(traj.theta_HS_C.max()),
        mean_theta_HS_C=float(traj.theta_HS_C.mean()),
        pct_intervals_capped=100.0 * float(traj.capped.mean()),
        span_hours=span_h,
        ageing_hours_annualised=ageing * (8760.0 / span_h) if span_h else float("nan"),
    )
    out["percent_LOL_annualised"] = _thermal.percent_loss_of_life(out["ageing_hours_annualised"])
    for t in tiers:
        out[f"hours_above_{int(t)}C"] = float((traj.theta_HS_C > t).sum() * dt_h)
    out["hours_cap_binding"] = float(traj.capped.sum() * dt_h)
    return out


def process(run_dir, subs, tp, tiers, ceiling, warmup, k2_emergency,
            envelope_side="both"):
    rows, notes = [], []
    for sub in subs:
        t = load_bau_series(run_dir, sub)
        if t is None or t.empty:
            notes.append(f"SKIP {sub}: no bau thermal.parquet (or no usable rows)")
            continue
        dt_h = tp["dt"] / 60.0
        kva = rated_kva(run_dir, sub)
        if kva != kva:
            notes.append(f"WARN {sub}: no rated kVA found; throughput columns will be NaN")
        k2, tha = t.K2_actual.tolist(), t.theta_A_C.tolist()
        if "is_export" not in t:
            t["is_export"] = False
        for sc in SCENARIOS:
            traj = recurse(k2, tha, tp, sc, k2_emergency=k2_emergency,
                           is_export=(t.is_export.tolist()
                                      if envelope_side == "export" else None))
            use = traj.iloc[warmup:] if warmup and len(traj) > warmup else traj
            span_h = len(use) * dt_h
            r = dict(substation=sub, scenario=sc, n_intervals=len(use),
                     first=str(t.timestamp.iloc[0]), last=str(t.timestamp.iloc[-1]),
                     rated_kva=kva,
                     throughput_kvah=throughput_kvah(use, dt_h, kva),
                     throughput_export_kvah=throughput_export_kvah(use, dt_h, kva),
                     export_intervals=int(use.is_export.astype(bool).sum())
                     if "is_export" in use else 0)
            r.update(summarise(use, dt_h, tiers, ceiling, span_h))
            rows.append(r)
    return pd.DataFrame(rows), notes


def validate(run_dir, df_reduced):
    """Compare reduced-form ageing against the solved ageing for the same window."""
    m = os.path.join(run_dir, "comparison", "metrics_by_substation.csv")
    if not os.path.exists(m):
        return pd.DataFrame(), [f"no metrics_by_substation.csv in {run_dir}"]
    solved = pd.read_csv(m)[["scenario", "substation", "ageing_hours", "peak_theta_HS_C"]]
    solved = solved.rename(columns={"ageing_hours": "solved_ageing_h",
                                    "peak_theta_HS_C": "solved_peak_C"})
    j = df_reduced.merge(solved, on=["substation", "scenario"], how="inner")
    j["reduced_ageing_h"] = j.ageing_hours
    j["ratio_reduced_over_solved"] = j.reduced_ageing_h / j.solved_ageing_h.replace(0, pd.NA)
    j["peak_error_C"] = j.peak_theta_HS_C - j.solved_peak_C
    return j[["substation", "scenario", "reduced_ageing_h", "solved_ageing_h",
              "ratio_reduced_over_solved", "peak_theta_HS_C", "solved_peak_C",
              "peak_error_C"]], []


def discover_subs(run_dir):
    d = os.path.join(run_dir, "scenarios", "bau")
    return sorted(os.path.basename(p.rstrip("/\\"))
                  for p in glob.glob(os.path.join(d, "*"))
                  if os.path.isdir(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bau-run", required=True, nargs="+",
                    help="one or more run directories whose scenarios/bau/<SUB>/thermal.parquet "
                         "supplies K2. Pass several (e.g. a summer run and a winter run) and their "
                         "ageing is SUMMED, with the recursion run separately on each so no false "
                         "join is created between non-contiguous spans.")
    ap.add_argument("--validate", default=None,
                    help="solved run directory to check the reduced form against "
                         "(usually the same as --bau-run)")
    ap.add_argument("--out", default="out/stage2")
    ap.add_argument("--subs", nargs="*", default=None, help="default: every substation in the run")
    ap.add_argument("--params", choices=sorted(PARAM_SETS), default="evoenergy")
    ap.add_argument("--theta-hs-max", type=float, default=None,
                    help="override the DTR limit (default from the parameter set: 120 C)")
    ap.add_argument("--k-emergency", type=float, default=2.0, help="DTR backstop, K not K^2")
    ap.add_argument("--tiers", type=float, nargs="*", default=list(DEFAULT_TIERS))
    ap.add_argument("--ceiling", type=float, default=DEFAULT_CEILING,
                    help="hot-spot above which ageing is reported as exposure, not loss of life")
    ap.add_argument("--warmup-intervals", type=int, default=48,
                    help="intervals simulated but excluded from the totals (cold-oil start)")
    ap.add_argument("--dtr-deployed-at", nargs="*", default=None,
                    help="substations where the DTR is actually deployed. Anywhere else, "
                         "doe_dtr is set equal to doe_static, which is both what a targeted "
                         "rollout means and what removes the reduced form's voltage bias. "
                         "Default: everywhere.")
    ap.add_argument("--envelope-side", choices=["both", "export"], default="both",
                    help="both (default) = two-sided envelope, matching the solved runs. "
                         "export = the envelope binds only in reverse flow, which is the "
                         "realistic deployment since residential import is inflexible. "
                         "Run both and report the difference.")
    ap.add_argument("--list-subs", action="store_true")
    a = ap.parse_args()

    if a.list_subs:
        for s in discover_subs(a.bau_run):
            print(s)
        return

    tp = dict(PARAM_SETS[a.params])
    if a.theta_hs_max is not None:
        tp["theta_HS_max"] = a.theta_hs_max
    subs = a.subs or sorted({x for r in a.bau_run for x in discover_subs(r)})
    if not subs:
        for r in a.bau_run:
            print("No substations found under", os.path.join(r, "scenarios", "bau"))
        return

    os.makedirs(a.out, exist_ok=True)
    k2_emerg = a.k_emergency ** 2

    print(f"Parameter set: {a.params}  {tp}")
    print(f"Envelope side: {a.envelope_side}"
          + ("  (two-sided, as the solved runs were)" if a.envelope_side == "both"
             else "  (reverse flow only; import runs uncapped)"))
    print(f"Substations: {len(subs)}   warm-up excluded: {a.warmup_intervals} intervals")

    parts, notes = [], []
    for r in a.bau_run:
        di, ni = process(r, subs, tp, a.tiers, a.ceiling, a.warmup_intervals, k2_emerg,
                         envelope_side=a.envelope_side)
        if not di.empty:
            di.insert(0, "bau_run", os.path.basename(r.rstrip("/\\")))
            parts.append(di)
        notes += ni
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(a.bau_run) > 1 and not df.empty:
        df.to_csv(os.path.join(a.out, "annual_ageing_by_run.csv"), index=False)
        sumcols = (["ageing_hours", "ageing_hours_below_ceiling", "hours_above_ceiling",
                    "span_hours", "n_intervals", "throughput_kvah",
                    "throughput_export_kvah", "export_intervals", "hours_cap_binding"]
                   + [f"hours_above_{int(t)}C" for t in a.tiers])
        g = df.groupby(["substation", "scenario"], as_index=False)
        agg = g[sumcols].sum()
        agg["peak_theta_HS_C"] = g["peak_theta_HS_C"].max()["peak_theta_HS_C"]
        agg["rated_kva"] = g["rated_kva"].max()["rated_kva"]
        agg["pct_intervals_capped"] = g["pct_intervals_capped"].mean()["pct_intervals_capped"]
        agg["percent_LOL"] = agg.ageing_hours.map(_thermal.percent_loss_of_life)
        agg["ageing_hours_annualised"] = agg.ageing_hours * (8760.0 / agg.span_hours)
        agg["percent_LOL_annualised"] = agg.ageing_hours_annualised.map(_thermal.percent_loss_of_life)
        agg["first"] = ""; agg["last"] = ""
        df = agg
        print(f"Combined {len(a.bau_run)} BAU runs; ageing summed, recursion run separately on each.")
    if df.empty:
        print("Nothing computed.")
        for n in notes: print(" ", n)
        return
    df.insert(0, "param_set", a.params)

    # Targeted deployment: where the DTR is not deployed the transformer runs the
    # conventional DOE, so doe_dtr == doe_static. Leaving doe_dtr at its reduced-form
    # value there would report the DTR as doing nothing (it drifts back to bau,
    # because the thermal cap never binds) which is the reduced form's known voltage
    # bias, not a result.
    df["dtr_deployed"] = True
    if a.dtr_deployed_at is not None:
        dep = set(a.dtr_deployed_at)
        cols = [c for c in df.columns if c not in ("substation", "scenario", "param_set",
                                                   "first", "last", "dtr_deployed")]
        stat = df[df.scenario == "doe_static"].set_index("substation")
        for i, r in df.iterrows():
            if r.scenario == "doe_dtr" and r.substation not in dep:
                df.at[i, "dtr_deployed"] = False
                if r.substation in stat.index:
                    for c in cols:
                        df.at[i, c] = stat.at[r.substation, c]
        n_off = int((~df[df.scenario == "doe_dtr"].dtr_deployed).sum())
        print(f"Targeted deployment: DTR modelled at {len(dep)} substation(s); "
              f"doe_dtr set equal to doe_static at {n_off} other(s).")

    ann = os.path.join(a.out, "annual_ageing.csv")
    df.to_csv(ann, index=False)

    span_d = float(df.span_hours.max()) / 24.0 if "span_hours" in df else 0.0
    print(f"\nSpan analysed: {span_d:.0f} days. Annualised columns scale this to 8760 h.")
    if span_d < 300:
        print("  CAUTION: naive scaling assumes the span is seasonally representative.")
        print("  Ageing is concentrated in hot high-PV days, so a span that already contains")
        print("  a full summer will OVERSTATE the year when doubled, and a winter-only span")
        print("  will understate it. Prefer the raw ageing_hours column plus an explicit,")
        print("  stated treatment of the uncovered months (register S7).")

    piv = df.pivot(index="substation", columns="scenario", values="ageing_hours_annualised")
    print("\n=== annualised equivalent ageing hours (scaled to 8760 h of span) ===")
    print(f"{'substation':13} {'bau':>14} {'doe_static':>12} {'doe_dtr':>12}   "
          f"{'dtr/static':>10} {'dtr/bau':>9}")
    for sub, r in piv.sort_values("bau", ascending=False).iterrows():
        b, s, d = r.get("bau"), r.get("doe_static"), r.get("doe_dtr")
        rr = lambda x, y: f"{x/y:.3f}" if (pd.notna(x) and pd.notna(y) and y) else "-"
        print(f"{sub:13} {b:14.1f} {s:12.2f} {d:12.2f}   {rr(d,s):>10} {rr(d,b):>9}")

    if "throughput_kvah" in df.columns and df.throughput_kvah.notna().any():
        tp_ = df.pivot(index="substation", columns="scenario", values="throughput_kvah")
        print("\n=== energy passed by the transformer (kVAh over the span) ===")
        print("    the DTR's BENEFIT: what it let through that a nameplate-limited DOE did not")
        print(f"{'substation':13} {'rated kVA':>9} {'bau':>14} {'doe_static':>14} {'doe_dtr':>14} "
              f"{'dtr - static':>13} {'% of static':>11}")
        kva = df.groupby("substation").rated_kva.max()
        for sub, r in tp_.sort_values("bau", ascending=False).iterrows():
            b, st, dt_ = r.get("bau"), r.get("doe_static"), r.get("doe_dtr")
            if pd.isna(st) or pd.isna(dt_):
                continue
            gain = dt_ - st
            pct = 100 * gain / st if st else float("nan")
            print(f"{sub:13} {kva.get(sub, float('nan')):9.0f} {b:14.0f} {st:14.0f} {dt_:14.0f} "
                  f"{gain:13.0f} {pct:10.2f}%")
        if "throughput_export_kvah" in df.columns and df.throughput_export_kvah.notna().any():
            te = df.pivot(index="substation", columns="scenario",
                          values="throughput_export_kvah")
            print("\n  --- restricted to REVERSE-FLOW intervals (recovered export) ---")
            print(f"  {'substation':13} {'static':>14} {'dtr':>14} {'dtr - static':>13} "
                  f"{'export share of':>16}")
            print(f"  {'':13} {'':>14} {'':>14} {'':>13} {'the total gain':>16}")
            for sub, r in te.sort_values("bau", ascending=False).iterrows():
                st, dt_ = r.get("doe_static"), r.get("doe_dtr")
                if pd.isna(st) or pd.isna(dt_):
                    continue
                gain_e = dt_ - st
                tot = tp_.loc[sub, "doe_dtr"] - tp_.loc[sub, "doe_static"]
                sh = 100 * gain_e / tot if tot else float("nan")
                print(f"  {sub:13} {st:14.0f} {dt_:14.0f} {gain_e:13.0f} {sh:15.1f}%")
            print("  Use the export column for the customer/market benefit. The remainder of")
            print("  the total gain is import-side headroom, which is not recovered PV export.")

        print("\n  kVAh, not kWh: S = K x S_rated, and at PV-export power factors the two are")
        print("  within a few per cent. Label it as apparent energy in the thesis.")
        print("  'dtr - static' is the quantity to set against the extra ageing the DTR costs.")

    print("\n=== two-regime check: how much BAU ageing sits above the "
          f"{a.ceiling:.0f} C validity ceiling ===")
    bau = df[df.scenario == "bau"]
    for _, r in bau.sort_values("ageing_hours", ascending=False).head(12).iterrows():
        frac = 100 * (1 - r.ageing_hours_below_ceiling / r.ageing_hours) if r.ageing_hours else 0
        print(f"{r.substation:13} peak {r.peak_theta_HS_C:6.1f} C   "
              f"{frac:5.1f}% of ageing above ceiling   "
              + "  ".join(f">{int(t)}C: {r[f'hours_above_{int(t)}C']:.1f} h" for t in a.tiers))
    print("\n  Where that fraction is large, do NOT report a replacement year from BAU "
          "loss of life.\n  Report the exposure hours above and cost them as unplanned "
          "failure instead (register S8).")

    if a.validate:
        vparams = "repo"
        print(f"\n=== validation against solved run {a.validate} ===")
        print(f"  using the '{vparams}' parameter set, because that is what the solve used;")
        print("  comparing against '%s' would conflate the reduced form with the "
              "parameter change." % a.params)
        print("  warm-up forced to 0 intervals: the solved ageing_hours in "
              "metrics_by_substation.csv\n  covers every interval, so excluding warm-up here "
              "would compare different spans.")
        vtp = dict(PARAM_SETS[vparams])
        if a.theta_hs_max is not None:
            vtp["theta_HS_max"] = a.theta_hs_max
        vdf, _ = process(a.validate, subs, vtp, a.tiers, a.ceiling, 0, k2_emerg,
                         envelope_side=a.envelope_side)
        vj, vnotes = validate(a.validate, vdf)
        notes += vnotes
        if vj.empty:
            print("  nothing to compare")
        else:
            vj.to_csv(os.path.join(a.out, "validation.csv"), index=False)
            print(f"  {'substation':13} {'scenario':11} {'reduced h':>11} {'solved h':>11} "
                  f"{'ratio':>7} {'peak err C':>11}")
            for _, r in vj.iterrows():
                rat = f"{r.ratio_reduced_over_solved:.3f}" if pd.notna(r.ratio_reduced_over_solved) else "-"
                print(f"  {r.substation:13} {r.scenario:11} {r.reduced_ageing_h:11.3f} "
                      f"{r.solved_ageing_h:11.3f} {rat:>7} {r.peak_error_C:11.2f}")
            bau_tot = vj[vj.scenario == "bau"].solved_ageing_h.sum()
            print(f"\n  --- agreement, weighted by how much ageing each substation carries ---")
            for sc in SCENARIOS:
                d = vj[vj.scenario == sc].copy()
                base = vj[vj.scenario == "bau"].set_index("substation").solved_ageing_h
                d["w"] = d.substation.map(base).fillna(0.0)
                mat = d[d.w >= 0.01 * bau_tot]      # substations carrying >=1% of run ageing
                if mat.empty:
                    print(f"  {sc:11} no substation carries >=1% of this run's ageing")
                    continue
                num = (mat.reduced_ageing_h).sum()
                den = (mat.solved_ageing_h).sum()
                print(f"  {sc:11} {len(mat)} substation(s) carry >=1% of run ageing; "
                      f"pooled reduced/solved = {num/den:.3f}"
                      f"   (per-sub range {mat.ratio_reduced_over_solved.min():.2f}"
                      f" to {mat.ratio_reduced_over_solved.max():.2f})")
            print("\n  Read it like this:")
            print("   bau  should be ~1.000. It has no cap in either the solve or the reduced")
            print("        form, so anything else means the recursion or the inputs disagree.")
            print("   dtr  near 1.000 at thermally bound substations = the analytic cap is sound.")
            print("   >1   is the expected direction where voltage binds: the reduced form lets")
            print("        through more loading than the real DOE did, so it understates benefit.")
            print("  Substations with near-zero ageing are excluded above; a ratio on 0.01 h is noise.")

    print(f"\nWritten: {ann}")
    if a.validate:
        print(f"Written: {os.path.join(a.out, 'validation.csv')}")
    if notes:
        print("\nNotes:")
        for n in notes: print("  ", n)


if __name__ == "__main__":
    main()
