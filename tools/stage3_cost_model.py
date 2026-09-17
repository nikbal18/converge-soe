"""Stage 3-5: annual ageing -> replacement year -> per-transformer cost.

Reads the Stage 2 outputs and turns them into the numbers the cost chapter
quotes. Nothing here needs the solver or the pipeline.

What it does, in order
----------------------
1. ANNUALISE with a bracket, not a point estimate. The runs cover ~183 days
   (a full summer and a full winter) and miss the shoulder seasons, so:
       lower = covered ageing + 2 x winter ageing   (shoulders age like winter)
       upper = covered ageing x 365 / span_days     (shoulders age like average)
   Both are reported. Never quote one alone.

2. TWO-REGIME RULE. Only ageing BELOW the validity ceiling is costed as wear.
   Above it the Arrhenius relation is not credible and the failure mode is
   dielectric, not gradual. Hours above the ceiling are reported as exposure
   and left unpriced, which makes the priced saving conservative.

3. REPLACEMENT YEAR.
       years_to_EOL = (EOL% - start%) / 100 x 180,000 h / annual ageing hours
       T_replacement = min(standard asset life, years_to_EOL)
   The min matters: a transformer that ages slowly is still replaced at its
   standard life. If ageing does not pull replacement inside the standard life,
   the deferral is ZERO and the script says so rather than inventing one.

4. ANNUITY.  A(T) = C x r / (1 - (1+r)^-T)
   Annual cost of a transformer replaced every T years. The difference between
   two scenarios' annuities is the annual cost of the ageing difference.

5. NET VALUE of the DTR = value of the extra energy it passes, minus the
   annuity cost of the extra ageing it causes.

Usage
-----
    python tools/stage3_cost_model.py --init            # write the params file
    # fill in cost_params.yaml, then
    python tools/stage3_cost_model.py \
        --stage2 out/stage2/lexcen out/stage2/saunders \
                 out/stage2/birrigai out/stage2/streeton \
        --params cost_params.yaml --out out/stage4
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

try:
    import yaml
except ImportError:                                         # noqa: BLE001
    yaml = None

NORMAL_LIFE_H = 180_000.0

TEMPLATE = """# Cost model parameters. Fill every null before running.
# Sources are named so each number is traceable in the thesis.

# AER Final Decision, Evoenergy distribution determination 2024-29.
# Nominal vanilla WACC, as a decimal (e.g. 0.0589 for 5.89 %).
wacc_nominal_vanilla: null

# AER repex model. Standard asset life for distribution transformers, years.
# A transformer that ages slowly is still replaced at this age.
standard_asset_life_years: null

# Installed replacement cost of one distribution transformer, AUD.
#
# THE ONE NUMBER TO CHANGE. Set this and the per-kVA table below is ignored.
# Currently a working estimate, not a sourced figure - replace it when you have
# Evoenergy Appendix 1.9 (CutlerMerz repex results) or the Category Analysis RIN.
unit_replacement_cost_flat_aud: 100000

# Optional: per-kVA bands, used ONLY if the flat cost above is null. The script
# picks the nearest band at or above the transformer's rating.
unit_replacement_cost_aud:
  250: null
  315: null
  500: null
  750: null
  1000: null

# Value of energy the DTR lets through that a nameplate-limited DOE would not.
# The same assumption the DOE runs used. It is an assumption, not a result.
energy_value_per_kwh: 0.10

# Normal insulation life, hours, at the 110 C reference hot-spot, CONTINUOUS.
# IEEE C57.91 Table 2 (NOT AS/NZS 60076.7, which takes a DP-based approach and
# does not publish this figure):
#     65,000 h  - 50 % retained tensile strength
#    135,000 h  - 25 % retained tensile strength
#    145,000 h  - 200 retained degree of polymerisation
#    180,000 h  - interpretation of distribution transformer functional life
#                 test data                                    <- the usual default
# This constant scales EVERY loss-of-life number linearly, and it decides whether
# ageing pulls replacement inside the standard asset life at all. Sweep it.
normal_insulation_life_hours: 180000.0

# How the network (DNSP) cost of insulation ageing is valued. TWO DIFFERENT
# VALUATIONS OF THE SAME PHYSICAL WEAR - report one as headline, the other as a
# bound, and NEVER add them.
#
#   depletion - user cost. The transformer contains `normal_insulation_life_hours`
#               of life and costs `unit_replacement_cost`, so each equivalent
#               ageing hour consumed is worth cost / life. Always non-zero.
#               Ignores WHEN the life is consumed.
#   deferral  - change in the NPV of the replacement cash flow. Zero unless the
#               ageing actually pulls the replacement date forward. Economically
#               rigorous; what a regulator would do in a RIT-D.
#
# Both are always computed and printed. This setting only chooses which one
# drives the reported net value.
network_cost_basis: depletion

# End of life as a percentage of the above.
end_of_life_percent_lol: 100.0

# Cumulative loss of life already consumed at the start, per cent.
# 0 unless Evoenergy condition assessment data has been joined.
starting_loss_of_life_percent: 0.0

# Sensitivity sweep.
sensitivity:
  starting_lol_percent: [0.0, 30.0, 60.0]
  wacc_absolute_delta: [-0.01, 0.0, 0.01]
  unit_cost_multiplier: [0.5, 1.0, 2.0]
  normal_insulation_life_hours: [65000.0, 135000.0, 180000.0]
  energy_value_per_kwh: [0.04, 0.10, 0.20]
"""


def annuity(cost, rate, years):
    """Annualised capital cost of an asset replaced every `years` years."""
    if not (cost == cost and rate == rate and years == years):
        return float("nan")
    if years <= 0:
        return float("nan")
    if rate == 0:
        return cost / years
    return cost * rate / (1.0 - (1.0 + rate) ** (-years))


def unit_cost_for(kva, table, flat=None):
    """Flat cost if one is set, else the nearest band at or above the rating."""
    if flat is not None and flat == flat:
        return float(flat)
    if kva != kva or not table:
        return float("nan")
    bands = sorted(float(k) for k in table if table[k] is not None)
    if not bands:
        return float("nan")
    for b in bands:
        if kva <= b * 1.001:
            return float(table[int(b)] if int(b) in table else table[b])
    return float(table[int(bands[-1])] if int(bands[-1]) in table else table[bands[-1]])


def load_stage2(dirs):
    """Combined annual_ageing.csv, plus the seasonal split where available."""
    main, byrun = [], []
    for d in dirs:
        p = os.path.join(d, "annual_ageing.csv")
        if not os.path.exists(p):
            print(f"  SKIP {d}: no annual_ageing.csv")
            continue
        m = pd.read_csv(p)
        m["stage2_dir"] = os.path.basename(os.path.normpath(d))
        main.append(m)
        q = os.path.join(d, "annual_ageing_by_run.csv")
        if os.path.exists(q):
            b = pd.read_csv(q)
            b["stage2_dir"] = os.path.basename(os.path.normpath(d))
            byrun.append(b)
    if not main:
        return pd.DataFrame(), pd.DataFrame()
    return (pd.concat(main, ignore_index=True),
            pd.concat(byrun, ignore_index=True) if byrun else pd.DataFrame())


def energy_column(main, prefer_export=True):
    """Which throughput column to price. Export-only is the customer benefit."""
    if prefer_export and "throughput_export_kvah" in main.columns \
            and main.throughput_export_kvah.notna().any():
        return "throughput_export_kvah"
    return "throughput_kvah" if "throughput_kvah" in main.columns else None


def seasonal_bracket(main, byrun, ecol=None):
    """Lower and upper annual ageing hours BELOW the ceiling, per sub+scenario.

    annual_ageing_by_run.csv is written BEFORE the --dtr-deployed-at
    substitution, so its doe_dtr rows are wrong wherever the DTR is not
    deployed. The deployment flag is re-applied here from the combined file.
    """
    key = ["stage2_dir", "substation", "scenario"]
    ecol = ecol or energy_column(main)
    cols = ["ageing_hours_below_ceiling", "span_hours"]
    if ecol and ecol in main.columns:
        cols.append(ecol)
    out = main.set_index(key)[cols].copy()
    out["span_days"] = out.span_hours / 24.0
    out["winter_below"] = float("nan")
    out["winter_throughput"] = float("nan")

    if not byrun.empty:
        dep = (main[main.scenario == "doe_dtr"]
               .set_index(["stage2_dir", "substation"]).get("dtr_deployed"))
        b = byrun.copy()
        subcols = [c for c in ("ageing_hours_below_ceiling", ecol)
                   if c and c in b.columns]
        if dep is not None:
            stat = (b[b.scenario == "doe_static"]
                    .set_index(["stage2_dir", "substation", "bau_run"])[subcols])
            for i, r in b.iterrows():
                if r.scenario != "doe_dtr":
                    continue
                if dep.get((r.stage2_dir, r.substation), True) in (False, "False"):
                    kk = (r.stage2_dir, r.substation, r.bau_run)
                    if kk in stat.index:
                        for c in subcols:
                            b.at[i, c] = stat.loc[kk, c]
        w = b[b.bau_run.str.contains("winter", case=False, na=False)]
        if not w.empty:
            out["winter_below"] = w.groupby(key)["ageing_hours_below_ceiling"].sum()
            if ecol and ecol in w.columns:
                out["winter_throughput"] = w.groupby(key)[ecol].sum()

    cov = out.ageing_hours_below_ceiling
    out["annual_lower_h"] = cov + 2.0 * out.winter_below.fillna(0.0)
    out["annual_upper_h"] = cov * (365.0 / out.span_days)
    # if the seasonal split is missing, the lower bound is not defined - use covered
    out.loc[out.winter_below.isna(), "annual_lower_h"] = cov[out.winter_below.isna()]
    return out.reset_index()


def build(main, byrun, P, ecol=None):
    ecol = ecol or energy_column(main)
    br = seasonal_bracket(main, byrun, ecol)
    df = main.merge(br, on=["stage2_dir", "substation", "scenario"],
                    suffixes=("", "_br"))
    life = float(P["standard_asset_life_years"])
    rate = float(P["wacc_nominal_vanilla"])
    eol = float(P["end_of_life_percent_lol"])
    start = float(P["starting_loss_of_life_percent"])
    price = float(P["energy_value_per_kwh"])
    table = {int(k): v for k, v in (P.get("unit_replacement_cost_aud") or {}).items()}
    flat = P.get("unit_replacement_cost_flat_aud")

    df["unit_cost_aud"] = df.rated_kva.map(lambda k: unit_cost_for(k, table, flat))
    nlife = float(P.get("normal_insulation_life_hours", NORMAL_LIFE_H))
    budget_h = max(eol - start, 0.0) / 100.0 * nlife

    depletion_rate = df.unit_cost_aud / nlife          # AUD per equivalent ageing hour
    df["depletion_rate_aud_per_h"] = depletion_rate

    for tag in ("lower", "upper"):
        a = df[f"annual_{tag}_h"]
        df[f"depletion_{tag}_aud_yr"] = a * depletion_rate
        yrs = budget_h / a.where(a > 0)
        df[f"years_to_eol_{tag}"] = yrs
        df[f"T_repl_{tag}"] = yrs.clip(upper=life).fillna(life)
        df[f"annuity_{tag}_aud"] = [
            annuity(c, rate, t) for c, t in zip(df.unit_cost_aud, df[f"T_repl_{tag}"])]
        df[f"ageing_limits_life_{tag}"] = yrs < life
        # Apparent energy passed, annualised on ITS OWN seasonal basis.
        # It must NOT be scaled by the ageing bracket: ageing is a strongly
        # non-linear function of temperature and its seasonal ratio differs per
        # scenario, so scaling energy by it makes doe_static and doe_dtr grow at
        # different rates and can turn a real gain negative.
        if ecol and ecol in df.columns:
            base = df[ecol]
            if tag == "upper":
                df[f"throughput_{tag}_kvah_yr"] = base * (365.0 / df.span_days)
            else:
                wt = df.get("winter_throughput")
                df[f"throughput_{tag}_kvah_yr"] = (
                    base + 2.0 * wt.fillna(0.0) if wt is not None
                    else base * (365.0 / df.span_days))
    return df


def deltas(df, P):
    """Per substation: BAU -> static, and static -> dtr, on both bracket ends."""
    price = float(P["energy_value_per_kwh"])
    rows = []
    for (d, sub), g in df.groupby(["stage2_dir", "substation"]):
        gi = g.set_index("scenario")
        if not {"bau", "doe_static", "doe_dtr"} <= set(gi.index):
            continue
        r = dict(stage2_dir=d, substation=sub,
                 rated_kva=gi.loc["bau", "rated_kva"],
                 unit_cost_aud=gi.loc["bau", "unit_cost_aud"],
                 dtr_deployed=gi.loc["doe_dtr", "dtr_deployed"],
                 exposure_h_above_ceiling_bau=gi.loc["bau", "hours_above_ceiling"],
                 bau_ageing_above_ceiling_h=gi.loc["bau", "ageing_hours"]
                 - gi.loc["bau", "ageing_hours_below_ceiling"])
        for tag in ("lower", "upper"):
            for sc in ("bau", "doe_static", "doe_dtr"):
                r[f"{sc}_annual_h_{tag}"] = gi.loc[sc, f"annual_{tag}_h"]
                r[f"{sc}_T_repl_{tag}"] = gi.loc[sc, f"T_repl_{tag}"]
                r[f"{sc}_annuity_{tag}"] = gi.loc[sc, f"annuity_{tag}_aud"]
                r[f"{sc}_depletion_{tag}"] = gi.loc[sc, f"depletion_{tag}_aud_yr"]
            # deferral basis: change in the NPV of the replacement stream
            r[f"value_of_any_doe_{tag}_deferral"] = (gi.loc["bau", f"annuity_{tag}_aud"]
                                                     - gi.loc["doe_static", f"annuity_{tag}_aud"])
            r[f"dtr_ageing_cost_{tag}_deferral"] = (gi.loc["doe_dtr", f"annuity_{tag}_aud"]
                                                    - gi.loc["doe_static", f"annuity_{tag}_aud"])
            # depletion basis: life consumed x cost per hour of life
            r[f"value_of_any_doe_{tag}_depletion"] = (gi.loc["bau", f"depletion_{tag}_aud_yr"]
                                                      - gi.loc["doe_static", f"depletion_{tag}_aud_yr"])
            r[f"dtr_ageing_cost_{tag}_depletion"] = (gi.loc["doe_dtr", f"depletion_{tag}_aud_yr"]
                                                     - gi.loc["doe_static", f"depletion_{tag}_aud_yr"])
            basis = P.get("network_cost_basis", "depletion")
            r[f"value_of_any_doe_{tag}_aud_yr"] = r[f"value_of_any_doe_{tag}_{basis}"]
            r[f"dtr_ageing_cost_{tag}_aud_yr"] = r[f"dtr_ageing_cost_{tag}_{basis}"]
            col = f"throughput_{tag}_kvah_yr"
            if col in gi.columns:
                gain = gi.loc["doe_dtr", col] - gi.loc["doe_static", col]
                r[f"dtr_energy_gain_{tag}_kvah_yr"] = gain
                r[f"dtr_energy_value_{tag}_aud_yr"] = gain * price
                r[f"dtr_net_value_{tag}_aud_yr"] = (gain * price
                                                    - r[f"dtr_ageing_cost_{tag}_aud_yr"])
        rows.append(r)
    return pd.DataFrame(rows)


def sensitivity(main, byrun, P, out_dir):
    sw = P.get("sensitivity") or {}
    ec = energy_column(main)
    rows = []
    prices = sw.get("energy_value_per_kwh", [P["energy_value_per_kwh"]])
    lives = sw.get("normal_insulation_life_hours",
                   [P.get("normal_insulation_life_hours", NORMAL_LIFE_H)])
    for pr in prices:
     for nl in lives:
      for s0 in sw.get("starting_lol_percent", [P["starting_loss_of_life_percent"]]):
        for dw in sw.get("wacc_absolute_delta", [0.0]):
            for m in sw.get("unit_cost_multiplier", [1.0]):
                Q = dict(P)
                Q["energy_value_per_kwh"] = pr
                Q["normal_insulation_life_hours"] = nl
                Q["starting_loss_of_life_percent"] = s0
                Q["wacc_nominal_vanilla"] = P["wacc_nominal_vanilla"] + dw
                Q["unit_replacement_cost_aud"] = {
                    k: (None if v is None else v * m)
                    for k, v in (P.get("unit_replacement_cost_aud") or {}).items()}
                fl = P.get("unit_replacement_cost_flat_aud")
                Q["unit_replacement_cost_flat_aud"] = None if fl is None else fl * m
                d = deltas(build(main, byrun, Q, ec), Q)
                if d.empty:
                    continue
                rows.append(dict(
                    energy_price=pr, normal_life_h=nl,
                    starting_lol_pct=s0, wacc=Q["wacc_nominal_vanilla"],
                    unit_cost_mult=m,
                    n_subs_ageing_limits_life=int(
                        build(main, byrun, Q, ec).ageing_limits_life_upper.sum()),
                    value_of_any_doe_lower=d.value_of_any_doe_lower_aud_yr.sum(),
                    value_of_any_doe_upper=d.value_of_any_doe_upper_aud_yr.sum(),
                    dtr_net_lower=d.get("dtr_net_value_lower_aud_yr",
                                        pd.Series(dtype=float)).sum(),
                    dtr_net_upper=d.get("dtr_net_value_upper_aud_yr",
                                        pd.Series(dtype=float)).sum()))
    s = pd.DataFrame(rows)
    if not s.empty:
        s.to_csv(os.path.join(out_dir, "sensitivity.csv"), index=False)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage2", nargs="*", default=[], help="Stage 2 output directories")
    ap.add_argument("--params", default="cost_params.yaml")
    ap.add_argument("--out", default="out/stage4")
    ap.add_argument("--init", action="store_true", help="write the params template and exit")
    a = ap.parse_args()

    if a.init:
        if os.path.exists(a.params):
            print(f"{a.params} already exists; not overwriting.")
            return 1
        with open(a.params, "w", encoding="utf-8") as fh:
            fh.write(TEMPLATE)
        print(f"Wrote {a.params}. Fill every null, then re-run without --init.")
        return 0

    if yaml is None:
        print("PyYAML is not installed: conda install pyyaml")
        return 1
    if not os.path.exists(a.params):
        print(f"No {a.params}. Run with --init first.")
        return 1
    P = yaml.safe_load(open(a.params, encoding="utf-8"))

    missing = [k for k in ("wacc_nominal_vanilla", "standard_asset_life_years")
               if P.get(k) is None]
    if (P.get("unit_replacement_cost_flat_aud") is None
            and all(v is None for v in (P.get("unit_replacement_cost_aud") or {}).values())):
        missing.append("unit_replacement_cost_flat_aud (or every band in "
                       "unit_replacement_cost_aud)")
    if missing:
        print("Cannot run. These are still null in " + a.params + ":")
        for m in missing:
            print("  -", m)
        print("\nwacc_nominal_vanilla : AER Final Decision Evoenergy 2024-29 (in the project root)")
        print("standard_asset_life  : AER repex model")
        print("unit_replacement_cost: EN24 Appendix 1.4 (Marsden Jacob, Jan 2023)")
        return 1

    if not a.stage2:
        print("Nothing to do: pass --stage2 <dir> [<dir> ...]")
        return 1
    os.makedirs(a.out, exist_ok=True)

    m, b = load_stage2(a.stage2)
    if m.empty:
        print("No Stage 2 outputs found.")
        return 1
    ecol = energy_column(m)
    print(f"Energy priced from column: {ecol}"
          + ("  (reverse flow only - recovered EXPORT)"
             if ecol == "throughput_export_kvah"
             else "  (TOTAL throughput, both directions - rerun Stage 2 for the export column)"))
    df = build(m, b, P, ecol)
    d = deltas(df, P)
    df.to_csv(os.path.join(a.out, "cost_model_detail.csv"), index=False)
    d.to_csv(os.path.join(a.out, "cost_per_transformer.csv"), index=False)

    print(f"WACC {P['wacc_nominal_vanilla']:.4f} | standard life "
          f"{P['standard_asset_life_years']} yr | normal insulation life "
          f"{float(P.get('normal_insulation_life_hours', NORMAL_LIFE_H)):,.0f} h "
          f"| EOL {P['end_of_life_percent_lol']} % LOL | start "
          f"{P['starting_loss_of_life_percent']} %")
    print(f"{len(d)} transformer(s) from {len(a.stage2)} Stage 2 directory(ies)\n")

    print("=== replacement year (lower / upper annual ageing bracket) ===")
    print(f"{'substation':13} {'kVA':>6} {'bau':>13} {'static':>13} {'dtr':>13}  ageing limits life?")
    for _, r in d.sort_values("bau_annual_h_upper", ascending=False).iterrows():
        fm = lambda lo, hi: f"{lo:5.0f}-{hi:<5.0f}"
        lim = []
        for sc in ("bau", "doe_static", "doe_dtr"):
            sub = df[(df.substation == r.substation) & (df.scenario == sc)]
            if len(sub) and bool(sub.iloc[0]["ageing_limits_life_upper"]):
                lim.append(sc)
        print(f"{r.substation:13} {r.rated_kva:6.0f} "
              f"{fm(r.bau_T_repl_upper, r.bau_T_repl_lower):>13} "
              f"{fm(r.doe_static_T_repl_upper, r.doe_static_T_repl_lower):>13} "
              f"{fm(r.doe_dtr_T_repl_upper, r.doe_dtr_T_repl_lower):>13}  "
              f"{', '.join(lim) if lim else 'none - all at standard life'}")

    basis = P.get("network_cost_basis", "depletion")
    rate = float(P.get("unit_replacement_cost_flat_aud") or 0) / float(
        P.get("normal_insulation_life_hours", NORMAL_LIFE_H))
    print(f"\n=== network cost basis: {basis.upper()} "
          f"(depletion rate {rate:,.4f} AUD per equivalent ageing hour) ===")
    print("  Both bases are computed. They are two valuations of the SAME wear:")
    print("  report one as headline and the other as a bound. Never add them.")

    print(f"\n=== per-transformer, AUD/yr (lower bracket .. upper bracket) ===")
    print(f"{'substation':13} {'DOE value (deplet.)':>21} {'DOE value (defer.)':>20} "
          f"{'DTR ageing (deplet.)':>21} {'DTR energy':>20} {'DTR net':>20}")
    show = d[(d.get('dtr_energy_value_upper_aud_yr', pd.Series(dtype=float)).fillna(0) != 0)
             | (d.value_of_any_doe_upper_depletion.fillna(0) != 0)]
    if show.empty:
        show = d
    for _, r in show.sort_values("value_of_any_doe_upper_depletion", ascending=False).iterrows():
        g = lambda k: r[k] if k in r and r[k] == r[k] else 0.0
        rng = lambda lo, hi: f"{lo:8,.0f} ..{hi:8,.0f}"
        print(f"{r.substation:13} "
              f"{rng(g('value_of_any_doe_lower_depletion'), g('value_of_any_doe_upper_depletion')):>21} "
              f"{rng(g('value_of_any_doe_lower_deferral'), g('value_of_any_doe_upper_deferral')):>20} "
              f"{rng(g('dtr_ageing_cost_lower_depletion'), g('dtr_ageing_cost_upper_depletion')):>21} "
              f"{rng(g('dtr_energy_value_lower_aud_yr'), g('dtr_energy_value_upper_aud_yr')):>20} "
              f"{rng(g('dtr_net_value_lower_aud_yr'), g('dtr_net_value_upper_aud_yr')):>20}")

    print(f"\n=== totals across all transformers, AUD/yr ===")
    for lab, dep, dfr in (("value of any DOE", "value_of_any_doe", "value_of_any_doe"),
                          ("DTR ageing cost", "dtr_ageing_cost", "dtr_ageing_cost")):
        print(f"  {lab:18} depletion {d[dep+'_lower_depletion'].sum():11,.0f} .."
              f"{d[dep+'_upper_depletion'].sum():11,.0f}   |   deferral "
              f"{d[dfr+'_lower_deferral'].sum():9,.0f} ..{d[dfr+'_upper_deferral'].sum():9,.0f}")
    if "dtr_energy_value_upper_aud_yr" in d:
        print(f"  {'DTR energy value':18} {d.dtr_energy_value_lower_aud_yr.sum():21,.0f} .."
              f"{d.dtr_energy_value_upper_aud_yr.sum():11,.0f}")
        print(f"  {'DTR NET ('+basis+')':18} {d.dtr_net_value_lower_aud_yr.sum():21,.0f} .."
              f"{d.dtr_net_value_upper_aud_yr.sum():11,.0f}")

    tot_exp = d.exposure_h_above_ceiling_bau.sum()
    print(f"\n=== unpriced, reported separately ===")
    print(f"  BAU exposure above the validity ceiling: {tot_exp:,.1f} hours over the covered span.")
    print(f"  This is NOT in any dollar figure above. The priced saving is therefore")
    print(f"  conservative: the avoided failure risk is strictly additional.")

    any_limit = df.ageing_limits_life_upper.any()
    if not any_limit:
        print("\n  NOTE: at NO substation does below-ceiling ageing pull replacement inside the")
        print("  standard asset life, so the deferral value is zero on the ageing channel alone.")
        print("  That is a finding, not a failure: the cost of having no envelope is failure")
        print("  risk (the exposure hours above), not accelerated wear. Say so plainly.")

    s = sensitivity(m, b, P, a.out)
    if not s.empty:
        print(f"\nSensitivity sweep: {len(s)} combination(s) -> {a.out}/sensitivity.csv")
        print(f"  value of any DOE  : {s.value_of_any_doe_lower.min():,.0f} .. "
              f"{s.value_of_any_doe_upper.max():,.0f} AUD/yr across the sweep")
        if "n_subs_ageing_limits_life" in s:
            g = s.groupby("normal_life_h").n_subs_ageing_limits_life.max()
            print("  substations where ageing pulls replacement inside the standard life,")
            print("  by normal-insulation-life constant:")
            for nl, n in g.items():
                print(f"    {nl:>10,.0f} h : {int(n)}")
            if g.min() == 0 < g.max():
                print("  The end-of-life constant DECIDES whether there is any deferral at all.")
                print("  That makes it the single most important assumption in the chapter.")
        if "dtr_net_lower" in s:
            print(f"  DTR net value     : {s.dtr_net_lower.min():,.0f} .. "
                  f"{s.dtr_net_upper.max():,.0f} AUD/yr across the sweep")
            if s.dtr_net_lower.min() < 0 < s.dtr_net_upper.max():
                print("  The DTR's net value CHANGES SIGN inside the plausible range. Report that.")

    print(f"\nWritten: {a.out}/cost_per_transformer.csv")
    print(f"         {a.out}/cost_model_detail.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
