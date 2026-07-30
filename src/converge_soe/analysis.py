"""Stage ⑧: metrics tables, comparison plots, and the PASS/FAIL sanity block.

Reads the streamed parquet outputs of a run (out/<FEEDER>/<RUN_ID>/scenarios/)
and writes everything into out/<FEEDER>/<RUN_ID>/comparison/. Never re-solves.
"""

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import plots as P

logger = logging.getLogger(__name__)


def _read(run_dir, scenario, sub, table):
    p = Path(run_dir) / "scenarios" / scenario / sub / f"{table}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def discover(run_dir):
    """{scenario: [substation, ...]} found under the run directory."""
    out = {}
    sdir = Path(run_dir) / "scenarios"
    for sc_dir in sorted(sdir.iterdir()) if sdir.exists() else []:
        subs = [p.name for p in sorted(sc_dir.iterdir())
                if (p / "doe.parquet").exists()]
        if subs:
            out[sc_dir.name] = subs
    return out


def run_analysis(run_dir, cfg, feeder_name="FEEDER", log=logger.info,
                 plot_substations=None):
    run_dir = Path(run_dir)
    comp = run_dir / "comparison"
    comp.mkdir(exist_ok=True)
    price = cfg.get("curtailment_price_per_kwh", 0.10)
    price_series = None
    if cfg.get("price_series_csv"):
        ps = pd.read_csv(cfg["price_series_csv"])
        price_series = pd.Series(ps.iloc[:, 1].values,
                                 index=pd.to_datetime(ps.iloc[:, 0]))
    theta_max = 120.0
    run_meta = {"run_id": run_dir.name, "feeder": feeder_name,
                "date": str(date.today()), "price": price}

    found = discover(run_dir)
    if not found:
        log("analysis: no scenario outputs found — nothing to do")
        return None
    subs_all = sorted({s for subs in found.values() for s in subs})

    # ---- common intervals --------------------------------------------------
    # Scenarios MUST be compared over the same set of intervals. A solver
    # failure in one scenario but not another makes the E_curt sums
    # incomparable: on GOLDCR_8HB_LEXCEN doe_dtr solved 46 steps and
    # doe_static 42, which made DTR look like it curtailed MORE than static
    # (504 vs 444 kWh) when on the 41 shared intervals it curtailed less
    # (443.06 vs 444.04) — i.e. the sanity check fired on an artefact.
    common_ts = {}
    for sc, subs in found.items():
        for sub in subs:
            d = _read(run_dir, sc, sub, "doe")
            if d is None or not len(d):
                continue
            ts = set(d["timestamp"].unique())
            common_ts[sub] = ts if sub not in common_ts else (common_ts[sub] & ts)

    dropped = {}
    for sc, subs in found.items():
        for sub in subs:
            d = _read(run_dir, sc, sub, "doe")
            if d is None or not len(d) or sub not in common_ts:
                continue
            n = d["timestamp"].nunique() - len(common_ts[sub])
            if n:
                dropped[(sc, sub)] = n
    if dropped:
        for (sc, sub), n in sorted(dropped.items()):
            logger.warning("analysis: %s/%s — %d interval(s) excluded so every "
                           "scenario is compared over the same %d intervals",
                           sc, sub, n, len(common_ts[sub]))

    def _restrict(df, sub):
        if df is None or not len(df) or sub not in common_ts:
            return df
        if "timestamp" not in df.columns:
            return df
        return df[df["timestamp"].isin(common_ts[sub])]

    # ---- metrics per substation × scenario --------------------------------
    rows = []
    thermal_cache = {}
    doe_cache = {}
    viol_cache = {}
    for sc, subs in found.items():
        for sub in subs:
            doe = _restrict(_read(run_dir, sc, sub, "doe"), sub)
            th = _restrict(_read(run_dir, sc, sub, "thermal"), sub)
            vl = _restrict(_read(run_dir, sc, sub, "viol"), sub)
            doe_cache[(sc, sub)] = doe
            thermal_cache[(sc, sub)] = th
            viol_cache[(sc, sub)] = vl
            r = {"scenario": sc, "substation": sub}
            if doe is not None and len(doe):
                c = M.curtailment(doe, price_per_kwh=price,
                                  price_series=price_series)
                dt_h = c.pop("dt_h")
                c.pop("per_nmi").to_csv(
                    comp / f"per_nmi_{sc}_{sub}.csv")
                r.update(c)
                r["dt_h"] = dt_h
            if th is not None and len(th):
                r.update(M.ageing(th, r.get("dt_h", 0.5)))
                if th["theta_HS_C"].notna().any():
                    theta_max = cfg_theta_max(cfg, theta_max)
            if sc == "doe_dtr" and th is not None and len(th):
                r.update(M.dtr_utilisation(th))
            rows.append(r)
    mdf = pd.DataFrame(rows)
    mdf.to_csv(comp / "metrics_by_substation.csv", index=False)

    # feeder totals
    num_cols = [c for c in mdf.columns
                if c not in ("scenario", "substation")
                and pd.api.types.is_numeric_dtype(mdf[c])]
    feeder_tot = mdf.groupby("scenario")[num_cols].sum(min_count=1)
    # non-additive columns: replace with sensible aggregates
    for c in ("peak_theta_HS_C", "max_headroom"):
        if c in mdf.columns:
            feeder_tot[c] = mdf.groupby("scenario")[c].max()
    for c in ("mean_F_AA", "mean_headroom", "share_above_nameplate", "dt_h"):
        if c in mdf.columns:
            feeder_tot[c] = mdf.groupby("scenario")[c].mean()
    feeder_tot.to_csv(comp / "metrics_feeder.csv")

    # wide scenario comparison with deltas — paste-ready
    wide = feeder_tot.T
    if {"doe_static", "doe_dtr"} <= set(wide.columns):
        wide["delta_dtr_minus_static"] = wide["doe_dtr"] - wide["doe_static"]
    if "bau" in wide.columns and "doe_dtr" in wide.columns:
        wide["delta_dtr_minus_bau"] = wide["doe_dtr"] - wide["bau"]
    wide.to_csv(comp / "scenario_comparison.csv")

    # ---- plots -------------------------------------------------------------
    try:
        P.curtailed_energy_by_scenario(mdf, comp, run_meta)
        P.curtailment_cost_by_scenario(mdf, comp, run_meta, price)
        P.energy_enabled_by_dtr(mdf, comp, run_meta)
        P.transformer_ageing_by_scenario(mdf, comp, run_meta)
        P.avoided_ageing(mdf, comp, run_meta)

        # per-substation detail plots: the worst substation by BAU peak θ_HS,
        # plus any requested via plot_substations
        worst = None
        if "peak_theta_HS_C" in mdf.columns and (mdf["scenario"] == "bau").any():
            worst = (mdf[mdf["scenario"] == "bau"]
                     .sort_values("peak_theta_HS_C", ascending=False)
                     ["substation"].iloc[0])
        wanted = set(plot_substations or [])
        if worst:
            wanted.add(worst)
        if not wanted and subs_all:
            wanted = {subs_all[0]}
        for sub in wanted:
            th_by_sc = {sc: thermal_cache[(sc, sub)]
                        for sc in found if thermal_cache.get((sc, sub)) is not None
                        and len(thermal_cache[(sc, sub)])}
            doe_by_sc = {sc: doe_cache[(sc, sub)]
                         for sc in found if doe_cache.get((sc, sub)) is not None}
            if "doe_dtr" in th_by_sc:
                P.dynamic_rating_vs_static(th_by_sc["doe_dtr"], sub, comp, run_meta)
            if th_by_sc:
                P.hotspot_timeseries(th_by_sc, sub, theta_max, comp, run_meta)
            if doe_by_sc:
                P.envelope_vs_desired(doe_by_sc, sub, comp, run_meta)

        th_all = {sc: pd.concat([thermal_cache[(sc, s)] for s in subs
                                 if thermal_cache.get((sc, s)) is not None],
                                ignore_index=True)
                  for sc, subs in found.items()
                  if any(thermal_cache.get((sc, s)) is not None for s in subs)}
        th_all = {sc: t for sc, t in th_all.items() if len(t)}
        if th_all:
            P.hotspot_duration_curve(th_all, theta_max, comp, run_meta)
            P.transformer_loading_duration_curve(th_all, comp, run_meta)
        doe_all = {sc: pd.concat([doe_cache[(sc, s)] for s in subs
                                  if doe_cache.get((sc, s)) is not None],
                                 ignore_index=True)
                   for sc, subs in found.items()}
        P.seasonal_curtailment_heatmap(doe_all, comp, run_meta)
        viol_all = {sc: (pd.concat([v for s in subs
                                    if (v := viol_cache.get((sc, s))) is not None],
                                   ignore_index=True)
                         if any(viol_cache.get((sc, s)) is not None for s in subs)
                         else pd.DataFrame())
                    for sc, subs in found.items()}
        P.violations_summary(viol_all, comp, run_meta)
    except Exception:
        logger.exception("plotting failed (metrics tables are still written)")

    # ---- sanity checks -----------------------------------------------------
    checks = sanity_checks(mdf, thermal_cache, doe_cache, found, theta_max,
                           run_dir)
    lines = [f"# Sanity checks — {feeder_name} {run_dir.name}", ""]
    n_fail = 0
    for name, ok, detail in checks:
        tag = "PASS" if ok else ("FAIL" if ok is False else "NOTE")
        n_fail += ok is False
        lines.append(f"- **{tag}** {name}" + (f" — {detail}" if detail else ""))
        log(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    (comp / "sanity_checks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_run_summary(run_dir, feeder_name, cfg, mdf, feeder_tot, checks)
    log(f"analysis written to {comp}")
    return mdf


def cfg_theta_max(cfg, default):
    try:
        from . import pipeline as pl
        return float(pl.load_transformer_params(cfg).get("theta_HS_max", default))
    except Exception:
        return default


def sanity_checks(mdf, thermal_cache, doe_cache, found, theta_max, run_dir):
    """The believe-the-results checklist. Returns [(name, ok|None, detail)]."""
    checks = []

    def get(sc, col):
        return mdf.loc[mdf["scenario"] == sc].set_index("substation")[col] \
            if col in mdf.columns and (mdf["scenario"] == sc).any() else None

    # E_curt(bau) == 0 by construction
    e_bau = get("bau", "E_curt_kwh")
    if e_bau is not None:
        ok = bool((e_bau.fillna(0) == 0).all())
        checks.append(("E_curt(bau) == 0 by construction", ok,
                       "" if ok else f"nonzero: {e_bau[e_bau != 0].to_dict()}"))

    # E_curt(doe_dtr) <= E_curt(doe_static) at every substation
    e_s, e_d = get("doe_static", "E_curt_kwh"), get("doe_dtr", "E_curt_kwh")
    if e_s is not None and e_d is not None:
        joined = pd.concat([e_s.rename("static"), e_d.rename("dtr")], axis=1).dropna()
        bad = joined[joined["dtr"] > joined["static"] * 1.001 + 0.01]
        checks.append(("E_curt(doe_dtr) ≤ E_curt(doe_static) at every substation",
                       bad.empty,
                       "" if bad.empty else
                       f"THE DTR IS MISCONFIGURED (a static limit still binds?): "
                       f"{bad.round(2).to_dict('index')}"))

    # peak theta_HS in doe_dtr <= theta_max + tol except already_over_limit
    viols = []
    for (sc, sub), th in thermal_cache.items():
        if sc != "doe_dtr" or th is None or not len(th):
            continue
        over = th[(th["theta_HS_C"] > theta_max + 0.5)
                  & (th["dtr_status"] != "already_over_limit")]
        if len(over):
            viols.append((sub, len(over)))
    checks.append((f"peak θ_HS(doe_dtr) ≤ θ_HS_max ({theta_max:g} °C) + tol, "
                   "except status=already_over_limit",
                   not viols, str(viols) if viols else ""))

    # envelopes contain zero
    bad_env = []
    for (sc, sub), doe in doe_cache.items():
        if sc == "bau" or doe is None or not len(doe):
            continue
        n = int(((doe["doe_lb_kw"] > 1e-6) | (doe["doe_ub_kw"] < -1e-6)).sum())
        if n:
            bad_env.append((sc, sub, n))
    checks.append(("doe_lb_kw ≤ 0 ≤ doe_ub_kw on every row", not bad_env,
                   str(bad_env) if bad_env else ""))

    # total desired export > 0
    any_doe = next((d for d in doe_cache.values() if d is not None and len(d)), None)
    if any_doe is not None:
        tot_exp = float(any_doe["p_des_kw"].clip(lower=0).sum())
        checks.append(("total desired export > 0 (the timeseries contains PV)",
                       tot_exp > 0,
                       f"{tot_exp:.1f} kW·steps" if tot_exp > 0 else
                       "no PV export in the data — every curtailment result "
                       "is trivial"))

    # energy balance (BAU): tx inflow vs sum of loads
    for (sc, sub), doe in doe_cache.items():
        if sc != "bau" or doe is None:
            continue
        br = pd.read_parquet(Path(run_dir) / "scenarios" / sc / sub / "branch.parquet") \
            if (Path(run_dir) / "scenarios" / sc / sub / "branch.parquet").exists() else None
        if br is None:
            continue
        # This is an INTERNAL consistency check on one scenario, not a
        # cross-scenario comparison, so both sides must cover the same
        # intervals. doe_cache is restricted to the intervals every scenario
        # solved; branch.parquet is not, so restrict it the same way or the
        # residual is pure sampling mismatch (43% on S_5402_AT).
        if "timestamp" in br.columns and len(doe):
            br = br[br["timestamp"].isin(set(doe["timestamp"].unique()))]
        # transformer = branch whose id appears in thermal table
        th = thermal_cache.get((sc, sub))
        if th is None or not len(th):
            continue
        tx_id = th["transformer_id"].iloc[0]
        tx = br[br["id"] == tx_id]
        inflow = tx["p_w_oel"].sum() / 1000.0
        loadsum = float(-doe["p_des_kw"].sum())
        rel = abs(inflow - loadsum) / max(abs(loadsum), 1e-9)
        checks.append((f"energy balance (bau, {sub}): tx inflow ≈ Σ loads + losses",
                       rel < 0.01 if loadsum else None,
                       f"residual {100 * rel:.2f} %"))
        break  # one substation is enough for the block

    # ageing ordering
    l_bau, l_s, l_d = (get("bau", "ageing_hours"), get("doe_static", "ageing_hours"),
                       get("doe_dtr", "ageing_hours"))
    if l_bau is not None and l_s is not None:
        j = pd.concat([l_bau.rename("bau"), l_s.rename("static")], axis=1).dropna()
        ok = bool((j["bau"] >= j["static"] - 1e-9).all())
        checks.append(("L(bau) ≥ L(doe_static) (DOE should not age the "
                       "transformer more than BAU)", ok,
                       "" if ok else str(j[j['bau'] < j['static']].to_dict())))
    if l_s is not None and l_d is not None:
        j = pd.concat([l_s.rename("static"), l_d.rename("dtr")], axis=1).dropna()
        more = j[j["dtr"] > j["static"]]
        checks.append(("L(doe_dtr) vs L(doe_static): NOT asserted either way — "
                       "DTR deliberately trades ageing for export headroom",
                       None,
                       f"DTR ages more at {len(more)}/{len(j)} substation(s); "
                       "this is expected behaviour, not a bug"))
    return checks


def write_run_summary(run_dir, feeder_name, cfg, mdf, feeder_tot, checks):
    run_dir = Path(run_dir)
    price = cfg.get("curtailment_price_per_kwh", 0.10)
    lines = [f"# Run summary — {feeder_name} / {run_dir.name}", ""]
    manifest = {}
    mp = run_dir / "_manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text())
    for sc, subs in manifest.items():
        n_ok = sum(1 for v in subs.values() if v.get("status") == "ok")
        lines.append(f"- **{sc}**: {n_ok}/{len(subs)} substations completed")
    lines += ["", f"Assumed flat curtailment price: **${price:.2f}/kWh** "
              "(assumption, not a result).", ""]
    if len(feeder_tot):
        lines += ["## Headline numbers (feeder totals)", "",
                  feeder_tot.round(3).to_markdown(), ""]
        t = feeder_tot
        if {"doe_static", "doe_dtr"} <= set(t.index) and "E_curt_kwh" in t.columns:
            en = t.at["doe_static", "E_curt_kwh"] - t.at["doe_dtr", "E_curt_kwh"]
            lines += [f"**Export enabled by the dynamic thermal rating: "
                      f"{en:.1f} kWh** over the run period "
                      f"(≈ ${en * price:.2f} at the assumed price).", ""]
    n_fail = sum(1 for _, ok, _ in checks if ok is False)
    lines += [f"## Sanity checks: "
              f"{'ALL PASS' if n_fail == 0 else f'{n_fail} FAILED'} "
              f"(see comparison/sanity_checks.md)", "",
              "Outputs: `comparison/` for tables and figures, "
              "`scenarios/<scenario>/<substation>/` for full per-timestep "
              "parquet, `preflight/` for the input checks. "
              "See docs/RESULTS_GUIDE.md for how to read every number.", ""]
    (run_dir / "RUN_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
