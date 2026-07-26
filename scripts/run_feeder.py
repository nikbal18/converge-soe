#!/usr/bin/env python3
"""
The single entry point: run the whole DOE pipeline for one feeder.

    python scripts/run_feeder.py --feeder GOLDCR_8HB_LEXCEN \
           --scenarios doe_dtr,doe_static,bau --jobs 8 --fast --resume

Stages (each skippable via --skip-*):
  ① build network  ② prepare timeseries  ③ select feeder & map NMIs
  ④ preflight      ⑤ pre-index           ⑥ solve  ⑦ (checkpointing, in ⑥)
  ⑧ analyse        ⑨ RUN_SUMMARY.md + config_resolved.yaml

CLI flags override config/default.yaml; the fully resolved configuration is
always written to the output folder so a run is reproducible.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from converge_soe import pipeline as pl        # noqa: E402
from converge_soe import preflight as pfl      # noqa: E402
from converge_soe import timeseries as tsm     # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feeder", default=None,
                    help="feeder id (as found by --list-feeders)")
    ap.add_argument("--feeder-json", default=None,
                    help="use this prebuilt feeder network.json instead of "
                         "building from data/xml")
    ap.add_argument("--meter", nargs="*", default=None,
                    help="meter CSV file(s); default: everything in data/meter/")
    ap.add_argument("--scenarios", default=None,
                    help="comma list from: doe_dtr,doe_static,bau")
    ap.add_argument("--only", default=None,
                    help='substation filter, e.g. "S 5402,S 5406"')
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--fast", action="store_true",
                    help="persistent mutable-param model (Path B)")
    ap.add_argument("--tx-limit", default=None, choices=["dtr", "static", "legacy"],
                    help="override the per-scenario transformer limit mode "
                         "(single-scenario debugging)")
    ap.add_argument("--values-are", default=None,
                    choices=["w", "kw", "kwh_per_interval"])
    ap.add_argument("--theta-a", type=float, default=None,
                    help="constant ambient °C (sets ambient.source=constant)")
    ap.add_argument("--envelope-abs-max", type=float, default=None)
    ap.add_argument("--flush-every", type=int, default=None)
    ap.add_argument("--csv", action="store_true", help="mirror parquet to CSV")
    ap.add_argument("--soft-limits", dest="soft_limits", action="store_true",
                    default=None)
    ap.add_argument("--no-soft-limits", dest="soft_limits", action="store_false")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--restart", action="store_true",
                    help="discard checkpoints and partial outputs")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-strict", action="store_true",
                    help="continue past preflight ERRORs")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-timeseries", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--skip-preindex", action="store_true")
    ap.add_argument("--skip-solve", action="store_true")
    ap.add_argument("--skip-analyse", action="store_true")
    ap.add_argument("--list-feeders", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and a runtime estimate; solve nothing")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    verbosity = 0 if args.quiet else 1 + args.verbose
    logging.basicConfig(
        level=(logging.WARNING if verbosity <= 1 else
               logging.INFO if verbosity == 2 else logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s")
    log = print if verbosity >= 1 else (lambda *a, **k: None)

    overrides = {}
    if args.scenarios:
        overrides["scenarios"] = args.scenarios.split(",")
    if args.jobs is not None:
        overrides["jobs"] = args.jobs
    if args.envelope_abs_max is not None:
        overrides["envelope_abs_max"] = args.envelope_abs_max
    if args.flush_every is not None:
        overrides["flush_every"] = args.flush_every
    if args.csv:
        overrides["csv_mirror"] = True
    if args.fast:
        overrides.setdefault("solver", {})["fast"] = True
    if args.soft_limits is not None:
        overrides.setdefault("solver", {})["soft_limits"] = args.soft_limits
    if args.values_are:
        overrides.setdefault("timeseries", {})["values_are"] = args.values_are
    if args.theta_a is not None:
        overrides["ambient"] = {"source": "constant", "constant_c": args.theta_a}
    if args.tx_limit:
        overrides["tx_limit"] = args.tx_limit

    cfg = pl.load_config(REPO, feeder=args.feeder, overrides=overrides)
    scenarios = cfg.get("scenarios", ["doe_dtr", "doe_static", "bau"])

    # ① build --------------------------------------------------------------
    if not args.skip_build and not args.feeder_json:
        pl.stage_build_network(REPO, cfg, log=log)
    if args.list_feeders:
        df = pl.list_feeders(REPO)
        print(df.to_string(index=False) if len(df) else
              "no feeders built — drop XMLs in data/xml and re-run")
        return 0

    # resolve the feeder network json
    if args.feeder_json:
        feeder_path = Path(args.feeder_json)
        feeder_name = args.feeder or feeder_path.stem.replace("_network", "")
    else:
        if not args.feeder:
            sys.exit("--feeder (or --feeder-json) is required; "
                     "see --list-feeders")
        feeder_name = args.feeder
        feeder_path = (REPO / "build" / "network" / "feeders"
                       / f"{feeder_name}_network.json")
        if not feeder_path.exists():
            sys.exit(f"{feeder_path} not found — run the build stage or pass "
                     "--feeder-json")
    feeder_ej = json.loads(feeder_path.read_text())

    # ② timeseries ---------------------------------------------------------
    df_long = pl.stage_prepare_timeseries(REPO, cfg, meter_files=args.meter,
                                          log=log)

    # ③ select -------------------------------------------------------------
    substations, nmi_index = pl.stage_select_feeder(
        feeder_ej, df_long, repo=REPO, feeder_name=feeder_name,
        only=args.only, log=log)

    # ambient + thermal params
    ambient = pl.load_ambient(REPO, cfg, sorted(df_long["timestamp"].unique()),
                              log=log)
    base_tp = pl.load_transformer_params(cfg, REPO)
    dt_min = tsm.modal_dt_minutes(df_long["timestamp"].unique())
    tparams_by_sub = {safe: pl.derive_tparams(sub_ej, base_tp, cfg, dt_min)
                      for safe, sub_ej in substations.items()}

    # run dir --------------------------------------------------------------
    run_dir, run_id = pl.make_run_dir(REPO, feeder_name, args.run_id)
    pl.write_resolved_config(run_dir, cfg)
    log(f"run: out/{feeder_name}/{run_id}  scenarios={','.join(scenarios)}")

    # ④ preflight ----------------------------------------------------------
    if not args.skip_preflight:
        findings, summary_rows = [], []
        for safe, sub_ej in substations.items():
            F = pfl.check_network(sub_ej, parent_feeder_ejson=feeder_ej,
                                  scope=safe)
            rows = nmi_index[nmi_index["substation"] == safe]
            if not rows.empty:
                sub_ids = set(rows["network_id"])
                F += pfl.check_timeseries(
                    df_long[df_long["load_id"].isin(set(rows["load_id"]))],
                    sub_ids, ambient=ambient,
                    min_coverage=cfg.get("timeseries", {}).get("min_coverage", 0.9),
                    scope=safe)
                b = tsm.pre_index(
                    df_long[df_long["load_id"].isin(set(rows["load_id"]))]
                    .assign(load_id=lambda d: d["load_id"].map(
                        dict(zip(rows["load_id"], rows["network_id"])))),
                    sorted(sub_ids), ambient=ambient)
                F += pfl.check_physical(b, sub_ej, tparams_by_sub[safe],
                                        scope=safe)
                F += pfl.check_model(b, cfg.get("envelope_abs_max", 50.0),
                                     cfg.get("solver", {}).get("soft_limits", True),
                                     scope=safe)
                summary_rows.append(pfl.summary_row(safe, F, b, sub_ej))
            findings += F
        pf_dir = run_dir / "preflight"
        pf_dir.mkdir(exist_ok=True)
        (pf_dir / "preflight_report.md").write_text(
            pfl.render_markdown(findings, f"Preflight — {feeder_name} {run_id}"))
        (pf_dir / "preflight_report.json").write_text(
            json.dumps(findings, indent=1, default=str))
        pd.DataFrame(summary_rows).to_csv(pf_dir / "preflight_summary.csv",
                                          index=False)
        pfl.print_findings(findings, log=log)
        v, n_err, _ = pfl.verdict(findings)
        strict = cfg.get("preflight", {}).get("strict", True) and not args.no_strict
        if n_err and strict:
            sys.exit(f"preflight BLOCKED with {n_err} error(s) — see "
                     f"{pf_dir / 'preflight_report.md'} (or pass --no-strict)")

    # ⑤ pre-index ----------------------------------------------------------
    bundles = pl.stage_preindex(substations, nmi_index, df_long, ambient,
                                repo=REPO, feeder_name=feeder_name, log=log)

    if args.dry_run:
        n_steps = next(iter(bundles.values()))["P"].shape[0] if bundles else 0
        n_runs = len(bundles) * len(scenarios)
        # pilot estimate: 20 timesteps of the first substation
        est = "unknown"
        if bundles:
            import time as _t
            from converge_soe import io as cio
            import tempfile
            safe0 = next(iter(bundles))
            b0 = {k: (v[:20] if getattr(v, "ndim", 0) >= 1 and len(v) >= 20 else v)
                  for k, v in bundles[safe0].items()}
            with tempfile.TemporaryDirectory() as td:
                w = cio.SubstationWriter(td, safe0, "doe_dtr")
                t0 = _t.perf_counter()
                from converge_soe import scenarios as scn
                scn.run_doe_scenario("doe_dtr", substations[safe0], b0,
                                     tparams_by_sub[safe0], cfg, w,
                                     fast=cfg.get("solver", {}).get("fast", False))
                w.close()
                per = (_t.perf_counter() - t0) / 20
            est = f"{per * n_steps * n_runs / max(cfg.get('jobs') if isinstance(cfg.get('jobs'), int) else 4, 1) / 60:.1f} min (~{per*1000:.0f} ms/step)"
        print(f"DRY RUN: {len(bundles)} substation(s) × {len(scenarios)} "
              f"scenario(s) × {n_steps} timesteps; estimated {est}")
        return 0

    # ⑥ solve --------------------------------------------------------------
    if not args.skip_solve:
        results = pl.stage_solve(substations, bundles, tparams_by_sub, cfg,
                                 run_dir, repo=REPO, feeder_name=feeder_name,
                                 scenarios=scenarios, resume=args.resume,
                                 restart=args.restart, verbosity=verbosity,
                                 log=log)
        n_bad = sum(1 for r in results if r["status"] != "ok")
        if n_bad:
            log(f"{n_bad} substation-run(s) did not complete cleanly — see "
                f"{run_dir / '_manifest.json'}")

    # ⑧ analyse ------------------------------------------------------------
    if not args.skip_analyse:
        from converge_soe import analysis
        analysis.run_analysis(run_dir, cfg, feeder_name=feeder_name, log=log)

    log(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
