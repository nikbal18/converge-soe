#!/usr/bin/env python3
"""
Stage ④ standalone: is this network and this data actually solvable?

    python scripts/preflight.py --network build/network/feeders/F_network.json
    python scripts/preflight.py --network examples/scenario_doe/network.json \
           --timeseries examples/scenario_doe/forecast_timeseries.csv --strict

Runs the NET/TS/PHY/MDL checks in seconds (no solving) and writes
preflight_report.{md,json} + preflight_summary.csv next to the network file
(or to --out). --strict exits nonzero on any ERROR.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from converge_soe import preflight as pfl        # noqa: E402
from converge_soe import timeseries as tsm       # noqa: E402
from converge_soe import pipeline as pl          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", required=True, help="network.json (ejson)")
    ap.add_argument("--parent-feeder", default=None,
                    help="parent feeder json for NET013 subtree diff")
    ap.add_argument("--timeseries", default=None,
                    help="long-format timeseries csv/parquet for TS/PHY checks")
    ap.add_argument("--transformer-params", default=None,
                    help="thermal params json/yaml (default: config class)")
    ap.add_argument("--theta-a", type=float, default=25.0)
    ap.add_argument("--out", default=None, help="report output directory")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    ej = json.loads(Path(args.network).read_text())
    parent = (json.loads(Path(args.parent_feeder).read_text())
              if args.parent_feeder else None)
    cfg = pl.load_config(REPO)

    findings = pfl.check_network(ej, parent_feeder_ejson=parent)

    bundle = None
    if args.timeseries:
        p = Path(args.timeseries)
        df = (pd.read_parquet(p) if p.suffix == ".parquet"
              else tsm.read_long(p))
        net_loads = [k for k, v in ej["components"].items() if "Load" in v]
        findings += pfl.check_timeseries(df, net_loads)
        mapping, _ = tsm.reconcile_load_ids(df["load_id"].unique(), net_loads)
        df2 = df[df["load_id"].isin(mapping)].copy()
        df2["load_id"] = df2["load_id"].map(mapping)
        bundle = tsm.pre_index(df2, sorted(set(mapping.values())))
        tp = None
        if args.transformer_params:
            tp_path = Path(args.transformer_params)
            tp = (json.loads(tp_path.read_text()) if tp_path.suffix == ".json"
                  else __import__("yaml").safe_load(tp_path.read_text()))
        else:
            try:
                tp = pl.load_transformer_params(cfg, REPO)
            except FileNotFoundError:
                pass
        if tp is not None:
            tp = pl.derive_tparams(ej, tp, cfg, float(bundle["dt_minutes"]))
        findings += pfl.check_physical(bundle, ej, tp,
                                       theta_A_const=args.theta_a)
        findings += pfl.check_model(bundle, cfg.get("envelope_abs_max", 50.0),
                                    cfg.get("solver", {}).get("soft_limits", True))

    out = Path(args.out) if args.out else Path(args.network).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "preflight_report.md").write_text(
        pfl.render_markdown(findings, f"Preflight — {Path(args.network).name}"), encoding="utf-8")
    (out / "preflight_report.json").write_text(
        json.dumps(findings, indent=1, default=str), encoding="utf-8")
    pd.DataFrame([pfl.summary_row(Path(args.network).stem, findings,
                                  bundle, ej)]).to_csv(
        out / "preflight_summary.csv", index=False)

    pfl.print_findings(findings)
    v, n_err, _ = pfl.verdict(findings)
    print(f"reports written to {out}")
    return 1 if (args.strict and n_err) else 0


if __name__ == "__main__":
    sys.exit(main())
