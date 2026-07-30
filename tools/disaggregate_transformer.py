#!/usr/bin/env python3
"""Standalone front-end to the synthetic-profile stage.

The pipeline runs this automatically as stage ⑤b (see
``converge_soe.synthetic`` and ``pipeline.stage_synthesise``), so this script
is only needed to inspect or export the profiles outside a full run — e.g. to
eyeball what a substation's unmapped NMIs will be given before solving.

    python tools/disaggregate_transformer.py \\
        --network data/xml/GOLDCR_8HB_LEXCEN_network.json \\
        --forecast data/meter/forecast_timeseries.csv \\
        --out examples/scenario_real

All the logic lives in ``converge_soe.synthetic``; this file only reads inputs,
calls it and writes CSV/JSON. Earlier versions of this script disaggregated the
whole transformer total into a fresh synthetic customer population — that
behaviour is now the *fallback* path inside ``synthetic.py``, used only when a
substation has fewer than ``min_donors`` usable donors.

NOTE: the outputs contain real metered profiles copied onto other NMI ids, and
the report contains a real-NMI -> real-NMI donor map. They are exactly as
confidential as the raw meter data and are covered by .gitignore.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from converge_soe import synthetic as sy          # noqa: E402
from converge_soe import timeseries as tsm        # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", type=Path,
                   default=REPO / "data/xml/GOLDCR_8HB_LEXCEN_network.json",
                   help="network JSON holding the Load population to fill in")
    p.add_argument("--forecast", type=Path,
                   default=REPO / "data/meter/forecast_timeseries.csv",
                   help="metered profiles: both the 'who has data' set and the "
                        "donor pool")
    p.add_argument("--out", type=Path, default=REPO / "examples/scenario_real")
    p.add_argument("--substation", default="",
                   help="name used to look up synthetic.transformer_series")
    p.add_argument("--config", type=Path, default=REPO / "config/default.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-donors", type=int, default=None)
    p.add_argument("--allow-feeder-donors", action="store_true", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    cfg = {}
    if args.config.exists():
        import yaml
        cfg = yaml.safe_load(args.config.read_text()) or {}
    scfg = sy.config(cfg)
    for key, val in (("seed", args.seed), ("min_donors", args.min_donors),
                     ("allow_feeder_donors", args.allow_feeder_donors)):
        if val is not None:
            scfg[key] = val

    netw = json.loads(args.network.read_text())
    fc = pd.read_csv(args.forecast)
    fc["load_id"] = fc["load_id"].astype(str)
    fc["timestamp"] = pd.to_datetime(fc["timestamp"])

    net_ids = set(sy.network_loads(netw))
    in_net = fc[fc["load_id"].isin(net_ids)]
    if in_net.empty:
        raise SystemExit(
            f"none of the {fc['load_id'].nunique()} ids in {args.forecast.name} "
            f"match a Load in {args.network.name} — wrong pair of files?")

    bundle = tsm.pre_index(in_net, sorted(in_net["load_id"].unique()))
    bank = sy._donor_bank({"all": tsm.pre_index(fc, sorted(fc["load_id"].unique()))})

    b, rep = sy.synthesise_substation(
        netw, bundle, scfg, substation=args.substation,
        rng=np.random.default_rng(scfg["seed"]), feeder_donors=bank)

    print(f"network Loads      : {rep['n_network_real_loads']}")
    print(f"with meter data    : {rep['n_with_data']}")
    print(f"unmapped (filled)  : {rep['n_gaps']}")
    print(f"method             : {rep['method']}  [{rep['flag']}]")
    print(f"donors             : {rep['n_donors']} "
          f"(local {rep.get('n_donors_local', 0)}, "
          f"feeder {rep.get('n_donors_from_feeder', 0)}, "
          f"reused {rep['n_donors_reused']})")
    for n in rep["notes"]:
        print(f"  ! {n}")

    if not rep["n_gaps"]:
        return 0

    syn = b["synthetic"]
    ids = np.array(list(map(str, b["load_ids"])))
    idx = tsm.timestamps_index(b)
    out = pd.concat([
        pd.DataFrame({"timestamp": idx.strftime("%Y-%m-%d %H:%M"),
                      "load_id": ids[j],
                      "real_power_w": np.round(b["P"][:, j], 2),
                      "reactive_power_var": np.round(b["Q"][:, j], 2)})
        for j in np.where(syn)[0]], ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "background_timeseries.csv"
    rep_path = args.out / "disaggregation_report.json"
    out.to_csv(csv_path, index=False)
    rep_path.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")

    real_kw = b["P"][:, ~syn].sum(axis=1).mean() / 1000.0
    syn_kw = b["P"][:, syn].sum(axis=1).mean() / 1000.0
    print(f"\nmean metered load  : {real_kw:8.1f} kW")
    print(f"mean synthetic load: {syn_kw:8.1f} kW")
    print(f"synthetic share    : {100*syn_kw/max(syn_kw+real_kw,1e-9):8.1f}%")
    print(f"\nwrote {csv_path}\nwrote {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
