#!/usr/bin/env python3
"""Why does the DOE optimiser report ~zero transformer current?

At 3x load the BAU tree-sweep power flow puts S_5409_AT at K2 = 1.687 and
theta_HS = 126.8 C, while the Pyomo DOE model reports K2_actual = 0.000 and
branch currents of ~3.5 A on a transformer rated ~348 A. Since
`scenarios.py` caps the actual-behaviour current at the solver's envelope
corner (`i2_pu = min(i2_pu, i2_corner)`), a near-zero corner current zeroes the
whole thermal trajectory — so the DTR limit can never bind and doe_dtr is
always identical to doe_static.

This solves ONE timestep and reports where the power went:

  * total load actually presented to the model
  * total envelope width granted
  * total nodal slack used (sof_bus_a_kw / sof_bus_r_kw) — these are the
    "virtual generator" variables added at every load bus with a hardcoded
    big_weight = 1000 penalty. If they are large, the model is balancing the
    network with slack instead of branch flow, which would explain currents
    collapsing to their 1e-8 lower bound.
  * resulting branch current at the transformer vs its rating

Run from the repo root, in a conda shell where ipopt is on PATH:

    python tools/diagnose_zero_current.py --substation S_5409_AT --scale 3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from converge_soe import pipeline as pl          # noqa: E402
from converge_soe import timeseries as tsm       # noqa: E402
from converge_soe.doe_solver import SoeSolver    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeder", default="GOLDCR_8HB_LEXCEN")
    ap.add_argument("--substation", default="S_5409_AT")
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--step", type=int, default=None,
                    help="timestep index; default = the peak-load step")
    ap.add_argument("--envelope-abs-max", type=float, default=50.0)
    ap.add_argument("--meter", nargs="*",
                    default=[str(REPO / "data/meter/forecast_timeseries.csv")],
                    help="meter CSV(s). Defaults to the long-format file — the "
                         "whole data/meter/ folder also contains the raw wide "
                         "export, which needs an explicit --values-are.")
    args = ap.parse_args()

    cfg = pl.load_config(REPO, feeder=args.feeder)
    cfg.setdefault("network", {})["fixed_lv_taps"] = True
    cfg["network"]["infeeder_v_setpoint_kv"] = 11.0
    cfg["scaling"] = {"import": args.scale, "export": args.scale}

    feeder_path = REPO / "build" / "network" / "feeders" / f"{args.feeder}_network.json"
    feeder_ej = pl.apply_network_overrides(
        json.loads(feeder_path.read_text(encoding="utf-8")), cfg,
        log=lambda *a, **k: None)

    df_long = pl.stage_prepare_timeseries(REPO, cfg, meter_files=args.meter,
                                          log=lambda *a, **k: None)
    df_long = pl.stage_scale_load(df_long, cfg, log=lambda *a, **k: None)
    substations, nmi_index = pl.stage_select_feeder(
        feeder_ej, df_long, repo=REPO, feeder_name=args.feeder,
        log=lambda *a, **k: None)
    bundles = pl.stage_preindex(substations, nmi_index, df_long, None,
                                repo=REPO, feeder_name=args.feeder,
                                log=lambda *a, **k: None)
    bundles, _ = pl.stage_synthesise(substations, bundles, cfg, repo=REPO,
                                     feeder_name=args.feeder,
                                     log=lambda *a, **k: None)

    safe = args.substation
    if safe not in bundles:
        raise SystemExit(f"{safe} not in {list(bundles)}")
    sub_ej, bundle = substations[safe], bundles[safe]
    load_ids = [str(x) for x in bundle["load_ids"]]
    P, Q = bundle["P"], bundle["Q"]
    syn = np.asarray(bundle.get("synthetic", np.zeros(len(load_ids), bool)), bool)

    i = args.step if args.step is not None else int(np.argmax(P.sum(axis=1)))
    f_t = pd.DataFrame({"real_power_w": P[i].astype(float),
                        "reactive_power_var": Q[i].astype(float)},
                       index=pd.Index(load_ids, name="load_id"))
    participants = [l for l, s in zip(load_ids, syn) if not s]

    print(f"substation {safe}, timestep {i}, scale x{args.scale:g}")
    print(f"  loads: {len(load_ids)} ({len(participants)} participants, "
          f"{int(syn.sum())} synthetic non-participants)")
    print(f"  total real power presented : {P[i].sum()/1000:10.2f} kW")
    print(f"    of which participants    : {P[i][~syn].sum()/1000:10.2f} kW")
    print(f"    of which synthetic       : {P[i][syn].sum()/1000:10.2f} kW")

    s = SoeSolver(sub_ej, f_t, envelope_abs_max=args.envelope_abs_max,
                  participant_load_ids=participants, quiet=True)
    status, res = s.solve()
    print(f"  solver status: {status}")
    if res is None:
        raise SystemExit(f"solve failed: {getattr(s,'last_solve_error','')}")

    m = s.model
    width = sum(m.p_inj_oe_kw[l, "oer"].value - m.p_inj_oe_kw[l, "oel"].value
                for l in s.partic_load_ids)
    slack_a = sum(m.sof_bus_a_kw[k].value or 0.0 for k in m.sof_bus_a_kw)
    slack_r = sum(m.sof_bus_r_kw[k].value or 0.0 for k in m.sof_bus_r_kw)
    print(f"  total envelope width granted: {width:10.2f} kW "
          f"({width/max(len(s.partic_load_ids),1):.1f} kW/NMI, cap "
          f"{args.envelope_abs_max:g})")
    print(f"  nodal slack used  active    : {slack_a:10.2f} kW   <-- if this is "
          f"large, the model is balancing with slack, not branch flow")
    print(f"  nodal slack used  reactive  : {slack_r:10.2f} kVAr")

    br = res.branch
    cc = [c for c in br.columns if "current" in c]
    print(f"  branch current max          : "
          f"{br[cc].abs().to_numpy().max():10.2f} A  over {len(br)} branches")
    tx = [(k, c["Transformer"]) for k, c in sub_ej["components"].items()
          if "Transformer" in c]
    for tid, t in tx:
        if tid in br.index:
            i_rated = t["s_max"] * 1e6 / (t["v_winding_base"][1] * 1000)
            i_a = max(abs(float(br.at[tid, c])) for c in cc)
            print(f"  transformer {tid}: {i_a:.2f} A of ~{i_rated:.0f} A rated "
                  f"= {100*i_a/i_rated:.1f}%")

    # --- where does the balance actually go? --------------------------------
    # branch_active_pu[b] == (load at its to_bus) + r*i^2 + sum(downstream)
    # so the transformer branch should accumulate the whole substation. If P is
    # ~0 there, the nodal injections never reached the balance. If P is large
    # but i^2 is ~0, the i^2*v^2 == P^2+Q^2 equality is being violated or the
    # reported current is a units problem.
    print("\n  --- per-branch variable values (pu on a 1 MVA base) ---")
    tx_ids = [t for t, _ in tx]
    rows = []
    for b_id in s.branches.index:
        for oe in ("oel", "oer"):
            p = m.branch_active_pu[b_id, oe].value
            q = m.branch_reactive_pu[b_id, oe].value
            i2 = m.square_current_pu[b_id, oe].value
            frm = s.branches.at[b_id, "from_bus_id"]
            v2 = m.square_voltage_pu[frm, oe].value
            rows.append({"branch": b_id, "oe": oe, "is_tx": b_id in tx_ids,
                         "P_pu": p, "Q_pu": q, "i2_pu": i2, "v2_pu": v2,
                         "resid": (i2 * v2) - (p * p + q * q)})
    bdf = pd.DataFrame(rows)
    print(f"  max |P_pu| over all branches : {bdf.P_pu.abs().max():.6f}")
    print(f"  max  i2_pu over all branches : {bdf.i2_pu.abs().max():.6e}")
    print(f"  max |i2*v2 - (P^2+Q^2)|      : {bdf.resid.abs().max():.3e}  "
          f"(should be ~0 — it is an equality constraint)")
    txr = bdf[bdf.is_tx]
    if len(txr):
        print("  transformer branches:")
        print(txr[["branch", "oe", "P_pu", "Q_pu", "i2_pu", "v2_pu",
                   "resid"]].to_string(index=False))

    tot_inj = sum(m.p_inj_oe_kw[l, "oer"].value for l in s.partic_load_ids)
    print(f"\n  sum p_inj_oe_kw at the 'oer' corner: {tot_inj:.1f} kW "
          f"(export) vs {P[i][syn].sum()/1000:.1f} kW synthetic consumption")
    print(f"  => net at the corner ~{tot_inj - P[i][syn].sum()/1000:.1f} kW, "
          f"which must cross the transformer")

    print("\ninterpretation")
    print("  slack ~0 and transformer current ~0  -> the branch-flow relaxation "
          "is not tight: current is free to sit at its 1e-8 bound because "
          "nothing in the objective pushes it up")
    print("  slack large                          -> the sof_bus_* variables "
          "(big_weight=1000, hardcoded) are absorbing the load; raise that "
          "penalty or constrain them")


if __name__ == "__main__":
    main()
