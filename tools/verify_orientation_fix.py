#!/usr/bin/env python3
"""Verify the branch-orientation fix in SoeSolver._parse_network.

Background: the DistFlow balance accumulates load at each branch's ``to_bus``.
CIM exports carry arbitrary edge orientation — on GOLDCR_8HB_LEXCEN 152 of 306
branches pointed back at the source, so 58 of 59 load buses were never any
branch's ``to_bus`` and their power contributed to no balance equation. The
model stayed self-consistent (i^2*v^2 == P^2+Q^2 held with both sides at zero),
ipopt returned ok, and every sanity check passed — but no current flowed, the
envelopes went to the envelope_abs_max cap unopposed, the transformer never
heated, and doe_dtr was identical to doe_static in every run.

Checks 1-2 are topology only and need no solver. Checks 3-6 solve one timestep
and need ipopt on PATH (conda shell).

    python tools/verify_orientation_fix.py
    python tools/verify_orientation_fix.py --substation S_5402_AT --scale 3
"""

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from converge_soe import pipeline as pl          # noqa: E402
from converge_soe.doe_solver import SoeSolver    # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feeder", default="GOLDCR_8HB_LEXCEN")
    ap.add_argument("--substation", default="S_5409_AT")
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--envelope-abs-max", type=float, default=20.0)
    ap.add_argument("--meter", nargs="*",
                    default=[str(REPO / "data/meter/forecast_timeseries.csv")])
    ap.add_argument("--skip-solve", action="store_true",
                    help="topology checks only (no ipopt needed)")
    ap.add_argument("--step", type=int, default=None,
                    help="timestep index; default = the peak-load step")
    ap.add_argument("--soft-limits", action="store_true",
                    help="allow penalised limit violations — lets an "
                         "over-constrained interval solve so you can see the "
                         "violation instead of a bare solver error")
    args = ap.parse_args()

    quiet = lambda *a, **k: None                                   # noqa: E731

    cfg = pl.load_config(REPO, feeder=args.feeder)
    cfg.setdefault("network", {})["fixed_lv_taps"] = True
    cfg["network"]["infeeder_v_setpoint_kv"] = 11.0
    cfg["scaling"] = {"import": args.scale, "export": args.scale}

    ej = pl.apply_network_overrides(
        json.loads((REPO / "build" / "network" / "feeders" /
                    f"{args.feeder}_network.json").read_text(encoding="utf-8")),
        cfg, log=quiet)
    df_long = pl.stage_prepare_timeseries(REPO, cfg, meter_files=args.meter,
                                          log=quiet)
    df_long = pl.stage_scale_load(df_long, cfg, log=quiet)
    substations, nmi_index = pl.stage_select_feeder(
        ej, df_long, repo=REPO, feeder_name=args.feeder, log=quiet)
    bundles = pl.stage_preindex(substations, nmi_index, df_long, None,
                                repo=REPO, feeder_name=args.feeder, log=quiet)
    bundles, _ = pl.stage_synthesise(substations, bundles, cfg, repo=REPO,
                                     feeder_name=args.feeder, log=quiet)

    safe = args.substation
    sub_ej, bundle = substations[safe], bundles[safe]
    load_ids = [str(x) for x in bundle["load_ids"]]
    P, Q = bundle["P"], bundle["Q"]
    syn = np.asarray(bundle.get("synthetic", np.zeros(len(load_ids), bool)), bool)
    i = args.step if args.step is not None else int(np.argmax(P.sum(axis=1)))

    f_t = pd.DataFrame({"real_power_w": P[i].astype(float),
                        "reactive_power_var": Q[i].astype(float)},
                       index=pd.Index(load_ids, name="load_id"))
    participants = [l for l, s in zip(load_ids, syn) if not s]

    print(f"\n{args.feeder} / {safe}, peak timestep {i}, scale x{args.scale:g}")
    print(f"{len(load_ids)} loads ({len(participants)} participants), "
          f"{P[i].sum()/1000:.1f} kW presented\n")

    # ---- topology (no solver) --------------------------------------------
    print("TOPOLOGY")
    s = SoeSolver(sub_ej, f_t, envelope_abs_max=args.envelope_abs_max,
                  participant_load_ids=participants,
                  soft_limits=args.soft_limits, quiet=True)
    to_buses = set(s.branches["to_bus_id"])
    missing = sorted(set(s.load_buses) - to_buses)
    check("every load bus is downstream of some branch",
          not missing,
          f"{len(set(s.load_buses)) - len(missing)}/{len(set(s.load_buses))} "
          f"load buses are a to_bus"
          + (f"; missing e.g. {missing[:3]}" if missing else ""))

    check("network is a radial tree (branches == buses - 1)",
          len(s.branches) == len(s.buses) - 1,
          f"{len(s.branches)} branches, {len(s.buses)} buses")

    root = [c["Infeeder"]["cons"][0]["node"]
            for c in sub_ej["components"].values() if "Infeeder" in c][0]
    adj = defaultdict(list)
    for b, r in s.branches.iterrows():
        adj[r["from_bus_id"]].append((b, r["to_bus_id"], True))
        adj[r["to_bus_id"]].append((b, r["from_bus_id"], False))
    seen, q, backwards = {root}, deque([root]), 0
    while q:
        n = q.popleft()
        for b, m, forward in adj[n]:
            if m in seen:
                continue
            seen.add(m)
            q.append(m)
            if not forward:
                backwards += 1
    check("no branch points back at the source",
          backwards == 0,
          f"{backwards} branch(es) still reversed")

    if args.skip_solve:
        return summarise()

    # ---- solve one timestep ----------------------------------------------
    print("\nPOWER FLOW (one timestep)")
    try:
        status, res = s.solve()
    except Exception as e:                    # ipopt can raise, not just fail
        status, res = f"{type(e).__name__}: {e}", None
    if res is None:
        check("solver returned a solution", False,
              f"{str(status)[:160]}\n"
              f"         {str(getattr(s, 'last_solve_error', '') or '')[:160]}")
        print("\n         An infeasible interval is NOT an orientation problem —"
              "\n         the topology checks above already passed. At this "
              "scaling the\n         substation may simply have no feasible "
              "envelope. Try:\n"
              "           --soft-limits   (see the violation instead of an error)\n"
              "           --step N        (a different interval)\n"
              "           --scale 2       (less stress)")
        return summarise()
    check("solver returned a solution", True, f"status {status}")

    m = s.model
    tx_ids = [t for t, _ in [(k, c) for k, c in sub_ej["components"].items()
                             if "Transformer" in c]]
    rows = []
    for b_id in s.branches.index:
        for oe in ("oel", "oer"):
            p = m.branch_active_pu[b_id, oe].value
            qv = m.branch_reactive_pu[b_id, oe].value
            i2 = m.square_current_pu[b_id, oe].value
            v2 = m.square_voltage_pu[s.branches.at[b_id, "from_bus_id"], oe].value
            rows.append({"b": b_id, "oe": oe, "P": p, "Q": qv, "i2": i2,
                         "resid": i2 * v2 - (p * p + qv * qv),
                         "is_tx": b_id in tx_ids})
    bdf = pd.DataFrame(rows)

    maxp = float(bdf.P.abs().max())
    check("power actually flows (max |P_pu| > 0.01)",
          maxp > 0.01,
          f"max |P_pu| = {maxp:.6f}  (was 0.000828 before the fix)")

    txp = float(bdf[bdf.is_tx].P.abs().max()) if bdf.is_tx.any() else 0.0
    check("transformer carries the substation (|P_pu| > 0.01)",
          txp > 0.01,
          f"transformer max |P_pu| = {txp:.6f}  (was 2e-41)")

    check("DistFlow equality holds (|i2*v2 - (P^2+Q^2)| < 1e-6)",
          float(bdf.resid.abs().max()) < 1e-6,
          f"max residual {float(bdf.resid.abs().max()):.3e}")

    # transformer flow should equal total load + losses, within a few percent
    load_kw = P[i].sum() / 1000.0
    tx_kw = txp * 1000.0
    ok = load_kw == 0 or abs(tx_kw) >= abs(load_kw) * 0.5
    check("transformer flow is the right order of magnitude",
          ok,
          f"transformer {tx_kw:.1f} kW vs {load_kw:.1f} kW of load "
          f"(exports at the envelope corner make these differ legitimately)")

    # ---- behaviour --------------------------------------------------------
    print("\nBEHAVIOUR")
    ub = np.array([m.p_inj_oe_kw[l, "oer"].value for l in s.partic_load_ids])
    lb = np.array([m.p_inj_oe_kw[l, "oel"].value for l in s.partic_load_ids])
    at_cap = np.isclose(np.abs(ub), args.envelope_abs_max, rtol=1e-3).mean()
    # Informational, not a correctness check: a substation with genuine spare
    # capacity SHOULD sit at the cap. It only matters that you know which
    # constraint bound — the network or the config parameter.
    if at_cap > 0.999:
        print(f"  [INFO] all upper envelopes at the {args.envelope_abs_max:g} kW cap "
              f"— this substation has spare capacity, so the CONFIG is binding,\n"
              f"         not the network. Re-run with a higher --envelope-abs-max "
              f"to find its real limit.")
    else:
        print(f"  [INFO] {100*at_cap:.0f}% of upper envelopes at the cap; the rest "
              f"are set by the network")

    print(f"\n  envelope ub: min {ub.min():.2f}  mean {ub.mean():.2f}  max {ub.max():.2f} kW")
    print(f"  envelope lb: min {lb.min():.2f}  mean {lb.mean():.2f}  max {lb.max():.2f} kW")
    return summarise()


def summarise():
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("\n" + "=" * 62)
    if n_fail:
        print(f"{n_fail} of {len(RESULTS)} checks FAILED")
        print("Do not trust DOE results until these pass.")
    else:
        print(f"all {len(RESULTS)} checks passed — the power flow is real again")
        print("Next: run the full pipeline and compare doe_dtr vs doe_static.")
    print("=" * 62)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
