#!/usr/bin/env python3
"""
Profile the DOE solve loop and attribute the wall time to pipeline stages.

Runs N timesteps (default 200) of examples/scenario_doe through the
per-timestep DoeSolver loop (the legacy runner pattern) under cProfile,
prints the top cumulative entries grouped into six buckets, and saves the
report. Re-run after each optimisation and record the delta — do not claim a
speedup you have not measured.

Buckets:
  build_model     Pyomo expression/model construction
  nl_write_spawn  .nl serialisation + ipopt subprocess spawn + .sol parse
  ipopt_solve     time actually inside the ipopt process (subprocess wait)
  pandas_slice    timeseries slicing / DataFrame scans in the loop
  network_build   ejson parsing and per-unit conversion per step
  extract         reading results back into DataFrames

Usage:
    python tools/profile_doe.py                       # 200 steps, plain path
    python tools/profile_doe.py -n 50 --fast          # persistent path
    python tools/profile_doe.py -o docs/profile_after.txt
"""

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCEN = REPO / "examples" / "scenario_doe"

BUCKETS = {
    "build_model": ("_build_opt_model", "constraint", "expr_", "numeric_expr",
                    "visitor", "ConstraintList", "construct", "_generate",
                    "relational_expr", "indexed_component", "pyomo/core"),
    "nl_write_spawn": ("write_nl", "nl_writer", "_presolve", "_apply_solver",
                       "subprocess", "popen", "Popen", "_execute_child",
                       "parse_sol", "sol_reader", "_postsolve",
                       "tempfile", "opt/base"),
    "ipopt_solve": ("wait", "poll", "communicate", "_try_wait", "selectors"),
    "pandas_slice": ("pandas", "frame.py", "series.py", "indexing.py",
                     "_libs", "reindex", "merge"),
    "network_build": ("_build_network_data", "_filter_input_data",
                      "_netw_components"),
    "extract": ("_extract_results", "warm_start_values"),
}


def bucket_of(key):
    filename, lineno, funcname = key
    hay = f"{filename}:{funcname}"
    for b in ("network_build", "extract", "build_model", "nl_write_spawn",
              "ipopt_solve", "pandas_slice"):
        for pat in BUCKETS[b]:
            if pat in hay:
                return b
    return "other"


def make_timeseries(n_steps):
    fc = pd.read_csv(SCEN / "forecast_timeseries.csv", dtype={"load_id": str})
    fc["timestamp"] = pd.to_datetime(fc["timestamp"], format="%H:%M")
    base = fc.copy()
    frames = []
    n_have = fc["timestamp"].nunique()
    reps = int(np.ceil(n_steps / n_have))
    for r in range(reps):
        f = base.copy()
        f["timestamp"] = f["timestamp"] + pd.Timedelta(minutes=30 * n_have * r)
        frames.append(f)
    fc = pd.concat(frames, ignore_index=True)
    keep = sorted(fc["timestamp"].unique())[:n_steps]
    return fc[fc["timestamp"].isin(keep)].copy()


def run_plain(netw, tp, fc, tx_limit):
    import converge_soe as csoe
    state = {}
    n_solved = 0
    for ts in sorted(fc["timestamp"].unique()):
        # NOTE: this .loc boolean scan over the whole frame each step is the
        # legacy pattern being measured (O(T^2) overall).
        f_t = (fc.loc[fc["timestamp"] == ts,
                      ["load_id", "real_power_w", "reactive_power_var"]]
               .set_index("load_id"))
        s = csoe.DoeSolver(netw, f_t, envelope_abs_max=50.0,
                           transformer_params=tp, theta_A=25.0,
                           thermal_state_in=state, tx_limit=tx_limit,
                           quiet=True)
        status, res = s.solve()
        if res is None:
            continue
        state = res.thermal_state
        n_solved += 1
    return n_solved


def run_fast(netw, tp, fc):
    from converge_soe.persistent_solver import PersistentDoeSolver
    tss = sorted(fc["timestamp"].unique())
    loads = sorted(fc["load_id"].unique())
    wide_p = fc.pivot(index="timestamp", columns="load_id", values="real_power_w").reindex(columns=loads)
    wide_q = fc.pivot(index="timestamp", columns="load_id", values="reactive_power_var").reindex(columns=loads)
    ps = PersistentDoeSolver(netw, loads, transformer_params=tp,
                             tx_limit="dtr", quiet=True)
    n_solved = 0
    for i, ts in enumerate(tss):
        res = ps.solve_step(wide_p.values[i], wide_q.values[i], theta_A=25.0)
        if res is not None:
            n_solved += 1
    return n_solved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--n-steps", type=int, default=200)
    ap.add_argument("--fast", action="store_true", help="profile the persistent path")
    ap.add_argument("--tx-limit", default="legacy",
                    choices=["legacy", "dtr", "static"])
    ap.add_argument("-o", "--output", default=str(REPO / "docs" / "profile_baseline.txt"))
    args = ap.parse_args()

    netw = json.loads((SCEN / "network.json").read_text())
    tp = json.loads((SCEN / "transformer_params.json").read_text())
    fc = make_timeseries(args.n_steps)

    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    if args.fast:
        n = run_fast(netw, tp, fc)
    else:
        n = run_plain(netw, tp, fc, args.tx_limit)
    prof.disable()
    wall = time.perf_counter() - t0

    stats = pstats.Stats(prof)
    stats.sort_stats("cumulative")

    # Bucket by *self* time so buckets don't double-count callees.
    totals = {}
    for key, (cc, nc, tt, ct, callers) in stats.stats.items():
        b = bucket_of(key)
        totals[b] = totals.get(b, 0.0) + tt
    tsum = sum(totals.values()) or 1.0

    buf = io.StringIO()
    mode = "fast/persistent" if args.fast else f"plain (tx_limit={args.tx_limit})"
    buf.write(f"DOE solve-loop profile — {n}/{args.n_steps} timesteps solved, "
              f"mode: {mode}\n")
    buf.write(f"wall time: {wall:.2f} s  ({wall / max(n,1) * 1000:.1f} ms/timestep)\n\n")
    buf.write("Self-time by bucket:\n")
    for b, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        buf.write(f"  {b:<16} {t:8.2f} s   {100 * t / tsum:5.1f} %\n")
    buf.write("\nTop 30 by cumulative time:\n")
    stats.stream = buf
    stats.print_stats(30)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(buf.getvalue())
    print(buf.getvalue()[:2200])
    print(f"\nFull report written to {out}")


if __name__ == "__main__":
    main()
