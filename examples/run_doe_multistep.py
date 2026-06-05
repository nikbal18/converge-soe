#!/usr/bin/env python3
"""
Multi-timestep DOE solver.

Reads forecast_timeseries.csv (columns: timestamp, load_id, real_power_w, reactive_power_var)
and runs DoeSolver for each timestep, carrying the transformer thermal state forward.

transformer_params.json keys (IEEE C57.91):
  tau_TO          - top-oil thermal time constant (min)
  tau_W           - winding thermal time constant (min)
  delta_theta_TO_R - rated top-oil temperature rise (°C)
  delta_theta_HS_R - rated hot-spot rise above top-oil at rated load (°C)
  R               - ratio of load loss to no-load loss
  n               - top-oil exponent (0.8 ONAN, 1.0 OFAF)
  m               - winding exponent (0.8 ONAN, 1.0 OFAF)
  I_rated         - rated secondary current of transformer (A)
  theta_HS_max    - hot-spot temperature limit (°C)
  dt              - timestep length, must match interval between rows (min)

Usage:
    python run_doe_multistep.py examples/scenario_doe examples/scenario_doe_multistep_output
"""

import argparse
import json
from pathlib import Path
import logging

import pandas as pd

import converge_soe as csoe

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('indir',  type=str, help='Input folder (network.json, forecast_timeseries.csv, transformer_params.json)')
parser.add_argument('outdir', type=str, help='Output folder')
args = parser.parse_args()

indir  = Path(args.indir)
outdir = Path(args.outdir)
outdir.mkdir(exist_ok=True, parents=True)

# --- Load inputs -----------------------------------------------------------------

with (indir / 'network.json').open() as f:
    netw = json.load(f)

with (indir / 'transformer_params.json').open() as f:
    transformer_params = json.load(f)

forecast_ts = pd.read_csv(
    indir / 'forecast_timeseries.csv',
    dtype={'load_id': str, 'timestamp': str}
)

# Ambient temperature — fixed here, but could be read from a CSV per timestep.
theta_A = 30.0  # °C

# ---------------------------------------------------------------------------------

csoe.logger.addHandler(logging.FileHandler(outdir / 'log.txt'))

thermal_state = {}   # carries forward between timesteps; empty = cold start

all_doe     = []
all_bus     = []
all_branch  = []
all_viol    = []
all_thermal = []

for timestamp in forecast_ts['timestamp'].unique():
    logging.info(f"--- Solving timestep {timestamp} ---")

    forecast_t = (
        forecast_ts.loc[forecast_ts['timestamp'] == timestamp,
                        ['load_id', 'real_power_w', 'reactive_power_var']]
        .set_index('load_id')
    )

    solver = csoe.DoeSolver(
        netw, forecast_t,
        envelope_abs_max=50.0,
        transformer_params=transformer_params,
        theta_A=theta_A,
        thermal_state_in=thermal_state,
        solver_options={}
    )
    status, results = solver.solve()

    if results is None:
        logging.warning(f"No solution at {timestamp} — skipping.")
        continue

    # Advance thermal state to next timestep
    thermal_state = results.thermal_state

    # Stamp each result frame with the current timestep and accumulate
    for df, store in [
        (results.doe,    all_doe),
        (results.bus,    all_bus),
        (results.branch, all_branch),
        (results.viol,   all_viol),
    ]:
        tmp = df.copy()
        tmp.insert(0, 'timestamp', timestamp)
        store.append(tmp)

    # Record transformer hot-spot state
    for tx_id, ts in thermal_state.items():
        all_thermal.append({
            'timestamp':        timestamp,
            'transformer_id':   tx_id,
            'delta_theta_TO_C': round(ts['delta_theta_TO'], 3),
            'delta_theta_HS_C': round(ts['delta_theta_HS'], 3),
            'theta_HS_C':       round(ts['theta_HS'],       3),
        })

# --- Save outputs ----------------------------------------------------------------

def save_frames(frames, filename):
    if frames:
        pd.concat(frames).to_csv(outdir / filename)

save_frames(all_doe,    'doe.csv')
save_frames(all_bus,    'bus.csv')
save_frames(all_branch, 'branch.csv')
save_frames(all_viol,   'viol.csv')

if all_thermal:
    pd.DataFrame(all_thermal).to_csv(outdir / 'thermal.csv', index=False)
    logging.info("Thermal state summary:")
    print(pd.DataFrame(all_thermal).to_string(index=False))

logging.info("Done.")
