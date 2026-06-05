#!/usr/bin/env python3
"""
disaggregate_transformer.py

Disaggregates aggregate transformer load data into synthetic NMI profiles
whose sum matches the transformer total at every timestep.

Solar customers (SOLAR_PENETRATION fraction of all customers) have a rooftop
PV system and can export to the grid, giving them negative net load at times.
Non-solar customers are pure consumers and are always non-negative.

Outputs two files ready for run_doe_multistep.py:
  - forecast_timeseries.csv   (per-NMI loads at each timestep)
  - network.json              (network topology with synthetic NMIs as Load components)

Edit the CONFIG section below to change settings.
"""

import json
import copy
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
# CONFIG — edit these values
# ==============================================================================

# Number of synthetic customers to generate.
N_CUSTOMERS = 20

# Fraction of customers with rooftop solar (can export to grid).
# 0.30 = 30% solar penetration. Change freely between 0.0 and 1.0.
SOLAR_PENETRATION = 0.30

# Input transformer timeseries CSV.
INPUT_CSV = r'C:\Users\nikki\OneDrive - Australian National University\Honours project\Individual Project\TX-Active-Power-Timeseries-june-march.csv'

# Base network.json to copy topology from (transformer, lines, buses).
# Load components in this file will be replaced with the synthetic NMIs.
NETWORK_JSON_IN = r'C:\Users\nikki\OneDrive - Australian National University\Honours project\Individual Project\converge-soe\examples\scenario_doe\network.json'

# Output folder — forecast_timeseries.csv and network.json will be written here.
OUTPUT_DIR = r'C:\Users\nikki\OneDrive - Australian National University\Honours project\Individual Project\converge-soe\examples\scenario_real'

# Which LV buses to place customers on, and what fraction goes to each.
# Keys must match Node IDs in network.json. Fractions must sum to 1.
BUS_SPLIT = {
    'bus_a': 0.5,
    'bus_b': 0.5,
}

# Assumed lagging power factor for reactive power (NEM12 records active only).
# Q = P * tan(arccos(POWER_FACTOR))
POWER_FACTOR = 0.95

# Use the measured METER_SUMMATION column (True) or the ESTIMATION column (False).
USE_MEASURED = True

# --- Diversity settings -------------------------------------------------------

# Spread of customer sizes. Higher = bigger difference between large and small
# customers. Drawn from a log-normal distribution with this σ.
SIZE_DIVERSITY = 0.6

# Temporal diversity: controls how much individual customers deviate from the
# aggregate shape before normalisation. Higher = customers peak at more
# different times. Expressed as a fraction of mean load.
TEMPORAL_DIVERSITY = 0.15

# Random seed — change to get a different disaggregation while keeping N fixed.
RANDOM_SEED = 42

# ==============================================================================
# END OF CONFIG
# ==============================================================================

np.random.seed(RANDOM_SEED)

N_SOLAR   = round(SOLAR_PENETRATION * N_CUSTOMERS)
N_NO_SOLAR = N_CUSTOMERS - N_SOLAR

# --- Load transformer data ----------------------------------------------------

df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
df.columns = df.columns.str.strip()

# Parse dates robustly (handles both "13/04/2026 00:00:00" and "1/06/2025 0:00")
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

load_col = 'METER_SUMMATION (kW)' if USE_MEASURED else 'ESTIMATION (kW)'

# Drop rows with missing load data (data gaps in the timeseries)
n_before = len(df)
df = df.dropna(subset=[load_col]).reset_index(drop=True)
n_dropped = n_before - len(df)
if n_dropped:
    print(f"  Note: dropped {n_dropped} rows with missing data.")

total_kw   = df[load_col].values
timestamps = df['date'].dt.strftime('%Y-%m-%d %H:%M')
T = len(total_kw)

print(f"Loaded {T} timesteps from {Path(INPUT_CSV).name}")
print(f"  Range : {timestamps.iloc[0]}  →  {timestamps.iloc[-1]}")
print(f"  Load  : {total_kw.min():.1f} – {total_kw.max():.1f} kW  (mean {total_kw.mean():.1f} kW)")
print(f"  Net export timesteps: {(total_kw < 0).sum()} ({100*(total_kw < 0).mean():.1f}% of the time)")

# --- Decompose aggregate into base consumption and solar generation ------------
#
# The aggregate (total_kw) = base_consumption - solar_generation at each step.
# When total_kw > 0: net consumption dominates, solar_generation = 0 in the
#   aggregate (or is outweighed by consumption).
# When total_kw < 0: solar generation exceeds consumption on the feeder.
#
# We model this as:
#   base_t(t)  = max(total_kw(t), 0)   — the visible consumption component
#   solar_t(t) = max(-total_kw(t), 0)  — the net solar export component
#
# By construction: base_t - solar_t = total_kw at every timestep.

base_t  = np.maximum(total_kw,  0.0)   # (T,) always >= 0
solar_t = np.maximum(-total_kw, 0.0)   # (T,) always >= 0

# --- Helper: generate diverse proportional profiles ---------------------------

window = max(1, round(60 / 15))   # smooth over ~1 hour of 15-min intervals

def make_profiles(n, total_t, size_diversity, temporal_diversity):
    """Generate n profiles that sum to total_t at every timestep."""
    if n == 0:
        return np.zeros((0, T))
    weights = np.random.lognormal(0.0, size_diversity, n)
    raw = np.zeros((n, T))
    for c in range(n):
        noise    = np.random.normal(0.0, temporal_diversity, T)
        smoothed = pd.Series(noise).rolling(window=window, center=True, min_periods=1).mean().values
        raw[c]   = np.maximum(weights[c] * (1.0 + smoothed), 1e-6)
    # Normalise column-wise; where total is 0, profiles are also 0
    col_sums = raw.sum(axis=0)
    profiles = np.where(total_t > 0,
                        raw / col_sums * total_t[np.newaxis, :],
                        0.0)
    return profiles

# --- Build individual profiles ------------------------------------------------
#
# ALL customers get a share of base_t  (their base consumption, always >= 0).
# SOLAR customers additionally get a share of solar_t subtracted from their load
# (their PV generation, which can make their net load negative).

base_profiles  = make_profiles(N_CUSTOMERS, base_t,  SIZE_DIVERSITY, TEMPORAL_DIVERSITY)
solar_profiles = make_profiles(N_SOLAR,     solar_t,  SIZE_DIVERSITY, TEMPORAL_DIVERSITY)

# Net load: solar customers = base - solar; non-solar customers = base only
net_profiles = base_profiles.copy()
for c in range(N_SOLAR):
    net_profiles[c] -= solar_profiles[c]

# Verify sum: base_t - solar_t = total_kw ✓
max_err = np.abs(net_profiles.sum(axis=0) - total_kw).max()
assert max_err < 1e-6, f"Sum mismatch: {max_err:.2e} kW"

# --- Assign customers to buses ------------------------------------------------

assert abs(sum(BUS_SPLIT.values()) - 1.0) < 1e-9, "BUS_SPLIT fractions must sum to 1."

bus_ids = []
for bus_id, frac in BUS_SPLIT.items():
    bus_ids.extend([bus_id] * round(frac * N_CUSTOMERS))
while len(bus_ids) < N_CUSTOMERS:
    bus_ids.append(list(BUS_SPLIT.keys())[-1])
bus_ids = bus_ids[:N_CUSTOMERS]

load_ids = [f'NMI_{c+1:03d}' for c in range(N_CUSTOMERS)]
tan_phi  = np.tan(np.arccos(POWER_FACTOR))

# --- Build forecast_timeseries.csv -------------------------------------------

records = []
for c in range(N_CUSTOMERS):
    for t in range(T):
        p_w   = float(net_profiles[c, t]) * 1000.0   # kW → W
        q_var = p_w * tan_phi
        records.append({
            'timestamp':          timestamps.iloc[t],
            'load_id':            load_ids[c],
            'real_power_w':       round(p_w,   2),
            'reactive_power_var': round(q_var, 2),
        })

forecast = pd.DataFrame(records)

# --- Build network.json -------------------------------------------------------

with open(NETWORK_JSON_IN) as f:
    netw = json.load(f)

netw_out = copy.deepcopy(netw)

# Remove existing Load components
netw_out['components'] = {
    k: v for k, v in netw_out['components'].items()
    if 'Load' not in v
}

# Add one Load component per synthetic NMI
for c in range(N_CUSTOMERS):
    netw_out['components'][load_ids[c]] = {
        'Load': {'cons': [{'node': bus_ids[c]}]}
    }

# --- Save outputs -------------------------------------------------------------

out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

forecast_path = out_dir / 'forecast_timeseries.csv'
network_path  = out_dir / 'network.json'

forecast.to_csv(forecast_path, index=False)

with open(network_path, 'w') as f:
    json.dump(netw_out, f, indent=2)

print(f"\nWrote {forecast_path}")
print(f"Wrote {network_path}")

# --- Summary ------------------------------------------------------------------

print(f"\nCustomer breakdown ({N_CUSTOMERS} total):")
print(f"  Solar (can export) : {N_SOLAR}  ({100*SOLAR_PENETRATION:.0f}%)")
print(f"  Non-solar          : {N_NO_SOLAR}")
for bus_id in BUS_SPLIT:
    print(f"  {bus_id}: {bus_ids.count(bus_id)} customers")

solar_export_frac = [
    (net_profiles[c] < 0).mean() for c in range(N_SOLAR)
]
if N_SOLAR:
    print(f"\nSolar customers export on average "
          f"{100*np.mean(solar_export_frac):.1f}% of timesteps "
          f"(range {100*min(solar_export_frac):.1f}–{100*max(solar_export_frac):.1f}%)")

mean_kw = net_profiles.mean(axis=1)
print(f"Mean net load range : {mean_kw.min():.2f} – {mean_kw.max():.2f} kW")
print(f"Sum check max error : {max_err:.2e} kW")
