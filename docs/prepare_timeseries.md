# scripts/prepare_timeseries.py

## What it does
Reads one or many raw meter exports — wide EvoEnergy format or already-long
CSVs, auto-detected — and normalises them into one tidy long table with
explicit units. Optionally fetches and caches the ambient temperature
series.

## Where it sits
Stage ② of the pipeline; run_feeder runs it automatically (cached).

## Inputs
| Input | Example | If wrong |
|---|---|---|
| `--meter` | `data/meter/lexcen_data.csv` | default: every CSV in data/meter |
| `--values-are` | `kwh_per_interval` | REQUIRED for wide exports — no silent unit guessing; wrong → TS006 warnings, ×2000 errors |
| `--fetch-ambient` | | needs internet; caches to data/ambient/*.csv |

Wide-export handling (the maintained version of the legacy translator):
active = ΣE* − ΣB* (captures PV reverse flow), `load_id = nmi_<NMI>`,
reactive from Q* channels when present else 0.4 × active, interval-ENDING
timestamps with 24:00 rolling to next-day 00:00.

## Outputs
`build/timeseries/all_nmis.parquet` — columns `timestamp, load_id,
real_power_w (W, load-positive), reactive_power_var (VAr)`; deduplicated,
sorted. Plus a printed summary. Cached against inputs + the unit choice.

## Assumptions and limitations
Reactive estimated as 0.4 × active when no Q channels exist; timestamps
parsed with an explicitly detected format (TS002).

## How to tell if the output is wrong
The printed min active power should be **negative** (PV export). ~2× or
~2000× off → the unit flag. NMI count much lower than expected → the file
wasn't tab-separated or the NmiSuffix channels weren't recognised.

## Worked example
```
python scripts/prepare_timeseries.py \
    --meter examples/scenario_doe/forecast_timeseries.csv
```
