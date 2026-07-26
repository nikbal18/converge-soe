# scripts/preflight.py

## What it does
Answers "is this network and this data actually solvable?" in seconds, with
no optimisation: 30+ checks over network structure (NET001–013), timeseries
(TS001–007), physical feasibility (PHY001–005) and model setup
(MDL001–002).

## Where it sits
Stage ④; run_feeder runs it per substation and aborts on ERROR unless
`--no-strict`. Standalone use is for poking at a single network/timeseries
pair.

## Inputs
`--network` (ejson), optional `--timeseries` (long csv/parquet),
`--parent-feeder` (enables NET013), `--transformer-params`, `--theta-a`,
`--strict` (exit nonzero on ERROR), `--out`.

## Outputs
`preflight_report.md` (grouped by severity, verdict at top: READY / READY
WITH WARNINGS (n) / BLOCKED (n errors)), `preflight_report.json` (machine),
`preflight_summary.csv` (per substation: n_loads, n_matched_nmis, peak_K,
n_over_rating, peak_theta_HS_openloop, is_radial, n_errors, n_warnings).

## Assumptions and limitations
PHY002/PHY004 use a lossless one-shot sweep — a screen, not a solve; small
excursions near limits can differ from the full model. PHY003's open-loop
trajectory assumes the configured thermal params and the data's modal Δt.

## How to tell if the output is wrong
It shouldn't pass a deliberately broken network —
`tests/test_preflight.py` feeds it missing nodes, mesh loops, shifted NMI
ids, dt mismatches, inverted voltage limits and an overloaded transformer
and asserts the right check id fires at the right severity.

## Worked example
```
python scripts/preflight.py --network examples/scenario_doe/network.json \
    --timeseries examples/scenario_doe/forecast_timeseries.csv
```
Every check id, trigger and fix: docs/TROUBLESHOOTING.md.
