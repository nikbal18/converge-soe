# scripts/run_feeder.py

## What it does
Runs the entire DOE pipeline for one feeder with one command: builds the
network from XMLs, prepares the meter timeseries, maps NMIs to substations,
preflights, pre-indexes, solves the three scenarios in parallel with
continuous checkpointing, and produces the analysis. It is the only command
most runs need.

## Where it sits
It is the front door to stages ①–⑨ (docs/PIPELINE.md); each stage is also
available standalone (`build_network.py`, `prepare_timeseries.py`,
`preflight.py`, `analyse_results.py`).

## Inputs
| Input | Example | If wrong / missing |
|---|---|---|
| `--feeder` | `GOLDCR_8HB_LEXCEN` | must match a feeder built from data/xml; `--list-feeders` shows them |
| `--feeder-json` | `build/network/feeders/F_network.json` | bypass stage ① with a prebuilt ejson; errors if unreadable |
| `--meter` | `data/meter/lexcen_data.csv` | default = every CSV in data/meter; none → hard error |
| `--values-are` | `kwh_per_interval` | REQUIRED for wide exports; wrong choice → TS006 magnitude warnings, results off by ×2000 |
| `--scenarios` | `doe_dtr,doe_static,bau` | unknown name → error |
| `--only` | `"S 5402,S 5406"` | substring filter on substation id/name; no match → nothing runs |
| `--jobs` | `8` | default cpu−1; substations are independent |
| `--fast` | | persistent mutable-param model (Path B), validated by the equivalence test |
| `--theta-a` | `25` | constant ambient; else data/ambient cache or open-meteo per config |
| `--resume` / `--restart` | | resume is default; stale checkpoints refuse and explain (fingerprint mismatch) |
| `--dry-run` | | prints substations × scenarios × timesteps + a runtime estimate from a 20-step pilot |
| config | `config/default.yaml` | CLI > feeder yaml > default; resolved copy always written to the run dir |

## Outputs
`out/<FEEDER>/<RUN_ID>/` — see docs/RESULTS_GUIDE.md for every column:
`config_resolved.yaml`, `preflight/preflight_report.{md,json}` +
`preflight_summary.csv`, `scenarios/<sc>/<SUB>/{doe,bus,branch,viol,thermal}
.parquet` + `run.log` + `failures.csv` + `_checkpoint.json`,
`_manifest.json`, `comparison/*` and `RUN_SUMMARY.md`.

Column dictionary highlights (doe.parquet): `timestamp`; `load_id` (network
Load id, `nmi_*`); `doe_lb_kw`/`doe_ub_kw` (envelope, kW, injection-positive,
±∞ for bau); `p_des_kw` (desired injection = −forecast/1000);
`has_data` (False = the reading was gap-filled). thermal.parquet:
`theta_HS_C` (actual-behaviour trajectory), `theta_HS_envelope_C`
(oel-corner trajectory), `i_max_pu`/`K2_max`/`dtr_status` (the DTR),
`K2_actual`.

## Assumptions and limitations
Radial single-phase-equivalent networks; myopic single-interval DTR;
constant Δt; flat curtailment price unless a series is given; K_emergency
backstop 2.0; soft limits on by default (violations quantified, not fatal);
variable bounds V∈[0.5,1.5] pu, I² ≥ 1e-8.

## How to tell if the output is wrong
`comparison/sanity_checks.md` must be all PASS. Eyeball: E_curt(bau)=0;
E_curt(dtr) ≤ E_curt(static); peak θ_HS(dtr) within 0.5 °C of the limit
during binding periods and never above; `n_failed` ≈ 0 in `_manifest.json`;
LV voltages within ±10 %; K duration curve reaching just above 1 only for
doe_dtr.

## Worked example
```
python scripts/run_feeder.py --feeder-json examples/scenario_doe/network.json \
       --feeder TOY --meter examples/scenario_doe/forecast_timeseries.csv \
       --theta-a 25 --jobs 1
cat out/TOY/*/RUN_SUMMARY.md
```
