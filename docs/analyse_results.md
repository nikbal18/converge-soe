# scripts/analyse_results.py

## What it does
Rebuilds every metric table, comparison figure and the PASS/FAIL sanity
block for an existing run from its streamed parquet outputs. Never
re-solves — figures can be regenerated in seconds, and each figure's data is
saved as a sibling CSV precisely so this stays true.

## Where it sits
Stage ⑧; run_feeder runs it at the end of every run.

## Inputs
`run_dir` (out/<FEEDER>/<RUN_ID>); optional `--plot-substations "A,B"` for
extra per-substation detail figures. Uses the run's own
`config_resolved.yaml` (price, thermal class) so re-analysis matches the
run.

## Outputs
`comparison/metrics_by_substation.csv`, `metrics_feeder.csv`,
`scenario_comparison.csv` (wide, with deltas — paste-ready),
`per_nmi_<sc>_<sub>.csv`, the 12 figures (docs/RESULTS_GUIDE.md) as
PNG+PDF+CSV, `sanity_checks.md`, and `RUN_SUMMARY.md` at the run root.

## Assumptions and limitations
The flat price is printed on every cost figure; ageing uses the
actual-behaviour θ trajectory (see RESULTS_GUIDE).

## How to tell if the output is wrong
sanity_checks.md is the answer to exactly that question. If a figure is
empty, the corresponding parquet is missing — check `_manifest.json` for a
failed substation.

## Worked example
```
python scripts/analyse_results.py out/TOY/$(ls out/TOY | head -1)
```
