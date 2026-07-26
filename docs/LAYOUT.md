# Repository layout

The organising principle is **lifecycle, not topic**: things you create by
hand, things the machine derives and can safely delete, and results you
keep.

```
converge-soe/
├── src/converge_soe/         THE LIBRARY — all real logic
│   ├── doe_solver.py           per-timestep DOE model (+ DTR modes)
│   ├── persistent_solver.py    model built once, params updated per step (--fast)
│   ├── thermal.py              C57.91 model, DTR inversion, ageing maths
│   ├── timeseries.py           load, normalise, pre-index NMI data
│   ├── preflight.py            all validation checks
│   ├── scenarios.py            doe_dtr / doe_static / bau definitions
│   ├── pipeline.py             the nine-stage orchestration
│   ├── metrics.py              curtailment, ageing, cost formulas
│   ├── plots.py                every figure
│   ├── io.py                   streaming parquet, checkpoints, resume
│   ├── analysis.py             stage ⑧: tables, plots, sanity checks
│   └── network/                CIM conversion, LV extraction, validation
├── scripts/                  THIN CLI WRAPPERS (argument parsing only)
│   ├── organise_repo.py  build_network.py  prepare_timeseries.py
│   ├── preflight.py      run_feeder.py     analyse_results.py
├── config/                   HAND-WRITTEN configuration
│   ├── default.yaml            all defaults, fully commented
│   ├── transformers/           one file per transformer class
│   └── feeders/<FEEDER>.yaml   per-feeder overrides (optional)
├── data/                     INPUTS YOU PROVIDE — gitignored except READMEs
│   ├── xml/  meter/  mapping/  ambient/
├── build/                    DERIVED — fully deletable, always regenerable
│   ├── network/  feeders/<F>/nmi_index.csv  timeseries/  manifest.json
├── out/<FEEDER>/<RUN_ID>/    RESULTS — keep forever
│   ├── config_resolved.yaml  preflight/  scenarios/  comparison/  RUN_SUMMARY.md
├── examples/
│   ├── legacy/                 the original runners — still work, reference
│   ├── scenario_doe/  scenario_1/
├── docs/   tests/   tools/
└── archive/                  anything without an obvious home — never delete
```

Rules of thumb:

* `data/` and `build/` and `out/` are gitignored (real network/customer data
  is confidential; derived things are regenerable; results are yours). The
  folder structure itself is documented by committed `README.md` files.
* If you can delete it and regenerate it with one command, it belongs in
  `build/`. If losing it would hurt, it belongs in `out/` or `data/`.
* `scripts/` never contains logic — every behaviour is importable from
  `converge_soe.*` so it can be tested.
* `archive/` holds old generated outputs (regression baselines under
  `archive/reference_outputs/`) and superseded originals moved by
  `scripts/organise_repo.py` — never delete it.
