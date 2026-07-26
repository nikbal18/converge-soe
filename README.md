# converge-soe — thermally-aware Dynamic Operating Envelopes

An honours-project extension of the [Project
Converge](https://arena.gov.au/projects/project-converge-act-distributed-energy-resources-demonstration-pilot/)
SOE optimisation engine. Research question: *what is the cost benefit of
thermally aware Dynamic Operating Envelopes — a transformer dynamic thermal
rating (DTR) driving the DOE — as a mechanism for managing DPV-driven
transformer ageing in ACT low-voltage networks?*

The engine computes per-customer operating envelopes on radial LV networks
with the transformer current limit set by an inverted IEEE C57.91 thermal
model, recomputed every interval from the transformer's thermal state and
the weather — instead of a single nameplate number. Three scenarios are
compared: `doe_dtr`, `doe_static`, and `bau`.

## Quick start

```bash
# 1. install (needs an ipopt executable — via conda on Windows; see
#    docs/TROUBLESHOOTING.md "ipopt not found")
pip install -e .

# 2. smoke test on the bundled toy scenario (no real data needed)
python scripts/run_feeder.py --feeder-json examples/scenario_doe/network.json \
       --feeder TOY --meter examples/scenario_doe/forecast_timeseries.csv \
       --theta-a 25 --jobs 1
cat out/TOY/*/RUN_SUMMARY.md

# 3. real data: drop CIM XMLs in data/xml/, meter CSVs in data/meter/, then
python scripts/build_network.py --list-feeders
python scripts/run_feeder.py --feeder <FEEDER> \
       --values-are kwh_per_interval --jobs 8 --fast
```

Everything about the pipeline: **docs/PIPELINE.md**. The DTR maths:
**docs/DTR_METHOD.md**. Reading results: **docs/RESULTS_GUIDE.md**. When
something breaks: **docs/TROUBLESHOOTING.md**. Folder layout:
**docs/LAYOUT.md**. Every script has its own doc in `docs/`.

## Repository shape (docs/LAYOUT.md)

`src/converge_soe/` is the library; `scripts/` are thin CLI wrappers;
`config/` is hand-written configuration; `data/` is your (gitignored,
confidential) inputs; `build/` is derived and deletable; `out/` is results;
`examples/legacy/` keeps the original runners working as the reference
implementation.

## The original converge-soe

converge-soe implements the shared operating envelope (SOE) optimisation
tested during Project Converge. The original SOE solver
(`converge_soe.SoeSolver`, `examples/legacy/run_scenario.py`,
`examples/scenario_1/`) is untouched. It requires the
[IPOPT](https://coin-or.github.io/Ipopt/) executable (with ASL) plus a
linear solver (MUMPS or MA27) on the PATH.

```python
import converge_soe as csoe
solver = csoe.SoeSolver(netw, forecast, offers, envelope_abs_max=50.0)
status, results = solver.solve()
```

## Licence

See LICENSE / NOTICE (Apache-2.0, unchanged from upstream converge-soe).
