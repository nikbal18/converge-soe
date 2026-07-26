# Examples

- `scenario_1/` — minimal 4-bus radial LV network for the original SOE solver
  (`legacy/run_scenario.py`). See its README for a guided tour.
- `scenario_doe/` — the same network set up for the DOE solver, with a
  half-day forecast timeseries and IEEE C57.91 transformer parameters. This is
  the tiny worked example used throughout `docs/` and by the test suite.
- `legacy/` — the original runner scripts, kept working as the reference
  implementation for A/B comparison against the new pipeline
  (`scripts/run_feeder.py`). See `legacy/RUN_DOE_GUIDE.md` for the manual
  route; `docs/PIPELINE.md` supersedes it.

Quick start (from the repo root, ipopt available):

    python examples/legacy/run_doe_scenario.py examples/scenario_doe /tmp/doe_out
