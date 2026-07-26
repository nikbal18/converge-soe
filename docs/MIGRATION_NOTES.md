# Migration notes: legacy runners → the pipeline

The legacy runners in `examples/legacy/` are unchanged and remain the
reference implementation. Where the new pipeline's numbers differ from
theirs, the differences are deliberate and listed here.

## Differences that change numbers

1. **The DTR scenario differs by design — that is the result.**
   Legacy behaviour (`tx_limit="legacy"`) imposes the static nameplate limit
   *and* the nonlinear thermal constraint; `doe_dtr` replaces the
   transformer limit with the inverted C57.91 rating. On the bundled
   fixture (3 summer days, 6 PV customers, 100 kVA transformer) this halves
   curtailed energy (201 → 101 kWh) while θ_HS rides exactly at the 120 °C
   limit. How much it differs on the real feeders, and when, is the thesis
   question — see `comparison/` of any run.

2. **Which thermal state is carried between intervals.** Legacy advanced the
   state on the 'oel' (max-consumption) envelope-corner current. The
   pipeline carries the transformer's *actual* state — customers following
   their forecast clipped into the granted envelope — which is what a real
   DTR controller measures, makes ageing comparable with BAU, and
   guarantees the delivered θ_HS respects the limit. The old-style
   trajectory is still recorded (`theta_HS_envelope_C`).

3. **Envelope allocation degeneracy is resolved.** With the legacy
   objective, the split of surplus network capacity between NMIs is
   mathematically degenerate — any split is optimal, so per-NMI curtailment
   was arbitrary (observed: all headroom handed to customers who couldn't
   use it, making the DTR look useless). The pipeline adds a small "usable
   export" reward (`solver.usable_export_weight`, default 0.01) so headroom
   goes to customers who can actually use it. Set it to 0 to reproduce
   legacy behaviour.

4. **Soft voltage/current limits (default ON).** Legacy returned
   "infeasible" and silently skipped the timestep; the pipeline solves with
   penalised slacks and reports the violation magnitude in viol.parquet.
   Feasible timesteps are unaffected.

5. **Variable bounds and warm starts (Phase 1.4).** `square_voltage_pu ∈
   [0.25, 2.25]`, `square_current_pu ≥ 1e-8`, initialised from the previous
   timestep. For feasible solves these change results only at solver
   tolerance level (verified: legacy-mode regression matches the baseline
   run to 3 decimals), but they remove the pow'(0,0.8) crash entirely.

## Equivalence that is enforced

* `tx_limit="legacy"` in the new solver reproduces the old combined
  behaviour — the multistep regression in `tests/test_legacy_regression.py`
  matches the pre-change baseline trajectory (θ_HS and envelope sums).
* The fast path (`--fast`, persistent solver) is validated against the
  plain path by `tests/test_fast_path_equivalence.py`: at tightened solver
  tolerance the two are bit-identical after rounding; at production
  tolerance they can differ by ~1e-2 kW on near-degenerate splits (interior
  point path sensitivity), which is far below meter accuracy.

## Practical changes

Parquet instead of CSV accumulation (use `--csv` for mirrors); results
stream to disk continuously with resume; ipopt output silenced by default
(`-vv` for the old firehose; failures always in failures.csv).
