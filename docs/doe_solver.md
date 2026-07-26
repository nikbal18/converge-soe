# src/converge_soe/doe_solver.py (module notes)

The per-timestep DOE optimisation model. Substantially extended in the DTR
rebuild — every extension is opt-in and the default construction reproduces
the legacy numbers.

New constructor parameters:

| Param | Default | Meaning |
|---|---|---|
| `tx_limit` | `"legacy"` | `legacy` = static nameplate limit + nonlinear thermal constraint (old behaviour); `dtr` = i_max replaced by the inverted C57.91 rating, no thermal constraint; `static` = nameplate only |
| `k_emergency`, `k2_floor` | 2.0, 1e-6 | DTR backstop and already-over-limit floor (docs/DTR_METHOD.md) |
| `soft_limits`, `soft_limit_penalty` | False, 1000 | penalised slacks on voltage/current limits (viol_v_pu, viol_i_pu) |
| `quiet` | False | print_level 0, sb=yes, no stdout redirect |
| `warm_start_in` | None | var values from `warm_start_values()` of the previous step |
| `network_cache` | None | parsed network data from `network_cache()` of a previous instance (Path A: parse once per substation) |
| `usable_export_weight` | 0.0 | small reward on min(doe_ub, desired export) — resolves the envelope-allocation degeneracy (docs/MIGRATION_NOTES.md §3) |
| `solver_name` | `"ipopt"` | any AMPL-interface solver Pyomo can find |

Other changes: `ApplicationError` and friends are caught in
`_solve_opt_model` (status `error`, reason in `.last_solve_error` — one bad
interval never kills a run); the O(B²) downstream-branch DataFrame scan is a
precomputed dict; `dtr_info` per transformer per timestep carries
`i_max_pu, K2_max, status` into thermal.parquet.
