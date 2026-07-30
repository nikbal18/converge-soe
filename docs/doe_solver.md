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

## Non-participant loads in the power balance

Non-participant loads DO enter the nodal balance, and always did. In
`_build_opt_model` the injection lists are assembled as:

```python
if load_id in self.partic_load_ids:
    a_bus_pu[bus, oe].append(-p_inj_oe_kw[load_id, oe] * _kw_to_pu)   # variable
elif load_id in self.forecast_load_ids:
    a_bus_pu[bus, oe].append(active_power)                            # constant
```

So a load with a forecast but no envelope is a fixed injection at its bus.
That is the mechanism synthetic NMIs (stage ⑤b) rely on: they load the network
via the `elif`, and never receive an envelope because `scenarios.py` passes
`participant_load_ids` = the non-synthetic ids.

**`_calculate_bus_loads_kw` is NOT that mechanism.** Despite the name, its
return values are used only to initialise the soft-slack variables
(`init_sof_a` / `init_sof_r`) — they appear in no constraint. It now also
accumulates active power for non-participants, which gives the slacks a better
starting point, but this is a warm-start improvement, not a physics change.
Do not read it as the background-load path.

Evidence that the `elif` path works: with 72 synthetic NMIs the transformer
branch current at the envelope corners is non-zero (3.55 A on S_5409_AT),
whereas with `--no-synthetic` — where every load is a participant and there
are no fixed injections at all — it is exactly 0.

### Warm-start initialisers are clamped

A converged `square_current_pu` is routinely ~1e-32: numerically zero, but
below its own 1e-8 lower bound. Feeding it back verbatim made Pyomo emit a
W1002 per variable per timestep (thousands per run). `init_from` now clamps
into bounds.

Other changes: `ApplicationError` and friends are caught in
`_solve_opt_model` (status `error`, reason in `.last_solve_error` — one bad
interval never kills a run); the O(B²) downstream-branch DataFrame scan is a
precomputed dict; `dtr_info` per transformer per timestep carries
`i_max_pu, K2_max, status` into thermal.parquet.
