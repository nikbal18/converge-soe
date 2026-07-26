# The dynamic thermal rating: method

*Written to lift directly into a thesis methods chapter.*

## 1. The problem with the previous formulation

The DOE solver used to impose two transformer constraints at once: a
**static** current limit derived from the nameplate rating,

```
i_max_a = s_max / v_secondary        →   square_current_pu ≤ i_max_pu²
```

and, separately, the IEEE C57.91 hot-spot temperature as a **nonlinear
constraint** on the same current variable. The static nameplate limit
therefore still bound the solution: whenever thermal conditions would have
permitted loading above nameplate — a cold night with cold oil — the static
constraint clipped it anyway. Dynamic thermal rating delivered no measurable
benefit. Worse, the `(K²)^0.8` terms are non-smooth at zero, which produced
the recorded solver crash (`Error evaluating constraint 7 gradient: can't
evaluate pow'(0,0.8)` → `Solver (ipopt) did not exit normally`).

## 2. The fix: invert the thermal model into a time-varying current limit

At each timestep the thermal state at the start of the interval
(ΔθTO₀, ΔθHS₀) and the ambient temperature θ_A are *known constants*. The
C57.91 end-of-interval hot-spot temperature is a strictly increasing,
continuous function of the squared per-unit loading K²:

```
θ_HS(K²) = θ_A
         + α_TO·ΔθTO_R·((K²·R + 1)/(R + 1))ⁿ + (1 − α_TO)·ΔθTO₀     [top-oil rise]
         + α_W ·ΔθHS_R·(K²)ᵐ                + (1 − α_W )·ΔθHS₀      [winding rise]

  α_TO = 1 − exp(−Δt/τ_TO)          α_W = 1 − exp(−Δt/τ_W)
```

Because θ_HS(K²) is strictly increasing, it has a unique inverse: the
largest loading that just reaches the limit,

```
K²_max(t) :  θ_HS(K²_max) = θ_HS_max
```

found numerically with `scipy.optimize.brentq` (a bracketed scalar
root-find, ~20 function evaluations, microseconds). Implementation:
`converge_soe.thermal.dtr_current_limit_pu`. The transformer branch's
`i_max_pu` in the optimisation is then simply **replaced** by

```
i_max_pu(t) = sqrt(K²_max(t) / c2),      c2 = (i_base_a / I_rated)²
```

and the nonlinear hot-spot constraint is removed from the model entirely.

### Bracketing and edge cases

* The search is bracketed on `[0, K_emergency²]` with **K_emergency = 2.0**
  (200 % of rated current, configurable as `thermal.k_emergency`). This is
  the hard mechanical / short-time backstop: bushings, tap changers and LV
  cabling do not get more capable when the oil is cold.
* `θ_HS(0) ≥ θ_HS_max` — the transformer is already over the limit from
  stored heat before any new load. The limit returned is
  `sqrt(k2_floor/c2)` (a tiny positive number, `thermal.k2_floor`), the
  status `already_over_limit` is recorded in `thermal.parquet`, and the
  problem stays feasible. Visible, not fatal.
* `θ_HS(K_emergency²) < θ_HS_max` — thermal never binds this interval; the
  emergency backstop is the limit, status `not_binding`.
* Otherwise, status `thermal_binding` — the interesting case.

## 3. Why this is the right change

1. **It is what dynamic thermal rating means.** The rating becomes a
   function of the transformer's present thermal state and the weather,
   recomputed every interval, instead of a single nameplate number. On a
   cold night the limit rises well above nameplate; on a hot afternoon
   after a loaded morning it falls below it.
2. **It removes the worst nonlinearity from the optimisation.** The
   `pow(0, 0.8)` gradient failure mode is eliminated completely; the
   transformer constraint becomes a simple bound, the Jacobian is better
   conditioned, and ipopt converges in fewer iterations.
3. **It is faster.** Fewer nonlinear constraints; the root-find costs
   microseconds.
4. **It is easier to defend and to plot.** `i_max_pu(t)` is an explicit,
   inspectable time series — see the `dynamic_rating_vs_static_<SUB>`
   figure: the dynamic rating curve against the flat nameplate line, with
   ambient temperature on a twin axis.

## 4. Which thermal state drives the rating

The state carried between intervals is the transformer's **actual** state:
every customer follows their forecast, clipped into the granted envelope
`[doe_lb, doe_ub]`. This is what a real DTR controller would see (a measured
oil temperature), it makes ageing directly comparable with the BAU
trajectory, and it guarantees the delivered θ_HS trajectory respects
θ_HS_max: each interval's limit is computed from the actual starting state,
and the actual current cannot exceed the granted envelope's binding corner.
(The legacy runner advanced the state on the 'oel' envelope-scenario current
instead — see `docs/MIGRATION_NOTES.md`. The envelope-scenario trajectory is
still recorded as `theta_HS_envelope_C`.)

The verification for the inversion is `tests/test_dtr_inversion.py`: feeding
`K²_max` back through the forward recursion returns θ_HS_max to 1 × 10⁻⁹.

## 5. Honest limitation: the rating is myopic

This is a *single-interval* dynamic rating. It guarantees θ_HS ≤ θ_HS_max at
the end of each interval given the current state, but it does not look
ahead: a greedy sequence of fully-utilised intervals heats the oil in a way
that can heavily restrict later intervals (visible in the fixture runs as a
falling K_max through a sunny afternoon). The previous formulation had
exactly the same property, so this is not a regression — but a
multi-interval look-ahead (model-predictive rating over a forecast horizon)
is the natural next step and is noted as future work in
`docs/RESULTS_GUIDE.md`.

## 6. Ageing accounting (IEEE C57.91)

```
F_AA(t)  = exp(15000/383 − 15000/(θ_HS(t) + 273))     (383 K = 110 °C reference)
L        = Σ_t F_AA(t)·Δt                              equivalent ageing hours
%LOL     = 100 · L / 180000                            (normal insulation life)
```

`converge_soe.thermal.faa / equivalent_ageing_hours / percent_loss_of_life`,
verified against hand-computed values in `tests/test_metrics.py`. Note that
the DTR *deliberately* trades some ageing for export headroom: L(doe_dtr) >
L(doe_static) is expected whenever the DTR runs the transformer above
nameplate; both should normally be far below L(bau).
