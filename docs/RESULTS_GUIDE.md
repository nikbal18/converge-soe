# Reading the results

Everything lands in `out/<FEEDER>/<RUN_ID>/`. Start with `RUN_SUMMARY.md`,
then `comparison/sanity_checks.md`, then the figures. This guide gives, for
every metric and plot: the formula, what a good result looks like, and what
a suspicious one looks like.

## Sign conventions

`P_des(i,t) = −real_power_w(i,t)/1000` kW — **injection-positive**, so PV
export is positive. It is what the customer would do unconstrained, and is
stored per row as `p_des_kw` in `doe.parquet`. Envelopes are injections too:
`doe_lb_kw ≤ 0 ≤ doe_ub_kw` always (an envelope must contain zero).

## Metrics (comparison/metrics_by_substation.csv)

| Column | Formula | Good | Suspicious |
|---|---|---|---|
| `E_curt_kwh` | Σᵢ Σₜ max(0, P_des − doe_ub)·Δt | 0 for bau (by construction); dtr ≤ static | dtr > static → a static limit still binds somewhere (misconfigured DTR) |
| `E_imp_kwh` | Σᵢ Σₜ max(0, −P_des − \|doe_lb\|)·Δt | usually small | large values → import-side congestion, check PHY001 |
| `E_des_export_kwh` | Σᵢ Σₜ max(0, P_des)·Δt | > 0 | 0 → no PV in the data; every curtailment number is trivial |
| `curtailment_cost` | E_curt × price | — | remember the price is an assumption (printed on every figure) |
| `ageing_hours` | Σₜ F_AA(θ_HS)·Δt | L(bau) ≥ L(static); L(dtr) between static and bau typically | L(static) > L(bau) → check which θ trajectory you're reading |
| `percent_LOL` | 100·L/180000 | tiny for short runs | — |
| `equivalent_days_of_life_lost_per_year` | L scaled to a year, in days | ≈365·F_AA_mean/… — 365 means ageing exactly at reference rate | ≫365 sustained → the transformer is being cooked |
| `peak_theta_HS_C` | max θ_HS (actual-behaviour trajectory) | ≤ θ_HS_max for doe_dtr | > θ_HS_max+0.5 in doe_dtr (excl. already_over_limit) → bug |
| `mean_headroom`, `share_above_nameplate` | headroom(t)=√K²_max = i_max_dtr/i_max_static | headroom > 1 most of the time in mild weather | always ≤ 1 → thermal params or ambient are off |

`E_enabled = E_curt(doe_static) − E_curt(doe_dtr)` is **the headline
number** — reported absolutely, as % of desired export
(`energy_enabled_by_dtr` figure), and in dollars in `RUN_SUMMARY.md`.

Note on the two θ_HS columns in `thermal.parquet`: `theta_HS_C`
(= `theta_HS_posthoc_C`) is the actual-behaviour trajectory used for ageing
and for carrying the DTR state; `theta_HS_envelope_C` is the trajectory the
solver's 'oel' envelope corner would produce (worst-case-ish, legacy
convention). BAU has a single trajectory.

## Plots (comparison/*.png, with .pdf and .csv siblings)

1. **curtailed_energy_by_scenario** — grouped bars. Expect: bau = 0,
   dtr ≤ static everywhere.
2. **curtailment_cost_by_scenario** — same in dollars, assumed price in the
   subtitle.
3. **energy_enabled_by_dtr** — horizontal bars, % of desired export
   annotated. This is the thesis headline figure.
4. **transformer_ageing_by_scenario** — log y-axis; bau can be orders of
   magnitude worse.
5. **avoided_ageing** — days of transformer life saved vs BAU.
6. **dynamic_rating_vs_static_<SUB>** — *the DTR money shot*: K_max(t) over
   a representative week against the flat nameplate line, ambient on a twin
   axis. Expect the rating to move opposite to ambient and to sag through
   loaded afternoons. A flat line at 1.0 means the DTR never engaged.
7. **hotspot_timeseries_<SUB>** — θ_HS for all scenarios + ambient +
   θ_HS_max. Expect doe_dtr to ride *at* the limit during curtailment
   periods (that is the rating doing its job), bau to break through it.
8. **hotspot_duration_curve** — sorted θ_HS. The area between bau and the
   DOE curves is the thermal protection delivered.
9. **transformer_loading_duration_curve** — sorted K, with K=1 marked.
   doe_dtr exceeding 1 for a small share of time IS the point.
10. **envelope_vs_desired_<SUB>** — one representative day: desired
    injection, upper envelopes, curtailment shaded.
11. **seasonal_curtailment_heatmap** — month × hour mean curtailment, one
    panel per DOE scenario, shared scale. Shows *when* the constraint binds
    (expect summer middays); this is what makes a full year of data worth
    having.
12. **violations_summary** — stacked bars by violation type and scenario.
    bau violations are recorded-not-prevented; DOE scenarios should show
    (near-)zero unless preflight PHY001 warned that background load alone
    exceeds ratings.

## The sanity block (comparison/sanity_checks.md)

Every run prints PASS/FAIL for: E_curt(bau)=0; E_curt(dtr) ≤ E_curt(static)
per substation; peak θ_HS(dtr) ≤ θ_HS_max+tol except `already_over_limit`;
envelopes contain zero; energy balance (bau) within 1 %; total desired
export > 0; L(bau) ≥ L(static). L(dtr) vs L(static) is *flagged, not
asserted* — the DTR trades ageing for headroom by design.

## Assumptions to keep in mind

Radial single-phase-equivalent network; myopic single-interval DTR (see
docs/DTR_METHOD.md §5); flat curtailment price unless `price_series_csv` is
set; constant Δt (preflight PHY005 errors on a mismatch); the
`K_emergency = 2.0` mechanical backstop; voltage/current soft-limit
penalties make violations quantified rather than fatal; the envelope
allocation uses a small "usable export" reward so headroom goes to customers
who can use it (`solver.usable_export_weight`, see docs/MIGRATION_NOTES.md).
