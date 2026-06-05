# converge-soe: SOE optimisation engine as used in Project Converge
# Edits as per honours topic: implementing thermal constrained modelling to SOEs. 
This repo is using the converge-soe framework as a base to build a model that can use thermal modelling of transformers to further optimise DOE allocation and allow for increased PV penetration.

It seeks to answer the question: "What is the regulatory cost benefit of thermally aware Dynamic Operating Envelopes as a mechanism for managing DPV-driven transformer ageing in ACT low-voltage networks?" 

The converge-soe model will be used to create a simpler DOE model,and then test cases constructed to determine potential benefits of DOEs with thermal constraints compared to DOEs without. 
# SOE Solver → DOE Solver: Change Summary

`converge-soe/src/converge_soe/soe_solver.py` → `doe_solver.py`

## Overview

The DOE solver is a refactored version of the SOE solver that shifts from a market-dispatch model (with customer offers and network support) to a pure dynamic operating envelope (DOE) model. The core power flow physics and IPOPT solver are unchanged; the differences are in what the optimiser is allowed to control, what constraints it must satisfy, and what it returns.

The SOE solver finds the widest envelopes that remain feasible given customer offer reservations, dispatching customers via the market to shift those envelopes when needed. The DOE solver treats all forecast loads as fixed, finds the widest feasible per-customer export/import bounds, and additionally constrains transformer temperature using the IEEE C57.91 thermal model.

---

## Change 1 — Constructor parameters

The `df_offers` parameter is removed entirely, since the DOE solver has no concept of customer offers or network support payments. Three new parameters are added to support the transformer thermal model.

| Location | SOE Solver | DOE Solver |
|---|---|---|
| `__init__` signature | `netw_ejson, df_forecasts, df_offers, envelope_abs_max=50.0, solver_options={}` | `netw_ejson, df_forecasts, envelope_abs_max=50.0, transformer_params=None, theta_A=20.0, thermal_state_in={}, solver_options={}` |
| New parameter: `transformer_params` | (not present) | Dict of IEEE C57.91 thermal model parameters: `tau_TO`, `tau_W`, `delta_theta_TO_R`, `delta_theta_HS_R`, `R`, `n`, `m`, `I_rated`, `theta_HS_max`, `dt`. If `None`, thermal constraints are skipped. |
| New parameter: `theta_A` | (not present) | Ambient temperature (°C) for the current timestep. Default `20.0`. |
| New parameter: `thermal_state_in` | (not present) | Dict keyed by transformer branch ID containing `delta_theta_TO` and `delta_theta_HS` from the previous period (for warm-start / multi-step use). Defaults to cold start (0 °C rises). |

---

## Change 2 — `_filter_input_data`

The offer filtering logic is removed. The SOE version intersected network load IDs with the offers dataframe to produce `offer_load_ids` and `df_offers_filt`. These are not present in the DOE version because there are no offers.

| Location | SOE Solver | DOE Solver |
|---|---|---|
| `_filter_input_data` | Filters forecasts AND offers. Produces: `forecast_load_ids`, `df_forecasts_filt`, `offer_load_ids`, `df_offers_filt` | Filters forecasts only. Produces: `forecast_load_ids`, `df_forecasts_filt` (no `offer_load_ids` or `df_offers_filt`) |

---

## Change 3 — Participant index set in `_build_opt_model`

In the SOE solver, only customers with active offers are treated as 'participants' whose envelope bounds are optimised. In the DOE solver, every customer with a forecast is a participant.

| Location | SOE Solver | DOE Solver |
|---|---|---|
| `partic_idxs` | `self.offer_load_ids` (customers with offers only) | `self.forecast_load_ids` (all forecast customers) |

---

## Change 4 — Bus active-power allocation in `_build_opt_model`

In the SOE solver, participants' active power in the bus balance equations is represented by the `p_inj_oe_kw` optimisation variable (so the solver sets their envelope bounds directly). Non-participants contribute their fixed forecast value. In the DOE solver, all forecast loads contribute a fixed forecast active power value; the `p_inj_oe_kw` variables still exist but are not linked into the power flow — they are purely output bounds found by the width-maximising objective.

| Location | SOE Solver | DOE Solver |
|---|---|---|
| Active power in `a_bus_pu` per load | If participant: `-p_inj_oe_kw` (variable). Elif forecast: fixed forecast value. | If forecast: fixed forecast active AND reactive power (both appended together). |
| Reactive power in `r_bus_pu` per load | If forecast: fixed reactive forecast | Same (unchanged) |

---

## Change 5 — Constraints in `_build_opt_model`

The SOE-specific offer and reservation constraints are removed. New transformer thermal constraints based on IEEE C57.91 are added. The core power flow constraints (voltage, branch active/reactive power, current limits) are unchanged.

| Location | SOE Solver | DOE Solver |
|---|---|---|
| SOE offer-cap constraints | `network_support_kw[con] <= offer_con_kw`, `network_support_kw[inj] <= offer_inj_kw` | (removed entirely) |
| SOE reservation constraints | `p_inj_oe_kw[oel] <= reservation_l + NS_inj - NS_con`, `p_inj_oe_kw[oer] >= reservation_u + NS_inj - NS_con` | (removed entirely) |
| Transformer thermal constraints (new) | (not present) | If `transformer_params` is provided, adds per-transformer per-OE constraint: `theta_A + delta_TO + delta_HS <= theta_HS_max` using IEEE C57.91 exact step-response discretisation of the two-layer ODE. Skipped if `transformer_params` is `None`. |

---

## Change 6 — Objective function

The SOE objective has three terms. The DOE objective drops the dispatch-cost term entirely (no offers, no payments) and retains only the envelope-width nudge and violation penalty.

| Location | SOE Solver | DOE Solver |
|---|---|---|
| Term 1 (dispatch cost) | `first_term`: sum of `network_support_kw * price * (5/60)` for each participant ($/kWh, 5-minute interval) | (removed) |
| Term 2 (envelope width) | `second_term`: `small_weight * sum(p_inj_oe_kw[oel] - p_inj_oe_kw[oer])` over `offer_load_ids` (pushes envelopes wider) | `width_term`: same formula but over `forecast_load_ids` (all customers, not just offerors) |
| Term 3 (violations) | `third_term`: `big_weight * sum` of all soft slack variables `sof_bus_a_kw` and `sof_bus_r_kw` | `viol_term`: identical formula (unchanged) |
| Objective | `minimize(first_term + second_term + third_term)` | `minimize(width_term + viol_term)` |

---

## Change 7 — `_extract_results`

The results dataframe for operating envelopes is simplified (no dispatch or payment columns). A new `thermal_state_out` dict is computed and returned so the caller can pass it as `thermal_state_in` to the next timestep for multi-step simulations.

| Location | SOE Solver | DOE Solver |
|---|---|---|
| Envelope results dataframe columns | `soe_lb_kw`, `soe_ub_kw`, `dispatch_kw`, `payment_dlr`. Indexed by `offer_load_ids`. | `doe_lb_kw`, `doe_ub_kw` (no dispatch or payment). Indexed by `forecast_load_ids`. |
| Thermal state output (new) | (not present) | `thermal_state_out` dict keyed by transformer branch ID. Each entry contains: `delta_theta_TO` (°C), `delta_theta_HS` (°C), `theta_HS` (absolute, °C). Only populated if `transformer_params` is set. |
| Return namedtuple | `Results(bus, branch, viol, soe)` | `Results(bus, branch, viol, soe, thermal_state)` |

---

## Change 8 — `_calculate_bus_loads_kw`

In the SOE solver, this method pre-computes both reactive and active background power at each bus (excluding participant active power, which is handled by the envelope variable). In the DOE solver, active power is appended to `a_bus_pu` directly inside `_build_opt_model` rather than pre-computed here, so `_calculate_bus_loads_kw` only returns the reactive power dict.

| Location | SOE Solver | DOE Solver |
|---|---|---|
| Active power accumulation | Adds `real_power_w * 1e-3` to `bus_ld_a_kw` for all forecast loads that are NOT in `offer_load_ids` (non-participant fixed load) | (removed from this method) Active power is appended directly into `a_bus_pu` inside `_build_opt_model` for all forecast loads. |
| Reactive power accumulation | Adds `reactive_power_var * 1e-3` for all forecast loads | Same (unchanged) |

---

## Summary of All Changes

| # | Location | Change |
|---|---|---|
| 1 | `__init__` signature | Removed `df_offers`; added `transformer_params`, `theta_A`, `thermal_state_in` |
| 2 | `_filter_input_data` | Removed offer filtering; no longer produces `offer_load_ids` or `df_offers_filt` |
| 3 | `_build_opt_model` — `partic_idxs` | Changed from `offer_load_ids` to `forecast_load_ids` |
| 4 | `_build_opt_model` — bus load allocation | All forecast loads now contribute fixed active power; participant envelope variable no longer in power flow equations |
| 5 | `_build_opt_model` — constraints | Removed SOE offer-cap and reservation constraints; added IEEE C57.91 transformer thermal constraints |
| 6 | `_build_opt_model` — objective | Removed dispatch-cost term (`first_term`); retained width nudge and violation penalty |
| 7 | `_extract_results` | Envelope output simplified to lb/ub only; added `thermal_state_out`; added `thermal_state` to return namedtuple |
| 8 | `_calculate_bus_loads_kw` | Active power accumulation removed; now only returns reactive power background |

# Original converge-soe README:
converge-soe is a python package that implements the shared operating envelope (SOE) optimisation as used in [Project Converge](https://arena.gov.au/projects/project-converge-act-distributed-energy-resources-demonstration-pilot/). It is an open-source version of the code that was tested during the project.

## Prerequisites
`converge-soe` requires the presence of the [IPOPT library](https://coin-or.github.io/Ipopt/). Installation instructions for IPOPT are provided [here](https://coin-or.github.io/Ipopt/INSTALL.html). Note that `converge-soe` requires that the optional ASL library is installed, as explained in the IPOPT installation instructions. A linear solver library must also be installed - both MUMPS and MA27 have been tested with `converge-soe`. The instructions provide links for both of these. MUMPS is fully open source, while MA27 is available free for research but requires a special licence.

## Installation
1. Install IPOPT (see above)
1. Make sure the `ipopt` executable is on your path.
1. Install the converge-soe package 
```
pip install -e .
```

## Use
```python
import converge_soe as csoe
from pyomo.opt import SolverStatus

solver = csoe.SoeSolver(
    netw, forecast, offers, envelope_abs_max=50.0, solver_options={'linear_solver': 'ma27'}
)
status, results = solver.solve()
is_ok = status == SolverStatus.ok
print(results.bus)
print(results.branch)
print(results.soe)
print(results.viol)
```

See the code in `examples/run_scenario.py` for a full example. This is invoked as follows:
```python
cd examples
python ./run_scenario.py -s 1.0 scenario_1 scenario_1_output
```
This will run the SOE engine, using input files in the `scenario_1` directory. Following successful completion, various output CSV and log files will be found in a newly created `scenario_1_output` directory.

The `-s 1.0` part of the command above scales all loads by a factor of 1.0 i.e. it keeps them unmodified. In `example_1`, the network is not overloaded and hence envelopes are permissive. We can see this by looking at `scenario_1_output/soe.csv`, which looks like this:
```csv
load_id,soe_lb_kw,soe_ub_kw,dispatch_kw,payment_dlr
load_1118,-49.999975,49.999975,0.0,0.0
load_1131,-49.999975,49.999975,0.0,0.0
...
```
Most loads have envelopes between -50 kW and 50 kW, and there is no network support dispatch.

We can induce a network support response and more restrictive envelopes by scaling the loads in the system - for example, by a factor of 2:
```python
python ./run_scenario.py -s 2.2 scenario_1 scenario_1_output
```
This produces the following output:
```
load_id,soe_lb_kw,soe_ub_kw,dispatch_kw,payment_dlr
load_1118,-49.999907,49.999991,0.0,0.0
...
load_541,-2.322306,49.999991,0.0,0.0
load_614,-4.445983,49.999991,0.0,0.0
...
```
We see that there are more restrictive lower (load) envelopes, but still no network support dispatches. Scaling the loads further, to 2.5, produces even more restrictive lower envelopes (zero), and includes generation network support dispatch to offset the excessive load:
```
load_id,soe_lb_kw,soe_ub_kw,dispatch_kw,payment_dlr
load_1118,0.0,49.999975,0.0,0.0
...
load_614,0.0,49.999975,3.247845,0.270654
load_632,0.0,49.999975,0.498675,0.041556
load_644,-2.625459,49.999975,0.0,0.0
...
```
