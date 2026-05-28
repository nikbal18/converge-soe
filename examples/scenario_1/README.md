# Scenario 1 — Simple 4-Bus Radial LV Network

This is a minimal working example you can use to practice running the SOE solver.

## Network Topology

```
[Infeeder: 11 kV slack]
        |
    [bus_hv]  (11 kV)
        |
   [TX1: 200 kVA, 11 kV / 415 V, ~4% impedance]
        |
    [bus_lv]  (415 V)
        |
   [line_1: 300 m, 1.0 Ω/km]
        |
    [bus_a]  (415 V) ── load 1001 (3 kW, non-participant)
        |               load 1002 (2 kW forecast, DER participant ☀️)
   [line_2: 300 m, 1.0 Ω/km]
        |
    [bus_b]  (415 V) ── load 1003 (1.5 kW forecast, DER participant ☀️)
                        load 1004 (4.5 kW, non-participant)
```

## What Each File Contains

### network.json
Describes the physical grid. Key things to notice:
- **units**: all values in the JSON are in kV / A / kW / Ω. The units block tells the solver to multiply by 1000 to convert kV→V and kW→W.
- **Infeeder**: the slack bus. `v_setpoint: 11.0` sets it to 11 kV (1.0 pu). The solver holds this bus at fixed voltage.
- **Node**: each bus has a `v_base` (kV) for per-unit conversion. bus_lv/bus_a/bus_b also have `v_min`/`v_max` voltage limits (in kV, so 0.390–0.440 kV = 0.94–1.06 pu).
- **Transformer tx1**: z is `[[r_pri, x_pri], [r_sec, x_sec]]` in Ω. The solver only uses the secondary impedance `[1]`. `nom_turns_ratio` is set so the voltage ratio is exactly 1.0 pu (11.0/0.415 = 26.506).
- **Line**: `z = [r_per_km, x_per_km]` in Ω/km, multiplied by `length` (km) inside the solver.
- **Load**: each customer is just a node connection. The power is in forecast.csv, not here.

### forecast.csv
One row per customer, giving their expected power for this 5-minute interval.
- `real_power_w`: active power consumption in watts (positive = consuming from grid)
- `reactive_power_var`: reactive power in VAr

Note: load 1002 and 1003 are participants, so their `real_power_w` is used for reactive power accounting only — their active power position is determined by the envelope variable `p_inj_oe_kw` in the optimisation.

### offers.csv
Only participants (1002, 1003) appear here. Non-participants have no offers.
- `injection`: JSON list of `[quantity_kw, price_$/kWh]` bid steps. Load 1002 offers up to 5 kW of solar export at $0.08/kWh.
- `consumption`: JSON list of `[quantity_kw, price_$/kWh]` bid steps. Load 1002 offers up to 2 kW of extra consumption at $0.05/kWh. Load 1003 has no consumption offer (`[]`).
- `reservation_l`: the minimum injection (kW) the customer is pre-entitled to. Negative means they can consume this much. `-5.0` → they can always import up to 5 kW.
- `reservation_u`: the maximum injection (kW) the customer is pre-entitled to. `5.0` → they can always export up to 5 kW (if the grid allows it).

## Running the Scenario

From the `converge-soe/examples/` directory:

```bash
python run_scenario.py scenario_1 scenario_1_output
```

Or with load scaling (e.g., 2× to stress the network):

```bash
python run_scenario.py -s 2.0 scenario_1 scenario_1_output
```

Note: you need IPOPT installed. The solver uses MUMPS by default. If you have MA27 (HSL), change the solver_options in run_scenario.py.

## What to Look For in the Outputs

**soe.csv** — the main result. One row per participant:
- `soe_lb_kw`: lower bound (most negative = max import allowed). Should be ≤ reservation_l if network is unconstrained.
- `soe_ub_kw`: upper bound (most positive = max export allowed). Should be ≥ reservation_u if unconstrained.
- `dispatch_kw`: any network support dispatched (+ve = injection, −ve = consumption).
- `payment_dlr`: cost of that dispatch.

**If the network is unconstrained:** soe_lb ≈ −5.0 and soe_ub ≈ 5.0 (or 4.0 for load 1003). The solver gives full reservation envelopes with no dispatch.

**If constrained (try `-s 2.0`):** soe_ub will be tightened below the reservation, and/or dispatch_kw will be non-zero as the solver buys network support from a participant.

**bus.csv** — voltage at each bus (pu) for both envelope scenarios:
- `voltage_pu_oel`: voltage when all participants are at their lower bound (max consumption)
- `voltage_pu_oer`: voltage when all participants are at their upper bound (max injection)

**branch.csv** — current (A) and power flows (W, VAr) per branch, for both scenarios.

**viol.csv** — any power balance violations (should be empty/zero for a feasible case).

## Things to Try

1. **Run baseline**: use `-s 1.0` (default). Check that soe.csv shows full reservation envelopes. viol.csv should be empty.
2. **Stress the network**: use `-s 2.5` or higher. Watch soe_ub tighten as line currents or voltage limits bind.
3. **Add a participant**: add load 1001 to offers.csv with an injection offer. Does its bus location give it a wider or tighter envelope than 1002?
4. **Change the line impedance**: make z = [2.0, 0.8] (more resistive). What happens to the envelopes at -s 1.0?
5. **Trace a constraint**: look at bus.csv `voltage_pu_oer` for bus_b when the scenario is tight. Is the voltage hitting v_max?
