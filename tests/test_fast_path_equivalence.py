"""Path A (plain, per-timestep model) vs Path B (persistent, mutable Params).

What must agree, and why the test is split into two tolerances:

* PER-FEEDER TOTALS (tight, 1e-6). The total envelope width the network can
  support at a given timestep is a physical quantity fixed by the voltage,
  current and thermal limits. Both paths solve the same problem, so these
  must match. Every headline metric in the study (curtailed energy, cost,
  transformer ageing, loading) is built from these totals.

* PER-NMI ALLOCATION (loose, 1e-2 kW). How the spare capacity is divided
  between customers is NOT uniquely determined. The objective's envelope-width
  term is symmetric across interchangeable participants, so any split summing
  to the same total is equally optimal — a degenerate optimum. Which one ipopt
  returns depends on its tie-break, which differs between the two code paths
  and between ipopt builds/platforms.

  Observed on Windows/ipopt 3.14 (anaconda) with weight 0: plain
  [0.85314, 0.687006] vs persistent [0.770073, 0.770073] — totals agree to
  1e-6, individual values differ by 8e-2 kW. Tightening the solver tolerance
  does not fix this; there is genuinely no unique answer to converge to.

  THE FIX is not a looser tolerance, it is to remove the degeneracy:
  usable_export_weight > 0 adds a small reward on min(doe_ub, desired export),
  which makes the allocation unique. config/default.yaml sets 0.01 and
  scenarios.py applies it, so every real pipeline run is already unique. This
  test therefore runs with the same weight and asserts the tight tolerance.

If this test starts failing again, check usable_export_weight first — at 0 the
per-NMI split is arbitrary and the failure is expected, not a bug. See
docs/MIGRATION_NOTES.md §3 and CHANGES.md §1.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import converge_soe as csoe
from converge_soe import persistent_solver as psm

REPO = Path(__file__).resolve().parent.parent
SCEN = REPO / "examples" / "scenario_doe"

TOL = {"tol": 1e-11, "acceptable_tol": 1e-11}

# Run the comparison the way the pipeline actually runs it. With weight 0
# (the bare library default) the envelope-width objective is symmetric across
# interchangeable participants, so the per-NMI split is a degenerate optimum
# and the two paths tie-break differently. config/default.yaml sets 0.01 and
# scenarios.py uses it, so every real run has a unique optimum — test that.
USABLE_EXPORT_WEIGHT = 0.01


def _ipopt_available():
    from pyomo.environ import SolverFactory
    try:
        return SolverFactory("ipopt").available(exception_flag=False)
    except Exception:
        return False


@pytest.mark.skipif(not _ipopt_available(), reason="ipopt not available")
def test_fast_path_equivalence(monkeypatch):
    netw = json.loads((SCEN / "network.json").read_text())
    tp = json.loads((SCEN / "transformer_params.json").read_text())
    tp["I_rated"] = 200e3 / 415.0
    fc = pd.read_csv(SCEN / "forecast_timeseries.csv", dtype={"load_id": str})
    fc["timestamp"] = pd.to_datetime(fc["timestamp"], format="%H:%M")
    tss = sorted(fc["timestamp"].unique())
    loads = sorted(fc["load_id"].unique())
    wp = fc.pivot(index="timestamp", columns="load_id",
                  values="real_power_w").reindex(columns=loads)
    wq = fc.pivot(index="timestamp", columns="load_id",
                  values="reactive_power_var").reindex(columns=loads)

    # plain path
    state, plain = {}, []
    for ts in tss:
        f_t = (fc.loc[fc["timestamp"] == ts,
                      ["load_id", "real_power_w", "reactive_power_var"]]
               .set_index("load_id"))
        s = csoe.DoeSolver(netw, f_t, transformer_params=tp, theta_A=25.0,
                           thermal_state_in=state, tx_limit="dtr",
                           quiet=True, soft_limits=True, solver_options=TOL,
                           usable_export_weight=USABLE_EXPORT_WEIGHT)
        _, res = s.solve()
        assert res is not None
        state = res.thermal_state
        plain.append((res.soe.copy(), state["tx1"]["theta_HS"]))

    # persistent path with the same tolerance
    monkeypatch.setattr(psm, "_WARM_START_OPTS",
                        {**psm._WARM_START_OPTS, **TOL})
    ps = psm.PersistentDoeSolver(
        json.loads((SCEN / "network.json").read_text()), loads,
        transformer_params=tp, tx_limit="dtr", quiet=True,
        usable_export_weight=USABLE_EXPORT_WEIGHT)
    for i, ts in enumerate(tss):
        r = ps.solve_step(wp.values[i], wq.values[i], theta_A=25.0)
        assert r.status == "ok", r.failure_reason
        p_doe, p_th = plain[i]
        fast, ref = r.doe.values, p_doe.values

        # 1. Totals per envelope side (doe_lb_kw, doe_ub_kw) must match tightly.
        #    This is the physical quantity every reported metric depends on.
        np.testing.assert_allclose(
            fast.sum(axis=0), ref.sum(axis=0), atol=1e-6, rtol=1e-6,
            err_msg=(f"timestep {i}: feeder-total envelope differs between the "
                     f"plain and persistent paths. This is a real "
                     f"disagreement, not the degenerate-allocation tie-break."))

        # 2. Per-NMI allocation. Unique because usable_export_weight > 0, so
        #    this can be tight. If it fails with weight 0 it means the optimum
        #    is degenerate (see module docstring); with weight 0.01 a failure
        #    is a genuine disagreement between the two paths.
        np.testing.assert_allclose(
            fast, ref, atol=1e-4, rtol=1e-4,
            err_msg=(f"timestep {i}: per-NMI envelopes differ despite "
                     f"usable_export_weight={USABLE_EXPORT_WEIGHT} breaking "
                     f"the allocation degeneracy."))

        # 3. Thermal trajectory follows the totals, so it must match tightly.
        assert abs(r.thermal_state["tx1"]["theta_HS"] - p_th) < 1e-3
