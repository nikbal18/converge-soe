"""tx_limit='legacy' must reproduce the pre-rebuild multistep trajectory.

Baseline captured from the unmodified solver on examples/scenario_doe
(constant 25 °C ambient): theta_HS and total upper envelope per timestep.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

import converge_soe as csoe

REPO = Path(__file__).resolve().parent.parent
SCEN = REPO / "examples" / "scenario_doe"

BASELINE = [  # (theta_HS after step, sum doe_ub_kw)
    (82.072, 28.088), (96.125, 28.823), (102.873, 29.051), (105.743, 27.282),
    (106.862, 26.226), (106.99, 25.504), (107.394, 25.377), (108.626, 25.677),
    (110.352, 25.862), (113.016, 25.969), (116.46, 25.906), (120.0, 25.666),
]


def _ipopt_available():
    from pyomo.environ import SolverFactory
    try:
        return SolverFactory("ipopt").available(exception_flag=False)
    except Exception:
        return False


@pytest.mark.skipif(not _ipopt_available(), reason="ipopt not available")
def test_legacy_mode_matches_baseline():
    netw = json.loads((SCEN / "network.json").read_text())
    tp = json.loads((SCEN / "transformer_params.json").read_text())
    fc = pd.read_csv(SCEN / "forecast_timeseries.csv", dtype={"load_id": str})
    fc["timestamp"] = pd.to_datetime(fc["timestamp"], format="%H:%M")
    state = {}
    for i, ts in enumerate(sorted(fc["timestamp"].unique())):
        f_t = (fc.loc[fc["timestamp"] == ts,
                      ["load_id", "real_power_w", "reactive_power_var"]]
               .set_index("load_id"))
        s = csoe.DoeSolver(netw, f_t, envelope_abs_max=50.0,
                           transformer_params=tp, theta_A=25.0,
                           thermal_state_in=state, quiet=True)
        _, res = s.solve()
        assert res is not None
        state = res.thermal_state
        th, ub = BASELINE[i]
        assert abs(state["tx1"]["theta_HS"] - th) < 0.02, (i, ts)
        assert abs(res.soe["doe_ub_kw"].sum() - ub) < 0.05, (i, ts)
