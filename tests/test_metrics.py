"""Hand-computed F_AA, %LOL, E_curt on a 3-timestep toy case, to 1e-9."""
import math

import numpy as np
import pandas as pd
import pytest

from converge_soe import metrics as M
from converge_soe import thermal


def test_faa_reference_point():
    assert thermal.faa(110.0) == pytest.approx(1.0, abs=1e-12)


def test_faa_hand_computed():
    # F_AA(98) = exp(15000/383 - 15000/371)
    expected = math.exp(15000.0 / 383.0 - 15000.0 / (98.0 + 273.0))
    assert thermal.faa(98.0) == pytest.approx(expected, abs=1e-12)


def test_ageing_hand_computed():
    th = pd.DataFrame({"theta_HS_C": [110.0, 98.0, 120.0]})
    dt_h = 0.5
    L = sum(math.exp(15000 / 383 - 15000 / (t + 273))
            for t in [110.0, 98.0, 120.0]) * dt_h
    out = M.ageing(th, dt_h)
    assert out["ageing_hours"] == pytest.approx(L, abs=1e-9)
    assert out["percent_LOL"] == pytest.approx(100 * L / 180000, abs=1e-12)
    assert out["peak_theta_HS_C"] == 120.0


def test_curtailment_hand_computed():
    # 1 NMI, 3 timesteps at 30 min: desired 5, 10, -2 kW; ub 4, 12, 0; lb -3
    doe = pd.DataFrame({
        "timestamp": pd.to_datetime(
            ["2026-01-01 00:00", "2026-01-01 00:30", "2026-01-01 01:00"]),
        "load_id": ["a"] * 3,
        "p_des_kw": [5.0, 10.0, -2.0],
        "doe_ub_kw": [4.0, 12.0, 0.0],
        "doe_lb_kw": [-3.0, -3.0, -3.0],
    })
    out = M.curtailment(doe, price_per_kwh=0.10)
    # curtailed: max(0,5-4)=1 at t0 only -> 1 kW * 0.5 h = 0.5 kWh
    assert out["E_curt_kwh"] == pytest.approx(0.5, abs=1e-9)
    # constrained import: -p_des=2 vs |lb|=3 -> 0
    assert out["E_imp_kwh"] == pytest.approx(0.0, abs=1e-9)
    # desired export: (5+10)*0.5
    assert out["E_des_export_kwh"] == pytest.approx(7.5, abs=1e-9)
    assert out["curtailment_cost"] == pytest.approx(0.05, abs=1e-9)


def test_bau_infinite_ub_gives_zero_curtailment():
    doe = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:30"]),
        "load_id": ["a", "a"],
        "p_des_kw": [50.0, 60.0],
        "doe_ub_kw": [np.inf, np.inf],
        "doe_lb_kw": [-np.inf, -np.inf],
    })
    out = M.curtailment(doe)
    assert out["E_curt_kwh"] == 0.0
    assert out["E_imp_kwh"] == 0.0
