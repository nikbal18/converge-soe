"""Phase 1.2 root-find: inversion consistency, monotonicity, branches."""
import math

import pytest

from converge_soe import thermal

TP = dict(tau_TO=60.0, tau_W=7.0, delta_theta_TO_R=55.0,
          delta_theta_HS_R=23.0, R=6.0, n=0.8, m=0.8, I_rated=481.9,
          theta_HS_max=120.0, dt=30.0)

STATES = [(25.0, 0.0, 0.0), (40.0, 50.0, 15.0), (10.0, 20.0, 5.0),
          (35.0, 60.0, 22.0), (0.0, 5.0, 1.0)]


@pytest.mark.parametrize("theta_A,dTO,dHS", STATES)
def test_forward_of_inverse_hits_limit_exactly(theta_A, dTO, dHS):
    k2, status = thermal.dtr_k2_max(TP, theta_A, dTO, dHS)
    if status != thermal.DTR_THERMAL_BINDING:
        pytest.skip("not the binding branch for this state")
    theta_end = thermal.get_coeffs(TP).theta_HS_end(k2, theta_A, dTO, dHS)
    assert abs(theta_end - TP["theta_HS_max"]) < 1e-9


def test_theta_strictly_increasing_in_k2():
    c = thermal.get_coeffs(TP)
    vals = [c.theta_HS_end(k2, 25.0, 30.0, 10.0)
            for k2 in [0.0, 0.1, 0.5, 1.0, 2.0, 4.0]]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_limit_monotone_decreasing_in_ambient():
    lims = [thermal.dtr_k2_max(TP, a, 30.0, 10.0)[0]
            for a in (0, 10, 20, 30, 40)]
    assert all(a > b for a, b in zip(lims, lims[1:]))


def test_already_over_limit_branch():
    # stored heat alone exceeds the limit: theta_A + rises decayed >= 120
    k2, status = thermal.dtr_k2_max(TP, 80.0, 80.0, 40.0)
    assert status == thermal.DTR_ALREADY_OVER
    assert k2 == pytest.approx(1e-6)


def test_not_binding_branch():
    # freezing ambient, cold transformer, tiny interval: never binds
    tp = dict(TP, dt=1.0)
    k2, status = thermal.dtr_k2_max(tp, -10.0, 0.0, 0.0, k2_emergency=4.0)
    assert status == thermal.DTR_NOT_BINDING
    assert k2 == 4.0


def test_forward_step_matches_legacy_recursion():
    import numpy as np
    # the exact discretisation from the legacy _extract_results
    K2, dTO_0, dHS_0, theta_A = 0.8, 12.0, 4.0, 22.0
    alpha_TO = 1 - np.exp(-TP["dt"] / TP["tau_TO"])
    alpha_W = 1 - np.exp(-TP["dt"] / TP["tau_W"])
    dTO_U = TP["delta_theta_TO_R"] * ((K2 * TP["R"] + 1) / (TP["R"] + 1)) ** TP["n"]
    dHS_U = TP["delta_theta_HS_R"] * K2 ** TP["m"]
    exp_TO = (dTO_U - dTO_0) * alpha_TO + dTO_0
    exp_HS = (dHS_U - dHS_0) * alpha_W + dHS_0
    st = thermal.forward_step(TP, K2, theta_A, dTO_0, dHS_0)
    assert st["delta_theta_TO"] == pytest.approx(exp_TO, abs=1e-12)
    assert st["delta_theta_HS"] == pytest.approx(exp_HS, abs=1e-12)
    assert st["theta_HS"] == pytest.approx(theta_A + exp_TO + exp_HS, abs=1e-12)
