"""IEEE C57.91 transformer thermal model, DTR inversion, and ageing maths.

This module owns everything thermal:

* the exact exponential step-response discretisation of the two-layer
  (top-oil + winding hot-spot) C57.91 clause-7 model,
* the inversion of that model into a per-interval **dynamic thermal rating**
  (the largest squared loading K² whose end-of-interval hot-spot temperature
  just reaches the limit), and
* the C57.91 insulation-ageing formulas (F_AA, equivalent ageing hours, %LOL).

The end-of-interval hot-spot temperature, given the thermal state at the
start of the interval (ΔθTO_0, ΔθHS_0) and ambient θ_A, is

    θ_HS(K²) = θ_A
             + α_TO·ΔθTO_R·((K²·R + 1)/(R + 1))^n + (1 − α_TO)·ΔθTO_0
             + α_W ·ΔθHS_R·(K²)^m               + (1 − α_W )·ΔθHS_0

    α_TO = 1 − exp(−Δt/τ_TO)        α_W = 1 − exp(−Δt/τ_W)

θ_HS is strictly increasing and continuous in K², so it can be inverted
numerically (scipy.optimize.brentq) for the largest loading that just
reaches θ_HS_max. That inverted limit is what `dtr_current_limit_pu`
returns; the optimisation model then needs only a simple bound on
square_current_pu instead of a nonlinear constraint — which also removes
the pow'(0,0.8) gradient failure recorded in the old output.log.

Parameter dict keys (same as the legacy transformer_params.json):
    tau_TO, tau_W        thermal time constants, minutes
    delta_theta_TO_R     rated top-oil rise over ambient, °C
    delta_theta_HS_R     rated hot-spot rise over top-oil, °C
    R                    rated load loss / no-load loss ratio
    n, m                 oil / winding exponents (0.8 ONAN)
    I_rated              rated secondary current, A
    theta_HS_max         hot-spot limit, °C
    dt                   interval length, minutes (must equal the data Δt)
"""

import math
from collections import namedtuple

from scipy.optimize import brentq

# C57.91 ageing constants
_B = 15000.0                 # K — ageing activation constant
_THETA_REF_K = 383.0         # K — 110 °C reference hot-spot temperature
NORMAL_INSULATION_LIFE_H = 180_000.0  # h — C57.91 normal insulation life

DtrResult = namedtuple("DtrResult", "i_max_pu status")

DTR_THERMAL_BINDING = "thermal_binding"
DTR_NOT_BINDING = "not_binding"
DTR_ALREADY_OVER = "already_over_limit"


class ThermalCoeffs:
    """Precomputed per-(params, dt) coefficients for the C57.91 step response.

    ``brentq`` costs ~20 function evaluations per call and the function is
    evaluated once per transformer per timestep, so everything that does not
    change between evaluations (α_TO, α_W, R+1, the rated rises) is computed
    once here and reused.
    """

    __slots__ = ("tau_TO", "tau_W", "dTO_R", "dHS_R", "R", "n", "m",
                 "I_rated", "theta_HS_max", "dt", "alpha_TO", "alpha_W",
                 "R_plus_1")

    def __init__(self, tp):
        self.tau_TO = float(tp["tau_TO"])
        self.tau_W = float(tp["tau_W"])
        self.dTO_R = float(tp["delta_theta_TO_R"])
        self.dHS_R = float(tp["delta_theta_HS_R"])
        self.R = float(tp["R"])
        self.n = float(tp["n"])
        self.m = float(tp["m"])
        self.I_rated = float(tp.get("I_rated", 0.0))
        self.theta_HS_max = float(tp["theta_HS_max"])
        self.dt = float(tp["dt"])
        self.alpha_TO = 1.0 - math.exp(-self.dt / self.tau_TO)
        self.alpha_W = 1.0 - math.exp(-self.dt / self.tau_W)
        self.R_plus_1 = self.R + 1.0

    # -- forward model -------------------------------------------------------
    def ultimate_rises(self, k2):
        """Ultimate (steady-state) rises for squared loading k2."""
        dTO_U = self.dTO_R * ((k2 * self.R + 1.0) / self.R_plus_1) ** self.n
        dHS_U = self.dHS_R * k2 ** self.m if k2 > 0.0 else 0.0
        return dTO_U, dHS_U

    def step(self, k2, dTO_0, dHS_0):
        """Advance the thermal state one interval at squared loading k2.

        Returns (dTO_new, dHS_new). Exact discretisation of the first-order
        ODEs — identical to the recursion in the legacy _extract_results.
        """
        dTO_U, dHS_U = self.ultimate_rises(k2)
        dTO_new = (dTO_U - dTO_0) * self.alpha_TO + dTO_0
        dHS_new = (dHS_U - dHS_0) * self.alpha_W + dHS_0
        return dTO_new, dHS_new

    def theta_HS_end(self, k2, theta_A, dTO_0, dHS_0):
        """End-of-interval hot-spot temperature at squared loading k2.

        Strictly increasing and continuous in k2 (both exponents > 0).
        """
        dTO_new, dHS_new = self.step(k2, dTO_0, dHS_0)
        return theta_A + dTO_new + dHS_new


_coeffs_cache = {}


def get_coeffs(tp):
    """Cached ThermalCoeffs for a params dict (keyed by its values)."""
    key = (tp["tau_TO"], tp["tau_W"], tp["delta_theta_TO_R"],
           tp["delta_theta_HS_R"], tp["R"], tp["n"], tp["m"],
           tp.get("I_rated"), tp["theta_HS_max"], tp["dt"])
    c = _coeffs_cache.get(key)
    if c is None:
        c = _coeffs_cache[key] = ThermalCoeffs(tp)
    return c


# ---------------------------------------------------------------------------
# Forward state advance (used after every solve, and for BAU / post-hoc runs)
# ---------------------------------------------------------------------------
def forward_step(tp, k2, theta_A, dTO_0=0.0, dHS_0=0.0):
    """Advance one interval; returns dict(delta_theta_TO, delta_theta_HS, theta_HS)."""
    c = get_coeffs(tp)
    dTO_new, dHS_new = c.step(k2, dTO_0, dHS_0)
    return {"delta_theta_TO": dTO_new, "delta_theta_HS": dHS_new,
            "theta_HS": theta_A + dTO_new + dHS_new}


def open_loop_trajectory(tp, k2_series, theta_A_series,
                         dTO_0=0.0, dHS_0=0.0):
    """Run the recursion forward over whole series (e.g. BAU currents).

    ``k2_series`` and ``theta_A_series`` are equal-length iterables. Returns a
    list of theta_HS values (°C), one per interval.
    """
    c = get_coeffs(tp)
    dTO, dHS = dTO_0, dHS_0
    out = []
    for k2, tha in zip(k2_series, theta_A_series):
        dTO, dHS = c.step(float(k2), dTO, dHS)
        out.append(float(tha) + dTO + dHS)
    return out


# ---------------------------------------------------------------------------
# The dynamic thermal rating: inversion of the forward model
# ---------------------------------------------------------------------------
def dtr_k2_max(tp, theta_A, dTO_0, dHS_0, k2_emergency=4.0, k2_floor=1e-6):
    """Largest K² whose end-of-interval θ_HS just reaches θ_HS_max.

    Returns (k2_max, status) where status is one of
    'thermal_binding' | 'not_binding' | 'already_over_limit'.

    * already_over_limit — θ_HS(0) ≥ θ_HS_max: the transformer is over the
      limit from stored heat before any new load. Returns ``k2_floor`` (a tiny
      positive limit) so the problem stays feasible; the status makes the
      condition visible in the outputs instead of fatal.
    * not_binding — θ_HS(k2_emergency) < θ_HS_max: thermal never binds this
      interval; the mechanical/short-time emergency backstop is the limit.
    * thermal_binding — the interesting case; found with brentq on the
      strictly-increasing scalar function.
    """
    c = get_coeffs(tp)
    theta_max = c.theta_HS_max

    f0 = c.theta_HS_end(0.0, theta_A, dTO_0, dHS_0)
    if f0 >= theta_max:
        return k2_floor, DTR_ALREADY_OVER

    f_hi = c.theta_HS_end(k2_emergency, theta_A, dTO_0, dHS_0)
    if f_hi < theta_max:
        return k2_emergency, DTR_NOT_BINDING

    k2_star = brentq(
        lambda k2: c.theta_HS_end(k2, theta_A, dTO_0, dHS_0) - theta_max,
        0.0, k2_emergency, xtol=1e-12, rtol=8.9e-16, maxiter=200,
    )
    return max(k2_star, k2_floor), DTR_THERMAL_BINDING


def dtr_current_limit_pu(tp, theta_A, dTO_0, dHS_0, c2,
                         k2_emergency=4.0, k2_floor=1e-6):
    """Return (i_max_pu, status) — the dynamic thermal current limit for this
    timestep, given the transformer's thermal state at the start of it.

    c2 = (i_base_a / I_rated)**2, so K2 = square_current_pu * c2.
    Uses scipy.optimize.brentq on a strictly-increasing scalar function.

    The returned per-unit limit is what the solver puts straight into the
    transformer branch's ``i_max_pu`` — the model then carries **no**
    nonlinear thermal constraint at all.
    """
    k2_max, status = dtr_k2_max(tp, theta_A, dTO_0, dHS_0,
                                k2_emergency=k2_emergency, k2_floor=k2_floor)
    return DtrResult(math.sqrt(k2_max / c2), status)


# ---------------------------------------------------------------------------
# Ageing (IEEE C57.91)
# ---------------------------------------------------------------------------
def faa(theta_HS_C):
    """Ageing acceleration factor at hot-spot temperature θ_HS (°C).

    F_AA = exp(15000/383 − 15000/(θ_HS + 273)); 1.0 at the 110 °C reference.
    """
    return math.exp(_B / _THETA_REF_K - _B / (theta_HS_C + 273.0))


def equivalent_ageing_hours(theta_HS_series_C, dt_hours):
    """L = Σ F_AA(θ_HS(t)) · Δt  — equivalent ageing hours over the series."""
    return sum(faa(t) for t in theta_HS_series_C) * dt_hours


def percent_loss_of_life(ageing_hours):
    """%LOL = 100 · L / 180,000 h (C57.91 normal insulation life)."""
    return 100.0 * ageing_hours / NORMAL_INSULATION_LIFE_H
