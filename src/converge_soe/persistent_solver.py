"""Persistent DOE solver: the Pyomo model is built once per substation and
only mutable Params change between timesteps (opt-in via --fast).

The plain path (doe_solver.DoeSolver) constructs thousands of symbolic
expressions in Python for every timestep. Here the ConcreteModel is built
ONCE; everything that varies per timestep is a mutable Param:

  * per-load fixed active power and reactive power,
  * the transformer's squared current limit (the Phase-1 dynamic thermal
    rating, recomputed each step from the carried thermal state), and
  * the participant export price.

Each timestep is then: update params → solve → read variables. No expression
construction at all.

Solver interface: uses appsi ipopt (pyomo.contrib.appsi) when importable and
available — it keeps the problem representation in memory between solves and
skips most of the classic interface's per-solve setup. Falls back to a single
re-used classic SolverFactory('ipopt') otherwise (still skips model
construction, still warm-starts). Which interface was chosen is exposed as
``.interface`` and logged once.

Warm start: the previous solution is simply left in the variable .value's,
and ipopt is given warm_start_init_point=yes with small bound-push values —
consecutive intervals are very similar, so iteration counts drop.

Supports tx_limit 'dtr' and 'static' only. The 'legacy' combined nonlinear
constraint stays on the plain path, which exists precisely to serve as the
reference implementation.

Equivalence with the plain path is enforced by
tests/test_fast_path_equivalence.py (doe bounds to 1e-4, theta_HS to 1e-3).
"""

import logging
from collections import namedtuple

import numpy as np
import pandas as pd
from pyomo.environ import (
    ConcreteModel, ConstraintList, NonNegativeReals, Objective, Param,
    Reals, SolverFactory, Var, minimize, value as pyo_value,
)
from pyomo.common.errors import ApplicationError

from . import thermal as _thermal
from .doe_solver import S_BASE_VA
from .doe_solver import SoeSolver as _DoeSolver, _oe_idxs, _kw_to_pu, _w_to_pu

logger = logging.getLogger(__name__)

StepResult = namedtuple(
    "StepResult",
    "status doe bus branch viol thermal_state dtr_info soft_viols failure_reason",
)

_WARM_START_OPTS = {
    "warm_start_init_point": "yes",
    "warm_start_bound_push": 1e-9,
    "warm_start_mult_bound_push": 1e-9,
    "mu_init": 1e-6,
}


class PersistentDoeSolver:
    def __init__(self, netw_ejson, load_ids, envelope_abs_max=50.0,
                 participant_load_ids=None, df_prices=None,
                 transformer_params=None, tx_limit="dtr",
                 k_emergency=2.0, k2_floor=1e-6,
                 soft_limits=True, soft_limit_penalty=1000.0,
                 quiet=True, solver_name="ipopt", use_appsi=None,
                 thermal_state_in=None, usable_export_weight=0.0):
        if tx_limit not in ("dtr", "static"):
            raise ValueError("PersistentDoeSolver supports tx_limit dtr|static; "
                             "'legacy' lives on the plain path")
        if tx_limit == "dtr" and transformer_params is None:
            raise ValueError("tx_limit='dtr' requires transformer_params")

        self.tx_limit = tx_limit
        self.transformer_params = transformer_params
        self.k2_emergency = float(k_emergency) ** 2
        self.k2_floor = float(k2_floor)
        self.soft_limits = bool(soft_limits)
        self.soft_limit_penalty = float(soft_limit_penalty)
        self.quiet = bool(quiet)
        self.solver_name = solver_name
        self.usable_export_weight = float(usable_export_weight)
        self.thermal_state = dict(thermal_state_in or {})
        self.load_ids = [str(x) for x in load_ids]

        # ---- network data: parsed once, via the plain solver's own builder
        # (guarantees identical per-unit conversion), with a zero forecast.
        df0 = pd.DataFrame(
            {"real_power_w": 0.0, "reactive_power_var": 0.0},
            index=pd.Index(self.load_ids, name="load_id"),
        )
        base = _DoeSolver.__new__(_DoeSolver)
        base.netw_ejson = netw_ejson
        base.netw_ejson["components"] = dict(sorted(netw_ejson["components"].items()))
        base.df_forecasts = df0
        base.envelope_abs_max = envelope_abs_max
        base.participant_load_ids_input = participant_load_ids
        base.df_prices = df_prices
        base.transformer_params = transformer_params
        base.theta_A = 20.0
        base.thermal_state_in = {}
        base.tx_limit = "static"      # static parse; DTR is applied via Params
        base.k_emergency = k_emergency
        base.k2_floor = k2_floor
        base.soft_limits = soft_limits
        base.soft_limit_penalty = soft_limit_penalty
        base.quiet = quiet
        base.warm_start_in = {}
        base.dtr_info = {}
        base._network_cache_in = None
        base.solver_name = solver_name
        base.solver_options = {}
        base._filter_input_data()
        base._build_network_data()
        self.net = base   # buses / branches / loads / transformers / partic ids

        self.partic_load_ids = base.partic_load_ids
        self.partic_prices = base.partic_prices
        self.forecast_load_ids = base.forecast_load_ids
        # position of each forecast load in the caller's array order
        self._pos = {lid: i for i, lid in enumerate(self.load_ids)}

        # Hoisted once (the plain path re-scans branches per constraint —
        # an O(B²) DataFrame scan): downstream branch ids per bus.
        self.downstream = {}
        for b_id, row in base.branches.iterrows():
            self.downstream.setdefault(row["from_bus_id"], []).append(b_id)

        self._build_model()
        self._make_solver(use_appsi)

    # ------------------------------------------------------------------
    def _build_model(self):
        net = self.net
        m = self.model = ConcreteModel()

        bus_idxs = list(net.buses.index)
        branch_idxs = list(net.branches.index)
        partic = self.partic_load_ids
        busld = net.load_buses

        bus_oe = [(b, oe) for b in bus_idxs for oe in _oe_idxs]
        branch_oe = [(b, oe) for b in branch_idxs for oe in _oe_idxs]
        partic_oe = [(l, oe) for l in partic for oe in _oe_idxs]
        busld_oe = [(b, oe) for b in busld for oe in _oe_idxs]

        # ---- mutable parameters (all that changes between timesteps)
        m.p_fixed_pu = Param(list(self.forecast_load_ids), mutable=True,
                             initialize=0.0, within=Reals)
        m.q_pu = Param(list(self.forecast_load_ids), mutable=True,
                       initialize=0.0, within=Reals)
        m.price = Param(partic, mutable=True, initialize=0.0, within=Reals)
        m.p_des_pos_kw = Param(partic, mutable=True, initialize=0.0, within=Reals)
        tx_ids = list(net.transformers.index)
        m.tx_imax2 = Param(tx_ids, mutable=True, initialize=1e6, within=Reals)

        # ---- variables (bounds as Phase 1.4)
        m.square_voltage_pu = Var(bus_oe, domain=NonNegativeReals,
                                  bounds=(0.25, 2.25), initialize=1.0)
        m.square_current_pu = Var(branch_oe, domain=NonNegativeReals,
                                  bounds=(1e-8, None), initialize=1e-8)
        m.branch_active_pu = Var(branch_oe, domain=Reals, initialize=0.0)
        m.branch_reactive_pu = Var(branch_oe, domain=Reals, initialize=0.0)

        def oe_bounds(mm, load_id, oe):
            return ((-net.envelope_abs_max, 0.0) if oe == "oel"
                    else (0.0, net.envelope_abs_max))
        m.p_inj_oe_kw = Var(partic_oe, domain=Reals, bounds=oe_bounds,
                            initialize=0.0)

        m.sof_bus_a_kw = Var([(b, oe, ci) for b, oe in busld_oe for ci in ("con", "inj")],
                             domain=NonNegativeReals, initialize=0.0)
        m.sof_bus_r_kw = Var([(b, oe, ci) for b, oe in busld_oe for ci in ("con", "inj")],
                             domain=NonNegativeReals, initialize=0.0)
        if self.soft_limits:
            m.viol_v_pu = Var(bus_oe, domain=NonNegativeReals, initialize=0.0)
            m.viol_i_pu = Var(branch_oe, domain=NonNegativeReals, initialize=0.0)
        if self.usable_export_weight > 0:
            m.usable_export_kw = Var(partic, domain=NonNegativeReals,
                                     initialize=0.0)

        # ---- bus load allocation (params / envelope vars)
        partic_set = set(partic)
        a_bus = {(b, oe): [] for b in bus_idxs for oe in _oe_idxs}
        r_bus = {(b, oe): [] for b in bus_idxs for oe in _oe_idxs}
        for load_id, lrow in net.loads.iterrows():
            bus_id = lrow["bus_id"]
            if load_id in partic_set:
                for oe in _oe_idxs:
                    r_bus[bus_id, oe].append(m.q_pu[load_id])
                    a_bus[bus_id, oe].append(-m.p_inj_oe_kw[load_id, oe] * _kw_to_pu)
            elif load_id in set(self.forecast_load_ids):
                for oe in _oe_idxs:
                    r_bus[bus_id, oe].append(m.q_pu[load_id])
                    a_bus[bus_id, oe].append(m.p_fixed_pu[load_id])
        for b in busld:
            for oe in _oe_idxs:
                a_bus[b, oe].append(
                    (m.sof_bus_a_kw[b, oe, "con"] - m.sof_bus_a_kw[b, oe, "inj"]) * _kw_to_pu)
                r_bus[b, oe].append(
                    (m.sof_bus_r_kw[b, oe, "con"] - m.sof_bus_r_kw[b, oe, "inj"]) * _kw_to_pu)

        # ---- constraints (identical maths to the plain model)
        m.c = ConstraintList()
        for oe in _oe_idxs:
            for bus_id, brow in net.buses.iterrows():
                if pd.notna(brow["v_mag_setpoint_pu"]):
                    m.c.add(m.square_voltage_pu[bus_id, oe] == brow["v_mag_setpoint_pu"] ** 2)
                else:
                    if pd.notna(brow["v_mag_max_pu"]):
                        rhs = brow["v_mag_max_pu"] ** 2
                        if self.soft_limits:
                            m.c.add(m.square_voltage_pu[bus_id, oe] <= rhs + m.viol_v_pu[bus_id, oe])
                        else:
                            m.c.add(m.square_voltage_pu[bus_id, oe] <= rhs)
                    if pd.notna(brow["v_mag_min_pu"]):
                        rhs = brow["v_mag_min_pu"] ** 2
                        if self.soft_limits:
                            m.c.add(m.square_voltage_pu[bus_id, oe] >= rhs - m.viol_v_pu[bus_id, oe])
                        else:
                            m.c.add(m.square_voltage_pu[bus_id, oe] >= rhs)

        tx_id_set = set(tx_ids)
        for oe in _oe_idxs:
            for branch_id, brow in net.branches.iterrows():
                to_b, from_b = brow["to_bus_id"], brow["from_bus_id"]
                down = self.downstream.get(to_b, [])
                m.c.add(
                    m.branch_active_pu[branch_id, oe] == sum(a_bus[to_b, oe])
                    + brow["r_pu"] * m.square_current_pu[branch_id, oe]
                    + sum(m.branch_active_pu[b, oe] for b in down)
                )
                m.c.add(
                    m.branch_reactive_pu[branch_id, oe] == sum(r_bus[to_b, oe])
                    + brow["x_pu"] * m.square_current_pu[branch_id, oe]
                    + sum(m.branch_reactive_pu[b, oe] for b in down)
                )
                vr = brow["voltage_ratio_pu"]
                lhs_v = (m.square_voltage_pu[to_b, oe] * vr ** 2
                         if pd.notna(vr) else m.square_voltage_pu[to_b, oe])
                m.c.add(
                    lhs_v - m.square_voltage_pu[from_b, oe] == -2 * (
                        m.branch_active_pu[branch_id, oe] * brow["r_pu"]
                        + m.branch_reactive_pu[branch_id, oe] * brow["x_pu"]
                    ) + (brow["r_pu"] ** 2 + brow["x_pu"] ** 2)
                    * m.square_current_pu[branch_id, oe]
                )
                m.c.add(
                    m.square_current_pu[branch_id, oe] * m.square_voltage_pu[from_b, oe]
                    == m.branch_active_pu[branch_id, oe] ** 2
                    + m.branch_reactive_pu[branch_id, oe] ** 2
                )
                # current limits: transformers via the mutable Param (the DTR),
                # lines via their static rating.
                if branch_id in tx_id_set:
                    if self.soft_limits:
                        m.c.add(m.square_current_pu[branch_id, oe]
                                <= m.tx_imax2[branch_id] + m.viol_i_pu[branch_id, oe])
                    else:
                        m.c.add(m.square_current_pu[branch_id, oe] <= m.tx_imax2[branch_id])
                elif pd.notna(brow["i_max_pu"]):
                    rhs = brow["i_max_pu"] ** 2
                    if self.soft_limits:
                        m.c.add(m.square_current_pu[branch_id, oe] <= rhs + m.viol_i_pu[branch_id, oe])
                    else:
                        m.c.add(m.square_current_pu[branch_id, oe] <= rhs)

        # usable-export reward constraints (u_i <= ub_i, u_i <= P_des_i+)
        if self.usable_export_weight > 0:
            for l in partic:
                m.c.add(m.usable_export_kw[l] <= m.p_inj_oe_kw[l, "oer"])
                m.c.add(m.usable_export_kw[l] <= m.p_des_pos_kw[l])

        # ---- objective (identical to the plain model, price as Param)
        benefit = -(5.0 / 60.0) * sum(
            m.price[l] * m.p_inj_oe_kw[l, "oer"] for l in partic)
        width = 0.001 * sum(
            m.p_inj_oe_kw[l, "oel"] - m.p_inj_oe_kw[l, "oer"] for l in partic)
        if self.usable_export_weight > 0:
            width = width - self.usable_export_weight * sum(
                m.usable_export_kw[l] for l in partic)
        viol = 1000.0 * sum(
            m.sof_bus_a_kw[i] + m.sof_bus_r_kw[i] for i in m.sof_bus_a_kw)
        if self.soft_limits:
            viol = viol + self.soft_limit_penalty * (
                sum(m.viol_v_pu[i] for i in m.viol_v_pu)
                + sum(m.viol_i_pu[i] for i in m.viol_i_pu))
        m.value = Objective(expr=benefit + width + viol, sense=minimize)

        # initial prices
        for l in partic:
            m.price[l] = float(self.partic_prices[l])

        # initial static tx limits
        for tx_id in tx_ids:
            m.tx_imax2[tx_id] = float(net.branches.at[tx_id, "i_max_pu"]) ** 2

    # ------------------------------------------------------------------
    def _make_solver(self, use_appsi):
        self._appsi = False
        if use_appsi is not False:
            try:
                from pyomo.contrib.appsi.solvers import Ipopt as AppsiIpopt
                s = AppsiIpopt()
                if s.available():
                    self._appsi = True
                    self.solver = s
                    opts = dict(_WARM_START_OPTS)
                    if self.quiet:
                        opts.update(print_level=0, sb="yes")
                    s.ipopt_options = opts
                    s.config.load_solution = False
                    logger.info("persistent solver: using appsi ipopt "
                                "(in-memory interface)")
                    self.interface = "appsi_ipopt"
                    return
            except Exception as e:            # pragma: no cover
                logger.info("appsi ipopt unavailable (%s); falling back", e)
        if use_appsi is True:
            raise RuntimeError("appsi ipopt requested but not available")
        self.solver = SolverFactory(self.solver_name)
        for k, v in _WARM_START_OPTS.items():
            self.solver.options[k] = v
        if self.quiet:
            self.solver.options["print_level"] = 0
            self.solver.options["sb"] = "yes"
        self.interface = "classic_" + self.solver_name
        logger.info("persistent solver: appsi not available — re-solving the "
                    "mutable-param model via the classic %s interface "
                    "(still skips model construction)", self.solver_name)

    # ------------------------------------------------------------------
    def solve_step(self, p_row_w, q_row_w, theta_A, price_row=None):
        """Solve one interval. p_row_w/q_row_w align with ``self.load_ids``.

        Returns a StepResult (status 'ok' or 'failed'); thermal state is
        carried internally and also returned.
        """
        m = self.model
        net = self.net
        tp = self.transformer_params

        # -- update load params
        for lid in self.forecast_load_ids:
            i = self._pos[lid]
            m.q_pu[lid] = float(q_row_w[i]) * _w_to_pu
            m.p_fixed_pu[lid] = float(p_row_w[i]) * _w_to_pu
        if self.usable_export_weight > 0:
            for lid in self.partic_load_ids:
                i = self._pos.get(lid)
                m.p_des_pos_kw[lid] = (max(-float(p_row_w[i]) / 1000.0, 0.0)
                                       if i is not None else 0.0)
        if price_row is not None:
            for lid in self.partic_load_ids:
                m.price[lid] = float(price_row.get(lid, 0.0)) \
                    if isinstance(price_row, dict) else float(price_row)

        # -- update transformer limit (the DTR, Phase 1)
        dtr_info = {}
        if self.tx_limit == "dtr":
            for tx_id, trow in net.transformers.iterrows():
                c2 = (trow["i_base_a"] / tp["I_rated"]) ** 2
                prev = self.thermal_state.get(tx_id, {})
                i_max_pu, status = _thermal.dtr_current_limit_pu(
                    tp, theta_A,
                    prev.get("delta_theta_TO", 0.0),
                    prev.get("delta_theta_HS", 0.0),
                    c2, k2_emergency=self.k2_emergency, k2_floor=self.k2_floor)
                m.tx_imax2[tx_id] = i_max_pu ** 2
                dtr_info[tx_id] = {"i_max_pu": i_max_pu,
                                   "K2_max": i_max_pu ** 2 * c2,
                                   "status": status, "c2": c2}

        # -- solve (previous solution left in .value's = warm start)
        try:
            if self._appsi:
                res = self.solver.solve(m)
                ok = str(res.termination_condition).endswith("optimal")
                if ok:
                    res.solution_loader.load_vars()
            else:
                res = self.solver.solve(m, tee=False)
                ok = res["Solver"][0].status in (
                    __import__("pyomo.opt", fromlist=["opt"]).SolverStatus.ok,)
        except (ApplicationError, ValueError, OSError, RuntimeError) as e:
            return StepResult("failed", None, None, None, None,
                              dict(self.thermal_state), dtr_info, [],
                              f"{type(e).__name__}: {e}")
        if not ok:
            return StepResult("failed", None, None, None, None,
                              dict(self.thermal_state), dtr_info, [],
                              f"termination={getattr(res, 'termination_condition', 'not ok')}")

        # -- extract
        doe = pd.DataFrame({
            "doe_lb_kw": [m.p_inj_oe_kw[l, "oel"].value for l in self.partic_load_ids],
            "doe_ub_kw": [m.p_inj_oe_kw[l, "oer"].value for l in self.partic_load_ids],
        }, index=pd.Index(self.partic_load_ids, name="load_id")).round(6)

        bus = pd.DataFrame({
            "voltage_pu_oel": [np.sqrt(m.square_voltage_pu[b, "oel"].value) for b in net.buses.index],
            "voltage_pu_oer": [np.sqrt(m.square_voltage_pu[b, "oer"].value) for b in net.buses.index],
        }, index=pd.Index(net.buses.index, name="id")).round(6)

        ib = net.branches["i_base_a"].values
        branch = pd.DataFrame({
            "current_a_oel": [np.sqrt(abs(m.square_current_pu[b, "oel"].value)) for b in net.branches.index] * np.ones(1),
            "current_a_oer": [np.sqrt(abs(m.square_current_pu[b, "oer"].value)) for b in net.branches.index] * np.ones(1),
            "p_w_oel": [m.branch_active_pu[b, "oel"].value * S_BASE_VA for b in net.branches.index],
            "p_w_oer": [m.branch_active_pu[b, "oer"].value * S_BASE_VA for b in net.branches.index],
            "q_va_oel": [m.branch_reactive_pu[b, "oel"].value * S_BASE_VA for b in net.branches.index],
            "q_va_oer": [m.branch_reactive_pu[b, "oer"].value * S_BASE_VA for b in net.branches.index],
        }, index=pd.Index(net.branches.index, name="id"))
        branch["current_a_oel"] *= ib
        branch["current_a_oer"] *= ib
        branch = branch.round(6)

        viol_rows = []
        for b in net.load_buses:
            for oe in _oe_idxs:
                va = sum(m.sof_bus_a_kw[b, oe, ci].value for ci in ("con", "inj"))
                vr = sum(m.sof_bus_r_kw[b, oe, ci].value for ci in ("con", "inj"))
                if va > 1e-6 or vr > 1e-6:
                    viol_rows.append({"id": b, "oe": oe,
                                      "viol_a_kw": round(va, 6),
                                      "viol_r_kw": round(vr, 6)})
        soft_viols = []
        if self.soft_limits:
            for (b, oe) in m.viol_v_pu:
                v = m.viol_v_pu[b, oe].value
                if v and v > 1e-6:
                    soft_viols.append({"kind": "voltage", "id": b, "oe": oe, "viol_pu": v})
            for (b, oe) in m.viol_i_pu:
                v = m.viol_i_pu[b, oe].value
                if v and v > 1e-6:
                    soft_viols.append({"kind": "current", "id": b, "oe": oe, "viol_pu": v})

        # -- advance thermal state ('oel' current, as the plain path)
        new_state = {}
        if tp is not None:
            for tx_id, trow in net.transformers.iterrows():
                c2 = (trow["i_base_a"] / tp["I_rated"]) ** 2
                prev = self.thermal_state.get(tx_id, {})
                k2 = m.square_current_pu[tx_id, "oel"].value * c2
                st = _thermal.forward_step(
                    tp, k2, theta_A,
                    prev.get("delta_theta_TO", 0.0),
                    prev.get("delta_theta_HS", 0.0))
                new_state[tx_id] = st
                if tx_id in dtr_info:
                    dtr_info[tx_id]["theta_HS"] = st["theta_HS"]
                    dtr_info[tx_id]["K2_actual"] = k2
            self.thermal_state = new_state

        return StepResult("ok", doe, bus, branch,
                          pd.DataFrame(viol_rows), dict(self.thermal_state),
                          dtr_info, soft_viols, None)
