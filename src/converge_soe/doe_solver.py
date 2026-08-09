from collections import namedtuple
import contextlib
import logging
import os
from pathlib import Path

from pyomo.environ import (
    ConcreteModel, ConstraintList, NonNegativeReals, Objective, minimize, pyomo, Reals, SolverFactory, Var
)
from pyomo.common.errors import ApplicationError
import numpy as np
import pandas as pd

from . import thermal as _thermal


logger = logging.getLogger(__name__)

#what goes into the solver?
#netw_json - dictionary describing the physical grid. 
#df_forecasts - for each customer, a df that describes each customer, what power they are expected to consume in the next 5-minute interval
#df_offers - df - which customers are willing to be dispatched and at what price?
# THE per-unit power base. Everything downstream derives from it: z_base,
# i_base, the kW/W conversions, and the branch-power reporting in scenarios.py
# and persistent_solver.py — which used to hardcode 1.0e6 in eight places, so
# changing this constant alone silently corrupted the thermal trajectory. They
# now import S_BASE_VA; change it HERE and only here.
#
# Why it matters numerically: LV customers draw kilowatts, so on a 1 MVA base
# the power variables sit at 1e-3 to 1e-6 pu. Combined with impedances from
# 5e-5 to 1.3 pu, the KKT matrix spans ~10 orders of magnitude and MUMPS
# struggles to factorise it ("EXIT: Error in step computation"). Lowering the
# base to 1e4 puts the power variables near 1. Re-run the sanity checks after
# any change: the DistFlow residual and the BAU energy balance will catch a
# mismatch immediately.
S_BASE_VA = 1.0e6
_s_base_va = S_BASE_VA

# Applied only to a RETRY, after a first attempt has already failed.
#   mu_strategy=adaptive     — vary the barrier parameter instead of the
#                              monotone schedule; much better on hard problems
#   bound_relax_factor       — let iterates sit slightly outside bounds rather
#                              than fighting an active bound with a huge
#                              multiplier
#   max_iter / max_cpu_time  — give the harder path room, but still bounded
#   honor_original_bounds=no — do not project the answer back onto the bounds
# (linear_system_scaling=mc19 would help too, but MC19 is part of HSL and is
#  not available in a MUMPS-only ipopt build.)
# NOTE: no max_cpu_time here. It used to carry 180 s, which OVERRODE a larger
# configured cap — so a retry after a 600 s timeout got less time than the
# attempt that had just timed out, guaranteeing a second timeout and burning
# 780 s per interval to reach the same answer. The retry inherits the
# configured cap instead.
_ROBUST_OPTIONS = {
    "mu_strategy": "adaptive",
    "bound_relax_factor": 1e-6,
    "honor_original_bounds": "no",
    "max_iter": 5000,
}
_kw_to_pu = 1000.0 / _s_base_va  # kW to pu
_w_to_pu = 1.0 / _s_base_va  # kW to pu
_pu_to_w = _s_base_va  # kW to pu

_oe_idxs = ['oel', 'oer']  # Operating envelopes: oe left (min injection) / oe right (max injection)
_ci_idxs = ['con', 'inj']  # Consumption or injection


class OutputLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.log(self.level, msg)

    def flush(self):
        pass

#when you create an SoeSolver, it runs filtering of input data, converts the network into clean pandas df, and builds the optimisation using Pyomo.
## this is a python dictionary describing the physical network, buses, lines, transformers and loads. 
class SoeSolver:
    def __init__(self, netw_ejson: dict, df_forecasts: pd.DataFrame,
                 envelope_abs_max=50.0, participant_load_ids=None,
                 df_prices: pd.DataFrame = None,
                 transformer_params: dict = None,
                 theta_A: float = 20.0, thermal_state_in: dict = None,
                 solver_options: dict = {},
                 tx_limit: str = "legacy",
                 k_emergency: float = 2.0, k2_floor: float = 1e-6,
                 soft_limits: bool = False, soft_limit_penalty: float = 1000.0,
                 quiet: bool = False, warm_start_in: dict = None,
                 solver_name: str = "ipopt", network_cache=None,
                 usable_export_weight: float = 0.0,
                 max_cpu_time: float = 120.0,
                 retry_on_failure: bool = True,
                 series_reduction: bool = False):
        self.netw_ejson = netw_ejson
        self.netw_ejson["components"] = dict(sorted(self.netw_ejson["components"].items()))  # For reprod/testing
        # pandas df with forecast power consumption for each customer.
        self.df_forecasts = df_forecasts
        # maximum envelope width
        self.envelope_abs_max = envelope_abs_max
        # None = every forecast load gets an envelope; otherwise only the supplied IDs do.
        self.participant_load_ids_input = participant_load_ids
        # Optional price table for participants: index=load_id, column 'export_price_kwh'.
        # Missing participants default to price 0 (no economic signal for that load).
        self.df_prices = df_prices
        # IEEE C57.91 transformer thermal model parameters (optional).
        # Keys: tau_TO, tau_W, delta_theta_TO_R, delta_theta_HS_R, R, n, m, I_rated, theta_HS_max, dt
        self.transformer_params = transformer_params
        # Ambient temperature (°C) for the current time step.
        self.theta_A = theta_A
        # Thermal state from the previous period, keyed by transformer branch ID.
        # Each value is a dict with 'delta_theta_TO' and 'delta_theta_HS' (°C).
        # Defaults to cold start (0 °C rises) for any transformer not present.
        self.thermal_state_in = thermal_state_in if thermal_state_in is not None else {}
        # Transformer current-limit mode:
        #   "legacy" — today's combined behaviour: static s_max-derived i_max_pu
        #              PLUS the nonlinear C57.91 hot-spot constraint (kept only
        #              so the Phase-9 equivalence tests have a reference).
        #   "dtr"    — the transformer i_max_pu is REPLACED by the inverted
        #              C57.91 dynamic thermal rating for this timestep, and the
        #              nonlinear hot-spot constraint is removed entirely.
        #   "static" — s_max-derived i_max_pu, no thermal constraint in the
        #              optimisation (thermal state still tracked post-hoc).
        if tx_limit not in ("legacy", "dtr", "static"):
            raise ValueError(f"tx_limit must be legacy|dtr|static, got {tx_limit!r}")
        if tx_limit == "dtr" and transformer_params is None:
            raise ValueError("tx_limit='dtr' requires transformer_params")
        self.tx_limit = tx_limit
        # Emergency backstop: bushings/tap changers/LV cabling cap the rating at
        # K_emergency x rated current no matter how cold the oil is.
        self.k_emergency = float(k_emergency)
        self.k2_floor = float(k2_floor)
        # Optional penalised slacks on voltage and current limits (MDL001): an
        # infeasible interval becomes a quantified violation instead of a
        # missing row. OFF by default for backwards compatibility.
        self.soft_limits = bool(soft_limits)
        self.soft_limit_penalty = float(soft_limit_penalty)
        # Silence ipopt (Phase 3): print_level=0, no stdout redirection.
        self.quiet = bool(quiet)
        # Optional warm start: dict of {var_name: {index: value}} from the
        # previous timestep's solution (see warm_start_values()).
        self.warm_start_in = warm_start_in or {}
        # Per-transformer DTR record for this timestep (populated in
        # _build_network_data when tx_limit == "dtr").
        self.dtr_info = {}
        self.solver_name = solver_name
        # Weight ($/kWh-ish) on the "usable export" reward: an auxiliary
        # variable u_i = min(doe_ub_i, max(P_des_i, 0)) is rewarded in the
        # objective, so envelope headroom is allocated to customers who can
        # actually use it. With weight 0 (the default, and the legacy
        # behaviour) the per-NMI allocation of surplus network capacity is
        # DEGENERATE — mathematically any split is optimal, so curtailment
        # metrics become arbitrary. The pipeline scenarios enable this.
        self.usable_export_weight = float(usable_export_weight)
        # Wall-clock cap per solve, and one retry with a more robust barrier
        # configuration when the first attempt fails. See _ROBUST_OPTIONS.
        self.max_cpu_time = float(max_cpu_time)
        self.retry_on_failure = bool(retry_on_failure)
        self.n_retries_used = 0
        # Merge line chains through nodes carrying no load. OFF by default so
        # enabling it is always a deliberate, recorded choice — see
        # _reduce_series_branches.
        self.series_reduction = bool(series_reduction)
        # Optional pre-parsed network data (Path A of the speed work): the
        # network is constant for a given substation, so the ejson parse and
        # per-unit conversion can be done once and reused for every timestep.
        # Obtain one from a first solver instance via .network_cache().
        self._network_cache_in = network_cache
        # Solver options defaults
        self.solver_options = {'print_level': 3, 'linear_solver': 'mumps'}

        # User supplied solver options
        for k, v in solver_options.items():
            self.solver_options[k] = v

        self.rebuild()
#when you get input data, do this:
    def rebuild(self):
        self._filter_input_data()
        self._build_network_data()
        self._build_opt_model()

    def dump_opt_model(self, filename):
        with open(filename, 'w+') as f:
            self.model.pprint(f, True)

    def solve(self):
        status = self._solve_opt_model()
        res = self._extract_results() if status in (pyomo.opt.SolverStatus.ok, pyomo.opt.SolverStatus.warning) else None
        return status, res

    def _filter_input_data(self):
        #only keep customers that arein both the network and the forecast/offer data.
        loads = _netw_components(self.netw_ejson, "Load")
        self.netw_load_ids_set = set(x[0] for x in loads)# setting the base values for all loads in the network

        self.forecast_load_ids = sorted(set.intersection(
            self.netw_load_ids_set, set(self.df_forecasts.index)
        ))
        self.df_forecasts_filt = self.df_forecasts.reindex(self.forecast_load_ids)

        if self.participant_load_ids_input is None:
            self.partic_load_ids = self.forecast_load_ids
        else:
            self.partic_load_ids = sorted(
                set(self.participant_load_ids_input) & self.netw_load_ids_set
            )

        # Build a price Series indexed by partic_load_ids; missing entries default to 0.
        if self.df_prices is not None:
            self.partic_prices = (
                self.df_prices['export_price_kwh']
                .reindex(self.partic_load_ids, fill_value=0.0)
            )
        else:
            self.partic_prices = pd.Series(0.0, index=self.partic_load_ids)

    def network_cache(self):
        '''Reusable parsed network data for subsequent DoeSolver instances.

        Pass as ``network_cache=`` to skip the ejson parse and per-unit
        conversion on every timestep (the network is constant per substation).
        '''
        return {
            "buses": self.buses,
            "branches": self._branches_pristine,
            "loads": self.loads,
            "load_buses": self.load_buses,
            "transformer_buses": self.transformer_buses,
            "downstream": self._downstream,
        }

    def _apply_cached_network(self, cache):
        self.buses = cache["buses"]
        # branches are copied because DTR mode overwrites transformer i_max_pu
        # per timestep; everything else is shared read-only.
        self._branches_pristine = cache["branches"]
        self.branches = cache["branches"].copy()
        self.loads = cache["loads"]
        self.load_buses = cache["load_buses"]
        self.transformer_buses = cache["transformer_buses"]
        self._downstream = cache["downstream"]

    def _build_network_data(self):
        #builds a dataframe for each bus, branch and loads. Converts everything to pu terms.
        if self._network_cache_in is not None:
            self._apply_cached_network(self._network_cache_in)
            self._apply_dtr_limits()
            sel_lines = self.branches["voltage_ratio_pu"].isna()
            self.lines = self.branches.loc[sel_lines]
            self.transformers = self.branches.loc[~sel_lines]
            return
        
        v_units = self.netw_ejson["units"]["voltage"]
        i_units = self.netw_ejson["units"]["current"]
        s_units = self.netw_ejson["units"]["power"]
        z_units = self.netw_ejson["units"]["impedance"]

        ej_infeeders = {cid: (ctp, cd) for cid, ctp, cd in _netw_components(self.netw_ejson, "Infeeder")}
        ej_nodes = {cid: (ctp, cd) for cid, ctp, cd in _netw_components(self.netw_ejson, "Node")}
        ej_lines = {cid: (ctp, cd) for cid, ctp, cd in _netw_components(self.netw_ejson, "Line")}
        ej_txs = {cid: (ctp, cd) for cid, ctp, cd in _netw_components(self.netw_ejson, "Transformer")}
        ej_loads = {cid: (ctp, cd) for cid, ctp, cd in _netw_components(self.netw_ejson, "Load")}

        def add_i(df):
            df["i"] = list(range(len(df)))

        buses_list = []
        for cid, (_, cd) in ej_nodes.items():
            try:
                v_mag_min_pu = cd["user_data"]["v_min"] / cd["v_base"]
            except KeyError:
                v_mag_min_pu = np.nan

            try:
                v_mag_max_pu = cd["user_data"]["v_max"] / cd["v_base"]
            except KeyError:
                v_mag_max_pu = np.nan

            buses_list.append(
                {
                    "id": cid,
                    "v_base_v": cd["v_base"] * v_units,
                    "v_mag_min_pu": v_mag_min_pu,
                    "v_mag_max_pu": v_mag_max_pu,
                    "v_mag_setpoint_pu": np.nan,
                }
            )

        self.buses = pd.DataFrame.from_records(buses_list).set_index("id")
        add_i(self.buses)

        for cid, (_, cd) in ej_infeeders.items():
            nd_id = cd["cons"][0]["node"]
            _, nd_dict = ej_nodes[nd_id]
            self.buses.loc[nd_id, "v_mag_setpoint_pu"] = cd["v_setpoint"] / nd_dict["v_base"]

        loads_list = []
        for cid, (_, cd) in ej_loads.items():
            nd_id = cd["cons"][0]["node"]
            loads_list.append(
                {
                    "load_id": cid,
                    "bus_id": nd_id,
                }
            )

        self.loads = pd.DataFrame.from_records(loads_list).set_index("load_id")
        add_i(self.loads)

        self.load_buses = list(dict.fromkeys(self.loads["bus_id"]))

        branches_list = []

        for cid, (_, cd) in ej_lines.items():
            nd_id_0 = cd["cons"][0]["node"]
            nd_id_1 = cd["cons"][1]["node"]
            length = cd["length"]
            z_ohm = cd["z"] * z_units
            z0_ohm = cd["z0"] * z_units
            branches_list.append(
                {
                    "id": cid,
                    "from_bus_id": nd_id_0,
                    "to_bus_id": nd_id_1,
                    "r_ohm": z_ohm[0] * length,
                    "x_ohm": z_ohm[1] * length,
                    "r0_ohm": z0_ohm[0] * length,
                    "x0_ohm": z0_ohm[1] * length,
                    # No rating in the export -> NO current constraint on this
                    # line (the constraint below is only added where i_max_pu
                    # is not NaN). The old 100 kA placeholder was equivalent
                    # physically — it never bound — but it was actively harmful
                    # numerically: it put i_max_pu = 40 alongside real limits of
                    # 0.04, a 1000x spread in the constraint matrix, and added
                    # hundreds of constraints per substation that could never
                    # be active. Preflight NET009 still reports the missing
                    # ratings, so the modelling gap stays visible.
                    "i_max_a": cd["i_max"] * i_units if "i_max" in cd else np.nan,
                    "transformer_bus_id": None,
                    # transformer_bus_id is just to_bus for transformers, but keep this for clarity / to make things
                    # more similar to previous code.
                    "voltage_ratio_pu": np.nan,
                }
            )

        self.transformer_buses = set()
        for cid, (_, cd) in ej_txs.items():
            nd_id_0 = cd["cons"][0]["node"]
            nd_id_1 = cd["cons"][1]["node"]
            self.transformer_buses.add(nd_id_1)
            z_ohm = cd["z"] * z_units
            z0_ohm = cd["z"] * z_units
            s_max_w = cd["s_max"] * s_units if "s_max" in cd else 1e9  # Default to large!
            i_max_a = s_max_w / (cd["v_winding_base"][1] * v_units)  # Max secondary current

            nom_tr = cd['nom_turns_ratio'][0]
            off_nom_tr = \
                1.0 + cd["taps"][0] * cd["tap_factor"] if cd["tap_side"] == "primary" else \
                1.0 / (1 + cd["taps"][0] * cd["tap_factor"])
            tr = nom_tr * off_nom_tr

            vg = cd['vector_group']
            assert vg[0] == vg[1]
            vr = tr

            nd_0 = ej_nodes[nd_id_0][1]
            nd_1 = ej_nodes[nd_id_1][1]

            vr_pu = vr * nd_1['v_base'] / nd_0['v_base']

            branches_list.append(
                {
                    "id": cid,
                    "from_bus_id": nd_id_0,
                    "to_bus_id": nd_id_1,
                    "transformer_bus_id": nd_id_1,  # Repeated, but leave for clarity wrt older code.
                    "r_ohm": z_ohm[1][0],  # TODO: Properly determine impedance side etc.
                    "x_ohm": z_ohm[1][1],  # TODO: Properly determine impedance side etc.
                    "r0_ohm": 0.0,
                    "x0_ohm": 0.0,
                    "i_max_a": i_max_a,
                    "voltage_ratio_pu": vr_pu,
                }
            )

        self.branches = pd.DataFrame.from_records(branches_list).set_index("id")
        self._orient_branches_from_source(ej_nodes)
        # getattr, not self.series_reduction: PersistentDoeSolver builds its
        # base via __new__ and sets attributes by hand, so anything added to
        # __init__ is absent there unless it is also set in persistent_solver.
        # Defaulting here keeps --fast working even if that list drifts.
        if getattr(self, "series_reduction", False):
            self._reduce_series_branches()
        add_i(self.branches)

        self.transformer_buses = list(self.transformer_buses)

        self.buses["is_transformer"] = False
        self.buses.loc[self.transformer_buses, "is_transformer"] = True

        def build_branch_base_and_pu(df):
            # Line v_base is defined as v_base of the to bus.
            df["v_base_v"] = self.buses.loc[df["to_bus_id"], "v_base_v"].tolist()
            df["z_base_ohm"] = df["v_base_v"].pow(2) / _s_base_va
            df["i_base_a"] = _s_base_va / df["v_base_v"]

            # TODO: following is the original code, but probably the 1/3 r0 + 2 r thing has already been taken care of
            # in converting model to single phase. Hopefully won"t make any difference and leaving code for now.
            df["r_pu"] = (1 / 3.0) * (df["r0_ohm"] + 2 * df["r_ohm"]) / df["z_base_ohm"]
            df["x_pu"] = (1 / 3.0) * (df["x0_ohm"] + 2 * df["x_ohm"]) / df["z_base_ohm"]

            sel_has_imax = pd.notna(df["i_max_a"])
            df.loc[sel_has_imax, "i_max_pu"] = df.loc[sel_has_imax, "i_max_a"] / df["i_base_a"]
            df.loc[~sel_has_imax, "i_max_pu"] = np.nan

        build_branch_base_and_pu(self.branches)

        # Keep a pristine copy (pre-DTR) plus the downstream-branch map for
        # network_cache() reuse across timesteps (Path A of the speed work).
        self._branches_pristine = self.branches.copy()
        self._downstream = {}
        for b_id, from_bus in self.branches["from_bus_id"].items():
            self._downstream.setdefault(from_bus, []).append(b_id)

        self._apply_dtr_limits()

        sel_lines = self.branches["voltage_ratio_pu"].isna()
        self.lines = self.branches.loc[sel_lines]
        self.transformers = self.branches.loc[~sel_lines]

    def _orient_branches_from_source(self, ej_nodes):
        '''Orient every branch away from the infeeder (from_bus = upstream).

        THIS IS LOAD-BEARING. The DistFlow balance is

            branch_active_pu[b] == sum(a_bus_pu[b.to_bus]) + r*i^2
                                   + sum(branch_active_pu[downstream of b.to_bus])

        which assumes each branch's ``to_bus`` is its DOWNSTREAM end. CIM
        exports carry arbitrary edge orientation: on GOLDCR_8HB_LEXCEN,
        152 of 306 branches pointed back towards the source, so 58 of 59 load
        buses were never any branch's ``to_bus``. Their power contributed to no
        balance at all and simply vanished — the model solved happily with
        ~0 A everywhere (the i^2*v^2 == P^2+Q^2 equality was satisfied because
        both sides were zero), every envelope went to the ``envelope_abs_max``
        cap unopposed, the transformer never heated, and doe_dtr was always
        identical to doe_static.

        BAU was never affected: ``scenarios.run_bau_scenario`` builds its own
        undirected adjacency and does a proper rooted traversal.

        Swapping a transformer's ends also inverts its voltage ratio, and the
        LV side (``transformer_bus_id``) follows the new ``to_bus``.
        '''
        from collections import defaultdict, deque

        srcs = [c[0]["cons"][0]["node"]
                for _, ctp, c in [(k, t, [d]) for k, t, d in
                                  _netw_components(self.netw_ejson, "Infeeder")]]
        if not srcs:
            logger.warning("no Infeeder: branch orientation left as imported; "
                           "the power balance may be wrong")
            return
        root = srcs[0]

        adj = defaultdict(list)
        for b_id, r in self.branches.iterrows():
            adj[r["from_bus_id"]].append((b_id, r["to_bus_id"]))
            adj[r["to_bus_id"]].append((b_id, r["from_bus_id"]))

        seen, flipped, visited = {root}, [], set()
        q = deque([root])
        while q:
            n = q.popleft()
            for b_id, m in adj[n]:
                if b_id in visited:
                    continue
                visited.add(b_id)
                if m in seen:        # closes a loop — not a radial tree
                    continue
                seen.add(m)
                q.append(m)
                if self.branches.at[b_id, "from_bus_id"] != n:
                    flipped.append(b_id)

        for b_id in flipped:
            row = self.branches.loc[b_id]
            self.branches.at[b_id, "from_bus_id"] = row["to_bus_id"]
            self.branches.at[b_id, "to_bus_id"] = row["from_bus_id"]
            vr = row["voltage_ratio_pu"]
            if pd.notna(vr) and vr != 0:
                self.branches.at[b_id, "voltage_ratio_pu"] = 1.0 / vr
                self.branches.at[b_id, "transformer_bus_id"] = row["from_bus_id"]

        unreached = len(self.buses) - len(seen)
        if flipped and not self.quiet:
            logger.info("oriented %d/%d branch(es) away from the source",
                        len(flipped), len(self.branches))
        if unreached:
            logger.warning(
                "%d bus(es) not reachable from the infeeder — their loads "
                "contribute to no branch balance and will be ignored by the "
                "power flow", unreached)

    def _reduce_series_branches(self):
        '''Merge chains of lines through nodes that carry nothing.

        A distribution LV model contains far more nodes than customers: poles,
        joints, cable-type changes. The largest substation measured here has
        950 branches for 148 customers, against 397 for 96 on a smaller one —
        2.4x the network for a comparable number of loads. The NLP scales
        superlinearly, so that turned ~12 s per interval into ~200 s.

        A node with exactly two line branches, no load, no transformer and no
        voltage setpoint injects nothing, so the two branches carry IDENTICAL
        current and can be replaced by one with

            r = r1 + r2,   x = x1 + x2,   i_max = min(i_max1, i_max2)

        This is exact, not an approximation: with no injection the voltage
        along the chain is monotonic in the flow, so the extreme voltage is at
        an endpoint and dropping the interior node's limit cannot hide a
        violation. The current limit is the tighter of the two because the same
        current passes through both.

        Preserved and never merged through: load buses, transformer terminals,
        the infeeder/setpoint bus, and any node joining three or more branches.
        '''
        from collections import defaultdict

        keep = set(self.load_buses) | set(self.transformer_buses)
        keep |= set(self.buses.index[self.buses["v_mag_setpoint_pu"].notna()])

        n_before = len(self.branches)
        merged_total = 0

        while True:
            inc = defaultdict(list)
            for b_id, r in self.branches.iterrows():
                inc[r["from_bus_id"]].append(b_id)
                inc[r["to_bus_id"]].append(b_id)

            merged_this_pass = 0
            done = set()
            for node, bids in inc.items():
                if node in keep or len(bids) != 2:
                    continue
                b1, b2 = bids
                if b1 in done or b2 in done or b1 == b2:
                    continue
                r1 = self.branches.loc[b1]
                r2 = self.branches.loc[b2]
                # lines only — a transformer carries a voltage ratio
                if pd.notna(r1["voltage_ratio_pu"]) or pd.notna(r2["voltage_ratio_pu"]):
                    continue
                # orient so the chain reads up -> node -> down
                if r1["to_bus_id"] == node and r2["from_bus_id"] == node:
                    up, dn = b1, b2
                elif r2["to_bus_id"] == node and r1["from_bus_id"] == node:
                    up, dn = b2, b1
                else:
                    continue    # not a clean series pair

                u, d = self.branches.at[up, "from_bus_id"], self.branches.at[dn, "to_bus_id"]
                if u == d:
                    continue

                for col in ("r_ohm", "x_ohm", "r0_ohm", "x0_ohm"):
                    if col in self.branches.columns:
                        self.branches.at[up, col] = (
                            self.branches.at[up, col] + self.branches.at[dn, col])
                i1, i2 = self.branches.at[up, "i_max_a"], self.branches.at[dn, "i_max_a"]
                self.branches.at[up, "i_max_a"] = np.nanmin([i1, i2])
                self.branches.at[up, "to_bus_id"] = d

                self.branches = self.branches.drop(index=dn)
                done.update((up, dn))
                merged_this_pass += 1

            merged_total += merged_this_pass
            if not merged_this_pass:
                break

        if merged_total:
            gone = set(self.buses.index) - (
                set(self.branches["from_bus_id"]) | set(self.branches["to_bus_id"]))
            self.buses = self.buses.drop(index=[b for b in gone if b not in keep])
            if not self.quiet:
                logger.info("series reduction: %d -> %d branches (%d merged), "
                            "%d buses remain", n_before, len(self.branches),
                            merged_total, len(self.buses))

    def _apply_dtr_limits(self):
        # Dynamic thermal rating (Phase 1): replace each transformer's static
        # nameplate i_max_pu with the limit obtained by inverting the C57.91
        # model for THIS timestep's thermal state and ambient temperature.
        # The nonlinear hot-spot constraint is then omitted from the model
        # entirely — the transformer limit is a simple bound.
        if self.tx_limit == "dtr" and self.transformer_params is not None:
            tp = self.transformer_params
            k2_emergency = self.k_emergency ** 2
            sel_tx = self.branches["voltage_ratio_pu"].notna()
            for tx_id in self.branches.index[sel_tx]:
                i_base_a = self.branches.at[tx_id, "i_base_a"]
                c2 = (i_base_a / tp["I_rated"]) ** 2
                prev = self.thermal_state_in.get(tx_id, {})
                dTO_0 = prev.get("delta_theta_TO", 0.0)
                dHS_0 = prev.get("delta_theta_HS", 0.0)
                i_max_pu, status = _thermal.dtr_current_limit_pu(
                    tp, self.theta_A, dTO_0, dHS_0, c2,
                    k2_emergency=k2_emergency, k2_floor=self.k2_floor,
                )
                self.branches.at[tx_id, "i_max_pu"] = i_max_pu
                self.branches.at[tx_id, "i_max_a"] = i_max_pu * i_base_a
                self.dtr_info[tx_id] = {
                    "i_max_pu": i_max_pu,
                    "K2_max": i_max_pu ** 2 * c2,
                    "status": status,
                    "c2": c2,
                }

    def _build_opt_model(self):
        #uses pyomo to build the actual optimization model. This includes:
        #variables: the unknowns the solver will find: envelope bounds, voltages, currents, power flows, network support dispatch. 
        #constraints - rules the solution must satisfy, eg. basic electronics principles. 
        #objectives- minimise cost of network support dispatches, with a small push towards wider envelopes and a large penalty for any constraint violations. 

        self.model = ConcreteModel()  # pyomo

        # Index sets -------------------------------------------------------------------------------------------------

        bus_idxs = self.buses.index
        branch_idxs = self.branches.index
        partic_idxs = self.partic_load_ids
        busld_idxs = self.load_buses

        bus_oe_idxs = [(bus_id, oe) for bus_id in bus_idxs for oe in _oe_idxs]
        branch_oe_idxs = [(branch_id, oe) for branch_id in branch_idxs for oe in _oe_idxs]
        partic_oe_idxs = [(load_id, oe) for load_id in partic_idxs for oe in _oe_idxs]
        partic_ci_idxs = [(load_id, ci) for load_id in partic_idxs for ci in _ci_idxs]
        busld_oe_ci_idxs = [(bus_id, oe, ci) for bus_id in busld_idxs for oe in _oe_idxs for ci in _ci_idxs]
        #sets up all of the indexes for each bus, node and branch. 

        # Calculate local active and reactive background load at each bus.
        # For participant NMIs, we don't include the active power forecast, as the active power will be
        # treated separately as the envelope limits.
        #don't double count participant loads as their power is already handled by the envelope variables, so leave it out of the background calculation
        bus_ld_a_kw, bus_ld_r_kw = self._calculate_bus_loads_kw(bus_idxs)

        # Variables --------------------------------------------------------------------------------------------------

        # Network variables.
        # Bounds (Phase 1.4): physically generous, numerically essential —
        # square_voltage_pu in [0.5^2, 1.5^2] and square_current_pu >= 1e-8
        # keep ipopt away from the pow(0, 0.8)-style singularities.
        # Initial values come from the previous timestep's solution when the
        # caller passes warm_start_in (consecutive intervals are similar), and
        # fall back to flat-start values otherwise.
        ws = self.warm_start_in

        def init_from(ws_key, fallback, lo=None, hi=None):
            """Initialiser from the previous solve, clamped into bounds.

            A converged square_current_pu is routinely ~1e-32 — numerically
            zero, but below the 1e-8 lower bound. Feeding it back verbatim made
            Pyomo emit a W1002 'outside the bounds' warning per variable per
            timestep (thousands per run) for what is a harmless rounding
            artefact. Clamping keeps the warm start while staying feasible.
            """
            prev = ws.get(ws_key) or {}

            def _init(m, *idx):
                v = prev.get(idx if len(idx) > 1 else idx[0], fallback)
                if lo is not None and v < lo:
                    return lo
                if hi is not None and v > hi:
                    return hi
                return v
            return _init

        self.model.square_voltage_pu = Var(
            bus_oe_idxs, name="square_voltage_pu", domain=NonNegativeReals,
            bounds=(0.5 ** 2, 1.5 ** 2),
            initialize=init_from("square_voltage_pu", 1.0,
                                 lo=0.5 ** 2, hi=1.5 ** 2)
        )

        # Lower bound on i^2. The 1e-8 floor exists to keep ipopt away from
        # (K2 + eps)**m with m ~ 0.8, whose derivative is unbounded at K2 = 0 —
        # but that term ONLY appears in the legacy thermal constraint, and it
        # already carries its own eps = 1e-6 guard. In dtr/static mode there is
        # no such term, and the floor is actively harmful: a lightly-loaded
        # branch genuinely wants i^2 ~ 1e-32, so the bound becomes active with
        # a huge multiplier and wrecks the conditioning of the KKT system.
        # (This is what the W1002 flood was telling us.) Zero is safe: the
        # variable is NonNegativeReals and the i^2*v^2 == P^2+Q^2 equality
        # pins it.
        _i2_lo = 1e-8 if self.tx_limit == "legacy" else 0.0
        self.model.square_current_pu = Var(
            branch_oe_idxs, name="square_current_pu", domain=NonNegativeReals,
            bounds=(_i2_lo, None),
            initialize=init_from("square_current_pu", max(_i2_lo, 1e-12),
                                 lo=_i2_lo)
        )
        self.model.branch_active_pu = Var(
            branch_oe_idxs, name="branch_active_pu", domain=Reals,
            initialize=init_from("branch_active_pu", 0.0)
        )
        self.model.branch_reactive_pu = Var(
            branch_oe_idxs, name="branch_reactive_pu", domain=Reals,
            initialize=init_from("branch_reactive_pu", 0.0)
        )

        # Operating envelope variables. These are power *injections*.
        def oe_bounds(m, load_id, oe): 
            #constrains how wide the bounds can be: can be between the minimum and 0 for import, and 0 and upper bound for export. 
            return (-self.envelope_abs_max, 0.0) if oe == 'oel' else (0.0, self.envelope_abs_max)
        # this is a variable for every combination of customer and scenario. this gives two variables per customer, one with the lower envelope bound and one the upper. 
        #This just combines them 
        self.model.p_inj_oe_kw = Var(
            partic_oe_idxs, name="p_inj_oe_kw", domain=Reals,
            bounds=oe_bounds,
            initialize=0.0
        )

        # Network support: also an optimization of customers bidding for network support. 
        # two variables for each participant: how much more to consume (above what they're currently doing) and how much more to inject.both always >=0.
        self.model.network_support_kw = Var(
            partic_ci_idxs, name="network_support_kw", domain=NonNegativeReals, initialize=0.0
        )

        def init_sof_a(m, bus_id, oe, ci):
            #initialisation variables for the soft slack variables. Pyomo lets you pass a function as initialise instead of a fixed number, so each customer can have a different guess. 
            p = bus_ld_a_kw[bus_id]
            if p > 0 and ci == 'inj':
                return p
            elif p < 0 and ci == 'con':
                return -p

            return 0.0

        def init_sof_r(m, bus_id, oe, ci):
            q = bus_ld_r_kw[bus_id]
            if q > 0 and ci == 'inj':
                return q
            elif q < 0 and ci == 'con':
                return -q

            return 0.0

        self.model.sof_bus_a_kw = Var(busld_oe_ci_idxs, name="sof_bus_a_kw", domain=NonNegativeReals,
                                      initialize=init_sof_a)
        #if a bus has positive background load, initialise the injuection slack. If not, initialise consumption slack. 
        self.model.sof_bus_r_kw = Var(busld_oe_ci_idxs, name="sof_bus_r_kw", domain=NonNegativeReals,
                                      initialize=init_sof_r)
        #same thing but for reactive power
        
        # Allocation of loads in buses

        a_bus_pu = {(bus_id, oe): [] for bus_id in bus_idxs for oe in _oe_idxs}  # Active power
        r_bus_pu = {(bus_id, oe): [] for bus_id in bus_idxs for oe in _oe_idxs}  # Active power

        for load_row in self.loads.itertuples():
            load_id = load_row.Index
            bus_id = load_row.bus_id

            if load_id in self.partic_load_ids:
                reactive_power = self.df_forecasts_filt.loc[load_id, "reactive_power_var"] * _w_to_pu
                for oe in _oe_idxs:
                    r_bus_pu[bus_id, oe].append(reactive_power)
                    a_bus_pu[(bus_id, oe)].append(-self.model.p_inj_oe_kw[load_id, oe] * _kw_to_pu)
            elif load_id in self.forecast_load_ids:
                reactive_power = self.df_forecasts_filt.loc[load_id, "reactive_power_var"] * _w_to_pu
                active_power = self.df_forecasts_filt.loc[load_id, "real_power_w"] * _w_to_pu
                for oe in _oe_idxs:
                    r_bus_pu[bus_id, oe].append(reactive_power)
                    a_bus_pu[(bus_id, oe)].append(active_power)

        # Allocation of soft variables
        #for each bus that has loads, and for each min and max envelope limit, it adds the net slack power at that bus. 
        #this accounts for if there is no combination of envelope settings that can satisfy all voltage and current limits simultaneously.
        #avoids a no solution found error. 
        for bus_id in self.load_buses:
            for oe in _oe_idxs:
                a_bus_pu[(bus_id, oe)].append(
                    (self.model.sof_bus_a_kw[bus_id, oe, 'con']-self.model.sof_bus_a_kw[bus_id, oe, 'inj']) * _kw_to_pu
                )
                r_bus_pu[(bus_id, oe)].append(
                    (self.model.sof_bus_r_kw[bus_id, oe, 'con']-self.model.sof_bus_r_kw[bus_id, oe, 'inj']) * _kw_to_pu
                )

        # Constraints ------------------------------------------------------------------------------------------------

        self.model.c = ConstraintList()

        # Optional slacks on the (otherwise hard) voltage and current limits
        # (MDL001). With soft limits an over-limit interval solves anyway and
        # the violation magnitude is reported, instead of the timestep silently
        # disappearing as "infeasible".
        if self.soft_limits:
            self.model.viol_v_pu = Var(
                bus_oe_idxs, name="viol_v_pu", domain=NonNegativeReals, initialize=0.0
            )
            self.model.viol_i_pu = Var(
                branch_oe_idxs, name="viol_i_pu", domain=NonNegativeReals, initialize=0.0
            )

        # Power flow constraints

        # Voltage
        # for each node, voltage has to be within limits. Infeeder has a fixed setpoint. Every other bus has to stay within the min and max values. 
        for oe in _oe_idxs:
            for bus_row in self.buses.itertuples():
                bus_id = bus_row.Index
                if pd.notna(bus_row.v_mag_setpoint_pu):
                    self.model.c.add(
                        self.model.square_voltage_pu[bus_id, oe] == pow(bus_row.v_mag_setpoint_pu, 2)
                    )
                else:
                    if self.soft_limits:
                        if pd.notna(bus_row.v_mag_max_pu):
                            self.model.c.add(
                                self.model.square_voltage_pu[bus_id, oe]
                                <= bus_row.v_mag_max_pu**2 + self.model.viol_v_pu[bus_id, oe]
                            )
                        if pd.notna(bus_row.v_mag_min_pu):
                            self.model.c.add(
                                self.model.square_voltage_pu[bus_id, oe]
                                >= bus_row.v_mag_min_pu**2 - self.model.viol_v_pu[bus_id, oe]
                            )
                    else:
                        if pd.notna(bus_row.v_mag_max_pu):
                            self.model.c.add(self.model.square_voltage_pu[bus_id, oe] <= bus_row.v_mag_max_pu**2)

                        if pd.notna(bus_row.v_mag_min_pu):
                            self.model.c.add(self.model.square_voltage_pu[bus_id, oe] >= bus_row.v_mag_min_pu**2)

        # Power flow
        # power in = power out. 
        for oe in _oe_idxs:
            for branch_row in self.branches.itertuples():
                branch_id = branch_row.Index
                to_bus_id = branch_row.to_bus_id
                from_bus_id = branch_row.from_bus_id

                # Line active is active injection into branch.
                # (precomputed once — the old per-branch DataFrame scan was
                # O(B²) across the whole constraint build)
                downstream_branch_ids = self._downstream.get(to_bus_id, [])
                # active power into a branch equals load at the destination plus resistive losses plus flows onto downstream branches. 
                self.model.c.add(
                    self.model.branch_active_pu[branch_id, oe] == sum(a_bus_pu[to_bus_id, oe]) +
                    branch_row.r_pu * self.model.square_current_pu[branch_id, oe]
                    + sum(self.model.branch_active_pu[bid, oe] for bid in downstream_branch_ids)
                )
                #for resistive and inductive losses. 
                self.model.c.add(
                    self.model.branch_reactive_pu[branch_id, oe] == sum(r_bus_pu[to_bus_id, oe]) +
                    branch_row.x_pu * self.model.square_current_pu[branch_id, oe]
                    + sum(self.model.branch_reactive_pu[bid, oe] for bid in downstream_branch_ids)
                )

                #linearised approximation of how voltage changes along a cable. Voltage at the far end is lower and drop depends on how much active power flows through the resistance
                #and how much reactive power flows through the reactance. For transformers, multiply by the voltage ratio for step up or down.
                if pd.notna(branch_row.voltage_ratio_pu):
                    self.model.c.add(
                        self.model.square_voltage_pu[to_bus_id, oe] * branch_row.voltage_ratio_pu**2 -
                        self.model.square_voltage_pu[from_bus_id, oe] == -2 * (
                            self.model.branch_active_pu[branch_id, oe] * branch_row.r_pu +
                            self.model.branch_reactive_pu[branch_id, oe] * branch_row.x_pu
                        ) + (
                            branch_row.r_pu**2 + branch_row.x_pu**2
                        ) * self.model.square_current_pu[branch_id, oe]
                    )
                else:
                    self.model.c.add(
                        self.model.square_voltage_pu[to_bus_id, oe] -
                        self.model.square_voltage_pu[from_bus_id, oe] == -2 * (
                            self.model.branch_active_pu[branch_id, oe] * branch_row.r_pu +
                            self.model.branch_reactive_pu[branch_id, oe] * branch_row.x_pu
                        ) + (
                            branch_row.r_pu**2 + branch_row.x_pu**2
                        ) * self.model.square_current_pu[branch_id, oe]
                    )

                #I^2  x V^2 = P^2 + Q^2
                self.model.c.add(
                    self.model.square_current_pu[branch_id, oe] *
                    self.model.square_voltage_pu[from_bus_id, oe] ==
                    self.model.branch_active_pu[branch_id, oe] * self.model.branch_active_pu[branch_id, oe] +
                    self.model.branch_reactive_pu[branch_id, oe] * self.model.branch_reactive_pu[branch_id, oe]
                )

                #current in a cable can't exceed rated maximum.
                if pd.notna(branch_row.i_max_pu):
                    if self.soft_limits:
                        self.model.c.add(
                            self.model.square_current_pu[branch_id, oe]
                            <= branch_row.i_max_pu**2 + self.model.viol_i_pu[branch_id, oe]
                        )
                    else:
                        self.model.c.add(self.model.square_current_pu[branch_id, oe] <= branch_row.i_max_pu**2)
        # everything is squared here, because working with V^2 and I^2 keeps the constraints in a form that avoids square roots.

        # Transformer thermal constraints (IEEE C57.91) --------------------------------------------------------------
        # LEGACY MODE ONLY. In "dtr" mode the thermal model is inverted into a
        # time-varying bound on square_current_pu (see _build_network_data) and
        # this nonlinear constraint is removed entirely; in "static" mode the
        # thermal model plays no part in the optimisation. Keeping the combined
        # constraint here under tx_limit="legacy" gives the Phase-9 equivalence
        # tests a reference for A/B comparison.
        if self.tx_limit == "legacy" and self.transformer_params is not None:
            tp = self.transformer_params
            R            = tp['R']
            n            = tp['n']
            m            = tp['m']
            dTO_R        = tp['delta_theta_TO_R']
            dHS_R        = tp['delta_theta_HS_R']
            I_rated      = tp['I_rated']
            theta_HS_max = tp['theta_HS_max']
            dt           = tp['dt']  # time-step length in minutes

            # Precompute scalar step-response coefficients
            alpha_TO = 1.0 - np.exp(-dt / tp['tau_TO'])
            alpha_W  = 1.0 - np.exp(-dt / tp['tau_W'])

            for tx_row in self.transformers.itertuples():
                tx_id    = tx_row.Index
                i_base_a = tx_row.i_base_a
                # K² = I²/I_rated² = square_current_pu * (i_base_a/I_rated)²
                c2 = (i_base_a / I_rated) ** 2

                prev   = self.thermal_state_in.get(tx_id, {})
                dTO_0  = prev.get('delta_theta_TO', 0.0)
                dHS_0  = prev.get('delta_theta_HS', 0.0)

                for oe in _oe_idxs:
                    K2 = self.model.square_current_pu[tx_id, oe] * c2

                    # Ultimate temperature rises — IEEE C57.91 exact form
                    delta_TO_U = dTO_R * ((K2 * R + 1) / (R + 1)) ** n
                    eps = 1e-6
                    delta_HS_U = dHS_R * (K2 + eps) ** m


                    # Step-response (exact discretisation of the first-order ODEs)
                    delta_TO = (delta_TO_U - dTO_0) * alpha_TO + dTO_0
                    delta_HS = (delta_HS_U - dHS_0) * alpha_W  + dHS_0

                    self.model.c.add(
                        self.theta_A + delta_TO + delta_HS <= theta_HS_max
                    )

        # Objective function -----------------------------------------------------------------------------------------

        # Economic surplus: maximise price × export envelope width (5-min interval = 5/60 h)
        benefit_term = -(5.0 / 60.0) * sum(
            self.partic_prices[load_id] * self.model.p_inj_oe_kw[load_id, 'oer']
            for load_id in self.partic_load_ids
        )

        small_weight = 0.001  # Nudge towards wider envelopes to break degeneracy
        width_term = small_weight * sum(
            self.model.p_inj_oe_kw[load_id, 'oel'] - self.model.p_inj_oe_kw[load_id, 'oer']
            for load_id in self.partic_load_ids
        )

        big_weight = 1000.0  # Violation penalty — last resort
        viol_term = big_weight * sum(
            self.model.sof_bus_a_kw[bus_id, oe, ci] + self.model.sof_bus_r_kw[bus_id, oe, ci]
            for bus_id in self.load_buses for oe in _oe_idxs for ci in _ci_idxs
        )

        # Usable-export reward (see __init__): u_i <= ub_i, u_i <= P_des_i+.
        if self.usable_export_weight > 0:
            p_des_pos = {
                lid: max(-self.df_forecasts_filt.loc[lid, "real_power_w"] / 1000.0, 0.0)
                if lid in self.df_forecasts_filt.index else 0.0
                for lid in self.partic_load_ids
            }
            self.model.usable_export_kw = Var(
                self.partic_load_ids, name="usable_export_kw",
                domain=NonNegativeReals, initialize=0.0)
            for lid in self.partic_load_ids:
                self.model.c.add(self.model.usable_export_kw[lid]
                                 <= self.model.p_inj_oe_kw[lid, 'oer'])
                self.model.c.add(self.model.usable_export_kw[lid]
                                 <= p_des_pos[lid])
            width_term = width_term - self.usable_export_weight * sum(
                self.model.usable_export_kw[lid] for lid in self.partic_load_ids)

        # Penalty on the soft voltage/current limit slacks (MDL001). Scaled by
        # the same order as the power-balance slacks so violations are used
        # only as a last resort but still solve.
        if self.soft_limits:
            viol_term = viol_term + self.soft_limit_penalty * (
                sum(self.model.viol_v_pu[i] for i in self.model.viol_v_pu)
                + sum(self.model.viol_i_pu[i] for i in self.model.viol_i_pu)
            )

        self.model.value = Objective(expr=benefit_term + width_term + viol_term, sense=minimize)
    #this is the IPOPT solver instance, calling the solver. 
    def _solve_opt_model(self):
        solver = SolverFactory(self.solver_name)
        for k, v in self.solver_options.items():
            solver.options[k] = v

        if self.quiet:
            # No stdout redirection, no banner, no iteration output. The string
            # formatting through the logging machinery on every solve is
            # measurable overhead across tens of thousands of timesteps.
            solver.options['print_level'] = 0
            solver.options['sb'] = 'yes'
            # Capture ipopt's own diagnosis to a scratch file. Without this the
            # only thing recorded on failure is Pyomo's wrapper ValueError
            # ("Cannot load a SolverResults object with bad status: error"),
            # which says nothing about WHY — restoration failure, too few
            # degrees of freedom, evaluation error and genuine infeasibility
            # all look identical. Costs one small file per failed solve.
            import tempfile as _tf
            _log = Path(_tf.gettempdir()) / f"ipopt_{os.getpid()}.log"
            solver.options['file_print_level'] = 5
            solver.options['output_file'] = str(_log)
            # Wall-clock cap. Without one, a single pathological interval ran
            # for 5189 s (86 minutes) and held a whole run hostage. Losing that
            # interval costs one row; losing the afternoon costs the run.
            solver.options.setdefault('max_cpu_time', float(self.max_cpu_time))

            def _ipopt_verdict():
                try:
                    if _log.exists():
                        lines = [ln.strip() for ln in
                                 _log.read_text(errors="replace").splitlines()
                                 if ln.strip()]
                        return " | ".join(lines[-3:])[:300]
                except Exception:
                    pass
                return ""

            try:
                results = solver.solve(self.model, tee=False)
            except (ApplicationError, ValueError, OSError) as e:
                first = f"{type(e).__name__}: {e}"
                verdict = _ipopt_verdict()

                # Retry once with a more robust barrier configuration. These
                # failures are dominated by "Error in step computation" — the
                # KKT factorisation failing on a marginally-conditioned model,
                # not genuine infeasibility — and an adaptive barrier with
                # relaxed bounds recovers most of them. Only failed intervals
                # pay this cost.
                # Do not retry a TIMEOUT with the same budget — it will simply
                # time out again and double the cost of an interval that was
                # never going to converge. Retry only failures that might be
                # fixed by a different barrier configuration.
                timed_out = "Maximum CPU Time" in verdict or "maxTimeLimit" in first
                if self.retry_on_failure and not timed_out:
                    for k, v in _ROBUST_OPTIONS.items():
                        solver.options[k] = v
                    try:
                        results = solver.solve(self.model, tee=False)
                        self.last_solve_error = None
                        self.n_retries_used = getattr(self, "n_retries_used", 0) + 1
                        logger.info("solver recovered on retry (first attempt: "
                                    "%s)", verdict or first)
                        return results['Solver'][0].status
                    except (ApplicationError, ValueError, OSError) as e2:
                        verdict = _ipopt_verdict() or verdict
                        first = f"{first} | retry: {type(e2).__name__}: {e2}"

                # One bad interval must never kill a year-long run: record and
                # report as an error status; the caller logs and continues.
                self.last_solve_error = first
                if verdict:
                    self.last_solve_error += f" — ipopt: {verdict}"
                logger.error("solver failed: %s", self.last_solve_error)
                return pyomo.opt.SolverStatus.error
        else:
            with contextlib.redirect_stdout(OutputLogger(logger, logging.INFO)):
                try:
                    results = solver.solve(self.model, tee=True)  # tee=True to see solver output
                except (ApplicationError, ValueError, OSError) as e:
                    self.last_solve_error = f"{type(e).__name__}: {e}"
                    logger.error("solver failed: %s", self.last_solve_error)
                    return pyomo.opt.SolverStatus.error

        # A solver that STOPPED is not a solver that SOLVED. Hitting
        # max_cpu_time or max_iter makes ipopt return termination
        # 'maxIterations'/'maxTimeLimit' with status *warning*, and solve()
        # accepts warning — so the current, non-converged iterate would be
        # extracted and written out as a valid DOE with no indication that it
        # is meaningless. That is worse than the failure it replaced, because
        # it is silent. Treat any limit termination as a failure.
        try:
            tc = results.solver.termination_condition
        except (AttributeError, KeyError, IndexError):
            tc = None
        _BAD_TC = {
            pyomo.opt.TerminationCondition.maxIterations,
            pyomo.opt.TerminationCondition.maxTimeLimit,
            pyomo.opt.TerminationCondition.maxEvaluations,
            pyomo.opt.TerminationCondition.userInterrupt,
            pyomo.opt.TerminationCondition.infeasible,
            pyomo.opt.TerminationCondition.unbounded,
        }
        if tc in _BAD_TC:
            self.last_solve_error = (
                f"solver stopped without converging: termination={tc}. "
                f"Raise solver.max_cpu_time (currently "
                f"{self.max_cpu_time:g}s) if this is a timeout on a large "
                f"substation.")
            logger.error("solver failed: %s", self.last_solve_error)
            return pyomo.opt.SolverStatus.error

        self.last_solve_error = None
        return results['Solver'][0].status
    #reads solved variables back out into df. 
    def _extract_results(self):
        # Network
        recs = []
        for bus_row in self.buses.itertuples():
            bus_id = bus_row.Index
            recs.append(
                {
                    "id": bus_id,
                    "voltage_pu_oel": np.sqrt(self.model.square_voltage_pu[bus_id, 'oel'].value),
                    "voltage_pu_oer": np.sqrt(self.model.square_voltage_pu[bus_id, 'oer'].value)
                }
            )

        results_bus = pd.DataFrame.from_records(recs).set_index("id").round(6) if len(recs) > 0 else pd.DataFrame()

        recs = []
        for branch_row in self.branches.itertuples():
            branch_id = branch_row.Index
            i_base_a = branch_row.i_base_a
            # Ensure positive for sqrt below.
            # Sometimes we can have a very tiny negative value due to tolerance.
            recs.append(
                {
                    "id": branch_id,
                    "current_a_oel": (np.sqrt(abs(self.model.square_current_pu[branch_id, 'oel'].value)) * i_base_a),
                    "current_a_oer": (np.sqrt(abs(self.model.square_current_pu[branch_id, 'oer'].value)) * i_base_a),
                    "p_w_oel": (self.model.branch_active_pu[branch_id, 'oel'].value * _pu_to_w),
                    "p_w_oer": (self.model.branch_active_pu[branch_id, 'oer'].value * _pu_to_w),
                    "q_va_oel": (self.model.branch_reactive_pu[branch_id, 'oel'].value * _pu_to_w),
                    "q_va_oer": (self.model.branch_reactive_pu[branch_id, 'oer'].value * _pu_to_w),
                }
            )

        results_branch = pd.DataFrame.from_records(recs).set_index("id").round(6) if len(recs) > 0 else pd.DataFrame()

        # Violations
        recs = []
        for bus_row in self.buses.loc[self.load_buses].itertuples():
            bus_id = bus_row.Index
            viol_a_oel = sum(self.model.sof_bus_a_kw[bus_id, 'oel', ci].value for ci in _ci_idxs)
            viol_a_oer = sum(self.model.sof_bus_a_kw[bus_id, 'oer', ci].value for ci in _ci_idxs)
            viol_r_oel = sum(self.model.sof_bus_r_kw[bus_id, 'oel', ci].value for ci in _ci_idxs)
            viol_r_oer = sum(self.model.sof_bus_r_kw[bus_id, 'oer', ci].value for ci in _ci_idxs)

            recs.append(
                {
                    "id": bus_id,
                    "viol_a_kw_oel": viol_a_oel,
                    "viol_a_kw_oer": viol_a_oer,
                    "viol_r_kw_oel": viol_r_oel,
                    "viol_r_kw_oer": viol_r_oer,
                }
            )

        if len(recs) > 0:
            results_viol = pd.DataFrame.from_records(recs).set_index("id").round(6)
            results_viol = results_viol.loc[(results_viol != 0).any(axis=1)]
        else:
            results_viol = pd.DataFrame()

        # Operating envelopes

        recs = []
        for load_id in self.partic_load_ids:
            recs.append({
                "load_id": load_id,
                "doe_lb_kw": self.model.p_inj_oe_kw[load_id, 'oel'].value,
                "doe_ub_kw": self.model.p_inj_oe_kw[load_id, 'oer'].value,
            })

        results_soe = pd.DataFrame.from_records(recs).set_index("load_id").round(6) if len(recs) > 0 else pd.DataFrame()

        # Transformer thermal state output — pass as thermal_state_in to the next period
        # (advanced with the ACTUAL solved current; the recursion is identical
        # in every tx_limit mode — only where the constraint is applied moved).
        thermal_state_out = {}
        if self.transformer_params is not None:
            tp      = self.transformer_params
            I_rated = tp['I_rated']

            for tx_row in self.transformers.itertuples():
                tx_id    = tx_row.Index
                i_base_a = tx_row.i_base_a
                c2       = (i_base_a / I_rated) ** 2

                prev   = self.thermal_state_in.get(tx_id, {})
                dTO_0  = prev.get('delta_theta_TO', 0.0)
                dHS_0  = prev.get('delta_theta_HS', 0.0)

                # 'oel' current, as in the original code. NOTE: with envelope
                # variables in the power flow the two oe scenarios can differ;
                # oel (max-consumption side) is the conservative choice and is
                # kept unchanged for backwards compatibility.
                K2_val = self.model.square_current_pu[tx_id, 'oel'].value * c2
                st = _thermal.forward_step(tp, K2_val, self.theta_A, dTO_0, dHS_0)
                thermal_state_out[tx_id] = st

                # Attach the DTR record for this timestep (evidence the dynamic
                # rating is doing something — written to thermal.parquet).
                if tx_id in self.dtr_info:
                    self.dtr_info[tx_id]['theta_HS'] = st['theta_HS']
                    self.dtr_info[tx_id]['K2_actual'] = K2_val

        return namedtuple("Results", "bus branch viol soe thermal_state")(
            results_bus, results_branch, results_viol, results_soe, thermal_state_out
        )
    def warm_start_values(self):
        '''Solved variable values, keyed for the next timestep's warm_start_in.

        Pass the returned dict as ``warm_start_in`` when constructing the
        solver for the next interval so variables initialise from this
        solution instead of a flat start.
        '''
        out = {}
        for name in ("square_voltage_pu", "square_current_pu",
                     "branch_active_pu", "branch_reactive_pu"):
            var = getattr(self.model, name)
            out[name] = {idx: var[idx].value for idx in var
                         if var[idx].value is not None}
        return out

    def extract_soft_violations(self, tol=1e-6):
        '''Nonzero voltage/current soft-limit slacks (needs soft_limits=True).

        Returns a list of dicts: {kind: 'voltage'|'current', id, oe, viol_pu}.
        '''
        rows = []
        if not self.soft_limits:
            return rows
        for (bid, oe) in self.model.viol_v_pu:
            v = self.model.viol_v_pu[bid, oe].value
            if v is not None and v > tol:
                rows.append({"kind": "voltage", "id": bid, "oe": oe, "viol_pu": v})
        for (bid, oe) in self.model.viol_i_pu:
            v = self.model.viol_i_pu[bid, oe].value
            if v is not None and v > tol:
                rows.append({"kind": "current", "id": bid, "oe": oe, "viol_pu": v})
        return rows

    # this is called at the start of the model, which pre-computes background load at each bus before the optimization model is built. 
    def _calculate_bus_loads_kw(self, bus_idxs):
        '''
        Calculate local active and reactive background load at each bus.

        For PARTICIPANT NMIs we don't include the active power forecast, as the
        active power is treated separately as the envelope limits.

        For NON-PARTICIPANT NMIs the active power IS background load and must be
        included, or they are electrically invisible to the power flow. Two
        populations land here:
          * NMIs with real meter data that were excluded from participation
            (participant_load_ids given as a subset);
          * NMIs with no meter data at all, given a synthetic profile by
            synthetic.py — no meter means no envelope can be issued to them,
            but they still load the network.
        Omitting them makes the feeder look unloaded and the envelopes come out
        far too generous.
        '''

        bus_ld_a_kw = {bus_id: 0.0 for bus_id in bus_idxs for oe in _oe_idxs}
        bus_ld_r_kw = {bus_id: 0.0 for bus_id in bus_idxs for oe in _oe_idxs}
        partic = set(self.partic_load_ids)
        for load_row in self.loads.itertuples():
            load_id = load_row.Index
            bus_id = load_row.bus_id

            if load_id in self.forecast_load_ids:
                bus_ld_r_kw[bus_id] += self.df_forecasts_filt.loc[load_id, "reactive_power_var"] * 1e-3
                if load_id not in partic:
                    bus_ld_a_kw[bus_id] += self.df_forecasts_filt.loc[load_id, "real_power_w"] * 1e-3

        return (bus_ld_a_kw, bus_ld_r_kw)

#convenience wrapper
def solve_soes(netw_ejson, df_forecasts_t, df_offers_t, solver_options={}):
    solver = SoeSolver(netw_ejson, df_forecasts_t, df_offers_t, solver_options=solver_options)
    status, results = solver.solve()
    return solver, status, results

#parses the JSON structure and gets the network dictionary. 
def _netw_components(netw_ejson, comp_type=None):
    comps = ((k1, k2, v2) for k1, v1 in netw_ejson["components"].items() for k2, v2 in v1.items())
    if comp_type is None:
        return list(comps)
    else:
        return [x for x in comps if x[1] == comp_type]
