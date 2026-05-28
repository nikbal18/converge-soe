from collections import namedtuple
import contextlib
import logging

from pyomo.environ import (
    ConcreteModel, ConstraintList, NonNegativeReals, Objective, minimize, pyomo, Reals, SolverFactory, Var
)
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

#what goes into the solver?
#netw_json - dictionary describing the physical grid. 
#df_forecasts - for each customer, a df that describes each customer, what power they are expected to consume in the next 5-minute interval
#df_offers - df - which customers are willing to be dispatched and at what price?
_s_base_va = 1.0e6
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
                 envelope_abs_max=50.0, solver_options: dict = {}):
        self.netw_ejson = netw_ejson
        self.netw_ejson["components"] = dict(sorted(self.netw_ejson["components"].items()))  # For reprod/testing
        # pandas df with forecast power consumption for each customer.
        self.df_forecasts = df_forecasts
        # maximum envelope width
        self.envelope_abs_max = envelope_abs_max
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

    def _build_network_data(self):
        #builds a dataframe for each bus, branch and loads. Converts everything to pu terms. 
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
                    "i_max_a": cd["i_max"] * i_units if "i_max" in cd else 100000.0,  # From old vers: 100 kA
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

        sel_lines = self.branches["voltage_ratio_pu"].isna()
        self.lines = self.branches.loc[sel_lines]
        self.transformers = self.branches.loc[~sel_lines]

    def _build_opt_model(self):
        #uses pyomo to build the actual optimization model. This includes:
        #variables: the unknowns the solver will find: envelope bounds, voltages, currents, power flows, network support dispatch. 
        #constraints - rules the solution must satisfy, eg. basic electronics principles. 
        #objectives- minimise cost of network support dispatches, with a small push towards wider envelopes and a large penalty for any constraint violations. 

        self.model = ConcreteModel()  # pyomo

        # Index sets -------------------------------------------------------------------------------------------------

        bus_idxs = self.buses.index
        branch_idxs = self.branches.index
        partic_idxs = self.forecast_load_ids
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

        # Network variables
        # All being setup with initial values
        self.model.square_voltage_pu = Var(
            bus_oe_idxs, name="square_voltage_pu", domain=NonNegativeReals, initialize=1.0
        )

        self.model.square_current_pu = Var(
            branch_oe_idxs, name="square_current_pu", domain=NonNegativeReals, initialize=0.0
        )
        self.model.branch_active_pu = Var(
            branch_oe_idxs, name="branch_active_pu", domain=Reals, initialize=0.0
        )
        self.model.branch_reactive_pu = Var(
            branch_oe_idxs, name="branch_reactive_pu", domain=Reals, initialize=0.0
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

            #reactive power: any load that has a forecast contributes it's reactive power forecast to the total (which is summed up later). 
            #reactive power always treated as a fixed known quantity, no envelopes used. 
            if load_id in self.forecast_load_ids:
                reactive_power = self.df_forecasts_filt.loc[load_id, "reactive_power_var"] * _w_to_pu
                for oe in _oe_idxs:
                    r_bus_pu[bus_id, oe].append(reactive_power)

            if load_id in self.forecast_load_ids:
                for oe in _oe_idxs:
                    a_bus_pu[(bus_id, oe)].append(-self.model.p_inj_oe_kw[load_id, oe] * _kw_to_pu)

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

        # SOE constraints

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
                downstream_branch_ids = self.branches.loc[self.branches["from_bus_id"] == to_bus_id].index
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
                    self.model.c.add(self.model.square_current_pu[branch_id, oe] <= branch_row.i_max_pu**2)
        # everything is squared here, because working with V^2 and I^2 keeps the constraints in a form that avoids square roots. 
        # Objective function -----------------------------------------------------------------------------------------
        #minimises the sum of three terms, with each representing a different priority. 
        #firstly, for every participant being dispatched, calculate the cost for the dispatch quantity. Summed separately for consumption and injection dispatches, then added together. 
        first_term = 0

        big_weight = 1000.0  # Dollars per kWh
        small_weight = 0.001  # Dollars per kWh
        #second term is the envelope width penalty, a small cost for smaller envelopes so tend towards wider envelopes. 
        second_term = small_weight * sum(
            self.model.p_inj_oe_kw[load_id, 'oel'] - self.model.p_inj_oe_kw[load_id, 'oer']
            for load_id in self.forecast_load_ids
        )

        #Third term is penalty for violation, $1000/kWh, only a last resort option. 
        third_term = big_weight * (
            sum(
                self.model.sof_bus_a_kw[bus_id, oe, ci] + self.model.sof_bus_r_kw[bus_id, oe, ci]
                for bus_id in self.load_buses for oe in _oe_idxs for ci in _ci_idxs
            )
        )

        self.model.value = Objective(expr=first_term + second_term + third_term, sense=minimize)
    #this is the IPOPT solver instance, calling the solver. 
    def _solve_opt_model(self):
        solver = SolverFactory("ipopt")
        for k, v in self.solver_options.items():
            solver.options[k] = v

        with contextlib.redirect_stdout(OutputLogger(logger, logging.INFO)):
            results = solver.solve(self.model, tee=True)  # tee=True to see solver output

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
        for load_id in self.forecast_load_ids:
            recs.append({
                "load_id": load_id,
                "doe_lb_kw": self.model.p_inj_oe_kw[load_id, 'oel'].value,
                "doe_ub_kw": self.model.p_inj_oe_kw[load_id, 'oer'].value,
            })

        results_soe = pd.DataFrame.from_records(recs).set_index("load_id").round(6) if len(recs) > 0 else pd.DataFrame()

        return namedtuple("Results", "bus branch viol soe")(results_bus, results_branch, results_viol, results_soe)
    # this is called at the start of the model, which pre-computes background load at each bus before the optimization model is built. 
    def _calculate_bus_loads_kw(self, bus_idxs):
        '''
        Calculate local active and reactive background load at each bus.
        For participant NMIs, we don't include the active power forecast, as the active power will be
        treated separately as the envelope limits.
        '''

        bus_ld_a_kw = {bus_id: 0.0 for bus_id in bus_idxs for oe in _oe_idxs}
        bus_ld_r_kw = {bus_id: 0.0 for bus_id in bus_idxs for oe in _oe_idxs}
        for load_row in self.loads.itertuples():
            load_id = load_row.Index
            bus_id = load_row.bus_id

            if load_id in self.forecast_load_ids:
                bus_ld_r_kw[bus_id] += self.df_forecasts_filt.loc[load_id, "reactive_power_var"] * 1e-3

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
