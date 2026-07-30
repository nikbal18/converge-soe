"""The three scenario definitions and the per-substation solve loops.

| Key          | Transformer limit                                   | Thermal model | Envelopes |
|--------------|-----------------------------------------------------|---------------|-----------|
| ``doe_dtr``  | i_max_pu from the inverted C57.91 model, recomputed | drives the    | solved    |
|              | each timestep, capped at K_emergency                | limit         |           |
| ``doe_static``| i_max_pu from nameplate s_max, constant            | post-hoc only | solved    |
| ``bau``      | not enforced — violations recorded, not prevented   | post-hoc      | none      |

BAU is NOT an optimisation: customers do exactly what the forecast says. It
is evaluated with a vectorised radial power-flow sweep over the whole year at
once (seconds, not hours), then the C57.91 recursion runs forward on the
resulting transformer currents. BAU is the ageing baseline and the
"no curtailment" reference — E_curt(bau) = 0 by construction.
"""

import logging

import numpy as np
import pandas as pd

from . import thermal as _thermal
from .doe_solver import SoeSolver as DoeSolver
from .doe_solver import S_BASE_VA
from .network import validate as _validate

logger = logging.getLogger(__name__)

SCENARIOS = ("doe_dtr", "doe_static", "bau")


def tx_limit_of(scenario):
    return {"doe_dtr": "dtr", "doe_static": "static"}[scenario]


# ---------------------------------------------------------------------------
# DOE scenarios (doe_dtr / doe_static)
# ---------------------------------------------------------------------------
def run_doe_scenario(scenario, sub_ej, bundle, transformer_params, cfg,
                     writer, start_after=None, thermal_state=None,
                     progress_cb=None, fast=False):
    """Solve every timestep of one DOE scenario for one substation, streaming
    results through ``writer`` (a converge_soe.io.SubstationWriter).

    ``bundle`` is the pre-indexed npz dict (stage ⑤) — the loop below touches
    no pandas at all for the timeseries.
    """
    tx_limit = tx_limit_of(scenario)
    idx = pd.DatetimeIndex(bundle["timestamps"].view("datetime64[ns]"))
    load_ids = [str(x) for x in bundle["load_ids"]]
    P, Q, theta = bundle["P"], bundle["Q"], bundle["theta_A"]

    # NMIs given a synthetic profile by synthetic.py have no meter, so no
    # envelope can be issued to them. They must still load the network — that
    # is handled by _calculate_bus_loads_kw, which adds active power for every
    # forecast load that is not a participant. When nothing is synthetic this
    # is None, i.e. exactly the previous behaviour.
    # synthetic.participants = true makes the synthetic NMIs participants too.
    # Physically that is wrong — you cannot issue an envelope to a customer
    # with no meter — but it isolates what the DOE could achieve if EVERY
    # customer were dispatchable, which is the upper bound on the mechanism.
    # With it false (default), synthetic NMIs are uncontrollable background
    # load, which is what limits the DOE in practice.
    _syn = np.asarray(
        bundle.get("synthetic", np.zeros(len(load_ids), dtype=bool)), dtype=bool)
    _all_participate = bool((cfg.get("synthetic", {}) or {}).get("participants", False))
    participants = (None if (_all_participate or not _syn.any())
                    else [lid for lid, s in zip(load_ids, _syn) if not s])
    state = dict(thermal_state or {})
    # THE carried thermal state is the transformer's ACTUAL state: every
    # customer follows their forecast, clipped into [doe_lb, doe_ub]. This is
    # what a real DTR would see (measured oil temperature), it is consistent
    # with the BAU trajectory (so ageing is comparable across scenarios), and
    # it guarantees the delivered θ_HS trajectory respects θ_HS_max: each
    # interval's limit is computed from the actual starting state and the
    # actual current cannot exceed the granted envelope current.
    # (The legacy runner carried the 'oel' envelope-scenario state instead —
    # see docs/MIGRATION_NOTES.md.)
    posthoc_state = state.pop("__posthoc__", {}) if isinstance(state, dict) else {}
    if posthoc_state:
        state = posthoc_state  # resume: the actual state is the carried one
    comps_t = {k: v for k, v in sub_ej["components"].items() if "Transformer" in v}
    inf_c = next((v["Infeeder"] for v in sub_ej["components"].values()
                  if "Infeeder" in v), None)
    v_root2 = 1.0
    if inf_c is not None:
        root_nd = sub_ej["components"][inf_c["cons"][0]["node"]]["Node"]
        v_root2 = (inf_c["v_setpoint"] / root_nd["v_base"]) ** 2
    # transformer per-unit impedance (for the loss correction of the actual
    # current; same construction as the solver's _build_network_data)
    tx_rx = {}
    z_units = sub_ej["units"].get("impedance", 1)
    v_units = sub_ej["units"]["voltage"]
    for tx_id, entry in comps_t.items():
        cd = entry["Transformer"]
        sec_nd = sub_ej["components"][cd["cons"][1]["node"]]["Node"]
        z_base = (sec_nd["v_base"] * v_units) ** 2 / S_BASE_VA
        r_pu = (2.0 / 3.0) * cd["z"][1][0] * z_units / z_base
        x_pu = (2.0 / 3.0) * cd["z"][1][1] * z_units / z_base
        tx_rx[tx_id] = (r_pu, x_pu)

    start_i = 0
    if start_after is not None:
        after = pd.Timestamp(start_after)
        start_i = int(np.searchsorted(idx.values, np.datetime64(after)) + 1)

    envelope_abs_max = cfg.get("envelope_abs_max", 50.0)
    k_emergency = cfg.get("thermal", {}).get("k_emergency", 2.0)
    k2_floor = cfg.get("thermal", {}).get("k2_floor", 1e-6)
    soft = cfg.get("solver", {}).get("soft_limits", True)
    soft_pen = cfg.get("solver", {}).get("soft_limit_penalty", 1000.0)
    solver_name = cfg.get("solver", {}).get("name", "ipopt")
    use_w = cfg.get("solver", {}).get("usable_export_weight", 0.01)
    solver_opts = dict(cfg.get("solver", {}).get("options", {}) or {})
    if cfg.get("solver", {}).get("linear_solver"):
        solver_opts.setdefault("linear_solver", cfg["solver"]["linear_solver"])
    max_cpu = float(cfg.get("solver", {}).get("max_cpu_time", 120.0))
    retry = bool(cfg.get("solver", {}).get("retry_on_failure", True))

    if fast:
        from .persistent_solver import PersistentDoeSolver
        ps = PersistentDoeSolver(
            sub_ej, load_ids, envelope_abs_max=envelope_abs_max,
            participant_load_ids=participants,
            transformer_params=transformer_params, tx_limit=tx_limit,
            k_emergency=k_emergency, k2_floor=k2_floor,
            soft_limits=soft, soft_limit_penalty=soft_pen,
            quiet=True, solver_name=solver_name, thermal_state_in=state,
            usable_export_weight=use_w)

    net_cache = None
    warm = None
    for i in range(start_i, len(idx)):
        ts = idx[i]
        theta_A = float(theta[i]) if np.isfinite(theta[i]) else \
            cfg.get("ambient", {}).get("constant_c", 25.0)

        if fast:
            ps.thermal_state = dict(state)   # DTR driven by the ACTUAL state
            r = ps.solve_step(P[i], Q[i], theta_A)
            if r.status != "ok":
                writer.record_failure(ts, "solve_failed", "error",
                                      r.failure_reason or "")
                if progress_cb:
                    progress_cb(i, ok=False)
                continue
            doe, bus, branch = r.doe, r.bus, r.branch
            envelope_state = r.thermal_state   # oel-advanced; recorded only
            dtr_info = r.dtr_info
            viol_rows = ([] if r.viol is None or r.viol.empty
                         else r.viol.to_dict("records"))
            soft_rows = r.soft_viols
        else:
            f_t = pd.DataFrame({"real_power_w": P[i].astype(float),
                                "reactive_power_var": Q[i].astype(float)},
                               index=pd.Index(load_ids, name="load_id"))
            s = DoeSolver(sub_ej, f_t, envelope_abs_max=envelope_abs_max,
                          participant_load_ids=participants,
                          transformer_params=transformer_params,
                          theta_A=theta_A, thermal_state_in=dict(state),
                          tx_limit=tx_limit, k_emergency=k_emergency,
                          k2_floor=k2_floor, soft_limits=soft,
                          soft_limit_penalty=soft_pen, quiet=True,
                          warm_start_in=warm, network_cache=net_cache,
                          solver_options=solver_opts, solver_name=solver_name,
                          usable_export_weight=use_w,
                          max_cpu_time=max_cpu, retry_on_failure=retry)
            if net_cache is None:
                net_cache = s.network_cache()
            status, res = s.solve()
            if res is None:
                writer.record_failure(ts, "solve_failed", str(status),
                                      getattr(s, "last_solve_error", "") or "")
                if progress_cb:
                    progress_cb(i, ok=False)
                continue
            warm = s.warm_start_values()
            doe, bus, branch = res.soe, res.bus, res.branch
            envelope_state = res.thermal_state   # oel-advanced; recorded only
            dtr_info = s.dtr_info
            viol_rows = []
            if not res.viol.empty:
                for bid, vr in res.viol.iterrows():
                    viol_rows.append({"id": bid, "oe": "",
                                      "viol_a_kw": float(vr.get("viol_a_kw_oel", 0)
                                                         + vr.get("viol_a_kw_oer", 0)),
                                      "viol_r_kw": float(vr.get("viol_r_kw_oel", 0)
                                                         + vr.get("viol_r_kw_oer", 0))})
            soft_rows = s.extract_soft_violations()

        ts_s = str(ts)
        p_des = -P[i] / 1000.0    # desired injection, kW (+ve = PV export)

        # ACTUAL-behaviour thermal step: this IS the carried state.
        new_state = {}
        k2_actual_by_tx = {}
        if transformer_params is not None and comps_t:
            lb = doe["doe_lb_kw"].reindex(load_ids).fillna(-np.inf).values
            ub = doe["doe_ub_kw"].reindex(load_ids).fillna(np.inf).values
            p_act_kw = np.clip(p_des, lb, ub)          # injection, kW
            p_w_tot = float((-p_act_kw * 1000.0).sum())
            q_var_tot = float(Q[i].sum())
            p_pu, q_pu = p_w_tot / S_BASE_VA, q_var_tot / S_BASE_VA
            for tx_id in (envelope_state or comps_t):
                info = dtr_info.get(tx_id, {})
                c2 = info.get("c2")
                if c2 is None:
                    tr = list(comps_t.values())[0]["Transformer"]
                    v_v = tr["v_winding_base"][1] * sub_ej["units"]["voltage"]
                    i_base = S_BASE_VA / v_v
                    c2 = (i_base / transformer_params["I_rated"]) ** 2
                # two fixed-point iterations of I²V² = (P+rI²)² + (Q+xI²)²
                # so the actual current includes transformer losses, matching
                # the solver's branch-flow convention
                r_pu, x_pu = tx_rx.get(tx_id, (0.0, 0.0))
                i2_pu = (p_pu ** 2 + q_pu ** 2) / v_root2
                for _ in range(2):
                    i2_pu = (((p_pu + r_pu * i2_pu) ** 2
                              + (q_pu + x_pu * i2_pu) ** 2) / v_root2)
                # The actual operating point lies inside the envelope box, so
                # its transformer current cannot exceed the binding corner the
                # model solved (which includes the full loss physics). Cap the
                # cheap lossless estimate there — this also guarantees the
                # delivered θ_HS trajectory respects the DTR limit.
                if tx_id in branch.index:
                    i_corner_a = max(float(branch.at[tx_id, "current_a_oel"]),
                                     float(branch.at[tx_id, "current_a_oer"]))
                    i2_corner = (i_corner_a / transformer_params["I_rated"]) ** 2 / c2
                    i2_pu = min(i2_pu, i2_corner)
                prev = state.get(tx_id, {})
                k2_actual_by_tx[tx_id] = i2_pu * c2
                new_state[tx_id] = _thermal.forward_step(
                    transformer_params, i2_pu * c2, theta_A,
                    prev.get("delta_theta_TO", 0.0),
                    prev.get("delta_theta_HS", 0.0))
            state = new_state
        writer.append("doe", [
            {"timestamp": ts_s, "load_id": lid,
             "doe_lb_kw": float(doe.at[lid, "doe_lb_kw"]),
             "doe_ub_kw": float(doe.at[lid, "doe_ub_kw"]),
             "p_des_kw": float(p_des[j]),
             "has_data": bool(bundle["mask"][i, j])}
            for j, lid in enumerate(load_ids) if lid in doe.index])
        writer.append("bus", [
            {"timestamp": ts_s, "id": bid,
             "voltage_pu_oel": float(row["voltage_pu_oel"]),
             "voltage_pu_oer": float(row["voltage_pu_oer"])}
            for bid, row in bus.iterrows()])
        writer.append("branch", [
            {"timestamp": ts_s, "id": bid,
             "current_a_oel": float(row["current_a_oel"]),
             "current_a_oer": float(row["current_a_oer"]),
             "p_w_oel": float(row["p_w_oel"]), "p_w_oer": float(row["p_w_oer"]),
             "q_va_oel": float(row["q_va_oel"]), "q_va_oer": float(row["q_va_oer"])}
            for bid, row in branch.iterrows()])
        vr = [{"timestamp": ts_s, "kind": "bus_balance", **v} for v in viol_rows]
        vr += [{"timestamp": ts_s, "kind": v["kind"], "id": v["id"],
                "oe": v["oe"], "viol_a_kw": np.nan, "viol_r_kw": np.nan,
                "viol_pu": float(v["viol_pu"])} for v in soft_rows]
        if vr:
            for row in vr:
                row.setdefault("viol_pu", np.nan)
            writer.append("viol", vr)
        writer.append("thermal", [
            {"timestamp": ts_s, "transformer_id": tx_id,
             "theta_HS_C": round(st["theta_HS"], 4),
             "theta_HS_posthoc_C": round(st["theta_HS"], 4),
             "theta_HS_envelope_C": round(
                 envelope_state.get(tx_id, {}).get("theta_HS", np.nan), 4)
                 if envelope_state else np.nan,
             "delta_theta_TO_C": round(st["delta_theta_TO"], 4),
             "delta_theta_HS_C": round(st["delta_theta_HS"], 4),
             "theta_A_C": theta_A,
             "i_max_pu": float(dtr_info.get(tx_id, {}).get("i_max_pu", np.nan)),
             "K2_max": float(dtr_info.get(tx_id, {}).get("K2_max", np.nan)),
             "K2_actual": float(k2_actual_by_tx.get(tx_id, np.nan)),
             "dtr_status": str(dtr_info.get(tx_id, {}).get("status", ""))}
            for tx_id, st in state.items()])
        writer.complete_timestep(ts_s, {**state, "__posthoc__": state})
        if progress_cb:
            progress_cb(i, ok=True)

    return state


# ---------------------------------------------------------------------------
# BAU — vectorised evaluation, no optimisation
# ---------------------------------------------------------------------------
def run_bau_scenario(sub_ej, bundle, transformer_params, cfg, writer,
                     progress_cb=None):
    """Evaluate BAU for the whole period at once.

    Radial lossless sweep vectorised over T: per branch, downstream
    P/Q accumulation, I ≈ S/V, V² drop along the tree; limits checked and
    violations recorded (never prevented). Then the thermal recursion runs
    forward on the transformer currents.
    """
    comps = _validate.components_by_type(sub_ej)
    nodes = comps["Node"]
    idx = pd.DatetimeIndex(bundle["timestamps"].view("datetime64[ns]"))
    load_ids = [str(x) for x in bundle["load_ids"]]
    P = bundle["P"].astype(np.float64)
    Q = bundle["Q"].astype(np.float64)
    theta = bundle["theta_A"].astype(np.float64)
    if np.isnan(theta).all():
        theta = np.full(len(idx), cfg.get("ambient", {}).get("constant_c", 25.0))
    T = len(idx)

    v_units = sub_ej["units"]["voltage"]
    i_units = sub_ej["units"]["current"]
    z_units = sub_ej["units"]["impedance"]
    s_base = S_BASE_VA

    inf = next(iter(comps["Infeeder"].values()))
    root = inf["cons"][0]["node"]

    # tree structure
    from collections import defaultdict
    adj = defaultdict(list)
    for cid, kind, n0, n1 in _validate.branch_endpoints(sub_ej):
        adj[n0].append((n1, cid, kind))
        adj[n1].append((n0, cid, kind))
    parent_edge, order, seen = {}, [root], {root}
    for cur in order:
        for nxt, cid, kind in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                parent_edge[nxt] = (cid, kind, cur)
                order.append(nxt)

    lid_pos = {l: j for j, l in enumerate(load_ids)}
    node_cols = defaultdict(list)
    for cid, cd in comps["Load"].items():
        if cid in lid_pos:
            node_cols[cd["cons"][0]["node"]].append(lid_pos[cid])

    # subtree columns bottom-up
    subtree = {n: list(node_cols.get(n, [])) for n in order}
    for n in reversed(order):
        if n != root:
            subtree[parent_edge[n][2]].extend(subtree[n])

    # per-branch flows [T]
    branch_rows = {}
    v2 = {root: np.full(T, (inf["v_setpoint"] / nodes[root]["v_base"]) ** 2)}
    viol_rows = []
    thermal_by_tx = {}

    for n in order:
        if n == root:
            continue
        cid, kind, par = parent_edge[n]
        cols = subtree[n]
        p_w = P[:, cols].sum(axis=1) if cols else np.zeros(T)
        q_var = Q[:, cols].sum(axis=1) if cols else np.zeros(T)
        p_pu, q_pu = p_w / s_base, q_var / s_base
        v_base_v = nodes[n]["v_base"] * v_units
        z_base = v_base_v ** 2 / s_base
        i_base_a = s_base / v_base_v

        if kind == "Line":
            cd = comps["Line"][cid]
            r = cd["z"][0] * z_units * cd.get("length", 1.0) / z_base
            x = cd["z"][1] * z_units * cd.get("length", 1.0) / z_base
            ratio2 = 1.0
            i_max_a = cd["i_max"] * i_units if "i_max" in cd else np.inf
        else:
            cd = comps["Transformer"][cid]
            r = cd["z"][1][0] * z_units / z_base
            x = cd["z"][1][1] * z_units / z_base
            try:
                tr = cd["nom_turns_ratio"][0] * (1.0 + cd["taps"][0] * cd["tap_factor"])
                ratio2 = (tr * nodes[n]["v_base"] / nodes[par]["v_base"]) ** 2
            except (KeyError, IndexError, TypeError):
                ratio2 = 1.0
            s_max_w = cd.get("s_max", 1e9 / sub_ej["units"]["power"]) * sub_ej["units"]["power"]
            i_max_a = s_max_w / (cd["v_winding_base"][1] * v_units)

        v2_n = (v2[par] - 2 * (p_pu * r + q_pu * x)) / ratio2
        v2[n] = np.maximum(v2_n, 1e-6)
        i2_pu = (p_pu ** 2 + q_pu ** 2) / v2[par]
        i_a = np.sqrt(i2_pu) * i_base_a
        branch_rows[cid] = (p_w, q_var, i_a)

        # violations: current over rating
        over = i_a > i_max_a
        if over.any():
            for t_i in np.where(over)[0]:
                viol_rows.append({"timestamp": str(idx[t_i]), "kind": "current",
                                  "id": cid, "oe": "bau",
                                  "viol_a_kw": np.nan, "viol_r_kw": np.nan,
                                  "viol_pu": float((i_a[t_i] - i_max_a) / i_base_a)})
        # violations: voltage outside limits
        ud = nodes[n].get("user_data") or {}
        if ud.get("v_min") is not None:
            vmin = ud["v_min"] / nodes[n]["v_base"]
            vmax = ud["v_max"] / nodes[n]["v_base"]
            v = np.sqrt(v2[n])
            for t_i in np.where((v < vmin - 1e-3) | (v > vmax + 1e-3))[0]:
                viol_rows.append({"timestamp": str(idx[t_i]), "kind": "voltage",
                                  "id": n, "oe": "bau",
                                  "viol_a_kw": np.nan, "viol_r_kw": np.nan,
                                  "viol_pu": float(min(abs(v[t_i] - vmin),
                                                       abs(v[t_i] - vmax)))})

        if kind == "Transformer" and transformer_params is not None:
            tp = transformer_params
            k2 = i2_pu * (i_base_a / tp["I_rated"]) ** 2
            traj = _thermal.open_loop_trajectory(tp, k2, theta)
            thermal_by_tx[cid] = (k2, traj)

    # ---- stream out (chunked so memory stays flat)
    p_des = -P / 1000.0
    for i in range(T):
        ts_s = str(idx[i])
        writer.append("doe", [
            {"timestamp": ts_s, "load_id": lid,
             "doe_lb_kw": -np.inf, "doe_ub_kw": np.inf,
             "p_des_kw": float(p_des[i, j]),
             "has_data": bool(bundle["mask"][i, j])}
            for j, lid in enumerate(load_ids)])
        writer.append("bus", [
            {"timestamp": ts_s, "id": n,
             "voltage_pu_oel": float(np.sqrt(v2[n][i])),
             "voltage_pu_oer": float(np.sqrt(v2[n][i]))}
            for n in order])
        writer.append("branch", [
            {"timestamp": ts_s, "id": cid,
             "current_a_oel": float(iarr[i]), "current_a_oer": float(iarr[i]),
             "p_w_oel": float(pw[i]), "p_w_oer": float(pw[i]),
             "q_va_oel": float(qv[i]), "q_va_oer": float(qv[i])}
            for cid, (pw, qv, iarr) in branch_rows.items()])
        writer.append("thermal", [
            {"timestamp": ts_s, "transformer_id": tx_id,
             "theta_HS_C": round(traj[i], 4),
             "theta_HS_posthoc_C": round(traj[i], 4),
             "delta_theta_TO_C": np.nan, "delta_theta_HS_C": np.nan,
             "theta_A_C": float(theta[i]),
             "i_max_pu": np.nan, "K2_max": np.nan,
             "K2_actual": float(k2[i]), "dtr_status": "bau"}
            for tx_id, (k2, traj) in thermal_by_tx.items()])
        writer.complete_timestep(ts_s, {})
        if progress_cb:
            progress_cb(i, ok=True)
    if viol_rows:
        writer.append("viol", viol_rows)
    return {}
