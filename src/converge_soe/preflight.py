"""Preflight: find the infeasibility before you spend an hour on it.

Thirty-plus fast checks over the network model, the timeseries, the physics,
and the optimisation setup. Runs in seconds — no solving. Every check emits a
finding::

    {id, severity, scope, message, detail, suggested_fix}

with severity ERROR / WARN / INFO. ``--strict`` makes any ERROR abort the run.

Check ids (see docs/TROUBLESHOOTING.md for every trigger and fix):
  NET001–NET013  network structure       TS001–TS007  timeseries
  PHY001–PHY005  physical feasibility    MDL001–MDL002 model setup
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from . import thermal as _thermal
from .network import validate as _validate

logger = logging.getLogger(__name__)

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"


def finding(fid, severity, scope, message, detail="", fix=""):
    return {"id": fid, "severity": severity, "scope": scope,
            "message": message, "detail": str(detail)[:2000],
            "suggested_fix": fix}


# ===========================================================================
# NET — network structure
# ===========================================================================
def check_network(ejson, parent_feeder_ejson=None, scope="network"):
    F = []
    comps = _validate.components_by_type(ejson)
    nodes = comps["Node"]
    lines = comps["Line"]
    txs = comps["Transformer"]
    loads = comps["Load"]
    infeeders = comps["Infeeder"]

    # NET001 every cons[].node exists
    dangling = []
    for ctype in ("Line", "Transformer", "Load", "Infeeder"):
        for cid, cd in comps[ctype].items():
            for con in cd.get("cons", []):
                if con.get("node") not in nodes:
                    dangling.append(f"{ctype} {cid} -> {con.get('node')}")
    if dangling:
        F.append(finding("NET001", ERROR, scope,
                         f"{len(dangling)} component connection(s) reference "
                         "nodes that do not exist",
                         dangling[:20],
                         "fix the ejson: every cons[].node must be a Node id"))

    # NET002 exactly one infeeder, on the transformer primary (if tx present)
    if len(infeeders) != 1:
        F.append(finding("NET002", ERROR, scope,
                         f"expected exactly 1 Infeeder, found {len(infeeders)}",
                         list(infeeders),
                         "no infeeder = no voltage reference; the model is "
                         "unbounded or infeasible"))
    elif txs:
        inf_node = next(iter(infeeders.values()))["cons"][0]["node"]
        primaries = {cd["cons"][0]["node"] for cd in txs.values()}
        secondaries = {cd["cons"][1]["node"] for cd in txs.values()}
        if inf_node in secondaries and inf_node not in primaries:
            F.append(finding("NET002", WARN, scope,
                             "Infeeder sits on a transformer SECONDARY node",
                             inf_node,
                             "expected on the primary (MV) node for an "
                             "extracted substation network"))

    # NET003 connectivity from infeeder
    if len(infeeders) == 1:
        inf_node = next(iter(infeeders.values()))["cons"][0]["node"]
        reach = _validate.reachable_from(ejson, inf_node)
        island = set(nodes) - reach
        if island:
            F.append(finding("NET003", ERROR, scope,
                             f"{len(island)} node(s) unreachable from the infeeder",
                             sorted(island)[:20],
                             "islanded nodes cannot satisfy power balance; "
                             "remove them or fix the topology"))

    # NET004 radial
    n_nodes, n_branches = len(nodes), len(lines) + len(txs)
    cycles = _validate.find_cycles(ejson)
    if cycles or (n_nodes and n_branches != n_nodes - 1):
        sev = ERROR if cycles else WARN
        F.append(finding("NET004", sev, scope,
                         f"network is not radial: {n_branches} branches for "
                         f"{n_nodes} nodes; {len(cycles)} cycle-closing branch(es)",
                         cycles[:20],
                         "the solver sums *downstream* branch flows, which is "
                         "only valid on a radial tree — break the loops"))

    # NET005 self loops / parallel branches
    seen_pairs = defaultdict(list)
    selfloops = []
    for cid, kind, n0, n1 in _validate.branch_endpoints(ejson):
        if n0 == n1 and n0 is not None:
            selfloops.append(cid)
        seen_pairs[frozenset((n0, n1))].append(cid)
    dupes = {tuple(v) for k, v in seen_pairs.items() if len(v) > 1}
    if selfloops:
        F.append(finding("NET005", ERROR, scope,
                         f"{len(selfloops)} self-loop branch(es)", selfloops[:10],
                         "remove; they create degenerate constraints"))
    if dupes:
        F.append(finding("NET005", ERROR, scope,
                         f"{len(dupes)} duplicate parallel branch group(s)",
                         list(dupes)[:10],
                         "merge parallel branches into one equivalent"))

    # NET006 impedances finite and > 1e-9 (checked in ohm-per-unit terms after
    # conversion; here in raw z entries)
    bad_z = []
    for cid, cd in lines.items():
        z = cd.get("z", [0, 0])
        try:
            r, x = float(z[0]), float(z[1])
            if not np.isfinite(r) or not np.isfinite(x) or (abs(r) <= 1e-9 and abs(x) <= 1e-9):
                bad_z.append(cid)
        except (TypeError, ValueError, IndexError):
            bad_z.append(cid)
    if bad_z:
        F.append(finding("NET006", ERROR, scope,
                         f"{len(bad_z)} line(s) with zero/invalid impedance",
                         bad_z[:20],
                         "zero-impedance branches make the voltage-drop "
                         "constraint degenerate"))

    # NET007 voltage limits present + sane; infeeder setpoint inside
    no_lim, inverted = [], []
    for nid, nd in nodes.items():
        ud = nd.get("user_data") or {}
        vmin, vmax = ud.get("v_min"), ud.get("v_max")
        if vmin is None or vmax is None:
            no_lim.append(nid)
        elif vmin >= vmax:
            inverted.append((nid, vmin, vmax))
    if no_lim:
        F.append(finding("NET007", WARN, scope,
                         f"{len(no_lim)} node(s) without v_min/v_max "
                         "(their voltage constraint is silently DROPPED — a "
                         "false pass, not a pass)",
                         no_lim[:20],
                         "add user_data.v_min/v_max (kV) or accept unlimited "
                         "voltage at these nodes"))
    if inverted:
        F.append(finding("NET007", ERROR, scope,
                         f"{len(inverted)} node(s) with v_min >= v_max",
                         inverted[:10], "swap or fix the limits"))
    if len(infeeders) == 1:
        inf = next(iter(infeeders.values()))
        nid = inf["cons"][0]["node"]
        nd = nodes.get(nid, {})
        ud = nd.get("user_data") or {}
        if ud.get("v_min") is not None and not (
                ud["v_min"] <= inf["v_setpoint"] <= ud["v_max"]):
            F.append(finding("NET007", ERROR, scope,
                             "infeeder setpoint outside its node's limits",
                             {"setpoint": inf["v_setpoint"], **ud},
                             "immediate infeasibility; fix the setpoint"))

    # NET008 v_base consistency along lines / across transformer
    vb_bad = []
    for cid, cd in lines.items():
        try:
            vb = [nodes[c["node"]]["v_base"] for c in cd["cons"]]
            if abs(vb[0] - vb[1]) > 1e-9:
                vb_bad.append((cid, vb))
        except KeyError:
            pass
    if vb_bad:
        F.append(finding("NET008", ERROR, scope,
                         f"{len(vb_bad)} line(s) join nodes of different v_base",
                         vb_bad[:10],
                         "a wrong base makes per-unit currents wrong by orders "
                         "of magnitude"))
    for cid, cd in txs.items():
        try:
            vb0 = nodes[cd["cons"][0]["node"]]["v_base"]
            vb1 = nodes[cd["cons"][1]["node"]]["v_base"]
            w0, w1 = cd["v_winding_base"]
            if abs(vb0 / vb1 - w0 / w1) / (w0 / w1) > 0.15:
                F.append(finding("NET008", WARN, scope,
                                 f"transformer {cid}: node v_base ratio "
                                 f"{vb0/vb1:.3f} vs winding ratio {w0/w1:.3f}",
                                 "", "check the voltage bases step correctly "
                                 "across the transformer"))
        except (KeyError, IndexError, ZeroDivisionError, TypeError):
            F.append(finding("NET011", ERROR, scope,
                             f"transformer {cid}: missing/invalid v_winding_base",
                             "", "add v_winding_base [kV_primary, kV_secondary]"))

    # NET009 explicit line i_max
    no_imax = [cid for cid, cd in lines.items() if "i_max" not in cd]
    if no_imax:
        F.append(finding("NET009", WARN, scope,
                         f"{len(no_imax)} line(s) without explicit i_max — the "
                         "solver falls back to 100 kA (i.e. NO limit at all)",
                         no_imax[:20],
                         "results look artificially feasible; add real ratings"))

    # NET010 explicit s_max
    no_smax = [cid for cid, cd in txs.items() if "s_max" not in cd]
    if no_smax:
        F.append(finding("NET010", ERROR, scope,
                         f"{len(no_smax)} transformer(s) without s_max — the "
                         "solver falls back to 1e9 W (no limit) and I_rated "
                         "cannot be derived",
                         no_smax, "add the nameplate rating"))

    # NET011 transformer fields
    for cid, cd in txs.items():
        missing = [k for k in ("nom_turns_ratio", "taps", "tap_factor",
                               "tap_side", "vector_group") if k not in cd]
        if missing:
            F.append(finding("NET011", ERROR, scope,
                             f"transformer {cid} missing fields {missing}", "",
                             "these crash _build_network_data with no context"))
        vg = cd.get("vector_group")
        if vg is not None and (not isinstance(vg, (list, tuple)) or len(vg) < 2
                               or vg[0] != vg[1]):
            F.append(finding("NET011", ERROR, scope,
                             f"transformer {cid}: vector_group {vg!r} — the "
                             "solver asserts vg[0] == vg[1]",
                             "", 'use e.g. ["YNyn0","YNyn0"] or "yy0" (str is '
                             "iterated per character and passes when uniform-"
                             "prefixed — prefer the 2-list form)"))

    # NET012 loads on nodes inside the network
    outside = [cid for cid, cd in loads.items()
               if cd["cons"][0].get("node") not in nodes]
    if outside:
        F.append(finding("NET012", ERROR, scope,
                         f"{len(outside)} Load(s) attached to nodes outside "
                         "the network — they are silently dropped by "
                         "_filter_input_data", outside[:20],
                         "reattach or remove them"))

    # NET013 diff against parent feeder
    if parent_feeder_ejson is not None:
        parent = _validate.components_by_type(parent_feeder_ejson)
        pn = set(parent["Node"])
        missing_nodes = [n for n in nodes if n not in pn]
        if missing_nodes:
            F.append(finding("NET013", WARN, scope,
                             f"{len(missing_nodes)} node(s) in the substation "
                             "subtree are not in the parent feeder",
                             missing_nodes[:20],
                             "extraction bug or missing LVNetwork XML"))

    if not F:
        F.append(finding("NET000", INFO, scope,
                         f"network structure OK: {n_nodes} nodes, {len(lines)} "
                         f"lines, {len(txs)} transformers, {len(loads)} loads"))
    return F


# ===========================================================================
# TS — timeseries
# ===========================================================================
def check_timeseries(df_long, network_load_ids, ambient=None,
                     min_coverage=0.90, scope="timeseries"):
    from . import timeseries as _ts
    F = []
    ids = df_long["load_id"].astype(str)

    # TS001 id match rate + nmi_ prefix detection
    mapping, rep = _ts.reconcile_load_ids(ids.unique(), network_load_ids)
    msg = (f"{rep['n_exact']}/{rep['n_ts_ids']} timeseries ids match network "
           f"loads exactly; {rep['n_prefix_fixed']} match after nmi_ prefix fixing")
    if rep["n_prefix_fixed"]:
        F.append(finding("TS001", WARN, scope, msg + " — prefix mismatch detected",
                         {"examples_fixed": dict(list(
                             {k: v for k, v in mapping.items() if k != v}.items())[:5]),
                          "ts_unmatched": rep["ts_unmatched"],
                          "network_unmatched": rep["network_unmatched"]},
                         "prepare_timeseries.py applies the fix automatically; "
                         "check the translator kept/stripped the prefix "
                         "consistently"))
    elif rep["n_exact"] == 0:
        F.append(finding("TS001", ERROR, scope,
                         "NO timeseries load ids match the network loads",
                         {"ts_examples": rep["ts_unmatched"],
                          "network_examples": rep["network_unmatched"]},
                         "ids must equal the Load component ids (nmi_<NMI>)"))
    else:
        F.append(finding("TS001", INFO, scope, msg,
                         {"ts_unmatched": rep["ts_unmatched"],
                          "network_unmatched": rep["network_unmatched"]}))

    # TS002 timestamps parse with explicit format
    fmt = _ts.detect_timestamp_format(df_long["timestamp"].astype(str).iloc[0]) \
        if not np.issubdtype(df_long["timestamp"].dtype, np.datetime64) else "already parsed"
    if fmt is None:
        F.append(finding("TS002", WARN, scope,
                         "timestamp format could not be inferred — parsing "
                         "falls back to slow per-element dateutil",
                         df_long['timestamp'].iloc[0],
                         "use ISO 8601 (2025-02-01T07:00)"))
    else:
        F.append(finding("TS002", INFO, scope, f"timestamp format: {fmt}"))

    ts = pd.to_datetime(df_long["timestamp"])

    # TS003 interval regularity + duplicates
    uniq = pd.Series(sorted(ts.unique()))
    d = uniq.diff().dropna().dt.total_seconds() / 60.0
    if len(d):
        dt_mode = d.mode().iat[0]
        gaps = d[d != dt_mode]
        if len(gaps):
            F.append(finding("TS003", WARN, scope,
                             f"irregular intervals: modal Δt {dt_mode:g} min but "
                             f"{len(gaps)} other gap(s) "
                             f"(min {gaps.min():g}, max {gaps.max():g} min)",
                             "", "the thermal model's dt is a constant — an "
                             "irregular series silently invalidates every "
                             "temperature; gaps become explicit zero-filled "
                             "steps in pre-indexing"))
        else:
            F.append(finding("TS003", INFO, scope,
                             f"regular Δt = {dt_mode:g} min, no gaps"))
    dup = df_long.duplicated(subset=["timestamp", "load_id"]).sum()
    if dup:
        F.append(finding("TS003", ERROR, scope,
                         f"{dup} duplicated (timestamp, load_id) pair(s)", "",
                         "deduplicate before pre-indexing"))

    # TS004 coverage per NMI
    n_steps = ts.nunique()
    cov = df_long.groupby("load_id")["timestamp"].nunique() / max(n_steps, 1)
    low = cov[cov < min_coverage]
    if len(low):
        F.append(finding("TS004", WARN, scope,
                         f"{len(low)} NMI(s) below {min_coverage:.0%} coverage",
                         low.sort_values().head(20).round(3).to_dict(),
                         "their gaps are filled with 0 W and masked"))

    # TS005 NaN / inf
    for col in ("real_power_w", "reactive_power_var"):
        vals = pd.to_numeric(df_long[col], errors="coerce")
        n_nan = int(vals.isna().sum())
        n_inf = int(np.isinf(vals.fillna(0)).sum())
        if n_nan or n_inf:
            F.append(finding("TS005", WARN, scope,
                             f"{col}: {n_nan} NaN/non-numeric, {n_inf} infinite",
                             "", "cleaned to 0 W and masked in pre-indexing"))

    # TS006 units
    for w in _ts.check_magnitudes(df_long):
        F.append(finding("TS006", WARN, scope, w, "",
                         "re-run prepare_timeseries with the right --values-are"))

    # TS007 ambient coverage
    if ambient is not None:
        amb = ambient.dropna()
        if len(amb) == 0:
            F.append(finding("TS007", ERROR, scope, "ambient series is empty",
                             "", "fetch or supply data/ambient"))
        else:
            if (amb.index.min() > ts.min()) or (amb.index.max() < ts.max()):
                F.append(finding("TS007", WARN, scope,
                                 "ambient series does not cover the full load "
                                 "timeseries range",
                                 {"ambient": [str(amb.index.min()), str(amb.index.max())],
                                  "loads": [str(ts.min()), str(ts.max())]},
                                 "edges are ffilled/bfilled — fetch the full range"))
    else:
        F.append(finding("TS007", INFO, scope,
                         "no ambient series supplied at preflight time "
                         "(constant θ_A assumed)"))
    return F


# ===========================================================================
# PHY — physical feasibility pre-screen
# ===========================================================================
def _tx_info(ejson):
    comps = _validate.components_by_type(ejson)
    out = {}
    s_units = ejson["units"]["power"]
    v_units = ejson["units"]["voltage"]
    for cid, cd in comps["Transformer"].items():
        s_max_w = cd.get("s_max", 1e9 / s_units) * s_units
        v_sec = cd["v_winding_base"][1] * v_units
        out[cid] = {"s_max_w": s_max_w, "v_sec_v": v_sec,
                    "i_rated_a": s_max_w / v_sec}
    return out


def check_physical(bundle, ejson, transformer_params=None, theta_A_const=25.0,
                   scope="physical"):
    """Cheap arithmetic on the pre-indexed arrays. High predictive value."""
    F = []
    P = bundle["P"].astype(np.float64)     # W, load-positive [T,N]
    Q = bundle["Q"].astype(np.float64)
    theta = bundle["theta_A"].astype(np.float64)
    if np.isnan(theta).all():
        theta = np.full(P.shape[0], theta_A_const)
    idx = pd.DatetimeIndex(bundle["timestamps"].view("datetime64[ns]"))
    dt_min = float(bundle["dt_minutes"])

    txs = _tx_info(ejson)
    if not txs:
        F.append(finding("PHY000", WARN, scope, "no transformer in network — "
                         "PHY001/003 skipped"))
        return F

    # PHY001 aggregate loading vs rating
    S_agg = np.hypot(P.sum(axis=1), Q.sum(axis=1))     # VA
    for tx_id, t in txs.items():
        K = S_agg / t["s_max_w"]
        n_over = int((K > 1.0).sum())
        worst = idx[np.argsort(-K)[:5]]
        msg = (f"{tx_id}: peak aggregate K = {K.max():.2f}, "
               f"{n_over}/{len(K)} interval(s) over rating")
        if n_over:
            F.append(finding("PHY001", WARN if K.max() < 1.5 else ERROR, scope,
                             msg, {"worst_timestamps": [str(w) for w in worst]},
                             "if the inflexible background load alone exceeds "
                             "the rating, no envelope choice is feasible — "
                             "expect violations (soft limits) at these times"))
        else:
            F.append(finding("PHY001", INFO, scope, msg))

    # PHY002 line loading (lossless downstream accumulation, I ≈ S/V_nom)
    comps = _validate.components_by_type(ejson)
    nodes = comps["Node"]
    v_units = ejson["units"]["voltage"]
    i_units = ejson["units"]["current"]
    # loads per node
    load_pos = {}
    lid_list = [str(x) for x in bundle["load_ids"]]
    lid_pos = {l: i for i, l in enumerate(lid_list)}
    for cid, cd in comps["Load"].items():
        if cid in lid_pos:
            load_pos.setdefault(cd["cons"][0]["node"], []).append(lid_pos[cid])
    # downstream node sets per line via BFS from infeeder
    inf = next(iter(comps["Infeeder"].values()), None)
    if inf is not None:
        root = inf["cons"][0]["node"]
        adj = defaultdict(list)
        for cid, kind, n0, n1 in _validate.branch_endpoints(ejson):
            adj[n0].append((n1, cid, kind))
            adj[n1].append((n0, cid, kind))
        parent_edge = {}
        order = [root]
        seen = {root}
        for cur in order:
            for nxt, cid, kind in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    parent_edge[nxt] = (cid, kind, cur)
                    order.append(nxt)
        # accumulate per-node subtree load indices bottom-up
        subtree = {n: list(load_pos.get(n, [])) for n in order}
        for n in reversed(order):
            if n == root:
                continue
            _, _, par = parent_edge[n]
            subtree[par].extend(subtree[n])
        n_lines_over = 0
        details = []
        for n, (cid, kind, par) in parent_edge.items():
            if kind != "Line":
                continue
            cd = comps["Line"][cid]
            if "i_max" not in cd:
                continue
            i_max_a = cd["i_max"] * i_units
            cols = subtree[n]
            if not cols:
                continue
            v_nom = nodes[n]["v_base"] * v_units
            S_line = np.hypot(P[:, cols].sum(axis=1), Q[:, cols].sum(axis=1))
            share = float((S_line / v_nom > i_max_a).mean())
            if share > 0:
                n_lines_over += 1
                if len(details) < 10:
                    details.append({"line": cid,
                                    "share_over": round(share, 4),
                                    "i_max_a": i_max_a})
        if n_lines_over:
            F.append(finding("PHY002", WARN, scope,
                             f"{n_lines_over} line(s) exceed their rating on "
                             "background load alone", details,
                             "expect current-limit violations on these branches"))
        else:
            F.append(finding("PHY002", INFO, scope,
                             "no line exceeds its rating on background load"))

    # PHY003 open-loop hot-spot on BAU currents + DTR preview
    if transformer_params is not None:
        for tx_id, t in txs.items():
            tp = dict(transformer_params)
            tp.setdefault("dt", dt_min)
            tp["I_rated"] = tp.get("I_rated") or t["i_rated_a"]
            K2 = (S_agg / t["s_max_w"]) ** 2 * (t["i_rated_a"] / tp["I_rated"]) ** 2
            traj = _thermal.open_loop_trajectory(tp, K2, theta)
            peak = max(traj)
            n_over = sum(1 for x in traj if x > tp["theta_HS_max"])
            msg = (f"{tx_id}: open-loop BAU peak θ_HS = {peak:.1f} °C, "
                   f"{n_over} interval(s) above the {tp['theta_HS_max']:g} °C limit")
            if n_over:
                F.append(finding("PHY003", WARN, scope, msg, "",
                                 "BAU already exceeds θ_HS_max — the DOE "
                                 "scenarios will curtail here; with soft "
                                 "limits the run proceeds and quantifies it"))
            else:
                F.append(finding("PHY003", INFO, scope, msg))
            # DTR preview: share of intervals with rating above nameplate
            dTO = dHS = 0.0
            c = _thermal.get_coeffs(tp)
            above = 0
            k2_nameplate = (t["i_rated_a"] / tp["I_rated"]) ** 2
            for k2, tha in zip(K2, theta):
                k2m, _st = _thermal.dtr_k2_max(tp, float(tha), dTO, dHS)
                if k2m > k2_nameplate:
                    above += 1
                dTO, dHS = c.step(float(k2), dTO, dHS)
            F.append(finding("PHY003", INFO, scope,
                             f"{tx_id}: dynamic rating above nameplate in "
                             f"{above}/{len(K2)} interval(s) "
                             f"({100*above/max(len(K2),1):.0f} % of the period)"))

    # PHY004 voltage screen at peak load and peak reverse flow
    if inf is not None:
        net_p = P.sum(axis=1)
        for label, t_idx in (("peak load", int(np.argmax(net_p))),
                             ("peak reverse flow", int(np.argmin(net_p)))):
            excursions = _voltage_screen(ejson, comps, parent_edge, order,
                                         subtree, P[t_idx], Q[t_idx], lid_list)
            if excursions:
                F.append(finding("PHY004", WARN, scope,
                                 f"voltage screen at {label} ({idx[t_idx]}): "
                                 f"{len(excursions)} node(s) outside limits "
                                 "before any envelope is applied",
                                 excursions[:10],
                                 "expect voltage-limit violations / tight "
                                 "envelopes at these times"))
            else:
                F.append(finding("PHY004", INFO, scope,
                                 f"voltage screen at {label}: all nodes inside limits"))

    # PHY005 transformer params sanity
    if transformer_params is not None:
        tp = transformer_params
        checks = []
        if not (tp["tau_TO"] > tp["tau_W"] > 0):
            checks.append(f"need tau_TO > tau_W > 0 (got {tp['tau_TO']}, {tp['tau_W']})")
        for k in ("n", "m"):
            if not (0 < tp[k] <= 1):
                checks.append(f"{k}={tp[k]} outside (0, 1]")
        if not tp["R"] > 0:
            checks.append(f"R={tp['R']} must be > 0")
        if not (100 <= tp["theta_HS_max"] <= 160):
            checks.append(f"theta_HS_max={tp['theta_HS_max']} outside [100, 160]")
        if tp.get("I_rated") is not None and tp["I_rated"] > 0:
            for tx_id, t in txs.items():
                ratio = tp["I_rated"] / t["i_rated_a"]
                if not (0.5 <= ratio <= 2.0):
                    checks.append(
                        f"I_rated {tp['I_rated']:.0f} A inconsistent with "
                        f"{tx_id} s_max/v_sec = {t['i_rated_a']:.0f} A "
                        f"(ratio {ratio:.2f}) — derive_i_rated is recommended")
        else:
            checks.append(f"I_rated={tp.get('I_rated')} must be > 0")
        if "dt" in tp and abs(tp["dt"] - dt_min) > 1e-6:
            F.append(finding("PHY005", ERROR, scope,
                             f"transformer params dt={tp['dt']:g} min but the "
                             f"timeseries modal Δt is {dt_min:g} min",
                             "", "a mismatch silently corrupts every thermal "
                             "result; the pipeline sets dt from the data"))
        if checks:
            F.append(finding("PHY005", ERROR, scope,
                             "transformer thermal parameters fail sanity checks",
                             checks, "fix config/transformers/*.yaml"))
        else:
            F.append(finding("PHY005", INFO, scope, "transformer params OK"))
    return F


def _voltage_screen(ejson, comps, parent_edge, order, subtree, p_row, q_row,
                    lid_list):
    """One-shot backward-forward sweep (lossless) — a screen, not a solve."""
    nodes = comps["Node"]
    v_units = ejson["units"]["voltage"]
    z_units = ejson["units"]["impedance"]
    s_base = 1.0e6
    inf = next(iter(comps["Infeeder"].values()))
    root = inf["cons"][0]["node"]
    v2 = {root: (inf["v_setpoint"] / nodes[root]["v_base"]) ** 2}

    excursions = []
    for n in order:
        if n == root:
            continue
        cid, kind, par = parent_edge[n]
        p_pu = sum(p_row[i] for i in subtree[n]) / s_base
        q_pu = sum(q_row[i] for i in subtree[n]) / s_base
        v_base_v = nodes[n]["v_base"] * v_units
        z_base = v_base_v ** 2 / s_base
        if kind == "Line":
            cd = comps["Line"][cid]
            r = cd["z"][0] * z_units * cd.get("length", 1.0) / z_base
            x = cd["z"][1] * z_units * cd.get("length", 1.0) / z_base
            ratio2 = 1.0
        else:
            cd = comps["Transformer"][cid]
            r = cd["z"][1][0] * z_units / z_base
            x = cd["z"][1][1] * z_units / z_base
            try:
                tr = cd["nom_turns_ratio"][0] * (1.0 + cd["taps"][0] * cd["tap_factor"])
                ratio2 = (tr * nodes[n]["v_base"] / nodes[par]["v_base"]) ** 2
            except (KeyError, IndexError, TypeError):
                ratio2 = 1.0
        v2_n = (v2[par] - 2 * (p_pu * r + q_pu * x)) / ratio2
        v2[n] = max(v2_n, 1e-6)
        ud = nodes[n].get("user_data") or {}
        if ud.get("v_min") is not None:
            vmin = ud["v_min"] / nodes[n]["v_base"]
            vmax = ud["v_max"] / nodes[n]["v_base"]
            v = v2[n] ** 0.5
            if v < vmin - 1e-3 or v > vmax + 1e-3:
                excursions.append({"node": n, "v_pu": round(v, 4),
                                   "limits": [round(vmin, 4), round(vmax, 4)]})
    return excursions


# ===========================================================================
# MDL — model setup
# ===========================================================================
def check_model(bundle, envelope_abs_max=50.0, soft_limits=True, scope="model"):
    F = []
    # MDL001 hard constraints
    if soft_limits:
        F.append(finding("MDL001", INFO, scope,
                         "soft voltage/current limits ON — over-limit "
                         "intervals become quantified violations in "
                         "viol.parquet instead of silently missing rows"))
    else:
        F.append(finding("MDL001", WARN, scope,
                         "soft limits OFF: only bus power balance has slacks; "
                         "a voltage or current violation returns infeasible "
                         "with no diagnostic and the timestep is skipped",
                         "", "run with --soft-limits (default ON)"))

    # MDL002 envelope_abs_max vs peak forecast
    P = np.abs(bundle["P"]).max(axis=0) / 1000.0   # kW per NMI
    over = {str(l): round(float(p), 2)
            for l, p in zip(bundle["load_ids"], P) if p > envelope_abs_max}
    if over:
        F.append(finding("MDL002", ERROR, scope,
                         f"envelope_abs_max = {envelope_abs_max:g} kW is "
                         f"smaller than the peak forecast of {len(over)} NMI(s)",
                         over,
                         "the envelope would be clipped by the parameter, not "
                         "the network — raise envelope_abs_max"))
    else:
        F.append(finding("MDL002", INFO, scope,
                         f"envelope_abs_max {envelope_abs_max:g} kW clears "
                         f"every NMI's peak forecast ({P.max():.1f} kW max)"))
    return F


# ===========================================================================
# Reporting
# ===========================================================================
def verdict(findings):
    n_err = sum(1 for f in findings if f["severity"] == ERROR)
    n_warn = sum(1 for f in findings if f["severity"] == WARN)
    if n_err:
        return f"BLOCKED ({n_err} error{'s' if n_err != 1 else ''})", n_err, n_warn
    if n_warn:
        return f"READY WITH WARNINGS ({n_warn})", n_err, n_warn
    return "READY", 0, 0


def print_findings(findings, log=print):
    v, n_err, n_warn = verdict(findings)
    log(f"preflight: {v}")
    for f in findings:
        if f["severity"] != INFO:
            log(f"  [{f['severity']:5s}] {f['id']} ({f['scope']}): {f['message']}")


def render_markdown(findings, title="Preflight report"):
    v, n_err, n_warn = verdict(findings)
    out = [f"# {title}", "", f"**Verdict: {v}**", ""]
    for sev in (ERROR, WARN, INFO):
        group = [f for f in findings if f["severity"] == sev]
        if not group:
            continue
        out.append(f"## {sev} ({len(group)})\n")
        for f in group:
            out.append(f"### {f['id']} — {f['scope']}\n")
            out.append(f"{f['message']}\n")
            if f["detail"]:
                out.append(f"```\n{f['detail']}\n```\n")
            if f["suggested_fix"]:
                out.append(f"*Fix:* {f['suggested_fix']}\n")
    return "\n".join(out)


def summary_row(substation, findings, bundle=None, ejson=None):
    """One row per substation for preflight_summary.csv."""
    v, n_err, n_warn = verdict(findings)
    row = {"substation": substation, "verdict": v.split(" (")[0],
           "n_errors": n_err, "n_warnings": n_warn}
    if ejson is not None:
        comps = _validate.components_by_type(ejson)
        row["n_loads"] = len(comps["Load"])
        row["is_radial"] = not _validate.find_cycles(ejson)
    if bundle is not None:
        row["n_matched_nmis"] = int(bundle["mask"].any(axis=0).sum())
        txi = _tx_info(ejson) if ejson is not None else {}
        if txi:
            t = next(iter(txi.values()))
            S = np.hypot(bundle["P"].sum(axis=1), bundle["Q"].sum(axis=1))
            row["peak_K"] = round(float(S.max() / t["s_max_w"]), 3)
            row["n_over_rating"] = int((S / t["s_max_w"] > 1).sum())
    for f in findings:
        if f["id"] == "PHY003" and "open-loop BAU peak" in f["message"]:
            try:
                row["peak_theta_HS_openloop"] = float(
                    f["message"].split("θ_HS = ")[1].split(" °C")[0])
            except (IndexError, ValueError):
                pass
            break
    return row
