#!/usr/bin/env python3
"""
Convert a Schneider DMS CIM RDF XML feeder export (EvoEnergy style) to a
converge-soe network.json ("ejson") file.

Produces the same processed form as the reference feeder ejson files
(e.g. _GOLDCR_8HB_LEXCEN-rad-sp-ot.json):
  * single-phase equivalent (all components phs ["A"])
  * open switches break the topology; de-energised islands are removed
  * closed switches become short low-impedance Lines (user_data.is_switch)
  * remaining loops are broken by dropping a closed-switch/line edge (warned)
  * an Infeeder is placed at the feeder-head node (stitchingInfo == "FH")

Loads:
  The DMS XML export contains no customer/NMI objects. By default a
  placeholder Load is created at every LV dead-end node so the network is
  immediately usable. Alternatively supply --loads loads.csv with columns:
      nmi          (required) meter id -> component id "nmi_<nmi>"
      substation   (optional) e.g. "S 5406" - attach within that substation's LV
      node         (optional) exact node id to attach to (wins over substation)
  NMIs are spread across the LV dead-end nodes of their substation.

Usage:
    python cim_to_network_json.py feeder.xml -o network.json
    python cim_to_network_json.py feeder.xml -o network.json --loads loads.csv
    python cim_to_network_json.py feeder.xml --lv-vmin 0.39 --lv-vmax 0.44
"""

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "cim": "http://iec.ch/TC57/2017/CIM-schema-cim16#",
    "sedms": "http://www.schneider-electric-dms.com/CIM16v33/2017/extension",
    "ee": "http://www.schneider-electric-dms.com/CIM16v33/EvoEnergy/extension",
}

# ACT Standard Grid (Mt Stromlo) origin, used to build the xy->latlong map.
ACT_GRID_FALSE_E = 200000.0
ACT_GRID_FALSE_N = 600000.0
ACT_GRID_LAT0 = -35.31773627
ACT_GRID_LON0 = 148.99792792
M_PER_DEG_LAT = 111320.0

# Conductor resistivity, ohm.mm^2/m (used only when the export's per-length
# impedance catalogue entry is zero/missing).
RESISTIVITY = {"aluminum": 0.028264, "aluminium": 0.028264, "copper": 0.017241}
DEFAULT_X_PER_M = 8.0e-5  # 0.08 ohm/km reactance fallback
SWITCH_LENGTH_M = 0.5
SWITCH_Z_PER_M = (1.0e-4, 1.0e-5)  # negligible impedance for closed switches

SWITCH_TAGS = ("Fuse", "Disconnector", "LoadBreakSwitch", "Breaker", "Jumper", "Sectionaliser", "Recloser")


def q(prefix, tag):
    return f"{{{NS[prefix]}}}{tag}"


def text_of(el, prefix, tag):
    c = el.find(q(prefix, tag))
    return c.text if c is not None else None


def ref_of(el, prefix, tag):
    c = el.find(q(prefix, tag))
    if c is None:
        return None
    r = c.get(q("rdf", "resource"))
    return r.lstrip("#") if r else None


def voltage_from_description(desc):
    """Parse '...:11,000 V' / 'In Service:400' style descriptions -> volts."""
    if not desc:
        return None
    m = re.search(r"([\d,]+)\s*V?\s*$", desc.replace(",", ""))
    if m:
        try:
            v = float(m.group(1))
            if 100 <= v <= 500000:
                return v
        except ValueError:
            pass
    return None


def parse_cim(path):
    tree = ET.parse(path)
    root = tree.getroot()
    data = {
        "circuit": None,
        "nodes": {},          # id -> dict
        "terminals": [],      # (equip_id, node_id, seq)
        "lines": {},
        "switches": {},
        "transformers": {},
        "tx_ends": defaultdict(dict),      # tx_id -> {endNumber: dict}
        "mesh_z": {},                      # tx_id -> dict
        "tap_changers": {},                # tx_end ref -> dict
        "pli": {},                         # id -> (r, x, r0, x0) per metre
        "wire_info": {},                   # id -> dict
        "feeder_objects": {},              # container id -> name (substation)
        "do_points": defaultdict(list),    # diagram-object id -> [(seq,x,y)]
        "do_target": {},                   # diagram-object id -> equipment id
        "terminal_node": {},               # terminal id -> node id
        "service_locs": {},                # service location id -> dict
        "usage_points": {},                # nmi -> dict
        "ders": [],                        # PV / battery info
        "circuit_master": False,
    }

    for el in root:
        tag = el.tag.split("}")[-1]
        eid = el.get(q("rdf", "ID"))
        if eid is None:
            continue

        if tag == "Circuit":
            ctype = text_of(el, "sedms", "Circuit.circuitType")
            is_master = (text_of(el, "sedms", "Circuit.isMaster") or "") == "True"
            if data["circuit"] is None or (is_master and not data["circuit_master"]) or ctype == "Feeder":
                data["circuit"] = {"id": eid, "name": text_of(el, "cim", "IdentifiedObject.name"),
                                   "type": ctype}
                data["circuit_master"] = is_master or ctype == "Feeder"

        elif tag == "ConnectivityNode":
            data["nodes"][eid] = {
                "name": text_of(el, "cim", "IdentifiedObject.name"),
                "desc": text_of(el, "cim", "IdentifiedObject.description"),
                "v": voltage_from_description(text_of(el, "cim", "IdentifiedObject.description")),
                "container": ref_of(el, "cim", "ConnectivityNode.ConnectivityNodeContainer"),
                "stitching": text_of(el, "sedms", "ConnectivityNode.stitchingInfo"),
                "is_ground": eid.startswith("GroundNode")
                or (text_of(el, "cim", "IdentifiedObject.description") == "BusnodeForGrounding"
                    and text_of(el, "sedms", "ConnectivityNode.stitchingInfo") != "FH"),
            }

        elif tag == "Terminal":
            equip = ref_of(el, "cim", "Terminal.ConductingEquipment")
            node = ref_of(el, "cim", "Terminal.ConnectivityNode")
            seq = text_of(el, "cim", "ACDCTerminal.sequenceNumber")
            if equip and node:
                data["terminals"].append((equip, node, int(seq or 1)))
                data["terminal_node"][eid] = node

        elif tag == "ACLineSegment":
            data["lines"][eid] = {
                "name": text_of(el, "cim", "IdentifiedObject.name"),
                "desc": text_of(el, "cim", "IdentifiedObject.description"),
                "v": voltage_from_description(text_of(el, "cim", "IdentifiedObject.description")),
                "length": float(text_of(el, "cim", "Conductor.length") or 0.0),
                "pli": ref_of(el, "cim", "ACLineSegment.PerLengthImpedance"),
                "wire": ref_of(el, "cim", "PowerSystemResource.AssetDatasheet"),
                "container": ref_of(el, "cim", "Equipment.EquipmentContainer"),
                "in_service": text_of(el, "sedms", "IdentifiedObject.serviceState") != "OutOfService",
            }

        elif tag in SWITCH_TAGS:
            if eid.startswith("GeneratedEarthSwitch"):
                continue
            data["switches"][eid] = {
                "kind": tag,
                "name": text_of(el, "cim", "IdentifiedObject.name"),
                "v": voltage_from_description(text_of(el, "cim", "IdentifiedObject.description")),
                "normal_open": (text_of(el, "cim", "Switch.normalOpen") or "false").lower() == "true",
                "status": text_of(el, "sedms", "Switch.extendedStatus"),
                "rated_a": float(text_of(el, "sedms", "Switch.ratedCurrent") or 0.0),
                "container": ref_of(el, "cim", "Equipment.EquipmentContainer"),
                "in_service": text_of(el, "sedms", "IdentifiedObject.serviceState") != "OutOfService",
            }

        elif tag == "PowerTransformer":
            data["transformers"][eid] = {
                "name": text_of(el, "cim", "IdentifiedObject.name"),
                "container": ref_of(el, "cim", "Equipment.EquipmentContainer"),
                "in_service": text_of(el, "sedms", "IdentifiedObject.serviceState") != "OutOfService",
            }

        elif tag == "PowerTransformerEnd":
            tx = ref_of(el, "cim", "PowerTransformerEnd.PowerTransformer")
            endn = int(text_of(el, "cim", "TransformerEnd.endNumber") or 1)
            data["tx_ends"][tx][endn] = {
                "id": eid,
                "terminal": ref_of(el, "cim", "TransformerEnd.Terminal"),
                "rated_s": float(text_of(el, "cim", "PowerTransformerEnd.ratedS") or 0.0),
                "rated_u": float(text_of(el, "cim", "PowerTransformerEnd.ratedU") or 0.0),
                "conn": text_of(el, "cim", "PowerTransformerEnd.connectionKind") or "Y",
                "clock": int(text_of(el, "cim", "PowerTransformerEnd.phaseAngleClock") or 0),
                "neutral_tap": int(text_of(el, "sedms", "TransformerEnd.neutralTap") or 1),
                "normal_tap": int(text_of(el, "sedms", "TransformerEnd.normalTap") or 1),
                "n_taps": int(text_of(el, "sedms", "TransformerEnd.numberOfTaps") or 1),
                "tap_pct": float(text_of(el, "sedms", "TransformerEnd.tapPercent") or 0.0),
            }

        elif tag == "TransformerMeshImpedance":
            tx = (ref_of(el, "cim", "TransformerMeshImpedance.FromTransformerEnd") or "").rsplit(".W", 1)[0]
            data["mesh_z"][tx] = {
                "z_pct": float(text_of(el, "sedms", "TransformerMeshImpedance.leakageImpedancePercent") or 0.0),
                "load_loss_kw": float(text_of(el, "sedms", "TransformerMeshImpedance.loadLoss") or 0.0),
                "no_load_loss_kw": float(text_of(el, "sedms", "TransformerMeshImpedance.noLoadLoss") or 0.0),
            }

        elif tag == "RatioTapChanger":
            end = ref_of(el, "cim", "RatioTapChanger.TransformerEnd")
            data["tap_changers"][end] = {
                "low": int(text_of(el, "cim", "TapChanger.lowStep") or 1),
                "high": int(text_of(el, "cim", "TapChanger.highStep") or 1),
                "neutral": int(text_of(el, "cim", "TapChanger.neutralStep") or 1),
                "normal": int(text_of(el, "cim", "TapChanger.normalStep") or 1),
            }

        elif tag == "PerLengthSequenceImpedance":
            data["pli"][eid] = (
                float(text_of(el, "cim", "PerLengthSequenceImpedance.r") or 0.0),
                float(text_of(el, "cim", "PerLengthSequenceImpedance.x") or 0.0),
                float(text_of(el, "cim", "PerLengthSequenceImpedance.r0") or 0.0),
                float(text_of(el, "cim", "PerLengthSequenceImpedance.x0") or 0.0),
            )

        elif tag == "WireInfo":
            data["wire_info"][eid] = {
                "material": (text_of(el, "cim", "WireInfo.material") or "").lower(),
                "size_mm2": float(text_of(el, "cim", "WireInfo.sizeDescription") or 0.0),
                "rated_a": float(text_of(el, "cim", "WireInfo.ratedCurrent") or 0.0),
            }

        elif tag == "ServiceLocation":
            data["service_locs"][eid] = {
                "terminal": ref_of(el, "cim", "ServiceLocation.Terminal"),
                "desc": text_of(el, "cim", "IdentifiedObject.description"),
                "name": text_of(el, "cim", "IdentifiedObject.name"),
            }

        elif tag == "UsagePoint":
            data["usage_points"][eid] = {
                "service_loc": ref_of(el, "cim", "UsagePoint.ServiceLocation"),
                "p_w": float(text_of(el, "sedms", "UsagePoint.p") or 0.0),
                "q_var": float(text_of(el, "sedms", "UsagePoint.q") or 0.0),
                "load_group": text_of(el, "sedms", "UsagePoint.loadGroup"),
                "phase": text_of(el, "cim", "UsagePoint.phaseCode"),
            }

        elif tag in ("DistributedGenerator", "EnergyStorage"):
            nm = text_of(el, "cim", "IdentifiedObject.name") or ""
            data["ders"].append({
                "nmi": nm.split(":")[0],
                "kind": "PV" if tag == "DistributedGenerator" else "Battery",
                "type": (text_of(el, "cim", "IdentifiedObject.description") or "").split(":")[-1],
            })

        elif tag == "FeederObject":
            data["feeder_objects"][eid] = text_of(el, "cim", "IdentifiedObject.name")

        elif tag == "DiagramObject":
            tgt = ref_of(el, "cim", "DiagramObject.IdentifiedObject")
            if tgt:
                data["do_target"][eid] = tgt

        elif tag == "DiagramObjectPoint":
            do = ref_of(el, "cim", "DiagramObjectPoint.DiagramObject")
            seq = int(text_of(el, "cim", "DiagramObjectPoint.sequenceNumber") or 0)
            x = float(text_of(el, "cim", "DiagramObjectPoint.xPosition") or 0.0)
            y = float(text_of(el, "cim", "DiagramObjectPoint.yPosition") or 0.0)
            if do:
                data["do_points"][do].append((seq, x, y))

    return data


def merge_data(datas):
    """Merge parsed CIM files; graphs join on shared ConnectivityNode ids."""
    base = datas[0]
    for d in datas[1:]:
        for key in ("nodes", "lines", "switches", "transformers", "mesh_z",
                    "tap_changers", "pli", "wire_info", "feeder_objects",
                    "do_target", "terminal_node", "service_locs", "usage_points"):
            for k, v in d[key].items():
                base[key].setdefault(k, v)
        for tx, ends in d["tx_ends"].items():
            base["tx_ends"].setdefault(tx, {}).update(ends)
        seen = set(base["terminals"])
        base["terminals"].extend(t for t in d["terminals"] if t not in seen)
        for k, v in d["do_points"].items():
            if k not in base["do_points"]:
                base["do_points"][k] = v
        base["ders"].extend(d["ders"])
        if base["circuit"] is None or (
                not base["circuit_master"] and d["circuit_master"]):
            base["circuit"], base["circuit_master"] = d["circuit"], d["circuit_master"]
    return base


def equipment_coords(data):
    """equipment id -> ordered [(x, y), ...] from its Geographic diagram object."""
    coords = {}
    for do_id, target in data["do_target"].items():
        pts = sorted(data["do_points"].get(do_id, []))
        if pts:
            coords[target] = [(x, y) for _, x, y in pts]
    return coords


def line_impedance_per_m(line, data, warnings):
    """Return ((r, x), (r0, x0)) per metre and a list of inferred fields."""
    inferred = []
    pli = data["pli"].get(line["pli"]) if line["pli"] else None
    r = x = r0 = x0 = 0.0
    if pli:
        r, x, r0, x0 = pli
    if r <= 0.0 and x <= 0.0:
        wi = data["wire_info"].get(line["wire"]) if line["wire"] else None
        if wi and wi["size_mm2"] > 0:
            rho = RESISTIVITY.get(wi["material"], RESISTIVITY["aluminum"])
            # 20C DC resistivity corrected to ~45C operating temperature
            r = 1.10 * rho / wi["size_mm2"]
            x = DEFAULT_X_PER_M
            inferred = ["z"]
        else:
            r, x = 4.0e-4, DEFAULT_X_PER_M  # generic fallback
            inferred = ["z"]
            warnings["no_z"] += 1
    if r0 <= 0.0 and x0 <= 0.0:
        r0, x0 = r, x
    return (r, x), (r0, x0), inferred


def build_network(data, args):
    warnings = defaultdict(int)
    coords = equipment_coords(data)
    node_v = {}
    nodes_used = set()

    # equipment -> [(node, terminal_seq)], node -> equipment
    equip_nodes = defaultdict(list)
    for equip, node, seq in data["terminals"]:
        equip_nodes[equip].append((node, seq))
    for equip in equip_nodes:
        equip_nodes[equip].sort(key=lambda t: t[1])

    live_nodes = {nid for nid, nd in data["nodes"].items() if not nd["is_ground"]}

    # ---- build edges -------------------------------------------------------
    edges = []  # (edge_id, n0, n1, kind, payload)

    for lid, ln in data["lines"].items():
        nds = [n for n, _ in equip_nodes.get(lid, []) if n in live_nodes]
        if len(nds) != 2 or not ln["in_service"]:
            warnings["line_skipped"] += 0 if ln["in_service"] else 1
            if len(nds) != 2:
                warnings["line_bad_terminals"] += 1
            continue
        edges.append((lid, nds[0], nds[1], "line", ln))

    for sid, sw in data["switches"].items():
        nds = [n for n, _ in equip_nodes.get(sid, []) if n in live_nodes]
        if len(nds) != 2:
            continue  # earth switches / single-ended
        is_open = sw["normal_open"] or (sw["status"] == "Open")
        if is_open or not sw["in_service"]:
            warnings["open_switches"] += 1
            continue
        edges.append((sid, nds[0], nds[1], "switch", sw))

    tx_edges = []
    for tid, tx in data["transformers"].items():
        ends = data["tx_ends"].get(tid, {})
        if len(ends) < 2 or not tx["in_service"]:
            warnings["tx_skipped"] += 1
            continue
        # Tx terminals reference the tx via ConductingEquipment; terminal
        # sequence number order corresponds to winding end number order.
        nds = [n for n, _ in equip_nodes.get(tid, []) if n in live_nodes]
        if len(nds) < 2:
            warnings["tx_skipped"] += 1
            continue
        edges.append((tid, nds[0], nds[1], "tx", tx))
        tx_edges.append(tid)

    # ---- merge parallel edges (per-phase R/W/B devices, parallel cables) ---
    # In the single-phase equivalent, parallel devices between the same node
    # pair collapse to one edge (prefer keeping a real line over a switch).
    by_pair = defaultdict(list)
    for e in edges:
        by_pair[frozenset((e[1], e[2]))].append(e)
    merged = []
    for pair, group in by_pair.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        group.sort(key=lambda e: {"line": 0, "tx": 1, "switch": 2}[e[3]])
        merged.append(group[0])
        warnings["parallel_edges_merged"] += len(group) - 1
    edges = merged

    # ---- connectivity from feeder head ------------------------------------
    fh = None
    for nid, nd in data["nodes"].items():
        if nd["stitching"] == "FH":
            fh = nid
            break
    if fh is None and args.source_node:
        fh = args.source_node
    if fh is None:
        sys.exit("ERROR: no feeder-head node (stitchingInfo=FH) found. "
                 "Use --source-node to specify the infeeder node id.")

    adj = defaultdict(list)
    for i, (eid, n0, n1, kind, _) in enumerate(edges):
        adj[n0].append((n1, i))
        adj[n1].append((n0, i))

    # BFS spanning tree; drop closed edges that would close a loop.
    visited = {fh}
    kept = set()
    dropped_loops = []
    queue = [fh]
    while queue:
        cur = queue.pop(0)
        for nxt, ei in adj[cur]:
            if ei in kept or ei in {d[0] for d in dropped_loops}:
                continue
            if nxt in visited:
                dropped_loops.append((ei, edges[ei][0]))
                continue
            visited.add(nxt)
            kept.add(ei)
            queue.append(nxt)

    n_unreached = sum(1 for e in edges if e[1] not in visited and e[2] not in visited)
    warnings["deenergised_edges_dropped"] = n_unreached
    warnings["loop_edges_dropped"] = len(dropped_loops)

    # ---- voltage propagation ----------------------------------------------
    # Node voltage: from node description where available, else propagate
    # across kept non-tx edges from neighbours.
    for nid in visited:
        node_v[nid] = data["nodes"].get(nid, {}).get("v")
    changed = True
    while changed:
        changed = False
        for ei in kept:
            eid, n0, n1, kind, payload = edges[ei]
            if kind == "tx":
                continue
            v0, v1 = node_v.get(n0), node_v.get(n1)
            ev = payload.get("v")
            fill = v0 or v1 or ev
            if fill:
                for n in (n0, n1):
                    if not node_v.get(n):
                        node_v[n] = fill
                        changed = True

    # Transformer side voltages force their nodes.
    for ei in kept:
        eid, n0, n1, kind, tx = edges[ei]
        if kind != "tx":
            continue
        ends = data["tx_ends"][eid]
        u1 = ends[min(ends)]["rated_u"]
        u2 = ends[max(ends)]["rated_u"]
        node_v[n0] = node_v.get(n0) or u1
        node_v[n1] = node_v.get(n1) or u2

    # ---- coordinates -------------------------------------------------------
    node_xy = {}
    for ei in kept:
        eid, n0, n1, kind, _ = edges[ei]
        pts = coords.get(eid)
        if not pts:
            # switches: try composite switch/feeder object container coords
            cont = edges[ei][4].get("container")
            pts = coords.get(cont)
        if not pts:
            continue
        node_xy.setdefault(n0, pts[0])
        node_xy.setdefault(n1, pts[-1])
    # fill any missing from neighbours
    for _ in range(3):
        for ei in kept:
            _, n0, n1, _, _ = edges[ei]
            if n0 in node_xy and n1 not in node_xy:
                node_xy[n1] = node_xy[n0]
            elif n1 in node_xy and n0 not in node_xy:
                node_xy[n0] = node_xy[n1]

    def to_latlong(xy):
        lat = ACT_GRID_LAT0 + (xy[1] - ACT_GRID_FALSE_N) / M_PER_DEG_LAT
        lon = ACT_GRID_LON0 + (xy[0] - ACT_GRID_FALSE_E) / (
            M_PER_DEG_LAT * math.cos(math.radians(ACT_GRID_LAT0)))
        return [lat, lon]

    # ---- emit components ---------------------------------------------------
    comps = {}
    PH = ["A"]

    def nid_out(n):
        return "_" + n

    mv_v = max((v for v in node_v.values() if v), default=11000.0)

    used_nodes = set()
    for ei in kept:
        _, n0, n1, _, _ = edges[ei]
        used_nodes.update((n0, n1))

    for n in sorted(used_nodes):
        v = node_v.get(n) or mv_v
        nd = {
            "phs": PH,
            "v_base": round(v / 1000.0, 6),
        }
        if n in node_xy:
            nd["xy"] = [node_xy[n][0] - ACT_GRID_FALSE_E, node_xy[n][1] - ACT_GRID_FALSE_N]
            nd["lat_long"] = to_latlong(node_xy[n])
        if v < 1000.0:
            if args.lv_vmin and args.lv_vmax:
                # Explicit absolute band (kV) overrides everything.
                nd["user_data"] = {"v_min": args.lv_vmin, "v_max": args.lv_vmax}
            elif args.lv_voltage_tolerance and args.lv_voltage_tolerance > 0:
                # Default: +/- tolerance around a nominal voltage. If
                # --lv-nominal-v is given (phase volts) it references that;
                # otherwise each node's own base voltage is the nominal.
                tol = args.lv_voltage_tolerance
                if args.lv_nominal_v:
                    nom_kv = args.lv_nominal_v * math.sqrt(3) / 1000.0  # phase V -> line-line kV
                else:
                    nom_kv = round(v / 1000.0, 6)
                nd["user_data"] = {"v_min": round(nom_kv * (1 - tol), 6),
                                   "v_max": round(nom_kv * (1 + tol), 6)}
        comps[nid_out(n)] = {"Node": nd}

    # Infeeder
    v_fh = (node_v.get(fh) or mv_v) / 1000.0
    comps["_" + (data["circuit"]["id"] if data["circuit"] else "INFEEDER")] = {
        "Infeeder": {
            "cons": [{"node": nid_out(fh), "phs": PH}],
            "v_setpoint": round(v_fh * args.v_setpoint_pu, 6),
        }
    }

    for ei in sorted(kept):
        eid, n0, n1, kind, payload = edges[ei]
        cons = [{"node": nid_out(n0), "phs": PH}, {"node": nid_out(n1), "phs": PH}]

        if kind == "line":
            (r, x), (r0, x0), inferred = line_impedance_per_m(payload, data, warnings)
            ln = {
                "cons": cons,
                "in_service": True,
                "length": max(payload["length"], 0.1),
                "z": [r, x],
                "z0": [r0, x0],
                "user_data": {"line_type": "line", "name": payload["name"]},
            }
            wi = data["wire_info"].get(payload["wire"]) if payload["wire"] else None
            if wi and wi["rated_a"] > 0:
                ln["i_max"] = wi["rated_a"] / 1000.0  # current units: kA
            if inferred:
                ln["user_data"]["inferred"] = inferred
            comps[nid_out(eid)] = {"Line": ln}

        elif kind == "switch":
            sw = {
                "cons": cons,
                "in_service": True,
                "length": SWITCH_LENGTH_M,
                "z": list(SWITCH_Z_PER_M),
                "z0": list(SWITCH_Z_PER_M),
                "user_data": {"is_switch": True, "obj_type": payload["kind"],
                              "name": payload["name"]},
            }
            if payload["rated_a"] > 0:
                sw["i_max"] = payload["rated_a"] / 1000.0
            comps[nid_out(eid)] = {"Line": sw}

        elif kind == "tx":
            ends = data["tx_ends"][eid]
            e1, e2 = ends[min(ends)], ends[max(ends)]
            u1, u2 = e1["rated_u"], e2["rated_u"]
            s_mva = (e1["rated_s"] or e2["rated_s"]) / 1e6
            mesh = data["mesh_z"].get(eid, {})
            z_pct = mesh.get("z_pct", 0.0)
            ll_kw = mesh.get("load_loss_kw", 0.0)
            # impedance in ohms referred to the secondary winding
            if s_mva > 0 and z_pct > 0:
                z_base = (u2 / 1000.0) ** 2 / s_mva  # kV^2/MVA = ohm
                z_pu = z_pct / 100.0
                r_pu = min((ll_kw / 1000.0) / s_mva if s_mva else 0.0, z_pu * 0.9)
                x_pu = math.sqrt(max(z_pu ** 2 - r_pu ** 2, 0.0))
                z_sec = [r_pu * z_base, x_pu * z_base]
            else:
                z_sec = [1e-6, 1e-6]
                warnings["tx_z_default"] += 1

            tc = data["tap_changers"].get(e1["id"]) or data["tap_changers"].get(e2["id"])
            if tc:
                tap_range = [tc["low"] - tc["neutral"], tc["high"] - tc["neutral"]]
                tap = tc["normal"] - tc["neutral"]
            else:
                tap_range = [e1["neutral_tap"] - e1["normal_tap"], e1["n_taps"] - e1["normal_tap"]]
                tap = e1["normal_tap"] - e1["neutral_tap"]

            sub = data["feeder_objects"].get(data["transformers"][eid]["container"], "")
            vg_actual = f'{e1["conn"]}{e2["conn"].lower()}{e2["clock"]}'

            comps[nid_out(eid)] = {"Transformer": {
                "cons": cons,
                "in_service": True,
                "nom_turns_ratio": [u1 / u2, 0.0],
                "s_max": s_mva,
                "tap_factor": (e1["tap_pct"] or 1.25) / 100.0,
                "tap_range": tap_range,
                "tap_side": "primary",
                "taps": [tap],
                "v_winding_base": [u1 / 1000.0, u2 / 1000.0],
                "vector_group": "yy0",  # solver requires vg[0]==vg[1]
                "z": [[0.0, 0.0], z_sec],
                "user_data": {
                    "name": (sub + " " + (data["transformers"][eid]["name"] or "")).strip(),
                    "substation": sub,
                    "s_rated": s_mva,
                    "vector_group_actual": vg_actual,
                    "inferred": ["vector_group"],
                    "normal_tap": tc["normal"] if tc else e1["normal_tap"],
                    "tap": tc["normal"] if tc else e1["normal_tap"],
                },
            }}

    # ---- loads --------------------------------------------------------------
    # LV dead-end nodes grouped by substation (via nearest tx subtree is done in
    # the extractor; here group by voltage only, plus substation via container).
    degree = defaultdict(int)
    for ei in kept:
        _, n0, n1, _, _ = edges[ei]
        degree[n0] += 1
        degree[n1] += 1
    tx_nodes = set()
    for ei in kept:
        eid, n0, n1, kind, _ = edges[ei]
        if kind == "tx":
            tx_nodes.update((n0, n1))
    lv_deadends = [n for n in used_nodes
                   if degree[n] == 1 and (node_v.get(n) or mv_v) < 1000.0
                   and n not in tx_nodes and n != fh]

    def sub_of_node(n):
        cont = data["nodes"].get(n, {}).get("container")
        return data["feeder_objects"].get(cont, "")

    ders_by_nmi = defaultdict(list)
    for der in data["ders"]:
        ders_by_nmi[der["nmi"]].append(der)

    n_loads = 0
    if data["usage_points"]:
        for up_id, up in sorted(data["usage_points"].items()):
            sl = data["service_locs"].get(up["service_loc"] or "", {})
            node = data["terminal_node"].get(sl.get("terminal") or "")
            if node not in used_nodes:
                warnings["load_unplaced"] += 1
                continue
            # DMS UsagePoint ids are the 10-digit market NMI plus an 11th
            # checksum digit. Strip it (unless --keep-nmi-checksum) so load ids
            # match interval-meter NMIs. Keep the raw id in user_data.
            if not args.keep_nmi_checksum and len(up_id) == 11 and up_id.isdigit():
                nmi = up_id[:10]
            else:
                nmi = up_id
            ud = {"nmi": nmi, "nmis": [nmi], "usage_point_id": up_id,
                  "load_group": up["load_group"],
                  "type": sl.get("desc"), "phase": up["phase"]}
            if ders_by_nmi.get(up_id):
                ud["der"] = ders_by_nmi[up_id]
            comps[f"nmi_{nmi}"] = {"Load": {
                "cons": [{"node": nid_out(node), "phs": PH}],
                "in_service": True,
                "s_nom": [[round(up["p_w"] / 1e6, 8), round(up["q_var"] / 1e6, 8)]],
                "wiring": "wye",
                "user_data": ud,
            }}
            n_loads += 1
    elif args.loads:
        by_sub = defaultdict(list)
        for n in lv_deadends:
            by_sub[sub_of_node(n)].append(n)
        rr = defaultdict(int)
        with open(args.loads, newline="") as f:
            for row in csv.DictReader(f):
                nmi = row.get("nmi") or row.get("NMI")
                if not nmi:
                    continue
                node = row.get("node")
                if not node:
                    sub = (row.get("substation") or row.get("transformer") or "").strip()
                    cands = by_sub.get(sub) or lv_deadends
                    if not cands:
                        warnings["load_unplaced"] += 1
                        continue
                    node = cands[rr[sub] % len(cands)]
                    rr[sub] += 1
                    node = nid_out(node)
                comps[f"nmi_{nmi}"] = {"Load": {
                    "cons": [{"node": node, "phs": PH}],
                    "in_service": True,
                    "s_nom": [[0.001, 0.0]],
                    "wiring": "wye",
                    "user_data": {"nmi": nmi, "nmis": [nmi]},
                }}
                n_loads += 1
    elif not args.no_placeholder_loads:
        for n in lv_deadends:
            comps[f"load{nid_out(n)}"] = {"Load": {
                "cons": [{"node": nid_out(n), "phs": PH}],
                "in_service": True,
                "s_nom": [[0.001, 0.0]],
                "wiring": "wye",
                "user_data": {"placeholder": True, "substation": sub_of_node(n)},
            }}
            n_loads += 1

    # ---- assemble ------------------------------------------------------------
    cos0 = math.cos(math.radians(ACT_GRID_LAT0))
    ejson = {
        "map": {
            "A": [[0.0, 1.0 / M_PER_DEG_LAT], [1.0 / (M_PER_DEG_LAT * cos0), 0.0]],
            "A_inv": [[0.0, M_PER_DEG_LAT * cos0], [M_PER_DEG_LAT, 0.0]],
            "b": [ACT_GRID_LAT0, ACT_GRID_LON0],
            "points": None,
            "user_data": {"docs": "latlong = A xy + b (xy = ACT Standard Grid - false origin)"},
        },
        "units": {"current": 1000.0, "energy": 1000000.0, "impedance": 1,
                  "length": 1, "power": 1000000.0, "voltage": 1000.0},
        "user_data": {
            "base_voltages": {"LV": 415, "LV0": 433, "MV": int(mv_v)},
            "source_xml": [str(x) for x in args.xml],
            "feeder": data["circuit"]["id"] if data["circuit"] else None,
            "feeder_name": data["circuit"]["name"] if data["circuit"] else None,
            "conversion_warnings": {k: v for k, v in warnings.items() if v},
            "loop_edges_dropped": [e for _, e in dropped_loops],
        },
        "voltage_type": "lg",
        "components": comps,
    }

    stats = {
        "Nodes": sum(1 for c in comps.values() if "Node" in c),
        "Lines": sum(1 for c in comps.values() if "Line" in c),
        "Transformers": sum(1 for c in comps.values() if "Transformer" in c),
        "Loads": n_loads,
        "Infeeders": 1,
    }
    return ejson, stats, dict(warnings)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("xml", nargs="+",
                   help="CIM RDF XML export(s): feeder XML plus any LV network XMLs")
    p.add_argument("-o", "--output", default=None, help="output network.json path")
    p.add_argument("--loads", default=None, help="CSV of loads (nmi[,substation][,node])")
    p.add_argument("--no-placeholder-loads", action="store_true",
                   help="don't auto-create loads at LV dead-end nodes")
    p.add_argument("--v-setpoint-pu", type=float, default=1.05,
                   help="infeeder voltage setpoint in pu (default 1.05)")
    p.add_argument("--lv-vmin", type=float, default=None,
                   help="explicit LV node v_min in kV, e.g. 0.36 (absolute override)")
    p.add_argument("--lv-vmax", type=float, default=None,
                   help="explicit LV node v_max in kV, e.g. 0.44 (absolute override)")
    p.add_argument("--lv-voltage-tolerance", type=float, default=0.10,
                   help="LV nodes get v_min/v_max = nominal*(1-/+tol). "
                        "Default 0.10 (+/-10%%). Pass 0 to disable voltage limits.")
    p.add_argument("--lv-nominal-v", type=float, default=None,
                   help="LV nominal PHASE voltage in V to centre the tolerance band "
                        "on (e.g. 230). Default: each node's own base voltage.")
    p.add_argument("--source-node", default=None,
                   help="node id of feeder head if no stitchingInfo=FH node exists")
    p.add_argument("--keep-nmi-checksum", action="store_true",
                   help="keep the full 11-digit UsagePoint id instead of "
                        "stripping the trailing NMI checksum digit")
    args = p.parse_args()

    data = merge_data([parse_cim(x) for x in args.xml])
    ejson, stats, warnings = build_network(data, args)

    out = args.output
    if not out:
        feeder = (data["circuit"]["id"] if data["circuit"] else "network")
        out = f"{feeder}_network.json"
    with open(out, "w") as f:
        json.dump(ejson, f, indent=1)

    print(f"Feeder: {ejson['user_data']['feeder']} ({ejson['user_data']['feeder_name']})")
    print("Components:", ", ".join(f"{k}={v}" for k, v in stats.items()))
    if warnings:
        print("Warnings:", ", ".join(f"{k}={v}" for k, v in warnings.items()))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
