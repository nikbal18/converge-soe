#!/usr/bin/env python3
"""
Extract the LV network below one distribution transformer from a
converge-soe network.json (ejson) file.

Starting at the chosen transformer's secondary (LV) node, walks the network
through Lines (never through another Transformer), and writes a new
network.json containing:
  * the transformer itself, its MV primary node, and a new Infeeder at the
    primary node (so the file is directly usable by the SOE/DOE solver), and
  * every Node, Line and Load at or below the transformer secondary.

The transformer can be identified by component id, exact user_data.name
(e.g. "S 5402 AT"), or a unique substring (e.g. "5402" or "S 5402").
Use --list to see all transformers in the file.

Usage:
    python -m converge_soe.network.extract_lv network.json --list
    python -m converge_soe.network.extract_lv network.json "S 5402" -o S5402_lv_network.json
    python -m converge_soe.network.extract_lv network.json "S 5402" --no-tx   # LV only,
        # Infeeder placed directly on the LV secondary node instead.
"""

import argparse
import json
import sys
from collections import defaultdict


def comp_type(entry):
    return next(iter(entry))


def find_transformer(comps, query):
    txs = {k: v["Transformer"] for k, v in comps.items() if "Transformer" in v}
    if query in txs:
        return query
    q = query.lower()
    exact = [k for k, t in txs.items()
             if t.get("user_data", {}).get("name", "").lower() == q]
    if len(exact) == 1:
        return exact[0]
    sub = [k for k, t in txs.items()
           if q in k.lower() or q in t.get("user_data", {}).get("name", "").lower()]
    if len(sub) == 1:
        return sub[0]
    if not sub:
        raise ValueError(f"no transformer matches '{query}'")
    opts = ", ".join(f"{k} ({txs[k].get('user_data', {}).get('name', '?')})" for k in sub)
    raise ValueError(f"'{query}' is ambiguous: {opts}")


def extract_components(comps, tx_id, no_tx=False, v_setpoint_pu=None):
    """Extract the LV subtree components below transformer ``tx_id``.

    Returns the components dict for a standalone, solvable substation network:
    the LV subtree plus (unless ``no_tx``) the transformer, its MV node, and a
    fresh Infeeder on the MV node.
    """
    tx = comps[tx_id]["Transformer"]
    primary_node, secondary_node = tx["cons"][0]["node"], tx["cons"][1]["node"]

    # Adjacency through Lines only; other Transformers are barriers.
    adj = defaultdict(set)
    for k, v in comps.items():
        if "Line" in v:
            c = v["Line"]["cons"]
            adj[c[0]["node"]].add(c[1]["node"])
            adj[c[1]["node"]].add(c[0]["node"])

    # BFS down from the secondary node.
    keep_nodes = {secondary_node}
    queue = [secondary_node]
    while queue:
        cur = queue.pop(0)
        for nxt in adj[cur]:
            if nxt == primary_node:
                continue  # don't walk back up through the MV side
            if nxt not in keep_nodes:
                keep_nodes.add(nxt)
                queue.append(nxt)

    out_comps = {}
    for k, v in comps.items():
        tp = comp_type(v)
        cd = v[tp]
        if tp == "Node":
            if k in keep_nodes:
                out_comps[k] = v
        elif tp == "Line":
            nds = [c["node"] for c in cd["cons"]]
            if all(n in keep_nodes for n in nds):
                out_comps[k] = v
        elif tp == "Load":
            if cd["cons"][0]["node"] in keep_nodes:
                out_comps[k] = v

    # Infeeder voltage setpoint: preserve the source network's pu setpoint.
    src_inf = next((v["Infeeder"] for v in comps.values() if "Infeeder" in v), None)
    if v_setpoint_pu is not None:
        pu = v_setpoint_pu
    elif src_inf:
        src_node = comps[src_inf["cons"][0]["node"]]["Node"]
        pu = src_inf["v_setpoint"] / src_node["v_base"]
    else:
        pu = 1.05

    if no_tx:
        v_base = comps[secondary_node]["Node"]["v_base"]
        out_comps[f"infeeder_{tx_id.lstrip('_')}"] = {"Infeeder": {
            "cons": [{"node": secondary_node, "phs": tx["cons"][1].get("phs", ["A"])}],
            "v_setpoint": round(v_base * pu, 6),
        }}
    else:
        out_comps[tx_id] = comps[tx_id]
        out_comps[primary_node] = comps[primary_node]
        v_base = comps[primary_node]["Node"]["v_base"]
        out_comps[f"infeeder_{tx_id.lstrip('_')}"] = {"Infeeder": {
            "cons": [{"node": primary_node, "phs": tx["cons"][0].get("phs", ["A"])}],
            "v_setpoint": round(v_base * pu, 6),
        }}
    return out_comps


def extract(ejson, transformer, no_tx=False, v_setpoint_pu=None, source_name=None):
    """Importable API: return a full substation ejson for ``transformer``.

    ``transformer`` may be a component id, exact user_data.name, or unique
    substring. Raises ValueError if no unique match.
    """
    comps = ejson["components"]
    tx_id = find_transformer(comps, transformer)
    tx = comps[tx_id]["Transformer"]
    tx_name = tx.get("user_data", {}).get("name", tx_id)

    out = {k: v for k, v in ejson.items() if k != "components"}
    out.setdefault("user_data", {})
    out["user_data"] = dict(out["user_data"] or {})
    if source_name:
        out["user_data"]["extracted_from"] = str(source_name)
    out["user_data"]["transformer"] = {"id": tx_id, "name": tx_name,
                                       "s_max": tx.get("s_max")}
    out["components"] = extract_components(comps, tx_id, no_tx=no_tx,
                                           v_setpoint_pu=v_setpoint_pu)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("network", help="input network.json (ejson)")
    p.add_argument("transformer", nargs="?", help="transformer id, name, or unique substring")
    p.add_argument("-o", "--output", default=None, help="output path (default <name>_lv_network.json)")
    p.add_argument("--list", action="store_true", help="list transformers and exit")
    p.add_argument("--no-tx", action="store_true",
                   help="exclude the transformer; put the Infeeder on the LV secondary node")
    p.add_argument("--v-setpoint-pu", type=float, default=None,
                   help="infeeder setpoint in pu (default: same pu as the source network's infeeder)")
    args = p.parse_args()

    with open(args.network) as f:
        ej = json.load(f)
    comps = ej["components"]

    if args.list or not args.transformer:
        print(f"{'id':40s} {'name':16s} {'s_max':>7s}  secondary node")
        for k, v in comps.items():
            if "Transformer" in v:
                t = v["Transformer"]
                print(f"{k:40s} {t.get('user_data', {}).get('name', '?'):16s} "
                      f"{t.get('s_max', '?'):>7}  {t['cons'][1]['node']}")
        return

    try:
        out = extract(ej, args.transformer, no_tx=args.no_tx,
                      v_setpoint_pu=args.v_setpoint_pu, source_name=args.network)
    except ValueError as e:
        sys.exit(f"ERROR: {e}. Use --list to see transformers.")

    tx_info = out["user_data"]["transformer"]
    out_path = args.output or f"{tx_info['name'].replace(' ', '_')}_lv_network.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)

    oc = out["components"]
    n_nodes = sum(1 for v in oc.values() if "Node" in v)
    n_lines = sum(1 for v in oc.values() if "Line" in v)
    n_loads = sum(1 for v in oc.values() if "Load" in v)
    print(f"Transformer: {tx_info['id']} ({tx_info['name']}), s_max={tx_info['s_max']}")
    print(f"LV subtree: {n_nodes} nodes, {n_lines} lines, {n_loads} loads"
          + ("" if args.no_tx else " (+ transformer, MV node, infeeder)"))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
