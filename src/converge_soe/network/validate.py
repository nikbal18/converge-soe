"""Structural validation of ejson network models.

Pure-graph checks used by converge_soe.preflight (NET001–NET013). Kept here so
they can also be run standalone right after conversion, before any timeseries
work:

    python -m converge_soe.network.validate network.json
"""

import json
import sys
from collections import defaultdict


def components_by_type(ejson):
    out = defaultdict(dict)
    for cid, entry in ejson["components"].items():
        for ctype, cd in entry.items():
            out[ctype][cid] = cd
    return out


def branch_endpoints(ejson):
    """[(branch_id, kind, node0, node1)] for every Line and Transformer."""
    res = []
    comps = components_by_type(ejson)
    for kind in ("Line", "Transformer"):
        for cid, cd in comps[kind].items():
            cons = cd.get("cons", [])
            n0 = cons[0].get("node") if len(cons) > 0 else None
            n1 = cons[1].get("node") if len(cons) > 1 else None
            res.append((cid, kind, n0, n1))
    return res


def adjacency(ejson):
    adj = defaultdict(set)
    for cid, kind, n0, n1 in branch_endpoints(ejson):
        if n0 is not None and n1 is not None:
            adj[n0].add(n1)
            adj[n1].add(n0)
    return adj


def reachable_from(ejson, start):
    adj = adjacency(ejson)
    seen = {start}
    queue = [start]
    while queue:
        cur = queue.pop(0)
        for nxt in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def find_cycles(ejson):
    """Return a list of cycles (each a list of branch ids) in the branch graph.

    Uses union-find over branch endpoints: any branch joining two already-
    connected nodes closes a cycle. Parallel branches count as cycles too.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cycles = []
    for cid, kind, n0, n1 in branch_endpoints(ejson):
        if n0 is None or n1 is None:
            continue
        r0, r1 = find(n0), find(n1)
        if r0 == r1:
            cycles.append(cid)
        else:
            parent[r0] = r1
    return cycles


def infeeder_nodes(ejson):
    comps = components_by_type(ejson)
    return {cid: cd["cons"][0]["node"] for cid, cd in comps["Infeeder"].items()}


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m converge_soe.network.validate network.json")
    with open(sys.argv[1]) as f:
        ej = json.load(f)

    # Import here to avoid a hard dependency loop at module import time.
    from converge_soe import preflight
    findings = preflight.check_network(ej)
    preflight.print_findings(findings)
    n_err = sum(1 for f in findings if f["severity"] == "ERROR")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
