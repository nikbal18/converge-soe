#!/usr/bin/env python3
"""
Batch-convert a folder of Schneider DMS CIM XML exports into converge-soe
network.json files.

Give it a folder containing any mix of feeder exports and LVNetwork exports
(any names, any nesting). It:
  1. classifies every XML by reading its cim:Circuit element,
  2. groups each LVNetwork export with its parent feeder,
  3. converts each feeder (feeder XML + all its LV XMLs merged) to
     <out>/feeders/<FEEDER>_network.json,
  4. extracts every distribution transformer's LV network to
     <out>/substations/<FEEDER>/<SUBSTATION>_lv_network.json,
  5. writes <out>/batch_report.csv listing, per feeder: component counts,
     conversion warnings, and any substations whose LVNetwork XML was
     missing from the input folder (so you know what to request).

Refactored from the original script version: conversion and extraction are
now direct function calls into converge_soe.network.cim_to_json and
converge_soe.network.extract_lv (no subprocesses — faster, and tracebacks
surface properly). The classify/group/convert steps are importable so
scripts/run_feeder.py can drive them programmatically.

Usage:
    python -m converge_soe.network.batch_convert xml_folder out_folder
    python -m converge_soe.network.batch_convert xml_folder out_folder --jobs 4
    python -m converge_soe.network.batch_convert xml_folder out_folder --no-extract --lv-vmin 0.39 --lv-vmax 0.44
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import cim_to_json, extract_lv

CIRCUIT_RE = re.compile(
    r'<cim:Circuit rdf:ID="([^"]+)">(.*?)</cim:Circuit>', re.S)
CTYPE_RE = re.compile(r"circuitType>([^<]*)<")
MASTER_RE = re.compile(r"isMaster>([^<]*)<")
CONNECTED_RE = re.compile(r"connectedCircuits>([^<\n]+)<")


def classify(path):
    """Return ('Feeder'|'LVNetwork'|None, circuit_id, lv_circuits_referenced)."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        return None, f"unreadable: {e}", set()
    kind = cid = None
    for m in CIRCUIT_RE.finditer(text[:400000]):
        body = m.group(2)
        ct = CTYPE_RE.search(body)
        ct = ct.group(1) if ct else None
        master = (MASTER_RE.search(body) or [None]) and (
            (MASTER_RE.search(body).group(1) if MASTER_RE.search(body) else "") == "True")
        if ct == "Feeder":
            kind, cid = "Feeder", m.group(1)
            break
        if ct == "LVNetwork" and master and kind is None:
            kind, cid = "LVNetwork", m.group(1)
    # circuits referenced via stitch nodes: "A#B" pairs anywhere in the file
    refs = set()
    for m in CONNECTED_RE.finditer(text):
        for part in m.group(1).split(","):
            part = part.strip()
            if "#" in part:
                a, b = part.split("#", 1)
                refs.add((a.strip(), b.strip()))
    return kind, cid, refs


def scan_folder(in_dir):
    """Classify every XML under ``in_dir`` and group LV exports with feeders.

    Returns dict with keys: feeders (id -> Path), lv_by_feeder
    (feeder id -> [Path]), classify_cache (Path -> (kind, cid, refs)),
    skipped ([(name, why)]), unmatched_lv ([(name, cid, parent)]).
    """
    in_dir = Path(in_dir)
    xmls = sorted(in_dir.rglob("*.xml"))

    classify_cache = {}
    feeders = {}                 # feeder circuit id -> path
    lv_files = {}                # lv circuit id -> path
    lv_parent = {}               # lv circuit id -> feeder circuit id
    feeder_expected = defaultdict(set)  # feeder id -> lv circuit ids it references
    skipped = []

    for x in xmls:
        kind, cid, refs = classify(x)
        classify_cache[x] = (kind, cid, refs)
        if kind == "Feeder":
            if cid in feeders:
                skipped.append((x.name, f"duplicate feeder {cid} (kept {feeders[cid].name})"))
            else:
                feeders[cid] = x
            for a, b in refs:
                if a == cid:
                    feeder_expected[cid].add(b)
        elif kind == "LVNetwork":
            if cid in lv_files:
                skipped.append((x.name, f"duplicate LV circuit {cid} (kept {lv_files[cid].name})"))
                continue
            lv_files[cid] = x
            parents = Counter(b for a, b in refs if a == cid)
            if parents:
                lv_parent[cid] = parents.most_common(1)[0][0]
        else:
            skipped.append((x.name, "no cim:Circuit found - not a DMS export?"))

    # attach LV files to feeders (by parent reference, else by feeder's own list)
    lv_by_feeder = defaultdict(list)
    unmatched_lv = []
    for cid, path in lv_files.items():
        parent = lv_parent.get(cid)
        if parent is None:
            parent = next((f for f, exp in feeder_expected.items() if cid in exp), None)
        if parent in feeders:
            lv_by_feeder[parent].append(path)
        else:
            unmatched_lv.append((path.name, cid, parent or "?"))

    return {
        "n_xmls": len(xmls),
        "feeders": feeders,
        "lv_by_feeder": {k: sorted(v) for k, v in lv_by_feeder.items()},
        "classify_cache": classify_cache,
        "skipped": skipped,
        "unmatched_lv": unmatched_lv,
    }


def convert_feeder(feeder_id, feeder_xml, lv_xmls, out_dir, sub_dir,
                   classify_cache=None, lv_vmin=None, lv_vmax=None,
                   v_setpoint_pu=None, no_extract=False):
    """Convert one feeder (direct function calls) and extract its substations.

    Returns a report row dict.
    """
    classify_cache = classify_cache or {}
    row = {"feeder": feeder_id, "xml": Path(feeder_xml).name,
           "n_lv_xmls": len(lv_xmls), "status": "", "nodes": "", "lines": "",
           "transformers": "", "loads": "", "warnings": "",
           "missing_lv_circuits": "", "substations_extracted": 0}
    out_dir = Path(out_dir)
    out_json = out_dir / f"{feeder_id}_network.json"

    kwargs = {}
    if lv_vmin is not None:
        kwargs["lv_vmin"] = lv_vmin
    if lv_vmax is not None:
        kwargs["lv_vmax"] = lv_vmax
    if v_setpoint_pu is not None:
        kwargs["v_setpoint_pu"] = v_setpoint_pu

    try:
        net, stats, warns = cim_to_json.convert(
            [feeder_xml, *lv_xmls], output=str(out_json), **kwargs)
    except Exception as e:  # keep going: one bad feeder must not kill the batch
        row["status"] = "CONVERT FAILED"
        row["warnings"] = f"{type(e).__name__}: {e}"[:400]
        return row

    comps = net["components"]
    counts = Counter(next(iter(v)) for v in comps.values())
    row.update(status="ok", nodes=counts["Node"], lines=counts["Line"],
               transformers=counts["Transformer"], loads=counts["Load"])
    row["warnings"] = "; ".join(
        f"{k}={v}" for k, v in net["user_data"].get("conversion_warnings", {}).items())

    # which substations on this feeder have no LVNetwork XML in the input?
    have = {classify_cache.get(Path(p), (None, "", None))[1] for p in lv_xmls}
    subs_on_feeder = {t["Transformer"]["user_data"].get("substation", "")
                      for t in comps.values() if "Transformer" in t}
    missing = sorted(s for s in subs_on_feeder
                     if s and f"{s}_LVNetwork" not in have)
    row["missing_lv_circuits"] = "; ".join(missing)

    if not no_extract:
        feeder_sub_dir = Path(sub_dir) / feeder_id
        feeder_sub_dir.mkdir(parents=True, exist_ok=True)
        for k, v in comps.items():
            if "Transformer" not in v:
                continue
            name = v["Transformer"].get("user_data", {}).get("name", k).replace(" ", "_")
            sub_out = feeder_sub_dir / f"{name}_lv_network.json"
            try:
                sub_ej = extract_lv.extract(net, k, source_name=out_json.name)
                with open(sub_out, "w") as f:
                    json.dump(sub_ej, f, indent=1)
                row["substations_extracted"] += 1
            except Exception as e:
                row["warnings"] += f"; extract {k} failed: {e}"
    return row


def run_batch(input_dir, output_dir, jobs=2, no_extract=False,
              lv_vmin=None, lv_vmax=None, v_setpoint_pu=None, log=print):
    """Convert every feeder found under ``input_dir``. Returns (rows, scan)."""
    out = Path(output_dir)
    feeders_dir = out / "feeders"
    subs_dir = out / "substations"
    feeders_dir.mkdir(parents=True, exist_ok=True)
    subs_dir.mkdir(parents=True, exist_ok=True)

    scan = scan_folder(input_dir)
    if scan["n_xmls"] == 0:
        raise FileNotFoundError(f"No .xml files found under {input_dir}")
    log(f"Scanning {scan['n_xmls']} XML files...")

    feeders = scan["feeders"]
    lv_by_feeder = scan["lv_by_feeder"]
    n_matched = sum(len(v) for v in lv_by_feeder.values())
    log(f"Found {len(feeders)} feeders, "
        f"{n_matched + len(scan['unmatched_lv'])} LV networks "
        f"({n_matched} matched to feeders), "
        f"{len(scan['skipped'])} files skipped.")
    for name, why in scan["skipped"][:10]:
        log(f"  skipped: {name}: {why}")
    for name, cid, parent in scan["unmatched_lv"][:10]:
        log(f"  unmatched LV: {name} ({cid}) -> feeder '{parent}' not in input")

    rows = []
    with ThreadPoolExecutor(max_workers=max(jobs, 1)) as ex:
        futs = {ex.submit(convert_feeder, fid, fx, lv_by_feeder.get(fid, []),
                          feeders_dir, subs_dir,
                          classify_cache=scan["classify_cache"],
                          lv_vmin=lv_vmin, lv_vmax=lv_vmax,
                          v_setpoint_pu=v_setpoint_pu,
                          no_extract=no_extract): fid
                for fid, fx in sorted(feeders.items())}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            log(f"[{i}/{len(futs)}] {row['feeder']}: {row['status']}, "
                f"{row['loads']} loads, {row['substations_extracted']} substations extracted"
                + (f", MISSING LV: {row['missing_lv_circuits'][:80]}" if row['missing_lv_circuits'] else ""))

    rows.sort(key=lambda r: r["feeder"])
    report = out / "batch_report.csv"
    if rows:
        with open(report, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_subs = sum(r["substations_extracted"] for r in rows)
    n_missing = sum(1 for r in rows if r["missing_lv_circuits"])
    log(f"\nDone: {n_ok}/{len(rows)} feeders converted, {n_subs} substation LV networks extracted.")
    if n_missing:
        log(f"{n_missing} feeders have substations with no LVNetwork XML - see {report.name}.")
    log(f"Report: {report}")
    return rows, scan


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", help="folder containing CIM XML exports (searched recursively)")
    p.add_argument("output_dir", help="folder for the network.json outputs")
    p.add_argument("--jobs", type=int, default=2, help="feeders converted in parallel (default 2)")
    p.add_argument("--no-extract", action="store_true", help="skip per-substation extraction")
    p.add_argument("--lv-vmin", type=float, default=None)
    p.add_argument("--lv-vmax", type=float, default=None)
    p.add_argument("--v-setpoint-pu", type=float, default=None)
    args = p.parse_args()

    try:
        run_batch(args.input_dir, args.output_dir, jobs=args.jobs,
                  no_extract=args.no_extract, lv_vmin=args.lv_vmin,
                  lv_vmax=args.lv_vmax, v_setpoint_pu=args.v_setpoint_pu)
    except FileNotFoundError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
