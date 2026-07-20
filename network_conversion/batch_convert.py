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

Requires cim_to_network_json.py and extract_lv_network.py in the same
directory as this script.

Usage:
    python batch_convert.py xml_folder out_folder
    python batch_convert.py xml_folder out_folder --jobs 4
    python batch_convert.py xml_folder out_folder --no-extract --lv-vmin 0.39 --lv-vmax 0.44
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONVERTER = HERE / "cim_to_network_json.py"
EXTRACTOR = HERE / "extract_lv_network.py"

CIRCUIT_RE = re.compile(
    r'<cim:Circuit rdf:ID="([^"]+)">(.*?)</cim:Circuit>', re.S)
CTYPE_RE = re.compile(r"circuitType>([^<]*)<")
MASTER_RE = re.compile(r"isMaster>([^<]*)<")
CONNECTED_RE = re.compile(r"connectedCircuits>([^<\n]+)<")


def classify(path):
    """Return ('Feeder'|'LVNetwork'|None, circuit_id, lv_circuits_referenced)."""
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
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


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def convert_feeder(feeder_id, feeder_xml, lv_xmls, out_dir, sub_dir, args):
    """Convert one feeder and extract its substations. Returns report row."""
    row = {"feeder": feeder_id, "xml": feeder_xml.name,
           "n_lv_xmls": len(lv_xmls), "status": "", "nodes": "", "lines": "",
           "transformers": "", "loads": "", "warnings": "",
           "missing_lv_circuits": "", "substations_extracted": 0}
    out_json = out_dir / f"{feeder_id}_network.json"
    cmd = [sys.executable, str(CONVERTER), str(feeder_xml),
           *[str(p) for p in lv_xmls], "-o", str(out_json)]
    if args.lv_vmin:
        cmd += ["--lv-vmin", str(args.lv_vmin)]
    if args.lv_vmax:
        cmd += ["--lv-vmax", str(args.lv_vmax)]
    if args.v_setpoint_pu:
        cmd += ["--v-setpoint-pu", str(args.v_setpoint_pu)]
    rc, log = run(cmd)
    if rc != 0 or not out_json.exists():
        row["status"] = "CONVERT FAILED"
        row["warnings"] = log.strip()[-400:]
        return row

    net = json.loads(out_json.read_text())
    comps = net["components"]
    counts = Counter(next(iter(v)) for v in comps.values())
    row.update(status="ok", nodes=counts["Node"], lines=counts["Line"],
               transformers=counts["Transformer"], loads=counts["Load"])
    row["warnings"] = "; ".join(
        f"{k}={v}" for k, v in net["user_data"].get("conversion_warnings", {}).items())

    # which substations on this feeder have no LVNetwork XML in the input?
    have = {classify_cache.get(p, (None, "", None))[1] for p in lv_xmls}
    subs_on_feeder = {t["Transformer"]["user_data"].get("substation", "")
                      for t in comps.values() if "Transformer" in t}
    missing = sorted(s for s in subs_on_feeder
                     if s and f"{s}_LVNetwork" not in have)
    row["missing_lv_circuits"] = "; ".join(missing)

    if not args.no_extract:
        feeder_sub_dir = sub_dir / feeder_id
        feeder_sub_dir.mkdir(parents=True, exist_ok=True)
        for k, v in comps.items():
            if "Transformer" not in v:
                continue
            name = v["Transformer"].get("user_data", {}).get("name", k).replace(" ", "_")
            sub_out = feeder_sub_dir / f"{name}_lv_network.json"
            rc, log = run([sys.executable, str(EXTRACTOR), str(out_json), k,
                           "-o", str(sub_out)])
            if rc == 0:
                row["substations_extracted"] += 1
            else:
                row["warnings"] += f"; extract {k} failed"
    return row


classify_cache = {}


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

    in_dir, out = Path(args.input_dir), Path(args.output_dir)
    feeders_dir = out / "feeders"
    subs_dir = out / "substations"
    feeders_dir.mkdir(parents=True, exist_ok=True)
    subs_dir.mkdir(parents=True, exist_ok=True)

    xmls = sorted(in_dir.rglob("*.xml"))
    if not xmls:
        sys.exit(f"No .xml files found under {in_dir}")
    print(f"Scanning {len(xmls)} XML files...")

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

    print(f"Found {len(feeders)} feeders, {len(lv_files)} LV networks "
          f"({sum(len(v) for v in lv_by_feeder.values())} matched to feeders), "
          f"{len(skipped)} files skipped.")
    for name, why in skipped[:10]:
        print(f"  skipped: {name}: {why}")
    for name, cid, parent in unmatched_lv[:10]:
        print(f"  unmatched LV: {name} ({cid}) -> feeder '{parent}' not in input")

    rows = []
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as ex:
        futs = {ex.submit(convert_feeder, fid, fx, sorted(lv_by_feeder.get(fid, [])),
                          feeders_dir, subs_dir, args): fid
                for fid, fx in sorted(feeders.items())}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            print(f"[{i}/{len(futs)}] {row['feeder']}: {row['status']}, "
                  f"{row['loads']} loads, {row['substations_extracted']} substations extracted"
                  + (f", MISSING LV: {row['missing_lv_circuits'][:80]}" if row['missing_lv_circuits'] else ""))

    rows.sort(key=lambda r: r["feeder"])
    report = out / "batch_report.csv"
    with open(report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_subs = sum(r["substations_extracted"] for r in rows)
    n_missing = sum(1 for r in rows if r["missing_lv_circuits"])
    print(f"\nDone: {n_ok}/{len(rows)} feeders converted, {n_subs} substation LV networks extracted.")
    if n_missing:
        print(f"{n_missing} feeders have substations with no LVNetwork XML - see {report.name}.")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
