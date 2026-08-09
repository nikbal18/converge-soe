#!/usr/bin/env python3
"""Work out which feeder each target substation sits on, and print the
run_feeder commands.

run_feeder.py solves ONE feeder at a time. With 8 feeder XMLs and 38 LV
substation XMLs in data/xml, working out by hand which feeder carries which
substation is error-prone — grepping the feeder XMLs gives false positives
(a four-digit id is a prefix of a five-digit one) and misses substations
whose name is encoded differently.

This reads the BUILT feeder JSONs, which carry the authoritative
Transformer.user_data.substation, intersects them with the substations present
in a meter file, and prints a ready-to-paste command per feeder. Substations
with no meter data are reported and excluded, since solving them would produce
100% synthetic customers.

    python scripts/build_network.py            # must run first
    python tools/plan_feeder_runs.py --meter data/meter/<sliced>.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def substations_in_meter(path):
    """Substation column of a raw (headerless) or headed wide export."""
    probe = pd.read_csv(path, nrows=1, header=None, dtype=str,
                        encoding="utf-8-sig")
    headed = str(probe.iloc[0, 0]).strip().lower() == "nmi"
    if headed:
        d = pd.read_csv(path, usecols=["Substation"], dtype=str,
                        encoding="utf-8-sig")
        col = d["Substation"]
    else:
        d = pd.read_csv(path, header=None, usecols=[6], names=["Substation"],
                        dtype=str, encoding="utf-8-sig")
        col = d["Substation"]
    return sorted({s.strip() for s in col.dropna().unique()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meter", type=Path, required=True)
    ap.add_argument("--values-are", default="kwh_per_interval")
    ap.add_argument("--infeeder-kv", default="11.0")
    ap.add_argument("--envelope-abs-max", default="65")
    ap.add_argument("--jobs", default="8")
    ap.add_argument("--tag", default="top20",
                    help="run-id prefix; the feeder name is appended")
    ap.add_argument("--extra", default="",
                    help="extra flags to append to every command")
    args = ap.parse_args()

    fdir = REPO / "build" / "network" / "feeders"
    jsons = sorted(fdir.glob("*_network.json")) if fdir.exists() else []
    if not jsons:
        raise SystemExit(f"no built feeders in {fdir} — run "
                         f"scripts/build_network.py first")

    targets = set(substations_in_meter(args.meter))
    print(f"{len(targets)} substation(s) present in {args.meter.name}:")
    print("  " + ", ".join(sorted(targets)) + "\n")

    found, plans = set(), []
    for p in jsons:
        ej = json.loads(p.read_text(encoding="utf-8"))
        feeder = ej.get("user_data", {}).get("feeder", p.stem.replace("_network", ""))
        subs = set()
        for comp in ej["components"].values():
            t = comp.get("Transformer")
            if not t:
                continue
            ud = t.get("user_data", {}) or {}
            name = (ud.get("substation") or ud.get("name") or "").strip()
            if name:
                subs.add(name)
        hit = sorted(subs & targets)
        found |= set(hit)
        if hit:
            plans.append((feeder, hit, len(subs)))

    missing = sorted(targets - found)
    if missing:
        print(f"!! {len(missing)} substation(s) in the meter file are in NO "
              f"built feeder — check the XML is present and the build ran:")
        print("   " + ", ".join(missing) + "\n")

    print("=" * 70)
    print(f"{len(plans)} feeder(s) to run, {len(found)} target substation(s)")
    print("=" * 70)
    for feeder, hit, total in plans:
        only = ",".join(hit)
        print(f"\n# {feeder}: {len(hit)} of {total} substations "
              f"({', '.join(hit)})")
        print(f"python scripts/run_feeder.py --feeder {feeder} \\")
        print(f"  --meter {args.meter.as_posix()} \\")
        print(f"  --values-are {args.values_are} \\")
        print(f"  --only \"{only}\" \\")
        print(f"  --infeeder-kv {args.infeeder_kv} "
              f"--envelope-abs-max {args.envelope_abs_max} "
              f"--jobs {args.jobs} \\")
        print(f"  --run-id {args.tag}_{feeder}"
              + (f" {args.extra}" if args.extra else ""))

    n_runs = sum(len(h) for _, h, _ in plans) * 3
    print(f"\n# {n_runs} substation-scenario solves in total. For scale: 54 "
          f"solves\n# over 144 intervals took ~60 min with --jobs 8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
