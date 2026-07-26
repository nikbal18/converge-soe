#!/usr/bin/env python3
"""
Stage ① standalone: convert every CIM XML in data/xml/ to network models.

    python scripts/build_network.py                # build (cached)
    python scripts/build_network.py --list-feeders # what did we find?

Wraps converge_soe.network.batch_convert; results land in build/network/.
The batch report's missing_lv_circuits column lists the LVNetwork exports
you still need to request.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from converge_soe import pipeline as pl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-feeders", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args()
    if args.force:
        m = pl.read_manifest(REPO)
        m.pop("network", None)
        import json
        pl._manifest_path(REPO).write_text(json.dumps(m))
    out = pl.stage_build_network(REPO, pl.load_config(REPO), log=print)
    if args.list_feeders:
        df = pl.list_feeders(REPO)
        print(df.to_string(index=False) if len(df) else "no feeders found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
