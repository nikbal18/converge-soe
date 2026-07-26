#!/usr/bin/env python3
"""
Stage ⑧ standalone: (re)build metrics tables, plots and sanity checks for an
existing run — never re-solves.

    python scripts/analyse_results.py out/GOLDCR_8HB_LEXCEN/20260726_101500
    python scripts/analyse_results.py out/F/RUN --plot-substations "S_5402_AT"
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from converge_soe import analysis, pipeline as pl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="out/<FEEDER>/<RUN_ID> directory")
    ap.add_argument("--plot-substations", default=None,
                    help="comma list of extra substations to plot in detail")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        sys.exit(f"{run_dir} not found")
    import yaml
    cfg_file = run_dir / "config_resolved.yaml"
    cfg = (yaml.safe_load(cfg_file.read_text()) if cfg_file.exists()
           else pl.load_config(REPO))
    feeder = run_dir.parent.name
    analysis.run_analysis(
        run_dir, cfg, feeder_name=feeder, log=print,
        plot_substations=(args.plot_substations.split(",")
                          if args.plot_substations else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
