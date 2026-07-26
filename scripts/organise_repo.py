#!/usr/bin/env python3
"""
One-shot, idempotent repository reorganisation for converge-soe.

Moves the organically-grown flat layout into the lifecycle-based layout
described in docs/LAYOUT.md:

  src/converge_soe/    the library (all real logic)
  scripts/             thin CLI wrappers
  config/              hand-written configuration
  data/                inputs you provide (gitignored)
  build/               derived, deletable
  out/                 results, keep forever
  examples/legacy/     the original runner scripts (still work from there)
  tools/               one-off utilities
  archive/             old generated outputs and anything without a home

Behaviour:
  * DRY-RUN BY DEFAULT — prints a table of every planned action and does
    nothing. Pass --apply to actually perform the moves.
  * Idempotent — running it twice reports "nothing to do". A source that no
    longer exists is skipped silently; a source whose destination already
    exists (e.g. the refactored copy is already in place) is moved to
    archive/superseded/ instead so no work is ever overwritten or lost.
  * Uses `git mv` when the file is tracked in a working git checkout (so
    history is preserved) and plain shutil.move otherwise.

Usage:
    python scripts/organise_repo.py            # show the plan
    python scripts/organise_repo.py --apply    # do it
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Windows consoles/pipes default to cp1252, which cannot encode the non-ASCII
# characters used in a few notes. Force UTF-8 so output survives redirection
# and subprocess capture (this is what broke tests/test_organise_repo.py).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


# ---------------------------------------------------------------------------
# The plan. (src_relative, dst_relative_or_None_for_delete, note)
# Globs allowed in src. dst ending in '/' means "into this directory".
# ---------------------------------------------------------------------------
MOVES = [
    # network_conversion → library package (renamed) --------------------------
    ("network_conversion/cim_to_network_json.py",
     "src/converge_soe/network/cim_to_json.py", "renamed; importable convert() API"),
    ("network_conversion/batch_convert.py",
     "src/converge_soe/network/batch_convert.py", "refactored: functions, not subprocesses"),
    ("network_conversion/extract_lv_network.py",
     "src/converge_soe/network/extract_lv.py", "renamed; importable extract() API"),
    ("network_conversion/work_computer_setup_guide.docx",
     "docs/setup/", "setup doc"),
    ("network_conversion/README.md",
     "archive/network_conversion_README.md", "superseded by docs/"),
    ("network_conversion/*.json",
     "archive/network_conversion_json/", "old generated outputs, not source"),

    # runnable examples → legacy ---------------------------------------------
    ("examples/run_scenario.py", "examples/legacy/", "legacy SOE runner"),
    ("examples/run_doe_scenario.py", "examples/legacy/", "legacy DOE single-step"),
    ("examples/run_doe_multistep.py", "examples/legacy/", "legacy DOE multistep"),
    ("examples/run_doe_feeder.py", "examples/legacy/", "legacy feeder driver"),
    ("examples/RUN_DOE_GUIDE.md", "examples/legacy/", "legacy manual guide"),
    ("examples/disaggregate_transformer.py", "tools/", "utility, not an example"),
    ("examples/scenario_2/data_translation/wide_to_long_translator.py",
     "examples/legacy/", "logic folded into converge_soe.timeseries; original kept"),

    # real meter data → data/ (gitignored inputs) ----------------------------
    ("examples/scenario_2/data_translation/lexcen_data.csv",
     "data/meter/", "raw NMI export — confidential, gitignored"),
    ("examples/scenario_2/data_translation/project_format.csv",
     "data/meter/", "raw NMI export — confidential, gitignored"),
    ("examples/scenario_2/forecast_timeseries.csv",
     "data/meter/", "derived from real NMI data — gitignored"),
    ("examples/scenario_2/network.json",
     "archive/reference_outputs/scenario_2_network.json", "real-data network"),

    # old generated outputs → regression baselines ---------------------------
    ("examples/scenario_1_output", "archive/reference_outputs/scenario_1_output", "baseline"),
    ("examples/scenario_1_doe_output", "archive/reference_outputs/scenario_1_doe_output", "baseline"),
    ("examples/scenario_doe_output", "archive/reference_outputs/scenario_doe_output", "baseline"),
    ("examples/scenario_doe_multistep_output",
     "archive/reference_outputs/scenario_doe_multistep_output", "baseline"),
    ("results", "archive/reference_outputs/results", "baseline"),

    # misplaced binaries ------------------------------------------------------
    ("bin/micromamba", "archive/bin/", "a binary does not belong in the repo"),
]

# transformer params: converted (JSON→YAML) rather than moved; the JSON stays
# because examples/legacy/run_doe_multistep.py reads it from the scenario dir.
CONVERT_JSON_YAML = (
    "examples/scenario_doe/transformer_params.json",
    "config/transformers/distribution_onan.yaml",
)

DELETES = [
    "output.log",                       # stale crash log (contents preserved in docs/TROUBLESHOOTING.md)
    "examples/scenario_2/break",        # stale
]

DELETE_GLOBS = [
    "**/__pycache__",
    "src/*.egg-info",
]

MKDIRS = [
    "src/converge_soe/network", "scripts", "config/transformers",
    "config/feeders", "data/xml", "data/meter", "data/mapping",
    "data/ambient", "docs", "docs/setup", "examples/legacy", "tools",
    "tests", "archive",
]


# ---------------------------------------------------------------------------
def is_git_checkout():
    try:
        r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                           cwd=REPO, capture_output=True, text=True, timeout=15)
        return r.returncode == 0 and r.stdout.strip() == "true"
    except (OSError, subprocess.TimeoutExpired):
        return False


def git_tracked(path):
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch",
                            str(path.relative_to(REPO))],
                           cwd=REPO, capture_output=True, text=True, timeout=15)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return False


def do_move(src, dst, use_git):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_git and git_tracked(src):
        r = subprocess.run(["git", "mv", str(src.relative_to(REPO)),
                            str(dst.relative_to(REPO))],
                           cwd=REPO, capture_output=True, text=True)
        if r.returncode == 0:
            return "git mv"
        # fall through to plain move (e.g. dst ignored by git)
    shutil.move(str(src), str(dst))
    return "move"


def convert_params_json_to_yaml(src, dst):
    import json
    p = json.loads(src.read_text())
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(f"""\
# Distribution transformer, ONAN cooling — IEEE C57.91 thermal parameters.
# Converted from examples/scenario_doe/transformer_params.json by
# scripts/organise_repo.py. One file per transformer class; select via
# config/default.yaml -> thermal.transformer_class.
#
# Sources: IEEE Std C57.91-2011 Annex G typical values for small ONAN
# distribution transformers; matched to the values used in the honours
# project to date. Review against the actual EvoEnergy fleet data sheets.

tau_TO: {p['tau_TO']}                # min — top-oil thermal time constant
tau_W: {p['tau_W']}                  # min — winding (hot-spot) thermal time constant
delta_theta_TO_R: {p['delta_theta_TO_R']}      # °C — top-oil rise over ambient at rated load
delta_theta_HS_R: {p['delta_theta_HS_R']}      # °C — hot-spot rise over top-oil at rated load
R: {p['R']}                     # — ratio of rated load loss to no-load loss
n: {p['n']}                     # — top-oil exponent (0.8 ONAN, 0.9 ONAF, 1.0 OFAF)
m: {p['m']}                     # — winding exponent  (0.8 ONAN, 0.9-1.0 forced)
I_rated: {p['I_rated']}              # A — rated secondary current; the pipeline normally
                            #     overrides this per substation from s_max/v_secondary
theta_HS_max: {p['theta_HS_max']}         # °C — hot-spot temperature limit (110 normal life,
                            #     120 planned loading beyond nameplate — C57.91 Table 8)
dt: {p['dt']}                    # min — timestep; MUST equal the timeseries interval.
                            #     The pipeline overrides this from the data and
                            #     preflight PHY005 errors on a mismatch.
""")


def plan():
    """Return a list of (action, src, dst, how, note) tuples."""
    actions = []
    use_git = is_git_checkout()

    for d in MKDIRS:
        p = REPO / d
        if not p.exists():
            actions.append(("mkdir", None, p, "mkdir", ""))

    for pattern, dst_rel, note in MOVES:
        matches = sorted(REPO.glob(pattern)) if any(ch in pattern for ch in "*?[") \
            else ([REPO / pattern] if (REPO / pattern).exists() else [])
        for src in matches:
            if not src.exists():
                continue
            if dst_rel.endswith("/"):
                dst = REPO / dst_rel / src.name
            else:
                dst = REPO / dst_rel
            if dst.exists():
                if src.resolve() == dst.resolve():
                    continue  # already in place
                sup = REPO / "archive" / "superseded" / src.relative_to(REPO)
                if sup.exists():
                    continue  # already archived on a previous run
                actions.append(("supersede", src, sup,
                                "git mv" if use_git else "move",
                                f"{note} (replacement already at {dst.relative_to(REPO)})"))
            else:
                actions.append(("move", src, dst,
                                "git mv" if use_git else "move", note))

    src_j, dst_y = (REPO / CONVERT_JSON_YAML[0]), (REPO / CONVERT_JSON_YAML[1])
    if src_j.exists() and not dst_y.exists():
        actions.append(("convert", src_j, dst_y, "json>yaml",
                        "original kept for examples/legacy/run_doe_multistep.py"))

    for d in DELETES:
        p = REPO / d
        if p.exists():
            actions.append(("delete", p, None, "rm", "stale"))
    for g in DELETE_GLOBS:
        for p in sorted(REPO.glob(g)):
            if "archive" in p.parts:
                continue
            actions.append(("delete", p, None, "rm -r", "generated"))

    return actions, use_git


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually perform the plan (default is dry-run)")
    args = ap.parse_args()

    actions, use_git = plan()

    if not actions:
        print("Nothing to do — repository already organised.")
        return

    mode = "APPLYING" if args.apply else "DRY RUN (pass --apply to execute)"
    print(f"organise_repo: {len(actions)} action(s) — {mode}")
    print(f"git checkout detected: {use_git}\n")
    wa = max(len(a[0]) for a in actions)
    for action, src, dst, how, note in actions:
        s = str(src.relative_to(REPO)) if src else ""
        d = str(dst.relative_to(REPO)) if dst else ""
        arrow = f"{s} -> {d}" if s and d else (d or s)
        print(f"  {action:<{wa}}  [{how:>8}]  {arrow}" + (f"   ({note})" if note else ""))

    if not args.apply:
        return

    print("\nExecuting...")
    n_done = 0
    for action, src, dst, how, note in actions:
        try:
            if action == "mkdir":
                dst.mkdir(parents=True, exist_ok=True)
            elif action in ("move", "supersede"):
                do_move(src, dst, use_git)
            elif action == "convert":
                convert_params_json_to_yaml(src, dst)
            elif action == "delete":
                if src.is_dir():
                    shutil.rmtree(src)
                else:
                    src.unlink()
            n_done += 1
        except Exception as e:
            print(f"  FAILED: {action} {src}: {e}", file=sys.stderr)

    # tidy: remove now-empty source dirs
    for d in ("network_conversion", "bin", "examples/scenario_2/data_translation",
              "examples/scenario_2"):
        p = REPO / d
        try:
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
                print(f"  removed empty dir {d}")
        except OSError:
            pass

    print(f"\nDone: {n_done}/{len(actions)} actions completed.")
    print("Re-run without --apply to confirm 'Nothing to do'.")


if __name__ == "__main__":
    main()
