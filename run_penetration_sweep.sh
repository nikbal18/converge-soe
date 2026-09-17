#!/usr/bin/env bash
# DPV penetration ladder — the run that answers SQ1.
#
#   bash run_penetration_sweep.sh 2>&1 | tee "run_pen_$(date +%F_%H%M).log"
#
# Re-run that exact line to continue. Every pass resumes from its checkpoints
# and skips what is already DONE.
#
# WHAT THIS IS
#   Three extra rungs (1.5x, 2.0x, 3.0x export) on top of the 1.0x runs you
#   already have. The question it answers: at what level of DPV does the
#   transformer thermal limit overtake LV voltage rise as the binding
#   constraint? That number is what makes the extrapolation section of the
#   Discussion possible, and it converts "voltage binds first" from a
#   limitation into a result.
#
# WHY 1.0x IS NOT IN THE LADDER
#   lexcen7_summer and peak7_summer ARE the 1.0x rung. They were produced by
#   run_all_feeders.py with the same flags this script uses, and their
#   config_resolved.yaml confirms scaling.export = 1.0. tools/aggregate_
#   penetration.py re-checks that claim against every setting that affects
#   comparability and refuses to aggregate if any of them drifted. Do not
#   re-run 1.0x unless that check fails.
#
# WHAT --scale-export ACTUALLY DOES
#   It multiplies every P < 0 (export) reading. That is more installed
#   capacity per EXISTING PV customer, not PV given to customers who have
#   none. Say so in the thesis: the ladder is an export-magnitude proxy for
#   penetration, and it understates the diversity a real rollout would bring
#   (new systems point in different directions, so a real 2x fleet has a
#   flatter, wider midday shoulder than a doubled one). Scaling happens at
#   stage 2b, BEFORE substation selection and donor sampling, so the synthetic
#   customers are scaled on the same terms as the metered ones.
#
# BEFORE YOU WALK AWAY
#   1. Pause OneDrive syncing (right-click the cloud icon -> 24 hours).
#      It held state.json open and killed a run outright on 8 Aug.
#   2. powercfg /change standby-timeout-ac 0
#      powercfg /change hibernate-timeout-ac 0
#   3. Laptop on mains. Screen lock is fine; sleep is not.
#   4. Run from the CONDA shell, not Git Bash's Store Python, or ipopt is not
#      on PATH and every solve fails. The guard below checks this for you.

set -u

# REQUIRED when piping through tee. The stage banners contain circled digits
# (stage 1, 5b); writing to a PIPE, Python falls back to the Windows locale
# (cp1252), which has no such characters, and every pass dies instantly with
# UnicodeEncodeError before doing any work.
export PYTHONIOENCODING=utf-8

# --- what to run ------------------------------------------------------------
# Override from the command line without editing the file:
#   LEVELS="1.5" FEEDERS="lexcen" bash run_penetration_sweep.sh
LEVELS="${LEVELS:-1.5 2.0 3.0}"
# Three feeders, because one feeder gives a crossover number with nothing to
# compare it against. Lexcen, Saunders and Birrigai all already have a complete
# 1.0x rung on the SAME 27 Dec - 3 Jan week with byte-identical settings, so all
# three ladder together for the price of the new rungs alone. They span metering
# coverage 44.3% / 31.6% / 28.6%, which is the variable most likely to move the
# crossover: fewer metered customers means less of the load the envelope can
# actually curtail.
#
# Not included: Streeton, Magenta, Wanganee and Monaro. Their only 1.0x runs use
# a different meter window (3 days in mid-December, not the 7-day week) so they
# are not rungs of this ladder without a re-run, and they only ever covered
# top-20 substations, which were selected on Gridsight's loss-of-life ranking
# and so are conditioned on having already had an overload event. Putting them
# in a crossover distribution would bias it low by construction.
# Wellington-Gurrang is excluded everywhere: 115 of 116 NMIs on its
# ageing-dominant substations are donor-sampled.
FEEDERS="${FEEDERS:-lexcen,saunders,birrigai}"

# RETRY=1 adds --retry-failed, which re-queues tasks left as ERROR, RUNNING or
# STALE_CHECKPOINT by a previous attempt. You need it after a Ctrl-C or a closed
# window: killing the console kills ipopt, which aborts with
#   forrtl: error (200): program aborting due to window-CLOSE event
# and the parent records the task as ERROR (returncode 2), not as interrupted.
# A plain re-run will NOT pick those up — only TIMEOUT, STALLED, PARTIAL and
# NO_OUTPUT resume automatically, because a genuine ERROR is meant to be read by
# a human before being retried. So read one log first, confirm it says
# window-CLOSE, then re-run with RETRY=1.
#   tail out/_run_all/<tag>/logs/<season>_<FEEDER>_<SUB>.log
RETRY="${RETRY:-}"
RETRY_FLAG=""
[ -n "$RETRY" ] && RETRY_FLAG="--retry-failed"

# RESTART=1 adds --restart, which discards a queued task's partial output and
# solves it from scratch. Needed when the kill left a TRUNCATED parquet (valid
# PAR1 header, no footer): that file can be neither resumed nor read by
# analyse_results, so the analysis stage dies with
#   ArrowInvalid: Parquet magic bytes not found in footer
# and the run ends with no metrics_by_substation.csv and no RUN_SUMMARY.md.
# --restart only touches tasks that are actually QUEUED, so paired with RETRY=1
# it rebuilds just the broken substations and leaves every DONE one alone.
# Find truncated files first with tools/check_parquet_footers.py.
RESTART="${RESTART:-}"
RESTART_FLAG=""
[ -n "$RESTART" ] && RESTART_FLAG="--restart"

# REPLAN=1 adds --replan. Needed ONLY when a tag already holds a plan and you
# want to widen it — e.g. pen15 was planned for lexcen alone and you now want
# saunders and birrigai in it. Without it the bare command continues the
# existing plan and the new feeders are never queued.
#
# IT IS NOT FREE. With --replan, tasks already marked DONE are no longer
# skipped (run_all_feeders.py line ~996), so every completed substation under
# that tag is solved again. Re-planning also rewrites each substation's .npz
# bundle during warm-up, which can stale the checkpoints from the first pass.
# Budget for a full re-solve of whatever that tag already finished.
REPLAN="${REPLAN:-}"
REPLAN_FLAG=""
[ -n "$REPLAN" ] && REPLAN_FLAG="--replan"

# The canonical max-demand summer week. Same file as the 1.0x rung.
# Summer only: the crossover is a midday-export question and the winter week
# has no midday export worth scaling. Running it would cost hours and answer
# nothing.
MET_S="data/meter/Gold_creek_summer_2023-12-27_2024-01-02.csv"

# Identical to the flags behind lexcen7_summer and peak7_summer. Do not change
# any of these for the sweep — if they drift, the ladder is not comparable with
# its own baseline and the aggregator will refuse it.
#   --skip-pilot     the window is fixed, so a pilot decides nothing
#   --no-feeder-lock otherwise only one task per feeder runs at a time
#   --jobs 2 --concurrency 10  20 processes on 20 cores
#   --yes            unattended, no confirmation pause
COMMON="--series-reduction --skip-pilot --no-feeder-lock"
COMMON="$COMMON --jobs 2 --concurrency 10 --yes"
COMMON="$COMMON --season summer --summer-meter $MET_S"

# --- guards -----------------------------------------------------------------
if [ ! -f "$MET_S" ]; then
  echo "!!! $MET_S not found. Slice it first with tools/slice_meter_export.py"
  echo "!!! (NOT prepare_wide_export.py — that one is for the full-span BAU)."
  exit 1
fi

echo ">>> checking the environment before committing a night to it"
python - <<'PYEOF' || exit 1
import shutil, sys
sys.path.insert(0, "src")
try:
    import converge_soe, pyomo          # noqa: F401
except Exception as e:
    sys.exit(f"!!! cannot import the package: {e}")
if shutil.which("ipopt") is None:
    sys.exit("!!! ipopt is not on PATH. You are probably in Git Bash's Store\n"
             "!!! Python instead of the conda env. Activate conda and retry.")
print("    ok: package imports, ipopt found at", shutil.which("ipopt"))
PYEOF

echo ">>> checking the ambient cache covers the window"
python scripts/fetch_ambient.py --check-only || {
  echo "!!! ambient check failed. Missing timestamps fall back to a FLAT 25 C,"
  echo "!!! which is not a DTR and would quietly invalidate the whole ladder."
  exit 1
}

# --- the ladder -------------------------------------------------------------
pass () {
  local lvl="$1" tag="$2" rc=0
  echo
  echo "=============================================================="
  echo ">>> penetration ${lvl}x  tag=${tag}  feeders=${FEEDERS}${RETRY:+  (retrying failed/interrupted tasks)}"
  echo ">>> started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=============================================================="
  # NO --run and NO --plan. run_all_feeders decides with
  #     do_plan = args.plan or args.replan or (not args.run and not have_plan)
  # so passing --run explicitly makes `not args.run` false and a FRESH tag
  # plans nothing, queues nothing, and exits 0 in three seconds having solved
  # nothing at all. The bare command plans when there is no plan and runs
  # either way, which is what every other orchestrator in this repo does.
  python scripts/run_all_feeders.py --feeders "$FEEDERS" \
    $COMMON $RETRY_FLAG $RESTART_FLAG $REPLAN_FLAG --tag "$tag" \
    --extra="--scale-export $lvl"
  rc=$?          # capture IMMEDIATELY — any command in between overwrites it
  echo "<<< ${tag} finished $(date '+%Y-%m-%d %H:%M:%S') with exit code $rc"

  # Exit code 0 is NOT proof that anything was solved — "nothing to run" exits
  # clean. Check for the artefacts instead.
  local made=0
  for d in out/*/"${tag}_summer"; do
    [ -f "$d/RUN_SUMMARY.md" ] && made=$((made + 1))
  done
  if [ "$made" -eq 0 ]; then
    echo "!!! ${tag} produced NO completed run directory. Exit code 0 here"
    echo "!!! usually means the planner queued nothing, not that it succeeded."
    echo "!!! Check the lines above for 'nothing to run'. If the tag already"
    echo "!!! holds a plan for different inputs, add --replan."
    rc=1
  else
    echo "    ${tag}: ${made} completed run director(y/ies)"
  fi
  if [ "$rc" -ne 0 ]; then
    echo "!!! ${tag} FAILED. Continuing to the next level so a bad rung does"
    echo "!!! not cost you the whole night — but do not aggregate until it is"
    echo "!!! either fixed or deliberately dropped from the ladder."
  fi
}

# SMALLEST MULTIPLIER FIRST. Higher penetration means more binding constraints
# and more soft-limit slack, so each rung is slower than the last. If the night
# runs out you want the low rungs complete rather than 3.0x half done — and the
# low rungs are where the crossover actually sits.
for lvl in $LEVELS; do
  pass "$lvl" "pen${lvl//./}"
done

echo
echo "ALL DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo
echo "Completion marker per run: 'analysis written to' in the log, and a"
echo "RUN_SUMMARY.md in the run directory. A run with scenario folders but no"
echo "RUN_SUMMARY.md was interrupted — the month_summer runs look exactly like"
echo "that, so check before you trust it."
echo
echo "Next:  python tools/aggregate_penetration.py --plot"
