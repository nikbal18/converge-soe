# Running a full-year job on Posit Workbench (VS Code)

Short answer: **resuming is one command.** Getting the job to survive the
night in the first place is the fiddly part, and VS Code on Workbench is the
worst combination for it. This doc is the checklist.

---

## Why VS Code on Workbench is the risky case

Workbench suspends or terminates sessions after an idle period (2 hours by
default). What counts as "idle" depends on the IDE, and this is the trap:

| IDE | On timeout | What counts as activity |
|---|---|---|
| RStudio Pro | **suspended**, restored with work intact | R console activity |
| VS Code / Positron | **terminated** — work is lost | UI interaction only |

For VS Code, "activity" means the VS Code API sees interaction: typing,
clicking, running a cell. **A long-running terminal command does not count.**
Worse, closing the browser tab severs the extension's connection to the
server, so the session looks idle even while it is computing hard.

So: starting `run_feeder.py` in the VS Code terminal and going home is the
one thing that reliably fails. Don't do that.

---

## The right way: a Workbench Job

Workbench Jobs run **remotely from your session**, so the job does not care
whether your session times out, your tab closes, or your laptop sleeps.
Available in VS Code sessions, not just RStudio Pro.

### 1. Write a launcher script

Workbench Jobs launch a *script*, not a command line, so put the invocation
in a file. Create `run_overnight.sh` in the repo root:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python scripts/run_feeder.py \
    --feeder GOLDCR_8HB_LEXCEN \
    --scenarios doe_dtr,doe_static,bau \
    --values-are kwh_per_interval \
    --jobs 8 --fast --resume \
    2>&1 | tee "run_$(date +%F_%H%M).log"
```

`--resume` is default but state it explicitly — this file is the record of
what you ran. Adjust `--jobs` to the cores your Workbench session actually
has, not your laptop's.

### 2. Launch it

1. Open the **Workbench Jobs** view.
2. Click **+** in the top-right — the Launcher options appear.
3. Under **Script Options**, use **Browse** to select `run_overnight.sh`.
4. Set the working directory to the repo root.
5. Launch.

The job appears in the Workbench Jobs view and updates automatically. You can
close the tab, close the laptop, go home.

### 3. Check on it

The Jobs view shows status and output. Independently of that:

```bash
# how far through, per substation/scenario
cat out/GOLDCR_8HB_LEXCEN/*/scenarios/*/*/_checkpoint.json | grep -E "substation|last_completed|n_timesteps|n_failed"

# anything failing?
wc -l out/GOLDCR_8HB_LEXCEN/*/scenarios/*/*/failures.csv
```

---

## If it dies anyway: resuming

This part genuinely is straightforward.

**Re-run the exact same command.** `--resume` is on by default. It reads
`_checkpoint.json` in each substation/scenario folder, restores the
transformer thermal state, and continues from the next timestamp. Results
already written to parquet are kept, not recomputed.

Checkpoints are written every 200 timesteps (`--flush-every`), so the most
you ever lose is the last few minutes.

### If it refuses to resume

You'll see a fingerprint mismatch error. That is the safety catch working:
the inputs changed since the checkpoint was written, so continuing would
splice two different runs together and silently produce a corrupt result.

The fingerprint covers the substation network JSON, that substation's
pre-indexed timeseries, the transformer params, and the resolved scenario
config. If you edited any of those — including `config/default.yaml` — the
old partial results are invalid.

Two options:

```bash
# start that substation over (correct choice if you meant to change inputs)
python scripts/run_feeder.py --feeder GOLDCR_8HB_LEXCEN ... --restart
```

or revert whatever you changed and resume normally. **Do not reach for
`--restart` just to make an error go away** — if you didn't knowingly change
an input, work out what changed first.

---

## Before betting a night on it

```bash
# 1. how long will this actually take? (20-timestep pilot, real extrapolation)
python scripts/run_feeder.py --feeder GOLDCR_8HB_LEXCEN --dry-run

# 2. does the data even pass preflight? (seconds, no solving)
python scripts/preflight.py --feeder GOLDCR_8HB_LEXCEN
cat out/GOLDCR_8HB_LEXCEN/*/preflight/preflight_report.md

# 3. does a small slice give sensible numbers? (~1 hour, not 12)
python scripts/run_feeder.py --feeder GOLDCR_8HB_LEXCEN \
       --only "S 5402,S 5406" --values-are kwh_per_interval --jobs 8 --fast
cat out/GOLDCR_8HB_LEXCEN/*/comparison/sanity_checks.md
```

Do all three. A twelve-hour run that fails preflight at hour zero, or
produces results that fail the sanity checks at hour twelve, costs a night
either way — and the first two commands take under a minute combined.

---

## When you come back

```bash
cat out/GOLDCR_8HB_LEXCEN/<RUN_ID>/RUN_SUMMARY.md
cat out/GOLDCR_8HB_LEXCEN/<RUN_ID>/comparison/sanity_checks.md
```

`sanity_checks.md` must be all PASS. The specific things to eyeball are in
`docs/run_feeder.md` ("How to tell if the output is wrong") and
`docs/RESULTS_GUIDE.md`.

Check `n_failed` in `_manifest.json` too — a handful of failed timesteps out
of 17,520 is tolerable and recorded in `failures.csv`; hundreds means
something is wrong with the network or the data, and preflight should have
caught it.

---

## Fallback if Workbench Jobs isn't available

Detaching from the terminal is second best — it survives the browser
disconnecting, but **not** the session being terminated, which is exactly
what VS Code sessions do on timeout:

```bash
nohup ./run_overnight.sh > nohup.log 2>&1 &
disown
```

Combined with `--resume`, this is usually survivable in practice: the job
gets killed at the idle timeout, you re-run it next morning, and it carries
on from the checkpoint. It just won't finish unattended in one go.

---

## Speed levers, in order of payoff

1. **`linear_solver=ma27`** (HSL, free academic licence) — typically 2–3×
   over MUMPS at this problem size. Set under `solver:` in
   `config/default.yaml`. The single biggest win for full-year runs; see
   `docs/profile_after.txt`.
2. **`--jobs N`** — substations are independent, so this scales nearly
   linearly. Match it to the Workbench session's cores.
3. **`--only`** — run a few substations at a time. Also makes each job short
   enough to dodge the timeout entirely.
4. **Hourly instead of 30-minute** for a first pass — halves the work, but
   blurs the thermal dynamics (τ_W is 7 minutes), so it is a scoping tool,
   not a result.

---

## References

- [Workbench session timeout (VS Code)](https://docs.posit.co/ide/server-pro/user/vs-code/guide/session-timeout.html)
- [Launching Workbench Jobs from VS Code](https://docs.posit.co/ide/server-pro/user/vs-code/guide/workbench-jobs.html)
- [Workbench session management](https://docs.posit.co/ide/server-pro/user/posit-workbench/guide/session-management.html)
- [Long-running sessions on Posit Workbench](https://support.posit.co/hc/en-us/articles/34336000247575-Long-running-Jupyter-sessions-on-Posit-Workbench)
