# DPV penetration sweep: step by step

Answers SQ1: how do different penetration scenarios affect thermal loading and
loss of life, and where does thermal overtake voltage as the binding
constraint? Right now only `scale2` on Lexcen speaks to this, and `scale2`
scaled demand as well as export, so it is a load stress test rather than a
penetration rung.

Every command below is meant to be typed into **Git Bash**. Copy the whole
block. Lines starting with `#` are comments and are safe to paste.

Two new files do the work:

- `run_penetration_sweep.sh` — the ladder
- `tools/aggregate_penetration.py` — the analysis and the crossover number

---

## Step 0: set up the shell (15 minutes, once)

### 0.1 Open Git Bash and check you are in conda

Start menu, type `Git Bash`, open it. Look at the prompt. You want to see
`(base)` at the front:

```
(base) nikki@nikkilaptop MINGW64 ~
$
```

If `(base)` is missing, run this once, then **close the window and open a new
Git Bash**:

```bash
conda init bash
```

Now confirm you have the right Python and that ipopt is visible:

```bash
which python
which ipopt
```

Both paths should contain `anaconda3`, something like
`/c/Users/nikki/anaconda3/python` and `/c/Users/nikki/anaconda3/Library/bin/ipopt`.
The exact paths do not matter. What matters is that they are both under
`anaconda3` and neither is under `WindowsApps`.

**If `which python` says anything containing `WindowsApps`, stop.** That is the
Microsoft Store Python. It has a different PATH, so it cannot see conda's ipopt,
and it sandboxes file access on top of that. Every solve would fail, but only
after the pipeline has already spent an hour building the network and preparing
timeseries, so you would lose the evening finding out. This is the single most
expensive mistake available in this workflow, which is why the sweep script
checks it for you and exits in the first two seconds if it is wrong.

Do not try to fix a Store Python by installing idaes, micromamba or the VC++
redistributable. That path ends in `0xC0000135` DLL errors. Just use conda's
Python.

### 0.2 Go to the working copy

```bash
cd "/c/Users/nikki/Desktop/Honours project/Individual Project/converge-soe"
pwd
```

The quotes matter, because the path has spaces in it. `pwd` should echo the
Desktop path back.

**Never the OneDrive copy.** Not even to read from. Those files are cloud
placeholders, so a `grep` or a `find` hydrates them and re-materialises the tree
you keep deleting.

### 0.3 Pause OneDrive

Not a terminal step. Right-click the OneDrive cloud icon in the system tray,
choose "Pause syncing", pick 24 hours.

OneDrive held `state.json` open on 8 August and killed a run outright. A sweep
is hours of solving; losing it to a file lock at hour three is worse than
losing it to anything else, because nothing in the log will tell you that is
what happened.

### 0.4 Stop Windows sleeping

```bash
powercfg -change standby-timeout-ac 0
powercfg -change hibernate-timeout-ac 0
```

Note the `-change`, not `/change`. Git Bash rewrites arguments that start with a
forward slash into Windows paths, so `/change` would arrive at `powercfg` as
something like `C:/Program Files/Git/change` and the command would fail in a
confusing way. `powercfg` accepts either prefix, so use the dash and avoid the
whole problem.

Screen lock is fine. Sleep is not: it suspends the solver mid-interval.

Put them back when the sweep is done:

```bash
powercfg -change standby-timeout-ac 30
powercfg -change hibernate-timeout-ac 30
```

### 0.5 Commit what you have before starting a long run

```bash
git status --short
```

Right now that lists your whole cost pipeline as untracked: `stage2_annual_
ageing.py`, `stage3_cost_model.py`, `check_binding_constraint.py`,
`which_meter_cached.py` and `cost_params.yaml`. Get them in before you start
hammering the machine for a night:

```bash
git add tools/stage2_annual_ageing.py tools/stage3_cost_model.py \
        tools/check_binding_constraint.py tools/which_meter_cached.py \
        tools/aggregate_penetration.py cost_params.yaml \
        run_penetration_sweep.sh PENETRATION_SWEEP_PLAN.md

git commit -m "Add cost translation pipeline and DPV penetration sweep"
```

Check `cost_params.yaml` has no confidential Evoenergy figures in it before you
commit it. `cost_params.yaml.old` is deliberately left out.

---

## Step 1: time one rung before committing the night to it (20 minutes)

```bash
LEVELS="1.5" FEEDERS="lexcen" bash run_penetration_sweep.sh 2>&1 | tee "run_pen15_$(date +%F_%H%M).log"
```

`LEVELS` and `FEEDERS` in front of the command override the script's defaults
for that one run without editing the file. `tee` shows you the output and writes
it to a timestamped log at the same time, so you can go away and still find out
what happened.

The baseline Lexcen week took roughly 25 minutes for 18 substations across three
scenarios. Do not assume the ladder scales from that, because higher export
means more binding constraints and more soft-limit slack, so every rung is
slower than the one below it.

Lexcen is the stopwatch because it is the smallest of the three feeders at 18
substations, against Birrigai's 21 and Saunders' 27, and it is the one whose
behaviour you already know.

**Then look at the clock and multiply.** All three feeders is 66 substations
against Lexcen's 18, so a full rung is roughly 3.7 times whatever this cost, and
the upper rungs are slower again. If it took 25 minutes, the whole thing is an
overnight run. If it took three hours, cut the ladder and not the feeders.
See step 2.

### Watching it from another window

Open a second Git Bash, `cd` to the same folder, and:

```bash
tail -f run_pen15_*.log
```

Ctrl-C stops watching. It does not stop the run.

---

## Step 2: run the rest of the ladder, on all three feeders

```bash
bash run_penetration_sweep.sh 2>&1 | tee "run_pen_$(date +%F_%H%M).log"
```

That is Lexcen, Saunders and Birrigai at 1.5x, 2.0x and 3.0x. The 1.5x Lexcen
rung from step 1 is already done and will be skipped, not redone.

If step 1 said the rungs are expensive, cut the ladder rather than the feeders:

```bash
LEVELS="1.5 2.0" bash run_penetration_sweep.sh 2>&1 | tee "run_pen_$(date +%F_%H%M).log"
```

Three feeders at two rungs beats one feeder at three. The second feeder tests
whether the crossover generalises, which is a different question with a
different answer. The third rung only extends a curve whose shape you can
already see.

**If it dies overnight**, re-run the exact same line. Everything checkpoints per
substation and scenario, resume is on by default, and finished work is skipped.
A failed rung does not stop the rungs after it.

---

## Step 3: check each rung actually finished

```bash
ls out/*/pen*_summer/RUN_SUMMARY.md
```

You want nine lines back: three feeders times three rungs.

A run directory with a `scenarios/` folder but no `RUN_SUMMARY.md` was
interrupted, not completed. Your `month_summer` runs look exactly like that,
which is how they sat around for a month looking finished. To see which rungs
exist at all, including the broken ones:

```bash
ls -d out/*/pen*_summer
```

The aggregator skips any rung with no `comparison/metrics_by_substation.csv` and
tells you it did, so a half-finished rung cannot quietly poison the result.

---

## Step 4: aggregate

```bash
python tools/aggregate_penetration.py --plot
```

That does all three feeders and the cross-feeder comparison. For one feeder on
its own:

```bash
python tools/aggregate_penetration.py --feeder lexcen --plot
```

It writes `out/penetration/<feeder>/` for each feeder:

- `penetration_by_substation.csv` — every substation at every rung
- `penetration_summary.csv` — per rung: how many substations are thermally
  bound, what share of BAU ageing sits at those, curtailment recovered
- `crossover.csv` — per substation, the export multiplier at which peak thermal
  utilisation reaches 95%
- `penetration_crossover.png` and `.pdf` — the two-panel figure

plus `out/penetration/crossover_by_feeder.csv`, the one-row-per-feeder table
that puts metering coverage next to median crossover.

### If it refuses to run

It will say `COMPARABILITY PROBLEMS` and stop. That is the guard doing its job,
not a bug. Your 1.0x rung ran on 9 August and the ladder ran in September, so it
compares every setting that could have drifted in between (series reduction,
tap handling, feeder-head voltage, envelope cap, synthetic seed, solver soft
limits, timeseries handling) and refuses if any of them differ. It also checks
`scaling.import` is still 1.0 on every rung, which is the difference between a
penetration ladder and another `scale2`, and that donor assignment did not shift
between rungs.

Read what it names. If you can explain the difference and argue in writing that
it is harmless:

```bash
python tools/aggregate_penetration.py --plot --force
```

If you cannot, re-run the rung it names instead. A crossover number that
silently includes configuration drift is worse than no crossover number.

---

## Step 5: read the results

```bash
cat out/penetration/crossover_by_feeder.csv
cat out/penetration/lexcen/penetration_summary.csv
```

Three numbers come out of this, and each has a home in the thesis:

1. **The crossover multiplier.** The median substation's crossing point. This is
   what makes the extrapolation section of Chapter 8 possible. Quote it to one
   decimal place and never outside the range you actually ran.

2. **The share of substations still voltage-limited at the top rung.** This is
   what converts "voltage binds first" from a limitation into a result. If most
   of a feeder is still voltage-bound at triple today's export, that is a
   finding about where DTR investment should and should not go, and it
   strengthens the targeting argument in §8.4.

3. **Recovered curtailment against penetration.** The second panel of the
   figure. If the DTR buys nothing at 1.0x and a lot at 2.0x, the recommendation
   is not "deploy DTR now" but "deploy it at the substations that will cross
   first, before they cross". That is a better recommendation and it comes
   straight out of your own numbers.

Then read the crossover against metering coverage across the three feeders:
Lexcen 44.3%, Saunders 31.6%, Birrigai 28.6%. Fewer metered customers means
less of the transformer's load the envelope can actually curtail, so the thermal
cap should arrive at a lower multiplier. If that shows up, it is a finding about
where DTR matters. If it does not, the crossover is a property of the network
rather than the metering, which is also worth a paragraph.

Three points is an observation, not a regression. Report the three numbers and
describe the direction. Do not fit a line to them.

Also check what happens to BAU ageing up the ladder. At 3.0x the BAU peaks go
further outside the Arrhenius validity range, so §7's two-regime treatment
(ageing below the tier, exposure hours above it) is carrying more weight. If the
top rung is almost entirely in the exposure regime, report exposure hours rather
than an ageing number the model cannot support.

---

## What is deliberately not in the ladder

**Streeton, Magenta, Wanganee, Monaro.** Their only 1.0x runs use a different
meter window, 3 days in mid-December rather than the 7-day week, so they are not
rungs of this ladder without re-running their baseline. Worse, they only ever
covered top-20 substations, and those were selected on Gridsight's
loss-of-insulation-life ranking, which conditions on having already had an
overload event. They sit at the thermal end of the distribution by construction,
so including them would pull the median crossover down for a reason that has
nothing to do with penetration. If you want them in, they need their own 1.0x
rung on the 7-day week and must be reported as a separate, explicitly biased
sample.

**Wellington-Gurrang**, everywhere: 115 of 116 NMIs on its ageing-dominant
substations are donor-sampled, so the envelope controls nothing there and
scaling it harder measures the soft current limit rather than the network.

**Winter.** The crossover is a midday-export question and the winter week has
no midday export worth scaling. Running it would cost hours and answer nothing.

---

## The limitation you have to state

`--scale-export` multiplies every export reading. That is **more installed
capacity per existing PV customer**, not PV given to customers who currently
have none. A real doubling of penetration adds systems at new sites with
different orientations, which produces a flatter, wider midday shoulder than
simply doubling the existing one.

So this ladder is an export-magnitude proxy, and it is a **conservative** one
for the peak: it concentrates the increase at the existing midday peak rather
than spreading it, so it reaches the thermal cap at a lower multiplier than a
real rollout would. Say that in the methodology, and say it again next to the
crossover number.

---

## If something goes wrong

**`UnicodeEncodeError` immediately, before any work.** The stage banners contain
circled digits and Windows falls back to cp1252 when writing to a pipe. The
script exports `PYTHONIOENCODING=utf-8` for you, so this should not happen. But
if you run a `python scripts/...` line by hand through `tee`, set it first:

```bash
export PYTHONIOENCODING=utf-8
```

**`ipopt: command not found`, or every solve fails.** You are in the Store
Python. Go back to step 0.1.

**The run stops with no error and the log just ends.** Almost always sleep or
OneDrive. Check steps 0.3 and 0.4, then re-run the same command to resume.

**`cannot determine meter file format`.** You pointed at a raw export rather
than a slice. The sliced weekly files have a header row; `Gold_creek_summer.csv`
does not. This sweep uses the slice, so you should not hit it here.

**Disk fills up.** Each rung writes per-timestep parquet for every
substation-scenario. Check before you start:

```bash
df -h /c
```
