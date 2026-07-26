# tools/profile_doe.py

## What it does
Measures where the solve loop's time actually goes: runs N timesteps of
examples/scenario_doe under cProfile and attributes self-time to six
buckets (model build, .nl write + subprocess spawn, ipopt solve, pandas
slicing, network rebuild, extraction).

## Where it sits
A development tool. The baseline is committed as docs/profile_baseline*.txt;
re-run after any optimisation and record the delta — do not claim a speedup
you have not measured.

## Inputs
`-n` steps (default 200), `--fast` (profile the persistent path),
`--tx-limit`, `-o` output path.

## Outputs
A text report: wall time, ms/timestep, self-time per bucket, top-30
cumulative entries.

## Assumptions and limitations
cProfile inflates everything (especially many small Python calls), so use
the bucket *shares* and the plain-vs-fast *ratio*, not absolute times. The
bundled scenario is tiny (5 buses) — on real substations model construction
takes a far larger share, subprocess spawn a smaller one.

## How to tell if the output is wrong
The bucket sum should be close to the wall time; an exploding "other"
bucket means the bucket patterns need updating for a new pyomo version.

## Worked example
```
python tools/profile_doe.py -n 100 -o docs/profile_after.txt
python tools/profile_doe.py -n 100 --fast --tx-limit dtr
```
