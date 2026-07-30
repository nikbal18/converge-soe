# The DOE pipeline, end to end

*Supersedes `examples/legacy/RUN_DOE_GUIDE.md` (kept as the legacy manual
route).* One command runs everything:

```
python scripts/run_feeder.py --feeder GOLDCR_8HB_LEXCEN \
       --scenarios doe_dtr,doe_static,bau --jobs 8 --fast --resume
```

## The nine stages

```
  data/xml/          data/meter/         config/
   (all feeders)     (a year of NMIs)    (transformer params, prices)
        │                  │                    │
        ▼                  ▼                    │
  ① BUILD NETWORK     ② PREPARE TIMESERIES      │
   XMLs → one JSON     messy CSVs → one tidy    │
   per feeder, plus    indexed table, units     │
   one per substation  fixed, NMIs matched      │
        │                  │                    │
        └────────┬─────────┘                    │
                 ▼                              │
          ③ SELECT FEEDER  ◄────────────────────┘
           pick one feeder; work out which NMIs
           sit under which substation automatically
                 │
                 ▼
          ④ PREFLIGHT  (seconds, no solving)
           "is this network and this data actually
            solvable?" — 30+ checks, one report
                 │
                 ▼
          ⑤ PRE-INDEX  (once, not per timestep)
           slice the year of data down to just this
           substation's NMIs, as a plain NumPy array
                 │
                 ▼
         ⑤b SYNTHESISE  (NMIs with no meter data)
           the meter export does not cover every NMI on
           the network. Those without data are not even
           columns after stage ⑤, so they load the
           network by exactly 0 kW and the envelopes come
           out far too generous. Each is given a real
           profile sampled from the metered NMIs on the
           same substation — they load the network but
           never receive an envelope. --no-synthetic
           reproduces the old behaviour.
                 │
                 ▼
          ⑥ SOLVE  × 3 scenarios × every substation
           doe_dtr  │  doe_static  │  bau
                 │
                 ▼
          ⑦ CHECKPOINT continuously — results hit disk
             as they are produced, never at the end
                 │
                 ▼
          ⑧ ANALYSE — metrics, comparison tables, plots
                 │
                 ▼
          ⑨ EXPLAIN — a doc per script, a results guide,
             and a PASS/FAIL sanity-check block
```

**Why it is shaped like this.** The legacy runners did everything inside one
loop: read the network, build the maths, slice the data, solve — 17,520
times per substation. Most of that work is identical every time. This
pipeline pushes everything that does not change (network parsing, unit
conversion, model structure, data slicing) *out* of the loop and does it
once. That is where the speed comes from, and it is why the stages are
separated: each one is cached (`build/manifest.json`), checked
(`preflight/`), and re-runnable on its own (`--skip-*` flags, or the
standalone scripts in `scripts/`).

## Where things live

See `docs/LAYOUT.md`. Short version: you put inputs in `data/`, everything
in `build/` is derived and deletable, everything in `out/` is a result you
keep. Each run writes `out/<FEEDER>/<RUN_ID>/` with `config_resolved.yaml`
(exactly what was run), per-substation parquet under `scenarios/`,
`preflight/`, `comparison/` (tables, figures, `sanity_checks.md`) and
`RUN_SUMMARY.md`.

## The three scenarios

| Key | Transformer limit | Thermal model | Envelopes |
|---|---|---|---|
| `doe_dtr` | `i_max_pu` from the inverted C57.91 model, recomputed each timestep, capped at `K_emergency` | drives the limit | solved |
| `doe_static` | `i_max_pu` from nameplate `s_max`, constant | post-hoc only | solved |
| `bau` | not enforced — violations recorded, not prevented | post-hoc | none |

BAU is not an optimisation: loads are fixed at forecast, a vectorised radial
power flow is evaluated for the whole period at once, and the thermal
recursion runs forward. Seconds for a full year.

## Typical sessions

First time with new data:

```
# 1. drop XMLs in data/xml/, meter CSVs in data/meter/
python scripts/build_network.py --list-feeders
python scripts/prepare_timeseries.py --values-are kwh_per_interval --fetch-ambient
python scripts/run_feeder.py --feeder <FEEDER> --dry-run     # plan + estimate
python scripts/run_feeder.py --feeder <FEEDER> --jobs 8 --fast
```

Interrupted run: just re-run the same command — `--resume` (default) picks
up from the last checkpoint and refuses stale checkpoints (add `--restart`
to discard them).

Just re-analysing: `python scripts/analyse_results.py out/<FEEDER>/<RUN_ID>`.

Worked example against the tiny bundled scenario (no real data needed):

```
python scripts/run_feeder.py --feeder-json examples/scenario_doe/network.json \
       --feeder TOY --meter examples/scenario_doe/forecast_timeseries.csv \
       --theta-a 25 --jobs 1
```

## Verbosity

| Level | Flag | Behaviour |
|---|---|---|
| 0 | `--quiet` | final summary and errors only |
| 1 | *(default)* | one progress bar over substation-runs; one line each on completion; warnings |
| 2 | `-v` | adds preflight summaries and stage detail |
| 3 | `-vv` | full solver detail to the per-substation `run.log` |

Failed timesteps are **always** recorded in
`scenarios/<sc>/<SUB>/failures.csv`, at every verbosity.
