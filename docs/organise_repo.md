# scripts/organise_repo.py

## What it does
One-shot, idempotent reorganisation of the repository into the lifecycle
layout (docs/LAYOUT.md). It prints a table of every planned move and does
nothing unless you pass `--apply`.

## Where it sits
Run once per checkout (usually once ever). Everything else assumes the new
layout.

## Inputs
| Flag | Effect |
|---|---|
| *(none)* | dry run: full plan, no changes |
| `--apply` | perform the plan |

No files are inputs; the plan is embedded (MOVES/DELETES tables in the
script). Running it in an already-organised repo prints "Nothing to do".

## Outputs / behaviour
* Uses `git mv` when the file is tracked in a working git checkout (history
  preserved), plain moves otherwise.
* A source whose destination already exists (e.g. the refactored copy is
  already in place) is moved to `archive/superseded/` — nothing is ever
  overwritten or lost, and a second run is a no-op.
* Converts `examples/scenario_doe/transformer_params.json` →
  `config/transformers/distribution_onan.yaml` (with units and sources); the
  JSON stays because the legacy multistep runner reads it.
* Deletes only: `output.log`, `examples/scenario_2/break`, `__pycache__`,
  `*.egg-info`.

## Assumptions and limitations
Never deletes data; `archive/` is append-only. It does not rewrite imports —
the refactored sources are already in place at their new paths.

## How to tell if it worked
`--apply` twice: the second run must say "Nothing to do".
`python -c "import converge_soe"` and
`python examples/legacy/run_doe_feeder.py --help` must both work.
`git status` should show only renames/moves.

## Worked example
```
python scripts/organise_repo.py           # read the plan
python scripts/organise_repo.py --apply
python scripts/organise_repo.py           # → Nothing to do
```
