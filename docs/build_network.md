# scripts/build_network.py

## What it does
Converts every CIM XML export in `data/xml/` into solver-ready network
models: one JSON per feeder plus one per substation, with a batch report of
counts, warnings, and the LVNetwork exports still missing.

## Where it sits
Stage ① of the pipeline; run_feeder runs it automatically (cached).

## Inputs
Any mix of feeder and LVNetwork XMLs under `data/xml/` (any names, any
nesting) — files are classified from their `cim:Circuit` element and LV
exports attach to their parent feeder via stitch-node references. Flags:
`--list-feeders`, `--force` (ignore the cache). A wrong/foreign XML is
skipped with a reason ("no cim:Circuit found").

## Outputs
`build/network/feeders/<FEEDER>_network.json`,
`build/network/substations/<FEEDER>/<SUB>_lv_network.json`,
`build/network/batch_report.csv` (columns: feeder, xml, n_lv_xmls, status,
nodes/lines/transformers/loads counts, warnings,
**missing_lv_circuits** — the exports to request — substations_extracted).
Cached against a SHA-256 of the XML inputs in `build/manifest.json`.

## Assumptions and limitations
Single-phase equivalent; open switches break topology; loops are broken by
dropping a closed edge (warned); `vector_group` forced to `yy0` (the solver
asserts vg[0]==vg[1]); line impedances missing from the catalogue are
estimated from conductor size and flagged `inferred`.

## How to tell if the output is wrong
Load counts of 0 with LV XMLs supplied → the LV exports didn't match their
feeder (check `unmatched LV` lines). Hundreds of `no_z` warnings → the
export lacks the impedance catalogue; results depend on estimated
impedances. Run `python -m converge_soe.network.validate <json>` or the
preflight for structural checks.

## Worked example
```
python scripts/build_network.py --list-feeders
```
