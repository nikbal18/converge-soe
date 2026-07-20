# Network conversion scripts

## cim_to_network_json.py — CIM XML → network.json

Converts a Schneider DMS CIM RDF XML feeder export into a converge-soe
network.json (ejson), in the same processed form as the reference feeder
files (single-phase, radial, open switches break topology, closed switches
become short `is_switch` lines, per-phase R/W/B devices merged).

```
# feeder only (no LV / placeholder loads):
python cim_to_network_json.py Lexcen_202604081113.xml -o network.json
# feeder + LV network exports (full network with real customers):
python cim_to_network_json.py Lexcen_202604081113.xml "S 5402_LVNetwork_202604081113.xml" \
    "S 5406_LVNetwork_202604081113.xml" ... -o network.json
python cim_to_network_json.py feeder.xml --lv-vmin 0.39 --lv-vmax 0.44
```

Accepts multiple XMLs: the feeder export plus any number of LVNetwork
circuit exports. Files merge automatically on their shared stitch-node IDs.

Loads, in priority order:
1. If LVNetwork XMLs are given, real customers are built from
   UsagePoint/ServiceLocation objects: component id `nmi_<NMI>`, placed at
   the exact service-point node, `s_nom` from the ADMS p/q estimate, and PV /
   battery systems recorded in `user_data.der`.
2. Else `--loads loads.csv` (columns: `nmi`, optional `substation`,
   optional `node`) spreads NMIs across LV dead-end nodes per substation.
3. Else a placeholder Load at each LV dead-end node
   (`--no-placeholder-loads` to disable).

Validation vs the reference ejson (S 5402): 33/34 NMIs identical, LV route
length within 6%, mean transformer-to-customer path resistance within 6%.

Notes:
- Transformer impedance is computed from leakage % + load loss (the reference
  files carry dummy 1e-6 values here — relevant for thermal work).
- `vector_group` is always "yy0" because soe_solver asserts vg[0]==vg[1];
  the actual CIM vector group is kept in `user_data.vector_group_actual`.
- Line impedance comes from PerLengthSequenceImpedance; where the catalogue
  entry is zero (most LV cables in the export) it is estimated from WireInfo
  conductor size/material and flagged `user_data.inferred: ["z"]`.
- Feeder head = node with `stitchingInfo=FH` (override with `--source-node`).
- Conversion warnings are recorded in the output's `user_data`.

## extract_lv_network.py — one transformer's LV network

Extracts everything at/below one distribution transformer's secondary into a
standalone solver-ready network.json (transformer + MV node + new Infeeder
included; use `--no-tx` to start at the LV busbar instead).

```
python extract_lv_network.py network.json --list
python extract_lv_network.py network.json "S 5402" -o S5402_lv.json
```

Works on both converted files and the original reference ejson files
(e.g. `_GOLDCR_8HB_LEXCEN-rad-sp-ot.json`).

## Important data caveat

The `Lexcen_202604081113.xml` export contains the MV feeder, all 18
distribution transformers and the substation LV boards, but NOT the
street-level LV reticulation or customers (~0.1 km LV wire vs 44 km in the
reference ejson, 0 customer objects vs 940 loads). If the other feeder XMLs
are the same export type, LV-level analysis needs either the reference-style
ejson files or separate LV network exports; the NMI/profile data alone gives
consumption but not network position.

## Example outputs

- `GOLDCR_8HB_LEXCEN_network.json` — converted from the Lexcen XML
- `S5402_AT_lv_network.json` — S 5402 LV network extracted from the
  reference ejson (70 nodes, 68 lines, 37 customer loads)

## batch_convert.py — whole-network batch conversion

Point it at a folder containing all your XML exports (feeder + LVNetwork
files mixed together, any names, subfolders fine):

```
python batch_convert.py xml_folder out_folder --jobs 6
```

Output structure:

```
out_folder/
  feeders/<FEEDER>_network.json                      one per feeder
  substations/<FEEDER>/<SUBSTATION>_lv_network.json  one per transformer
  batch_report.csv                                   counts, warnings, and
                                                     missing LVNetwork XMLs
```

Files are classified and grouped automatically from their cim:Circuit
elements — the LV exports attach to their parent feeder via the stitch-node
references, so filenames don't matter. The report's missing_lv_circuits
column tells you exactly which LVNetwork exports still need to be requested.
