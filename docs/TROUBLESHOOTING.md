# Troubleshooting

Every preflight check id, its trigger and its fix — plus the classic traps.

## The classics

### `can't evaluate pow'(0,0.8)` → `Solver (ipopt) did not exit normally`
The legacy combined thermal constraint evaluated `(K²)^0.8` at K²=0, where
the gradient is singular. Fixed structurally: with `tx_limit="dtr"` the
thermal model is inverted into a bound (docs/DTR_METHOD.md) and this
constraint no longer exists; variable lower bounds (`square_current_pu ≥
1e-8`) protect the legacy path too. If you ever see it again you are running
`tx_limit="legacy"` with an old checkout.

### The `nmi_` prefix mismatch
Meter exports carry bare NMIs (`7001009197`); network Load ids are
`nmi_7001009197`. Symptom: `n_nmis_with_data = 0`, empty results. The
pipeline reconciles both directions automatically and reports what it did
(TS001); `build/feeders/<F>/nmi_index.csv` records every assignment.

### The kWh-vs-W trap
Wide meter exports are kWh-per-30-min-interval; the solver wants average W
(× 2000 difference!). `prepare_timeseries.py` refuses to guess: pass
`--values-are kwh_per_interval` (or `kw`, or `w`). TS006 flags implausible
magnitudes afterwards. If DOE numbers look exactly 2× or 2000× off, this is
it.

### The `dt` mismatch
The thermal model's `dt` must equal the timeseries interval; a mismatch
silently corrupts every temperature. The pipeline always sets `dt` from the
data's modal Δt; preflight PHY005 makes a mismatch an ERROR.

### ipopt not found (conda vs Store Python)
On the laptop **ipopt is installed via conda**, so scripts must run from a
conda-activated shell (`conda activate <env>` in Anaconda Prompt / Git Bash
with conda on PATH). The Microsoft Store Python does not see it. The legacy
runners also try `import idaes` which registers the IDAES-bundled ipopt if
present (`idaes get-extensions` downloads it once). Check with:
`python -c "from pyomo.environ import SolverFactory; print(SolverFactory('ipopt').available())"`.

### OneDrive
Do not point outputs or Pyomo temp files at a OneDrive-synced folder — every
`.nl` write triggers a cloud sync. The pipeline puts Pyomo temp files in the
system temp dir; keep the working repo itself on a local (non-synced) path
if solve performance matters.

## Preflight check reference

| id | severity | trigger | fix |
|---|---|---|---|
| NET001 | ERROR | a `cons[].node` references a missing Node | fix the ejson; dangling refs crash or silently drop constraints |
| NET002 | ERROR/WARN | not exactly one Infeeder / infeeder on tx secondary | one Infeeder on the MV primary node |
| NET003 | ERROR | nodes unreachable from the infeeder | remove islands or fix topology |
| NET004 | ERROR | cycles / n_branches ≠ n_nodes−1 | the model sums *downstream* flows — valid only on a radial tree; break loops |
| NET005 | ERROR | self-loops or duplicate parallel branches | merge/remove |
| NET006 | ERROR | line impedance zero/invalid | zero-impedance branches degenerate the voltage-drop constraint |
| NET007 | WARN/ERROR | missing v_min/v_max (constraint silently dropped — false pass) / v_min ≥ v_max / setpoint outside limits | add limits; fix setpoint |
| NET008 | ERROR/WARN | v_base inconsistent along a line or across the transformer | wrong bases make pu currents wrong by orders of magnitude |
| NET009 | WARN | line without explicit i_max (100 kA fallback = no limit) | add real ratings |
| NET010 | ERROR | transformer without s_max (1e9 fallback; I_rated underivable) | add nameplate rating |
| NET011 | ERROR | missing tx fields / vector_group vg[0]≠vg[1] | the solver has a bare assert here; fix the fields |
| NET012 | ERROR | Loads attached outside the subtree (silently dropped) | reattach or remove |
| NET013 | WARN | substation subtree nodes missing from parent feeder | extraction bug or missing LVNetwork XML |
| TS001 | ERROR/WARN | load_id match rate; nmi_ prefix mismatch detected explicitly | see prefix trap above |
| TS002 | WARN | timestamp format not inferrable (slow dateutil fallback) | use ISO 8601 |
| TS003 | ERROR/WARN | duplicated (timestamp, load_id); irregular Δt | dedupe; gaps become masked zero-fills |
| TS004 | WARN | NMI coverage below 90 % | gaps are filled with 0 W and masked |
| TS005 | WARN | NaN / infinite power values | cleaned to 0 W and masked |
| TS006 | WARN | implausible magnitudes / no PV export / \|Q\|>\|P\| | check --values-are and channel handling |
| TS007 | WARN/ERROR | ambient series missing or not covering the range | prepare_timeseries --fetch-ambient |
| PHY001 | WARN/ERROR | aggregate background load exceeds tx rating (K>1) | no envelope choice is feasible there — expect quantified violations |
| PHY002 | WARN | line exceeds rating on background load alone | expect current violations on those branches |
| PHY003 | WARN | open-loop BAU θ_HS above the limit | the DOE scenarios will curtail here; also reports the DTR-above-nameplate share |
| PHY004 | WARN | voltage screen outside limits at peak load / reverse flow | expect voltage-bound envelopes at those times |
| PHY005 | ERROR | thermal params fail sanity (τ ordering, exponents, θ_HS_max range, I_rated consistency, **dt ≠ data Δt**) | fix config/transformers/*.yaml; let the pipeline derive I_rated and dt |
| MDL001 | WARN/INFO | soft limits off: violations become silent infeasibility | run with soft limits (default ON) |
| MDL002 | ERROR | envelope_abs_max smaller than a NMI's peak forecast | raise it — otherwise the parameter, not the network, clips the envelope |

## Resume refuses to continue
`StaleResumeError`: the checkpoint's `inputs_fingerprint` no longer matches
(network, timeseries, thermal params or solver config changed). The partial
results came from different inputs; `--restart` discards them. A silently
stale resume would be worse than no resume.

## A substation shows status "error" in _manifest.json
Read `out/.../scenarios/<sc>/<SUB>/run.log` — workers log full tracebacks
there. Individual timestep failures (not fatal) are in `failures.csv`.
