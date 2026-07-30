# CHANGES — the DOE pipeline rebuild

## Added

* **`src/converge_soe/synthetic.py` + stage ⑤b** — synthetic profiles for
  network NMIs with no interval-meter data. Stage ③ only indexes NMIs that
  matched the meter export, so the rest were not columns in the bundle at all
  and contributed 0 kW to the power flow: the feeder looked unloaded and the
  envelopes came out far too generous. On the Gold Creek LV network this was
  **72 of 121 Loads (60%), carrying ~61% of the real feeder load**.
  Each gap is filled with a real profile sampled from the metered NMIs on the
  same substation (rather than a mean profile, which would smooth away the
  peak and minimum-demand intervals that bind the constraints), matching
  rooftop PV via `user_data.der`, drawing without replacement, and reporting a
  diversity flag plus the full donor→target map. Where a substation has fewer
  than `min_donors` usable donors, it falls back to disaggregating that
  substation's measured transformer demand *minus* its metered NMIs.
  Config under `synthetic:`; `--no-synthetic` reproduces the old behaviour so
  the difference can be quantified.
  Note: `ESTIMATION − METER_SUMMATION` is **not** the unmetered load — the
  ratio is a near-constant ~1.03 gross-up for missing meter *reads*, rising to
  ~2.3 when `METER_COUNT` collapses, and being multiplicative it flips sign
  with the feeder. See the module docstring.

* **`src/converge_soe/thermal.py`** — IEEE C57.91 forward model, the DTR
  inversion (`dtr_current_limit_pu`, brentq on the strictly-increasing
  end-of-interval θ_HS), open-loop trajectories, and the ageing formulas.
  Verified: inversion round-trips to 1e-9 (`tests/test_dtr_inversion.py`).
* **`src/converge_soe/persistent_solver.py`** — model built once per
  substation, mutable Params per timestep, appsi-ipopt when available,
  warm starts (`--fast`; equivalence-gated).
* **`src/converge_soe/timeseries.py`** — wide/long meter ingestion with
  explicit units, `nmi_` prefix reconciliation, and per-substation
  pre-indexing to NumPy bundles (the timestep loop contains no pandas).
* **`src/converge_soe/preflight.py`** + `network/validate.py` — the
  NET/TS/PHY/MDL check suite and reports.
* **`src/converge_soe/io.py`** — streaming parquet writers (row-group per
  flush), atomic fingerprinted checkpoints, resume (append-safe),
  failures.csv, run manifest.
* **`src/converge_soe/scenarios.py`** — doe_dtr / doe_static / bau; BAU is
  a vectorised power-flow evaluation, not an optimisation.
* **`src/converge_soe/pipeline.py`**, **`analysis.py`**, **`metrics.py`**,
  **`plots.py`** — the nine-stage orchestration, metric tables, 12 figures,
  and the PASS/FAIL sanity block.
* **`scripts/`** — organise_repo, build_network, prepare_timeseries,
  preflight, run_feeder (the single entry point), analyse_results; console
  entry points csoe-build/run/analyse.
* **`tools/profile_doe.py`**, docs for every script, PIPELINE / DTR_METHOD /
  RESULTS_GUIDE / TROUBLESHOOTING / LAYOUT / MIGRATION_NOTES, tests
  (33 passing), config/default.yaml + config/transformers/.

## Changed in doe_solver.py (backwards compatible)

Default construction reproduces the legacy numbers
(`tests/test_legacy_regression.py`). New opt-in parameters: `tx_limit`
(legacy|dtr|static — dtr replaces the transformer i_max with the inverted
C57.91 rating and removes the nonlinear hot-spot constraint entirely),
`soft_limits` (+penalty), `quiet`, `warm_start_in`, `network_cache`,
`usable_export_weight`, `solver_name`, `k_emergency`, `k2_floor`.
Always-on fixes (Phase 1.4): variable bounds (V² ∈ [0.25, 2.25],
I² ≥ 1e-8 — kills the pow'(0,0.8) crash), solve exceptions caught and
reported instead of raised, the O(B²) downstream-branch scan replaced by a
precomputed map, truncated `_netw_components` tail repaired (the synced
copy ended in a bare `return`).

## Deliberate deviations from the spec, and why

1. **`E_curt(dtr) ≤ E_curt(static)` needed an objective fix to be a fair
   test.** With the legacy width-only objective the per-NMI allocation of
   surplus capacity is mathematically degenerate; observed: all DTR headroom
   handed to customers who couldn't use it, making E_curt identical between
   scenarios. Added a small "usable export" reward
   (`solver.usable_export_weight`, default 0.01 in the pipeline, 0 in the
   library) so envelopes follow desired export. Without something like this
   the headline metric is arbitrary. Details: docs/MIGRATION_NOTES.md §3.
2. **The carried thermal state is the actual (clipped-forecast) state,**
   not the 'oel' envelope-corner state the legacy extraction advanced. It is
   what a physical DTR would measure, makes ageing comparable with BAU, and
   guarantees the delivered θ_HS respects the limit. The envelope-corner
   trajectory is still recorded (`theta_HS_envelope_C`). Spec said "advance
   exactly as _extract_results does today" — kept verbatim in
   `tx_limit="legacy"`; changed for the dtr/static scenarios for the
   reasons above.
3. **The fast path is not the big win on this hardware.** Measured
   (docs/profile_after.txt): pyomo 6.10 builds the 150-NMI model in 0.09 s;
   >95 % of per-step time is inside ipopt. The spec's estimate that model
   construction dominates did not hold here. Path B exists, is correct
   (bit-identical at tight tolerance), and will pay off where construction
   is slower; the levers that actually moved wall-clock were solver options
   (mu_strategy=adaptive, tol=1e-6: 151→96 iterations, ~1.9×), warm starts,
   parallel substations, and the O(T²) pandas elimination (~minutes to
   hours saved at year scale — measured).
4. **`K_emergency = 2.0` kept, but flagged.** For ONAN distribution
   transformers a 2.0 short-time backstop is defensible (C57.91 short-time
   loading tables), but bushing/tap-changer limits are asset-specific;
   worth checking against EvoEnergy data sheets. Configurable per
   transformer class file.
5. **Myopic single-interval DTR kept** (as specified); the greedy-heating
   effect is clearly visible in the fixture (K_max sags through loaded
   afternoons). Multi-interval look-ahead noted as future work.
6. **Soft-limits approach agreed with** — implemented as specified, default
   ON in the pipeline, OFF in the raw library class for backwards
   compatibility.
7. **BAU energy-balance sanity check** uses the lossless sweep, so the "≈ Σ
   loads + losses within 1 %" check is applied to the BAU scenario only
   (where it closes exactly); for DOE scenarios the corner flows are not an
   energy balance of actual behaviour.

## Verification summary

33 tests pass: DTR inversion (round-trip 1e-9, monotonicity, all three
status branches), fast-path equivalence (bit-identical at tol=1e-11),
legacy regression (matches the pre-change baseline trajectory), preflight
fixtures (each broken input fires the right check id), interrupt/resume
(byte-identical concatenated parquet + stale-checkpoint refusal),
metrics hand-computed to 1e-9, organise_repo idempotence. End-to-end run on
the PV-rich fixture: all sanity checks PASS; doe_dtr halves curtailment vs
doe_static (201 → 101 kWh) with θ_HS riding exactly at the 120 °C limit,
while BAU would have peaked at 141.6 °C.
