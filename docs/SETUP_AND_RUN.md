# Setup and run: feeder-wide DOE for a summer day

End-to-end guide to install the toolchain on a fresh machine and run the
thermal-aware DOE (Dynamic Operating Envelope) pipeline on one feeder for a
single day. The worked example below uses a **summer** day, because that is when
transformer hot-spot temperature and PV reverse flow are most likely to bind.

> Data note: all real network exports and meter data are confidential and are
> excluded from this repository by `.gitignore`. Keep those files in a local
> `data/` folder (also git-ignored). The commands below use placeholder file
> names like `feeder.xml` and `meter_export.csv`.

---

## 1. One-time install

The solver (Pyomo) shells out to the **ipopt** executable, which is not a Python
package — it must be installed separately. The reliable cross-platform route is
conda-forge.

### If you have conda / Anaconda (Windows or Linux)

```
conda install -c conda-forge ipopt
```

### If you do NOT have conda (e.g. a locked-down Linux server)

Use micromamba, which needs no admin rights:

```
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
micromamba create -y -n ipopt-env -c conda-forge ipopt
export PATH="$HOME/micromamba/envs/ipopt-env/bin:$PATH"
```

### Then, in the same environment, install the project

From the repository root:

```
pip install -e .
pip install requests
```

`requests` is only needed for the optional live ambient-temperature fetch.

### Verify the solver is visible to Pyomo

```
python -c "from pyomo.environ import SolverFactory; print(SolverFactory('ipopt').available())"
```

`True` means you are ready. (On the Microsoft Store build of Python, ipopt
installed by conda lives in a different environment — run from the Anaconda
Prompt, or `conda init bash` and reopen the shell, so `python` and `ipopt` come
from the same conda environment.)

---

## 2. Prepare the confidential inputs (local only)

Put these in a local `data/` folder (git-ignored). None of them are committed.

1. **Network exports** — the feeder CIM/DMS XML plus one LV-network XML per
   substation you want to model. The LV exports carry the customer UsagePoints.
2. **Meter data** — a wide interval-meter export covering the day you want to
   study, with one row per NMI / day / channel and `Data_HH_MM` interval columns.

---

## 3. Run the pipeline (worked summer-day example)

All commands run from the repository root.

### Step 1 — Build the network model from the exports

Merges the feeder and LV exports, places the real NMIs on their nodes (the
11-digit UsagePoint id is reduced to the 10-digit market NMI automatically), and
applies LV voltage limits of +/-10% by default.

```
python network_conversion/cim_to_network_json.py \
  "data/Lexcen_202604081113.xml" \
  "data/S 5402_LVNetwork_202604081113.xml" \
  "data/S 5406_LVNetwork_202604081113.xml" \
  "data/S 5408_LVNetwork_202604081114.xml" \
  "data/S 5409_LVNetwork_202604081114.xml" \
  "data/S 5549_LVNetwork_202604081113.xml" \
  "data/S 5550_LVNetwork_202604081114.xml" \
  "data/S 5846_LVNetwork_202604081113.xml" \
  "data/S 6158_LVNetwork_202604081114.xml" \
  "data/S 6918_LVNetwork_202604081114.xml" \
  "data/S 6921_LVNetwork_202604081114.xml" \
  "data/S 8029_LVNetwork_202604081114.xml" \
  "data/S 8082_LVNetwork_202604081114.xml" \
  "data/S 8199_LVNetwork_202604081114.xml" \
  "data/S 8391_LVNetwork_202604081114.xml" \
  "data/S 8511_LVNetwork_202604081114.xml" \
  "data/S 8673_LVNetwork_202604081114.xml" \
  "data/S 8701_LVNetwork_202604081114.xml" \
  "data/S 8712_LVNetwork_202604081114.xml" \
  -o data/feeder_network.json
```

Useful options:

- `--lv-voltage-tolerance 0.10` — LV band as a fraction of nominal (default 0.10
  = +/-10%). Pass `0` to disable voltage limits.
- `--lv-nominal-v 230` — centre the band on 230 V phase instead of each node's
  own base voltage. Tighter, and may need a lower `--v-setpoint-pu`.
- `--v-setpoint-pu 1.0` — feeder-head voltage setpoint (default 1.05). Lower this
  if voltage limits make the solve infeasible.

Check the NMIs were placed:

```
python -c "import json; c=json.load(open('data/feeder_network.json'))['components']; print(sum(k.startswith('nmi_') for k in c), 'NMI loads')"
```

### Step 2 — Translate the meter export into a solver timeseries

Nets active power as (import − export) so PV reverse flow is captured, uses the
reactive channels, and picks a single day. **Choose a summer date** present in
your export.

```
python examples/scenario_2/data_translation/wide_to_long_translator.py \
  -i data/lexcen_year_data.csv \
  -o data/forecast_timeseries.csv \
  --reactive-from-q
```

  --day 01/06/2023 \

- `--day DD/MM/YYYY` — the summer day to study (omit to use all days in the file).
- `--values-are-kw` — only if your meter values are already average kW rather
  than kWh per interval.

Sanity check the printout: the minimum active power should be negative (PV
export), and NMI count / timesteps should look right.

### Step 3 — Run the feeder-wide DOE solve

Runs the DOE for each chosen substation across every timestep, carrying the
transformer thermal state forward, with real ambient temperature for the day.

```
python examples/run_doe_feeder.py \
  data/feeder_network.json \
  data/forecast_timeseries.csv \
  out/summer_day \
  --transformer-params examples/scenario_doe/transformer_params.json \
  --open-meteo
```
  --only "SUBSTATION_A,SUBSTATION_B" \
- `--only "..."` — comma list of substation id/name substrings; omit to run every
  substation on the feeder.
- `--transformer-params ...` — IEEE C57.91 thermal parameters; omit for a plain
  DOE with no thermal limit. `I_rated` is derived per substation from each
  transformer rating.
- `--open-meteo` — fetch the day's real ambient temperature (needs internet). For
  a summer day this pulls the hot-weather profile automatically. Offline
  alternative: `--theta-a 35`.

Outputs land in `out/summer_day/<SUBSTATION>/`: `doe.csv` (the envelopes),
`bus.csv`, `branch.csv`, `thermal.csv`, plus a `feeder_summary.csv`.

### Step 4 — Analyse curtailment and its cause

Compares the DOE against unconstrained operation, and (with `--network`)
attributes each curtailed timestep to voltage, current (line/transformer
ampacity) or thermal (hot-spot) limits.

```
python examples/compare_doe_vs_normal.py out/summer_day \
  --network data/feeder_network.json \
  --transformer-params examples/scenario_doe/transformer_params.json \
  --timeseries data/forecast_timeseries.csv
```

Reads: `pct_curtailed`, `export_headroom_lost_kwh`, `actual_binding_events`
(whether real demand was cut), `curtail_cause`, and peak voltage/current/thermal
utilisation per substation.

---

## 4. Reading the results

- `feeder_summary.csv` — per substation: loads, NMIs with data, timesteps solved
  (out of the day's total), peak hot-spot temperature, and max violation.
- `doe.csv` — per NMI per timestep: `doe_lb_kw` (max import) and `doe_ub_kw` (max
  export). Values below the unconstrained cap mean the network curtailed that
  customer.
- `thermal.csv` — transformer hot-spot temperature trajectory over the day.

### What to expect on a summer vs winter day

On a cold day the transformer barely warms, so the **current** (ampacity) limit
usually binds first and the thermal constraint stays slack. On a hot summer day
the ambient is higher and midday PV export drives reverse flow, so the
**hot-spot temperature** and **voltage** limits are far more likely to become the
binding constraint — which is the case this pipeline is built to study.

### If timesteps go infeasible

Hard voltage limits with too high a feeder-head setpoint can make individual
timesteps infeasible; the runner skips those and continues (see
`n_timesteps_solved`). If many are skipped, lower `--v-setpoint-pu` (try 1.0,
then 0.98) when building the network in Step 1 so the LV sits mid-band.
