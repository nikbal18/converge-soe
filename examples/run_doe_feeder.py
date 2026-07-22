#!/usr/bin/env python3
"""
Feeder-wide multi-timestep DOE runner.

The DoeSolver runs one transformer (substation) at a time. This driver takes a
whole feeder network.json, and for EVERY distribution transformer on that feeder
it:
  1. extracts the LV subtree below that transformer (the substation network),
  2. filters the load timeseries down to the NMIs that sit on that substation,
  3. runs the multi-timestep DOE solve (carrying transformer thermal state
     forward between timesteps), and
  4. writes per-substation results plus one feeder-level summary.

It reuses the exact extraction logic of extract_lv_network.py and the exact
multistep logic of run_doe_multistep.py, so results are identical to running
those two tools by hand for each substation.

--------------------------------------------------------------------------------
INPUTS
  feeder.json               A converge-soe network.json for the whole feeder
                            (output of cim_to_network_json.py). Must contain the
                            real NMI Loads, i.e. re-convert with --loads / usage
                            points once you have exported meter data.

  timeseries.csv            Long-format load timeseries for ALL NMIs on the
                            feeder. Columns (header row required):
                              timestamp        parseable by pandas (ISO 8601 is
                                               safest, e.g. 2025-02-01T07:00)
                              load_id          MUST match the Load component id
                                               in the network json. With real
                                               meter data that id is "nmi_<NMI>".
                              real_power_w     active power, WATTS, load +ve
                              reactive_power_var  reactive power, VAr
                            The driver filters this file per substation, so one
                            file for the whole feeder is fine.

  outdir                    Output folder (created if missing).

OPTIONS
  --substations-dir DIR     Use pre-extracted <SUB>_lv_network.json files from
                            this folder (e.g. batch_convert's out/substations/
                            <FEEDER>/) instead of extracting on the fly.
  --transformer-params P    JSON template of IEEE C57.91 thermal params (see
                            examples/scenario_doe/transformer_params.json).
                            I_rated and dt are auto-set per substation unless you
                            also pass --fixed-i-rated. Omit this flag entirely to
                            run a plain DOE with no thermal constraint.
  --fixed-i-rated           Keep I_rated from the params file instead of deriving
                            it from each transformer's s_max.
  --theta-a DEG             Constant ambient temperature, °C (default 25).
  --open-meteo              Fetch real ambient temperature from Open-Meteo ERA5
                            archive for the timeseries date range (overrides
                            --theta-a; needs internet).
  --envelope-abs-max KW     Max envelope half-width per NMI (default 50).
  --only SUBSTR             Only run substations whose id/name contains SUBSTR
                            (repeatable-style comma list, e.g. "S 5402,S 5406").

USAGE
  python run_doe_feeder.py feeders/GOLDCR_8HB_LEXCEN_network.json \\
         nmi_timeseries.csv  out/lexcen \\
         --transformer-params examples/scenario_doe/transformer_params.json \\
         --open-meteo
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Register the IDAES-bundled ipopt solver in-process. Needed when ipopt isn't on
# PATH (e.g. the Microsoft Store build of Python sandboxes ~/.idaes/bin).
# Harmless if idaes isn't installed.
try:
    import idaes  # noqa: F401
except ImportError:
    pass

import converge_soe as csoe

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("run_doe_feeder")

# Central Canberra — used only for the optional Open-Meteo ambient fetch.
SITE_LAT, SITE_LON = -35.2035, 149.1548


# ---------------------------------------------------------------------------
# LV subtree extraction (mirrors network_conversion/extract_lv_network.py)
# ---------------------------------------------------------------------------
def extract_substation(comps, tx_id, v_setpoint_pu=None):
    """Return a substation ejson `components` dict for the LV network below tx_id.

    Walks Lines only (never through another Transformer) from the transformer's
    secondary node, and includes the transformer, its MV node, and a fresh
    Infeeder on the MV node so the result is directly solvable.
    """
    tx = comps[tx_id]["Transformer"]
    primary_node = tx["cons"][0]["node"]
    secondary_node = tx["cons"][1]["node"]

    adj = defaultdict(set)
    for k, v in comps.items():
        if "Line" in v:
            c = v["Line"]["cons"]
            adj[c[0]["node"]].add(c[1]["node"])
            adj[c[1]["node"]].add(c[0]["node"])

    keep_nodes = {secondary_node}
    queue = [secondary_node]
    while queue:
        cur = queue.pop(0)
        for nxt in adj[cur]:
            if nxt == primary_node:
                continue  # don't climb back up the MV side
            if nxt not in keep_nodes:
                keep_nodes.add(nxt)
                queue.append(nxt)

    out = {}
    for k, v in comps.items():
        tp = next(iter(v))
        cd = v[tp]
        if tp == "Node" and k in keep_nodes:
            out[k] = v
        elif tp == "Line" and all(c["node"] in keep_nodes for c in cd["cons"]):
            out[k] = v
        elif tp == "Load" and cd["cons"][0]["node"] in keep_nodes:
            out[k] = v

    # Preserve the source infeeder's pu setpoint unless overridden.
    src_inf = next((v["Infeeder"] for v in comps.values() if "Infeeder" in v), None)
    if v_setpoint_pu is not None:
        pu = v_setpoint_pu
    elif src_inf:
        src_node = comps[src_inf["cons"][0]["node"]]["Node"]
        pu = src_inf["v_setpoint"] / src_node["v_base"]
    else:
        pu = 1.05

    out[tx_id] = comps[tx_id]
    out[primary_node] = comps[primary_node]
    v_base = comps[primary_node]["Node"]["v_base"]
    out[f"infeeder_{tx_id.lstrip('_')}"] = {"Infeeder": {
        "cons": [{"node": primary_node, "phs": tx["cons"][0].get("phs", ["A"])}],
        "v_setpoint": round(v_base * pu, 6),
    }}
    return out


def substation_ejson(feeder_ej, tx_id):
    """Build a full substation ejson (metadata + extracted components)."""
    out = {k: v for k, v in feeder_ej.items() if k != "components"}
    out["components"] = extract_substation(feeder_ej["components"], tx_id)
    return out


def load_ids_of(ejson):
    return [k for k, v in ejson["components"].items() if "Load" in v]


def tx_secondary_current_a(feeder_ej, tx_id):
    """Rated secondary current (A), matching cim_to_network_json's i_max formula:
       I = s_max_va / v_secondary_v  (single-phase equivalent)."""
    net = feeder_ej
    s_units = net["units"]["power"]
    v_units = net["units"]["voltage"]
    tx = net["components"][tx_id]["Transformer"]
    s_va = tx.get("s_max", 0.25) * s_units
    v_sec = tx["v_winding_base"][1] * v_units
    return s_va / v_sec


# ---------------------------------------------------------------------------
# Ambient temperature (optional Open-Meteo archive fetch; else constant)
# ---------------------------------------------------------------------------
def build_temp_series(timestamps, theta_a_const, use_open_meteo):
    if not use_open_meteo:
        return None  # signal: use constant theta_a_const
    import requests
    start = min(timestamps).date()
    end = max(timestamps).date()
    log.info(f"Fetching Open-Meteo temperature {start} → {end}")
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={"latitude": SITE_LAT, "longitude": SITE_LON,
                "hourly": "temperature_2m", "start_date": str(start),
                "end_date": str(end), "timezone": "UTC+11"},
        timeout=30,
    )
    r.raise_for_status()
    om = r.json()
    idx = pd.to_datetime(om["hourly"]["time"]).tz_localize(None)
    s = pd.Series(om["hourly"]["temperature_2m"], index=idx, dtype=float)
    return s.resample("5min").interpolate(method="time")


# ---------------------------------------------------------------------------
# Per-substation multistep solve (mirrors run_doe_multistep.py)
# ---------------------------------------------------------------------------
def run_substation(sub_ej, fc_sub, tparams, theta_a_const, temp_series,
                   envelope_abs_max, sub_outdir):
    sub_outdir.mkdir(parents=True, exist_ok=True)
    timestamps = sorted(fc_sub["timestamp"].unique())

    thermal_state = {}
    all_doe, all_bus, all_branch, all_viol, all_thermal = [], [], [], [], []

    for ts in timestamps:
        fc_t = (fc_sub.loc[fc_sub["timestamp"] == ts,
                           ["load_id", "real_power_w", "reactive_power_var"]]
                .set_index("load_id"))

        theta_a = (float(temp_series.asof(ts)) if temp_series is not None
                   else theta_a_const)

        solver = csoe.DoeSolver(
            sub_ej, fc_t,
            envelope_abs_max=envelope_abs_max,
            transformer_params=tparams,
            theta_A=theta_a,
            thermal_state_in=thermal_state,
            solver_options={},
        )
        try:
            status, results = solver.solve()
        except Exception as e:
            log.warning(f"    solver error at {ts}: {e} — skipping timestep")
            continue
        if results is None:
            log.warning(f"    no solution at {ts} — skipping timestep")
            continue

        thermal_state = results.thermal_state
        for df, store in [(results.soe, all_doe), (results.bus, all_bus),
                          (results.branch, all_branch), (results.viol, all_viol)]:
            tmp = df.copy()
            tmp.insert(0, "timestamp", ts)
            store.append(tmp)
        for tx_id, tstate in thermal_state.items():
            all_thermal.append({
                "timestamp": ts, "transformer_id": tx_id,
                "delta_theta_TO_C": round(tstate["delta_theta_TO"], 3),
                "delta_theta_HS_C": round(tstate["delta_theta_HS"], 3),
                "theta_HS_C": round(tstate["theta_HS"], 3),
            })

    def save(frames, name):
        if frames:
            pd.concat(frames).to_csv(sub_outdir / name)

    save(all_doe, "doe.csv")
    save(all_bus, "bus.csv")
    save(all_branch, "branch.csv")
    save(all_viol, "viol.csv")
    if all_thermal:
        pd.DataFrame(all_thermal).to_csv(sub_outdir / "thermal.csv", index=False)

    peak_hs = max((t["theta_HS_C"] for t in all_thermal), default=None)
    max_viol = 0.0
    for vf in all_viol:
        cols = [c for c in vf.columns if c.startswith("viol_")]
        if cols:
            max_viol = max(max_viol, float(vf[cols].abs().max().max()))
    return {"n_timesteps_solved": len({t["timestamp"] for t in all_thermal})
            or len(all_bus),
            "peak_hotspot_C": peak_hs, "max_violation_kw": round(max_viol, 4)}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feeder", help="feeder network.json")
    ap.add_argument("timeseries", help="long-format NMI timeseries csv")
    ap.add_argument("outdir", help="output folder")
    ap.add_argument("--substations-dir", default=None,
                    help="use pre-extracted <SUB>_lv_network.json files here")
    ap.add_argument("--transformer-params", default=None,
                    help="IEEE C57.91 params json (omit = no thermal model)")
    ap.add_argument("--fixed-i-rated", action="store_true",
                    help="keep I_rated from params instead of deriving from s_max")
    ap.add_argument("--theta-a", type=float, default=25.0,
                    help="constant ambient °C (default 25)")
    ap.add_argument("--open-meteo", action="store_true",
                    help="fetch ambient temperature from Open-Meteo archive")
    ap.add_argument("--envelope-abs-max", type=float, default=50.0)
    ap.add_argument("--only", default=None,
                    help="comma list of substring filters on tx id/name")
    args = ap.parse_args()

    feeder_ej = json.loads(Path(args.feeder).read_text())
    comps = feeder_ej["components"]

    fc = pd.read_csv(args.timeseries, dtype={"load_id": str})
    fc["timestamp"] = pd.to_datetime(fc["timestamp"])
    ts_all = sorted(fc["timestamp"].unique())

    # dt (minutes) between consecutive timesteps — needed by the thermal model.
    dt_min = ((pd.Series(ts_all).diff().dropna().dt.total_seconds() / 60.0)
              .mode().iat[0]) if len(ts_all) > 1 else 30.0

    base_params = None
    if args.transformer_params:
        base_params = json.loads(Path(args.transformer_params).read_text())
        base_params["dt"] = dt_min
        log.info(f"Thermal model ON (dt={dt_min:g} min, "
                 f"I_rated {'fixed' if args.fixed_i_rated else 'per-substation'})")
    else:
        log.info("Thermal model OFF (no --transformer-params)")

    temp_series = build_temp_series(ts_all, args.theta_a, args.open_meteo)

    tx_items = [(k, v["Transformer"]) for k, v in comps.items() if "Transformer" in v]
    if args.only:
        subs = [s.strip().lower() for s in args.only.split(",")]
        tx_items = [(k, t) for k, t in tx_items
                    if any(s in k.lower()
                           or s in t.get("user_data", {}).get("name", "").lower()
                           for s in subs)]

    log.info(f"Feeder {feeder_ej['user_data'].get('feeder')} — "
             f"{len(tx_items)} substation(s) to run")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    subs_dir = Path(args.substations_dir) if args.substations_dir else None

    summary = []
    for tx_id, tx in tx_items:
        name = tx.get("user_data", {}).get("name", tx_id)
        sub = tx.get("user_data", {}).get("substation", name)
        safe = name.replace(" ", "_").replace("/", "_")
        log.info(f"=== {name} ({tx_id}) ===")

        # Substation network: pre-extracted file if available, else extract now.
        sub_ej = None
        if subs_dir is not None:
            cand = subs_dir / f"{safe}_lv_network.json"
            if cand.exists():
                sub_ej = json.loads(cand.read_text())
        if sub_ej is None:
            sub_ej = substation_ejson(feeder_ej, tx_id)

        lids = load_ids_of(sub_ej)
        fc_sub = fc[fc["load_id"].isin(lids)].copy()
        n_matched = fc_sub["load_id"].nunique()
        log.info(f"    {len(lids)} loads in network, "
                 f"{n_matched} with timeseries data")
        if n_matched == 0:
            log.warning(f"    no NMI timeseries for {name} — skipping")
            summary.append({"substation": sub, "transformer": name,
                            "n_loads": len(lids), "n_nmis_with_data": 0,
                            "n_timesteps_solved": 0, "peak_hotspot_C": None,
                            "max_violation_kw": None, "status": "no_data"})
            continue

        tparams = None
        if base_params is not None:
            tparams = dict(base_params)
            if not args.fixed_i_rated:
                tparams["I_rated"] = round(tx_secondary_current_a(feeder_ej, tx_id), 3)

        stats = run_substation(sub_ej, fc_sub, tparams, args.theta_a,
                               temp_series, args.envelope_abs_max,
                               outdir / safe)
        summary.append({"substation": sub, "transformer": name,
                        "n_loads": len(lids), "n_nmis_with_data": n_matched,
                        **stats, "status": "ok"})
        log.info(f"    done: peak hotspot {stats['peak_hotspot_C']} °C, "
                 f"max viol {stats['max_violation_kw']} kW")

    pd.DataFrame(summary).to_csv(outdir / "feeder_summary.csv", index=False)
    log.info(f"\nFeeder summary written to {outdir / 'feeder_summary.csv'}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
