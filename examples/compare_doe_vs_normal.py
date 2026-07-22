#!/usr/bin/env python3
"""
Compare DOE results against "normal operation", quantify curtailment, and
(optionally) diagnose WHY each curtailment happened.

"Normal operation" = each customer is free to inject/consume up to the full
connection envelope (+/- CAP kW, the widest envelope the solver ever hands out,
i.e. the unconstrained value). Wherever the DOE hands out a NARROWER envelope,
the network has curtailed that customer at that timestep.

Three views:
  1. Headroom curtailment  — how far each envelope is pulled in below +/-CAP.
     Always available from the DOE files alone.
  2. Binding curtailment   — whether the customer's ACTUAL net power (from the
     forecast timeseries) would be cut by the envelope. Needs --timeseries.
  3. Cause diagnosis        — for each curtailed timestep, is the binding limit
     VOLTAGE, CURRENT (line/transformer ampacity) or THERMAL (transformer
     hot-spot)? Needs --network (and --transformer-params for the hot-spot
     limit). Reads the bus.csv / branch.csv / thermal.csv the solver writes.

Works on a whole output folder (out/test_day/, with <SUB>/{doe,bus,branch,
thermal}.csv inside) or a single doe.csv. Reusable on any run.

Sign convention (matches the solver): injection j = -real_power_w/1000 (export
positive). Envelope is doe_lb_kw <= j <= doe_ub_kw.

Usage:
    python compare_doe_vs_normal.py out/test_day
    python compare_doe_vs_normal.py out/test_day --timeseries examples/scenario_2/forecast_timeseries.csv
    python compare_doe_vs_normal.py out/test_day \\
        --network GOLDCR_8HB_LEXCEN_network.json \\
        --transformer-params examples/scenario_doe/transformer_params.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

TOL = 1e-3     # kW; ignore sub-watt numerical slack
ATLIM = 0.99   # a utilisation >= this counts as "at its limit"


def collect_doe_files(path):
    p = Path(path)
    if p.is_file():
        return [(p.parent.name or p.stem, p)]
    return [(f.parent.name, f) for f in sorted(p.glob("**/doe.csv"))]


def interval_hours(ts):
    u = pd.Series(sorted(ts.unique()))
    if len(u) < 2:
        return 0.5
    return u.diff().dropna().dt.total_seconds().mode().iat[0] / 3600.0


# --- cause diagnosis --------------------------------------------------------
def network_limits(netw):
    """Return {bus_id: (v_min_pu, v_max_pu)}, {branch_id: i_max_a}."""
    comps = netw["components"]
    vu, iu, su = (netw["units"][k] for k in ("voltage", "current", "power"))
    vlim, ilim = {}, {}
    for cid, v in comps.items():
        if "Node" in v:
            nd = v["Node"]; vb = nd.get("v_base"); ud = nd.get("user_data", {}) or {}
            vmn, vmx = ud.get("v_min"), ud.get("v_max")
            vlim[cid] = (vmn / vb if vmn and vb else None,
                         vmx / vb if vmx and vb else None)
        elif "Line" in v and "i_max" in v["Line"]:
            ilim[cid] = v["Line"]["i_max"] * iu
        elif "Transformer" in v and "s_max" in v["Transformer"]:
            tx = v["Transformer"]
            ilim[cid] = tx["s_max"] * su / (tx["v_winding_base"][1] * vu)
    return vlim, ilim


def diagnose(subdir, vlim, ilim, theta_hs_max):
    """timestamp -> v_util, i_util, t_util, cause (per-substation)."""
    def rd(name):
        f = subdir / name
        return pd.read_csv(f) if f.exists() else None
    bus, branch, thermal = rd("bus.csv"), rd("branch.csv"), rd("thermal.csv")
    recs = {}

    if bus is not None:
        for _, r in bus.iterrows():
            vmn, vmx = vlim.get(r["id"], (None, None))
            u = 0.0
            for oe in ("voltage_pu_oel", "voltage_pu_oer"):
                val = r.get(oe)
                if pd.notna(val):
                    if vmx:
                        u = max(u, val / vmx)
                    if vmn and val > 0:
                        u = max(u, vmn / val)
            recs.setdefault(r["timestamp"], {})["v"] = max(recs.get(r["timestamp"], {}).get("v", 0.0), u)

    if branch is not None:
        for _, r in branch.iterrows():
            imx = ilim.get(r["id"])
            if not imx:
                continue
            u = max(abs(r.get("current_a_oel", 0)) / imx, abs(r.get("current_a_oer", 0)) / imx)
            recs.setdefault(r["timestamp"], {})["i"] = max(recs.get(r["timestamp"], {}).get("i", 0.0), u)

    if thermal is not None and theta_hs_max:
        for _, r in thermal.iterrows():
            u = r["theta_HS_C"] / theta_hs_max
            recs.setdefault(r["timestamp"], {})["t"] = max(recs.get(r["timestamp"], {}).get("t", 0.0), u)

    rows = []
    for ts, dd in recs.items():
        v, i, t = dd.get("v", 0.0), dd.get("i", 0.0), dd.get("t", 0.0)
        cands = {"voltage": v, "current": i, "thermal": t}
        top = max(cands, key=cands.get)
        rows.append({"timestamp": ts, "v_util": v, "i_util": i, "t_util": t,
                     "cause": top if cands[top] >= ATLIM else "other/soft"})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doe", help="DOE output folder or a single doe.csv")
    ap.add_argument("--cap", type=float, default=None,
                    help="normal-operation envelope half-width kW (default: widest envelope seen)")
    ap.add_argument("--timeseries", default=None,
                    help="forecast_timeseries.csv to also test binding curtailment")
    ap.add_argument("--network", default=None,
                    help="feeder network.json — enables voltage/current/thermal cause diagnosis")
    ap.add_argument("--transformer-params", default=None,
                    help="thermal params json (for the hot-spot limit; default 120 C)")
    ap.add_argument("-o", "--output", default=None,
                    help="per-substation report csv (default: alongside input)")
    args = ap.parse_args()

    files = collect_doe_files(args.doe)
    if not files:
        raise SystemExit(f"No doe.csv found under {args.doe}")
    frames = []
    for sub, f in files:
        df = pd.read_csv(f, dtype={"load_id": str}); df["substation"] = sub
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    dt_h = interval_hours(d["timestamp"])

    cap = args.cap if args.cap is not None else round(
        max(d["doe_ub_kw"].max(), (-d["doe_lb_kw"]).max()), 3)
    print(f"Loaded {len(files)} substation(s), {len(d):,} NMI-timesteps, interval {dt_h*60:g} min")
    print(f"Normal-operation envelope (CAP) = +/-{cap:g} kW "
          f"({'given' if args.cap else 'inferred as widest envelope'})\n")

    d["export_curt_kw"] = (cap - d["doe_ub_kw"]).clip(lower=0)
    d["import_curt_kw"] = (cap + d["doe_lb_kw"]).clip(lower=0)
    d["curtailed"] = (d["export_curt_kw"] > TOL) | (d["import_curt_kw"] > TOL)

    have_ts = False
    if args.timeseries:
        ts = pd.read_csv(args.timeseries, dtype={"load_id": str})
        ts["timestamp"] = pd.to_datetime(ts["timestamp"])
        ts["inj_kw"] = -ts["real_power_w"] / 1000.0
        d = d.merge(ts[["load_id", "timestamp", "inj_kw"]], on=["load_id", "timestamp"], how="left")
        d["bind_over_kw"] = ((d["inj_kw"] - d["doe_ub_kw"]).clip(lower=0)
                             + (d["doe_lb_kw"] - d["inj_kw"]).clip(lower=0))
        have_ts = d["inj_kw"].notna().any()

    # cause diagnosis
    diag = {}
    if args.network:
        theta_hs_max = 120.0
        if args.transformer_params:
            theta_hs_max = json.loads(Path(args.transformer_params).read_text()).get("theta_HS_max", 120.0)
        vlim, ilim = network_limits(json.loads(Path(args.network).read_text()))
        for sub, f in files:
            dg = diagnose(f.parent, vlim, ilim, theta_hs_max)
            if not dg.empty:
                dg["timestamp"] = pd.to_datetime(dg["timestamp"])
                diag[sub] = dg

    rows = []
    for sub, g in d.groupby("substation"):
        rec = {
            "substation": sub,
            "pct_curtailed": round(100 * g["curtailed"].mean(), 1),
            "min_export_envelope_kw": round(g["doe_ub_kw"].min(), 2),
            "export_headroom_lost_kwh": round(g["export_curt_kw"].sum() * dt_h, 1),
            "import_headroom_lost_kwh": round(g["import_curt_kw"].sum() * dt_h, 1),
        }
        if have_ts:
            gb = g.dropna(subset=["inj_kw"])
            rec["actual_binding_events"] = int((gb["bind_over_kw"] > TOL).sum())
            rec["actual_energy_curtailed_kwh"] = round(gb["bind_over_kw"].sum() * dt_h, 2)
        if sub in diag:
            dg = diag[sub]
            curt_ts = g.loc[g["curtailed"], "timestamp"].unique()
            cc = dg[dg["timestamp"].isin(curt_ts)]["cause"].value_counts().to_dict()
            rec["curtail_cause"] = ", ".join(f"{k}:{v}" for k, v in cc.items()) or "-"
            rec["peak_v_util%"] = round(100 * dg["v_util"].max())
            rec["peak_i_util%"] = round(100 * dg["i_util"].max())
            rec["peak_t_util%"] = round(100 * dg["t_util"].max())
        rows.append(rec)
    rep = pd.DataFrame(rows)

    print(rep.to_string(index=False))
    print()
    tot = d["curtailed"].sum()
    print(f"TOTAL: {tot:,} of {len(d):,} NMI-timesteps curtailed "
          f"({100*tot/len(d):.1f}%) vs unconstrained +/-{cap:g} kW.")
    if have_ts:
        nb = int((d["bind_over_kw"] > TOL).sum())
        print(f"       {nb} events where the envelope would actually cut real demand "
              f"({round(d['bind_over_kw'].sum()*dt_h,2)} kWh).")
    if diag:
        print("       cause = which limit was at >=99% on the curtailed timesteps "
              "(voltage / current = line-or-transformer ampacity / thermal = hot-spot).")

    out = Path(args.output) if args.output else (
        (Path(args.doe) if Path(args.doe).is_dir() else Path(args.doe).parent) / "curtailment_report.csv")
    rep.to_csv(out, index=False)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
