"""Synthetic PV-rich test fixture: a small substation where curtailment
really happens and the DTR has headroom to unlock.

Layout: 11 kV infeeder → 100 kVA transformer → LV busbar → two feeders of
three customers each. Every customer has PV; midday desired export well
exceeds what the (thermally hot) transformer can take on a summer afternoon,
while cool nights leave DTR headroom.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

S_MAX_KVA = 100.0
V_SEC_KV = 0.415
N_LOADS = 6


def make_network():
    comps = {
        "inf1": {"Infeeder": {"v_setpoint": 11.0, "cons": [{"node": "bus_hv"}]}},
        "bus_hv": {"Node": {"v_base": 11.0}},
        "bus_lv": {"Node": {"v_base": V_SEC_KV,
                            "user_data": {"v_min": 0.373, "v_max": 0.457}}},
        "tx1": {"Transformer": {
            "cons": [{"node": "bus_hv"}, {"node": "bus_lv"}],
            "z": [[0.0, 0.0], [0.0067, 0.034]],
            "nom_turns_ratio": [11.0 / V_SEC_KV],
            "taps": [0], "tap_factor": 0.025, "tap_side": "primary",
            "vector_group": ["YNyn0", "YNyn0"],
            "s_max": S_MAX_KVA, "v_winding_base": [11.0, V_SEC_KV],
        }},
    }
    for fdr, bus in (("a", "bus_a"), ("b", "bus_b")):
        comps[bus] = {"Node": {"v_base": V_SEC_KV,
                               "user_data": {"v_min": 0.373, "v_max": 0.457}}}
        comps[f"line_{fdr}"] = {"Line": {
            "cons": [{"node": "bus_lv"}, {"node": bus}],
            "length": 0.15, "z": [0.32, 0.08], "z0": [0.64, 0.16],
            "i_max": 300}}
    for i in range(N_LOADS):
        bus = "bus_a" if i < N_LOADS // 2 else "bus_b"
        comps[f"nmi_700100{i:04d}"] = {"Load": {"cons": [{"node": bus}]}}
    return {"units": {"voltage": 1000, "current": 1, "power": 1000,
                      "impedance": 1},
            "components": comps}


def make_timeseries(n_days=3, dt_min=30, seed=7, pv_kw=25.0, base_kw=1.2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05", periods=n_days * 24 * 60 // dt_min,
                        freq=f"{dt_min}min")
    hours = idx.hour + idx.minute / 60.0
    sun = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None) ** 1.5
    rows = []
    net = make_network()
    lids = [k for k, v in net["components"].items() if "Load" in v]
    for j, lid in enumerate(lids):
        size = 0.7 + 0.6 * rng.random()
        load = base_kw * size * (1 + 0.5 * np.cos((hours - 19) / 24 * 2 * np.pi)
                                 + 0.15 * rng.standard_normal(len(idx)))
        pv = pv_kw * size * sun * (0.9 + 0.1 * rng.random(len(idx)))
        p_w = (load - pv) * 1000.0          # load-positive W; negative = export
        rows.append(pd.DataFrame({
            "timestamp": idx, "load_id": lid,
            "real_power_w": np.round(p_w, 1),
            "reactive_power_var": np.round(np.abs(p_w) * 0.2, 1)}))
    return pd.concat(rows, ignore_index=True)


def make_ambient(timestamps):
    idx = pd.DatetimeIndex(sorted(pd.unique(timestamps)))
    hours = idx.hour + idx.minute / 60.0
    day = (idx.dayofyear - idx.dayofyear.min()).values
    temp = (26 + 2 * day                          # heatwave building
            + 8 * np.sin((hours - 9) / 24 * 2 * np.pi))
    return pd.Series(temp, index=idx, name="temperature_c")


def write_fixture(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    net = make_network()
    (outdir / "network.json").write_text(json.dumps(net, indent=1))
    ts = make_timeseries()
    ts.to_csv(outdir / "forecast_timeseries.csv", index=False)
    amb = make_ambient(ts["timestamp"])
    amb.rename_axis("timestamp").to_csv(outdir / "ambient.csv")
    return outdir


if __name__ == "__main__":
    import sys
    write_fixture(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dtr_fixture")
    print("fixture written")
