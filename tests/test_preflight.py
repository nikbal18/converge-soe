"""Deliberately broken fixtures must fire the right check id at the right
severity."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from converge_soe import preflight as pfl
from converge_soe import timeseries as tsm

REPO = Path(__file__).resolve().parent.parent
SCEN = REPO / "examples" / "scenario_doe"


@pytest.fixture()
def net():
    return json.loads((SCEN / "network.json").read_text())


@pytest.fixture()
def df():
    return tsm.read_long(SCEN / "forecast_timeseries.csv")


def ids(findings, severity=None):
    return {f["id"] for f in findings
            if severity is None or f["severity"] == severity}


def test_clean_network_has_no_errors(net):
    assert ids(pfl.check_network(net), "ERROR") == set()


def test_missing_node_fires_net001(net):
    net["components"]["nmi_x"] = {"Load": {"cons": [{"node": "bus_GONE"}]}}
    F = pfl.check_network(net)
    assert "NET001" in ids(F, "ERROR")
    assert "NET012" in ids(F, "ERROR")


def test_mesh_loop_fires_net004(net):
    net["components"]["loop"] = {"Line": {
        "cons": [{"node": "bus_a"}, {"node": "bus_b"}],
        "length": 0.1, "z": [1, 1], "z0": [1, 1]}}
    assert "NET004" in ids(pfl.check_network(net), "ERROR")


def test_inverted_voltage_limits_fire_net007(net):
    net["components"]["bus_b"]["Node"]["user_data"] = {"v_min": 0.5, "v_max": 0.4}
    assert "NET007" in ids(pfl.check_network(net), "ERROR")


def test_missing_smax_fires_net010(net):
    del net["components"]["tx1"]["Transformer"]["s_max"]
    assert "NET010" in ids(pfl.check_network(net), "ERROR")


def test_no_infeeder_fires_net002(net):
    del net["components"]["inf1"]
    assert "NET002" in ids(pfl.check_network(net), "ERROR")


def test_shifted_nmi_ids_fire_ts001(df):
    df2 = df.copy()
    df2["load_id"] = "nmi_" + df2["load_id"]     # timeseries has prefix,
    F = pfl.check_timeseries(df2, ["1001", "1002", "1003", "1004"])
    f = next(f for f in F if f["id"] == "TS001")
    assert f["severity"] == "WARN"
    assert "prefix" in f["message"]


def test_duplicate_rows_fire_ts003(df):
    df2 = pd.concat([df, df.iloc[:4]], ignore_index=True)
    F = pfl.check_timeseries(df2, ["1001", "1002", "1003", "1004"])
    assert any(f["id"] == "TS003" and f["severity"] == "ERROR" for f in F)


def test_dt_mismatch_fires_phy005(net, df):
    b = tsm.pre_index(df, ["1001", "1002", "1003", "1004"])
    tp = dict(tau_TO=60.0, tau_W=7.0, delta_theta_TO_R=55.0,
              delta_theta_HS_R=23.0, R=6.0, n=0.8, m=0.8,
              I_rated=200e3 / 415, theta_HS_max=120.0, dt=5.0)  # data is 30
    F = pfl.check_physical(b, net, tp)
    assert any(f["id"] == "PHY005" and f["severity"] == "ERROR"
               and "dt" in f["message"] for f in F)


def test_overloaded_bau_fires_phy001(net, df):
    df2 = df.copy()
    df2["real_power_w"] *= 60     # ~60x load: way over the 200 kVA rating
    b = tsm.pre_index(df2, ["1001", "1002", "1003", "1004"])
    F = pfl.check_physical(b, net, None)
    assert any(f["id"] == "PHY001" and f["severity"] in ("WARN", "ERROR")
               and "over rating" in f["message"] for f in F)


def test_envelope_abs_max_too_small_fires_mdl002(df):
    b = tsm.pre_index(df, ["1001", "1002", "1003", "1004"])
    F = pfl.check_model(b, envelope_abs_max=2.0)
    assert any(f["id"] == "MDL002" and f["severity"] == "ERROR" for f in F)


def test_verdicts():
    F = [pfl.finding("X1", pfl.ERROR, "s", "m")]
    assert pfl.verdict(F)[0].startswith("BLOCKED")
    F = [pfl.finding("X1", pfl.WARN, "s", "m")]
    assert pfl.verdict(F)[0].startswith("READY WITH WARNINGS")
    assert pfl.verdict([pfl.finding("X", pfl.INFO, "s", "m")])[0] == "READY"
