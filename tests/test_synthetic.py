"""Synthetic profiles for NMIs with no meter data (stage ⑤b).

Covers donor sampling, the transformer-disaggregation fallback, prefix
reconciliation, diversity flags and the bundle-splicing invariants.
"""

import json

import numpy as np
import pandas as pd
import pytest

from converge_soe import synthetic as sy
from converge_soe import timeseries as tsm

T = 48
FREQ = "30min"
START = "2023-06-02 00:30"


def _index():
    return pd.date_range(START, periods=T, freq=FREQ)


def _network(n_with_data=8, n_gaps=3, der_on=(), placeholders=0):
    """Loads named d0.. (metered) and g0.. (gaps), all on one bus."""
    comps = {"bus_lv": {"Node": {"v_base": 0.415}}}
    for i in range(n_with_data):
        comps[f"nmi_d{i}"] = {"Load": {"cons": [{"node": "bus_lv"}]}}
    for i in range(n_gaps):
        ud = {}
        if f"g{i}" in der_on:
            ud["der"] = [{"nmi": f"g{i}"}]
        comps[f"nmi_g{i}"] = {"Load": {"cons": [{"node": "bus_lv"}],
                                       "user_data": ud}}
    for i in range(placeholders):
        comps[f"loadph{i}"] = {"Load": {"cons": [{"node": "bus_lv"}],
                                        "user_data": {"placeholder": True}}}
    return {"components": comps}


def _bundle(n, pv_from=None, complete=True):
    """A bundle with n metered columns; columns >= pv_from export."""
    idx = _index()
    rng = np.random.default_rng(0)
    P = rng.uniform(200.0, 800.0, (T, n)).astype(np.float32)
    if pv_from is not None:
        # midday export for the PV cohort
        P[20:28, pv_from:] = -rng.uniform(100.0, 500.0, (8, n - pv_from))
    Q = (P * 0.1).astype(np.float32)
    mask = np.ones((T, n), dtype=bool)
    if not complete:
        mask[0, 0] = False
    return {
        "timestamps": idx.values.astype("datetime64[ns]").astype(np.int64),
        "load_ids": np.array([f"nmi_d{i}" for i in range(n)], dtype="<U32"),
        "P": P, "Q": Q, "mask": mask,
        "theta_A": np.full(T, 25.0, dtype=np.float32),
        "dt_minutes": np.float64(30.0),
    }


# ---------------------------------------------------------------------------
# Donor sampling
# ---------------------------------------------------------------------------
def test_gaps_filled_without_reuse():
    sub = _network(n_with_data=8, n_gaps=3)
    b, rep = sy.synthesise_substation(sub, _bundle(8), sy.config({}), "S1",
                                      np.random.default_rng(1))
    assert rep["method"] == "donor_sampling"
    assert rep["n_gaps"] == 3
    assert rep["n_donors"] == 8
    assert rep["n_donors_reused"] == 0
    assert rep["flag"] == sy.OK
    assert b["P"].shape == (T, 11)


def test_bundle_invariants():
    sub = _network(n_with_data=8, n_gaps=3)
    b, _ = sy.synthesise_substation(sub, _bundle(8), sy.config({}), "S1",
                                    np.random.default_rng(1))
    ids = list(map(str, b["load_ids"]))
    assert ids == sorted(ids), "load_ids must stay sorted"
    for k in ("P", "Q", "mask"):
        assert b[k].shape == (T, len(ids))
    syn = b["synthetic"]
    assert syn.sum() == 3
    # mask is the record of MEASURED data and must stay False where synthetic
    assert not b["mask"][:, syn].any()
    assert b["mask"][:, ~syn].all()
    # every synthetic column is a copy of some real column
    for j in np.where(syn)[0]:
        assert any(np.allclose(b["P"][:, j], b["P"][:, k])
                   for k in np.where(~syn)[0])


def test_no_gaps_is_a_noop():
    sub = _network(n_with_data=8, n_gaps=0)
    b0 = _bundle(8)
    b, rep = sy.synthesise_substation(sub, b0, sy.config({}), "S1",
                                      np.random.default_rng(1))
    assert rep["n_gaps"] == 0
    assert rep["method"] == "none"
    assert b["P"].shape == b0["P"].shape


def test_prefix_mismatch_is_not_a_gap():
    """Network ids carry nmi_, bundle ids do not — zero gaps, not eight."""
    sub = _network(n_with_data=8, n_gaps=0)
    b0 = _bundle(8)
    b0["load_ids"] = np.array([f"d{i}" for i in range(8)], dtype="<U32")
    _, rep = sy.synthesise_substation(sub, b0, sy.config({}), "S1",
                                      np.random.default_rng(1))
    assert rep["n_gaps"] == 0


def test_placeholders_skipped_by_default():
    sub = _network(n_with_data=8, n_gaps=2, placeholders=5)
    _, rep = sy.synthesise_substation(sub, _bundle(8), sy.config({}), "S1",
                                      np.random.default_rng(1))
    assert rep["n_gaps"] == 2
    assert rep["n_network_placeholder_loads"] == 5

    scfg = sy.config({"synthetic": {"include_placeholder_loads": True}})
    _, rep2 = sy.synthesise_substation(sub, _bundle(8), scfg, "S1",
                                       np.random.default_rng(1))
    assert rep2["n_gaps"] == 7


def test_incomplete_donor_excluded():
    sub = _network(n_with_data=8, n_gaps=2)
    _, rep = sy.synthesise_substation(sub, _bundle(8, complete=False),
                                      sy.config({}), "S1",
                                      np.random.default_rng(1))
    assert rep["n_donors"] == 7, "the partially-masked donor must be dropped"


def test_pv_matching_uses_der():
    """A target with user_data.der must get a donor that actually exports."""
    sub = _network(n_with_data=8, n_gaps=2, der_on=("g0", "g1"))
    b, rep = sy.synthesise_substation(sub, _bundle(8, pv_from=4),
                                      sy.config({}), "S1",
                                      np.random.default_rng(3))
    assert rep["n_targets_with_der_data"] == 2
    ids = list(map(str, b["load_ids"]))
    for cid in ("nmi_g0", "nmi_g1"):
        assert (b["P"][:, ids.index(cid)] < 0).any()


def test_more_gaps_than_donors_warns_and_reuses():
    sub = _network(n_with_data=6, n_gaps=20)
    _, rep = sy.synthesise_substation(sub, _bundle(6), sy.config({}), "S1",
                                      np.random.default_rng(1))
    assert rep["n_donors_reused"] > 0
    assert rep["flag"] == sy.WARN
    assert any("coincidence" in n for n in rep["notes"])


def test_feeder_donors_widen_the_pool():
    sub = _network(n_with_data=6, n_gaps=20)
    bank_ids = [f"nmi_x{i}" for i in range(30)]
    rng = np.random.default_rng(5)
    bank = (bank_ids,
            rng.uniform(200, 800, (T, 30)).astype(np.float32),
            rng.uniform(20, 80, (T, 30)).astype(np.float32))

    scfg = sy.config({"synthetic": {"allow_feeder_donors": True}})
    _, rep = sy.synthesise_substation(sub, _bundle(6), scfg, "S1",
                                      np.random.default_rng(1),
                                      feeder_donors=bank)
    assert rep["n_donors_local"] == 6
    assert rep["n_donors_from_feeder"] == 30
    assert rep["n_donors_reused"] == 0
    assert rep["flag"] == sy.OK


def _bank(n, seed=5):
    rng = np.random.default_rng(seed)
    return ([f"nmi_x{i}" for i in range(n)],
            rng.uniform(200, 800, (T, n)).astype(np.float32),
            rng.uniform(20, 80, (T, n)).astype(np.float32))


def test_feeder_pool_used_even_when_local_would_suffice():
    """Default is feeder-wide: consumption does not vary much geographically,
    and the larger pool avoids reuse. Widening is not gated on the local pool
    being too small."""
    sub = _network(n_with_data=8, n_gaps=3)
    _, rep = sy.synthesise_substation(sub, _bundle(8), sy.config({}), "S1",
                                      np.random.default_rng(1),
                                      feeder_donors=_bank(30))
    assert rep["n_donors_local"] == 8
    assert rep["n_donors_from_feeder"] == 30
    assert rep["n_donors"] == 38


def test_local_only_when_disabled():
    sub = _network(n_with_data=8, n_gaps=3)
    scfg = sy.config({"synthetic": {"allow_feeder_donors": False}})
    _, rep = sy.synthesise_substation(sub, _bundle(8), scfg, "S1",
                                      np.random.default_rng(1),
                                      feeder_donors=_bank(30))
    assert rep["n_donors_from_feeder"] == 0
    assert rep["n_donors"] == 8


def test_min_donors_measured_against_widened_pool():
    """A substation with 2 local donors must NOT fall back when the feeder has
    plenty — the guard is against a dataless feeder, not a thin substation."""
    sub = _network(n_with_data=2, n_gaps=6)
    _, rep = sy.synthesise_substation(sub, _bundle(2), sy.config({}), "S1",
                                      np.random.default_rng(1),
                                      feeder_donors=_bank(30))
    assert rep["method"] == "donor_sampling"
    assert rep["n_donors"] == 32


def test_substations_without_bundles_are_skipped():
    """A substation with no metered NMIs has no bundle from stage ⑤ and must be
    left alone: on a partial feeder export those are substations whose LV
    network was deliberately not supplied, not ones whose customers are
    missing. Synthesising them would invent a whole substation.
    """
    subs = {"S_HAS_DATA": _network(n_with_data=8, n_gaps=3),
            "S_NOT_SUPPLIED": _network(n_with_data=0, n_gaps=40)}
    bundles = {"S_HAS_DATA": _bundle(8)}          # stage ⑤ produced only this

    out, reports = sy.synthesise_all(subs, bundles, {},
                                     log=lambda *a, **k: None)
    assert set(out) == {"S_HAS_DATA"}
    assert [r["substation"] for r in reports] == ["S_HAS_DATA"]
    assert reports[0]["n_gaps"] == 3


def test_deterministic_for_a_given_seed():
    sub = _network(n_with_data=8, n_gaps=4)
    out = [sy.synthesise_substation(sub, _bundle(8), sy.config({}), "S1",
                                    np.random.default_rng(7))[1]
           ["donor_assignment"] for _ in range(2)]
    assert out[0] == out[1]


# ---------------------------------------------------------------------------
# Transformer fallback
# ---------------------------------------------------------------------------
def _tx_csv(tmp_path, kw):
    p = tmp_path / "tx.csv"
    pd.DataFrame({"date": _index().strftime("%d/%m/%Y %H:%M"),
                  "METER_SUMMATION (kW)": kw}).to_csv(p, index=False)
    return p


def test_fallback_sums_to_transformer_demand(tmp_path):
    """Metered + synthetic must reproduce the transformer series exactly."""
    sub = _network(n_with_data=2, n_gaps=6)
    b0 = _bundle(2)
    metered_kw = b0["P"].sum(axis=1) / 1000.0
    tx_kw = metered_kw + 40.0                      # 40 kW is unmetered
    scfg = sy.config({"synthetic": {
        "transformer_series": {"S1": str(_tx_csv(tmp_path, tx_kw))}}})

    b, rep = sy.synthesise_substation(sub, b0, scfg, "S1",
                                      np.random.default_rng(1))
    assert rep["method"] == "transformer_fallback"
    assert rep["flag"] == sy.FALLBACK
    total_kw = b["P"].sum(axis=1) / 1000.0
    np.testing.assert_allclose(total_kw, tx_kw, rtol=1e-4, atol=1e-3)


def test_fallback_without_series_raises():
    sub = _network(n_with_data=2, n_gaps=6)
    with pytest.raises(ValueError, match="min_donors"):
        sy.synthesise_substation(sub, _bundle(2), sy.config({}), "S1",
                                 np.random.default_rng(1))


def test_warn_zero_fills_with_zeros():
    sub = _network(n_with_data=2, n_gaps=6)
    scfg = sy.config({"synthetic": {"on_missing": "warn_zero"}})
    b, rep = sy.synthesise_substation(sub, _bundle(2), scfg, "S1",
                                      np.random.default_rng(1))
    assert rep["method"] == "zero_fill"
    assert not b["P"][:, b["synthetic"]].any()
    assert any(f["id"] == "SY003" and f["severity"] == "ERROR"
               for f in sy.findings([rep]))


def test_estimation_column_warns_as_circular(tmp_path, caplog):
    sub = _network(n_with_data=2, n_gaps=4)
    b0 = _bundle(2)
    p = tmp_path / "tx.csv"
    pd.DataFrame({"date": _index().strftime("%d/%m/%Y %H:%M"),
                  "ESTIMATION (kW)": b0["P"].sum(axis=1) / 1000.0 + 10.0
                  }).to_csv(p, index=False)
    scfg = sy.config({"synthetic": {"transformer_series": {
        "S1": {"path": str(p), "column": "ESTIMATION (kW)"}}}})
    with caplog.at_level("WARNING"):
        sy.synthesise_substation(sub, b0, scfg, "S1", np.random.default_rng(1))
    assert "circular" in caplog.text


# ---------------------------------------------------------------------------
# Whole-feeder driver and npz round-trip
# ---------------------------------------------------------------------------
def test_synthesise_all_and_npz_roundtrip(tmp_path):
    subs = {"S1": _network(n_with_data=8, n_gaps=3)}
    bundles, reports = sy.synthesise_all(subs, {"S1": _bundle(8)}, {},
                                         log=lambda *a, **k: None)
    assert reports[0]["n_gaps"] == 3

    p = tmp_path / "S1.npz"
    np.savez_compressed(p, **bundles["S1"])
    back = tsm.load_npz(p)
    assert back["synthetic"].sum() == 3
    np.testing.assert_array_equal(back["P"], bundles["S1"]["P"])
    # a second pass finds nothing left to do
    _, rep2 = sy.synthesise_substation(subs["S1"], back, sy.config({}), "S1",
                                       np.random.default_rng(1))
    assert rep2["n_gaps"] == 0


def test_disabled_is_passthrough():
    b0 = {"S1": _bundle(8)}
    out, reports = sy.synthesise_all({"S1": _network(8, 3)}, b0,
                                     {"synthetic": {"enabled": False}},
                                     log=lambda *a, **k: None)
    assert reports == []
    assert out["S1"]["P"].shape == (T, 8)


def test_non_participants_become_background_active_load():
    """The whole point: a load with no envelope must still load the network.

    Regression for _calculate_bus_loads_kw, which used to initialise
    bus_ld_a_kw to zero and never increment it, so synthetic (and any
    non-participating) NMIs were electrically invisible.
    """
    from fixtures import make_network
    from converge_soe.doe_solver import SoeSolver

    netw = make_network()
    lids = [c for c, comp in netw["components"].items()
            for t in comp if t == "Load"]
    f = pd.DataFrame({"real_power_w": [5000.0] * len(lids),
                      "reactive_power_var": [500.0] * len(lids)},
                     index=pd.Index(lids, name="load_id"))

    # Everyone participates: active power is the envelope variable, not
    # background load. Unchanged from before the fix.
    s_all = SoeSolver(netw, f, quiet=True)
    a_all, r_all = s_all._calculate_bus_loads_kw(s_all.buses.index)
    assert sum(a_all.values()) == pytest.approx(0.0)
    assert sum(r_all.values()) == pytest.approx(len(lids) * 0.5)

    # Three of six are non-participants -> 3 x 5 kW of background load.
    s_sub = SoeSolver(netw, f, participant_load_ids=lids[:3], quiet=True)
    a_sub, r_sub = s_sub._calculate_bus_loads_kw(s_sub.buses.index)
    assert sum(a_sub.values()) == pytest.approx(15.0)
    assert sum(r_sub.values()) == pytest.approx(len(lids) * 0.5)


def test_load_scaling():
    from converge_soe import pipeline as pl

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-06-02 00:30"] * 4),
        "load_id": ["a", "b", "c", "d"],
        "real_power_w": [1000.0, -500.0, 0.0, 2000.0],
        "reactive_power_var": [100.0, -50.0, 0.0, 200.0],
    })

    assert pl.stage_scale_load(df, {}, log=lambda *a, **k: None) is df

    # import x2, export unchanged
    out = pl.stage_scale_load(df, {"scaling": {"import": 2.0}},
                              log=lambda *a, **k: None)
    assert list(out["real_power_w"]) == [2000.0, -500.0, 0.0, 4000.0]
    assert list(out["reactive_power_var"]) == [200.0, -50.0, 0.0, 400.0]

    # export x3, import unchanged — the two are independent
    out = pl.stage_scale_load(df, {"scaling": {"export": 3.0}},
                              log=lambda *a, **k: None)
    assert list(out["real_power_w"]) == [1000.0, -1500.0, 0.0, 2000.0]

    # both, and the input is not mutated
    out = pl.stage_scale_load(df, {"scaling": {"import": 2.0, "export": 2.0}},
                              log=lambda *a, **k: None)
    assert list(out["real_power_w"]) == [2000.0, -1000.0, 0.0, 4000.0]
    assert list(df["real_power_w"]) == [1000.0, -500.0, 0.0, 2000.0]

    with pytest.raises(ValueError):
        pl.stage_scale_load(df, {"scaling": {"import": -1.0}},
                            log=lambda *a, **k: None)


def test_scaling_warns_past_envelope_cap():
    from converge_soe import pipeline as pl
    msgs = []
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2023-06-02 00:30"]),
        "load_id": ["a"], "real_power_w": [30_000.0],
        "reactive_power_var": [0.0]})
    pl.stage_scale_load(df, {"scaling": {"import": 3.0},
                             "envelope_abs_max": 50.0},
                        log=lambda m: msgs.append(m))
    assert any("envelope_abs_max" in m for m in msgs), \
        "90 kW peak against a 50 kW cap must warn — the envelope would be " \
        "clipped by the parameter, not the network"


def test_branches_oriented_away_from_source():
    """Every load bus must be some branch's to_bus, or its power vanishes.

    The DistFlow balance accumulates load at each branch's to_bus. CIM exports
    carry arbitrary edge orientation — on GOLDCR_8HB_LEXCEN 152 of 306 branches
    pointed back at the source and 58 of 59 load buses were never a to_bus, so
    the model solved with ~0 A everywhere and the envelopes went to the cap
    unopposed.
    """
    from fixtures import make_network
    from converge_soe.doe_solver import SoeSolver

    netw = make_network()
    lids = [c for c, comp in netw["components"].items()
            for t in comp if t == "Load"]
    f = pd.DataFrame({"real_power_w": [5000.0] * len(lids),
                      "reactive_power_var": [500.0] * len(lids)},
                     index=pd.Index(lids, name="load_id"))

    # Reverse every branch in the source ejson; orientation must be recovered.
    flipped = json.loads(json.dumps(netw))
    for comp in flipped["components"].values():
        for kind in ("Line", "Transformer"):
            if kind in comp and len(comp[kind].get("cons", [])) == 2:
                comp[kind]["cons"] = list(reversed(comp[kind]["cons"]))

    for label, ej in (("as-built", netw), ("all reversed", flipped)):
        s = SoeSolver(ej, f, quiet=True)
        to_buses = set(s.branches["to_bus_id"])
        missing = [b for b in set(s.load_buses) if b not in to_buses]
        assert not missing, f"{label}: load buses not downstream of any branch: {missing}"
        # the tree must stay connected: one branch per non-root bus
        assert len(s.branches) == len(s.buses) - 1


def test_series_reduction_preserves_the_electrical_model():
    """Merging line chains through empty nodes must change nothing that matters.

    LV models carry far more nodes than customers — poles, joints, cable
    changes. A node with two lines, no load and no transformer injects nothing,
    so both branches carry the same current and collapse to one. The test is
    that every load bus survives and the source-to-load impedance is unchanged.
    """
    from collections import defaultdict, deque
    from fixtures import make_network
    from converge_soe.doe_solver import SoeSolver

    netw = make_network()
    # splice 4 empty poles into the line feeding bus_a
    line = netw["components"]["line_a"]["Line"]
    src, dst = line["cons"][0]["node"], line["cons"][1]["node"]
    z, z0 = list(line["z"]), list(line.get("z0", [0.0, 0.0]))
    seg = [z[0] / 5.0, z[1] / 5.0]
    seg0 = [z0[0] / 5.0, z0[1] / 5.0]
    prev = src
    for k in range(4):
        nd = f"pole_{k}"
        netw["components"][nd] = {"Node": {"v_base": 0.415,
                                           "user_data": {"v_min": 0.373,
                                                         "v_max": 0.457}}}
        netw["components"][f"seg_{k}"] = {"Line": {
            "cons": [{"node": prev}, {"node": nd}],
            "length": 0.03, "z": list(seg), "z0": list(seg0),
            # deliberately TIGHTER than line_a so the merge must take the min
            "i_max": 250}}
        prev = nd
    line["cons"] = [{"node": prev}, {"node": dst}]
    line["z"], line["z0"] = list(seg), list(seg0)

    lids = [c for c, comp in netw["components"].items()
            for t in comp if t == "Load"]
    f = pd.DataFrame({"real_power_w": [5000.0] * len(lids),
                      "reactive_power_var": [500.0] * len(lids)},
                     index=pd.Index(lids, name="load_id"))

    plain = SoeSolver(netw, f, quiet=True)
    red = SoeSolver(netw, f, quiet=True, series_reduction=True)

    assert len(red.branches) < len(plain.branches), "nothing was merged"
    assert set(red.load_buses) == set(plain.load_buses)

    root = [c["Infeeder"]["cons"][0]["node"]
            for c in netw["components"].values() if "Infeeder" in c][0]

    def path_z(s):
        down = defaultdict(list)
        for b, r in s.branches.iterrows():
            down[r["from_bus_id"]].append((b, r["to_bus_id"]))
        z, q = {root: (0.0, 0.0)}, deque([root])
        while q:
            n = q.popleft()
            for b, m in down.get(n, []):
                z[m] = (z[n][0] + s.branches.at[b, "r_pu"],
                        z[n][1] + s.branches.at[b, "x_pu"])
                q.append(m)
        return z

    za, zb = path_z(plain), path_z(red)
    for bus in plain.load_buses:
        assert bus in zb, f"load bus {bus} lost in reduction"
        assert zb[bus][0] == pytest.approx(za[bus][0], abs=1e-12)
        assert zb[bus][1] == pytest.approx(za[bus][1], abs=1e-12)

    # the merged branch takes the TIGHTER current limit
    assert red.branches["i_max_pu"].min() == pytest.approx(
        plain.branches["i_max_pu"].min(), rel=1e-9)


def test_series_reduction_is_off_by_default():
    from fixtures import make_network
    from converge_soe.doe_solver import SoeSolver
    netw = make_network()
    lids = [c for c, comp in netw["components"].items()
            for t in comp if t == "Load"]
    f = pd.DataFrame({"real_power_w": [1000.0] * len(lids),
                      "reactive_power_var": [100.0] * len(lids)},
                     index=pd.Index(lids, name="load_id"))
    assert SoeSolver(netw, f, quiet=True).series_reduction is False


def test_findings_report_the_share():
    sub = _network(n_with_data=8, n_gaps=3)
    _, rep = sy.synthesise_substation(sub, _bundle(8), sy.config({}), "S1",
                                      np.random.default_rng(1))
    f = sy.findings([rep])
    assert f and f[0]["id"] == "SY001"
    assert "27%" in f[0]["message"]        # 3 of 11
