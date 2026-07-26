"""Interrupt-and-resume must produce output identical to an uninterrupted
run (and refuse stale checkpoints)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from converge_soe import io as cio
from converge_soe import scenarios as scen
from converge_soe import timeseries as tsm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from fixtures import make_network, make_timeseries, make_ambient  # noqa: E402

TP = dict(tau_TO=60.0, tau_W=7.0, delta_theta_TO_R=55.0,
          delta_theta_HS_R=23.0, R=6.0, n=0.8, m=0.8,
          I_rated=100e3 / 415, theta_HS_max=120.0, dt=30.0)

CFG = {"envelope_abs_max": 50.0,
       "thermal": {"k_emergency": 2.0, "k2_floor": 1e-6},
       "solver": {"soft_limits": True, "usable_export_weight": 0.01},
       "flush_every": 5, "ambient": {"constant_c": 25.0}}


def _ipopt_available():
    from pyomo.environ import SolverFactory
    try:
        return SolverFactory("ipopt").available(exception_flag=False)
    except Exception:
        return False


def make_bundle(n_steps=12):
    ts = make_timeseries(n_days=1)
    net = make_network()
    lids = sorted(ts["load_id"].unique())
    amb = make_ambient(ts["timestamp"])
    keep = sorted(ts["timestamp"].unique())[:n_steps]
    ts = ts[ts["timestamp"].isin(keep)]
    return net, tsm.pre_index(ts, lids, ambient=amb)


def run_span(net, bundle, outdir, start_after=None, state=None,
             n_steps_then_stop=None):
    """Run doe_dtr, optionally stopping after n steps (the 'Ctrl-C')."""
    w = cio.SubstationWriter(outdir, "tx1", "doe_dtr",
                             inputs_fingerprint="sha256:fx",
                             flush_every=CFG["flush_every"])
    stopped = {}

    class Stop(Exception):
        pass

    n_done = [0]

    def cb(i, ok):
        n_done[0] += 1
        if n_steps_then_stop and n_done[0] >= n_steps_then_stop:
            raise Stop()

    try:
        scen.run_doe_scenario("doe_dtr", net, bundle, TP, CFG, w,
                              start_after=start_after, thermal_state=state,
                              progress_cb=cb)
    except Stop:
        pass
    w.flush()
    w.close()
    return w


@pytest.mark.skipif(not _ipopt_available(), reason="ipopt not available")
def test_interrupt_resume_identical(tmp_path):
    net, bundle = make_bundle()

    # uninterrupted reference
    ref_dir = tmp_path / "ref"
    run_span(net, bundle, ref_dir)
    ref = pd.read_parquet(ref_dir / "doe.parquet")
    ref_th = pd.read_parquet(ref_dir / "thermal.parquet")

    # interrupted after 6 steps, then resumed
    res_dir = tmp_path / "resumed"
    run_span(net, bundle, res_dir, n_steps_then_stop=6)
    ck = cio.read_checkpoint(res_dir)
    assert ck is not None and ck["n_timesteps_completed"] >= 5
    start_after, state, n_done, n_fail = cio.resume_state(res_dir, "sha256:fx")
    w = cio.SubstationWriter(res_dir, "tx1", "doe_dtr",
                             inputs_fingerprint="sha256:fx",
                             flush_every=CFG["flush_every"],
                             resume_existing=True)
    w.n_completed = n_done
    w.last_completed_timestamp = start_after
    scen.run_doe_scenario("doe_dtr", net, bundle, TP, CFG, w,
                          start_after=start_after, thermal_state=state)
    w.close()

    got = pd.read_parquet(res_dir / "doe.parquet")
    got_th = pd.read_parquet(res_dir / "thermal.parquet")
    # concatenated output identical to the uninterrupted run
    pd.testing.assert_frame_equal(
        got.sort_values(["timestamp", "load_id"]).reset_index(drop=True),
        ref.sort_values(["timestamp", "load_id"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(
        got_th.sort_values("timestamp").reset_index(drop=True),
        ref_th.sort_values("timestamp").reset_index(drop=True))


def test_stale_checkpoint_refused(tmp_path):
    d = tmp_path / "s"
    w = cio.SubstationWriter(d, "tx1", "doe_dtr", inputs_fingerprint="sha256:a")
    w.complete_timestep("2026-01-01T00:00", {})
    w.close()
    with pytest.raises(cio.StaleResumeError):
        cio.resume_state(d, "sha256:DIFFERENT")
    # --restart path
    assert cio.resume_state(d, "sha256:DIFFERENT", restart=True)[0] is None
