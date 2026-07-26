"""Pipeline orchestration: the nine stages behind scripts/run_feeder.py.

  ① build network      ② prepare timeseries   ③ select feeder / map NMIs
  ④ preflight          ⑤ pre-index            ⑥ solve (3 scenarios × subs)
  ⑦ checkpoint (continuous, inside ⑥)         ⑧ analyse   ⑨ explain (docs)

Everything that does not change per timestep is pushed out of the solve loop
and cached in build/ (fingerprinted via build/manifest.json). Scripts under
scripts/ are thin argument-parsing wrappers around these functions.
"""

import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import io as cio
from . import preflight as pfl
from . import scenarios as scen
from . import timeseries as tsm
from .network import batch_convert, extract_lv, validate as netval

logger = logging.getLogger("converge_soe.pipeline")

REPO = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(repo=REPO, feeder=None, overrides=None):
    cfg = {}
    default = Path(repo) / "config" / "default.yaml"
    if default.exists():
        cfg = yaml.safe_load(default.read_text()) or {}
    if feeder:
        fcfg = Path(repo) / "config" / "feeders" / f"{feeder}.yaml"
        if fcfg.exists():
            _deep_update(cfg, yaml.safe_load(fcfg.read_text()) or {})
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def _deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_transformer_params(cfg, repo=REPO):
    cls = cfg.get("thermal", {}).get("transformer_class", "distribution_onan")
    p = Path(repo) / "config" / "transformers" / f"{cls}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"transformer class file not found: {p}")
    return yaml.safe_load(p.read_text())


# ---------------------------------------------------------------------------
# Manifest-based caching
# ---------------------------------------------------------------------------
def _manifest_path(repo):
    return Path(repo) / "build" / "manifest.json"


def read_manifest(repo=REPO):
    p = _manifest_path(repo)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def update_manifest(repo, key, fingerprint):
    p = _manifest_path(repo)
    p.parent.mkdir(parents=True, exist_ok=True)
    m = read_manifest(repo)
    m[key] = {"fingerprint": fingerprint, "updated": datetime.now().isoformat()}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=1))
    os.replace(tmp, p)


def cache_fresh(repo, key, fingerprint, *outputs):
    m = read_manifest(repo).get(key)
    return (m is not None and m.get("fingerprint") == fingerprint
            and all(Path(o).exists() for o in outputs))


# ---------------------------------------------------------------------------
# Stage ① — build network
# ---------------------------------------------------------------------------
def stage_build_network(repo=REPO, cfg=None, log=logger.info):
    cfg = cfg or {}
    xml_dir = Path(repo) / "data" / "xml"
    out_dir = Path(repo) / "build" / "network"
    xmls = sorted(xml_dir.rglob("*.xml"))
    if not xmls:
        log("stage ① build network: no XMLs in data/xml — skipped "
            "(supply --feeder-json to use a prebuilt network)")
        return None
    fp = tsm.file_fingerprint(*xmls)
    if cache_fresh(repo, "network", fp, out_dir / "batch_report.csv"):
        log("stage ① build network: cached (inputs unchanged)")
        return out_dir
    rows, scan = batch_convert.run_batch(xml_dir, out_dir, jobs=2, log=log)
    missing = [(r["feeder"], r["missing_lv_circuits"]) for r in rows
               if r["missing_lv_circuits"]]
    if missing:
        log("MISSING LVNetwork XMLs (request these):")
        for f, m in missing:
            log(f"  {f}: {m}")
    update_manifest(repo, "network", fp)
    return out_dir


def list_feeders(repo=REPO):
    """Table of every feeder found in build/network (or data-derived)."""
    rows = []
    fdir = Path(repo) / "build" / "network" / "feeders"
    for p in sorted(fdir.glob("*_network.json")) if fdir.exists() else []:
        ej = json.loads(p.read_text())
        comps = netval.components_by_type(ej)
        rows.append({"feeder": ej.get("user_data", {}).get("feeder", p.stem),
                     "file": p.name,
                     "substations": len(comps["Transformer"]),
                     "loads": len(comps["Load"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage ② — prepare timeseries
# ---------------------------------------------------------------------------
def stage_prepare_timeseries(repo=REPO, cfg=None, meter_files=None,
                             log=logger.info):
    cfg = cfg or {}
    tcfg = cfg.get("timeseries", {})
    meter_dir = Path(repo) / "data" / "meter"
    files = ([Path(f) for f in meter_files] if meter_files else
             [p for p in sorted(meter_dir.glob("*.csv"))
              if p.name != "README.md"])
    if not files:
        raise FileNotFoundError(
            "no meter CSVs found (data/meter/ is empty and no --meter given)")
    out = Path(repo) / "build" / "timeseries" / "all_nmis.parquet"
    fp = tsm.file_fingerprint(*files) + f"|{tcfg.get('values_are')}"
    if cache_fresh(repo, "timeseries", fp, out):
        log("stage ② prepare timeseries: cached (inputs unchanged)")
        return pd.read_parquet(out)
    df = tsm.prepare(files, values_are=tcfg.get("values_are"),
                     reactive_from_q=tcfg.get("reactive_from_q", True),
                     pf_ratio=tcfg.get("pf_ratio", 0.4))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    update_manifest(repo, "timeseries", fp)
    log(f"stage ② prepare timeseries: {df['load_id'].nunique()} NMIs × "
        f"{df['timestamp'].nunique()} timesteps -> {out}")
    return df


def load_ambient(repo, cfg, timestamps, log=logger.info):
    """Ambient series per config: cache | constant | open_meteo."""
    acfg = cfg.get("ambient", {})
    source = acfg.get("source", "cache")
    if source == "constant":
        c = float(acfg.get("constant_c", 25.0))
        return pd.Series(c, index=pd.DatetimeIndex(timestamps))
    cache_dir = Path(repo) / acfg.get("cache_dir", "data/ambient")
    csvs = [p for p in sorted(cache_dir.glob("*.csv"))]
    if csvs:
        frames = []
        for p in csvs:
            d = pd.read_csv(p)
            cols = {c.lower(): c for c in d.columns}
            tcol = cols.get("timestamp") or list(d.columns)[0]
            vcol = cols.get("temperature_c") or list(d.columns)[1]
            ts, _ = tsm.parse_timestamps(d[tcol].astype(str))
            frames.append(pd.Series(pd.to_numeric(d[vcol], errors="coerce").values,
                                    index=ts))
        s = pd.concat(frames).sort_index()
        return s[~s.index.duplicated()]
    if source == "open_meteo":
        return fetch_open_meteo(cfg, timestamps, cache_dir, log=log)
    log("no ambient cache found — falling back to constant "
        f"{acfg.get('constant_c', 25.0)} °C (set ambient.source or run "
        "prepare_timeseries --fetch-ambient)")
    return pd.Series(float(acfg.get("constant_c", 25.0)),
                     index=pd.DatetimeIndex(timestamps))


def fetch_open_meteo(cfg, timestamps, cache_dir, log=logger.info):
    import requests
    acfg = cfg.get("ambient", {})
    ts = pd.DatetimeIndex(timestamps)
    start, end = ts.min().date(), ts.max().date()
    log(f"fetching Open-Meteo ERA5 ambient {start} → {end}")
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={"latitude": acfg.get("site_lat", -35.2035),
                "longitude": acfg.get("site_lon", 149.1548),
                "hourly": "temperature_2m", "start_date": str(start),
                "end_date": str(end), "timezone": "UTC+11"},
        timeout=60)
    r.raise_for_status()
    om = r.json()
    idx = pd.to_datetime(om["hourly"]["time"])
    s = pd.Series(om["hourly"]["temperature_2m"], index=idx, dtype=float)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"open_meteo_{start}_{end}.csv"
    s.rename("temperature_c").rename_axis("timestamp").to_csv(out)
    log(f"cached ambient to {out}")
    return s


# ---------------------------------------------------------------------------
# Stage ③ — select feeder, map NMIs to substations
# ---------------------------------------------------------------------------
def stage_select_feeder(feeder_ej, df_long, repo=REPO, feeder_name="FEEDER",
                        mapping_csv=None, only=None, log=logger.info):
    """Extract every substation and assign NMIs.

    Precedence: (a) data/mapping/loads.csv substation column,
    (b) membership of the extracted LV subtree, (c) report the unmatched.
    Writes build/feeders/<FEEDER>/nmi_index.csv recording which rule assigned
    each NMI. Returns (substations: {name: sub_ej}, nmi_index: DataFrame).
    """
    comps = feeder_ej["components"]
    tx_items = [(k, v["Transformer"]) for k, v in comps.items()
                if "Transformer" in v]
    if only:
        subs_filter = [s.strip().lower() for s in only.split(",")]
        tx_items = [(k, t) for k, t in tx_items
                    if any(s in k.lower()
                           or s in t.get("user_data", {}).get("name", "").lower()
                           for s in subs_filter)]

    # mapping file (rule a)
    map_rule_a = {}
    mp = Path(mapping_csv) if mapping_csv else Path(repo) / "data" / "mapping" / "loads.csv"
    if mp.exists():
        md = pd.read_csv(mp, dtype=str)
        cols = {c.lower(): c for c in md.columns}
        if "nmi" in cols and "substation" in cols:
            for _, r in md.iterrows():
                map_rule_a[f"nmi_{r[cols['nmi']]}"] = str(r[cols["substation"]]).strip()
                map_rule_a[str(r[cols["nmi"]])] = str(r[cols["substation"]]).strip()

    ts_ids = set(df_long["load_id"].astype(str).unique())
    mapping, rep = tsm.reconcile_load_ids(ts_ids, set(
        k for k, v in comps.items() if "Load" in v))

    substations = {}
    index_rows = []
    assigned = set()
    for tx_id, tx in tx_items:
        name = tx.get("user_data", {}).get("name", tx_id)
        safe = name.replace(" ", "_").replace("/", "_")
        sub_ej = extract_lv.extract(feeder_ej, tx_id, source_name=feeder_name)
        sub_loads = {k for k, v in sub_ej["components"].items() if "Load" in v}
        substations[safe] = sub_ej
        subname = tx.get("user_data", {}).get("substation", name)
        for ts_id, net_id in mapping.items():
            rule = None
            if map_rule_a.get(ts_id) == subname or map_rule_a.get(net_id) == subname:
                rule = "mapping_csv"
            elif net_id in sub_loads:
                rule = "lv_subtree"
            if rule:
                assigned.add(ts_id)
                index_rows.append({"load_id": ts_id, "network_id": net_id,
                                   "substation": safe, "rule": rule})

    unmatched = sorted(ts_ids - assigned)
    for u in unmatched:
        index_rows.append({"load_id": u, "network_id": mapping.get(u, ""),
                           "substation": "", "rule": "UNASSIGNED"})
    nmi_index = pd.DataFrame(index_rows)
    out = Path(repo) / "build" / "feeders" / feeder_name / "nmi_index.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    nmi_index.to_csv(out, index=False)

    empty = [s for s in substations
             if not (nmi_index["substation"] == s).any()]
    log(f"stage ③ select feeder: {len(substations)} substation(s), "
        f"{len(assigned)}/{len(ts_ids)} NMIs assigned"
        + (f", {len(unmatched)} unmatched" if unmatched else "")
        + (f", {len(empty)} substation(s) with zero NMIs: {empty}" if empty else ""))
    return substations, nmi_index


# ---------------------------------------------------------------------------
# Stage ⑤ — pre-index
# ---------------------------------------------------------------------------
def stage_preindex(substations, nmi_index, df_long, ambient, repo=REPO,
                   feeder_name="FEEDER", log=logger.info):
    """One .npz per substation, cached against inputs."""
    bundles = {}
    for safe, sub_ej in substations.items():
        rows = nmi_index[nmi_index["substation"] == safe]
        if rows.empty:
            continue
        # rename timeseries ids to network ids once, up front
        ren = dict(zip(rows["load_id"], rows["network_id"]))
        sub_df = df_long[df_long["load_id"].isin(ren)].copy()
        sub_df["load_id"] = sub_df["load_id"].map(ren)
        out = (Path(repo) / "build" / "timeseries" / "by_substation"
               / feeder_name / f"{safe}.npz")
        bundles[safe] = tsm.pre_index(sub_df, sorted(set(ren.values())),
                                      ambient=ambient, out_path=out)
        log(f"stage ⑤ pre-index: {safe}: "
            f"{bundles[safe]['P'].shape[0]} steps × "
            f"{bundles[safe]['P'].shape[1]} NMIs -> {out.name}")
    return bundles


# ---------------------------------------------------------------------------
# Stage ⑥ — solve
# ---------------------------------------------------------------------------
def _solve_one(args):
    """Worker: one (scenario, substation). Runs in its own process."""
    (scenario, safe, sub_ej, bundle_path, tparams, cfg, outdir, fingerprint,
     code_version, resume, restart, verbosity) = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # keep Pyomo temp files off any cloud-synced folder
    import tempfile
    from pyomo.common.tempfiles import TempfileManager
    TempfileManager.tempdir = tempfile.gettempdir()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"csoe.{scenario}.{safe}")
    fh = logging.FileHandler(outdir / "run.log")
    fh.setLevel(logging.DEBUG if verbosity >= 2 else logging.INFO)
    lg.addHandler(fh)

    bundle = tsm.load_npz(bundle_path)
    t0 = time.perf_counter()
    try:
        if restart:
            cio.fresh_output_dir(outdir)
        start_after, state, n_done, n_fail = (None, {}, 0, 0)
        if resume and not restart:
            start_after, state, n_done, n_fail = cio.resume_state(
                outdir, fingerprint)
        writer = cio.SubstationWriter(outdir, safe, scenario,
                                      inputs_fingerprint=fingerprint,
                                      code_version=code_version,
                                      flush_every=cfg.get("flush_every", 200),
                                      csv_mirror=cfg.get("csv_mirror", False),
                                      resume_existing=start_after is not None)
        writer.n_completed, writer.n_failed = n_done, n_fail
        writer.last_completed_timestamp = start_after

        if scenario == "bau":
            scen.run_bau_scenario(sub_ej, bundle, tparams, cfg, writer)
        else:
            scen.run_doe_scenario(scenario, sub_ej, bundle, tparams, cfg,
                                  writer, start_after=start_after,
                                  thermal_state=state,
                                  fast=cfg.get("solver", {}).get("fast", False))
        writer.close()
        dt = time.perf_counter() - t0
        return {"scenario": scenario, "substation": safe, "status": "ok",
                "n_completed": writer.n_completed,
                "n_failed": writer.n_failed, "seconds": round(dt, 1)}
    except cio.StaleResumeError as e:
        return {"scenario": scenario, "substation": safe,
                "status": "stale_checkpoint", "error": str(e)}
    except Exception as e:   # worker must report, not crash the pool
        lg.exception("substation failed")
        return {"scenario": scenario, "substation": safe, "status": "error",
                "error": f"{type(e).__name__}: {e}"}


def stage_solve(substations, bundles, tparams_by_sub, cfg, run_dir,
                repo=REPO, feeder_name="FEEDER", scenarios=None,
                resume=True, restart=False, verbosity=1, log=logger.info):
    scenarios = scenarios or cfg.get("scenarios", list(scen.SCENARIOS))
    jobs = cfg.get("jobs", "auto")
    if jobs in ("auto", None):
        jobs = max((os.cpu_count() or 2) - 1, 1)
    code_version = cio.git_code_version(repo)

    tasks = []
    for safe in substations:
        if safe not in bundles:
            log(f"  skipping {safe}: no NMI data")
            continue
        bundle_path = (Path(repo) / "build" / "timeseries" / "by_substation"
                       / feeder_name / f"{safe}.npz")
        for sc in scenarios:
            outdir = Path(run_dir) / "scenarios" / sc / safe
            fp = tsm.fingerprint(
                json.dumps(substations[safe], sort_keys=True, default=str).encode(),
                bundle_path.read_bytes(),
                json.dumps(tparams_by_sub.get(safe), sort_keys=True).encode(),
                json.dumps({k: cfg.get(k) for k in
                            ("envelope_abs_max", "thermal", "solver")},
                           sort_keys=True, default=str).encode(),
            )
            tasks.append((sc, safe, substations[safe], str(bundle_path),
                          tparams_by_sub.get(safe), cfg, str(outdir), fp,
                          code_version, resume, restart, verbosity))

    results = []
    use_tqdm = verbosity >= 1
    bar = None
    if use_tqdm:
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(tasks), unit="run", desc="solving")
        except ImportError:
            bar = None

    if jobs <= 1 or len(tasks) <= 1:
        for t in tasks:
            r = _solve_one(t)
            results.append(r)
            _report_result(r, run_dir, bar, verbosity, log)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_solve_one, t): t for t in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                _report_result(r, run_dir, bar, verbosity, log)
    if bar is not None:
        bar.close()
    return results


def _report_result(r, run_dir, bar, verbosity, log):
    cio.update_run_manifest(run_dir, r["substation"], r["scenario"],
                            r["status"], **{k: v for k, v in r.items()
                                            if k not in ("substation",
                                                         "scenario", "status")})
    if bar is not None:
        bar.update(1)
    if r["status"] != "ok":
        log(f"  {r['scenario']}/{r['substation']}: {r['status'].upper()}: "
            f"{r.get('error', '')[:300]}")
    elif verbosity >= 1:
        msg = (f"  {r['scenario']}/{r['substation']}: {r['n_completed']} steps"
               + (f", {r['n_failed']} failed" if r.get("n_failed") else "")
               + f" in {r.get('seconds', '?')}s")
        (bar.write if bar is not None else log)(msg)


# ---------------------------------------------------------------------------
# Run directory helpers
# ---------------------------------------------------------------------------
def make_run_dir(repo, feeder_name, run_id=None):
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path(repo) / "out" / feeder_name / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d, run_id


def write_resolved_config(run_dir, cfg):
    (Path(run_dir) / "config_resolved.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False))


def derive_tparams(sub_ej, base_params, cfg, dt_minutes):
    """Per-substation thermal params: dt from the data, I_rated from s_max."""
    if base_params is None:
        return None
    tp = dict(base_params)
    tp["dt"] = float(dt_minutes)
    if cfg.get("thermal", {}).get("derive_i_rated", True):
        comps = netval.components_by_type(sub_ej)
        tx = next(iter(comps["Transformer"].values()), None)
        if tx is not None and "s_max" in tx:
            s_w = tx["s_max"] * sub_ej["units"]["power"]
            v_v = tx["v_winding_base"][1] * sub_ej["units"]["voltage"]
            tp["I_rated"] = round(s_w / v_v, 3)
    return tp
