"""Curtailment, ageing and cost metrics (the formulas live here AND in
docs/RESULTS_GUIDE.md — keep them in sync).

Sign conventions: ``P_des(i,t) = −real_power_w/1000`` kW, injection-positive,
so PV export is positive; it is what the customer would do unconstrained
(stored per row as ``p_des_kw`` in doe.parquet).

    Curtailed export      E_curt(s)  = Σ_i Σ_t max(0, P_des − doe_ub_kw)   · Δt   [kWh]
    Constrained import    E_imp(s)   = Σ_i Σ_t max(0, −P_des − |doe_lb_kw|)· Δt   [kWh]
    Export enabled by DTR E_enabled  = E_curt(doe_static) − E_curt(doe_dtr)       [kWh]
    Curtailment cost                 = E_curt(s) × price                          [$]

``doe_ub_kw = +∞`` for bau, so E_curt(bau) = 0 by construction.

Transformer ageing (IEEE C57.91):

    F_AA(t)  = exp(15000/383 − 15000/(θ_HS(t) + 273))     (383 K = 110 °C ref)
    L(s)     = Σ_t F_AA(t) · Δt                            equivalent ageing hours
    %LOL(s)  = 100 · L(s) / 180000                         (normal insulation life)
    ΔL       = L(bau) − L(s)                               avoided ageing

Dynamic rating utilisation (the most direct evidence the DTR works):

    headroom(t) = i_max_pu_dtr(t) / i_max_pu_static        >1 = extra capacity
    share of period with headroom > 1

With ``thermal.derive_i_rated`` (the default), K = 1 at nameplate, so
headroom(t) = √(K2_max(t)) straight out of thermal.parquet.
"""

import numpy as np
import pandas as pd

from . import thermal as _thermal

HOURS_PER_YEAR = 8766.0


def dt_hours_of(doe_df):
    ts = pd.to_datetime(doe_df["timestamp"]).sort_values().unique()
    if len(ts) < 2:
        return 0.5
    d = pd.Series(ts).diff().dropna().dt.total_seconds() / 3600.0
    return float(d.mode().iat[0])


# ---------------------------------------------------------------------------
# Curtailment
# ---------------------------------------------------------------------------
def curtailment(doe_df, dt_h=None, price_per_kwh=0.10, price_series=None):
    """Curtailment metrics from one substation-scenario doe table.

    Returns dict with E_curt_kwh, E_imp_kwh, E_des_export_kwh, cost columns,
    plus per-NMI breakdowns.
    """
    df = doe_df.copy()
    dt_h = dt_h if dt_h is not None else dt_hours_of(df)
    ub = df["doe_ub_kw"].replace([np.inf], np.nan)
    curt_kw = (df["p_des_kw"] - ub).clip(lower=0).fillna(0.0)
    lb_mag = df["doe_lb_kw"].abs().replace([np.inf], np.nan)
    imp_kw = ((-df["p_des_kw"]) - lb_mag).clip(lower=0).fillna(0.0)
    des_exp_kw = df["p_des_kw"].clip(lower=0)

    if price_series is not None:
        pr = pd.to_datetime(df["timestamp"]).map(price_series).fillna(price_per_kwh)
    else:
        pr = price_per_kwh

    per_nmi = (pd.DataFrame({
        "load_id": df["load_id"],
        "E_curt_kwh": curt_kw * dt_h,
        "E_imp_kwh": imp_kw * dt_h,
        "E_des_export_kwh": des_exp_kw * dt_h,
        "curtailment_cost": curt_kw * dt_h * pr,
    }).groupby("load_id").sum())

    return {
        "E_curt_kwh": float(per_nmi["E_curt_kwh"].sum()),
        "E_imp_kwh": float(per_nmi["E_imp_kwh"].sum()),
        "E_des_export_kwh": float(per_nmi["E_des_export_kwh"].sum()),
        "curtailment_cost": float(per_nmi["curtailment_cost"].sum()),
        "per_nmi": per_nmi,
        "dt_h": dt_h,
    }


# ---------------------------------------------------------------------------
# Ageing
# ---------------------------------------------------------------------------
def ageing(thermal_df, dt_h):
    """Equivalent ageing hours, %LOL, and life-days-per-year equivalents.

    Uses theta_HS_posthoc_C (θ_HS under the actual clipped-forecast
    behaviour — comparable across scenarios) when present, else the model
    trajectory theta_HS_C (envelope-worst-case).
    """
    if ("theta_HS_posthoc_C" in thermal_df.columns
            and thermal_df["theta_HS_posthoc_C"].notna().any()):
        th = thermal_df["theta_HS_posthoc_C"].astype(float).values
    else:
        th = thermal_df["theta_HS_C"].astype(float).values
    faa = np.exp(15000.0 / 383.0 - 15000.0 / (th + 273.0))
    L = float(faa.sum() * dt_h)
    period_h = len(th) * dt_h
    per_year = L * (HOURS_PER_YEAR / period_h) if period_h else np.nan
    return {
        "ageing_hours": L,
        "percent_LOL": _thermal.percent_loss_of_life(L),
        "peak_theta_HS_C": float(np.max(th)) if len(th) else np.nan,
        "mean_F_AA": float(faa.mean()) if len(th) else np.nan,
        # days of insulation life consumed per calendar year of this operating
        # pattern (24 days/yr = ageing exactly at the C57.91 reference rate is
        # 365; above 365 means ageing faster than real time)
        "equivalent_days_of_life_lost_per_year":
            float(per_year / 24.0) if period_h else np.nan,
    }


# ---------------------------------------------------------------------------
# Dynamic-rating utilisation
# ---------------------------------------------------------------------------
def dtr_utilisation(thermal_dtr_df):
    k2 = thermal_dtr_df["K2_max"].astype(float)
    headroom = np.sqrt(k2.clip(lower=0))
    ok = headroom.notna() & np.isfinite(headroom)
    if not ok.any():
        return {"mean_headroom": np.nan, "share_above_nameplate": np.nan,
                "n_already_over_limit": 0, "n_thermal_binding": 0,
                "n_not_binding": 0}
    st = thermal_dtr_df["dtr_status"].astype(str)
    return {
        "mean_headroom": float(headroom[ok].mean()),
        "min_headroom": float(headroom[ok].min()),
        "max_headroom": float(headroom[ok].max()),
        "share_above_nameplate": float((headroom[ok] > 1.0).mean()),
        "n_already_over_limit": int((st == "already_over_limit").sum()),
        "n_thermal_binding": int((st == "thermal_binding").sum()),
        "n_not_binding": int((st == "not_binding").sum()),
    }


# ---------------------------------------------------------------------------
# Energy balance
# ---------------------------------------------------------------------------
def energy_balance(branch_df, doe_df, sub_ej, tol=0.01):
    """Transformer inflow ≈ Σ loads + losses, within tol (sanity check)."""
    tx_ids = [k for k, v in sub_ej["components"].items() if "Transformer" in v]
    if not tx_ids:
        return None
    tx = branch_df[branch_df["id"] == tx_ids[0]]
    if tx.empty:
        return None
    inflow_kwh = tx["p_w_oel"].sum() / 1000.0
    # loads: use the oel-side world (participants at lower bound = max import).
    # Balance is only exactly closed for BAU (fixed loads); for DOE scenarios
    # compare against total branch flows into load buses instead — here we
    # report the BAU-style residual which the sanity check applies to bau only.
    load_kwh = -doe_df["p_des_kw"].sum()
    if load_kwh == 0:
        return None
    rel = abs(inflow_kwh - load_kwh) / max(abs(load_kwh), 1e-9)
    return {"tx_inflow_sum_kw": inflow_kwh, "load_sum_kw": load_kwh,
            "relative_residual": rel, "within_tol": bool(rel <= tol)}
