"""Every figure the analysis produces.

Conventions (docs/RESULTS_GUIDE.md documents what each plot should look
like): one figure per file, PNG at 200 dpi AND PDF for LaTeX, consistent
colours (BAU grey, static DOE blue, DTR DOE orange), every plot titled,
axis-labelled with units, legended, with a footer naming run id / feeder /
date. Each plot's underlying data is saved as a sibling .csv so figures can
be rebuilt without re-solving.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

logger = logging.getLogger(__name__)

COLORS = {"bau": "#8a8a8a", "doe_static": "#1f77b4", "doe_dtr": "#ff7f0e"}
LABELS = {"bau": "BAU", "doe_static": "DOE (static rating)",
          "doe_dtr": "DOE (dynamic thermal rating)"}


def _footer(ax, run_meta):
    ax.figure.text(0.99, 0.005,
                   f"run {run_meta.get('run_id', '?')} · "
                   f"feeder {run_meta.get('feeder', '?')} · "
                   f"{run_meta.get('date', '')}",
                   ha="right", va="bottom", fontsize=6, color="#888888")


def _save(fig, outdir, name, data=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(outdir / f"{name}.png", dpi=200)
    fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)
    if data is not None:
        data.to_csv(outdir / f"{name}.csv")
    logger.info("plot %s", name)


def _grouped_bar(ax, table, ylabel):
    subs = list(table.index)
    scen_cols = list(table.columns)
    x = np.arange(len(subs))
    w = 0.8 / max(len(scen_cols), 1)
    for i, sc in enumerate(scen_cols):
        ax.bar(x + i * w - 0.4 + w / 2, table[sc].values, w,
               label=LABELS.get(sc, sc), color=COLORS.get(sc, None))
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


# 1 ─ curtailed energy by scenario ------------------------------------------
def curtailed_energy_by_scenario(metrics_df, outdir, run_meta):
    t = metrics_df.pivot_table(index="substation", columns="scenario",
                               values="E_curt_kwh")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _grouped_bar(ax, t, "Curtailed export [kWh]")
    ax.set_title("Curtailed export energy by scenario")
    _footer(ax, run_meta)
    _save(fig, outdir, "curtailed_energy_by_scenario", t)


# 2 ─ curtailment cost -------------------------------------------------------
def curtailment_cost_by_scenario(metrics_df, outdir, run_meta, price):
    t = metrics_df.pivot_table(index="substation", columns="scenario",
                               values="curtailment_cost")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _grouped_bar(ax, t, "Curtailment cost [$]")
    ax.set_title("Curtailment cost by scenario")
    ax.text(0.5, 1.001, f"assumed flat price: ${price:.2f}/kWh "
            "(an assumption, not a result)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
            color="#666666")
    _footer(ax, run_meta)
    _save(fig, outdir, "curtailment_cost_by_scenario", t)


# 3 ─ energy enabled by DTR --------------------------------------------------
def energy_enabled_by_dtr(metrics_df, outdir, run_meta):
    t = metrics_df.pivot_table(index="substation", columns="scenario",
                               values="E_curt_kwh")
    if not {"doe_static", "doe_dtr"} <= set(t.columns):
        return
    en = (t["doe_static"] - t["doe_dtr"]).sort_values()
    des = metrics_df.pivot_table(index="substation", columns="scenario",
                                 values="E_des_export_kwh")["doe_dtr"]
    pct = 100 * en / des.replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(en) + 1.5)))
    ax.barh(en.index, en.values, color=COLORS["doe_dtr"])
    for i, (sub, v) in enumerate(en.items()):
        ax.text(v, i, f"  {pct[sub]:.1f}% of desired export"
                if np.isfinite(pct[sub]) else "  n/a",
                va="center", fontsize=7)
    ax.set_xlabel("Export enabled by DTR, E_curt(static) − E_curt(dtr) [kWh]")
    ax.set_title("Additional export enabled by the dynamic thermal rating")
    _footer(ax, run_meta)
    _save(fig, outdir, "energy_enabled_by_dtr",
          pd.DataFrame({"E_enabled_kwh": en, "pct_of_desired_export": pct}))


# 4 ─ ageing by scenario -----------------------------------------------------
def transformer_ageing_by_scenario(metrics_df, outdir, run_meta):
    t = metrics_df.pivot_table(index="substation", columns="scenario",
                               values="ageing_hours")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _grouped_bar(ax, t, "Equivalent ageing [hours]")
    ax.set_yscale("log")
    ax.set_title("Transformer insulation ageing by scenario (log scale)")
    _footer(ax, run_meta)
    _save(fig, outdir, "transformer_ageing_by_scenario", t)


# 5 ─ avoided ageing ---------------------------------------------------------
def avoided_ageing(metrics_df, outdir, run_meta):
    t = metrics_df.pivot_table(index="substation", columns="scenario",
                               values="ageing_hours")
    if "bau" not in t.columns:
        return
    rows = {}
    for sc in ("doe_static", "doe_dtr"):
        if sc in t.columns:
            rows[sc] = (t["bau"] - t[sc]) / 24.0
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _grouped_bar(ax, d, "Avoided ageing vs BAU [days of life]")
    ax.set_title("Transformer life saved relative to BAU")
    _footer(ax, run_meta)
    _save(fig, outdir, "avoided_ageing", d)


# 6 ─ THE DTR money shot -----------------------------------------------------
def dynamic_rating_vs_static(thermal_dtr, sub, outdir, run_meta, week=None):
    th = thermal_dtr.copy()
    th["timestamp"] = pd.to_datetime(th["timestamp"])
    th = th.sort_values("timestamp")
    if week is not None:
        th = th[(th["timestamp"] >= week[0]) & (th["timestamp"] < week[1])]
    elif len(th) > 7 * 48:
        # a representative week: the one containing the peak dynamic rating
        i_peak = th["K2_max"].idxmax()
        t_peak = th.loc[i_peak, "timestamp"]
        th = th[(th["timestamp"] >= t_peak - pd.Timedelta(days=3))
                & (th["timestamp"] <= t_peak + pd.Timedelta(days=4))]
    K = np.sqrt(th["K2_max"].clip(lower=0))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(th["timestamp"], K, color=COLORS["doe_dtr"],
            label="dynamic rating K_max(t)")
    ax.axhline(1.0, color=COLORS["doe_static"], ls="--",
               label="static nameplate (K = 1)")
    ax.set_ylabel("Transformer rating [× rated current]")
    ax2 = ax.twinx()
    ax2.plot(th["timestamp"], th["theta_A_C"], color="#999999", lw=0.8,
             label="ambient")
    ax2.set_ylabel("Ambient [°C]", color="#777777")
    ax.set_title(f"Dynamic vs static transformer rating — {sub}")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    fig.autofmt_xdate()
    _footer(ax, run_meta)
    _save(fig, outdir, f"dynamic_rating_vs_static_{sub}",
          th[["timestamp", "K2_max", "theta_A_C", "dtr_status"]].set_index("timestamp"))


# 7 ─ hotspot timeseries -----------------------------------------------------
def hotspot_timeseries(thermal_by_scenario, sub, theta_max, outdir, run_meta):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = {}
    for sc, th in thermal_by_scenario.items():
        th = th.copy()
        th["timestamp"] = pd.to_datetime(th["timestamp"])
        th = th.sort_values("timestamp")
        ax.plot(th["timestamp"], th["theta_HS_C"], color=COLORS.get(sc),
                label=LABELS.get(sc, sc), lw=0.9)
        data[sc] = th.set_index("timestamp")["theta_HS_C"]
        amb = th.set_index("timestamp")["theta_A_C"]
    ax.plot(amb.index, amb.values, color="#bbbbbb", lw=0.7, label="ambient")
    ax.axhline(theta_max, color="red", ls=":", label=f"θ_HS_max = {theta_max:g} °C")
    ax.set_ylabel("Hot-spot temperature [°C]")
    ax.set_title(f"Transformer hot-spot temperature — {sub}")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    _footer(ax, run_meta)
    _save(fig, outdir, f"hotspot_timeseries_{sub}", pd.DataFrame(data))


# 8 ─ hotspot duration curve -------------------------------------------------
def hotspot_duration_curve(thermal_by_scenario, theta_max, outdir, run_meta):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = {}
    for sc, th in thermal_by_scenario.items():
        v = np.sort(th["theta_HS_C"].values)[::-1]
        pct = 100 * np.arange(len(v)) / max(len(v), 1)
        ax.plot(pct, v, color=COLORS.get(sc), label=LABELS.get(sc, sc))
        data[sc] = pd.Series(v)
    ax.axhline(theta_max, color="red", ls=":", label=f"θ_HS_max = {theta_max:g} °C")
    ax.set_xlabel("Share of period [%]")
    ax.set_ylabel("Hot-spot temperature [°C]")
    ax.set_title("Hot-spot temperature duration curve")
    ax.legend(fontsize=8)
    _footer(ax, run_meta)
    _save(fig, outdir, "hotspot_duration_curve", pd.DataFrame(data))


# 9 ─ transformer loading duration curve ------------------------------------
def transformer_loading_duration_curve(thermal_by_scenario, outdir, run_meta):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = {}
    for sc, th in thermal_by_scenario.items():
        k = np.sort(np.sqrt(th["K2_actual"].clip(lower=0).values))[::-1]
        pct = 100 * np.arange(len(k)) / max(len(k), 1)
        ax.plot(pct, k, color=COLORS.get(sc), label=LABELS.get(sc, sc))
        data[sc] = pd.Series(k)
    ax.axhline(1.0, color="black", ls=":", lw=0.8, label="K = 1 (nameplate)")
    ax.set_xlabel("Share of period [%]")
    ax.set_ylabel("Loading K = I / I_rated")
    ax.set_title("Transformer loading duration curve")
    ax.legend(fontsize=8)
    _footer(ax, run_meta)
    _save(fig, outdir, "transformer_loading_duration_curve", pd.DataFrame(data))


# 10 ─ envelope vs desired ---------------------------------------------------
def envelope_vs_desired(doe_by_scenario, sub, outdir, run_meta, day=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = {}
    base = next(iter(doe_by_scenario.values())).copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    if day is None:
        # representative day = the one with the largest desired export
        daily = base.groupby(base["timestamp"].dt.date)["p_des_kw"] \
            .apply(lambda s: s.clip(lower=0).sum())
        day = daily.idxmax()
    for sc, doe in doe_by_scenario.items():
        d = doe.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d = d[d["timestamp"].dt.date == day]
        agg = d.groupby("timestamp").agg(
            p_des=("p_des_kw", "sum"), ub=("doe_ub_kw", "sum"))
        if sc == "bau":
            ax.plot(agg.index, agg["p_des"], color=COLORS["bau"],
                    label="desired injection (forecast)")
            data["p_des"] = agg["p_des"]
            continue
        ax.plot(agg.index, agg["ub"], color=COLORS.get(sc),
                label=f"{LABELS.get(sc, sc)} upper envelope")
        data[f"ub_{sc}"] = agg["ub"]
        curt = (agg["p_des"] - agg["ub"]).clip(lower=0)
        ax.fill_between(agg.index, agg["ub"], agg["ub"] + curt,
                        color=COLORS.get(sc), alpha=0.2)
    ax.set_ylabel("Power [kW, injection +ve]")
    ax.set_title(f"Envelopes vs desired injection — {sub}, {day} "
                 "(shaded = curtailed)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    _footer(ax, run_meta)
    _save(fig, outdir, f"envelope_vs_desired_{sub}", pd.DataFrame(data))


# 11 ─ seasonal curtailment heatmap ------------------------------------------
def seasonal_curtailment_heatmap(doe_by_scenario, outdir, run_meta):
    scens = [s for s in ("doe_static", "doe_dtr") if s in doe_by_scenario]
    if not scens:
        return
    mats, vmax = {}, 0.0
    for sc in scens:
        d = doe_by_scenario[sc].copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        curt = (d["p_des_kw"] - d["doe_ub_kw"].replace([np.inf], np.nan)) \
            .clip(lower=0).fillna(0)
        d = d.assign(curt=curt)
        pivot = d.pivot_table(index=d["timestamp"].dt.month,
                              columns=d["timestamp"].dt.hour,
                              values="curt", aggfunc="mean")
        mats[sc] = pivot
        if pivot.size:
            vmax = max(vmax, float(np.nanmax(pivot.values)))
    fig, axes = plt.subplots(1, len(scens), figsize=(6 * len(scens), 4),
                             squeeze=False)
    for ax, sc in zip(axes[0], scens):
        pv = mats[sc]
        im = ax.imshow(pv.values, aspect="auto", origin="lower",
                       vmin=0, vmax=vmax or 1, cmap="YlOrRd",
                       extent=(pv.columns.min() - 0.5, pv.columns.max() + 0.5,
                               pv.index.min() - 0.5, pv.index.max() + 0.5))
        ax.set_title(LABELS.get(sc, sc), fontsize=9)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Month")
    fig.colorbar(im, ax=axes[0].tolist(), label="Mean curtailment [kW]")
    fig.suptitle("When does the constraint bind? Mean curtailment by "
                 "month × hour", fontsize=10)
    _footer(axes[0][0], run_meta)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "seasonal_curtailment_heatmap.png", dpi=200)
    fig.savefig(outdir / "seasonal_curtailment_heatmap.pdf")
    plt.close(fig)
    for sc in scens:
        mats[sc].to_csv(outdir / f"seasonal_curtailment_heatmap_{sc}.csv")


# 12 ─ violations summary ----------------------------------------------------
def violations_summary(viol_by_scenario, outdir, run_meta):
    rows = []
    for sc, v in viol_by_scenario.items():
        if v is None or v.empty:
            rows.append({"scenario": sc})
            continue
        counts = v.groupby("kind").size()
        rows.append({"scenario": sc, **counts.to_dict()})
    t = pd.DataFrame(rows).set_index("scenario").fillna(0)
    kinds = [c for c in t.columns]
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(t))
    for k in kinds:
        ax.bar(t.index, t[k].values, bottom=bottom, label=k)
        bottom += t[k].values
    ax.set_ylabel("Violation count [interval·component]")
    ax.set_title("Recorded limit violations by type and scenario")
    if kinds:
        ax.legend(fontsize=8)
    _footer(ax, run_meta)
    _save(fig, outdir, "violations_summary", t)
