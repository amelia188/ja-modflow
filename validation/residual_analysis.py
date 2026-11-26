#!/usr/bin/env python3
"""
residual_analysis.py -- checks 1, 2 and 4 of the validation plan in one run:

  1. Parameter plausibility   (K, Ss, Sy per layer, active cells, vs. ranges you supply)
  2. Residual analysis        (statistics, per-well table, diagnostic plots, seasonal bias)
  4. Water budget             (discrepancy and the size of each flow term)

Usage
-----
    # Rio Cobre: run_model.py has been run; the simulation folder is found automatically
    python residual_analysis.py --obs observations.csv --ranges plausible_ranges_riocobre.csv

    # any MODFLOW 6 model
    python residual_analysis.py --sim-ws path/to/model --obs observations.csv [--model gwf_name]

Outputs (default: <sim-ws>/validation/residual_analysis/):
    summary.json, per_well.csv, residuals.csv, parameter_check.csv (if --ranges),
    fig1_sim_vs_obs.png, fig2_residual_map.png, fig3_residuals_time.png,
    fig4_residual_hist.png, fig5_monthly_bias.png (dated data), report.md

Residual = observed - simulated: positive means the model is too low.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mfval

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, POS, NEG = "#222222", "#888888", "#2166ac", "#b2182b"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_sim_vs_obs(df, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    d = df.dropna(subset=["sim_head"])
    lo, hi = np.nanmin([d.obs_head.min(), d.sim_head.min()]), np.nanmax([d.obs_head.max(), d.sim_head.max()])
    pad = 0.03 * (hi - lo) if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, zorder=1, label="1:1")
    ax.scatter(d.obs_head, d.sim_head, s=14, color=INK, alpha=0.6, lw=0, zorder=2)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Observed head (m)")
    ax.set_ylabel("Simulated head (m)")
    ax.set_title("Simulated vs observed heads")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_residual_map(df, gwf, path):
    """Mean residual per well. Left: whole model with constant-head cells; right: zoom on the wells."""
    g = df.dropna(subset=["residual"]).groupby("well_id").agg(
        res=("residual", "mean"), i=("i", "first"), j=("j", "first"))
    mg = gwf.modelgrid
    xc, yc = mg.xcellcenters, mg.ycellcenters
    act = (mfval._idomain(gwf)[0] > 0).astype(float)
    chd_xy = None
    chd = gwf.get_package("chd")
    if chd is not None:
        try:
            cid = list(chd.stress_period_data.get_data(0)["cellid"])
            chd_xy = ([xc[c[1], c[2]] for c in cid], [yc[c[1], c[2]] for c in cid])
        except Exception:
            pass
    x = np.array([xc[int(i), int(j)] for i, j in zip(g.i, g.j)])
    y = np.array([yc[int(i), int(j)] for i, j in zip(g.i, g.j)])
    lim = max(abs(g.res).max(), 1e-6) if len(g) else 1.0
    span = max(np.ptp(x), np.ptp(y), 2000) * 0.65 if len(g) else 0
    cx, cy = (x.mean(), y.mean()) if len(g) else (0, 0)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.0, 5.0), gridspec_kw={"width_ratios": [1, 1.5]})
    for ax, zoom in ((a0, False), (a1, True)):
        ax.contourf(xc, yc, act, levels=[0.5, 1.5], colors=["#f2f2ee"])
        ax.contour(xc, yc, act, levels=[0.5], colors=[MUTED], linewidths=0.8)
        if chd_xy and not zoom:
            ax.scatter(*chd_xy, s=3, color="#4a7fb0", lw=0, label="constant head")
        if len(g):
            sc = ax.scatter(x, y, c=g.res, cmap="RdBu", vmin=-lim, vmax=lim,
                            s=(30 + 120 * abs(g.res) / lim) if zoom else 14, edgecolor=INK, lw=0.4, zorder=3)
        ax.set_aspect("equal")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=6)
    a0.set_title("Whole model", fontsize=9)
    if chd_xy:
        a0.legend(frameon=False, loc="lower left", fontsize=7, markerscale=3)
    if len(g):
        a0.add_patch(plt.Rectangle((cx - span, cy - span), 2 * span, 2 * span, fill=False, ec=INK, lw=0.8))
        a1.set_xlim(cx - span, cx + span); a1.set_ylim(cy - span, cy + span)
        for wid, xi, yi in zip(g.index, x, y):
            a1.annotate(wid, (xi, yi), xytext=(6, 4), textcoords="offset points", fontsize=6.5, color=INK)
        cb = fig.colorbar(sc, ax=a1, shrink=0.85)
        cb.set_label("Mean residual, observed − simulated (m)")
    a1.set_title("Wells (blue: model too low, red: too high)", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_residuals_time(df, path):
    use_date = "date" in df and df["date"].notna().any()
    tcol = "date" if use_date else "time"
    d = df.dropna(subset=["residual"])
    worst = d.groupby("well_id")["residual"].apply(lambda r: np.sqrt(np.mean(r ** 2))).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(0, color=MUTED, lw=1)
    single = d.groupby("well_id")[tcol].nunique().max() <= 1
    if single:
        # one measurement per well: a residual-by-well bar chart says more than a time series
        w = worst.index
        r = d.groupby("well_id")["residual"].mean().reindex(w)
        ax.bar(range(len(w)), r, color=[POS if v > 0 else NEG for v in r])
        ax.set_xticks(range(len(w))); ax.set_xticklabels(w, rotation=90, fontsize=7)
        ax.set_ylabel("Residual, observed − simulated (m)")
        ax.set_title("Residual by well (one measurement each; worst first)")
    else:
        for wid in d.well_id.unique():
            g = d[d.well_id == wid].sort_values(tcol)
            ax.plot(g[tcol], g.residual, color="#cccccc", lw=0.8)
        for wid in worst.index[:3]:
            g = d[d.well_id == wid].sort_values(tcol)
            ax.plot(g[tcol], g.residual, lw=1.6, marker="o", ms=2.5, label=f"{wid} (RMSE {worst[wid]:.2f} m)")
        ax.set_xlabel("Date" if use_date else "Model time (days)")
        ax.set_ylabel("Residual, observed − simulated (m)")
        ax.set_title("Residuals over time (three worst wells highlighted)")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_hist(df, path):
    r = df.residual.dropna()
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.hist(r, bins=min(30, max(5, len(r) // 3)), color=INK, alpha=0.75)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(r.mean(), color=NEG, lw=1.4, label=f"mean {r.mean():+.2f} m")
    ax.set_xlabel("Residual, observed − simulated (m)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_monthly_bias(mb, path):
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.axhline(0, color=MUTED, lw=1)
    ax.bar(mb.index, mb["mean"], color=[POS if v > 0 else NEG for v in mb["mean"]])
    for m_, row in mb.iterrows():
        ax.text(m_, row["mean"], f"n={int(row['count'])}", ha="center",
                va="bottom" if row["mean"] >= 0 else "top", fontsize=6, color=MUTED)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS)
    ax.set_ylabel("Mean residual (m)")
    ax.set_title("Mean residual by calendar month")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def verdicts(m, wells, budget, pcheck, df, mb):
    """Plain-language flags against common rules of thumb."""
    v = []
    if m["n"] == 0:
        return ["No observation could be compared with the model; see the data problems below."]
    if m["nrmse_pct"] <= 5:
        v.append(f"Fit: NRMSE {m['nrmse_pct']:.1f}% is good (≤ 5%).")
    elif m["nrmse_pct"] <= 10:
        v.append(f"Fit: NRMSE {m['nrmse_pct']:.1f}% is acceptable (5–10%).")
    else:
        v.append(f"Fit: NRMSE {m['nrmse_pct']:.1f}% is weak (> 10%); review before relying on predictions.")
    if np.isfinite(m["r2"]) and m["r2"] < 0:
        v.append(f"R² is {m['r2']:.2f}: the model explains less of the head variation than the mean of the "
                 f"observations would. With a small head range ({m['obs_range_m']:.1f} m) this usually means a bias "
                 f"or a wrong gradient, not noise.")
    if abs(m["mean_error_m"]) > 0.25 * m["rmse_m"]:
        side = "low" if m["mean_error_m"] > 0 else "high"
        v.append(f"Bias: mean error {m['mean_error_m']:+.2f} m, so the model runs systematically {side}. "
                 f"Check recharge, the constant-head level and K.")
    else:
        v.append(f"Bias: mean error {m['mean_error_m']:+.2f} m is small relative to RMSE.")
    trend_col = next((c for c in ("residual_trend_m_per_year", "residual_trend_m_per_time") if c in wells), None)
    if trend_col:
        tcol = "date" if trend_col.endswith("year") else "time"
        span = ((df["date"].max() - df["date"].min()).days / 365.25) if tcol == "date" else (df["time"].max() - df["time"].min())
        drift_total = wells[trend_col].abs() * span
        n_drift = int((drift_total > 0.5 * m["rmse_m"]).sum())
        if n_drift:
            v.append(f"Drift: {n_drift} well(s) show residuals trending over time. A model that repeats an average "
                     f"year cannot follow a trend, so check abstraction history, land-use change or storage.")
    if mb is not None and len(mb) >= 3:
        amp = mb["mean"].max() - mb["mean"].min()
        # compare with the scatter left after removing each well's own bias, so a large overall
        # bias does not hide a seasonal pattern
        d = df.dropna(subset=["residual"])
        sd_within = float((d["residual"] - d.groupby("well_id")["residual"].transform("mean")).std())
        if amp > max(0.25, 0.5 * sd_within):
            v.append(f"Season: mean residual swings {amp:.2f} m between months "
                     f"({MONTHS[int(mb['mean'].idxmax()) - 1]} most positive, {MONTHS[int(mb['mean'].idxmin()) - 1]} most "
                     f"negative). Check recharge timing and specific yield.")
    if "max_abs_period_discrepancy_pct" in budget:
        b = budget["max_abs_period_discrepancy_pct"]
        v.append(f"Water budget: worst period discrepancy {b:.3f}% ({'OK' if b < 1 else 'too high, check solver settings'}; target < 1%).")
    if pcheck is not None and len(pcheck):
        bad = pcheck[pcheck.status != "OK"]
        v.append("Parameters: all within plausible ranges." if bad.empty else
                 f"Parameters: {len(bad)} outside plausible ranges ({', '.join(f'{r.param} layer {r.layer}' for r in bad.itertuples())}). "
                 f"A good fit with implausible parameters suggests compensating errors.")
    return v


def _fmt(x, f):
    return "n/a" if x is None or not np.isfinite(x) else format(x, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-ws", default=None, help="Folder with the MODFLOW 6 simulation (already run). "
                                                   "Default: the run_model.py WORKSPACE")
    ap.add_argument("--obs", required=True, help="Observation CSV")
    ap.add_argument("--model", default=None, help="GWF model name if the simulation has several")
    ap.add_argument("--ranges", default=None, help="CSV of plausible ranges: param,layer,min,max[,source]")
    ap.add_argument("--include-spinup", action="store_true",
                    help="Keep observations that fall in the spin-up cycles (excluded by default)")
    ap.add_argument("--out", default=None, help="Output folder (default: <sim-ws>/validation/residual_analysis)")
    a = ap.parse_args()

    sim, gwf = mfval.load_model(a.sim_ws, a.model)
    sim_ws = Path(gwf.model_ws)
    out = Path(a.out) if a.out else sim_ws / "validation" / "residual_analysis"
    out.mkdir(parents=True, exist_ok=True)
    rec = mfval.run_record(sim_ws)

    obs = mfval.load_obs(a.obs, gwf, sim)
    df_all = mfval.simulated_at_obs(gwf, obs)
    df_all.to_csv(out / "residuals.csv", index=False)
    problems = mfval.data_problems(df_all)
    df = df_all if a.include_spinup else df_all[~df_all["in_spinup"]]

    m = mfval.fit_metrics(df)
    wells = mfval.per_well_metrics(df)
    wells.to_csv(out / "per_well.csv")
    mb = mfval.monthly_bias(df)
    budget = mfval.budget_check(gwf)
    params = mfval.parameter_summary(gwf)
    pcheck = mfval.check_ranges(params, a.ranges) if a.ranges else None
    if pcheck is not None:
        pcheck.to_csv(out / "parameter_check.csv", index=False)

    figs = []
    if m["n"]:
        plot_sim_vs_obs(df, out / "fig1_sim_vs_obs.png")
        plot_residual_map(df, gwf, out / "fig2_residual_map.png")
        plot_residuals_time(df, out / "fig3_residuals_time.png")
        plot_hist(df, out / "fig4_residual_hist.png")
        figs = ["fig1_sim_vs_obs.png", "fig2_residual_map.png", "fig3_residuals_time.png", "fig4_residual_hist.png"]
        if mb is not None:
            plot_monthly_bias(mb, out / "fig5_monthly_bias.png")
            figs.append("fig5_monthly_bias.png")

    v = verdicts(m, wells, budget, pcheck, df, mb)
    summary = {"run": rec, "sim_ws": str(sim_ws), "observations_file": str(Path(a.obs).resolve()),
               "residual_convention": "observed - simulated", "overall": m, "water_budget": budget,
               "parameters": params, "verdicts": v, "data_problems": problems,
               "not_compared": int(df_all.sim_head.isna().sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    head = f"Model: `{gwf.name}`"
    if rec:
        head += f" · version {rec.get('model_version') or '?'} · run {rec.get('run_started') or '?'}"
        commit = rec.get("git_commit")
        head += f" · commit `{commit[:10]}`" if commit else " · no Git commit recorded"
        if rec.get("git_uncommitted_changes"):
            head += " (uncommitted changes)"
    rates = budget.get("final_period_rates_m3_per_day", {})
    md = ["# Residual analysis report", "", head, "",
          f"Observations compared: {m['n']} of {len(df_all)} · wells: {df.dropna(subset=['residual']).well_id.nunique()}"
          f" · residual = observed − simulated (positive: model too low)", "",
          "## Verdicts", *[f"- {x}" for x in v], ""]
    if problems:
        md += ["## Data problems", *[f"- {p}" for p in problems], ""]
    md += ["## Overall fit",
           "| Metric | Value |", "| --- | --- |",
           f"| RMSE | {_fmt(m['rmse_m'], '.3f')} m |", f"| NRMSE (RMSE / observed range) | {_fmt(m['nrmse_pct'], '.2f')} % |",
           f"| Mean error (bias) | {_fmt(m['mean_error_m'], '+.3f')} m |", f"| MAE | {_fmt(m['mae_m'], '.3f')} m |",
           f"| R² | {_fmt(m['r2'], '.3f')} |", f"| Observed range | {_fmt(m['obs_range_m'], '.2f')} m |", ""]
    if len(wells):
        md += ["## Worst five wells",
               "| Well | n | RMSE (m) | Mean error (m) |", "| --- | --- | --- | --- |",
               *[f"| {w} | {int(r.n)} | {_fmt(r.rmse_m, '.3f')} | {_fmt(r.mean_error_m, '+.3f')} |"
                 for w, r in wells.head(5).iterrows()], ""]
    if rates:
        md += ["## Water budget, final period (m³/day)", "| Term | In | Out |", "| --- | --- | --- |",
               *[f"| {t} | {rates['in'].get(t, 0):,.0f} | {rates['out'].get(t, 0):,.0f} |"
                 for t in sorted(set(rates["in"]) | set(rates["out"]))],
               "", f"Worst period discrepancy: {budget['max_abs_period_discrepancy_pct']:.3f} %", ""]
    if pcheck is not None and len(pcheck):
        md += ["## Parameters (active cells)", "| Parameter | Layer | Model range | Plausible range | Status |",
               "| --- | --- | --- | --- | --- |",
               *[f"| {r.param} | {r.layer} | {r.model_min:g} – {r.model_max:g} | {r.plausible_min:g} – {r.plausible_max:g} | {r.status} |"
                 for r in pcheck.itertuples()], ""]
    if figs:
        md += ["## Figures", *[f"![]({f})" for f in figs]]
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(v))
    if problems:
        print("\nData problems:\n  " + "\n  ".join(problems))
    print(f"\nReport written to {out / 'report.md'}")


if __name__ == "__main__":
    main()
