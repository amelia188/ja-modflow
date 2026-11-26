#!/usr/bin/env python3
"""
split_sample.py -- check 3 of the validation plan: does the model predict data it was NOT calibrated on?

Workflow
--------
1. SPLIT the observations into a calibration set and a held-out test set:

     # temporal: calibrate on the first 70% of the record, test on the last 30%
     python split_sample.py split --obs obs.csv --mode temporal --train-frac 0.7 --out splits

   With a `date` column the temporal split is by calendar date (earlier years calibrate, later
   years test), which is what matters for a model that repeats an average year.

     # spatial: k-fold leave-wells-out (each fold holds out about 1/k of the wells)
     python split_sample.py split --obs obs.csv --mode wells --k 5 --out splits

   If you calibrate with PEST/PEST++, add --pst model.pst (and an `obsnme` column in obs.csv):
   a copy of the control file is written per split with held-out observations weighted 0.

2. RECALIBRATE on the training set only, with your usual method (PEST, PEST++, manual), or with
   the built-in quick option below, which fits K and recharge multipliers:

     python split_sample.py recalibrate --obs splits/temporal/train.csv \
         --out runs/temporal [--sim-ws model] [--k-layers 1 2] [--recharge] [--max-runs 60]

   --sim-ws defaults to the run_model.py WORKSPACE. One Rio Cobre run takes about half a minute,
   so --max-runs 60 is about half an hour.

3. EVALUATE the recalibrated model on both sets:

     python split_sample.py evaluate --sim-ws runs/temporal --train splits/temporal/train.csv \
         --test splits/temporal/test.csv --out results/temporal

   For k-fold, run recalibrate + evaluate per fold, then summarise:

     python split_sample.py summarise --results results/fold_*

Reading the result: if test-set RMSE stays close to training RMSE (ratio below about 1.5), the
model has real predictive skill. A large jump means it is fitting noise or compensating for a
structural error.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mfval

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
TRAIN_C, TEST_C, MUTED = "#4d4d4d", "#d6604d", "#999999"


# --------------------------------------------------------------------------- #
# 1. Split
# --------------------------------------------------------------------------- #
def cmd_split(a):
    obs = pd.read_csv(a.obs)
    out = Path(a.out)
    splits = {}
    if a.mode == "temporal":
        tcol = "date" if "date" in obs.columns else "time" if "time" in obs.columns else None
        if tcol is None:
            raise SystemExit("A temporal split needs a date or time column (month alone repeats every year).")
        t = pd.to_datetime(obs[tcol]) if tcol == "date" else obs[tcol]
        if t.nunique() < 2:
            raise SystemExit("All observations share one time: use --mode wells instead.")
        cut = t.quantile(a.train_frac)
        tr, te = obs[t <= cut], obs[t > cut]
        splits["temporal"] = (tr, te)
        print(f"Temporal split at {tcol} = {cut if tcol == 'date' else format(cut, 'g')}: "
              f"{len(tr)} training, {len(te)} test observations")
        if tcol == "time":
            print("  note: with model time, a split inside one simulated year separates SEASONS, not years. "
                  "Give dates to split by year.")
    else:
        wells = np.array(sorted(obs.well_id.astype(str).unique()))
        if len(wells) < a.k:
            raise SystemExit(f"Only {len(wells)} wells for {a.k} folds: lower --k.")
        rng = np.random.default_rng(a.seed)
        rng.shuffle(wells)
        for f, held in enumerate(np.array_split(wells, a.k)):
            te = obs[obs.well_id.isin(held)]
            tr = obs[~obs.well_id.isin(held)]
            splits[f"fold_{f + 1}"] = (tr, te)
            print(f"fold_{f + 1}: holds out {len(held)} wells ({', '.join(map(str, held))})")

    for name, (tr, te) in splits.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        tr.to_csv(d / "train.csv", index=False)
        te.to_csv(d / "test.csv", index=False)
        if a.pst:
            write_pst_copy(a.pst, te, d / f"{Path(a.pst).stem}_{name}.pst")
    print(f"Splits written to {out}/")


def write_pst_copy(pst_path, test, out_path):
    if not Path(pst_path).exists():
        print(f"  (PEST file {pst_path} not found: skipping control file)")
        return
    try:
        import pyemu
    except ImportError:
        print("  (pyemu not installed: skipping PEST control file; pip install pyemu)")
        return
    if "obsnme" not in test.columns:
        print("  (no obsnme column: skipping PEST control file)")
        return
    pst = pyemu.Pst(str(pst_path))
    names = set(test.obsnme.str.lower())
    hit = pst.observation_data.obsnme.str.lower().isin(names)
    pst.observation_data.loc[hit, "weight"] = 0.0
    pst.write(str(out_path))
    print(f"  wrote {out_path} ({hit.sum()} held-out observations set to weight 0)")


# --------------------------------------------------------------------------- #
# 2. Quick recalibration (multipliers on K per layer and on recharge)
# --------------------------------------------------------------------------- #
def run_with_multipliers(base_ws, work_ws, model, k_layers, k_mults, rch_mult, exe):
    sim, gwf = mfval.load_model(base_ws, model, exe)
    sim.set_sim_path(str(work_ws))
    k = np.array(gwf.npf.k.array, dtype=float)
    for lay, mlt in zip(k_layers, k_mults):
        k[lay - 1] = k[lay - 1] * mlt
    gwf.npf.k.set_data(k)
    if gwf.npf.k33.array is not None and k_layers:     # keep the K33/K anisotropy ratio
        k33 = np.array(gwf.npf.k33.array, dtype=float)
        for lay, mlt in zip(k_layers, k_mults):
            k33[lay - 1] = k33[lay - 1] * mlt
        gwf.npf.k33.set_data(k33)
    if rch_mult is not None:
        rch = gwf.get_package("rcha")
        if rch is None:
            raise ValueError("--recharge needs an array recharge package (RCHA).")
        data = rch.recharge.get_data()
        rch.recharge.set_data({kper: arr * rch_mult for kper, arr in data.items() if arr is not None})
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    if not ok:
        return None, None
    for f in ("grid.npz",):                       # keep the run usable by the export and the 3D viewer
        if (Path(base_ws) / f).exists():
            shutil.copy2(Path(base_ws) / f, Path(work_ws) / f)
    return sim.get_model(gwf.name), sim


def cmd_recalibrate(a):
    from scipy.optimize import minimize

    import time
    sim0, gwf0 = mfval.load_model(a.sim_ws, a.model)
    base_ws, out = Path(gwf0.model_ws), Path(a.out)
    exe = a.exe or mfval.default_exe()
    train = mfval.load_obs(a.obs, gwf0, sim0)
    layers = a.k_layers or [1]
    n_k = len(layers)
    n_par = n_k + (1 if a.recharge else 0)
    history = []
    work = out.parent / (out.name + "_trial")
    t0 = time.time()

    def objective(logm):
        k_mults = [float(x) for x in 10 ** logm[:n_k]]
        rch_mult = float(10 ** logm[n_k]) if a.recharge else None
        gwf, _ = run_with_multipliers(base_ws, work, a.model, layers, k_mults, rch_mult, exe)
        if gwf is None:
            rmse = np.inf
        else:
            d = mfval.simulated_at_obs(gwf, train)
            d = d[~d["in_spinup"]]
            mt = mfval.fit_metrics(d)
            # cells that go dry lose their observations: penalise so the search cannot "win" by drying wells
            rmse = mt["rmse_m"] * (1 + d["sim_head"].isna().mean()) if mt["n"] else np.inf
        history.append({"run": len(history) + 1, "k_mults": k_mults, "rch_mult": rch_mult, "train_rmse_m": rmse})
        eta = (time.time() - t0) / len(history) * (a.max_runs - len(history))
        print(f"  run {len(history):3d}/{a.max_runs}: K x {np.round(k_mults, 3).tolist()}"
              f"{'' if rch_mult is None else f', R x {rch_mult:.3f}'}  ->  train RMSE {rmse:.4f} m"
              f"   (about {eta / 60:.0f} min left at most)")
        return rmse if np.isfinite(rmse) else 1e6

    # Start from the current calibration (multipliers = 1). The search works in log10 space, and
    # SciPy's default first steps from 0 are only 0.00025 (x 1.0006); start with x 10**step instead.
    x0 = np.zeros(n_par)
    simplex = np.vstack([x0] + [x0 + a.step * np.eye(n_par)[p] for p in range(n_par)])
    lim = np.log10(a.max_factor)
    print(f"Recalibrating on {len(train)} training observations with {exe}; "
          f"multipliers between 1/{a.max_factor:g} and {a.max_factor:g}")
    res = minimize(objective, x0, method="Nelder-Mead", bounds=[(-lim, lim)] * n_par,
                   options={"maxfev": a.max_runs, "xatol": 0.005, "fatol": 1e-3, "initial_simplex": simplex})
    best_run = min(history, key=lambda h: h["train_rmse_m"])       # the best run actually made
    k_mults, rch_mult = best_run["k_mults"], best_run["rch_mult"]
    if out.exists():
        shutil.rmtree(out)
    run_with_multipliers(base_ws, out, a.model, layers, k_mults, rch_mult, exe)
    shutil.rmtree(work, ignore_errors=True)
    rec = mfval.run_record(base_ws)
    info = {"base_run": str(base_ws), "base_model_version": rec.get("model_version"),
            "base_git_commit": rec.get("git_commit"), "observations": str(Path(a.obs).resolve()),
            "k_layers": layers, "k_multipliers": k_mults, "recharge_multiplier": rch_mult,
            "train_rmse_m": float(best_run["train_rmse_m"]), "model_runs": len(history),
            "converged": bool(res.success), "message": str(res.message)}
    (out / "recalibration.json").write_text(json.dumps(info, indent=2))
    pd.DataFrame(history).to_csv(out / "recalibration_history.csv", index=False)
    print(f"Best: {info}\nRecalibrated model written to {out}/")


# --------------------------------------------------------------------------- #
# 3. Evaluate
# --------------------------------------------------------------------------- #
def cmd_evaluate(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sim, gwf = mfval.load_model(a.sim_ws, a.model)
    tr = mfval.simulated_at_obs(gwf, mfval.load_obs(a.train, gwf, sim)); tr["set"] = "train"
    te = mfval.simulated_at_obs(gwf, mfval.load_obs(a.test, gwf, sim)); te["set"] = "test"
    for p in mfval.data_problems(pd.concat([tr, te])):
        print("  data problem: " + p)
    tr, te = tr[~tr["in_spinup"]], te[~te["in_spinup"]]
    if not mfval.fit_metrics(te)["n"]:
        raise SystemExit("No test observation could be compared with the model.")
    mtr, mte = mfval.fit_metrics(tr), mfval.fit_metrics(te)
    ratio = mte["rmse_m"] / mtr["rmse_m"] if mtr["rmse_m"] > 0 else np.nan
    verdict = ("Good predictive skill: test error close to calibration error." if ratio <= 1.5 else
               "Moderate: test error noticeably higher; check the wells or period that degrade." if ratio <= 2.5 else
               "Weak: the model fits its calibration data much better than new data (possible overfitting or structural error).")
    res = {"run": mfval.run_record(gwf.model_ws), "recalibration": _read_json(Path(gwf.model_ws) / "recalibration.json"),
           "train": mtr, "test": mte, "test_to_train_rmse_ratio": float(ratio), "verdict": verdict}
    (out / "split_result.json").write_text(json.dumps(res, indent=2))
    pd.concat([tr, te]).to_csv(out / "split_residuals.csv", index=False)
    plot_split(tr, te, mtr, mte, out / "split_sim_vs_obs.png")
    plot_split_time(tr, te, out / "split_timeseries.png")
    print(f"Train RMSE {mtr['rmse_m']:.3f} m | Test RMSE {mte['rmse_m']:.3f} m | ratio {ratio:.2f}\n{verdict}")


def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def plot_split(tr, te, mtr, mte, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    allv = pd.concat([tr, te]).dropna(subset=["sim_head"])
    lo, hi = min(allv.obs_head.min(), allv.sim_head.min()), max(allv.obs_head.max(), allv.sim_head.max())
    pad = 0.03 * (hi - lo) if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.scatter(tr.obs_head, tr.sim_head, s=9, color=TRAIN_C, alpha=0.45, lw=0, label=f"Calibration (RMSE {mtr['rmse_m']:.2f} m)")
    ax.scatter(te.obs_head, te.sim_head, s=12, color=TEST_C, alpha=0.85, lw=0, label=f"Held out (RMSE {mte['rmse_m']:.2f} m)")
    ax.set_xlabel("Observed head (m)"); ax.set_ylabel("Simulated head (m)")
    ax.set_title("Calibration vs held-out performance")
    ax.set_aspect("equal"); ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_split_time(tr, te, path, n=4):
    both = pd.concat([tr, te])
    tcol = "date" if "date" in both and both["date"].notna().any() else "time"
    if both.groupby("well_id")[tcol].nunique().max() <= 1:
        return                                     # one measurement per well: nothing to plot over time
    wells = both.groupby("well_id").size().sort_values(ascending=False).index[:n]
    fig, axes = plt.subplots(len(wells), 1, figsize=(6.4, 1.7 * len(wells)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, w in zip(axes, wells):
        g = both[both.well_id == w].sort_values(tcol)
        ax.plot(g[tcol], g.sim_head, color=TRAIN_C, lw=1.2, label="Simulated")
        for s, c in (("train", TRAIN_C), ("test", TEST_C)):
            gg = g[g.set == s]
            ax.scatter(gg[tcol], gg.obs_head, s=10, color=c, zorder=3, label=f"Observed ({'calibration' if s == 'train' else 'held out'})")
        ax.set_ylabel(f"{w}\nhead (m)")
    axes[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    axes[-1].set_xlabel("Date" if tcol == "date" else "Model time (days)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def cmd_summarise(a):
    rows = []
    for d in sorted(glob.glob(a.results)):
        f = Path(d) / "split_result.json"
        if f.exists():
            r = json.loads(f.read_text())
            rows.append({"split": Path(d).name, "train_rmse_m": r["train"]["rmse_m"],
                         "test_rmse_m": r["test"]["rmse_m"], "ratio": r["test_to_train_rmse_ratio"]})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    if len(df):
        print(f"\nMean test/train ratio across {len(df)} splits: {df.ratio.mean():.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split"); s.add_argument("--obs", required=True)
    s.add_argument("--mode", choices=["temporal", "wells"], default="temporal")
    s.add_argument("--train-frac", type=float, default=0.7); s.add_argument("--k", type=int, default=5)
    s.add_argument("--seed", type=int, default=42); s.add_argument("--pst", default=None)
    s.add_argument("--out", default="splits"); s.set_defaults(func=cmd_split)

    r = sub.add_parser("recalibrate"); r.add_argument("--sim-ws", default=None, help="Default: the run_model.py WORKSPACE")
    r.add_argument("--obs", required=True)
    r.add_argument("--model", default=None); r.add_argument("--out", required=True)
    r.add_argument("--k-layers", type=int, nargs="+", default=None, help="Layers whose K gets its own multiplier (default: 1)")
    r.add_argument("--recharge", action="store_true", help="Also fit a recharge multiplier (RCHA package)")
    r.add_argument("--max-runs", type=int, default=60)
    r.add_argument("--step", type=float, default=0.1, help="First search step in log10 units (0.1 = x 1.26)")
    r.add_argument("--max-factor", type=float, default=10.0, help="Multipliers stay within 1/F and F (default 10)")
    r.add_argument("--exe", default=None, help="mf6 executable (default: run_model.py's, else mf6 on PATH)")
    r.set_defaults(func=cmd_recalibrate)

    e = sub.add_parser("evaluate"); e.add_argument("--sim-ws", required=True); e.add_argument("--model", default=None)
    e.add_argument("--train", required=True); e.add_argument("--test", required=True)
    e.add_argument("--out", default="split_results"); e.set_defaults(func=cmd_evaluate)

    m = sub.add_parser("summarise"); m.add_argument("--results", required=True, help="Glob of evaluate output folders")
    m.set_defaults(func=cmd_summarise)

    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
