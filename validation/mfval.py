"""
mfval.py -- shared helpers for validating a MODFLOW 6 model built with FloPy.

Works with any MODFLOW 6 model, and knows the Upper Rio Cobre model (run_model.py, v22+):
when run_model.py is found (../modflowandfiles, this folder, or RIOCOBRE_MODEL_DIR), the
simulation folder, the mf6 executable and the calendar default to that model's own.

Observation CSV (one row per measurement). Location: either

    layer, row, col          1-based model cell
    x, y [, layer]           model coordinates (JAD2001 metres for Rio Cobre); layer defaults to 1

and time, one of:

    time       model time (days from the start of the simulation)
    date       calendar date (YYYY-MM-DD). The Rio Cobre model simulates a typical year that
               repeats N_CYCLES times; a date is compared with the same day of the year in the
               FINAL cycle, so measurements from different years land on the same model year.
    month      1-12: compared with the END of that month in the final cycle (as run_model.py does)

plus well_id and obs_head (m). An optional `obsnme` column carries PEST observation names.

Sign convention: residual = observed - simulated (Anderson, Woessner & Hunt 2015).
Positive mean error: the model is too LOW. run_model.py prints simulated - observed, the same
numbers with the opposite sign.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import flopy

DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


# --------------------------------------------------------------------------- #
# The Rio Cobre model, if present
# --------------------------------------------------------------------------- #
def find_run_model():
    """Import run_model.py if it can be found; return the module or None."""
    here = Path(__file__).resolve().parent
    candidates = [os.environ.get("RIOCOBRE_MODEL_DIR"), here.parent / "modflowandfiles", here, Path.cwd()]
    for c in candidates:
        if c and (Path(c) / "run_model.py").is_file():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            try:
                import run_model
                return run_model
            except Exception as e:          # a broken import should not stop a generic validation
                print(f"  (found run_model.py in {c} but could not import it: {e})")
                return None
    return None


def default_sim_ws():
    m = find_run_model()
    return getattr(m, "WORKSPACE", None) if m else None


def default_exe():
    m = find_run_model()
    if m and hasattr(m, "resolve_mf6_path"):
        try:
            return m.resolve_mf6_path()
        except Exception:
            pass
    return "mf6"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_model(sim_ws=None, model_name=None, exe=None):
    """Load a MODFLOW 6 simulation and return (sim, gwf). `exe` is used for re-runs."""
    sim_ws = sim_ws or default_sim_ws()
    if sim_ws is None:
        raise ValueError("Pass --sim-ws (run_model.py was not found to supply a default).")
    if not (Path(sim_ws) / "mfsim.nam").exists():
        raise FileNotFoundError(f"No mfsim.nam in {sim_ws}. Run the model first (python run_model.py).")
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(sim_ws), exe_name=exe or default_exe(), verbosity_level=0)
    names = sim.model_names
    if model_name is None:
        if len(names) != 1:
            raise ValueError(f"Several models in simulation, pass --model. Found: {names}")
        model_name = names[0]
    return sim, sim.get_model(model_name)


def run_record(sim_ws):
    """model_version and Git commit from run_config.json (written by run_model.py v22+), if present."""
    p = Path(sim_ws) / "run_config.json"
    if not p.exists():
        return {}
    try:
        r = json.loads(p.read_text())
    except Exception:
        return {}
    git = r.get("git") or {}
    return {"model_version": r.get("model_version"), "run_started": r.get("run_started"),
            "git_commit": git.get("commit"), "git_uncommitted_changes": git.get("uncommitted_changes"),
            "mf6": (r.get("software") or {}).get("mf6")}


# --------------------------------------------------------------------------- #
# Calendar: dates and months -> model time
# --------------------------------------------------------------------------- #
def model_calendar(sim):
    """Period start/end times from TDIS, and where the final 12-month cycle starts (or None)."""
    pd_ = sim.tdis.perioddata.array
    perlen = np.array([float(p[0]) for p in pd_])
    ends = np.cumsum(perlen)
    cal = {"perlen": perlen, "period_end": ends, "total": float(ends[-1]), "final_cycle_start": None}
    if len(perlen) >= 12 and np.allclose(perlen[-12:], DAYS_PER_MONTH):
        cal["final_cycle_start"] = float(ends[-13]) if len(perlen) > 12 else 0.0
    return cal


def _month_start_day(m):
    return float(sum(DAYS_PER_MONTH[:m - 1]))


def date_to_model_time(dates, cal):
    """Same day of the year in the final cycle. Day d of a month is the end of day d."""
    if cal["final_cycle_start"] is None:
        raise ValueError("A `date` or `month` column needs a model whose last 12 periods are the months "
                         "Jan-Dec. Give model `time` instead.")
    d = pd.to_datetime(dates)
    day = np.minimum(d.dt.day.to_numpy(), [DAYS_PER_MONTH[m - 1] for m in d.dt.month])   # 29 Feb -> 28 Feb
    frac = (d.dt.hour.to_numpy() + d.dt.minute.to_numpy() / 60) / 24
    return cal["final_cycle_start"] + np.array([_month_start_day(m) for m in d.dt.month]) + day - 1 + frac \
        + np.where(frac == 0, 1.0, 0.0)            # a bare date means end of that day


def month_to_model_time(months, cal):
    if cal["final_cycle_start"] is None:
        raise ValueError("A `month` column needs a model whose last 12 periods are the months Jan-Dec.")
    months = np.asarray(months, int)
    if ((months < 1) | (months > 12)).any():
        raise ValueError("`month` must be 1-12.")
    return cal["final_cycle_start"] + np.array([_month_start_day(m) + DAYS_PER_MONTH[m - 1] for m in months])


def load_obs(csv_path, gwf=None, sim=None):
    """Read observations; give each a 0-based (k, i, j) and a model `time`.

    Adds columns: k, i, j, time (if given as date/month), month (when known), location_ok.
    """
    df = pd.read_csv(csv_path)
    need = {"well_id", "obs_head"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Observation file is missing columns: {sorted(missing)}")
    df["well_id"] = df["well_id"].astype(str)

    # ---- time
    if "time" not in df.columns:
        if sim is None:
            raise ValueError("date/month observations need the simulation; pass sim.")
        cal = model_calendar(sim)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["time"] = date_to_model_time(df["date"], cal)
            df["month"] = df["date"].dt.month
        elif "month" in df.columns:
            df["time"] = month_to_model_time(df["month"], cal)
        else:
            raise ValueError("Give a time, date or month column.")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # ---- location
    if {"layer", "row", "col"} <= set(df.columns):
        df["k"] = df["layer"].astype(int) - 1
        df["i"] = df["row"].astype(int) - 1
        df["j"] = df["col"].astype(int) - 1
    elif {"x", "y"} <= set(df.columns):
        if gwf is None:
            raise ValueError("x/y observations need the model grid; pass gwf.")
        ii, jj = [], []
        for x, y in zip(df["x"], df["y"]):
            try:
                i, j = gwf.modelgrid.intersect(float(x), float(y))
            except Exception:
                i, j = np.nan, np.nan               # outside the grid
            ii.append(i); jj.append(j)
        df["i"], df["j"] = ii, jj
        df["k"] = (df["layer"].astype(int) - 1) if "layer" in df.columns else 0
    else:
        raise ValueError("Give either layer,row,col or x,y columns.")

    df["location_ok"] = df[["i", "j"]].notna().all(axis=1)
    if gwf is not None:
        idom = _idomain(gwf)
        ok = df["location_ok"].to_numpy().copy()
        for n in np.where(ok)[0]:
            k, i, j = int(df.k.iat[n]), int(df.i.iat[n]), int(df.j.iat[n])
            if not (0 <= k < idom.shape[0] and 0 <= i < idom.shape[1] and 0 <= j < idom.shape[2]) or idom[k, i, j] < 1:
                ok[n] = False
        df["location_ok"] = ok
    return df


def _idomain(gwf):
    dis = gwf.dis
    shape = (dis.nlay.get_data(), dis.nrow.get_data(), dis.ncol.get_data())
    idom = dis.idomain.array
    return np.ones(shape, int) if idom is None else np.asarray(idom, int).reshape(shape)


# --------------------------------------------------------------------------- #
# Simulated values at observations
# --------------------------------------------------------------------------- #
def simulated_at_obs(gwf, obs):
    """Return a copy of `obs` with sim_head, residual (obs - sim) and quality flags.

    Heads are interpolated linearly in time between saved head times (month ends for Rio Cobre).
    Flags: outside_grid_or_inactive, dry (head at or below the cell bottom: with Newton, MODFLOW 6
    keeps a head in dry cells instead of HDRY), outside_times, in_spinup (before the final cycle).
    """
    hds = gwf.output.head()
    times = np.array(hds.get_times())
    botm = np.asarray(gwf.dis.botm.array, float)
    botm = botm.reshape((-1,) + botm.shape[-2:]) if botm.ndim >= 2 else botm
    out = obs.copy()
    sim_vals = np.full(len(out), np.nan)
    dry = np.zeros(len(out), bool)

    good = out[out["location_ok"]] if "location_ok" in out else out
    for (k, i, j), idx in good.groupby(["k", "i", "j"]).groups.items():
        k, i, j = int(k), int(i), int(j)
        series = hds.get_ts((k, i, j))                    # columns: time, head
        t, h = series[:, 0], series[:, 1].astype(float)
        h = np.where(np.abs(h) > 1e20, np.nan, h)         # inactive / HDRY
        pos = out.index.get_indexer(idx)
        sim_vals[pos] = np.interp(out.loc[idx, "time"], t, h)
        dry[pos] = sim_vals[pos] <= botm[k, i, j] + 1e-6

    sim_vals[dry] = np.nan
    out["sim_head"] = sim_vals
    out["residual"] = out["obs_head"] - out["sim_head"]
    out["outside_grid_or_inactive"] = ~out["location_ok"] if "location_ok" in out else False
    out["dry"] = dry
    out["outside_times"] = (out["time"] < times.min()) | (out["time"] > times.max())
    try:
        cal = model_calendar(gwf.simulation)
        fcs = cal["final_cycle_start"]
        out["in_spinup"] = (out["time"] <= fcs) if fcs is not None else False
    except Exception:
        out["in_spinup"] = False
    return out


def data_problems(df):
    """Plain-language list of observations that could not be compared, and why."""
    msgs = []
    for col, text in (("outside_grid_or_inactive", "outside the active model area"),
                      ("dry", "in cells the model has dry"),
                      ("outside_times", "outside the simulated time range (nearest saved head used)"),
                      ("in_spinup", "in the spin-up cycles, before the final cycle")):
        if col in df and df[col].any():
            w = sorted(df.loc[df[col], "well_id"].unique())
            msgs.append(f"{int(df[col].sum())} observation(s) at {len(w)} well(s) {text}: "
                        f"{', '.join(w[:10])}{'…' if len(w) > 10 else ''}")
    return msgs


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def fit_metrics(df):
    """Standard head-calibration statistics (residual = observed - simulated)."""
    d = df.dropna(subset=["sim_head", "obs_head"])
    r = d["residual"].to_numpy()
    o = d["obs_head"].to_numpy()
    s = d["sim_head"].to_numpy()
    if len(r) == 0:
        return {"n": 0, "mean_error_m": np.nan, "mae_m": np.nan, "rmse_m": np.nan,
                "nrmse_pct": np.nan, "r2": np.nan, "obs_range_m": np.nan}
    rng = o.max() - o.min()
    ss_res = np.sum((o - s) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    return {
        "n": int(len(r)),
        "mean_error_m": float(r.mean()),            # bias: + means model too low
        "mae_m": float(np.abs(r).mean()),
        "rmse_m": float(np.sqrt((r ** 2).mean())),
        "nrmse_pct": float(100 * np.sqrt((r ** 2).mean()) / rng) if rng > 0 else np.nan,
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "obs_range_m": float(rng),
    }


def per_well_metrics(df):
    rows = []
    for w, g in df.groupby("well_id"):
        m = fit_metrics(g)
        m["well_id"] = w
        gg = g.dropna(subset=["residual"])
        # Trend in residuals: per year of calendar date if dates are given, else per model time unit.
        if "date" in gg and gg["date"].notna().sum() > 2 and gg["date"].dt.year.nunique() > 1:
            yrs = (gg["date"] - gg["date"].min()).dt.days / 365.25
            m["residual_trend_m_per_year"] = float(np.polyfit(yrs, gg["residual"], 1)[0])
        elif len(gg) > 2 and gg["time"].nunique() > 2:
            m["residual_trend_m_per_time"] = float(np.polyfit(gg["time"], gg["residual"], 1)[0])
        rows.append(m)
    return pd.DataFrame(rows).set_index("well_id").sort_values("rmse_m", ascending=False)


def monthly_bias(df):
    """Mean residual per calendar month: a seasonal pattern points at recharge timing or Sy."""
    if "month" not in df:
        return None
    d = df.dropna(subset=["residual"])
    if d["month"].nunique() < 3:
        return None
    return d.groupby("month")["residual"].agg(["mean", "count"])


# --------------------------------------------------------------------------- #
# Water budget
# --------------------------------------------------------------------------- #
def budget_check(gwf):
    """Per-period and cumulative budget discrepancy from the MODFLOW 6 list file."""
    lst = Path(gwf.model_ws) / f"{gwf.name}.lst"
    if not lst.exists():
        return {"error": f"list file not found: {lst}"}
    mlb = flopy.utils.Mf6ListBudget(str(lst))
    inc, cum = mlb.get_dataframes(diff=False)
    pct_col = [c for c in inc.columns if "PERCENT" in c.upper()]
    if not pct_col:
        return {"error": "no PERCENT_DISCREPANCY column in list file"}
    pct = inc[pct_col[0]].astype(float)
    flows_in = {c[:-3]: float(cum[c].iloc[-1]) for c in cum.columns if c.endswith("_IN") and c != "TOTAL_IN"}
    flows_out = {c[:-4]: float(cum[c].iloc[-1]) for c in cum.columns if c.endswith("_OUT") and c != "TOTAL_OUT"}
    rate_in = {c[:-3]: float(inc[c].iloc[-1]) for c in inc.columns if c.endswith("_IN") and c != "TOTAL_IN"}
    rate_out = {c[:-4]: float(inc[c].iloc[-1]) for c in inc.columns if c.endswith("_OUT") and c != "TOTAL_OUT"}
    return {
        "max_abs_period_discrepancy_pct": float(pct.abs().max()),
        "final_cumulative_discrepancy_pct": float(cum[[c for c in cum.columns if "PERCENT" in c.upper()][0]].iloc[-1]),
        "final_period_rates_m3_per_day": {"in": rate_in, "out": rate_out},
        "cumulative_inflows_m3": flows_in,
        "cumulative_outflows_m3": flows_out,
    }


# --------------------------------------------------------------------------- #
# Parameter plausibility
# --------------------------------------------------------------------------- #
def parameter_summary(gwf):
    """Min/median/max of key aquifer parameters per layer, over ACTIVE cells only."""
    out = {}
    idom = _idomain(gwf)
    nlay = idom.shape[0]

    def per_layer(arr):
        a = np.broadcast_to(np.asarray(arr, float).reshape((nlay, -1)) if np.size(arr) > 1
                            else np.full(idom.reshape(nlay, -1).shape, float(np.asarray(arr).ravel()[0])),
                            idom.reshape(nlay, -1).shape)
        act = idom.reshape(nlay, -1) > 0
        return [_stats(a[l][act[l]]) for l in range(nlay)]

    npf = gwf.get_package("npf")
    if npf is not None and npf.k.array is not None:
        out["k"] = per_layer(npf.k.array)
    sto = gwf.get_package("sto")
    if sto is not None:
        for name in ("ss", "sy"):
            arr = getattr(sto, name).array
            if arr is not None:
                out[name] = per_layer(arr)
    return out


def _stats(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {"min": np.nan, "median": np.nan, "max": np.nan}
    return {"min": float(np.nanmin(a)), "median": float(np.nanmedian(a)), "max": float(np.nanmax(a))}


def check_ranges(params, ranges_csv):
    """Compare parameters with plausible ranges from a CSV: param,layer,min,max[,source]."""
    ranges = pd.read_csv(ranges_csv, comment="#")
    rows = []
    for _, r in ranges.iterrows():
        name, lay = str(r["param"]).lower(), int(r["layer"]) - 1
        if name not in params or lay >= len(params[name]):
            continue
        s = params[name][lay]
        ok = (s["min"] >= r["min"]) and (s["max"] <= r["max"])
        rows.append({"param": name, "layer": lay + 1, "model_min": s["min"], "model_max": s["max"],
                     "plausible_min": r["min"], "plausible_max": r["max"],
                     "source": r.get("source", ""), "status": "OK" if ok else "OUTSIDE RANGE"})
    return pd.DataFrame(rows)
