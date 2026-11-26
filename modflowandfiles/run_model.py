# -*- coding: utf-8 -*-
"""
run_model.py  —  Upper Rio Cobre groundwater model, version v22 (MODFLOW 6)
---------------------------------------------------------------------------
Single-layer transient model of the Upper Rio Cobre aquifer, Jamaica.

  - 200 m x 200 m grid from the DEM and the Upper Rio Cobre SWMU shapefile
  - 1 steady-state spin-up + N_CYCLES years of 12 monthly transient periods
  - Monthly recharge = IDW rainfall x 0.17 (Curtis 2025, Table 4.8), from recharge_model.py
  - Southern constant head (CHD); all other edges no-flow (catchment divide)
  - Calibration against 13 observed heads on the final cycle (RMSE, MAE, R2)

Version history (the model's versions are Git tags; see setup_versions.py)
  v19  MODFLOW-2005, steady state, uniform recharge
  v20  MODFLOW-2005, transient monthly recharge (April 2026 report)
  v21  MODFLOW-2005, end-of-month heads, 3 cycles, no-flow north boundary, observed heads
  v22  THIS FILE. MODFLOW 6 port, with:
         * outputs in runs/<MODEL_NAME>/ next to this script, wherever Python is started
         * run_config.json saved beside every run's outputs (settings, input checksums,
           Git commit, software versions) so any result can be traced to its inputs
         * grid.npz saved beside every run (grid, georeference, final IDOMAIN/CHD cells);
           the export script and 3D viewer read it instead of rebuilding the grid
         * flow from MODFLOW 6's own cell-centred specific discharge (DATA-SPDIS),
           which uses each cell's own K and gives values at basin-edge cells
         * output control saves heads/budgets at the LAST step of each period by name,
           so the (period, step) / (step, period) mix-up of v21 cannot happen
         * Newton-Raphson formulation for robust drying and rewetting of unconfined cells
         * the duplicate flowline tracer removed (export_hydrowindy.py does this)

Run recharge_model.py first to generate recharge_grid_12months.npy.
"""

import os
import sys
import json
import shutil
import hashlib
import datetime
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point

try:
    import flopy
    from flopy.utils import HeadFile, CellBudgetFile
except ModuleNotFoundError as e:
    raise SystemExit("FloPy is not installed. Install with: conda install -c conda-forge flopy") from e

# ==============================================================================
# CONFIGURATION  (every UPPER_CASE value below is copied into run_config.json)
# ==============================================================================

MODEL_VERSION = "v22"
MODEL_NAME    = "riocobre_v22"              # also the run folder name: runs/<MODEL_NAME>/

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SHP_PATH     = os.path.join(BASE_DIR, "Upper_RioCobre_SWMU.shp")
DEM_PATH     = os.path.join(BASE_DIR, "riocobre2.tif")
RECHARGE_NPY = os.path.join(BASE_DIR, "rainfall_outputs", "recharge_grid_12months.npy")
RUNS_DIR     = os.path.join(BASE_DIR, "runs")
WORKSPACE    = os.path.join(RUNS_DIR, MODEL_NAME)   # built from BASE_DIR, not os.getcwd()

# MODFLOW 6 executable: found on PATH ("mf6"), else this path, else env var MF6_EXE.
# Install once with:  get-modflow :flopy    (FloPy's downloader)
EXE_MF6 = os.environ.get("MF6_EXE", r"C:\Users\ameli\Github\projects\ja-modflow\mf6.exe")

WEB_DATA_DIR = None   # e.g. r"C:\Users\ameli\Github\ja-modflow\hydro-windy-app\data"
SHOW_PLOTS   = False  # True opens each figure in a window (pauses the run until closed)

# Grid
DELR = DELC    = 200.0
NLAY           = 1
MIN_THICKNESS  = 10.0
BOTM_ELEVATION = 0.0

# Aquifer properties
HK     = 200.0   # horizontal hydraulic conductivity (m/day); scalar or (nrow, ncol) array
VKA    = 200.0   # vertical hydraulic conductivity (m/day)
SY     = 0.20    # specific yield (also effective porosity for seepage velocity)
SS     = 1e-5    # specific storage (1/m)

# Time: 1 steady-state spin-up + N_CYCLES x 12 monthly periods
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_LABELS   = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
N_CYCLES = 3
NPER     = 1 + 12 * N_CYCLES
PERLEN   = [365.0] + [float(d) for d in DAYS_PER_MONTH] * N_CYCLES
NSTP     = [1] + [30] * (12 * N_CYCLES)
STEADY   = [True] + [False] * (12 * N_CYCLES)

# Boundaries
CH_ELEV_SOUTH = 52.0   # constant head on the southern edge (m)
CH_ELEV_NORTH = None   # None = no-flow (catchment divide); if used, must exceed ~81 m
INITIAL_HEAD  = 62.0

# Solver (IMS). Heads converge to OUTER_DVCLOSE (m); flows to INNER_RCLOSE (m3/day)
OUTER_DVCLOSE = 1e-3
INNER_DVCLOSE = 1e-4
INNER_RCLOSE  = 1.0

CALIB_MONTH    = 12    # month (1-12) of the final cycle used for calibration statistics
FLOWLINE_MONTH = 12    # month used for the streamline figure

# Calibration targets with observed heads (Curtis 2025)
CALIBRATION_TARGETS = [
    {"id": 1,  "x": 748740.4205, "y": 660747.1651, "obs_head": 77.0},
    {"id": 2,  "x": 749538.4976, "y": 661301.9048, "obs_head": 77.3},
    {"id": 3,  "x": 748862.8926, "y": 661688.6217, "obs_head": 77.8},
    {"id": 4,  "x": 746973.2362, "y": 663735.7684, "obs_head": 78.8},
    {"id": 5,  "x": 750878.5184, "y": 661012.7804, "obs_head": 77.4},
    {"id": 6,  "x": 751545.3569, "y": 660868.9354, "obs_head": 77.1},
    {"id": 7,  "x": 751449.8523, "y": 664156.1805, "obs_head": 78.2},
    {"id": 8,  "x": 752455.0110, "y": 665860.7871, "obs_head": 79.1},
    {"id": 9,  "x": 742297.6009, "y": 669060.8732, "obs_head": 81.1},
    {"id": 10, "x": 748370.4468, "y": 666657.6147, "obs_head": 79.4},
    {"id": 11, "x": 752487.1230, "y": 662971.9867, "obs_head": 78.0},
    {"id": 12, "x": 752370.8074, "y": 662119.7216, "obs_head": 77.6},
    {"id": 13, "x": 745848.6474, "y": 661594.7216, "obs_head": 78.0},
]

# ==============================================================================
# TIME HELPERS  (the only code that knows how periods, steps and cycles are numbered)
# ==============================================================================

def final_cycle_sp(month):
    """Stress period index (0-based) of a month (1-12) in the final transient cycle."""
    return 1 + 12 * (N_CYCLES - 1) + (month - 1)


def kstpkper_end(sp):
    """(kstp, kper) of the last time step in stress period sp, as FloPy's readers want it."""
    return (NSTP[sp] - 1, sp)


def get_heads(hds, sp):
    """End-of-period heads for layer 1; inactive / dry cells as NaN."""
    h = hds.get_data(kstpkper=kstpkper_end(sp))[0].astype(float).copy()
    h[np.abs(h) > 1e20] = np.nan
    return h

# ==============================================================================
# GRID
# ==============================================================================

def assign_dem_to_modflow_top(shapefile_path, dem_path, delr, delc):
    """Basin mask + DEM land surface on the model grid. Row 0 is the NORTH edge."""
    print("Loading files and harmonizing CRS...")
    with rasterio.open(dem_path) as src:
        dem_crs, dem_array, dem_nodata = src.crs, src.read(1), src.nodata
        transform, dem_h, dem_w = src.transform, src.height, src.width

    gdf = gpd.read_file(shapefile_path).to_crs(dem_crs)
    minx, miny, maxx, maxy = gdf.total_bounds
    ncol = int((maxx - minx) / delr)
    nrow = int((maxy - miny) / delc)

    xc = minx + delr * (np.arange(ncol) + 0.5)
    yc = maxy - delc * (np.arange(nrow) + 0.5)
    X, Y = np.meshgrid(xc, yc)

    try:
        basin = gdf.union_all()
    except AttributeError:
        basin = gdf.unary_union
    try:
        import shapely
        mask = np.asarray(shapely.contains_xy(basin, X, Y), dtype=bool)
    except AttributeError:
        from shapely.prepared import prep
        pb = prep(basin)
        mask = np.array([[pb.contains(Point(x, y)) for x in xc] for y in yc])

    elevation = np.full((nrow, ncol), np.nan)
    ii, jj = np.nonzero(mask)
    r, c = rowcol(transform, X[ii, jj], Y[ii, jj])
    r, c = np.asarray(r), np.asarray(c)
    inb = (r >= 0) & (r < dem_h) & (c >= 0) & (c < dem_w)
    vals = np.full(len(ii), np.nan)
    vals[inb] = dem_array[r[inb], c[inb]]
    if dem_nodata is not None:
        vals[vals == dem_nodata] = np.nan
    elevation[ii, jj] = vals

    good = elevation[mask & ~np.isnan(elevation)]
    if len(good):
        med = np.median(good)
        print(f"Filled {int(np.sum(np.isnan(elevation) & mask))} NoData cells with median elevation: {med:.2f} m")
        elevation[np.isnan(elevation) & mask] = med
    else:
        elevation[np.isnan(elevation)] = 0.0
        print("WARNING: No valid elevation data found!")

    min_active = np.min(elevation[mask]) if np.any(mask) else 0.0
    top = np.where(mask, elevation, min_active)
    active = np.where(mask, 1, 0)
    print(f"Grid: {nrow} rows x {ncol} cols, {mask.sum()} active cells")
    print(f"TOP: min={np.min(top[mask]):.1f} m, max={np.max(top[mask]):.1f} m")
    return top, active, nrow, ncol, minx, maxy, dem_crs.to_wkt()


def chd_cells(active_2d):
    """(row, col, head) of every constant-head cell. The one place the CHD rule lives."""
    nrow = active_2d.shape[0]
    cells = [(nrow - 1, int(j), CH_ELEV_SOUTH) for j in np.where(active_2d[nrow - 1] == 1)[0]]
    if CH_ELEV_NORTH is not None:
        cells += [(0, int(j), CH_ELEV_NORTH) for j in np.where(active_2d[0] == 1)[0]]
    return cells


def save_grid(path, top, botm, active, nrow, ncol, minx, maxy, crs_wkt, chd):
    """grid.npz: everything downstream code needs to place and read this run's cells."""
    ibound = active.copy()
    for r, c, _ in chd:
        ibound[r, c] = -1
    np.savez_compressed(
        path, model_version=MODEL_VERSION, model_name=MODEL_NAME,
        nrow=nrow, ncol=ncol, delr=DELR, delc=DELC, minx=minx, maxy=maxy, crs_wkt=crs_wkt,
        top=top, botm=np.asarray(botm, float), idomain=active, ibound=ibound,
        chd=np.array(chd, float).reshape(-1, 3))
    print(f"✓ Grid saved: {path}")


def load_grid(workspace=None):
    """Read grid.npz from a run folder (default: this version's). Returns a plain dict."""
    path = os.path.join(workspace or WORKSPACE, "grid.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run run_model.py first.")
    z = np.load(path, allow_pickle=False)
    g = {k: z[k] for k in z.files}
    for k in ("nrow", "ncol"):
        g[k] = int(g[k])
    for k in ("delr", "delc", "minx", "maxy"):
        g[k] = float(g[k])
    for k in ("crs_wkt", "model_version", "model_name"):
        g[k] = str(g[k])
    g["active"] = g["idomain"] != 0
    g["xc"] = g["minx"] + g["delr"] * (np.arange(g["ncol"]) + 0.5)
    g["yc"] = g["maxy"] - g["delc"] * (np.arange(g["nrow"]) + 0.5)
    return g

# ==============================================================================
# RECHARGE
# ==============================================================================

def load_recharge_array(npy_path, nrow, ncol):
    if os.path.exists(npy_path):
        rch = np.load(npy_path)
        print(f"Loaded recharge array: shape={rch.shape}, mean={np.mean(rch[rch > 0]):.6f} m/day")
        if rch.shape != (12, nrow, ncol):
            raise ValueError(f"Recharge array shape {rch.shape} does not match (12, {nrow}, {ncol}). "
                             f"Re-run recharge_model.py.")
        return rch
    print(f"WARNING: {npy_path} not found. Falling back to uniform 1e-4 m/day.")
    return np.full((12, nrow, ncol), 1e-4)

# ==============================================================================
# CALIBRATION
# ==============================================================================

def find_calibration_cells(calibration_targets, minx, maxy, nrow, ncol, active_2d, verbose=True):
    if verbose:
        print("\n" + "=" * 60 + "\nLOCATING CALIBRATION TARGETS\n" + "=" * 60)
    cells = []
    for t in calibration_targets:
        col = int(np.floor((t['x'] - minx) / DELR))
        row = int(np.floor((maxy - t['y']) / DELC))
        if 0 <= row < nrow and 0 <= col < ncol and active_2d[row, col] != 0:
            cells.append((row, col, t['x'], t['y'], t['id'], t['obs_head']))
            if verbose:
                print(f"Target {t['id']:>2}: -> Cell ({row}, {col}), obs={t['obs_head']:.1f} m")
        elif verbose:
            print(f"Target {t['id']:>2}: -> outside grid or inactive")
    if verbose:
        print(f"\n✓ Located {len(cells)} targets in active cells")
    return cells


def calibration_stats(sim, obs):
    """One definition of the statistics, used by the CSV, the plot and the export."""
    sim, obs = np.asarray(sim, float), np.asarray(obs, float)
    ok = np.isfinite(sim)
    res = sim[ok] - obs[ok]
    if not ok.any():
        return {"n": 0, "rmse": None, "mae": None, "me": None, "r2": None, "scaled_rmse": None}
    ss_tot = np.sum((obs[ok] - obs[ok].mean()) ** 2)
    rng = obs[ok].max() - obs[ok].min()
    rmse = float(np.sqrt(np.mean(res ** 2)))
    return {"n": int(ok.sum()), "rmse": rmse, "mae": float(np.mean(np.abs(res))),
            "me": float(np.mean(res)), "max_abs": float(np.max(np.abs(res))),
            "r2": float(1 - np.sum(res ** 2) / ss_tot) if ss_tot > 0 else None,
            "scaled_rmse": float(rmse / rng) if rng > 0 else None}


def extract_calibration_heads(head_2d, cells, top):
    data = []
    for row, col, x, y, tid, obs in cells:
        sim = head_2d[row, col]
        ok = bool(np.isfinite(sim))
        data.append({'target_id': tid, 'row': row, 'col': col, 'x': x, 'y': y,
                     'observed_head': obs, 'simulated_head': float(sim) if ok else np.nan,
                     'residual': float(sim - obs) if ok else np.nan, 'land_surface': float(top[row, col]),
                     'status': 'VALID' if ok else 'DRY/INACTIVE'})
    return data


def export_calibration_data(data, workspace):
    import pandas as pd
    path = os.path.join(workspace, 'calibration_results.csv')
    pd.DataFrame(data).to_csv(path, index=False)
    print(f"✓ Calibration data saved: {path}")
    st = calibration_stats([d['simulated_head'] for d in data], [d['observed_head'] for d in data])
    if st["n"]:
        print(f"\n{'=' * 44}\nCALIBRATION (end of {MONTH_LABELS[CALIB_MONTH - 1]}, cycle {N_CYCLES})\n{'=' * 44}")
        print(f"  Valid targets: {st['n']}/{len(data)}")
        print(f"  RMSE {st['rmse']:.3f} m   MAE {st['mae']:.3f} m   ME {st['me']:+.3f} m (simulated - observed)")
        r2 = f"{st['r2']:.3f}" if st['r2'] is not None else "n/a"
        print(f"  R2 {r2}   scaled RMSE {100 * st['scaled_rmse']:.1f}% of observed range")
    return st

# ==============================================================================
# RUN RECORD
# ==============================================================================

def sha256(path, chunk=1 << 20):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=BASE_DIR, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=BASE_DIR, capture_output=True,
                                    text=True, timeout=10).stdout.strip())
        tag = subprocess.run(["git", "describe", "--tags", "--always"], cwd=BASE_DIR, capture_output=True,
                             text=True, timeout=10).stdout.strip() or None
        if commit is None:                      # not a Git repository (yet)
            return {"commit": None, "describe": None, "uncommitted_changes": None}
        return {"commit": commit, "describe": tag, "uncommitted_changes": dirty}
    except Exception:
        return {"commit": None, "describe": None, "uncommitted_changes": None}


def write_run_config(workspace, exe):
    """run_config.json: the settings, inputs and software that produced this run."""
    cfg = {}
    for k, v in globals().items():
        if k.isupper() and not k.startswith("_"):
            if isinstance(v, np.ndarray):
                v = {"array_shape": list(v.shape), "mean": float(np.nanmean(v))}
            try:
                json.dumps(v); cfg[k] = v
            except TypeError:
                cfg[k] = str(v)
    try:
        mf6v = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        mf6v = None
    record = {
        "model_version": MODEL_VERSION,
        "run_started": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git_state(),
        "software": {"python": sys.version.split()[0], "flopy": flopy.__version__,
                     "numpy": np.__version__, "mf6": mf6v},
        "inputs_sha256": {"dem": sha256(DEM_PATH), "basin_shp": sha256(SHP_PATH),
                          "recharge_npy": sha256(RECHARGE_NPY)},
        "config": cfg,
    }
    path = os.path.join(workspace, "run_config.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"✓ Run record saved: {path}")
    if record["git"]["uncommitted_changes"]:
        print("  NOTE: the working copy has uncommitted changes; commit before runs you will report.")
    return record

# ==============================================================================
# MODFLOW 6 BUILD
# ==============================================================================

def resolve_mf6_path():
    p = shutil.which("mf6")
    if p:
        return p
    if os.path.isfile(EXE_MF6):
        if os.name != "nt":
            os.chmod(EXE_MF6, 0o755)
        return EXE_MF6
    raise FileNotFoundError("MODFLOW 6 executable not found. Install with:  get-modflow :flopy  "
                            "or set MF6_EXE to the path of mf6.exe.")


def build_model(top, botm, active, nrow, ncol, minx, maxy, recharge, chd, exe):
    sim = flopy.mf6.MFSimulation(sim_name=MODEL_NAME, sim_ws=WORKSPACE, exe_name=exe, version="mf6")
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=NPER,
                          perioddata=[(PERLEN[sp], NSTP[sp], 1.0) for sp in range(NPER)])
    flopy.mf6.ModflowIms(sim, complexity="COMPLEX", outer_dvclose=OUTER_DVCLOSE, outer_maximum=200,
                         inner_maximum=300, inner_dvclose=INNER_DVCLOSE, rcloserecord=[INNER_RCLOSE, "strict"],
                         linear_acceleration="BICGSTAB", under_relaxation="DBD")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=MODEL_NAME, save_flows=True,
                               newtonoptions="UNDER_RELAXATION")
    flopy.mf6.ModflowGwfdis(gwf, length_units="METERS", nlay=NLAY, nrow=nrow, ncol=ncol,
                            delr=DELR, delc=DELC, top=top, botm=botm, idomain=active,
                            xorigin=minx, yorigin=maxy - nrow * DELC)
    flopy.mf6.ModflowGwfic(gwf, strt=np.where(active == 1, INITIAL_HEAD, BOTM_ELEVATION))
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=HK, k33=VKA, save_specific_discharge=True)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, sy=SY, ss=SS,
                            steady_state={0: True}, transient={1: True})

    flopy.mf6.ModflowGwfchd(gwf, stress_period_data={0: [((0, r, c), h) for r, c, h in chd]})

    chd_mask = np.zeros((nrow, ncol), bool)
    for r, c, _ in chd:
        chd_mask[r, c] = True
    mean_rch = recharge.mean(axis=0)
    rch = {}
    for sp in range(NPER):
        a = (mean_rch if sp == 0 else recharge[(sp - 1) % 12]).copy()
        a[(active == 0) | chd_mask] = 0.0
        rch[sp] = a
    flopy.mf6.ModflowGwfrcha(gwf, recharge=rch)

    flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f"{MODEL_NAME}.hds", budget_filerecord=f"{MODEL_NAME}.cbc",
                           saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
                           printrecord=[("BUDGET", "LAST")])
    return sim, gwf

# ==============================================================================
# RUN AND READ
# ==============================================================================

def run_and_read(sim):
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True, report=True)
    if not ok:
        lst = os.path.join(WORKSPACE, f"{MODEL_NAME}.lst")
        try:
            with open(lst, errors="ignore") as f:
                print("\nMODFLOW 6 LIST FILE TAIL\n" + "=" * 40 + "\n" + "".join(f.readlines()[-120:]))
        except Exception:
            pass
        raise RuntimeError("MODFLOW 6 failed; see the list file above.")
    return (HeadFile(os.path.join(WORKSPACE, f"{MODEL_NAME}.hds")),
            CellBudgetFile(os.path.join(WORKSPACE, f"{MODEL_NAME}.cbc"), precision="double"))


def check_cycle_convergence(hds):
    if N_CYCLES < 2:
        print("Cycle convergence: only 1 cycle.")
        return None
    sp = final_cycle_sp(12)
    d = np.abs(get_heads(hds, sp) - get_heads(hds, sp - 12))
    print(f"\nCycle convergence (Dec, cycle {N_CYCLES} vs {N_CYCLES - 1}): "
          f"max |Δh| = {np.nanmax(d):.3f} m, mean |Δh| = {np.nanmean(d):.3f} m")
    if np.nanmax(d) > 0.1:
        print("  NOTE: heads still drifting between cycles; consider more cycles.")
    return float(np.nanmax(d))


BUDGET_TERMS = {"RCHA": "RECHARGE", "RCH": "RECHARGE", "CHD": "CONSTANT HEAD",
                "STO-SS": "STORAGE", "STO-SY": "STORAGE"}   # MODFLOW 6 record -> report name
NON_BUDGET = ("FLOW-JA-FACE", "DATA-SPDIS", "DATA-SAT")


def budget_terms(cbc, sp):
    """In and out (m3/day) per budget term for one saved period, MODFLOW 6 names merged."""
    out = {}
    for name in cbc.get_unique_record_names():
        n = name.decode().strip() if isinstance(name, bytes) else str(name).strip()
        if n in NON_BUDGET:
            continue
        term = BUDGET_TERMS.get(n, n)
        for rec in cbc.get_data(kstpkper=kstpkper_end(sp), text=n):
            if getattr(rec, "dtype", None) is not None and rec.dtype.names and "q" in rec.dtype.names:
                q = np.asarray(rec["q"], float)
            else:
                q = np.asarray(rec, float).ravel()
            q = q[np.isfinite(q) & (np.abs(q) < 1e20)]
            i, o = out.get(term, (0.0, 0.0))
            out[term] = (i + q[q > 0].sum(), o - q[q < 0].sum())
    return out


def water_budget_check(cbc, sp):
    """Percent discrepancy for one saved period, from the cell-by-cell budget."""
    terms = budget_terms(cbc, sp)
    tot_in = sum(i for i, _ in terms.values()); tot_out = sum(o for _, o in terms.values())
    disc = 100 * (tot_in - tot_out) / ((tot_in + tot_out) / 2) if (tot_in + tot_out) else 0.0
    return tot_in, tot_out, disc

# ==============================================================================
# FLOW VECTORS  (from MODFLOW 6's own specific discharge)
# ==============================================================================

def flow_vectors(cbc, sp, nrow, ncol):
    """Cell-centred specific discharge (Darcy flux, m/day) for layer 1 at the end of period sp.
    qx is east-positive, qy NORTH-positive. Inactive cells are NaN.
    Read straight from MODFLOW 6's DATA-SPDIS record, which MODFLOW computes from each cell's
    own conductances: correct with zoned K, and defined at every active cell including the
    basin edge. Record nodes are full-grid (user) numbers, so no model object is needed;
    the export script and the 3D viewer call this same function."""
    rec = cbc.get_data(text="DATA-SPDIS", kstpkper=kstpkper_end(sp))[0]
    qx = np.full(nrow * ncol, np.nan); qy = np.full(nrow * ncol, np.nan)
    idx = np.asarray(rec["node"], int) - 1
    qx[idx] = rec["qx"]; qy[idx] = rec["qy"]
    qx = qx.reshape(nrow, ncol); qy = qy.reshape(nrow, ncol)
    return qx, qy, np.hypot(qx, qy)

# ==============================================================================
# PLOTTING
# ==============================================================================

def _finish(fig, path):
    fig.savefig(path, dpi=250, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"✓ Saved: {path}")


def plot_seasonal_heads(hds, cells, extent):
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    xs = [c[2] for c in cells]; ys = [c[3] for c in cells]
    for m_, ax in zip(range(1, 13), axes.flatten()):
        im = ax.imshow(get_heads(hds, final_cycle_sp(m_)), cmap="viridis", origin="upper", extent=extent)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Head (m)")
        ax.scatter(xs, ys, c="red", s=25, edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_title(f"{MONTH_LABELS[m_ - 1]} (end of month)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Simulated heads, monthly, cycle {N_CYCLES} ({MODEL_VERSION}, MODFLOW 6)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _finish(fig, os.path.join(WORKSPACE, "seasonal_head_maps.png"))


def plot_head_and_flow(hds, cbc, cells, nrow, ncol, minx, maxy, extent, month):
    label = MONTH_LABELS[month - 1]
    sp = final_cycle_sp(month)
    h2d = get_heads(hds, sp)
    qx, qy, qmag = flow_vectors(cbc, sp, nrow, ncol)
    # streamplot wants y increasing upwards: flip rows (row 0 is north); qy is already north-positive
    x = minx + DELR * (np.arange(ncol) + 0.5)
    y = (maxy - nrow * DELC) + DELC * (np.arange(nrow) + 0.5)
    X, Y = np.meshgrid(x, y)
    U = np.flipud(np.nan_to_num(qx)); V = np.flipud(np.nan_to_num(qy))
    lw = np.clip(np.flipud(np.nan_to_num(2.0 * qmag / np.nanmax(qmag), nan=0.5)), 0.5, 2.5)
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(h2d, cmap="viridis", origin="upper", extent=extent, alpha=0.5)
    plt.colorbar(im, ax=ax, label="Hydraulic head (m)", shrink=0.8)
    ax.streamplot(X, Y, U, V, density=1.5, color="#0b8fb3", linewidth=lw, arrowsize=1.5, zorder=2)
    ax.text(0.02, 0.98, f"Darcy flux max {np.nanmax(qmag):.4f} m/day\nmean {np.nanmean(qmag):.4f} m/day",
            transform=ax.transAxes, va="top", fontsize=10, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    for _, _, x0, y0, tid, _ in cells:
        ax.scatter(x0, y0, c="red", s=120, marker="*", edgecolors="yellow", linewidth=1.5, zorder=3)
        ax.annotate(str(tid), (x0, y0), xytext=(6, 6), textcoords="offset points", color="yellow",
                    fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="red", alpha=0.7))
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_title(f"{label} (end of month, cycle {N_CYCLES}): head and groundwater flow "
                 f"(MODFLOW 6 specific discharge)", fontsize=12)
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    _finish(fig, os.path.join(WORKSPACE, f"{label.lower()}_head_flow_streamlines.png"))


def plot_head_hydrograph(hds, cells):
    sps = list(range(1, NPER))
    H = np.array([get_heads(hds, sp) for sp in sps])
    t = np.arange(1, len(sps) + 1)
    fig, ax = plt.subplots(figsize=(14, 6))
    for row, col, x, y, tid, obs in cells:
        ax.plot(t, H[:, row, col], marker="o", markersize=3, label=f"Target {tid}")
    for k in range(1, N_CYCLES):
        ax.axvline(12 * k + 0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_xticks(t); ax.set_xticklabels([MONTH_LABELS[(i - 1) % 12][0] for i in t], fontsize=8)
    ax.set_title(f"Head at calibration targets, {N_CYCLES} cycles (dotted = cycle boundary)")
    ax.set_ylabel("Simulated head (m)"); ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.4); plt.tight_layout()
    _finish(fig, os.path.join(WORKSPACE, "calibration_head_hydrographs.png"))


def plot_calibration_scatter(data, stats):
    valid = [d for d in data if d["status"] == "VALID"]
    if not valid:
        return
    obs = np.array([d["observed_head"] for d in valid]); sim = np.array([d["simulated_head"] for d in valid])
    ids = [d["target_id"] for d in valid]; res = sim - obs
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax.scatter(obs, sim, c="steelblue", s=80, edgecolors="black", linewidths=0.5, zorder=3)
    for i, tid in enumerate(ids):
        ax.annotate(str(tid), (obs[i], sim[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    lims = [min(obs.min(), sim.min()) - 1, max(obs.max(), sim.max()) + 1]
    ax.plot(lims, lims, "r--", label="1:1"); ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed head (m)"); ax.set_ylabel("Simulated head (m)"); ax.legend(); ax.grid(alpha=0.4)
    r2 = f"{stats['r2']:.3f}" if stats["r2"] is not None else "n/a"
    ax.text(0.05, 0.95, f"RMSE = {stats['rmse']:.2f} m\nME (sim − obs) = {stats['me']:+.2f} m\nR² = {r2}\nn = {stats['n']}",
            transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_title(f"Simulated vs observed, end of {MONTH_LABELS[CALIB_MONTH - 1]}")
    ax2.bar([str(i) for i in ids], res, color=["tomato" if r > 0 else "steelblue" for r in res],
            edgecolor="black", linewidth=0.5)
    ax2.axhline(0, color="black"); ax2.set_xlabel("Target"); ax2.set_ylabel("Residual, sim − obs (m)")
    ax2.set_title("Residuals"); ax2.grid(alpha=0.4, axis="y"); plt.tight_layout()
    _finish(fig, os.path.join(WORKSPACE, "calibration_scatter.png"))

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"\n{'=' * 60}\nUpper Rio Cobre model {MODEL_VERSION} (MODFLOW 6)\n{'=' * 60}\n")
    os.makedirs(WORKSPACE, exist_ok=True)
    exe = resolve_mf6_path()
    write_run_config(WORKSPACE, exe)

    top, active, nrow, ncol, minx, maxy, crs_wkt = assign_dem_to_modflow_top(SHP_PATH, DEM_PATH, DELR, DELC)
    extent = [minx, minx + ncol * DELR, maxy - nrow * DELC, maxy]
    min_top = np.min(top[active == 1])
    botm = [min_top - MIN_THICKNESS] if BOTM_ELEVATION >= min_top - MIN_THICKNESS else [BOTM_ELEVATION]
    print(f"Layer bottom: {botm[0]:.1f} m")

    recharge = load_recharge_array(RECHARGE_NPY, nrow, ncol)
    cells = find_calibration_cells(CALIBRATION_TARGETS, minx, maxy, nrow, ncol, active)
    chd = chd_cells(active)
    print(f"CHD: {len(chd)} cells" + ("" if CH_ELEV_NORTH else "; north boundary no-flow"))
    save_grid(os.path.join(WORKSPACE, "grid.npz"), top, botm, active, nrow, ncol, minx, maxy, crs_wkt, chd)

    print("\nBuilding and running MODFLOW 6...")
    sim, gwf = build_model(top, botm, active, nrow, ncol, minx, maxy, recharge, chd, exe)
    hds, cbc = run_and_read(sim)
    print("✓ MODFLOW 6 completed")

    conv = check_cycle_convergence(hds)
    worst = max((water_budget_check(cbc, sp)[2] for sp in range(NPER)), key=abs)
    print(f"Water budget: worst period discrepancy {worst:+.4f}%")

    h_cal = get_heads(hds, final_cycle_sp(CALIB_MONTH))
    data = extract_calibration_heads(h_cal, cells, top)
    stats = export_calibration_data(data, WORKSPACE)

    plot_seasonal_heads(hds, cells, extent)
    plot_head_and_flow(hds, cbc, cells, nrow, ncol, minx, maxy, extent, FLOWLINE_MONTH)
    plot_head_hydrograph(hds, cells)
    plot_calibration_scatter(data, stats)

    # append results to the run record
    rec_path = os.path.join(WORKSPACE, "run_config.json")
    rec = json.load(open(rec_path))
    rec["results"] = {"run_finished": datetime.datetime.now().isoformat(timespec="seconds"),
                      "calibration": stats, "cycle_convergence_max_m": conv,
                      "worst_budget_discrepancy_pct": worst}
    json.dump(rec, open(rec_path, "w"), indent=2)

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"Grid {nrow} x {ncol}, {int(active.sum())} active cells, {NPER} periods "
          f"(1 spin-up + {N_CYCLES} x 12)")
    print(f"K {HK} m/d, Sy {SY}, CHD south {CH_ELEV_SOUTH} m, north "
          f"{'no-flow' if CH_ELEV_NORTH is None else CH_ELEV_NORTH}")
    print(f"Results: {WORKSPACE}\n✓ MODEL COMPLETE")


if __name__ == "__main__":
    main()
