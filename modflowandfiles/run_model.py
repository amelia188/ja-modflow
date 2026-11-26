# -*- coding: utf-8 -*-
"""
run_modflow_transient_recharge_v21.py
-------------------------------------
MODFLOW Single-Layer Transient Model — Upper Rio Cobre Aquifer, Jamaica

What this model does:
  - Builds a 200m x 200m finite difference grid from DEM and watershed shapefile
  - Runs 1 steady-state spin-up + N_CYCLES years of 12 transient monthly periods
  - Applies spatially distributed monthly recharge (IDW-interpolated rainfall x 0.17)
    derived from 26 years of observed rainfall data (2000-2025)
  - Recharge coefficient 0.17 from Curtis (2025) SWAT water balance calibration
  - Calibrates against 13 observed heads using the FINAL cycle (end of month)
  - Produces calibration statistics (RMSE, MAE, R2) and diagnostic plots
  - Traces groundwater flowlines and exports them as GeoJSON (WGS84) for the
    HydroWindy web visualizer

Run recharge_model.py first to generate recharge_grid_12months.npy.

Changes from run_modflow_transient_recharge.py (v20 lineage):
  1. Output control saves the LAST time step of each stress period, so
     "December" heads are end-of-December (previously day 1 of December).
  2. Transient year repeated N_CYCLES times; calibration and plots use the
     final cycle so results no longer depend on the steady-state spin-up.
     A convergence check compares December heads between the last two cycles.
  3. Northern boundary is no-flow by default (catchment divide). The previous
     northern CHD (CH_ELEV + 20 = 72 m) sat below every observed head and
     drove flow north. Set CH_ELEV_NORTH to a value to restore a CHD.
  4. Grid cell centres placed at exact DELR/DELC spacing from (minx, maxy),
     matching how calibration targets and flowlines are georeferenced
     (previously np.linspace gave slightly non-uniform spacing).
  5. Streamline plot no longer mirrored north-south.
  6. Dry/inactive heads (+/-1e30) masked consistently.
  7. New: flowline tracing (RK2) + GeoJSON export.
  8. Faster vectorised basin masking and DEM sampling.
"""

import os
import json
import shutil
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
    raise SystemExit(
        "FloPy is not installed. Install with: conda install -c conda-forge flopy"
    ) from e

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SHP_PATH     = os.path.join(BASE_DIR, "Upper_RioCobre_SWMU.shp")
DEM_PATH     = os.path.join(BASE_DIR, "riocobre2.tif")
RECHARGE_NPY = os.path.join(BASE_DIR, "rainfall_outputs", "recharge_grid_12months.npy")
EXE_MF2005   = r"C:\Users\ameli\Github\projects\ja-modflow\mf2005.exe"

MODEL_NAME = "mf2005_riocobre_v21"
WORKSPACE  = os.path.join(os.getcwd(), "mf2005_riocobre_v21")
os.makedirs(WORKSPACE, exist_ok=True)

# Optional: folder of the HydroWindy web project's data directory.
# If set, flowlines.json is copied there automatically.
WEB_DATA_DIR = None   # e.g. r"C:\Users\ameli\Github\projects\hydrowindy\data"

# Grid parameters
DELR = DELC    = 200.0
NLAY           = 1
MIN_THICKNESS  = 10.0
BOTM_ELEVATION = 0.0

# Aquifer properties
HK     = [200.0]  # Horizontal hydraulic conductivity (m/day) — re-check after boundary fix
VKA    = [200.0]  # Vertical hydraulic conductivity (m/day)
SY     = [0.20]   # Specific yield (also used as effective porosity for seepage velocity)
SS     = 1e-5     # Specific storage
LAYTYP = [1]      # Convertible (unconfined)

# Stress periods: 1 steady-state spin-up + N_CYCLES x 12 transient monthly
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_LABELS   = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
N_CYCLES = 3
NPER     = 1 + 12 * N_CYCLES
PERLEN   = [365] + DAYS_PER_MONTH * N_CYCLES
NSTP     = [1] + [30] * (12 * N_CYCLES)
STEADY   = [True] + [False] * (12 * N_CYCLES)

# Boundaries
CBC_UNIT      = 53
CH_ELEV_SOUTH = 52.0   # Constant head on southern edge (m)
CH_ELEV_NORTH = None   # None = no-flow (catchment divide). If used, must exceed ~81 m (max obs head)
INITIAL_HEAD  = 62.0

# Which month (1-12) of the final cycle to calibrate against / export flowlines for
CALIB_MONTH    = 12
FLOWLINE_MONTH = 12

# Flowline tracing
FLOWLINE_SEED_SPACING = 4      # seed every Nth cell
FLOWLINE_STEP         = 50.0   # tracer step length (m)
FLOWLINE_MAX_STEPS    = 3000
FLOWLINE_MIN_V        = 1e-6   # stop below this seepage velocity (m/day)

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
# STRESS PERIOD / HEAD HELPERS
# ==============================================================================

def final_cycle_sp(month):
    """Stress period index for a month (1-12) in the final transient cycle."""
    return 1 + 12 * (N_CYCLES - 1) + (month - 1)


def kstpkper_end(sp):
    """(kstp, kper) of the last time step in stress period sp."""
    return (NSTP[sp] - 1, sp)


def get_heads(hds, sp):
    """End-of-period heads for layer 1 with inactive/dry cells set to NaN."""
    h = hds.get_data(kstpkper=kstpkper_end(sp))[0].astype(float).copy()
    h[np.abs(h) > 1e20] = np.nan
    return h

# ==============================================================================
# GRID GENERATION
# ==============================================================================

def assign_dem_to_modflow_top(shapefile_path, dem_path, delr, delc):
    print("Loading files and harmonizing CRS...")

    with rasterio.open(dem_path) as src:
        dem_crs    = src.crs
        dem_array  = src.read(1)
        dem_nodata = src.nodata
        transform  = src.transform
        dem_h, dem_w = src.height, src.width

    gdf = gpd.read_file(shapefile_path).to_crs(dem_crs)
    minx, miny, maxx, maxy = gdf.total_bounds

    ncol = int((maxx - minx) / delr)
    nrow = int((maxy - miny) / delc)

    # Exact cell centres (row 0 = north), consistent with target/flowline georeferencing
    xc = minx + delr * (np.arange(ncol) + 0.5)
    yc = maxy - delc * (np.arange(nrow) + 0.5)
    X, Y = np.meshgrid(xc, yc)

    # Basin mask (vectorised)
    try:
        basin = gdf.union_all()
    except AttributeError:          # geopandas < 1.0
        basin = gdf.unary_union
    try:
        import shapely
        mask = np.asarray(shapely.contains_xy(basin, X, Y), dtype=bool)
    except AttributeError:          # shapely < 2.0
        from shapely.prepared import prep
        pb = prep(basin)
        mask = np.array([[pb.contains(Point(x, y)) for x in xc] for y in yc])

    # DEM sampling (vectorised)
    elevation = np.full((nrow, ncol), np.nan)
    ii, jj = np.nonzero(mask)
    r, c = rowcol(transform, X[ii, jj], Y[ii, jj])
    r, c = np.asarray(r), np.asarray(c)
    inb  = (r >= 0) & (r < dem_h) & (c >= 0) & (c < dem_w)
    vals = np.full(len(ii), np.nan)
    vals[inb] = dem_array[r[inb], c[inb]]
    if dem_nodata is not None:
        vals[vals == dem_nodata] = np.nan
    elevation[ii, jj] = vals

    active_elevations = elevation[mask & ~np.isnan(elevation)]
    if len(active_elevations) > 0:
        median_elev = np.median(active_elevations)
        fill_count  = np.sum(np.isnan(elevation) & mask)
        elevation[np.isnan(elevation) & mask] = median_elev
        print(f"Filled {fill_count} NoData cells with median elevation: {median_elev:.2f} m")
    else:
        elevation[np.isnan(elevation)] = 0.0
        print("WARNING: No valid elevation data found!")

    min_active_elev = np.min(elevation[mask]) if np.any(mask) else 0.0
    TOP_array       = np.where(mask, elevation, min_active_elev)
    ibound_mask_2d  = np.where(mask, 1, 0)

    print(f"Grid: {nrow} rows x {ncol} cols, {mask.sum()} active cells")
    print(f"TOP: min={np.min(TOP_array[mask]):.1f} m, "
          f"max={np.max(TOP_array[mask]):.1f} m")

    return TOP_array, ibound_mask_2d, nrow, ncol, minx, maxy, dem_crs.to_wkt()


# ==============================================================================
# RECHARGE LOADING
# ==============================================================================

def load_recharge_array(npy_path, nrow, ncol):
    """Load monthly recharge array from recharge_model.py output."""
    if os.path.exists(npy_path):
        rch = np.load(npy_path)
        print(f"Loaded recharge array: shape={rch.shape}, "
              f"mean={np.mean(rch[rch > 0]):.6f} m/day")
        if rch.shape != (12, nrow, ncol):
            raise ValueError(
                f"Recharge array shape {rch.shape} does not match "
                f"expected (12, {nrow}, {ncol}). Re-run recharge_model.py."
            )
        return rch
    print(f"WARNING: {npy_path} not found. Falling back to uniform 1e-4 m/day.")
    return np.full((12, nrow, ncol), 1e-4)


# ==============================================================================
# CALIBRATION TARGETS
# ==============================================================================

def find_calibration_cells(calibration_targets, minx, maxy, nrow, ncol, ibound_2d):
    print("\n" + "=" * 60)
    print("LOCATING CALIBRATION TARGETS")
    print("=" * 60)

    calibration_cells = []
    for t in calibration_targets:
        col = int(np.floor((t['x'] - minx) / DELR))
        row = int(np.floor((maxy - t['y']) / DELC))

        if 0 <= row < nrow and 0 <= col < ncol:
            if ibound_2d[row, col] != 0:
                calibration_cells.append(
                    (row, col, t['x'], t['y'], t['id'], t['obs_head'])
                )
                print(f"Target {t['id']:>2}: ({t['x']:.1f}, {t['y']:.1f}) "
                      f"-> Cell ({row}, {col}), obs={t['obs_head']:.1f} m")
            else:
                print(f"Target {t['id']:>2}: -> INACTIVE CELL")
        else:
            print(f"Target {t['id']:>2}: -> OUTSIDE GRID")

    print(f"\n✓ Located {len(calibration_cells)} targets in active cells")
    return calibration_cells


def extract_calibration_heads(head_2d, calibration_cells, top_arr):
    """Extract simulated heads (NaN-masked 2D array) and residuals vs observed."""
    data = []
    for row, col, x, y, tid, obs_head in calibration_cells:
        sim   = head_2d[row, col]
        valid = not np.isnan(sim)
        data.append({
            'target_id':      tid,
            'row':            row,
            'col':            col,
            'x':              x,
            'y':              y,
            'observed_head':  obs_head,
            'simulated_head': float(sim) if valid else np.nan,
            'residual':       float(sim - obs_head) if valid else np.nan,
            'land_surface':   top_arr[row, col],
            'status':         'VALID' if valid else 'DRY/INACTIVE'
        })
    return data


def export_calibration_data(calibration_data, workspace):
    """Export calibration data to CSV and print statistics."""
    import pandas as pd

    df = pd.DataFrame(calibration_data)
    csv_file = os.path.join(workspace, 'calibration_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ Calibration data saved: {csv_file}")

    valid = [d for d in calibration_data if d['status'] == 'VALID']
    if not valid:
        print("No valid targets — no statistics computed.")
        return

    sim    = np.array([d['simulated_head'] for d in valid])
    obs    = np.array([d['observed_head']  for d in valid])
    res    = sim - obs
    rmse   = np.sqrt(np.mean(res**2))
    mae    = np.mean(np.abs(res))
    ss_res = np.sum((obs - sim)**2)
    ss_tot = np.sum((obs - np.mean(obs))**2)
    r2     = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    print(f"\n{'=' * 40}")
    print(f"CALIBRATION STATISTICS (end of {MONTH_LABELS[CALIB_MONTH - 1]}, cycle {N_CYCLES})")
    print(f"{'=' * 40}")
    print(f"  Valid targets:  {len(valid)}/{len(calibration_data)}")
    print(f"  RMSE:           {rmse:.3f} m")
    print(f"  MAE:            {mae:.3f} m")
    print(f"  R\u00b2:             {r2:.3f}")
    print(f"  Mean residual:  {np.mean(res):.3f} m")
    print(f"  Max residual:   {np.max(np.abs(res)):.3f} m")
    print(f"  Obs range:      {obs.min():.1f} to {obs.max():.1f} m")
    print(f"  Sim range:      {sim.min():.1f} to {sim.max():.1f} m")
    print(f"{'=' * 40}")


# ==============================================================================
# MODFLOW MODEL BUILD
# ==============================================================================

def resolve_mf2005_path(explicit_path):
    p = shutil.which("mf2005")
    if p:
        return p
    if os.path.isfile(explicit_path):
        if os.name != "nt":
            os.chmod(explicit_path, 0o755)
        return explicit_path
    raise FileNotFoundError("MODFLOW-2005 executable not found.")


def build_model(model_name, top_arr, botm_list, ibound_mask_2d,
                nrow, ncol, recharge_array):
    exe = resolve_mf2005_path(EXE_MF2005)
    mf  = flopy.modflow.Modflow(modelname=model_name,
                                model_ws=WORKSPACE, exe_name=exe)

    # DIS
    flopy.modflow.ModflowDis(
        mf, nlay=NLAY, nrow=nrow, ncol=ncol,
        delr=DELR, delc=DELC, top=top_arr, botm=botm_list,
        nper=NPER, perlen=PERLEN, nstp=NSTP, steady=STEADY
    )

    # BAS — copy so the 2D active mask is not modified in place
    ibound  = np.expand_dims(ibound_mask_2d, axis=0).copy()
    chd_spd = []

    south_cols = np.where(ibound[0, nrow - 1, :] == 1)[0]
    for j in south_cols:
        ibound[0, nrow - 1, j] = -1
        chd_spd.append([0, nrow - 1, j, CH_ELEV_SOUTH, CH_ELEV_SOUTH])
    print(f"CHD south: {len(south_cols)} cells (H={CH_ELEV_SOUTH} m)")

    if CH_ELEV_NORTH is not None:
        north_cols = np.where(ibound[0, 0, :] == 1)[0]
        for j in north_cols:
            ibound[0, 0, j] = -1
            chd_spd.append([0, 0, j, CH_ELEV_NORTH, CH_ELEV_NORTH])
        print(f"CHD north: {len(north_cols)} cells (H={CH_ELEV_NORTH} m)")
    else:
        print("North boundary: no-flow (catchment divide)")

    strt = np.where(ibound != 0,
                    np.full((NLAY, nrow, ncol), INITIAL_HEAD), 1e30)
    flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt, hnoflo=1e30)

    if chd_spd:
        flopy.modflow.ModflowChd(
            mf, stress_period_data={sp: chd_spd for sp in range(NPER)},
            ipakcb=CBC_UNIT
        )

    # LPF
    flopy.modflow.ModflowLpf(
        mf, hk=HK, vka=VKA, laytyp=LAYTYP, sy=SY, ss=SS,
        laywet=[1], wetfct=0.1, iwetit=1, ihdwet=0, wetdry=[1.0],
        ipakcb=CBC_UNIT
    )

    # RCH: spin-up uses mean annual; transient periods cycle through the 12 months
    mean_rch = recharge_array.mean(axis=0)
    rch_spd  = {}
    for sp in range(NPER):
        rch_2d = (mean_rch if sp == 0 else recharge_array[(sp - 1) % 12]).copy()
        rch_2d[ibound[0] == 0]  = 0.0
        rch_2d[ibound[0] == -1] = 0.0     # no recharge on CHD cells
        rch_spd[sp] = rch_2d
    flopy.modflow.ModflowRch(mf, rech=rch_spd, ipakcb=CBC_UNIT)

    # Solver
    flopy.modflow.ModflowPcg(
        mf, hclose=0.01, rclose=10.0,
        mxiter=5000, iter1=1000, relax=0.97
    )

    # Output control — save the LAST time step of every stress period.
    # NOTE: ModflowOc keys are (period, step), the reverse of HeadFile's
    # kstpkper = (step, period) used in get_heads().
    flopy.modflow.ModflowOc(
        mf,
        stress_period_data={
            (sp, NSTP[sp] - 1): ["save head", "save budget", "print budget"]
            for sp in range(NPER)
        },
        compact=True
    )

    return mf, ibound


# ==============================================================================
# RUN AND READ
# ==============================================================================

def run_and_read(mf):
    mf.write_input()
    ok, _ = mf.run_model(silent=False, report=True)

    if not ok:
        lst = os.path.join(mf.model_ws, f"{mf.name}.list")
        try:
            with open(lst, "r", errors="ignore") as f:
                print("\nMODFLOW LIST FILE TAIL\n" + "=" * 40)
                print("".join(f.readlines()[-200:]))
        except Exception:
            pass
        raise RuntimeError(f"MODFLOW failed for {mf.name}")

    hds = HeadFile(os.path.join(mf.model_ws, f"{mf.name}.hds"))
    cbc = CellBudgetFile(os.path.join(mf.model_ws, f"{mf.name}.cbc"))
    return hds, cbc


def check_cycle_convergence(hds):
    """Compare end-of-December heads between the last two cycles."""
    if N_CYCLES < 2:
        print("Cycle convergence: only 1 cycle — increase N_CYCLES to check.")
        return
    sp_last = final_cycle_sp(12)
    sp_prev = sp_last - 12
    d = np.abs(get_heads(hds, sp_last) - get_heads(hds, sp_prev))
    print(f"\nCycle convergence (Dec, cycle {N_CYCLES} vs {N_CYCLES - 1}): "
          f"max |Δh| = {np.nanmax(d):.3f} m, mean |Δh| = {np.nanmean(d):.3f} m")
    if np.nanmax(d) > 0.1:
        print("  NOTE: heads still drifting between cycles — consider increasing N_CYCLES.")


# ==============================================================================
# FLOW VECTORS
# ==============================================================================

def calculate_flow_vectors(head_2d, ibound_2d, nrow, ncol):
    """Darcy flux (m/day). qx east-positive; qy positive toward increasing row (south)."""
    h    = head_2d
    dhdx = (h[:, 1:] - h[:, :-1]) / DELR
    dhdy = (h[1:, :] - h[:-1, :]) / DELC

    qx = np.full((nrow, ncol), np.nan)
    qy = np.full((nrow, ncol), np.nan)

    qx[:, 1:-1] = -HK[0] * 0.5 * (dhdx[:, :-1] + dhdx[:, 1:])
    qx[:, 0]    = -HK[0] * dhdx[:, 0]
    qx[:, -1]   = -HK[0] * dhdx[:, -1]

    qy[1:-1, :] = -HK[0] * 0.5 * (dhdy[:-1, :] + dhdy[1:, :])
    qy[0, :]    = -HK[0] * dhdy[0, :]
    qy[-1, :]   = -HK[0] * dhdy[-1, :]

    dry     = np.isnan(h) | (ibound_2d == 0)
    qx[dry] = np.nan
    qy[dry] = np.nan

    return qx, qy, np.sqrt(qx**2 + qy**2)


# ==============================================================================
# FLOWLINES → GEOJSON (HydroWindy)
# ==============================================================================

def trace_flowlines(head_2d, ibound_2d, nrow, ncol, minx, maxy,
                    seed_spacing=FLOWLINE_SEED_SPACING, step=FLOWLINE_STEP,
                    max_steps=FLOWLINE_MAX_STEPS, min_v=FLOWLINE_MIN_V):
    """Trace downgradient streamlines (RK2 midpoint, fixed step in metres).
    Returns a list of (path_xy[N,2], mean_seepage_velocity_m_per_day)."""
    from scipy.interpolate import RegularGridInterpolator

    qx, qy_south, _ = calculate_flow_vectors(head_2d, ibound_2d, nrow, ncol)
    vx = qx / SY[0]           # seepage velocity, east-positive
    vy = -qy_south / SY[0]    # seepage velocity, north-positive

    xc = minx + DELR * (np.arange(ncol) + 0.5)
    yc = maxy - DELC * (np.arange(nrow) + 0.5)   # descending (row 0 = north)

    fx = RegularGridInterpolator((yc[::-1], xc), vx[::-1],
                                 bounds_error=False, fill_value=np.nan)
    fy = RegularGridInterpolator((yc[::-1], xc), vy[::-1],
                                 bounds_error=False, fill_value=np.nan)

    def vel(p):
        pt = [[p[1], p[0]]]
        return np.array([fx(pt)[0], fy(pt)[0]])

    lines = []
    for i in range(0, nrow, seed_spacing):
        for j in range(0, ncol, seed_spacing):
            if ibound_2d[i, j] != 1 or np.isnan(vx[i, j]):
                continue
            p = np.array([xc[j], yc[i]])
            path, speeds = [p.copy()], []
            for _ in range(max_steps):
                v1 = vel(p); s1 = np.hypot(*v1)
                if not np.isfinite(s1) or s1 < min_v:
                    break
                v2 = vel(p + 0.5 * step * v1 / s1); s2 = np.hypot(*v2)
                if not np.isfinite(s2) or s2 < min_v:
                    break
                p = p + step * v2 / s2
                path.append(p.copy()); speeds.append(s2)
            if len(path) >= 3:
                lines.append((np.array(path), float(np.mean(speeds))))

    print(f"✓ Traced {len(lines)} flowlines")
    return lines


def export_flowlines_geojson(lines, src_crs_wkt, out_path, step=FLOWLINE_STEP):
    """Reproject flowlines to WGS84 and write a GeoJSON FeatureCollection."""
    from pyproj import Transformer

    tf = Transformer.from_crs(src_crs_wkt, "EPSG:4326", always_xy=True)
    features = []
    for k, (path, v_mean) in enumerate(lines):
        lon, lat = tf.transform(path[:, 0], path[:, 1])
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[round(float(a), 6), round(float(b), 6)]
                                for a, b in zip(lon, lat)]
            },
            "properties": {
                "id": k,
                "mean_velocity_m_per_day": round(v_mean, 4),
                "length_m": round((len(path) - 1) * step, 1)
            }
        })

    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print(f"✓ Flowlines GeoJSON saved: {out_path}")

    if WEB_DATA_DIR:
        os.makedirs(WEB_DATA_DIR, exist_ok=True)
        dest = os.path.join(WEB_DATA_DIR, "flowlines.json")
        shutil.copyfile(out_path, dest)
        print(f"✓ Copied to web project: {dest}")


# ==============================================================================
# PLOTTING
# ==============================================================================

def plot_seasonal_heads(hds, calibration_cells, extent):
    print("\nPlotting seasonal head maps (final cycle)...")
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    axes = axes.flatten()

    xs = [x for _, _, x, y, _, _ in calibration_cells]
    ys = [y for _, _, x, y, _, _ in calibration_cells]

    for m in range(1, 13):
        h2d = get_heads(hds, final_cycle_sp(m))
        ax  = axes[m - 1]
        im  = ax.imshow(h2d, cmap='viridis', origin='upper', extent=extent)
        plt.colorbar(im, ax=ax, shrink=0.7, label='Head (m)')
        ax.scatter(xs, ys, c='red', s=25, edgecolors='white',
                   linewidths=0.5, zorder=3)
        ax.set_title(f"{MONTH_LABELS[m - 1]} (end of month)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'Simulated Hydraulic Heads — Monthly, Cycle {N_CYCLES}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(WORKSPACE, 'seasonal_head_maps.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {out}")


def plot_head_and_flow(hds, ibound_2d, calibration_cells, nrow, ncol,
                       minx, maxy, extent, month=12):
    label = MONTH_LABELS[month - 1]
    print(f"\nPlotting {label} head + flow streamlines...")

    h2d = get_heads(hds, final_cycle_sp(month))
    qx, qy, qmag = calculate_flow_vectors(h2d, ibound_2d, nrow, ncol)

    # streamplot needs ascending y; arrays have row 0 = north, so flip them
    x_coords = minx + DELR * (np.arange(ncol) + 0.5)
    y_coords = (maxy - nrow * DELC) + DELC * (np.arange(nrow) + 0.5)
    X, Y     = np.meshgrid(x_coords, y_coords)

    qx_plot = np.flipud(np.where(np.isnan(qx), 0.0, qx))
    qy_plot = np.flipud(np.where(np.isnan(qy), 0.0, -qy))   # north-positive

    lw = 2.0 * (qmag / np.nanmax(qmag))
    lw = np.clip(np.where(np.isnan(lw), 0.5, lw), 0.5, 2.5)

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(h2d, cmap='viridis', origin='upper', extent=extent, alpha=0.5)
    plt.colorbar(im, ax=ax, label='Hydraulic Head (m)', shrink=0.8)

    ax.streamplot(X, Y, qx_plot, qy_plot, density=1.5, color='cyan',
                  linewidth=np.flipud(lw), arrowsize=1.5, zorder=2)

    ax.text(0.02, 0.98,
            f'Darcy flux max: {np.nanmax(qmag):.4f} m/day\n'
            f'Darcy flux mean: {np.nanmean(qmag):.4f} m/day',
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for _, _, x, y, tid, _ in calibration_cells:
        ax.scatter(x, y, c='red', s=120, marker='*',
                   edgecolors='yellow', linewidth=1.5, zorder=3)
        ax.annotate(str(tid), (x, y), xytext=(6, 6),
                    textcoords='offset points', color='yellow',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='red', alpha=0.7), zorder=3)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(f'{label} (end of month, cycle {N_CYCLES}) — '
                 'Hydraulic Head with Groundwater Flow Lines', fontsize=13)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.grid(True, alpha=0.2, color='white', linestyle='--')
    plt.tight_layout()

    out = os.path.join(WORKSPACE, f'{label.lower()}_head_flow_streamlines.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {out}")


def plot_head_hydrograph(hds, calibration_cells):
    """Heads at targets through ALL transient cycles (shows spin-up convergence)."""
    print("\nPlotting head hydrographs...")

    sps   = list(range(1, NPER))
    heads = np.array([get_heads(hds, sp) for sp in sps])   # (nsp, nrow, ncol)
    t     = np.arange(1, len(sps) + 1)

    fig, ax = plt.subplots(figsize=(14, 6))
    for row, col, x, y, tid, obs_head in calibration_cells:
        ax.plot(t, heads[:, row, col], marker='o', markersize=3, label=f'Target {tid}')

    for k in range(1, N_CYCLES):
        ax.axvline(12 * k + 0.5, color='grey', linestyle=':', linewidth=1)

    ax.set_xticks(t)
    ax.set_xticklabels([MONTH_LABELS[(i - 1) % 12][0] for i in t], fontsize=8)
    ax.set_title(f'Head at Calibration Targets — {N_CYCLES} transient cycles '
                 '(dotted lines = cycle boundaries)', fontsize=13)
    ax.set_xlabel('Month')
    ax.set_ylabel('Simulated Head (m)')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    out = os.path.join(WORKSPACE, 'calibration_head_hydrographs.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {out}")


def plot_calibration_scatter(calibration_data, workspace):
    """Scatter plot of simulated vs observed heads + residuals bar chart."""
    print("\nPlotting calibration scatter...")

    valid = [d for d in calibration_data if d['status'] == 'VALID']
    if not valid:
        print("No valid targets — skipping scatter.")
        return
    obs = np.array([d['observed_head']  for d in valid])
    sim = np.array([d['simulated_head'] for d in valid])
    ids = [d['target_id'] for d in valid]
    res = sim - obs

    rmse   = np.sqrt(np.mean(res**2))
    ss_res = np.sum((obs - sim)**2)
    ss_tot = np.sum((obs - np.mean(obs))**2)
    r2     = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(obs, sim, c='steelblue', s=80,
               edgecolors='black', linewidths=0.5, zorder=3)
    for i, tid in enumerate(ids):
        ax.annotate(str(tid), (obs[i], sim[i]),
                    xytext=(4, 4), textcoords='offset points', fontsize=8)

    lims = [min(obs.min(), sim.min()) - 1, max(obs.max(), sim.max()) + 1]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='1:1 line')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Observed Head (m)', fontsize=12)
    ax.set_ylabel('Simulated Head (m)', fontsize=12)
    ax.set_title(f'Simulated vs Observed — end of {MONTH_LABELS[CALIB_MONTH - 1]}',
                 fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.text(0.05, 0.95,
            f'RMSE = {rmse:.2f} m\nR\u00b2 = {r2:.3f}\nn = {len(valid)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2 = axes[1]
    colors = ['tomato' if r > 0 else 'steelblue' for r in res]
    ax2.bar([str(i) for i in ids], res,
            color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_xlabel('Target ID', fontsize=12)
    ax2.set_ylabel('Residual — Simulated minus Observed (m)', fontsize=12)
    ax2.set_title('Head Residuals at Calibration Targets', fontsize=13)
    ax2.grid(True, alpha=0.4, axis='y')

    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(facecolor='tomato',    label='Overestimated (Sim > Obs)'),
        Patch(facecolor='steelblue', label='Underestimated (Sim < Obs)')
    ], fontsize=9)

    plt.tight_layout()
    out = os.path.join(workspace, 'calibration_scatter.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: {out}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"\n{'=' * 60}")
    print("MODFLOW Transient Recharge Model — Upper Rio Cobre (v21)")
    print(f"{'=' * 60}\n")

    # Grid
    print("Generating model grid...")
    TOP_ARR, IBOUND_2D, NROW, NCOL, minx, maxy, crs_wkt = \
        assign_dem_to_modflow_top(SHP_PATH, DEM_PATH, DELR, DELC)
    EXTENT = [minx, minx + NCOL * DELR, maxy - NROW * DELC, maxy]

    # Thickness check
    min_active_top = np.min(TOP_ARR[IBOUND_2D == 1])
    if BOTM_ELEVATION >= min_active_top - MIN_THICKNESS:
        BOTM = [min_active_top - MIN_THICKNESS]
        print(f"[GEOMETRY] Adjusted BOTM to {BOTM[0]:.1f} m")
    else:
        BOTM = [BOTM_ELEVATION]
    print(f"Layer bottom: {BOTM[0]:.1f} m")

    # Recharge
    print("\nLoading recharge array...")
    recharge_array = load_recharge_array(RECHARGE_NPY, NROW, NCOL)

    # Calibration targets
    calibration_cells = find_calibration_cells(
        CALIBRATION_TARGETS, minx, maxy, NROW, NCOL, IBOUND_2D
    )

    # Build and run
    print("\nBuilding MODFLOW model...")
    mf, IBOUND_3D = build_model(MODEL_NAME, TOP_ARR, BOTM, IBOUND_2D,
                                NROW, NCOL, recharge_array)
    IBOUND_MODEL = IBOUND_3D[0]   # includes -1 for CHD cells

    print("Running MODFLOW...")
    hds, cbc = run_and_read(mf)
    print("✓ MODFLOW completed successfully")

    check_cycle_convergence(hds)

    # Calibration on final cycle
    h_cal    = get_heads(hds, final_cycle_sp(CALIB_MONTH))
    cal_data = extract_calibration_heads(h_cal, calibration_cells, TOP_ARR)
    export_calibration_data(cal_data, WORKSPACE)

    # Plots
    plot_seasonal_heads(hds, calibration_cells, EXTENT)
    plot_head_and_flow(hds, IBOUND_MODEL, calibration_cells, NROW, NCOL,
                       minx, maxy, EXTENT, month=CALIB_MONTH)
    plot_head_hydrograph(hds, calibration_cells)
    plot_calibration_scatter(cal_data, WORKSPACE)

    # Flowlines for HydroWindy
    print("\nTracing flowlines for web export...")
    h_flow = get_heads(hds, final_cycle_sp(FLOWLINE_MONTH))
    lines  = trace_flowlines(h_flow, IBOUND_MODEL, NROW, NCOL, minx, maxy)
    export_flowlines_geojson(lines, crs_wkt,
                             os.path.join(WORKSPACE, "flowlines.json"))

    # Summary
    valid_heads = [d['simulated_head'] for d in cal_data if d['status'] == 'VALID']
    print(f"\n{'=' * 60}")
    print("MODEL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Grid:           {NROW} rows x {NCOL} cols x {NLAY} layer")
    print(f"Cell size:      {DELR} m x {DELC} m")
    print(f"Active cells:   {np.sum(IBOUND_2D == 1)}")
    print(f"Stress periods: {NPER} (1 spin-up + {N_CYCLES} x 12 monthly transient)")
    print(f"Recharge:       spatially distributed (IDW, 0.17 x rainfall)")
    print(f"K:              {HK[0]} m/d,  Sy: {SY[0]}")
    print(f"CHD south:      {CH_ELEV_SOUTH} m")
    print(f"North boundary: {'no-flow' if CH_ELEV_NORTH is None else f'CHD {CH_ELEV_NORTH} m'}")
    if valid_heads:
        print(f"Head range ({MONTH_LABELS[CALIB_MONTH - 1]}, cycle {N_CYCLES}): "
              f"{min(valid_heads):.1f} to {max(valid_heads):.1f} m")
    print(f"Flowlines:      {len(lines)} exported")
    print(f"Results:        {WORKSPACE}")
    print("✓ MODEL COMPLETE")


if __name__ == "__main__":
    main()
