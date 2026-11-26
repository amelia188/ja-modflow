# -*- coding: utf-8 -*-
"""
export_hydrowindy.py
--------------------
Exports everything the HydroWindy web app needs from a COMPLETED model run (v22, MODFLOW 6).
It does not re-run MODFLOW. It reads, from runs/<MODEL_NAME>/:
  grid.npz   grid, georeference and final IDOMAIN/CHD cells, saved by run_model.py
  .hds       heads;   .cbc  budget, including MODFLOW 6 specific discharge (DATA-SPDIS)
so it never rebuilds the grid or repeats the boundary rules, and it can be run from any folder.

Usage:
  Put this file next to run_model.py and run it (from any folder) after the model.

Writes (to <workspace>/hydrowindy_data/, and copies to WEB_DATA_DIR if set):
  metadata.json          run info, calibration stats, grid georeference, ranges, budget
  grids.json             per-cell land surface + 12 monthly head / depth-to-water / recharge
  flowlines_01..12.json  forward flowlines per month (with travel time per vertex)
  capture.json           backward pathlines from each well, split into travel-time bands
  contours.json          head contours for all 12 months
  wells.geojson          calibration wells with observed + 12 monthly simulated heads
  basin.geojson          Upper Rio Cobre SWMU boundary
  chd.geojson            constant-head boundary cells as lines
  stations.geojson       rainfall stations (if merged_rainfall_stations.csv is found)

Flowlines/capture zones use a 2D RK2 tracer on MODFLOW 6's cell-centred specific discharge,
divided by Sy as effective porosity. Good for visualisation; for report-grade travel times
use MODPATH 7.
"""

import os
import sys
import json
import glob
import shutil
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_model as m   # v22 settings and helpers; importing it has no side effects

from flopy.utils import HeadFile, CellBudgetFile
from pyproj import Transformer, CRS
from scipy.interpolate import RegularGridInterpolator

# ==============================================================================
# CONFIGURATION
# ==============================================================================

OUT_DIR      = os.path.join(m.WORKSPACE, "hydrowindy_data")
WEB_DATA_DIR = m.WEB_DATA_DIR   # override here if you like, e.g. r"...\hydro-windy-app\data"

SEED_SPACING   = 4        # forward flowline seed every Nth cell
TRACE_STEP     = 50.0     # tracer step (m)
MAX_STEPS      = 4000
MIN_V          = 1e-6     # m/day
DECIMATE       = 4        # keep every Nth traced vertex in the output (step 50 m -> 200 m)

CAPTURE_BANDS_YR  = [0, 1, 5, 10, 25]   # travel-time band edges (years)
CAPTURE_PARTICLES = 16                  # particles released around each well
CAPTURE_RADIUS    = 30.0                # release ring radius (m)

STATIONS_CSV_NAME = "merged_rainfall_stations.csv"

MONTHS = m.MONTH_LABELS
DAYS   = m.DAYS_PER_MONTH

# ==============================================================================
# HELPERS
# ==============================================================================

def rnd(v, nd):
    """Round a scalar; NaN/inf -> None (valid JSON)."""
    v = float(v)
    return round(v, nd) + 0.0 if np.isfinite(v) else None   # +0.0 turns -0.0 into 0.0


def flat(arr, nd):
    return [rnd(v, nd) for v in np.asarray(arr, float).ravel()]


def save_json(obj, name):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"), allow_nan=False)
    print(f"  ✓ {name:<22} {os.path.getsize(path) / 1024:8.0f} KB")
    return name


def nice_interval(span, target=12):
    raw = span / max(target, 1)
    for s in [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100]:
        if s >= raw:
            return s
    return 100.0


def decimate_idx(n, k):
    idx = list(range(0, n, k))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


# ==============================================================================
# VELOCITY FIELD + VECTORISED TRACER
# ==============================================================================

class Grid:
    pass


def make_vel_fn(qx, qy, G):
    """Interpolated seepage velocity from MODFLOW 6 specific discharge (qy north-positive)."""
    sy = np.asarray(m.SY, float)
    vx = qx / sy                 # seepage velocity, east-positive
    vy = qy / sy                 # seepage velocity, north-positive
    ya = G.yc[::-1]
    fx = RegularGridInterpolator((ya, G.xc), vx[::-1], bounds_error=False, fill_value=np.nan)
    fy = RegularGridInterpolator((ya, G.xc), vy[::-1], bounds_error=False, fill_value=np.nan)

    def vel(P):
        q = np.column_stack([P[:, 1], P[:, 0]])
        return np.column_stack([fx(q), fy(q)])

    return vel, vx


def trace(seeds, vel, backward=False, max_time=None):
    """Trace all seeds at once. Returns lists of paths (N,2) and cumulative times (days)."""
    sign  = -1.0 if backward else 1.0
    P     = np.asarray(seeds, float).copy()
    n     = len(P)
    T     = np.zeros(n)
    alive = np.ones(n, bool)
    paths = [[P[i].copy()] for i in range(n)]
    times = [[0.0] for _ in range(n)]

    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(MAX_STEPS):
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            v1 = sign * vel(P[idx]); s1 = np.hypot(v1[:, 0], v1[:, 1])
            ok = np.isfinite(s1) & (s1 > MIN_V)
            mid = P[idx] + 0.5 * TRACE_STEP * v1 / np.where(ok, s1, 1.0)[:, None]
            v2 = sign * vel(mid); s2 = np.hypot(v2[:, 0], v2[:, 1])
            ok &= np.isfinite(s2) & (s2 > MIN_V)
            alive[idx[~ok]] = False
            idx, v2, s2 = idx[ok], v2[ok], s2[ok]
            if idx.size == 0:
                break
            P[idx] += TRACE_STEP * v2 / s2[:, None]
            T[idx] += TRACE_STEP / s2
            for i in idx:
                paths[i].append(P[i].copy())
                times[i].append(T[i])
            if max_time is not None:
                alive[idx[T[idx] >= max_time]] = False

    return [np.array(p) for p in paths], [np.array(t) for t in times]


# ==============================================================================
# EXPORTERS
# ==============================================================================

def export_flowlines(G, H, to_ll):
    print("\nFlowlines (12 months)...")
    seeds = [(G.xc[j], G.yc[i])
             for i in range(0, G.nrow, SEED_SPACING)
             for j in range(0, G.ncol, SEED_SPACING)
             if G.ib[i, j] == 1]
    seeds = np.array(seeds)
    files, all_v = [], []

    for k in range(12):
        vel, vx = make_vel_fn(*G.flow[k], G)
        paths, times = trace(seeds, vel)
        feats = []
        for path, t in zip(paths, times):
            if len(path) < 3 or t[-1] <= 0:
                continue
            idx = decimate_idx(len(path), DECIMATE)
            lon, lat = to_ll.transform(path[idx, 0], path[idx, 1])
            length = (len(path) - 1) * TRACE_STEP
            v_mean = length / t[-1]
            all_v.append(v_mean)
            feats.append({
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [[round(float(a), 5), round(float(b), 5)]
                                             for a, b in zip(lon, lat)]},
                "properties": {
                    "id": len(feats),
                    "v":  round(v_mean, 4),          # mean seepage velocity (m/day)
                    "len": round(length, 1),          # m
                    "tt": round(float(t[-1]), 1),     # total travel time (days)
                    "t":  [round(float(t[i]), 1) for i in idx],        # cumulative days per vertex
                }
            })
        files.append(save_json({"type": "FeatureCollection", "features": feats},
                               f"flowlines_{k + 1:02d}.json"))

    v = np.array(all_v) if all_v else np.array([0.0, 1.0])
    return files, [rnd(np.percentile(v, 5), 5), rnd(np.percentile(v, 95), 5)]


def export_capture(G, H, cells, to_ll):
    print("\nCapture zones (mean annual flow field)...")
    qx_mean = np.nanmean(np.array([f[0] for f in G.flow]), axis=0)
    qy_mean = np.nanmean(np.array([f[1] for f in G.flow]), axis=0)
    vel, _ = make_vel_fn(qx_mean, qy_mean, G)
    max_t  = CAPTURE_BANDS_YR[-1] * 365.25
    ang    = np.linspace(0, 2 * np.pi, CAPTURE_PARTICLES, endpoint=False)
    feats  = []

    for row, col, x, y, tid, _ in cells:
        seeds = np.column_stack([x + CAPTURE_RADIUS * np.cos(ang),
                                 y + CAPTURE_RADIUS * np.sin(ang)])
        paths, times = trace(seeds, vel, backward=True, max_time=max_t)
        for path, t in zip(paths, times):
            if len(path) < 2:
                continue
            ty = t / 365.25
            for b in range(len(CAPTURE_BANDS_YR) - 1):
                t0, t1 = CAPTURE_BANDS_YR[b], CAPTURE_BANDS_YR[b + 1]
                a = max(int(np.searchsorted(ty, t0, "left")) - 1, 0)
                e = int(np.searchsorted(ty, t1, "right"))
                seg = path[a:e]
                if len(seg) < 2:
                    continue
                seg = seg[decimate_idx(len(seg), 2)]
                lon, lat = to_ll.transform(seg[:, 0], seg[:, 1])
                feats.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString",
                                 "coordinates": [[round(float(p), 5), round(float(q), 5)]
                                                 for p, q in zip(lon, lat)]},
                    "properties": {"well": tid, "band": b, "t0": t0, "t1": t1}
                })
    return save_json({"type": "FeatureCollection", "features": feats,
                      "bands_years": CAPTURE_BANDS_YR}, "capture.json")


def export_contours(G, H, to_ll):
    print("\nHead contours...")
    import contourpy
    hmin = float(np.nanmin(H)); hmax = float(np.nanmax(H))
    interval = nice_interval(hmax - hmin)
    levels = np.arange(np.ceil(hmin / interval) * interval, hmax + 1e-9, interval)
    ya = G.yc[::-1]
    feats = []
    for k in range(12):
        z = np.ma.masked_invalid(H[k][::-1])
        gen = contourpy.contour_generator(G.xc, ya, z, line_type=contourpy.LineType.Separate)
        for lev in levels:
            for line in gen.lines(float(lev)):
                if len(line) < 2:
                    continue
                lon, lat = to_ll.transform(line[:, 0], line[:, 1])
                feats.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString",
                                 "coordinates": [[round(float(a), 5), round(float(b), 5)]
                                                 for a, b in zip(lon, lat)]},
                    "properties": {"month": k + 1, "level": round(float(lev), 2)}
                })
    name = save_json({"type": "FeatureCollection", "features": feats}, "contours.json")
    return name, interval


def export_grids(G, H, D, R):
    print("\nGrids...")
    return save_json({
        "nrow": G.nrow, "ncol": G.ncol,
        "top":      flat(np.where(G.active, G.top, np.nan), 1),
        "head":     [flat(H[k], 2) for k in range(12)],
        "depth":    [flat(D[k], 2) for k in range(12)],
        "recharge": [flat(R[k], 2) for k in range(12)],
    }, "grids.json")


def export_wells(G, H, cells, to_ll):
    print("\nWells...")
    feats = []
    for row, col, x, y, tid, obs in cells:
        lon, lat = to_ll.transform(x, y)
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
            "properties": {
                "id": tid, "row": int(row), "col": int(col),
                "obs": obs,
                "sim": [rnd(H[k][row, col], 3) for k in range(12)],
                "land": rnd(G.top[row, col], 2),
            }
        })
    return save_json({"type": "FeatureCollection", "features": feats}, "wells.geojson")


def export_basin(crs_wkt):
    print("\nBasin boundary...")
    import geopandas as gpd
    gdf = gpd.read_file(m.SHP_PATH).to_crs(crs_wkt)
    gdf["geometry"] = gdf.geometry.simplify(20)
    gdf = gdf[["geometry"]].to_crs(4326)
    return save_json(json.loads(gdf.to_json()), "basin.geojson")


def export_chd(G, to_ll):
    print("\nBoundary cells...")
    feats = []
    rows = []
    for r in sorted({int(c[0]) for c in G.chd}):
        heads = [c[2] for c in G.chd if int(c[0]) == r]
        rows.append((r, float(np.mean(heads)), "south" if r >= G.nrow // 2 else "north"))
    for r, h, side in rows:
        cols = np.where(G.ib[r] == -1)[0]
        if cols.size == 0:
            continue
        runs = np.split(cols, np.where(np.diff(cols) > 1)[0] + 1)
        for run in runs:
            xs = [G.minx + run[0] * m.DELR, G.minx + (run[-1] + 1) * m.DELR]
            ys = [G.yc[r], G.yc[r]]
            lon, lat = to_ll.transform(xs, ys)
            feats.append({
                "type": "Feature",
                "geometry": {"type": "LineString",
                             "coordinates": [[round(float(a), 6), round(float(b), 6)]
                                             for a, b in zip(lon, lat)]},
                "properties": {"head": h, "side": side, "cells": int(run.size)}
            })
    return save_json({"type": "FeatureCollection", "features": feats}, "chd.geojson")


def export_stations():
    print("\nRainfall stations...")
    import pandas as pd
    hits = glob.glob(os.path.join(m.BASE_DIR, "**", STATIONS_CSV_NAME), recursive=True)
    if not hits:
        print(f"  – {STATIONS_CSV_NAME} not found; skipping stations")
        return None
    try:
        df = pd.read_csv(hits[0])
        cols = list(df.columns)
        low  = {c: str(c).lower() for c in cols}
        num  = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

        lat_c = next((c for c in num if "lat" in low[c] and df[c].between(17, 19).all()), None)
        lon_c = next((c for c in num if "lon" in low[c] and df[c].between(-79, -76).all()), None)
        if lat_c is None:
            lat_c = next((c for c in num if df[c].between(17, 19).all()), None)
        if lon_c is None:
            lon_c = next((c for c in num if df[c].between(-79, -76).all()), None)
        name_c = next((c for c in cols if any(s in low[c] for s in ("station", "name"))
                       and not pd.api.types.is_numeric_dtype(df[c])), None)
        if name_c is None:
            name_c = next((c for c in cols if not pd.api.types.is_numeric_dtype(df[c])), None)

        mon_c = []
        for mo in MONTHS:
            c = next((c for c in num if low[c].startswith(mo.lower())), None)
            mon_c.append(c)

        if lat_c is None or lon_c is None:
            print(f"  – could not find decimal lat/lon columns in {hits[0]}; skipping")
            return None

        monthly = None
        scale = 1.0
        if all(mon_c):
            vals = df[mon_c].to_numpy(float)
            # Columns may be multi-year totals: rescale if the annual sum is implausible
            if np.nanmedian(vals.sum(axis=1)) > 8000:
                scale = 1.0 / 26.0
                print("  – monthly columns look like 26-year totals; dividing by 26")
            monthly = vals * scale

        feats = []
        for i, r in df.iterrows():
            props = {"name": str(r[name_c]) if name_c else f"Station {i + 1}"}
            if monthly is not None:
                props["monthly"] = [rnd(v, 1) for v in monthly[i]]
                props["annual"] = rnd(np.nansum(monthly[i]), 0)
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point",
                             "coordinates": [round(float(r[lon_c]), 6), round(float(r[lat_c]), 6)]},
                "properties": props
            })
        return save_json({"type": "FeatureCollection", "features": feats}, "stations.geojson")
    except Exception as e:
        print(f"  – stations skipped ({e})")
        return None


def export_budget(cbc):
    """Monthly water budget (m3/day) from the cell budget file, final cycle, MODFLOW 6 terms merged."""
    print("\nWater budget...")
    try:
        months, terms = [], []
        for k in range(1, 13):
            bt = m.budget_terms(cbc, m.final_cycle_sp(k))
            for t in bt:
                if t not in terms:
                    terms.append(t)
            months.append({"month": k, "in": {t: rnd(v[0], 1) for t, v in bt.items()},
                           "out": {t: rnd(v[1], 1) for t, v in bt.items()}})
        print(f"  ✓ terms: {', '.join(terms)}")
        return {"units": "m3/day", "terms": terms, "months": months}
    except Exception as e:
        print(f"  – budget skipped ({e})")
        return None


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(f"\n{'=' * 60}\nHydroWindy export — {m.MODEL_NAME}\n{'=' * 60}")
    hds_path = os.path.join(m.WORKSPACE, f"{m.MODEL_NAME}.hds")
    cbc_path = os.path.join(m.WORKSPACE, f"{m.MODEL_NAME}.cbc")
    if not os.path.exists(hds_path):
        raise SystemExit(f"No head file at {hds_path}. Run run_model.py first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Grid exactly as the model saved it (no rebuilding, no repeated boundary rules)
    g = m.load_grid()
    top, nrow, ncol, minx, maxy, crs_wkt = g["top"], g["nrow"], g["ncol"], g["minx"], g["maxy"], g["crs_wkt"]
    G = Grid()
    G.top, G.nrow, G.ncol, G.minx, G.maxy = top, nrow, ncol, minx, maxy
    G.active = g["active"]
    G.ib = g["ibound"]
    G.chd = g["chd"]
    G.xc, G.yc = g["xc"], g["yc"]
    ib2d = g["idomain"]

    to_ll = Transformer.from_crs(crs_wkt, "EPSG:4326", always_xy=True)
    try:
        proj4 = CRS.from_wkt(crs_wkt).to_proj4()
    except Exception:
        proj4 = None

    # Heads, depth to water, recharge — final cycle
    hds = HeadFile(hds_path)
    cbc = CellBudgetFile(cbc_path, precision="double")
    G.flow = [m.flow_vectors(cbc, m.final_cycle_sp(k), nrow, ncol)[:2] for k in range(1, 13)]
    H = [np.where(G.active, m.get_heads(hds, m.final_cycle_sp(k)), np.nan) for k in range(1, 13)]
    D = [np.where(G.active, top - H[k], np.nan) for k in range(12)]
    rch = m.load_recharge_array(m.RECHARGE_NPY, nrow, ncol)
    R = [np.where(G.active, rch[k] * DAYS[k] * 1000.0, np.nan) for k in range(12)]   # mm/month

    cells = m.find_calibration_cells(m.CALIBRATION_TARGETS, minx, maxy, nrow, ncol, ib2d, verbose=False)

    # Exports
    flow_files, v_range = export_flowlines(G, H, to_ll)
    capture_file = export_capture(G, H, cells, to_ll)
    contour_file, interval = export_contours(G, H, to_ll)
    grids_file = export_grids(G, H, D, R)
    wells_file = export_wells(G, H, cells, to_ll)
    basin_file = export_basin(crs_wkt)
    chd_file = export_chd(G, to_ll)
    stations_file = export_stations()
    budget = export_budget(cbc)

    # Calibration stats at the calibration month: same definition as the model's CSV
    k = m.CALIB_MONTH - 1
    st = m.calibration_stats([H[k][c[0], c[1]] for c in cells], [c[5] for c in cells])
    calib = {"month": m.CALIB_MONTH, "n": st["n"],
             "rmse": rnd(st["rmse"], 3) if st["rmse"] is not None else None,
             "mae": rnd(st["mae"], 3) if st["mae"] is not None else None,
             "r2": rnd(st["r2"], 3) if st["r2"] is not None else None,
             "mean_residual": rnd(st["me"], 3) if st["me"] is not None else None}

    # Georeference
    x0, x1 = minx, minx + ncol * m.DELR
    y0, y1 = maxy - nrow * m.DELC, maxy
    cx, cy = to_ll.transform([x0, x1, x1, x0], [y1, y1, y0, y0])   # TL, TR, BR, BL
    corners = [[round(a, 6), round(b, 6)] for a, b in zip(cx, cy)]
    lons, lats = [c[0] for c in corners], [c[1] for c in corners]

    def rng(arrs, lo=0, hi=100, nd=2):
        a = np.concatenate([np.ravel(x) for x in arrs]); a = a[np.isfinite(a)]
        return [rnd(np.percentile(a, lo), nd), rnd(np.percentile(a, hi), nd)]

    meta = {
        "run": {
            "model": m.MODEL_NAME,
            "exported": datetime.datetime.now().isoformat(timespec="minutes"),
            "version": m.MODEL_VERSION, "engine": "MODFLOW 6",
            "k_m_per_day": (float(m.HK) if np.ndim(m.HK) == 0 else f"zoned, mean {np.mean(m.HK):.0f}"),
            "sy": float(np.mean(m.SY)),
            "n_cycles": m.N_CYCLES,
            "chd_south_m": m.CH_ELEV_SOUTH,
            "north_boundary": "no-flow" if m.CH_ELEV_NORTH is None else f"CHD {m.CH_ELEV_NORTH} m",
            "recharge": "IDW rainfall x 0.17 (Curtis 2025)",
        },
        "calibration": calib,
        "months": MONTHS,
        "grid": {
            "nrow": nrow, "ncol": ncol, "delr": m.DELR, "delc": m.DELC,
            "minx": minx, "maxy": maxy, "proj4": proj4, "corners": corners,
        },
        "bounds": [[min(lons), min(lats)], [max(lons), max(lats)]],
        "ranges": {
            "head": rng(H), "depth": rng(D, 2, 98, 1), "recharge": rng(R),
            "velocity": v_range,
        },
        "contour_interval": interval,
        "capture_bands_years": CAPTURE_BANDS_YR,
        "budget": budget,
        "files": {
            "grids": grids_file, "wells": wells_file, "basin": basin_file,
            "chd": chd_file, "stations": stations_file, "contours": contour_file,
            "capture": capture_file, "flowlines": flow_files,
        },
    }
    save_json(meta, "metadata.json")

    if WEB_DATA_DIR:
        os.makedirs(WEB_DATA_DIR, exist_ok=True)
        for f in os.listdir(OUT_DIR):
            shutil.copyfile(os.path.join(OUT_DIR, f), os.path.join(WEB_DATA_DIR, f))
        print(f"\n✓ Copied to web project: {WEB_DATA_DIR}")

    print(f"\n✓ Export complete: {OUT_DIR}")


if __name__ == "__main__":
    main()
