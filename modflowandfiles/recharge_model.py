# -*- coding: utf-8 -*-
"""
recharge_model.py
-----------------
Builds a spatially distributed monthly recharge array for MODFLOW.

Pipeline:
  1. Load per-station monthly recharge CSV (output of rainfall_idw_vscode.py)
  2. IDW interpolate each month to the MODFLOW 200m grid
  3. Mask to active cells only
  4. Save recharge array as .npy file: shape (12, NROW, NCOL), units m/day
  5. Save diagnostic plots

Recharge = mean monthly rainfall * 0.17 / 1000 / days_in_month  [m/day]
(coefficient from Curtis 2025, Table 4.8 - SWAT water balance)

Run this BEFORE run_model.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import pyproj
from shapely.geometry import Point
from scipy.spatial.distance import cdist

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input files
RECHARGE_CSV  = os.path.join(BASE_DIR, "rainfall_outputs", "monthly_recharge_per_station.csv")
SHP_PATH      = os.path.join(BASE_DIR, "Upper_RioCobre_SWMU.shp")
DEM_PATH      = os.path.join(BASE_DIR, "riocobre2.tif")

# Output
OUTPUT_DIR    = os.path.join(BASE_DIR, "rainfall_outputs")
RECHARGE_NPY  = os.path.join(OUTPUT_DIR, "recharge_grid_12months.npy")

# MODFLOW grid (must match run_model.py)
DELR = DELC = 200.0

# Month metadata
MONTH_COLS = ['Jan', 'Feb', 'Mar', 'April', 'May', 'Jun',
              'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

IDW_POWER = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# STEP 1 — BUILD MODFLOW GRID (same logic as run_model.py)
# ==============================================================================

def build_modflow_grid(shp_path, dem_path, delr, delc):
    """
    Recreate the MODFLOW active cell mask and grid coordinates.
    Returns: ibound_2d, nrow, ncol, minx, miny, maxx, maxy
    """
    print("Building MODFLOW grid...")

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs

    gdf = gpd.read_file(shp_path).to_crs(dem_crs)
    minx, miny, maxx, maxy = gdf.total_bounds

    ncol = int((maxx - minx) / delr)
    nrow = int((maxy - miny) / delc)

    # Exact 200 m cell centres from the north-west corner, as in run_model.py (v21+).
    # (np.linspace(minx + delr/2, maxx - delr/2, ncol) stretched the spacing to fit the bounds.)
    x = minx + delr * (np.arange(ncol) + 0.5)
    y = maxy - delc * (np.arange(nrow) + 0.5)
    X, Y = np.meshgrid(x, y)

    mask = np.zeros((nrow, ncol), dtype=bool)
    for i in range(nrow):
        for j in range(ncol):
            if gdf.contains(Point(X[i, j], Y[i, j])).any():
                mask[i, j] = True

    ibound_2d = np.where(mask, 1, 0)
    print(f"Grid: {nrow} rows x {ncol} cols, {mask.sum()} active cells")

    return ibound_2d, nrow, ncol, minx, miny, maxx, maxy, X, Y


# ==============================================================================
# STEP 2 — IDW INTERPOLATION TO MODFLOW GRID
# ==============================================================================

def idw_interpolation(known_points, known_values, query_points, power=2):
    """
    IDW interpolation from station points to MODFLOW cell centroids.
    known_points: (N, 2) projected coordinates [x, y]
    query_points: (M, 2) projected coordinates [x, y]
    Returns: (M,) interpolated values
    """
    result = np.zeros(len(query_points))
    for i, pt in enumerate(query_points):
        dists = cdist([pt], known_points)[0]
        if 0.0 in dists:
            result[i] = known_values[np.where(dists == 0)[0][0]]
        else:
            w = 1.0 / (dists ** power)
            result[i] = np.sum(w / w.sum() * known_values)
    return result


def project_stations(recharge_df, shp_path):
    """
    Project station lat/lon coordinates to the shapefile's projected CRS.
    Returns: projected x, y arrays
    """
    prj_file = shp_path.replace('.shp', '.prj')
    try:
        with open(prj_file, 'r') as f:
            crs = pyproj.CRS.from_wkt(f.read())
    except Exception:
        crs = pyproj.CRS.from_epsg(32618)

    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_proj, y_proj = transformer.transform(
        recharge_df['LONG_DD'].values,
        recharge_df['LAT_DD'].values
    )
    return x_proj, y_proj


def build_recharge_grid(recharge_df, ibound_2d, nrow, ncol, X, Y, shp_path):
    """
    IDW interpolate each month's recharge values to the MODFLOW grid.
    Returns: recharge_array shape (12, NROW, NCOL) in m/day
    """
    print("\nProjecting station coordinates to MODFLOW CRS...")
    x_stns, y_stns = project_stations(recharge_df, shp_path)
    known_points = np.column_stack([x_stns, y_stns])

    # Only interpolate to active cells
    active_rows, active_cols = np.where(ibound_2d == 1)
    query_points = np.column_stack([
        X[active_rows, active_cols],
        Y[active_rows, active_cols]
    ])

    recharge_array = np.zeros((12, nrow, ncol), dtype=float)

    print("Interpolating monthly recharge to MODFLOW grid...")
    for m_idx, (month, days) in enumerate(zip(MONTH_COLS, DAYS_PER_MONTH)):
        col_name = f'rch_{month}_m_per_day'
        if col_name not in recharge_df.columns:
            print(f"  Warning: column '{col_name}' not found, using zeros.")
            continue

        known_values = recharge_df[col_name].values
        valid = ~np.isnan(known_values)

        if valid.sum() < 2:
            print(f"  Warning: fewer than 2 valid stations for {month}, skipping.")
            continue

        interpolated = idw_interpolation(
            known_points[valid], known_values[valid], query_points, power=IDW_POWER
        )

        # Assign to active cells
        recharge_array[m_idx, active_rows, active_cols] = interpolated

        mean_rch = interpolated.mean()
        print(f"  {MONTH_LABELS[m_idx]:>3}: mean recharge = {mean_rch:.6f} m/day  "
              f"({mean_rch * days * 1000:.1f} mm/month)")

    # Zero out inactive cells
    for m in range(12):
        recharge_array[m][ibound_2d == 0] = 0.0

    return recharge_array


# ==============================================================================
# STEP 3 — DIAGNOSTIC PLOTS
# ==============================================================================

def plot_recharge_maps(recharge_array, ibound_2d, minx, miny, maxx, maxy, output_dir):
    """Plot monthly recharge maps as a 3x4 grid."""
    print("\nPlotting monthly recharge maps...")

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    axes = axes.flatten()

    masked = recharge_array.copy().astype(float)
    for m in range(12):
        masked[m][ibound_2d == 0] = np.nan

    vmin = np.nanmin(masked[masked > 0])
    vmax = np.nanmax(masked)

    for m in range(12):
        ax = axes[m]
        im = ax.imshow(masked[m], cmap='Blues', origin='upper',
                       extent=[minx, maxx, miny, maxy],
                       vmin=vmin, vmax=vmax)
        ax.set_title(MONTH_LABELS[m], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8, label='m/day')

    fig.suptitle('Monthly Recharge Distribution on MODFLOW Grid\n'
                 '(Rainfall × 0.17, IDW interpolated)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(output_dir, 'monthly_recharge_maps.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {out}")


def plot_mean_annual_recharge(recharge_array, ibound_2d, minx, miny, maxx, maxy, output_dir):
    """Plot mean annual recharge map."""
    print("Plotting mean annual recharge map...")

    annual = recharge_array.sum(axis=0) * np.array(DAYS_PER_MONTH)[:, None, None]
    annual = annual.sum(axis=0) * 1000  # convert to mm/year — wait, sum already done
    # Correct: sum(rch_m/day * days) over 12 months, then * 1000 for mm
    annual_mm = np.zeros_like(recharge_array[0])
    for m in range(12):
        annual_mm += recharge_array[m] * DAYS_PER_MONTH[m] * 1000

    annual_mm[ibound_2d == 0] = np.nan

    plt.figure(figsize=(10, 8))
    im = plt.imshow(annual_mm, cmap='Blues', origin='upper',
                    extent=[minx, maxx, miny, maxy])
    plt.colorbar(im, label='Annual Recharge (mm/year)')
    plt.title('Mean Annual Recharge on MODFLOW Grid\n'
              f'(Active cells mean: {np.nanmean(annual_mm):.1f} mm/year)',
              fontsize=13)
    plt.xlabel('Easting (m)')
    plt.ylabel('Northing (m)')
    plt.tight_layout()

    out = os.path.join(output_dir, 'annual_recharge_map.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {out}")


def plot_seasonal_cycle(recharge_array, ibound_2d, output_dir):
    """Plot spatially averaged seasonal recharge cycle."""
    print("Plotting seasonal recharge cycle...")

    mean_monthly = []
    for m in range(12):
        active_vals = recharge_array[m][ibound_2d == 1]
        mean_monthly.append(active_vals.mean() * DAYS_PER_MONTH[m] * 1000)  # mm/month

    plt.figure(figsize=(10, 5))
    plt.bar(MONTH_LABELS, mean_monthly, color='steelblue', edgecolor='white')
    plt.title('Spatially Averaged Monthly Recharge — MODFLOW Grid\n'
              f'Total annual: {sum(mean_monthly):.1f} mm/year', fontsize=13)
    plt.xlabel('Month')
    plt.ylabel('Mean Recharge (mm/month)')
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()

    out = os.path.join(output_dir, 'seasonal_recharge_cycle.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {out}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("RECHARGE MODEL — Upper Rio Cobre")
    print("=" * 60)

    # Step 1: Load per-station recharge
    print(f"\nLoading recharge data from: {RECHARGE_CSV}")
    recharge_df = pd.read_csv(RECHARGE_CSV)
    print(f"Loaded {len(recharge_df)} stations.")
    print(recharge_df[['Stations', 'LAT_DD', 'LONG_DD']].to_string())

    # Step 2: Build MODFLOW grid
    ibound_2d, nrow, ncol, minx, miny, maxx, maxy, X, Y = build_modflow_grid(
        SHP_PATH, DEM_PATH, DELR, DELC
    )

    # Step 3: IDW interpolate to MODFLOW grid
    recharge_array = build_recharge_grid(
        recharge_df, ibound_2d, nrow, ncol, X, Y, SHP_PATH
    )

    # Step 4: Save array
    np.save(RECHARGE_NPY, recharge_array)
    print(f"\nRecharge array saved: {RECHARGE_NPY}")
    print(f"Shape: {recharge_array.shape}  (12 months x {nrow} rows x {ncol} cols)")

    # Step 5: Diagnostic plots
    plot_recharge_maps(recharge_array, ibound_2d, minx, miny, maxx, maxy, OUTPUT_DIR)
    plot_mean_annual_recharge(recharge_array, ibound_2d, minx, miny, maxx, maxy, OUTPUT_DIR)
    plot_seasonal_cycle(recharge_array, ibound_2d, OUTPUT_DIR)

    # Summary
    active_annual = []
    for m in range(12):
        active_annual.append(recharge_array[m][ibound_2d == 1].mean() * DAYS_PER_MONTH[m] * 1000)

    print("\n" + "=" * 60)
    print("RECHARGE SUMMARY")
    print("=" * 60)
    for m, (label, val) in enumerate(zip(MONTH_LABELS, active_annual)):
        print(f"  {label:>3}: {val:6.1f} mm/month")
    print(f"  Annual total: {sum(active_annual):.1f} mm/year")
    print(f"\nRecharge array ready for MODFLOW: {RECHARGE_NPY}")


if __name__ == "__main__":
    main()
