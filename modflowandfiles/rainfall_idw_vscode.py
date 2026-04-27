# -*- coding: utf-8 -*-
"""
Rainfall IDW Interpolation - Upper Rio Cobre Catchment
Adapted from Colab notebook for VS Code / local Python environment.

Inputs:
    - Totals_Daily_rainfall_st_catherine_2000-2025.xlsx
    - rio_cobre_stns.xlsx
    - Upper_RioCobre_SWMU.shp

Outputs:
    - rainfall_idw_map.png        : IDW map with basin boundary and stations
    - monthly_correlation.png     : monthly rainfall correlation heatmap
    - merged_rainfall_stations.csv: per-station monthly + annual rainfall
    - monthly_recharge_per_station.csv: recharge = rainfall * 0.17 (m/day)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pyproj
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from thefuzz import fuzz

# ==============================================================================
# CONFIGURATION — update paths here if needed
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAINFALL_FILE  = os.path.join(BASE_DIR, "Totals_Daily_rainfall_st_catherine_2000-2025.xlsx")
STATIONS_FILE  = os.path.join(BASE_DIR, "rio_cobre_stns.xlsx")
SHP_PATH       = os.path.join(BASE_DIR, "Upper_RioCobre_SWMU.shp")
PRJ_PATH       = os.path.join(BASE_DIR, "Upper_RioCobre_SWMU.prj")
OUTPUT_DIR     = os.path.join(BASE_DIR, "rainfall_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RECHARGE_COEFF = 0.17          # SWAT-derived deep recharge fraction (Curtis 2025)
NUM_YEARS      = 26            # 2000-2025
IDW_POWER      = 2
IDW_GRID_CELLS = 100
IDW_BUFFER_DEG = 0.02

MONTH_ORDER = ['Jan', 'Feb', 'Mar', 'April', 'May', 'Jun',
               'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# ==============================================================================
# STEP 1 — LOAD AND PROCESS DAILY RAINFALL DATA
# ==============================================================================

def load_rainfall_data(filepath):
    """
    Load multi-sheet Excel file (one sheet per year).
    Aggregate daily/monthly values to total monthly rainfall per station.
    Returns DataFrame with columns: Stations, Jan, Feb, ..., Dec
    """
    print("\n--- STEP 1: Loading rainfall data ---")
    try:
        xls = pd.ExcelFile(filepath)
        sheet_names = xls.sheet_names
        print(f"Found {len(sheet_names)} sheets: {sheet_names}")
    except Exception as e:
        raise FileNotFoundError(f"Could not load rainfall file: {e}")

    # Skip non-year sheets (e.g. 'Trend Analysis')
    year_sheets = [s for s in sheet_names if 'year' in s.lower() or s.strip().isdigit()]
    print(f"Processing {len(year_sheets)} year sheets...")

    all_years_data = []

    for sheet_name in year_sheets:
        try:
            df_year = xls.parse(sheet_name)

            # Find station column
            station_col = None
            for candidate in ['Stations', 'STATION', 'Station']:
                if candidate in df_year.columns:
                    station_col = candidate
                    break
            if station_col is None:
                print(f"  Warning: No station column in '{sheet_name}', skipping.")
                continue

            df_year[station_col] = df_year[station_col].astype(str).str.strip().str.lower()

            # Find which month columns are present
            present_months = [m for m in MONTH_ORDER if m in df_year.columns]
            if not present_months:
                print(f"  Warning: No month columns in '{sheet_name}', skipping.")
                continue

            # Replace nodata flag and coerce to numeric
            df_year = df_year.replace(-99.0, np.nan).replace(-99, np.nan)
            for col in present_months:
                df_year[col] = pd.to_numeric(df_year[col], errors='coerce')

            df_clean = df_year[[station_col] + present_months].copy()
            df_clean = df_clean.rename(columns={station_col: 'Stations'})
            all_years_data.append(df_clean)

            print(f"  Processed {sheet_name}: {len(df_clean)} stations")

        except Exception as e:
            print(f"  Error in sheet '{sheet_name}': {e}, skipping.")

    if not all_years_data:
        raise RuntimeError("No rainfall data could be processed.")

    combined = pd.concat(all_years_data, ignore_index=True)

    # Sum monthly rainfall across all years per station
    # Use only MONTH_ORDER columns that are present
    available_months = [m for m in MONTH_ORDER if m in combined.columns]
    rainfall_df = combined.groupby('Stations')[available_months].sum().reset_index()

    print(f"\nTotal monthly rainfall computed for {len(rainfall_df)} stations.")
    print(rainfall_df[['Stations'] + available_months[:4]].head())

    return rainfall_df, available_months


# ==============================================================================
# STEP 2 — LOAD STATION COORDINATES AND CONVERT DMS TO DECIMAL DEGREES
# ==============================================================================

def dms_to_dd(dms_str):
    """Convert 'DD MM SS' string to decimal degrees."""
    try:
        parts = str(dms_str).strip().split()
        d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        return d + m / 60 + s / 3600
    except Exception:
        return np.nan


def load_station_coordinates(filepath):
    """
    Load station coordinates from Excel file.
    Converts LAT and LONG from DMS format to decimal degrees.
    """
    print("\n--- STEP 2: Loading station coordinates ---")
    try:
        rio_cobre_df = pd.read_excel(filepath)
        print(f"Loaded {len(rio_cobre_df)} stations.")
        print("Columns:", rio_cobre_df.columns.tolist())
    except Exception as e:
        raise FileNotFoundError(f"Could not load stations file: {e}")

    rio_cobre_df['LAT_DD']  = rio_cobre_df['LAT'].apply(dms_to_dd)
    rio_cobre_df['LONG_DD'] = rio_cobre_df['LONG'].apply(dms_to_dd) * -1  # West = negative

    rio_cobre_df['STATION_NA'] = rio_cobre_df['STATION_NA'].astype(str).str.strip().str.lower()

    print("Converted coordinates (first 5 rows):")
    print(rio_cobre_df[['STATION_NA', 'LAT_DD', 'LONG_DD']].head())

    return rio_cobre_df


# ==============================================================================
# STEP 3 — MERGE RAINFALL AND COORDINATES WITH FUZZY MATCHING
# ==============================================================================

def merge_rainfall_coordinates(rainfall_df, rio_cobre_df, month_cols,
                                match_threshold=50):
    """
    Merge rainfall data with station coordinates.
    Uses exact match first, then fuzzy matching for unmatched stations.
    """
    print("\n--- STEP 3: Merging rainfall and coordinates ---")

    # Exact merge first
    merged = pd.merge(
        rainfall_df, rio_cobre_df,
        left_on='Stations', right_on='STATION_NA',
        how='inner'
    )
    print(f"Exact matches: {len(merged)} stations")

    # Fuzzy match for unmatched rainfall stations
    unmatched_rain = rainfall_df[~rainfall_df['Stations'].isin(merged['Stations'])]
    unmatched_coords = rio_cobre_df[~rio_cobre_df['STATION_NA'].isin(merged['STATION_NA'])]

    if not unmatched_rain.empty and not unmatched_coords.empty:
        print(f"Attempting fuzzy matching for {len(unmatched_rain)} unmatched rainfall stations...")
        fuzzy_pairs = []

        for r_name in unmatched_rain['Stations']:
            best_score, best_match = 0, None
            for c_name in unmatched_coords['STATION_NA']:
                score = fuzz.token_set_ratio(r_name, c_name)
                if score > best_score:
                    best_score, best_match = score, c_name
            if best_match and best_score >= match_threshold:
                fuzzy_pairs.append((r_name, best_match, best_score))
                print(f"  '{r_name}' -> '{best_match}' (score: {best_score})")

        # Add fuzzy-matched rows
        for r_name, c_name, score in fuzzy_pairs:
            r_row = unmatched_rain[unmatched_rain['Stations'] == r_name].copy()
            c_row = unmatched_coords[unmatched_coords['STATION_NA'] == c_name]
            if not c_row.empty:
                r_row['STATION_NA'] = c_name
                r_row['LAT_DD']     = c_row['LAT_DD'].values[0]
                r_row['LONG_DD']    = c_row['LONG_DD'].values[0]
                merged = pd.concat([merged, r_row], ignore_index=True)

    # Compute average annual rainfall
    merged['Average_Annual_Rainfall_mm'] = merged[month_cols].sum(axis=1, skipna=True) / NUM_YEARS

    print(f"\nFinal merged dataset: {len(merged)} stations")
    print(merged[['Stations', 'LAT_DD', 'LONG_DD', 'Average_Annual_Rainfall_mm']].to_string())

    return merged


# ==============================================================================
# STEP 4 — IDW INTERPOLATION
# ==============================================================================

def idw_interpolation(known_points, known_values, interpolation_points, power=2):
    """
    Inverse Distance Weighting interpolation.
    known_points: (N, 2) array of [lat, lon]
    known_values: (N,) array of values
    interpolation_points: (M, 2) array of [lat, lon]
    Returns: (M,) interpolated values
    """
    known_points_arr = np.array(known_points)
    interp_arr       = np.array(interpolation_points)
    result           = np.zeros(len(interp_arr))

    for i, pt in enumerate(interp_arr):
        distances = cdist([pt], known_points_arr)[0]
        if 0.0 in distances:
            result[i] = known_values[np.where(distances == 0)[0][0]]
        else:
            weights = 1.0 / (distances ** power)
            result[i] = np.sum(weights / weights.sum() * known_values)

    return result


def build_idw_grid(merged_df, shp_path, value_col='Average_Annual_Rainfall_mm'):
    """
    Build IDW interpolation grid covering the catchment extent.
    Returns: lon_mesh, lat_mesh, interpolated_grid, transformer, custom_crs
    """
    print(f"\n--- STEP 4: IDW interpolation of '{value_col}' ---")

    # Load CRS from .prj file
    prj_file = shp_path.replace('.shp', '.prj')
    try:
        with open(prj_file, 'r') as f:
            custom_crs = pyproj.CRS.from_wkt(f.read())
        print(f"Loaded CRS: {custom_crs.name}")
    except Exception as e:
        print(f"CRS load failed ({e}), falling back to EPSG:32618")
        custom_crs = pyproj.CRS.from_epsg(32618)

    transformer = pyproj.Transformer.from_crs("EPSG:4326", custom_crs, always_xy=True)

    # Get geographic bounds from shapefile
    gdf_geo = gpd.read_file(shp_path)
    if gdf_geo.crs and gdf_geo.crs.to_epsg() != 4326:
        gdf_geo = gdf_geo.to_crs(epsg=4326)
    min_lon_shp, min_lat_shp, max_lon_shp, max_lat_shp = gdf_geo.total_bounds

    # Grid extent covers both stations and shapefile
    min_lat = min(merged_df['LAT_DD'].min(), min_lat_shp) - IDW_BUFFER_DEG
    max_lat = max(merged_df['LAT_DD'].max(), max_lat_shp) + IDW_BUFFER_DEG
    min_lon = min(merged_df['LONG_DD'].min(), min_lon_shp) - IDW_BUFFER_DEG
    max_lon = max(merged_df['LONG_DD'].max(), max_lon_shp) + IDW_BUFFER_DEG

    lat_grid = np.linspace(min_lat, max_lat, IDW_GRID_CELLS)
    lon_grid = np.linspace(min_lon, max_lon, IDW_GRID_CELLS)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    interp_points = np.column_stack([lat_mesh.flatten(), lon_mesh.flatten()])

    # Run IDW
    known_pts = merged_df[['LAT_DD', 'LONG_DD']].values
    known_vals = merged_df[value_col].values
    valid = ~np.isnan(known_vals)
    print(f"Running IDW with {valid.sum()} valid stations...")

    interpolated = idw_interpolation(known_pts[valid], known_vals[valid],
                                     interp_points, power=IDW_POWER)
    grid = interpolated.reshape(lat_mesh.shape)

    print("IDW interpolation complete.")
    return lon_mesh, lat_mesh, grid, transformer, custom_crs


# ==============================================================================
# STEP 5 — RECHARGE COMPUTATION (per station, monthly)
# ==============================================================================

def compute_monthly_recharge(merged_df, month_cols):
    """
    Compute monthly recharge per station using SWAT recharge coefficient.
    Returns DataFrame with per-station monthly recharge in m/day.
    """
    print("\n--- STEP 5: Computing monthly recharge ---")

    recharge_df = merged_df[['Stations', 'LAT_DD', 'LONG_DD']].copy()

    for month, days in zip(month_cols, DAYS_PER_MONTH[:len(month_cols)]):
        # Mean monthly rainfall (mm/month) = total / num_years
        mean_monthly_mm = merged_df[month] / NUM_YEARS
        # Recharge (m/day) = rainfall_mm * coeff / 1000 / days
        recharge_df[f'rch_{month}_m_per_day'] = (
            mean_monthly_mm * RECHARGE_COEFF / 1000.0 / days
        )

    print(f"Monthly recharge computed for {len(recharge_df)} stations.")
    print(recharge_df[['Stations'] + [f'rch_{m}_m_per_day' for m in month_cols[:3]]].head())

    return recharge_df


# ==============================================================================
# STEP 6 — PLOTTING
# ==============================================================================

def plot_idw_map(lon_mesh, lat_mesh, grid, merged_df,
                 transformer, custom_crs, shp_path, output_dir):
    """Plot IDW rainfall map with basin outline and station locations."""
    print("\n--- STEP 6a: Plotting IDW rainfall map ---")

    gdf_proj = gpd.read_file(shp_path).to_crs(custom_crs)

    # Reproject grid to projected CRS
    x_proj, y_proj = transformer.transform(lon_mesh, lat_mesh)

    # Mask grid to shapefile
    pts_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x_proj.flatten(), y_proj.flatten()),
        crs=custom_crs
    )
    within = gpd.sjoin(pts_gdf, gdf_proj, how='inner', predicate='intersects')
    mask = np.zeros(len(x_proj.flatten()), dtype=bool)
    mask[within.index.values] = True
    grid_masked = np.where(mask.reshape(x_proj.shape), grid, np.nan)

    # Station projected coordinates
    x_stns, y_stns = transformer.transform(
        merged_df['LONG_DD'].values, merged_df['LAT_DD'].values
    )

    fig, ax = plt.subplots(figsize=(14, 12))

    mesh = ax.pcolormesh(x_proj, y_proj, grid_masked,
                         cmap='viridis', shading='auto', zorder=1)
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Average Annual Rainfall (mm)', fontsize=14)

    ax.contour(x_proj, y_proj, grid_masked,
               levels=10, colors='black', alpha=0.5, linewidths=0.7, zorder=2)

    gdf_proj.plot(ax=ax, facecolor='none', edgecolor='blue',
                  linewidth=2, alpha=1.0, zorder=6)

    ax.scatter(x_stns, y_stns, color='red', s=80, edgecolors='black',
               zorder=4, label='Rainfall stations')

    for i, name in enumerate(merged_df['Stations']):
        ax.text(x_stns[i], y_stns[i], name.title(),
                fontsize=9, ha='left', va='bottom', weight='bold',
                bbox=dict(facecolor='white', alpha=0.5,
                          edgecolor='none', boxstyle='round,pad=0.2'),
                zorder=5)

    ax.set_title('IDW Interpolation of Average Annual Rainfall (2000-2025)\nUpper Rio Cobre Catchment',
                 fontsize=16)
    ax.set_xlabel('Easting (m)', fontsize=13)
    ax.set_ylabel('Northing (m)', fontsize=13)
    ax.legend(fontsize=12)
    plt.tight_layout()

    out = os.path.join(output_dir, 'rainfall_idw_map.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {out}")


def plot_monthly_correlation(merged_df, month_cols, output_dir):
    """Plot monthly rainfall correlation heatmap across stations."""
    print("\n--- STEP 6b: Plotting monthly correlation heatmap ---")

    corr = merged_df[month_cols].corr()

    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Monthly Rainfall (2000-2025)', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Month')
    plt.tight_layout()

    out = os.path.join(output_dir, 'monthly_correlation.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {out}")


def plot_mean_monthly_bar(merged_df, month_cols, output_dir):
    """Bar chart of mean monthly rainfall across all stations."""
    print("\n--- STEP 6c: Plotting mean monthly rainfall ---")

    mean_monthly = (merged_df[month_cols].sum(axis=0, skipna=True) /
                    (NUM_YEARS * len(merged_df)))

    plt.figure(figsize=(10, 5))
    plt.bar(MONTH_LABELS[:len(month_cols)], mean_monthly.values, color='steelblue')
    plt.title('Mean Monthly Rainfall - All Stations (2000-2025)', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Mean Monthly Rainfall (mm)')
    plt.tight_layout()

    out = os.path.join(output_dir, 'mean_monthly_rainfall.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {out}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("RAINFALL IDW PROCESSING — Upper Rio Cobre")
    print("=" * 60)

    # Step 1: Load rainfall
    rainfall_df, month_cols = load_rainfall_data(RAINFALL_FILE)

    # Step 2: Load coordinates
    rio_cobre_df = load_station_coordinates(STATIONS_FILE)

    # Step 3: Merge
    merged_df = merge_rainfall_coordinates(rainfall_df, rio_cobre_df, month_cols)

    # Step 4: IDW interpolation
    lon_mesh, lat_mesh, rain_grid, transformer, custom_crs = build_idw_grid(
        merged_df, SHP_PATH, value_col='Average_Annual_Rainfall_mm'
    )

    # Step 5: Recharge computation
    recharge_df = compute_monthly_recharge(merged_df, month_cols)

    # Step 6: Plots
    plot_idw_map(lon_mesh, lat_mesh, rain_grid, merged_df,
                 transformer, custom_crs, SHP_PATH, OUTPUT_DIR)
    plot_monthly_correlation(merged_df, month_cols, OUTPUT_DIR)
    plot_mean_monthly_bar(merged_df, month_cols, OUTPUT_DIR)

    # Save outputs
    csv_merged = os.path.join(OUTPUT_DIR, 'merged_rainfall_stations.csv')
    merged_df.to_csv(csv_merged, index=False)
    print(f"\nSaved merged data: {csv_merged}")

    csv_recharge = os.path.join(OUTPUT_DIR, 'monthly_recharge_per_station.csv')
    recharge_df.to_csv(csv_recharge, index=False)
    print(f"Saved recharge data: {csv_recharge}")

    print("\n" + "=" * 60)
    print("DONE. All outputs saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
