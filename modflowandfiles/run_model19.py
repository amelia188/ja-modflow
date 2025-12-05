# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 20:41:09 2025

@author: 20002213
"""

# -*- coding: utf-8 -*-
"""
MODFLOW Single-Layer Model with Integrated DEM Processing, Varying Thickness, 
Calibration Targets, and Flow Vector Visualization
Created on Tue Nov 18 21:00:00 2025
@author: 20002213
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from shapely.geometry import Point

try:
    import flopy
    from flopy.utils import HeadFile, CellBudgetFile
except ModuleNotFoundError as e:
    raise SystemExit(
        "FloPy is not installed. Install with: conda install -c conda-forge flopy numpy matplotlib pandas"
    ) from e

# ==============================================================================
# ----------------------------- CONFIGURATION ----------------------------------
# ==============================================================================

# File Paths
SHP_PATH = "Upper_RioCobre_SWMU.shp"
DEM_PATH = "riocobre2.tif"

# MODFLOW Executable
EXE_MF2005 = r"C:\Users\ameli\Github\projects\ja-modflow\mf2005.exe"

# Workspace setup
FILE_DIR = os.getcwd()
WORKSPACE = os.path.join(FILE_DIR, "mf2005_riocobre_model")
os.makedirs(WORKSPACE, exist_ok=True)

# Grid parameters
DELR = DELC = 200.0              # Cell size in meters (using your working value)
NLAY = 1                         # Single layer
MIN_THICKNESS = 10.0             # Minimum thickness required

# Aquifer bottom (single layer)
BOTM_ELEVATION = 0.0 

# MODFLOW Parameters (Single Layer)
HK       = [20.0]                # Horizontal K in m/d
VKA      = [20.0]                # Vertical K in m/d
SY       = [0.20]                # Specific yield
SS       = 1e-5                  # Specific storage
LAYTYP   = [1]                   # Convertible (unconfined)

# Stresses
RCH_RATE = 1e-4                  # Recharge rate (m/d)
CBC_UNIT = 53                    # Cell-by-cell budget unit (for flow vectors)
CH_ELEV  = 5.0                   # Constant Head elevation (m)

CALIBRATION_TARGETS = [
    (748740.4205, 660747.1651),
    (749538.4976, 661301.9048),
    (748862.8926, 661688.6217),
    (748862.8926, 661688.6217),   # duplicate kept only if you want it
    (746973.2362, 663735.7684),
    (750878.5184, 661012.7804),
    (751545.3569, 660868.9354),
    (751449.8523, 664156.1805),
    (752455.0110, 665860.7871),
    (742297.6009, 669060.8732),
    (748370.4468, 666657.6147),
    (752487.1230, 662971.9867),
    (752370.8074, 662119.7216),
    (745848.6474, 661594.7216)
]

# ==============================================================================
# ------------------------------ GRID GENERATION -------------------------------
# ==============================================================================

def assign_dem_to_modflow_top(shapefile_path, dem_path, delr, delc):
    
    """
    DEM processing and grid definition.
    """
    print("Loading files and harmonizing CRS...")
    
    # Load shapefile and DEM info
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_array = src.read(1)
        dem_nodata = src.nodata

    gdf = gpd.read_file(shapefile_path).to_crs(dem_crs)

    # Define grid from shapefile bounds
    minx, miny, maxx, maxy = gdf.total_bounds

    ncol = int((maxx - minx) / delr)
    nrow = int((maxy - miny) / delc)

    # Generate grid cell centroids
    x = np.linspace(minx + delr / 2, maxx - delr / 2, ncol)
    y = np.linspace(maxy - delc / 2, miny + delc / 2, nrow)
    X, Y = np.meshgrid(x, y)

    # Mask grid cells inside shapefile
    mask = np.zeros((nrow, ncol), dtype=bool)
    for i in range(nrow):
        for j in range(ncol):
            point = Point(X[i, j], Y[i, j])
            if gdf.contains(point).any():
                mask[i, j] = True

    # Sample DEM values at active cell centroids
    elevation = np.full((nrow, ncol), np.nan)
    with rasterio.open(dem_path) as src:
        for i in range(nrow):
            for j in range(ncol):
                if mask[i, j]:
                    x_coord = X[i, j]
                    y_coord = Y[i, j]
                    try:
                        row, col = src.index(x_coord, y_coord)
                        if 0 <= row < src.height and 0 <= col < src.width:
                            val = dem_array[row, col]
                            if dem_nodata is not None and val == dem_nodata:
                                elevation[i, j] = np.nan
                            else:
                                elevation[i, j] = val
                    except IndexError:
                        elevation[i, j] = np.nan

    # Handle NoData values - use median of active area
    active_elevations = elevation[mask & ~np.isnan(elevation)]
    if len(active_elevations) > 0:
        median_elev = np.median(active_elevations)
        fill_count = np.sum(np.isnan(elevation) & mask)
        elevation[np.isnan(elevation) & mask] = median_elev
        print(f"Filled {fill_count} NoData cells with median elevation: {median_elev:.2f} m")
    else:
        elevation[np.isnan(elevation)] = 0.0
        print("WARNING: No valid elevation data found!")

    # Set inactive areas to a background value
    min_active_elev = np.min(elevation[mask]) if np.any(mask) else 0.0
    TOP_array = np.where(mask, elevation, min_active_elev)
    ibound_mask_2d = np.where(mask, 1, 0)

    print(f"Grid Dimensions: {nrow} rows, {ncol} columns")
    print(f"Active cells: {np.sum(mask)}")
    print(f"TOP array - Min: {np.min(TOP_array[mask]):.1f} m, Max: {np.max(TOP_array[mask]):.1f} m")

    return TOP_array, ibound_mask_2d, nrow, ncol, minx, miny, maxx, maxy

# Run Grid Generation
print("Generating model grid from DEM and shapefile...")
TOP_ARR, IBOUND_MASK_2D, NROW, NCOL, minx, miny, maxx, maxy = \
    assign_dem_to_modflow_top(SHP_PATH, DEM_PATH, DELR, DELC)

# Thickness enforcement
min_active_top = np.min(TOP_ARR[IBOUND_MASK_2D == 1]) if np.any(IBOUND_MASK_2D == 1) else BOTM_ELEVATION + MIN_THICKNESS

# Check if the fixed bottom elevation would result in a layer thinner than MIN_THICKNESS
if BOTM_ELEVATION >= min_active_top - MIN_THICKNESS:
    suggested_botm = min_active_top - MIN_THICKNESS
    print(f"\n[GEOMETRY WARNING] Adjusting BOTM from {BOTM_ELEVATION:.1f}m to {suggested_botm:.1f}m")
    BOTM = [suggested_botm] # Use this single adjusted value
else:
    BOTM = [BOTM_ELEVATION] # Use the original fixed bottom value

BOTM_SINGLE_VALUE = BOTM[0] # The single elevation value for BOTM

# ==============================================================================
# ------------------------ CALIBRATION TARGETS FUNCTIONS -----------------------
# ==============================================================================

def find_calibration_cells(calibration_targets, minx, miny, maxx, maxy, nrow, ncol, delr, delc):
    """
    Find which grid cells contain calibration targets
    Returns: list of (row, col, x, y, target_id) for each target
    """
    print("\n" + "="*60)
    print("LOCATING CALIBRATION TARGETS")
    print("="*60)
    
    calibration_cells = []
    
    for i, (x_target, y_target) in enumerate(calibration_targets):
        # Calculate row and column indices
        col = int((x_target - minx) / delr)
        row = int((maxy - y_target) / delc)  # Note: y-axis is flipped
        
        # Check if within bounds and active
        if 0 <= row < nrow and 0 <= col < ncol:
            if IBOUND_MASK_2D[row, col] != 0:
                 calibration_cells.append((row, col, x_target, y_target, i+1))
                 print(f"Target {i+1}: ({x_target:.1f}, {y_target:.1f}) → Cell ({row}, {col})")
            else:
                 print(f"Target {i+1}: ({x_target:.1f}, {y_target:.1f}) → INACTIVE CELL ({row}, {col})")
        else:
            print(f"Target {i+1}: ({x_target:.1f}, {y_target:.1f}) → OUTSIDE GRID BOUNDS")

    
    print(f"\n✓ Located {len(calibration_cells)} calibration targets within active model cells")
    return calibration_cells

def extract_calibration_heads(head, calibration_cells):
    """
    Extract simulated heads at calibration target locations
    """
    calibration_data = []
    
    for row, col, x, y, target_id in calibration_cells:
        simulated_head = head[0, row, col]
        if simulated_head < 1e20:  # Valid cell
            calibration_data.append({
                'target_id': target_id,
                'row': row,
                'col': col,
                'x': x,
                'y': y,
                'simulated_head': simulated_head,
                'land_surface': TOP_ARR[row, col],
                'status': 'VALID'
            })
        else:
            calibration_data.append({
                'target_id': target_id,
                'row': row,
                'col': col,
                'x': x,
                'y': y,
                'simulated_head': np.nan,
                'land_surface': TOP_ARR[row, col],
                'status': 'INACTIVE_CELL'
            })
    
    return calibration_data

def plot_calibration_targets(calibration_cells, head=None):
    """
    Plot calibration target locations on the model
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Targets on topography
    ax1 = axes[0]
    im = ax1.imshow(TOP_ARR, cmap='terrain', origin='upper', 
                    extent=[minx, maxx, miny, maxy], alpha=0.8)
    plt.colorbar(im, ax=ax1, label='Elevation (m)')
    
    # Plot calibration targets
    for row, col, x, y, target_id in calibration_cells:
        ax1.scatter(x, y, color='red', s=100, marker='*', edgecolors='white', linewidth=2)
        ax1.annotate(f'{target_id}', (x, y), xytext=(5, 5), textcoords='offset points',
                     fontweight='bold', fontsize=10, color='white',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
    
    ax1.set_title("Calibration Targets on Topography", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Targets on simulated heads (if available)
    ax2 = axes[1]
    if head is not None:
        head_plot = head[0].copy()
        head_plot[head_plot > 1e20] = np.nan
        im2 = ax2.imshow(head_plot, cmap='viridis', origin='upper',
                         extent=[minx, maxx, miny, maxy], alpha=0.8)
        plt.colorbar(im2, ax=ax2, label='Head (m)')
        plot_title = "Calibration Targets on Simulated Heads"
    else:
        # If no head data, just show topography
        im2 = ax2.imshow(TOP_ARR, cmap='terrain', origin='upper',
                         extent=[minx, maxx, miny, maxy], alpha=0.8)
        plt.colorbar(im2, ax=ax2, label='Elevation (m)')
        plot_title = "Calibration Targets (Heads Not Available)"
    
    # Plot calibration targets
    for row, col, x, y, target_id in calibration_cells:
        ax2.scatter(x, y, color='red', s=100, marker='*', edgecolors='white', linewidth=2)
        ax2.annotate(f'{target_id}', (x, y), xytext=(5, 5), textcoords='offset points',
                     fontweight='bold', fontsize=10, color='white',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
    
    ax2.set_title(plot_title, fontsize=14, fontweight='bold')
    ax2.set_xlabel("Easting (m)")
    ax2.set_ylabel("Northing (m)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(WORKSPACE, "calibration_targets.png"), dpi=300, bbox_inches='tight')
    plt.show()

def export_calibration_data(calibration_data, workspace):
    """
    Export calibration data to CSV file
    """
    import pandas as pd
    
    df = pd.DataFrame(calibration_data)
    csv_file = os.path.join(workspace, 'calibration_data.csv')
    df.to_csv(csv_file, index=False)
    
    print(f"✓ Calibration data exported to: {csv_file}")
    
    # Print summary
    valid_targets = [d for d in calibration_data if d['status'] == 'VALID']
    print(f"\nCalibration Summary:")
    print(f"  Total targets: {len(calibration_data)}")
    print(f"  Valid targets: {len(valid_targets)}")
    
    if valid_targets:
        heads = [d['simulated_head'] for d in valid_targets]
        print(f"  Head range at targets: {min(heads):.1f} to {max(heads):.1f} m")
        print(f"  Mean head at targets: {np.mean(heads):.1f} m")
    
    return csv_file

# ==============================================================================
# --------------------------- MODFLOW FUNCTIONS --------------------------------
# ==============================================================================

def resolve_mf2005_path(explicit_path: str) -> str:
    
    """Return a working path to mf2005."""
    p = shutil.which("mf2005")
    if p:
        return p
    if os.path.isfile(explicit_path):
        if os.name != "nt":
            os.chmod(explicit_path, 0o755)
        return explicit_path
    raise FileNotFoundError("MODFLOW-2005 executable not found.")

def build_model(model_name: str, dem_top_arr, botm_list, dem_ibound_mask_2d, nlay: int):
    exe = resolve_mf2005_path(EXE_MF2005)
    mf = flopy.modflow.Modflow(modelname=model_name, model_ws=WORKSPACE, exe_name=exe)

    # Discretization
    flopy.modflow.ModflowDis(
        mf, nlay=nlay, nrow=NROW, ncol=NCOL,
        delr=DELR, delc=DELC, top=dem_top_arr, botm=botm_list,
        nper=1, perlen=1.0, nstp=1, steady=True
    )

    # IBOUND (single layer)
    ibound = np.expand_dims(dem_ibound_mask_2d, axis=0)  # Shape: (1, NROW, NCOL)
    
    # Constant Head Boundary on southern edge
    chd_spd = []
    # Check only active cells on the southern-most row
    chd_cols = np.where(ibound[0, NROW - 1, :] == 1)[0]
    
    if len(chd_cols) > 0:
        for j in chd_cols:
            ibound[0, NROW - 1, j] = -1 # Set to Constant Head
            chd_spd.append([0, NROW - 1, j, CH_ELEV, CH_ELEV])
        print(f"Applying {len(chd_spd)} Constant Head cells (H={CH_ELEV}m) on southern boundary.")
    
    # Starting heads - use reasonable initial guess
    active_top = dem_top_arr[dem_ibound_mask_2d == 1]
    if len(active_top) > 0:
        initial_head = np.percentile(active_top, 30)  # Conservative initial head
    else:
        initial_head = CH_ELEV + 10.0
        
    strt = np.full((nlay, NROW, NCOL), initial_head, dtype=float)
    strt = np.where(ibound == 1, strt, 1e30) # Only set start head in active cells

    flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt, hnoflo=1e30)
    
    # Apply CHD package
    if len(chd_spd) > 0:
        flopy.modflow.ModflowChd(mf, stress_period_data={0: chd_spd}, ipakcb=CBC_UNIT)

    # LPF Package
    flopy.modflow.ModflowLpf(
        mf,
        hk=HK, vka=VKA, laytyp=LAYTYP, sy=SY, ss=SS,
        laywet=[1], wetfct=0.1, iwetit=1, ihdwet=0, wetdry=[1.0],
        ipakcb=CBC_UNIT
    )

    # Recharge
    rch_array = np.where(dem_ibound_mask_2d == 1, RCH_RATE, 0.0)
    flopy.modflow.ModflowRch(mf, rech=rch_array, ipakcb=CBC_UNIT)

    # Solver
    flopy.modflow.ModflowPcg(
        mf, hclose=0.01, rclose=1.0, mxiter=2000, iter1=500
    )
    
    # Output control
    spd = {(0, 0): ["save head", "save budget", "print head", "print budget"]}
    flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)

    return mf

def run_and_read(mf: "flopy.modflow.Modflow") -> tuple[np.ndarray, CellBudgetFile]:
    
    mf.write_input()
    ok, _ = mf.run_model(silent=False, report=True)
    if not ok:
        lst = os.path.join(mf.model_ws, f"{mf.name}.list")
        try:
            with open(lst, "r", errors="ignore") as f:
                print("\n" + "="*40 + "\nMODFLOW LIST FILE TAIL\n" + "="*40)
                print("".join(f.readlines()[-200:]))
        except Exception:
            pass
        raise RuntimeError(f"MODFLOW failed for {mf.name}")
    
    # Read Heads
    hds = HeadFile(os.path.join(mf.model_ws, f"{mf.name}.hds"))
    head = hds.get_data(kstpkper=(0, 0))
    
    # Read Cell-by-Cell Budget (CBC)
    cbc_file = os.path.join(mf.model_ws, f"{mf.name}.cbc")
    cbc = CellBudgetFile(cbc_file)
    
    return head, cbc

def calculate_flow_vectors(head: np.ndarray, hk: float, delr: float, delc: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Darcy velocity components (qx, qy) and magnitude using head gradient.
    """
    
    # Head array for layer 1 (shape NROW, NCOL)
    h_2d = head[0]
    
    # Head gradient in X direction
    dhdx = (h_2d[:, 1:] - h_2d[:, :-1]) / DELR
    
    # Head gradient in Y direction
    dhdy = (h_2d[1:, :] - h_2d[:-1, :]) / DELC
    
    # Calculate Darcy velocity components (qx, qy)
    qx = np.full((NROW, NCOL), np.nan)
    qy = np.full((NROW, NCOL), np.nan)
    
    # QX: Average gradient of faces
    qx[:, 1:-1] = -HK[0] * 0.5 * (dhdx[:, :-1] + dhdx[:, 1:])
    qx[:, 0] = -HK[0] * dhdx[:, 0]
    qx[:, -1] = -HK[0] * dhdx[:, -1]
    
    # QY: Average gradient of faces
    qy[1:-1, :] = -HK[0] * 0.5 * (dhdy[:-1, :] + dhdy[1:, :])
    qy[0, :] = -HK[0] * dhdy[0, :]
    qy[-1, :] = -HK[0] * dhdy[-1, :]
    
    # Filter inactive and dry cells
    dry_mask = (h_2d > 1e20) | (IBOUND_MASK_2D == 0)
    qx[dry_mask] = np.nan
    qy[dry_mask] = np.nan
    
    # Calculate magnitude
    qmag = np.sqrt(qx**2 + qy**2)
    
    return qx, qy, qmag

def plot_flow_lines(head, qx, qy, qmag, x_coords, y_coords, calibration_cells=None):
    """
    Draw groundwater flow lines using Darcy velocity vectors (streamlines).
    """
    print("\nPlotting flow lines (streamlines)...")

    head_plot = head[0].copy()
    head_plot[head_plot > 1e20] = np.nan

    # Meshgrid
    X, Y = np.meshgrid(x_coords, y_coords)

    plt.figure(figsize=(14, 10))

    # Background hydraulic head
    im = plt.imshow(head_plot, cmap="viridis", origin="upper",
                    extent=(minx, maxx, miny, maxy), alpha=0.8)
    plt.colorbar(im, label="Hydraulic Head (m)", shrink=0.8)

    # Scale linewidth by magnitude
    lw = 3.0 * (qmag / np.nanmax(qmag))
    lw[np.isnan(lw)] = 0.1

    # Streamlines
    plt.streamplot(
        X, Y,
        qx, qy,
        density=1.2,
        color='white',
        linewidth=lw,
        arrowsize=1.2
    )

    # Calibration targets
    if calibration_cells:
        for row, col, x, y, target_id in calibration_cells:
            plt.scatter(x, y, color='red', s=150, marker='*',
                        edgecolors='white', linewidth=2)
            plt.annotate(f'{target_id}', (x, y), xytext=(8, 8),
                         textcoords='offset points', color='white',
                         fontweight='bold', fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor='red', alpha=0.8))

    plt.title("Groundwater Flow Lines Derived from Darcy Velocity", fontsize=15, fontweight='bold')
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(WORKSPACE, "flow_lines_streamplot.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Flow lines saved: {out}")

def plot_head_distribution_with_vectors(head, qx, qy, qmag, x_coords, y_coords, calibration_cells=None):
    """
    Plot head distribution with flow vectors and calibration targets
    """
    print("\nPlotting head distribution with flow vectors...")
    
    head_plot = head[0].copy()
    head_plot[head_plot > 1e20] = np.nan
    
    plt.figure(figsize=(14, 10))
    
    # Head distribution
    im = plt.imshow(head_plot, cmap="viridis", origin="upper",
                    extent=(minx, maxx, miny, maxy), alpha=0.8)
    plt.colorbar(im, label="Hydraulic Head (m)", shrink=0.8)
    
    # Flow vectors (sparse for clarity)
    stream_density = 5
    x_vecs, y_vecs = x_coords[::stream_density], y_coords[::stream_density]
    qx_vecs = qx[::stream_density, ::stream_density]
    qy_vecs = qy[::stream_density, ::stream_density]
    qmag_vecs = qmag[::stream_density, ::stream_density]
    
    # Mask out vectors with low magnitude
    min_mag_plot = 1e-4
    qx_vecs[qmag_vecs < min_mag_plot] = 0.0
    qy_vecs[qmag_vecs < min_mag_plot] = 0.0

    # Plot vectors
    plt.quiver(x_vecs, y_vecs, qx_vecs, qy_vecs, 
               color='white', scale=5, scale_units='dots', 
               angles='xy', minshaft=2, headwidth=4, headlength=6,
               alpha=0.8)
    
    # Calibration targets
    if calibration_cells:
        for row, col, x, y, target_id in calibration_cells:
            if head[0, row, col] < 1e20:  # Valid cell
                plt.scatter(x, y, color='red', s=150, marker='*', 
                           edgecolors='white', linewidth=2)
                plt.annotate(f'{target_id}', (x, y), xytext=(8, 8), 
                           textcoords='offset points', color='white',
                           fontweight='bold', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", 
                                   facecolor='red', alpha=0.8))
    
    plt.title("Simulated Hydraulic Heads with Darcy Flow Vectors", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.tight_layout()
    
    out = os.path.join(WORKSPACE, "head_distribution_with_vectors.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Head distribution plot saved: {out}")

def run_model_and_process_results(model_name: str):
    # Find calibration target cells
    calibration_cells = find_calibration_cells(CALIBRATION_TARGETS, minx, miny, maxx, maxy, NROW, NCOL, DELR, DELC)
    
    # Plot model setup with calibration targets
    print("Plotting model setup with calibration targets...")
    plot_calibration_targets(calibration_cells)
    
    # Plot model setup
    print("Plotting model setup...")
    plot_model_setup()
    
    # Run model
    try:
        print("Building MODFLOW model...")
        model = build_model(model_name, dem_top_arr=TOP_ARR, botm_list=BOTM, 
                            dem_ibound_mask_2d=IBOUND_MASK_2D, nlay=NLAY)
        
        print("Running MODFLOW...")
        head, cbc = run_and_read(model) 
        print("✓ MODFLOW completed successfully\n")
    except RuntimeError as e:
        print(f"\n[CRITICAL ERROR] Model failed: {e}")
        return None, None, None, None, None
    
    # Extract calibration data
    calibration_data = extract_calibration_heads(head, calibration_cells)
    export_calibration_data(calibration_data, WORKSPACE)
    
    # Calculate flow vectors
    print("\nCalculating Darcy flow vectors from head gradients...")
    qx, qy, qmag = calculate_flow_vectors(head, HK[0], DELR, DELC)
    print("✓ Flow vectors calculated.")
    
    return head, calibration_cells, qx, qy, qmag

# ==============================================================================
# ---------------------------- PLOTTING FUNCTIONS ------------------------------
# ==============================================================================

def plot_model_setup():
    """Plot the model setup including grid and topography"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Grid with active cells
    ax1 = axes[0]
    try:
        gdf = gpd.read_file(SHP_PATH).to_crs(f'EPSG:{2213}')
    except:
        print("Warning: Could not reproject shapefile for plot. Using default CRS.")
        gdf = gpd.read_file(SHP_PATH)
        
    # Plot shapefile outline
    gdf.boundary.plot(ax=ax1, color='black', linewidth=2, label='Watershed boundary')
    
    # Plot grid cell outlines and highlight active cells
    active_count = 0
    for i in range(NROW):
        for j in range(NCOL):
            x0 = minx + j * DELR
            y0 = maxy - (i + 1) * DELC
            
            # Draw grid cell outline
            rect_outline = plt.Rectangle((x0, y0), DELR, DELC,
                                         edgecolor='gray', facecolor='none', linewidth=0.2)
            ax1.add_patch(rect_outline)
            
            if IBOUND_MASK_2D[i, j] == 1:
                # Highlight active cells
                rect_fill = plt.Rectangle((x0, y0), DELR, DELC,
                                          facecolor='blue', edgecolor='darkblue', 
                                          linewidth=0.3, alpha=0.6)
                ax1.add_patch(rect_fill)
                active_count += 1
    
    ax1.set_xlim(minx, maxx)
    ax1.set_ylim(miny, maxy)
    ax1.set_aspect('equal')
    ax1.set_title(f"Model Grid\n{active_count} active cells of {NROW*NCOL} total")
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot 2: Topography
    ax2 = axes[1]
    im = ax2.imshow(TOP_ARR, cmap='terrain', origin='upper', 
                    extent=[minx, maxx, miny, maxy])
    plt.colorbar(im, ax=ax2, label='Elevation (m)')
    gdf.boundary.plot(ax=ax2, color='red', linewidth=2)
    ax2.set_title("Land Surface Topography")
    ax2.set_xlabel("Easting (m)")
    ax2.set_ylabel("Northing (m)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(WORKSPACE, "model_setup.png"), dpi=300, bbox_inches='tight')
    plt.show()

# ==============================================================================
# ---------------------------------- MAIN RUN ----------------------------------
# ==============================================================================

def main():
    print(f"\n{'='*60}")
    print(f"MODFLOW Single-Layer Model: Rio Cobre Aquifer")
    print(f"{'='*60}\n")
    print(f"Layer Bottom Elevation: {BOTM_SINGLE_VALUE} m")
    
    # Execute model run and initial processing
    head, calibration_cells, qx, qy, qmag = run_model_and_process_results("mf2005_riocobre")
    
    if head is None:
        return
    
    # Create coordinate arrays
    x_coords = np.arange(NCOL) * DELR + minx + DELR/2
    y_coords = np.arange(NROW) * DELC + miny + DELC/2
    
    # Generate plots
    plot_flow_lines(head, qx, qy, qmag, x_coords, y_coords, calibration_cells)
    plot_head_distribution_with_vectors(head, qx, qy, qmag, x_coords, y_coords, calibration_cells)
    
    # Summary
    print(f"\n{'='*60}")
    print("MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"Grid: {NROW} rows × {NCOL} columns × {NLAY} layer")
    print(f"Cell size: {DELR} m × {DELC} m")
    print(f"Active cells: {np.sum(IBOUND_MASK_2D == 1)}")
    print(f"Layer Bottom: {BOTM_SINGLE_VALUE} m")
    
    print(f"\nAquifer properties: K={HK[0]} m/d, Sy={SY[0]}")
    
    valid_heads = head[head < 1e20]
    if len(valid_heads) > 0:
        print(f"Head range: {np.min(valid_heads):.1f} to {np.max(valid_heads):.1f} m")
    
    # Flow Vector Summary
    valid_qmag = qmag[~np.isnan(qmag)]
    if len(valid_qmag) > 0:
        print(f"\nDarcy Velocity (Active Cells):")
        print(f"  Range: {np.min(valid_qmag):.4f} to {np.max(valid_qmag):.4f} m/d")
        print(f"  Mean: {np.mean(valid_qmag):.4f} m/d")
    
    print(f"\n✓ MODEL COMPLETE")
    print(f"Results in: {WORKSPACE}")

if __name__ == "__main__":
    main()