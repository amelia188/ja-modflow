# -*- coding: utf-8 -*-
"""
aquifer_3d.py  (version 2)
--------------------------
Interactive 3D model of the Upper Rio Cobre hydrogeology, built on the v21 MODFLOW grid.

What you see
  * Five hydrostratigraphic units as smooth solids clipped to the real basin outline:
      valley alluvium · red marine / valley clay · White Limestone (karst aquifer)
      · Yellow Limestone (basal aquitard) · volcanic basement
  * The SATURATED ZONE: wherever rock lies below the simulated water table it is tinted
    blue, so every cut shows how much of each aquifer is full of water
  * The water table (v21 heads, any month), its trace on the cut face, flowlines coloured
    by seepage velocity, the 13 calibration wells (observed vs simulated head)
  * Context: river network derived from the DEM, towns, optional faults and imagery,
    a 5 km scale bar and a true-elevation ruler
  * A movable slicing plane, an exploded view, layer toggles, preset views

Run it next to run_model.py, from any folder, after the model (and optionally the export)
      python aquifer_3d.py                          # interactive window
      python aquifer_3d.py --month 9 --ve 20
      python aquifer_3d.py --screenshot fig.png --view ns --explode 0.6
      python aquifer_3d.py --export-vtk aquifer_vtk # ParaView files, true elevations
  Keys in the window: 1 overview · 2 north-south cut · 3 west-east cut · 4 map view ·
                      e toggle exploded view · s save screenshot
  Needs: pyvista >= 0.38, scipy, geopandas, rasterio, pyproj (all but pyvista come with v21).

------------------------------------------------------------------------------------------
LAYER GEOMETRY: rules now, real surfaces when you trace them
------------------------------------------------------------------------------------------
The DEM gives the land surface. Below it, each surface comes from (in order of preference)
  1. a GeoTIFF you supply in SURFACE_FILES, or
  2. contours you digitise from Wishart (2001) and list in CONTOUR_FILES, or
  3. a schematic rule (numbers from Wishart 2001, Table 4), clearly labelled on screen.

How to digitise Wishart's maps in QGIS (about 15 minutes each)
  * Export the figure page as PNG (Fig. 3.5 White Limestone top, 3.6 White Limestone
    thickness, 3.10 alluvium thickness).
  * Raster > Georeferencer: add 4+ points at the axis tick marks. The axes are the JAD69
    Jamaica National Grid, EPSG:24200 (e.g. Bog Walk is about 249 250 E, 161 000 N).
  * New line layer (EPSG:24200), field "value" (real). Trace each contour line that
    crosses the Upper Rio Cobre and type its value. Save as a shapefile or GeoPackage.
  * Put the path in CONTOUR_FILES below. The script reprojects to the model grid,
    interpolates between contours, and uses the result instead of the rule.
"""

import os
import sys
import json
import heapq
import argparse
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_model as m   # v22 settings and helpers (no side effects on import)

try:
    import pyvista as pv
    import vtk
except ModuleNotFoundError as e:
    raise SystemExit("PyVista is not installed. Install with: pip install pyvista") from e
pv.global_theme.allow_empty_mesh = True

# ==============================================================================
# CONFIGURATION
# ==============================================================================

VE_DEFAULT    = 15          # vertical exaggeration
MONTH_DEFAULT = 12          # month of the final cycle for heads
MESH_SPACING  = 250.0       # m between mesh vertices (smaller = smoother, slower)

# --- 1. Real surfaces as GeoTIFF (elevation, m asl, any CRS) ------------------
SURFACE_FILES = {
    "alluvium_base": None,
    "clay_base":     None,
    "wl_base":       None,
    "yl_base":       None,
}

# --- 2. Digitised contours: (path, value_field). Any CRS (Wishart = EPSG:24200) --
CONTOUR_FILES = {
    "wl_top":             None,   # Fig 3.5 altitude of top of White Limestone, m asl
    "wl_thickness":       None,   # Fig 3.6 thickness of White Limestone, m
    "alluvium_thickness": None,   # Fig 3.10 thickness of alluvium, m (coastal; 0 where absent)
}

# --- 3. Schematic rules (sources in brackets) ----------------------------------
VALLEY_FILL    = True
VALLEY_RELIEF  = 40.0    # m above the lowest active cell that counts as valley floor
ALLUVIUM_MAX   = 30.0    # m  [productive alluvium = upper 20-30 m, Wishart p.62]
CLAY_MAX       = 12.0    # m  [red marine / valley clay <2-24 m, K ~ 1e-6 m/d, p.62]
WL_BASE_NORTH  = 50.0    # m asl at the north edge  [WL modelled 400 m asl to -250 m, thickens south, p.45]
WL_BASE_SOUTH  = -250.0  # m asl at the south edge
WL_MIN_THICK   = 30.0    # m
YL_THICK       = 60.0    # m  [Yellow Limestone basal aquitard 21-152 m, Table 4]
FLOOR_BELOW_YL = 150.0   # m of basement shown below the deepest Yellow Limestone

# --- Context layers -------------------------------------------------------------
IMAGERY_FILE   = None    # optional RGB GeoTIFF (e.g. exported from QGIS) draped on the land
FAULTS_FILE    = None    # optional line shapefile of faults (any CRS)
RIVER_MIN_KM2  = 4.0     # drainage area that counts as a channel
TOWNS = [                # name, latitude, longitude (approximate town centres)
    ("Spanish Town", 17.991, -76.957),
    ("Bog Walk",     18.102, -77.006),
    ("Linstead",     18.137, -77.032),
    ("Ewarton",      18.183, -77.087),
]

# --- Look -------------------------------------------------------------------------
UNITS = [  # name, colour, aquifer? (gets saturated-zone tint), role
    ("Valley alluvium",          "#e3d08f", True,  "aquifer"),
    ("Red marine / valley clay", "#a8452f", False, "aquitard"),
    ("White Limestone",          "#e4e8ec", True,  "karst aquifer"),
    ("Yellow Limestone",         "#b5762a", False, "basal aquitard"),
    ("Volcanic basement",        "#5d5266", False, "impermeable"),
]
WATER = "#1f6fb5"
TINT_STRENGTH = 0.55       # how blue the saturated rock becomes
BG_BOTTOM, BG_TOP = "#eef2f5", "#c9d8e4"
INK = "#14202b"
USE_SSAO = False           # soft ambient occlusion: try True on a good GPU; it can add blotches
FLOW_STRIDE = 6

# ==============================================================================
# SMALL HELPERS
# ==============================================================================

def hex_rgb(h):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def blend(c1, c2, t):
    return tuple((1 - t) * hex_rgb(c1) + t * hex_rgb(c2))


def fill_nan_nearest(a):
    from scipy.ndimage import distance_transform_edt
    mask = np.isnan(a)
    if not mask.any():
        return a
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def smooth(a, passes=3):
    a = a.copy()
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        st = np.stack([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a = np.where(np.isnan(a), np.nan, np.nanmean(st, axis=0))
    return a


class Grid:
    pass


def build_grid():
    """The grid exactly as the model saved it (grid.npz); rebuilt from the DEM only if missing."""
    G = Grid()
    try:
        g = m.load_grid()
        G.top, G.nrow, G.ncol, G.minx, G.maxy, G.crs_wkt = g["top"], g["nrow"], g["ncol"], g["minx"], g["maxy"], g["crs_wkt"]
        G.active, G.xc, G.yc = g["active"], g["xc"], g["yc"]
    except FileNotFoundError:
        print("grid.npz not found (model not run yet): building the grid from the DEM; geology only.")
        top, act, nrow, ncol, minx, maxy, crs_wkt = m.assign_dem_to_modflow_top(m.SHP_PATH, m.DEM_PATH, m.DELR, m.DELC)
        G.top, G.nrow, G.ncol, G.minx, G.maxy, G.crs_wkt = top, nrow, ncol, minx, maxy, crs_wkt
        G.active = act != 0
        G.xc = minx + m.DELR * (np.arange(ncol) + 0.5)
        G.yc = maxy - m.DELC * (np.arange(nrow) + 0.5)
    return G


def sampler(G, A):
    """Bilinear sampler for a cell-centred grid; NaNs filled from the nearest valid cell."""
    from scipy.interpolate import RegularGridInterpolator
    F = fill_nan_nearest(np.asarray(A, float))
    f = RegularGridInterpolator((G.yc[::-1], G.xc), F[::-1], bounds_error=False, fill_value=None)
    return lambda x, y: f(np.column_stack([np.asarray(y), np.asarray(x)]))

# ==============================================================================
# SURFACES
# ==============================================================================

def raster_to_grid(path, G):
    import rasterio
    from rasterio.warp import transform as warp_transform
    with rasterio.open(path) as src:
        X, Y = np.meshgrid(G.xc, G.yc)
        xs, ys = X.ravel(), Y.ravel()
        if src.crs and src.crs.to_wkt() != G.crs_wkt:
            xs, ys = warp_transform(G.crs_wkt, src.crs, xs, ys)
        v = np.array([s[0] for s in src.sample(zip(xs, ys))], float)
        if src.nodata is not None:
            v[v == src.nodata] = np.nan
    return v.reshape(G.nrow, G.ncol)


def contours_to_grid(spec, G):
    """Digitised contour lines -> smooth surface on the model grid."""
    import geopandas as gpd
    from scipy.interpolate import griddata
    path, field = spec
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(24200)                    # Wishart's maps: JAD69 National Grid
    gdf = gdf.to_crs(G.crs_wkt)
    xs, ys, vs = [], [], []
    for geom, val in zip(gdf.geometry, gdf[field].astype(float)):
        parts = getattr(geom, "geoms", [geom])
        for g in parts:
            n = max(int(g.length // 100), 2)
            for d in np.linspace(0, g.length, n):
                p = g.interpolate(d); xs.append(p.x); ys.append(p.y); vs.append(val)
    X, Y = np.meshgrid(G.xc, G.yc)
    lin = griddata((xs, ys), vs, (X, Y), method="linear")
    near = griddata((xs, ys), vs, (X, Y), method="nearest")
    out = np.where(np.isnan(lin), near, lin)
    return smooth(out, passes=4)


def build_surfaces(G):
    """Six bounding surfaces (cell-centred, NaN outside), top to bottom, plus provenance notes."""
    act = G.active
    top = np.where(act, G.top, np.nan)
    notes = {"Land surface": "DEM"}

    # alluvium + clay
    if CONTOUR_FILES["alluvium_thickness"]:
        t_all = np.clip(contours_to_grid(CONTOUR_FILES["alluvium_thickness"], G), 0, None)
        notes["Alluvium base"] = "digitised contours"
    elif VALLEY_FILL:
        zmin = np.nanmin(top)
        frac = smooth(np.where(act, np.clip((zmin + VALLEY_RELIEF - top) / VALLEY_RELIEF, 0, 1), np.nan))
        t_all = ALLUVIUM_MAX * np.sqrt(np.nan_to_num(frac))
        notes["Alluvium base"] = "rule: valley fill"
    else:
        t_all = np.zeros_like(top); notes["Alluvium base"] = "none"
    t_cly = np.where(t_all > 1.0, CLAY_MAX * np.sqrt(np.clip(t_all / max(ALLUVIUM_MAX, 1), 0, 1)), 0.0)

    s1 = top - t_all
    if SURFACE_FILES["alluvium_base"]:
        s1 = np.minimum(top, raster_to_grid(SURFACE_FILES["alluvium_base"], G)); notes["Alluvium base"] = "GeoTIFF"
    s2 = s1 - t_cly
    notes["Clay base"] = "rule: under alluvium"
    if CONTOUR_FILES["wl_top"]:
        # Limestone top from contours. Where it lies below the land, the cover is clay at the
        # base (up to CLAY_MAX) with alluvium above it; where it reaches the land, limestone outcrops.
        # Cover is only allowed where alluvium is known to exist (valley rule or traced alluvium map);
        # elsewhere the limestone outcrops and its top is the land surface.
        has_cover = t_all > 1.0
        wl_top = np.where(has_cover, np.minimum(top, contours_to_grid(CONTOUR_FILES["wl_top"], G)), top)
        cover = np.clip(top - wl_top, 0, None)
        clay = np.where(cover > 1.0, np.minimum(cover, CLAY_MAX), 0.0)
        s1 = top - (cover - clay)
        s2 = wl_top
        notes["Alluvium base"] = "digitised WL-top contours (cover minus basal clay)"
        notes["Clay base"] = "digitised WL-top contours"
    if SURFACE_FILES["clay_base"]:
        s2 = np.minimum(s1, raster_to_grid(SURFACE_FILES["clay_base"], G)); notes["Clay base"] = "GeoTIFF"

    if CONTOUR_FILES["wl_thickness"]:
        s3 = s2 - np.clip(contours_to_grid(CONTOUR_FILES["wl_thickness"], G), WL_MIN_THICK, None)
        notes["White Limestone base"] = "digitised thickness contours"
    else:
        yfrac = (G.maxy - G.yc) / (G.nrow * m.DELC)
        s3 = np.repeat((WL_BASE_NORTH + (WL_BASE_SOUTH - WL_BASE_NORTH) * yfrac)[:, None], G.ncol, axis=1)
        notes["White Limestone base"] = f"rule: dips {WL_BASE_NORTH:+.0f} m (N) to {WL_BASE_SOUTH:+.0f} m (S)"
    if SURFACE_FILES["wl_base"]:
        s3 = raster_to_grid(SURFACE_FILES["wl_base"], G); notes["White Limestone base"] = "GeoTIFF"
    s3 = np.minimum(s3, s2 - WL_MIN_THICK)

    s4 = s3 - YL_THICK
    notes["Yellow Limestone base"] = f"rule: {YL_THICK:.0f} m below WL base"
    if SURFACE_FILES["yl_base"]:
        s4 = np.minimum(s3, raster_to_grid(SURFACE_FILES["yl_base"], G)); notes["Yellow Limestone base"] = "GeoTIFF"

    floor = np.nanmin(np.where(act, s4, np.nan)) - FLOOR_BELOW_YL
    s5 = np.full_like(top, floor)
    S = [np.where(act, s, np.nan) for s in (top, s1, s2, s3, s4, s5)]
    return S, notes, floor


def schematic(notes):
    return any(v.startswith("rule") for k, v in notes.items() if k != "Land surface")

# ==============================================================================
# MESHES
# ==============================================================================

def basin_polygon(G):
    import geopandas as gpd
    gdf = gpd.read_file(m.SHP_PATH).to_crs(G.crs_wkt)
    try:
        poly = gdf.union_all()
    except AttributeError:
        poly = gdf.unary_union
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly.buffer(0).simplify(30)


def triangulate(poly, spacing):
    """Triangle mesh filling the basin: densified outline + regular interior points."""
    import shapely
    from scipy.spatial import Delaunay
    ring = poly.exterior
    nb = max(int(ring.length // (spacing * 0.6)), 50)
    bpts = np.array([ring.interpolate(d).coords[0] for d in np.linspace(0, ring.length, nb, endpoint=False)])
    minx, miny, maxx, maxy = poly.bounds
    gx, gy = np.meshgrid(np.arange(minx, maxx, spacing), np.arange(miny, maxy, spacing))
    gx, gy = gx.ravel(), gy.ravel()
    inner = poly.buffer(-spacing * 0.45)
    keep = shapely.contains_xy(inner, gx, gy)
    pts = np.vstack([bpts, np.column_stack([gx[keep], gy[keep]])])
    tri = Delaunay(pts).simplices
    c = pts[tri].mean(axis=1)
    tri = tri[shapely.contains_xy(poly, c[:, 0], c[:, 1])]
    return pts, tri, len(bpts)


def unit_solid(pts, tri, zt, zb, zrel_t=None, zrel_b=None, min_thick=0.3):
    """Wedge (prism) mesh between two surfaces. Clipping it gives true cut faces."""
    n = len(pts)
    P = np.vstack([np.column_stack([pts, zb]), np.column_stack([pts, zt])])
    thick = (zt - zb)[tri].max(axis=1)
    t = tri[thick > min_thick]
    if len(t) == 0:
        return None
    cells = np.column_stack([np.full(len(t), 6), t, t + n]).ravel()
    ug = pv.UnstructuredGrid(cells, np.full(len(t), vtk.VTK_WEDGE, np.uint8), P)
    ug.point_data["elev_m"] = P[:, 2]
    if zrel_t is not None:
        ug.point_data["above_wt_m"] = np.concatenate([zrel_b, zrel_t])
    return ug.clean()


def surface_mesh(pts, tri, z):
    faces = np.column_stack([np.full(len(tri), 3), tri]).ravel()
    return pv.PolyData(np.column_stack([pts, z]), faces)


def read_heads(month):
    from flopy.utils import HeadFile
    path = os.path.join(m.WORKSPACE, f"{m.MODEL_NAME}.hds")
    return m.get_heads(HeadFile(path), m.final_cycle_sp(month)) if os.path.exists(path) else None


def drainage(G):
    """D8 river network from the DEM: priority-flood pit filling, then flow accumulation."""
    z = np.where(G.active, G.top, np.inf).astype(float)
    nr, nc = z.shape
    filled = z.copy()
    done = ~G.active.copy()
    pq = []
    for i in range(nr):
        for j in range(nc):
            if G.active[i, j]:
                edge = (i in (0, nr - 1) or j in (0, nc - 1) or
                        not G.active[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].all())
                if edge:
                    heapq.heappush(pq, (z[i, j], i, j)); done[i, j] = True
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while pq:
        h, i, j = heapq.heappop(pq)
        for di, dj in nb:
            a, b = i + di, j + dj
            if 0 <= a < nr and 0 <= b < nc and not done[a, b]:
                done[a, b] = True
                filled[a, b] = max(z[a, b], h + 1e-3)
                heapq.heappush(pq, (filled[a, b], a, b))
    # D8 receivers
    rec = -np.ones((nr, nc, 2), int)
    for i in range(nr):
        for j in range(nc):
            if not G.active[i, j]:
                continue
            best, bi = 0.0, None
            for di, dj in nb:
                a, b = i + di, j + dj
                if 0 <= a < nr and 0 <= b < nc and G.active[a, b]:
                    s = (filled[i, j] - filled[a, b]) / (1.4142 if di and dj else 1.0)
                    if s > best:
                        best, bi = s, (a, b)
            if bi:
                rec[i, j] = bi
    order = np.argsort(-np.where(G.active, filled, -np.inf), axis=None)
    acc = np.where(G.active, 1.0, 0.0)
    for k in order:
        i, j = divmod(int(k), nc)
        if not G.active[i, j]:
            continue
        a, b = rec[i, j]
        if a >= 0:
            acc[a, b] += acc[i, j]
    return rec, acc


def river_mesh(G, land_at):
    rec, acc = drainage(G)
    thr = RIVER_MIN_KM2 * 1e6 / (m.DELR * m.DELC)
    pts, lines, width = [], [], []
    for i, j in zip(*np.where((acc >= thr) & G.active)):
        a, b = rec[i, j]
        if a < 0:
            continue
        x0, y0, x1, y1 = G.xc[j], G.yc[i], G.xc[b], G.yc[a]
        xs, ys = np.linspace(x0, x1, 5), np.linspace(y0, y1, 5)       # follow the surface between cells
        zs = land_at(xs, ys) + 12.0
        n = len(pts)
        pts += list(zip(xs, ys, zs))
        lines += [5] + list(range(n, n + 5))
        width += [np.log10(acc[i, j] * m.DELR * m.DELC / 1e6)] * 5
    if not pts:
        return None
    pd = pv.PolyData(np.array(pts)); pd.lines = np.array(lines)
    pd.point_data["log_area"] = np.array(width)
    return pd


def lines_from_file(path, G, land_at, offset=6.0):
    import geopandas as gpd
    gdf = gpd.read_file(path).to_crs(G.crs_wkt)
    pts, lines, n = [], [], 0
    for geom in gdf.geometry:
        for g in getattr(geom, "geoms", [geom]):
            seg = g.segmentize(150) if hasattr(g, "segmentize") else g
            c = np.array(seg.coords)[:, :2]
            z = land_at(c[:, 0], c[:, 1]) + offset
            pts.append(np.column_stack([c, z])); lines.append(np.r_[len(c), np.arange(n, n + len(c))]); n += len(c)
    if not pts:
        return None
    pd = pv.PolyData(np.vstack(pts)); pd.lines = np.concatenate(lines)
    return pd


def flowlines_mesh(G, head_at, month):
    path = os.path.join(m.WORKSPACE, "hydrowindy_data", f"flowlines_{month:02d}.json")
    if not os.path.exists(path):
        return None
    from pyproj import Transformer
    to_xy = Transformer.from_crs("EPSG:4326", G.crs_wkt, always_xy=True)
    pts, lines, vel, n = [], [], [], 0
    for ft in json.load(open(path))["features"][::FLOW_STRIDE]:
        c = np.array(ft["geometry"]["coordinates"])
        if len(c) < 2:
            continue
        x, y = to_xy.transform(c[:, 0], c[:, 1])
        z = head_at(x, y) + 3.0
        pts.append(np.column_stack([x, y, z])); lines.append(np.r_[len(c), np.arange(n, n + len(c))])
        vel += [ft["properties"].get("v", np.nan)] * len(c); n += len(c)
    if not pts:
        return None
    pd = pv.PolyData(np.vstack(pts)); pd.lines = np.concatenate(lines)
    pd.point_data["velocity_m_per_day"] = np.array(vel, float)
    return pd


def imagery_texture(path, G):
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(path) as src:
        img = np.dstack([src.read(b) for b in (1, 2, 3)])
        b = transform_bounds(src.crs, G.crs_wkt, *src.bounds)
    if img.dtype != np.uint8:
        img = np.clip(img / np.nanpercentile(img, 99) * 255, 0, 255).astype(np.uint8)
    return pv.Texture(np.flipud(img)), b

# ==============================================================================
# SCENE
# ==============================================================================

GROUP = {  # which unit each overlay moves with in the exploded view
    "water": 2, "wt_line": 2, "flow": 2, "well_casing": 2, "well_obs": 2, "well_sim": 2,
    "rivers": 0, "faults": 0, "towns": 0, "imagery": 0,
}


class Scene:
    def __init__(self, month=MONTH_DEFAULT, ve=VE_DEFAULT, explode=0.0, off_screen=False):
        self.G = G = build_grid()
        self.S, self.notes, self.floor = build_surfaces(G)
        self.poly = basin_polygon(G)
        self.pts, self.tri, self.nb = triangulate(self.poly, MESH_SPACING)
        self.ve, self.explode = float(ve), float(explode)
        self.cells = m.find_calibration_cells(m.CALIBRATION_TARGETS, G.minx, G.maxy, G.nrow, G.ncol,
                                              G.active.astype(int), verbose=False)
        self.land_at = sampler(G, self.S[0])
        self.Z = [sampler(G, s)(self.pts[:, 0], self.pts[:, 1]) for s in self.S]
        for k in range(1, len(self.Z)):                    # interpolation must not cross
            self.Z[k] = np.minimum(self.Z[k], self.Z[k - 1])
        self.zrange = (self.floor, float(np.nanmax(self.Z[0])))
        self.plane = None
        self.objs, self.style, self.display, self.actors = {}, {}, {}, {}
        self.p = pv.Plotter(off_screen=off_screen, window_size=(1600, 1000), lighting="none")
        self._static_context()
        self.set_month(month, build_only=True)

    # ---------------- data ----------------
    def _static_context(self):
        G = self.G
        self.objs["rivers"] = river_mesh(G, self.land_at)
        if FAULTS_FILE:
            self.objs["faults"] = lines_from_file(FAULTS_FILE, G, self.land_at)
        from pyproj import Transformer
        import shapely
        to_xy = Transformer.from_crs("EPSG:4326", G.crs_wkt, always_xy=True)
        self.towns = []
        for name, lat, lon in TOWNS:
            x, y = to_xy.transform(lon, lat)
            if shapely.contains_xy(self.poly, x, y):
                self.towns.append((name, x, y, float(self.land_at([x], [y])[0])))
        if self.towns:
            self.objs["towns"] = pv.PolyData(np.array([(x, y, z + 8) for _, x, y, z in self.towns]))
        if IMAGERY_FILE:
            tex, b = imagery_texture(IMAGERY_FILE, G)
            surf = surface_mesh(self.pts, self.tri, self.Z[0] + 1.0)
            surf.active_texture_coordinates = np.column_stack([(self.pts[:, 0] - b[0]) / (b[2] - b[0]),
                                                               (self.pts[:, 1] - b[1]) / (b[3] - b[1])])
            self.objs["imagery"] = surf; self.texture = tex

    def set_month(self, month, build_only=False):
        self.month = month
        self.head = read_heads(month)
        G, pts = self.G, self.pts
        if self.head is not None:
            self.head_at = sampler(G, np.where(G.active, self.head, np.nan))
            H = self.head_at(pts[:, 0], pts[:, 1])
        else:
            self.head_at, H = None, None
        for k, (name, color, aquifer, _) in enumerate(UNITS):
            zt, zb = self.Z[k], self.Z[k + 1]
            rel = (zt - H, zb - H) if (H is not None and aquifer) else (None, None)
            self.objs[f"unit{k}"] = unit_solid(pts, self.tri, zt, zb, *rel)
        if H is not None:
            self.objs["water"] = surface_mesh(pts, self.tri, H)
            self.objs["flow"] = flowlines_mesh(G, self.head_at, month)
        casing, obs, sim, self.labels = [], [], [], []
        for row, col, x, y, tid, ob in self.cells:
            land = float(self.land_at([x], [y])[0])
            si = float(self.head_at([x], [y])[0]) if self.head_at else np.nan
            casing.append(pv.Line((x, y, land + 10), (x, y, min(ob, si if np.isfinite(si) else ob) - 30)))
            obs.append((x, y, ob))
            if np.isfinite(si):
                sim.append((x, y, si))
            self.labels.append((f"Well {tid}", (x, y, land + 25), 2))
        self.labels += [(n, (x, y, z + 40), 5) for n, x, y, z in self.towns]
        self.objs["well_casing"] = casing[0].merge(casing[1:]) if casing else None
        self.objs["well_obs"] = pv.PolyData(np.array(obs)) if obs else None
        self.objs["well_sim"] = pv.PolyData(np.array(sim)) if sim else None
        if not build_only:
            self.refresh(); self._set_title()

    # ---------------- transform + clip ----------------
    def _offset(self, key):
        if key.startswith("unit"):
            k = int(key[4:])
        else:
            k = GROUP.get(key, 0)
        gap = 0.12 * (self.zrange[1] - self.zrange[0]) * self.ve
        return (len(UNITS) - 1 - k) * gap * self.explode

    def _display_mesh(self, key, mesh):
        out = mesh.scale([1.0, 1.0, self.ve], inplace=False)
        dz = self._offset(key)
        if dz:
            out = out.translate((0, 0, dz), inplace=False)
        if self.plane is not None and key not in ("towns",):
            n, o = self.plane
            out = out.clip(normal=n, origin=o, invert=True)
        return out

    def refresh(self):
        for key, mesh in self.objs.items():
            if mesh is None:
                continue
            d = self._display_mesh(key, mesh)
            if key in self.display:
                self.display[key].copy_from(d)
            else:
                self.display[key] = d
        self._update_wt_line()
        self._update_labels()
        self._update_ruler()
        self.p.render()

    def _update_wt_line(self):
        line = pv.PolyData()
        if self.plane is not None and self.objs.get("water") is not None:
            n, o = self.plane
            w = self.objs["water"].scale([1, 1, self.ve], inplace=False).translate((0, 0, self._offset("water")), inplace=False)
            line = w.slice(normal=n, origin=o)
            if line.n_points:
                nn = np.array(n, float); nn /= np.linalg.norm(nn)
                line = line.translate(nn * 60.0, inplace=False)
        if "wt_line" in self.display:
            self.display["wt_line"].copy_from(line)
        else:
            self.display["wt_line"] = line

    # ---------------- labels with collision avoidance ----------------
    def _update_labels(self):
        pts = []
        for text, (x, y, z), pri in self.labels:
            key = "towns" if not text.startswith("Well") else "well_casing"
            pts.append((x, y, z * self.ve + self._offset(key)))
        if not hasattr(self, "_label_pd"):
            self._label_pd = pv.PolyData(np.array(pts) if pts else np.zeros((0, 3)))
            arr = vtk.vtkStringArray(); arr.SetName("text")
            pr = vtk.vtkIntArray(); pr.SetName("priority")
            for text, _, pri in self.labels:
                arr.InsertNextValue(text); pr.InsertNextValue(pri)
            self._label_pd.GetPointData().AddArray(arr); self._label_pd.GetPointData().AddArray(pr)
        else:
            self._label_pd.points = np.array(pts)
            self._label_pd.Modified()

    def _add_label_actor(self):
        hier = vtk.vtkPointSetToLabelHierarchy()
        hier.SetInputData(self._label_pd)
        hier.SetLabelArrayName("text"); hier.SetPriorityArrayName("priority")
        tp = hier.GetTextProperty()
        tp.SetFontSize(14); tp.SetColor(*hex_rgb(INK)); tp.SetBold(False); tp.SetFontFamilyToArial()
        mapper = vtk.vtkLabelPlacementMapper()
        mapper.SetInputConnection(hier.GetOutputPort())
        mapper.SetShapeToRoundedRect(); mapper.SetStyleToFilled()
        mapper.SetBackgroundColor(1, 1, 1); mapper.SetBackgroundOpacity(0.85); mapper.SetMargin(4)
        actor = vtk.vtkActor2D(); actor.SetMapper(mapper)
        self.p.add_actor(actor, name="labels"); self.actors["labels"] = actor

    # ---------------- ruler + scale bar ----------------
    def _ruler_geometry(self):
        b = self.poly.bounds
        x0, y0 = b[2] + 2500, b[1] - 1500          # south-east corner, beside the usual cut faces
        zlo, zhi = self.zrange
        step = 100 if (zhi - zlo) < 1200 else 200
        ticks = np.arange(np.ceil(zlo / step) * step, zhi + 1, step)
        ve = self.ve
        segs = [((x0, y0, zlo * ve), (x0, y0, zhi * ve))]
        segs += [((x0, y0, t * ve), (x0 + 600, y0, t * ve)) for t in ticks]
        segs += [((x0, y0 + 2000, zlo * ve), (x0, y0 + 7000, zlo * ve))]                    # 5 km bar, N-S
        segs += [((x0, y, zlo * ve - 40 * ve), (x0, y, zlo * ve + 40 * ve)) for y in (y0 + 2000, y0 + 7000)]
        P, L = [], []
        for a, c in segs:
            L += [2, len(P), len(P) + 1]; P += [a, c]
        pd = pv.PolyData(np.array(P, float)); pd.lines = np.array(L)
        labels = [(f"{int(t)} m", (x0 + 800, y0, t * ve)) for t in ticks]
        labels += [("5 km", (x0, y0 + 4500, zlo * ve - 90 * ve))]
        return pd, labels

    def _update_ruler(self):
        pd, labels = self._ruler_geometry()
        if "ruler" in self.display:
            self.display["ruler"].copy_from(pd)
        else:
            self.display["ruler"] = pd
        self._ruler_labels = labels
        if "ruler_text" in self.actors:
            self.p.remove_actor(self.actors["ruler_text"])
            self.actors["ruler_text"] = self.p.add_point_labels(
                np.array([p for _, p in labels]), [t for t, _ in labels], font_size=11, point_size=0,
                show_points=False, shape=None, text_color=INK, always_visible=True, name="ruler_text")

    # ---------------- build window ----------------
    def build(self):
        p = self.p
        p.set_background(BG_BOTTOM, top=BG_TOP)
        self.refresh()
        mat = dict(smooth_shading=True, split_sharp_edges=True, feature_angle=35,
                   ambient=0.28, diffuse=0.75, specular=0.08, specular_power=12)
        for k, (name, color, aquifer, _) in enumerate(UNITS):
            key = f"unit{k}"
            if key not in self.display:
                continue
            mesh = self.display[key]
            if aquifer and "above_wt_m" in mesh.point_data:
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap([blend(color, WATER, TINT_STRENGTH), hex_rgb(color)])
                self.actors[key] = p.add_mesh(mesh, scalars="above_wt_m", cmap=cmap, clim=[-0.5, 0.5],
                                              interpolate_before_map=True, show_scalar_bar=False, name=key, **mat)
            else:
                self.actors[key] = p.add_mesh(mesh, color=color, name=key, **mat)
            # thin outline on unit boundaries makes the cut read like a drawn section
        if "imagery" in self.display:
            self.actors["imagery"] = p.add_mesh(self.display["imagery"], texture=self.texture, name="imagery",
                                                smooth_shading=True, split_sharp_edges=True, ambient=0.35)
        if "water" in self.display:
            self.actors["water"] = p.add_mesh(self.display["water"], color=WATER, opacity=0.35,
                                              smooth_shading=True, split_sharp_edges=True, specular=0.4, name="water")
        self.actors["wt_line"] = p.add_mesh(self.display["wt_line"], color=WATER, line_width=7,
                                            render_lines_as_tubes=True, name="wt_line")
        if self.display.get("flow") is not None:
            self.actors["flow"] = p.add_mesh(self.display["flow"], scalars="velocity_m_per_day", cmap="viridis",
                                             log_scale=True, line_width=2.5, render_lines_as_tubes=True,
                                             scalar_bar_args=dict(title="Flow velocity (m/day)\n", vertical=False,
                                                                  position_x=0.74, position_y=0.86, width=0.22,
                                                                  height=0.06, color=INK, fmt="%.2g", n_labels=4,
                                                                  title_font_size=12, label_font_size=10),
                                             name="flow")
        if self.display.get("rivers") is not None:
            self.actors["rivers"] = p.add_mesh(self.display["rivers"], color="#2b7bbd", line_width=3,
                                               lighting=False, name="rivers")
        if self.display.get("faults") is not None:
            self.actors["faults"] = p.add_mesh(self.display["faults"], color="#c0392b", line_width=3, name="faults")
        if self.display.get("towns") is not None:
            self.actors["towns"] = p.add_mesh(self.display["towns"], color=INK, point_size=11,
                                              render_points_as_spheres=True, name="towns")
        if self.display.get("well_casing") is not None:
            self.actors["well_casing"] = p.add_mesh(self.display["well_casing"], color=INK, line_width=3,
                                                    render_lines_as_tubes=True, name="well_casing")
            self.actors["well_obs"] = p.add_mesh(self.display["well_obs"], color="#e8a33d", point_size=15,
                                                 render_points_as_spheres=True, name="well_obs")
            if self.display.get("well_sim") is not None:
                self.actors["well_sim"] = p.add_mesh(self.display["well_sim"], color=WATER, point_size=15,
                                                     render_points_as_spheres=True, name="well_sim")
        self.actors["ruler"] = p.add_mesh(self.display["ruler"], color=INK, line_width=2, name="ruler")
        self.actors["ruler_text"] = p.add_point_labels(
            np.array([q for _, q in self._ruler_labels]), [t for t, _ in self._ruler_labels], font_size=11,
            point_size=0, show_points=False, shape=None, text_color=INK, always_visible=True, name="ruler_text")
        self._add_label_actor()
        self._lights()
        if USE_SSAO:
            try:
                p.enable_ssao(radius=0.02 * self._span(), bias=0.5, kernel_size=128)
            except Exception:
                pass
        p.enable_anti_aliasing("ssaa")
        self._set_title()
        if schematic(self.notes):
            p.add_text("Layers below the land surface are partly SCHEMATIC: "
                       + "; ".join(f"{k}: {v}" for k, v in self.notes.items() if v.startswith("rule")),
                       position="lower_left", font_size=8, color="#566573", name="caveat")
        self._add_toggles()
        self.view("overview")

    def _span(self):
        b = self.poly.bounds
        return max(b[2] - b[0], b[3] - b[1])

    def _lights(self):
        p = self.p
        b = self.poly.bounds
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        span = self._span()
        ztop = self.zrange[1] * self.ve
        key = pv.Light(position=(cx - 0.8 * span, cy + 1.0 * span, ztop + 1.2 * span),
                       focal_point=(cx, cy, 0), intensity=0.85, light_type="scene light")
        fill = pv.Light(position=(cx + span, cy - span, ztop + 0.4 * span), focal_point=(cx, cy, 0),
                        intensity=0.30, light_type="scene light")
        head = pv.Light(light_type="headlight", intensity=0.25)
        for L in (key, fill, head):
            p.add_light(L)

    def _set_title(self):
        mon = m.MONTH_LABELS[self.month - 1]
        extra = "" if self.head is not None else "  (no .hds found: geology only)"
        self.p.add_text(f"Upper Rio Cobre aquifer system  ·  water table end of {mon}  ·  VE ×{self.ve:g}{extra}",
                        position="upper_left", font_size=13, color=INK, name="title")

    def _add_toggles(self):
        p = self.p
        rows = [(f"unit{k}", f"{n}  ({r})", c) for k, (n, c, _, r) in enumerate(UNITS)]
        rows += [("water", "Water table  (blue rock = saturated)", WATER),
                 ("flow", "Flowlines, coloured by velocity", "#35b779"),
                 ("wells", "Wells: orange = observed, blue = simulated head", "#e8a33d"),
                 ("rivers", "Rivers (from DEM)", "#2b7bbd"),
                 ("towns", "Towns and labels", INK),
                 ("faults", "Faults", "#c0392b"), ("imagery", "Imagery", "#7a8f6a")]
        H = p.window_size[1]
        y = H - 78
        for key, label, color in rows:
            keys = {"wells": ["well_casing", "well_obs", "well_sim"], "water": ["water", "wt_line"],
                    "towns": ["towns", "labels"]}.get(key, [key])
            if not any(k in self.actors for k in keys):
                continue

            def cb(state, keys=keys):
                for k in keys:
                    if k in self.actors:
                        self.actors[k].SetVisibility(state)
                p.render()
            p.add_checkbox_button_widget(cb, value=True, position=(16, y), size=20, border_size=2,
                                         color_on=color, color_off="#c8d0d8", background_color=INK)
            p.add_text(label, position=(46, y + 2), font_size=10, color=INK, name=f"lbl_{key}")
            y -= 30

    # ---------------- views ----------------
    def set_plane(self, normal, origin=None):
        b = self.poly.bounds
        if origin is None:
            origin = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 0.0)
        self.plane = (tuple(normal), tuple(origin)) if normal is not None else None
        self.refresh()
        if getattr(self, "plane_widget", None) is not None and normal is not None:
            self.plane_widget.SetNormal(*normal); self.plane_widget.SetOrigin(*origin)

    def view(self, which):
        p = self.p
        b = self.poly.bounds
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        cz = 0.5 * (self.zrange[0] + self.zrange[1]) * self.ve + self._offset("unit2")
        span = self._span()
        if which == "overview":
            eye = (cx + 0.55 * span, cy - 0.95 * span, cz + 0.75 * span)
        elif which == "ns":            # cut face (x-normal) seen level from the east: reads like a drawn section
            eye = (cx + 1.5 * span, cy, cz)
        elif which == "we":            # cut face (y-normal) seen level from the north
            eye = (cx, cy + 1.5 * span, cz)
        else:                          # map view, north up
            eye = (cx, cy, cz + 2.0 * span)
        up = (0, 1, 0) if which == "map" else (0, 0, 1)
        p.camera_position = [eye, (cx, cy, cz), up]
        if which == "overview":
            p.disable_parallel_projection()
        else:                          # sections and map read like drawings: no perspective
            p.enable_parallel_projection()
        p.reset_camera(); p.camera.zoom({"overview": 1.25, "map": 1.15}.get(which, 1.85))
        if which in ("ns", "we", "map"):   # slide the drawing right, clear of the tick-box panel
            shift = np.array({"ns": (0, -1, 0), "we": (1, 0, 0), "map": (-1, 0, 0)}[which], float) * 0.10 * span
            cam = p.camera
            cam.position = tuple(np.array(cam.position) + shift)
            cam.focal_point = tuple(np.array(cam.focal_point) + shift)
        p.render()

    # ---------------- interaction ----------------
    def add_widgets(self):
        p = self.p

        def on_plane(normal, origin):
            self.plane = (tuple(normal), tuple(origin))
            self.refresh()

        b = self.poly.bounds
        origin = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 0.0)
        self.plane_widget = p.add_plane_widget(on_plane, normal="x", origin=origin, color="#0b5a8a",
                                               outline_translation=False, implicit=True,
                                               interaction_event="end", outline_opacity=0.25)
        if self.plane is None:
            on_plane((1.0, 0.0, 0.0), origin)

        def on_ve(v):
            self.ve = float(round(v)); self.refresh(); self._set_title()
            self.plane_widget.PlaceWidget(p.bounds); p.reset_camera(); p.render()

        def on_explode(v):
            self.explode = float(v); self.refresh(); self.plane_widget.PlaceWidget(p.bounds); p.render()

        sl = dict(interaction_event="end", style="modern", color=INK, title_height=0.017)
        p.add_slider_widget(on_ve, [1, 50], value=self.ve, title="Vertical exaggeration", fmt="%.0f",
                            pointa=(0.30, 0.06), pointb=(0.44, 0.06), **sl)
        p.add_slider_widget(on_explode, [0, 1], value=self.explode, title="Exploded view", fmt="%.2f",
                            pointa=(0.48, 0.06), pointb=(0.62, 0.06), **sl)
        if self.head is not None:
            p.add_slider_widget(lambda v: (int(round(v)) != self.month) and self.set_month(int(round(v))),
                                [1, 12], value=self.month, title="Month", fmt="%.0f",
                                pointa=(0.66, 0.13), pointb=(0.80, 0.13), **sl)

        p.add_key_event("1", lambda: self.view("overview"))
        p.add_key_event("2", lambda: (self.set_plane((1.0, 0.0, 0.0)), self.view("ns")))
        p.add_key_event("3", lambda: (self.set_plane((0.0, -1.0, 0.0)), self.view("we")))
        p.add_key_event("4", lambda: self.view("map"))
        p.add_key_event("e", lambda: on_explode(0.0 if self.explode > 0 else 0.6))
        p.add_key_event("s", lambda: (p.screenshot("aquifer_3d_view.png"), print("saved aquifer_3d_view.png")))
        p.add_text("drag plane to cut · 1 overview · 2 N–S cut · 3 W–E cut · 4 map · e explode · s save",
                   position=(int(0.62 * p.window_size[0]), p.window_size[1] - 30), font_size=9,
                   color="#566573", name="help")

# ==============================================================================
# MAIN
# ==============================================================================

def report(sc):
    names = [u[0] for u in UNITS]
    print("\nUnit thickness on the mesh (m):")
    for k, n in enumerate(names):
        t = sc.Z[k] - sc.Z[k + 1]
        print(f"  {n:<26} max {np.nanmax(t):7.1f}   present over {100 * (t > 0.5).mean():5.1f}% of the basin")
    print("\nSurface sources:")
    for k, v in sc.notes.items():
        print(f"  {k:<22} {v}")


def export_vtk(sc, folder):
    os.makedirs(folder, exist_ok=True)
    for key, mesh in sc.objs.items():
        if mesh is None:
            continue
        ext = ".vtu" if isinstance(mesh, pv.UnstructuredGrid) else ".vtp"
        name = UNITS[int(key[4:])][0].split()[0].lower() if key.startswith("unit") else key
        mesh.save(os.path.join(folder, f"{key}_{name}{ext}"))
    print(f"✓ VTK files written to {folder} (true elevations; open in ParaView)")


def main():
    ap = argparse.ArgumentParser(description="3D model of the Upper Rio Cobre aquifer system")
    ap.add_argument("--month", type=int, default=MONTH_DEFAULT)
    ap.add_argument("--ve", type=float, default=VE_DEFAULT)
    ap.add_argument("--explode", type=float, default=0.0)
    ap.add_argument("--view", choices=["overview", "ns", "we", "map"], default="ns")
    ap.add_argument("--screenshot", help="render without a window and save to this PNG")
    ap.add_argument("--export-vtk", help="write ParaView files to this folder and exit")
    a = ap.parse_args()

    sc = Scene(a.month, a.ve, a.explode, off_screen=bool(a.screenshot))
    report(sc)
    if a.export_vtk:
        export_vtk(sc, a.export_vtk); return
    if a.view == "ns":
        sc.plane = ((1.0, 0.0, 0.0), ((sc.poly.bounds[0] + sc.poly.bounds[2]) / 2, 0.0, 0.0))
    elif a.view == "we":
        sc.plane = ((0.0, -1.0, 0.0), (0.0, (sc.poly.bounds[1] + sc.poly.bounds[3]) / 2, 0.0))
    sc.build()
    sc.view(a.view)
    if a.screenshot:
        sc.p.screenshot(a.screenshot); print(f"✓ Saved {a.screenshot}"); return
    sc.add_widgets()
    sc.p.show()


if __name__ == "__main__":
    main()
