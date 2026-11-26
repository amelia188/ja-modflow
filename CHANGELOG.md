# Upper Rio Cobre groundwater model: changelog

Model versions are Git tags (`git tag`, `git diff v21 v22`). The model lives at one path,
`modflowandfiles/run_model.py`. Each run writes `runs/<model>/run_config.json` recording the settings,
input checksums, software versions and Git commit that produced it.

## v22 (November 2026): MODFLOW 6 port

- **Engine:** MODFLOW 6 (`flopy.mf6`), Newton-Raphson formulation for unconfined drying and rewetting.
  Output control saves heads and budgets at the **LAST** step of each period by name.
- **Outputs:** `runs/<MODEL_NAME>/` beside the scripts, built from `BASE_DIR`, so they no longer depend on
  the folder Python is started from.
- **Run record:** `run_config.json` (settings, SHA-256 of DEM, shapefile and recharge, Git commit, Python,
  FloPy and MODFLOW versions, calibration results).
- **Grid:** `grid.npz` saved with every run (grid, georeference, IDOMAIN, final IBOUND, CHD cells). The
  export and the 3D viewer read it instead of rebuilding the grid and repeating the CHD rule.
- **Flow:** cell-centred specific discharge from MODFLOW 6 (`DATA-SPDIS`). It uses each cell's own
  conductance, so it is correct with zoned K and defined at basin-edge cells, which were NaN in v21.
- **Code tidy:** one definition of the calibration statistics, used by the CSV, the plots and the export.
  The duplicate flowline tracer was removed. `SHOW_PLOTS` switch added.
- **Bug fix inherited from v20/v21:** LPF rewetting (`laywet=1`, `wetdry=1.0` in every cell) activated all
  cells *outside* the basin (4,777 cells in the test grid). MODFLOW-2005 therefore simulated flow across
  the full rectangle.

- **Validation scripts** (`validation/`): residual analysis, water budget, parameter plausibility and
  split-sample tests, reading the v22 run folder directly. Observations can be dated; dates are compared
  with the same day of the final cycle. `run_model.py` now labels its mean error as simulated − observed
  (the validation reports use observed − simulated).

### Verification of the port (synthetic Rio Cobre-style basin, 104 x 232 cells, 37 periods)

| Comparison | Mean head difference | Max |
|---|---|---|
| Analytical Dupuit strip: MODFLOW-2005 vs exact | 0.000 m | 0.000 m |
| Analytical Dupuit strip: MODFLOW 6 (Newton) vs exact | 0.004 m RMS | 0.005 m |
| Steady state, v21 settings vs MODFLOW 6 | +1.33 m | 2.17 m |
| Steady state, rewetting fixed vs MODFLOW 6 (standard formulation) | 0.000 m | 0.000 m |
| Transient, v21 as run vs v22, every saved month | +1.38 m | 2.27 m |
| Transient, v21 with rewetting fixed vs v22 | +0.07 m | 0.10 m |

The remaining ~7 cm comes from the Newton formulation (upstream-weighted saturated thickness) and v21's
looser solver tolerances (`hclose=0.01`, `rclose=10`). Water budget discrepancy in v22: under 0.02% in
every period.

**Consequence:** v20 and v21 calibration results, including the move to K = 200 m/d, were obtained with
the rewetting bug. Recalibrate on v22 before verification.

To rerun v21 correctly for comparison, change its LPF line to
`wetdry=np.where(ibound[0] != 0, 1.0, 0.0)`.

## v21 (October 2026)

End-of-month heads (output control at the last step), 3 transient cycles with a convergence check,
no-flow northern boundary, exact cell-centre georeferencing, observed heads with RMSE/MAE/R²,
vectorised grid build, HydroWindy flowline export.

## v20 (April 2026)

1 steady-state spin-up + 12 transient monthly periods, IDW recharge x 0.17 (Curtis 2025, Table 4.8),
K = 50 m/d, CHD 100 m, 30 time steps per month.

## v19

MODFLOW-2005, one steady-state period, uniform recharge 1e-4 m/d, K = 20 m/d, CHD 5 m.
