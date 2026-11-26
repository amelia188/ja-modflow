# Validation scripts (MODFLOW 6, Upper Rio Cobre v22+)

Scripts for checking that a calibrated MODFLOW 6 model is not "garbage", for the verification and
validation steps of the NASA systems-engineering process. They work with any MODFLOW 6 model and
know the Rio Cobre model: put this `validation/` folder next to `modflowandfiles/` and the
simulation folder (`modflowandfiles/runs/riocobre_v22`), the mf6 executable and the calendar are
taken from `run_model.py`.

| Script | Validation checks |
| --- | --- |
| `residual_analysis.py` | 1. parameter plausibility, 2. residual analysis, 4. water budget |
| `split_sample.py` | 3. prediction of held-out data (by year, or leave-wells-out) |

```
ja-modflow/
├── modflowandfiles/run_model.py       → writes runs/riocobre_v22/
└── validation/                        ← these scripts
    ├── mfval.py  residual_analysis.py  split_sample.py
    ├── plausible_ranges_riocobre.csv
    └── tests/  make_test_model.py  test_riocobre.py
```

## Install

```bash
pip install flopy pandas numpy matplotlib scipy
pip install pyemu          # optional, only if you calibrate with PEST/PEST++
```

## Your observation file

One row per measurement. Location as model coordinates (JAD2001 metres, as in
`CALIBRATION_TARGETS`) or a model cell; time as a calendar date, a month, or model time:

```csv
well_id,date,obs_head,x,y
BW-03,2014-03-12,77.41,749538.5,661301.9
BW-03,2014-09-30,78.02,749538.5,661301.9
```

| Column | Meaning |
| --- | --- |
| `well_id`, `obs_head` | required; head in metres above datum |
| `x, y` [+ `layer`] or `layer, row, col` | location (1-based layer/row/col) |
| `date` | **recommended.** The model repeats an average year 3 times; a date is compared with the same day of the year in the final cycle, so readings from 2012 and 2019 both meet the model's typical year |
| `month` | 1–12, compared with the end of that month (what `run_model.py` does for its December targets) |
| `time` | model days from the start of the simulation (the steady-state spin-up is the first 365 days) |
| `obsnme` | optional PEST observation name |

Measurements that fall in the spin-up cycles, outside the basin, or in cells the model has dry are
listed under **Data problems** in the report rather than silently dropped.

**Sign convention:** residual = observed − simulated, so positive means the model is too low.
`run_model.py` prints simulated − observed: the same numbers with the opposite sign (now labelled).

## 1. Residual analysis (start here)

```bash
cd validation
python residual_analysis.py --obs ../data/observations.csv --ranges plausible_ranges_riocobre.csv
```

Writes `runs/riocobre_v22/validation/residual_analysis/report.md` (or `--out`): verdicts in plain
language, statistics, data problems, worst wells, the final-period water budget, the parameter
check and up to five figures (simulated vs observed, residual map, residuals over time, residual
distribution, mean residual by month). The report header records the model version and the Git
commit from `run_config.json`, so every report traces back to the exact code that produced it.

`plausible_ranges_riocobre.csv`: the K range for karst limestone is from Freeze & Cherry (1979).
**The Ss and Sy rows are general placeholders**: replace them with site values and sources before
reporting.

## 2. Split-sample test (the real test of predictive skill)

```bash
# by year: the earliest 70% of the record calibrates, the latest 30% tests
python split_sample.py split --obs ../data/observations.csv --mode temporal --train-frac 0.7 --out splits

# recalibrate on the training data only (your PEST setup, or the quick built-in option)
python split_sample.py recalibrate --obs splits/temporal/train.csv --out runs/temporal --recharge

# evaluate on both sets
python split_sample.py evaluate --sim-ws runs/temporal \
    --train splits/temporal/train.csv --test splits/temporal/test.csv --out results/temporal
```

For spatial checks use `--mode wells --k 5`, run recalibrate and evaluate per fold, then
`python split_sample.py summarise --results "results/fold_*"`. With only the 13 December targets
(one reading per well), the leave-wells-out test is the only split possible.

The built-in `recalibrate` fits multipliers on K (per chosen layer, K33 scaled with it) and on
recharge (RCHA) with Nelder–Mead, starting from the current calibration with first steps of × 1.26
(`--step 0.1`, log10 units) and keeping multipliers within × 1/10 to × 10 (`--max-factor`). One Rio
Cobre run takes about 35 s, so the default `--max-runs 60` is up to about 35 minutes; the progress
line shows the time left. The recalibrated run keeps `grid.npz`, so the HydroWindy export and the
3D viewer can read it. It is a quick check, not a replacement for a full PEST calibration.

**Reading the result:** test/train RMSE ratio ≤ 1.5 means good predictive skill; 1.5–2.5 is
moderate; above 2.5 the model fits its calibration data much better than new data.

## Tested

- `tests/test_riocobre.py` (needs a finished v22 run): the December targets reproduce
  `run_model.py`'s calibration exactly (same cells and heads; RMSE 2.885 m on the synthetic test
  basin), dates map to the right day of the final cycle (29 February included), and a well outside
  the basin is reported.
- `tests/make_test_model.py` builds a synthetic two-layer transient aquifer (a "truth" model and a
  deliberately mis-calibrated one). The residual analysis flags the bias and the drifting wells.
  Recalibrating on 70% of the record converges in 36 runs to K × 1.27 and recharge × 0.92, against
  true values of × 1.25 and × 0.91 (train RMSE 0.048 m, held-out 0.055 m, ratio 1.14). K and
  recharge trade off, so always check parameter plausibility too.
