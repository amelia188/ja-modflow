"""Check the validation scripts against a finished Rio Cobre run (run_model.py v22+).

    python tests/test_riocobre.py            # uses run_model.py's WORKSPACE

1. The 13 Curtis (2025) targets at end of December must give exactly the statistics that
   run_model.py wrote to calibration_results.csv (same cells, same heads, sign flipped).
2. Dated observations are mapped to the same day of the final cycle, and a well outside the
   basin is reported, not silently dropped.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mfval  # noqa: E402

m = mfval.find_run_model()
assert m is not None, "run_model.py not found (set RIOCOBRE_MODEL_DIR)"
sim, gwf = mfval.load_model()
ws = Path(gwf.model_ws)

# 1 ---- December targets reproduce run_model.py
tg = pd.DataFrame([{"well_id": f"T{d['id']:02d}", "month": 12, "obs_head": d["obs_head"], "x": d["x"], "y": d["y"]}
                   for d in m.CALIBRATION_TARGETS])
tg.to_csv(ws / "_test_targets.csv", index=False)
df = mfval.simulated_at_obs(gwf, mfval.load_obs(ws / "_test_targets.csv", gwf, sim))
ref = pd.read_csv(ws / "calibration_results.csv")
mine = df.set_index(df.well_id.str[1:].astype(int))["sim_head"].sort_index()
theirs = ref.set_index("target_id")["simulated_head"].sort_index()
assert np.allclose(mine.values, theirs.values, atol=1e-6, equal_nan=True), (mine, theirs)
st = m.calibration_stats(theirs.values, ref.set_index("target_id")["observed_head"].sort_index().values)
mt = mfval.fit_metrics(df)
assert abs(mt["rmse_m"] - st["rmse"]) < 1e-9 and abs(mt["mean_error_m"] + st["me"]) < 1e-9
print(f"1. December targets match run_model.py: RMSE {mt['rmse_m']:.3f} m, ME {mt['mean_error_m']:+.3f} m (obs - sim)")

# 2 ---- dates -> final cycle; outside well flagged
cal = mfval.model_calendar(sim)
t = mfval.date_to_model_time(pd.Series(pd.to_datetime(["2003-12-31", "1999-01-01", "2016-02-29"])), cal)
end = cal["total"]
assert np.isclose(t[0], end) and np.isclose(t[1], end - 364) and np.isclose(t[2], end - 365 + 59), t
d2 = pd.DataFrame([{"well_id": "IN", "date": "2010-06-15", "obs_head": 77, "x": tg.x[0], "y": tg.y[0]},
                   {"well_id": "OUT", "date": "2010-06-15", "obs_head": 77, "x": 1.0, "y": 1.0}])
d2.to_csv(ws / "_test_dated.csv", index=False)
r2 = mfval.simulated_at_obs(gwf, mfval.load_obs(ws / "_test_dated.csv", gwf, sim))
assert np.isfinite(r2.sim_head[0]) and np.isnan(r2.sim_head[1]) and r2.outside_grid_or_inactive[1]
print("2. Dates map to the final cycle; the outside well is reported:", mfval.data_problems(r2)[0])
for f in ("_test_targets.csv", "_test_dated.csv"):
    (ws / f).unlink()
print("All checks passed.")
