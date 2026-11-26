"""Synthetic 2-layer transient aquifer: a 'truth' model makes observations; a slightly wrong
'calibrated' model is what the validation scripts examine."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, flopy

here = Path(__file__).parent
_local = here.parent / "bin" / "mf6"
exe = str(_local.resolve()) if _local.exists() else "mf6"   # else use mf6 on your PATH
nlay, nrow, ncol, d = 2, 30, 30, 100.0
nper = 36

def build(ws, k1, k2, rch_scale):
    sim = flopy.mf6.MFSimulation(sim_name="aq", sim_ws=str(ws), exe_name=exe)
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=[(30.0, 1, 1.0)] * nper, time_units="days")
    flopy.mf6.ModflowIms(sim, complexity="simple", outer_dvclose=1e-6, inner_dvclose=1e-6)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="aq", save_flows=True)
    top = 60 + 0.4 * np.tile(np.arange(ncol), (nrow, 1))      # rises inland (east)
    flopy.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol, delr=d, delc=d,
                            top=top, botm=[np.full((nrow, ncol), 20.0), np.full((nrow, ncol), -20.0)])
    flopy.mf6.ModflowGwfnpf(gwf, k=[k1, k2], icelltype=[1, 0])
    flopy.mf6.ModflowGwfsto(gwf, sy=0.12, ss=1e-5, iconvert=[1, 0], transient={0: True})
    flopy.mf6.ModflowGwfic(gwf, strt=40.0)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(l, r, 0), 30.0] for l in range(nlay) for r in range(nrow)])  # river on west edge
    rch = {p: rch_scale * (0.0012 + 0.0010 * np.sin(2 * np.pi * p / 12)) * np.ones((nrow, ncol)) for p in range(nper)}
    flopy.mf6.ModflowGwfrcha(gwf, recharge=rch)
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data={p: [[(1, 15, 18), -1500.0 if p >= 12 else 0.0]] for p in range(nper)})
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="aq.hds", budget_filerecord="aq.cbc",
                           saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")], printrecord=[("BUDGET", "ALL")])
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True)
    assert ok, f"model failed in {ws}"
    return gwf

truth = build(here / "truth", 15.0, 5.0, 1.0)
calib = build(here / "calibrated", 12.0, 5.0, 1.1)      # K a bit low, recharge a bit high

rng = np.random.default_rng(3)
wells = [(f"W{n+1:02d}", 1 if n % 3 else 2, int(r), int(c))
         for n, (r, c) in enumerate(zip(rng.integers(2, 28, 12), rng.integers(3, 28, 12)))]
hds = truth.output.head()
rows = []
for w, lay, r, c in wells:
    ts = hds.get_ts((lay - 1, r, c))
    for t, h in ts:
        rows.append({"well_id": w, "time": t, "obs_head": round(h + rng.normal(0, 0.05), 3),
                     "layer": lay, "row": r + 1, "col": c + 1, "obsnme": f"{w.lower()}_{int(t)}"})
pd.DataFrame(rows).to_csv(here / "obs.csv", index=False)
pd.DataFrame([{"param": "k", "layer": 1, "min": 5, "max": 50}, {"param": "k", "layer": 2, "min": 1, "max": 20},
              {"param": "sy", "layer": 1, "min": 0.05, "max": 0.3}]).to_csv(here / "plausible_ranges.csv", index=False)
print("built truth + calibrated models,", len(rows), "observations")
