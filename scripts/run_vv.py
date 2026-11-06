"""Verification & validation runs. Writes results/vv_results.json and results/*.npz.

    python scripts/run_vv.py            # full set (several minutes)
    python scripts/run_vv.py --quick    # coarse meshes only
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from simplefvm import SimpleSolver, ghia  # noqa: E402

OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

quick = "--quick" in sys.argv
dc = "--dc" in sys.argv                      # second-order convection by deferred correction
gamma = 1.0 if dc else 0.0
tag = "_dc" if dc else ""
cases = [(100, 32), (100, 64), (100, 128), (400, 128), (1000, 128)]
if quick:
    cases = [(100, 32), (100, 64)]
elif not dc:
    cases += [(100, 256)]
only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
if only:                                       # e.g. --only=100x128,400x128
    keep = {tuple(map(int, c.split("x"))) for c in only[0].split(",")}
    cases = [c for c in cases if c in keep]

json_path = os.path.join(OUT, f"vv_results{tag}.json")
results = json.load(open(json_path)) if (only and os.path.exists(json_path)) else []
results = [r for r in results if (r["Re"], r["M"]) not in {tuple(c) for c in cases}]
for Re, M in cases:
    print(f"Re {Re}, {M}x{M}")
    s = SimpleSolver(M=M, Re=Re, gamma_dc=gamma)
    s.run(max_outer=40000, tol=1e-6, verbose=2000)
    y, u = s.centerline_u()
    x, v = s.centerline_v()
    err = {"ghia_" + k: v_ for k, v_ in ghia.compare(s, Re).items()}
    rec = {"Re": Re, "M": M, "scheme": "central (deferred correction)" if dc else "upwind",
           "converged": s.converged, "iterations": s.iterations,
           "wall_time_s": round(s.wall_time, 2), "max_cell_divergence": s.max_divergence(),
           "u_min": float(u.min()), "y_at_u_min": float(y[np.argmin(u)]),
           "v_peak_max": float(v.max()), "v_peak_min": float(v.min()), **err}
    print("   ", rec)
    results.append(rec)
    np.savez_compressed(os.path.join(OUT, f"cavity_Re{Re}_M{M}{tag}.npz"), X=s.X, Y=s.Y,
                        X_cor=s.X_cor, Y_cor=s.Y_cor, U=s.fields.U, V=s.fields.V, P=s.fields.P,
                        psi=s.stream_function(), res_u=np.array(s.history["u"]),
                        res_v=np.array(s.history["v"]), res_m=np.array(s.history["mass"]))
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=1)
print("done")
