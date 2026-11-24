"""Full de Vahl Davis Table V runs with tight tolerance.

Writes results/dvd_boussinesq.json incrementally (after each run).
WARNING: long runs — Ra=1e6 M=128 can take hours. Total ~6-8 hours.

    python scripts/run_dvd.py
"""
import sys, os, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

from simplefvm import NaturalConvectionSolver

DVD_REF = {
    1e3: dict(u_max=3.649, y_u=0.813, v_max=3.697, x_v=0.178, Nu=1.117),
    1e4: dict(u_max=16.178, y_u=0.823, v_max=19.617, x_v=0.119, Nu=2.238),
    1e5: dict(u_max=34.73, y_u=0.855, v_max=68.59, x_v=0.066, Nu=4.509),
    1e6: dict(u_max=64.63, y_u=0.850, v_max=219.36, x_v=0.0379, Nu=8.817),
}

MESHES = {1e3: (32, 64, 128), 1e4: (32, 64, 128), 1e5: (32, 64, 128), 1e6: (32, 64, 128)}

JSON_PATH = os.path.join(OUT, "dvd_boussinesq.json")

# load existing results
if os.path.exists(JSON_PATH):
    results = json.load(open(JSON_PATH))
else:
    results = {}
existing = set(results.keys())

print("run_dvd.py — estimated total: ~6-8 hours")
print("Writing to", JSON_PATH)
print("=" * 70)

for Ra in (1e3, 1e4, 1e5, 1e6):
    for M in MESHES[Ra]:
        key = f"Ra_{int(Ra):05d}_M{M}"
        if key in existing:
            print(f"  SKIP {key} (already done)")
            continue

        alpha_T = 1.0
        if Ra >= 1e6 and M >= 128:
            alpha_T = 0.8

        t0 = time.perf_counter()
        try:
            s = NaturalConvectionSolver(M=M, Ra=Ra, Pr=0.71,
                                         alpha_u=0.7, alpha_p=0.3,
                                         alpha_T=alpha_T)
            s.run(max_outer=50000, tol=1e-8, verbose=1000)
        except Exception as e:
            print(f"  {key}: ERROR {e}")
            continue
        dt = round(time.perf_counter() - t0, 1)

        Nu_h = float(np.sum(s.energy.wall_heat_flux("W", s.fields, s.T,
                                                      s.Gamma_T, cp=1.0)[0]))
        Nu_c = float(np.sum(s.energy.wall_heat_flux("E", s.fields, s.T,
                                                      s.Gamma_T, cp=1.0)[0]))
        u_max, y_u = s.max_u_mid()
        v_max, x_v = s.max_v_mid()
        psi_mid, psi_max = s.psi_mid()
        ref = DVD_REF[Ra]

        def pct(v, r):
            return float(abs(v - r) / abs(r) * 100)

        rec = dict(Ra=Ra, M=M, converged=s.converged, iterations=s.iterations,
                   wall_time_s=dt,
                   Nu_h=Nu_h, Nu_c=Nu_c,
                   Nu_bal=abs(abs(Nu_h) - abs(Nu_c)),
                   u_max=u_max, y_u=y_u, v_max=v_max, x_v=x_v,
                   psi_mid=psi_mid, psi_max=psi_max,
                   u_err_pct=pct(u_max, ref["u_max"]),
                   v_err_pct=pct(v_max, ref["v_max"]),
                   Nu_err_pct=pct(abs(Nu_h), ref["Nu"]),
                   max_div=s.max_divergence())

        results[key] = rec
        with open(JSON_PATH, "w") as fh:
            json.dump(results, fh, indent=1)

        print(f"  {key}  its={s.iterations:5d}  {dt:7.1f}s  "
              f"Nu={abs(Nu_h):.4f}({rec['Nu_err_pct']:5.2f}%)  "
              f"u={u_max:.3f}({rec['u_err_pct']:5.2f}%)@y={y_u:.3f}  "
              f"v={v_max:.3f}({rec['v_err_pct']:5.2f}%)@x={x_v:.3f}")

print("\nDone — all results in", JSON_PATH)