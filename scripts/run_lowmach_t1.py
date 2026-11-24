"""Le Quere et al. (2005) case T1: constant-property low-Mach natural convection.

Writes results/lowmach_t1.json incrementally after each run.
    python scripts/run_lowmach_t1.py
"""
import sys, os, json, time, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
JSON_PATH = os.path.join(OUT, "lowmach_t1.json")
from simplefvm.lowmach import LowMachSolver
from simplefvm.properties import cavity_length, mu_star, P0

if os.path.exists(JSON_PATH):
    results = json.load(open(JSON_PATH))
else:
    results = {}
existing = set(results.keys())

T1_REF = dict(Nu=8.85978, Pbar_P0=0.856338)
EPS = 0.6; RA = 1e6; L = cavity_length(RA, EPS, mu_star)

print("run_lowmach_t1.py")
print(f"  T1: eps={EPS}, Ra={RA}, L={L:.5f} m")
print(f"  Reference: Nu={T1_REF['Nu']}, Pbar/P0={T1_REF['Pbar_P0']}")
print("  Writing to", JSON_PATH, "\n")

for M in (32, 64, 128, 256):
    key = f"M{M}"
    if key in existing:
        print(f"  SKIP {key} (already done)")
        continue
    t0 = time.perf_counter()
    s = LowMachSolver(M=M, eps=EPS, Ra=RA, L=L, p_solver="direct",
                       alpha_u=0.7, alpha_p=0.25, alpha_T=0.9)   # fastest converged set at M=32; result is relaxation-independent
    s.run(max_outer=60000, tol=1e-8, verbose=5000)
    dt = round(time.perf_counter() - t0, 1)
    Nu_h = s.nu("W"); Nu_c = s.nu("E"); Pb = s.Pbar/P0
    rec = dict(M=M, converged=s.converged, iterations=s.iterations,
               wall_time_s=dt, Nu_h=Nu_h, Nu_c=Nu_c,
               Nu_err_pct=abs(Nu_h/T1_REF['Nu']-1)*100,
               Pbar_P0=Pb,
               Pbar_err_pct=abs(Pb/T1_REF['Pbar_P0']-1)*100,
               max_div=s.max_divergence(), status=s.status)
    results[key] = rec
    with open(JSON_PATH, "w") as fh: json.dump(results, fh, indent=1)
    print(f"  {key} its={s.iterations:5d} {dt:7.1f}s Nu={Nu_h:.5f}({rec['Nu_err_pct']:.2f}%)"
          f" Pbar/P0={Pb:.6f}({rec['Pbar_err_pct']:.2f}%) {s.status}")
    s_per_it = dt/max(s.iterations,1)
    print(f"    {s_per_it:.4f} s/iter")

# Richardson on 64/128 with observed order from 32/64/128
for mlist in [(32, 64, 128), (64, 128, 256)]:
    k1, k2, k3 = f"M{mlist[2]}", f"M{mlist[1]}", f"M{mlist[0]}"
    if all(k in results for k in (k1, k2, k3)):
        f1, f2, f3 = results[k1], results[k2], results[k3]
        for qty in ("Nu_h", "Pbar_P0"):
            v1, v2, v3 = f1[qty], f2[qty], f3[qty]
            if abs(v2-v3)>1e-30 and abs(v1-v2)>1e-30:
                p = np.log(abs((v3-v2)/(v2-v1)))/np.log(2)
                rich = v1 + (v1-v2)/(2**p-1)
                f1[f"{qty}_order"] = float(p)
                f1[f"{qty}_rich"] = float(rich)
        results[k1] = f1
        with open(JSON_PATH, "w") as fh: json.dump(results, fh, indent=1)

print("\nDone —", JSON_PATH)