"""Le Quere et al. (2005) T2/T3: Sutherland-property low-Mach natural convection.

Writes results/lowmach_sutherland.json incrementally after each run.
    python scripts/run_lowmach_t2.py
"""
import sys, os, json, time, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
JSON_PATH = os.path.join(OUT, "lowmach_sutherland.json")
from simplefvm.lowmach import LowMachSolver
from simplefvm.properties import cavity_length, mu_sutherland, P0

if os.path.exists(JSON_PATH):
    results = json.load(open(JSON_PATH))
else:
    results = {}
existing = set(results.keys())

REF = {2: dict(Nu=8.6866, Pbar_P0=0.924487),
       3: dict(Nu=16.2410, Pbar_P0=0.92263)}

EPS = 0.6
mu0 = mu_sutherland(600.0)  # for cavity_length
L = cavity_length(1e6, EPS, mu0)

print("run_lowmach_t2.py — Sutherland properties")
print(f"  eps={EPS}, Ra={1e6}, L={L:.5f} m\n")

for case_id, Ra_val, meshes in [(2, 1e6, (32, 64, 128, 256)),
                                  (3, 1e7, (64, 128))]:
    ref = REF[case_id]
    L = cavity_length(Ra_val, EPS, mu0)   # L depends on Ra (T3 is Ra 1e7)
    print(f"--- T{case_id} (Ra={Ra_val}, Nu_ref={ref['Nu']}, Pb_ref={ref['Pbar_P0']}) ---")
    for M in meshes:
        key = f"T{case_id}_M{M}"
        if key in existing:
            print(f"  SKIP {key}")
            continue
        t0 = time.perf_counter()
        s = LowMachSolver(M=M, eps=EPS, Ra=Ra_val, L=L, p_solver="direct",
                           alpha_u=0.7, alpha_p=0.25, alpha_T=0.9,
                           properties="sutherland")
        s.run(max_outer=60000, tol=1e-8, verbose=5000)
        dt = round(time.perf_counter()-t0, 1)
        Nu_W = s.nu("W"); Nu_E = s.nu("E"); Pb = s.Pbar/P0
        rec = dict(Ra=Ra_val, M=M, converged=s.converged, iterations=s.iterations,
                   wall_time_s=dt, Nu_W=Nu_W, Nu_E=Nu_E, Pbar_P0=Pb,
                   Nu_err_pct=abs(Nu_W/ref['Nu']-1)*100,
                   Pbar_err_pct=abs(Pb/ref['Pbar_P0']-1)*100,
                   max_div=s.max_divergence(), status=s.status)
        results[key] = rec
        with open(JSON_PATH,"w") as fh: json.dump(results, fh, indent=1)
        print(f"  {key} its={s.iterations:5d} {dt:7.1f}s Nu_W={Nu_W:.5f} Nu_E={Nu_E:.5f}"
              f" ({rec['Nu_err_pct']:.2f}%) PB={Pb:.6f}({rec['Pbar_err_pct']:.2f}%) {s.status}")

    # Richardson
    ks = [f"T{case_id}_M{m}" for m in meshes]
    if len(ks)>=3 and all(k in results for k in ks[-3:]):
        f1,f2,f3=results[ks[-1]],results[ks[-2]],results[ks[-3]]
        for qty in ("Nu_W","Pbar_P0"):
            v1,v2,v3=f1[qty],f2[qty],f3[qty]
            if abs(v2-v3)>1e-30 and abs(v1-v2)>1e-30:
                p=np.log(abs((v3-v2)/(v2-v1)))/np.log(2)
                f1[f"{qty}_order"]=float(p)
                f1[f"{qty}_rich"]=float(v1+(v1-v2)/(2**p-1))
        results[ks[-1]]=f1
        with open(JSON_PATH,"w") as fh: json.dump(results,fh,indent=1)

print("\nDone —", JSON_PATH)