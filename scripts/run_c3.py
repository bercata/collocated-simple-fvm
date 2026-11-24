"""Demirdzic C3 skewed-cavity natural convection runs.

Writes results/demirdzic_c3.json incrementally after each run.
Stop on convergence, not on a small cap. Richardson on 64/128 with
observed order from 32/64/128 after each Pr finishes.

    python scripts/run_c3.py
"""
import sys, os, json, time, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
JSON_PATH = os.path.join(OUT, "demirdzic_c3.json")
from simplefvm import NaturalConvectionSolver

if os.path.exists(JSON_PATH):
    results = json.load(open(JSON_PATH))
else:
    results = {}
existing = set(results.keys())

DEMIRDZIC_C3 = {
    0.1: dict(Nu=5.98493, Nu_max=8.6778, yNu=0.45644, psi_min=-9.68706e-3,
              psi_x=0.78926, psi_y=0.178134),
    10: dict(Nu=7.58013, Nu_max=12.471, yNu=0.17028, psi_min=-1.662127e-3,
             psi_x=0.57276, psi_y=0.319989),
}

def pct(v, r):
    return abs(v - r) / abs(r) * 100

def richardson(f1, f2, f3, h_ratio=2.0):
    """Return (observed_order, richardson_extrapolated).
       f1, f2, f3 are values on fine, medium, coarse meshes."""
    if abs(f2 - f3) < 1e-30 or abs(f1 - f2) < 1e-30:
        return 0.0, f1
    p = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(h_ratio)
    rich = f1 + (f1 - f2) / (h_ratio ** p - 1)
    return float(p), float(rich)

print("run_c3.py — Demirdzic C3, beta=45")
print("Writing to", JSON_PATH)
print("=" * 70)

# ---- Pr=10 first (faster) ----
for Pr, mu, k, meshes in [(10, 1e-3, 1e-4, (32, 64, 128)),
                            (0.1, 1e-4, 1e-3, (32, 64, 128)),
                            (10, 1e-3, 1e-4, (32, 64, 128, 256))]:   # M=256 last; 32-128 are reused
    # Pr 0.1 (Gr 1e7): 0.5/0.2/0.7 gives a SIMPLE limit cycle at M>=64 (PROMPT 23-3e); 0.3/0.1/0.5 converges
    alpha_u = 0.6 if Pr == 10 else 0.3
    alpha_p = 0.3 if Pr == 10 else 0.1
    alpha_T = 0.8 if Pr == 10 else 0.5
    data_Pr = {}

    for M in meshes:
        key = f"Pr{Pr}_M{M}"
        if key in results and results[key].get("converged"):
            print(f"  SKIP {key} (already done)")
            data_Pr[str(M)] = results[key]
            continue

        t0 = time.perf_counter()
        s = NaturalConvectionSolver(M=M, shape="parallelogram", beta_deg=45,
                                     L=1.0, H=1.0, rho=1.0, mu=mu, k=k,
                                     cp=1.0, g_beta=0.1, T_h=1.0, T_c=0.0,
                                     T_ref=0.5, alpha_u=alpha_u,
                                     alpha_p=alpha_p, alpha_T=alpha_T,
                                     p_solver="direct", n_nonorth_correctors=2,
                                     inner_sweeps=4)
        s.run(max_outer=60000 if Pr == 10 else 120000, tol=1e-8, verbose=2000)
        dt = round(time.perf_counter() - t0, 1)
        Nu_c = s.nu("E"); Nu_h = s.nu("W")
        _, Nu_max_val, yNu = s.nu_local("W")   # Demirdzic Table VI: Nu_max and y* on the HOT wall (W)
        psi_m, psi_x, psi_y = s.psi_min()
        ref = DEMIRDZIC_C3[Pr]
        rec = dict(Pr=Pr, M=M, converged=s.converged, iterations=s.iterations,
                   wall_time_s=dt, Nu_c=Nu_c, Nu_h=Nu_h,
                   Nu_bal=abs(Nu_h - Nu_c),
                   Nu_max=Nu_max_val, yNu=yNu,
                   Nu_err_pct=pct(Nu_c, ref["Nu"]),
                   Nu_max_err_pct=pct(Nu_max_val, ref["Nu_max"]),
                   psi_min=psi_m, psi_x=psi_x, psi_y=psi_y,
                   psi_err_pct=pct(abs(psi_m), abs(ref["psi_min"])),
                   max_div=s.max_divergence(),
                   status=s.status)
        results[key] = rec
        with open(JSON_PATH, "w") as fh:
            json.dump(results, fh, indent=1)
        data_Pr[str(M)] = rec

        s_per_it = dt / max(s.iterations, 1)
        print(f"  {key} its={s.iterations:5d} {dt:7.1f}s  "
              f"Nu={Nu_c:.4f}({rec['Nu_err_pct']:.2f}%)  "
              f"Nu_max={Nu_max_val:.3f}@y*={yNu:.4f}  "
              f"psi={psi_m:.6e}  {s.status}  "
              f"({s_per_it:.3f}s/iter)")
        if s_per_it > 0:
            for eM in meshes:
                if eM > M:
                    est = round(s_per_it * (eM / M) ** 2.3, 0)
                    print(f"     -> estimated M={eM}: ~{est:.0f}s")

    # Richardson on Nu, Nu_max, psi_min for this Pr
    for mlist in [(32, 64, 128), (64, 128, 256)]:
        s1, s2, s3 = str(mlist[2]), str(mlist[1]), str(mlist[0])
        if all(k in data_Pr for k in (s1, s2, s3)):
            f1, f2, f3 = data_Pr[s1], data_Pr[s2], data_Pr[s3]
            for qty in ("Nu_c", "Nu_max", "psi_min"):
                v1, v2, v3 = f1[qty], f2[qty], f3[qty]
                p, rich = richardson(v1, v2, v3)
                f1[f"{qty}_order"] = p
                f1[f"{qty}_rich"] = rich
                f1[f"{qty}_gci_128"] = float(
                    1.25 * abs(v1 - v2) / abs(v1) / (2 ** p - 1) * 100
                ) if abs(p) > 1e-6 and abs(v1) > 1e-30 else None
            results[f"Pr{Pr}_M{mlist[2]}"] = f1
    # Save updated
    with open(JSON_PATH, "w") as fh:
        json.dump(results, fh, indent=1)

print("\nDone —", JSON_PATH)