"""Part A3: Project case M32 with fixed unbalanced_force trap."""
import sys, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simplefvm.lowmach import LowMachSolver
from simplefvm.properties import cavity_length, mu_sutherland, P0

EPS = 0.6; RA = 1e5; BETA = 60; PROPERTIES = "sutherland"; NN = 2
mu0 = mu_sutherland(600.0); L = cavity_length(RA, EPS, mu0)

def run(label, traps_set):
    s = LowMachSolver(M=32, shape="parallelogram", beta_deg=BETA,
                      eps=EPS, Ra=RA, L=L, properties=PROPERTIES,
                      alpha_u=0.4, alpha_p=0.15, alpha_T=0.6,
                      p_solver="direct", n_nonorth_correctors=NN,
                      traps=traps_set)
    t0 = time.perf_counter()
    s.run(max_outer=20000, tol=1e-8, verbose=0)
    dt = time.perf_counter() - t0
    psi_min, _, _ = s.psi_min()
    _, Nu_hot, y_hot = s.nu_local("W")
    _, Nu_cold, y_cold = s.nu_local("E")
    return {"conv": s.converged, "its": s.iterations, "time": dt,
            "Nu": s.nu("W"), "Pbar_P0": float(s.Pbar/P0),
            "psi_min": psi_min, "Nu_hot": Nu_hot, "y_hot": y_hot,
            "Nu_cold": Nu_cold, "y_cold": y_cold}

print("PART A3: Project case M32 (eps=0.6, Ra=1e5, beta=60, sutherland)")
b = run("baseline", ())
print(f"  baseline:        Nu={b['Nu']:.4f}  Pbar/P0={b['Pbar_P0']:.4f}  psi_min={b['psi_min']:.4e}  y_hot={b['y_hot']:.4f}  y_cold={b['y_cold']:.4f}  its={b['its']}  {b['time']:.1f}s")
t = run("unbalanced_force", ("unbalanced_force",))
print(f"  trap (fixed):    Nu={t['Nu']:.4f}  Pbar/P0={t['Pbar_P0']:.4f}  psi_min={t['psi_min']:.4e}  y_hot={t['y_hot']:.4f}  y_cold={t['y_cold']:.4f}  its={t['its']}  {t['time']:.1f}s")
print()
print("  Changes vs baseline:")
for k in ["Nu", "Pbar_P0", "psi_min", "Nu_hot", "Nu_cold"]:
    pct = 100*(t[k] - b[k]) / (abs(b[k]) + 1e-30)
    print(f"    {k:>12s}: {t[k]:15.6e}  ({pct:+.2f}%)")
print(f"  psi_min sign: {'NEGATIVE (correct)' if t['psi_min'] < 0 else 'POSITIVE (wrong!)'}")
print(f"  y_hot (must be ~0.20): {t['y_hot']:.4f}")
print(f"  y_cold (must be ~0.77): {t['y_cold']:.4f}")