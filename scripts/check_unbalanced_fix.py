"""Sanity tests for the unbalanced_force fix (Part A2 + A3).

A2: hydrostatic test — uniform T, no heating, M=32.
A3: project case M=32 (eps=0.6, Ra=1e5, beta=60, sutherland, n_nonorth=2).
"""
import sys, os, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simplefvm.lowmach import LowMachSolver
from simplefvm.properties import cavity_length, mu_sutherland

EPS = 0.6
RA = 1e5
BETA = 60
PROPERTIES = "sutherland"
N_NONORTH = 2
mu0 = mu_sutherland(600.0)
L = cavity_length(RA, EPS, mu0)
MAX_OUTER = {32: 20000, 64: 40000}


def make_solver(M, traps_set=(), au=0.4, ap=0.15, aT=0.6):
    return LowMachSolver(M=M, shape="parallelogram", beta_deg=BETA,
                         eps=EPS, Ra=RA, L=L, properties=PROPERTIES,
                         alpha_u=au, alpha_p=ap, alpha_T=aT,
                         p_solver="direct", n_nonorth_correctors=N_NONORTH,
                         traps=traps_set)


# ==================================================================== A2: hydrostatic
print("=" * 65)
print("PART A2: Hydrostatic test (uniform T=T0, M=32, beta=60)")
print("=" * 65)

def hydrostatic_test(label, traps_set):
    s = make_solver(32, traps_set=traps_set)
    s.T[:] = s.T0
    s.Pbar = s.P0
    s.rho_cell[:] = s.rho_ref
    s.fields.rho[:] = s.rho_ref
    s.fields.U[:, :] = 0.0
    s.fields.V[:, :] = 0.0
    s.fields.P[:, :] = 0.0

    if "unbalanced_force" in s.traps:
        s.pcorr.body_force_rc = False

    for it in range(10):
        r = s.step()
        s.T[:] = s.T0
        s.rho_cell[:] = s.rho_ref
        s.fields.rho[:] = s.rho_ref
    umax = float(np.max(np.abs(s.fields.U[1:-1, 1:-1])))
    vmax = float(np.max(np.abs(s.fields.V[1:-1, 1:-1])))
    print(f"  {label:>25s}: max |U| = {umax:.3e}  max |V| = {vmax:.3e}")
    return umax, vmax

u_b, v_b = hydrostatic_test("baseline", ())
u_t, v_t = hydrostatic_test("unbalanced_force (fixed)", ("unbalanced_force",))

print("\n  Summary: baseline max|u| = {:.3e}, trap max|u| = {:.3e}".format(
    max(u_b, v_b), max(u_t, v_t)))
print("  Expected: baseline ~ round-off (<1e-12), trap small but >0 (non-zero).")


# ==================================================================== A3: project case M32
print("\n" + "=" * 65)
print("PART A3: Project case M32 with fixed unbalanced_force")
print("=" * 65)

def run_m32(label, traps_set):
    s = make_solver(32, traps_set=traps_set)
    t0 = time.perf_counter()
    s.run(max_outer=MAX_OUTER[32], tol=1e-8, verbose=0)
    dt = time.perf_counter() - t0
    psi_min, _, _ = s.psi_min()
    _, Nu_hot, y_hot = s.nu_local("W")
    _, Nu_cold, y_cold = s.nu_local("E")
    nuw = s.nu("W")
    pbar_p0 = float(s.Pbar / s.P0)
    return {
        "converged": s.converged, "its": s.iterations, "time": dt,
        "Nu_W": nuw, "Pbar_P0": pbar_p0,
        "psi_min": psi_min,
        "Nu_max_hot": Nu_hot, "y_hot": y_hot,
        "Nu_max_cold": Nu_cold, "y_cold": y_cold,
        "status": s.status,
    }

b32 = run_m32("baseline", ())
t32 = run_m32("unbalanced_force (fixed)", ("unbalanced_force",))

fields = ["conv?", "its", "time_s", "Nu_W", "Pbar/P0", "psi_min",
          "Nu_max_hot", "y_hot", "Nu_max_cold", "y_cold"]
print()
print(f"{'metric':>18s} {'baseline':>18s} {'unbalanced_force':>18s} {'change%':>10s}")
print("-" * 65)

baseline_data = [
    "Y" if b32["converged"] else "N",
    b32["its"], b32["time"],
    b32["Nu_W"], b32["Pbar_P0"], b32["psi_min"],
    b32["Nu_max_hot"], b32["y_hot"],
    b32["Nu_max_cold"], b32["y_cold"],
]
trap_data = [
    "Y" if t32["converged"] else "N",
    t32["its"], t32["time"],
    t32["Nu_W"], t32["Pbar_P0"], t32["psi_min"],
    t32["Nu_max_hot"], t32["y_hot"],
    t32["Nu_max_cold"], t32["y_cold"],
]

for i, fld in enumerate(fields):
    bv = baseline_data[i]
    tv = trap_data[i]
    if isinstance(bv, (int, float)) and isinstance(tv, (int, float)) and abs(bv) > 1e-30:
        chg = 100.0 * (tv - bv) / bv
    else:
        chg = float("nan")
    bstr = f"{bv:.6e}" if isinstance(bv, float) and abs(bv) < 1e-3 else (
        f"{bv:.6f}" if isinstance(bv, float) else str(bv))
    tstr = f"{tv:.6e}" if isinstance(tv, float) and abs(tv) < 1e-3 else (
        f"{tv:.6f}" if isinstance(tv, float) else str(tv))
    cstr = f"{chg:.2f}" if not np.isnan(chg) else "-"
    print(f"{fld:>18s} {bstr:>18s} {tstr:>18s} {cstr:>10s}")

print(f"\n  Key check: psi_min sign must be negative (like baseline -6.45e-4)")
print(f"             Nu_max_hot y* must be near 0.20 (baseline 0.203)")
print(f"             Nu_max_cold y* must be near 0.77 (baseline 0.766)")
print(f"             Status: {t32['status']}")