"""Check script for the consistent Rhie-Chow (Majumdar 1988) fix.

Run: python scripts/check_lowmach.py
"""
import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simplefvm import SimpleSolver, ghia


def run_silent(solver, max_outer=20000, tol=1e-9):
    solver.run(max_outer=max_outer, tol=tol, verbose=0)
    return solver


def report(solver, label):
    psi = solver.stream_function()
    psi_min = float(psi.min())
    y, u = solver.centerline_u()
    u_min = float(u.min())
    x, v = solver.centerline_v()
    v_max = float(v.max())
    v_min = float(v.min())
    max_div = solver.max_divergence()
    wall_mass = wall_mass_flux(solver)
    its = solver.iterations
    print(f"  {label:>30s}  psi_min={psi_min:+.6e}  U_min={u_min:+.6e}  V_max={v_max:+.6e}  V_min={v_min:+.6e}  "
          f"mass={max_div:.1e}  wall={wall_mass:.1e}  its={its:5d}")
    return {"psi_min": psi_min, "U_min": u_min, "V_max": v_max, "V_min": v_min,
            "max_div": max_div, "wall_mass": wall_mass, "its": its}


def wall_mass_flux(solver):
    mx = solver.fields.mflux_x
    my = solver.fields.mflux_y
    return float(np.abs(mx[0, :].sum() + mx[-1, :].sum() + my[:, 0].sum() + my[:, -1].sum()))


# ======================================================= Check 1: parallelogram, alpha independence
print("=" * 80)
print("Check 1: 45 deg parallelogram, Re=1000, M=32, gamma_dc=1, n_nonorth_correctors=2, tol=1e-9")
print("-" * 80)

for rc_label, rc in [("consistent_rc=True", True), ("consistent_rc=False", False)]:
    print(f"\n  --- {rc_label} ---")
    for au, ap in [(0.7, 0.3), (0.5, 0.2)]:
        s = SimpleSolver(M=32, Re=1000, shape="parallelogram", beta_deg=45,
                         gamma_dc=1.0, alpha_u=au, alpha_p=ap,
                         n_nonorth_correctors=2, consistent_rc=rc)
        run_silent(s, max_outer=40000, tol=1e-9)
        report(s, f"alpha_u={au}, alpha_p={ap} {rc_label}")


# ======================================================= Check 2: square cavity, alpha independence
print("\n" + "=" * 80)
print("Check 2: square cavity, Re=400, M=32, tol=1e-9")
print("-" * 80)

for rc_label, rc in [("consistent_rc=True", True), ("consistent_rc=False", False)]:
    print(f"\n  --- {rc_label} ---")
    for au, ap in [(0.7, 0.3), (0.5, 0.2)]:
        s = SimpleSolver(M=32, Re=400, alpha_u=au, alpha_p=ap, consistent_rc=rc)
        run_silent(s, max_outer=20000, tol=1e-9)
        report(s, f"alpha_u={au}, alpha_p={ap} {rc_label}")


# ======================================================= Check 3: Ghia Re=100 (square) and Demirdzic 45deg Re=100
print("\n" + "=" * 80)
print("Check 3: Ghia Re=100 M=32 and 45deg parallelogram Re=100 M=32, gamma_dc=1, n_nonorth_correctors=2")
print("-" * 80)

for rc_label, rc in [("consistent_rc=True (new)", True), ("consistent_rc=False (old)", False)]:
    print(f"\n  --- {rc_label} ---")

    # Ghia square cavity
    s = SimpleSolver(M=32, Re=100, gamma_dc=1.0, consistent_rc=rc)
    run_silent(s, max_outer=20000, tol=1e-9)
    err = ghia.compare(s, 100)
    report(s, f"Ghia Re=100 {rc_label}")
    print(f"           Ghia errors: u_rms={err['u_rms']:.6e}  v_rms={err['v_rms']:.6e}")

    # Demirdzic 45 degree parallelogram
    s2 = SimpleSolver(M=32, Re=100, shape="parallelogram", beta_deg=45,
                      gamma_dc=1.0, n_nonorth_correctors=2, consistent_rc=rc)
    run_silent(s2, max_outer=20000, tol=1e-9)
    report(s2, f"Demirdzic 45deg Re=100 {rc_label}")


print("\nDone.")