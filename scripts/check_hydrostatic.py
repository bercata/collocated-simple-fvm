"""Hydrostatic test: frozen T=tanh, energy skipped, balanced+RC=True.

Run: python scripts/check_hydrostatic.py
"""
import sys, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simplefvm import setUpMesh, Stencil2D, SimpleFields2D
from simplefvm.solvers import (MomentumEquationSolver, RhieChowPressureCorrection,
                                PressureVelocityCorrector,
                                compute_balanced_body_force, compute_face_forces)


def run(M, shape, beta, max_iter=2000):
    Ra = 1e6; Pr = 0.71; buoy = Ra * Pr
    X, Y, Xc, Yc = setUpMesh(M, 2.0, 2.0, 1.0, shape, beta_deg=beta)
    s = Stencil2D(X, Y, Xc, Yc)
    f = SimpleFields2D(X, Y, Xc, Yc, rho0=1.0, mu0=Pr, lid_velocity=0.0)
    f.mu[:] = Pr
    T = 0.5 + 0.5 * np.tanh((Y - 0.5) / 0.08)
    fx = np.zeros_like(T); fy = np.zeros_like(T)
    compute_balanced_body_force(T, buoy, 0.5, s, fx, fy)
    fy_fx, fy_fy = compute_face_forces(T, buoy, 0.5)
    fx_fx = np.zeros_like(fy_fx); fx_fy = np.zeros_like(fy_fy)
    vel_scale = np.sqrt(Ra)
    mom = MomentumEquationSolver(s, X, Y, Xc, Yc, inner_sweeps=2)
    c = (M // 2 + 1, M // 2 + 1)
    pcorr = RhieChowPressureCorrection(s, X, Y, Xc, Yc,
                                        reference=(c[0], c[1], 0.0),
                                        body_force_rc=True)
    corr = PressureVelocityCorrector(s, Xc, Yc, gauge=c)
    for it in range(max_iter):
        _, _, *_ = mom.solve(f, component="u", vel_scale=vel_scale)
        _, _, *_ = mom.solve(f, component="v", fx=fx, fy=fy, vel_scale=vel_scale)
        pp = pcorr.solve(f, fx=fx, fy=fy,
                         fx_face_x=fx_fx, fy_face_x=fy_fx,
                         fx_face_y=fx_fy, fy_face_y=fy_fy)
        corr.apply(f, pp, alpha_p=0.3, fx=fx, fy=fy)
        if it % 200 == 0 and it > 0:
            um = np.max(np.abs(f.U[1:-1, 1:-1]))
            vm = np.max(np.abs(f.V[1:-1, 1:-1]))
            if um < 1e-9 and vm < 1e-9:
                break
    return (np.max(np.abs(f.U[1:-1, 1:-1])),
            np.max(np.abs(f.V[1:-1, 1:-1])),
            np.max(np.abs(f.mflux_x)),
            np.max(np.abs(f.mflux_y)),
            mom.residual["u"], mom.residual["v"], it + 1)


if __name__ == "__main__":
    print("Hydrostatic test: frozen T = 0.5 + 0.5 tanh((y-0.5)/0.08), Ra=1e6")
    print("-" * 70)
    print(f"{'shape':>16s} {'M':>4s} {'|u|':>10s} {'|v|':>10s} {'|mfx|':>10s} {'|mfy|':>10s} {'res_u':>10s} {'res_v':>10s} {'its':>5s}")
    for shape, beta, Ms in [("rectangular", None, (32, 64)),
                              ("parallelogram", 45, (32,)),
                              ("parallelogram", 60, (32,))]:
        for M in Ms:
            um, vm, mfx, mfy, ru, rv, itn = run(M, shape, beta)
            tag = f"{shape}" if beta is None else f"para{beta}"
            print(f"{tag:>16s} {M:4d} {um:10.2e} {vm:10.2e} {mfx:10.2e} {mfy:10.2e} {ru:10.2e} {rv:10.2e} {itn:5d}")

    # one old-force comparison for square M=32
    M = 32
    X, Y, Xc, Yc = setUpMesh(M, 2.0, 2.0, 1.0, "rectangular")
    s = Stencil2D(X, Y, Xc, Yc)
    f = SimpleFields2D(X, Y, Xc, Yc, rho0=1.0, mu0=0.71, lid_velocity=0.0)
    f.mu[:] = 0.71
    T = 0.5 + 0.5 * np.tanh((Y - 0.5) / 0.08)
    fx_n = None; fy_n = 710000.0 * (T - 0.5)
    mom = MomentumEquationSolver(s, X, Y, Xc, Yc, inner_sweeps=2)
    c = (M // 2 + 1, M // 2 + 1)
    pcorr = RhieChowPressureCorrection(s, X, Y, Xc, Yc,
                                        reference=(c[0], c[1], 0.0),
                                        body_force_rc=False)
    corr = PressureVelocityCorrector(s, Xc, Yc, gauge=c)
    for it in range(500):
        _, _, *_ = mom.solve(f, component="u")
        _, _, *_ = mom.solve(f, component="v", fx=fx_n, fy=fy_n)
        pp = pcorr.solve(f, fx=fx_n, fy=fy_n)
        corr.apply(f, pp, alpha_p=0.3, fx=fx_n, fy=fy_n)
    print(f"{'old (square)':>16s} {32:4d} "
          f"{np.max(np.abs(f.U[1:-1,1:-1])):10.2e} "
          f"{np.max(np.abs(f.V[1:-1,1:-1])):10.2e} "
          f"{np.max(np.abs(f.mflux_x)):10.2e} "
          f"{np.max(np.abs(f.mflux_y)):10.2e} "
          f"{mom.residual['u']:10.2e} {mom.residual['v']:10.2e} {500:5d}")
    print("\nDone.")