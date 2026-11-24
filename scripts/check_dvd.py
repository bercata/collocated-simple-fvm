"""Stage 2 checks: body-force pressure ghosts, DVD Table V, regression.

Run: python scripts/check_dvd.py
"""
import sys, os, time, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simplefvm import (NaturalConvectionSolver, SimpleSolver, setUpMesh,
                       Stencil2D, SimpleFields2D)
from simplefvm.solvers import (MomentumEquationSolver, RhieChowPressureCorrection,
                                PressureVelocityCorrector)


# ================================================================== Check A
def check_a():
    print("=" * 70)
    print("Check A: hydrostatic with nonlinear T=tanh, Ra=1e6, ghost fix + RC")
    print("-" * 70)

    def run(M, use_ghost, use_rc, iters=200):
        Pr = 0.71; L = 1.0; Ra = 1e6; buoy = Ra * Pr
        X, Y, Xc, Yc = setUpMesh(M, 2.0, 2.0, L, "rectangular")
        s = Stencil2D(X, Y, Xc, Yc)
        f = SimpleFields2D(X, Y, Xc, Yc, rho0=1.0, mu0=Pr, lid_velocity=0.0)
        f.mu[:] = Pr
        T = 0.5 + 0.5 * np.tanh((Y - 0.5) / 0.08)
        fx_cell = np.zeros_like(T)
        fy_cell = buoy * (T - 0.5)
        fx_rc = fx_cell if use_rc else None
        fy_rc = fy_cell if use_rc else None
        fx_g = fx_cell if use_ghost else None
        fy_g = fy_cell if use_ghost else None
        mom = MomentumEquationSolver(s, X, Y, Xc, Yc, inner_sweeps=2)
        c = (M // 2 + 1, M // 2 + 1)
        pcorr = RhieChowPressureCorrection(s, X, Y, Xc, Yc,
                                            reference=(c[0], c[1], 0.0),
                                            body_force_rc=use_rc)
        corr = PressureVelocityCorrector(s, Xc, Yc, gauge=c)
        for _ in range(iters):
            _, f.Ap_u, *_ = mom.solve(f, component="u", fx=fx_cell, fy=fy_cell)
            _, f.Ap_v, *_ = mom.solve(f, component="v", fx=fx_cell, fy=fy_cell)
            pcorr.solve(f, fx=fx_rc, fy=fy_rc)
            corr.apply(f, f.p_prime, alpha_p=0.3, fx=fx_g, fy=fy_g)
        return (np.max(np.abs(f.U[1:-1, 1:-1])),
                np.max(np.abs(f.V[1:-1, 1:-1])),
                np.max(np.abs(f.mflux_x)),
                np.max(np.abs(f.mflux_y)))

    print("  ghost_fix  rc        M   max|u|     max|v|     max|mflux_x|  max|mflux_y|")
    for gfix in (True, False):
        for rc in (True, False):
            for M in (32, 64):
                um, vm, mfx, mfy = run(M, gfix, rc)
                print(f"  {str(gfix):5s}   {str(rc):5s}   {M:3d}  "
                      f"{um:.2e}  {vm:.2e}  {mfx:.2e}  {mfy:.2e}")


# ================================================================== Check B
def check_b():
    print("\n" + "=" * 70)
    print("Check B: de Vahl Davis Table V  (converged to stage-2 criteria)")
    print("-" * 70)

    DVD_REF = {1e3: dict(Nu=1.117, u=3.649, y_u=0.813, v=3.697, x_v=0.178),
               1e4: dict(Nu=2.238, u=16.178, y_u=0.823, v=19.617, x_v=0.119),
               1e5: dict(Nu=4.509, u=34.73, y_u=0.855, v=68.59, x_v=0.066)}

    print("  Ra     M   its   time(s)  Nu        Nu_err  u_max(y)   v_max(x)   status    res(u/v/mass)  dT")
    for Ra in (1e3, 1e4, 1e5):
        t0 = time.perf_counter()
        s = NaturalConvectionSolver(M=32, Ra=Ra)
        s.run(max_outer=5000, tol=1e-8, verbose=0)
        dt = round(time.perf_counter() - t0, 1)
        Nu_h = float(np.sum(s.energy.wall_heat_flux("W", s.fields, s.T,
                                                      s.Gamma_T, cp=1.0)[0]))
        Nu_c = float(np.sum(s.energy.wall_heat_flux("E", s.fields, s.T,
                                                      s.Gamma_T, cp=1.0)[0]))
        u_m, y_u = s.max_u_mid()
        v_m, x_v = s.max_v_mid()
        ref = DVD_REF[Ra]
        dT = np.max(np.abs(s.T[1:-1, 1:-1] - s._T_old[1:-1, 1:-1]))
        r_u = s.history["u"][-1]
        r_v = s.history["v"][-1]
        r_m = s.history["mass"][-1]
        Nu_bal = abs(abs(Nu_h) - abs(Nu_c))
        print(f"  {int(Ra):5d}  {32:3d}  {s.iterations:5d}  {dt:7.1f}  "
              f"{abs(Nu_h):.4f}  {abs(abs(Nu_h)/ref['Nu']-1)*100:5.2f}%%   "
              f"{u_m:.3f}({y_u:.3f})  {v_m:.3f}({x_v:.3f})  "
              f"{s.status:10s}  {r_u:.1e}/{r_v:.1e}/{r_m:.1e}  {dT:.1e}")


# ================================================================== Check D
def check_d():
    print("\n" + "=" * 70)
    print("Check D: regression")
    print("-" * 70)
    s = SimpleSolver(M=32, Re=1000, shape="parallelogram", beta_deg=45, gamma_dc=1.0,
                     n_nonorth_correctors=2)
    s.run(max_outer=40000, tol=1e-9, verbose=0)
    psi = s.stream_function()
    psi_min = float(psi.min())
    print(f"  psi_min = {psi_min:+.10e}  (target -0.0490243555)")
    print("  pytest: run separately (python -m pytest -q tests/)")


if __name__ == "__main__":
    check_a()
    check_b()
    check_d()
    print("\nDone.")