"""Hydrostatic test for lowmach unbalanced_force fix (Part A2).

Uses core momentum+pressure solver with a frozen stratified T profile.
rho = P0/(R*T), f = -(rho - rho_ref)*g.
Balanced: face-consistent body force -> round-off velocity (hydrostatic balance).
Trap: plain cell-centre force -> small but non-zero spurious currents.
"""
import sys, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simplefvm import setUpMesh, Stencil2D, SimpleFields2D
from simplefvm.solvers import (MomentumEquationSolver,
                                RhieChowPressureCorrection,
                                PressureVelocityCorrector)

P0_val = 101325.0; T0_val = 600.0; R_gas = 287.0; g_val = 9.81
rho0 = P0_val / (R_gas * T0_val)


def run_test(M, shape, beta, balanced, max_iter=4000):
    X, Y, Xc, Yc = setUpMesh(M, 2.0, 2.0, 1.0, shape, beta_deg=beta)
    s = Stencil2D(X, Y, Xc, Yc)
    Pr = 0.71
    f = SimpleFields2D(X, Y, Xc, Yc, rho0=rho0, mu0=Pr, lid_velocity=0.0)
    f.mu[:] = Pr

    Y_arr = Y  # includes ghosts
    eps_T = 0.01  # small temperature stratification
    T = T0_val + eps_T * T0_val * np.tanh((Y_arr - 1.0) / 0.25)

    rho = P0_val / (R_gas * T + 1e-30)
    rho_cell = rho.copy()

    if balanced:
        bu = -float(g_val)
        M = s.M
        Tx = 0.5*(T[:-1, 1:-1] + T[1:, 1:-1])
        Ty = 0.5*(T[1:-1, :-1] + T[1:-1, 1:])
        Tx[0, :] = T[0, 1:-1]; Tx[-1, :] = T[-1, 1:-1]
        Ty[:, 0] = T[1:-1, 0]; Ty[:, -1] = T[1:-1, -1]
        rho_x = P0_val/(R_gas*Tx + 1e-30)
        rho_y = P0_val/(R_gas*Ty + 1e-30)
        fy_x = bu*(rho_x - rho0)
        fy_y = bu*(rho_y - rho0)

        fx = np.zeros_like(T); fy = np.zeros_like(T)
        fd_i = fy_x[1:M, :]*s.dx_y[1:M, :]
        sx_i, sy_i = s.Sx_x[1:M, :], s.Sx_y[1:M, :]
        fx[1:M, 1:-1] += fd_i*sx_i/(2.0*s.vol[:M-1, :])
        fy[1:M, 1:-1] += fd_i*sy_i/(2.0*s.vol[:M-1, :])
        fx[2:M+1, 1:-1] += fd_i*sx_i/(2.0*s.vol[1:M, :])
        fy[2:M+1, 1:-1] += fd_i*sy_i/(2.0*s.vol[1:M, :])
        fd_j = fy_y[:, 1:M]*s.dy_y[:, 1:M]
        sx_j, sy_j = s.Sy_x[:, 1:M], s.Sy_y[:, 1:M]
        fx[1:-1, 1:M] += fd_j*sx_j/(2.0*s.vol[:, :M-1])
        fy[1:-1, 1:M] += fd_j*sy_j/(2.0*s.vol[:, :M-1])
        fx[1:-1, 2:M+1] += fd_j*sx_j/(2.0*s.vol[:, 1:M])
        fy[1:-1, 2:M+1] += fd_j*sy_j/(2.0*s.vol[:, 1:M])
        fd = fy_x[0, :]*s.dx_y[0, :]
        fx[1, 1:-1] += fd*s.Sx_x[0, :]/s.vol[0, :]
        fy[1, 1:-1] += fd*s.Sx_y[0, :]/s.vol[0, :]; fy[0, 1:-1] = fy_x[0, :]
        fd = fy_x[M, :]*s.dx_y[M, :]
        fx[M, 1:-1] += fd*s.Sx_x[M, :]/s.vol[M-1, :]
        fy[M, 1:-1] += fd*s.Sx_y[M, :]/s.vol[M-1, :]; fy[-1, 1:-1] = fy_x[M, :]
        fd = fy_y[:, 0]*s.dy_y[:, 0]
        fx[1:-1, 1] += fd*s.Sy_x[:, 0]/s.vol[:, 0]
        fy[1:-1, 1] += fd*s.Sy_y[:, 0]/s.vol[:, 0]; fy[1:-1, 0] = fy_y[:, 0]
        fd = fy_y[:, M]*s.dy_y[:, M]
        fx[1:-1, M] += fd*s.Sy_x[:, M]/s.vol[:, M-1]
        fy[1:-1, M] += fd*s.Sy_y[:, M]/s.vol[:, M-1]; fy[1:-1, -1] = fy_y[:, M]

        fy_fx = fy_x; fy_fy = fy_y
        fx_fx = np.zeros_like(fy_x); fx_fy = np.zeros_like(fy_y)
        body_force_rc = True
        corr_fx = fx; corr_fy = fy
    else:
        bu = -float(g_val)
        fx = np.zeros_like(T)
        fy = bu * (rho_cell - rho0)
        fx_fx = fy_fx = fx_fy = fy_fy = None
        body_force_rc = False
        corr_fx = corr_fy = None

    vel_scale = np.sqrt(float(g_val * 1.0))
    mom = MomentumEquationSolver(s, X, Y, Xc, Yc, inner_sweeps=2)
    c = (M // 2 + 1, M // 2 + 1)
    pcorr = RhieChowPressureCorrection(s, X, Y, Xc, Yc,
                                        reference=(c[0], c[1], 0.0),
                                        body_force_rc=body_force_rc)
    corr = PressureVelocityCorrector(s, Xc, Yc, gauge=c)

    for it in range(max_iter):
        _, _, *_ = mom.solve(f, component="u", vel_scale=vel_scale)
        _, _, *_ = mom.solve(f, component="v", fx=fx, fy=fy, vel_scale=vel_scale)
        pp = pcorr.solve(f, fx=fx, fy=fy,
                         fx_face_x=fx_fx, fy_face_x=fy_fx,
                         fx_face_y=fx_fy, fy_face_y=fy_fy)
        corr.apply(f, pp, alpha_p=0.3, fx=corr_fx, fy=corr_fy)
        if it % 500 == 0 and it > 0:
            um = np.max(np.abs(f.U[1:-1, 1:-1]))
            vm = np.max(np.abs(f.V[1:-1, 1:-1]))
            if um < 1e-9 and vm < 1e-9:
                break
    return (np.max(np.abs(f.U[1:-1, 1:-1])),
            np.max(np.abs(f.V[1:-1, 1:-1])),
            it + 1)


if __name__ == "__main__":
    M = 32
    print("PART A2: Hydrostatic test (stratified T, M=32, beta=60)")
    print(f"  T(y) = T0 * (1 + 0.01 * tanh((y-1)/0.25)), T0 = {T0_val} K")
    print("  rho = P0/(R*T), f = -(rho - rho_ref) * g")
    print("-" * 60)
    print(f"{'label':>28s} {'|u|':>12s} {'|v|':>12s} {'its':>6s}")
    um_b, vm_b, its_b = run_test(M, "parallelogram", 60, balanced=True)
    print(f"{'baseline (balanced)':>28s} {um_b:12.3e} {vm_b:12.3e} {its_b:6d}")
    um_t, vm_t, its_t = run_test(M, "parallelogram", 60, balanced=False)
    print(f"{'trap (unbalanced, fixed)':>28s} {um_t:12.3e} {vm_t:12.3e} {its_t:6d}")
    print("\n  Expected: baseline ~ 0.0, trap > 0 (spurious current)")
    print(f"  Ratio trap/baseline |v|: {vm_t/(vm_b+1e-30):.1f}")
    print("\nDone.")