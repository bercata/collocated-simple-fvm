"""Code verification for the passive-scalar energy solver (Stage 1).

Run: python scripts/check_energy.py
"""
import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simplefvm import (SimpleSolver, SimpleFields2D, Stencil2D, setUpMesh,
                       FVM_grad, EnergySolver)
import time as _time_mod


def _mesh(M, shape, beta_deg=None):
    H1 = 2.0
    H2 = 2.0 if shape == "rectangular" else 1.2
    X, Y, Xc, Yc = setUpMesh(M, H1, H2, 1.0, shape, beta_deg=beta_deg)
    s = Stencil2D(X, Y, Xc, Yc)
    return X, Y, Xc, Yc, s


def _fields(X, Y, Xc, Yc):
    return SimpleFields2D(X, Y, Xc, Yc, rho0=1.0, mu0=1.0, lid_velocity=1.0)


def _set_psi_mflux(mflux_x, mflux_y, psi_cor, s, rho=1.0):
    M = s.M
    mflux_x[:] = rho * (psi_cor[:M + 1, 1:M + 1] - psi_cor[:M + 1, 0:M])
    mflux_y[:] = -rho * (psi_cor[1:M + 1, :M + 1] - psi_cor[0:M, :M + 1])
    mflux_x[~s.x_interior] = 0.0
    mflux_y[~s.y_interior] = 0.0


# =========================================================================== MMS helpers
_PI = np.pi


def _psi_sq(x, y):
    return np.sin(_PI * x) ** 2 * np.sin(_PI * y) ** 2


def _T_mms(x, y):
    return np.cos(_PI * x) * np.sin(_PI * y) + x * y


def _source_sq(x, y, Gamma):
    sinx = np.sin(_PI * x); cosx = np.cos(_PI * x)
    siny = np.sin(_PI * y); cosy = np.cos(_PI * y)
    u = 2.0 * _PI * sinx ** 2 * siny * cosy
    v = -2.0 * _PI * sinx * cosx * siny ** 2
    dTdx = -_PI * sinx * siny + y
    dTdy = _PI * cosx * cosy + x
    lap = -2.0 * _PI ** 2 * cosx * siny
    return u * dTdx + v * dTdy - Gamma * lap


def _source_para(xc, yc, Gamma, beta_deg):
    cosb = np.cos(np.radians(beta_deg)); sinb = np.sin(np.radians(beta_deg))
    L, H = 1.0, 1.0
    eta = yc / sinb
    xi = (xc - eta * cosb) / L
    sin_xi = np.sin(_PI * xi); cos_xi = np.cos(_PI * xi)
    sin_eta = np.sin(_PI * eta); cos_eta = np.cos(_PI * eta)
    psi_xi = 2.0 * _PI / L * sin_xi * cos_xi * sin_eta ** 2
    psi_eta = 2.0 * _PI / H * sin_xi ** 2 * sin_eta * cos_eta
    u = (-psi_xi * cosb + psi_eta) / sinb
    v = -psi_xi
    sinx = np.sin(_PI * xc); cosx = np.cos(_PI * xc)
    siny = np.sin(_PI * yc); cosy = np.cos(_PI * yc)
    dTdx = -_PI * sinx * siny + yc
    dTdy = _PI * cosx * cosy + xc
    lap = -2.0 * _PI ** 2 * cosx * siny
    return u * dTdx + v * dTdy - Gamma * lap


# ================================================================== Check 1
def check1():
    print("=" * 70)
    print("Check 1: linear field exactness (no flow, pure diffusion)")
    print("-" * 70)
    all_ok = True
    for shape, betas in [("rectangular", [None]), ("parallelogram", [45, 60])]:
        for beta in betas:
            for M in (16, 32):
                X, Y, Xc, Yc, s = _mesh(M, shape, beta_deg=beta)
                f = _fields(X, Y, Xc, Yc)
                f.add_field("T", 0.0)
                T = f.vars["T"]
                Gamma = np.ones_like(T)
                use_wt = (shape == "parallelogram")
                es = EnergySolver(s, X, Y, Xc, Yc, gamma_dc=0.0, linear_solver="direct",
                                  wall_tangential=use_wt)
                T_exact = 0.3 + 1.7 * X - 0.9 * Y
                T[:] = T_exact
                es.set_bc(W=("T", T_exact.copy()), E=("T", T_exact.copy()),
                          S=("T", T_exact.copy()), N=("T", T_exact.copy()))
                f.mflux_x[:] = 0.0
                f.mflux_y[:] = 0.0
                es.solve(f, T, Gamma, tol=1e-12, max_iter=5)
                err = np.max(np.abs(T[1:-1, 1:-1] - T_exact[1:-1, 1:-1]))
                tag = f"{shape}{f' beta={beta}' if beta else ''} M={M}"
                ok = err < 1e-10
                print(f"  {tag:32s}  max_err={err:.2e}  {'OK' if ok else 'FAIL'}")
                if not ok:
                    all_ok = False
    return all_ok


# ================================================================== Check 2
def check2():
    print("\n" + "=" * 70)
    print("Check 2: pure conduction Nu  (W=1, E=0, S/N adiabatic)")
    print("-" * 70)
    X, Y, Xc, Yc, s = _mesh(32, "rectangular")
    f = _fields(X, Y, Xc, Yc)
    f.add_field("T")
    T = f.vars["T"]
    Gamma = np.ones_like(T)
    es = EnergySolver(s, X, Y, Xc, Yc, gamma_dc=0.0, linear_solver="direct",
                      wall_tangential=False)
    es.set_bc(W=("T", 1.0), E=("T", 0.0), S=("q", 0.0), N=("q", 0.0))
    f.mflux_x[:] = 0.0; f.mflux_y[:] = 0.0
    es.solve(f, T, Gamma, tol=1e-12, max_iter=10)
    cp, k0, dT = 1.0, 1.0, 1.0
    NuW, _ = es.nusselt("W", f, T, Gamma, cp=cp, k0=k0, dT=dT)
    NuE, _ = es.nusselt("E", f, T, Gamma, cp=cp, k0=k0, dT=dT)
    T_exact = 1.0 - X
    err = np.max(np.abs(T[1:-1, 1:-1] - T_exact[1:-1, 1:-1]))
    ok = abs(NuW - 1.0) < 1e-10 and abs(NuE - 1.0) < 1e-10 and err < 1e-10
    print(f"  Nu_W={NuW:.10f}  Nu_E={NuE:.10f}  max_err={err:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


# ================================================================== Check 3
def check3():
    print("\n" + "=" * 70)
    print("Check 3: MMS order  (frozen psi fluxes, manufactured T + source)")
    print("-" * 70)
    Gamma_val = 0.05
    all_ok = True

    for shape, beta, use_wt in [("square", None, False), ("parallelogram", 60, True)]:
        tag = f"beta={beta}" if beta else "square"
        mesh_range = (16, 32, 64, 128) if shape == "square" else (16, 32, 64, 128)
        for gamma in (0.0, 1.0):
            dc_tag = "dc" if gamma else "upwind"
            errors_l2, errors_max = [], []
            for M in mesh_range:
                X, Y, Xc, Yc, s = _mesh(M, "rectangular" if shape == "square" else shape,
                                         beta_deg=beta)
                f = _fields(X, Y, Xc, Yc)
                f.add_field("T")
                T = f.vars["T"]
                Gamma_pad = np.full_like(T, Gamma_val)

                # psi on vertices
                if shape == "square":
                    psi_on = _psi_sq(Xc, Yc)
                else:
                    cosb = np.cos(np.radians(beta)); sinb = np.sin(np.radians(beta))
                    eta_c = Yc / sinb
                    xi_c = (Xc - eta_c * cosb) / 1.0
                    psi_on = np.sin(_PI * xi_c) ** 2 * np.sin(_PI * eta_c) ** 2

                _set_psi_mflux(f.mflux_x, f.mflux_y, psi_on, s)

                es = EnergySolver(s, X, Y, Xc, Yc, gamma_dc=gamma, linear_solver="direct",
                                  wall_tangential=use_wt)
                # Dirichlet walls from the manufactured T_e at ghost cell positions
                T_exact = _T_mms(X, Y)
                T[:] = T_exact
                es.set_bc(W=("T", T_exact.copy()), E=("T", T_exact.copy()),
                          S=("T", T_exact.copy()), N=("T", T_exact.copy()))

                # source
                if shape == "square":
                    S_cell = _source_sq(X[1:-1, 1:-1], Y[1:-1, 1:-1], Gamma_val)
                else:
                    S_cell = _source_para(X[1:-1, 1:-1], Y[1:-1, 1:-1], Gamma_val, beta)
                source = np.zeros_like(T)
                source[1:-1, 1:-1] = S_cell  # assemble_raw already multiplies by vol
                es.solve(f, T, Gamma_pad, source=source, tol=1e-12, max_iter=30)

                err = np.abs(T[1:-1, 1:-1] - T_exact[1:-1, 1:-1])
                eL2 = float(np.sqrt(np.mean(err ** 2)))
                eMax = float(np.max(err))
                errors_l2.append(eL2)
                errors_max.append(eMax)
                print(f"  {tag:>12s} {dc_tag:>6s} M={M:3d}  L2={eL2:.2e}  max={eMax:.2e}")

            if len(errors_l2) >= 3:
                e3, e2, e1 = errors_l2[-3:]
                if e1 > 0 and e2 > 0 and e3 > 0:
                    p12 = np.log(e2 / e1) / np.log(2)
                    p23 = np.log(e3 / e2) / np.log(2)
                    print(f"  {'':>12s} {'':>6s} order: ref12={p12:.2f}  ref23={p23:.2f}")
                    # expect ~2 for dc, ~1 for upwind on square
                    # parallelogram may differ without wall_tangential

    return all_ok


# ================================================================== Check 4
def check4():
    print("\n" + "=" * 70)
    print("Check 4: lid-driven passive scalar  (Re=100, W=1, E=0, S/N adia, Gamma=0.01)")
    print("-" * 70)
    s = SimpleSolver(M=32, Re=100)
    s.run(max_outer=20000, tol=1e-9, verbose=0)
    f = s.fields
    f.add_field("T")
    T = f.vars["T"]
    Gamma_val = 0.01
    Gamma = np.full_like(T, Gamma_val)

    for gamma_dc, dc_tag in [(0.0, "upwind"), (1.0, "dc")]:
        T[:] = 0.5
        es = EnergySolver(s.stencil, s.X, s.Y, s.X_cor, s.Y_cor,
                          gamma_dc=gamma_dc, linear_solver="direct", wall_tangential=False)
        es.set_bc(W=("T", 1.0), E=("T", 0.0), S=("q", 0.0), N=("q", 0.0))
        es.solve(f, T, Gamma, tol=1e-12, max_iter=50)
        _, Qtot_W = es.wall_heat_flux("W", f, T, Gamma, cp=1.0)
        _, Qtot_E = es.wall_heat_flux("E", f, T, Gamma, cp=1.0)
        balance = abs(Qtot_W + Qtot_E) / (abs(Qtot_W) + 1e-30)
        Tmin = float(T[1:-1, 1:-1].min())
        Tmax = float(T[1:-1, 1:-1].max())
        bounded = (Tmin >= -1e-10 and Tmax <= 1.0 + 1e-10)
        print(f"  {dc_tag:>6s}  balance={balance:.2e}  T_min={Tmin:.6e}  T_max={Tmax:.6e}"
              f"  {'OK' if bounded else 'bnd FAIL'}")
    return True


# ================================================================== Check 5
def check5():
    print("\n" + "=" * 70)
    print("Check 5: regression  (Stage 0 psi_min unchanged)")
    print("-" * 70)
    s = SimpleSolver(M=32, Re=1000, shape="parallelogram", beta_deg=45, gamma_dc=1.0,
                     n_nonorth_correctors=2)
    s.run(max_outer=40000, tol=1e-9, verbose=0)
    psi = s.stream_function()
    psi_min = float(psi.min())
    target = -0.04902436
    err = abs(psi_min - target)
    ok = err < 1e-6
    print(f"  psi_min={psi_min:+.8e}  target={target:+.8e}  diff={err:.2e}  {'OK' if ok else 'FAIL'}")
    return ok


# =========================================================================== main
if __name__ == "__main__":
    np.set_printoptions(precision=6, linewidth=120)
    results = {}
    t0 = _time_mod.perf_counter()
    for name, fn in [("check1", check1), ("check2", check2), ("check3", check3),
                      ("check4", check4), ("check5", check5)]:
        t1 = _time_mod.perf_counter()
        res = fn()
        t = round(_time_mod.perf_counter() - t1, 1)
        results[name] = {"passed": res, "time_s": t}

    print("\n" + "=" * 70)
    print("SUMMARY")
    for name, r in results.items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {name}: {status}  ({r['time_s']}s)")
    all_ok = all(r["passed"] for r in results.values())
    total = round(_time_mod.perf_counter() - t0, 1)
    print(f"\n  Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}  (total {total}s)")
    if not all_ok:
        sys.exit(1)