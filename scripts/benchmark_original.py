"""Cost of one SIMPLE outer iteration: first-version code vs. this package (same mesh).

The first-version code is executed from original/simple_original.py (cells 1-15 only).
"""
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

src = open(os.path.join(ROOT, "original", "simple_original.py"), encoding="utf-8").read()
src = src.split("# %% cell 16")[0]          # definitions only, not the run cell
ns = {}
exec(compile(src, "simple_original", "exec"), ns)


def one_outer_original(M):
    X, Y, Xc, Yc = ns["setUpMesh"](M, 2.0, 2.0, 1.0, "rectangular")
    fields = ns["SimpleFields2D"](X, Y, Xc, Yc, rho0=1.0, mu0=0.01)
    st = ns["Stencil2D"](X, Y)
    mom = ns["MomentumEquationSolver"](st, X, Y, Xc, Yc)
    rc = ns["RhieChowPressureCorrection"](st, X, Y, Xc, Yc, reference=(1, 1, 0.0), tol=1e-6, max_inner=551)
    corr = ns["PressureVelocityCorrector"](st, Xc, Yc)
    t = time.perf_counter()
    U, Ap_u, *_ = mom.solve(fields, alpha=0.7, component="u"); fields.U, fields.Ap_u = U, Ap_u
    V, Ap_v, *_ = mom.solve(fields, alpha=0.7, component="v"); fields.V, fields.Ap_v = V, Ap_v
    fields.DpU = np.zeros_like(Ap_u); fields.DpU[1:-1, 1:-1] = 1.0 / (Ap_u[1:-1, 1:-1] + 1e-20)
    fields.DpV = np.zeros_like(Ap_v); fields.DpV[1:-1, 1:-1] = 1.0 / (Ap_v[1:-1, 1:-1] + 1e-20)
    p = rc.solve(fields)
    corr.apply(fields, p, alpha_p=0.3)
    return time.perf_counter() - t


def one_outer_new(M, n=50):
    from simplefvm import SimpleSolver
    s = SimpleSolver(M=M, Re=100)
    s.step()
    t = time.perf_counter()
    for _ in range(n):
        s.step()
    return (time.perf_counter() - t) / n


if __name__ == "__main__":
    M = 20
    t_old = one_outer_original(M)
    t_new = one_outer_new(M)
    print(f"{M}x{M}: first-version code {t_old:.2f} s per outer iteration, "
          f"this package {t_new * 1000:.1f} ms, speed-up {t_old / t_new:.0f}x")
