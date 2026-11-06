"""Code-verification tests. Run with `python -m pytest -q` (or `python tests/test_solver.py`)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simplefvm import (FVM_grad, SimpleSolver, SparseDirect, Stencil2D, TDMA, ghia,  # noqa: E402
                       setUpMesh)


def _mesh(M=16, shape="rectangular"):
    X, Y, Xc, Yc = setUpMesh(M, 2.0, 1.2 if shape != "rectangular" else 2.0, 1.0, shape)
    return X, Y, Xc, Yc, Stencil2D(X, Y, Xc, Yc)


def test_geometry_closed_cells():
    """Face normals of every cell sum to zero and volumes add up to the domain area."""
    for shape in ("rectangular", "linear", "quadratic", "crazy"):
        X, Y, Xc, Yc, s = _mesh(24, shape)
        sx = s.Sx_x[1:] - s.Sx_x[:-1] + s.Sy_x[:, 1:] - s.Sy_x[:, :-1]
        sy = s.Sx_y[1:] - s.Sx_y[:-1] + s.Sy_y[:, 1:] - s.Sy_y[:, :-1]
        assert np.max(np.abs(sx)) < 1e-12 and np.max(np.abs(sy)) < 1e-12, shape
        area = np.trapezoid(Yc[:, -1], Xc[:, -1])
        assert abs(s.vol.sum() - area) < 1e-12, shape


def test_gauss_gradient_exact_for_linear_field():
    X, Y, Xc, Yc, s = _mesh(20)
    phi = 3.0 * X - 2.0 * Y + 1.0          # ghosts hold the exact face values
    gx, gy = FVM_grad(X, Y, Xc, Yc, s).compute_gradient(phi)
    assert np.allclose(gx[1:-1, 1:-1], 3.0) and np.allclose(gy[1:-1, 1:-1], -2.0)


def test_linear_solvers_agree():
    """ADI-TDMA (many sweeps) and the direct solver give the same answer on a Poisson problem."""
    M = 24
    shp = (M + 2, M + 2)
    A = {k: np.zeros(shp) for k in ("Aw", "Ae", "As", "An", "Ap", "Sp")}
    I = (slice(1, -1), slice(1, -1))
    for k in ("Aw", "Ae", "As", "An"):
        A[k][I] = 1.0
    A["Ap"][I] = 4.0
    A["Sp"][I] = 1.0 / M**2
    phi_t = np.zeros(shp)
    TDMA.solve_line(phi_t, **A, sweeps=1000)   # line-Jacobi ADI converges slowly on Poisson

    phi_d = SparseDirect.solve(np.zeros(shp), **A)
    assert np.max(np.abs(phi_t - phi_d)) < 1e-8


def test_cavity_re100_mass_conservation_and_ghia():
    s = SimpleSolver(M=32, Re=100)
    assert s.run(max_outer=5000, tol=1e-6, verbose=0)
    assert s.max_divergence() < 1e-10                 # continuity per cell
    err = ghia.compare(s, 100)
    assert err["u_rms"] < 0.02 and err["v_rms"] < 0.015, err


def test_deferred_correction_is_more_accurate_than_upwind():
    up = SimpleSolver(M=32, Re=100)
    dc = SimpleSolver(M=32, Re=100, gamma_dc=1.0)
    assert up.run(max_outer=5000, verbose=0) and dc.run(max_outer=5000, verbose=0)
    e_up, e_dc = ghia.compare(up, 100), ghia.compare(dc, 100)
    assert e_dc["u_rms"] < 0.25 * e_up["u_rms"], (e_up, e_dc)


def test_curved_lid_runs_and_conserves_mass():
    s = SimpleSolver(M=32, Re=100, H1=2.0, H2=1.2, shape="quadratic")
    assert s.run(max_outer=8000, tol=1e-6, verbose=0)
    assert s.max_divergence() < 1e-10


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
