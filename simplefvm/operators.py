"""Finite-volume operators (vectorised; same class names and call signatures as the course version)."""
from __future__ import annotations

import numpy as np

from .stencil import Stencil2D, x_face_avg, y_face_avg, x_face_value, y_face_value


def _interior(a):
    return a[1:-1, 1:-1]


class FVM_grad:
    """Cell-centre gradient by the Gauss theorem: grad(phi) = sum_f phi_f S_f / V.

    Face values are the arithmetic mean of the two cells; on a wall face the ghost value
    (which sits on the face centre) is used directly. On a uniform rectangular mesh this is
    the same central difference as the course version.
    """

    def __init__(self, X, Y, X_cor, Y_cor, stencil: Stencil2D):
        self.s = stencil
        self.m, self.n = X.shape

    def compute_gradient(self, phi):
        s = self.s
        fx = x_face_value(phi)            # (M+1, M)
        fy = y_face_value(phi)            # (M, M+1)
        gx = np.zeros((self.m, self.n))
        gy = np.zeros((self.m, self.n))
        sum_x = (fx[1:] * s.Sx_x[1:] - fx[:-1] * s.Sx_x[:-1]
                 + fy[:, 1:] * s.Sy_x[:, 1:] - fy[:, :-1] * s.Sy_x[:, :-1])
        sum_y = (fx[1:] * s.Sx_y[1:] - fx[:-1] * s.Sx_y[:-1]
                 + fy[:, 1:] * s.Sy_y[:, 1:] - fy[:, :-1] * s.Sy_y[:, :-1])
        gx[1:-1, 1:-1] = sum_x / s.vol
        gy[1:-1, 1:-1] = sum_y / s.vol
        return gx, gy


class FVM_div:
    """Convective face mass fluxes F_f = (rho u . S)_f for the momentum (or any scalar) equation.

    If face mass fluxes from the Rhie-Chow step are supplied (``mflux``), they are used: they are
    the fluxes that satisfy continuity, so the transported quantity is conserved cell by cell.
    Without them (first iteration) the course-version estimate from averaged cell velocities is
    used. Wall faces carry no mass.
    """

    def __init__(self, X, Y, X_cor, Y_cor, stencil: Stencil2D):
        self.s = stencil
        self.m, self.n = X.shape

    def face_mass_fluxes(self, U, V, rho):
        s = self.s
        mx = x_face_avg(rho) * (x_face_avg(U) * s.Sx_x + x_face_avg(V) * s.Sx_y)
        my = y_face_avg(rho) * (y_face_avg(U) * s.Sy_x + y_face_avg(V) * s.Sy_y)
        mx[~s.x_interior] = 0.0
        my[~s.y_interior] = 0.0
        return mx, my

    def compute_face_fluxes(self, phi, U, V, rho, mflux=None):
        mx, my = mflux if mflux is not None else self.face_mass_fluxes(U, V, rho)
        out = {k: np.zeros((self.m, self.n)) for k in ("flux_e", "flux_w", "flux_n", "flux_s")}
        out["flux_e"][1:-1, 1:-1] = mx[1:]
        out["flux_w"][1:-1, 1:-1] = mx[:-1]
        out["flux_n"][1:-1, 1:-1] = my[:, 1:]
        out["flux_s"][1:-1, 1:-1] = my[:, :-1]
        return out


class FVM_laplacian:
    """Diffusive conductances D_f = Gamma_f |S_f| / |d_f| (orthogonal part only).

    On a wall face |d_f| is the distance from the cell centre to the wall (half a cell), which
    gives the correct one-sided wall gradient. Non-orthogonal correction is not included; it
    is exact on the rectangular cavity and an approximation on the curved-lid meshes.
    """

    def __init__(self, X, Y, X_cor, Y_cor, stencil: Stencil2D):
        self.s = stencil
        self.m, self.n = X.shape

    def compute_face_diffusion(self, phi, Gamma):
        s = self.s
        Dx = x_face_avg(Gamma) * s.Sx_mag / s.dx_mag
        Dy = y_face_avg(Gamma) * s.Sy_mag / s.dy_mag
        out = {k: np.zeros((self.m, self.n)) for k in ("Diff_e", "Diff_w", "Diff_n", "Diff_s")}
        out["Diff_e"][1:-1, 1:-1] = Dx[1:]
        out["Diff_w"][1:-1, 1:-1] = Dx[:-1]
        out["Diff_n"][1:-1, 1:-1] = Dy[:, 1:]
        out["Diff_s"][1:-1, 1:-1] = Dy[:, :-1]
        return out


def cell_area(X_cor, Y_cor, i, j):
    """Area of interior cell (i, j) by the shoelace formula (scalar helper, kept for user code)."""
    x0, y0 = X_cor[i - 1, j - 1], Y_cor[i - 1, j - 1]
    x1, y1 = X_cor[i, j - 1], Y_cor[i, j - 1]
    x2, y2 = X_cor[i, j], Y_cor[i, j]
    x3, y3 = X_cor[i - 1, j], Y_cor[i - 1, j]
    return 0.5 * abs(x0 * y1 + x1 * y2 + x2 * y3 + x3 * y0 - y0 * x1 - y1 * x2 - y2 * x3 - y3 * x0)
