"""Finite-volume operators (vectorised; same class names and call signatures as the first version)."""
from __future__ import annotations

import numpy as np

from .stencil import Stencil2D, x_face_avg, y_face_avg, x_face_value, y_face_value


def _interior(a):
    return a[1:-1, 1:-1]


class FVM_grad:
    """Cell-centre gradient by the Gauss theorem: grad(phi) = sum_f phi_f S_f / V.

    Face values are the arithmetic mean of the two cells; on a wall face the ghost value
    (which sits on the face centre) is used directly. On a uniform rectangular mesh this is
    the same central difference as the first version.
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
    Without them (first iteration) the first-version estimate from averaged cell velocities is
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
    """Diffusive conductances using the over-relaxed decomposition D_f = Gamma_f |S_f|^2 / (S_f.d_f).

    On an orthogonal mesh S_f.d_f = |S_f| |d_f| and this reduces to the standard form.
    Non-orthogonal correction is added as an explicit source term via
    ``nonorthogonal_correction_source``.
    """

    def __init__(self, X, Y, X_cor, Y_cor, stencil: Stencil2D):
        self.s = stencil
        self.m, self.n = X.shape

    def _conductance_x(self, Gamma):
        s = self.s
        denom = np.where(s.x_interior, s.Sx_dot_dx, 1.0)
        Dx = x_face_avg(Gamma) * s.Sx_mag * s.Sx_mag / denom
        Dx[~s.x_interior] = x_face_avg(Gamma)[~s.x_interior] * s.Sx_mag[~s.x_interior] ** 2 / s.Sx_dot_dx[~s.x_interior]
        return Dx

    def _conductance_y(self, Gamma):
        s = self.s
        denom = np.where(s.y_interior, s.Sy_dot_dy, 1.0)
        Dy = y_face_avg(Gamma) * s.Sy_mag * s.Sy_mag / denom
        Dy[~s.y_interior] = y_face_avg(Gamma)[~s.y_interior] * s.Sy_mag[~s.y_interior] ** 2 / s.Sy_dot_dy[~s.y_interior]
        return Dy

    def compute_face_diffusion(self, phi, Gamma):
        s = self.s
        Dx = self._conductance_x(Gamma)
        Dy = self._conductance_y(Gamma)
        out = {k: np.zeros((self.m, self.n)) for k in ("Diff_e", "Diff_w", "Diff_n", "Diff_s")}
        out["Diff_e"][1:-1, 1:-1] = Dx[1:]
        out["Diff_w"][1:-1, 1:-1] = Dx[:-1]
        out["Diff_n"][1:-1, 1:-1] = Dy[:, 1:]
        out["Diff_s"][1:-1, 1:-1] = Dy[:, :-1]
        return out

    def nonorthogonal_correction_source(self, grad_phi_x, grad_phi_y, Gamma):
        """Explicit non-orthogonal diffusion source (deferred correction) per interior cell.

        grad_phi_x, grad_phi_y: cell-centred gradient components, padded (M+2, M+2).
        Gamma: diffusivity field, padded (M+2, M+2).
        Returns array shape (M+2, M+2) with the divergence in interior cells.
        """
        s = self.s
        gx = grad_phi_x
        gy = grad_phi_y
        eps = 1e-30

        # x-faces
        sx, sy = s.Sx_x, s.Sx_y
        sm2 = s.Sx_mag * s.Sx_mag
        alpha_x = np.divide(sm2, s.Sx_dot_dx + eps)
        noc_x = sx - alpha_x * s.dx_x
        noc_y = sy - alpha_x * s.dx_y
        gx_f = 0.5 * (gx[:-1, 1:-1] + gx[1:, 1:-1])
        gy_f = 0.5 * (gy[:-1, 1:-1] + gy[1:, 1:-1])
        flux_noc_x = 0.5 * (Gamma[:-1, 1:-1] + Gamma[1:, 1:-1]) * (noc_x * gx_f + noc_y * gy_f)
        flux_noc_x[~s.x_interior] = 0.0

        # y-faces
        sx, sy = s.Sy_x, s.Sy_y
        sm2 = s.Sy_mag * s.Sy_mag
        alpha_y = np.divide(sm2, s.Sy_dot_dy + eps)
        noc_x = sx - alpha_y * s.dy_x
        noc_y = sy - alpha_y * s.dy_y
        gx_f = 0.5 * (gx[1:-1, :-1] + gx[1:-1, 1:])
        gy_f = 0.5 * (gy[1:-1, :-1] + gy[1:-1, 1:])
        flux_noc_y = 0.5 * (Gamma[1:-1, :-1] + Gamma[1:-1, 1:]) * (noc_x * gx_f + noc_y * gy_f)
        flux_noc_y[~s.y_interior] = 0.0

        src = np.zeros((self.m, self.n))
        src[1:-1, 1:-1] = (flux_noc_x[1:] - flux_noc_x[:-1]
                           + flux_noc_y[:, 1:] - flux_noc_y[:, :-1])
        return src


def cell_area(X_cor, Y_cor, i, j):
    """Area of interior cell (i, j) by the shoelace formula (scalar helper, kept for user code)."""
    x0, y0 = X_cor[i - 1, j - 1], Y_cor[i - 1, j - 1]
    x1, y1 = X_cor[i, j - 1], Y_cor[i, j - 1]
    x2, y2 = X_cor[i, j], Y_cor[i, j]
    x3, y3 = X_cor[i - 1, j], Y_cor[i - 1, j]
    return 0.5 * abs(x0 * y1 + x1 * y2 + x2 * y3 + x3 * y0 - y0 * x1 - y1 * x2 - y2 * x3 - y3 * x0)
