"""Linear-system assembly and boundary-condition helpers (first-version API, vectorised)."""
from __future__ import annotations

import numpy as np


class FVLinearSystem:
    """Holds the five-point coefficients (Aw, Ae, As, An, Ap, Sp) on the padded grid.

    Workflow (unchanged):
        1) assemble_raw(...)            first-order upwind convection + central diffusion
        2) boundary conditions          walls enter through the ghost values
        3) apply_under_relax(phi_old)   Patankar under-relaxation
    """

    def __init__(self, Nx: int, Ny: int):
        self.Nx, self.Ny = Nx, Ny
        shp = (Nx, Ny)
        self.Aw = np.zeros(shp)
        self.Ae = np.zeros(shp)
        self.As = np.zeros(shp)
        self.An = np.zeros(shp)
        self.Ap = np.zeros(shp)
        self.Sp = np.zeros(shp)

    def assemble_raw(self, flux_e, flux_w, flux_n, flux_s,
                     diff_e, diff_w, diff_n, diff_s,
                     source, X_cor, Y_cor, Dp=None, vol=None):
        """Fill the coefficients without under-relaxation.

        ``source`` is a volumetric source (per unit volume); ``Dp`` is the pressure gradient
        component, which enters as -Dp * V. ``vol`` (interior cell volumes) can be passed to
        avoid recomputing it.
        """
        I = (slice(1, -1), slice(1, -1))
        for arr in (self.Aw, self.Ae, self.As, self.An, self.Ap, self.Sp):
            arr.fill(0.0)
        if vol is None:
            from .stencil import Stencil2D  # noqa: F401  (vol from corners)
            x0, y0 = X_cor[:-1, :-1], Y_cor[:-1, :-1]
            x1, y1 = X_cor[1:, :-1], Y_cor[1:, :-1]
            x2, y2 = X_cor[1:, 1:], Y_cor[1:, 1:]
            x3, y3 = X_cor[:-1, 1:], Y_cor[:-1, 1:]
            vol = 0.5 * np.abs(x0 * y1 + x1 * y2 + x2 * y3 + x3 * y0 - y0 * x1 - y1 * x2 - y2 * x3 - y3 * x0)

        self.Ae[I] = diff_e[I] + np.maximum(-flux_e[I], 0.0)
        self.Aw[I] = diff_w[I] + np.maximum(flux_w[I], 0.0)
        self.An[I] = diff_n[I] + np.maximum(-flux_n[I], 0.0)
        self.As[I] = diff_s[I] + np.maximum(flux_s[I], 0.0)
        # Ap = sum of neighbour coefficients (first version). This is consistent because the
        # convective fluxes are the Rhie-Chow fluxes, which satisfy continuity cell by cell.
        self.Ap[I] = self.Ae[I] + self.Aw[I] + self.An[I] + self.As[I]
        self.Sp[I] = source[I] * vol
        if Dp is not None:
            self.Sp[I] -= Dp[I] * vol
        return self.Aw, self.Ae, self.As, self.An, self.Ap, self.Sp

    def apply_under_relax(self, phi_old, alpha: float = 1.0):
        """Ap <- Ap / alpha;  Sp <- Sp + (1 - alpha)/alpha * Ap_raw * phi_old."""
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        Ap_raw = self.Ap.copy()
        self.Ap[:] = Ap_raw / alpha
        self.Sp[:] += (1.0 - alpha) / alpha * Ap_raw * phi_old

    def residual(self, phi):
        """Algebraic residual on interior cells: Ap phi - sum(a_nb phi_nb) - Sp."""
        c = (slice(1, -1), slice(1, -1))
        return (self.Ap[c] * phi[c] - self.Aw[c] * phi[:-2, 1:-1] - self.Ae[c] * phi[2:, 1:-1]
                - self.As[c] * phi[1:-1, :-2] - self.An[c] * phi[1:-1, 2:] - self.Sp[c])


class BCManager:
    """Boundary conditions for additional transported scalars (future flamelet Z, C).

    Directions follow the package convention: i = 0 west, i = nx-1 east, j = 0 south,
    j = ny-1 north. (The first version mapped i = 0 to 'north'; that is corrected here.)
    Supported per-side types: 'dirichlet' (value), 'neumann' (zero gradient).
    """

    def __init__(self, bc_dict):
        self.bc_dict = bc_dict  # e.g. {'west': {'type': 'dirichlet', 'value': 1.0}, ...}

    def apply_ghosts(self, phi):
        sides = {"west": (0, slice(1, -1), 1), "east": (-1, slice(1, -1), -2),
                 "south": (slice(1, -1), 0, 1), "north": (slice(1, -1), -1, -2)}
        for side, bc in self.bc_dict.items():
            a, b, inner = sides[side]
            if bc["type"] == "dirichlet":
                phi[a, b] = bc["value"]
            elif bc["type"] == "neumann":
                phi[a, b] = phi[inner, b] if side in ("west", "east") else phi[a, inner]
            else:
                raise ValueError(f"Unknown BC type {bc['type']!r} on {side}")
        return phi
