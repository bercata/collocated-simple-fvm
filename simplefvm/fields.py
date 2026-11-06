"""Field container (same name and fields as the course version, plus stored face mass fluxes)."""
from __future__ import annotations

import numpy as np


class SimpleFields2D:
    """Cell-centred fields with one ghost layer; ghosts hold wall values.

    The lid (north ghost row) moves with speed ``lid_velocity`` along the local wall tangent.
    For a flat lid this is u = lid_velocity, v = 0, exactly as in the course version.
    ``vars`` holds extra transported scalars (for example Z and C for a flamelet extension).
    """

    def __init__(self, X, Y, X_cor, Y_cor, rho0=1.0, mu0=1.0, lid_velocity=1.0):
        self.X, self.Y, self.X_cor, self.Y_cor = X, Y, X_cor, Y_cor
        self.m, self.n = X.shape
        M = self.m - 2
        shp = (self.m, self.n)
        self.U = np.zeros(shp)
        self.V = np.zeros(shp)
        self.P = np.zeros(shp)
        self.p_prime = np.zeros(shp)
        self.mu = np.full(shp, float(mu0))
        self.rho = np.full(shp, float(rho0))
        self.DpU = np.zeros(shp)
        self.DpV = np.zeros(shp)
        self.Ap_u = np.ones(shp)
        self.Ap_v = np.ones(shp)
        # face mass fluxes from the Rhie-Chow step: x-faces (M+1, M), y-faces (M, M+1)
        self.mflux_x = np.zeros((M + 1, M))
        self.mflux_y = np.zeros((M, M + 1))

        # moving lid: unit tangent of each top face
        tx = X_cor[1:, -1] - X_cor[:-1, -1]
        ty = Y_cor[1:, -1] - Y_cor[:-1, -1]
        t = np.hypot(tx, ty)
        self.U[1:-1, -1] = lid_velocity * tx / t
        self.V[1:-1, -1] = lid_velocity * ty / t
        self.U[0, -1] = self.U[-1, -1] = lid_velocity   # corner ghosts (unused)
        self.lid_velocity = lid_velocity

        self.vars = {}

    def add_field(self, name, initial=0.0):
        self.vars[name] = np.full((self.m, self.n), float(initial))

    def get_field(self, name):
        return self.vars[name]
