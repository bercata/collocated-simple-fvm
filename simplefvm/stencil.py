"""Stencil and pre-computed finite-volume geometry.

The course version recomputed every face length and distance inside the double loops of
every operator, on every iteration. Here they are computed once, as arrays, when the
Stencil2D is built. The per-cell methods (cell_type, get_neighbors, get_face_centers) are
kept for readability and for user code that wants them.

Face storage
------------
x-faces (between cell (i, j) and (i+1, j)): arrays of shape (M+1, M), index [i, j-1],
    i = 0..M is the vertical grid line; i = 0 and i = M are the west/east walls.
y-faces (between cell (i, j) and (i, j+1)): arrays of shape (M, M+1), index [i-1, j],
    j = 0..M is the grid line; j = 0 is the bottom wall, j = M is the lid.
Face normal vectors (Sx, Sy) have the face length as magnitude and point to +i or +j.
"""
from __future__ import annotations

import numpy as np


class Coordicate2D:  # name kept from the course version
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Stencil2D:
    def __init__(self, X, Y, X_cor=None, Y_cor=None):
        self.X = X
        self.Y = Y
        self.nx, self.ny = X.shape
        self.m, self.n = X.shape  # aliases used by BCManager
        self.M = self.nx - 2
        if X_cor is not None:
            self._build_metrics(X_cor, Y_cor)

    # ---------------------------------------------------------------- geometry
    def _build_metrics(self, Xc, Yc):
        M = self.M
        X, Y = self.X, self.Y
        self.X_cor, self.Y_cor = Xc, Yc

        # x-faces: edge from vertex (i, j-1) to (i, j)
        dx = Xc[:, 1:] - Xc[:, :-1]
        dy = Yc[:, 1:] - Yc[:, :-1]
        self.Sx_x, self.Sx_y = dy, -dx                      # normal towards +i
        self.Sx_mag = np.hypot(dx, dy)
        px, py = X[0:M + 1, 1:M + 1], Y[0:M + 1, 1:M + 1]   # owner (west side)
        nx_, ny_ = X[1:M + 2, 1:M + 1], Y[1:M + 2, 1:M + 1]  # neighbour (east side)
        self.dx_x, self.dx_y = nx_ - px, ny_ - py
        self.dx_mag = np.hypot(self.dx_x, self.dx_y)

        # y-faces: edge from vertex (i-1, j) to (i, j)
        dx = Xc[1:, :] - Xc[:-1, :]
        dy = Yc[1:, :] - Yc[:-1, :]
        self.Sy_x, self.Sy_y = -dy, dx                      # normal towards +j
        self.Sy_mag = np.hypot(dx, dy)
        px, py = X[1:M + 1, 0:M + 1], Y[1:M + 1, 0:M + 1]   # owner (south side)
        nx_, ny_ = X[1:M + 1, 1:M + 2], Y[1:M + 1, 1:M + 2]  # neighbour (north side)
        self.dy_x, self.dy_y = nx_ - px, ny_ - py
        self.dy_mag = np.hypot(self.dy_x, self.dy_y)

        # cell volumes (shoelace, corners SW -> SE -> NE -> NW)
        x0, y0 = Xc[:-1, :-1], Yc[:-1, :-1]
        x1, y1 = Xc[1:, :-1], Yc[1:, :-1]
        x2, y2 = Xc[1:, 1:], Yc[1:, 1:]
        x3, y3 = Xc[:-1, 1:], Yc[:-1, 1:]
        self.vol = 0.5 * np.abs(x0 * y1 + x1 * y2 + x2 * y3 + x3 * y0
                                - y0 * x1 - y1 * x2 - y2 * x3 - y3 * x0)

        # masks of interior faces (walls are False)
        self.x_interior = np.zeros((M + 1, M), bool)
        self.x_interior[1:M, :] = True
        self.y_interior = np.zeros((M, M + 1), bool)
        self.y_interior[:, 1:M] = True

    # ---------------------------------------------------------- per-cell helpers
    def cell_type(self, i: int, j: int) -> str:
        if 0 < i < self.nx - 1 and 0 < j < self.ny - 1:
            return "interior"
        if j == self.ny - 1:
            return "NW" if i == 0 else "NE" if i == self.nx - 1 else "north"
        if j == 0:
            return "SW" if i == 0 else "SE" if i == self.nx - 1 else "south"
        if i == 0:
            return "west"
        if i == self.nx - 1:
            return "east"
        raise ValueError(f"Unexpected cell indices: ({i},{j})")

    def get_neighbors(self, i: int, j: int) -> dict:
        nbr = {}
        if i > 0:
            nbr["W"] = (i - 1, j)
        if i < self.nx - 1:
            nbr["E"] = (i + 1, j)
        if j > 0:
            nbr["S"] = (i, j - 1)
        if j < self.ny - 1:
            nbr["N"] = (i, j + 1)
        return nbr

    def get_face_centers(self, i: int, j: int) -> dict:
        faces = {}
        for d, (ni, nj) in self.get_neighbors(i, j).items():
            faces[d] = Coordicate2D(0.5 * (self.X[i, j] + self.X[ni, nj]),
                                    0.5 * (self.Y[i, j] + self.Y[ni, nj]))
        return faces


# ---------------------------------------------------------------- face helpers
def x_face_avg(phi):
    """Arithmetic mean of owner/neighbour across every x-face, shape (M+1, M)."""
    return 0.5 * (phi[:-1, 1:-1] + phi[1:, 1:-1])


def y_face_avg(phi):
    """Arithmetic mean across every y-face, shape (M, M+1)."""
    return 0.5 * (phi[1:-1, :-1] + phi[1:-1, 1:])


def x_face_value(phi):
    """Face value with the wall faces taken from the ghost cell (the wall value)."""
    f = x_face_avg(phi)
    f[0, :] = phi[0, 1:-1]
    f[-1, :] = phi[-1, 1:-1]
    return f


def y_face_value(phi):
    f = y_face_avg(phi)
    f[:, 0] = phi[1:-1, 0]
    f[:, -1] = phi[1:-1, -1]
    return f
