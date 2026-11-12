"""Structured 2-D mesh with a (possibly curved) top wall.

Index convention used everywhere in this package (unchanged from the first version):

    i  -> x direction (west -> east),  j -> y direction (south -> north)
    cell-centred arrays have shape (M+2, M+2): interior cells 1..M, one ghost layer
    corner (vertex) arrays have shape (M+1, M+1)

Ghost "cells" sit on the boundary face centres, so a ghost value is the wall value.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def formfunction(x, shape: str, H1: float, H2: float, L: float):
    """Height of the top wall at position x (the cavity spans 0 <= y <= formfunction(x))."""
    if shape == "linear":
        return (H1 / 2) + (H2 / 2 - H1 / 2) * (x / L)
    if shape == "rectangular":
        return (H1 / 2) * np.ones_like(x)
    if shape == "quadratic":
        a = (H1 / 2 - H2 / 2) / L**2
        return a * (x - L) ** 2 + H2 / 2
    if shape == "crazy":
        return (H1 / 2 - H2 / 2) * np.sin(np.pi * x / L) + (H1 + H2) / 4
    raise ValueError(f"Unknown shape: {shape}")


def setUpMesh(M: int, H1: float, H2: float, L: float, shape: str
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y (cell centres incl. ghosts, (M+2, M+2)) and X_cor, Y_cor (vertices, (M+1, M+1)).

    Vertical grid lines are equally spaced in x; each one is divided into M equal
    cells between y = 0 and the top wall, so the mesh follows a curved lid.
    """
    x_nodes = np.linspace(0.0, L, M + 1)
    top = np.asarray(formfunction(x_nodes, shape, H1, H2, L), dtype=float)
    X_cor = np.repeat(x_nodes[:, None], M + 1, axis=1)
    Y_cor = top[:, None] * np.linspace(0.0, 1.0, M + 1)[None, :]

    X = np.zeros((M + 2, M + 2))
    Y = np.zeros_like(X)
    # interior centres (average of the four vertices)
    X[1:M + 1, 1:M + 1] = 0.25 * (X_cor[:-1, :-1] + X_cor[1:, :-1] + X_cor[1:, 1:] + X_cor[:-1, 1:])
    Y[1:M + 1, 1:M + 1] = 0.25 * (Y_cor[:-1, :-1] + Y_cor[1:, :-1] + Y_cor[1:, 1:] + Y_cor[:-1, 1:])
    # ghosts on the face centres: south (j = 0) and north (j = M+1)
    X[1:M + 1, 0] = 0.5 * (X_cor[:-1, 0] + X_cor[1:, 0])
    Y[1:M + 1, 0] = 0.5 * (Y_cor[:-1, 0] + Y_cor[1:, 0])
    X[1:M + 1, M + 1] = 0.5 * (X_cor[:-1, -1] + X_cor[1:, -1])
    Y[1:M + 1, M + 1] = 0.5 * (Y_cor[:-1, -1] + Y_cor[1:, -1])
    # ghosts on the face centres: west (i = 0) and east (i = M+1)
    X[0, 1:M + 1] = 0.5 * (X_cor[0, :-1] + X_cor[0, 1:])
    Y[0, 1:M + 1] = 0.5 * (Y_cor[0, :-1] + Y_cor[0, 1:])
    X[M + 1, 1:M + 1] = 0.5 * (X_cor[-1, :-1] + X_cor[-1, 1:])
    Y[M + 1, 1:M + 1] = 0.5 * (Y_cor[-1, :-1] + Y_cor[-1, 1:])
    # corner ghosts (never used by a stencil)
    X[0, 0], Y[0, 0] = X_cor[0, 0], Y_cor[0, 0]
    X[0, -1], Y[0, -1] = X_cor[0, -1], Y_cor[0, -1]
    X[-1, 0], Y[-1, 0] = X_cor[-1, 0], Y_cor[-1, 0]
    X[-1, -1], Y[-1, -1] = X_cor[-1, -1], Y_cor[-1, -1]
    return X, Y, X_cor, Y_cor


def unit_square(M: int, L: float = 1.0):
    """Square cavity of side L (the 'rectangular' shape has height H1/2)."""
    return setUpMesh(M, 2.0 * L, 2.0 * L, L, "rectangular")
