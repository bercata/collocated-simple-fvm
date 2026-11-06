"""Benchmark data of Ghia, Ghia & Shin (1982), J. Comput. Phys. 48, 387-411, Tables I and II.

Table I: u along the vertical line through the cavity centre (x = 0.5).
Table II: v along the horizontal line through the cavity centre (y = 0.5).

Table I was cross-checked against an independent transcription
(https://gist.github.com/ivan-pi/3e9326d18a366ffe6a8e5bfda6353219). The Re = 100 column of
Table II was checked against the Ghia markers plotted in arXiv:2407.10093 (Fig. 12).
The Re = 400 entry at x = 0.9063 (v = -0.23827) is widely suspected to be a misprint in the
original table; it is kept as published and flagged in the plots.
"""
import numpy as np

Y_U = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
                0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
U = {
    100: np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
                   -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192,
                   -0.03717, 0.00000]),
    400: np.array([1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135,
                   -0.11477, -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266,
                   -0.08186, 0.00000]),
    1000: np.array([1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702,
                    -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196,
                    -0.18109, 0.00000]),
}

X_V = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
                0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
V = {
    100: np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445,
                   -0.24533, 0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10091,
                   0.09233, 0.00000]),
    400: np.array([0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993,
                   -0.38598, 0.05186, 0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713,
                   0.18360, 0.00000]),
    1000: np.array([0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665,
                    -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012,
                    0.27485, 0.00000]),
}
SUSPECT_V = {400: [5]}   # index into X_V of the suspected misprint


def compare(solver, Re):
    """RMS and max deviation of the solver's centre-line profiles from Ghia (interior points)."""
    y, u = solver.centerline_u()
    x, v = solver.centerline_v()
    order_u, order_v = np.argsort(y), np.argsort(x)
    u_i = np.interp(Y_U, y[order_u], u[order_u])
    v_i = np.interp(X_V, x[order_v], v[order_v])
    mask_u = (Y_U > 0) & (Y_U < 1)
    mask_v = (X_V > 0) & (X_V < 1)
    for k in SUSPECT_V.get(Re, []):
        mask_v[k] = False
    du = u_i[mask_u] - U[Re][mask_u]
    dv = v_i[mask_v] - V[Re][mask_v]
    return {"u_rms": float(np.sqrt(np.mean(du ** 2))), "u_max": float(np.max(np.abs(du))),
            "v_rms": float(np.sqrt(np.mean(dv ** 2))), "v_max": float(np.max(np.abs(dv)))}
