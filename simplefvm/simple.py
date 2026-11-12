"""SIMPLE outer loop (the same sequence as the first version's run cell)."""
from __future__ import annotations

import time

import numpy as np

from .fields import SimpleFields2D
from .mesh import setUpMesh
from .solvers import (ConvergenceMonitor, MomentumEquationSolver, PressureVelocityCorrector,
                      RhieChowPressureCorrection)
from .stencil import Stencil2D


class SimpleSolver:
    """Lid-driven cavity with an optional curved lid.

    Re = rho * U_lid * L / mu, with L the cavity width.
    """

    def __init__(self, M=64, Re=100.0, L=1.0, H1=2.0, H2=2.0, shape="rectangular",
                 rho=1.0, lid_velocity=1.0, alpha_u=0.7, alpha_p=0.3, inner_sweeps=2,
                 p_solver="direct", gamma_dc=0.0):
        self.M, self.Re, self.L = M, Re, L
        self.alpha_u, self.alpha_p = alpha_u, alpha_p
        X, Y, Xc, Yc = setUpMesh(M, H1, H2, L, shape)
        self.X, self.Y, self.X_cor, self.Y_cor = X, Y, Xc, Yc
        mu = rho * lid_velocity * L / Re
        self.fields = SimpleFields2D(X, Y, Xc, Yc, rho0=rho, mu0=mu, lid_velocity=lid_velocity)
        self.stencil = Stencil2D(X, Y, Xc, Yc)
        self.momentum = MomentumEquationSolver(self.stencil, X, Y, Xc, Yc, inner_sweeps=inner_sweeps,
                                               gamma_dc=gamma_dc)
        c = (M // 2 + 1, M // 2 + 1)
        self.pcorr = RhieChowPressureCorrection(self.stencil, X, Y, Xc, Yc,
                                                reference=(c[0], c[1], 0.0), linear_solver=p_solver)
        self.corrector = PressureVelocityCorrector(self.stencil, Xc, Yc, gauge=c)
        self.mass_scale = rho * lid_velocity * L        # lid-driven mass-flux scale
        self.history = {"u": [], "v": [], "mass": []}
        self.wall_time = 0.0
        self.iterations = 0

    def step(self):
        f = self.fields
        _, f.Ap_u, *_ = self.momentum.solve(f, alpha=self.alpha_u, component="u")
        _, f.Ap_v, *_ = self.momentum.solve(f, alpha=self.alpha_u, component="v")
        p_prime = self.pcorr.solve(f)
        self.corrector.apply(f, p_prime, alpha_p=self.alpha_p)
        r = (self.momentum.residual["u"], self.momentum.residual["v"],
             self.pcorr.mass_residual / self.mass_scale)
        for k, v in zip(("u", "v", "mass"), r):
            self.history[k].append(v)
        return r

    def run(self, max_outer=20000, tol=1e-6, verbose=500):
        mon = ConvergenceMonitor(atol=tol, rtol=0.0, max_it=max_outer + 1)
        t0 = time.perf_counter()
        converged = False
        for it in range(1, max_outer + 1):
            r = self.step()
            if verbose and it % verbose == 0:
                print(f"  it {it:6d}  res u {r[0]:.2e}  v {r[1]:.2e}  mass {r[2]:.2e}")
            if it > 10 and mon.record(max(r)):
                converged = True
                break
        self.iterations = it
        self.wall_time = time.perf_counter() - t0
        self.converged = converged
        return converged

    # ------------------------------------------------------------ post-processing
    def centerline_u(self):
        """u along the vertical centre line x = L/2 (interpolated between the two middle columns
        when M is even), with the wall values added. Returns (y, u)."""
        f, M = self.fields, self.M
        if M % 2 == 0:
            k = M // 2
            u = 0.5 * (f.U[k, :] + f.U[k + 1, :])
            y = 0.5 * (self.Y[k, :] + self.Y[k + 1, :])
        else:
            k = M // 2 + 1
            u, y = f.U[k, :], self.Y[k, :]
        return y, u

    def centerline_v(self):
        """v along the horizontal centre line y = H/2. Returns (x, v)."""
        f, M = self.fields, self.M
        if M % 2 == 0:
            k = M // 2
            v = 0.5 * (f.V[:, k] + f.V[:, k + 1])
            x = 0.5 * (self.X[:, k] + self.X[:, k + 1])
        else:
            k = M // 2 + 1
            v, x = f.V[:, k], self.X[:, k]
        return x, v

    def max_divergence(self):
        """Largest per-cell net mass outflow of the stored face fluxes, scaled by rho U L."""
        mx, my = self.fields.mflux_x, self.fields.mflux_y
        div = mx[1:] - mx[:-1] + my[:, 1:] - my[:, :-1]
        return float(np.max(np.abs(div)) / self.mass_scale)

    def stream_function(self):
        """Stream function on the vertices from the x-face mass fluxes (psi = 0 on the walls)."""
        mx = self.fields.mflux_x / self.fields.rho[0, 0]
        psi = np.zeros_like(self.X_cor)
        psi[:, 1:] = np.cumsum(mx, axis=1)      # psi = integral of u dy up each vertical grid line
        return psi
