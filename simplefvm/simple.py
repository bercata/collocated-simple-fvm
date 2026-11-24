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
                 p_solver="direct", gamma_dc=0.0, beta_deg=None,
                 n_nonorth_correctors=0, consistent_rc=True, body_force_rc=False,
                 mass_scale=None, vel_scale=None):
        self.M, self.Re, self.L = M, Re, L
        self.alpha_u, self.alpha_p = alpha_u, alpha_p
        self.shape = shape
        self.beta_deg = beta_deg
        self.H1, self.H2 = H1, H2
        side_len = H1 / 2.0 if shape == "parallelogram" else None
        X, Y, Xc, Yc = setUpMesh(M, H1, H2, L, shape, beta_deg=beta_deg,
                                  side_length=side_len)
        self.X, self.Y, self.X_cor, self.Y_cor = X, Y, Xc, Yc
        mu = rho * lid_velocity * L / Re if Re != 0 else 0.0
        self.fields = SimpleFields2D(X, Y, Xc, Yc, rho0=rho, mu0=mu, lid_velocity=lid_velocity)
        self.stencil = Stencil2D(X, Y, Xc, Yc)
        self.momentum = MomentumEquationSolver(self.stencil, X, Y, Xc, Yc, inner_sweeps=inner_sweeps,
                                               gamma_dc=gamma_dc)
        c = (M // 2 + 1, M // 2 + 1)
        self.pcorr = RhieChowPressureCorrection(self.stencil, X, Y, Xc, Yc,
                                                reference=(c[0], c[1], 0.0), linear_solver=p_solver,
                                                n_nonorth_correctors=n_nonorth_correctors,
                                                consistent_rc=consistent_rc,
                                                body_force_rc=body_force_rc)
        self.corrector = PressureVelocityCorrector(self.stencil, Xc, Yc, gauge=c)
        if mass_scale is not None:
            self.mass_scale = mass_scale
        else:
            self.mass_scale = rho * lid_velocity * L
        self.history = {"u": [], "v": [], "mass": []}
        self.wall_time = 0.0
        self.iterations = 0
        self.status = "not run"
        self._grid_sequencing_first = False

    @classmethod
    def init_from(cls, coarse, M=None, alpha_u=None, alpha_p=None):
        """Create a finer solver initialised from a coarser converged solver.

        U, V, P are interpolated bilinearly in logical (i, j) index coordinates.
        Face mass fluxes are rebuilt from the interpolated velocities.
        """
        if M is None:
            M = 2 * coarse.M
        if coarse.shape != "parallelogram":
            raise ValueError("init_from only supported for parallelogram meshes")
        alpha_u = alpha_u if alpha_u is not None else coarse.alpha_u
        alpha_p = alpha_p if alpha_p is not None else coarse.alpha_p

        fine = cls(
            M=M, Re=coarse.Re, L=coarse.L,
            H1=coarse.H1, H2=coarse.H2,
            shape=coarse.shape,
            rho=coarse.fields.rho[1, 1],
            lid_velocity=coarse.fields.lid_velocity,
            alpha_u=alpha_u, alpha_p=alpha_p,
            gamma_dc=coarse.momentum.gamma_dc,
            beta_deg=coarse.beta_deg,
            n_nonorth_correctors=coarse.pcorr.n_nonorth_correctors,
            consistent_rc=coarse.pcorr.consistent_rc,
        )
        fine._grid_sequencing_first = True

        fc = coarse.fields
        ff = fine.fields
        Mc = coarse.M
        L = coarse.L
        side_len = coarse.H1 / 2.0
        cosb = np.cos(np.radians(coarse.beta_deg))
        sinb = np.sin(np.radians(coarse.beta_deg))

        Xf, Yf = fine.X, fine.Y
        eta_f = Yf / (side_len * sinb)
        xi_f = (Xf - Yf * cosb / sinb) / L

        i_float = np.clip(xi_f * Mc, 0.0, float(Mc))
        j_float = np.clip(eta_f * Mc, 0.0, float(Mc))
        i_lo = np.floor(i_float).astype(int)
        j_lo = np.floor(j_float).astype(int)
        i_hi = np.clip(i_lo + 1, 0, Mc + 1)
        j_hi = np.clip(j_lo + 1, 0, Mc + 1)
        tx = i_float - i_lo
        ty = j_float - j_lo

        def interp(cf):
            v00 = cf[i_lo, j_lo]
            v10 = cf[i_hi, j_lo]
            v01 = cf[i_lo, j_hi]
            v11 = cf[i_hi, j_hi]
            return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 \
                 + (1.0 - tx) * ty * v01 + tx * ty * v11

        ff.U[...] = interp(fc.U)
        ff.V[...] = interp(fc.V)
        ff.P[...] = interp(fc.P)

        # clamp boundaries
        tx_lid = fine.X_cor[1:, -1] - fine.X_cor[:-1, -1]
        ty_lid = fine.Y_cor[1:, -1] - fine.Y_cor[:-1, -1]
        t_lid = np.hypot(tx_lid, ty_lid)
        ff.U[1:-1, -1] = ff.lid_velocity * tx_lid / t_lid
        ff.V[1:-1, -1] = ff.lid_velocity * ty_lid / t_lid
        ff.U[1:-1, 0] = 0.0; ff.V[1:-1, 0] = 0.0
        ff.U[0, :] = 0.0; ff.V[0, :] = 0.0
        ff.U[-1, :] = 0.0; ff.V[-1, :] = 0.0
        ff.U[0, -1] = ff.U[-1, -1] = ff.lid_velocity

        ff.mflux_x, ff.mflux_y = fine.momentum.div.face_mass_fluxes(ff.U, ff.V, ff.rho)

        return fine

    def step(self, fx=None, fy=None, fx_face_x=None, fy_face_x=None,
             fx_face_y=None, fy_face_y=None):
        f = self.fields
        U_old = f.U.copy()
        V_old = f.V.copy()
        mflux_x_old = f.mflux_x.copy()
        mflux_y_old = f.mflux_y.copy()
        first = self._grid_sequencing_first
        self._grid_sequencing_first = False
        _, f.Ap_u, *_ = self.momentum.solve(f, alpha=self.alpha_u, component="u",
                                             fx=fx, fy=fy)
        _, f.Ap_v, *_ = self.momentum.solve(f, alpha=self.alpha_u, component="v",
                                             fx=fx, fy=fy)
        p_prime = self.pcorr.solve(f, U_old=U_old, V_old=V_old,
                                   mflux_x_old=mflux_x_old, mflux_y_old=mflux_y_old,
                                   alpha_u=self.alpha_u, first_iteration=first,
                                   fx=fx, fy=fy,
                                   fx_face_x=fx_face_x, fy_face_x=fy_face_x,
                                   fx_face_y=fx_face_y, fy_face_y=fy_face_y)
        self.corrector.apply(f, p_prime, alpha_p=self.alpha_p, fx=fx, fy=fy)
        r = (self.momentum.residual["u"], self.momentum.residual["v"],
             self.pcorr.mass_residual / self.mass_scale)
        for k, v in zip(("u", "v", "mass"), r):
            self.history[k].append(v)
        return r

    def run(self, max_outer=20000, tol=1e-6, verbose=500):
        mon = ConvergenceMonitor(atol=tol, rtol=0.0, max_it=max_outer + 1)
        t0 = time.perf_counter()
        converged = False
        diverged_at = None
        for it in range(1, max_outer + 1):
            r = self.step()
            if verbose and it % verbose == 0:
                print(f"  it {it:6d}  res u {r[0]:.2e}  v {r[1]:.2e}  mass {r[2]:.2e}")
            if not np.all(np.isfinite(r)):
                diverged_at = it
                break
            if it > 10 and mon.record(max(r)):
                converged = True
                break
        self.iterations = it
        self.wall_time = time.perf_counter() - t0
        self.converged = converged
        if diverged_at is not None:
            self.status = f"diverged at iteration {diverged_at}"
        elif converged:
            self.status = "converged"
        else:
            self.status = "max_outer reached"
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


class NaturalConvectionSolver(SimpleSolver):
    """Boussinesq natural convection in a rectangular or parallelogram cavity.

    Physical constructor (all dimensional unless noted):
        M=32, shape='rectangular', beta_deg=None, L=1.0, H=1.0,
        rho=1.0, mu=0.71, k=1.0, cp=1.0, g_beta=0.0,
        T_h=1.0, T_c=0.0, T_ref=0.5,
        alpha_u=0.8, alpha_p=0.2, alpha_T=1.0, gamma_dc=1.0,
        inner_sweeps=4, p_solver='direct', n_nonorth_correctors=0,
        balanced_force=True

    Body force:  f_y = g_beta * (T - T_ref)
    BCs: W wall T_h, E wall T_c, S/N adiabatic (horizontal walls),
         all walls no-slip.  Gravity -y.

    Use :meth:`from_Ra_Pr` for the standard non-dimensional de Vahl Davis setup.
    """
    # ------------------------------------------------------------------
    def __init__(self, M=32, shape="rectangular", beta_deg=None,
                 L=1.0, H=1.0, rho=1.0, mu=0.71, k=1.0, cp=1.0,
                 g_beta=0.0, T_h=1.0, T_c=0.0, T_ref=0.5,
                 alpha_u=0.8, alpha_p=0.2, alpha_T=1.0, gamma_dc=1.0,
                 inner_sweeps=4, p_solver="direct", n_nonorth_correctors=0,
                 balanced_force=True):
        self.shape_nc = shape
        self.beta_deg = beta_deg
        self.g_beta = g_beta
        self.T_h = T_h
        self.T_c = T_c
        self.T_ref = T_ref
        self.k = k
        self.cp_nc = cp
        self.rho_nc = rho
        self.mu_nc = mu
        self.alpha_T = alpha_T

        vel_scale = np.sqrt(float(abs(g_beta) * abs(T_h - T_c) * L / max(rho, 1e-30)))
        if vel_scale < 1e-30:
            vel_scale = float(k / max(rho * cp * L, 1e-30))  # thermal diffusion scale
        Re = rho * vel_scale * L / max(mu, 1e-30)

        super().__init__(M=M, Re=Re, L=L, H1=2.0 * L, H2=2.0 * H,
                         shape=shape, beta_deg=beta_deg,
                         rho=rho, lid_velocity=0.0,
                         alpha_u=alpha_u, alpha_p=alpha_p,
                         inner_sweeps=inner_sweeps, p_solver=p_solver,
                         gamma_dc=gamma_dc, body_force_rc=True,
                         n_nonorth_correctors=n_nonorth_correctors,
                         mass_scale=rho * vel_scale * L)
        self.vel_scale = vel_scale

        from .energy import EnergySolver
        self.energy = EnergySolver(self.stencil, self.X, self.Y, self.X_cor, self.Y_cor,
                                   gamma_dc=gamma_dc, alpha_T=alpha_T,
                                   linear_solver=p_solver,
                                   wall_tangential=False)
        self.energy.set_bc(W=("T", T_h), E=("T", T_c), S=("q", 0.0), N=("q", 0.0))
        self.energy.wall_tangential = True  # zero effect for uniform T_wall
        self.fields.add_field("T")
        self.T = self.fields.vars["T"]
        self.Gamma_T = np.full_like(self.T, float(k / cp))
        self.fields.mu[:] = mu
        self.history["T"] = []
        self._T_old = None
        self.balanced_force = balanced_force

    # ------------------------------------------------------------------ step
    def step(self):
        from .solvers import compute_balanced_body_force, compute_face_forces
        f = self.fields
        T = self.T
        buoy = float(self.g_beta)
        if self.balanced_force:
            fx = np.zeros_like(T); fy = np.zeros_like(T)
            compute_balanced_body_force(T, buoy, self.T_ref, self.stencil, fx, fy)
        else:
            fx = None
            fy = buoy * (T - self.T_ref)
        fy_face_x, fy_face_y = compute_face_forces(T, buoy, self.T_ref)

        U_old = f.U.copy(); V_old = f.V.copy()
        mfx_old = f.mflux_x.copy(); mfy_old = f.mflux_y.copy()
        _, f.Ap_u, *_ = self.momentum.solve(f, alpha=self.alpha_u, component="u",
                                             fx=fx, fy=fy, vel_scale=self.vel_scale)
        _, f.Ap_v, *_ = self.momentum.solve(f, alpha=self.alpha_u, component="v",
                                             fx=fx, fy=fy, vel_scale=self.vel_scale)
        p_prime = self.pcorr.solve(f, U_old=U_old, V_old=V_old,
                                   mflux_x_old=mfx_old, mflux_y_old=mfy_old,
                                   alpha_u=self.alpha_u, first_iteration=False,
                                   fx=fx, fy=fy,
                                   fx_face_x=np.zeros_like(fy_face_x), fy_face_x=fy_face_x,
                                   fx_face_y=np.zeros_like(fy_face_y), fy_face_y=fy_face_y)
        self.corrector.apply(f, p_prime, alpha_p=self.alpha_p, fx=fx, fy=fy)
        r = (self.momentum.residual["u"], self.momentum.residual["v"],
             self.pcorr.mass_residual / self.mass_scale)
        for k, v in zip(("u", "v", "mass"), r):
            self.history[k].append(v)
        # energy step
        self.energy.apply_bc(T, self.Gamma_T)
        self.energy.step(f, T, self.Gamma_T)
        self.history["T"].append(self._T_old)
        return r

    # ------------------------------------------------------------------ from_Ra_Pr
    @classmethod
    def from_Ra_Pr(cls, M, Ra, Pr, **kwargs):
        """Non-dimensional de Vahl Davis setup: L=1, rho=1, k=cp=1, mu=Pr, g_beta=Ra*Pr."""
        return cls(M=M, L=1.0, H=1.0, rho=1.0, mu=Pr, k=1.0, cp=1.0,
                   g_beta=float(Ra * Pr), T_h=1.0, T_c=0.0, T_ref=0.5, **kwargs)

    # ------------------------------------------------------------------ run
    def run(self, max_outer=50000, tol=1e-8, verbose=500):
        mon = ConvergenceMonitor(atol=tol, rtol=0.0, max_it=max_outer + 1)
        t0 = time.perf_counter()
        converged = False
        for it in range(1, max_outer + 1):
            self._T_old = self.T.copy()
            r = self.step()
            T_change = np.max(np.abs(self.T[1:-1, 1:-1] - self._T_old[1:-1, 1:-1]))
            if verbose and it % verbose == 0:
                Nu_h, _ = self.energy.wall_heat_flux("W", self.fields, self.T,
                                                      self.Gamma_T, cp=self.cp_nc)
                Nu_c, _ = self.energy.wall_heat_flux("E", self.fields, self.T,
                                                      self.Gamma_T, cp=self.cp_nc)
                print(f"  it {it:6d}  res u {r[0]:.2e}  v {r[1]:.2e}  mass {r[2]:.2e}"
                      f"  dT {T_change:.2e}  Nu {float(np.sum(Nu_h)):.4f}/{float(np.sum(Nu_c)):.4f}")
            if not np.all(np.isfinite(r)):
                self.status = f"diverged at iteration {it}"
                self.iterations = it
                self.wall_time = time.perf_counter() - t0
                return False
            if it > 10 and mon.record(max(r)):
                if T_change < 1e-10:
                    Nu_h = float(np.sum(self.energy.wall_heat_flux(
                        "W", self.fields, self.T, self.Gamma_T, cp=self.cp_nc)[0]))
                    Nu_c = float(np.sum(self.energy.wall_heat_flux(
                        "E", self.fields, self.T, self.Gamma_T, cp=self.cp_nc)[0]))
                    if abs(abs(Nu_h) - abs(Nu_c)) / (abs(Nu_h) + 1e-30) < 1e-6:
                        converged = True
                        break
        self.iterations = it
        self.wall_time = time.perf_counter() - t0
        self.converged = converged
        self.status = "converged" if converged else "max_outer reached"
        return converged

    # ------------------------------------------------------------------ Nusselt
    def nu(self, side):
        """Mean Nusselt number on *side* (W or E).  Nu = |Q|/(k*(T_h-T_c))."""
        Q_f, Q_total = self.energy.wall_heat_flux(side, self.fields, self.T,
                                                   self.Gamma_T, cp=self.cp_nc)
        return float(abs(Q_total) / (self.k * abs(self.T_h - self.T_c)))

    def nu_local(self, side):
        """Per-face Nu_f = Q_f/(|S_f|*k*(T_h-T_c)) and the maximum with its position.

        Returns (Nu_f_array, Nu_max, y_star) where y_star is 0..1 along the wall
        (0=bottom, 1=top).  On a slanted wall y_star uses the ghost-cell coordinate.
        """
        Q_f, _ = self.energy.wall_heat_flux(side, self.fields, self.T,
                                             self.Gamma_T, cp=self.cp_nc)
        denom = self.k * abs(self.T_h - self.T_c)
        if side in ("W", "E"):
            S_mag = self.stencil.Sx_mag[0, :] if side == "W" else self.stencil.Sx_mag[-1, :]
            Y_g = (self.Y[0, 1:-1] if side == "W" else self.Y[-1, 1:-1])
        else:
            S_mag = self.stencil.Sy_mag[:, 0] if side == "S" else self.stencil.Sy_mag[:, -1]
            Y_g = (self.Y[1:-1, 0] if side == "S" else self.Y[1:-1, -1])
        Nu_f = np.where(S_mag > 0, abs(Q_f) / (S_mag * denom), 0.0)
        i_max = int(np.argmax(Nu_f))
        Nu_max = float(Nu_f[i_max])
        y_star = float((Y_g[i_max] - Y_g[0]) / max(Y_g[-1] - Y_g[0], 1e-30))
        return Nu_f, Nu_max, y_star

    # ------------------------------------------------------------------ centre-line / psi
    def max_u_mid(self):
        y, u = self.centerline_u()
        i = np.argmax(u)
        if i <= 0 or i >= len(u) - 1:
            return float(u[i]), float(y[i])
        i = np.clip(i, 1, len(u) - 2)
        y_opt, u_opt = _parabola_peak(y[i - 1:i + 2], u[i - 1:i + 2])
        return float(u_opt), float(y_opt)

    def max_v_mid(self):
        x, v = self.centerline_v()
        i = np.argmax(v)
        if i <= 0 or i >= len(v) - 1:
            return float(v[i]), float(x[i])
        i = np.clip(i, 1, len(v) - 2)
        x_opt, v_opt = _parabola_peak(x[i - 1:i + 2], v[i - 1:i + 2])
        return float(v_opt), float(x_opt)

    def psi_min(self):
        """psi minimum on vertices and its (x, y) via quadratic fit around the min vertex."""
        psi = self.stream_function()
        idx = np.unravel_index(np.argmin(psi), psi.shape)
        i0, j0 = int(idx[0]), int(idx[1])
        if 1 <= i0 < psi.shape[0] - 1 and 1 <= j0 < psi.shape[1] - 1:
            ii = slice(max(1, i0 - 1), min(psi.shape[0] - 1, i0 + 2))
            jj = slice(max(1, j0 - 1), min(psi.shape[1] - 1, j0 + 2))
            local = psi[ii, jj]
            xi = self.X_cor[ii, jj]
            yi = self.Y_cor[ii, jj]
            min_idx = np.argmin(local)
            return float(psi[i0, j0]), float(xi.flat[min_idx]), float(yi.flat[min_idx])
        return float(psi[i0, j0]), float(self.X_cor[i0, j0]), float(self.Y_cor[i0, j0])


def _parabola_peak(x, y):
    """Fit y = a x^2 + b x + c through three points, return peak (x0, y0)."""
    x1, x2, x3 = [float(v) for v in x]
    y1, y2, y3 = [float(v) for v in y]
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(denom) < 1e-30:
        return x2, y2
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3 ** 2 * (y1 - y2) + x2 ** 2 * (y3 - y1) + x1 ** 2 * (y2 - y3)) / denom
    x0 = -b / (2.0 * a) if abs(a) > 1e-30 else x2
    x0 = np.clip(x0, x1, x3)
    c = y1 - a * x1 ** 2 - b * x1
    y0 = a * x0 ** 2 + b * x0 + c
    return x0, y0
