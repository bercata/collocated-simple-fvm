"""Low-Mach natural convection solver (Le Quere et al. 2005)."""
from __future__ import annotations
import numpy as np
import time
from .simple import NaturalConvectionSolver
from .stencil import x_face_avg, y_face_avg
from .solvers import ConvergenceMonitor
from .properties import mu_sutherland, k_thermal, mu_star, cavity_length


class LowMachSolver(NaturalConvectionSolver):
    """Low-Mach-number natural convection with ideal-gas density.

    rho = Pbar/(R*T),  div(rho*u)=0,  f=(rho-rho_ref)*g,  full stress tensor.

    Known limitation: the extra-stress face fluxes use averaged cell gradients;
    the wall-cell Gauss gradient is first-order on non-orthogonal meshes, so
    the extra-stress source has an O(1) local error in the second cell layer
    next to a wall on the parallelogram.  The term scales with div u and is
    small (a few percent of the stress magnitude).  A grid study (T1/T2)
    should determine whether it affects the observed convergence order.

    traps : iterable of str, optional
        One or more implementation-trap names to activate.  Empty (default)
        gives bit-identical results to the baseline solver.
        See ``trap_study.py`` for the full list.
    """

    def __init__(self, M=32, shape="rectangular", beta_deg=None,
                 eps=0.6, Ra=1e6, L=None,
                 P0=101325.0, T0=600.0, R_gas=287.0, cp_val=1004.5,
                 Pr_val=0.71, g_val=9.81,
                 alpha_u=0.6, alpha_p=0.2, alpha_T=0.8,
                 gamma_dc=1.0, inner_sweeps=4,
                 p_solver="direct", n_nonorth_correctors=0,
                 properties="constant", traps=()):
        # --- store ---
        self.P0 = P0; self.T0 = T0; self.R_gas = R_gas
        self.cp_val = cp_val; self.Pr_val = Pr_val; self.g_val = g_val
        self.eps = eps; self.Ra = Ra; self._prop_mode = properties
        if L is None:
            mu0 = mu_star if properties == "constant" else mu_sutherland(T0)
            L = cavity_length(Ra, eps, mu0)
        self.L = L; self.Th = T0*(1.0+eps); self.Tc = T0*(1.0-eps)
        rho0 = P0/(R_gas*T0); self.rho_ref = rho0
        self.mu0 = mu_star if properties == "constant" else mu_sutherland(T0)
        self.k0 = self.mu0*cp_val/Pr_val
        vel_scale = np.sqrt(float(g_val*L*abs(2.0*eps)))
        self.rich_nc = False

        # build parent
        n_nonorth = n_nonorth_correctors if shape == "parallelogram" else 0
        super().__init__(M=M, shape=shape, beta_deg=beta_deg,
                         L=L, H=L, rho=rho0, mu=self.mu0, k=self.k0,
                         cp=cp_val, g_beta=g_val, T_h=self.Th,
                         T_c=self.Tc, T_ref=T0,
                         alpha_u=alpha_u, alpha_p=alpha_p,
                         alpha_T=alpha_T, gamma_dc=gamma_dc,
                         inner_sweeps=inner_sweeps, p_solver=p_solver,
                         n_nonorth_correctors=n_nonorth)
        self.mass_scale = rho0*vel_scale*L
        self.vel_scale = vel_scale

        # --- low-Mach state ---
        self.Pbar = P0
        self.m0 = rho0*np.sum(self.stencil.vol)
        self.rho_cell = np.full_like(self.fields.U, float(rho0))
        self.history["Pbar"] = []
        self._first_step = True
        # initial temperature field = T0 everywhere
        self.T[:] = T0
        # set ghost T for energy BCs
        self.energy.apply_bc(self.T, self.Gamma_T)

        # --- traps ---
        self.traps = set(traps)
        if "no_majumdar" in self.traps:
            self.pcorr.consistent_rc = False
        if "unbalanced_force" in self.traps:
            self.pcorr.body_force_rc = False
        if "no_wall_tangential" in self.traps:
            self.energy.wall_tangential = False

    # ================================================================ grid sequencing
    @classmethod
    def init_from(cls, coarse, M=None, alpha_u=None, alpha_p=None, alpha_T=None,
                  traps=None):
        """Create a finer solver (M = 2 * coarse.M) initialised from a converged solver.

        U, V, P and T are interpolated bilinearly in the logical (i, j) coordinates
        (the parallelogram mesh is uniform in logical space).  Pbar and the density
        rho = Pbar/(R*T) are copied/recomputed, and the face mass fluxes are rebuilt
        from the interpolated fields so the first iteration starts mass-consistent.
        Relaxation factors default to the coarse solver's values but can be overridden
        (e.g. to lower them when a finer mesh stalls).
        """
        if M is None:
            M = 2 * coarse.M
        if coarse.shape != "parallelogram":
            raise ValueError("init_from only supported for parallelogram meshes")
        alpha_u = coarse.alpha_u if alpha_u is None else alpha_u
        alpha_p = coarse.alpha_p if alpha_p is None else alpha_p
        alpha_T = coarse.alpha_T if alpha_T is None else alpha_T
        _traps = traps if traps is not None else coarse.traps
        fine = cls(M=M, shape=coarse.shape, beta_deg=coarse.beta_deg,
                   eps=coarse.eps, Ra=coarse.Ra, L=coarse.L,
                   P0=coarse.P0, T0=coarse.T0, R_gas=coarse.R_gas,
                   cp_val=coarse.cp_val, Pr_val=coarse.Pr_val, g_val=coarse.g_val,
                   alpha_u=alpha_u, alpha_p=alpha_p, alpha_T=alpha_T,
                   gamma_dc=coarse.momentum.gamma_dc,
                   inner_sweeps=coarse.momentum.inner_sweeps,
                   p_solver=coarse.pcorr.linear_solver,
                   n_nonorth_correctors=coarse.pcorr.n_nonorth_correctors,
                   properties=coarse._prop_mode,
                   traps=_traps)

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
            return ((1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10
                    + (1.0 - tx) * ty * v01 + tx * ty * v11)

        ff.U[...] = interp(fc.U)
        ff.V[...] = interp(fc.V)
        ff.P[...] = interp(fc.P)
        # no-slip walls: ghost velocity is the wall value (zero)
        ff.U[0, :] = ff.U[-1, :] = 0.0
        ff.V[0, :] = ff.V[-1, :] = 0.0
        ff.U[:, 0] = ff.U[:, -1] = 0.0
        ff.V[:, 0] = ff.V[:, -1] = 0.0

        # temperature: interpolate, then re-apply the wall conditions on the ghosts
        fine.T[...] = interp(coarse.T)
        mu_cell = mu_sutherland(fine.T)
        fine.fields.mu[:] = mu_cell
        fine.Gamma_T[:] = k_thermal(fine.T, mu_cell) / fine.cp_val
        fine.energy.apply_bc(fine.T, fine.Gamma_T)
        if "no_wall_tangential" not in fine.traps:
            fine.energy.apply_adiabatic_tangential(fine.T, fine.Gamma_T, n_iter=3)
        fine.Gamma_T[:] = k_thermal(fine.T, mu_sutherland(fine.T)) / fine.cp_val

        # thermodynamic state and consistent face mass fluxes
        fine.Pbar = coarse.Pbar
        if "boussinesq_rho" in fine.traps:
            fine.rho_cell[:] = fine.rho_ref
            fine.fields.rho[:] = fine.rho_ref
        else:
            fine._update_density()
        ff.mflux_x, ff.mflux_y = fine.momentum.div.face_mass_fluxes(ff.U, ff.V, ff.rho)
        return fine

    # ================================================================ helpers
    def _update_density(self):
        rho_new = self.Pbar/(self.R_gas*self.T + 1e-30)
        if getattr(self, '_rho_old', None) is None:
            self._rho_old = rho_new.copy()
        # under-relaxed density update
        alpha_rho = 0.8
        rho_relaxed = self._rho_old + alpha_rho*(rho_new - self._rho_old)
        self._rho_old = rho_relaxed
        self.rho_cell[:] = rho_relaxed
        self.fields.rho[:] = self.rho_cell

    def _update_Pbar(self):
        V = self.stencil.vol
        T_c = self.T[1:-1, 1:-1]
        self.Pbar = self.m0*self.R_gas/np.sum(V/(T_c + 1e-30))

    def _compute_face_properties(self):
        """Return face-mu arrays (mu_x, mu_y) shapes (M+1,M) and (M,M+1)."""
        T = self.T
        Tx = 0.5*(T[:-1, 1:-1] + T[1:, 1:-1])
        Ty = 0.5*(T[1:-1, :-1] + T[1:-1, 1:])
        # wall faces: use ghost T directly (baseline) or cell/ghost average (trap)
        if "wall_props_avg" in self.traps:
            Tx[0, :] = 0.5*(T[0, 1:-1] + T[1, 1:-1])
            Tx[-1, :] = 0.5*(T[-1, 1:-1] + T[-2, 1:-1])
            Ty[:, 0] = 0.5*(T[1:-1, 0] + T[1:-1, 1])
            Ty[:, -1] = 0.5*(T[1:-1, -1] + T[1:-1, -2])
        else:
            Tx[0, :] = T[0, 1:-1]; Tx[-1, :] = T[-1, 1:-1]
            Ty[:, 0] = T[1:-1, 0]; Ty[:, -1] = T[1:-1, -1]
        if self._prop_mode == "sutherland":
            mu_x = mu_sutherland(Tx); mu_y = mu_sutherland(Ty)
            gamma_x = mu_x*self.cp_val/self.Pr_val/self.cp_val  # = mu_x/Pr
            gamma_y = mu_y/self.Pr_val
        else:
            mu_x = np.full_like(Tx, self.mu0); mu_y = np.full_like(Ty, self.mu0)
            gamma_x = np.full_like(Tx, self.mu0/self.Pr_val)
            gamma_y = np.full_like(Ty, self.mu0/self.Pr_val)
        return mu_x, mu_y, gamma_x, gamma_y

# ================================================================ stress
    def _stress_source(self, component, mu_x, mu_y):
        f = self.fields; s = self.stencil
        gux, guy = self.momentum.grad.compute_gradient(f.U)
        gvx, gvy = self.momentum.grad.compute_gradient(f.V)
        div_u = gux + gvy
        if component == "u":
            qx = mu_x*((x_face_avg(gux) - (2.0/3.0)*x_face_avg(div_u))*s.Sx_x
                       + x_face_avg(gvx)*s.Sx_y)
            qy = mu_y*((y_face_avg(gux) - (2.0/3.0)*y_face_avg(div_u))*s.Sy_x
                       + y_face_avg(gvx)*s.Sy_y)
        else:
            qx = mu_x*(x_face_avg(guy)*s.Sx_x
                       + (x_face_avg(gvy) - (2.0/3.0)*x_face_avg(div_u))*s.Sx_y)
            qy = mu_y*(y_face_avg(guy)*s.Sy_x
                       + (y_face_avg(gvy) - (2.0/3.0)*y_face_avg(div_u))*s.Sy_y)
        qx[~s.x_interior] = 0.0
        qy[~s.y_interior] = 0.0
        src = np.zeros((s.nx, s.ny))
        src[1:-1, 1:-1] = (qx[1:] - qx[:-1] + qy[:, 1:] - qy[:, :-1])
        return src

# ================================================================ step
    def step(self):
        f = self.fields; T = self.T; s = self.stencil; buoy = float(self.g_val)
        traps = self.traps

        # ---- density ----
        if "boussinesq_rho" in traps:
            self.rho_cell[:] = self.rho_ref
            f.rho[:] = self.rho_ref
        else:
            self._update_density()

        # ---- Sutherland or constant cell properties ----
        if self._prop_mode == "sutherland":
            mu_cell = mu_sutherland(T)
            k_cell = k_thermal(T, mu_cell)
        else:
            mu_cell = np.full_like(T, self.mu0)
            k_cell = np.full_like(T, self.k0)
        f.mu[:] = mu_cell
        gamma_cell = k_cell / self.cp_val
        self.Gamma_T[:] = gamma_cell

        # face properties
        mu_x, mu_y, gamma_x, gamma_y = self._compute_face_properties()

        # ---- face forces ----
        Tx = 0.5*(T[:-1, 1:-1] + T[1:, 1:-1])
        Ty = 0.5*(T[1:-1, :-1] + T[1:-1, 1:])
        Tx[0, :] = T[0, 1:-1]; Tx[-1, :] = T[-1, 1:-1]
        Ty[:, 0] = T[1:-1, 0]; Ty[:, -1] = T[1:-1, -1]
        if "boussinesq_rho" in traps:
            rho_x = self.rho_ref*(1.0 - (Tx - self.T0)/self.T0)
            rho_y = self.rho_ref*(1.0 - (Ty - self.T0)/self.T0)
        else:
            rho_x = self.Pbar/(self.R_gas*Tx + 1e-30)
            rho_y = self.Pbar/(self.R_gas*Ty + 1e-30)
        fy_x = -buoy*(rho_x - self.rho_ref)
        fy_y = -buoy*(rho_y - self.rho_ref)

        # ---- cell forces ----
        if "unbalanced_force" in traps:
            # plain cell body force (no balanced construction, no face-force Rhie-Chow)
            if "boussinesq_rho" in traps:
                fy_cell = buoy*self.rho_ref*(T - self.T0)/self.T0
            else:
                fy_cell = -buoy*(self.rho_cell - self.rho_ref)
            fx = np.zeros_like(T)
            fy = fy_cell
            corr_fx = None; corr_fy = None
            pcorr_fx_fx = None; pcorr_fy_fx = None
            pcorr_fx_fy = None; pcorr_fy_fy = None
        else:
            # balanced cell force
            M = s.M
            fx = np.zeros_like(T); fy = np.zeros_like(T)
            fd_i = fy_x[1:M, :]*s.dx_y[1:M, :]
            sx_i, sy_i = s.Sx_x[1:M, :], s.Sx_y[1:M, :]
            fx[1:M, 1:-1] += fd_i*sx_i/(2.0*s.vol[:M-1, :])
            fy[1:M, 1:-1] += fd_i*sy_i/(2.0*s.vol[:M-1, :])
            fx[2:M+1, 1:-1] += fd_i*sx_i/(2.0*s.vol[1:M, :])
            fy[2:M+1, 1:-1] += fd_i*sy_i/(2.0*s.vol[1:M, :])
            fd_j = fy_y[:, 1:M]*s.dy_y[:, 1:M]
            sx_j, sy_j = s.Sy_x[:, 1:M], s.Sy_y[:, 1:M]
            fx[1:-1, 1:M] += fd_j*sx_j/(2.0*s.vol[:, :M-1])
            fy[1:-1, 1:M] += fd_j*sy_j/(2.0*s.vol[:, :M-1])
            fx[1:-1, 2:M+1] += fd_j*sx_j/(2.0*s.vol[:, 1:M])
            fy[1:-1, 2:M+1] += fd_j*sy_j/(2.0*s.vol[:, 1:M])
            fd = fy_x[0, :]*s.dx_y[0, :]
            fx[1, 1:-1] += fd*s.Sx_x[0, :]/s.vol[0, :]
            fy[1, 1:-1] += fd*s.Sx_y[0, :]/s.vol[0, :]; fy[0, 1:-1] = fy_x[0, :]
            fd = fy_x[M, :]*s.dx_y[M, :]
            fx[M, 1:-1] += fd*s.Sx_x[M, :]/s.vol[M-1, :]
            fy[M, 1:-1] += fd*s.Sx_y[M, :]/s.vol[M-1, :]; fy[-1, 1:-1] = fy_x[M, :]
            fd = fy_y[:, 0]*s.dy_y[:, 0]
            fx[1:-1, 1] += fd*s.Sy_x[:, 0]/s.vol[:, 0]
            fy[1:-1, 1] += fd*s.Sy_y[:, 0]/s.vol[:, 0]; fy[1:-1, 0] = fy_y[:, 0]
            fd = fy_y[:, M]*s.dy_y[:, M]
            fx[1:-1, M] += fd*s.Sy_x[:, M]/s.vol[:, M-1]
            fy[1:-1, M] += fd*s.Sy_y[:, M]/s.vol[:, M-1]; fy[1:-1, -1] = fy_y[:, M]
            corr_fx = fx; corr_fy = fy
            fx_fx = np.zeros_like(fy_x); fx_fy = np.zeros_like(fy_y)
            pcorr_fx_fx = fx_fx; pcorr_fy_fx = fy_x
            pcorr_fx_fy = fx_fy; pcorr_fy_fy = fy_y

        # ---- momentum ----
        if "no_extra_stress" in traps:
            S_u = np.zeros_like(T); S_v = np.zeros_like(T)
        else:
            S_u = self._stress_source("u", mu_x, mu_y)
            S_v = self._stress_source("v", mu_x, mu_y)
        U_old = f.U.copy(); V_old = f.V.copy()
        mfx_old = f.mflux_x.copy(); mfy_old = f.mflux_y.copy()
        _, Ap_u, *_ = self.momentum.solve(f, alpha=self.alpha_u, component="u",
                                           fx=fx, fy=fy, vel_scale=self.vel_scale,
                                           stress_source=S_u,
                                           mu_face_x=mu_x, mu_face_y=mu_y)
        f.Ap_u = Ap_u
        _, Ap_v, *_ = self.momentum.solve(f, alpha=self.alpha_u, component="v",
                                           fx=fx, fy=fy, vel_scale=self.vel_scale,
                                           stress_source=S_v,
                                           mu_face_x=mu_x, mu_face_y=mu_y)
        f.Ap_v = Ap_v

        first = self._first_step; self._first_step = False
        if "no_majumdar" in traps:
            # consistent_rc already set to False in __init__
            pass
        p_prime = self.pcorr.solve(f, U_old=U_old, V_old=V_old,
                                    mflux_x_old=mfx_old, mflux_y_old=mfy_old,
                                    alpha_u=self.alpha_u, first_iteration=first,
                                    fx=fx, fy=fy,
                                    fx_face_x=pcorr_fx_fx, fy_face_x=pcorr_fy_fx,
                                    fx_face_y=pcorr_fx_fy, fy_face_y=pcorr_fy_fy)
        self.corrector.apply(f, p_prime, alpha_p=self.alpha_p, fx=corr_fx, fy=corr_fy)

        r = (self.momentum.residual["u"], self.momentum.residual["v"],
             self.pcorr.mass_residual / self.mass_scale)
        for k, v in zip(("u", "v", "mass"), r):
            self.history[k].append(v)

        # ---- energy ----
        self.energy.apply_bc(T, self.Gamma_T)
        if "no_wall_tangential" in traps:
            ls = self.energy._assemble(f, T, self.Gamma_T,
                                        Gamma_face_x=gamma_x, Gamma_face_y=gamma_y)
            self.energy._solve_linear(ls, T)
        else:
            self.energy.apply_adiabatic_tangential(T, self.Gamma_T, n_iter=3)
            self.energy.step(f, T, self.Gamma_T,
                              Gamma_face_x=gamma_x, Gamma_face_y=gamma_y)

        # ---- Pbar ----
        if "pbar_fixed" in traps:
            pass  # Pbar stays at its current value (P0 at init)
        else:
            self._update_Pbar()
        self.history["Pbar"].append(float(self.Pbar))
        return r

    # ================================================================ run
    def run(self, max_outer=50000, tol=1e-8, verbose=500):
        mon = ConvergenceMonitor(atol=tol, rtol=0.0, max_it=max_outer+1)
        t0 = time.perf_counter()
        converged = False
        for it in range(1, max_outer+1):
            T_old = self.T.copy()
            P_old = self.Pbar
            r = self.step()
            dT = np.max(np.abs(self.T[1:-1, 1:-1] - T_old[1:-1, 1:-1]))
            dP = abs(self.Pbar - P_old)/self.P0
            if verbose and it % verbose == 0:
                print(f"  it {it:6d}  u {r[0]:.2e}  v {r[1]:.2e}  mass {r[2]:.2e}"
                      f"  dT {dT:.2e}  dPbar {dP:.2e}  Pbar/P0 {self.Pbar/self.P0:.6f}")
            if not np.all(np.isfinite(r)):
                self.status = f"diverged at {it}"; self.iterations = it
                self.wall_time = time.perf_counter()-t0; return False
            if it > 10 and mon.record(max(r)):
                if dT/self.T0 < 1e-12 and dP < 1e-12:
                    Nu_h = self.nu("W"); Nu_c = self.nu("E")
                    if abs(abs(Nu_h)-abs(Nu_c))/(abs(Nu_h)+1e-30) < 1e-6:
                        converged = True; break
        self.iterations = it; self.wall_time = time.perf_counter()-t0
        self.converged = converged
        self.status = "converged" if converged else "max_outer reached"
        return converged

    # ================================================================ post-processing
    def nu_local(self, side):
        """Per-face local Nusselt number with correct L scaling.

        Nu_f = Q_f/|S_f| * L/(k0*dT),  consistent with nu().
        y_star is the distance along the wall from the bottom corner divided by
        the wall length (0 = bottom corner, 1 = top corner).
        """
        Q_f, _ = self.energy.wall_heat_flux(side, self.fields, self.T,
                                             self.Gamma_T, cp=self.cp_val)
        denom = self.k0 * abs(self.Th - self.Tc)
        if side in ("W", "E"):
            S_mag = self.stencil.Sx_mag[0, :] if side == "W" else self.stencil.Sx_mag[-1, :]
            Y_g = (self.Y[0, 1:-1] if side == "W" else self.Y[-1, 1:-1])
            X_g = (self.X[0, 1:-1] if side == "W" else self.X[-1, 1:-1])
            i_wall = 0 if side == "W" else -1
            x_bot, y_bot = self.X_cor[i_wall, 0], self.Y_cor[i_wall, 0]
            x_top, y_top = self.X_cor[i_wall, -1], self.Y_cor[i_wall, -1]
            wall_len = np.hypot(x_top - x_bot, y_top - y_bot)
        else:
            S_mag = self.stencil.Sy_mag[:, 0] if side == "S" else self.stencil.Sy_mag[:, -1]
            Y_g = (self.Y[1:-1, 0] if side == "S" else self.Y[1:-1, -1])
            X_g = (self.X[1:-1, 0] if side == "S" else self.X[1:-1, -1])
            j_wall = 0 if side == "S" else -1
            x_bot, y_bot = self.X_cor[0, j_wall], self.Y_cor[0, j_wall]
            x_top, y_top = self.X_cor[-1, j_wall], self.Y_cor[-1, j_wall]
            wall_len = np.hypot(x_top - x_bot, y_top - y_bot)
        Nu_f = np.where(S_mag > 0, abs(Q_f) * self.L / (S_mag * denom), 0.0)
        i_max = int(np.argmax(Nu_f))
        Nu_max = float(Nu_f[i_max])
        dist = np.hypot(X_g - x_bot, Y_g - y_bot)
        y_star = float(dist[i_max] / max(wall_len, 1e-30))
        return Nu_f, Nu_max, y_star

    def psi_volume(self):
        """Psi from volume flux (U.S) instead of mass flux — post-processing only."""
        s = self.stencil
        mx_vol = (x_face_avg(self.fields.U) * s.Sx_x + x_face_avg(self.fields.V) * s.Sx_y)
        my_vol = (y_face_avg(self.fields.U) * s.Sy_x + y_face_avg(self.fields.V) * s.Sy_y)
        mx_vol[~s.x_interior] = 0.0
        my_vol[~s.y_interior] = 0.0
        psi = np.zeros_like(self.X_cor)
        psi[:, 1:] = np.cumsum(mx_vol, axis=1)
        idx = np.unravel_index(np.argmin(psi), psi.shape)
        i0, j0 = int(idx[0]), int(idx[1])
        if 1 <= i0 < psi.shape[0] - 1 and 1 <= j0 < psi.shape[1] - 1:
            ii = slice(max(1, i0 - 1), min(psi.shape[0] - 1, i0 + 2))
            jj = slice(max(1, j0 - 1), min(psi.shape[1] - 1, j0 + 2))
            local = psi[ii, jj]
            xi = self.X_cor[ii, jj]; yi = self.Y_cor[ii, jj]
            min_idx = np.argmin(local)
            return float(psi[i0, j0]), float(xi.flat[min_idx]), float(yi.flat[min_idx])
        return float(psi[i0, j0]), float(self.X_cor[i0, j0]), float(self.Y_cor[i0, j0])

    def total_mass(self):
        """Sum of rho*V over all interior cells divided by m0, minus 1."""
        V = self.stencil.vol
        total = np.sum(self.fields.rho[1:-1, 1:-1] * V)
        return float(total / self.m0 - 1.0)

    # ================================================================ save / load
    def save(self, path):
        """Save solver state to a .npz file for restart."""
        f = self.fields
        state = {
            "U": f.U, "V": f.V, "P": f.P, "T": self.T,
            "rho": f.rho, "rho_cell": self.rho_cell,
            "mflux_x": f.mflux_x, "mflux_y": f.mflux_y,
            "Pbar": self.Pbar,
            "_rho_old": getattr(self, "_rho_old", self.rho_cell.copy()),
            "_first_step": self._first_step,
            "iterations": getattr(self, "iterations", 0),
            "wall_time": getattr(self, "wall_time", 0.0),
            "converged": getattr(self, "converged", False),
            "status": getattr(self, "status", "fresh"),
            "alpha_u": self.alpha_u, "alpha_p": self.alpha_p,
            "alpha_T": self.alpha_T,
        }
        for k, v in self.history.items():
            if isinstance(v, list):
                state[f"hist_{k}"] = np.array(v, dtype=float)
        np.savez_compressed(path, **state)

    @classmethod
    def from_state(cls, path, eps=None, Ra=None, beta_deg=None):
        """Restore a solver from a .npz file saved by ``save()``."""
        data = np.load(path, allow_pickle=False)
        M = data["U"].shape[0] - 2
        P0 = 101325.0; T0 = 600.0; R_gas = 287.0
        cp_val = 1004.5; Pr_val = 0.71; g_val = 9.81
        _eps = eps if eps is not None else 0.6
        _Ra = Ra if Ra is not None else 1e5
        _beta = beta_deg if beta_deg is not None else 60
        au = float(data["alpha_u"]); ap = float(data["alpha_p"])
        aT = float(data["alpha_T"])
        solver = cls(M=M, shape="parallelogram", beta_deg=_beta,
                     eps=_eps, Ra=_Ra, L=None,
                     P0=P0, T0=T0, R_gas=R_gas,
                     cp_val=cp_val, Pr_val=Pr_val, g_val=g_val,
                     alpha_u=au, alpha_p=ap, alpha_T=aT,
                     p_solver="direct", n_nonorth_correctors=0,
                     properties="sutherland")
        f = solver.fields
        f.U[:] = data["U"]; f.V[:] = data["V"]; f.P[:] = data["P"]
        solver.T[:] = data["T"]
        f.rho[:] = data["rho"]; solver.rho_cell[:] = data["rho_cell"]
        f.mflux_x[:] = data["mflux_x"]; f.mflux_y[:] = data["mflux_y"]
        solver.Pbar = float(data["Pbar"])
        solver._rho_old = data["_rho_old"].copy()
        solver._first_step = bool(data["_first_step"])
        solver.iterations = int(data["iterations"])
        solver.wall_time = float(data["wall_time"])
        solver.converged = bool(data["converged"])
        solver.status = str(data["status"])
        # restore history
        for k in list(data.keys()):
            if k.startswith("hist_"):
                name = k[5:]
                solver.history[name] = list(data[k])
        # rebuild consistent mu/Gamma
        mu_cell = mu_sutherland(solver.T)
        f.mu[:] = mu_cell
        solver.Gamma_T[:] = k_thermal(solver.T, mu_cell) / solver.cp_val
        solver.energy.apply_bc(solver.T, solver.Gamma_T)
        solver.energy.apply_adiabatic_tangential(solver.T, solver.Gamma_T, n_iter=3)
        solver.Gamma_T[:] = k_thermal(solver.T, mu_sutherland(solver.T)) / solver.cp_val

        # restore face-Gamma arrays and tangential flux that wall_heat_flux reads
        mu_x, mu_y, gamma_x, gamma_y = solver._compute_face_properties()
        solver.energy._Gamma_face_x = gamma_x
        solver.energy._Gamma_face_y = gamma_y
        solver.energy._wall_tangential_source(solver.T, solver.Gamma_T)
        return solver