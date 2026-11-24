"""Energy (temperature) equation as a passive scalar, solved on frozen flow fields."""
from __future__ import annotations

import numpy as np

from .linsys import FVLinearSystem
from .operators import FVM_grad, FVM_div, FVM_laplacian, deferred_correction_source
from .solvers import SparseDirect, TDMA


class EnergySolver:
    """Steady energy equation: div(mflux * T) = div(Gamma * grad T) + source.

    Convection uses the stored Rhie-Chow face mass fluxes; deferred correction
    (Khosla-Rubin) centres the convective term.  Diffusion uses the over-relaxed
    decomposition with an explicit non-orthogonal source.

    Wall conditions per side (``"W"``, ``"E"``, ``"S"``, ``"N"``):
       ``("T", value)``  Dirichlet  (T_ghost = value)
       ``("q", 0.0)``    adiabatic  (T_ghost = T_interior, zero normal flux exactly)
    """

    def __init__(self, stencil, X, Y, X_cor, Y_cor, gamma_dc=1.0, alpha_T=1.0,
                 linear_solver="direct", wall_tangential=True):
        self.s = stencil
        self.X, self.Y, self.X_cor, self.Y_cor = X, Y, X_cor, Y_cor
        self.gamma_dc = gamma_dc
        self.alpha_T = alpha_T
        self.linear_solver = linear_solver
        self.wall_tangential = wall_tangential
        self.grad = FVM_grad(X, Y, X_cor, Y_cor, stencil)
        self.div = FVM_div(X, Y, X_cor, Y_cor, stencil)
        self.lap = FVM_laplacian(X, Y, X_cor, Y_cor, stencil)
        self.bc = {}
        self._q_t = {}
        self._Gamma_face_x = None
        self._Gamma_face_y = None

    def set_bc(self, W=None, E=None, S=None, N=None):
        """Set wall conditions.  Pass ``None`` to leave a side at its previous setting."""
        for side, val in zip(("W", "E", "S", "N"), (W, E, S, N)):
            if val is not None:
                self.bc[side] = val

    # ------------------------------------------------------------ ghost cells
    _GHOST = {
        "W": ((0, slice(1, -1)), (1, slice(1, -1))),
        "E": ((-1, slice(1, -1)), (-2, slice(1, -1))),
        "S": ((slice(1, -1), 0), (slice(1, -1), 1)),
        "N": ((slice(1, -1), -1), (slice(1, -1), -2)),
    }

    def apply_bc(self, T, Gamma):
        """Set ghost-cell values of *T* and *Gamma* from the wall conditions.

        For Dirichlet walls only *T* is set; the ghost *Gamma* is left to the
        caller (e.g. ``k(T_ghost)/cp`` for variable properties) so that the
        wall conductances, the face-Gamma arrays and the Nusselt numbers agree.
        For adiabatic walls the ghost *Gamma* is copied from the interior cell.
        """
        for side, (ghost_idx, interior_idx) in self._GHOST.items():
            bc_val = self.bc.get(side)
            if bc_val is None:
                continue
            btype, val = bc_val
            if btype == "T":
                try:
                    T[ghost_idx] = val[ghost_idx]
                except (IndexError, TypeError):
                    T[ghost_idx] = val
            elif btype == "q":
                T[ghost_idx] = T[interior_idx]
                Gamma[ghost_idx] = Gamma[interior_idx]
            else:
                raise ValueError(f"Unknown BC type {btype!r} on side {side}")

    def apply_adiabatic_tangential(self, T, Gamma, n_iter=3):
        """Enforce dT/dn = 0 on adiabatic walls of a non-orthogonal mesh.

        T_ghost = T_P + (g_P - (g_P . n) n) . d_w

        where g_P is the Gauss gradient of the wall cell, n is the unit
        wall-face normal (S_w/|S_w|, sign does not matter for projection),
        d_w = x_wall_face - x_P.
        """
        s = self.s; X, Y = self.X, self.Y
        for _ in range(n_iter):
            gx, gy = self.grad.compute_gradient(T)
            for side in ("W", "E", "S", "N"):
                bc = self.bc.get(side)
                if bc is None or bc[0] != "q":
                    continue
                gi, ii = self._GHOST[side]
                if side in ("W", "E"):
                    k = 0 if side == "W" else -1
                    Sx, Sy = s.Sx_x[k, :], s.Sx_y[k, :]
                else:
                    k = 0 if side == "S" else -1
                    Sx, Sy = s.Sy_x[:, k], s.Sy_y[:, k]
                nm = np.hypot(Sx, Sy)
                nx, ny = Sx / (nm + 1e-30), Sy / (nm + 1e-30)
                gpx, gpy = gx[ii], gy[ii]
                gn = gpx * nx + gpy * ny
                tx, ty = gpx - gn * nx, gpy - gn * ny
                dwx, dwy = X[gi] - X[ii], Y[gi] - Y[ii]
                T[gi] = T[ii] + tx * dwx + ty * dwy
        # consistency for Dirichlet walls (corner ghosts may have been touched)
        for side in ("W", "E", "S", "N"):
            bc = self.bc.get(side)
            if bc is not None and bc[0] == "T":
                gi, ii = self._GHOST[side]
                try:    T[gi] = bc[1][gi]
                except (IndexError, TypeError): T[gi] = bc[1]

    # ------------------------------------------------------------ assembly
    def _assemble(self, fields, T, Gamma, source=None,
                  Gamma_face_x=None, Gamma_face_y=None):
        s = self.s
        f = fields
        mflux = (f.mflux_x, f.mflux_y)
        fluxes = self.div.compute_face_fluxes(T, f.U, f.V, f.rho, mflux=mflux)
        diffs = self.lap.compute_face_diffusion(T, Gamma,
                                                Gamma_face_x=Gamma_face_x,
                                                Gamma_face_y=Gamma_face_y)
        self._Gamma_face_x = Gamma_face_x
        self._Gamma_face_y = Gamma_face_y

        # Zero wall conductances for adiabatic walls: they contribute nothing.
        for side, (gidx, iidx) in self._GHOST.items():
            bc_val = self.bc.get(side)
            if bc_val is None or bc_val[0] != "q":
                continue
            if side == "W":
                diffs["Diff_w"][1, 1:-1] = 0.0
            elif side == "E":
                diffs["Diff_e"][-2, 1:-1] = 0.0
            elif side == "S":
                diffs["Diff_s"][1:-1, 1] = 0.0
            elif side == "N":
                diffs["Diff_n"][1:-1, -2] = 0.0

        ls = FVLinearSystem(s.nx, s.ny)
        src = np.zeros((s.nx, s.ny)) if source is None else source
        ls.assemble_raw(fluxes["flux_e"], fluxes["flux_w"], fluxes["flux_n"], fluxes["flux_s"],
                        diffs["Diff_e"], diffs["Diff_w"], diffs["Diff_n"], diffs["Diff_s"],
                        src, self.X_cor, self.Y_cor, vol=s.vol)

        if self.gamma_dc > 0.0:
            dcsrc = deferred_correction_source(T, f.mflux_x, f.mflux_y, self.gamma_dc)
            ls.Sp[1:-1, 1:-1] -= dcsrc[1:-1, 1:-1]

        gtx, gty = self.grad.compute_gradient(T)
        src_noc = self.lap.nonorthogonal_correction_source(gtx, gty, Gamma)
        ls.Sp[1:-1, 1:-1] += src_noc[1:-1, 1:-1]

        if self.wall_tangential:
            wt_src = self._wall_tangential_source(T, Gamma)
            ls.Sp[1:-1, 1:-1] += wt_src[1:-1, 1:-1]

        if self.alpha_T < 1.0:
            ls.apply_under_relax(T, alpha=self.alpha_T)

        return ls

    def _adjust_boundary(self, ls, T):
        """Compensate SparseDirect for adiabatic walls.

        SparseDirect zeroes the wall off-diagonal entry and adds D·T_ghost to
        the RHS.  For adiabatic walls T_ghost = T_P so the net flux is zero:
        D·T_P in the diagonal and D·T_P on the RHS should cancel.  We enforce
        this directly by subtracting D from both Ap and Sp.
        Dirichlet walls are left alone — the standard SparseDirect handling is
        correct for them.
        """
        for side, (gidx, _iidx) in self._GHOST.items():
            bc_val = self.bc.get(side)
            if bc_val is None or bc_val[0] != "q":
                continue
            if side == "W":
                Dw = ls.Aw[1, 1:-1]
                ls.Ap[1, 1:-1] -= Dw
                ls.Sp[1, 1:-1] -= Dw * T[1, 1:-1]
            elif side == "E":
                Dw = ls.Ae[-2, 1:-1]
                ls.Ap[-2, 1:-1] -= Dw
                ls.Sp[-2, 1:-1] -= Dw * T[-2, 1:-1]
            elif side == "S":
                Dw = ls.As[1:-1, 1]
                ls.Ap[1:-1, 1] -= Dw
                ls.Sp[1:-1, 1] -= Dw * T[1:-1, 1]
            elif side == "N":
                Dw = ls.An[1:-1, -2]
                ls.Ap[1:-1, -2] -= Dw
                ls.Sp[1:-1, -2] -= Dw * T[1:-1, -2]

    def _solve_linear(self, ls, T):
        if self.linear_solver == "direct":
            SparseDirect.solve(T, Aw=ls.Aw, Ap=ls.Ap, Ae=ls.Ae, As=ls.As, An=ls.An, Sp=ls.Sp)
        else:
            TDMA.solve_line(T, Aw=ls.Aw, Ap=ls.Ap, Ae=ls.Ae, As=ls.As, An=ls.An, Sp=ls.Sp,
                            sweeps=100)

    # ------------------------------------------------------------ wall tangential
    def _wall_tangential_source(self, T, Gamma):
        """Tangential wall-flux correction using the energy-only copy of
        the non-orthogonal source formula.

        For each Dirichlet wall face *f*:
           t_f = S_f - alpha_f * d_f   (tangent to the wall)
           s_hat_f = unit vector along the wall (ghost cell to ghost cell)
           dT/ds_f = central difference along the wall ghost values
           Q_{t,f} = Gamma_f * (t_f . s_hat_f) * dT/ds_f

        The wall entries of *dnoc_x* / *dnoc_y* are filled with Q_t and the
        standard divergence formula (east/north +, west/south -) gives the
        correct cell source.  *self._q_t* is stored for wall_heat_flux.
        """
        s = self.s
        eps = 1e-30
        dnoc_x = np.zeros_like(s.x_interior, dtype=float)
        dnoc_y = np.zeros_like(s.y_interior, dtype=float)
        self._q_t = {}

        # ---- helpers for one wall ----
        def _wall_dtds(Tg):
            Xg = self.X[gidx]
            Yg = self.Y[gidx]
            n = len(Tg)
            dTds = np.zeros(n)
            s_hat_x = np.zeros(n)
            s_hat_y = np.zeros(n)
            for k in range(n):
                if k == 0:
                    dX = Xg[1] - Xg[0]; dY = Yg[1] - Yg[0]
                elif k == n - 1:
                    dX = Xg[-1] - Xg[-2]; dY = Yg[-1] - Yg[-2]
                else:
                    dX = Xg[k + 1] - Xg[k - 1]; dY = Yg[k + 1] - Yg[k - 1]
                dm = np.hypot(dX, dY) + eps
                dTds[k] = (Tg[min(k + 1, n - 1)] - Tg[max(k - 1, 0)]) / dm
                s_hat_x[k] = dX / dm
                s_hat_y[k] = dY / dm
            return dTds, s_hat_x, s_hat_y

        # --- west wall ---
        bw = self.bc.get("W")
        if bw is not None and bw[0] == "T":
            gidx = (0, slice(1, -1))
            sm2 = s.Sx_mag[0, :] ** 2
            alpha = np.divide(sm2, s.Sx_dot_dx[0, :] + eps)
            t_x = s.Sx_x[0, :] - alpha * s.dx_x[0, :]
            t_y = s.Sx_y[0, :] - alpha * s.dx_y[0, :]
            dTds, sh_x, sh_y = _wall_dtds(T[0, 1:-1])
            t_dot_s = t_x * sh_x + t_y * sh_y
            Gamma_f = 0.5 * (Gamma[0, 1:-1] + Gamma[1, 1:-1])
            dnoc_x[0, :] = Gamma_f * t_dot_s * dTds
            self._q_t["W"] = dnoc_x[0, :].copy()

        # --- east wall ---
        bw = self.bc.get("E")
        if bw is not None and bw[0] == "T":
            gidx = (-1, slice(1, -1))
            sm2 = s.Sx_mag[-1, :] ** 2
            alpha = np.divide(sm2, s.Sx_dot_dx[-1, :] + eps)
            t_x = s.Sx_x[-1, :] - alpha * s.dx_x[-1, :]
            t_y = s.Sx_y[-1, :] - alpha * s.dx_y[-1, :]
            dTds, sh_x, sh_y = _wall_dtds(T[-1, 1:-1])
            t_dot_s = t_x * sh_x + t_y * sh_y
            Gamma_f = 0.5 * (Gamma[-1, 1:-1] + Gamma[-2, 1:-1])
            dnoc_x[-1, :] = Gamma_f * t_dot_s * dTds
            self._q_t["E"] = dnoc_x[-1, :].copy()

        # --- south wall ---
        bw = self.bc.get("S")
        if bw is not None and bw[0] == "T":
            gidx = (slice(1, -1), 0)
            sm2 = s.Sy_mag[:, 0] ** 2
            alpha = np.divide(sm2, s.Sy_dot_dy[:, 0] + eps)
            t_x = s.Sy_x[:, 0] - alpha * s.dy_x[:, 0]
            t_y = s.Sy_y[:, 0] - alpha * s.dy_y[:, 0]
            dTds, sh_x, sh_y = _wall_dtds(T[1:-1, 0])
            t_dot_s = t_x * sh_x + t_y * sh_y
            Gamma_f = 0.5 * (Gamma[1:-1, 0] + Gamma[1:-1, 1])
            dnoc_y[:, 0] = Gamma_f * t_dot_s * dTds
            self._q_t["S"] = dnoc_y[:, 0].copy()

        # --- north wall ---
        bw = self.bc.get("N")
        if bw is not None and bw[0] == "T":
            gidx = (slice(1, -1), -1)
            sm2 = s.Sy_mag[:, -1] ** 2
            alpha = np.divide(sm2, s.Sy_dot_dy[:, -1] + eps)
            t_x = s.Sy_x[:, -1] - alpha * s.dy_x[:, -1]
            t_y = s.Sy_y[:, -1] - alpha * s.dy_y[:, -1]
            dTds, sh_x, sh_y = _wall_dtds(T[1:-1, -1])
            t_dot_s = t_x * sh_x + t_y * sh_y
            Gamma_f = 0.5 * (Gamma[1:-1, -1] + Gamma[1:-1, -2])
            dnoc_y[:, -1] = Gamma_f * t_dot_s * dTds
            self._q_t["N"] = dnoc_y[:, -1].copy()

        src = np.zeros((s.nx, s.ny))
        src[1:-1, 1:-1] = (dnoc_x[1:] - dnoc_x[:-1]
                           + dnoc_y[:, 1:] - dnoc_y[:, :-1])
        return src

    # ------------------------------------------------------------ solve / step
    def solve(self, fields, T, Gamma, source=None, tol=1e-8, max_iter=50,
              Gamma_face_x=None, Gamma_face_y=None):
        """Iterate the deferred terms to convergence with frozen mass fluxes."""
        for it in range(max_iter):
            T_old = T.copy()
            self.apply_bc(T, Gamma)
            self.apply_adiabatic_tangential(T, Gamma, n_iter=3)
            ls = self._assemble(fields, T, Gamma, source=source,
                                Gamma_face_x=Gamma_face_x,
                                Gamma_face_y=Gamma_face_y)
            self._solve_linear(ls, T)
            if np.max(np.abs(T[1:-1, 1:-1] - T_old[1:-1, 1:-1])) < tol:
                return it + 1
        return max_iter

    def step(self, fields, T, Gamma, source=None,
             Gamma_face_x=None, Gamma_face_y=None):
        """One assembly + solve (no iteration of deferred terms)."""
        self.apply_bc(T, Gamma)
        self.apply_adiabatic_tangential(T, Gamma, n_iter=3)
        ls = self._assemble(fields, T, Gamma, source=source,
                            Gamma_face_x=Gamma_face_x,
                            Gamma_face_y=Gamma_face_y)
        self._solve_linear(ls, T)

    # ------------------------------------------------------------ wall heat flux
    def _wall_conductance_x(self, Gamma, i):
        """Wall conductance D = Gamma_f * |S|^2 / (S.d) for x-face at grid line *i*."""
        s = self.s
        if self._Gamma_face_x is not None:
            Gamma_f = self._Gamma_face_x[i, :]
        else:
            interior_idx_offset = 1 if i == 0 else -1
            Gamma_f = 0.5 * (Gamma[i, 1:-1] + Gamma[i + interior_idx_offset, 1:-1])
        return Gamma_f * s.Sx_mag[i, :] ** 2 / s.Sx_dot_dx[i, :]

    def _wall_conductance_y(self, Gamma, j):
        s = self.s
        if self._Gamma_face_y is not None:
            Gamma_f = self._Gamma_face_y[:, j]
        else:
            interior_idx_offset = 1 if j == 0 else -1
            Gamma_f = 0.5 * (Gamma[1:-1, j] + Gamma[1:-1, j + interior_idx_offset])
        return Gamma_f * s.Sy_mag[:, j] ** 2 / s.Sy_dot_dy[:, j]

    def wall_heat_flux(self, side, fields, T, Gamma, cp=1.0):
        """Per-face wall heat flux Q_f (positive = heat INTO the fluid).

        Returns ``(Q_f, Q_total)``.  Adiabatic walls return exactly zero.
        """
        bc_val = self.bc.get(side)
        if bc_val is not None and bc_val[0] == "q":
            n = T.shape[1] - 2 if side in ("W", "E") else T.shape[0] - 2
            return np.zeros(n), 0.0
        # ensure adiabatic ghosts are consistent before computing flux
        self.apply_adiabatic_tangential(T, Gamma, n_iter=1)
        s = self.s
        if side == "W":
            D = self._wall_conductance_x(Gamma, 0)
            T_ghost, T_P = T[0, 1:-1], T[1, 1:-1]
            S_mag = s.Sx_mag[0, :]
            Q_t = self._q_t.get("W", np.zeros_like(D))
        elif side == "E":
            D = self._wall_conductance_x(Gamma, -1)
            T_ghost, T_P = T[-1, 1:-1], T[-2, 1:-1]
            S_mag = s.Sx_mag[-1, :]
            Q_t = self._q_t.get("E", np.zeros_like(D))
            Q_t = -Q_t
        elif side == "S":
            D = self._wall_conductance_y(Gamma, 0)
            T_ghost, T_P = T[1:-1, 0], T[1:-1, 1]
            S_mag = s.Sy_mag[:, 0]
            Q_t = self._q_t.get("S", np.zeros_like(D))
        elif side == "N":
            D = self._wall_conductance_y(Gamma, -1)
            T_ghost, T_P = T[1:-1, -1], T[1:-1, -2]
            S_mag = s.Sy_mag[:, -1]
            Q_t = self._q_t.get("N", np.zeros_like(D))
            Q_t = -Q_t
        else:
            raise ValueError(f"Unknown side {side!r}")
        Q_f = cp * D * (T_ghost - T_P) + cp * Q_t
        Q_total = float(np.sum(Q_f))
        return Q_f, Q_total

    def nusselt(self, side, fields, T, Gamma, cp, k0, dT, L_ref=None):
        """Mean and local Nusselt number on a wall.

        ``Nu_mean = |Q_side| / (k0 * dT)`` (per unit depth, streamwise L cancels).
        ``Nu_local = Q_f / |S_f| * L_ref / (k0 * dT)`` (if *L_ref* is given).
        Returns ``(Nu_mean, Nu_local)``.
        """
        Q_f, Q_total = self.wall_heat_flux(side, fields, T, Gamma, cp=cp)
        Nu_mean = abs(Q_total) / (k0 * dT)
        if L_ref is not None:
            if side in ("W", "E"):
                S_mag = self.s.Sy_mag[:, 0] if side == "W" else self.s.Sy_mag[:, -1]
            else:
                S_mag = self.s.Sx_mag[0, :] if side == "S" else self.s.Sx_mag[-1, :]
            Nu_local = np.where(S_mag > 0, Q_f / S_mag * L_ref / (k0 * dT), 0.0)
        else:
            Nu_local = None
        return Nu_mean, Nu_local