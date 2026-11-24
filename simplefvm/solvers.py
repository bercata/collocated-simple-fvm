"""Linear solvers and the SIMPLE building blocks (first-version classes, vectorised and corrected)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .linsys import FVLinearSystem
from .operators import FVM_div, FVM_grad, FVM_laplacian, deferred_correction_source
from .stencil import x_face_avg, y_face_avg


# ============================================================================ linear solvers
class TDMA:
    """Line-by-line Thomas solver.

    One call performs ``sweeps`` alternating-direction passes: all x-lines are solved at once
    (vectorised over j), then all y-lines (vectorised over i). Ghost values are fixed boundary
    data. The first version swept one line at a time in Python, which is the same method
    but roughly two orders of magnitude slower.
    """

    @staticmethod
    def _sweep_axis0(phi, Aw, Ap, Ae, As, An, Sp):
        I = (slice(1, -1), slice(1, -1))
        rhs = Sp[I] + As[I] * phi[1:-1, :-2] + An[I] * phi[1:-1, 2:]
        aw, ap, ae = Aw[I], Ap[I], Ae[I]
        n = ap.shape[0]
        P = np.empty_like(ap)
        Q = np.empty_like(ap)
        P[0] = ae[0] / ap[0]
        Q[0] = (rhs[0] + aw[0] * phi[0, 1:-1]) / ap[0]
        for k in range(1, n):
            denom = ap[k] - aw[k] * P[k - 1]
            P[k] = ae[k] / denom
            Q[k] = (rhs[k] + aw[k] * Q[k - 1]) / denom
        phi[n, 1:-1] = P[n - 1] * phi[n + 1, 1:-1] + Q[n - 1]
        for k in range(n - 2, -1, -1):
            phi[k + 1, 1:-1] = P[k] * phi[k + 2, 1:-1] + Q[k]

    @classmethod
    def solve_line(cls, phi, *, Aw, Ap, Ae, As, An, Sp, sweeps=1):
        for _ in range(sweeps):
            cls._sweep_axis0(phi, Aw, Ap, Ae, As, An, Sp)                     # x-lines
            cls._sweep_axis0(phi.T, As.T, Ap.T, An.T, Aw.T, Ae.T, Sp.T)       # y-lines
        return phi


class SparseDirect:
    """Direct sparse solve of the five-point system on the interior cells (scipy SuperLU).

    Used for the pressure-correction equation, where an exact solve per outer iteration is
    cheaper overall than many TDMA sweeps. ``pin=(i, j, value)`` fixes one cell, which removes
    the constant null space of the all-Neumann pressure-correction problem.
    """

    @staticmethod
    def factorize(phi, *, Aw, Ap, Ae, As, An, pin=None):
        I = (slice(1, -1), slice(1, -1))
        M = phi.shape[0] - 2
        aP = Ap[I].copy()
        aE, aW, aN, aS = Ae[I].copy(), Aw[I].copy(), An[I].copy(), As[I].copy()
        aE[-1, :] = 0.0; aW[0, :] = 0.0; aN[:, -1] = 0.0; aS[:, 0] = 0.0
        if pin is not None:
            pi, pj, val = pin
            k = (pi - 1, pj - 1)
            aP[k] = 1.0
            aE[k] = aW[k] = aN[k] = aS[k] = 0.0
        N = M * M
        A = sp.diags(
            [aP.ravel(), -aE[:-1].ravel(), -aW[1:].ravel(), -aN.ravel()[:-1], -aS.ravel()[1:]],
            [0, M, -M, 1, -1], shape=(N, N), format="csc")
        return spla.splu(A)

    @staticmethod
    def solve(phi, *, Aw, Ap, Ae, As, An, Sp, pin=None):
        I = (slice(1, -1), slice(1, -1))
        M = phi.shape[0] - 2
        aP = Ap[I].copy()
        aE, aW, aN, aS = Ae[I].copy(), Aw[I].copy(), An[I].copy(), As[I].copy()
        rhs = Sp[I].copy()
        rhs[-1, :] += aE[-1, :] * phi[-1, 1:-1]; aE[-1, :] = 0.0
        rhs[0, :] += aW[0, :] * phi[0, 1:-1];    aW[0, :] = 0.0
        rhs[:, -1] += aN[:, -1] * phi[1:-1, -1]; aN[:, -1] = 0.0
        rhs[:, 0] += aS[:, 0] * phi[1:-1, 0];    aS[:, 0] = 0.0
        if pin is not None:
            pi, pj, val = pin
            k = (pi - 1, pj - 1)
            aP[k], rhs[k] = 1.0, val
            aE[k] = aW[k] = aN[k] = aS[k] = 0.0
        N = M * M
        A = sp.diags(
            [aP.ravel(), -aE[:-1].ravel(), -aW[1:].ravel(), -aN.ravel()[:-1], -aS.ravel()[1:]],
            [0, M, -M, 1, -1], shape=(N, N), format="csc")
        phi[I] = spla.spsolve(A, rhs.ravel()).reshape(M, M)
        return phi


# ======================================================================= convergence monitor
class ConvergenceMonitor:
    """Tracks scaled residual histories and tests convergence.

    ``update`` keeps the first-version signature (algebraic residual of one equation). The
    SIMPLE driver uses ``record`` with the already scaled residuals of all equations.
    """

    def __init__(self, atol=1e-6, rtol=1e-3, max_it=1000, norm="L1"):
        self.atol, self.rtol, self.max_it, self.norm = atol, rtol, max_it, norm
        self.reset()

    def reset(self):
        self.hist = []
        self.iter = 0

    def _norm(self, R):
        if self.norm == "L1":
            return float(np.sum(np.abs(R)))
        if self.norm == "L2":
            return float(np.sqrt(np.mean(R ** 2)))
        return float(np.max(np.abs(R)))

    def update(self, Ap, Aw, Ae, As, An, Sp, phi):
        c = (slice(1, -1), slice(1, -1))
        R = (Ap[c] * phi[c] - Aw[c] * phi[:-2, 1:-1] - Ae[c] * phi[2:, 1:-1]
             - As[c] * phi[1:-1, :-2] - An[c] * phi[1:-1, 2:] - Sp[c])
        return self.record(self._norm(R))

    def record(self, res):
        self.hist.append(res)
        self.iter += 1
        if res < self.atol or res / (self.hist[0] + 1e-30) < self.rtol:
            return True
        if self.iter >= self.max_it:
            raise RuntimeError(f"SIMPLE did not converge: residual {res:.3e} after {self.iter} iterations")
        return False


# ========================================================================= momentum equation
class MomentumEquationSolver:
    """Assembles and (partially) solves one momentum component with TDMA sweeps.

    The first version solved each momentum system to convergence with up to 550 sweeps per
    outer iteration. In SIMPLE that work is wasted, because the coefficients change at the
    next outer iteration; a few sweeps per outer iteration is the standard choice.
    """

    def __init__(self, stencil, X, Y, X_cor, Y_cor, inner_sweeps=2, gamma_dc=0.0):
        self.gamma_dc = gamma_dc     # 0: first-order upwind (first version); 1: central via deferred correction
        self.stencil = stencil
        self.X, self.Y, self.X_cor, self.Y_cor = X, Y, X_cor, Y_cor
        self.grad = FVM_grad(X, Y, X_cor, Y_cor, stencil)
        self.div = FVM_div(X, Y, X_cor, Y_cor, stencil)
        self.lap = FVM_laplacian(X, Y, X_cor, Y_cor, stencil)
        self.tdma2d = TDMA()
        self.inner_sweeps = inner_sweeps
        self.residual = {}

    def solve(self, fields, *, alpha: float = 0.7, component: str = "u",
              fx=None, fy=None, vel_scale=None, stress_source=None,
              mu_face_x=None, mu_face_y=None):
        comp = component.lower()
        if comp not in ("u", "v"):
            raise ValueError("component must be 'u' or 'v'")
        var = fields.U if comp == "u" else fields.V
        gx, gy = self.grad.compute_gradient(fields.P)
        dp = gx if comp == "u" else gy

        fluxes = self.div.compute_face_fluxes(var, fields.U, fields.V, fields.rho,
                                              mflux=(fields.mflux_x, fields.mflux_y))
        diffs = self.lap.compute_face_diffusion(var, fields.mu,
                                                Gamma_face_x=mu_face_x,
                                                Gamma_face_y=mu_face_y)

        Nx, Ny = var.shape
        ls = FVLinearSystem(Nx, Ny)
        ls.assemble_raw(fluxes["flux_e"], fluxes["flux_w"], fluxes["flux_n"], fluxes["flux_s"],
                        diffs["Diff_e"], diffs["Diff_w"], diffs["Diff_n"], diffs["Diff_s"],
                        np.zeros((Nx, Ny)), self.X_cor, self.Y_cor, Dp=dp, vol=self.stencil.vol)
        if self.gamma_dc > 0.0:
            self._deferred_correction(ls, var, fields.mflux_x, fields.mflux_y)

        # non-orthogonal diffusion correction
        gvx, gvy = self.grad.compute_gradient(var)
        src_noc = self.lap.nonorthogonal_correction_source(gvx, gvy, fields.mu,
                                                           Gamma_face_x=mu_face_x,
                                                           Gamma_face_y=mu_face_y)
        ls.Sp[1:-1, 1:-1] += src_noc[1:-1, 1:-1]

        # body force (per unit volume): Sp += f * vol
        if comp == "u" and fx is not None:
            ls.Sp[1:-1, 1:-1] += fx[1:-1, 1:-1] * self.stencil.vol
        if comp == "v" and fy is not None:
            ls.Sp[1:-1, 1:-1] += fy[1:-1, 1:-1] * self.stencil.vol
        # extra stress source (explicit, full tensor minus laplacian)
        if stress_source is not None:
            ls.Sp[1:-1, 1:-1] += stress_source[1:-1, 1:-1]

        ls.apply_under_relax(phi_old=var, alpha=alpha)

        # scaled residual
        R = ls.residual(var)
        if vel_scale is not None:
            scale = np.sum(np.abs(ls.Ap[1:-1, 1:-1])) * vel_scale + 1e-30
        else:
            scale = np.sum(np.abs(ls.Ap[1:-1, 1:-1] * var[1:-1, 1:-1])) + 1e-30
        self.residual[comp] = float(np.sum(np.abs(R)) / scale)

        self.tdma2d.solve_line(var, Aw=ls.Aw, Ap=ls.Ap, Ae=ls.Ae, As=ls.As, An=ls.An, Sp=ls.Sp,
                               sweeps=self.inner_sweeps)
        return var, ls.Ap, ls.Aw, ls.Ae, ls.As, ls.An, ls.Sp

    def _deferred_correction(self, ls, phi, mx, my):
        """Khosla-Rubin deferred correction: the matrix keeps the upwind coefficients, and
        gamma * F_f (phi_central - phi_upwind) from the current iterate is added to the source.
        At convergence the convective term is central (second order) for gamma = 1.
        Wall faces carry no mass flux, so they contribute nothing."""
        dcsrc = deferred_correction_source(phi, mx, my, self.gamma_dc)
        ls.Sp[1:-1, 1:-1] -= dcsrc[1:-1, 1:-1]


# ====================================================================== pressure correction
class RhieChowPressureCorrection:
    """Pressure-correction (p') equation on the collocated grid with Rhie-Chow face fluxes.

    Face velocity (normal component), as in the first version but along the face normal:
        u_f = avg(u).n + avg(1/a_P) V_f [ avg(grad p).d_hat - (p_N - p_P)/|d| ]
    p' equation: a_f = rho_f avg(1/a_P) |S_f|^2 on interior faces and zero on walls
    (zero-gradient p'), source = - net mass outflow of the cell.
    After the solve the face mass fluxes are corrected with the full p' and stored on the
    fields; the next momentum assembly uses them.
    """

    def __init__(self, stencil, X, Y, X_cor, Y_cor, reference=(1, 1, 0.0),
                 tol=1e-8, max_inner=500, linear_solver="direct",
                 n_nonorth_correctors=0, consistent_rc=True, body_force_rc=False):
        self.s = stencil
        self.X, self.Y, self.X_cor, self.Y_cor = X, Y, X_cor, Y_cor
        self.reference = reference
        self.tol, self.max_inner = tol, max_inner
        self.linear_solver = linear_solver
        self.n_nonorth_correctors = n_nonorth_correctors
        self.consistent_rc = consistent_rc
        self.body_force_rc = body_force_rc
        self.grad = FVM_grad(X, Y, X_cor, Y_cor, stencil)
        self.m, self.n = X.shape
        self.mass_residual = np.inf

    def _face_flux_x(self, f, gx, gy, DpU, U_old=None, V_old=None, mflux_old=None,
                     alpha_u=None, first_iteration=False, fx=None, fy=None,
                     fx_face=None, fy_face=None):
        s = self.s
        un_bar = (x_face_avg(f.U) * s.Sx_x + x_face_avg(f.V) * s.Sx_y) / s.Sx_mag
        g_avg = (x_face_avg(gx) * s.dx_x + x_face_avg(gy) * s.dx_y) / s.dx_mag
        g_dir = (f.P[1:, 1:-1] - f.P[:-1, 1:-1]) / s.dx_mag
        if self.body_force_rc and fx is not None and fy is not None:
            bf_avg = (x_face_avg(fx) * s.dx_x + x_face_avg(fy) * s.dx_y) / s.dx_mag
            if fx_face is not None and fy_face is not None:
                bf_dir = (fx_face * s.dx_x + fy_face * s.dx_y) / s.dx_mag
            else:
                bf_dir = bf_avg
            g_avg = g_avg - bf_avg
            g_dir = g_dir - bf_dir
        dpf = x_face_avg(DpU)
        rho_f = x_face_avg(f.rho)
        un = un_bar + dpf * (s.dx_mag * s.Sx_mag) * (g_avg - g_dir)
        if self.consistent_rc and not first_iteration and alpha_u is not None and mflux_old is not None:
            un_bar_old = (x_face_avg(U_old) * s.Sx_x + x_face_avg(V_old) * s.Sx_y) / s.Sx_mag
            un_face_old = mflux_old / (rho_f * s.Sx_mag + 1e-30)
            un = un + (1.0 - alpha_u) * (un_face_old - un_bar_old)
        m = rho_f * un * s.Sx_mag
        a = rho_f * dpf * s.Sx_mag ** 2
        m[~s.x_interior] = 0.0
        a[~s.x_interior] = 0.0
        return m, a

    def _face_flux_y(self, f, gx, gy, DpV, U_old=None, V_old=None, mflux_old=None,
                     alpha_u=None, first_iteration=False, fx=None, fy=None,
                     fx_face=None, fy_face=None):
        s = self.s
        un_bar = (y_face_avg(f.U) * s.Sy_x + y_face_avg(f.V) * s.Sy_y) / s.Sy_mag
        g_avg = (y_face_avg(gx) * s.dy_x + y_face_avg(gy) * s.dy_y) / s.dy_mag
        g_dir = (f.P[1:-1, 1:] - f.P[1:-1, :-1]) / s.dy_mag
        if self.body_force_rc and fx is not None and fy is not None:
            bf_avg = (y_face_avg(fx) * s.dy_x + y_face_avg(fy) * s.dy_y) / s.dy_mag
            if fx_face is not None and fy_face is not None:
                bf_dir = (fx_face * s.dy_x + fy_face * s.dy_y) / s.dy_mag
            else:
                bf_dir = bf_avg
            g_avg = g_avg - bf_avg
            g_dir = g_dir - bf_dir
        dpf = y_face_avg(DpV)
        rho_f = y_face_avg(f.rho)
        un = un_bar + dpf * (s.dy_mag * s.Sy_mag) * (g_avg - g_dir)
        if self.consistent_rc and not first_iteration and alpha_u is not None and mflux_old is not None:
            un_bar_old = (y_face_avg(U_old) * s.Sy_x + y_face_avg(V_old) * s.Sy_y) / s.Sy_mag
            un_face_old = mflux_old / (rho_f * s.Sy_mag + 1e-30)
            un = un + (1.0 - alpha_u) * (un_face_old - un_bar_old)
        m = rho_f * un * s.Sy_mag
        a = rho_f * dpf * s.Sy_mag ** 2
        m[~s.y_interior] = 0.0
        a[~s.y_interior] = 0.0
        return m, a

    def solve(self, fields, U_old=None, V_old=None, mflux_x_old=None, mflux_y_old=None,
              alpha_u=None, first_iteration=False, fx=None, fy=None,
              fx_face_x=None, fy_face_x=None, fx_face_y=None, fy_face_y=None):
        s = self.s
        f = fields
        DpU = np.zeros_like(f.U); DpU[1:-1, 1:-1] = 1.0 / (f.Ap_u[1:-1, 1:-1] + 1e-30)
        DpV = np.zeros_like(f.V); DpV[1:-1, 1:-1] = 1.0 / (f.Ap_v[1:-1, 1:-1] + 1e-30)
        for D in (DpU, DpV):
            D[0, :], D[-1, :], D[:, 0], D[:, -1] = D[1, :], D[-2, :], D[:, 1], D[:, -2]
        f.DpU, f.DpV = DpU, DpV

        gx, gy = self.grad.compute_gradient(f.P)
        mx, ax = self._face_flux_x(f, gx, gy, DpU, U_old, V_old, mflux_x_old,
                                   alpha_u, first_iteration, fx, fy,
                                   fx_face_x, fy_face_x)
        my, ay = self._face_flux_y(f, gx, gy, DpV, U_old, V_old, mflux_y_old,
                                   alpha_u, first_iteration, fx, fy,
                                   fx_face_y, fy_face_y)

        shp = (self.m, self.n)
        Aw, Ae, As, An, Ap, Sp = (np.zeros(shp) for _ in range(6))
        I = (slice(1, -1), slice(1, -1))
        Ae[I], Aw[I] = ax[1:], ax[:-1]
        An[I], As[I] = ay[:, 1:], ay[:, :-1]
        Ap[I] = Ae[I] + Aw[I] + An[I] + As[I]
        Sp0 = -(mx[1:] - mx[:-1] + my[:, 1:] - my[:, :-1])
        self.mass_residual = float(np.sum(np.abs(Sp0)))

        eps = 1e-30

        # alpha = |S_f|^2 / (S_f . d_f)  (over-relaxed factor)
        alpha_x = np.divide(s.Sx_mag * s.Sx_mag, s.Sx_dot_dx + eps)
        alpha_y = np.divide(s.Sy_mag * s.Sy_mag, s.Sy_dot_dy + eps)

        dnoc_x = np.zeros_like(mx)
        dnoc_y = np.zeros_like(my)

        p_prime = np.zeros(shp)
        M = shp[0] - 2
        if self.linear_solver == "direct":
            lu = SparseDirect.factorize(p_prime, Aw=Aw, Ap=Ap, Ae=Ae, As=As, An=An,
                                        pin=self.reference)
        for k in range(self.n_nonorth_correctors + 1):
            div_noc = dnoc_x[1:] - dnoc_x[:-1] + dnoc_y[:, 1:] - dnoc_y[:, :-1]
            Sp[I] = Sp0 + div_noc

            if self.linear_solver == "direct":
                rhs = Sp[I].copy()
                rhs[-1, :] += Ae[I][-1, :] * p_prime[-1, 1:-1]
                rhs[0, :] += Aw[I][0, :] * p_prime[0, 1:-1]
                rhs[:, -1] += An[I][:, -1] * p_prime[1:-1, -1]
                rhs[:, 0] += As[I][:, 0] * p_prime[1:-1, 0]
                ri, rj, rv = self.reference
                rhs[ri - 1, rj - 1] = rv
                p_prime[I] = lu.solve(rhs.ravel()).reshape(M, M)
            else:
                ri, rj, rv = self.reference
                Ap[ri, rj], Sp[ri, rj] = 1.0, rv
                Aw[ri, rj] = Ae[ri, rj] = As[ri, rj] = An[ri, rj] = 0.0
                for _ in range(self.max_inner):
                    TDMA.solve_line(p_prime, Aw=Aw, Ap=Ap, Ae=Ae, As=As, An=An, Sp=Sp)
                    R = (Ap[I] * p_prime[I] - Aw[I] * p_prime[:-2, 1:-1]
                         - Ae[I] * p_prime[2:, 1:-1] - As[I] * p_prime[1:-1, :-2]
                         - An[I] * p_prime[1:-1, 2:] - Sp[I])
                    if np.max(np.abs(R)) < self.tol:
                        break

            if k < self.n_nonorth_correctors:
                # zero-gradient p' ghosts BEFORE gradient (were stale/zero from init)
                p_prime[0, :], p_prime[-1, :] = p_prime[1, :], p_prime[-2, :]
                p_prime[:, 0], p_prime[:, -1] = p_prime[:, 1], p_prime[:, -2]
                gpx, gpy = self.grad.compute_gradient(p_prime)
                # x-face non-orthogonal mass flux:
                #   dnoc = rho_f * (V/a_P)_f * (grad p')_f . (S_f - alpha * d_f)
                #   (V/a_P)_f = dpf * V_face  with V_face = Sx_dot_dx
                noc_xvec = s.Sx_x - alpha_x * s.dx_x
                noc_yvec = s.Sx_y - alpha_x * s.dx_y
                gx_f = 0.5 * (gpx[:-1, 1:-1] + gpx[1:, 1:-1])
                gy_f = 0.5 * (gpy[:-1, 1:-1] + gpy[1:, 1:-1])
                dnoc_x = (x_face_avg(f.rho) * x_face_avg(f.DpU)
                          * s.Sx_dot_dx
                          * (noc_xvec * gx_f + noc_yvec * gy_f))
                dnoc_x[~s.x_interior] = 0.0
                # y-face
                noc_xvec = s.Sy_x - alpha_y * s.dy_x
                noc_yvec = s.Sy_y - alpha_y * s.dy_y
                gx_f = 0.5 * (gpx[1:-1, :-1] + gpx[1:-1, 1:])
                gy_f = 0.5 * (gpy[1:-1, :-1] + gpy[1:-1, 1:])
                dnoc_y = (y_face_avg(f.rho) * y_face_avg(f.DpV)
                          * s.Sy_dot_dy
                          * (noc_xvec * gx_f + noc_yvec * gy_f))
                dnoc_y[~s.y_interior] = 0.0

        # zero-gradient p' in ghosts
        p_prime[0, :], p_prime[-1, :] = p_prime[1, :], p_prime[-2, :]
        p_prime[:, 0], p_prime[:, -1] = p_prime[:, 1], p_prime[:, -2]

        # face mass fluxes with implicit part + explicit non-orthogonal part
        # both use the dnoc that was used in the last p' solve
        f.mflux_x = (mx - ax * (p_prime[1:, 1:-1] - p_prime[:-1, 1:-1])
                     - dnoc_x)
        f.mflux_y = (my - ay * (p_prime[1:-1, 1:] - p_prime[1:-1, :-1])
                     - dnoc_y)

        div_corr = np.abs(f.mflux_x[1:] - f.mflux_x[:-1]
                          + f.mflux_y[:, 1:] - f.mflux_y[:, :-1])
        self.post_corr_imbalance = float(np.max(div_corr))

        f.p_prime = p_prime
        return p_prime


class PressureVelocityCorrector:
    """SIMPLE correction step (same order as the first version):
    1. P <- P + alpha_p p'   2. u, v <- u, v - (V/a_P) grad p'   3. P ghosts
    4. gauge: subtract the pressure of one interior cell.

    Without body force the ghosts are zero-gradient; with body force they
    satisfy the hydrostatic balance  dp/dn = f.n  at each wall.
    """

    def __init__(self, stencil, X_cor, Y_cor, gauge=(2, 3)):
        self.s = stencil
        self.X = stencil.X
        self.Y = stencil.Y
        self.grad = FVM_grad(stencil.X, stencil.Y, X_cor, Y_cor, stencil)
        self.gauge = gauge

    def apply(self, fields, p_prime, alpha_p=0.3, fx=None, fy=None):
        f = fields
        I = (slice(1, -1), slice(1, -1))
        f.P[:, :] += alpha_p * p_prime
        gpx, gpy = self.grad.compute_gradient(p_prime)
        f.U[I] -= f.DpU[I] * self.s.vol * gpx[I]
        f.V[I] -= f.DpV[I] * self.s.vol * gpy[I]
        P = f.P
        self._set_pressure_ghosts(P, fx, fy)
        P -= P[self.gauge]

    def _set_pressure_ghosts(self, P, fx=None, fy=None):
        """Set pressure ghosts.  Without body force: zero-gradient (existing behaviour).
        With body force: P_ghost = P_P + f_w . (x_ghost - x_P)."""
        X, Y = self.X, self.Y
        if fx is None or fy is None:
            P[:, 0], P[:, -1] = P[:, 1], P[:, -2]
            P[0, :], P[-1, :] = P[1, :], P[-2, :]
            return
        # S wall (j=0): ghost below first interior cell
        P[1:-1, 0] = (P[1:-1, 1]
                      + fx[1:-1, 0] * (X[1:-1, 0] - X[1:-1, 1])
                      + fy[1:-1, 0] * (Y[1:-1, 0] - Y[1:-1, 1]))
        # N wall (j=-1): ghost above last interior cell
        P[1:-1, -1] = (P[1:-1, -2]
                       + fx[1:-1, -1] * (X[1:-1, -1] - X[1:-1, -2])
                       + fy[1:-1, -1] * (Y[1:-1, -1] - Y[1:-1, -2]))
        # W wall (i=0): ghost left of first interior cell
        P[0, 1:-1] = (P[1, 1:-1]
                      + fx[0, 1:-1] * (X[0, 1:-1] - X[1, 1:-1])
                      + fy[0, 1:-1] * (Y[0, 1:-1] - Y[1, 1:-1]))
        # E wall (i=-1): ghost right of last interior cell
        P[-1, 1:-1] = (P[-2, 1:-1]
                       + fx[-1, 1:-1] * (X[-1, 1:-1] - X[-2, 1:-1])
                       + fy[-1, 1:-1] * (Y[-1, 1:-1] - Y[-2, 1:-1]))


# ========================================================================= balanced body force
def compute_balanced_body_force(T, buoy, T_ref, stencil, fx_out, fy_out):
    """Fill *fx_out*, *fy_out* (padded) with the face-consistent body force.

    The force at each interior cell is built from face-interpolated values
    so that  p_N - p_P = f_f . d_f  on every face (discrete hydrostatic)
    implies  f_balanced = Gauss_gradient(p).  The formula (Francois et al. / Mencinger & Zun):

        f_P = 1/(2 V_P) Σ_interior (f_f . d_f) S_f  +  1/V_P Σ_walls (f_w . d_f) S_f

    All S_f and d_f come from the stencil (+index).  The sign rule (both d and S
    point OUT of cell P, product unchanged) means every face contributes
    +(f . d) S to both adjacent cells — no minus sign for west/south faces.

    *stencil.vol* has shape (M, M) — interior cells only, 0-indexed.
    Interior cell (i, j) in 1-based indexing maps to vol[i-1, j-1].

    Wall faces use the ghost (wall-face) temperature directly, not 0.5*(T_g + T_P).
    Ghost force values (wall FACE force) are stored in the output for use by
    _set_pressure_ghosts.
    """
    s = stencil
    M = s.M
    fx_out.fill(0.0)
    fy_out.fill(0.0)

    # ---- face-interpolated y-force (fx=0 for buoyancy) ----
    # interior: arithmetic mean of the two cell values
    Tx = 0.5 * (T[:-1, 1:-1] + T[1:, 1:-1])       # x-faces  (M+1, M)
    Ty = 0.5 * (T[1:-1, :-1] + T[1:-1, 1:])       # y-faces  (M, M+1)
    # wall faces: use ghost value directly  (T at wall face centre)
    Tx[0, :] = T[0, 1:-1];  Tx[M, :] = T[-1, 1:-1]
    Ty[:, 0] = T[1:-1, 0];  Ty[:, M] = T[1:-1, -1]
    fy_x = buoy * (Tx - T_ref)
    fy_y = buoy * (Ty - T_ref)

    # ---- interior x-faces (grid_line i = 1 .. M-1) ----
    fd = fy_x[1:M, :] * s.dx_y[1:M, :]             # (M-1, M)
    sx = s.Sx_x[1:M, :]; sy = s.Sx_y[1:M, :]
    fx_out[1:M, 1:-1] += fd * sx / (2.0 * s.vol[:M - 1, :])
    fy_out[1:M, 1:-1] += fd * sy / (2.0 * s.vol[:M - 1, :])
    fx_out[2:M + 1, 1:-1] += fd * sx / (2.0 * s.vol[1:M, :])
    fy_out[2:M + 1, 1:-1] += fd * sy / (2.0 * s.vol[1:M, :])

    # ---- interior y-faces (grid_line j = 1 .. M-1) ----
    fd = fy_y[:, 1:M] * s.dy_y[:, 1:M]             # (M, M-1)
    sx = s.Sy_x[:, 1:M]; sy = s.Sy_y[:, 1:M]
    fx_out[1:-1, 1:M] += fd * sx / (2.0 * s.vol[:, :M - 1])
    fy_out[1:-1, 1:M] += fd * sy / (2.0 * s.vol[:, :M - 1])
    fx_out[1:-1, 2:M + 1] += fd * sx / (2.0 * s.vol[:, 1:M])
    fy_out[1:-1, 2:M + 1] += fd * sy / (2.0 * s.vol[:, 1:M])

    # ---- wall faces (all four, one block) ----
    # West (i=0)
    fd = fy_x[0, :] * s.dx_y[0, :]
    fx_out[1, 1:-1] += fd * s.Sx_x[0, :] / s.vol[0, :]
    fy_out[1, 1:-1] += fd * s.Sx_y[0, :] / s.vol[0, :]
    fy_out[0, 1:-1] = fy_x[0, :]     # west ghost fy  (wall FACE force)
    # East (i=M)
    fd = fy_x[M, :] * s.dx_y[M, :]
    fx_out[M, 1:-1] += fd * s.Sx_x[M, :] / s.vol[M - 1, :]
    fy_out[M, 1:-1] += fd * s.Sx_y[M, :] / s.vol[M - 1, :]
    fy_out[-1, 1:-1] = fy_x[M, :]    # east ghost fy
    # South (j=0)
    fd = fy_y[:, 0] * s.dy_y[:, 0]
    fx_out[1:-1, 1] += fd * s.Sy_x[:, 0] / s.vol[:, 0]
    fy_out[1:-1, 1] += fd * s.Sy_y[:, 0] / s.vol[:, 0]
    fy_out[1:-1, 0] = fy_y[:, 0]     # south ghost fy
    # North (j=M)
    fd = fy_y[:, M] * s.dy_y[:, M]
    fx_out[1:-1, M] += fd * s.Sy_x[:, M] / s.vol[:, M - 1]
    fy_out[1:-1, M] += fd * s.Sy_y[:, M] / s.vol[:, M - 1]
    fy_out[1:-1, -1] = fy_y[:, M]    # north ghost fy


def compute_face_forces(T, buoy, T_ref):
    """Return (fy_face_x, fy_face_y) with the body-force y-component at FACE centres.

    fy_face_x  shape (M+1, M)   — x-face y-force
    fy_face_y  shape (M, M+1)   — y-face y-force

    Interior faces use linear interpolation; wall faces use the ghost (wall-face) T.
    """
    M = T.shape[0] - 2
    Tx = 0.5 * (T[:-1, 1:-1] + T[1:, 1:-1])        # (M+1, M)
    Ty = 0.5 * (T[1:-1, :-1] + T[1:-1, 1:])        # (M, M+1)
    Tx[0, :] = T[0, 1:-1];  Tx[M, :] = T[-1, 1:-1]
    Ty[:, 0] = T[1:-1, 0];  Ty[:, M] = T[1:-1, -1]
    return buoy * (Tx - T_ref), buoy * (Ty - T_ref)
