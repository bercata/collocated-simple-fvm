"""Linear solvers and the SIMPLE building blocks (first-version classes, vectorised and corrected)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .linsys import FVLinearSystem
from .operators import FVM_div, FVM_grad, FVM_laplacian
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
    def solve(phi, *, Aw, Ap, Ae, As, An, Sp, pin=None):
        I = (slice(1, -1), slice(1, -1))
        M = phi.shape[0] - 2
        aP = Ap[I].copy()
        aE, aW, aN, aS = Ae[I].copy(), Aw[I].copy(), An[I].copy(), As[I].copy()
        rhs = Sp[I].copy()
        # couplings to ghost cells move to the right-hand side
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

    def solve(self, fields, *, alpha: float = 0.7, component: str = "u"):
        comp = component.lower()
        if comp not in ("u", "v"):
            raise ValueError("component must be 'u' or 'v'")
        var = fields.U if comp == "u" else fields.V
        gx, gy = self.grad.compute_gradient(fields.P)
        dp = gx if comp == "u" else gy

        fluxes = self.div.compute_face_fluxes(var, fields.U, fields.V, fields.rho,
                                              mflux=(fields.mflux_x, fields.mflux_y))
        diffs = self.lap.compute_face_diffusion(var, fields.mu)

        Nx, Ny = var.shape
        ls = FVLinearSystem(Nx, Ny)
        ls.assemble_raw(fluxes["flux_e"], fluxes["flux_w"], fluxes["flux_n"], fluxes["flux_s"],
                        diffs["Diff_e"], diffs["Diff_w"], diffs["Diff_n"], diffs["Diff_s"],
                        np.zeros((Nx, Ny)), self.X_cor, self.Y_cor, Dp=dp, vol=self.stencil.vol)
        if self.gamma_dc > 0.0:
            self._deferred_correction(ls, var, fields.mflux_x, fields.mflux_y)
        ls.apply_under_relax(phi_old=var, alpha=alpha)

        # scaled residual before the inner solve (relaxation terms cancel at phi_old)
        R = ls.residual(var)
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
        g = self.gamma_dc
        P, N = phi[:-1, 1:-1], phi[1:, 1:-1]                       # x-faces
        cx = mx * (0.5 * (P + N) - np.where(mx > 0, P, N))
        P, N = phi[1:-1, :-1], phi[1:-1, 1:]                       # y-faces
        cy = my * (0.5 * (P + N) - np.where(my > 0, P, N))
        net = cx[1:] - cx[:-1] + cy[:, 1:] - cy[:, :-1]            # net outflow correction per cell
        ls.Sp[1:-1, 1:-1] -= g * net


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
                 tol=1e-8, max_inner=500, linear_solver="direct"):
        self.s = stencil
        self.X, self.Y, self.X_cor, self.Y_cor = X, Y, X_cor, Y_cor
        self.reference = reference
        self.tol, self.max_inner = tol, max_inner
        self.linear_solver = linear_solver
        self.grad = FVM_grad(X, Y, X_cor, Y_cor, stencil)
        self.m, self.n = X.shape
        self.mass_residual = np.inf

    def _face_flux_x(self, f, gx, gy, DpU):
        s = self.s
        un_bar = (x_face_avg(f.U) * s.Sx_x + x_face_avg(f.V) * s.Sx_y) / s.Sx_mag
        g_avg = (x_face_avg(gx) * s.dx_x + x_face_avg(gy) * s.dx_y) / s.dx_mag
        g_dir = (f.P[1:, 1:-1] - f.P[:-1, 1:-1]) / s.dx_mag
        dpf = x_face_avg(DpU)
        rho_f = x_face_avg(f.rho)
        un = un_bar + dpf * (s.dx_mag * s.Sx_mag) * (g_avg - g_dir)
        m = rho_f * un * s.Sx_mag
        a = rho_f * dpf * s.Sx_mag ** 2
        m[~s.x_interior] = 0.0
        a[~s.x_interior] = 0.0
        return m, a

    def _face_flux_y(self, f, gx, gy, DpV):
        s = self.s
        un_bar = (y_face_avg(f.U) * s.Sy_x + y_face_avg(f.V) * s.Sy_y) / s.Sy_mag
        g_avg = (y_face_avg(gx) * s.dy_x + y_face_avg(gy) * s.dy_y) / s.dy_mag
        g_dir = (f.P[1:-1, 1:] - f.P[1:-1, :-1]) / s.dy_mag   # y-difference (was an x-difference)
        dpf = y_face_avg(DpV)
        rho_f = y_face_avg(f.rho)
        un = un_bar + dpf * (s.dy_mag * s.Sy_mag) * (g_avg - g_dir)
        m = rho_f * un * s.Sy_mag
        a = rho_f * dpf * s.Sy_mag ** 2
        m[~s.y_interior] = 0.0
        a[~s.y_interior] = 0.0
        return m, a

    def solve(self, fields):
        f = fields
        DpU = np.zeros_like(f.U); DpU[1:-1, 1:-1] = 1.0 / (f.Ap_u[1:-1, 1:-1] + 1e-30)
        DpV = np.zeros_like(f.V); DpV[1:-1, 1:-1] = 1.0 / (f.Ap_v[1:-1, 1:-1] + 1e-30)
        # 1/a_P on the ghosts: copy the adjacent cell so wall-adjacent averages stay one-sided
        for D in (DpU, DpV):
            D[0, :], D[-1, :], D[:, 0], D[:, -1] = D[1, :], D[-2, :], D[:, 1], D[:, -2]
        f.DpU, f.DpV = DpU, DpV

        gx, gy = self.grad.compute_gradient(f.P)
        mx, ax = self._face_flux_x(f, gx, gy, DpU)
        my, ay = self._face_flux_y(f, gx, gy, DpV)

        shp = (self.m, self.n)
        Aw, Ae, As, An, Ap, Sp = (np.zeros(shp) for _ in range(6))
        I = (slice(1, -1), slice(1, -1))
        Ae[I], Aw[I] = ax[1:], ax[:-1]
        An[I], As[I] = ay[:, 1:], ay[:, :-1]
        Ap[I] = Ae[I] + Aw[I] + An[I] + As[I]
        Sp[I] = -(mx[1:] - mx[:-1] + my[:, 1:] - my[:, :-1])
        self.mass_residual = float(np.sum(np.abs(Sp[I])))

        p_prime = np.zeros(shp)
        if self.linear_solver == "direct":
            SparseDirect.solve(p_prime, Aw=Aw, Ap=Ap, Ae=Ae, As=As, An=An, Sp=Sp, pin=self.reference)
        else:
            ri, rj, rv = self.reference
            Ap[ri, rj], Sp[ri, rj] = 1.0, rv
            Aw[ri, rj] = Ae[ri, rj] = As[ri, rj] = An[ri, rj] = 0.0
            for _ in range(self.max_inner):
                TDMA.solve_line(p_prime, Aw=Aw, Ap=Ap, Ae=Ae, As=As, An=An, Sp=Sp)
                R = (Ap[I] * p_prime[I] - Aw[I] * p_prime[:-2, 1:-1] - Ae[I] * p_prime[2:, 1:-1]
                     - As[I] * p_prime[1:-1, :-2] - An[I] * p_prime[1:-1, 2:] - Sp[I])
                if np.max(np.abs(R)) < self.tol:
                    break

        # zero-gradient p' in the ghosts (the first version left them at 0, i.e. p' = 0 on walls)
        p_prime[0, :], p_prime[-1, :] = p_prime[1, :], p_prime[-2, :]
        p_prime[:, 0], p_prime[:, -1] = p_prime[:, 1], p_prime[:, -2]

        # corrected, continuity-satisfying face mass fluxes
        f.mflux_x = mx - ax * (p_prime[1:, 1:-1] - p_prime[:-1, 1:-1])
        f.mflux_y = my - ay * (p_prime[1:-1, 1:] - p_prime[1:-1, :-1])
        f.p_prime = p_prime
        return p_prime


class PressureVelocityCorrector:
    """SIMPLE correction step (same order as the first version):
    1. P <- P + alpha_p p'   2. u, v <- u, v - (V/a_P) grad p'   3. zero-gradient P ghosts
    4. gauge: subtract the pressure of one interior cell.
    """

    def __init__(self, stencil, X_cor, Y_cor, gauge=(2, 3)):
        self.s = stencil
        self.grad = FVM_grad(stencil.X, stencil.Y, X_cor, Y_cor, stencil)
        self.gauge = gauge

    def apply(self, fields, p_prime, alpha_p=0.3):
        f = fields
        I = (slice(1, -1), slice(1, -1))
        f.P[:, :] += alpha_p * p_prime
        gpx, gpy = self.grad.compute_gradient(p_prime)
        f.U[I] -= f.DpU[I] * self.s.vol * gpx[I]
        f.V[I] -= f.DpV[I] * self.s.vol * gpy[I]
        P = f.P
        P[:, 0], P[:, -1] = P[:, 1], P[:, -2]
        P[0, :], P[-1, :] = P[1, :], P[-2, :]
        P -= P[self.gauge]
