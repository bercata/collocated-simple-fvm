"""Regression tests for p' ghost fix (Task 3 + Task 4)."""
import sys, os, time, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simplefvm import SimpleSolver, NaturalConvectionSolver
from simplefvm.lowmach import LowMachSolver
from simplefvm.properties import cavity_length, mu_sutherland

# ========================================================== Test 1: Demirdzic lid 45 deg Re=100
print("=" * 60)
print("REGRESSION 1: Demirdzic lid-driven 45 deg Re=100 M=32")
print("=" * 60)

def run_demirdzic(n_nonorth):
    s = SimpleSolver(M=32, Re=100, shape="parallelogram", beta_deg=45,
                     gamma_dc=1.0, n_nonorth_correctors=n_nonorth)
    s.run(max_outer=50000, tol=1e-9, verbose=0)
    psi = s.stream_function()
    return float(psi.min()), s.converged

psi_2, c2 = run_demirdzic(2)
# n_nonorth=0 may diverge at Re=100 without correctors; try n_nonorth=1 instead
psi_1, c1 = run_demirdzic(1)
print(f"  n_nonorth=2: psi_min = {psi_2:+.10e}  converged={c2}")
print(f"  n_nonorth=1: psi_min = {psi_1:+.10e}  converged={c1}")
diff = abs(psi_2 - psi_1) / (abs(psi_1) + 1e-30)
print(f"  |diff| = {abs(psi_2-psi_1):.2e}, relative = {diff:.2e}")
print(f"  MATCH to 8 digits: {diff < 1e-7} (correctors do not change converged answer)")

# ========================================================== Test 2: Demirdzic lid 45 deg Re=1000 (check_dvd)
print("\n" + "=" * 60)
print("REGRESSION 2: Demirdzic lid-driven 45 deg Re=1000 M=32 n_nonorth=2")
print("=" * 60)

s = SimpleSolver(M=32, Re=1000, shape="parallelogram", beta_deg=45,
                 gamma_dc=1.0, n_nonorth_correctors=2)
s.run(max_outer=40000, tol=1e-9, verbose=0)
psi_min = float(s.stream_function().min())
target = -0.0490243555
print(f"  psi_min = {psi_min:+.10e}  (target {target:+.10e})")
print(f"  |diff| = {abs(psi_min-target):.2e}")

# ========================================================== Test 3: C3 Pr=10 M=32
print("\n" + "=" * 60)
print("REGRESSION 3: C3 Pr=10 M=32 Nu=8.18006")
print("=" * 60)

s = NaturalConvectionSolver(M=32, shape="parallelogram", beta_deg=45,
                             L=1.0, H=1.0, rho=1.0, mu=1e-3, k=1e-4,
                             cp=1.0, g_beta=0.1, T_h=1.0, T_c=0.0,
                             T_ref=0.5, alpha_u=0.6, alpha_p=0.3,
                             alpha_T=0.8, p_solver="direct",
                             n_nonorth_correctors=2, inner_sweeps=4)
s.run(max_outer=60000, tol=1e-8, verbose=2000)
Nu = abs(s.nu("E"))
print(f"  Nu = {Nu:.6f}  (target 8.18006 from run_c3.py M=32)")
print(f"  |diff| = {abs(Nu - 8.18006):.2e}")

# ========================================================== Test 4: Project case M32 n_nonorth 0 vs 2
print("\n" + "=" * 60)
print("REGRESSION 4: Project case M32 n_nonorth 0 vs 2 (its + time)")
print("=" * 60)

EPS = 0.6; RA = 1e5; BETA = 60
mu0 = mu_sutherland(600.0); L = cavity_length(RA, EPS, mu0)

for nn in (2, 0):
    s = LowMachSolver(M=32, shape="parallelogram", beta_deg=BETA,
                      eps=EPS, Ra=RA, L=L, properties="sutherland",
                      alpha_u=0.4, alpha_p=0.15, alpha_T=0.6,
                      p_solver="direct", n_nonorth_correctors=nn)
    t0 = time.perf_counter()
    s.run(max_outer=20000, tol=1e-8, verbose=0)
    dt = time.perf_counter() - t0
    Nu = s.nu("W"); Pb = s.Pbar/s.P0
    psi_min, _, _ = s.psi_min()
    print(f"  n_nonorth={nn}: its={s.iterations:5d}  {dt:6.1f}s  "
          f"Nu={Nu:.10f}  Pbar/P0={Pb:.10f}  psi_min={psi_min:.10e}  "
          f"Nu_bal={abs(Nu-s.nu('E')):.2e}")

print("\nDone.")