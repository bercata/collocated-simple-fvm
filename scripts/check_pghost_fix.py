"""Test p' ghost fix: n_nonorth=2 (fixed) from M128 start, 3000 its."""
import sys, os, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results")
from simplefvm.lowmach import LowMachSolver

START = os.path.join(OUT, "state_skew60_M128_start.npz")
s = LowMachSolver.from_state(START)

print(f"n_nonorth_correctors = {s.pcorr.n_nonorth_correctors}")
print("M128, 3000 its, n_nonorth=2 (fixed code)")
print(f"{'it':>6s} {'u_res':>10s} {'mass_res':>10s}")

t0 = time.perf_counter()
for it in range(1, 3001):
    r = s.step()
    if it in (1000, 2000, 3000):
        print(f"{it:6d} {r[0]:10.2e} {r[2]:10.2e}")
dt = time.perf_counter() - t0
print(f"\n  {dt:.1f}s  Nu_W={s.nu('W'):.6f}  Pbar/P0={s.Pbar/s.P0:.6f}")
print("  Expected (from n_nonorth=0 run): u~9e-8 mass~7e-8 at 3000")
nu_diff = abs(s.nu("W") - s.nu("E"))
print(f"  Nu_W-Nu_E = {nu_diff:.2e}")