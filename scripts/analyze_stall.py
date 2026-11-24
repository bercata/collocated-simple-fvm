"""Diagnose M128 stall: hotspot scan + variant comparison.

Usage:
  python scripts/analyze_stall.py --scan         # tasks 2+3 (setup + hotspot scan)
  python scripts/analyze_stall.py --setup-only    # task 2 only
  python scripts/analyze_stall.py --scan-only     # task 3 only (from saved start)
  python scripts/analyze_stall.py --compare       # task 4 (user runs, ~2 h)
  python scripts/analyze_stall.py --check-mass    # cheap check 4: mass imbalance (200 its)
  python scripts/analyze_stall.py --check-corner  # cheap check 5: corner time series (400 its)
"""
import sys, os, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
from simplefvm.lowmach import LowMachSolver
from simplefvm.properties import cavity_length, mu_sutherland

EPS = 0.6; RA = 1e5; BETA = 60
mu0 = mu_sutherland(600.0); L = cavity_length(RA, EPS, mu0)
AU, AP, AT = 0.4, 0.15, 0.6

START_PATH = os.path.join(OUT, "state_skew60_M128_start.npz")

# ==================================================================== setup
def make_solver(M, au=AU, ap=AP, aT=AT, traps=()):
    return LowMachSolver(M=M, shape="parallelogram", beta_deg=BETA,
                         eps=EPS, Ra=RA, L=L, properties="sutherland",
                         alpha_u=au, alpha_p=ap, alpha_T=aT,
                         p_solver="direct", n_nonorth_correctors=2,
                         traps=traps)

def setup_start():
    import os
    if os.path.exists(START_PATH):
        print(f"[setup] M128 start already exists: {START_PATH}")
        return LowMachSolver.from_state(START_PATH)
    print("[setup] M32 cold start...")
    t0 = time.perf_counter()
    s32 = make_solver(32)
    s32.run(max_outer=20000, tol=1e-8, verbose=2000)
    print(f"  M32: its={s32.iterations} {time.perf_counter()-t0:.1f}s Nu={s32.nu('W'):.4f} {s32.status}")
    print("[setup] M64 from M32...")
    t0 = time.perf_counter()
    s64 = LowMachSolver.init_from(s32)
    s64.run(max_outer=40000, tol=1e-8, verbose=2000)
    print(f"  M64: its={s64.iterations} {time.perf_counter()-t0:.1f}s Nu={s64.nu('W'):.4f} {s64.status}")
    s64.save(os.path.join(OUT, "state_skew60_M64_tmp.npz"))
    print("[setup] M128 from M64...")
    t0 = time.perf_counter()
    s128 = LowMachSolver.init_from(s64)
    s128.save(START_PATH)
    print(f"  M128 start saved -> {START_PATH} ({time.perf_counter()-t0:.1f}s)")
    return s128

# ==================================================================== hotspot scan
def hotspot_scan(solver, n_its=3000):
    print(f"\n[scan] running {n_its} its from M128 start, logging every 10 its...")
    t0 = time.perf_counter()
    from collections import Counter
    dt_cells = Counter(); du_cells = Counter(); mass_cells = Counter()
    nu_gaps = []
    for it in range(1, n_its + 1):
        T_old = solver.T.copy(); U_old = solver.fields.U.copy()
        r = solver.step()
        dT = np.abs(solver.T - T_old); dU = np.abs(solver.fields.U - U_old)
        f = solver.fields; s = solver.stencil
        mfx_W = f.mflux_x[:-1, :]; mfx_E = f.mflux_x[1:, :]
        mfy_S = f.mflux_y[:, :-1]; mfy_N = f.mflux_y[:, 1:]
        div_m = mfx_E - mfx_W + mfy_N - mfy_S
        mass_per_cell = np.abs(div_m) / (f.rho[1:-1, 1:-1] * s.vol + 1e-30)
        if it % 10 == 0:
            dt_cells[tuple(x + 1 for x in np.unravel_index(np.argmax(dT[1:-1, 1:-1]), dT[1:-1, 1:-1].shape))] += 1
            du_cells[tuple(x + 1 for x in np.unravel_index(np.argmax(dU[1:-1, 1:-1]), dU[1:-1, 1:-1].shape))] += 1
            mass_cells[tuple(x + 1 for x in np.unravel_index(np.argmax(mass_per_cell), mass_per_cell.shape))] += 1
            nu_gaps.append((it, solver.nu("W") - solver.nu("E")))
        if it % 100 == 0:
            print(f"  it {it:5d}  u {r[0]:.2e}  v {r[1]:.2e}  mass {r[2]:.2e}  dT_max {float(np.max(dT[1:-1,1:-1])):.2e}")
    wall_t = time.perf_counter() - t0
    print(f"\n  Scan done: {n_its} its in {wall_t:.1f}s")
    M = solver.M
    corners = {
        (1, 1): "bottom-left   (acute  60 deg, hot+adiab)",
        (1, M): "top-left      (obtuse 120 deg, hot+adiab)",
        (M, 1): "bottom-right  (obtuse 120 deg, cold+adiab)",
        (M, M): "top-right     (acute  60 deg, cold+adiab)",
    }
    def desc(i, j):
        c = corners.get((i, j), "")
        parts = [p for p in [
            "west (hot)  wall cell" if i == 1 else None,
            "east (cold) wall cell" if i == M else None,
            "south (adiab) wall cell" if j == 1 else None,
            "north (adiab) wall cell" if j == M else None,
            "cell next to hot wall" if i == 2 else None,
            "cell next to cold wall" if i == M - 1 else None,
            "cell next to south wall" if j == 2 else None,
            "cell next to north wall" if j == M - 1 else None,
        ] if p is not None]
        tag = c if c else (", ".join(parts) if parts else f"interior (i={i}, j={j})")
        return tag
    for name, counter in [("max |dT|", dt_cells), ("max |dU|", du_cells), ("max |mass imb|", mass_cells)]:
        print(f"\n  {name} -- top 5 most frequent cells:")
        for (i, j), count in counter.most_common(5):
            print(f"    ({i:3d},{j:3d}) {count:4d} ({100.0*count/(n_its//10):5.1f}%)  {desc(i, j)}")
    gaps = np.array([g for _, g in nu_gaps])
    signs = np.sign(gaps - np.median(gaps))
    sc = np.sum(np.abs(np.diff(signs))) // 2
    period = (2 * n_its / max(sc, 1)) if sc > 0 else float('inf')
    print(f"\n  Nu_W-Nu_E: min={np.min(gaps):+.2e} max={np.max(gaps):+.2e} mean={np.mean(gaps):+.2e}")
    print(f"  Sign changes: {sc}, est period ~{period:.0f} its")

# ==================================================================== task 4: compare variants
def _patch_density(solver, alpha_rho):
    """Monkey-patch _update_density to use a different alpha_rho than the hardcoded 0.8."""
    def patched():
        rho_new = solver.Pbar / (solver.R_gas * solver.T + 1e-30)
        if getattr(solver, '_rho_old', None) is None:
            solver._rho_old = rho_new.copy()
        rho_r = solver._rho_old + alpha_rho * (rho_new - solver._rho_old)
        solver._rho_old = rho_r
        solver.rho_cell[:] = rho_r
        solver.fields.rho[:] = solver.rho_cell
    solver._update_density = patched
    # also clobber _rho_old so the first call picks up the current rho properly
    solver._rho_old = solver.rho_cell.copy()

def compare_variants(start_path):
    print("\n" + "=" * 60)
    print("TASK 4: VARIANT COMPARISON (3000 its each from M128 start)")
    print("=" * 60)

    variants = [
        ("(a) baseline 0.4/0.15/0.6",       AU,  AP,  AT,  None, None,       None,           "a_baseline"),
        ("(g) no_majumdar",                  AU,  AP,  AT,  ("no_majumdar",),  None,       None,           "g_nomajumdar"),
        ("(h) n_nonorth_correctors 0",       AU,  AP,  AT,  None, 0,           None,           "h_nonorth0"),
        ("(i) energy conv upwind (dc=0)",    AU,  AP,  AT,  None, None,       "e_dc0",        "i_edc0"),
        ("(j) momentum conv upwind (dc=0)",  AU,  AP,  AT,  None, None,       "m_dc0",        "j_mdc0"),
        ("(k) density relax 0.5",            AU,  AP,  AT,  None, None,       "rho_05",       "k_rho05"),
        ("(k2) density relax 1.0",           AU,  AP,  AT,  None, None,       "rho_10",       "k2_rho10"),
        ("(l) Pbar frozen",                  AU,  AP,  AT,  ("pbar_fixed",),   None,       None,           "l_pbar_fixed"),
        ("(m) no_extra_stress",              AU,  AP,  AT,  ("no_extra_stress",), None,   None,           "m_noextrastress"),
    ]

    EST_MIN_PER = 3.0
    print(f"  {len(variants)} variants, ~{len(variants)*EST_MIN_PER:.0f} min each, total ~{len(variants)*EST_MIN_PER/60:.1f} h\n")

    for label, au, ap, aT, traps, n_nonorth_override, special, tag in variants:
        print(f"\n[{label}]")
        solver = LowMachSolver.from_state(start_path)

        # --- relaxation override (must reach all sub-solvers) ---
        solver.alpha_u = au; solver.alpha_p = ap; solver.alpha_T = aT
        solver.energy.alpha_T = aT  # EnergySolver keeps its own copy (energy.py:160/318)

        # --- traps ---
        if traps:
            for t in traps:
                solver.traps.add(t)
                if t == "no_wall_tangential":
                    solver.energy.wall_tangential = False
                elif t == "no_majumdar":
                    solver.pcorr.consistent_rc = False             # lowmach __init__:77
                elif t == "unbalanced_force":
                    solver.pcorr.body_force_rc = False             # lowmach __init__:79
                elif t == "no_extra_stress":
                    pass  # checked in step()
                elif t == "pbar_fixed":
                    pass  # checked in step()

        # --- n_nonorth_correctors override (h) ---
        if n_nonorth_override is not None:
            solver.pcorr.n_nonorth_correctors = n_nonorth_override

        # --- gamma_dc overrides (i/j) ---
        if special == "e_dc0":
            solver.energy.gamma_dc = 0.0
        elif special == "m_dc0":
            solver.momentum.gamma_dc = 0.0

        # --- density relaxation override (k/k2) ---
        if special == "rho_05":
            _patch_density(solver, 0.5)
        elif special == "rho_10":
            _patch_density(solver, 1.0)

        # run
        t0 = time.perf_counter()
        records = []
        # seed _prev_T for correct dT at it==1000
        solver._prev_T = solver.T.copy()
        for it in range(1, 3001):
            r = solver.step()
            if it in (1000, 2000, 3000):
                dT_val = float(np.max(np.abs(solver.T[1:-1, 1:-1] - solver._prev_T[1:-1, 1:-1])))
                records.append((it, r[0], r[2], dT_val, solver.nu("W") - solver.nu("E")))
                # sample last 100 its slope: save T at it-100 (done below)
            if it == 900:
                solver._T_900 = solver.T.copy()
            if it == 1900:
                solver._T_1900 = solver.T.copy()
            if it == 2900:
                solver._T_2900 = solver.T.copy()
        wall_t = time.perf_counter() - t0

        # slope of dT over last 1000 its of each segment
        slopes = []
        for snap_key in ['_T_900', '_T_1900', '_T_2900']:
            T_snap = getattr(solver, snap_key, None)
            if T_snap is not None:
                slopes.append(float(np.max(np.abs(solver.T[1:-1, 1:-1] - T_snap[1:-1, 1:-1]))) / 100.0)
            else:
                slopes.append(float('nan'))

        print(f"      {'it':>5s} {'u_res':>10s} {'mass_res':>10s} {'dT':>10s} {'Nu_gap':>10s} {'dT_slope':>10s}")
        for idx, (itn, ures, mres, dt_val, nugap) in enumerate(records):
            sl_str = f"{slopes[idx]:.2e}" if not np.isnan(slopes[idx]) else "-"
            print(f"      {itn:5d} {ures:10.2e} {mres:10.2e} {dt_val:10.2e} {nugap:10.2e} {sl_str:>10s}")
        print(f"      time: {wall_t:.1f}s")

        ur = [r[1] for r in records]; mr = [r[2] for r in records]; dts = [r[3] for r in records]
        def trend(x): return "falls" if x[-1] < 0.9*x[0] else ("flat" if x[-1] < 1.1*x[0] else "grows")
        print(f"      trend: u->{trend(ur)}, mass->{trend(mr)}, dT->{trend(dts)}")

# ==================================================================== cheap checks

def check_mass(start_path):
    """Cheap check 4: signed mass imbalance sum + reference-cell value, 200 its."""
    print("CHEAP CHECK 4: signed mass imbalance per iteration (200 its)")
    solver = LowMachSolver.from_state(start_path)
    M = solver.M
    cref = (M // 2, M // 2)  # reference cell (65, 65 for M=128)
    print(f"  Domain M={M}, reference cell=({cref[0]+1},{cref[1]+1})")
    print(f"  {'it':>5s} {'sum_div_m':>12s} {'ref_div_m':>12s}")
    for it in range(1, 201):
        solver.step()
        f = solver.fields
        div_m = f.mflux_x[1:, :] - f.mflux_x[:-1, :] + f.mflux_y[:, 1:] - f.mflux_y[:, :-1]
        sum_div = float(np.sum(div_m))
        ref_div = float(div_m[cref[0], cref[1]])
        if it <= 10 or it % 50 == 0:
            print(f"  {it:5d} {sum_div:12.3e} {ref_div:12.3e}")
    print("  Expected: sum_div_m ~ 0 (round-off, telescoping sum of interior fluxes)")


def check_corner(start_path):
    """Cheap check 5: corner cell (127,128) T/U/V/P + Pbar time series, 400 its."""
    print("CHEAP CHECK 5: corner cell (127,128) T/U/V/P + Pbar time series, 400 its")
    solver = LowMachSolver.from_state(start_path)
    M = solver.M
    # (127, 128) in interior indices = array indices [127, 128] (ghosts at 0 and 129)
    ai, aj = M - 1, M
    print(f"  Corner cell array=({ai},{aj})")
    data = []
    for it in range(1, 401):
        solver.step()
        Ti = float(solver.T[ai, aj])
        Ui = float(solver.fields.U[ai, aj])
        Vi = float(solver.fields.V[ai, aj])
        Pi = float(solver.fields.P[ai, aj])
        Pbar = float(solver.Pbar)
        data.append((it, Ti, Ui, Vi, Pi, Pbar))
        if it <= 5 or it % 100 == 0:
            print(f"  it={it:4d}  T={Ti:.4f}  U={Ui:+.4e}  V={Vi:+.4e}  P={Pi:+.4e}  Pbar={Pbar:.6f}")

    arr = np.array(data)
    def amp(col):
        a = arr[:, col]
        return float(np.max(a) - np.min(a))
    print(f"\n  Amplitude (max-min): T={amp(1):.4e} K  U={amp(2):.4e}  V={amp(3):.4e}  P={amp(4):.4e}  Pbar={amp(5):.4e}")

    # Pbar period
    pbar = arr[:, 5]; pm = np.median(pbar)
    sc = np.sum(np.abs(np.diff(np.sign(pbar - pm)))) // 2
    period = (2 * 400 / max(sc, 1)) if sc > 0 else float('inf')
    print(f"  Pbar sign changes: {sc}, est period ~{period:.0f} its")
    print(f"  Pbar range: [{np.min(pbar):.6f}, {np.max(pbar):.6f}]")

# ==================================================================== main
if __name__ == "__main__":
    if "--check-mass" in sys.argv:
        if not os.path.exists(START_PATH):
            print(f"ERROR: M128 start not found at {START_PATH}")
            sys.exit(1)
        check_mass(START_PATH)
    elif "--check-corner" in sys.argv:
        if not os.path.exists(START_PATH):
            print(f"ERROR: M128 start not found at {START_PATH}")
            sys.exit(1)
        check_corner(START_PATH)
    elif "--setup-only" in sys.argv:
        print("ANALYZE STALL -- Task 2 only: setup M128 start state")
        setup_start()
    elif "--scan-only" in sys.argv:
        if not os.path.exists(START_PATH):
            print(f"ERROR: M128 start not found at {START_PATH}")
            sys.exit(1)
        print("ANALYZE STALL -- Task 3 only: hotspot scan")
        solver = LowMachSolver.from_state(START_PATH)
        hotspot_scan(solver, n_its=3000)
    elif "--scan" in sys.argv:
        print("ANALYZE STALL -- Tasks 2+3: setup + hotspot scan")
        solver = setup_start()
        hotspot_scan(solver, n_its=3000)
    elif "--compare" in sys.argv:
        if not os.path.exists(START_PATH):
            print(f"ERROR: M128 start not found at {START_PATH}")
            sys.exit(1)
        compare_variants(START_PATH)
    else:
        print("Usage:")
        print("  python scripts/analyze_stall.py --check-mass     # cheap check 4 (200 its)")
        print("  python scripts/analyze_stall.py --check-corner   # cheap check 5 (400 its)")
        print("  python scripts/analyze_stall.py --setup-only     # task 2 only")
        print("  python scripts/analyze_stall.py --scan-only      # task 3 only")
        print("  python scripts/analyze_stall.py --scan           # tasks 2+3")
        print("  python scripts/analyze_stall.py --compare        # task 4 (user runs, ~2 h)")