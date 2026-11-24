"""Trap order study: convergence order of trap-induced errors.

Labels: baseline, wall_props_avg, no_wall_tangential, no_extra_stress.
Meshes: 32 -> 64 -> 128 with init_from, max_outer 80000, tol 1e-8.
Loads M32/64 from trap_study_m64.json; reruns M64 for init to M128.

Usage (USER runs):
    python scripts/trap_order_study.py          # full study 32->64->128
    python scripts/trap_order_study.py --smoke  # quick smoke M8->16 (one label)
"""
import sys, os, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
JSON_PATH = os.path.join(OUT, "trap_order_study.json")
TRAP_JSON = os.path.join(OUT, "trap_study_m64.json")

from simplefvm.lowmach import LowMachSolver
from simplefvm.properties import cavity_length, mu_sutherland, P0

# ------------------------------------------------------------------- project case
EPS = 0.6
RA = 1e5
BETA = 60
PROPERTIES = "sutherland"
N_NONORTH = 0   # 2 stalls at M128; converged answer identical
mu0 = mu_sutherland(600.0)
L = cavity_length(RA, EPS, mu0)

LABELS = ["baseline", "wall_props_avg", "no_wall_tangential", "no_extra_stress"]
TRAP_SETS = {  # label -> traps tuple
    "baseline": (),
    "wall_props_avg": ("wall_props_avg",),
    "no_wall_tangential": ("no_wall_tangential",),
    "no_extra_stress": ("no_extra_stress",),
}
RELAX = (0.4, 0.15, 0.6)
MAX_OUTER = 80000
TOL = 1e-8
MESHES = [32, 64, 128]
SMOKE_MESHES = [8, 16]

EST_HOURS = 5.5  # total estimate for full study


# ------------------------------------------------------------------- helpers
def make_solver(M, traps_set=()):
    return LowMachSolver(M=M, shape="parallelogram", beta_deg=BETA,
                         eps=EPS, Ra=RA, L=L, properties=PROPERTIES,
                         alpha_u=RELAX[0], alpha_p=RELAX[1], alpha_T=RELAX[2],
                         p_solver="direct", n_nonorth_correctors=N_NONORTH,
                         traps=traps_set)


def record(solver, label, dt):
    psi_min, _, _ = solver.psi_min()
    _, Nu_hot, y_hot = solver.nu_local("W")
    _, Nu_cold, y_cold = solver.nu_local("E")
    return {
        "label": label, "M": solver.M,
        "converged": solver.converged, "iterations": solver.iterations,
        "wall_time_s": round(dt, 1), "status": solver.status,
        "Nu_W": solver.nu("W"), "Pbar_P0": float(solver.Pbar / P0),
        "psi_min": psi_min,
        "Nu_max_hot": Nu_hot, "y_Nu_max_hot": y_hot,
        "Nu_max_cold": Nu_cold, "y_Nu_max_cold": y_cold,
    }


def run_mesh(M, traps_set, init_solver=None):
    if init_solver is not None:
        s = LowMachSolver.init_from(init_solver, M=M, traps=traps_set,
                                     alpha_u=RELAX[0], alpha_p=RELAX[1],
                                     alpha_T=RELAX[2])
    else:
        s = make_solver(M, traps_set=traps_set)
    t0 = time.perf_counter()
    s.run(max_outer=MAX_OUTER, tol=TOL, verbose=0)
    dt = time.perf_counter() - t0
    r = record(s, "?", dt)
    print(f"    [M{M}] its={s.iterations:6d}  {dt:8.1f}s  Nu_W={r['Nu_W']:.6f}"
          f"  Pbar/P0={r['Pbar_P0']:.6f}  {s.status}")
    return s, r


# ------------------------------------------------------------------- load existing M32/M64 from trap_study_m64.json
def load_existing():
    if not os.path.exists(TRAP_JSON):
        return {}
    return json.load(open(TRAP_JSON))


def get_existing_data(label, M, existing):
    entry = existing.get(label, {})
    mkey = f"M{M}"
    return entry.get(mkey)


# ------------------------------------------------------------------- order computation
def observed_order(f1, f2, f3, r=2):
    """p = log((f1-f2)/(f2-f3)) / log(r)"""
    d12 = f1 - f2
    d23 = f2 - f3
    if abs(d23) < 1e-30 or abs(d12) < 1e-30:
        return None
    return np.log(abs(d12 / d23)) / np.log(r)


def richardson_value(f3, f2, p, r=2):
    """f_h=0 = f3 + (f3 - f2) / (r^p - 1)"""
    if p is None:
        return None
    return f3 + (f3 - f2) / (r**p - 1)


# ------------------------------------------------------------------- main
def initialise_results():
    if os.path.exists(JSON_PATH):
        return json.load(open(JSON_PATH))
    return {}


def save_results(data):
    with open(JSON_PATH, "w") as fh:
        json.dump(data, fh, indent=1)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    existing = load_existing()

    all_results = initialise_results()

    if smoke:
        label = "wall_props_avg"
        traps_set = TRAP_SETS[label]
        M1, M2 = SMOKE_MESHES
        print(f"TRAP ORDER STUDY SMOKE TEST (M{M1} -> M{M2})")
        print(f"  Label: {label}  traps={traps_set}")
        print(f"  L = {L:.5f} m  eps={EPS}  Ra={RA:.0e}  beta={BETA}deg  n_nonorth={N_NONORTH}")
        print("  Expected time: ~1-2 min\n")

        entry = all_results.get(label, {})
        s1, r1 = run_mesh(M1, traps_set)
        entry[f"M{M1}"] = r1
        all_results[label] = entry; save_results(all_results)

        s2, r2 = run_mesh(M2, traps_set, init_solver=s1)
        entry[f"M{M2}"] = r2
        all_results[label] = entry; save_results(all_results)

        print(f"\n  Smoke results for {label}:")
        for k in ["converged", "iterations", "Nu_W", "Pbar_P0", "psi_min"]:
            print(f"    M{M1}: {k}={r1[k]}   M{M2}: {k}={r2[k]}")

    else:
        n_labels = len(LABELS)
        print("TRAP ORDER STUDY (M32 -> M64 -> M128 grid sequencing)")
        print(f"  L = {L:.5f} m  eps={EPS}  Ra={RA:.0e}  beta={BETA}deg  n_nonorth={N_NONORTH}")
        print(f"  Labels: {LABELS}")
        print(f"  Meshes: {MESHES}")
        print(f"  Running {n_labels} labels; M128 ~1.4 h each; total ~{EST_HOURS} h")
        print(f"  Reusing existing M32/64 from {os.path.basename(TRAP_JSON)}\n")

        for label in LABELS:
            traps_set = TRAP_SETS[label]
            print(f"\n[{label}] traps={traps_set}")

            entry = all_results.get(label, {})
            if not entry:
                # try to load M32 from trap_study_m64.json
                r32_existing = get_existing_data(label, 32, existing)
                if r32_existing and r32_existing.get("converged"):
                    entry["M32"] = r32_existing
                    print(f"    [M32] loaded from trap_study_m64.json  Nu_W={r32_existing.get('Nu_W'):.6f}")

            # M64: always re-run to get a solver for init_to_M128
            print("    Running M64 for init_from...")
            s32 = make_solver(32, traps_set=traps_set)
            s32.run(max_outer=MAX_OUTER, tol=TOL, verbose=0)
            if not s32.converged:
                print(f"    ERROR: M32 did not converge ({s32.status})")
                entry["M64"] = {"converged": False, "status": f"M32 fail: {s32.status}"}
                entry["M128"] = {"converged": False, "status": "skipped (no M32)"}
                all_results[label] = entry; save_results(all_results)
                continue
            s64, r64 = run_mesh(64, traps_set, init_solver=s32)
            entry["M64"] = r64
            all_results[label] = entry; save_results(all_results)

            if not s64.converged:
                entry["M128"] = {"converged": False, "status": f"M64 did not converge ({s64.status})"}
                all_results[label] = entry; save_results(all_results)
                continue

            # M128
            print("    Running M128 (~1.4 h)...")
            s128, r128 = run_mesh(128, traps_set, init_solver=s64)
            entry["M128"] = r128
            all_results[label] = entry; save_results(all_results)

        # ---- compute and print order study ----
        print("\n" + "=" * 80)
        print("ORDER STUDY RESULTS")
        print("=" * 80)

        outputs = ["Nu_W", "Pbar_P0", "psi_min", "Nu_max_hot", "Nu_max_cold"]
        for label in LABELS:
            entry = all_results.get(label, {})
            if not entry:
                continue
            r32 = entry.get("M32", {})
            r64 = entry.get("M64", {})
            r128 = entry.get("M128", {})
            if not (r32.get("converged") and r64.get("converged") and r128.get("converged")):
                print(f"\n  [{label}] incomplete, skipping order calc")
                continue
            print(f"\n  [{label}] observed order p and Richardson value:")
            for out in outputs:
                v32 = r32.get(out)
                v64 = r64.get(out)
                v128 = r128.get(out)
                if v32 is None or v64 is None or v128 is None:
                    continue
                p = observed_order(v32, v64, v128)
                rv = richardson_value(v128, v64, p) if p is not None else None
                p_str = f"{p:.3f}" if p is not None else "-"
                rv_str = f"{rv:.6e}" if rv is not None else "-"
                print(f"    {out:>16s}: p={p_str:>6s}  R_h0={rv_str}")

        # ---- trap error ratios ----
        print("\n" + "=" * 80)
        print("TRAP ERROR vs BASELINE: e(M) = trap(M) - baseline(M)")
        print("=" * 80)
        baseline = all_results.get("baseline", {})
        for label in LABELS:
            if label == "baseline":
                continue
            entry = all_results.get(label, {})
            print(f"\n  [{label}]:")
            e_vals = {}
            for M in MESHES:
                mkey = f"M{M}"
                br = baseline.get(mkey, {})
                tr = entry.get(mkey, {})
                if not (br.get("converged") and tr.get("converged")):
                    print(f"    M{M}: skipped (incomplete)")
                    continue
                e_nu = tr.get("Nu_W", 0) - br.get("Nu_W", 0)
                e_psi = tr.get("psi_min", 0) - br.get("psi_min", 0)
                e_vals[M] = (e_nu, e_psi)
                print(f"    M{M:3d}:  e(Nu_W)={e_nu:+.6e}  e(psi_min)={e_psi:+.6e}")
            # ratios
            for out_name, idx in [("Nu_W", 0), ("psi_min", 1)]:
                pairs = [(32, 64), (64, 128)]
                ratios = []
                for Mm, Mf in pairs:
                    if Mm in e_vals and Mf in e_vals:
                        e_m = e_vals[Mm][idx]
                        e_f = e_vals[Mf][idx]
                        if abs(e_f) > 1e-30:
                            r = abs(e_m / e_f)
                            ratios.append(f"e({Mm})/e({Mf})={r:.2f}")
                        else:
                            ratios.append(f"e({Mm})/e({Mf})=inf")
                if ratios:
                    print(f"      {out_name}: {', '.join(ratios)}  (2=1st-order, 4=2nd-order)")

    print(f"\nDone - {JSON_PATH}")