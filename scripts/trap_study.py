"""Trap-size study: run baseline and each trap M32 -> M64, record all outputs.

Usage (USER runs this):
    python scripts/trap_study.py                        # full study M32 -> M64
    python scripts/trap_study.py --smoke                # M16 quick smoke test of one trap
    python scripts/trap_study.py --rerun LABEL          # redo one label at M32 -> M64
"""
import sys, os, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
JSON_PATH = os.path.join(OUT, "trap_study_m64.json")
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
M_SMOKE = 16

MAX_OUTER = {32: 20000, 64: 40000}
EST_MIN_PER_RUN = 8.0   # rough estimate


# ------------------------------------------------------------------- trap definitions
# Each entry: (label, traps_set, alpha_u, alpha_p, alpha_T)
# label is the run key; None -> post-processing only
TRAPS = [
    ("baseline",     (),                               0.4, 0.15, 0.6),
    ("pbar_fixed",   ("pbar_fixed",),                  0.4, 0.15, 0.6),
    ("boussinesq_rho", ("boussinesq_rho",),            0.4, 0.15, 0.6),
    ("wall_props_avg", ("wall_props_avg",),            0.4, 0.15, 0.6),
    ("no_majumdar_a",  ("no_majumdar",),               0.4, 0.15, 0.6),
    ("no_majumdar_b",  ("no_majumdar",),               0.3, 0.10, 0.5),
    ("no_extra_stress", ("no_extra_stress",),          0.4, 0.15, 0.6),
    ("unbalanced_force", ("unbalanced_force",),        0.4, 0.15, 0.6),
    ("no_wall_tangential", ("no_wall_tangential",),    0.4, 0.15, 0.6),
    # post-processing only:
    ("psi_volume",    (),                               0.4, 0.15, 0.6),
]

POST_ONLY = {"psi_volume"}


# ------------------------------------------------------------------- helpers
def make_solver(M, traps_set, au, ap, aT):
    return LowMachSolver(M=M, shape="parallelogram", beta_deg=BETA,
                         eps=EPS, Ra=RA, L=L, properties=PROPERTIES,
                         alpha_u=au, alpha_p=ap, alpha_T=aT,
                         p_solver="direct", n_nonorth_correctors=N_NONORTH,
                         traps=traps_set)


def record_run(s, label, dt):
    r = {}
    r["label"] = label
    r["traps"] = list(s.traps)
    r["M"] = s.M
    r["converged"] = s.converged
    r["iterations"] = s.iterations
    r["wall_time_s"] = round(dt, 1)
    r["status"] = s.status
    r["Nu_W"] = s.nu("W")
    r["Nu_E"] = s.nu("E")
    r["Nu_reldiff"] = abs(r["Nu_W"] - r["Nu_E"]) / (abs(r["Nu_W"]) + 1e-30)
    r["Pbar_P0"] = float(s.Pbar / P0)
    r["mass_imbalance"] = float(np.sum(s.fields.rho[1:-1, 1:-1] * s.stencil.vol) / s.m0 - 1.0)
    psi_min, psi_x, psi_y = s.psi_min()
    r["psi_min"] = psi_min
    _, Nu_hot, y_hot = s.nu_local("W")
    _, Nu_cold, y_cold = s.nu_local("E")
    r["Nu_max_hot"] = Nu_hot
    r["y_Nu_max_hot"] = y_hot
    r["Nu_max_cold"] = Nu_cold
    r["y_Nu_max_cold"] = y_cold
    return r


def run_single(label, traps_set, au, ap, aT, smoke=False, skip=False):
    """Run M32 cold start and M64 from M32 (or M16 for smoke).  Returns records."""
    recs = {}

    if smoke:
        rec = _run_mesh(label, traps_set, au, ap, aT, M_SMOKE, skip=skip)
        recs[f"M{M_SMOKE}"] = rec
        return recs

    # M32 cold start
    rec32 = _run_mesh(label, traps_set, au, ap, aT, 32, skip=skip)
    recs["M32"] = rec32
    if not rec32.get("converged"):
        recs["M64"] = {"converged": False, "status": f"M32 did not converge ({rec32.get('status')})"}
        return recs

    # M64 from M32
    rec64 = _run_mesh(label, traps_set, au, ap, aT, 64, init_from=True, skip=skip)
    recs["M64"] = rec64
    return recs


def _run_mesh(label, traps_set, au, ap, aT, M, init_from=False, skip=False):
    key = f"M{M}"
    if skip:
        print(f"    [{key}] skipped (already converged)")
        return {"converged": True, "status": "pre-converged (skipped)", "iterations": 0, "wall_time_s": 0.0}

    if init_from:
        s32 = make_solver(32, traps_set, au, ap, aT)
        # We need to load the stored M32 results and restore the solver.
        # Since we already ran M32 in the same process, we can reuse the solver.
        # When skip=True, we need cold-start M32 then init_from.
        s32.run(max_outer=MAX_OUTER[32], tol=1e-8, verbose=0)
        if not s32.converged:
            return {"converged": False, "status": f"M32 did not converge ({s32.status})"}
        s = LowMachSolver.init_from(s32, M=M, traps=traps_set,
                                     alpha_u=au, alpha_p=ap, alpha_T=aT)
    else:
        s = make_solver(M, traps_set, au, ap, aT)

    t0 = time.perf_counter()
    s.run(max_outer=MAX_OUTER.get(M, 40000), tol=1e-8, verbose=0)
    dt = time.perf_counter() - t0
    rec = record_run(s, label, dt)
    print(f"    [{key}] its={s.iterations:6d}  {dt:8.1f}s  Nu_W={rec['Nu_W']:.6f}"
          f"  Pbar/P0={rec['Pbar_P0']:.6f}  {s.status}")
    return rec


def initialise_results():
    if os.path.exists(JSON_PATH):
        return json.load(open(JSON_PATH))
    return {}


def save_results(results):
    with open(JSON_PATH, "w") as fh:
        json.dump(results, fh, indent=1)


def compute_line(results):
    """Extract values for one run at M64 from the results dict."""
    rec = results.get("M64", {})
    return {
        "converged": rec.get("converged"),
        "its": rec.get("iterations", 0),
        "time_s": rec.get("wall_time_s", 0.0),
        "Nu_W": rec.get("Nu_W"),
        "Nu_E": rel_diff(rec.get("Nu_W"), rec.get("Nu_E")),
        "Pbar_P0": rec.get("Pbar_P0"),
        "mass_imb": rec.get("mass_imbalance"),
        "psi_min": rec.get("psi_min"),
        "Nu_max_hot": rec.get("Nu_max_hot"),
        "y_hot": rec.get("y_Nu_max_hot"),
        "Nu_max_cold": rec.get("Nu_max_cold"),
        "y_cold": rec.get("y_Nu_max_cold"),
        "status": rec.get("status", ""),
    }


def rel_diff(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / (abs(a) + 1e-30)


# ------------------------------------------------------------------- psi_volume
def compute_psi_volume_post(s):
    """Post-processing: psi_min from volume flux, returned as a dict compatible with record_run."""
    psi_min_v, psi_x_v, psi_y_v = s.psi_volume()
    return {
        "converged": True,
        "iterations": s.iterations,
        "wall_time_s": s.wall_time,
        "status": "post-processed",
        "Nu_W": s.nu("W"),
        "Nu_E": s.nu("E"),
        "Nu_reldiff": abs(s.nu("W") - s.nu("E")) / (abs(s.nu("W")) + 1e-30),
        "Pbar_P0": float(s.Pbar / P0),
        "mass_imbalance": s.total_mass(),
        "psi_min": psi_min_v,
        "Nu_max_hot": None, "y_Nu_max_hot": None,
        "Nu_max_cold": None, "y_Nu_max_cold": None,
        "label": "psi_volume",
    }


# ------------------------------------------------------------------- table
def print_table(all_results, smoke=False):
    """Print a results table for the study."""
    M_str = f"M{M_SMOKE}" if smoke else "M64"
    headers = ["label", "conv?", "its", "Nu_W", "|dNu|/Nu_W", "Pbar/P0",
               "mass_imb", "psi_min", "Nu_max_hot", "y_hot", "Nu_max_cold", "y_cold", "status"]
    rows = []
    baseline = None

    for label, traps_set, au, ap, aT in TRAPS:
        if label not in all_results:
            continue
        entry = all_results[label]
        if M_str not in entry:
            continue
        rec = entry[M_str]
        row = [label, "Y" if rec.get("converged") else "N",
               rec.get("iterations", "?"),
               f"{rec.get('Nu_W'):.6f}" if rec.get("Nu_W") is not None else "-",
               f"{rec.get('Nu_reldiff'):.2e}" if rec.get("Nu_reldiff") is not None else "-",
               f"{rec.get('Pbar_P0'):.6f}" if rec.get("Pbar_P0") is not None else "-",
               f"{rec.get('mass_imbalance'):.2e}" if rec.get("mass_imbalance") is not None else "-",
               f"{rec.get('psi_min'):.6e}" if rec.get('psi_min') is not None else "-",
               f"{rec.get('Nu_max_hot'):.6f}" if rec.get('Nu_max_hot') is not None else "-",
               f"{rec.get('y_Nu_max_hot'):.6f}" if rec.get('y_Nu_max_hot') is not None else "-",
               f"{rec.get('Nu_max_cold'):.6f}" if rec.get('Nu_max_cold') is not None else "-",
               f"{rec.get('y_Nu_max_cold'):.6f}" if rec.get('y_Nu_max_cold') is not None else "-",
               rec.get("status", "-")]
        if label == "baseline":
            baseline = rec
        rows.append(row)

    for h in headers:
        print(f"{h:>20s}", end="")
    print()
    for row in rows:
        for i, v in enumerate(row):
            print(f"{str(v):>20s}", end="")
        print()

    if baseline and len(rows) > 1:
        print("\nRelative change vs baseline (%):")
        print(f"{'label':>25s} {'Nu_W':>10s} {'Pbar/P0':>10s} {'psi_min':>10s}"
              f" {'Nu_hot':>10s} {'Nu_cold':>10s}")
        bw = baseline.get("Nu_W")
        bp = baseline.get("Pbar_P0")
        bpsi = baseline.get("psi_min")
        bnh = baseline.get("Nu_max_hot")
        bnc = baseline.get("Nu_max_cold")
        for row in rows:
            label = row[0]
            if label == "baseline":
                continue
            rec = all_results.get(label, {}).get(M_str, {})
            vals = [
                f"{100*(rec.get('Nu_W')-bw)/bw:.2f}" if bw and rec.get("Nu_W") is not None else "-",
                f"{100*(rec.get('Pbar_P0')-bp)/bp:.2f}" if bp and rec.get("Pbar_P0") is not None else "-",
                f"{100*(rec.get('psi_min')-bpsi)/(abs(bpsi)+1e-30):.2f}" if bpsi and rec.get("psi_min") is not None else "-",
                f"{100*(rec.get('Nu_max_hot')-bnh)/(abs(bnh)+1e-30):.2f}" if bnh and rec.get('Nu_max_hot') is not None else "-",
                f"{100*(rec.get('Nu_max_cold')-bnc)/(abs(bnc)+1e-30):.2f}" if bnc and rec.get('Nu_max_cold') is not None else "-",
            ]
            print(f"{label:>25s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s} {vals[3]:>10s} {vals[4]:>10s}")


# ===================================================================== main
if __name__ == "__main__":
    smoke = "--smoke" in sys.argv

    # --rerun LABEL  -> force redo just that label
    rerun_label = None
    for i, a in enumerate(sys.argv):
        if a == "--rerun":
            if i + 1 < len(sys.argv):
                rerun_label = sys.argv[i + 1]
            break

    all_results = initialise_results()

    # helper: filter TRAPS to only the requested label when --rerun is given
    def _active_traps():
        for t in TRAPS:
            if rerun_label is not None:
                if t[0] != rerun_label:
                    continue
            yield t

    if rerun_label is not None:
        # remove existing entries so the label is re-run from scratch
        if rerun_label in all_results:
            del all_results[rerun_label]
            save_results(all_results)
            print(f"[--rerun] removed existing entry for '{rerun_label}'")
        # check label is valid
        known = {t[0] for t in TRAPS}
        if rerun_label not in known:
            print(f"ERROR: unknown label '{rerun_label}'. Known: {sorted(known)}")
            sys.exit(1)

    if smoke:
        M_RUN = M_SMOKE
        print(f"TRAP STUDY SMOKE TEST (M={M_RUN})")
        print(f"  L = {L:.5f} m  eps={EPS}  Ra={RA:.0e}  beta={BETA}deg  n_nonorth={N_NONORTH}")
        print(f"  Running baseline + one trap (no_extra_stress) at M{M_RUN} cold start")
        print("  Expected time: ~0.5 min per run\n")

        for label, traps_set, au, ap, aT in _active_traps():
            if label not in ("baseline", "no_extra_stress"):
                continue
            if not rerun_label and label in all_results and f"M{M_RUN}" in all_results[label] and all_results[label][f"M{M_RUN}"].get("converged"):
                print(f"\n[{label}] already converged at M{M_RUN}, skipping")
                continue
            print(f"\n[{label}] M={M_RUN} traps={traps_set} relax=({au},{ap},{aT})")
            recs = run_single(label, traps_set, au, ap, aT, smoke=True)
            all_results[label] = recs
            save_results(all_results)

        print_table(all_results, smoke=True)

    else:
        eta = len([t for t in _active_traps() if t[0] not in POST_ONLY])
        pp = sum(1 for t in _active_traps() if t[0] in POST_ONLY)
        total_min = eta * EST_MIN_PER_RUN
        if rerun_label:
            print(f"TRAP STUDY --rerun {rerun_label} (M32 -> M64)")
        else:
            print("TRAP STUDY (M32 -> M64 grid sequencing)")
        print(f"  L = {L:.5f} m  eps={EPS}  Ra={RA:.0e}  beta={BETA}deg  n_nonorth={N_NONORTH}")
        print(f"  {eta} solver runs + {pp} post-processing")
        print(f"  Expected total time: ~{total_min:.0f} min (M32: ~1-2 min + M64: ~6-8 min each)\n")

        baseline_rec = None
        for label, traps_set, au, ap, aT in _active_traps():
            print(f"\n[{label}] traps={traps_set} relax=({au},{ap},{aT})")

            if label in POST_ONLY:
                # post-processing: use baseline solver
                if baseline_rec is None:
                    print("    Running baseline first for psi_volume post-processing...")
                    baseline_recs = run_single("baseline", (), 0.4, 0.15, 0.6)
                    all_results["baseline"] = baseline_recs
                    save_results(all_results)
                    if "M64" in baseline_recs and baseline_recs["M64"].get("converged"):
                        baseline_rec = baseline_recs["M64"]
                if baseline_rec and baseline_rec.get("converged"):
                    # Reconstruct solver from baseline results to compute psi_volume
                    print("    Post-processing psi_volume from baseline...")
                    _s = make_solver(64, (), 0.4, 0.15, 0.6)
                    _s32 = make_solver(32, (), 0.4, 0.15, 0.6)
                    _s32.run(max_outer=MAX_OUTER[32], tol=1e-8, verbose=0)
                    _s = LowMachSolver.init_from(_s32, M=64, traps=())
                    _s.run(max_outer=MAX_OUTER[64], tol=1e-8, verbose=0)
                    pp_rec = compute_psi_volume_post(_s)
                    all_results["psi_volume"] = {"M64": pp_rec}
                    save_results(all_results)
                    print(f"    psi_min(volume) = {pp_rec['psi_min']:.6e}")
                    print(f"    psi_min(mass)   = {baseline_rec['psi_min']:.6e}")
            else:
                skip = False
                if not rerun_label and label in all_results and "M64" in all_results[label] and all_results[label]["M64"].get("converged"):
                    skip = True
                    print("    M64 already converged, skipping")

                recs = run_single(label, traps_set, au, ap, aT, smoke=False, skip=skip)
                all_results[label] = recs
                save_results(all_results)
                if label == "baseline":
                    baseline_rec = recs.get("M64", {})

        print_table(all_results, smoke=False)

    print(f"\nDone - {JSON_PATH}")