"""Low-Mach natural convection in a skewed (parallelogram) cavity, with grid sequencing.

Default (project) case: rhombus beta=60, eps=0.6, Ra=1e5 (Sutherland).
For other cases pass --beta, --eps, --meshes etc.

Each mesh is started from the previous (converged) mesh with LowMachSolver.init_from,
so the iteration count (and wall time) is much lower than a cold start.  The stable
relaxation is 0.4/0.15/0.6; if a finer mesh stalls, drop to 0.3/0.1/0.5.

Writes <out_dir>/lowmach_<tag>.json incrementally after each run.
    python scripts/run_lowmach_skew.py
"""
import argparse, sys, os, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def resolve_out_dir(arg, root=ROOT):
    d = arg if os.path.isabs(arg) else os.path.join(root, arg)
    os.makedirs(d, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser(description="Low-Mach skewed-cavity benchmark (grid sequencing)")
    ap.add_argument("--beta", type=float, default=60.0, help="parallelogram angle in degrees")
    ap.add_argument("--eps", type=float, default=0.6, help="non-dimensional temperature amplitude")
    ap.add_argument("--ra", type=float, default=1e5, help="Rayleigh number")
    ap.add_argument("--meshes", type=int, nargs="+", default=[32, 64, 128, 256],
                    help="mesh sizes to run (default: 32 64 128 256)")
    ap.add_argument("--out-dir", default="results/",
                    help="output directory (default: results/)")
    ap.add_argument("--tag", default="skew60",
                    help="case tag for file naming (default: skew60)")
    args = ap.parse_args()

    BETA = args.beta
    EPS = args.eps
    RA = args.ra
    MESHES = tuple(args.meshes)
    TAG = args.tag

    OUT = resolve_out_dir(args.out_dir)
    JSON_PATH = os.path.join(OUT, f"lowmach_{TAG}.json")
    STATE_DIR = os.path.join(OUT, f"state_{TAG}")
    os.makedirs(STATE_DIR, exist_ok=True)

    from simplefvm.lowmach import LowMachSolver
    from simplefvm.properties import cavity_length, mu_sutherland, P0

    ALPHA_U, ALPHA_P, ALPHA_T = 0.4, 0.15, 0.6
    MAX_OUTER = {32: 20000, 64: 40000, 128: 80000, 256: 240000}
    EST_TIME = {32: "~1 min", 64: "~7.5 min", 128: "~2.2 h", 256: "~7.7 h"}

    mu0 = mu_sutherland(600.0)
    L = cavity_length(RA, EPS, mu0)

    if os.path.exists(JSON_PATH):
        results = json.load(open(JSON_PATH))
    else:
        results = {}

    print(f"run_lowmach_skew.py - beta {BETA} rhombus, Sutherland, eps {EPS}, Ra {RA:.0e} (grid sequencing)")
    print(f"  L = {L:.5f} m   relaxation = {ALPHA_U}/{ALPHA_P}/{ALPHA_T}")
    print("  expected time per mesh (started from the previous mesh):")
    for M in MESHES:
        est = EST_TIME.get(M, "?")
        print(f"    M{M:3d}: {est}")

    def richardson(f1, f2, f3, h_ratio=2.0):
        if abs(f2 - f3) < 1e-30 or abs(f1 - f2) < 1e-30:
            return 0.0, f1
        p = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(h_ratio)
        rich = f1 + (f1 - f2) / (h_ratio ** p - 1.0)
        return float(p), float(rich)

    current = None
    for M in MESHES:
        key = f"M{M}"
        max_outer = MAX_OUTER.get(M, int(20000 * (M / 32) ** 2.0))
        if current is None:
            s = LowMachSolver(M=M, shape="parallelogram", beta_deg=BETA, eps=EPS, Ra=RA, L=L,
                              properties="sutherland", alpha_u=ALPHA_U, alpha_p=ALPHA_P,
                              alpha_T=ALPHA_T, p_solver="direct", n_nonorth_correctors=0)
        else:
            s = LowMachSolver.init_from(current)

        already = key in results and results[key].get("converged")
        t0 = time.perf_counter()
        s.run(max_outer=max_outer, tol=1e-8, verbose=2000)
        dt = round(time.perf_counter() - t0, 1)
        current = s
        STATE_PATH = os.path.join(STATE_DIR, f"state_{TAG}_M{M}.npz")
        s.save(STATE_PATH)
        print(f"    state saved -> {STATE_PATH}")

        if already:
            print(f"  {key} its={s.iterations:6d} {dt:8.1f}s (already converged; "
                  f"re-run to provide the next mesh's start)")
            continue

        if not s.converged:
            print(f"  {key} NOT CONVERGED after {s.iterations} iterations ({dt:.1f}s)")
            print(f"  status: {s.status}")
            print(f"  max_div: {s.max_divergence():.6e}")
            print(f"  last residuals: u={s.history['u'][-1]:.4e} v={s.history['v'][-1]:.4e} mass={s.history['mass'][-1]:.4e}")
            raise SystemExit(1)

        NuW = s.nu("W"); NuE = s.nu("E"); Pb = s.Pbar / P0
        psi_min, psi_x, psi_y = s.psi_min()
        _, Nu_max, ystar = s.nu_local("W")
        rec = dict(M=M, converged=s.converged, iterations=s.iterations,
                   wall_time_s=dt, Nu_W=NuW, Nu_E=NuE, Nu_bal=abs(NuW - NuE),
                   Pbar_P0=Pb, psi_min=psi_min, Nu_max=Nu_max, y_Nu_max=ystar,
                   max_div=s.max_divergence(), status=s.status)
        results[key] = rec
        if "config" not in results:
            results["config"] = {"beta_deg": BETA, "eps": EPS, "ra": RA,
                                  "alpha_u": ALPHA_U, "alpha_p": ALPHA_P, "alpha_T": ALPHA_T}
        with open(JSON_PATH, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"  {key} its={s.iterations:6d} {dt:8.1f}s  Nu_W={NuW:.6f} Nu_E={NuE:.6f} "
              f"Pbar/P0={Pb:.6f} psi_min={psi_min:.6e} Nu_max={Nu_max:.3f}@{ystar:.4f} "
              f"{s.status}")

    ks = ["M64", "M128", "M256"]
    if all(k in results and results[k].get("converged") for k in ks):
        f256, f128, f64 = results[ks[2]], results[ks[1]], results[ks[0]]
        for qty in ("Nu_W", "Pbar_P0", "psi_min", "Nu_max"):
            v256, v128, v64 = f256[qty], f128[qty], f64[qty]
            p, rich = richardson(v256, v128, v64)
            f256[f"{qty}_order"] = p
            f256[f"{qty}_rich"] = rich
        results[ks[2]] = f256
        with open(JSON_PATH, "w") as fh:
            json.dump(results, fh, indent=1)

    print("\nDone -", JSON_PATH)


if __name__ == "__main__":
    main()
