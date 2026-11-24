"""Post-process hidden-case states and produce a reference JSON.

Reads the run JSON and state files from --out-dir / --tag.  Uses the Richardson
orders from the project-case skew60 reference to extrapolate M64->M128.

Usage:
  python scripts/postprocess_hidden.py --out-dir D:\\Ai_training\\hidden-ref --tag hidden
"""
import argparse, sys, os, json, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SKEW60_REF = os.path.join(ROOT, "results", "skew60_reference.json")
with open(SKEW60_REF) as fh:
    _skew60 = json.load(fh)
PROJECT_ORDERS = {qty: v["order"] for qty, v in _skew60["richardson"].items()
                  if "order" in v}

from simplefvm.lowmach import LowMachSolver
from simplefvm.properties import P0


def load_state(path, eps, Ra, beta_deg):
    s = LowMachSolver.from_state(path, eps=eps, Ra=Ra, beta_deg=beta_deg)
    NuW = s.nu("W"); NuE = s.nu("E")
    Pb = s.Pbar / P0
    psi_min, _, _ = s.psi_min()
    nubal = abs(NuW - NuE) / (abs(NuW) + 1e-30)
    _, Nu_hot, y_hot = s.nu_local("W")
    _, Nu_cold, y_cold = s.nu_local("E")
    ratio = Nu_hot / Nu_cold if abs(Nu_cold) > 1e-30 else float("nan")
    return dict(Nu_W=NuW, Nu_E=NuE, Nu_bal=nubal, Pbar_P0=Pb,
                psi_min=psi_min,
                Nu_max_hot=Nu_hot, y_Nu_max_hot=y_hot,
                Nu_max_cold=Nu_cold, y_Nu_max_cold=y_cold,
                Nu_max_ratio=ratio)


def two_mesh_extrap(f64, f128, p):
    denom = 2.0**p - 1.0
    if abs(denom) < 1e-30:
        return f128
    return f128 + (f128 - f64) / denom


def observed_order(f32, f64, f128):
    num = abs(f32 - f64)
    den = abs(f64 - f128)
    if den < 1e-30 or num < 1e-30:
        return None
    return float(np.log(num / den) / np.log(2.0))


def main():
    ap = argparse.ArgumentParser(description="Post-process hidden-case states")
    ap.add_argument("--out-dir", default="results/",
                    help="directory with the run JSON and states (default: results/)")
    ap.add_argument("--tag", default="hidden",
                    help="case tag (default: hidden)")
    args = ap.parse_args()

    TAG = args.tag
    OUT = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    JSON_PATH = os.path.join(OUT, f"lowmach_{TAG}.json")
    STATE_DIR = os.path.join(OUT, f"state_{TAG}")
    REF_PATH = os.path.join(OUT, f"{TAG}_reference.json")

    with open(JSON_PATH) as fh:
        run_data = json.load(fh)
    config = run_data.get("config", {})
    EPS = config.get("eps", 0.6)
    RA = config.get("ra", 1e5)
    BETA = config.get("beta_deg", 60)

    MESHES = sorted(int(k[1:]) for k in run_data if k.startswith("M") and k[1:].isdigit())
    print(f"Case: beta={BETA}  eps={EPS}  Ra={RA:.0e}  meshes={MESHES}")

    recs = {}
    for M in MESHES:
        path = os.path.join(STATE_DIR, f"state_{TAG}_M{M}.npz")
        d = load_state(path, eps=EPS, Ra=RA, beta_deg=BETA)
        d["M"] = M
        recs[f"M{M}"] = d

    fmt_hdr = f"{'M':>4s} {'Nu_W':>14s} {'Nu_E':>14s} {'Nu_bal':>10s} {'Pbar/P0':>12s} {'psi_min':>14s} {'Nu_hot':>10s} {'y_hot':>8s} {'Nu_cold':>10s} {'y_cold':>8s} {'ratio':>10s}"
    fmt_row = "{M:4d} {Nu_W:14.10f} {Nu_E:14.10f} {Nu_bal:10.2e} {Pbar_P0:12.10f} {psi_min:14.10e} {Nu_max_hot:10.7f} {y_Nu_max_hot:8.5f} {Nu_max_cold:10.7f} {y_Nu_max_cold:8.5f} {Nu_max_ratio:10.6f}"

    print(fmt_hdr)
    for M in MESHES:
        print(fmt_row.format(**recs[f"M{M}"]))

    QUANTITIES = ["Nu_W", "Pbar_P0", "psi_min", "Nu_max_hot", "Nu_max_cold"]

    richardson = {}
    if "M64" in recs and "M128" in recs:
        f64 = recs["M64"]; f128 = recs["M128"]

        for qty in QUANTITIES:
            p_project = PROJECT_ORDERS.get(qty)
            ext_project = two_mesh_extrap(f64[qty], f128[qty], p_project) if p_project else None

            ext_p15 = two_mesh_extrap(f64[qty], f128[qty], 1.5)
            ext_p20 = two_mesh_extrap(f64[qty], f128[qty], 2.0)

            if ext_project is not None and abs(ext_project) > 1e-30:
                band = max(abs(ext_p15 - ext_project), abs(ext_p20 - ext_project)) / abs(ext_project)
            else:
                band = None

            m64_err = None
            if ext_project is not None and abs(ext_project) > 1e-30:
                m64_err = abs(f64[qty] - ext_project) / abs(ext_project) * 100.0

            richardson[qty] = {
                "values": [f64[qty], f128[qty]],
                "project_order": p_project,
                "rich": ext_project,
                "extrap_band_pct": band,
                "m64_err_pct": m64_err,
            }

        hot_rich = richardson.get("Nu_max_hot", {}).get("rich")
        cold_rich = richardson.get("Nu_max_cold", {}).get("rich")
        if hot_rich and cold_rich and abs(cold_rich) > 1e-30:
            richardson["Nu_max_ratio"] = {"rich": hot_rich / cold_rich}

    obs_order = {}
    if "M32" in recs and "M64" in recs and "M128" in recs:
        f32 = recs["M32"]; f64 = recs["M64"]; f128 = recs["M128"]
        for qty in QUANTITIES:
            p = observed_order(f32[qty], f64[qty], f128[qty])
            obs_order[qty] = {
                "order": p,
                "note": "pre-asymptotic, not used",
                "values": [f32[qty], f64[qty], f128[qty]],
            }
        obs_order["Nu_max_ratio"] = {
            "order": observed_order(f32["Nu_max_ratio"], f64["Nu_max_ratio"], f128["Nu_max_ratio"]),
            "note": "pre-asymptotic, not used",
            "values": [f32["Nu_max_ratio"], f64["Nu_max_ratio"], f128["Nu_max_ratio"]],
        }

    out = {"meshes": recs}
    if richardson:
        out["richardson"] = richardson
    if obs_order:
        out["observed_order"] = obs_order

    with open(REF_PATH, "w") as fh:
        json.dump(out, fh, indent=1,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    print(f"\nWritten -> {REF_PATH}")


if __name__ == "__main__":
    main()
