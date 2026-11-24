"""Re-create README tables with the consistent RC (lowmach) code into results/*_lowmach.json.

Runs: Ghia RMS at M=128, GCI Re=100, Demirdzic 45 deg at 32/64/128 (Re 100 and Re 1000).
WARNING: long run — the M=256 upwind case alone is ~90 min. Total: ~3.5 hours.

    python scripts/refresh_tables.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from simplefvm import SimpleSolver, ghia

OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

# Demirdzic 1992 reference values (45-degree parallelogram)
DEMIRDZIC_PSI_MIN = {
    100: -0.070226,
    1000: -0.053507,   # Demirdzic, Lilek & Peric (1992), Table I, 320 x 320 CV
}

EST_TOTAL = "~3.5 hours"
print(f"refresh_tables.py starting — estimated total: {EST_TOTAL}")
print("=" * 70)


# -- helpers ----------------------------------------------------------------
def wall_clock(start):
    return round(time.perf_counter() - start, 1)


def wall_mass_flux(solver):
    mx, my = solver.fields.mflux_x, solver.fields.mflux_y
    return float(np.abs(mx[0, :].sum() + mx[-1, :].sum() + my[:, 0].sum() + my[:, -1].sum())
                  / solver.mass_scale)


def run_one(Re, M, *, shape="rectangular", gamma_dc=0.0, beta_deg=None,
            n_nonorth_correctors=0, alpha_u=0.7, alpha_p=0.3, tol=1e-6):
    s = SimpleSolver(M=M, Re=Re, shape=shape, gamma_dc=gamma_dc, beta_deg=beta_deg,
                     n_nonorth_correctors=n_nonorth_correctors,
                     alpha_u=alpha_u, alpha_p=alpha_p)
    s.run(max_outer=40000, tol=tol, verbose=0)
    y, u = s.centerline_u()
    x, v = s.centerline_v()
    wflux = wall_mass_flux(s)
    err = ghia.compare(s, Re) if shape == "rectangular" else {
        "u_rms": None, "u_max": None, "v_rms": None, "v_max": None}
    psi_min = float(s.stream_function().min())
    return dict(Re=Re, M=M, shape=shape, scheme="central (dc)" if gamma_dc else "upwind",
                gamma_dc=gamma_dc, beta_deg=beta_deg, alpha_u=alpha_u, alpha_p=alpha_p,
                n_nonorth_correctors=n_nonorth_correctors, tol=tol,
                converged=s.converged, iterations=s.iterations,
                wall_time_s=round(s.wall_time, 1),
                psi_min=psi_min, u_min=float(u.min()),
                v_max=float(v.max()), v_min=float(v.min()),
                max_cell_divergence=s.max_divergence(),
                wall_mass_flux=wflux, **{"ghia_" + k: v for k, v in err.items()})


# ===================================================== Ghia RMS, M=128, both schemes
print("\n--- Ghia RMS at M=128 ---")
ghia_table = []
for scheme, gamma in [("upwind", 0.0), ("dc", 1.0)]:
    tag = "upwind" if gamma == 0 else "dc"
    for Re in (100, 400, 1000):
        t0 = time.perf_counter()
        r = run_one(Re, 128, gamma_dc=gamma)
        print(f"  Re={Re:4d} {tag:>6s}  its={r['iterations']:5d}  {wall_clock(t0):6.1f}s  "
              f"div={r['max_cell_divergence']:.1e}  wall_flux={r['wall_mass_flux']:.1e}  "
              f"u_rms={r['ghia_u_rms']}  v_rms={r['ghia_v_rms']}")
        ghia_table.append(r)

with open(os.path.join(OUT, "ghia_rms_128_lowmach.json"), "w") as fh:
    json.dump(ghia_table, fh, indent=1)
print("  -> ghia_rms_128_lowmach.json written")

# ===================================================== GCI Re=100
print("\n--- GCI Re=100 ---")
gci_runs = {}
for scheme, gamma in [("upwind", 0.0), ("dc", 1.0)]:
    tag = "upwind" if gamma == 0 else "dc"
    meshes = (32, 64, 128, 256) if gamma == 0 else (32, 64, 128)
    values = {}
    for M in meshes:
        t0 = time.perf_counter()
        r = run_one(100, M, gamma_dc=gamma)
        values[str(M)] = r
        print(f"  Re=100 {tag:>6s} M={M:3d}  its={r['iterations']:5d}  {wall_clock(t0):7.1f}s  "
              f"div={r['max_cell_divergence']:.1e}  wall_flux={r['wall_mass_flux']:.1e}  "
              f"u_min={r['u_min']:+f}")
    umins = [values[str(m)]["u_min"] for m in meshes]
    if len(umins) >= 3:
        f3, f2, f1 = umins[-3], umins[-2], umins[-1]
        rho = 2.0
        p = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(rho)
        gci_val = float(1.25 * abs((f1 - f2) / f1) / (rho ** p - 1) * 100)
        rich = float(f1 + (f1 - f2) / (rho ** p - 1))
        gci_runs[tag] = {
            "quantity": "min u on vertical centre line, Re = 100",
            "meshes": list(meshes[-3:]), "values": [f3, f2, f1],
            "observed_order": float(p), "richardson_extrapolated": rich,
            "GCI_fine_percent": gci_val, "ghia_value": float(ghia.U[100].min())
        }

with open(os.path.join(OUT, "gci_re100_lowmach.json"), "w") as fh:
    json.dump(gci_runs, fh, indent=1)
print("  -> gci_re100_lowmach.json written")

# ===================================================== Demirdzic 45 deg parallelogram
print("\n--- Demirdzic 45 deg parallelogram, M=32/64/128, tol=1e-9 ---")
dem = []
for Re in (100, 1000):
    alpha_u = 0.7 if Re == 100 else 0.5
    alpha_p = 0.3 if Re == 100 else 0.2
    for M in (32, 64, 128):
        t0 = time.perf_counter()
        r = run_one(Re, M, shape="parallelogram", beta_deg=45, gamma_dc=1.0,
                    n_nonorth_correctors=2, alpha_u=alpha_u, alpha_p=alpha_p, tol=1e-9)
        ref = DEMIRDZIC_PSI_MIN.get(Re)
        if ref is not None:
            err_pct = float(abs(r["psi_min"] - ref) / abs(ref) * 100)
            ref_str = f"  err={err_pct:.4e} %"
        else:
            err_pct = None
            ref_str = "  (no Demirdzic ref value in repo)"
        r["demirdzic_psi_min_ref"] = ref
        r["demirdzic_psi_min_err_pct"] = err_pct
        print(f"  Re={Re:4d} M={M:3d}  its={r['iterations']:5d}  {wall_clock(t0):7.1f}s  "
              f"psi_min={r['psi_min']:+.8e}  div={r['max_cell_divergence']:.1e}  "
              f"wall_flux={r['wall_mass_flux']:.1e}{ref_str}")
        dem.append(r)

with open(os.path.join(OUT, "demirdzic_45_lowmach.json"), "w") as fh:
    json.dump(dem, fh, indent=1)
print("  -> demirdzic_45_lowmach.json written")

print("\nAll results written to results/*_lowmach.json")