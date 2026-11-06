"""Figures, error tables and the grid-convergence (GCI) table from results/*.npz.

Run `python scripts/run_vv.py` (upwind) and `python scripts/run_vv.py --dc` (central by deferred
correction) first.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from simplefvm import SimpleSolver, ghia  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
COL = {100: "#1f77b4", 400: "#d62728", 1000: "#2ca02c"}
SCHEMES = {"": "upwind (1st order)", "_dc": "central, deferred correction (2nd order)"}


def path(Re, M, tag):
    return os.path.join(RES, f"cavity_Re{Re}_M{M}{tag}.npz")


def load(Re, M, tag=""):
    d = np.load(path(Re, M, tag))
    s = SimpleSolver.__new__(SimpleSolver)
    s.M, s.X, s.Y = M, d["X"], d["Y"]
    s.fields = type("F", (), {"U": d["U"], "V": d["V"]})()
    return s, d


def centre(s):
    return SimpleSolver.centerline_u(s), SimpleSolver.centerline_v(s)


summary = {}

# ------------------------------------------------ Ghia comparison, 128 x 128, both schemes
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))
for Re in (100, 400, 1000):
    for tag, ls in (("", "--"), ("_dc", "-")):
        if not os.path.exists(path(Re, 128, tag)):
            continue
        s, _ = load(Re, 128, tag)
        (y, u), (x, v) = centre(s)
        ax[0].plot(u, y, ls, color=COL[Re], lw=1.2)
        ax[1].plot(x, v, ls, color=COL[Re], lw=1.2)
        summary.setdefault(f"Re{Re}_128", {})[SCHEMES[tag]] = ghia.compare(s, Re)
    ax[0].plot(ghia.U[Re], ghia.Y_U, "o", mfc="none", color=COL[Re], ms=5, label=f"Re = {Re}")
    ax[1].plot(ghia.X_V, ghia.V[Re], "o", mfc="none", color=COL[Re], ms=5, label=f"Re = {Re}")
ax[1].plot(ghia.X_V[5], ghia.V[400][5], "x", color="k", ms=8, label="suspected misprint in Ghia")
ax[0].plot([], [], "k-", label="central (deferred correction)")
ax[0].plot([], [], "k--", label="upwind")
ax[0].set(xlabel="u / U_lid", ylabel="y / L", title="u on the vertical centre line, 128x128")
ax[1].set(xlabel="x / L", ylabel="v / U_lid", title="v on the horizontal centre line, 128x128")
ax[0].legend(fontsize=7.5, title="circles: Ghia et al. (1982)", title_fontsize=7.5)
ax[1].legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ghia_centerlines.png"))
plt.close(fig)

# ------------------------------------------------ mesh convergence, Re = 100
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
gci = {}
for tag, mk in (("", "s--"), ("_dc", "o-")):
    meshes = [m for m in (32, 64, 128, 256) if os.path.exists(path(100, m, tag))]
    errs, umins = [], []
    for M in meshes:
        s, _ = load(100, M, tag)
        (y, u), _ = centre(s)
        if tag == "_dc":
            ax[0].plot(u, y, label=f"{M}x{M} (central)")
        errs.append(ghia.compare(s, 100)["u_rms"])
        umins.append(float(u.min()))
    if meshes:
        summary[f"Re100_mesh{tag or '_upwind'}"] = dict(zip(map(str, meshes), errs))
    if len(umins) >= 3:
        f3, f2, f1 = umins[-3], umins[-2], umins[-1]
        r = 2.0
        p = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(r)
        gci[SCHEMES[tag]] = {
            "quantity": "min u on vertical centre line, Re = 100", "meshes": meshes[-3:],
            "values": [f3, f2, f1], "observed_order": float(p),
            "richardson_extrapolated": float(f1 + (f1 - f2) / (r ** p - 1)),
            "GCI_fine_percent": float(1.25 * abs((f1 - f2) / f1) / (r ** p - 1) * 100),
            "ghia_value": float(ghia.U[100].min())}
        # self-convergence: error of min(u) against this scheme's Richardson-extrapolated value
        h = 1.0 / np.array(meshes)
        e = np.abs(np.array(umins) - gci[SCHEMES[tag]]["richardson_extrapolated"])
        ax[1].loglog(h[e > 0], e[e > 0], mk, label=f"{SCHEMES[tag]}, p = {p:.2f}")
        ax[1].loglog(h, e[0] * (h / h[0]) ** (2 if tag else 1), ":", color="grey", lw=1)
ax[0].plot(ghia.U[100], ghia.Y_U, "ko", mfc="none", label="Ghia et al.")
ax[0].set(xlabel="u / U_lid", ylabel="y / L", title="Re = 100, central scheme, mesh refinement")
ax[0].legend(fontsize=8)
ax[1].plot([], [], ":", color="grey", label="ideal slope (1 and 2)")
ax[1].set(xlabel="cell size h / L", ylabel="|min u - Richardson extrapolation|",
          title="Re = 100: observed order of accuracy")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "mesh_convergence_re100.png"))
plt.close(fig)
with open(os.path.join(RES, "gci_re100.json"), "w") as fh:
    json.dump(gci, fh, indent=1)

# ------------------------------------------------ residual histories (central, 128)
fig, ax = plt.subplots(figsize=(6, 4))
for Re in (100, 400, 1000):
    tag = "_dc" if os.path.exists(path(Re, 128, "_dc")) else ""
    _, d = load(Re, 128, tag)
    ax.semilogy(np.maximum.reduce([d["res_u"], d["res_v"], d["res_m"]]), color=COL[Re], label=f"Re = {Re}")
ax.set(xlabel="SIMPLE outer iteration", ylabel="max scaled residual (u, v, mass)",
       title="Iterative convergence, 128x128, central scheme")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG, "residuals.png"))
plt.close(fig)

# ------------------------------------------------ streamlines (central, 128)
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, Re in zip(ax, (100, 400, 1000)):
    tag = "_dc" if os.path.exists(path(Re, 128, "_dc")) else ""
    _, d = load(Re, 128, tag)
    psi = d["psi"]
    neg = np.linspace(psi.min(), 0, 14)[:-1]
    pos = np.geomspace(1e-6, max(psi.max(), 2e-6), 5) if psi.max() > 1e-6 else np.array([])
    a.contour(d["X_cor"], d["Y_cor"], psi, levels=np.unique(np.concatenate([neg, pos])),
              colors="k", linewidths=0.7)
    a.set(aspect="equal", title=f"Re = {Re}", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
    a.grid(False)
fig.suptitle("Stream function, 128x128 (corner eddies included)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "streamlines.png"))
plt.close(fig)

# ------------------------------------------------ curved lids (demo, central)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
demo = {}
for a, (shape, H1, H2) in zip(ax, (("quadratic", 2.0, 1.2), ("crazy", 1.4, 1.0))):
    # inclined lid cells: neglected non-orthogonal terms need stronger under-relaxation
    s = SimpleSolver(M=64, Re=100, H1=H1, H2=H2, shape=shape, gamma_dc=1.0, alpha_u=0.5, alpha_p=0.2)
    cache = os.path.join(RES, f"curved_{shape}.npz")
    if os.path.exists(cache):                       # reuse a finished run
        c = np.load(cache)
        s.fields.U, s.fields.V = c["U"], c["V"]
        s.fields.mflux_x, s.fields.mflux_y = c["mx"], c["my"]
        demo[shape] = json.loads(str(c["meta"]))
    else:
        s.run(max_outer=20000, tol=1e-6, verbose=0)
        demo[shape] = {"converged": s.converged, "iterations": s.iterations,
                       "wall_time_s": round(s.wall_time, 1), "max_cell_divergence": s.max_divergence()}
        np.savez_compressed(cache, U=s.fields.U, V=s.fields.V, mx=s.fields.mflux_x,
                            my=s.fields.mflux_y, meta=json.dumps(demo[shape]))
    speed = np.hypot(s.fields.U, s.fields.V)[1:-1, 1:-1]
    pc = a.pcolormesh(s.X_cor, s.Y_cor, speed, shading="flat", cmap="viridis")
    a.contour(s.X_cor, s.Y_cor, s.stream_function(), levels=12, colors="w", linewidths=0.6)
    a.set(aspect="equal", title=f"lid shape '{shape}', Re = 100, 64x64")
    a.grid(False)
    fig.colorbar(pc, ax=a, label="|u| / U_lid", shrink=0.8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "curved_lids.png"))
plt.close(fig)

summary["gci"] = gci
summary["curved_lid_demo"] = demo
with open(os.path.join(RES, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary, indent=1))
