"""Test from_state() fix (Task 2+3+4): load all 4 states, verify Nu etc., round-trip, Richardson.

Usage:
  python scripts/postprocess_skew.py          # print table + write skew60_reference.json
  python scripts/postprocess_skew.py --test   # also run round-trip + 10-step stability
"""
import sys, os, json, time, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results"); OUT = RES
STATE_DIR = os.path.join(RES, "state_skew60")
JSON_PATH = os.path.join(OUT, "skew60_reference.json")

from simplefvm.lowmach import LowMachSolver
from simplefvm.properties import P0

def get_before(s, key):
    if key == "Nu_W": return s.nu("W")
    if key == "Nu_E": return s.nu("E")
    if key == "Pbar_P0": return s.Pbar / P0
    if key == "psi_min":
        v, _, _ = s.psi_min(); return v
    if key == "Nu_max_hot":
        _, v, _ = s.nu_local("W"); return v
    if key == "y_Nu_max_hot":
        _, _, v = s.nu_local("W"); return v
    if key == "Nu_max_cold":
        _, v, _ = s.nu_local("E"); return v
    if key == "y_Nu_max_cold":
        _, _, v = s.nu_local("E"); return v
    return None

def load_and_check(M, do_step=False):
    path = os.path.join(STATE_DIR, f"state_skew60_M{M}.npz")
    s = LowMachSolver.from_state(path)

    if do_step:
        s.step()

    NuW = s.nu("W"); NuE = s.nu("E")
    Pb = s.Pbar / P0
    psi_min, _, _ = s.psi_min()
    nubal = abs(NuW - NuE) / (abs(NuW) + 1e-30)

    # nu_local on W (hot wall) — LowMachSolver override
    _, Nu_hot, y_hot = s.nu_local("W")
    _, Nu_cold, y_cold = s.nu_local("E")

    Nu_max_ratio = Nu_hot / Nu_cold if abs(Nu_cold) > 1e-30 else 0.0
    return dict(M=M, Nu_W=NuW, Nu_E=NuE, Nu_bal=nubal, Pbar_P0=Pb,
                psi_min=psi_min,
                Nu_max_hot=Nu_hot, y_Nu_max_hot=y_hot,
                Nu_max_cold=Nu_cold, y_Nu_max_cold=y_cold,
                Nu_max_ratio=Nu_max_ratio)

# ================================================================ Task 2
print("=" * 75)
print("TASK 2: Post-processing after load (no step)")
print("=" * 75)

MESHES = (32, 64, 128, 256)
# expected values
EXPECTED = {
    32:  dict(Nu_W=4.1977953380, Pbar_P0=0.9290607835, Nu_max_hot=6.9155619, y_hot=0.20313, Nu_max_cold=5.6286065, y_cold=0.76562),
    64:  dict(Nu_W=4.0916080322, Pbar_P0=0.9317179852, Nu_max_hot=6.6604802, y_hot=0.21094, Nu_max_cold=5.5506285, y_cold=0.78906),
    128: dict(Nu_W=4.0533345092, Pbar_P0=0.9336090799, Nu_max_hot=6.5772268, y_hot=0.21484, Nu_max_cold=5.4694944, y_cold=0.77734),
    256: dict(Nu_W=4.0430729238, Pbar_P0=0.9341544013, Nu_max_hot=6.5557552, y_hot=0.21680, Nu_max_cold=5.4448459, y_cold=0.77539),
}

fmt_hdr = f"{'M':>4s} {'Nu_W':>14s} {'Nu_E':>14s} {'Nu_bal':>10s} {'Pbar/P0':>12s} {'psi_min':>14s} {'Nu_hot':>10s} {'y_hot':>8s} {'Nu_cold':>10s} {'y_cold':>8s} {'ratio':>10s}"
fmt_row = "{M:4d} {Nu_W:14.10f} {Nu_E:14.10f} {Nu_bal:10.2e} {Pbar_P0:12.10f} {psi_min:14.10e} {Nu_max_hot:10.7f} {y_Nu_max_hot:8.5f} {Nu_max_cold:10.7f} {y_Nu_max_cold:8.5f} {Nu_max_ratio:10.6f}"

print(fmt_hdr)
all_recs = []
for M in MESHES:
    d = load_and_check(M)
    all_recs.append(d)
    exp = EXPECTED[M]
    ok_Nu = abs(d["Nu_W"] - exp["Nu_W"]) / (abs(exp["Nu_W"]) + 1e-30) < 1e-7
    ok_Pb = abs(d["Pbar_P0"] - exp["Pbar_P0"]) < 1e-10
    ok_hot = abs(d["Nu_max_hot"] - exp["Nu_max_hot"]) / (abs(exp["Nu_max_hot"]) + 1e-30) < 1e-4
    ok_cold = abs(d["Nu_max_cold"] - exp["Nu_max_cold"]) / (abs(exp["Nu_max_cold"]) + 1e-30) < 1e-4
    ok_bal = d["Nu_bal"] < 1e-7
    all_ok = ok_Nu and ok_Pb and ok_hot and ok_cold and ok_bal
    print(fmt_row.format(**d), f"  {'OK' if all_ok else 'FAIL'}"
          f"  Nu:{'OK' if ok_Nu else 'FAIL'}"
          f"  Pb:{'OK' if ok_Pb else 'FAIL'}"
          f"  hot:{'OK' if ok_hot else 'FAIL'}"
          f"  cold:{'OK' if ok_cold else 'FAIL'}"
          f"  bal:{'OK' if ok_bal else 'FAIL'}")

# ================================================================ Task 3: round-trip
do_test = "--test" in sys.argv
if do_test:
    print("\n" + "=" * 75)
    print("TASK 3: Round-trip test (M=32)")
    print("=" * 75)

    from simplefvm.properties import cavity_length, mu_sutherland
    EPS = 0.6; RA = 1e5; BETA = 60
    mu0 = mu_sutherland(600.0); L = cavity_length(RA, EPS, mu0)

    # fresh solve
    print("  Running M32 to convergence...")
    t0 = time.perf_counter()
    s1 = LowMachSolver(M=32, shape="parallelogram", beta_deg=BETA,
                        eps=EPS, Ra=RA, L=L, properties="sutherland",
                        alpha_u=0.4, alpha_p=0.15, alpha_T=0.6,
                        p_solver="direct", n_nonorth_correctors=2)
    s1.run(max_outer=20000, tol=1e-8, verbose=0)
    print(f"    its={s1.iterations} {time.perf_counter()-t0:.1f}s Nu={s1.nu('W'):.10f}")

    vals_before = {k: get_before(s1, k) for k in ["Nu_W", "Nu_E", "Pbar_P0", "psi_min",
                                                    "Nu_max_hot", "y_Nu_max_hot",
                                                    "Nu_max_cold", "y_Nu_max_cold"]}

    # save and reload
    tmp = os.path.join(OUT, "_roundtrip_tmp.npz")
    s1.save(tmp)
    s2 = LowMachSolver.from_state(tmp)
    os.remove(tmp)

    vals_after = {k: get_before(s2, k) for k in vals_before}

    print(f"  {'metric':>16s} {'before':>18s} {'after':>18s} {'|diff|':>12s} {'bit?':>6s}")
    all_bit = True
    for k in vals_before:
        a, b = vals_before[k], vals_after[k]
        diff = abs(a - b) if a is not None and b is not None else float('nan')
        bit = "YES" if diff == 0.0 or (diff < 1e-15) else "NO"
        if bit == "NO":
            all_bit = False
        print(f"  {k:>16s} {a:18.10e} {b:18.10e} {diff:12.2e} {bit:>6s}")

    # 10 more steps — Nu must not move by more than 1e-9 relative
    Nu0 = s2.nu("W")
    for _ in range(10):
        s2.step()
    Nu10 = s2.nu("W")
    drift = abs(Nu10 - Nu0) / (abs(Nu0) + 1e-30)
    print(f"\n  Nu before 10 steps: {Nu0:.10f}")
    print(f"  Nu after  10 steps: {Nu10:.10f}")
    print(f"  Drift: {drift:.2e}  {'OK' if drift < 1e-9 else 'FAIL'}")
    print(f"  Bit-identical round-trip: {'YES' if all_bit else 'NO (some diffs)'}")

# ================================================================ Task 4: Richardson
print("\n" + "=" * 75)
print("TASK 4: Richardson extrapolation on 64/128/256")
print("=" * 75)

rec64 = all_recs[1]; rec128 = all_recs[2]; rec256 = all_recs[3]

def richardson(f1, f2, f3, h_ratio=2.0):
    if abs(f2 - f3) < 1e-30 or abs(f1 - f2) < 1e-30:
        return None, None, None, None
    p = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(h_ratio)
    rich = f1 + (f1 - f2) / (h_ratio**p - 1.0)
    gci_fine = 1.25 * abs(f1 - f2) / ((h_ratio**p - 1.0) * abs(rich) + 1e-30) * 100
    return float(p), float(rich), float(gci_fine)

EXPECTED_RICH = {
    "Nu_W": 4.0393138, "Pbar_P0": 0.93437537, "psi_min": -6.3649465e-4,
    "Nu_max_hot": 6.5482931, "Nu_max_cold": 5.4340902,
}

rich_results = {}
for qty in ["Nu_W", "Pbar_P0", "psi_min", "Nu_max_hot", "Nu_max_cold"]:
    f1 = rec256[qty]; f2 = rec128[qty]; f3 = rec64[qty]
    p, rich, gci = richardson(f1, f2, f3)
    rich_results[qty] = {"order": p, "rich": rich, "GCI_fine": gci}

print(f"  {'qty':>16s} {'p':>8s} {'R_h0':>18s} {'exp_R_h0':>18s} {'GCI%':>8s}")
for qty in ["Nu_W", "Pbar_P0", "psi_min", "Nu_max_hot", "Nu_max_cold"]:
    rr = rich_results[qty]
    exp = EXPECTED_RICH.get(qty)
    p_str = f"{rr['order']:.3f}" if rr['order'] is not None else "-"
    rich_str = f"{rr['rich']:.10e}" if rr['rich'] is not None else "-"
    exp_str = f"{exp:.10e}" if exp is not None else "-"
    gci_str = f"{rr['GCI_fine']:.2f}" if rr['GCI_fine'] is not None else "-"
    match = "OK" if exp is not None and rr['rich'] is not None and abs(rr['rich'] - exp) / (abs(exp) + 1e-30) < 1e-4 else ""
    print(f"  {qty:>16s} {p_str:>8s} {rich_str:>18s} {exp_str:>18s} {gci_str:>8s} {match}")

# hot/cold ratio
hot_rich = rich_results.get("Nu_max_hot", {}).get("rich")
cold_rich = rich_results.get("Nu_max_cold", {}).get("rich")
if hot_rich and cold_rich:
    ratio = hot_rich / cold_rich
    print(f"  {'hot/cold ratio':>16s} {'':>8s} {ratio:18.10f}")

# write JSON
all_data = {f"M{M}": r for M, r in zip(MESHES, all_recs)}
all_data["richardson"] = {}
for qty in rich_results:
    all_data["richardson"][qty] = {
        "values": [rec64[qty], rec128[qty], rec256[qty]],
        "order": rich_results[qty]["order"],
        "rich": rich_results[qty]["rich"],
        "GCI_fine_pct": rich_results[qty]["GCI_fine"],
    }
if hot_rich and cold_rich and abs(cold_rich) > 1e-30:
    all_data["richardson"]["Nu_max_ratio"] = {"rich": hot_rich / cold_rich}
with open(JSON_PATH, "w") as fh:
    json.dump(all_data, fh, indent=1, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
print(f"\n  Written -> {JSON_PATH}")
print("\nDone.")