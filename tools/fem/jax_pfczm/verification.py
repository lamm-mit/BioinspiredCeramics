"""Analytic / benchmark verification cases and the verification suite.

Run ``python -m jax_pfczm --level quick`` (or ``production``), or the console script
``jax-pfczm-verify``, to print all checks.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .bcs import bar_bc, crack_path, crack_tip_x, surfing_bc
from .grid import Grid
from .models import Driving, Material, ModelKind, PhaseFieldModel, Softening, discrete_toughness_factor
from .solver import PhaseFieldSolver


# ---------------------------------------------------------------------------
# Verification cases
# ---------------------------------------------------------------------------
def verify_at2_bar(nx: int = 20, l: float = 0.1, E: float = 1.0, nu: float = 0.2, Gc: float = 1.0,
                   n_steps: int = 80) -> dict:
    """Homogeneous AT2 bar: sigma(eps) vs the analytic homogeneous solution; peak vs (3/16) sqrt(3E'Gc/l)."""
    L = 1.0
    grid = Grid(nx, 1, L / nx)
    mat = Material.homogeneous(grid, E, nu, Gc)
    model = PhaseFieldModel("at2", length=l)
    solver = PhaseFieldSolver(grid, model)
    Ep = E / (1 - nu**2)
    eps_c = math.sqrt(Gc / (3 * Ep * l))
    loads = np.linspace(0, 2.0 * eps_c * L, n_steps + 1)[1:]
    bc, right = bar_bc(grid)
    res = solver.run(mat, bc, loads, observers={"F": lambda u, d, f: f[right, 0].sum()})
    sig = res["F"] / grid.height
    eps = loads / L
    x = Ep * eps**2 * l / Gc
    sig_an = Ep * eps / (1 + x) ** 2
    sc_an = 3.0 / 16.0 * math.sqrt(3 * Ep * Gc / l)
    pre = eps <= eps_c * (1 + 1e-9)  # homogeneous branch is stable up to the peak (localises after)
    err_curve = float(np.max(np.abs(sig[pre] - sig_an[pre])) / sc_an)
    eb = float(np.max(np.abs(res["external_work"][pre] - res["total"][pre])) / max(res["external_work"][pre][-1], 1e-300))
    return {"case": "at2_homogeneous_bar", "l": l, "h": grid.h, "E_plane_strain": Ep,
            "sigma_peak": float(sig.max()), "sigma_peak_analytic": sc_an,
            "rel_err_peak": float(abs(sig.max() - sc_an) / sc_an), "max_curve_err_over_sigma_c": err_curve,
            "energy_balance_rel_err": eb, "all_converged": bool(res["converged"].all()),
            "eps": eps.tolist(), "sigma": sig.tolist(), "sigma_analytic": sig_an.tolist()}


def verify_pfczm_bar(b: float, h: float, softening: Softening = "linear", E: float = 1.0, Gf: float = 1.0,
                     ft: float = 1.0, L: float = 1.0, weak: float = 0.02, n_steps: int = 120) -> dict:
    """PF-CZM bar (nu = 0) with a slightly weakened central column (f_t*(1-weak)).

    Checks: peak stress = f_t (independent of b); dissipated energy -> G_f * area * discrete factor;
    softening branch vs the target traction-separation law; energy balance W_ext = E_el + Gamma.
    """
    nx = int(round(L / h))
    grid = Grid(nx, 1, L / nx)
    mat = Material.homogeneous(grid, E, 0.0, Gf, ft)
    mat.ft[nx // 2] *= (1 - weak)
    model = PhaseFieldModel("pfczm", length=b, softening=softening)
    solver = PhaseFieldSolver(grid, model)
    wc = 2 * Gf / ft if softening == "linear" else 12.0 * Gf / ft
    u_pk = ft * (1 - weak) * L / E
    loads = np.concatenate([np.linspace(0, 0.9 * u_pk, 10)[1:], np.linspace(0.9 * u_pk, 1.1 * u_pk, 41)[1:],
                            np.linspace(1.1 * u_pk, u_pk + 1.05 * wc, n_steps)[1:]])
    bc, right = bar_bc(grid)
    res = solver.run(mat, bc, loads, observers={"F": lambda u, d, f: f[right, 0].sum()})
    sig = res["F"] / grid.height
    area = grid.height
    fac = discrete_toughness_factor("pfczm", grid.h / b)
    Gam = res["dissipated"][-1]
    ip = int(np.argmax(sig))
    wopen = loads - sig * L / E
    post = np.arange(len(sig)) > ip
    if softening == "linear":
        target = np.maximum(ft * (1 - wopen / wc), 0.0)
    else:
        target = ft * np.exp(-ft * wopen / Gf)
    tsl_err = float(np.max(np.abs(sig[post] - target[post])) / ft) if post.any() else float("nan")
    eb = float(np.max(np.abs(res["external_work"] - res["total"])) / max(res["external_work"][-1], 1e-300))
    return {"case": f"pfczm_bar_{softening}", "b": b, "h": grid.h, "sigma_peak": float(sig.max()),
            "ft_weak_column": ft * (1 - weak), "rel_err_peak": float(abs(sig.max() - ft * (1 - weak)) / ft),
            "dissipated": float(Gam), "Gf_area": Gf * area, "discrete_factor": fac,
            "rel_err_dissipated_vs_Gf": float(abs(Gam / (Gf * area) - 1.0)),
            "rel_err_dissipated_vs_Gf_discrete": float(abs(Gam / (Gf * area * fac) - 1.0)),
            "tsl_max_err_over_ft": tsl_err, "energy_balance_rel_err": eb,
            "all_converged": bool(res["converged"].all()), "final_max_d": float(res["max_d"][-1]),
            "opening": wopen.tolist(), "sigma": sig.tolist()}


def verify_surfing(kind: ModelKind = "at2", h: float = 0.5, length: float = 1.0, Lx: float = 30.0,
                   Ly: float = 12.0, E: float = 1.0, nu: float = 0.3, Gc: float = 1.0, ft: float = 1.0,
                   G_applied_over_Gc: float = 2.0, a0: float = 6.0, advance: float = 12.0, dt: float = 0.5,
                   n_ramp: int = 8, softening: Softening = "linear", driving: Optional[Driving] = None,
                   verbose: bool = False) -> dict:
    """Edge-notched strip under surfing (translating K-field) BCs: stable steady crack growth.

    Measures dGamma/da over the steady part (-> G_eff = G_c * discrete factor), the
    crack-path deviation from the notch plane and the energy balance.
    """
    nx, ny = int(round(Lx / h)), int(round(Ly / h))
    if ny % 2:
        ny += 1
    grid = Grid(nx, ny, h, 0.0, -ny * h / 2)
    mat = Material.homogeneous(grid, E, nu, Gc, ft)
    model = PhaseFieldModel(kind, length=length, softening=softening, driving=driving)
    solver = PhaseFieldSolver(grid, model, cg_tol=1e-10)
    Ep = E / (1 - nu**2)
    fac = discrete_toughness_factor(kind, h / length)
    K = math.sqrt(G_applied_over_Gc * Gc * fac * Ep)
    jmid = ny // 2
    nodes = grid.nodes
    pre = (np.isclose(nodes[:, 1], 0.0)) & (nodes[:, 0] <= a0 + 1e-9)
    bc = surfing_bc(grid, K, E, nu, a0, ramp=1.0)
    loads = np.concatenate([np.linspace(0, 1.0, n_ramp + 1)[1:], 1.0 + np.arange(dt, advance + 1e-9, dt)])
    res = solver.run(mat, bc, loads, d_fixed_one=pre, verbose=verbose,
                     observers={"a": lambda u, d, f: crack_tip_x(grid, d, jmid)})
    a = res["a"]
    Gam = res["dissipated"]
    steady = loads >= 1.0 + advance / 2.0
    if steady.sum() >= 3 and np.ptp(a[steady]) > 2 * h:
        slope = float(np.polyfit(a[steady], Gam[steady], 1)[0])
    else:
        slope = float("nan")
    xs, ys = crack_path(grid, res["d"], x_from=a0 + length)
    dev = float(np.max(np.abs(ys))) if ys.size else float("nan")
    eb = float(abs(res["external_work"][-1] - res["total"][-1]) / max(res["external_work"][-1], 1e-300))
    return {"case": f"surfing_{kind}", "h": h, "length": length, "h_over_length": h / length,
            "G_eff_measured": slope, "discrete_factor": fac, "G_eff_expected": Gc * fac,
            "rel_err_vs_discrete": float(abs(slope / (Gc * fac) - 1.0)),
            "bourdin_factor_1_plus_h_over_2l": 1 + h / (2 * length) if kind == "at2" else None,
            "crack_advance": float(a[-1] - a0), "max_path_deviation": dev,
            "energy_balance_rel_err": eb, "all_converged": bool(res["converged"].all()),
            "a": a.tolist(), "Gamma": Gam.tolist(), "loads": loads.tolist(), "d": res["d"], "grid": grid}


def run_verification_suite(level: str = "production") -> dict:
    """All solver verification cases (analytic / benchmark) with pass/fail checks."""
    prod = level == "production"
    out = {"level": level, "cases": [], "checks": []}

    def chk(name, ok, metric, tol, details):
        out["checks"].append({"name": name, "passed": bool(ok), "metric": float(metric), "tolerance": tol,
                              "details": details})

    r = verify_at2_bar()
    out["cases"].append({k: v for k, v in r.items() if not isinstance(v, list)})
    chk("AT2 bar peak stress = (3/16)sqrt(3E'Gc/l)", r["rel_err_peak"] < 1e-3, r["rel_err_peak"], 1e-3,
        f"sigma_peak {r['sigma_peak']:.6f} vs analytic {r['sigma_peak_analytic']:.6f}")
    chk("AT2 bar pre-peak sigma(eps) = analytic homogeneous solution", r["max_curve_err_over_sigma_c"] < 1e-3,
        r["max_curve_err_over_sigma_c"], 1e-3, "max |sigma - sigma_an| / sigma_c")
    chk("AT2 bar energy balance (pre-peak)", r["energy_balance_rel_err"] < 5e-3, r["energy_balance_rel_err"], 5e-3,
        "|W_ext - (E_el + Gamma)| / W_ext")
    for soft in ("linear", "exponential"):
        for b in ((0.1, 0.05) if prod else (0.1,)):
            # quick level: 80 steps (the original used 50, where the exponential-bar energy balance was 0.01009,
            # just above its 0.01 tolerance; 80 steps give 4.3e-3).  Production is unchanged.
            r = verify_pfczm_bar(b=b, h=0.0125 if prod else 0.025, softening=soft, n_steps=80)
            out["cases"].append({k: v for k, v in r.items() if not isinstance(v, list)})
            chk(f"PF-CZM {soft} bar peak = f_t (b={b})", abs(r["sigma_peak"] - 1.0) < 0.025,
                abs(r["sigma_peak"] - 1.0), 0.025, "peak stress / f_t - 1 (f_t = 1; one column 2% weaker)")
            chk(f"PF-CZM {soft} bar dissipated = G_f A (b={b})", r["rel_err_dissipated_vs_Gf"] < 0.08,
                r["rel_err_dissipated_vs_Gf"], 0.08, "includes diffuse pre-localisation damage")
            chk(f"PF-CZM {soft} bar softening law (b={b})", r["tsl_max_err_over_ft"] < 0.08,
                r["tsl_max_err_over_ft"], 0.08, "max |sigma - target(w)| / f_t on the softening branch")
            chk(f"PF-CZM {soft} bar energy balance (b={b})", r["energy_balance_rel_err"] < 0.01,
                r["energy_balance_rel_err"], 0.01, "|W_ext - (E_el + Gamma)| / W_ext")
    hs = (0.5, 0.25) if prod else (0.5,)
    conv = []
    for h in hs:
        r = verify_surfing("at2", h=h, length=1.0, Lx=30, Ly=12, a0=6, advance=12, dt=0.5)
        rr = {k: v for k, v in r.items() if not isinstance(v, (list, np.ndarray, Grid))}
        out["cases"].append(rr)
        conv.append(rr)
        chk(f"AT2 surfing SEN crack path straight (h/l={h})", r["max_path_deviation"] <= h + 1e-9,
            r["max_path_deviation"], h, "max |y_crack| (notch plane y=0)")
        chk(f"AT2 surfing energy balance (h/l={h})", r["energy_balance_rel_err"] < 0.02,
            r["energy_balance_rel_err"], 0.02, "|W_ext - (E_el + Gamma)| / W_ext at end")
        b_f = 1 + h / 2
        chk(f"AT2 surfing dGamma/da = G_c (1 + h/2l) (h/l={h})", abs(r["G_eff_measured"] / b_f - 1) < 0.05,
            abs(r["G_eff_measured"] / b_f - 1), 0.05, f"measured {r['G_eff_measured']:.4f}, Bourdin et al. 2008 {b_f:.4f}")
    out["at2_mesh_convergence"] = [{"h_over_l": c["h_over_length"], "G_eff_over_Gc": c["G_eff_measured"],
                                    "bourdin_1_plus_h_over_2l": c["bourdin_factor_1_plus_h_over_2l"]} for c in conv]
    if len(conv) >= 2:
        e = [c["G_eff_measured"] - 1.0 for c in conv]
        chk("AT2 G_eff -> G_c under mesh refinement (monotone)", all(e[i + 1] < e[i] for i in range(len(e) - 1)),
            e[-1], None, f"G_eff/G_c - 1 = {[round(x, 4) for x in e]}")
    for drv in (("principal_stress", "isotropic") if prod else ()):
        r = verify_surfing("pfczm", h=0.5, length=1.0, Lx=36, Ly=16, ft=0.5, a0=6, advance=14, dt=0.5,
                           driving=drv)
        rr = {k: v for k, v in r.items() if not isinstance(v, (list, np.ndarray, Grid))}
        rr["driving"] = drv
        out["cases"].append(rr)
        chk(f"PF-CZM surfing crack path straight ({drv})", r["max_path_deviation"] <= 0.5 + 1e-9,
            r["max_path_deviation"], 0.5, "max |y_crack|")
        chk(f"PF-CZM surfing dGamma/da = G_f ({drv})", r["rel_err_vs_discrete"] < 0.2, r["rel_err_vs_discrete"], 0.2,
            f"measured {r['G_eff_measured']:.4f} vs {r['G_eff_expected']:.4f}")
    out["all_passed"] = all(c["passed"] for c in out["checks"])
    return out


def main(argv=None):  # pragma: no cover - CLI
    import argparse
    import json

    import jax

    ap = argparse.ArgumentParser(description="jax_pfczm solver verification suite")
    ap.add_argument("--level", default="quick", choices=["quick", "production"])
    ap.add_argument("--out", default=None, help="optional JSON output path")
    a = ap.parse_args(argv)
    v = run_verification_suite(a.level)
    v["device"] = str(jax.devices()[0])
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(v, fh, indent=1, default=float)
    for c in v["checks"]:
        print(("PASS" if c["passed"] else "FAIL"), c["name"], "metric=%.4g" % c["metric"], "tol=", c["tolerance"])
    print("all_passed:", v["all_passed"])
    return 0 if v["all_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
