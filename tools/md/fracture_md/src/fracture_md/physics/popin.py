"""Hertzian pop-in analysis: experimental and MD spherical-indentation onsets on the same footing.

Why this module exists (docs/POPIN_VALIDATION.md)
--------------------------------------------------
Spherical-indentation pop-in is the one regime where an MD indentation and a laboratory test can be compared
directly: below the first pop-in both are elastic sphere contacts, and the first pop-in of a well-polished,
low-dislocation-density crystal marks the nucleation of plasticity (or of a crack) in a stressed volume that may
be defect-free.  Experiments report it as an *apparent* Hertzian maximum shear stress computed from the pop-in
load P, the tip radius R and the reduced modulus E_r with LINEAR isotropic Hertz theory.  To compare an MD onset
with that number the MD load must go through exactly the same formula; the true local stress at the onset is a
different quantity: for Vashishta 2007 the largest coarse-grained resolved shear just before yield is 0.5-0.6 of the
linear-Hertz value on (001) (twin sense, subsurface; gapfill G05) but 0.9-1.0 of it on (111) (anti-twin systems just
under the contact; docs/POPIN_VALIDATION.md section 7.5).  Apparent stresses compare with apparent stresses only, and
a laboratory apparent stress above the DFT ideal shear strength cannot be a stress the crystal sustained.

Formulas (rigid or elastic sphere on an isotropic half-space; Johnson, Contact Mechanics, 1985, section 4.2)
----------------------------------------------------------------------------------------------------------
* reduced modulus  1/E_r = (1 - nu_s^2)/E_s + (1 - nu_i^2)/E_i   (= 1/M_s + (1 - nu_i^2)/E_i with the sample's
  indentation modulus M_s = E_s/(1 - nu_s^2); a rigid tip: E_r = M_s);  diamond E_i = 1141 GPa, nu_i = 0.07.
* load-depth  P = (4/3) E_r R^(1/2) h^(3/2);  contact radius a = (3 P R / (4 E_r))^(1/3);  h = a^2 / R.
* pressures  p_m = P / (pi a^2),  p0 = 3 P / (2 pi a^2) = 1.5 p_m = (6 P E_r^2 / (pi^3 R^2))^(1/3) = (2 E_r/pi)(a/R).
* maximum shear stress (on the axis, depth z ~ 0.48 a)  tau_max = k(nu) p0, conventionally k = 0.31 (nu = 0.30);
  :func:`tau_max_over_p0` evaluates k(nu) exactly from the on-axis field
      sigma_z/p0 = -1/(1 + zeta^2),
      sigma_r/p0 = -(1 + nu)(1 - zeta atan(1/zeta)) + 1/(2 (1 + zeta^2)),   zeta = z/a,
      tau = (sigma_r - sigma_z)/2.
* E_r-free variant (uses the measured depth instead of the modulus; a^2 = R h):  p0 = 1.5 P/(pi R h), i.e.
  tau_max = 0.31 x 1.5 P/(pi R h) = 0.465 P/(pi R h) (the "0.47 P/(pi R h)" of Goel et al. 2014).

Units: lengths in angstrom, loads in nN, stresses and moduli in GPa (1 nN/A^2 = 100 GPa); experiment helpers take
mN and nm.  Nothing here is a measurement: values computed from literature inputs are labelled "recomputed".
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.optimize import minimize_scalar

GPA_PER_NN_PER_A2 = 100.0  # 1 nN / A^2 = 1e11 Pa
NN_PER_MN = 1.0e6
A_PER_NM = 10.0
DIAMOND_E_GPA = 1141.0  # diamond indenter, the nanoindentation convention (Oliver & Pharr 1992)
DIAMOND_NU = 0.07
TAU_MAX_OVER_P0_CONVENTIONAL = 0.31  # Johnson 1985 for nu = 0.30 (z = 0.48 a)
DATA_FILE = Path(__file__).resolve().parents[3] / "data" / "indentation" / "popin_experiments.json"


# ============================================================================================ moduli
def reduced_modulus_GPa(*, M_sample_GPa: Optional[float] = None, E_sample_GPa: Optional[float] = None,
                        nu_sample: Optional[float] = None, E_tip_GPa: float = DIAMOND_E_GPA,
                        nu_tip: float = DIAMOND_NU, rigid_tip: bool = False) -> float:
    """E_r from the sample's indentation modulus M_s (or E_s and nu_s) and the tip (diamond by default)."""
    if M_sample_GPa is None:
        if E_sample_GPa is None or nu_sample is None:
            raise ValueError("give M_sample_GPa, or E_sample_GPa and nu_sample")
        M_sample_GPa = float(E_sample_GPa) / (1.0 - float(nu_sample) ** 2)
    if rigid_tip:
        return float(M_sample_GPa)
    return 1.0 / (1.0 / float(M_sample_GPa) + (1.0 - nu_tip ** 2) / float(E_tip_GPa))


def vlassak_nix_modulus_cubic_GPa(c11: float, c12: float, c44: float,
                                  normal: Sequence[int] = (0, 0, 1)) -> float:
    """Vlassak-Nix indentation modulus M = 2 / <G33(phi)> of a cubic crystal indented along ``normal`` ((0, 0, 1) or
    (1, 1, 1)), from the Stroh half-space compliance of physics.contact_box (the project's M(001) = 331.8 GPa for
    Vashishta at 300 K and 426.2 GPa for the experimental 3C C_ij 395/123/236 come from the same code)."""
    from fracture_md.physics.contact_box import LayerMaterial, cubic_stiffness

    C = cubic_stiffness(float(c11), float(c12), float(c44))
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    if not np.allclose(n, [0.0, 0.0, 1.0]):
        trial = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(n, np.cross(trial, n))
        x /= np.linalg.norm(x)
        rows = np.array([x, np.cross(n, x), n])  # lab axes in crystal coordinates
        C = np.einsum("ia,jb,kc,ld,abcd->ijkl", rows, rows, rows, rows, C)
    return float(LayerMaterial(C).M_VN)


# ============================================================================================ Hertz field
def hertz_axis_stresses(zeta: Any, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """On-axis (sigma_r = sigma_theta, sigma_z) / p0 of the Hertz sphere at depth zeta = z / a (Johnson eq. 3.45)."""
    z = np.asarray(zeta, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        za = np.where(z > 0, z * np.arctan(1.0 / np.where(z > 0, z, 1.0)), 0.0)
    s_z = -1.0 / (1.0 + z ** 2)
    s_r = -(1.0 + nu) * (1.0 - za) + 0.5 / (1.0 + z ** 2)
    return s_r, s_z


def tau_max_over_p0(nu: float) -> tuple[float, float]:
    """(k, zeta*) with tau_max = k p0 the maximum principal shear stress on the axis and zeta* = z*/a its depth."""
    def neg_tau(zeta: float) -> float:
        s_r, s_z = hertz_axis_stresses(zeta, nu)
        return -0.5 * float(s_r - s_z)

    res = minimize_scalar(neg_tau, bounds=(1.0e-6, 3.0), method="bounded", options={"xatol": 1.0e-10})
    return float(-res.fun), float(res.x)


# ============================================================================================ contact
def hertz_from_load(P_nN: Any, R_A: float, Er_GPa: float) -> dict[str, Any]:
    """Contact radius a (A), depth h = a^2 / R (A), p_m and p0 (GPa) of a Hertz contact at load P."""
    P = np.asarray(P_nN, float)
    Er = float(Er_GPa) / GPA_PER_NN_PER_A2  # nN / A^2
    a = np.cbrt(3.0 * P * float(R_A) / (4.0 * Er))
    pm = GPA_PER_NN_PER_A2 * P / (math.pi * a ** 2)
    return {"a_A": a, "h_A": a ** 2 / float(R_A), "pm_GPa": pm, "p0_GPa": 1.5 * pm}


def hertz_from_depth(h_A: Any, R_A: float, Er_GPa: float) -> dict[str, Any]:
    """Load P (nN), contact radius a = sqrt(R h) (A), p_m and p0 (GPa) of a Hertz contact at depth h."""
    h = np.clip(np.asarray(h_A, float), 0.0, None)
    P = 4.0 / 3.0 * float(Er_GPa) / GPA_PER_NN_PER_A2 * math.sqrt(float(R_A)) * h ** 1.5
    a = np.sqrt(float(R_A) * h)
    p0 = 2.0 * float(Er_GPa) / math.pi * a / float(R_A)
    return {"P_nN": P, "a_A": a, "p0_GPa": p0, "pm_GPa": p0 / 1.5}


def p0_from_load_and_depth(P_nN: Any, R_A: float, h_A: Any) -> np.ndarray:
    """E_r-free Hertz p0 = 1.5 P / (pi R h) (GPa), with a^2 = R h from the measured depth."""
    return 1.5 * GPA_PER_NN_PER_A2 * np.asarray(P_nN, float) / (math.pi * float(R_A) * np.asarray(h_A, float))


def implied_reduced_modulus_GPa(P_nN: Any, R_A: float, h_A: Any) -> np.ndarray:
    """E_r that makes (P, h) a Hertz point for radius R: 3 P / (4 R^(1/2) h^(3/2))."""
    return GPA_PER_NN_PER_A2 * 3.0 * np.asarray(P_nN, float) / (4.0 * math.sqrt(R_A) * np.asarray(h_A, float) ** 1.5)


def implied_radius_A(P_nN: Any, h_A: Any, Er_GPa: float) -> np.ndarray:
    """R that makes (P, h) a Hertz point for E_r: (3 P / (4 E_r))^2 / h^3."""
    Er = float(Er_GPa) / GPA_PER_NN_PER_A2
    return (3.0 * np.asarray(P_nN, float) / (4.0 * Er)) ** 2 / np.asarray(h_A, float) ** 3


def popin_stresses(P_nN: Any, R_A: float, *, Er_GPa: Optional[float] = None, h_A: Any = None,
                   nu: Optional[float] = None) -> dict[str, Any]:
    """Apparent Hertz stresses at a pop-in load, every variant that the inputs allow.

    * ``from_load`` (needs E_r): a, h_Hertz, p_m, p0, tau_max = 0.31 p0 (and k(nu) p0 with ``nu``);
    * ``from_load_and_depth`` (needs the measured depth h): p0 = 1.5 P/(pi R h), tau = 0.31 p0, and the E_r that the
      (P, h) pair implies for this R (a check of R and E_r: it should be close to the stated E_r).
    Arrays broadcast (e.g. a [min, max] load range).
    """
    out: dict[str, Any] = {"R_A": float(R_A), "P_nN": np.asarray(P_nN, float),
                           "k_conventional": TAU_MAX_OVER_P0_CONVENTIONAL}
    k_nu = None
    if nu is not None:
        k_nu, z_nu = tau_max_over_p0(float(nu))
        out.update(nu=float(nu), k_nu=k_nu, zeta_at_tau_max=z_nu)
    if Er_GPa is not None:
        c = hertz_from_load(P_nN, R_A, Er_GPa)
        c["tau_max_GPa"] = TAU_MAX_OVER_P0_CONVENTIONAL * c["p0_GPa"]
        if k_nu is not None:
            c["tau_max_k_nu_GPa"] = k_nu * c["p0_GPa"]
        c["a_over_R"] = c["a_A"] / float(R_A)
        out["from_load"] = dict(c, Er_GPa=float(Er_GPa))
    if h_A is not None:
        p0 = p0_from_load_and_depth(P_nN, R_A, h_A)
        d = {"h_A": np.asarray(h_A, float), "p0_GPa": p0, "pm_GPa": p0 / 1.5,
             "tau_max_GPa": TAU_MAX_OVER_P0_CONVENTIONAL * p0,
             "implied_Er_GPa": implied_reduced_modulus_GPa(P_nN, R_A, h_A)}
        if k_nu is not None:
            d["tau_max_k_nu_GPa"] = k_nu * p0
        out["from_load_and_depth"] = d
    return out


def experiment_popin_stresses(P_mN: Any, R_nm: float, *, Er_GPa: Optional[float] = None, h_nm: Any = None,
                              nu: Optional[float] = None) -> dict[str, Any]:
    """:func:`popin_stresses` with laboratory units (mN, nm in; GPa, nm out)."""
    r = popin_stresses(np.asarray(P_mN, float) * NN_PER_MN, float(R_nm) * A_PER_NM, Er_GPa=Er_GPa,
                       h_A=None if h_nm is None else np.asarray(h_nm, float) * A_PER_NM, nu=nu)
    for key in ("from_load", "from_load_and_depth"):
        if key in r:
            for q in ("a_A", "h_A"):
                if q in r[key]:
                    r[key][q.replace("_A", "_nm")] = np.asarray(r[key].pop(q)) / A_PER_NM
    r["R_nm"] = float(R_nm)
    r["P_mN"] = np.asarray(P_mN, float)
    r.pop("R_A")
    r.pop("P_nN")
    return r


def json_ready(x: Any) -> Any:
    """numpy -> plain JSON types (rounded to 6 significant digits for floats)."""
    if isinstance(x, Mapping):
        return {k: json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_ready(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_ready(x.tolist()) if x.ndim else json_ready(x.item())
    if isinstance(x, (np.floating, float)):
        return float(f"{float(x):.6g}") if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


# ============================================================================================ data set
def load_experiments(path: Union[str, Path, None] = None) -> dict[str, Any]:
    """data/indentation/popin_experiments.json (curated literature pop-in data with recomputed Hertz stresses)."""
    return json.loads(Path(path or DATA_FILE).read_text())


def recompute_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the Hertz stresses of one data-set entry from its stated inputs (the ``inputs`` block)."""
    inp = entry["inputs"]
    if inp.get("tip_radius_nm") is None or inp.get("popin_load_mN") is None:
        return {}
    return json_ready(experiment_popin_stresses(inp["popin_load_mN"], inp["tip_radius_nm"],
                                                Er_GPa=inp.get("Er_GPa"), h_nm=inp.get("popin_depth_nm"),
                                                nu=inp.get("nu_sample")))


def included_tau_range_GPa(data: Optional[Mapping[str, Any]] = None, *, key: str = "from_load") -> tuple[float, float]:
    """(min, max) of the recomputed apparent tau_max = 0.31 p0 over the entries marked ``use: include``."""
    d = data or load_experiments()
    vals: list[float] = []
    for e in d["entries"]:
        if e.get("use") != "include":
            continue
        t = ((e.get("recomputed") or {}).get(key) or {}).get("tau_max_GPa")
        if t is None:
            continue
        vals += [float(v) for v in np.atleast_1d(t)]
    if not vals:
        raise ValueError(f"no included entry has recomputed.{key}.tau_max_GPa")
    return min(vals), max(vals)


def md_popin_summary(P_nN: float, R_A: float, h_A: float, *, M_GPa: Sequence[float], nu: Optional[float] = None,
                     ) -> dict[str, Any]:
    """The experiment-equivalent apparent stresses of an MD onset (load P at depth h, sphere R): linear Hertz from the
    load with each modulus in ``M_GPa`` (e.g. the potential's half-space M and the MD Hertz-fit M; a rigid tip, so
    E_r = M), the E_r-free p0 from (P, h), and the geometric mean pressure P / (pi (2 R h - h^2))."""
    out: dict[str, Any] = {"P_nN": float(P_nN), "R_A": float(R_A), "h_A": float(h_A), "h_over_R": float(h_A) / float(R_A)}
    out["from_load"] = {f"M={m:.1f}": json_ready(popin_stresses(P_nN, R_A, Er_GPa=m, nu=nu)["from_load"])
                        for m in M_GPa}
    out["from_load_and_depth"] = json_ready(popin_stresses(P_nN, R_A, h_A=h_A, nu=nu)["from_load_and_depth"])
    a2 = max(2.0 * R_A * h_A - h_A ** 2, 1.0e-12)
    out["pm_geometric_GPa"] = GPA_PER_NN_PER_A2 * float(P_nN) / (math.pi * a2)
    return out


# ============================================================================================ MD load curves
# A displacement-controlled MD sphere shows a pop-in as a LOAD DROP at fixed depth (a load-controlled laboratory
# indenter shows the same event as a depth excursion at fixed load).  The functions below turn the loading branch
# (h, P) of one run into: a windowed load, its noise sigma, a Hertz fit with a free depth-zero offset h0, the first
# load drop above max(rel_drop x running maximum, n_sigma x sigma), and the apparent Hertz stresses at that onset
# through the experimental formula (popin_stresses).  statistics_over_seeds summarises a set of runs.

def bin_load_curve(h_A: Any, P_nN: Any, dh_A: float, h_lo_A: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean depth and load in depth bins [h_lo + k dh, h_lo + (k+1) dh) (a fixed time window at constant speed: dh =
    speed x window).  Returns (h centres, mean loads, sample counts); empty bins are dropped."""
    h = np.asarray(h_A, float)
    P = np.asarray(P_nN, float)
    k = np.floor((h - h_lo_A) / float(dh_A)).astype(int)
    ok = k >= 0
    k, hh, PP = k[ok], h[ok], P[ok]
    if k.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    n = np.bincount(k)
    s = np.bincount(k, PP)
    sh = np.bincount(k, hh)
    nz = n > 0
    return sh[nz] / n[nz], s[nz] / n[nz], n[nz]


def local_quadratic_baseline(h_A: Any, P_nN: Any, half_width_A: float) -> np.ndarray:
    """At each sample, the value of a quadratic fitted to the samples within +-half_width (NaN with fewer than 5)."""
    h = np.asarray(h_A, float)
    P = np.asarray(P_nN, float)
    out = np.full_like(P, np.nan)
    for i, x in enumerate(h):
        m = np.abs(h - x) <= half_width_A
        if m.sum() >= 5:
            out[i] = np.polyfit(h[m] - x, P[m], 2)[-1]
    return out


def load_noise_sigma(h_A: Any, P_nN: Any, h_lo_A: float, h_hi_A: float, half_width_A: float = 0.75) -> float:
    """Robust load noise (nN): 1.4826 x the median absolute deviation of the residual from a local quadratic over
    h_lo <= h <= h_hi (NaN with fewer than 5 samples).  At 300 K it includes the slab-breathing ringing, which is
    deterministic and correlated between seeds."""
    h = np.asarray(h_A, float)
    P = np.asarray(P_nN, float)
    ref = local_quadratic_baseline(h, P, half_width_A)
    m = (h >= h_lo_A) & (h <= h_hi_A) & np.isfinite(ref)
    if m.sum() < 5:
        return float("nan")
    r = P[m] - ref[m]
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def hertz_fit_offset(h_A: Any, P_nN: Any, R_A: float, *, h_max_A: float, h_min_A: Optional[float] = None,
                     h0_bounds_A: tuple[float, float] = (-1.5, 1.5), M_GPa: Optional[float] = None) -> dict[str, Any]:
    """Least-squares Hertz fit with a free depth-zero offset h0 (the surface position is not known to better than the
    thermal roughness, and the tip-surface force law adds a constant shift):

        P = (4/3) M sqrt(R) (h - h0)^(3/2)      over h_min <= h <= h_max  ((h - h0) clipped at 0)

    Variable projection: at fixed h0 the best M is linear, so the residual is a 1-D function of h0, scanned on a
    0.01 A grid over ``h0_bounds_A`` and refined by a bounded Brent search.  With ``M_GPa`` given only h0 is fitted.
    Returns M_GPa (for a rigid tip, the reduced modulus E_r), h0_A, rms_nN, n, the 1-sigma standard errors M_se_GPa
    and h0_se_A from the Gauss-Newton covariance (residual variance x (J^T J)^-1), and ``h0_at_bound``."""
    h = np.asarray(h_A, float)
    P = np.asarray(P_nN, float)
    m = h <= float(h_max_A) + 1e-12
    if h_min_A is not None:
        m &= h >= float(h_min_A) - 1e-12
    hh, PP = h[m], P[m]
    lo, hi = (float(v) for v in h0_bounds_A)
    base = {"n": int(hh.size), "h_max_A": float(h_max_A), "h_min_A": None if h_min_A is None else float(h_min_A),
            "R_A": float(R_A), "model": "P = 4/3 M sqrt(R) (h - h0)^1.5, h0 free", "h0_bounds_A": [lo, hi]}
    if hh.size < 3:
        return {**base, "error": "fewer than 3 samples in the fit window"}
    k = 4.0 / 3.0 * math.sqrt(float(R_A)) / GPA_PER_NN_PER_A2  # P = k M x

    def sse(h0: float) -> tuple[float, float]:
        x = np.clip(hh - h0, 0.0, None) ** 1.5
        xx = float(x @ x)
        if xx <= 0.0:
            return math.inf, float("nan")
        c = float(x @ PP) / xx if M_GPa is None else k * float(M_GPa)
        r = PP - c * x
        return float(r @ r), c

    grid = np.arange(lo, hi + 1e-9, 0.01)
    s = np.array([sse(g)[0] for g in grid])
    if not np.isfinite(s).any():
        return {**base, "error": "no samples above any h0 in the bounds"}
    i = int(np.nanargmin(np.where(np.isfinite(s), s, np.nan)))
    res = minimize_scalar(lambda g: sse(g)[0], bounds=(float(grid[max(i - 1, 0)]), float(grid[min(i + 1, len(grid) - 1)])),
                          method="bounded", options={"xatol": 1.0e-7})
    h0 = float(res.x) if res.fun <= s[i] else float(grid[i])
    S, c = sse(h0)
    M = c / k
    d = np.clip(hh - h0, 0.0, None)
    cols = [-1.5 * k * M * d ** 0.5] if M_GPa is not None else [k * d ** 1.5, -1.5 * k * M * d ** 0.5]
    J = np.column_stack(cols)
    dof = max(hh.size - J.shape[1], 1)
    try:
        cov = (S / dof) * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(J.shape[1], np.nan)
    M_se = float("nan") if M_GPa is not None else float(se[0])
    h0_se = float(se[-1])
    return {**base, "M_GPa": float(M), "h0_A": h0, "rms_nN": math.sqrt(S / hh.size), "M_se_GPa": M_se,
            "h0_se_A": h0_se, "M_fixed": M_GPa is not None,
            "h0_at_bound": bool(min(abs(h0 - lo), abs(h0 - hi)) < 0.011)}


def detect_load_drops(h_A: Any, P_nN: Any, *, noise_nN: float = 0.0, rel_drop: float = 0.02, n_sigma: float = 3.0,
                      min_peak_nN: float = 20.0) -> list[dict[str, float]]:
    """Pop-in candidates on a displacement-controlled loading branch (use the windowed load).

    A sample is in a drop when the load lies below its running maximum by more than the threshold
    max(rel_drop x running maximum, n_sigma x noise_nN) and the running maximum exceeds ``min_peak_nN``.
    Consecutive samples form one event, reported with the peak before it (``h_peak_A``, ``P_peak_nN``: the pop-in
    depth and load), the first sample past the threshold (``h_drop_A``), the largest drop (``max_drop_nN`` and
    ``max_drop_frac`` of the running maximum), the threshold at the first sample and the depth span of the event."""
    h = np.asarray(h_A, float)
    P = np.asarray(P_nN, float)
    if h.size == 0:
        return []
    runmax = np.maximum.accumulate(P)
    drop = runmax - P
    thr = np.maximum(float(rel_drop) * runmax, float(n_sigma) * float(noise_nN))
    cand = np.where((drop > thr) & (runmax > float(min_peak_nN)))[0]
    out: list[dict[str, float]] = []
    if cand.size:
        for e in np.split(cand, np.where(np.diff(cand) > 1)[0] + 1):
            ipk = int(np.argmax(P[: e[0] + 1]))
            out.append({"h_peak_A": float(h[ipk]), "P_peak_nN": float(P[ipk]), "h_drop_A": float(h[e[0]]),
                        "h_end_A": float(h[e[-1]]), "n_samples": int(e.size),
                        "max_drop_nN": float(drop[e].max()), "max_drop_frac": float((drop[e] / runmax[e]).max()),
                        "threshold_nN": float(thr[e[0]]),
                        "threshold_rule": "relative" if rel_drop * runmax[e[0]] >= n_sigma * noise_nN else "noise"})
    return out


def popin_onset(P_nN: float, h_A: float, R_A: float, *, Er_GPa: float, h0_A: float = 0.0,
                nu: Optional[float] = None) -> dict[str, Any]:
    """Apparent Hertz stresses at one MD onset (load P at nominal depth h under a sphere of radius R), computed as an
    experimentalist computes them from a pop-in load: a = (3 P R / 4 E_r)^(1/3), p0 = 3 P / (2 pi a^2),
    tau_max = 0.31 p0 (and k(nu) p0 with ``nu``); a rigid MD tip has E_r = the crystal's indentation modulus.  Also the
    E_r-free variant p0 = 1.5 P / (pi R (h - h0)) with the fitted depth zero h0, and the geometric mean pressure
    P / (pi (2 R (h - h0) - (h - h0)^2))."""
    he = max(float(h_A) - float(h0_A), 1.0e-9)
    r = popin_stresses(P_nN, R_A, Er_GPa=Er_GPa, h_A=he, nu=nu)
    fl, fd = r["from_load"], r["from_load_and_depth"]
    out = {"P_nN": float(P_nN), "h_A": float(h_A), "h0_A": float(h0_A), "h_eff_A": he, "R_A": float(R_A),
           "h_over_R": float(h_A) / float(R_A), "Er_GPa": float(Er_GPa), "a_A": float(fl["a_A"]),
           "a_over_R": float(fl["a_over_R"]), "h_Hertz_A": float(fl["h_A"]), "pm_GPa": float(fl["pm_GPa"]),
           "p0_GPa": float(fl["p0_GPa"]), "tau_max_GPa": float(fl["tau_max_GPa"]),
           "p0_Er_free_GPa": float(fd["p0_GPa"]), "tau_max_Er_free_GPa": float(fd["tau_max_GPa"]),
           "Er_implied_by_P_h_GPa": float(fd["implied_Er_GPa"]),
           "pm_geometric_GPa": GPA_PER_NN_PER_A2 * float(P_nN) / (math.pi * max(2.0 * R_A * he - he ** 2, 1e-12))}
    if nu is not None:
        out["nu"] = float(nu)
        out["tau_max_k_nu_GPa"] = float(fl["tau_max_k_nu_GPa"])
        out["k_nu"] = float(r["k_nu"])
    return out


def analyze_load_curve(h_A: Any, P_nN: Any, R_A: float, *, dh_A: float, Er_GPa: Optional[float] = None,
                       nu: Optional[float] = None, fit_h_max_A: float = 3.0, rel_drop: float = 0.02,
                       n_sigma: float = 3.0, min_peak_nN: float = 20.0,
                       noise_window_A: Optional[tuple[float, float]] = None) -> dict[str, Any]:
    """The pop-in analysis of one loading branch (depth h in A increasing, load P in nN on the sphere).

    1. windowed load: :func:`bin_load_curve` with bin ``dh_A`` (speed x time window);
    2. noise: :func:`load_noise_sigma` over ``noise_window_A`` (default 1 A to max(0.2 R, 2 A); widened to
       0.5 A-max(0.3 R, 3 A) when that has fewer than 5 bins);
    3. elastic part: :func:`hertz_fit_offset` of the raw samples over h <= ``fit_h_max_A`` (M = E_r, h0);
    4. pop-in: the first event of :func:`detect_load_drops` on the windowed load;
    5. stresses at the pop-in (:func:`popin_onset`) with the fitted E_r (as an experimentalist uses the measured E_r)
       and, when given, with ``Er_GPa`` (e.g. the potential's own half-space modulus).
    Without a drop, ``popin`` is None and ``no_popin_bound`` gives the same stresses at the deepest windowed point
    (the onset lies beyond it)."""
    h = np.asarray(h_A, float)
    P = np.asarray(P_nN, float)
    hb, Pb, nb = bin_load_curve(h, P, dh_A)
    lo, hi = noise_window_A or (1.0, max(0.2 * R_A, 2.0))
    sig = load_noise_sigma(hb, Pb, lo, hi)
    noise_window = [lo, hi]
    if not math.isfinite(sig) and noise_window_A is None:
        noise_window = [0.5, max(0.3 * R_A, 3.0)]
        sig = load_noise_sigma(hb, Pb, *noise_window)
    fit = hertz_fit_offset(h, P, R_A, h_max_A=fit_h_max_A)
    ev = detect_load_drops(hb, Pb, noise_nN=sig if math.isfinite(sig) else 0.0, rel_drop=rel_drop, n_sigma=n_sigma,
                           min_peak_nN=min_peak_nN)
    Er_fit = fit.get("M_GPa")
    h0 = fit.get("h0_A", 0.0) if Er_fit is not None else 0.0
    moduli = {k: v for k, v in (("fit", Er_fit), ("given", Er_GPa)) if v is not None}

    def onset(Pc: float, hc: float) -> dict[str, Any]:
        return {k: popin_onset(Pc, hc, R_A, Er_GPa=v, h0_A=h0, nu=nu) for k, v in moduli.items()}

    out: dict[str, Any] = {
        "R_A": float(R_A), "dh_A": float(dh_A), "n_samples": int(h.size), "n_bins": int(hb.size),
        "h_max_A": float(h.max()) if h.size else None, "noise_sigma_nN": sig, "noise_window_A": noise_window,
        "drop_rule": {"rel_drop": rel_drop, "n_sigma": n_sigma, "min_peak_nN": min_peak_nN},
        "hertz_fit": fit, "Er_fit_GPa": Er_fit, "Er_given_GPa": Er_GPa, "load_drop_events": ev[:5],
        "n_load_drop_events": len(ev), "popin": None, "no_popin_bound": None}
    if ev:
        e = ev[0]
        out["popin"] = {"event": e, "P_c_nN": e["P_peak_nN"], "h_c_A": e["h_peak_A"],
                        "stresses": onset(e["P_peak_nN"], e["h_peak_A"])}
    elif hb.size:
        out["no_popin_bound"] = {"P_nN": float(Pb[-1]), "h_A": float(hb[-1]),
                                 "stresses": onset(float(Pb[-1]), float(hb[-1])),
                                 "note": "no load drop up to the deepest windowed point: the pop-in lies beyond it"}
    return out


def statistics_over_seeds(values: Sequence[Optional[float]], *, confidence: float = 0.95) -> dict[str, Any]:
    """Summary of one quantity over seeds (None / NaN entries are counted as missing): n, mean, sample standard
    deviation, standard error, Student-t confidence interval of the mean, median, min, max."""
    from scipy.stats import t as student_t

    v = np.array([float(x) for x in values if x is not None and math.isfinite(float(x))])
    out: dict[str, Any] = {"n": int(v.size), "n_missing": int(len(values) - v.size)}
    if v.size == 0:
        return out
    out.update(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
    if v.size >= 2:
        sd = float(v.std(ddof=1))
        sem = sd / math.sqrt(v.size)
        q = float(student_t.ppf(0.5 + confidence / 2.0, v.size - 1))
        out.update(std=sd, sem=sem, ci=[float(v.mean() - q * sem), float(v.mean() + q * sem)], confidence=confidence,
                   cv=sd / abs(float(v.mean())) if v.mean() != 0 else None)
    return out


__all__ = [
    "A_PER_NM", "DATA_FILE", "DIAMOND_E_GPA", "DIAMOND_NU", "GPA_PER_NN_PER_A2", "NN_PER_MN",
    "TAU_MAX_OVER_P0_CONVENTIONAL", "analyze_load_curve", "bin_load_curve", "detect_load_drops",
    "experiment_popin_stresses", "hertz_axis_stresses", "hertz_fit_offset", "hertz_from_depth", "hertz_from_load",
    "implied_radius_A", "implied_reduced_modulus_GPa", "included_tau_range_GPa", "json_ready", "load_experiments",
    "load_noise_sigma", "local_quadratic_baseline", "md_popin_summary", "p0_from_load_and_depth", "popin_onset",
    "popin_stresses", "recompute_entry", "reduced_modulus_GPa", "statistics_over_seeds", "tau_max_over_p0",
    "vlassak_nix_modulus_cubic_GPa",
]
