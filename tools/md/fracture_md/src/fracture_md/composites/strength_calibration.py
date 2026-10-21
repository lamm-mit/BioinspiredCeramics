"""Iteration 2: calibration of the absolute flexural-strength model and toughening levers.

The design layer's strength model (toughening.predict_flexural_strength) is Griffith,
sigma_f = K_model / (Y sqrt(pi a)), with an ASSUMED flaw depth a = 25 um [15-50].  It
over-predicts printed resins (214 MPa for the Midas proxy vs 120-150 MPa in the 510(k)).

Here the single free parameter a_eff (an *effective* flaw depth that also absorbs the
offset of the analytic K model) is fitted per material class to published 3-point flexural
strengths, using the SAME K model the design layer uses (analytic median of
toughening.predict_toughness, strong interface, Ba-glass filler at the material's filler
fraction).  Prediction interval: log-normal, Student t on the class scatter of log a_eff.

Also: short-fibre crack-bridging (frictional pull-out, L < l_c) toughening, Kitano fibre
maximum packing for the paste viscosity, Halpin-Tsai / 2D-random fibre modulus.

Every output is a DERIVED_MODEL estimate.  No value here is a measurement of our material.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from fracture_md.composites import toughening as TG

# ---------------------------------------------------------------------------
# Density assumptions for wt% -> vol% (stated ranges)
# ---------------------------------------------------------------------------
RHO_FILLER = (2.5, 2.65, 2.8)  # g/cm^3: silanised dental glass (Ba glass 2.80 [brief T12]; silica ~2.2-2.6) - ASSUMED range
RHO_RESIN = (1.15, 1.20, 1.25)  # g/cm^3 cured dimethacrylate - ASSUMED (brief uses 1.2)


def wt_to_vol(w: float, rho_f: float = RHO_FILLER[1], rho_r: float = RHO_RESIN[1]) -> float:
    return (w / rho_f) / (w / rho_f + (1 - w) / rho_r)


def wt_to_vol_range(w: float) -> tuple[float, float, float]:
    """(low, central, high) vol fraction for a filler weight fraction over the density ranges."""
    lo = wt_to_vol(w, RHO_FILLER[2], RHO_RESIN[0])
    hi = wt_to_vol(w, RHO_FILLER[0], RHO_RESIN[2])
    return lo, wt_to_vol(w), hi


def density_to_vol(rho_c: float) -> float:
    return (rho_c - RHO_RESIN[1]) / (RHO_FILLER[1] - RHO_RESIN[1])


# ---------------------------------------------------------------------------
# Calibration dataset (every entry cited; brief = docs/dental/RESEARCH_BRIEF.md)
# ---------------------------------------------------------------------------
@dataclass
class StrengthDatum:
    id: str
    product: str
    klass: str  # printed_commercial | printed_enhanced_silanisation | milled_or_bulk
    sigma_mpa: float
    sigma_range: tuple[float, float]
    test: str
    conditioning: str
    phi: float
    phi_range: tuple[float, float]
    phi_basis: str
    source: str
    k_ic_measured: Optional[float] = None
    k_ic_source: str = ""
    use_in_fit: bool = True
    notes: str = ""


def _ash(w):
    lo, c, hi = wt_to_vol_range(w)
    return c, (lo, hi)


def calibration_dataset() -> list[StrengthDatum]:
    ash33 = _ash(0.333)
    cc_phi = (density_to_vol(1.65), (density_to_vol(1.60) * 0.95, density_to_vol(1.70) * 1.05))
    midas = _ash(0.77)
    cera = _ash(0.77)
    return [
        StrengthDatum("crowntec_tayeb2025", "Saremco Crowntec", "printed_commercial", 140.9, (140.9, 140.9), "ISO 4049 3-point",
                      "ISO 4049 (24 h water 37 C)", _ash(0.332)[0], _ash(0.332)[1], "ash 33.2 wt% (Tayeb 2025) -> vol, assumed densities",
                      "Tayeb 2025 J Funct Biomater 16:468 doi:10.3390/jfb16120468 [brief S15]"),
        StrengthDatum("varseo_cp_tayeb2025", "BEGO VarseoSmile Crown plus", "printed_commercial", 128.0, (128.0, 128.0), "ISO 4049 3-point",
                      "ISO 4049 (24 h water 37 C)", ash33[0], ash33[1], "ash 33.3 wt% -> vol", "Tayeb 2025 [brief S15]"),
        StrengthDatum("formlabs_pcr_tayeb2025", "Formlabs Permanent Crown", "printed_commercial", 143.6, (143.6, 143.6), "ISO 4049 3-point",
                      "ISO 4049 (24 h water 37 C)", _ash(0.334)[0], _ash(0.334)[1], "ash 33.4 wt% -> vol", "Tayeb 2025 [brief S15]"),
        StrengthDatum("crowntec_salonen2025", "Saremco Crowntec", "printed_commercial", 102.0, (102.0, 102.0), "ISO 4049 3-point",
                      "ISO 4049 (24 h water 37 C)", _ash(0.332)[0], _ash(0.332)[1], "same product ash (Tayeb 2025)",
                      "Salonen 2025 Biomater Investig Dent 12:43584 doi:10.2340/biid.v12.43584 [brief S16b]",
                      k_ic_measured=0.9, k_ic_source="SENB, dry, 0 deg print (Salonen 2025)"),
        StrengthDatum("ceramic_crown_chen2026", "SprintRay Ceramic Crown", "printed_commercial", 130.7, (127.06, 134.26),
                      "3-point 1.2x4x14 mm, 12 mm span (90/0 deg build)", "24 h water 37 C", cc_phi[0], cc_phi[1],
                      "density 1.6-1.7 g/cm^3 (Mfr, brief S7) -> vol, assumed phase densities (~30 vol%; >50 wt% Mfr)",
                      "Chen 2026 J Prosthodont Res doi:10.2186/jpr.JPR_D_25_00232 [brief S12]"),
        StrengthDatum("ceramic_crown_hoffmann2025_24h", "SprintRay Ceramic Crown (printed on Midas)", "printed_commercial", 97.4, (97.4, 97.4),
                      "3-point 1.2x4x12 mm, 10 mm span", "24 h water (130 initial; 84.5 after 10k TC)", cc_phi[0], cc_phi[1],
                      "as above", "Hoffmann 2025 Materials 18:4662 doi:10.3390/ma18204662 [brief C12]"),
        StrengthDatum("midas_restore_510k", "SprintRay Midas Restore", "printed_commercial", 135.0, (120.0, 150.0), "ISO 10477 3-point",
                      "per ISO 10477 (510(k) bench test; conditioning not stated in the summary)", midas[0], midas[1],
                      ">75 wt% glass-ceramic + zirconia (Mfr) taken as 77 [75-80] wt% -> vol; zirconia share unknown (would lower vol%)",
                      "FDA K260661 510(k) summary https://www.accessdata.fda.gov/cdrh_docs/pdf26/K260661.pdf"),
        StrengthDatum("magnus_sanchez2026", "experimental 'Magnus' printable composite", "printed_enhanced_silanisation", 209.7, (209.7, 209.7),
                      "3-point", "wet, 36 C", 0.53, (0.52, 0.54), "reported ~70 wt% / 53 vol%",
                      "Sanchez 2026 Dent Mater doi:10.1016/j.dental.2026.06.014 [brief S17]", use_in_fit=False,
                      notes="single point: its own class (novel silanisation); not pooled with commercial printed resins"),
        StrengthDatum("lava_ultimate_tpp", "3M Lava Ultimate (milled)", "milled_or_bulk", 204.0, (185.0, 223.0), "ISO 6872 modified 3-point",
                      "manufacturer TPP (conditioning not stated)", 0.65, (0.64, 0.66), "reported 80 wt% / 65 vol%",
                      "3M Lava Ultimate Technical Product Profile [brief C14]", k_ic_measured=2.02, k_ic_source="V-notch (TPP)"),
        StrengthDatum("filtek_supreme_ultra_tpp", "3M Filtek Supreme Ultra (direct, lab-cured bars)", "milled_or_bulk", 165.1, (151.5, 178.7),
                      "internal 3-point", "manufacturer (not stated)", 0.633, (0.62, 0.64), "reported 78.5 wt% / 63.3 vol%",
                      "3M Filtek Supreme Ultra TPP [brief C17]", k_ic_measured=1.84, k_ic_source="notched short-rod (TPP)"),
        StrengthDatum("cerasmart270", "GC Cerasmart 270 (milled)", "milled_or_bulk", 243.0, (202.0, 246.0), "JIS T 6526 3-point",
                      "dry (202 after 7 d water)", cera[0], cera[1], "77 wt% (Mfr) -> vol, assumed densities",
                      "GC Cerasmart 270 documentation [brief C16]"),
    ]


EXCLUDED = [
    {"product": "Grandio blocs (244.5 ISO 4049, Tayeb 2025; 219.3 wet, Sanchez 2026)", "reason": "filler vol% not in the verified brief"},
    {"product": "Brilliant Crios (225.5; 151 after 24 h water)", "reason": "filler vol% not in the verified brief"},
    {"product": "Shofu Block HC (155)", "reason": "filler vol% not in the verified brief"},
    {"product": "Rodin Sculpture 2.0 (160-195 Mfr)", "reason": "filler fraction not published"},
    {"product": "VarseoSmile TriniQ (120 Mfr)", "reason": "30-50 wt% only; duplicate chemistry of Crown plus"},
]

# 7 d water / 24 h ratio for the >= 160 MPa-after-7-d target: Chen 2026 (90 deg 108.32/127.06 = 0.85,
# 0 deg 142.39/134.26 = 1.06) and Cerasmart 270 (202/243 = 0.83, dry -> 7 d).  Central 0.91 [0.83-1.06].
AGEING_7D_FACTOR = (0.91, 0.83, 1.06)


def k_model(phi: float, d50_um: float = 0.7) -> float:
    """K_Ic of the design layer's analytic model (median of analytic models, strong interface, glass filler)."""
    spec = {"fillers": [{"kind": "glass_filler", "volume_fraction": phi, "d50_um": d50_um}], "interface": "strong"}
    return float(TG.predict_toughness(spec, use_phase_field=False)["value"])


def a_eff_for(d: StrengthDatum, K: Optional[float] = None) -> float:
    K = k_model(d.phi) if K is None else K
    return TG.critical_flaw_size(K, d.sigma_mpa)


@dataclass
class ClassFit:
    klass: str
    n: int
    a_eff_um: float  # geometric mean
    log_a_sd: float
    sigma_log_pred: float  # sd of log(strength) for a new material incl. parameter uncertainty
    t_95: float
    members: list[dict] = field(default_factory=list)
    empirical_mean_mpa: float = 0.0
    empirical_log_sd: float = 0.0
    slope_mpa_per_phi: float = float("nan")
    notes: str = ""


_T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812}  # one-sided 95 %


def fit_classes(data: Optional[list[StrengthDatum]] = None) -> dict[str, ClassFit]:
    data = data or calibration_dataset()
    out = {}
    for kl in sorted({d.klass for d in data}):
        mem = [d for d in data if d.klass == kl and (d.use_in_fit or kl == "printed_enhanced_silanisation")]
        la = np.array([math.log(a_eff_for(d)) for d in mem])
        n = len(mem)
        sd = float(np.std(la, ddof=1)) if n > 1 else float("nan")
        t = _T95.get(max(n - 1, 1), 1.645)
        s_pred = 0.5 * sd * math.sqrt(1 + 1 / n) if n > 1 else float("nan")
        ls = np.log([d.sigma_mpa for d in mem])
        phis = np.array([d.phi for d in mem])
        slope = float(np.polyfit(phis, [d.sigma_mpa for d in mem], 1)[0]) if n > 2 and np.ptp(phis) > 0.05 else float("nan")
        out[kl] = ClassFit(kl, n, float(math.exp(la.mean())), sd, s_pred, t,
                           [{"id": d.id, "phi": d.phi, "sigma": d.sigma_mpa, "K_model": k_model(d.phi), "a_eff_um": a_eff_for(d),
                             "K_measured": d.k_ic_measured,
                             "a_from_measured_K_um": TG.critical_flaw_size(d.k_ic_measured, d.sigma_mpa) if d.k_ic_measured else None}
                            for d in mem],
                           float(math.exp(ls.mean())), float(np.std(ls, ddof=1)) if n > 1 else float("nan"), slope)
    return out


def predict_strength(K: float, fit: ClassFit, *, sigma_log: Optional[float] = None, t: Optional[float] = None) -> dict:
    """Calibrated 3-point strength: median K/(Y sqrt(pi a_eff)), 90 % prediction interval (log-normal, t)."""
    s = fit.sigma_log_pred if sigma_log is None else sigma_log
    tt = fit.t_95 if t is None else t
    med = TG.griffith_strength(K, fit.a_eff_um)
    return {"value": med, "low": med * math.exp(-tt * s), "high": med * math.exp(tt * s), "sigma_log": s, "t": tt,
            "a_eff_um": fit.a_eff_um, "K_used": K, "class": fit.klass,
            "interval": "90 % two-sided prediction interval (one-sided t_0.95 each side), log-normal"}


def prob_exceeds(pred: dict, threshold: float, factor: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> float:
    """P(strength x factor > threshold) with log-normal strength and an independent log-uniform-ish factor
    (factor range treated as +-1.645 sd in log)."""
    from math import erf, log, sqrt

    mu = log(pred["value"] * factor[0])
    sf = (log(factor[2]) - log(factor[1])) / (2 * 1.645) if factor[2] > factor[1] else 0.0
    s = sqrt(pred["sigma_log"] ** 2 + sf**2)
    z = (mu - log(threshold)) / s
    return 0.5 * (1 + erf(z / sqrt(2)))


# ---------------------------------------------------------------------------
# Short-fibre levers
# ---------------------------------------------------------------------------
def critical_fibre_length(sigma_f_mpa: float, d_um: float, tau_mpa: float) -> float:
    """Kelly-Tyson critical length l_c = sigma_f d / (2 tau) [um]."""
    return sigma_f_mpa * d_um / (2 * tau_mpa)


def pullout_toughening(V_f: float, L_um: float, d_um: float, tau_mpa: float, *, sigma_f_mpa: float = 2000.0,
                       orientation_factor: float = 2 / math.pi) -> float:
    """Steady-state frictional pull-out work Delta G [J/m^2] (= MPa um).

    Derivation (Kelly 1970 / Hull & Clyne 1996 ch. 8 form): N = V_f/(pi r^2) fibres per unit area,
    work pi d tau l^2 / 2 per fibre with embedded length l.  L < l_c: l ~ U(0, L/2) -> <l^2> = L^2/12 ->
    Delta G = V_f tau L^2 / (6 d).  L > l_c: only fibres with l < l_c/2 pull out (fraction l_c/L) ->
    Delta G = V_f tau l_c^3 / (6 d L).  x orientation factor (2D random in-plane: 2/pi crossing density;
    snubbing and inclined pull-out neglected).
    """
    lc = critical_fibre_length(sigma_f_mpa, d_um, tau_mpa)
    if L_um <= lc:
        G = V_f * tau_mpa * L_um**2 / (6 * d_um)
    else:
        G = V_f * tau_mpa * lc**3 / (6 * d_um * L_um)
    return G * orientation_factor


def kitano_phi_max(aspect_ratio: float) -> float:
    """Kitano, Kataoka & Shirota, Rheol Acta 20 (1981) 207: phi_m = 0.54 - 0.0125 r (fibres, r = L/d)."""
    return max(0.54 - 0.0125 * aspect_ratio, 0.05)


def fibre_viscosity_factor(V_f_in_paste: float, aspect_ratio: float) -> float:
    """Kitano relative viscosity (1 - phi/phi_m)^-2 of fibres suspended in the particulate paste."""
    pm = kitano_phi_max(aspect_ratio)
    if V_f_in_paste >= pm:
        return float("inf")
    return (1 - V_f_in_paste / pm) ** -2


def halpin_tsai(Em: float, Ef: float, V_f: float, xi: float) -> float:
    eta = (Ef / Em - 1) / (Ef / Em + xi)
    return Em * (1 + xi * eta * V_f) / (1 - eta * V_f)


def fibre_modulus_2d_random(Em: float, Ef: float, V_f: float, aspect_ratio: float) -> dict:
    """In-plane quasi-isotropic modulus: E = 3/8 E_L + 5/8 E_T (Tsai & Pagano) with Halpin-Tsai
    E_L (xi = 2 L/d) and E_T (xi = 2).  Out-of-plane (build direction) ~ E_T."""
    EL = halpin_tsai(Em, Ef, V_f, 2 * aspect_ratio)
    ET = halpin_tsai(Em, Ef, V_f, 2.0)
    return {"E_inplane": 0.375 * EL + 0.625 * ET, "E_L": EL, "E_T": ET, "E_build_direction": ET,
            "anisotropy_inplane_over_build": (0.375 * EL + 0.625 * ET) / ET}


FIBRE_DEFAULT = {"d_um": 6.0, "L_um": 90.0, "E_GPa": 72.0, "sigma_f_MPa": (1500.0, 2000.0, 3000.0),
                 "tau_MPa": (5.0, 10.0, 20.0),
                 "notes": ("E-glass fibre E 72 GPa and strength 1.5-3.5 GPa (textbook ranges, not re-verified); frictional "
                           "sliding stress tau 5-20 MPa ASSUMED (no verified dental glass-fibre/dimethacrylate value here)")}


def loo_check(data: Optional[list[StrengthDatum]] = None, klass: str = "printed_commercial") -> list[dict]:
    """Leave-one-out: refit the class without each member and predict it (90 % PI coverage check)."""
    data = data or calibration_dataset()
    mem = [d for d in data if d.klass == klass and d.use_in_fit]
    out = []
    for d in mem:
        rest = [x for x in data if x is not d]
        f = fit_classes(rest)[klass]
        p = predict_strength(k_model(d.phi), f)
        out.append({"id": d.id, "measured": d.sigma_mpa, "predicted": p["value"], "low": p["low"], "high": p["high"],
                    "inside_90PI": bool(p["low"] <= d.sigma_mpa <= p["high"])})
    return out


# ---------------------------------------------------------------------------
# Composite with short fibres (two-scale: fibres in the particulate paste)
# ---------------------------------------------------------------------------
def fibre_composite_estimates(paste: dict, V_f: float, fibre: dict = FIBRE_DEFAULT, *, nu: float = 0.27) -> dict:
    """Fibre-level estimates on top of a particulate paste evaluation.

    paste: {"E": (v, lo, hi) GPa, "K": (v, lo, hi) MPa m^0.5, "eta": (v, lo, hi) Pa s, "sorption": (v, lo, hi)}
    with V_f the fibre volume fraction of the composite (fibres replace paste volume)."""
    d, L = fibre["d_um"], fibre["L_um"]
    ar = L / d
    out = {"V_f": V_f, "aspect_ratio": ar, "kitano_phi_max": kitano_phi_max(ar)}
    Eres = [fibre_modulus_2d_random(E, fibre["E_GPa"], V_f, ar) for E in paste["E"]]
    out["E_inplane"] = tuple(r["E_inplane"] for r in Eres)
    out["E_build_direction"] = tuple(r["E_build_direction"] for r in Eres)
    out["anisotropy"] = Eres[0]["anisotropy_inplane_over_build"]
    Ks = []
    for (K, E) in ((paste["K"][0], out["E_inplane"][0]), (paste["K"][1], out["E_inplane"][1]), (paste["K"][2], out["E_inplane"][2])):
        Gm = TG.g_from_k(K, paste["E"][0], nu)
        Ks.append((Gm, E))
    dG = {tau: pullout_toughening(V_f, L, d, tau, sigma_f_mpa=fibre["sigma_f_MPa"][1]) for tau in fibre["tau_MPa"]}
    out["delta_G_pullout_J_m2"] = dG
    out["l_c_um"] = {tau: critical_fibre_length(fibre["sigma_f_MPa"][1], d, tau) for tau in fibre["tau_MPa"]}
    # plateau toughness (fully developed bridging zone): G = (1 - V_f) G_paste + Delta G
    tl, tc, th = fibre["tau_MPa"]
    Kc = TG.k_from_g((1 - V_f) * Ks[0][0] + dG[tc], out["E_inplane"][0], nu)
    Klo = TG.k_from_g((1 - V_f) * Ks[1][0] + dG[tl], out["E_inplane"][1], nu)
    Khi = TG.k_from_g((1 - V_f) * Ks[2][0] + dG[th], out["E_inplane"][2], nu)
    out["K_plateau"] = (Kc, Klo, Khi)
    # initiation toughness (no bridging yet) = paste K scaled by stiffness
    out["K_initiation"] = tuple(TG.k_from_g((1 - V_f) * Ks[i][0], out["E_inplane"][i], nu) for i in range(3))
    out["bridging_zone_length_um"] = 0.5 * L
    phi_paste = V_f  # fibres suspended in the paste occupy V_f of the total volume
    fac = fibre_viscosity_factor(phi_paste, ar)
    out["kitano_viscosity_factor"] = fac
    out["eta"] = tuple(e * fac for e in paste["eta"])
    out["sorption"] = tuple(s * (1 - V_f) for s in paste["sorption"])
    out["notes"] = ["K_plateau needs ~L/2 of crack growth to develop (R-curve); strength-controlling flaws "
                    "(a_eff ~25-50 um) see only part of it -> strength gain bounded by [K_initiation, K_plateau]",
                    "fibres unbreakable (l_c = sigma_f d / 2 tau >> L for tau <= 20 MPa)",
                    "sorption: fibres assumed non-absorbing; sizing/interface water uptake not modelled",
                    fibre["notes"]]
    return out
