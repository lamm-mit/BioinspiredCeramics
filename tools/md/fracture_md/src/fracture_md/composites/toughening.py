"""Analytical toughness / strength models for particle-filled (dental) resins.

Every function states its assumptions.  Units: E in GPa, K in MPa*m^0.5,
G in J/m^2, lengths in um unless noted, stresses in MPa.

Models
------
* Plane-strain K <-> G:  G = K^2 (1 - nu^2) / E   (Irwin).
* Composite modulus: Mori-Tanaka for spheres (= Hashin-Shtrikman lower bound for
  stiff particles in a compliant matrix; Mori & Tanaka 1973, Hashin & Shtrikman 1963).
* Crack-path rule of mixtures (this work, stated assumption): a planar crack
  crossing a random section meets filler over an area fraction = phi (Delesse);
  over the filler it either cuts the particle (G_p) or goes around it along the
  interface (hemispherical debond, area ratio 2 -> 2 G_int), whichever is cheaper.
  G_path = (1 - phi) G_m + phi * min(G_p, 2 G_int).
* Crack deflection (Faber & Evans, Acta Metall 31 (1983) 565-576, spheres):
  G/G_0 ~= 1 + 0.87 phi as commonly linearised from their sphere curve; the
  linearisation is used only up to phi = 0.3 and held constant above (the
  deflection increment saturates at higher phi) -- screening estimate.
  Applied only when the crack goes around particles.
* Crack bowing/pinning (Lange, Phil Mag 22 (1970) 983; Evans, Phil Mag 26 (1972)
  1327; Green, J Am Ceram Soc 66 (1983) C-4): line-tension form
  G/G_m = 1 + T/(G_m lambda) with lambda the Fullman mean free path
  4 r (1 - phi)/(3 phi) and T = tau r G_m, tau in [0.25, 1] an *assumed*
  sensitivity range (not a value taken from those papers).  Only active for
  particles clearly tougher than the matrix (K_p > 1.5 K_m) with a strong interface.
* Griffith flaw-size strength: sigma_f = K_Ic / (Y sqrt(pi a)); Y = 0.713 (deepest
  point of a semicircular surface flaw of depth a, i.e. 1.12 * 2/pi) or 2/pi
  (embedded penny crack of radius a, exact).
* Weibull size scaling between the ISO 4049 3-point bar and the ISO 6872
  piston-on-three-balls disc: sigma_2/sigma_1 = (A_eff,1 / A_eff,2)^(1/m) (surface
  flaws; principle of independent action for the biaxial disc; thin-plate
  moments of a simply supported disc under a central uniform pressure disc, with
  the overhang correction consistent with the ISO 6872 formula).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

SQRT_UM = 1e-3  # sqrt(1 um) in sqrt(m)

# ---------------------------------------------------------------------------
# Constituent inputs (value, range, source).  Ranges are the uncertainty used.
# ---------------------------------------------------------------------------
# One matrix constituent set for the whole composites package: the cured-resin elastic constants are taken
# from homogenization.CONSTITUENTS["resin"] (E 2.5 [2.0-3.5] GPa, nu 0.35), so that the Mori-Tanaka/FFT
# moduli, the phase-field matrix and the K <-> G conversions here all use the same matrix.  (Before the
# final-solver pass this module used its own E 3.0 [2.5-4.0] GPa; see docs/dental/FRACTURE.md.)
from fracture_md.composites.homogenization import CONSTITUENTS as _HOMOG_CONSTITUENTS  # noqa: E402

_RESIN = _HOMOG_CONSTITUENTS["resin"]
MATRIX_E_GPA: float = float(_RESIN.E[1])
MATRIX_E_RANGE_GPA: tuple[float, float] = (float(_RESIN.E[0]), float(_RESIN.E[2]))
MATRIX_NU: float = float(_RESIN.nu[1])

CONSTITUENTS: dict[str, dict] = {
    "matrix_dimethacrylate": {
        "E_GPa": MATRIX_E_GPA, "E_range": MATRIX_E_RANGE_GPA, "nu": MATRIX_NU,
        "K_Ic": 0.9, "K_Ic_range": (0.7, 1.2), "ft_MPa": 50.0, "ft_range": (40.0, 80.0),
        "source": ("K_Ic range 0.7-1.2 MPa m^0.5 for cured Bis-GMA/UDMA/TEGDMA networks as specified in the "
                   "project brief (dental-fracture.md); cf. Ferracane, Antonio & Matsumoto, J Dent Res 66 (1987) "
                   "1140 (variables affecting K_Ic of dental composites) -- exact unfilled-resin values not "
                   "re-verified in this session.  E and nu are shared with homogenization.CONSTITUENTS['resin'] "
                   "(E 2.5 [2.0-3.5] GPa, nu 0.35); tensile strength 40-80 MPa and the elastic constants are "
                   "assumptions typical of cured dimethacrylates (not literature-verified here)."),
    },
    "glass_filler": {
        "E_GPa": 72.2, "E_range": (65.0, 80.0), "nu": 0.168,
        "K_Ic": 0.75, "K_Ic_range": (0.7, 0.8), "ft_MPa": 300.0, "ft_range": (150.0, 1000.0),
        "source": ("E 72.2 GPa, nu 0.168: amorphous silica, Pabst & Gregorova, Ceramics-Silikaty 57 (2013) 167 "
                   "[brief T14]; Ba-aluminoborosilicate dental-glass E not found (brief) -> range 65-80 GPa "
                   "assumed.  K_Ic 0.7-0.8 MPa m^0.5 typical of silicate glasses (e.g. Wiederhorn, J Am Ceram "
                   "Soc 52 (1969) 99, fracture surface energies of glasses).  Particle tensile strength "
                   "150-1000 MPa is an assumption (flaw-controlled micro-particles)."),
    },
    "zirconia_filler": {
        "E_GPa": 205.0, "E_range": (190.0, 215.0), "nu": 0.31,
        "K_Ic": 4.5, "K_Ic_range": (4.0, 5.0), "ft_MPa": 800.0, "ft_range": (500.0, 1200.0),
        "source": ("E 205 GPa, nu 0.31: t-ZrO2, Kisi & Howard, J Am Ceram Soc 81 (1998) 1682 [brief T15]. "
                   "K_Ic 4-5 MPa m^0.5 is the range for bulk 3Y-TZP (e.g. Kelly & Denry, Dent Mater 24 (2008) "
                   "289) and includes transformation toughening that is NOT expected in sub-micron filler "
                   "particles -> an upper bound for the particle.  Particle strength 500-1200 MPa assumed."),
    },
}

INTERFACES: dict[str, dict] = {
    # ratios relative to the matrix (f_t, G_c) of the interphase band
    "strong": {"ft_ratio": 1.0, "Gc_ratio": 1.0,
               "note": "well-silanised (gamma-MPS) glass: failure cohesive in the matrix/interphase (assumed)"},
    "intermediate": {"ft_ratio": 0.5, "Gc_ratio": 0.3, "note": "partially hydrolysed silane layer (assumed)"},
    "weak": {"ft_ratio": 0.2, "Gc_ratio": 0.1,
             "note": ("unsilanised or water-degraded interface; zirconia with silane only (needs MDP) "
                      "(assumed ratios; interfacial-strength literature, e.g. Debnath et al. Dent Mater 19 (2003) "
                      "441 microbond IFSS, supports strong silane dependence but values not re-verified)")},
}


# ---------------------------------------------------------------------------
# Elasticity helpers
# ---------------------------------------------------------------------------
def plane_strain_modulus(E: float, nu: float) -> float:
    return E / (1.0 - nu * nu)


def g_from_k(K_MPa_sqrt_m: float, E_GPa: float, nu: float) -> float:
    """Plane-strain G_c [J/m^2] from K_Ic [MPa m^0.5] and E [GPa]."""
    return (K_MPa_sqrt_m * 1e6) ** 2 / (plane_strain_modulus(E_GPa, nu) * 1e9)


def k_from_g(G_J_m2: float, E_GPa: float, nu: float) -> float:
    """Plane-strain K_Ic [MPa m^0.5] from G_c [J/m^2]."""
    return math.sqrt(G_J_m2 * plane_strain_modulus(E_GPa, nu) * 1e9) / 1e6


def k_ic_from_rg(R_G: float, E_c_GPa: float, nu_c: float, *, R_G_range: Optional[tuple[float, float]] = None,
                 E_c_range_GPa: Optional[tuple[float, float]] = None, K_m: Optional[float] = None) -> dict:
    """Composite K_Ic = sqrt(R_G G_m E'_c) (plane strain), G_m from the shared matrix constituent set.

    One E model is used for the value and both range ends: the caller passes the composite modulus
    (e.g. the FFT estimate) with its own [lo, hi]; the R_G and E ranges are combined as independent
    min/max envelopes (not a probability interval).
    """
    mc = CONSTITUENTS["matrix_dimethacrylate"]
    Km = mc["K_Ic"] if K_m is None else K_m
    Gm = g_from_k(Km, mc["E_GPa"], mc["nu"])
    val = k_from_g(R_G * Gm, E_c_GPa, nu_c)
    rr = R_G_range or (R_G, R_G)
    er = E_c_range_GPa or (E_c_GPa, E_c_GPa)
    lo = k_from_g(rr[0] * Gm, er[0], nu_c)
    hi = k_from_g(rr[1] * Gm, er[1], nu_c)
    return {"value": val, "low": lo, "high": hi, "unit": "MPa*m^0.5", "G_m_J_m2": Gm, "K_m": Km,
            "E_m_GPa": mc["E_GPa"], "nu_m": mc["nu"], "E_c_GPa": E_c_GPa, "E_c_range_GPa": er, "R_G_range": rr,
            "method": "K = sqrt(R_G * G_m * E_c/(1-nu_c^2)); G_m = K_m^2 (1-nu_m^2)/E_m (shared matrix set)"}


def mori_tanaka_spheres(Em: float, num: float, Ep: float, nup: float, phi: float) -> tuple[float, float]:
    """Mori-Tanaka effective (E, nu) of isotropic spheres in an isotropic matrix."""
    Km, Gm = Em / (3 * (1 - 2 * num)), Em / (2 * (1 + num))
    Kp, Gp = Ep / (3 * (1 - 2 * nup)), Ep / (2 * (1 + nup))
    K = Km + phi * (Kp - Km) / (1 + (1 - phi) * (Kp - Km) / (Km + 4 * Gm / 3))
    f = Gm * (9 * Km + 8 * Gm) / (6 * (Km + 2 * Gm))
    G = Gm + phi * (Gp - Gm) / (1 + (1 - phi) * (Gp - Gm) / (Gm + f))
    E = 9 * K * G / (3 * K + G)
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    return E, nu


def fullman_mean_free_path(r: float, phi: float) -> float:
    """Mean edge-to-edge free path between spheres of radius r (Fullman 1953): 4 r (1-phi) / (3 phi)."""
    return 4.0 * r * (1.0 - phi) / (3.0 * max(phi, 1e-12))


# ---------------------------------------------------------------------------
# Toughening models (return G_c of the composite in J/m^2)
# ---------------------------------------------------------------------------
def crack_path_rule_of_mixtures(phi: float, G_m: float, G_p: float, G_int: float) -> tuple[float, str]:
    """G_path = (1-phi) G_m + phi min(G_p, 2 G_int); returns (G, 'cut'|'around')."""
    around = 2.0 * G_int
    if G_p <= around:
        return (1 - phi) * G_m + phi * G_p, "cut"
    return (1 - phi) * G_m + phi * around, "around"


def faber_evans_spheres_ratio(phi: float) -> float:
    """G/G_0 for crack deflection by spheres, linearised 1 + 0.87 phi, phi capped at 0.3 (screening)."""
    return 1.0 + 0.87 * min(max(phi, 0.0), 0.3)


def lange_bowing_ratio(phi: float, r_um: float, tau: float = 0.5) -> float:
    """G/G_m = 1 + T/(G_m lambda) with T = tau r G_m -> 1 + tau r / lambda (tau assumed in [0.25, 1])."""
    if phi <= 0:
        return 1.0
    return 1.0 + tau * r_um / fullman_mean_free_path(r_um, phi)


# ---------------------------------------------------------------------------
# Strength models
# ---------------------------------------------------------------------------
Y_SEMICIRCULAR_SURFACE = 1.12 * 2.0 / math.pi  # 0.713 (sigma_f = K / (Y sqrt(pi a)))
Y_PENNY_EMBEDDED = 2.0 / math.pi


def griffith_strength(K_MPa_sqrt_m: float, a_um: float, Y: float = Y_SEMICIRCULAR_SURFACE) -> float:
    """sigma_f [MPa] = K_Ic / (Y sqrt(pi a)), a in um."""
    return K_MPa_sqrt_m / (Y * math.sqrt(math.pi * a_um * 1e-6))


def critical_flaw_size(K_MPa_sqrt_m: float, sigma_MPa: float, Y: float = Y_SEMICIRCULAR_SURFACE) -> float:
    """a_c [um] = (K / (Y sigma))^2 / pi."""
    return (K_MPa_sqrt_m / (Y * sigma_MPa)) ** 2 / math.pi * 1e6


def effective_area_3pt(span_mm: float = 20.0, width_mm: float = 2.0, thickness_mm: float = 2.0,
                       m: float = 10.0) -> float:
    """Weibull effective surface of a 3-point bar [mm^2]: (L/(m+1)) (b + t/(m+1))  (tension face + sides)."""
    return span_mm / (m + 1) * (width_mm + thickness_mm / (m + 1))


def effective_volume_3pt(span_mm: float = 20.0, width_mm: float = 2.0, thickness_mm: float = 2.0,
                         m: float = 10.0) -> float:
    """Weibull effective volume of a 3-point bar [mm^3]: L b t / (2 (m+1)^2)."""
    return span_mm * width_mm * thickness_mm / (2 * (m + 1) ** 2)


def _disc_moments(r, P, a, c, R, nu):
    """Radial/tangential moments of a disc on a support circle a, uniform pressure over r<=c, radius R."""
    k = P / (4 * math.pi)
    r = np.asarray(r, float)
    inner = r <= c
    Mr = np.where(inner, k * ((1 + nu) * math.log(a / c) + 1 - (1 - nu) * c * c / (4 * a * a) - (3 + nu) * r * r / (4 * c * c)),
                  k * ((1 + nu) * np.log(a / np.maximum(r, 1e-30)) + (1 - nu) * c * c / (4 * np.maximum(r, 1e-30) ** 2) * (1 - r * r / (a * a))))
    Mt = np.where(inner, k * ((1 + nu) * math.log(a / c) + 1 - (1 - nu) * c * c / (4 * a * a) - (1 + 3 * nu) * r * r / (4 * c * c)),
                  k * ((1 + nu) * np.log(a / np.maximum(r, 1e-30)) + (1 - nu) - (1 - nu) * c * c / (4 * np.maximum(r, 1e-30) ** 2) * (1 + r * r / (a * a))))
    # overhang (R > a): uniform edge-moment correction, consistent with the ISO 6872 centre-stress formula
    dM = k * (1 - nu) * (-0.5 + (a * a - 0.5 * c * c) / (2 * R * R) + c * c / (4 * a * a))
    return Mr + dM, Mt + dM


def iso6872_center_stress(P_N: float, t_mm: float, a_mm: float = 5.0, c_mm: float = 0.7, R_mm: float = 7.0,
                          nu: float = 0.3) -> float:
    """ISO 6872 piston-on-three-balls centre stress [MPa] = -0.2387 P (X - Y) / b^2."""
    X = (1 + nu) * math.log((c_mm / R_mm) ** 2) + (1 - nu) / 2 * (c_mm / R_mm) ** 2
    Yv = (1 + nu) * (1 + math.log((a_mm / R_mm) ** 2)) + (1 - nu) * (a_mm / R_mm) ** 2
    return -0.2387 * P_N * (X - Yv) / t_mm**2


def effective_area_biaxial(m: float = 10.0, a_mm: float = 5.0, c_mm: float = 0.7, R_mm: float = 7.0,
                           t_mm: float = 1.2, nu: float = 0.3, n: int = 4000) -> float:
    """Weibull effective tensile surface [mm^2] of the ISO 6872 disc (PIA, surface inside the support circle)."""
    r = np.linspace(0, a_mm, n)
    Mr, Mt = _disc_moments(r, 1.0, a_mm, c_mm, R_mm, nu)
    smax = max(Mr[0], Mt[0])
    integrand = (np.clip(Mr / smax, 0, None) ** m + np.clip(Mt / smax, 0, None) ** m) * 2 * math.pi * r
    return float(np.trapezoid(integrand, r))


def weibull_scale(sigma_1: float, A_1: float, A_2: float, m: float) -> float:
    """Mean strength in geometry 2 from geometry 1: sigma_2 = sigma_1 (A_1/A_2)^(1/m)."""
    return sigma_1 * (A_1 / A_2) ** (1.0 / m)


def biaxial_to_3pt(sigma_biaxial: float, m: float = 10.0, **kw) -> float:
    """Equivalent ISO 4049 3-point bar strength from an ISO 6872 biaxial strength (surface flaws)."""
    return weibull_scale(sigma_biaxial, effective_area_biaxial(m, **kw), effective_area_3pt(m=m), m)


def threept_to_biaxial(sigma_3pt: float, m: float = 10.0, **kw) -> float:
    return weibull_scale(sigma_3pt, effective_area_3pt(m=m), effective_area_biaxial(m, **kw), m)


# ---------------------------------------------------------------------------
# Design-agent interface
# ---------------------------------------------------------------------------
@dataclass
class FillerSpec:
    kind: Literal["glass_filler", "zirconia_filler"]
    volume_fraction: float
    d50_um: float = 1.0


@dataclass
class CompositeSpec:
    fillers: list[FillerSpec]
    interface: Literal["strong", "intermediate", "weak"] = "strong"
    matrix: str = "matrix_dimethacrylate"
    matrix_K_Ic: Optional[float] = None  # override central value
    flaw_size_um: tuple[float, float, float] = (15.0, 25.0, 50.0)  # (low, central, high) surface-flaw depth
    weibull_m: tuple[float, float, float] = (8.0, 10.0, 15.0)
    notes: str = ""

    @property
    def phi(self) -> float:
        return sum(f.volume_fraction for f in self.fillers)


def _as_spec(spec) -> CompositeSpec:
    if isinstance(spec, CompositeSpec):
        return spec
    d = dict(spec)
    fillers = [f if isinstance(f, FillerSpec) else FillerSpec(**f) for f in d.pop("fillers")]
    for k in ("flaw_size_um", "weibull_m"):
        if k in d:
            d[k] = tuple(d[k])
    return CompositeSpec(fillers=fillers, **d)


_PF_RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "dental", "fracture_results.json")


def load_phase_field_ratios(path: Optional[str] = None) -> dict:
    """Toughening ratios G_eff/G_eff,matrix from the phase-field study (if the results file exists)."""
    p = os.path.abspath(path or _PF_RESULTS)
    if not os.path.exists(p):
        return {}
    with open(p) as fh:
        data = json.load(fh)
    return {c["case"]: c for c in data.get("microstructure_study", {}).get("cases", []) if c.get("is_base_case", True)}


def _toughness_models(sp: CompositeSpec, Km: float, Em: float, num: float, tau: float, G_int_ratio: float):
    """Returns dict model -> (G_c J/m^2, E_c GPa, nu_c)."""
    Gm = g_from_k(Km, Em, num)
    phi = sp.phi
    # composite modulus: sequential Mori-Tanaka over filler kinds (volume-weighted particle properties)
    if phi > 0:
        Ep = sum(CONSTITUENTS[f.kind]["E_GPa"] * f.volume_fraction for f in sp.fillers) / phi
        nup = sum(CONSTITUENTS[f.kind]["nu"] * f.volume_fraction for f in sp.fillers) / phi
    else:
        Ep, nup = Em, num
    Ec, nuc = mori_tanaka_spheres(Em, num, Ep, nup, phi)
    # particle fracture energy (volume-weighted)
    Gp = sum(g_from_k(CONSTITUENTS[f.kind]["K_Ic"], CONSTITUENTS[f.kind]["E_GPa"], CONSTITUENTS[f.kind]["nu"])
             * f.volume_fraction for f in sp.fillers) / phi if phi > 0 else Gm
    G_int = G_int_ratio * Gm
    out = {}
    out["modulus_only"] = (Gm, Ec, nuc)
    Gpath, mode = crack_path_rule_of_mixtures(phi, Gm, Gp, G_int)
    out["path_rule_of_mixtures"] = (Gpath, Ec, nuc)
    fe = faber_evans_spheres_ratio(phi) if mode == "around" else 1.0
    out["path_rom_x_faber_evans"] = (Gpath * fe, Ec, nuc)
    bow = 1.0
    for f in sp.fillers:
        Kp = CONSTITUENTS[f.kind]["K_Ic"]
        if Kp > 1.5 * Km and sp.interface == "strong":
            bow += (lange_bowing_ratio(f.volume_fraction, f.d50_um / 2, tau) - 1.0)
    out["matrix_x_lange_bowing"] = (Gm * bow, Ec, nuc)
    return out, mode


def predict_toughness(spec, *, use_phase_field: bool = True, results_path: Optional[str] = None) -> dict:
    """Composite K_Ic estimate (MPa m^0.5) with range, method and assumptions.

    Central value: phase-field toughening ratio (if a matching case exists in
    data/dental/fracture_results.json) applied to the matrix G_c, converted with
    the Mori-Tanaka plane-strain modulus; otherwise the median of the analytic
    models.  Range: min/max over the analytic models x input ranges (matrix K_Ic,
    E, bowing coefficient tau).
    """
    sp = _as_spec(spec)
    mc = CONSTITUENTS[sp.matrix]
    Km0 = sp.matrix_K_Ic or mc["K_Ic"]
    iface = INTERFACES[sp.interface]
    central_models, mode = _toughness_models(sp, Km0, mc["E_GPa"], mc["nu"], 0.5, iface["Gc_ratio"])
    K_models = {k: k_from_g(G, E, nu) for k, (G, E, nu) in central_models.items()}
    samples = []
    for Km in (mc["K_Ic_range"][0], Km0, mc["K_Ic_range"][1]):
        for Em in (mc["E_range"][0], mc["E_GPa"], mc["E_range"][1]):
            for tau in (0.25, 1.0):
                mm, _ = _toughness_models(sp, Km, Em, mc["nu"], tau, iface["Gc_ratio"])
                samples += [k_from_g(G, E, nu) for (G, E, nu) in mm.values()]
    method = "median of analytic models"
    central = float(np.median(list(K_models.values())))
    pf_used = None
    if use_phase_field:
        pf = load_phase_field_ratios(results_path)
        key = _match_phase_field_case(sp, pf)
        if key is not None:
            R = float(pf[key]["toughening_ratio_G"])
            Gm = g_from_k(Km0, mc["E_GPa"], mc["nu"])
            _, Ec, nuc = central_models["modulus_only"]
            central = k_from_g(R * Gm, Ec, nuc)
            samples.append(central)
            af = pf[key].get("particle_area_fraction")
            method = (f"phase-field (PF-CZM) toughening ratio R_G={R:.3f} from case '{key}' (achieved particle area "
                      f"fraction {af if af is None else round(af, 3)}) x matrix G_c, K via Mori-Tanaka E' at phi={sp.phi:.2f}")
            pf_used = key
    lo, hi = float(min(samples)), float(max(samples))
    Ec = central_models["modulus_only"][1]
    return {
        "value": central, "range": (lo, hi), "unit": "MPa*m^0.5", "method": method,
        "models": K_models, "crack_path_mode_over_particles": mode, "phase_field_case": pf_used,
        "E_composite_GPa": Ec, "matrix_G_c_J_m2": g_from_k(Km0, mc["E_GPa"], mc["nu"]),
        "assumptions": [
            "plane-strain LEFM, K = sqrt(G E') with E' of the Mori-Tanaka composite",
            f"matrix K_Ic {mc['K_Ic_range'][0]}-{mc['K_Ic_range'][1]} MPa m^0.5 (central {Km0}), E {mc['E_range']} GPa",
            f"interface '{sp.interface}': G_int/G_m = {iface['Gc_ratio']}, f_t,int/f_t,m = {iface['ft_ratio']} ({iface['note']})",
            "crack-path rule of mixtures (Delesse area fraction = volume fraction; hemispherical debond area ratio 2)",
            "Faber-Evans sphere deflection linearised 1+0.87 phi, phi capped at 0.3; Lange bowing coefficient tau in [0.25,1] (assumed)",
            "no matrix plasticity / crazing, no R-curve, no water ageing, no transformation toughening of zirconia",
        ],
    }


def _match_phase_field_case(sp: CompositeSpec, pf: dict, tol: float = 0.1) -> Optional[str]:
    """Single-filler specs only: same filler and interface, achieved particle area fraction within tol of phi."""
    kinds = {f.kind.split("_")[0] for f in sp.fillers}
    if len(kinds) != 1:
        return None
    kind = kinds.pop()
    best, dbest = None, tol + 1e-12
    for key, c in pf.items():
        if c.get("toughening_ratio_G") is None or c.get("filler") != kind or c.get("interface") != sp.interface:
            continue
        af = c.get("particle_area_fraction", c.get("phi_target"))
        if af is None:
            continue
        if abs(af - sp.phi) <= dbest:
            best, dbest = key, abs(af - sp.phi)
    return best


def _pf_key(sp: CompositeSpec) -> str:
    kinds = "+".join(sorted({f.kind.split("_")[0] for f in sp.fillers}))
    return f"{kinds}_{int(round(100 * sp.phi))}_{sp.interface}"


def predict_flexural_strength(spec, *, toughness: Optional[dict] = None, **kw) -> dict:
    """ISO 4049 3-point flexural strength estimate (MPa) from Griffith with printing-defect flaw sizes,
    plus the Weibull-equivalent ISO 6872 biaxial strength (for comparison with SprintRay's biaxial claim)."""
    sp = _as_spec(spec)
    tk = toughness or predict_toughness(sp, **kw)
    a_lo, a_c, a_hi = sp.flaw_size_um
    m_lo, m_c, m_hi = sp.weibull_m
    s_c = griffith_strength(tk["value"], a_c)
    s_lo = griffith_strength(tk["range"][0], a_hi)
    s_hi = griffith_strength(tk["range"][1], a_lo)
    biax = threept_to_biaxial(s_c, m_c)
    biax_range = (threept_to_biaxial(s_lo, m_hi), threept_to_biaxial(s_hi, m_lo))
    return {
        "value": s_c, "range": (s_lo, s_hi), "unit": "MPa", "test": "ISO 4049 3-point bar (2x2x25 mm, span 20 mm)",
        "biaxial_equivalent": biax, "biaxial_range": (min(biax_range), max(biax_range)),
        "biaxial_test": "ISO 6872 piston-on-3-balls (support circle 10 mm, piston 1.4 mm, disc 14 mm x 1.2 mm)",
        "method": "Griffith sigma_f = K_Ic/(Y sqrt(pi a)), Y = 0.713 (semicircular surface flaw); Weibull area scaling",
        "K_Ic_used": tk["value"], "K_Ic_range": tk["range"],
        "assumptions": [
            f"critical surface-flaw depth {a_lo}-{a_hi} um (central {a_c}) representing printing defects / "
            "agglomerates / layer-interface flaws (assumed, not measured)",
            f"Weibull modulus {m_lo}-{m_hi} (central {m_c}) assumed for size scaling",
            "strength limited by LEFM fracture from the largest flaw; no subcritical crack growth (water, fatigue)",
        ] + tk["assumptions"],
    }


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def _default_candidate(name: str = "particle-filled dimethacrylate composite"):
    from fracture_md.schema.material import Candidate, CandidateKind, Composition

    return Candidate(id="dental_composite_generic", name=name, kind=CandidateKind.FORMULATION,
                     composition=Composition(formula="resin-composite", elements={}))


def toughness_record(pred: dict, candidate=None):
    from fracture_md.records import make_record
    from fracture_md.schema.common import Conditions, Uncertainty, UncertaintyKind
    from fracture_md.schema.properties import Derivation, MethodSpec, PropertyName, PropertyOrigin, PropertyScale
    from fracture_md.units import Unit, q

    cand = candidate or _default_candidate()
    lo, hi = pred["range"]
    return make_record(
        name=PropertyName.FRACTURE_TOUGHNESS_ESTIMATE, value=pred["value"], unit=Unit.MPA_SQRT_M, candidate=cand,
        method=MethodSpec(name="composite toughness model ensemble", description=pred["method"],
                          assumptions=pred["assumptions"]),
        origin=PropertyOrigin.DERIVED_MODEL, scale=PropertyScale.MACROSCOPIC,
        conditions=Conditions(temperature=q(310.15, Unit.KELVIN), loading="quasistatic", environment="dry (no ageing)"),
        uncertainty=[Uncertainty(kind=UncertaintyKind.INTER_POTENTIAL_SPREAD, value=(hi - lo) / 2, unit=Unit.MPA_SQRT_M,
                                 interval="range", samples=[lo, hi],
                                 description="spread over analytic models x constituent input ranges (model-form proxy)")],
        derivation=Derivation(model="K_c = sqrt(G_c E'_c); G_c from crack-path ROM / Faber-Evans / Lange / phase-field ratio",
                              equation="G = K^2 (1-nu^2)/E", assumptions=pred["assumptions"]),
        notes=json.dumps({k: round(v, 4) for k, v in pred["models"].items()}),
    )


def flexural_record(pred: dict, candidate=None):
    from fracture_md.records import make_record
    from fracture_md.schema.common import Conditions, Uncertainty, UncertaintyKind
    from fracture_md.schema.properties import Derivation, MethodSpec, PropertyName, PropertyOrigin, PropertyScale
    from fracture_md.units import Unit, q

    cand = candidate or _default_candidate()
    lo, hi = pred["range"]
    return make_record(
        name=PropertyName.FLEXURAL_STRENGTH_ESTIMATE, value=pred["value"], unit=Unit.MPA, candidate=cand,
        method=MethodSpec(name="Griffith flaw-size strength + Weibull scaling", description=pred["method"],
                          assumptions=pred["assumptions"]),
        origin=PropertyOrigin.DERIVED_MODEL, scale=PropertyScale.MACROSCOPIC,
        conditions=Conditions(temperature=q(310.15, Unit.KELVIN), loading="quasistatic",
                              stress_state=pred["test"], environment="dry (no ageing)"),
        uncertainty=[Uncertainty(kind=UncertaintyKind.UNQUANTIFIED, value=(hi - lo) / 2, unit=Unit.MPA, interval="range",
                                 samples=[lo, hi], description="flaw size x toughness range")],
        derivation=Derivation(model="sigma_f = K_Ic / (Y sqrt(pi a))", equation="Y = 0.713 semicircular surface flaw",
                              assumptions=pred["assumptions"]),
        notes=f"ISO 6872 biaxial equivalent {pred['biaxial_equivalent']:.1f} MPa (range {pred['biaxial_range'][0]:.0f}-{pred['biaxial_range'][1]:.0f})",
    )
