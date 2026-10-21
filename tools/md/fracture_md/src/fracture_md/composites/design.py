"""Formulation design for a 3D-printable ceramic-filled dental composite resin.

Integrates the dental modules (it re-implements none of their physics):

* ``constituents`` / ``property_models`` - hardness, viscosity, cure depth, sorption,
  solubility, conversion, shrinkage;
* ``homogenization`` - effective modulus (two-scale Mori-Tanaka x FFT correction; full FFT
  for the lead, see ``lead_fft_spec``);
* ``toughening`` (+ the ``phase_field`` study through ``predict_toughness``) - K_Ic,
  3-point flexural strength and its ISO 6872 biaxial equivalent;
* ``bonding`` - weakest-link crown/cement/adhesive/tooth assembly estimate;
* ``benchmarks`` - SprintRay Midas Restore / Ceramic Crown comparison;
* ``discovery.ranking`` - multi-objective Pareto ranking (no single score).

Design space: a seeded Latin hypercube over continuous variables (filler vol%, matrix
blend, zirconia and nano shares, fumed silica, photoinitiator loading, print temperature)
with categorical variables (modality, diluent, photoinitiator, silane, printing route).

Everything returned here is a DERIVED_MODEL estimate; ranges are propagated from the
constituent/model ranges each module states.  Nothing here is a measurement.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

import numpy as np

from fracture_md.composites import benchmarks as BM
from fracture_md.composites import bonding as BD
from fracture_md.composites import constituents as C
from fracture_md.composites import property_models as PM
from fracture_md.composites import strength_calibration as SC
from fracture_md.composites import toughening as TG
from fracture_md.composites.homogenization import effective_modulus
from fracture_md.discovery.ranking import (
    CandidateEvaluation,
    Objective,
    ObjectiveValue,
    probability_better,
    rank_candidates,
)
from fracture_md.records import make_record
from fracture_md.schema.common import CheckResult, Citation, Conditions, Uncertainty, UncertaintyKind, ValidationStatus
from fracture_md.schema.formulation import Component, ComponentRole, Formulation, ProcessingRoute
from fracture_md.schema.material import Candidate, CandidateKind, Composition
from fracture_md.schema.properties import Derivation, MethodSpec, PropertyName, PropertyOrigin, PropertyScale
from fracture_md.units import Unit, q

# ---------------------------------------------------------------------------
# Assumptions that belong to the design layer (stated, with ranges)
# ---------------------------------------------------------------------------
ROUTE_LIMITS_PA_S = {
    # open-vat DLP recoating guideline from the brief (<~3 Pa s at 30 1/s at print temperature)
    "open_vat_dlp": {"limit": 3.0, "status": "brief guideline (DENTAL_CONTEXT / RESEARCH_BRIEF)"},
    # press-fed / heated-cartridge route (Midas-like).  NO verified number: ASSUMED 50 Pa s
    # (range 20-100) - a paste that can be pressed through a heated capsule and spread into a
    # 50 um layer.  Must be measured on the actual printer (protocol P-VISC).
    "press_fed_cartridge": {"limit": 50.0, "range": (20.0, 100.0), "status": "ASSUMED (no verified limit)"},
}
LAYER_UM = 50.0
DOSE_MJ_CM2 = 60.0
WAVELENGTH_NM = 405.0
JAMMING_PA_S = 1.0e3  # 95th-percentile paste viscosity above this = the KD model jams in >5 % of samples
ISO4049_FLEX_MIN_MPA = 100.0  # ISO 4049:2019 extra-orally cured, occlusal (brief)
ISO4049_SORPTION_MAX = 40.0
ISO4049_SOLUBILITY_MAX = 7.5
MOTS8_HV_RATIO = (0.86, 0.84, 0.88)  # Yon 2023 via constituents.json (8-MOTS vs gamma-MPS, composite HV)
MOTS8_FLEX_RATIO = (1.0, 0.69, 1.09)  # Yon 2023: 1.01 (3 wt%) ... 0.69 (12 wt% silane)
RECOMMENDED_INTAGLIO = "APA 50 um 2 bar + silane/MDP universal (Scotchbond Universal Plus)"
SPRINTRAY_INTAGLIO = "APA 50 um 1 bar + Adhese Universal + Variolink Esthetic DC"

# Midas Restore public numbers used as ranking thresholds (from data/dental/targets.json via the brief)
MIDAS = {
    "hv": 82.0,  # HV, load not stated (manufacturer claim)
    "flex_biaxial": 180.0,  # MPa mean, ISO 6872 (manufacturer claim)
    "flex_3pt_hi": 150.0,  # MPa, 510(k) 120-150 (ISO 10477)
    "modulus": 10.5,  # GPa (claim)
    "sorption_lo": 22.0,  # ug/mm^3, 510(k) 22-24
    "solubility_lo": 1.0,  # 1.0-2.5
    # bond_assembly removed from the Midas comparison (final-lead review): the assembly bond is set by the
    # intaglio/cement protocol, not by the material; see bonding.compare_protocols and BONDING.md.
}

# Processing class -> calibrated Griffith effective-flaw class (strength_calibration.fit_classes).
#   commercial: pre-silanised filler as bought, standard dispersion (7 printed products, a_eff ~48 um)
#   enhanced:   Magnus-class silanisation + dispersion (ONE published point, Sanchez 2026; a_eff ~24 um;
#               scatter borrowed from the commercial class - assumption)
PROCESSING_CLASS_FIT = {"commercial": "printed_commercial", "enhanced": "printed_enhanced_silanisation"}
SCORE_KIND = ("ordinal score (assumption-based): P that the displayed estimate exceeds the Midas number, with the "
              "stated range read as a normal/log-normal 90 % interval and the Midas number treated as exact; "
              "it ranks candidates, it is not a calibrated probability")


# ---------------------------------------------------------------------------
# Design variables
# ---------------------------------------------------------------------------
@dataclass
class DesignPoint:
    id: str
    phi_target: float  # cured filler volume fraction
    modality: Literal["mono", "bi", "tri"]
    zirconia_share: float  # share of filler volume as 0.4 um zirconia (radiopacity / hardness)
    nano_share: float  # share of filler volume as nano filler (tri only)
    nano_kind: Literal["fumed silica", "nano zirconia"]
    fumed_silica_wt: float  # extra OX50 in wt% of the paste (thixotropy); 0-0.02
    udma_share: float  # UDMA / (UDMA + Bis-EMA)
    diluent: Literal["TEGDMA", "D3MA"]
    diluent_share: float  # diluent / all monomers
    photoinitiator: Literal["BAPO", "TPO"]
    pi_wt_resin: float  # photoinitiator wt% of the resin (organic) phase
    silane: Literal["gamma-MPS", "8-MOTS"]
    route: Literal["open_vat_dlp", "press_fed_cartridge"]
    print_temperature_c: float
    dose_mj_cm2: float = 60.0  # per-layer exposure at 405 nm
    zirconia_size: Literal["0.4", "nano"] = "0.4"  # micron zirconia scatters strongly (n 2.15)
    notes: str = ""
    # explicit design variable: filler silanisation/dispersion process (sets the calibrated a_eff class)
    processing_class: Literal["commercial", "enhanced"] = "commercial"


def _filler_volume_shares(p: DesignPoint) -> dict[str, float]:
    """Constituent name -> share of the total filler volume (before the fumed-silica additive)."""
    zr = p.zirconia_share
    nano = p.nano_share if p.modality == "tri" else 0.0
    glass = max(1.0 - zr - nano, 0.0)
    out: dict[str, float] = {}
    if p.modality == "mono":
        out["ba glass 0.7"] = glass
    else:
        # bimodal glass: coarse 1.5 um + fine 0.4 um in 65:35 by volume (near the de Larrard optimum
        # for a ~4:1 size ratio; stated design choice, not optimised further)
        out["ba glass 1.5"] = 0.65 * glass
        out["ba glass 0.4"] = 0.35 * glass
    if zr > 0:
        zkey = "zirconia 0.4" if p.zirconia_size == "0.4" else "nano zirconia"
        out[zkey] = out.get(zkey, 0.0) + zr
    if nano > 0:
        out[p.nano_kind] = out.get(p.nano_kind, 0.0) + nano
    return out


def build_formulation(p: DesignPoint, db=None) -> Formulation:
    """Weigh out a formulation hitting the target *cured* filler volume fraction (2 fixed-point passes)."""
    db = db or C.load_constituents()
    shares = _filler_volume_shares(p)
    mons = {"UDMA": (1 - p.diluent_share) * p.udma_share, "Bis-EMA": (1 - p.diluent_share) * (1 - p.udma_share),
            p.diluent: p.diluent_share}
    pi = p.pi_wt_resin / 100.0
    bht = 0.0005

    def weigh(phi_fill: float) -> list[Component]:
        # masses per unit composite volume (cured basis)
        rho_c = {n: C.component_density(C.make_component(n, 0.1, db), db, "cured") for n in mons}
        rho_m = 1.0 / sum(w / rho_c[n] for n, w in mons.items())  # cured monomer blend density
        m_resin = (1.0 - phi_fill) * rho_m
        treat = "3 wt% gamma-MPS (pre-silanised)" if p.silane == "gamma-MPS" else "8-MOTS (8-methacryloxyoctyl trimethoxysilane), 3 wt%"
        masses: list[tuple[str, float, dict]] = [(n, w * m_resin * (1 - pi - bht), {}) for n, w in mons.items() if w > 0]
        masses.append((p.photoinitiator, pi * m_resin, {}))
        masses.append(("BHT", bht * m_resin, {}))
        for n, s in shares.items():
            rho = db.lookup(n).get("density", Unit.G_PER_CM3)
            tr = treat if "zirconia" not in n else "10-MDP + silane (zirconia needs MDP)"
            masses.append((n, phi_fill * s * rho, {"surface_treatment": tr}))
        tot = sum(m for _, m, _ in masses)
        fs = p.fumed_silica_wt
        fumed = db.lookup("fumed silica").name
        comps = []
        for n, m, kw in masses:
            w = m / tot * (1 - fs)
            if fs > 0 and db.lookup(n).name == fumed:
                w += fs
            comps.append(C.make_component(n, w, db, **kw))
        if fs > 0 and not any(c.name == fumed for c in comps):
            comps.append(C.make_component("fumed silica", fs, db, surface_treatment=treat))
        # exact renormalisation (floating point)
        tot = sum(c.weight_fraction for c in comps)
        return [c.model_copy(update={"weight_fraction": c.weight_fraction / tot}) for c in comps]

    T = p.print_temperature_c + 273.15
    proc = ProcessingRoute(
        method="DLP", wavelength_nm=WAVELENGTH_NM, layer_thickness=q(LAYER_UM, Unit.MICROMETER),
        print_temperature=q(T, Unit.KELVIN), exposure_dose=q(p.dose_mj_cm2, Unit.MJ_PER_CM2),
        post_cure="405 nm + heat, 60 C, N2 atmosphere, 2 x 10 min (assumed; tune by P-DC)",
        notes=f"route {p.route}",
    )
    phi_fill = p.phi_target
    for _ in range(3):
        comps = weigh(phi_fill)
        f = Formulation(name=p.id, components=comps, processing=proc)
        phi_now = C.filler_volume_fraction(f, db, "cured")
        phi_fill = min(max(phi_fill + (p.phi_target - phi_now), 0.05), 0.8)
    return C.with_volume_fraction(f, db)


def candidate_for(p: DesignPoint, f: Formulation) -> Candidate:
    return Candidate(id=f"dental_{p.id}", name=f"printable composite {p.id}", kind=CandidateKind.FORMULATION,
                     composition=Composition(formula="filled dimethacrylate resin", elements={}), formulation=f,
                     hypothesis="high, multimodal filler loading in a low-sorption matrix raises hardness/modulus while "
                                "staying printable in a heated vat or press-fed route",
                     mechanism_tags=["filler_loading", "multimodal_packing"], proposed_by="design_agent")


# ---------------------------------------------------------------------------
# Specs for the other modules
# ---------------------------------------------------------------------------
_HOM_MATERIAL = {"ba_glass": "ba_glass", "silica": "silica", "fumed_silica": "silica", "zirconia": "zirconia"}


def inorganic_volume_fractions(f: Formulation, db=None, state: str = "cured") -> dict[str, float]:
    """Per-filler volume fraction on the INORGANIC basis (silane mass moved to the organic phase at
    C.SILANE_DENSITY), consistent with C.filler_volume_fraction(basis='inorganic'); non-fillers -> 0."""
    db = db or C.load_constituents()
    v, v_tot = {}, 0.0
    for c in f.components:
        rho = C.component_density(c, db, state)
        if c.role == ComponentRole.FILLER:
            s_ = C.silane_mass_fraction_on_filler(c)
            v[c.name] = c.weight_fraction * (1.0 - s_) / rho
            v_tot += v[c.name] + c.weight_fraction * s_ / C.SILANE_DENSITY[0]
        else:
            v_tot += c.weight_fraction / rho
    return {k: x / v_tot for k, x in v.items()}


def homogenization_spec(f: Formulation, db=None) -> dict:
    """Explicit spec for homogenization.effective_modulus (the Formulation path maps Ba glass
    to 'silica' by name matching; we pass the Ba-glass constituent explicitly)."""
    db = db or C.load_constituents()
    vf = inorganic_volume_fractions(f, db, "cured")
    fillers = []
    for c in f.components:
        if c.role != ComponentRole.FILLER:
            continue
        k = db.lookup(c.name)
        fillers.append({"material": _HOM_MATERIAL.get(k.filler_type or "", "silica"), "volume_fraction": vf[c.name],
                        "d50_um": k.get("d50", Unit.MICROMETER), "name": c.name})
    return {"fillers": fillers, "matrix": "resin"}


def toughness_spec(f: Formulation, silane: str, db=None) -> dict:
    db = db or C.load_constituents()
    vf = inorganic_volume_fractions(f, db, "cured")
    fl = []
    for c in f.components:
        if c.role != ComponentRole.FILLER:
            continue
        k = db.lookup(c.name)
        kind = "zirconia_filler" if (k.filler_type or "").startswith("zirconia") else "glass_filler"
        fl.append({"kind": kind, "volume_fraction": vf[c.name], "d50_um": k.get("d50", Unit.MICROMETER)})
    # merge by kind (the analytic models use kind-level volume fractions)
    merged: dict[str, dict] = {}
    for x in fl:
        m = merged.setdefault(x["kind"], {"kind": x["kind"], "volume_fraction": 0.0, "d50_um": 0.0})
        m["d50_um"] = (m["d50_um"] * m["volume_fraction"] + x["d50_um"] * x["volume_fraction"]) / (m["volume_fraction"] + x["volume_fraction"])
        m["volume_fraction"] += x["volume_fraction"]
    return {"fillers": list(merged.values()), "interface": "strong"}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class Pred:
    value: float
    low: float
    high: float
    unit: str
    interval_z: float = 1.645
    model: str = ""
    status: str = ValidationStatus.MODEL_ESTIMATE.value
    notes: str = ""

    def ov(self) -> ObjectiveValue:
        return ObjectiveValue(self.value, self.low, self.high, ValidationStatus(self.status), self.interval_z, self.notes)


def _from_mp(mp: PM.ModelPrediction, z: float, factor=(1.0, 1.0, 1.0), note: str = "") -> Pred:
    # the class-range check (printed_experimental class spans phi 0-0.52) is an EXTRAPOLATION flag, reported in the
    # note, not a failed prediction; any other failing hardness check marks the prediction FAILED
    failing = [c.name for c in mp.checks if not c.passed]
    hard = [n for n in failing if "within_class_calibration_range" not in n]
    st = ValidationStatus.FAILED.value if hard and mp.name.value.startswith("vickers") else ValidationStatus.MODEL_ESTIMATE.value
    if failing and mp.name.value.startswith("vickers"):
        note = (note + "; " if note else "") + "EXTRAPOLATION/flags: " + ", ".join(failing)
    return Pred(mp.value * factor[0], mp.low * factor[1], mp.high * factor[2], mp.unit.value, z, mp.model, st, note)


@dataclass
class Evaluation:
    point: DesignPoint
    formulation: Formulation
    phi_cured: float
    phi_uncured: float
    filler_wt: float
    preds: dict[str, Pred]
    constraints: dict[str, dict]
    violations: list[str]
    extra: dict = field(default_factory=dict)
    model_predictions: dict = field(default_factory=dict, repr=False)  # raw ModelPrediction objects

    def summary(self) -> dict:
        return {"id": self.point.id, "design": asdict(self.point), "phi_cured": self.phi_cured, "phi_uncured": self.phi_uncured,
                "filler_wt": self.filler_wt, "predictions": {k: asdict(v) for k, v in self.preds.items()},
                "constraints": self.constraints, "violations": self.violations, "extra": self.extra,
                "recipe_wt": {c.name: c.weight_fraction for c in self.formulation.components}}


def _proxy_pred(mp: PM.ModelPrediction) -> Pred:
    """Proxy-calibrated sorption/solubility (paired ratio to the Midas-like proxy x the Midas 510(k) value)."""
    i = mp.inputs
    return Pred(i["proxy_calibrated_value"], i["proxy_calibrated_low"], i["proxy_calibrated_high"], mp.unit.value, 1.645,
                mp.model + " x paired Midas-proxy calibration", notes="proxy-calibrated (raw model fails the product anchor check)")


def calibrated_strength_preds(K: tuple[float, float, float], processing_class: str, factor=(1.0, 1.0, 1.0),
                              k_note: str = "") -> dict[str, Pred]:
    """Calibrated 3-point (24 h and 7 d water) and ISO 6872 biaxial strength.

    sigma = K/(0.713 sqrt(pi a_eff)) with a_eff of the processing class (strength_calibration.fit_classes);
    90 % prediction interval log-normal with one-sided t_0.95 (enhanced class, n=1: commercial scatter
    borrowed - assumption).  K = (value, low, high): the K range, when given, is folded in by taking the
    PI ends at K_low / K_high.  7 d water factor strength_calibration.AGEING_7D_FACTOR; biaxial by Weibull
    effective-area scaling (m 10, ends m 15 / 8)."""
    fits = SC.fit_classes()
    fit = fits[PROCESSING_CLASS_FIT[processing_class]]
    com = fits["printed_commercial"]
    sig = fit.sigma_log_pred if math.isfinite(fit.sigma_log_pred) else com.sigma_log_pred
    t = fit.t_95 if math.isfinite(fit.sigma_log_pred) else com.t_95
    mid = SC.predict_strength(K[0], fit, sigma_log=sig, t=t)
    lo = SC.predict_strength(K[1], fit, sigma_log=sig, t=t)["low"]
    hi = SC.predict_strength(K[2], fit, sigma_log=sig, t=t)["high"]
    v24 = (mid["value"] * factor[0], lo * factor[1], hi * factor[2])
    ag = SC.AGEING_7D_FACTOR
    v7 = (v24[0] * ag[0], v24[1] * ag[1], v24[2] * ag[2])
    bx = (TG.threept_to_biaxial(v24[0], m=10.0), TG.threept_to_biaxial(v24[1], m=15.0), TG.threept_to_biaxial(v24[2], m=8.0))
    m = (f"calibrated Griffith, class {fit.klass} (a_eff {fit.a_eff_um:.1f} um, sigma_log {sig:.3f}, t {t:.3f}); {k_note}")
    return {"flex_3pt": Pred(*v24, "MPa", 1.645, m + "; 24 h"),
            "flex_3pt_7d": Pred(*v7, "MPa", 1.645, m + "; x 7 d water factor 0.91 [0.83-1.06]"),
            "flex_biaxial": Pred(*bx, "MPa", 1.645, m + "; ISO 6872 Weibull-equivalent (m 10; ends m 15/8)")}


def evaluate_point(p: DesignPoint, db=None) -> Evaluation:
    db = db or C.load_constituents()
    f = build_formulation(p, db)
    phi = C.filler_volume_fraction(f, db, "cured")
    phi_u = C.filler_volume_fraction(f, db, "uncured")
    mots = p.silane == "8-MOTS"
    # modulus first (feeds the Johnson hardness cross-check)
    E, nu, unc, method = effective_modulus(homogenization_spec(f, db))
    Etrip = (E, unc["E_min"], unc["E_max"])
    mp = PM.evaluate_formulation(f, dose_mj_cm2=p.dose_mj_cm2, E_composite_gpa=Etrip, db=db)
    route = ROUTE_LIMITS_PA_S[p.route]
    visc = PM.predict_viscosity(f, temperature_k=p.print_temperature_c + 273.15, limit_pa_s=route["limit"], db=db)
    mp["viscosity"] = visc
    hv_f = MOTS8_HV_RATIO if mots else (1.0, 1.0, 1.0)
    fx_f = MOTS8_FLEX_RATIO if mots else (1.0, 1.0, 1.0)
    tk = TG.predict_toughness(toughness_spec(f, p.silane, db))
    fs = TG.predict_flexural_strength(toughness_spec(f, p.silane, db), toughness=tk)  # uncalibrated Griffith (secondary)
    preds = {
        "hv": _from_mp(mp["hardness_empirical"], 1.96, hv_f, "HV0.2-0.5 dry/24 h calibration; 8-MOTS factor applied" if mots else "HV0.2-0.5 dry/24 h calibration"),
        "hv_johnson_lower": _from_mp(mp["hardness_johnson"], 1.645),
        "flex_3pt_uncalibrated": Pred(fs["value"] * fx_f[0], fs["range"][0] * fx_f[1], fs["range"][1] * fx_f[2], "MPa", 1.645,
                                      fs["method"] + " (UNCALIBRATED; secondary)"),
        "modulus": Pred(E, unc["E_min"], unc["E_max"], "GPa", 1.645, method),
        "k_ic": Pred(tk["value"], tk["range"][0], tk["range"][1], "MPa*m^0.5", 1.645, tk["method"]),
        "sorption_raw": _from_mp(mp["water_sorption"], 1.645, note="raw model; fails the high-filler product anchor check"),
        "solubility_raw": _from_mp(mp["water_solubility"], 1.645, note="raw model; fails the product anchor check"),
        "sorption": _proxy_pred(mp["water_sorption"]),
        "solubility": _proxy_pred(mp["water_solubility"]),
        "viscosity": _from_mp(visc, 1.645),
        "cure_depth": _from_mp(mp["cure_depth"], 1.645),
        "dc": _from_mp(mp["degree_of_conversion"], 1.645),
        "shrinkage": _from_mp(mp["shrinkage"], 1.645),
    }
    K_cal = SC.k_model(phi)  # the K basis the a_eff classes were fitted with (calibration-consistent)
    preds.update(calibrated_strength_preds((K_cal, K_cal, K_cal), p.processing_class, fx_f,
                                           f"K = strength_calibration.k_model(phi={phi:.3f}) (calibration basis)"))
    ba = BD.assess_bonding(f, "superficial_dentin", intaglio_protocol=RECOMMENDED_INTAGLIO, aged=True,
                           restoration_flexural_strength_mpa=preds["flex_3pt"].value, db=db)
    a = ba.assembly_estimate
    preds["bond_assembly"] = Pred(a.value, a.low, a.high, "MPa", 1.645, a.model, notes=a.notes)
    mp["bond_assembly"] = a
    mp["bond_cement_restoration"] = ba.cement_restoration
    # refractive-index mismatch of micron fillers (volume-weighted |n_p - n_resin,uncured|)
    n0 = PM.resin_refractive_index(f, "uncured", db)
    vf = C.formulation_volume_fractions(f, db, "uncured")
    num = den = 0.0
    for c in f.components:
        if c.role == ComponentRole.FILLER and (db.lookup(c.name).get("d50", Unit.MICROMETER) or 1) >= 0.1:
            num += vf[c.name] * abs((c.refractive_index or n0) - n0)
            den += vf[c.name]
    dn = num / den if den else 0.0
    # constraints
    cons: dict[str, dict] = {}
    viol: list[str] = []

    def con(name, ok, detail):
        cons[name] = {"passed": bool(ok), "details": detail}
        if not ok:
            viol.append(f"{name}: {detail}")

    lim = route["limit"]
    con("printable", preds["viscosity"].value <= lim,
        f"eta {preds['viscosity'].value:.3g} Pa s at {p.print_temperature_c:.0f} C vs {p.route} limit {lim} Pa s ({route['status']})")
    con("printable_robust", preds["viscosity"].high < JAMMING_PA_S,
        f"95th-percentile eta {preds['viscosity'].high:.3g} Pa s (limit {JAMMING_PA_S:g}: above it some parameter samples of the "
        "paste model exceed maximum packing - nano-filler bound-layer jamming risk)")
    con("cure_depth_ge_2x_layer", preds["cure_depth"].value >= 2 * LAYER_UM,
        f"C_d {preds['cure_depth'].value:.0f} um vs 2 x {LAYER_UM:.0f} um at {p.dose_mj_cm2:.0f} mJ/cm^2, {WAVELENGTH_NM:.0f} nm")
    con("refractive_index_cure_feasible", not (dn > 0.03 and preds["cure_depth"].low < LAYER_UM),
        f"volume-weighted |dn| of micron fillers {dn:.3f}; C_d 5th pct {preds['cure_depth'].low:.0f} um vs one layer {LAYER_UM:.0f} um")
    con("iso4049_sorption", preds["sorption"].value <= ISO4049_SORPTION_MAX,
        f"{preds['sorption'].value:.1f} [{preds['sorption'].low:.1f}-{preds['sorption'].high:.1f}] ug/mm^3 vs <= {ISO4049_SORPTION_MAX}")
    con("iso4049_solubility", preds["solubility"].value <= ISO4049_SOLUBILITY_MAX,
        f"{preds['solubility'].value:.2f} ug/mm^3 vs <= {ISO4049_SOLUBILITY_MAX}")
    con("iso4049_flexural_min", preds["flex_3pt"].value >= ISO4049_FLEX_MIN_MPA,
        f"{preds['flex_3pt'].value:.0f} MPa vs >= {ISO4049_FLEX_MIN_MPA:.0f} MPa (ISO 4049 extra-orally cured); "
        "ISO sets no K_Ic minimum for polymer-based restoratives (K_Ic >= 2 is a project objective, not a constraint)")
    con("hardness_calibration_domain", mp["hardness_empirical"].checks[0].passed,
        mp["hardness_empirical"].checks[0].details)
    extra = {"delta_n_micron_fillers": dn, "n_resin_uncured": n0, "E_method": method, "E_fft_correction": unc.get("fft_correction"),
             "nu": nu, "k_ic_models": tk["models"], "phase_field_case": tk["phase_field_case"],
             "bond_weakest_interface": ba.interface_level.weakest_interface,
             "bond_origin_probability": ba.interface_level.origin_probability,
             "tpo_cmr_flag": p.photoinitiator == "TPO", "processing_class": p.processing_class,
             "strength_basis": preds["flex_3pt"].model}
    if p.photoinitiator == "TPO":
        extra["warnings"] = ["TPO is classified CMR 1B in the EU (reprotoxic); BAPO preferred"]
    return Evaluation(p, f, phi, phi_u, C.filler_weight_fraction(f), preds, cons, viol, extra, mp)


# ---------------------------------------------------------------------------
# Design space
# ---------------------------------------------------------------------------
def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    u = (np.argsort(rng.random((d, n)), axis=1).T + rng.random((n, d))) / n
    return u


def design_space(n: int = 96, seed: int = 20260928) -> list[DesignPoint]:
    """Seeded Latin hypercube over the continuous variables; categorical variables drawn
    uniformly (modality weighted towards bi/tri because phi > 0.55 needs multimodal filler)."""
    rng = np.random.default_rng(seed)
    u = latin_hypercube(n, 9, rng)
    pts = []
    for i in range(n):
        modality = rng.choice(["mono", "bi", "tri"], p=[0.2, 0.35, 0.45])
        pts.append(DesignPoint(
            id=f"lhs{i:03d}",
            phi_target=float(0.30 + 0.35 * u[i, 0]),
            modality=str(modality),
            zirconia_share=float(0.15 * u[i, 1]),
            nano_share=float(0.05 + 0.15 * u[i, 2]),
            nano_kind=str(rng.choice(["fumed silica", "nano zirconia"])),
            fumed_silica_wt=float(0.02 * u[i, 3]),
            udma_share=float(0.20 + 0.60 * u[i, 4]),
            diluent=str(rng.choice(["TEGDMA", "D3MA"])),
            diluent_share=float(0.10 + 0.20 * u[i, 5]),
            photoinitiator=str(rng.choice(["BAPO", "TPO"], p=[0.7, 0.3])),
            pi_wt_resin=float(0.3 + 0.7 * u[i, 6]),
            silane=str(rng.choice(["gamma-MPS", "8-MOTS"], p=[0.7, 0.3])),
            route=str(rng.choice(["open_vat_dlp", "press_fed_cartridge"])),
            print_temperature_c=float(25.0 + 35.0 * u[i, 7]),
            dose_mj_cm2=float(40.0 + 110.0 * u[i, 8]),
            zirconia_size=str(rng.choice(["0.4", "nano"])),
        ))
    return pts


def engineered_points() -> list[DesignPoint]:
    """Hand-designed points following the modules' findings: phi ~0.60 for a median 85 HV;
    multimodal filler above 0.55; refractive-index matching (Bis-EMA-rich matrix, n ~1.50-1.51,
    against Ba glass n 1.53) and nano- rather than micron zirconia so that scattering does not
    kill the cure depth; low-sorption D3MA diluent; BAPO (not TPO); heated vat or press-fed."""
    base = dict(zirconia_share=0.06, zirconia_size="nano", nano_share=0.10, nano_kind="fumed silica", fumed_silica_wt=0.0,
                udma_share=0.35, diluent="D3MA", diluent_share=0.15, photoinitiator="BAPO", pi_wt_resin=0.4, silane="gamma-MPS",
                dose_mj_cm2=100.0)
    out = []
    for phi in (0.55, 0.58, 0.60, 0.62):
        out.append(DesignPoint(id=f"eng_tri_vat_phi{int(round(phi * 100))}", phi_target=phi, modality="tri", route="open_vat_dlp",
                               print_temperature_c=60.0, **base))
        out.append(DesignPoint(id=f"eng_tri_press_phi{int(round(phi * 100))}", phi_target=phi, modality="tri", route="press_fed_cartridge",
                               print_temperature_c=50.0, **base))
    b2 = dict(base, nano_share=0.0)
    out.append(DesignPoint(id="eng_bi_vat_phi58", phi_target=0.58, modality="bi", route="open_vat_dlp", print_temperature_c=60.0, **b2))
    out.append(DesignPoint(id="eng_bi_vat_phi58_tegdma", phi_target=0.58, modality="bi", route="open_vat_dlp", print_temperature_c=60.0,
                           **dict(b2, diluent="TEGDMA")))
    out.append(DesignPoint(id="eng_tri_vat_phi60_udma", phi_target=0.60, modality="tri", route="open_vat_dlp", print_temperature_c=60.0,
                           **dict(base, udma_share=0.70)))
    out.append(DesignPoint(id="eng_tri_vat_phi60_zr04", phi_target=0.60, modality="tri", route="open_vat_dlp", print_temperature_c=60.0,
                           **dict(base, zirconia_size="0.4")))
    return out


def midas_proxy_point() -> DesignPoint:  # noqa: D401
    """Midas-Restore-like proxy (composition NOT public: >75 wt% glass-ceramic + zirconia,
    ~56 vol% by the brief's densities).  Used only for same-model paired comparisons."""
    return DesignPoint(id="midas_proxy", phi_target=0.56, modality="bi", zirconia_share=0.10, nano_share=0.0, nano_kind="fumed silica",
                       fumed_silica_wt=0.0, udma_share=0.7, diluent="TEGDMA", diluent_share=0.15, photoinitiator="TPO",
                       pi_wt_resin=0.5, silane="gamma-MPS", route="press_fed_cartridge", print_temperature_c=50.0, dose_mj_cm2=100.0,
                       notes="PROXY: public composition unknown; >75 wt% glass-ceramic+zirconia, heated cartridge")


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
OBJECTIVES = [
    Objective("hv", "strength", "max", "HV", log_scale=True, description="Vickers hardness (HV0.2 calibration)"),
    Objective("flex_3pt", "strength", "max", "MPa", log_scale=True, description="ISO 4049 3-point flexural strength"),
    Objective("modulus", "strength", "max", "GPa", description="effective Young's modulus (flexural-modulus proxy)"),
    Objective("k_ic", "fracture", "max", "MPa*m^0.5", description="K_Ic estimate"),
    Objective("g_sim", "fracture", "max", "J/m^2", required=False, description="phase-field fracture energy (lead only)"),
    Objective("sorption", "durability", "min", "ug/mm^3", description="water sorption ISO 4049"),
    Objective("solubility", "durability", "min", "ug/mm^3", log_scale=True, description="water solubility ISO 4049"),
]

# comparisons vs Midas: (objective, threshold, direction, log, test-method note)
MIDAS_COMPARISONS = [
    ("hv", MIDAS["hv"], "max", True, "Midas load not stated; prediction is HV0.2-0.5 calibration"),
    ("flex_biaxial", MIDAS["flex_biaxial"], "max", True, "biaxial ISO 6872 claim vs our Weibull-converted estimate"),
    ("flex_3pt", MIDAS["flex_3pt_hi"], "max", True, "510(k) 3-point ISO 10477 (120-150, upper end) vs our calibrated 24 h ISO 4049 bar"),
    ("modulus", MIDAS["modulus"], "max", False, "claimed modulus (method not stated) vs elastic E_eff"),
    ("sorption", MIDAS["sorption_lo"], "min", False, "510(k) ISO 4049-type sorption vs our proxy-calibrated estimate"),
    ("solubility", MIDAS["solubility_lo"], "min", True, "510(k) solubility vs our proxy-calibrated estimate"),
]


def to_ranking_input(e: Evaluation) -> CandidateEvaluation:
    obj = {k: e.preds[k].ov() for k in ("hv", "flex_3pt", "modulus", "k_ic", "sorption", "solubility")}
    if "g_sim" in e.preds:
        obj["g_sim"] = e.preds["g_sim"].ov()
    return CandidateEvaluation(e.point.id, obj, list(e.violations))


def midas_scorecard(e: Evaluation) -> dict:
    """Per-property probability (and range-margin) of beating the Midas Restore number."""
    out = {}
    for key, thr, d, lg, note in MIDAS_COMPARISONS:
        pr = e.preds[key]
        pb = probability_better(pr.ov(), thr, d, lg)
        margin_beyond_uncertainty = (pr.low > thr) if d == "max" else (pr.high < thr)
        out[key] = {"predicted": pr.value, "low": pr.low, "high": pr.high, "midas": thr, "P_better": pb, "P_better_kind": SCORE_KIND,
                    "quantity": pr.model,
                    "beats_with_margin": bool(margin_beyond_uncertainty), "beats_median": bool(pr.value > thr if d == "max" else pr.value < thr),
                    "note": note}
    return out


def choose_lead(evals: list[Evaluation], ranking) -> tuple[str, list[dict]]:
    """Lead = feasible, overall Pareto tier 0 (median mode), maximising (1) the number of Midas
    properties beaten with margin beyond the stated range, (2) the number with P_better >= 0.8,
    (3) the sum of P_better, (4) BAPO + gamma-MPS preferred (no harmonised CMR classification found for BAPO -
    ECHA not directly verified, screening flag reported, skin sensitiser H317 - and a verified silane).
    P_better is an ordinal, assumption-based score (SCORE_KIND); the protocol-driven assembly bond is NOT scored.
    Returns (lead id, ordered table)."""
    table = []
    for e in evals:
        if e.violations or ranking.tiers_overall.get(e.point.id, 99) != 0:
            continue
        sc = midas_scorecard(e)
        table.append({"id": e.point.id, "n_margin": sum(v["beats_with_margin"] for v in sc.values()),
                      "n_p80": sum(v["P_better"] >= 0.8 for v in sc.values()),
                      "sum_p": sum(v["P_better"] for v in sc.values()),
                      "safe_chem": int(e.point.photoinitiator == "BAPO") + int(e.point.silane == "gamma-MPS")})
    table.sort(key=lambda r: (r["n_margin"], r["n_p80"], r["safe_chem"], r["sum_p"]), reverse=True)
    return (table[0]["id"] if table else ""), table


def run_sweep(n_lhs: int = 96, seed: int = 20260928, include_engineered: bool = True):
    db = C.load_constituents()
    pts = design_space(n_lhs, seed) + (engineered_points() if include_engineered else [])
    evals = [evaluate_point(p, db) for p in pts]
    ranking = rank_candidates([to_ranking_input(e) for e in evals], OBJECTIVES, "median")
    ranking_iv = rank_candidates([to_ranking_input(e) for e in evals], OBJECTIVES, "interval")
    lead, table = choose_lead(evals, ranking)
    return evals, ranking, ranking_iv, lead, table


# ---------------------------------------------------------------------------
# Records for the lead
# ---------------------------------------------------------------------------
def records_for_evaluation(e: Evaluation, cand: Optional[Candidate] = None) -> list:
    """PropertyRecords (make_record) for every prediction of an evaluation."""
    cand = cand or candidate_for(e.point, e.formulation)
    recs = []
    mp = e.model_predictions
    for key in ("hardness_empirical", "hardness_johnson", "viscosity", "cure_depth", "water_sorption", "water_solubility",
                "degree_of_conversion", "shrinkage", "bond_assembly", "bond_cement_restoration"):
        if key in mp:
            m = mp[key]
            if key == "hardness_empirical" and e.point.silane == "8-MOTS":
                m = m.model_copy(update={"value": e.preds["hv"].value, "low": e.preds["hv"].low, "high": e.preds["hv"].high,
                                         "assumptions": m.assumptions + ["x 8-MOTS/gamma-MPS HV ratio 0.86 [0.84-0.88] (Yon 2023)"]})
            recs.append(m.to_record(cand))
    fs = TG.predict_flexural_strength(toughness_spec(e.formulation, e.point.silane))
    tk = TG.predict_toughness(toughness_spec(e.formulation, e.point.silane))
    recs.append(TG.toughness_record(tk, cand))
    recs.append(TG.flexural_record(fs, cand))
    recs.append(scalar_record(PropertyName.FILLER_VOLUME_FRACTION, e.phi_cured, Unit.DIMENSIONLESS, cand,
                              "wt->vol conversion with cured densities", "phi = sum(w_i/rho_i) over fillers / sum(w/rho)",
                              ["cured monomer density = rho/(1 - homopolymer shrinkage) (constituents.py)"], (e.phi_cured - 0.01, e.phi_cured + 0.01)))
    recs.append(scalar_record(PropertyName.FILLER_WEIGHT_FRACTION, e.filler_wt, Unit.DIMENSIONLESS, cand, "recipe", "sum of filler weight fractions",
                              ["as weighed"], (e.filler_wt, e.filler_wt)))
    return recs


def scalar_record(name: PropertyName, value: float, unit: Unit, cand: Candidate, model: str, equation: str, assumptions: list[str],
                  rng: tuple[float, float], checks: list[CheckResult] = (), scale=PropertyScale.MACROSCOPIC, notes: str = "",
                  citations: list[Citation] = ()):
    rec = make_record(
        name=name, value=value, unit=unit, candidate=cand,
        method=MethodSpec(name=model, engine="fracture_md.composites.design", description=equation, assumptions=assumptions),
        origin=PropertyOrigin.DERIVED_MODEL, scale=scale,
        conditions=Conditions(temperature=q(310.15, Unit.KELVIN), loading="quasistatic", environment="model estimate"),
        uncertainty=[Uncertainty(kind=UncertaintyKind.REFERENCE_DISCREPANCY if rng[0] != rng[1] else UncertaintyKind.UNQUANTIFIED,
                                 value=0.5 * (rng[1] - rng[0]), unit=unit, interval="range", samples=[rng[0], rng[1]],
                                 description=f"range [{rng[0]:.4g}, {rng[1]:.4g}]")],
        checks=list(checks),
        derivation=Derivation(model=model, equation=equation, assumptions=assumptions),
        notes=notes,
    )
    rec.citations = list(citations)
    return rec


def benchmark_rows(recs: list) -> list:
    """benchmarks.compare for 3-point and (separately) the biaxial equivalent."""
    rows = BM.compare(recs)
    # biaxial-equivalent copy of the flexural record for comparison with Midas's biaxial claim
    fl = [r for r in recs if r.name == PropertyName.FLEXURAL_STRENGTH_ESTIMATE]
    return rows, fl


# ---------------------------------------------------------------------------
# Lead characterisation: full-PSD FFT homogenisation (run on the GPU container)
# ---------------------------------------------------------------------------
def lead_fft_case(e: Evaluation, db=None) -> dict:
    """Two-scale RVE spec for the lead's actual PSD: micron fillers resolved as log-normal
    sphere families (cv 0.15, as in the homogenisation module), nano fillers (d50 < 0.1 um)
    homogenised into the matrix by Mori-Tanaka (two-scale treatment of HOMOGENIZATION.md)."""
    db = db or C.load_constituents()
    spec = homogenization_spec(e.formulation, db)
    micro = [f for f in spec["fillers"] if f["d50_um"] >= 0.1]
    nano = [f for f in spec["fillers"] if f["d50_um"] < 0.1]
    phi_micro = sum(f["volume_fraction"] for f in micro)
    return {"micro": micro, "nano": nano, "phi_micro": phi_micro,
            "nano_in_matrix": [(f["material"], f["volume_fraction"] / (1.0 - phi_micro)) for f in nano],
            "phi_total": phi_micro + sum(f["volume_fraction"] for f in nano)}


def run_lead_fft(case: dict, n: int = 96, seeds=(0, 1), gap: float = 0.025, n_particles: int = 120,
                 which_list=("nominal", "min", "max")) -> dict:
    """FFT (CG Lippmann-Schwinger) 6x6 stiffness of the lead RVE for each seed and constituent set."""
    from fracture_md.composites.homogenization import (CONSTITUENTS as HC, MATRIX_PHASE, IsotropicPhase,
                                                        E_nu_from_KG, fft_homogenize, hashin_shtrikman_bounds,
                                                        mori_tanaka_spheres, nanofilled_matrix)
    from fracture_md.composites.microstructure import SphereFamily, pack_spheres, voxelize

    fams = [SphereFamily(f["name"], f["material"], f["d50_um"], f["volume_fraction"], 0.15) for f in case["micro"]]
    out = {"case": case, "n": n, "gap_um": gap, "runs": []}
    for seed in seeds:
        p = pack_spheres(fams, n_particles=n_particles, seed=seed, min_gap=gap)
        gap_used = gap
        if not p.converged and gap > 0:
            p = pack_spheres(fams, n_particles=n_particles, seed=seed, min_gap=0.0)
            gap_used = 0.0
        v = voxelize(p, n)
        for which in which_list:
            i = {"min": 0, "nominal": 1, "max": 2}[which]
            resin = IsotropicPhase("resin", HC["resin"].E[i], HC["resin"].nu[1])
            nano_ph = [IsotropicPhase(k, HC[k].E[i], HC[k].nu[1]) for k, _ in case["nano_in_matrix"]]
            mat = nanofilled_matrix(resin, nano_ph, [x for _, x in case["nano_in_matrix"]])
            ph = {MATRIX_PHASE: mat}
            for nm in v.phase_names:
                if nm != MATRIX_PHASE:
                    ph[nm] = IsotropicPhase(nm, HC[nm].E[i], HC[nm].nu[1])
            r = fft_homogenize(v, ph, tol=1e-6)
            incl = [k for k in v.phase_names if k != MATRIX_PHASE]
            Emt, numt = E_nu_from_KG(*mori_tanaka_spheres(mat, [ph[k] for k in incl], [v.volume_fractions[k] for k in incl]))
            hs = hashin_shtrikman_bounds([ph[k] for k in v.phase_names], [v.volume_fractions[k] for k in v.phase_names])
            Ehs_lo, Ehs_hi = E_nu_from_KG(*hs["lower"])[0], E_nu_from_KG(*hs["upper"])[0]
            out["runs"].append({
                "seed": seed, "which": which, "gap_used_um": gap_used, "packing_converged": bool(p.converged),
                "n_particles": int(len(p.radii)), "box_um": float(p.box), "voxel_volume_fractions": v.volume_fractions,
                "matrix_eff_E_GPa": mat.E, "matrix_eff_nu": mat.nu,
                "C_eff_GPa": np.round(r.C_eff, 5).tolist(), "E_fft": r.metrics["E_hill"], "nu_fft": r.metrics["nu_hill"],
                "E_MT": Emt, "E_HS_lower": Ehs_lo, "E_HS_upper": Ehs_hi,
                "within_HS": bool(Ehs_lo * (1 - 1e-6) <= r.metrics["E_hill"] <= Ehs_hi * (1 + 1e-6)),
                "A_U": r.metrics["universal_anisotropy_index"], "dev_iso": r.metrics["relative_deviation_from_isotropy"],
                "iterations": r.iterations, "max_residual": max(r.residuals), "symmetry_error": r.symmetry_error,
                "converged": bool(r.converged), "wall_time_s": r.wall_time_s, "device": r.device})
    return out


# ---------------------------------------------------------------------------
# Iteration 2 additions (new functions only; nothing above is changed)
# ---------------------------------------------------------------------------
def random_fibres(width: float, height: float, area_fraction: float, length: float, diameter: float, *,
                  seed: int = 0, min_gap: float = 2.0, exclude=None, max_tries: int = 20000,
                  angle_range: tuple[float, float] = (0.0, math.pi)) -> list[tuple[float, float, float]]:
    """Random sequential addition of non-overlapping rectangular (2D) short fibres.

    Returns (x_centre, y_centre, theta) with theta the fibre-axis angle from +x.  The model plane is the
    printed layer plane (layer-wise printing aligns fibres in-plane), orientation uniform in angle_range.
    Overlap test: separating-axis test on rectangles inflated by min_gap/2.
    """
    rng = np.random.default_rng(seed)
    target = area_fraction * width * height
    out: list[tuple[float, float, float]] = []
    placed, tries = 0.0, 0
    hl, hw = 0.5 * length + 0.5 * min_gap, 0.5 * diameter + 0.5 * min_gap

    def corners(x, y, t):
        c, s = math.cos(t), math.sin(t)
        ax, ay = np.array([c, s]), np.array([-s, c])
        ctr = np.array([x, y])
        return [ctr + sx * hl * ax + sy * hw * ay for sx in (-1, 1) for sy in (-1, 1)], (ax, ay)

    def overlap(a, b):
        ca, axa = a
        cb, axb = b
        for ax in axa + axb:
            pa = [float(np.dot(p, ax)) for p in ca]
            pb = [float(np.dot(p, ax)) for p in cb]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
        return True

    geoms = []
    while placed < target and tries < max_tries:
        tries += 1
        x, y = rng.uniform(0, width), rng.uniform(0, height)
        t = rng.uniform(*angle_range)
        if exclude is not None and exclude(x, y, t):
            continue
        g = corners(x, y, t)
        if any(overlap(g, h) for h in geoms):
            continue
        geoms.append(g)
        out.append((x, y, t))
        # in-box area (clip by sampling along the fibre axis)
        s = np.linspace(-0.5 * length, 0.5 * length, 41)
        px, py = x + s * math.cos(t), y + s * math.sin(t)
        placed += float(np.mean((px >= 0) & (px <= width) & (py >= 0) & (py <= height))) * length * diameter
    return out


def rasterize_fibres(grid, fibres, length: float, diameter: float, matrix, fibre, interface=None,
                     interface_thickness: float = 0.0, void_mask=None, fibre_damageable: bool = False):
    """Phase labels as phase_field.rasterize_microstructure: 0 matrix, 1 fibre, 2 interface band, 3 void.

    Rectangle signed distance (outside): max(|s| - L/2, |t| - d/2) in fibre-local coordinates.
    ``fibre_damageable=False`` makes the fibres unbreakable (E-glass fibre strength 1.5-3.5 GPa is far
    above the stresses reached; also needed because a brittle fibre's l_ch << b violates the PF-CZM
    length condition)."""
    from fracture_md.composites.phase_field import Material, PhaseProps

    c = grid.centers
    phase = np.zeros(grid.nel, int)
    dist = np.full(grid.nel, np.inf)
    for (x, y, t) in fibres:
        dx, dy = c[:, 0] - x, c[:, 1] - y
        s = dx * math.cos(t) + dy * math.sin(t)
        tt = -dx * math.sin(t) + dy * math.cos(t)
        dist = np.minimum(dist, np.maximum(np.abs(s) - 0.5 * length, np.abs(tt) - 0.5 * diameter))
    phase[dist <= 0] = 1
    if interface is not None and interface_thickness > 0:
        phase[(dist > 0) & (dist <= interface_thickness)] = 2
    if void_mask is not None:
        phase[np.asarray(void_mask, bool)] = 3
    fprops = PhaseProps(E=fibre.E, nu=fibre.nu, Gc=fibre.Gc, ft=fibre.ft, damageable=fibre_damageable)
    props = {0: matrix, 1: fprops, 2: interface or matrix,
             3: PhaseProps(E=matrix.E * 1e-9, nu=matrix.nu, Gc=matrix.Gc, ft=matrix.ft, damageable=False)}
    mat = Material(E=np.zeros(grid.nel), nu=np.zeros(grid.nel), Gc=np.zeros(grid.nel), ft=np.zeros(grid.nel),
                   damageable=np.zeros(grid.nel, bool), phase=phase)
    for k, p in props.items():
        sel = phase == k
        mat.E[sel], mat.nu[sel], mat.Gc[sel], mat.ft[sel], mat.damageable[sel] = p.E, p.nu, p.Gc, p.ft, p.damageable
    return mat


def build_fibre_specimen(area_fraction: float, *, matrix, fibre, interface=None, h: float = 2.0,
                         width: float = 192.0, height: float = 128.0, a0_frac: float = 0.2, length: float = 90.0,
                         diameter: float = 6.0, interface_thickness: Optional[float] = None, seed: int = 1,
                         min_gap: Optional[float] = None, fibre_damageable: bool = False):
    """Clamped-SENT specimen of short fibres in a homogenised particulate-composite matrix (two-scale)."""
    from fracture_md.composites.phase_field import Grid, notch_void_mask

    grid = Grid(int(round(width / h)), int(round(height / h)), h)
    a0 = a0_frac * grid.width
    jn = int(grid.height / 2 / h)
    y_notch = (jn + 0.5) * h
    void = notch_void_mask(grid, a0, y_notch)
    gap = 2.0 * h if min_gap is None else min_gap
    ith = h if interface_thickness is None else interface_thickness

    def excl(x, y, t):  # keep fibres (plus band) off the notch row
        s = np.linspace(-0.5 * length, 0.5 * length, 61)
        px, py = x + s * math.cos(t), y + s * math.sin(t)
        return bool(np.any((px < a0 + gap) & (np.abs(py - y_notch) < 0.5 * diameter + gap + h)))

    fibres = random_fibres(grid.width, grid.height, area_fraction, length, diameter, seed=seed, min_gap=gap,
                           exclude=excl) if area_fraction > 0 else []
    mat = rasterize_fibres(grid, fibres, length, diameter, matrix, fibre, interface if fibres else None,
                           ith if fibres else 0.0, void, fibre_damageable)
    solid = mat.phase != 3
    return grid, mat, {"a0": a0, "y_notch": y_notch, "fibres": fibres,
                       "fibre_area_fraction": float(np.mean(mat.phase[solid] == 1)),
                       "interface_area_fraction": float(np.mean(mat.phase[solid] == 2))}


# ---------------------------------------------------------------------------
# Final-lead additions: local refinement, phase-field K with lead-id check
# ---------------------------------------------------------------------------
FM_R60_BASE = dict(  # lhs094 (FM-R60 v1) chemistry, data/dental/design/lead_characterisation.json
    modality="bi", zirconia_share=0.05775486531705373, nano_share=0.09964739826798627, nano_kind="fumed silica",
    fumed_silica_wt=0.013530585620405863, udma_share=0.3920663842535784, diluent="D3MA", diluent_share=0.2566054958967243,
    photoinitiator="BAPO", pi_wt_resin=0.43892396790885946, silane="gamma-MPS", route="press_fed_cartridge",
    print_temperature_c=54.154002732372355, dose_mj_cm2=94.34465929591914, zirconia_size="nano")


def refinement_points(base: Optional[dict] = None, phis=(0.58, 0.60, 0.62, 0.64), fumed=(0.0, 0.0135),
                      zr_shares=(0.0, 0.0578), classes=("commercial", "enhanced")) -> list[DesignPoint]:
    """Local refinement around the Pareto front on the press-fed route: phi (inorganic basis) x fumed
    silica x nano-zirconia share x processing class, keeping the FM-R60 chemistry.  The jamming
    robustness check is the existing 'printable_robust' constraint of evaluate_point."""
    base = dict(base or FM_R60_BASE)
    out = []
    for cl in classes:
        for phi in phis:
            for fs in fumed:
                for zr in zr_shares:
                    d = dict(base, fumed_silica_wt=fs, zirconia_share=zr, route="press_fed_cartridge")
                    pid = f"ref_{'es' if cl == 'enhanced' else 'cs'}_phi{int(round(phi * 100))}_fs{int(round(fs * 1e4)):03d}_zr{int(round(zr * 100)):02d}"
                    out.append(DesignPoint(id=pid, phi_target=phi, processing_class=cl, **d))
    return out


def load_phase_field_k(path, lead_id: str, *, required_family: str = "FM-R60") -> dict:
    """Phase-field K_Ic for the lead from data/dental/final/lead_rg_refinement.json (or a raw file with a
    'lead' key).  Raises ValueError when the file's lead id / family does not match (lead-id check)."""
    import json
    from pathlib import Path

    d = json.loads(Path(path).read_text())
    tag = str(d.get("lead", d.get("lead_id", "")))
    if not tag and "specimen" in d:
        tag = "FM-R60 (lhs094)"  # lead_rg_refinement.json predates the lead key; its specimen is the lhs094 case
    if lead_id not in tag and required_family not in tag and "lhs094" not in tag:
        raise ValueError(f"phase-field data in {path} are for '{tag}', not lead '{lead_id}' (family {required_family})")
    k = d["K_Ic"] if "K_Ic" in d else d["k_ic_pf"]
    return {"value": float(k["value"]), "low": float(k["low"]), "high": float(k["high"]), "source": str(path), "tag": tag}


def apply_lead_strength(e: Evaluation, K: tuple[float, float, float], k_note: str) -> Evaluation:
    """Replace the lead's strength predictions with the calibrated class a_eff and the lead's phase-field K."""
    fx = MOTS8_FLEX_RATIO if e.point.silane == "8-MOTS" else (1.0, 1.0, 1.0)
    e.preds.update(calibrated_strength_preds(K, e.point.processing_class, fx, k_note))
    e.extra["strength_basis"] = e.preds["flex_3pt"].model
    return e


def run_refinement(n_lhs: int = 96, seed: int = 20260928):
    """Sweep (LHS + engineered) plus the local refinement; returns (evals, ranking, ranking_iv, lead, table)."""
    db = C.load_constituents()
    pts = design_space(n_lhs, seed) + engineered_points() + refinement_points()
    evals = [evaluate_point(p, db) for p in pts]
    ranking = rank_candidates([to_ranking_input(e) for e in evals], OBJECTIVES, "median")
    ranking_iv = rank_candidates([to_ranking_input(e) for e in evals], OBJECTIVES, "interval")
    lead, table = choose_lead(evals, ranking)
    return evals, ranking, ranking_iv, lead, table
