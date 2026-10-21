"""Design of a DIRECT (hand-placed, light-cured in the cavity) ceramic-filled composite.

The user relaxed the 3D-printing requirement.  This module re-uses the harness models
(no physics is re-implemented) and replaces the printing constraints by direct-placement ones:

* handling window (sculptable paste) instead of open-vat printability - the Newtonian
  Krieger-Dougherty model in ``property_models`` cannot represent a yield stress, so handling is
  scored RELATIVE to a Filtek-Supreme-Ultra-like proxy paste computed with the same model;
* camphorquinone / amine curing at 470 nm with a 1000-1200 mW/cm^2 LED for 10-20 s; ISO 4049
  depth of cure (= half the scraped cured length) from the Jacobs working curve with the
  ``property_models`` scattering model evaluated at 470 nm and CQ absorption (S35);
* in-situ degree of conversion (no post-cure) from the homopolymer-DC rule; its effect on
  hardness (``dc_ratio``), matrix toughness / strength and (via the light-cured homopolymer basis)
  water sorption is applied explicitly;
* polymerisation shrinkage (``property_models.predict_shrinkage``) and shrinkage STRESS with a
  constrained-elastic post-gel model (Davidson & Feilzer 1997 [S46]) compared against the aged
  dentin bond (Weibull, ``bonding.tooth_interface``);
* ISO 4049:2019 Type 1 (occlusal) limits for intra-orally light-activated materials.

Everything returned is a DERIVED_MODEL estimate (MODEL_ESTIMATE); nothing is a measurement.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from fracture_md.composites import bonding as BD
from fracture_md.composites import constituents as C
from fracture_md.composites import property_models as PM
from fracture_md.composites import toughening as TG
from fracture_md.composites.design import MIDAS, homogenization_spec, toughness_spec
from fracture_md.composites.homogenization import effective_modulus
from fracture_md.composites.strength_calibration import AGEING_7D_FACTOR
from fracture_md.discovery.ranking import CandidateEvaluation, Objective, ObjectiveValue, rank_candidates
from fracture_md.records import make_record
from fracture_md.schema.common import CheckResult, Citation, Conditions, Uncertainty, UncertaintyKind
from fracture_md.schema.formulation import ComponentRole, Formulation, ProcessingRoute
from fracture_md.schema.material import Candidate, CandidateKind, Composition
from fracture_md.schema.properties import Derivation, MethodSpec, PropertyName, PropertyOrigin, PropertyScale
from fracture_md.units import Unit, q

ROOT = Path(__file__).resolve().parents[3]
DIRECT_DATA = ROOT / "data" / "dental" / "direct"
TARGETS = ROOT / "data" / "dental" / "targets.json"

# ---------------------------------------------------------------------------
# Direct-placement conditions and limits (stated, with sources)
# ---------------------------------------------------------------------------
WAVELENGTH_NM = 470.0  # CQ absorption maximum ~468-470 nm (brief 3.2 [S35])
IRRADIANCE_MW_CM2 = (1100.0, 1000.0, 1200.0)  # LED curing light (user brief)
EXPOSURE_S = (20.0, 10.0, 20.0)  # per increment (user brief); nominal 20 s
INCREMENT_MM = 2.0  # conventional incremental technique
ISO_DOC_MIN_MM = 1.5  # ISO 4049:2019 5.2.8, non-opaque (targets.json)
ISO_FLEX_MIN_MPA = 80.0  # ISO 4049:2019 Table 1, Type 1 (occlusal), intra-orally light-activated (targets.json)
ISO_SORPTION_MAX = 40.0
ISO_SOLUBILITY_MAX = 7.5
# Critical (gel-point) radiant exposure of a CQ/amine dimethacrylate paste for the ISO scrape test.
# ASSUMED 1 J/cm^2 [0.3-3]: no verified working-curve value for a CQ composite in the source base.
EC_MJ_CM2 = (1000.0, 300.0, 3000.0)
TRANSPORT_FRACTION = (0.07, 0.03, 0.15)  # same assumption as property_models.predict_cure_depth
# In-situ DC of light-cured direct composites: 55-70 % (user brief; Ferracane 2011 review [S34]).
DC_DIRECT_RANGE = (55.0, 70.0)
DC_REF_DIRECT = 62.5  # centre of that range = the DC basis of the "direct" hardness class and the matrix K_Ic
# Shrinkage-stress model (Davidson & Feilzer 1997 [S46]: stress = constrained post-gel shrinkage x modulus,
# relieved by pre-gel flow and wall compliance).  Parameters are ASSUMED and wide.
POST_GEL_FRACTION = (0.30, 0.20, 0.50)  # share of the volumetric shrinkage after gelation (ASSUMED)
C_FACTOR = (4.0, 3.0, 5.0)  # bonded / unbonded surface ratio of a class I/II increment (user brief)
C_HALF = 2.0  # constraint factor kappa = C/(C + C_half) (ASSUMED interpolation: free (C=0) -> 0, rigid (C->inf) -> 1)
RELAXATION = (0.35, 0.20, 0.50)  # viscoelastic relaxation + tooth/adhesive compliance during cure (ASSUMED)
STRESS_BOND_FRACTION_MAX = 0.25  # shrinkage stress must stay below 25 % of the aged dentin uTBS median (design rule)
DEBOND_P_MAX = 0.05  # and the Weibull probability that it alone debonds the dentin interface below 5 %
ADHESIVE = "universal, etch-and-rinse mode (Scotchbond Universal)"  # selective-enamel-etch + 10-MDP universal
HANDLING_RATIO = (0.33, 3.0)  # paste viscosity within 1/3-3x of the Filtek-Supreme-Ultra-like proxy = sculptable

CITES = {
    "S34": Citation(key="S34", text="Ferracane JL. Resin composite - state of the art. Dent Mater 2011;27:29-38", doi="10.1016/j.dental.2010.10.020"),
    "S35": Citation(key="S35", text="Neumann MG et al. Molar extinction coefficients and the photon absorption efficiency of dental photoinitiators and light curing units. J Dent 2005;33:525-532", doi="10.1016/j.jdent.2004.11.013"),
    "S46": Citation(key="S46", text="Davidson CL, Feilzer AJ. Polymerization shrinkage and polymerization shrinkage stress in polymer-based restoratives. J Dent 1997;25:435-440", doi="10.1016/S0300-5712(96)00063-2"),
    "S47": Citation(key="S47", text="Chung KH, Greener EH. Correlation between degree of conversion, filler concentration and mechanical properties of posterior composite resins. J Oral Rehabil 1990;17:487-494", doi="10.1111/j.1365-2842.1990.tb01419.x"),
    "ISO4049": Citation(key="ISO4049", text="ISO 4049:2019 Dentistry - Polymer-based restorative materials (Table 1; 5.2.8 depth of cure; 5.2.10 sorption/solubility)",
                        url="https://cdn.standards.iteh.ai/samples/67596/66dff80a23e94f87844dd3990fe7f440/ISO-4049-2019.pdf"),
    "C17": Citation(key="C17", text="Solventum/3M Filtek Supreme Ultra Technical Product Profile (165.1 +/- 13.6 MPa, 11.35 GPa, K_Ic 1.84, 78.5 wt% / 63.3 vol%)",
                    url="https://multimedia.3m.com/mws/media/1363018O/3m-filtek-supreme-ultra-universal-restorative-technical-product-profile.pdf"),
    "ScougallVilchis2009": Citation(key="ScougallVilchis2009", text="Scougall-Vilchis RJ et al. 2009: HV0.3 of direct composites (dry, 5 d) as stored in data/dental/hardness_calibration.json"),
}

# Best direct composites with verified data in the project (evidence level stated).
DIRECT_BENCHMARKS = [
    {"product": "3M Filtek Supreme Ultra", "property": "flex_3pt", "value": 165.1, "range": (151.5, 178.7), "unit": "MPa", "method": "internal 3-point (TPP)", "evidence": "Mfr", "cite": "C17"},
    {"product": "3M Filtek Supreme Ultra", "property": "modulus", "value": 11.35, "range": None, "unit": "GPa", "method": "TPP", "evidence": "Mfr", "cite": "C17"},
    {"product": "3M Filtek Supreme Ultra", "property": "k_ic", "value": 1.84, "range": None, "unit": "MPa*m^0.5", "method": "notched short-rod (TPP)", "evidence": "Mfr", "cite": "C17"},
    {"product": "3M Filtek Supreme", "property": "hv", "value": 84.5, "range": (81.7, 87.3), "unit": "HV0.3", "method": "optical Vickers, dry 5 d", "evidence": "Indep", "cite": "ScougallVilchis2009"},
    {"product": "3M Filtek Z250", "property": "hv", "value": 87.2, "range": (84.4, 90.0), "unit": "HV0.3", "method": "optical Vickers, dry 5 d", "evidence": "Indep", "cite": "ScougallVilchis2009"},
    {"product": "Kuraray Clearfil AP-X", "property": "hv", "value": 112.0, "range": None, "unit": "HV0.3", "method": "optical Vickers, dry 5 d", "evidence": "Indep", "cite": "ScougallVilchis2009"},
    {"product": "Kuraray Clearfil Majesty", "property": "hv", "value": 57.7, "range": None, "unit": "HV0.3", "method": "optical Vickers, dry 5 d", "evidence": "Indep", "cite": "ScougallVilchis2009"},
]


@lru_cache(maxsize=1)
def load_direct_db() -> C.ConstituentDatabase:
    """Harness constituent database + CQ / EDMAB (data/dental/direct/direct_constituents.json)."""
    base = C.load_constituents()
    raw = json.loads((DIRECT_DATA / "direct_constituents.json").read_text())
    cits = {k: Citation(key=k, **v) for k, v in raw["citations"].items()}
    cons = {}
    for cid, c in raw["constituents"].items():
        cons[cid] = C.Constituent(id=cid, name=c["name"], kind=c["kind"], role=ComponentRole(c["role"]), aliases=c.get("aliases", []),
                                  cas=c.get("cas"), properties={k: C._parse_prop(v) for k, v in c["properties"].items()}, notes=c.get("notes", ""))
    return base.model_copy(update={"constituents": {**base.constituents, **cons}, "citations": {**base.citations, **cits}})


# ---------------------------------------------------------------------------
# Design variables and formulation builder
# ---------------------------------------------------------------------------
@dataclass
class DirectPoint:
    id: str
    phi_target: float  # cured inorganic filler volume fraction
    modality: Literal["bi", "tri"]
    zirconia_share: float  # nano-zirconia share of filler volume (radiopacity / hardness)
    nano_share: float  # nano-filler share (tri only)
    nano_kind: Literal["fumed silica", "nanocluster"]
    bisgma: float  # monomer weight shares (normalised)
    udma: float
    bisema: float
    tegdma: float
    cq_wt_resin: float = 0.4  # CQ wt% of the organic phase
    amine_ratio: float = 2.0  # EDMAB / CQ by weight
    glass: Literal["ba glass 0.7", "ba glass 0.4"] = "ba glass 0.7"  # fine glass of the bimodal pair
    notes: str = ""


def _filler_shares(p: DirectPoint) -> dict[str, float]:
    zr = p.zirconia_share
    nano = p.nano_share if p.modality == "tri" else 0.0
    glass = max(1.0 - zr - nano, 0.0)
    out = {"ba glass 1.5": 0.65 * glass, p.glass: 0.35 * glass}  # coarse:fine 65:35 by volume (as design.py)
    if zr > 0:
        out["nano zirconia"] = zr
    if nano > 0:
        out[p.nano_kind] = out.get(p.nano_kind, 0.0) + nano
    return out


def build_direct_formulation(p: DirectPoint, db=None, fillers: Optional[dict[str, float]] = None) -> Formulation:
    """Weigh out a paste hitting the target cured inorganic filler volume fraction (fixed-point on phi)."""
    db = db or load_direct_db()
    shares = fillers or _filler_shares(p)
    mons = {k: v for k, v in {"Bis-GMA": p.bisgma, "UDMA": p.udma, "Bis-EMA": p.bisema, "TEGDMA": p.tegdma}.items() if v > 0}
    tot = sum(mons.values())
    mons = {k: v / tot for k, v in mons.items()}
    cq, am, bht = p.cq_wt_resin / 100.0, p.amine_ratio * p.cq_wt_resin / 100.0, 0.0005
    treat = "3 wt% gamma-MPS (pre-silanised)"
    rho_c = {n: C.component_density(C.make_component(n, 0.1, db), db, "cured") for n in mons}
    rho_m = 1.0 / sum(w / rho_c[n] for n, w in mons.items())

    def weigh(phi_fill: float) -> Formulation:
        m_resin = (1.0 - phi_fill) * rho_m
        masses = [(n, w * m_resin * (1 - cq - am - bht), {}) for n, w in mons.items()]
        masses += [("CQ", cq * m_resin, {}), ("EDMAB", am * m_resin, {}), ("BHT", bht * m_resin, {})]
        for n, s in shares.items():
            tr = "10-MDP + silane (zirconia needs MDP)" if "zirconia" in n and "cluster" not in n else treat
            masses.append((n, phi_fill * s * db.lookup(n).get("density", Unit.G_PER_CM3), {"surface_treatment": tr}))
        mt = sum(m for _, m, _ in masses)
        comps = [C.make_component(n, m / mt, db, **kw) for n, m, kw in masses]
        s = sum(c.weight_fraction for c in comps)
        comps = [c.model_copy(update={"weight_fraction": c.weight_fraction / s}) for c in comps]
        proc = ProcessingRoute(method="direct_placement", wavelength_nm=WAVELENGTH_NM, layer_thickness=q(INCREMENT_MM * 1000.0, Unit.MICROMETER),
                               exposure_dose=q(IRRADIANCE_MW_CM2[0] * EXPOSURE_S[0], Unit.MJ_PER_CM2),
                               post_cure="none (in-situ light cure only)", notes="LED 1000-1200 mW/cm^2, 10-20 s per 2 mm increment")
        return Formulation(name=p.id, components=comps, processing=proc)

    phi_fill = p.phi_target
    for _ in range(4):
        f = weigh(phi_fill)
        phi_fill = min(max(phi_fill + (p.phi_target - C.filler_volume_fraction(f, db, "cured")), 0.05), 0.85)
    return C.with_volume_fraction(f, db)


def candidate_for(p: DirectPoint, f: Formulation) -> Candidate:
    return Candidate(id=f"direct_{p.id}", name=f"direct composite {p.id}", kind=CandidateKind.FORMULATION,
                     composition=Composition(formula="filled dimethacrylate resin (CQ/amine)", elements={}), formulation=f,
                     hypothesis="multimodal Ba-glass/nano filler at 60-70 vol% in a low-TEGDMA matrix gives hardness/stiffness above the best "
                                "direct composites while keeping the handling, depth of cure and shrinkage stress of a universal composite",
                     mechanism_tags=["filler_loading", "multimodal_packing", "direct_placement"], proposed_by="direct_design_agent")


# ---------------------------------------------------------------------------
# Direct-specific models
# ---------------------------------------------------------------------------
def cq_absorption_per_um(f: Formulation, db=None, eps: Optional[float] = None) -> float:
    """alpha_a = ln(10) eps_CQ c_CQ (1/um) at 470 nm; c in mol/L of paste (EDMAB non-absorbing)."""
    db = db or load_direct_db()
    rho = 1.0 / sum(c.weight_fraction / C.component_density(c, db, "uncured") for c in f.components)
    cq = db.lookup("CQ")
    e = cq.get("molar_absorptivity_470", Unit.DIMENSIONLESS) if eps is None else eps
    w = sum(c.weight_fraction for c in f.components if db.lookup(c.name).id == cq.id)
    conc = w * rho * 1000.0 / cq.get("molar_mass", Unit.AMU)
    return math.log(10.0) * e * conc * 1e-4


def predict_depth_of_cure(f: Formulation, radiant_exposure_mj_cm2: Optional[float] = None, n_samples: int = 400, seed: int = 0, db=None) -> PM.ModelPrediction:
    """ISO 4049 depth of cure (mm) = half the Jacobs cured length L = D_p ln(H/H_c) at 470 nm.
    1/D_p = CQ absorption + the property_models scattering term (anomalous-diffraction Q, transport fraction)."""
    db = db or load_direct_db()
    H = radiant_exposure_mj_cm2 or IRRADIANCE_MW_CM2[0] * EXPOSURE_S[0]
    classes = PM.filler_classes(f, db, "uncured")
    n0 = PM.resin_refractive_index(f, "uncured", db)
    cq = db.lookup("CQ").prop("molar_absorptivity_470")

    def doc(ec, ftr, eps):
        dp_um = 1.0 / (cq_absorption_per_um(f, db, eps) + PM.scattering_attenuation_per_um(classes, n0, WAVELENGTH_NM, ftr) + 1e-12)
        return 0.5 * dp_um * math.log(H / ec) / 1000.0 if H > ec else 0.0, dp_um

    v, dp = doc(EC_MJ_CM2[0], TRANSPORT_FRACTION[0], cq.value.value)
    rng = np.random.default_rng(seed)
    lo_e, hi_e = cq.range()
    s = [doc(math.exp(rng.uniform(math.log(EC_MJ_CM2[1]), math.log(EC_MJ_CM2[2]))), rng.uniform(*TRANSPORT_FRACTION[1:]),
             rng.uniform(lo_e, hi_e))[0] for _ in range(n_samples)]
    lo, hi = PM._pct(np.array(s))
    chk = CheckResult(name="iso4049_depth_of_cure", passed=bool(v >= ISO_DOC_MIN_MM), metric=v, tolerance=ISO_DOC_MIN_MM, unit=Unit.MILLIMETER,
                      details=f"ISO DoC {v:.2f} mm [{lo:.2f}-{hi:.2f}] vs >= {ISO_DOC_MIN_MM} mm at {H / 1000:.0f} J/cm^2, 470 nm")
    return PM.ModelPrediction(
        name=PropertyName.CURE_DEPTH_ESTIMATE, value=v, low=lo, high=hi, unit=Unit.MILLIMETER,
        interval="5-95 percentile of H_c (log-uniform), transport fraction and eps_CQ sampling",
        model="Jacobs working curve at 470 nm (CQ Beer-Lambert absorption + property_models anomalous-diffraction scattering); ISO 4049 DoC = L/2",
        equation="DoC = 0.5 D_p ln(H/H_c); 1/D_p = ln10 eps_CQ c_CQ + sum_i (3/2) phi_i Q_i f_tr / d_i",
        assumptions=[
            f"H = {H / 1000:.0f} J/cm^2 at the surface (1000-1200 mW/cm^2 x 10-20 s); tip distance, spectrum mismatch and cavity walls ignored",
            f"gel-point radiant exposure H_c = {EC_MJ_CM2[0] / 1000:.1f} J/cm^2 [{EC_MJ_CM2[1] / 1000:.1f}-{EC_MJ_CM2[2] / 1000:.1f}] ASSUMED (no verified CQ working curve)",
            "eps_CQ(470 nm) 28 L/mol/cm (S35), range 20-46 ASSUMED; amine and inhibitor do not absorb; CQ photobleaching ignored (conservative)",
            "refractive indices at 470 nm taken equal to the tabulated (sodium-D) values - dispersion ignored",
            "transport fraction f_tr 0.03-0.15 ASSUMED as in property_models.predict_cure_depth (dominant uncertainty)",
        ],
        citations=[CITES["S35"], CITES["ISO4049"], PM.CITES["Jacobs1992"], PM.CITES["GH1997"]],
        inputs={"H_mJ_cm2": H, "D_p_um": dp, "n_resin_uncured": n0, "alpha_cq_per_um": cq_absorption_per_um(f, db)},
        checks=[chk], conditions_note="470 nm LED, ISO 4049 scrape test",
    )


def shrinkage_stress(E_gpa: tuple[float, float, float], S_vol_pct: tuple[float, float, float], c_factor=C_FACTOR) -> dict:
    """Constrained post-gel shrinkage stress (MPa) (Davidson & Feilzer 1997 concept):
    sigma = E x (S_vol/3) x f_postgel x kappa(C) x (1 - relaxation), kappa = C/(C + C_half).
    Nominal uses the nominal inputs; low/high combine the extremes (conservative envelope)."""
    def s(E, S, fpg, C_, rel):
        return E * 1000.0 * (S / 100.0 / 3.0) * fpg * (C_ / (C_ + C_HALF)) * (1.0 - rel)
    v = s(E_gpa[0], S_vol_pct[0], POST_GEL_FRACTION[0], c_factor[0], RELAXATION[0])
    lo = s(E_gpa[1], S_vol_pct[1], POST_GEL_FRACTION[1], c_factor[1], RELAXATION[2])
    hi = s(E_gpa[2], S_vol_pct[2], POST_GEL_FRACTION[2], c_factor[2], RELAXATION[1])
    return {"value": v, "low": lo, "high": hi, "unit": "MPa",
            "model": "constrained post-gel shrinkage x composite modulus x constraint factor kappa(C) x (1 - relaxation) [S46]",
            "assumptions": [f"post-gel fraction {POST_GEL_FRACTION} ASSUMED", f"C-factor {c_factor}, kappa = C/(C+{C_HALF}) ASSUMED form",
                            f"relaxation/compliance {RELAXATION} ASSUMED", "final (not gel-time) modulus used: conservative"]}


def debond_probability(stress_mpa: float, aged: bool = True) -> tuple[float, float]:
    """(P that the shrinkage stress alone exceeds the dentin-adhesive interface strength, aged dentin median)."""
    wb = BD.tooth_interface("superficial_dentin", ADHESIVE, aged)
    p = 1.0 - math.exp(-((max(stress_mpa, 0.0) / wb.sigma0_mpa) ** wb.m))
    med = wb.sigma0_mpa * math.log(2.0) ** (1.0 / wb.m)
    return p, med


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class DPred:
    value: float
    low: float
    high: float
    unit: str
    model: str = ""
    notes: str = ""

    def ov(self, z: float = 1.645) -> ObjectiveValue:
        return ObjectiveValue(self.value, self.low, self.high, interval_z=z, note=self.notes)


@dataclass
class DirectEvaluation:
    point: DirectPoint
    formulation: Formulation
    phi_cured: float
    filler_wt: float
    preds: dict[str, DPred]
    constraints: dict[str, dict]
    violations: list[str]
    extra: dict = field(default_factory=dict)
    model_predictions: dict = field(default_factory=dict, repr=False)

    def summary(self) -> dict:
        return {"id": self.point.id, "design": asdict(self.point), "phi_cured": self.phi_cured, "filler_wt": self.filler_wt,
                "predictions": {k: asdict(v) for k, v in self.preds.items()}, "constraints": self.constraints, "violations": self.violations,
                "extra": self.extra, "recipe_wt": {c.name: c.weight_fraction for c in self.formulation.components}}


def _mp(m: PM.ModelPrediction, f=(1.0, 1.0, 1.0), notes: str = "") -> DPred:
    return DPred(m.value * f[0], m.low * f[1], m.high * f[2], m.unit.value, m.model, notes)


def evaluate_direct(p: DirectPoint, db=None, reference_viscosity: Optional[float] = None, n_samples: int = 200,
                    fillers: Optional[dict[str, float]] = None) -> DirectEvaluation:
    db = db or load_direct_db()
    f = build_direct_formulation(p, db, fillers)
    phi = C.filler_volume_fraction(f, db, "cured")
    E, nu, unc, method = effective_modulus(homogenization_spec(f, db))
    # in-situ DC (no post-cure) and its ratio to the direct-class basis
    dc = PM.predict_degree_of_conversion(f, post_cure_gain_pp=(0.0, 0.0, 0.0), db=db)
    dcr = (dc.value / DC_REF_DIRECT, dc.low / DC_REF_DIRECT, dc.high / DC_REF_DIRECT)
    hv = PM.predict_vickers_empirical(phi, dc_ratio=dcr[0], material_class="direct", label_load="HV0.2-HV0.5")
    hv_lo = PM.predict_vickers_empirical(phi, dc_ratio=min(dcr), material_class="direct").low
    hv_hi = PM.predict_vickers_empirical(phi, dc_ratio=max(dcr), material_class="direct").high
    # toughness / strength: matrix K_Ic scaled linearly with the DC ratio (ASSUMED, cf. Chung & Greener 1990 [S47])
    ts = toughness_spec(f, "gamma-MPS", db)
    ts["matrix_K_Ic"] = TG.CONSTITUENTS["matrix_dimethacrylate"]["K_Ic"] * float(np.clip(dcr[0], 0.8, 1.15))
    tk = TG.predict_toughness(ts)
    fs = TG.predict_flexural_strength(ts, toughness=tk)
    sorp = PM.predict_water_sorption(f, db)
    solu = PM.predict_water_solubility(f, db)
    shr = PM.predict_shrinkage(f, db)
    visc = PM.predict_viscosity(f, temperature_k=296.15, limit_pa_s=1e9, n_samples=n_samples, db=db)
    doc = predict_depth_of_cure(f, n_samples=n_samples, db=db)
    doc10 = predict_depth_of_cure(f, radiant_exposure_mj_cm2=IRRADIANCE_MW_CM2[1] * EXPOSURE_S[1], n_samples=n_samples, db=db)
    st = shrinkage_stress((E, unc["E_min"], unc["E_max"]), (shr.value, shr.low, shr.high))
    p_deb, dentin_med = debond_probability(st["value"])
    p_deb_hi, _ = debond_probability(st["high"])
    a7 = AGEING_7D_FACTOR
    preds = {
        "hv": DPred(hv.value, hv_lo, hv_hi, "HV", hv.model, "direct-class calibration (dry/5 d, HV0.2-0.5) x DC ratio"),
        "modulus": DPred(E, unc["E_min"], unc["E_max"], "GPa", method),
        "flex_3pt": DPred(fs["value"], fs["range"][0], fs["range"][1], "MPa", fs["method"], "24 h / dry basis"),
        "flex_3pt_7d_water": DPred(fs["value"] * a7[0], fs["range"][0] * a7[1], fs["range"][1] * a7[2], "MPa", fs["method"] + " x 7 d water factor",
                                   "AGEING_7D_FACTOR 0.91 [0.83-1.06] (strength_calibration)"),
        "k_ic": DPred(tk["value"], tk["range"][0], tk["range"][1], "MPa*m^0.5", tk["method"]),
        "sorption": _mp(sorp, notes="raw rule of mixtures on light-cured homopolymer data (appropriate DC basis for a direct material)"),
        "sorption_proxy": DPred(sorp.inputs["proxy_calibrated_value"], sorp.inputs["proxy_calibrated_low"], sorp.inputs["proxy_calibrated_high"],
                                "ug/mm^3", "paired ratio to Midas proxy x Midas 510(k)", "cross-check only (Midas is printed + post-cured)"),
        "solubility": _mp(solu),
        "shrinkage": _mp(shr),
        "shrinkage_stress": DPred(st["value"], st["low"], st["high"], "MPa", st["model"]),
        "dc": DPred(dc.value, dc.low, dc.high, "%", "volume-weighted homopolymer DC, no post-cure"),
        "depth_of_cure": _mp(doc, notes="20 J/cm^2 (1000 mW/cm^2 x 20 s ~ nominal 1100 x 20 s)"),
        "depth_of_cure_10s": _mp(doc10, notes="10 J/cm^2 (1000 mW/cm^2 x 10 s)"),
        "viscosity": _mp(visc, notes="Newtonian KD at 23 C, ~30 1/s; NO yield stress"),
    }
    cons, viol = {}, []

    def con(name, ok, detail):
        cons[name] = {"passed": bool(ok), "details": detail}
        if not ok:
            viol.append(f"{name}: {detail}")

    con("iso4049_depth_of_cure", doc.value >= ISO_DOC_MIN_MM, doc.checks[0].details)
    con("iso4049_flexural_min", fs["value"] * a7[0] >= ISO_FLEX_MIN_MPA, f"{fs['value'] * a7[0]:.0f} MPa (7 d water) vs >= {ISO_FLEX_MIN_MPA:.0f} (Type 1, intra-orally cured)")
    con("iso4049_sorption", sorp.value <= ISO_SORPTION_MAX, f"{sorp.value:.1f} ug/mm^3 vs <= {ISO_SORPTION_MAX}")
    con("iso4049_solubility", solu.value <= ISO_SOLUBILITY_MAX, f"{solu.value:.2f} ug/mm^3 vs <= {ISO_SOLUBILITY_MAX}")
    con("dc_in_situ_range", DC_DIRECT_RANGE[0] <= dc.value <= DC_DIRECT_RANGE[1] + 5.0, f"DC {dc.value:.1f} % vs direct range {DC_DIRECT_RANGE} % (+5 tolerance)")
    con("shrinkage_stress_vs_bond", st["value"] <= STRESS_BOND_FRACTION_MAX * dentin_med and p_deb <= DEBOND_P_MAX,
        f"sigma_s {st['value']:.1f} MPa vs {STRESS_BOND_FRACTION_MAX:.0%} of aged dentin uTBS median {dentin_med:.1f} MPa; P_debond {p_deb:.3f} (high-stress case {p_deb_hi:.3f})")
    con("paste_not_jammed", visc.inputs["phi_eff"] < 0.98 * visc.inputs["phi_m"], f"phi_eff {visc.inputs['phi_eff']:.3f} vs phi_m {visc.inputs['phi_m']:.3f}")
    if reference_viscosity is not None:
        r = visc.value / reference_viscosity
        con("handling_window", HANDLING_RATIO[0] <= r <= HANDLING_RATIO[1],
            f"eta/eta_ref {r:.2f} vs sculptable window {HANDLING_RATIO} relative to the Filtek-Supreme-Ultra-like proxy (model has no yield stress)")
    con("hardness_class_domain", all(c.passed for c in hv.checks), "; ".join(c.details for c in hv.checks))
    extra = {"dc_ratio": dcr, "E_method": method, "nu": nu, "k_ic_models": tk["models"], "phase_field_case": tk["phase_field_case"],
             "dentin_uTBS_aged_median": dentin_med, "P_debond_shrinkage": p_deb, "P_debond_shrinkage_high": p_deb_hi,
             "D_p_um": doc.inputs["D_p_um"], "n_resin_uncured": doc.inputs["n_resin_uncured"],
             "viscosity_ratio_to_ref": (visc.value / reference_viscosity) if reference_viscosity else None,
             "hv_milled_class_optimistic": hv.inputs["hv_milled_class_optimistic"], "shrinkage_stress_assumptions": st["assumptions"],
             "biaxial_equivalent_MPa": (fs["biaxial_equivalent"], *fs["biaxial_range"])}
    mps = {"hv": hv, "cure_depth": doc, "water_sorption": sorp, "water_solubility": solu, "shrinkage": shr, "viscosity": visc, "dc": dc,
           "toughness": tk, "flexural": fs}
    return DirectEvaluation(p, f, phi, C.filler_weight_fraction(f), preds, cons, viol, extra, mps)


# ---------------------------------------------------------------------------
# Reference proxies, design space, ranking
# ---------------------------------------------------------------------------
def filtek_proxy_point() -> tuple[DirectPoint, dict[str, float]]:
    """Filtek-Supreme-Ultra-like proxy: 63.3 vol% (TPP) zirconia-silica nanoclusters + ~40 nm silica in a
    Bis-GMA/UDMA/Bis-EMA/TEGDMA matrix.  Filler split and monomer shares are APPROXIMATE (not verified)."""
    p = DirectPoint(id="filtek_supreme_ultra_proxy", phi_target=0.633, modality="tri", zirconia_share=0.0, nano_share=0.2,
                    nano_kind="fumed silica", bisgma=0.25, udma=0.30, bisema=0.30, tegdma=0.15, notes="benchmark proxy, composition approximate")
    return p, {"nanocluster": 0.8, "fumed silica": 0.2}


@lru_cache(maxsize=1)
def reference_viscosity() -> float:
    p, fl = filtek_proxy_point()
    f = build_direct_formulation(p, load_direct_db(), fl)
    return PM.predict_viscosity(f, temperature_k=296.15, limit_pa_s=1e9, n_samples=1, db=load_direct_db()).value


def design_space(n: int = 72, seed: int = 20260928) -> list[DirectPoint]:
    rng = np.random.default_rng(seed)
    u = (np.argsort(rng.random((8, n)), axis=1).T + rng.random((n, 8))) / n  # Latin hypercube
    pts = []
    for i in range(n):
        m = rng.dirichlet([2.0, 2.0, 2.0, 1.2])  # Bis-GMA, UDMA, Bis-EMA, TEGDMA
        m[3] = 0.05 + 0.25 * m[3] / max(m.max(), 1e-9) * 0.5  # TEGDMA 5-17.5 %: allowed but penalised
        pts.append(DirectPoint(
            id=f"dlhs{i:03d}", phi_target=float(0.55 + 0.17 * u[i, 0]), modality=str(rng.choice(["bi", "tri"], p=[0.35, 0.65])),
            zirconia_share=float(0.12 * u[i, 1]), nano_share=float(0.05 + 0.15 * u[i, 2]), nano_kind=str(rng.choice(["fumed silica", "nanocluster"])),
            bisgma=float(m[0]), udma=float(m[1]), bisema=float(m[2]), tegdma=float(m[3]),
            cq_wt_resin=float(0.2 + 0.4 * u[i, 3]), amine_ratio=float(1.0 + 1.0 * u[i, 4]),
            glass=str(rng.choice(["ba glass 0.7", "ba glass 0.4"]))))
    return pts


def engineered_points() -> list[DirectPoint]:
    """Hand-designed points: Bis-EMA-rich matrix (resin n ~1.50-1.51 vs Ba glass 1.53 for depth of cure),
    low TEGDMA (sorption, shrinkage), nano-zirconia for radiopacity without micron-ZrO2 scattering."""
    out = []
    base = dict(modality="tri", zirconia_share=0.06, nano_share=0.10, nano_kind="fumed silica", bisgma=0.20, udma=0.35, bisema=0.35, tegdma=0.10,
                cq_wt_resin=0.4, amine_ratio=2.0)
    for phi in (0.60, 0.63, 0.66, 0.68, 0.70):
        out.append(DirectPoint(id=f"deng_phi{int(round(phi * 100))}", phi_target=phi, **base))
    out.append(DirectPoint(id="deng_phi66_cluster", phi_target=0.66, **dict(base, nano_kind="nanocluster", nano_share=0.15)))
    out.append(DirectPoint(id="deng_phi66_tegdma20", phi_target=0.66, **dict(base, bisema=0.25, tegdma=0.20)))
    out.append(DirectPoint(id="deng_phi66_nobisgma", phi_target=0.66, **dict(base, bisgma=0.0, udma=0.50, bisema=0.40)))
    return out


OBJECTIVES = [
    Objective("hv", "mechanics", "max", "HV"), Objective("modulus", "mechanics", "max", "GPa"),
    Objective("flex_3pt_7d_water", "mechanics", "max", "MPa"),
    Objective("k_ic", "toughness", "max", "MPa*m^0.5"),
    Objective("sorption", "durability", "min", "ug/mm^3"), Objective("shrinkage_stress", "durability", "min", "MPa"),
]


def rank(evals: list[DirectEvaluation]):
    ce = [CandidateEvaluation(e.point.id, {o.name: e.preds[o.name].ov() for o in OBJECTIVES}, list(e.violations)) for e in evals]
    return rank_candidates(ce, OBJECTIVES)


def choose_lead(evals: list[DirectEvaluation], ranking) -> DirectEvaluation:
    """Feasible, overall Pareto tier 0; maximise the minimum normalised margin over
    (HV / 90, E / 12 GPa, sigma_7d / 160 MPa, stress_ref / stress) - a stated tie-break, not a score."""
    feas = [e for e in evals if not e.violations and ranking.tiers_overall.get(e.point.id, 99) == 0]
    if not feas:
        feas = [e for e in evals if not e.violations] or evals
    ref = next((e for e in evals if e.point.id == "filtek_supreme_ultra_proxy"), None)
    s_ref = ref.preds["shrinkage_stress"].value if ref else 5.0

    def margin(e):
        pr = e.preds
        return min(pr["hv"].value / 90.0, pr["modulus"].value / 12.0, pr["flex_3pt_7d_water"].value / 160.0, s_ref / pr["shrinkage_stress"].value)
    return max(feas, key=margin)


def run_direct_sweep(n_lhs: int = 72, seed: int = 20260928, n_samples: int = 200):
    db = load_direct_db()
    eta_ref = reference_viscosity()
    fp, fl = filtek_proxy_point()
    evals = [evaluate_direct(fp, db, eta_ref, n_samples, fillers=fl)]
    evals += [evaluate_direct(p, db, eta_ref, n_samples) for p in engineered_points() + design_space(n_lhs, seed)]
    rk = rank(evals)
    return evals, rk, choose_lead(evals, rk)


# ---------------------------------------------------------------------------
# Benchmarks, verification, records
# ---------------------------------------------------------------------------
def benchmark_table(e: DirectEvaluation) -> list[dict]:
    """Lead vs Midas Restore (the user's target; printed crown material) and vs the best direct composites."""
    pr = e.preds
    rows = []

    def row(prop, ours, ref, ref_label, evidence, direction="max", note=""):
        lo_ok = ours.low >= ref if direction == "max" else ours.high <= ref
        med_ok = ours.value >= ref if direction == "max" else ours.value <= ref
        verdict = "beats (whole range)" if lo_ok else ("beats at median, range overlaps" if med_ok else "does not beat")
        rows.append({"property": prop, "ours": ours.value, "ours_low": ours.low, "ours_high": ours.high, "unit": ours.unit, "reference": ref,
                     "reference_product": ref_label, "evidence": evidence, "verdict": verdict, "note": note})

    row("Vickers hardness", pr["hv"], MIDAS["hv"], "Midas Restore 82 HV (load not stated)", "Mfr", note="loads differ: ours HV0.2-0.5 calibration")
    row("Vickers hardness", pr["hv"], 87.2, "Filtek Z250 87.2 HV0.3 (dry)", "Indep")
    row("Vickers hardness", pr["hv"], 112.0, "Clearfil AP-X 112 HV0.3 (dry)", "Indep")
    row("3-point flexural (7 d water)", pr["flex_3pt_7d_water"], MIDAS["flex_3pt_hi"], "Midas 510(k) 120-150 MPa (upper)", "510(k)")
    row("3-point flexural (24 h)", pr["flex_3pt"], 165.1, "Filtek Supreme Ultra 165.1 MPa", "Mfr")
    row("Flexural/Young's modulus", pr["modulus"], MIDAS["modulus"], "Midas 10.5 GPa", "Mfr")
    row("Flexural/Young's modulus", pr["modulus"], 11.35, "Filtek Supreme Ultra 11.35 GPa", "Mfr")
    row("K_Ic", pr["k_ic"], 1.84, "Filtek Supreme Ultra 1.84 (short-rod)", "Mfr")
    row("K_Ic", pr["k_ic"], 2.0, "project target 2.0", "target")
    row("Water sorption", pr["sorption"], MIDAS["sorption_lo"], "Midas 510(k) 22-24 ug/mm^3", "510(k)", "min")
    row("Water sorption", pr["sorption"], 12.0, "project target <= 12", "target", "min")
    return rows


def verification_checks(db=None) -> list[CheckResult]:
    """Analytic / benchmark reproduction checks that must pass before the design numbers are used."""
    db = db or load_direct_db()
    out = []
    # 1. Jacobs analytic limit: no filler -> D_p = 1 / (ln10 eps c) exactly
    comps = [C.make_component(n, w, db) for n, w in (("Bis-GMA", 0.4955), ("TEGDMA", 0.4955), ("CQ", 0.005), ("EDMAB", 0.0040))]
    f0 = Formulation(name="unfilled", components=comps, processing=ProcessingRoute(method="direct_placement", wavelength_nm=470.0))
    rho = 1.0 / sum(c.weight_fraction / C.component_density(c, db, "uncured") for c in f0.components)
    c_mol = 0.005 * rho * 1000.0 / 166.22
    dp_ref = 1.0 / (math.log(10.0) * 28.0 * c_mol) * 1e4  # um
    m = predict_depth_of_cure(f0, radiant_exposure_mj_cm2=20000.0, n_samples=5, db=db)
    doc_ref = 0.5 * dp_ref * math.log(20000.0 / EC_MJ_CM2[0]) / 1000.0
    out.append(CheckResult(name="jacobs_unfilled_analytic", passed=abs(m.inputs["D_p_um"] / dp_ref - 1) < 1e-6 and abs(m.value / doc_ref - 1) < 1e-6,  # 1e-12 regulariser in 1/D_p
                           metric=m.inputs["D_p_um"], tolerance=dp_ref, unit=Unit.MICROMETER,
                           details=f"D_p {m.inputs['D_p_um']:.1f} um vs analytic {dp_ref:.1f} um; DoC {m.value:.2f} vs {doc_ref:.2f} mm"))
    # 2. shrinkage-stress model: elastic fully-constrained limit is Hooke's law
    s = E_ = 10.0
    hooke = E_ * 1000.0 * (3.0 / 100.0 / 3.0)
    st = shrinkage_stress((E_, E_, E_), (3.0, 3.0, 3.0), c_factor=(1e12, 1e12, 1e12))
    lim = st["value"] / (POST_GEL_FRACTION[0] * (1.0 - RELAXATION[0]))
    out.append(CheckResult(name="shrinkage_stress_hooke_limit", passed=abs(lim / hooke - 1) < 1e-6, metric=lim, tolerance=hooke, unit=Unit.MPA,
                           details=f"kappa->1, f_pg = 1, no relaxation: {lim:.1f} MPa = E eps_lin {hooke:.1f} MPa (E {s} GPa, S 3 %)"))
    # 3. benchmark reproduction: Filtek-Supreme-Ultra-like proxy vs TPP / HV data
    fp, fl = filtek_proxy_point()
    e = evaluate_direct(fp, db, reference_viscosity(), 100, fillers=fl)
    for key, meas, lab in (("flex_3pt", 165.1, "Filtek Supreme Ultra 165.1 MPa (TPP)"), ("modulus", 11.35, "Filtek Supreme Ultra 11.35 GPa (TPP)"),
                           ("k_ic", 1.84, "Filtek Supreme Ultra 1.84 MPa m^0.5 (TPP)"), ("hv", 84.5, "Filtek Supreme 84.5 HV0.3 (Scougall-Vilchis 2009)")):
        pr = e.preds[key]
        out.append(CheckResult(name=f"filtek_proxy_{key}", passed=bool(pr.low <= meas <= pr.high), metric=pr.value, tolerance=meas,
                               details=f"proxy {pr.value:.3g} [{pr.low:.3g}-{pr.high:.3g}] {pr.unit} vs measured {lab}"))
    # 4. ISO numbers used here equal those in targets.json
    t = json.loads(TARGETS.read_text())["iso_requirements"]
    vals = {(r["property"], r["value"]) for r in t if not isinstance(r["value"], list)}
    ok = all(x in vals for x in (("flexural_strength_min", ISO_FLEX_MIN_MPA), ("water_sorption_max", ISO_SORPTION_MAX),
                                  ("water_solubility_max", ISO_SOLUBILITY_MAX), ("depth_of_cure_min", ISO_DOC_MIN_MM)))
    out.append(CheckResult(name="iso4049_limits_match_targets", passed=ok, details="80 MPa (Type 1, intra-oral), 40 / 7.5 ug/mm^3, DoC 1.5 mm vs targets.json"))
    return out


def _record(name: PropertyName, pr: DPred, unit: Unit, cand: Candidate, assumptions: list[str], citations=(), checks=()):
    rec = make_record(
        name=name, value=pr.value, unit=unit, candidate=cand,
        method=MethodSpec(name=pr.model[:200] or "model", engine="fracture_md.composites.direct_design", description=pr.model, assumptions=assumptions),
        origin=PropertyOrigin.DERIVED_MODEL, scale=PropertyScale.MACROSCOPIC,
        conditions=Conditions(temperature=q(310.15, Unit.KELVIN), loading="quasistatic", environment="model estimate, in-situ light-cured (no post-cure)"),
        uncertainty=[Uncertainty(kind=UncertaintyKind.REFERENCE_DISCREPANCY, value=0.5 * (pr.high - pr.low), unit=unit, interval="range",
                                 samples=[pr.low, pr.high], description=f"range [{pr.low:.4g}, {pr.high:.4g}]")],
        checks=list(checks), derivation=Derivation(model=pr.model or "model", equation=pr.model or "see module", assumptions=assumptions), notes=pr.notes)
    rec.citations = list(citations)
    return rec


def records_for(e: DirectEvaluation) -> list:
    cand = candidate_for(e.point, e.formulation)
    mp = e.model_predictions
    recs = [mp["hv"].model_copy(update={"value": e.preds["hv"].value, "low": e.preds["hv"].low, "high": e.preds["hv"].high}).to_record(cand),
            mp["cure_depth"].to_record(cand), mp["water_sorption"].to_record(cand), mp["water_solubility"].to_record(cand), mp["shrinkage"].to_record(cand)]
    dc_a = [f"no post-cure; DC ratio {e.extra['dc_ratio'][0]:.3f} to the direct basis {DC_REF_DIRECT} % applied to matrix K_Ic (ASSUMED linear)"]
    recs.append(TG.toughness_record(mp["toughness"], cand))
    recs.append(TG.flexural_record(mp["flexural"], cand))
    recs.append(_record(PropertyName.EFFECTIVE_YOUNGS_MODULUS, e.preds["modulus"], Unit.GPA, cand, ["homogenization.effective_modulus (MT x FFT correction)"]))
    recs.append(_record(PropertyName.DEGREE_OF_CONVERSION_ESTIMATE, e.preds["dc"], Unit.PERCENT, cand, dc_a + ["volume-weighted homopolymer DC"], [CITES["S34"]]))
    recs.append(_record(PropertyName.FILLER_VOLUME_FRACTION, DPred(e.phi_cured, e.phi_cured - 0.01, e.phi_cured + 0.01, "1", "wt->vol, cured densities, inorganic basis"),
                        Unit.DIMENSIONLESS, cand, ["silane mass counted as organic"]))
    return recs


# ===========================================================================
# FLOWABLE direct composite (target: 3M Filtek Supreme Flowable, 3M TDS)
# ===========================================================================
# Anchors from the user's 3M technical data sheet ("3M internal data"; test methods not public):
FLOWABLE_ANCHORS = {
    "filtek_supreme_flowable": {"cusp_deflection_um": 13.3, "k_ic": 1.4, "gloss_retention_pct": 66.0, "radiopacity_mmAl": 2.1},
    "flowable_average": {"cusp_deflection_um": 13.4, "k_ic": 1.5, "gloss_retention_pct": 43.1, "radiopacity_mmAl": 2.8},
}
FLOWABLE_DIRECTION = {"cusp_deflection_um": "min", "k_ic": "max", "gloss_retention_pct": "max", "radiopacity_mmAl": "max"}
# Flowable handling window, Newtonian-equivalent apparent viscosity at ~30 1/s, 23 C.  Literature values of
# commercial flowables span roughly 1-100 Pa s depending on shear rate (strongly shear-thinning); the window
# below is a DESIGN RULE (stated, not a measured product spec): the KD model output (no yield stress) must be
# 0.3-3x that of the model's own flowable-average proxy, i.e. RELATIVE, as for the sculptable pastes above.
FLOW_HANDLING_RATIO = (0.3, 3.0)
FLOW_PHI_RANGE = (0.35, 0.60)


@dataclass
class FlowPoint:
    id: str
    phi_target: float
    fillers: dict  # filler name -> volume share of the filler
    bisgma: float = 0.25
    udma: float = 0.35
    bisema: float = 0.15
    tegdma: float = 0.25
    cq_wt_resin: float = 0.4
    amine_ratio: float = 2.0
    notes: str = ""

    def as_direct(self) -> DirectPoint:
        return DirectPoint(id=self.id, phi_target=self.phi_target, modality="tri", zirconia_share=0.0, nano_share=0.0, nano_kind="nanocluster",
                           bisgma=self.bisgma, udma=self.udma, bisema=self.bisema, tegdma=self.tegdma, cq_wt_resin=self.cq_wt_resin,
                           amine_ratio=self.amine_ratio, notes=self.notes)


def _norm(d: dict) -> dict:
    d = {k: v for k, v in d.items() if v > 1e-9}
    s = sum(d.values())
    return {k: v / s for k, v in d.items()}


def flowable_proxies() -> list[FlowPoint]:
    """Benchmark proxies (compositions APPROXIMATE; used to calibrate/check the models, not claimed exact).

    * Filtek Supreme Flowable (3M): zirconia/silica nanoclusters + discrete ~20-75 nm silica + 4-11 nm zirconia +
      YbF3 (0.1-5 um), ~65 wt% filler, Bis-GMA/TEGDMA/Procrylat (Bis-EMA-like) resin - 3M product literature,
      split between filler types NOT public -> the YbF3 share is fixed by the radiopacity calibration (radiopacity.py).
    * flowable average: Ba-aluminoborosilicate glass ~0.7 um + fumed silica, ~60-65 wt% (typical of the six named
      products), Bis-GMA/UDMA/TEGDMA.
    """
    return [
        # 3M TDS: 46 vol% / ~65 wt% inorganic; clusters + discrete 20/75 nm silica + YbF3.  With the harness densities
        # even a YbF3-free cluster/silica mix gives 66 wt% at 46 vol%, so the density balance only allows ~2 % YbF3 by
        # filler volume; cluster:silica 1:1 ASSUMED.  Radiopacity is then PREDICTED (not fitted) and checked vs 2.1 mm Al.
        FlowPoint("filtek_supreme_flowable_proxy", 0.46, _norm({"nanocluster": 0.49, "fumed silica": 0.49, "ybf3": 0.02}),
                  bisgma=0.35, udma=0.0, bisema=0.30, tegdma=0.35, notes="benchmark proxy (composition approximate; Procrylat modelled as Bis-EMA)"),
        FlowPoint("flowable_average_proxy", 0.42, _norm({"ba glass 0.7": 0.88, "fumed silica": 0.12}),
                  bisgma=0.35, udma=0.25, bisema=0.0, tegdma=0.40, notes="benchmark proxy (composition approximate)"),
    ]


def flowable_space(n: int = 80, seed: int = 20260929) -> list[FlowPoint]:
    """Latin-hypercube flowable design space: 35-60 vol% inorganic filler of ZrO2-SiO2 nanoclusters,
    Ba glass 0.4/0.7 um, YbF3 (radiopacity), nano-ZrO2 and fumed silica; UDMA/Bis-EMA/Bis-GMA/TEGDMA matrix."""
    rng = np.random.default_rng(seed)
    u = (np.argsort(rng.random((7, n)), axis=1).T + rng.random((n, 7))) / n
    pts = []
    for i in range(n):
        yb, zr, fs = 0.02 + 0.16 * u[i, 1], 0.08 * u[i, 2], 0.03 + 0.12 * u[i, 3]
        rest = max(1.0 - yb - zr - fs, 0.0)
        cl = rest * u[i, 4]
        glass = str(rng.choice(["ba glass 0.7", "ba glass 0.4"]))
        m = rng.dirichlet([2.0, 2.5, 2.0, 2.0])
        teg = 0.12 + 0.23 * u[i, 5]
        m = m[:3] / m[:3].sum() * (1 - teg)
        pts.append(FlowPoint(f"flhs{i:03d}", float(FLOW_PHI_RANGE[0] + (FLOW_PHI_RANGE[1] - FLOW_PHI_RANGE[0]) * u[i, 0]),
                             _norm({"nanocluster": cl, glass: rest - cl, "ybf3": yb, "nano zirconia": zr, "fumed silica": fs}),
                             bisgma=float(m[0]), udma=float(m[1]), bisema=float(m[2]), tegdma=float(teg), cq_wt_resin=float(0.25 + 0.35 * u[i, 6])))
    return pts


def flowable_engineered() -> list[FlowPoint]:
    """Hand-designed nanocluster/YbF3 flowables (the concept: Filtek-like nano surface for polish, more YbF3 for
    radiopacity, more filler for K_Ic and less shrinkage, UDMA-rich low-TEGDMA matrix for sorption/shrinkage)."""
    out = []
    for phi in (0.48, 0.52, 0.55, 0.58):
        for yb in (0.10, 0.14):
            out.append(FlowPoint(f"fleng_phi{int(round(phi * 100))}_yb{int(round(yb * 100))}", phi,
                                 _norm({"nanocluster": 0.70 - yb, "fumed silica": 0.08, "nano zirconia": 0.06, "ybf3": yb, "ba glass 0.4": 0.16}),
                                 bisgma=0.20, udma=0.45, bisema=0.15, tegdma=0.20))
    return out


def materials_for_restoration(evals: list[DirectEvaluation]) -> dict:
    """E (homogenization.effective_modulus) and volumetric shrinkage (property_models.predict_shrinkage) for the
    restoration (cusp-deflection) agent."""
    out = {}
    for e in evals:
        pr = e.preds
        out[e.point.id] = {"E_GPa": pr["modulus"].value, "E_GPa_low": pr["modulus"].low, "E_GPa_high": pr["modulus"].high,
                           "nu": e.extra["nu"], "shrinkage_vol_pct": pr["shrinkage"].value, "shrinkage_vol_pct_low": pr["shrinkage"].low,
                           "shrinkage_vol_pct_high": pr["shrinkage"].high, "phi_inorganic_vol": e.phi_cured, "filler_wt": e.filler_wt,
                           "dc_pct": pr["dc"].value, "notes": e.point.notes}
    return out


def evaluate_flowable(fp: FlowPoint, db=None, n_samples: int = 100) -> DirectEvaluation:
    return evaluate_direct(fp.as_direct(), db, None, n_samples, fillers=fp.fillers)


# ---------------------------------------------------------------------------
# Flowable evaluation on the four 3M metrics (kept separate) + ISO 4049 + handling
# ---------------------------------------------------------------------------
from fracture_md.composites import polish as PL  # noqa: E402
from fracture_md.composites import radiopacity as RO  # noqa: E402

RESTORATION_PRED = ROOT / "data" / "dental" / "restoration" / "cusp_deflection_predictions.json"
ISO_FLEX_MIN_FLOW = 80.0  # ISO 4049:2019 Type 1 (claimed for occlusal surfaces, as Filtek Flowable's small occlusal indication); 50 MPa otherwise
ISO_RADIOPACITY_MIN = 1.0  # ISO 4049: >= same thickness of Al when radiopacity is claimed
ENAMEL_MMAL = 2.0  # ~2 mm Al per mm (textbook); model gives 1.90
FLOW_KEEP = ("iso4049_flexural_min", "iso4049_sorption", "iso4049_solubility", "dc_in_situ_range")
FLOW_CITES = {
    "3M_TDS": PL.CITES["3M_TDS"],
    "Thalacker2007": Citation(key="Thalacker2007", text="Thalacker C et al. Cusp deflection of flowable composites (aluminium block, 4 x 4 mm cavity, LVDT). CED-IADR 2007 abstract #0289 (as cited in the 3M TDS)"),
    "Lassila2019": Citation(key="Lassila2019", text="Lassila L, Keulemans F, Vallittu PK, Garoushi S. Characterization of a new fiber-reinforced flowable composite. Odontology 2019;107:342-352 (everX Flow K_Ic 2.6 MPa m^0.5)",
                            doi="10.1007/s10266-018-0405-y"),
}


def _crowding(visc_mp) -> float:
    return visc_mp.inputs["phi_eff"] / visc_mp.inputs["phi_m"]


@lru_cache(maxsize=1)
def flowable_calibration() -> dict:
    """Evaluate the two benchmark proxies once: gloss calibration, cusp-deflection scaling, DoC / crowding references."""
    db = load_direct_db()
    fil, avg = flowable_proxies()
    ef, ea = evaluate_flowable(fil, db, 60), evaluate_flowable(avg, db, 60)
    A = FLOWABLE_ANCHORS
    gcal = PL.calibrate({"filtek_supreme_flowable": (fil.fillers, A["filtek_supreme_flowable"]["gloss_retention_pct"]),
                         "flowable_average": (avg.fillers, A["flowable_average"]["gloss_retention_pct"])})
    # cusp-deflection FALLBACK scaling (used only if the restoration FE file is absent): delta = k E S_vol, k from the average anchor
    es_a = ea.preds["modulus"].value * ea.preds["shrinkage"].value
    k = A["flowable_average"]["cusp_deflection_um"] / es_a
    return {"gloss": gcal, "k_cusp": k, "doc_filtek": ef.preds["depth_of_cure"].value, "crowding_filtek": _crowding(ef.model_predictions["viscosity"]),
            "crowding_average": _crowding(ea.model_predictions["viscosity"]), "eta_average_23C": ea.preds["viscosity"].value,
            "filtek_eval": ef, "average_eval": ea}


def _load_fe() -> dict:
    if RESTORATION_PRED.exists():
        try:
            d = json.loads(RESTORATION_PRED.read_text())
            return d.get("predictions", d.get("candidates", d))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def cusp_deflection(e: DirectEvaluation, cal: dict, fe: Optional[dict] = None) -> DPred:
    fe = _load_fe() if fe is None else fe
    r = fe.get(e.point.id) if isinstance(fe, dict) else None
    if isinstance(r, dict) and "cusp_deflection_um" in r:
        return DPred(r["cusp_deflection_um"], r.get("lo", r.get("cusp_deflection_um_low", r["cusp_deflection_um"])),
                     r.get("hi", r.get("cusp_deflection_um_high", r["cusp_deflection_um"])), "um",
                     "FE (restoration agent): aluminium block 4x4 mm, 2 x 2 mm increments, post-gel shrinkage eigenstrain", "physics (FE), from data/dental/restoration")
    pr = e.preds
    k = cal["k_cusp"]
    v = k * pr["modulus"].value * pr["shrinkage"].value
    # range: model inputs' own ranges (E low/high, S low/high), not combined with calibration error
    return DPred(v, k * pr["modulus"].low * pr["shrinkage"].low, k * pr["modulus"].high * pr["shrinkage"].high, "um",
                 "scaling delta = k E S_vol, k calibrated on the flowable-average anchor 13.4 um (FALLBACK; FE not available)",
                 "calibrated scaling (labelled fallback)")


def evaluate_flowable_full(fp: FlowPoint, db=None, n_samples: int = 60, fe: Optional[dict] = None) -> DirectEvaluation:
    db = db or load_direct_db()
    cal = flowable_calibration()
    e = evaluate_flowable(fp, db, n_samples)
    rad = RO.predict_radiopacity(e.formulation, db)
    gl = PL.predict_gloss_retention(fp.fillers, cal["gloss"])
    cd = cusp_deflection(e, cal, fe)
    v23 = e.model_predictions["viscosity"]
    v37 = PM.predict_viscosity(e.formulation, temperature_k=310.15, limit_pa_s=1e9, n_samples=max(n_samples, 20), db=db)
    e.preds["radiopacity"] = DPred(rad.value, rad.low, rad.high, "mm Al", rad.model, "physics (X-ray attenuation), 1 mm specimen")
    e.preds["gloss_retention"] = DPred(gl.value, gl.low, gl.high, "%", gl.method, "calibrated empirical" + ("" if gl.in_domain else " - OUT OF DOMAIN"))
    e.preds["cusp_deflection"] = cd
    e.preds["viscosity_37C"] = DPred(v37.value, v37.low, v37.high, "Pa*s", v37.model, "Newtonian KD at 37 C")
    viol = [v for v in e.violations if v.split(":")[0] in FLOW_KEEP]
    cons = {k: v for k, v in e.constraints.items() if k in FLOW_KEEP}
    # ISO 4049 flexural for flowables re-stated at 80 MPa (Type 1) - same value as ISO_FLEX_MIN_MPA
    crowd = _crowding(v23)

    def con(name, ok, detail):
        cons[name] = {"passed": bool(ok), "details": detail}
        if not ok:
            viol.append(f"{name}: {detail}")
    doc10 = e.preds["depth_of_cure_10s"]
    con("iso4049_depth_of_cure_10s", doc10.value >= ISO_DOC_MIN_MM,
        f"model ISO DoC {doc10.value:.2f} mm at 10 J/cm^2 (1000 mW/cm^2 x 10 s, the TDS protocol) vs >= {ISO_DOC_MIN_MM} mm; model is NOT validated on flowables (see proxy checks)")
    eta_r = v23.value / cal["eta_average_23C"]
    con("flowable_handling", FLOW_HANDLING_RATIO[0] <= eta_r <= FLOW_HANDLING_RATIO[1] and crowd < 0.98,
        f"KD eta(23 C) {v23.value:.3g} Pa s = {eta_r:.2f} x flowable-average proxy (window {FLOW_HANDLING_RATIO}); crowding phi_eff/phi_m {crowd:.3f} < 0.98")
    con("iso4049_radiopacity", rad.value >= ISO_RADIOPACITY_MIN, f"{rad.value:.2f} mm Al vs >= {ISO_RADIOPACITY_MIN} (ISO 4049 claim) ")
    e.violations = viol
    e.constraints = cons
    e.extra.update({"crowding": crowd, "d_eff_wear_um": gl.d_eff_um, "gloss_in_domain": gl.in_domain, "density_cured": rad.density,
                    "viscosity_23C_Pa_s": v23.value, "viscosity_37C_Pa_s": v37.value, "cusp_source": cd.notes})
    return e


FLOW_OBJECTIVES = [
    Objective("cusp_deflection", "target_3m", "min", "um"), Objective("k_ic", "target_3m", "max", "MPa*m^0.5"),
    Objective("gloss_retention", "target_3m", "max", "%"), Objective("radiopacity", "target_3m", "max", "mm Al"),
    Objective("sorption", "durability", "min", "ug/mm^3"), Objective("solubility", "durability", "min", "ug/mm^3"),
]


def flowable_engineered_v2() -> list[FlowPoint]:
    """Concept of this run: ALL-NANO wear units (ZrO2-SiO2 clusters, optional discrete 20 nm ZrO2, no glass) for polish,
    YbF3 + ZrO2 for radiopacity above the flowable average, higher filler load than Filtek for K_Ic and lower
    shrinkage, UDMA/Bis-EMA-rich low-TEGDMA matrix (sorption)."""
    out = []
    mats = {"bema": (0.25, 0.0, 0.55, 0.20), "mix": (0.30, 0.10, 0.40, 0.20)}  # (Bis-GMA, UDMA, Bis-EMA, TEGDMA): resin n ~1.52 matches the clusters
    for mk, (bg, ud, be, tg) in mats.items():
        for phi in (0.48, 0.50, 0.52, 0.54, 0.56):
            for yb in (0.05, 0.08, 0.11):
                for zr in (0.0, 0.04):
                    out.append(FlowPoint(f"flnano_{mk}_phi{int(round(phi * 100))}_yb{int(round(yb * 100))}_zr{int(round(zr * 100))}", phi,
                                         _norm({"nanocluster": 1.0 - yb - zr, "nano zirconia": zr, "ybf3": yb}),
                                         bisgma=bg, udma=ud, bisema=be, tegdma=tg, notes="all-nano concept"))
    return out


def run_flowable_sweep(n_lhs: int = 80, seed: int = 20260929, n_samples: int = 60):
    db = load_direct_db()
    fe = _load_fe()
    pts = flowable_proxies() + flowable_engineered() + flowable_engineered_v2() + flowable_space(n_lhs, seed)
    evals = [evaluate_flowable_full(p, db, n_samples, fe) for p in pts]
    ce = [CandidateEvaluation(e.point.id, {o.name: e.preds[o.name].ov() for o in FLOW_OBJECTIVES}, list(e.violations)) for e in evals]
    rk = rank_candidates(ce, FLOW_OBJECTIVES)
    return evals, rk, choose_flowable_lead(evals)


def _margins(e: DirectEvaluation, anchor: str) -> dict:
    A = FLOWABLE_ANCHORS[anchor]
    pr = e.preds
    return {"cusp_deflection_um": A["cusp_deflection_um"] / pr["cusp_deflection"].value, "k_ic": pr["k_ic"].value / A["k_ic"],
            "gloss_retention_pct": pr["gloss_retention"].value / A["gloss_retention_pct"], "radiopacity_mmAl": pr["radiopacity"].value / A["radiopacity_mmAl"]}


def choose_flowable_lead(evals: list[DirectEvaluation]) -> DirectEvaluation:
    """Feasible (no violations), not a proxy; maximise the minimum over the four metrics of the ratio to Filtek Supreme
    Flowable (a stated tie-break: 'beat Filtek on all four by the largest common margin'); ties -> the average anchors."""
    feas = [e for e in evals if not e.violations and not e.point.id.endswith("_proxy")] or [e for e in evals if not e.point.id.endswith("_proxy")]
    return max(feas, key=lambda e: (round(min(_margins(e, "filtek_supreme_flowable").values()), 3), min(_margins(e, "flowable_average").values())))


def flowable_verdicts(e: DirectEvaluation) -> list[dict]:
    keys = {"cusp_deflection_um": "cusp_deflection", "k_ic": "k_ic", "gloss_retention_pct": "gloss_retention", "radiopacity_mmAl": "radiopacity"}
    kind = {"cusp_deflection_um": "calibrated scaling / FE", "k_ic": "mesoscale simulation-informed toughening model (toughening.predict_toughness)",
            "gloss_retention_pct": "calibrated empirical (two anchors)", "radiopacity_mmAl": "physics (X-ray attenuation)"}
    rows = []
    for m, pk in keys.items():
        pr = e.preds[pk]
        d = FLOWABLE_DIRECTION[m]
        for anc in ("filtek_supreme_flowable", "flowable_average"):
            ref = FLOWABLE_ANCHORS[anc][m]
            whole = pr.low > ref if d == "max" else pr.high < ref
            med = pr.value > ref if d == "max" else pr.value < ref
            rows.append({"metric": m, "vs": anc, "ours": pr.value, "low": pr.low, "high": pr.high, "unit": pr.unit, "reference": ref, "better_is": "higher" if d == "max" else "lower",
                         "verdict": "beats (whole range)" if whole else ("beats at nominal, range overlaps" if med else "does not beat"),
                         "method_type": kind[m] if m != "cusp_deflection_um" else pr.notes})
    return rows


def flowable_proxy_checks() -> list[CheckResult]:
    """Benchmark reproduction on the Filtek / average proxies (radiopacity and K_Ic are predictions, not fits)."""
    cal = flowable_calibration()
    out = []
    for lab, e in (("filtek_supreme_flowable", cal["filtek_eval"]), ("flowable_average", cal["average_eval"])):
        db = load_direct_db()
        r = RO.predict_radiopacity(e.formulation, db)
        A = FLOWABLE_ANCHORS[lab]
        out.append(CheckResult(name=f"radiopacity_{lab}_proxy", passed=bool(r.low <= A["radiopacity_mmAl"] <= r.high), metric=r.value, tolerance=A["radiopacity_mmAl"],
                               unit=Unit.MILLIMETER, details=f"{r.value:.2f} [{r.low:.2f}-{r.high:.2f}] vs 3M TDS {A['radiopacity_mmAl']} mm Al, error {100 * (r.value / A['radiopacity_mmAl'] - 1):+.0f} %"))
        k = e.preds["k_ic"]
        out.append(CheckResult(name=f"k_ic_{lab}_proxy", passed=bool(k.low <= A["k_ic"] <= k.high), metric=k.value, tolerance=A["k_ic"], unit=Unit.MPA_SQRT_M,
                               details=f"{k.value:.2f} [{k.low:.2f}-{k.high:.2f}] vs 3M TDS {A['k_ic']} MPa m^0.5, error {100 * (k.value / A['k_ic'] - 1):+.0f} %"))
    ef = cal["filtek_eval"]
    cd = cal["k_cusp"] * ef.preds["modulus"].value * ef.preds["shrinkage"].value
    out.append(CheckResult(name="cusp_scaling_filtek_second_anchor", passed=abs(cd / 13.3 - 1) < 0.10, metric=cd, tolerance=13.3, unit=Unit.MICROMETER,
                           details=f"k E S calibrated on 13.4 um (average) predicts Filtek proxy {cd:.2f} vs 13.3 um, error {100 * (cd / 13.3 - 1):+.1f} %"))
    fe = _load_fe()
    r = fe.get("filtek_supreme_flowable_proxy") if isinstance(fe, dict) else None
    if isinstance(r, dict) and "cusp_deflection_um" in r:
        v, lo, hi = r["cusp_deflection_um"], r.get("lo", r["cusp_deflection_um"]), r.get("hi", r["cusp_deflection_um"])
        out.append(CheckResult(name="cusp_fe_filtek_second_anchor", passed=bool(lo <= 13.3 <= hi), metric=v, tolerance=13.3, unit=Unit.MICROMETER,
                               details=f"restoration FE (calibrated on the 13.4 um average anchor) predicts Filtek proxy {v:.2f} [{lo:.2f}-{hi:.2f}] vs 13.3 um, error {100 * (v / 13.3 - 1):+.0f} %"))
    out += PL.verification_checks(cal["gloss"])
    return out
