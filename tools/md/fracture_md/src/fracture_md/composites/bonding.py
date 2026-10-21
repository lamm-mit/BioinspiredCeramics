"""Bonding of a printed composite crown to the tooth: the bonded assembly, literature-calibrated
Weibull bond-strength distributions per interface, a weakest-link assembly estimate, and the
effect of the formulation (exposed filler fraction, filler chemistry vs primer) on the
cement-restoration interface.  See docs/dental/BONDING.md.

Assembly (indirect crown), numbering as in RESEARCH_BRIEF.md section 5.3::

    tooth (enamel margin + dentin core)            <- the PULP is not a bonding substrate
      | interface 1: tooth <-> adhesive (hybrid layer / etched enamel / 10-MDP-Ca salt)
    adhesive (universal, 10-MDP)
      | interface 2: adhesive <-> resin cement (co-polymerised)
    dual-cure resin cement, ~50-100 um
      | interface 3: cement <-> treated intaglio (50 um Al2O3 @ ~1 bar + silane/10-MDP primer)
    printed composite crown

All values are MODEL_ESTIMATE; literature inputs are stated with their test method, because
shear (SBS/micro-shear) and tensile (uTBS) numbers are not interchangeable.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Sequence

import numpy as np
from pydantic import Field

from fracture_md.composites import constituents as C
from fracture_md.composites.property_models import ModelPrediction
from fracture_md.schema.common import CheckResult, Citation, StrictModel
from fracture_md.schema.formulation import Formulation
from fracture_md.schema.properties import PropertyName
from fracture_md.units import Unit

CIT = {
    "B46": Citation(key="B46", text="De Munck J et al. Meta-analytical review of parameters involved in dentin bonding. J Dent Res 2012;91:351-357", doi="10.1177/0022034511431251"),
    "B36": Citation(key="B36", text="Tang C et al. J Adhes Dent 2024 (dentin uTBS immediate vs 50,000 thermocycles)", doi="10.3290/j.jad.b4949669"),
    "B47": Citation(key="B47", text="Pouyanfar H et al. Open Access Maced J Med Sci 2018 (enamel uTBS: Scotchbond Universal + etch 65.8 +/- 32.1; Clearfil SE 44.9 +/- 15.9 MPa)", doi="10.3889/oamjms.2018.427"),
    "B37": Citation(key="B37", text="ISO/TS 11405:2015 (CV of micro-tests typically 20-50%; Weibull needs n >= 15)"),
    "B41": Citation(key="B41", text="De Munck J et al. Dent Mater 2013;29:635-644 (Weibull m 2.7-4.8 for uTBS vs 3.8-11.5 for chevron-notch)", doi="10.1016/j.dental.2013.03.010"),
    "B18": Citation(key="B18", text="Hammamy M et al. Materials 2025;18:265 (SprintRay Ceramic Crown: APA 2 bar + Scotchbond Universal Plus 35.4 MPa vs 19.5 unabraded; APA + silane 18.7 vs 6.8)", doi="10.3390/ma18020265"),
    "S12": Citation(key="S12", text="Chen J et al. J Prosthodont Res 2026 (Ceramic Crown SBS: APA 0.1 MPa + Adhese Universal/Variolink Esthetic DC 15.1 (24 h) / 14.8 (20k TC) MPa)", doi="10.2186/jpr.JPR_D_25_00232"),
    "B19": Citation(key="B19", text="Dederichs M et al. J Dent 2025;155:105621 (Formlabs Permanent Crown + RelyX Unicem 2: APA 1/2/3 bar 25.6/28.1/30.2 vs glass beads 10.1 MPa)", doi="10.1016/j.jdent.2025.105621"),
    "B50": Citation(key="B50", text="Kang et al. Polymers 2023;15:1390 (TC-80: 24.7 -> 22.5 MPa after 10k TC)", doi="10.3390/polym15061390"),
    "B51": Citation(key="B51", text="Sacu et al. J Prosthodont 2026 (Crowntec: 110 um APA 14.1; HF 9.5; untreated 7.1 MPa)", doi="10.1111/jopr.70173"),
    "B23": Citation(key="B23", text="Montenbruck M et al. J Prosthet Dent 2025;134:842.e1 (Crowntec/3Delta Crown + Variolink Esthetic LC + Adhese Universal on bovine dentin: 6.57-10.58 MPa after 10k TC, mostly adhesive)", doi="10.1016/j.prosdent.2025.04.043"),
    "B35": Citation(key="B35", text="Braga RR et al. Adhesion to tooth structure: a critical review of 'macro' test methods. Dent Mater 2010;26:e38-e49 (shear tests concentrate stress; cohesive substrate failures)", doi="10.1016/j.dental.2009.11.150"),
    "B43": Citation(key="B43", text="Soares FZM et al. Dent Mater 2016;32:1385 (bovine vs human teeth: no difference, meta-analysis)", doi="10.1016/j.dental.2016.09.019"),
    "B28": Citation(key="B28", text="Yoshida Y et al. J Dent Res 2000;79:709 (adhesion-decalcification concept)", doi="10.1177/00220345000790020301"),
    "B30": Citation(key="B30", text="Yoshida Y et al. J Dent Res 2012;91:376 (MDP-Ca nano-layering)", doi="10.1177/0022034512437375"),
    "B33": Citation(key="B33", text="Yoshihara K et al. Dent Mater 2016;32:1218 (silane in universal adhesives ineffective unless fresh)", doi="10.1016/j.dental.2016.07.002"),
    "S43": Citation(key="S43", text="Lung CYK, Matinlinna JP. Dent Mater 2012;28:467 (zirconia bonds poorly to silane; MDP needed)", doi="10.1016/j.dental.2012.02.009"),
    "B11": Citation(key="B11", text="Murray PE et al. Int Endod J 2003;36:33 (odontoblast loss vs remaining dentin thickness; >= 0.5 mm RDT)", doi="10.1046/j.0143-2885.2003.00609.x"),
    "B13": Citation(key="B13", text="Zach L, Cohen G. Oral Surg 1965;19:515 (+5.5 C intrapulpal threshold; primary not accessed)", doi="10.1016/0030-4220(65)90015-0"),
    "B15": Citation(key="B15", text="Schweikl H, Spagnuolo G, Schmalz G. J Dent Res 2006;85:870 (TEGDMA/HEMA: GSH depletion, ROS, apoptosis)", doi="10.1177/154405910608501001"),
    "B10": Citation(key="B10", text="Costa CAS, Hebling J, Hanks CT. Dent Mater 2000;16:188 (adhesives on exposed pulp: persistent inflammation)", doi="10.1016/s0109-5641(00)00008-7"),
    "B16": Citation(key="B16", text="Uctasli et al. Eur J Oral Sci 2026 (ISO 10993-5 >= 70% viability; ISO 7405 dentin barrier 0.4 mm)", doi="10.1111/eos.70055"),
    "B54": Citation(key="B54", text="Tseng et al. J Dent Sci 2025;20:410 (CZM: enamel-composite sigma_c 34.5 MPa, G_c 50 J/m^2; dentin-composite 17.0 MPa, 10 J/m^2)", doi="10.1016/j.jds.2024.05.033"),
    "B42": Citation(key="B42", text="Kagaoan et al. J Dent 2024;144:104873 (printed crown-cement interfacial fracture energy 1.77 +/- 0.96 J/m^2)", doi="10.1016/j.jdent.2024.104873"),
    "B24": Citation(key="B24", text="May LG et al. Dent Mater 2012;28:e99 (cement thickness 50 vs 500 um: 673.5 vs 300.6 N)", doi="10.1016/j.dental.2012.04.033"),
}


# ---------------------------------------------------------------------------
# Weibull helpers
# ---------------------------------------------------------------------------
def weibull_m_from_cv(cv: float) -> float:
    """Weibull modulus with the given coefficient of variation (solves CV(m) = cv)."""
    def cv_of(m):
        g1, g2 = math.gamma(1 + 1 / m), math.gamma(1 + 2 / m)
        return math.sqrt(g2 - g1 * g1) / g1
    lo, hi = 0.3, 200.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if cv_of(mid) > cv:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def weibull_scale_from_mean(mean: float, m: float) -> float:
    return mean / math.gamma(1 + 1 / m)


class WeibullBond(StrictModel):
    """Bond-strength distribution of one interface: P_f(s) = 1 - exp(-(s/sigma0)^m)."""

    interface: str
    sigma0_mpa: float
    m: float
    test_basis: Literal["uTBS", "SBS", "micro-shear", "tensile_assembly", "assumed"]
    source_mean_mpa: Optional[float] = None
    source_cv: Optional[float] = None
    citations: list[Citation] = Field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_mean_cv(cls, interface: str, mean: float, cv: float, **kw) -> "WeibullBond":
        m = weibull_m_from_cv(cv)
        return cls(interface=interface, sigma0_mpa=weibull_scale_from_mean(mean, m), m=m, source_mean_mpa=mean, source_cv=cv, **kw)

    def survival(self, s: np.ndarray) -> np.ndarray:
        return np.exp(-((np.asarray(s) / self.sigma0_mpa) ** self.m))

    def mean(self) -> float:
        return self.sigma0_mpa * math.gamma(1 + 1 / self.m)

    def quantile(self, p: float) -> float:
        return self.sigma0_mpa * (-math.log(1 - p)) ** (1 / self.m)


class AssemblyResult(StrictModel):
    median_mpa: float
    mean_mpa: float
    p10_mpa: float
    p90_mpa: float
    origin_probability: dict[str, float]  # probability that each interface fails first
    weakest_interface: str


def weakest_link(links: Sequence[WeibullBond], s_max: Optional[float] = None, n: int = 4000) -> AssemblyResult:
    """Series system (all interfaces carry the same nominal stress in a uTBS stick):
    S(s) = prod_i S_i(s).  Origin probabilities P_i = int f_i(s) prod_{j!=i} S_j(s) ds."""
    s_max = s_max or 3.0 * max(l.sigma0_mpa for l in links)
    s = np.linspace(0.0, s_max, n)
    Ss = np.array([l.survival(s) for l in links])
    S = np.prod(Ss, axis=0)
    F = 1.0 - S
    pdf_total = np.gradient(F, s)
    median = float(np.interp(0.5, F, s))
    mean = float(np.trapezoid(S, s))
    p10, p90 = float(np.interp(0.1, F, s)), float(np.interp(0.9, F, s))
    origin = {}
    for i, l in enumerate(links):
        fi = np.gradient(1.0 - Ss[i], s)
        others = np.prod(np.delete(Ss, i, axis=0), axis=0) if len(links) > 1 else np.ones_like(s)
        origin[l.interface] = float(np.trapezoid(fi * others, s))
    tot = sum(origin.values()) or 1.0
    origin = {k: v / tot for k, v in origin.items()}
    del pdf_total
    return AssemblyResult(median_mpa=median, mean_mpa=mean, p10_mpa=p10, p90_mpa=p90, origin_probability=origin,
                          weakest_interface=max(origin, key=origin.get))


# ---------------------------------------------------------------------------
# Literature-calibrated interface distributions
# ---------------------------------------------------------------------------
# Dentin/enamel uTBS (interface 1) by adhesive class: (1-day mean, aged mean) MPa; aged = 1 year
# water (De Munck 2012 meta-model, SiC-ground) or 50k TC (Tang 2024) as stated.
ADHESIVE_CLASSES: dict[str, dict] = {
    "3-step etch-and-rinse (OptiBond FL)": {"dentin": (49.7, 44.8), "aged": "1 year water", "cite": ["B46", "B36"]},
    "2-step self-etch (Clearfil SE Bond)": {"dentin": (45.4, 38.6), "aged": "1 year water", "cite": ["B46", "B36"]},
    "universal, etch-and-rinse mode (Scotchbond Universal)": {"dentin": (48.3, 34.5), "aged": "50k thermocycles", "cite": ["B36"]},
    "universal, self-etch mode (Scotchbond Universal)": {"dentin": (24.1, 28.6), "aged": "50k thermocycles", "cite": ["B36"]},
    "1-step self-etch (class mean)": {"dentin": (34.4, 23.2), "aged": "1 year water", "cite": ["B46"]},
}
ENAMEL_ETCHED_MEAN = 65.8  # Scotchbond Universal + H3PO4 etch [B47]; Clearfil SE 44.9
ENAMEL_SELF_ETCH_MEAN = 44.9
DEEP_DENTIN_FACTOR = (0.8, 0.65, 0.9)  # ASSUMED: deep dentin has more/wider tubules and water -> lower uTBS (brief 6.2); no verified number
UTBS_CV = (0.35, 0.20, 0.50)  # ISO/TS 11405 & ADM: CV 20-50%

# Interface 3 (cement -> printed resin), shear-type tests on SprintRay Ceramic Crown / printed resins.
INTAGLIO_PROTOCOLS: dict[str, dict] = {
    "untreated": {"mean": 7.0, "aged_factor": 0.9, "basis": "SBS", "cite": ["B18", "B51"], "primer": "none",
                  "notes": "6.8 (Ceramic Crown unabraded + silane), 7.1 (Crowntec untreated); adhesive failures"},
    "APA 50 um 1 bar + Adhese Universal + Variolink Esthetic DC": {"mean": 15.1, "aged_factor": 0.98, "basis": "SBS", "cite": ["S12"],
                  "primer": "universal_adhesive_silane", "notes": "SprintRay's recommended protocol; 14.8 after 20k TC"},
    "APA 50 um + silane primer": {"mean": 18.7, "aged_factor": 0.91, "basis": "SBS", "cite": ["B18"], "primer": "silane",
                  "notes": "aged factor from TC-80 24.7 -> 22.5 after 10k TC [B50] (ASSUMED transferable)"},
    "APA 50 um 2 bar + silane/MDP universal (Scotchbond Universal Plus)": {"mean": 35.4, "aged_factor": 1.0, "basis": "SBS", "cite": ["B18"],
                  "primer": "silane_mdp", "notes": "value already after 3 months water"},
    "APA 50 um 2 bar + self-adhesive cement (RelyX Unicem 2)": {"mean": 28.1, "aged_factor": 0.91, "basis": "SBS", "cite": ["B19"],
                  "primer": "none", "notes": "Formlabs Permanent Crown; 25.6/28.1/30.2 at 1/2/3 bar"},
    "APA + Clearfil Ceramic Primer Plus + Panavia V5": {"mean": 42.1, "aged_factor": 1.0, "basis": "SBS", "cite": [], "primer": "silane_mdp",
                  "notes": "SprintRay's best protocol on Ceramic Crown, MANUFACTURER CLAIM (brief section 5); ageing regime not stated - aged_factor 1.0 is ASSUMED"},
}
# Relative chemical compatibility of primer with exposed filler (1 = forms a stable bond).
# Expert judgement from the chemistry in brief 5.4 (silane bonds Si-OH glass/silica; zirconia
# needs 10-MDP), NOT measured coefficients.
PRIMER_COMPATIBILITY = {
    "none": {"ba_glass": 0.0, "silica": 0.0, "fumed_silica": 0.0, "zirconia": 0.0, "zirconia_silica": 0.0, "ybf3": 0.0, "unknown": 0.0},
    "silane": {"ba_glass": 1.0, "silica": 1.0, "fumed_silica": 1.0, "zirconia": 0.2, "zirconia_silica": 0.7, "ybf3": 0.2, "unknown": 0.5},
    "universal_adhesive_silane": {"ba_glass": 0.6, "silica": 0.6, "fumed_silica": 0.6, "zirconia": 0.8, "zirconia_silica": 0.7, "ybf3": 0.3, "unknown": 0.5},
    "silane_mdp": {"ba_glass": 1.0, "silica": 1.0, "fumed_silica": 1.0, "zirconia": 1.0, "zirconia_silica": 1.0, "ybf3": 0.4, "unknown": 0.7},
}
REFERENCE_SURFACE_FILLER = 0.33  # ASSUMED effective bondable filler fraction of SprintRay Ceramic Crown (>50 wt% Si-based)
ASSEMBLY_FACTOR = (0.8, 0.6, 0.98)  # interface-level tests -> whole-assembly tensile; calibrated on Montenbruck 2025 [B23]:
# measured 6.57-10.58 MPa / interface-level estimate 10.8 MPa for a Crowntec-like resin -> 0.61-0.98


def bondable_surface_fraction(formulation: Formulation, primer: str, db=None) -> float:
    """Area-weighted exposed filler fraction x primer compatibility.  After air abrasion the
    exposed area fraction of filler is taken equal to the filler volume fraction (stereology:
    A_A = V_V for a random section); each filler type's share is weighted by its compatibility."""
    db = db or C.load_constituents()
    vf = C.formulation_volume_fractions(formulation, db, "cured")
    comp = PRIMER_COMPATIBILITY[primer]
    out = 0.0
    for c in formulation.components:
        if c.role.value != "filler":
            continue
        k = db.find(c.name)
        ftype = (k.filler_type if k else None) or "unknown"
        out += vf[c.name] * comp.get(ftype, comp["unknown"])
    return out


def cement_restoration_bond(
    formulation: Formulation,
    protocol: str = "APA 50 um 2 bar + silane/MDP universal (Scotchbond Universal Plus)",
    aged: bool = True,
    cohesive_limit_mpa: Optional[float] = None,
    db=None,
) -> tuple[WeibullBond, float, float, str]:
    """Interface-3 bond distribution for our formulation.  Literature mean for the protocol
    (Ceramic Crown / printed resins) scaled by (phi_bond / phi_bond,ref)^0.5, clipped to
    [0.7, 1.3], ASSUMED exponent; capped by the restoration's cohesive strength if given.
    Returns (distribution, low mean, high mean, note)."""
    p = INTAGLIO_PROTOCOLS[protocol]
    base = p["mean"] * (p["aged_factor"] if aged else 1.0)
    if p["primer"] == "none":
        scale = 1.0
        note = "no primer: bond independent of filler chemistry (micromechanical + cement only)"
    else:
        fb = bondable_surface_fraction(formulation, p["primer"], db)
        ref = REFERENCE_SURFACE_FILLER * PRIMER_COMPATIBILITY[p["primer"]]["silica"]
        scale = min(1.3, max(0.7, (fb / ref) ** 0.5 if ref > 0 else 1.0))
        note = f"bondable filler fraction {fb:.2f} vs reference {ref:.2f} -> scale {scale:.2f}"
    mean = base * scale
    capped = False
    if cohesive_limit_mpa is not None and mean > cohesive_limit_mpa:
        mean, capped = cohesive_limit_mpa, True
        note += f"; capped at restoration cohesive limit {cohesive_limit_mpa:.0f} MPa (failure moves into the printed resin)"
    wb = WeibullBond.from_mean_cv(
        "3: cement-restoration", mean, UTBS_CV[0], test_basis="SBS" if p["basis"] == "SBS" else "uTBS",
        citations=[CIT[k] for k in p["cite"]], notes=f"{protocol}; {p['notes']}; {note}",
    )
    return wb, mean * 0.75, mean * 1.2, note + (" (capped)" if capped else "")


def tooth_interface(substrate: Literal["enamel", "superficial_dentin", "deep_dentin"], adhesive: str, aged: bool = True) -> WeibullBond:
    a = ADHESIVE_CLASSES[adhesive]
    if substrate == "enamel":
        mean = ENAMEL_ETCHED_MEAN if "etch-and-rinse" in adhesive else ENAMEL_SELF_ETCH_MEAN
        cites = [CIT["B47"]]
        cv = 0.45
    else:
        mean = a["dentin"][1 if aged else 0]
        if substrate == "deep_dentin":
            mean *= DEEP_DENTIN_FACTOR[0]
        cites = [CIT[k] for k in a["cite"]]
        cv = UTBS_CV[0]
    return WeibullBond.from_mean_cv(f"1: tooth({substrate})-adhesive", mean, cv, test_basis="uTBS", citations=cites,
                                    notes=f"{adhesive}; {'aged: ' + a['aged'] if aged else '1 day'}")


def adhesive_cement_interface() -> WeibullBond:
    return WeibullBond(interface="2: adhesive-cement", sigma0_mpa=60.0, m=5.0, test_basis="assumed",
                       notes="ASSUMED: co-polymerised methacrylate layers; rarely the failure origin (brief 5.3)")


class BondAssessment(StrictModel):
    substrate: str
    adhesive: str
    intaglio_protocol: str
    links: list[WeibullBond]
    interface_level: AssemblyResult
    assembly_estimate: ModelPrediction
    cement_restoration: ModelPrediction
    notes: list[str] = Field(default_factory=list)
    comparison_basis: str = ("protocol assessment for ONE material: compare protocol A vs protocol B on the same restoration; "
                             "NOT a material property to score against another product (Midas cemented the same way scores the same)")


# Best manufacturer protocol datum (SprintRay Ceramic Crown, Clearfil Ceramic Primer Plus + Panavia V5; manufacturer claim,
# brief section 5): kept separately - it is a protocol effect, not a material property.
BEST_PROTOCOL_DATUM = {"product": "SprintRay Ceramic Crown", "protocol": "Clearfil Ceramic Primer Plus + Panavia V5",
                       "value_mpa": 42.1, "test": "shear bond, cement to restoration", "evidence": "manufacturer claim"}


def compare_protocols(formulation: Formulation, protocol_a: str, protocol_b: str,
                      substrate: Literal["enamel", "superficial_dentin", "deep_dentin"] = "superficial_dentin",
                      aged: bool = True, db=None) -> dict:
    """Same material, protocol A vs protocol B (assembly and cement-restoration medians, ratio B/A)."""
    a = assess_bonding(formulation, substrate, intaglio_protocol=protocol_a, aged=aged, db=db)
    b = assess_bonding(formulation, substrate, intaglio_protocol=protocol_b, aged=aged, db=db)
    return {"basis": "same material, protocol A vs protocol B", "protocol_a": protocol_a, "protocol_b": protocol_b,
            "assembly_a_mpa": a.assembly_estimate.value, "assembly_b_mpa": b.assembly_estimate.value,
            "cement_restoration_a_mpa": a.cement_restoration.value, "cement_restoration_b_mpa": b.cement_restoration.value,
            "ratio_b_over_a": b.assembly_estimate.value / a.assembly_estimate.value,
            "best_protocol_datum": BEST_PROTOCOL_DATUM}


def assess_bonding(
    formulation: Formulation,
    substrate: Literal["enamel", "superficial_dentin", "deep_dentin"] = "superficial_dentin",
    adhesive: str = "universal, etch-and-rinse mode (Scotchbond Universal)",
    intaglio_protocol: str = "APA 50 um 2 bar + silane/MDP universal (Scotchbond Universal Plus)",
    aged: bool = True,
    restoration_flexural_strength_mpa: Optional[float] = None,
    db=None,
) -> BondAssessment:
    """Weakest-link bond estimate for crown resin / cement / adhesive / tooth.

    Interface-level: series Weibull system of interfaces 1-3 (literature uTBS/SBS inputs).
    Assembly-level (what a uTBS stick of the whole assembly after 10k TC would read): the
    interface-level median times ASSEMBLY_FACTOR (0.6-0.98), calibrated on the only published
    printed-crown/cement/dentin assembly test (Montenbruck 2025: 6.6-10.6 MPa, where the
    interface-level estimate for a Crowntec-like resin is ~10.8 MPa)."""
    coh = None if restoration_flexural_strength_mpa is None else restoration_flexural_strength_mpa / 1.5
    l3, m3lo, m3hi, note3 = cement_restoration_bond(formulation, intaglio_protocol, aged, coh, db)
    links = [tooth_interface(substrate, adhesive, aged), adhesive_cement_interface(), l3]
    res = weakest_link(links)
    f0, fl, fh = ASSEMBLY_FACTOR
    # range: interface-level p10..p90 is the specimen scatter; model range uses factor range and interface-3 mean range
    l3_lo = WeibullBond.from_mean_cv(l3.interface, m3lo, UTBS_CV[0], test_basis=l3.test_basis)
    l3_hi = WeibullBond.from_mean_cv(l3.interface, m3hi, UTBS_CV[0], test_basis=l3.test_basis)
    med_lo = weakest_link([links[0], links[1], l3_lo]).median_mpa
    med_hi = weakest_link([links[0], links[1], l3_hi]).median_mpa
    cites = [c for l in links for c in l.citations] + [CIT["B23"], CIT["B35"], CIT["B41"]]
    checks = [CheckResult(name="assembly_target_20MPa_dentin", passed=bool(res.median_mpa * f0 >= 20.0), metric=res.median_mpa * f0,
                          tolerance=20.0, unit=Unit.MPA, details="project target: assembly uTBS >= 20 MPa to dentin after 10k TC")]
    assembly = ModelPrediction(
        name=PropertyName.BOND_STRENGTH_ESTIMATE,
        value=res.median_mpa * f0, low=med_lo * fl, high=med_hi * fh, unit=Unit.MPA,
        interval="assembly-factor range x interface-3 mean range (specimen scatter reported separately)",
        model="weakest-link Weibull series of interfaces 1-3 x assembly factor",
        equation="S(s) = prod_i exp(-(s/sigma0_i)^m_i); sigma_assembly = F_asm * median(S); m_i from CV (20-50%)",
        assumptions=[
            "each interface carries the same nominal tensile stress (uTBS stick); shear-test (SBS) inputs for interface 3 used as if tensile",
            "shear tests concentrate stress and misrepresent interfacial strength (Braga 2010); uTBS CV 20-50%, Weibull m ~ 2.7-4.8",
            "F_asm = 0.6-0.98 (nominal 0.8) calibrated on ONE study (Montenbruck 2025) - the dominant uncertainty",
            f"interface 3 scaled by bondable filler fraction ({note3})",
            "adhesive-cement interface ASSUMED strong (sigma0 60 MPa, m 5)",
        ],
        citations=cites,
        inputs={"interface_level_median": res.median_mpa, "F_asm": f0, **{f"P_origin[{k}]": v for k, v in res.origin_probability.items()}},
        checks=checks,
        conditions_note=f"{substrate}, {'after 10k TC / aged' if aged else '24 h'}",
        notes=f"weakest interface: {res.weakest_interface}",
    )
    cr = ModelPrediction(
        name=PropertyName.BOND_STRENGTH_ESTIMATE, value=l3.mean(), low=m3lo, high=m3hi, unit=Unit.MPA,
        interval="scaling-exponent / reference-fraction uncertainty (+20/-25 %)",
        model="literature SBS for the intaglio protocol scaled by bondable exposed-filler fraction",
        equation="sigma_3 = sigma_protocol,ref (phi_bond/phi_bond,ref)^0.5, clipped to x[0.7, 1.3]",
        assumptions=["reference = SprintRay Ceramic Crown / printed resins with the same protocol", "exponent 0.5 and compatibility table are expert judgement"],
        citations=l3.citations + [CIT["S43"], CIT["B33"]],
        inputs={"protocol_mean_MPa": INTAGLIO_PROTOCOLS[intaglio_protocol]["mean"]},
        checks=[CheckResult(name="cement_restoration_target_25MPa", passed=bool(l3.mean() >= 25.0), metric=l3.mean(), tolerance=25.0, unit=Unit.MPA)],
        conditions_note="notched-edge/shear, aged" if aged else "24 h",
    )
    notes = [
        "the pulp is not a bonding substrate; protect it via remaining dentin thickness >= 0.5 mm, limited heat (+5.5 C threshold, secondary), low residual monomer (ISO 10993-5, ISO 7405)",
        "once interface 3 is properly treated, failures move cohesively into the printed resin: the resin's own strength/toughness limit the assembly",
    ]
    return BondAssessment(substrate=substrate, adhesive=adhesive, intaglio_protocol=intaglio_protocol, links=links,
                          interface_level=res, assembly_estimate=assembly, cement_restoration=cr, notes=notes)


def montenbruck_check(db=None) -> CheckResult:
    """CONSISTENCY check (not an independent validation: F_asm was fitted to this same study):
    interface-level estimate for a Crowntec-like printed resin (~19 vol%
    glass/silica) with APA + Adhese Universal / Variolink, dentin, aged, times F_asm should
    fall in the measured 6.57-10.58 MPa (Montenbruck 2025)."""
    from fracture_md.schema.formulation import Formulation
    comps = [C.make_component("Bis-EMA", 0.50), C.make_component("UDMA", 0.164), C.make_component("TPO", 0.004),
             C.make_component("ba glass 0.7", 0.302), C.make_component("fumed silica", 0.03)]
    f = Formulation(name="crowntec-like", components=comps)
    a = assess_bonding(f, "superficial_dentin", "universal, etch-and-rinse mode (Scotchbond Universal)",
                       "APA 50 um 1 bar + Adhese Universal + Variolink Esthetic DC", True, db=db)
    v = a.assembly_estimate.value
    return CheckResult(name="assembly_factor_consistency_Montenbruck2025", passed=bool(6.57 * 0.9 <= v <= 10.58 * 1.1), metric=v,
                       unit=Unit.MPA, details=f"consistency only (F_asm fitted to this study): predicted {v:.1f} MPa vs measured 6.57-10.58 MPa (interface-level median {a.interface_level.median_mpa:.1f})")


# ---------------------------------------------------------------------------
# Structured substrate / pulp facts (for reports and agents)
# ---------------------------------------------------------------------------
TOOTH_SUBSTRATES = {
    "enamel": {
        "composition": "~96 wt% carbonated hydroxyapatite, ~3 wt% water, ~1 wt% organic (~92 vol% mineral)",
        "structure": "prisms (rods) 4-6 um with interrod enamel; HAp crystallites 30-40 nm wide",
        "bonding": "micromechanical after 35-37% H3PO4 etch (resin tags) + chemical 10-MDP-Ca salt formation",
        "typical_uTBS_MPa": "44.9 (self-etch) to 65.8 +/- 32.1 (etched) [B47]",
    },
    "dentin_superficial": {
        "composition": "70 wt% mineral / 20 wt% organic (~90% type I collagen) / 10 wt% water",
        "structure": "~20,000 tubules/mm^2, ~0.9 um diameter; tubule volume ~4% near the DEJ",
        "bonding": "hybrid layer (resin-infiltrated collagen); mild self-etch keeps HAp for MDP-Ca chemistry",
        "typical_uTBS_MPa": "34-50 at 1 day for gold-standard adhesives [B46, B36]",
    },
    "dentin_deep": {
        "structure": "~45,000 tubules/mm^2, ~2.5 um diameter; tubule volume up to ~28% near the pulp; pulpal fluid pressure ~25-30 mmHg",
        "bonding": "wetter, weaker, more variable; hybrid-layer collagen degrades by MMPs/hydrolysis",
        "typical_uTBS_MPa": "lower than superficial (not quantified in verified sources; DEEP_DENTIN_FACTOR is assumed)",
    },
    "pulp": {
        "is_bonding_substrate": False,
        "why": "soft vascularised, innervated connective tissue; adhesives on exposed pulp cause persistent inflammation [B10]",
        "protect_by": [
            "remaining dentin thickness >= 0.5 mm (odontoblast loss 13.6% at 0.5-2.5 mm RDT vs 33.7% at <0.5 mm) [B11]",
            "limit intrapulpal temperature rise (+5.5 C classic threshold, secondary sources) - printed-crown exotherm is extra-oral; only cement light-curing is intra-oral [B13]",
            "minimise residual monomer (TEGDMA/HEMA -> GSH depletion, ROS, apoptosis) [B15]: full post-cure, low-TEGDMA matrix",
            "ISO 10993-5 cytotoxicity (>= 70% viability) and ISO 7405 dentin-barrier test (0.4 mm dentin disc) [B16]",
        ],
    },
}
