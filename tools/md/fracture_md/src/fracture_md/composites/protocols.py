"""Typed laboratory test protocols for printed dental composites.

Every protocol lists all parameters needed to run it, the standard it follows, the specimen
geometry, conditioning (24 h / 7 d water at 37 C, thermocycling), sample size, statistics and
the ``*_MEASURED`` PropertyName(s) it fills.  Measured values enter the results database only
as ``origin=EXPERIMENTAL`` records (the PropertyRecord validator enforces this).

``default_test_plan()`` returns the full characterisation plan used in
docs/dental/TEST_PROTOCOLS.md; ``sample_size_two_groups`` gives n per group for a two-sided
comparison at a given coefficient of variation.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import Field, model_validator

from fracture_md.schema.common import Citation, StrictModel
from fracture_md.schema.properties import MEASURED_ONLY_NAMES, PropertyName
from fracture_md.units import Quantity, Unit, q

# single source: every *_MEASURED name (adds R-curve, fatigue life and the ISO 14577 H_IT / E_IT names)
_MEASURED_ONLY = MEASURED_ONLY_NAMES


class Conditioning(StrictModel):
    """Specimen storage / ageing before testing."""

    label: str  # e.g. "24h_water_37C"
    medium: Literal["dry", "distilled_water", "artificial_saliva", "ethanol_75"] = "distilled_water"
    temperature: Quantity = Field(default_factory=lambda: q(310.15, Unit.KELVIN))  # 37 C
    duration: Quantity = Field(default_factory=lambda: q(86400.0, Unit.SECOND))  # 24 h
    thermocycles: int = 0
    thermocycle_low: Optional[Quantity] = None  # K
    thermocycle_high: Optional[Quantity] = None  # K
    dwell_per_bath: Optional[Quantity] = None  # s
    notes: str = ""


def water_24h() -> Conditioning:
    return Conditioning(label="24h_water_37C")


def water_7d() -> Conditioning:
    return Conditioning(label="7d_water_37C", duration=q(7 * 86400.0, Unit.SECOND))


def thermocycled_10k() -> Conditioning:
    return Conditioning(
        label="24h_water_then_10k_TC_5_55C",
        thermocycles=10000,
        thermocycle_low=q(278.15, Unit.KELVIN),
        thermocycle_high=q(328.15, Unit.KELVIN),
        dwell_per_bath=q(30.0, Unit.SECOND),
        notes="~10,000 cycles is the usual provisional equivalent of ~1 year in service (Gale & Darvell 1999)",
    )


class SpecimenSpec(StrictModel):
    geometry: str  # e.g. "bar 2 x 2 x 25 mm"
    dimensions: dict[str, Quantity] = Field(default_factory=dict)
    build_orientations: list[str] = Field(default_factory=lambda: ["0deg (layers parallel to long axis)", "90deg"])
    post_cure: str = "exactly per IFU (record unit, time, temperature, atmosphere)"
    surface_preparation: str = ""
    tested_surface: Literal["top", "bottom", "top_and_bottom", "cross_section", "not_applicable"] = "not_applicable"


class StatisticsPlan(StrictModel):
    n_per_group: int
    analysis: str
    weibull: bool = False
    alpha: float = 0.05
    power: float = 0.8
    notes: str = ""


class LabProtocol(StrictModel):
    """Base class: common fields of every lab protocol."""

    id: str
    title: str
    standard: str
    fills: list[PropertyName]
    specimen: SpecimenSpec
    conditioning: list[Conditioning] = Field(default_factory=lambda: [water_24h(), water_7d()])
    statistics: StatisticsPlan
    equipment: list[str] = Field(default_factory=list)
    acceptance: list[str] = Field(default_factory=list)  # standard minimums / project targets
    report_items: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    # True when no *_MEASURED PropertyName exists for the quantity (e.g. DC, cytotoxicity):
    # results are then stored as raw data attached to the co-measured records.
    raw_data_only: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def _fills_measured(self) -> "LabProtocol":
        bad = [p.value for p in self.fills if p not in _MEASURED_ONLY]
        if bad:
            raise ValueError(f"lab protocols may only fill *_MEASURED properties, got {bad}")
        if not self.fills and not self.raw_data_only:
            raise ValueError("protocol must state which PropertyName(s) it fills (or set raw_data_only)")
        return self


# ---------------------------------------------------------------------------
# Specific protocols
# ---------------------------------------------------------------------------
_HV_FORCE_N = {"HV0.1": 0.9807, "HV0.2": 1.961, "HV0.3": 2.942, "HV0.5": 4.903, "HV1": 9.807}


class VickersProtocol(LabProtocol):
    load_designation: Literal["HV0.1", "HV0.2", "HV0.3", "HV0.5", "HV1"] = "HV0.2"
    secondary_load: Optional[Literal["HV0.1", "HV0.2", "HV0.3", "HV0.5", "HV1"]] = "HV0.5"
    force_application_time: Quantity = Field(default_factory=lambda: q(7.0, Unit.SECOND))  # 2-8 s
    dwell: Quantity = Field(default_factory=lambda: q(15.0, Unit.SECOND))
    indents_per_specimen: int = 5
    specimens: int = 5
    polish_final: str = "wet SiC P2000-P4000 then 1 um diamond/alumina; remove >=200 um of the as-printed skin"
    min_spacing_diagonals: float = 3.0
    min_edge_distance_diagonals: float = 3.0
    read_within_s: float = 10.0  # read diagonals promptly (viscoelastic recovery)
    temperature_test: Quantity = Field(default_factory=lambda: q(296.15, Unit.KELVIN))
    depth_of_cure_ratio_min: float = 0.7  # ISO 10477 bottom/top >= 0.7 (project target 0.8)
    also_iso14577: bool = True  # H_IT, E_IT, C_IT by instrumented indentation

    def force_newton(self) -> float:
        return _HV_FORCE_N[self.load_designation]

    def designation(self) -> str:
        return f"{self.load_designation}/{int(self.dwell.magnitude(Unit.SECOND))}"


class FlexuralProtocol(LabProtocol):
    span: Quantity
    crosshead_speed_mm_per_min: float = 0.75
    loading: Literal["3-point", "4-point"] = "3-point"
    minimum_required: Optional[Quantity] = None


class BiaxialProtocol(LabProtocol):
    support_ball_circle_diameter: Quantity
    ball_diameter: Quantity
    piston_diameter: Quantity
    disc_diameter: Quantity
    disc_thickness: Quantity
    crosshead_speed_mm_per_min: float = 1.0
    poisson_ratio_for_eval: float = 0.3


class ToughnessProtocol(LabProtocol):
    method: Literal["SEVNB", "SENB", "CNB"] = "SEVNB"
    notch_depth_ratio: float = 0.3  # a/W
    notch_root_radius_max: Quantity = Field(default_factory=lambda: q(10.0, Unit.MICROMETER))
    span: Quantity = Field(default_factory=lambda: q(16.0, Unit.MILLIMETER))
    crosshead_speed_mm_per_min: float = 0.5


class SorptionProtocol(LabProtocol):
    disc_diameter: Quantity = Field(default_factory=lambda: q(15.0, Unit.MILLIMETER))
    disc_thickness: Quantity = Field(default_factory=lambda: q(1.0, Unit.MILLIMETER))
    desiccation: str = "desiccator 37 C then 23 C, weigh daily to constant mass m1 (<0.1 mg change in 24 h)"
    immersion: str = "7 d distilled water 37 C -> m2; re-desiccate to constant mass m3"
    formulae: str = "W_sp = (m2 - m3)/V ; W_sl = (m1 - m3)/V  [ug/mm^3]"
    limits: str = "ISO 4049/10477: W_sp <= 40, W_sl <= 7.5 ug/mm^3; project: <= 12 and <= 1.0"


class CureDepthProtocol(LabProtocol):
    method: Literal["working_curve_windowpane", "ISO4049_scrape", "hardness_profile"] = "working_curve_windowpane"
    wavelength_nm: float = 405.0
    doses_mJ_cm2: list[float] = Field(default_factory=lambda: [5, 10, 20, 40, 80, 160])
    fit: str = "C_d = D_p ln(E/E_c); report D_p (um) and E_c (mJ/cm^2) with 95% CI; confirm bottom/top HV >= 0.7 (ISO 10477)"


class ConversionProtocol(LabProtocol):
    technique: str = "ATR-FTIR, aliphatic C=C 1638 cm^-1 vs aromatic 1608 cm^-1 (or C=O 1720 for non-aromatic blends)"
    resolution_cm1: float = 4.0
    scans: int = 32
    timepoints: list[str] = Field(default_factory=lambda: ["green (after print)", "after IFU post-cure", "top and bottom surfaces"])


class BondProtocol(LabProtocol):
    test: Literal["uTBS", "SBS_notched_edge", "microshear"] = "uTBS"
    substrates: list[str] = Field(default_factory=list)
    beam_area: Optional[Quantity] = None  # mm^2 as MILLIMETER^2 not in unit table -> store side length
    beam_side: Optional[Quantity] = None
    button_diameter: Optional[Quantity] = None
    crosshead_speed_mm_per_min: float = 1.0
    restoration_treatment: str = ""
    adhesive_and_cement: str = ""
    failure_mode_classes: list[str] = Field(
        default_factory=lambda: [
            "adhesive at tooth-adhesive (interface 1)",
            "adhesive at cement-restoration (interface 3)",
            "cohesive in dentin/enamel",
            "cohesive in cement",
            "cohesive in printed restoration",
            "mixed",
        ]
    )
    pretest_failures: str = "counted as 0 MPa (not excluded)"


class CytotoxicityProtocol(LabProtocol):
    cell_line: str = "L929 mouse fibroblasts (or human gingival fibroblasts)"
    extraction: str = "ISO 10993-12: 3 cm^2/mL (thickness <0.5 mm) or 1.25 cm^2/mL, 37 C, 24 h in complete medium"
    assay: str = "MTT/XTT; viability >= 70% of blank = non-cytotoxic"
    also: str = "ISO 7405 dentin-barrier test (0.4 mm dentin disc) for the cement/adhesive assembly"


class ViscosityProtocol(LabProtocol):
    geometry: str = "cone-plate 40 mm / 1 deg or parallel plate 25 mm, 0.5 mm gap"
    temperatures_K: list[float] = Field(default_factory=lambda: [298.15, 303.15, 313.15, 323.15, 333.15])
    shear_rates_per_s: list[float] = Field(default_factory=lambda: [1, 10, 30, 100])
    thixotropy: str = "3-interval thixotropy test (0.1 / 100 / 0.1 1/s); report recovery time (target < ~10 s per layer)"


# ---------------------------------------------------------------------------
# Plan and statistics helpers
# ---------------------------------------------------------------------------
def _z(p: float) -> float:
    """Standard-normal quantile (Acklam-free: bisection on erf, adequate for n planning)."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sample_size_two_groups(cv: float, relative_difference: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """n per group to detect a relative difference of means with a two-sided t-test
    (normal approximation, equal CV): n = 2 (z_{1-a/2} + z_{power})^2 (cv / d)^2, rounded up,
    minimum 5.  E.g. cv=0.3 (uTBS), d=0.25 -> n=23 per group."""
    if cv <= 0 or relative_difference <= 0:
        raise ValueError("cv and relative_difference must be positive")
    n = 2.0 * (_z(1.0 - alpha / 2.0) + _z(power)) ** 2 * (cv / relative_difference) ** 2
    return max(5, math.ceil(n))


CIT = {
    "ISO6507": Citation(key="ISO6507-1:2023", text="ISO 6507-1:2023 Metallic materials - Vickers hardness test - Part 1",
                        url="https://cdn.standards.iteh.ai/samples/83898/e08ffc7223ac43bdba0d0a326bd8f891/ISO-6507-1-2023.pdf"),
    "ISO14577": Citation(key="ISO14577-1", text="ISO 14577-1 Instrumented indentation test; HV = 0.0945 H_IT"),
    "ISO4049": Citation(key="ISO4049:2019", text="ISO 4049:2019 Dentistry - Polymer-based restorative materials",
                        url="https://www.iso.org/standard/67596.html"),
    "ISO10477": Citation(key="ISO10477:2020", text="ISO 10477:2020 Dentistry - Polymer-based crown and veneering materials",
                         url="https://www.iso.org/standard/80007.html"),
    "ISO6872": Citation(key="ISO6872", text="ISO 6872 Dentistry - Ceramic materials (piston-on-three-balls biaxial flexure)"),
    "ISO23146": Citation(key="ISO23146", text="ISO 23146 Fine ceramics - fracture toughness by SEVNB"),
    "ISO11405": Citation(key="ISO/TS11405:2015", text="ISO/TS 11405:2015 Dentistry - Testing of adhesion to tooth structure",
                         url="https://cdn.standards.iteh.ai/samples/62898/c8afc682cda547d797cf339c726837ce/ISO-TS-11405-2015.pdf"),
    "ISO29022": Citation(key="ISO29022:2013", text="ISO 29022:2013 Dentistry - Adhesion - Notched-edge shear bond strength test",
                         url="https://cdn.standards.iteh.ai/samples/45285/c6456927fa5c4edfb3d4d21f1651127b/ISO-29022-2013.pdf"),
    "ISO10993": Citation(key="ISO10993-5", text="ISO 10993-5 Biological evaluation of medical devices - in vitro cytotoxicity"),
    "ISO7405": Citation(key="ISO7405", text="ISO 7405 Dentistry - Evaluation of biocompatibility of medical devices used in dentistry"),
    "Armstrong2017": Citation(key="B38", text="Armstrong S et al. Academy of Dental Materials guidance on microtensile bond strength testing. Dent Mater 2017;33:133-143",
                              doi="10.1016/j.dental.2016.11.015"),
    "GaleDarvell1999": Citation(key="B44", text="Gale MS, Darvell BW. Thermal cycling procedures for laboratory testing of dental restorations. J Dent 1999;27:89-99",
                                doi="10.1016/s0300-5712(98)00037-2"),
    "Chen2026": Citation(key="S12", text="Chen J et al. J Prosthodont Res 2026 (HV0.2/10 s, 7 d water protocol)", doi="10.2186/jpr.JPR_D_25_00232"),
    "Jacobs1992": Citation(key="Jacobs1992", text="Jacobs PF. Rapid Prototyping & Manufacturing: Fundamentals of Stereolithography. SME, 1992 (working curve C_d = D_p ln(E/E_c))"),
}


def default_test_plan() -> list[LabProtocol]:
    """The full characterisation plan (see docs/dental/TEST_PROTOCOLS.md)."""
    mm, um, s = Unit.MILLIMETER, Unit.MICROMETER, Unit.SECOND
    plan: list[LabProtocol] = []
    plan.append(
        VickersProtocol(
            id="P-HV",
            title="Vickers hardness (low-force) + ISO 14577 instrumented indentation",
            standard="ISO 6507-1:2023 (HV0.2/15 primary, HV0.5/20 secondary); ISO 14577-1 (H_IT, E_IT, C_IT)",
            fills=[PropertyName.VICKERS_HARDNESS_MEASURED],
            specimen=SpecimenSpec(
                geometry="printed disc 10 x 10 x 2-3 mm (clinical build orientation + 90 deg)",
                dimensions={"side": q(10.0, mm), "thickness": q(2.5, mm)},
                surface_preparation="wet SiC to P4000 then 1 um polish; state tested surface",
                tested_surface="top_and_bottom",
            ),
            conditioning=[water_24h(), water_7d(), thermocycled_10k()],
            statistics=StatisticsPlan(
                n_per_group=25,
                analysis="5 specimens x 5 indents per condition/orientation/surface; mixed model (specimen random effect); report mean, SD, 95% CI",
                notes="Report designation with load and dwell (e.g. '87 HV0.2/15'); depth of cure: bottom/top >= 0.7 (ISO 10477), target >= 0.8",
            ),
            equipment=["micro-Vickers tester, calibrated per ISO 6507-2", "instrumented indenter (Vickers/Berkovich), 1-10 N"],
            acceptance=["target >= 85 HV0.2/15 after 7 d water (stretch 100)", "bottom/top HV >= 0.8 (ISO 10477 minimum 0.7)"],
            report_items=["d1, d2 per indent", "load, dwell, time-to-read", "top vs bottom", "build orientation", "post-cure unit"],
            citations=[CIT["ISO6507"], CIT["ISO14577"], CIT["ISO10477"], CIT["Chen2026"]],
        )
    )
    plan.append(
        FlexuralProtocol(
            id="P-FS3-4049",
            title="3-point flexural strength and modulus (ISO 4049 bars)",
            standard="ISO 4049:2019 clause 7.11",
            fills=[PropertyName.FLEXURAL_STRENGTH_MEASURED, PropertyName.FLEXURAL_MODULUS_MEASURED],
            specimen=SpecimenSpec(
                geometry="bar 2 x 2 x 25 mm, edges chamfered, printed in 0 and 90 deg orientation",
                dimensions={"width": q(2.0, mm), "height": q(2.0, mm), "length": q(25.0, mm)},
                surface_preparation="wet-ground P1200 on 4 long faces",
            ),
            conditioning=[water_24h(), water_7d(), thermocycled_10k()],
            span=q(20.0, mm),
            crosshead_speed_mm_per_min=0.75,
            minimum_required=q(100.0, Unit.MPA),
            statistics=StatisticsPlan(
                n_per_group=15, weibull=True,
                analysis="sigma = 3FL/(2bh^2); E_f from initial slope; Weibull m, sigma_0 (MLE, 95% CI; n>=15 per ADM/ISO practice)",
            ),
            acceptance=["ISO 4049 Type 1 Class 2 Group 2: >= 100 MPa", "project: >= 160 MPa after 7 d water, <= 15% loss after 10k TC"],
            citations=[CIT["ISO4049"]],
        )
    )
    plan.append(
        FlexuralProtocol(
            id="P-FS3-10477",
            title="3-point flexural strength (ISO 10477, for 510(k)-style comparison with Midas Restore 120-150 MPa)",
            standard="ISO 10477:2020 clause 7.4",
            fills=[PropertyName.FLEXURAL_STRENGTH_MEASURED],
            specimen=SpecimenSpec(geometry="bar 2 x 2 x 25 mm (ISO 10477)", dimensions={"length": q(25.0, mm)}),
            conditioning=[water_24h()],
            span=q(20.0, mm),
            crosshead_speed_mm_per_min=1.0,
            minimum_required=q(50.0, Unit.MPA),
            statistics=StatisticsPlan(n_per_group=15, weibull=True, analysis="as ISO 4049 protocol"),
            acceptance=[">= 50 MPa (ISO 10477)"],
            citations=[CIT["ISO10477"]],
        )
    )
    plan.append(
        BiaxialProtocol(
            id="P-BFS",
            title="Biaxial flexural strength, piston-on-three-balls (for comparison with Midas 180 MPa mean)",
            standard="ISO 6872 (piston-on-three-balls)",
            fills=[PropertyName.BIAXIAL_FLEXURAL_STRENGTH_MEASURED],
            specimen=SpecimenSpec(
                geometry="disc 12-16 mm diameter x 1.2 +/- 0.2 mm",
                dimensions={"diameter": q(14.0, mm), "thickness": q(1.2, mm)},
                surface_preparation="both faces ground/polished to 15 um; tensile face = intaglio-equivalent face",
            ),
            conditioning=[water_24h(), water_7d()],
            support_ball_circle_diameter=q(10.0, mm),
            ball_diameter=q(3.2, mm),
            piston_diameter=q(1.4, mm),
            disc_diameter=q(14.0, mm),
            disc_thickness=q(1.2, mm),
            statistics=StatisticsPlan(n_per_group=30, weibull=True, analysis="ISO 6872 formula; Weibull m >= 10 target (n=30 for m CI)"),
            acceptance=["project: mean >= 220 MPa, Weibull m >= 10"],
            citations=[CIT["ISO6872"]],
            notes="Midas Restore's 180 MPa is a manufacturer claim on this geometry; compare like with like.",
        )
    )
    plan.append(
        ToughnessProtocol(
            id="P-KIC",
            title="Fracture toughness K_Ic (SEVNB)",
            standard="ISO 23146 (SEVNB) adapted; SENB per Fujishima & Ferracane 1996 as alternative",
            fills=[PropertyName.FRACTURE_TOUGHNESS_MEASURED],
            specimen=SpecimenSpec(geometry="bar 3 x 4 x 20 mm, V-notch a/W 0.2-0.3 sharpened with razor + 1 um diamond paste",
                                  dimensions={"width": q(4.0, mm), "thickness": q(3.0, mm), "length": q(20.0, mm)}),
            conditioning=[water_24h(), water_7d()],
            statistics=StatisticsPlan(n_per_group=10, analysis="K_Ic from ISO 23146 compliance function; notch root radius verified optically"),
            acceptance=["project: >= 2.0 MPa m^0.5"],
            citations=[CIT["ISO23146"]],
        )
    )
    plan.append(
        SorptionProtocol(
            id="P-WSP",
            title="Water sorption and solubility",
            standard="ISO 4049:2019 clause 7.12 / ISO 10477:2020 clauses 7.6-7.7",
            fills=[PropertyName.WATER_SORPTION_MEASURED, PropertyName.WATER_SOLUBILITY_MEASURED],
            specimen=SpecimenSpec(geometry="disc 15.0 +/- 0.1 mm x 1.0 +/- 0.1 mm", dimensions={"diameter": q(15.0, mm), "thickness": q(1.0, mm)}),
            conditioning=[water_7d()],
            statistics=StatisticsPlan(n_per_group=5, analysis="mean +/- SD; ISO pass/fail per specimen"),
            acceptance=["ISO: <= 40 / <= 7.5 ug/mm^3", "project: <= 12 / <= 1.0 ug/mm^3"],
            citations=[CIT["ISO4049"], CIT["ISO10477"]],
        )
    )
    plan.append(
        CureDepthProtocol(
            id="P-CD",
            title="Cure depth / working curve at the printer wavelength",
            standard="Jacobs working curve (windowpane); ISO 4049 clause 7.10 scrape test as a cross-check; ISO 10477 hardness ratio",
            fills=[PropertyName.CURE_DEPTH_MEASURED],
            specimen=SpecimenSpec(geometry="single exposures on a glass slide/vat film, thickness by micrometer after IPA rinse"),
            conditioning=[Conditioning(label="as_cured", medium="dry", duration=q(0.0, s))],
            statistics=StatisticsPlan(n_per_group=3, analysis="3 replicates per dose; least-squares fit of C_d vs ln E"),
            acceptance=["C_d(E_layer) >= 2 x layer thickness (e.g. >= 100 um for 50 um layers)"],
            citations=[CIT["Jacobs1992"], CIT["ISO4049"]],
        )
    )
    plan.append(
        ConversionProtocol(
            id="P-DC",
            title="Degree of conversion (ATR-FTIR)",
            standard="ATR-FTIR (no ISO standard); same specimens as P-HV",
            fills=[],
            raw_data_only=True,
            specimen=SpecimenSpec(geometry="P-HV discs, top and bottom", tested_surface="top_and_bottom"),
            conditioning=[Conditioning(label="as_post_cured", medium="dry", duration=q(0.0, s))],
            statistics=StatisticsPlan(n_per_group=5, analysis="DC = 1 - (A1638/A1608)_cured/(A1638/A1608)_uncured; correlate with HV (expect r ~ 0.8)"),
            acceptance=["project: DC >= 80% after IFU post-cure"],
            notes="No *_MEASURED PropertyName exists for DC; DC values are stored as raw data attached to the "
                  "co-measured VICKERS_HARDNESS_MEASURED records (same specimens).",
        )
    )
    uTBS_cond = [water_24h(), thermocycled_10k()]
    plan.append(
        BondProtocol(
            id="P-uTBS",
            title="Microtensile bond strength of the full assembly (crown resin / cement / adhesive / tooth)",
            standard="ISO/TS 11405:2015 + Academy of Dental Materials uTBS guidance (Armstrong 2017)",
            fills=[PropertyName.BOND_STRENGTH_MEASURED],
            specimen=SpecimenSpec(
                geometry="non-trimmed beams ~1 x 1 mm cut from bonded human third molars (donors 16-40 y) or bovine incisors (<=5 y)",
                dimensions={"beam_side": q(1.0, mm)},
                surface_preparation="flat superficial dentin (~1 mm below DEJ) and deep dentin (~1 mm above pulp horn); enamel groups separate; P400 SiC smear layer",
            ),
            conditioning=uTBS_cond,
            substrates=["enamel (etched)", "superficial dentin", "deep dentin"],
            beam_side=q(1.0, mm),
            crosshead_speed_mm_per_min=1.0,
            restoration_treatment="50 um Al2O3 air abrasion at 1 bar (0.1 MPa), 10 mm, 10 s; ultrasonic/Ivoclean; silane + 10-MDP primer (e.g. Monobond Plus / Clearfil Ceramic Primer Plus)",
            adhesive_and_cement="Adhese Universal (selective enamel etch) + Variolink Esthetic DC; comparator: SprintRay Ceramic Crown with the same protocol",
            statistics=StatisticsPlan(
                n_per_group=sample_size_two_groups(0.35, 0.30),
                weibull=True,
                analysis="tooth as statistical unit (>=5 teeth/group), beams nested; pre-test failures = 0 MPa; Weibull sigma_0, m, Pf10; failure modes by stereomicroscope 40x + SEM",
                notes="n from sample_size_two_groups(cv=0.35, 30% difference, alpha 0.05, power 0.8)",
            ),
            acceptance=["project: assembly uTBS >= 20 MPa to dentin and >= 30 MPa to enamel after 10k TC", "<10% pre-test failures", "failures not at interface 3"],
            citations=[CIT["ISO11405"], CIT["Armstrong2017"], CIT["GaleDarvell1999"]],
        )
    )
    plan.append(
        BondProtocol(
            id="P-SBS",
            title="Notched-edge shear bond strength, cement to printed restoration (interface 3)",
            standard="ISO 29022:2013 (notched-edge, 2.38 mm button) applied to the printed-resin substrate",
            fills=[PropertyName.BOND_STRENGTH_MEASURED],
            test="SBS_notched_edge",
            specimen=SpecimenSpec(geometry="printed resin block, intaglio-equivalent face treated as in P-uTBS"),
            conditioning=[water_24h(), thermocycled_10k()],
            button_diameter=q(2.38, mm),
            restoration_treatment="as P-uTBS; groups: untreated / APA only / APA + silane-MDP primer",
            adhesive_and_cement="Variolink Esthetic DC (+ Adhese Universal) and Panavia V5 + Clearfil Ceramic Primer Plus",
            statistics=StatisticsPlan(n_per_group=15, weibull=True, analysis="ISO 29022: n=15; report failure modes; shear tests over-concentrate stress (Braga 2010)"),
            acceptance=["project: >= 25 MPa after 10k TC with cohesive failure in the printed resin"],
            citations=[CIT["ISO29022"]],
        )
    )
    plan.append(
        CytotoxicityProtocol(
            id="P-CYTO",
            title="In vitro cytotoxicity of post-cured specimens (extract test) and dentin-barrier test",
            standard="ISO 10993-5 (+ ISO 10993-12 extraction); ISO 7405 dentin barrier",
            fills=[],
            raw_data_only=True,
            specimen=SpecimenSpec(geometry="discs post-cured per IFU, cleaned as for clinical use"),
            conditioning=[Conditioning(label="extraction_24h_37C", medium="distilled_water")],
            statistics=StatisticsPlan(n_per_group=3, analysis="triplicate wells x 3 independent runs; viability vs blank"),
            acceptance=["viability >= 70% (non-cytotoxic)"],
            citations=[CIT["ISO10993"], CIT["ISO7405"]],
            notes="No *_MEASURED PropertyName exists for cytotoxicity; results are stored as raw data (viability %, "
                  "extraction ratio) with the candidate's provenance.",
        )
    )
    plan.append(
        ViscosityProtocol(
            id="P-VISC",
            title="Paste rheology at print temperature",
            standard="rotational rheometry (no ISO standard); report at 30 1/s",
            fills=[PropertyName.VISCOSITY_MEASURED],
            specimen=SpecimenSpec(geometry="uncured paste, degassed, 1 mL"),
            conditioning=[Conditioning(label="as_mixed_after_24h_rest", medium="dry", duration=q(86400.0, s))],
            statistics=StatisticsPlan(n_per_group=3, analysis="flow curve fit (Herschel-Bulkley / Cross); eta(30 1/s, T)"),
            acceptance=["open-vat DLP: <= 3 Pa s at 30 1/s at print temperature; else press-fed route"],
        )
    )
    plan.extend(biocompatibility_protocols())
    return plan


# ---------------------------------------------------------------------------
# Biocompatibility (ISO 10993-1) and radiopacity
# ---------------------------------------------------------------------------
# ISO 10993-1 endpoints for a surface device, permanent contact (> 30 d) with tooth (mucosa-
# contacting surface).  "X" = to be addressed (test or justified by chemical characterisation).
ISO10993_1_ENDPOINTS: list[dict] = [
    {"endpoint": "chemical characterisation / leachables", "standard": "ISO 10993-18, -12, -17 (TRA)", "required": True, "protocol": "P-ELUTE"},
    {"endpoint": "cytotoxicity", "standard": "ISO 10993-5 (+ ISO 7405 dentin barrier)", "required": True, "protocol": "P-CYTO"},
    {"endpoint": "sensitisation", "standard": "ISO 10993-10", "required": True, "protocol": "P-SENS"},
    {"endpoint": "irritation / intracutaneous reactivity", "standard": "ISO 10993-23", "required": True, "protocol": "P-IRR"},
    {"endpoint": "material-mediated pyrogenicity", "standard": "ISO 10993-11", "required": False, "protocol": None,
     "note": "usually justified by chemistry for tooth-surface devices; decide with the test house"},
    {"endpoint": "acute systemic toxicity", "standard": "ISO 10993-11", "required": True, "protocol": None,
     "note": "may be addressed by P-ELUTE + toxicological risk assessment (ISO 10993-17)"},
    {"endpoint": "subacute / subchronic / chronic toxicity", "standard": "ISO 10993-11", "required": True, "protocol": None,
     "note": "permanent contact: address by TRA of P-ELUTE data or test"},
    {"endpoint": "genotoxicity", "standard": "ISO 10993-3 (Ames + in vitro mammalian)", "required": True, "protocol": "P-GENO"},
    {"endpoint": "implantation", "standard": "ISO 10993-6", "required": False, "protocol": None,
     "note": "normally not required for surface (tooth) contact; justify"},
    {"endpoint": "carcinogenicity", "standard": "ISO 10993-3", "required": False, "protocol": None,
     "note": "address by genotoxicity + TRA (BAPO regulatory watch)"},
]


def biocompatibility_protocols() -> list[LabProtocol]:
    """P-ELUTE, P-SENS, P-GENO, P-IRR and P-RADIO (see docs/dental/TEST_PROTOCOLS.md)."""
    d = Unit.SECOND
    day = 86400.0
    cit = [CIT["ISO10993"]]
    disc = SpecimenSpec(geometry="discs post-cured per IFU, cleaned as for clinical use; surface/volume ratio per ISO 10993-12")
    elute_cond = [Conditioning(label=f"extract_{n}d_{m}", medium=m, duration=q(n * day, d)) for m in ("distilled_water", "ethanol_75")
                  for n in (1, 7, 30)]
    return [
        LabProtocol(
            id="P-ELUTE", title="Residual monomer and leachables (HPLC / GC-MS of extracts)",
            standard="ISO 10993-12 (extraction), ISO 10993-18 (chemical characterisation), ISO 10993-17 (TRA)",
            fills=[], raw_data_only=True, specimen=disc, conditioning=elute_cond,
            statistics=StatisticsPlan(n_per_group=3, analysis="quantify UDMA, Bis-EMA, TEGDMA, D3MA, BAPO and its photoproducts "
                                      "(2,4,6-trimethylbenzoic acid, phenylphosphinic species), BHT; ug/cm^2 vs time"),
            acceptance=["each analyte below its tolerable-intake-derived allowable limit (ISO 10993-17 TRA)",
                        "cumulative 30 d elution reported; no unexplained peaks above the analytical evaluation threshold"],
            citations=cit,
            notes="Water and 75% ethanol (food-simulating worst case) at 37 C, 1/7/30 d. DC is predicted ~80% so residual monomer matters.",
        ),
        LabProtocol(
            id="P-SENS", title="Skin sensitisation (methacrylates and BAPO H317)", standard="ISO 10993-10 (GPMT or LLNA; or in chemico/in vitro per ISO/TS 10993-10 alternatives)",
            fills=[], raw_data_only=True, specimen=disc, conditioning=[Conditioning(label="extract_polar_nonpolar_72h", duration=q(3 * day, d))],
            statistics=StatisticsPlan(n_per_group=10, analysis="GPMT: 10 test + 5 control animals; Magnusson-Kligman grading"),
            acceptance=["no sensitisation (grade 0 in test animals, or negative in the validated alternative)"], citations=cit,
        ),
        LabProtocol(
            id="P-GENO", title="Genotoxicity", standard="ISO 10993-3 (bacterial reverse mutation OECD 471 + in vitro micronucleus OECD 487)",
            fills=[], raw_data_only=True, specimen=disc, conditioning=[Conditioning(label="extract_polar_nonpolar_72h", duration=q(3 * day, d))],
            statistics=StatisticsPlan(n_per_group=3, analysis="per OECD guideline; positive/negative controls"),
            acceptance=["negative in both assays"], citations=cit,
        ),
        LabProtocol(
            id="P-IRR", title="Irritation (oral mucosa / intracutaneous)", standard="ISO 10993-23",
            fills=[], raw_data_only=True, specimen=disc, conditioning=[Conditioning(label="extract_polar_nonpolar_72h", duration=q(3 * day, d))],
            statistics=StatisticsPlan(n_per_group=3, analysis="in vitro reconstructed-epidermis model or intracutaneous test per ISO 10993-23"),
            acceptance=["non-irritant"], citations=cit,
        ),
        LabProtocol(
            id="P-RADIO", title="Radiopacity (ZrO2 / Ba glass)", standard="ISO 4049:2019 clause 7.14 (aluminium step wedge)",
            fills=[], raw_data_only=True,
            specimen=SpecimenSpec(geometry="disc 15 mm dia x 1.0 +/- 0.1 mm", dimensions={"thickness": q(1.0, Unit.MILLIMETER)}),
            conditioning=[Conditioning(label="dry_24h", medium="dry")],
            statistics=StatisticsPlan(n_per_group=3, analysis="optical density vs Al step wedge; mm Al equivalent"),
            acceptance=["radiopacity >= that of the same thickness of aluminium (ISO 4049); claim-level if stated by the manufacturer"],
            citations=[CIT["ISO4049"]],
        ),
    ]
