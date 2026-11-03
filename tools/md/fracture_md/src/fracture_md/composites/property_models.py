"""Property models for printable ceramic-filled dental composites.

Every model returns a :class:`ModelPrediction` (value, low/high range, unit, model name,
equation, assumptions, citations, inputs) that turns into a PropertyRecord via
``prediction.to_record(candidate)`` -> ``records.make_record(origin=DERIVED_MODEL)``, so the
validation status is always MODEL_ESTIMATE (or FAILED if an attached check fails).

Models (see docs/dental/PROPERTY_MODELS.md for equations, calibration and limitations):

* Vickers hardness
  - (a) empirical: ln HV = ln HV_m + a*phi fitted to ``data/dental/hardness_calibration.json``
    (Braem-type exponential rule), 95 % prediction interval from the OLS fit;
  - (b) physics cross-check: Johnson expanding-cavity model with composite E (Hashin-Shtrikman
    lower bound = Mori-Tanaka for spheres) and a composite compressive yield stress;
  - (c) unit conversions HV <-> GPa, ISO 6507 HV = 0.1891 F/d^2, ISO 14577 HV = 0.0945 H_IT.
* Water sorption / solubility: matrix rule of mixtures x (1 - phi) x interface factor.
* Viscosity: Grunberg-Nissan log mixing + Arrhenius for the monomer blend, Krieger-Dougherty
  for the suspension with phi_m from the de Larrard linear packing model and a bound-layer
  (effective volume) penalty for nanoparticles.
* Cure depth: Jacobs working curve with D_p from Beer-Lambert absorption (initiator/absorber)
  plus particle scattering (Griffith-Halloran geometric form, anomalous-diffraction Q).
* Polymerisation shrinkage and degree of conversion: rule-of-mixtures estimates.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from pydantic import Field

from fracture_md.composites import constituents as C
from fracture_md.records import make_record
from fracture_md.schema.common import (
    CheckResult,
    Citation,
    Conditions,
    StrictModel,
    Uncertainty,
    UncertaintyKind,
)
from fracture_md.schema.formulation import ComponentRole, Formulation
from fracture_md.schema.material import Candidate
from fracture_md.schema.properties import (
    Derivation,
    MethodSpec,
    PropertyName,
    PropertyOrigin,
    PropertyRecord,
    PropertyScale,
)
from fracture_md.units import ISO14577_HV_PER_MPA, Unit, convert, q

REPO_ROOT = Path(__file__).resolve().parents[3]
HARDNESS_CALIBRATION_PATH = REPO_ROOT / "data" / "dental" / "hardness_calibration.json"
GAS_CONSTANT = 8.314462618  # J/(mol K)
KGF_N = 9.80665

CITES = {
    "ISO6507": Citation(key="ISO6507-1:2023", text="ISO 6507-1:2023 Vickers hardness test: HV = 0.1891 F/d^2 (F in N, d in mm)",
                        url="https://cdn.standards.iteh.ai/samples/83898/e08ffc7223ac43bdba0d0a326bd8f891/ISO-6507-1-2023.pdf"),
    "ISO14577": Citation(key="S60", text="ISO 14577-1 conversion HV = 0.0945 H_IT (KLA application note)",
                         url="https://www.kla.com/wp-content/uploads/KLA_AppNote_ISO_14577.pdf"),
    "Braem1989": Citation(key="S48", text="Braem M et al. Mechanical properties and filler fraction of dental composites. Dent Mater 1989;5:346-348",
                          doi="10.1016/0109-5641(89)90128-0"),
    "Chantler1999": Citation(key="T13", text="Chantler PM, Hu X, Boyd NM. Dent Mater 1999;15:144 (exponential rules break down for E_f/E_m > 10)",
                             doi="10.1016/S0109-5641(99)00024-x"),
    "Johnson1970": Citation(key="T2", text="Johnson KL. The correlation of indentation experiments. J Mech Phys Solids 1970;18:115-126; Johnson KL, Contact Mechanics (CUP 1985)",
                            doi="10.1016/0022-5096(70)90029-3"),
    "Olah2024": Citation(key="Olah2024", text="Olah F, Horvath R, Reger M. Heliyon 2024;10:e37461 (Johnson ECM, spherical form H/sigma_y = 2/3[1+ln(E a/(3 sigma_y R))])",
                         url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11425130/"),
    "Koch2009": Citation(key="T3", text="Koch T, Seidler S. Strain 2009;45:26 (polymer hardness correlates with compressive yield)",
                         doi="10.1111/j.1475-1305.2008.00468.x"),
    "HS1963": Citation(key="T36-HS", text="Hashin Z, Shtrikman S. J Mech Phys Solids 1963;11:127", doi="10.1016/0022-5096(63)90060-7"),
    "MT1973": Citation(key="T36-MT", text="Mori T, Tanaka K. Acta Metall 1973;21:571", doi="10.1016/0001-6160(73)90064-3"),
    "Pukanszky1990": Citation(key="Pukanszky1990", text="Pukanszky B. Influence of interface interaction on the ultimate tensile properties of polymer composites. Composites 1990;21:255-262",
                              doi="10.1016/0010-4361(90)90240-W"),
    "KD1959": Citation(key="S40", text="Krieger IM, Dougherty TJ. Trans Soc Rheol 1959;3:137-152", doi="10.1122/1.548848"),
    "GN1949": Citation(key="GrunbergNissan1949", text="Grunberg L, Nissan AH. Mixture law for viscosity. Nature 1949;164:799-800", doi="10.1038/164799b0"),
    "deLarrard1999": Citation(key="deLarrard1999", text="de Larrard F. Concrete Mixture Proportioning: A Scientific Approach. E&FN Spon 1999 (linear packing density model; interaction functions a = sqrt(1-(1-r)^1.02), b = 1-(1-r)^1.5)"),
    "Taylor1998": Citation(key="Taylor1998", text="Taylor DF, Kalachandra S, Sankarapandian M, McGrath JE. Relationship between filler and matrix resin characteristics and the properties of uncured composite pastes. Biomaterials 1998;19:197-204 (max loading 83 vol% hybrid vs 24 vol% microfiller)",
                           doi="10.1016/s0142-9612(98)80001-x"),
    "Lee2006": Citation(key="Lee2006", text="Lee JH, Um CM, Lee IB. Rheological properties of resin composites according to variations in monomer and filler composition. Dent Mater 2006;22:515-526",
                        doi="10.1016/j.dental.2005.05.008"),
    "S38": Citation(key="S38", text="Simunovic L et al. Rheology of dental photopolymers for SLA/DLP/MSLA 3D printing. Polymers 2025;17:2706", doi="10.3390/polym17192706"),
    "S24": Citation(key="S24", text="Griffith ML, Halloran JW. Freeform fabrication of ceramics via stereolithography. J Am Ceram Soc 1996;79:2601 (<3 Pa s at 30 1/s limit as cited in later literature)",
                    doi="10.1111/j.1151-2916.1996.tb09022.x"),
    "GH1997": Citation(key="S41", text="Griffith ML, Halloran JW. Scattering of ultraviolet radiation in turbid suspensions. J Appl Phys 1997;81:2538-2546 (cure depth controlled by Delta n^2 and d/phi)",
                       doi="10.1063/1.364311"),
    "Wang2025": Citation(key="Wang2025", text="Wang ZH et al. Materials 2025;18:1556 (secondary form D_p ~ (2 d50/3)(lambda/phi)(n0^2/Delta n^2) citing Griffith & Halloran)",
                         doi="10.3390/ma18071556"),
    "Jacobs1992": Citation(key="Jacobs1992", text="Jacobs PF. Rapid Prototyping & Manufacturing: Fundamentals of Stereolithography. SME 1992 (C_d = D_p ln(E/E_c))"),
    "vdHulst1957": Citation(key="vandeHulst1957", text="van de Hulst HC. Light Scattering by Small Particles. Wiley 1957 (anomalous diffraction: Q = 2 - (4/rho) sin rho + (4/rho^2)(1 - cos rho), rho = 2x|m-1|)"),
    "Gajewski2012": Citation(key="S31", text="Gajewski VE et al. Monomers used in resin composites. Braz Dent J 2012;23:508-514", doi="10.1590/S0103-64402012000500007"),
    "Kleverlaan2005": Citation(key="Kleverlaan2005", text="Kleverlaan CJ, Feilzer AJ. Polymerization shrinkage and contraction stress of dental resin composites. Dent Mater 2005;21:1150-1157 (shrinkage linear in resin content)",
                               doi="10.1016/j.dental.2005.02.004"),
    "Ferracane1985": Citation(key="S50", text="Ferracane JL. Correlation between hardness and degree of conversion. Dent Mater 1985;1:11-14", doi="10.1016/S0109-5641(85)80058-0"),
    "S45": Citation(key="S45", text="Aktug Karademir S et al. BMC Oral Health 2025;25:304 (DC vs HV r = 0.80 in printed permanent resins)", doi="10.1186/s12903-025-05664-5"),
    "S44": Citation(key="S44", text="Ferracane JL. Hygroscopic and hydrolytic effects in dental polymer networks. Dent Mater 2006;22:211-222", doi="10.1016/j.dental.2005.05.005"),
    "S43": Citation(key="S43", text="Lung CYK, Matinlinna JP. Dent Mater 2012;28:467-477 (zirconia bonds poorly to silanes)", doi="10.1016/j.dental.2012.02.009"),
    "S20": Citation(key="S20", text="ISO 4049:2019 (sorption <= 40, solubility <= 7.5 ug/mm^3)", url="https://www.iso.org/standard/67596.html"),
}


# ===========================================================================
# Prediction container
# ===========================================================================
class ModelPrediction(StrictModel):
    name: PropertyName
    value: float
    low: float
    high: float
    unit: Unit
    interval: str = "range"  # e.g. "95% prediction interval", "5-95 percentile of parameter sampling"
    model: str
    equation: str = ""
    assumptions: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    inputs: dict[str, float] = Field(default_factory=dict)
    checks: list[CheckResult] = Field(default_factory=list)
    conditions_note: str = ""
    notes: str = ""

    def derivation(self, input_record_ids: Sequence[str] = ()) -> Derivation:
        return Derivation(model=self.model, equation=self.equation, assumptions=list(self.assumptions),
                          input_record_ids=list(input_record_ids))

    def to_record(
        self,
        candidate: Candidate,
        conditions: Optional[Conditions] = None,
        input_record_ids: Sequence[str] = (),
        protocol_id: Optional[str] = None,
    ) -> PropertyRecord:
        unc = Uncertainty(
            kind=UncertaintyKind.FIT_STANDARD_ERROR if "prediction interval" in self.interval else UncertaintyKind.UNQUANTIFIED
            if self.low == self.high else UncertaintyKind.REFERENCE_DISCREPANCY,
            value=0.5 * (self.high - self.low),
            unit=self.unit,
            interval="range",
            samples=[self.low, self.high],
            description=f"{self.interval}: [{self.low:.4g}, {self.high:.4g}] {self.unit.value}",
        )
        rec = make_record(
            name=self.name,
            value=self.value,
            unit=self.unit,
            candidate=candidate,
            method=MethodSpec(name=self.model, protocol_id=protocol_id, engine="fracture_md.composites.property_models",
                              description=self.equation, assumptions=list(self.assumptions)),
            origin=PropertyOrigin.DERIVED_MODEL,
            scale=PropertyScale.MACROSCOPIC,
            conditions=conditions or Conditions(temperature=q(310.15, Unit.KELVIN), loading="quasistatic",
                                                environment=self.conditions_note or "model estimate"),
            uncertainty=[unc],
            checks=self.checks,
            derivation=self.derivation(input_record_ids),
            notes=self.notes,
        )
        rec.citations = list(self.citations)
        return rec


def _pct(samples: np.ndarray, lo: float = 5.0, hi: float = 95.0) -> tuple[float, float]:
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


# ===========================================================================
# (c) Hardness conversions
# ===========================================================================
HV_LOAD_N = {"HV0.1": 0.9807, "HV0.2": 1.961, "HV0.3": 2.942, "HV0.5": 4.903, "HV1": 9.807, "HV3": 29.42, "HV5": 49.03}


def hv_to_gpa(hv: float) -> float:
    """HV (kgf/mm^2, sloped contact area) -> GPa: H = HV * 9.80665e-3.  82 HV -> 0.804 GPa.
    Vickers (sloped-area) pressures only: for H_IT / MD projected-area hardness use h_it_mpa_to_hv
    (units.h_it_to_hv); the unit table would overstate HV by 7.9 %."""
    return float(convert(hv, Unit.HV, Unit.GPA))


def gpa_to_hv(h_gpa: float) -> float:
    """GPa (Vickers load / sloped area) -> HV.  Never for a projected-area hardness (see hv_to_gpa)."""
    return float(convert(h_gpa, Unit.GPA, Unit.HV))


def hv_from_force_diagonal(force_n: float, diagonal_mm: float) -> float:
    """ISO 6507-1: HV = 0.1891 F / d^2 (F in N, d = mean diagonal in mm).
    0.1891 = 2 sin(68 deg) / 9.80665."""
    return 2.0 * math.sin(math.radians(68.0)) / KGF_N * force_n / diagonal_mm**2


def diagonal_from_hv(hv: float, force_n: float) -> float:
    """Mean indentation diagonal (mm) expected for a hardness at a test force."""
    return math.sqrt(2.0 * math.sin(math.radians(68.0)) / KGF_N * force_n / hv)


def hv_to_h_it_mpa(hv: float) -> float:
    """ISO 14577: HV = 0.0945 H_IT (H_IT in N/mm^2 = MPa, projected area)."""
    return hv / ISO14577_HV_PER_MPA


def h_it_mpa_to_hv(h_it_mpa: float) -> float:
    return ISO14577_HV_PER_MPA * h_it_mpa


def mean_pressure_projected_mpa_to_hv(p_mpa: float) -> float:
    """Mean contact pressure on the projected area (as in the Johnson ECM) -> HV (= 0.0945 p)."""
    return h_it_mpa_to_hv(p_mpa)


# ===========================================================================
# (a) Empirical hardness calibration
# ===========================================================================
class CalibrationPoint(StrictModel):
    material: str
    filler_type: str
    phi: Optional[float]  # None when the source gives no usable filler content (held out)
    phi_low: Optional[float]
    phi_high: Optional[float]
    phi_source: str  # "reported", "computed" or "unknown"
    hv: float
    load: str
    method: str
    include_in_fit: bool
    citation: str
    notes: str = ""
    material_class: str = "unknown"  # milled / direct / printed / experimental
    material_group: str = ""  # same material from different labs -> averaged before fitting


HV_CLASSES = ("printed_experimental", "direct", "milled")  # first = reference class of the offset model


def hv_class(material_class: str) -> str:
    """Map a calibration material class to the offset-model class."""
    return "printed_experimental" if material_class in ("printed", "experimental", "printed_experimental") else material_class


class HardnessCalibration(StrictModel):
    ln_hv_m: float
    a: float
    cov: list[list[float]]
    s: float  # residual standard error in ln HV
    dof: int
    r2: float
    t95: float
    points: list[CalibrationPoint]
    residuals_ln: list[float]
    citations: dict[str, Citation]
    source_path: str
    # class-offset model ln HV = b0 + a phi + d_class (reference class printed_experimental)
    class_coef: list[float] = Field(default_factory=list)  # [b0, a, d_direct, d_milled]
    class_cov: list[list[float]] = Field(default_factory=list)
    class_s: float = 0.0
    class_dof: int = 0
    class_t95: float = 1.96
    class_phi_range: dict[str, tuple[float, float]] = Field(default_factory=dict)
    class_residuals_ln: dict[str, list[float]] = Field(default_factory=dict)

    def class_offset(self, material_class: str) -> float:
        c = hv_class(material_class)
        if c not in HV_CLASSES:
            raise ValueError(f"unknown hardness material class {material_class!r}; use one of {HV_CLASSES}")
        return 0.0 if c == HV_CLASSES[0] else self.class_coef[1 + HV_CLASSES.index(c)]

    def predict_class(self, phi: float, material_class: str = "printed_experimental", matrix_hv: Optional[float] = None,
                      dc_ratio: float = 1.0) -> tuple[float, float, float]:
        """Median HV and 95 % prediction interval at phi for one material class (offset model)."""
        c = hv_class(material_class)
        x = np.zeros(len(self.class_coef))
        x[0], x[1] = 1.0, phi
        if c != HV_CLASSES[0]:
            x[1 + HV_CLASSES.index(c)] = 1.0
        mu = float(x @ np.asarray(self.class_coef))
        se = math.sqrt(self.class_s**2 + float(x @ np.asarray(self.class_cov) @ x))
        if matrix_hv:
            mu += math.log(matrix_hv) - (self.class_coef[0] + self.class_offset(c))
        mu += math.log(dc_ratio)
        return math.exp(mu), math.exp(mu - self.class_t95 * se), math.exp(mu + self.class_t95 * se)

    @property
    def hv_m(self) -> float:
        return math.exp(self.ln_hv_m)

    def predict_ln(self, phi: float) -> tuple[float, float]:
        """(mean ln HV, standard error of a NEW observation) at filler volume fraction phi."""
        x = np.array([1.0, phi])
        cov = np.asarray(self.cov)
        se_pred = math.sqrt(self.s**2 + float(x @ cov @ x))
        return self.ln_hv_m + self.a * phi, se_pred

    def predict(self, phi: float, matrix_hv: Optional[float] = None, dc_ratio: float = 1.0) -> tuple[float, float, float]:
        """Median HV and 95 % prediction interval at phi.

        matrix_hv replaces the fitted intercept (a known unfilled-matrix HV); dc_ratio scales
        the matrix hardness linearly with conversion relative to the calibration set
        (Ferracane 1985: hardness linear in DC within one resin)."""
        mu, se = self.predict_ln(phi)
        shift = (math.log(matrix_hv) - self.ln_hv_m) if matrix_hv else 0.0
        mu = mu + shift + math.log(dc_ratio)
        return math.exp(mu), math.exp(mu - self.t95 * se), math.exp(mu + self.t95 * se)


def _t_quantile_975(dof: int) -> float:
    # Student t 0.975 quantiles (exact table for small dof; normal limit beyond).
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
             12: 2.179, 14: 2.145, 16: 2.120, 18: 2.101, 20: 2.086, 22: 2.074, 24: 2.064, 26: 2.056, 28: 2.048,
             30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
    if dof in table:
        return table[dof]
    keys = sorted(table)
    for k0, k1 in zip(keys, keys[1:]):
        if k0 < dof < k1:
            return table[k0] + (table[k1] - table[k0]) * (dof - k0) / (k1 - k0)
    return 1.96


def calibration_points(path: Path | str | None = None) -> tuple[list[CalibrationPoint], dict[str, Citation]]:
    raw = json.loads(Path(path or HARDNESS_CALIBRATION_PATH).read_text())
    rm = raw["matrix_density"]
    fd = raw["filler_density"]
    cits = {k: Citation(key=k, **v) for k, v in raw["citations"].items()}
    pts = []
    for p in raw["points"]:
        if p["filler_volume_fraction_reported"] is not None:
            phi = float(p["filler_volume_fraction_reported"])
            lo, hi, src = phi - 0.02, phi + 0.02, "reported"
        elif p["filler_weight_fraction"] is None:
            phi, lo, hi, src = None, None, None, "unknown"
        else:
            w = float(p["filler_weight_fraction"])
            if p.get("filler_weight_basis") == "as_weighed_silanised":
                w = w * (1.0 - float(p.get("silane_wt_on_filler", 0.0)))  # ash (inorganic) basis
            if w == 0.0:
                phi, lo, hi, src = 0.0, 0.0, 0.0, "reported"
            else:
                d = fd[p["filler_type"]]
                phi = C.two_phase_volume_fraction(w, d["value"], rm["value"])
                a = C.two_phase_volume_fraction(w, d["high"], rm["low"])
                b = C.two_phase_volume_fraction(w, d["low"], rm["high"])
                lo, hi, src = min(a, b), max(a, b), "computed"
        pts.append(CalibrationPoint(material=p["material"], filler_type=p["filler_type"], phi=phi, phi_low=lo, phi_high=hi,
                                    phi_source=src, hv=float(p["hv"]), load=p["load"], method=p["method"],
                                    include_in_fit=bool(p["include_in_fit"]), citation=p["citation"], notes=p.get("notes", ""),
                                    material_class=p.get("material_class", "unknown"),
                                    material_group=p.get("material_group", p["material"])))
    return pts, cits


@lru_cache(maxsize=4)
def _fit_cached(path: str) -> HardnessCalibration:
    pts, cits = calibration_points(path)
    fit_all = [p for p in pts if p.include_in_fit]
    if any(p.phi is None for p in fit_all):
        raise ValueError("calibration points in the fit need a filler fraction")
    # average duplicate materials (same material, different labs) in ln HV: one point per material
    groups: dict[str, list[CalibrationPoint]] = {}
    for p in fit_all:
        groups.setdefault(p.material_group or p.material, []).append(p)
    gx = np.array([float(np.mean([p.phi for p in g])) for g in groups.values()])
    gy = np.array([float(np.mean(np.log([p.hv for p in g]))) for g in groups.values()])
    gc = [hv_class(g[0].material_class) for g in groups.values()]
    x = gx
    y = gy
    A = np.c_[np.ones_like(x), x]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    dof = len(x) - 2
    s = float(math.sqrt(r @ r / dof))
    cov = (s**2) * np.linalg.inv(A.T @ A)
    r2 = 1.0 - float(r @ r) / float((y - y.mean()) @ (y - y.mean()))
    # class-offset model
    Ac = np.column_stack([np.ones_like(x), x] + [np.array([1.0 if c == k else 0.0 for c in gc]) for k in HV_CLASSES[1:]])
    cc, *_ = np.linalg.lstsq(Ac, y, rcond=None)
    rc = y - Ac @ cc
    dofc = len(x) - Ac.shape[1]
    sc = float(math.sqrt(rc @ rc / dofc))
    covc = (sc**2) * np.linalg.inv(Ac.T @ Ac)
    rng_c = {k: (float(min(x[i] for i, c in enumerate(gc) if c == k)), float(max(x[i] for i, c in enumerate(gc) if c == k)))
             for k in HV_CLASSES if k in gc}
    res_c = {k: [float(rc[i]) for i, c in enumerate(gc) if c == k] for k in HV_CLASSES}
    return HardnessCalibration(ln_hv_m=float(coef[0]), a=float(coef[1]), cov=cov.tolist(), s=s, dof=dof, r2=r2,
                               t95=_t_quantile_975(dof), points=pts, residuals_ln=r.tolist(), citations=cits,
                               source_path=str(path), class_coef=cc.tolist(), class_cov=covc.tolist(), class_s=sc,
                               class_dof=dofc, class_t95=_t_quantile_975(dofc), class_phi_range=rng_c,
                               class_residuals_ln=res_c)


def fit_hardness_calibration(path: Path | str | None = None) -> HardnessCalibration:
    return _fit_cached(str(path or HARDNESS_CALIBRATION_PATH))


def predict_vickers_empirical(
    phi: float,
    matrix_hv: Optional[float] = None,
    dc_ratio: float = 1.0,
    phi_uncertainty: float = 0.02,
    water_ageing_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    calibration: Optional[HardnessCalibration] = None,
    label_load: str = "HV0.2-HV0.5",
    material_class: Optional[str] = "printed_experimental",
) -> ModelPrediction:
    """Empirical Vickers hardness (model a).  Range = 95 % prediction interval of the fit,
    widened by the filler-fraction uncertainty and an optional water-ageing factor
    (nominal, low, high multipliers; default no ageing = dry-stored calibration).

    material_class selects the class-offset model (ln HV = b0 + a phi + d_class):
    "printed_experimental" (default; lab-made / printed, light-cured, no heat-pressure
    polymerisation - the right class for a new printed resin), "direct" or "milled"
    (heat/pressure-polymerised CAD/CAM blocks).  The milled-class value is returned in
    ``inputs['hv_milled_class_optimistic']`` as an optimistic bound.  material_class=None
    uses the legacy pooled fit (all classes, no offset)."""
    cal = calibration or fit_hardness_calibration()
    if material_class is None:
        pr = lambda ph: cal.predict(ph, matrix_hv, dc_ratio)  # noqa: E731
    else:
        pr = lambda ph: cal.predict_class(ph, material_class, matrix_hv, dc_ratio)  # noqa: E731
    v, lo, hi = pr(phi)
    _, lo2, _ = pr(max(phi - phi_uncertainty, 0.0))
    _, _, hi2 = pr(phi + phi_uncertainty)
    milled = cal.predict_class(phi, "milled", matrix_hv, dc_ratio)
    pooled = cal.predict(phi, matrix_hv, dc_ratio)
    wn, wl, wh = water_ageing_factor
    fit_pts = [p for p in cal.points if p.include_in_fit]
    phis = [p.phi for p in fit_pts]
    checks = [CheckResult(name="phi_within_calibration_range", passed=min(phis) <= phi <= max(phis),
                          metric=phi, details=f"calibration spans phi = {min(phis):.2f}-{max(phis):.2f} (all classes)")]
    if material_class is not None:
        c = hv_class(material_class)
        a, b = cal.class_phi_range[c]
        checks.append(CheckResult(name="phi_within_class_calibration_range", passed=bool(a - 1e-9 <= phi <= b + 1e-9), metric=phi,
                                  details=f"class {c} spans phi = {a:.2f}-{b:.2f}; outside = extrapolation of the class offset"))
    cls_txt = "pooled (no class offset)" if material_class is None else f"class {hv_class(material_class)}"
    return ModelPrediction(
        name=PropertyName.VICKERS_HARDNESS_ESTIMATE,
        value=v * wn, low=min(lo, lo2) * wl, high=max(hi, hi2) * wh, unit=Unit.HV,
        interval=f"95% prediction interval of the calibration fit (+/- {phi_uncertainty} in phi)",
        model="empirical exponential filler rule (Braem-type), OLS fit to published HV vs filler vol%",
        equation=(f"ln HV = ln HV_m + a phi; HV_m = {cal.hv_m:.2f} HV, a = {cal.a:.3f}, s = {cal.s:.3f} (ln units), n = {cal.dof + 2}, R^2 = {cal.r2:.3f}"
                  if material_class is None else
                  f"ln HV = b0 + a phi + d_class; b0 = {cal.class_coef[0]:.3f}, a = {cal.class_coef[1]:.3f}, "
                  f"d_direct = {cal.class_coef[2]:.3f}, d_milled = {cal.class_coef[3]:.3f}, s = {cal.class_s:.3f}, n = {cal.class_dof + 4} materials; {cls_txt}"),
        assumptions=[
            f"calibration on optical Vickers {label_load} of dry or unspecified-storage commercial/experimental composites; HV1 and ISO 14577-derived values held out",
            "filler vol% computed from ash wt% with assumed densities where not reported (see hardness_calibration.json)",
            "matrix hardness enters only through the intercept (or matrix_hv override) and a linear DC scaling",
            "filler type, dispersion and silane quality are absorbed in the residual scatter (s); poorly dispersed printed resins sit below the fit",
            "interpolator, not physics: exponential rules break down for E_f/E_m > 10 (Chantler 1999)",
            f"prediction for {cls_txt}; milled CAD/CAM blocks (heat/pressure polymerised) sit ~{math.exp(cal.class_coef[3]) if cal.class_coef else 1:.2f}x above lab-made/printed light-cured materials at equal phi - the milled-class value is an optimistic bound only",
            "duplicate materials from different labs averaged (one point per material); phi on the inorganic (ash) basis",
            "dry calibration: after 7 d water expect lower HV (pass water_ageing_factor)",
        ],
        citations=[CITES["Braem1989"], CITES["Chantler1999"], CITES["ISO6507"]] + list(cal.citations.values()),
        inputs={"phi": phi, "matrix_hv": matrix_hv or cal.hv_m, "dc_ratio": dc_ratio, "water_ageing_nominal": wn,
                "hv_milled_class_optimistic": milled[0] * wn, "hv_milled_class_high": milled[2] * wh, "hv_pooled_fit": pooled[0] * wn},
        checks=checks,
        notes=f"{cls_txt}: {v * wn:.1f} HV; optimistic bound (milled-block class) {milled[0] * wn:.1f} HV; legacy pooled fit {pooled[0] * wn:.1f} HV",
        conditions_note="dry / 24 h storage, 23 C test (calibration conditions) unless water_ageing_factor given",
    )


# ===========================================================================
# Micromechanics helpers (analytic; the FFT solver lives in composites.homogenization)
# ===========================================================================
def _kg(E: float, nu: float) -> tuple[float, float]:
    return E / (3.0 * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def hashin_shtrikman_lower(E_m: float, nu_m: float, fillers: Sequence[tuple[float, float, float]]) -> tuple[float, float]:
    """Multi-phase Hashin-Shtrikman lower bound with the (softest) matrix as reference
    (identical to Mori-Tanaka for spherical inclusions).  fillers = [(phi_i, E_i, nu_i)].
    Returns (E_eff, nu_eff) in the units of E."""
    Km, Gm = _kg(E_m, nu_m)
    phi_m = 1.0 - sum(f[0] for f in fillers)
    if phi_m <= 0:
        raise ValueError("filler volume fractions must sum to < 1")
    zeta = Gm / 6.0 * (9.0 * Km + 8.0 * Gm) / (Km + 2.0 * Gm)
    k_terms = [(phi_m, Km)] + [(p, _kg(E, nu)[0]) for p, E, nu in fillers]
    g_terms = [(phi_m, Gm)] + [(p, _kg(E, nu)[1]) for p, E, nu in fillers]
    K = 1.0 / sum(p / (k + 4.0 * Gm / 3.0) for p, k in k_terms) - 4.0 * Gm / 3.0
    G = 1.0 / sum(p / (g + zeta) for p, g in g_terms) - zeta
    E = 9.0 * K * G / (3.0 * K + G)
    nu = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))
    return E, nu


def johnson_ecm_hardness_mpa(E_mpa: float, sigma_y_mpa: float, tan_beta: float = math.tan(math.radians(19.7))) -> float:
    """Johnson expanding-cavity model, conical-indenter form:
    p_m = (2/3) sigma_y [1 + ln(E tan(beta) / (3 sigma_y))]  (mean pressure on the projected area).
    tan_beta=1 gives the simplified form H = (2/3) sigma_y [1 + ln(E/(3 sigma_y))].
    Clamped below at the Hertz-like elastic limit ln(...) >= 0."""
    arg = E_mpa * tan_beta / (3.0 * sigma_y_mpa)
    return (2.0 / 3.0) * sigma_y_mpa * (1.0 + math.log(max(arg, 1.0)))


def pukanszky_yield(sigma_y_m: float, phi: float, B: float) -> float:
    """Pukanszky (1990) composite yield stress: sigma_y = sigma_y,m (1-phi)/(1+2.5 phi) exp(B phi)."""
    return sigma_y_m * (1.0 - phi) / (1.0 + 2.5 * phi) * math.exp(B * phi)


def predict_vickers_johnson(
    phi: float,
    E_m_gpa: tuple[float, float, float] = (2.5, 2.0, 3.5),
    nu_m: float = 0.35,
    sigma_yc_m_mpa: tuple[float, float, float] = (120.0, 100.0, 140.0),
    fillers: Optional[Sequence[tuple[float, float, float]]] = None,
    E_f_gpa: tuple[float, float, float] = (70.0, 65.0, 80.0),
    nu_f: float = 0.22,
    B_interface: tuple[float, float, float] = (3.0, 2.0, 4.0),
    E_composite_gpa: Optional[tuple[float, float, float]] = None,
    n_samples: int = 2000,
    seed: int = 0,
) -> ModelPrediction:
    """Physics cross-check (model b).  Tuples are (nominal, low, high).  Composite E from the
    HS lower bound (or ``E_composite_gpa`` from the FFT homogenisation), composite compressive
    yield from the Pukanszky interface model with parameter B (ASSUMED range), hardness from
    the Johnson ECM (cone, beta = 19.7 deg), converted to HV with HV = 0.0945 p_m[MPa]."""
    rng = np.random.default_rng(seed)

    def draw(t):
        return rng.uniform(t[1], t[2], n_samples)

    Em, sy, Bs, Ef = draw(E_m_gpa), draw(sigma_yc_m_mpa), draw(B_interface), draw(E_f_gpa)
    Ec_draw = draw(E_composite_gpa) if E_composite_gpa else None

    def one(em, s_y, b, ef, ec=None):
        if ec is None:
            fl = [(p, ef if fillers is None else E, nu) for p, E, nu in (fillers or [(phi, ef, nu_f)])]
            ec, _ = hashin_shtrikman_lower(em, nu_m, fl)
        syc = pukanszky_yield(s_y, phi, b)
        p = johnson_ecm_hardness_mpa(ec * 1000.0, syc)
        return mean_pressure_projected_mpa_to_hv(p), ec, syc

    hv_nom, ec_nom, syc_nom = one(E_m_gpa[0], sigma_yc_m_mpa[0], B_interface[0], E_f_gpa[0],
                                  E_composite_gpa[0] if E_composite_gpa else None)
    samples = np.array([one(Em[i], sy[i], Bs[i], Ef[i], None if Ec_draw is None else Ec_draw[i])[0] for i in range(n_samples)])
    lo, hi = _pct(samples)
    # verification: phi = 0 limit vs the measured unfilled printed resin (16.4 HV0.3, Lim 2026 [S14])
    Km, _ = _kg(E_m_gpa[0], nu_m)
    hv0 = mean_pressure_projected_mpa_to_hv(johnson_ecm_hardness_mpa(E_m_gpa[0] * 1000.0, sigma_yc_m_mpa[0]))
    checks = [CheckResult(name="ecm_unfilled_limit_vs_measured_16.4HV", passed=bool(abs(hv0 / 16.4 - 1.0) <= 0.3), metric=hv0,
                          tolerance=0.3, unit=Unit.HV, details="phi=0 prediction vs unfilled printed resin NextDent C&B 16.4 HV0.3 [S14]")]
    return ModelPrediction(
        name=PropertyName.VICKERS_HARDNESS_ESTIMATE,
        value=hv_nom, low=lo, high=hi, unit=Unit.HV, checks=checks,
        interval="5-95 percentile over uniform sampling of E_m, sigma_y,m, E_f, B",
        model="Johnson expanding-cavity model with HS-lower-bound composite modulus and Pukanszky composite yield",
        equation="p_m = (2/3) sigma_y,c [1 + ln(E_c tan(19.7 deg)/(3 sigma_y,c))]; sigma_y,c = sigma_y,m (1-phi)/(1+2.5phi) exp(B phi); HV = 0.0945 p_m[MPa]",
        assumptions=[
            "matrix compressive yield 100-140 MPa (brief section 4.4 illustration uses 120 MPa; ASSUMED range, to be measured by compression test)",
            "cone form of Johnson's ECM with beta = 19.7 deg (Vickers-equivalent cone) inferred from Johnson 1985; only the spherical form was seen verbatim (Olah 2024)",
            "Pukanszky B = 2-4 is an ASSUMED range for well-silanised fillers (B is an interface-interaction parameter, not measured here)",
            "homogenised continuum: indent diagonal (~50-120 um) >> particle size (~0.5-1.5 um)",
            "limitations: polymers are viscoelastic/pressure-sensitive (hardness depends on dwell and load), rigid particles also carry load directly (ECM with a homogenised yield stress ignores this), and Pukanszky was derived for tensile yield",
        ],
        citations=[CITES["Johnson1970"], CITES["Olah2024"], CITES["Koch2009"], CITES["HS1963"], CITES["MT1973"], CITES["Pukanszky1990"], CITES["ISO14577"]],
        inputs={"phi": phi, "E_composite_GPa_nominal": ec_nom, "sigma_y_composite_MPa_nominal": syc_nom},
        notes="cross-check of the empirical model, not a calibrated predictor",
    )


# ===========================================================================
# Formulation helpers
# ===========================================================================
def _db() -> C.ConstituentDatabase:
    return C.load_constituents()


def matrix_monomer_volume_fractions(formulation: Formulation, db=None, state="cured") -> dict[str, float]:
    db = db or _db()
    mons = [c for c in formulation.components if c.role in (ComponentRole.BASE_MONOMER, ComponentRole.DILUENT_MONOMER)]
    rho = {c.name: C.component_density(c, db, state) for c in mons}
    return C.weight_to_volume_fractions({c.name: c.weight_fraction for c in mons}, rho)


class FillerClass(StrictModel):
    name: str
    filler_type: str
    volume_fraction: float  # of the composite
    d50_um: float
    refractive_index: float
    E_gpa: Optional[float] = None
    nu: Optional[float] = None
    beta_packing: float  # monomodal (virtual) maximum packing fraction of this class alone


def filler_classes(formulation: Formulation, db=None, state="cured") -> list[FillerClass]:
    db = db or _db()
    vf = C.formulation_volume_fractions(formulation, db, state)
    out = []
    for comp in formulation.components:
        if comp.role != ComponentRole.FILLER:
            continue
        c = db.find(comp.name)
        d50 = comp.particle_d50.magnitude(Unit.MICROMETER) if comp.particle_d50 else (c.get("d50", Unit.MICROMETER) if c else None)
        n = comp.refractive_index if comp.refractive_index is not None else (c.get("refractive_index", Unit.DIMENSIONLESS) if c else None)
        if d50 is None or n is None:
            raise ValueError(f"filler {comp.name!r} needs d50 and refractive index")
        out.append(FillerClass(
            name=comp.name, filler_type=(c.filler_type if c else "unknown") or "unknown", volume_fraction=vf[comp.name],
            d50_um=d50, refractive_index=n,
            E_gpa=c.get("youngs_modulus", Unit.GPA) if c else None, nu=c.get("poisson_ratio", Unit.DIMENSIONLESS) if c else None,
            beta_packing=(c.get("packing_fraction_monomodal", Unit.DIMENSIONLESS, 0.60) if c else 0.60),  # type: ignore[arg-type]
        ))
    return out


def predict_hardness_for_formulation(formulation: Formulation, dc_ratio: float = 1.0, **kw) -> ModelPrediction:
    phi = C.filler_volume_fraction(formulation, state="cured")
    return predict_vickers_empirical(phi, dc_ratio=dc_ratio, **kw)


# ===========================================================================
# Viscosity (printability)
# ===========================================================================
def matrix_viscosity(
    monomer_weights: dict[str, float],
    temperature_k: float,
    db=None,
    eta25: Optional[dict[str, float]] = None,
    ea_over_r: Optional[dict[str, float]] = None,
) -> float:
    """Monomer-blend viscosity (Pa s): Grunberg-Nissan log mixing on mole fractions (no
    interaction term), each monomer eta_i(T) = eta_i(298.15 K) exp[(Ea_i/R)(1/T - 1/298.15)].
    ``eta25`` / ``ea_over_r`` override database values per monomer (used for sampling)."""
    db = db or _db()
    mols, lns = {}, {}
    for name, w in monomer_weights.items():
        c = db.lookup(name)
        mols[name] = w / c.get("molar_mass", Unit.AMU)
        e25 = (eta25 or {}).get(name, c.get("viscosity_25C", Unit.PA_S))
        ear = (ea_over_r or {}).get(name, c.get("viscosity_Ea_over_R", Unit.KELVIN))
        lns[name] = math.log(e25) + ear * (1.0 / temperature_k - 1.0 / 298.15)
    tot = sum(mols.values())
    return math.exp(sum(mols[k] / tot * lns[k] for k in mols))


def _sample_monomer_params(monomer_names: Iterable[str], rng, db=None) -> tuple[dict[str, float], dict[str, float]]:
    """Log-uniform eta_25 and uniform Ea/R within each monomer's stored [low, high] range."""
    db = db or _db()
    e25, ea = {}, {}
    for n in monomer_names:
        c = db.lookup(n)
        lo, hi = c.prop("viscosity_25C").range(Unit.PA_S)
        e25[n] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        lo, hi = c.prop("viscosity_Ea_over_R").range(Unit.KELVIN)
        ea[n] = rng.uniform(lo, hi)
    return e25, ea


# Grunberg-Nissan (G12 = 0) check against a measured binary: Bis-GMA/TEGDMA 50/50 wt% = 0.204 Pa s
# at 25 C (Alrahlah 2021).  The ideal rule over-predicts such mixtures (see docs); the matrix
# viscosity sampling therefore includes a multiplicative model-error factor.
GN_MODEL_ERROR_RANGE = (0.45, 1.1)


def de_larrard_packing(classes: Sequence[tuple[float, float, float]]) -> float:
    """Virtual packing density of a multimodal mix (de Larrard linear packing model).
    classes = [(solid volume share y_i, diameter d_i, monomodal packing beta_i)], any order.
    gamma_i = beta_i / (1 - sum_{j coarser} [1 - beta_i + b_ij beta_i (1 - 1/beta_j)] y_j
                          - sum_{j finer} [1 - a_ij beta_i / beta_j] y_j);   phi_m = min_i gamma_i
    with a_ij = sqrt(1 - (1 - d_j/d_i)^1.02) (loosening), b_ij = 1 - (1 - d_i/d_j)^1.50 (wall)."""
    cl = sorted(classes, key=lambda t: -t[1])
    tot = sum(t[0] for t in cl)
    ys = [t[0] / tot for t in cl]
    gam = []
    for i, (_, di, bi) in enumerate(cl):
        s = 0.0
        for j, (_, dj, bj) in enumerate(cl):
            if j == i:
                continue
            if dj >= di and j < i:  # coarser class j exerts a wall effect on i
                b = 1.0 - (1.0 - di / dj) ** 1.50
                s += (1.0 - bi + b * bi * (1.0 - 1.0 / bj)) * ys[j]
            else:  # finer class j loosens i
                a = math.sqrt(max(0.0, 1.0 - (1.0 - dj / di) ** 1.02))
                s += (1.0 - a * bi / bj) * ys[j]
        gam.append(bi / (1.0 - s))
    return min(gam)


def krieger_dougherty(eta_m: float, phi: float, phi_m: float, intrinsic: float = 2.5) -> float:
    """eta = eta_m (1 - phi/phi_m)^(-[eta] phi_m); +inf at/after phi_m."""
    if phi >= phi_m:
        return math.inf
    return eta_m * (1.0 - phi / phi_m) ** (-intrinsic * phi_m)


def effective_volume_fraction(phi: float, d_um: float, bound_layer_nm: float) -> float:
    """Bound/adsorbed-layer (silane + immobilised monomer) effective volume: phi (1 + delta/r)^3."""
    r_nm = 0.5 * d_um * 1000.0
    return phi * (1.0 + bound_layer_nm / r_nm) ** 3


def suspension_viscosity(
    eta_m: float, classes: Sequence[FillerClass], bound_layer_nm: float = 2.0, beta_scale: float = 1.0
) -> tuple[float, float, float]:
    """(eta, phi_eff_total, phi_m) of a filled paste at high shear (Newtonian KD estimate)."""
    eff = [(effective_volume_fraction(c.volume_fraction, c.d50_um, bound_layer_nm), c.d50_um, min(0.74, c.beta_packing * beta_scale))
           for c in classes if c.volume_fraction > 0]
    if not eff:
        return eta_m, 0.0, 1.0
    phi_eff = sum(e[0] for e in eff)
    phi_m = de_larrard_packing(eff)
    return krieger_dougherty(eta_m, phi_eff, phi_m), phi_eff, phi_m


def predict_viscosity(
    formulation: Formulation,
    temperature_k: Optional[float] = None,
    limit_pa_s: float = 3.0,
    n_samples: int = 400,
    seed: int = 0,
    db=None,
    calibrated: bool = True,
    shear_rate_s: float = 30.0,
) -> ModelPrediction:
    """Paste viscosity at print temperature (Pa s), nominally at ~30 1/s (Newtonian KD model).

    calibrated=True (default) multiplies the KD estimate by the filled-paste correction factor
    K_paste (fitted so that a SprintRay-Ceramic-Crown-like 52 wt% paste gives the claimed
    2.5-6 Pa s at 30 C; range 2.5/raw - 6/raw) and adds a fumed-silica network yield-stress term
    eta_app = eta + tau_y / shear_rate, tau_y ~ U(0, 30 Pa x (wt% fumed silica)^2) (ASSUMED, widens the upper tail).
    Range from sampling each monomer's eta_25 and Ea/R within its stored range, a
    Grunberg-Nissan model-error factor (GN_MODEL_ERROR_RANGE), beta_packing (x0.93-1.05) and the
    bound layer (1-5 nm).  Check: viscosity <= limit (open-vat DLP ~3 Pa s at 30 1/s)."""
    db = db or _db()
    T = temperature_k or (formulation.processing.print_temperature.magnitude(Unit.KELVIN)
                          if formulation.processing.print_temperature else 298.15)
    mw = C.monomer_weight_fractions(formulation)
    classes = filler_classes(formulation, db, state="uncured")
    eta_m = matrix_viscosity(mw, T, db)
    eta, phi_eff, phi_m = suspension_viscosity(eta_m, classes, 2.0)
    if calibrated:
        k0, kl, kh = paste_correction_factor()
        fs_pct = 100.0 * sum(c.weight_fraction for c in formulation.components
                             if c.role == ComponentRole.FILLER and "fumed" in db.lookup(c.name).name.lower())
        tau_max = FUMED_SILICA_TAU_Y_PA_PER_WT2 * fs_pct**2
    else:
        k0 = kl = kh = 1.0
        tau_max = 0.0
    eta_raw = eta
    eta = eta * k0 + 0.25 * tau_max / shear_rate_s
    rng = np.random.default_rng(seed)
    s = []
    for _ in range(n_samples):
        e25, ea = _sample_monomer_params(mw, rng, db)
        em = matrix_viscosity(mw, T, db, e25, ea) * rng.uniform(*GN_MODEL_ERROR_RANGE)
        e, _, _ = suspension_viscosity(em, classes, rng.uniform(1.0, 5.0), rng.uniform(0.93, 1.05))
        e = e * rng.uniform(kl, kh) + rng.uniform(0.0, tau_max) / shear_rate_s
        s.append(min(e, 1e6))
    lo, hi = _pct(np.array(s))
    phi = sum(c.volume_fraction for c in classes)
    checks = [CheckResult(name="printable_open_vat", passed=bool(eta <= limit_pa_s), metric=min(eta, 1e9), tolerance=limit_pa_s,
                          unit=Unit.PA_S, details=f"nominal eta {eta:.3g} Pa s vs limit {limit_pa_s} Pa s at {T - 273.15:.0f} C")]
    return ModelPrediction(
        name=PropertyName.VISCOSITY_ESTIMATE,
        value=min(eta, 1e9), low=lo, high=hi, unit=Unit.PA_S,
        interval="5-95 percentile of parameter sampling",
        model="Grunberg-Nissan/Arrhenius matrix + Krieger-Dougherty suspension, de Larrard packing, bound-layer penalty",
        equation="ln eta_m = sum x_i ln eta_i(T); eta = eta_m (1 - phi_eff/phi_m)^(-2.5 phi_m); phi_eff = sum phi_i (1 + delta/r_i)^3",
        assumptions=[
            "Newtonian high-shear estimate standing in for eta at ~30 1/s; shear thinning, thixotropy and fumed-silica networks are not modelled",
            "no interaction term in the Grunberg-Nissan rule (G12 = 0); Arrhenius temperature dependence per monomer; the ideal rule over-predicts Bis-GMA/TEGDMA 50/50 (0.204 Pa s measured, Alrahlah 2021) by ~1.8x, covered by a 0.45-1.1 model-error factor in the range",
            "Ea/R for UDMA, Bis-EMA, TEGDMA, D3MA are ASSUMED (no verified data); Bis-GMA derived from Alrahlah 2021",
            "[eta] = 2.5 (Einstein, spheres); irregular milled glass raises [eta] - partly absorbed in the monomodal packing beta (ASSUMED values in constituents.json)",
            "bound layer delta = 1-5 nm (ASSUMED: silane layer + immobilised monomer); dominates for nanoparticles",
            "de Larrard interaction-function exponents (1.02, 1.50) taken from de Larrard 1999 (not re-verified in this session)",
            "uncured monomer densities used for the paste volume fractions",
            ("filled-paste correction K_paste = %.2f [%.2f-%.2f] applied (calibrated on ONE filled printable paste: SprintRay Ceramic Crown "
             "claim 2.5-6 Pa s at 30 C, proxy composition 52 wt%% Ba glass); fumed-silica network yield stress tau_y = 0-30 Pa x (wt%%)^2 "
             "(ASSUMED) added as tau_y/shear_rate" % (k0, kl, kh)) if calibrated else "UNCALIBRATED Krieger-Dougherty estimate (calibrated=False)",
            "open-vat printability is NOT established by this model: measure P-VISC (rotational rheometry at print temperature)",
        ],
        citations=[CITES["GN1949"], CITES["KD1959"], CITES["deLarrard1999"], CITES["Taylor1998"], CITES["Lee2006"], CITES["S38"], CITES["S24"]]
        + _db().cite(sorted({k for n in mw for k in _db().lookup(n).prop("viscosity_25C").citation_keys})),
        inputs={"temperature_K": T, "eta_matrix_Pa_s": eta_m, "phi_filler_uncured": phi, "phi_eff": phi_eff, "phi_m": phi_m,
                "eta_uncalibrated_Pa_s": eta_raw, "K_paste": k0, "K_paste_low": kl, "K_paste_high": kh,
                "tau_y_max_Pa": tau_max, "shear_rate_1_s": shear_rate_s},
        checks=checks,
        conditions_note=f"uncured paste at {T - 273.15:.1f} C",
    )


FUMED_SILICA_TAU_Y_PA_PER_WT2 = 30.0  # ASSUMED yield stress scale of a fumed-silica network, Pa per (wt%)^2
PASTE_CALIBRATION = {"product": "SprintRay Ceramic Crown (proxy)", "temperature_K": 303.15, "low": 2.5, "high": 6.0,
                     "source": "manufacturer claim, targets.json / RESEARCH_BRIEF (>50 wt% Si-based filler, 2.5-6 Pa s at 30 C)"}


@lru_cache(maxsize=1)
def paste_correction_factor() -> tuple[float, float, float]:
    """(nominal, low, high) multiplier on the KD estimate from the Ceramic-Crown-like paste."""
    f = proxy_formulation(PASTE_CALIBRATION["product"])
    raw = predict_viscosity(f, temperature_k=PASTE_CALIBRATION["temperature_K"], n_samples=1, calibrated=False).value
    lo, hi = PASTE_CALIBRATION["low"], PASTE_CALIBRATION["high"]
    return math.sqrt(lo * hi) / raw, lo / raw, hi / raw


def viscosity_temperature_check(db=None) -> CheckResult:
    """Heated-vat check (Lim 2025, TC-80DP crown resin): 2.36 Pa s at RT -> 0.49 at 50 C (ratio 4.8).
    Compares the model's 25 -> 50 C ratio for a Ceramic-Crown-like paste (within 2x)."""
    f = proxy_formulation(PASTE_CALIBRATION["product"])
    r = (predict_viscosity(f, temperature_k=298.15, n_samples=1, calibrated=False).value
         / predict_viscosity(f, temperature_k=323.15, n_samples=1, calibrated=False).value)
    return CheckResult(name="viscosity_temperature_ratio_TC80DP", passed=bool(4.8 / 2 <= r <= 4.8 * 2), metric=r, tolerance=2.0,
                       details=f"model eta(25 C)/eta(50 C) = {r:.2f} vs TC-80DP 2.36/0.49 = 4.82 (composition of TC-80DP unknown; RT assumed 25 C)")


# ===========================================================================
# Cure depth (Jacobs working curve)
# ===========================================================================
def anomalous_diffraction_q(d_um: float, wavelength_nm: float, n_particle: float, n_medium: float) -> float:
    """van de Hulst anomalous-diffraction extinction efficiency, rho = 2 x |m - 1|, x = pi d n0/lambda."""
    x = math.pi * d_um * 1000.0 * n_medium / wavelength_nm
    rho = 2.0 * x * abs(n_particle / n_medium - 1.0)
    if rho < 1e-4:
        return rho**2 / 2.0
    return 2.0 - 4.0 / rho * math.sin(rho) + 4.0 / rho**2 * (1.0 - math.cos(rho))


def scattering_attenuation_per_um(classes: Sequence[FillerClass], n_resin: float, wavelength_nm: float, transport_fraction: float) -> float:
    """alpha_s = sum_i (3/2) phi_i Q_i f_tr / d_i  (1/um): Beer-Lambert with geometric cross
    sections, i.e. D_p,s = (2/3) d/(phi Q) per class (the Griffith-Halloran geometric form)."""
    return sum(1.5 * c.volume_fraction * anomalous_diffraction_q(c.d50_um, wavelength_nm, c.refractive_index, n_resin)
               * transport_fraction / c.d50_um for c in classes if c.volume_fraction > 0)


def absorption_attenuation_per_um(formulation: Formulation, wavelength_nm: float, db=None) -> float:
    """alpha_a = ln(10) sum_i eps_i c_i  (Napierian, 1/um) from photoinitiators/absorbers in the
    composite (c_i in mol/L of composite).  Molar absorptivities (L/mol/cm) from constituents.json."""
    db = db or _db()
    rho_c = 1.0 / sum(c.weight_fraction / C.component_density(c, db, "uncured") for c in formulation.components)  # g/cm^3
    key = "molar_absorptivity_405" if abs(wavelength_nm - 405) <= abs(wavelength_nm - 385) else "molar_absorptivity_385"
    a_per_cm = 0.0
    for comp in formulation.components:
        if comp.role not in (ComponentRole.PHOTOINITIATOR, ComponentRole.UV_ABSORBER):
            continue
        c = db.find(comp.name)
        if c is None or not c.has(key):
            continue
        conc = comp.weight_fraction * rho_c * 1000.0 / c.get("molar_mass", Unit.AMU)  # mol/L
        a_per_cm += math.log(10.0) * c.get(key, Unit.DIMENSIONLESS) * conc
    return a_per_cm * 1e-4


def resin_refractive_index(formulation: Formulation, state: str = "uncured", db=None) -> float:
    db = db or _db()
    vf = matrix_monomer_volume_fractions(formulation, db, "uncured" if state == "uncured" else "cured")
    key = "refractive_index" if state == "uncured" else "refractive_index_cured"
    return sum(v * db.lookup(n).get(key, Unit.DIMENSIONLESS) for n, v in vf.items())


def predict_cure_depth(
    formulation: Formulation,
    dose_mj_cm2: Optional[float] = None,
    critical_dose_mj_cm2: tuple[float, float, float] = (10.0, 5.0, 20.0),
    transport_fraction: tuple[float, float, float] = (0.07, 0.03, 0.15),
    layer_thickness_um: Optional[float] = None,
    n_samples: int = 400,
    seed: int = 0,
    db=None,
) -> ModelPrediction:
    """Cure depth C_d = D_p ln(E/E_c) (um) at the print wavelength.
    1/D_p = alpha_absorption + alpha_scattering (uncured resin index as the medium).
    Check: C_d >= 2 x layer thickness."""
    db = db or _db()
    pr = formulation.processing
    lam = pr.wavelength_nm or 405.0
    E = dose_mj_cm2 or (pr.exposure_dose.magnitude(Unit.MJ_PER_CM2) if pr.exposure_dose else 50.0)
    layer = layer_thickness_um or (pr.layer_thickness.magnitude(Unit.MICROMETER) if pr.layer_thickness else 50.0)
    classes = filler_classes(formulation, db, state="uncured")
    n0 = resin_refractive_index(formulation, "uncured", db)
    aa = absorption_attenuation_per_um(formulation, lam, db)

    def cd(ec, ftr):
        dp = 1.0 / (aa + scattering_attenuation_per_um(classes, n0, lam, ftr) + 1e-12)
        return dp * math.log(E / ec) if E > ec else 0.0, dp

    v, dp = cd(critical_dose_mj_cm2[0], transport_fraction[0])
    rng = np.random.default_rng(seed)
    s = [cd(rng.uniform(*critical_dose_mj_cm2[1:]), rng.uniform(*transport_fraction[1:]))[0] for _ in range(n_samples)]
    lo, hi = _pct(np.array(s))
    checks = [CheckResult(name="cure_depth_ge_2x_layer", passed=bool(v >= 2.0 * layer), metric=v, tolerance=2.0 * layer,
                          unit=Unit.MICROMETER, details=f"C_d {v:.0f} um vs 2 x {layer:.0f} um layer at {E:.0f} mJ/cm^2, {lam:.0f} nm")]
    return ModelPrediction(
        name=PropertyName.CURE_DEPTH_ESTIMATE,
        value=v, low=lo, high=hi, unit=Unit.MICROMETER,
        interval="5-95 percentile of E_c and transport-fraction sampling",
        model="Jacobs working curve with Beer-Lambert absorption + Griffith-Halloran-type scattering (anomalous-diffraction Q)",
        equation="C_d = D_p ln(E/E_c); 1/D_p = ln10 sum eps_i c_i + sum_i (3/2) phi_i Q_i f_tr / d_i; Q = ADA(x = pi d n0/lambda, m = n_p/n0)",
        assumptions=[
            "D_p,s = (2/3) d/(phi Q) per class is Beer-Lambert with geometric cross sections (the geometric form used by Griffith & Halloran); Q from van de Hulst's anomalous-diffraction approximation reproduces their Delta n^2 dependence at small rho",
            "f_tr = fraction of the extinction that is lost for curing (strongly forward-peaked scattering from micron particles): ASSUMED 0.03-0.15, not fitted; this is the dominant uncertainty",
            "E_c = 5-20 mJ/cm^2 (ASSUMED typical for BAPO/TPO methacrylate resins); measure the working curve (P-CD)",
            "uncured resin index used for the medium; index rises on cure so the mismatch changes during exposure",
            "absorbers/initiators from constituents.json molar absorptivities; oxygen inhibition and multiple scattering beyond f_tr ignored",
        ],
        citations=[CITES["Jacobs1992"], CITES["GH1997"], CITES["S24"], CITES["Wang2025"], CITES["vdHulst1957"]],
        inputs={"dose_mJ_cm2": E, "wavelength_nm": lam, "n_resin_uncured": n0, "D_p_um": dp, "alpha_abs_per_um": aa,
                "layer_um": layer},
        checks=checks,
        conditions_note=f"{lam:.0f} nm, {E:.0f} mJ/cm^2",
    )


# ===========================================================================
# Water sorption / solubility, shrinkage, conversion
# ===========================================================================
def _matrix_rule(formulation: Formulation, key: str, db=None, state="cured") -> tuple[float, float, float, list[str]]:
    db = db or _db()
    vf = matrix_monomer_volume_fractions(formulation, db, state)
    v = lo = hi = 0.0
    keys: set[str] = set()
    for n, f in vf.items():
        p = db.lookup(n).prop(key)
        a, b = p.range()
        v += f * p.magnitude()
        lo += f * a
        hi += f * b
        keys.update(p.citation_keys)
    return v, lo, hi, sorted(keys)


def _interface_factor(formulation: Formulation, db=None) -> tuple[float, float, float, str]:
    """Water uptake enhancement by filler interfaces (ASSUMED): 1.0 for silanised glass/silica;
    zirconia (poorly silanisable) surfaces add k = 0-1 (nominal 0.5) times their share of the
    filler surface area (area ~ phi_i / d_i)."""
    classes = filler_classes(formulation, db, state="cured")
    area = {c.name: c.volume_fraction / c.d50_um for c in classes}
    tot = sum(area.values()) or 1.0
    zr = sum(area[c.name] for c in classes if c.filler_type == "zirconia") / tot
    return 1.0 + 0.5 * zr, 1.0, 1.0 + 1.0 * zr, f"zirconia share of filler surface area = {zr:.2f}"


# Network-quality factors: CQ-cured homopolymer sorption/solubility (Gajewski, Sideridou) versus
# heat-post-cured printed/milled networks.  Calibrated on two anchors (see PROPERTY_MODELS.md):
#   Saremco Crowntec (TDS, ISO 4049): 9 / 0.6 ug/mm^3 at ~19 vol% glass/silica, Bis-EMA-rich matrix
#   VITA Enamic (Mfr, ISO 10477): 5.7 / <=1.2 ug/mm^3 at 75 vol% ceramic, UDMA/TEGDMA polymer
# Rule-of-mixtures predictions with the homopolymer data over-estimate these by ~1.7-2.2x
# (sorption) and ~4-15x (solubility).
SORPTION_NETWORK_FACTOR = (0.55, 0.45, 0.80)  # nominal, low, high
SOLUBILITY_NETWORK_FACTOR = (0.10, 0.05, 0.26)


def _raw_water_sorption(formulation: Formulation, db=None) -> ModelPrediction:
    db = db or _db()
    phi = C.filler_volume_fraction(formulation, db, "cured")
    wm, wl, wh, keys = _matrix_rule(formulation, "water_sorption_homopolymer", db)
    f, fl, fh, note = _interface_factor(formulation, db)
    n0, nl, nh = SORPTION_NETWORK_FACTOR
    v, lo, hi = (1 - phi) * wm * f * n0, (1 - phi) * wl * fl * nl, (1 - phi) * wh * fh * nh
    checks = [CheckResult(name="iso4049_sorption_limit", passed=bool(hi <= 40.0), metric=v, tolerance=40.0, unit=Unit.MICROGRAM_PER_MM3,
                          details="upper end of the range vs ISO 4049/10477 limit 40 ug/mm^3")]
    return ModelPrediction(
        name=PropertyName.WATER_SORPTION_ESTIMATE, value=v, low=lo, high=hi, unit=Unit.MICROGRAM_PER_MM3,
        interval="homopolymer-value range x interface-factor range x network-factor range",
        model="matrix rule of mixtures x (1 - phi) x interface factor x post-cure network factor",
        equation="W_sp = (1 - phi) [sum_i v_i W_sp,i] F_int F_net;  F_int = 1 + k A_ZrO2/A_filler (k 0-1); F_net = 0.45-0.80 (nominal 0.55)",
        assumptions=[
            "fillers absorb no water; homopolymer sorption values (CQ-cured, ISO 4049) mixed by monomer volume fraction",
            "F_net calibrated on two heat-cured anchors (Crowntec TDS 9 ug/mm^3; VITA Enamic 5.7 ug/mm^3): printed + post-cured networks absorb ~0.45-0.6x the homopolymer rule",
            "silanised glass/silica interfaces add no uptake; zirconia surfaces (poor silane bonding, Lung & Matinlinna 2012) raise uptake by k in 0-1 (ASSUMED; motivated by Midas Restore 22-24 and Rodin Sculpture 2.0 29.7 ug/mm^3 at >60-75 wt% filler)",
            f"{note}",
        ],
        citations=[CITES["Gajewski2012"], CITES["S44"], CITES["S43"], CITES["S20"]] + db.cite(keys),
        inputs={"phi": phi, "matrix_sorption_rule": wm, "F_int": f, "F_net": n0}, checks=checks,
        conditions_note="ISO 4049: 7 d water 37 C, 15 x 1 mm discs",
    )


def _raw_water_solubility(formulation: Formulation, db=None) -> ModelPrediction:
    db = db or _db()
    phi = C.filler_volume_fraction(formulation, db, "cured")
    wm, wl, wh, keys = _matrix_rule(formulation, "water_solubility_homopolymer", db)
    n0, nl, nh = SOLUBILITY_NETWORK_FACTOR
    v, lo, hi = (1 - phi) * wm * n0, (1 - phi) * wl * nl, (1 - phi) * wh * nh
    checks = [CheckResult(name="iso4049_solubility_limit", passed=bool(hi <= 7.5), metric=v, tolerance=7.5, unit=Unit.MICROGRAM_PER_MM3)]
    return ModelPrediction(
        name=PropertyName.WATER_SOLUBILITY_ESTIMATE, value=max(v, 0.0), low=max(lo, 0.0), high=max(hi, 0.0), unit=Unit.MICROGRAM_PER_MM3,
        interval="homopolymer-value range x network-factor range",
        model="matrix rule of mixtures x (1 - phi) x post-cure network factor",
        equation="W_sl = (1 - phi) [sum_i v_i W_sl,i] F_net,sl;  F_net,sl = 0.05-0.26 (nominal 0.10)",
        assumptions=[
            "fillers insoluble (Ba-glass ion leaching ignored)",
            "homopolymer solubilities conflict between sources by up to 10x (TEGDMA 2.4 vs 27.5; Bis-EMA 2.1 vs 14.2 ug/mm^3)",
            "F_net,sl calibrated on Crowntec (0.6) and VITA Enamic (<=1.2): elution from a fully post-cured network is far below CQ-cured homopolymers",
        ],
        citations=[CITES["Gajewski2012"], CITES["S20"]] + db.cite(keys),
        inputs={"phi": phi, "matrix_solubility_rule": wm, "F_net_sl": n0}, checks=checks, conditions_note="ISO 4049 7 d water 37 C",
    )


# ---------------------------------------------------------------------------
# Anchor check and proxy calibration against high-filler printed products
# ---------------------------------------------------------------------------
_GLASS_TREAT = "3 wt% gamma-MPS (pre-silanised)"
# PROXY compositions (public compositions are unknown; see LEAD_MATERIAL / design midas_proxy):
#   Midas Restore: >75 wt% glass-ceramic + zirconia, heated cartridge (proxy = design midas_proxy recipe)
#   Rodin Sculpture 2.0: >60 wt% ceramic (proxy: 62 wt% Ba glass, UDMA/TEGDMA)
#   SprintRay Ceramic Crown: >50 wt% Si-based filler (proxy: 52 wt% Ba glass, UDMA/Bis-EMA/TEGDMA)
_PROXY_RECIPES: dict[str, list[tuple[str, float, Optional[str]]]] = {
    "SprintRay Midas Restore (proxy)": [
        ("UDMA", 0.13817, None), ("Bis-EMA(4)", 0.05922, None), ("TEGDMA", 0.03483, None), ("TPO", 0.00117, None),
        ("BHT", 0.00012, None), ("Ba-Al-borosilicate glass 1.5 um (silanised)", 0.40151, _GLASS_TREAT),
        ("Ba-Al-borosilicate glass 0.4 um (silanised)", 0.21620, _GLASS_TREAT),
        ("Zirconia 0.4 um (MDP/silane treated)", 0.14879, "10-MDP + silane (zirconia needs MDP)")],
    "Rodin Sculpture 2.0 (proxy)": [
        ("UDMA", 0.2660, None), ("TEGDMA", 0.1100, None), ("TPO", 0.0038, None), ("BHT", 0.0002, None),
        ("ba glass 1.5", 0.4092, _GLASS_TREAT), ("ba glass 0.4", 0.2108, _GLASS_TREAT)],
    "SprintRay Ceramic Crown (proxy)": [
        ("UDMA", 0.3120, None), ("Bis-EMA", 0.1200, None), ("TEGDMA", 0.0438, None), ("TPO", 0.004, None), ("BHT", 0.0002, None),
        ("ba glass 1.5", 0.3432, _GLASS_TREAT), ("ba glass 0.4", 0.1768, _GLASS_TREAT)],
}
# measured water sorption / solubility (ug/mm^3) of the real products (targets.json / brief)
SORPTION_ANCHORS = {"SprintRay Midas Restore (proxy)": (23.0, 22.0, 24.0), "Rodin Sculpture 2.0 (proxy)": (29.7, 29.7, 29.7),
                    "SprintRay Ceramic Crown (proxy)": (17.4, 17.4, 17.4)}
SOLUBILITY_ANCHORS = {"SprintRay Midas Restore (proxy)": (1.75, 1.0, 2.5)}
PROXY_RATIO_UNCERTAINTY = (0.8, 1.25)  # ASSUMED paired-ratio uncertainty (composition of the proxy unknown)


def proxy_formulation(name: str, db=None) -> Formulation:
    db = db or _db()
    comps = [C.make_component(n, w, db, **({"surface_treatment": t} if t else {})) for n, w, t in _PROXY_RECIPES[name]]
    tot = sum(c.weight_fraction for c in comps)
    comps = [c.model_copy(update={"weight_fraction": c.weight_fraction / tot}) for c in comps]
    from fracture_md.schema.formulation import ProcessingRoute
    return Formulation(name=name, components=comps, processing=ProcessingRoute(
        wavelength_nm=405, layer_thickness=q(50, Unit.MICROMETER), print_temperature=q(303.15, Unit.KELVIN),
        exposure_dose=q(100, Unit.MJ_PER_CM2)))


@lru_cache(maxsize=2)
def _anchor_table(kind: str) -> tuple[tuple[str, float, float, float], ...]:
    fn, anchors = (_raw_water_sorption, SORPTION_ANCHORS) if kind == "sorption" else (_raw_water_solubility, SOLUBILITY_ANCHORS)
    out = []
    for n, (m, _, _) in anchors.items():
        pr = fn(proxy_formulation(n)).value
        out.append((n, pr, m, pr / m))
    return tuple(out)


def water_uptake_anchor_check(kind: str = "sorption") -> CheckResult:
    """Fails when the raw model is off by more than 2x for any high-filler printed product proxy."""
    tab = _anchor_table(kind)
    worst = max(tab, key=lambda r: abs(math.log(r[3])))
    ok = all(0.5 <= r[3] <= 2.0 for r in tab)
    det = "; ".join(f"{n}: model {p:.2f} vs measured {m:.2f} ug/mm^3 (ratio {r:.2f})" for n, p, m, r in tab)
    return CheckResult(name=f"{kind}_anchor_high_filler_products", passed=bool(ok), metric=worst[3], tolerance=2.0,
                       details=f"raw model within 2x of product anchors? {det}. Proxy compositions are approximate.")


def _proxy_calibrated(raw: ModelPrediction, kind: str) -> tuple[float, float, float, float]:
    tab = _anchor_table(kind)
    midas = [r for r in tab if r[0].startswith("SprintRay Midas")][0]
    _, lo_m, hi_m = (SORPTION_ANCHORS if kind == "sorption" else SOLUBILITY_ANCHORS)[midas[0]]
    ratio = raw.value / midas[1]
    return ratio * midas[2], ratio * lo_m * PROXY_RATIO_UNCERTAINTY[0], ratio * hi_m * PROXY_RATIO_UNCERTAINTY[1], ratio


def _with_proxy(raw: ModelPrediction, kind: str) -> ModelPrediction:
    v, lo, hi, ratio = _proxy_calibrated(raw, kind)
    chk = water_uptake_anchor_check(kind)
    raw.inputs.update({"proxy_calibrated_value": v, "proxy_calibrated_low": lo, "proxy_calibrated_high": hi, "ratio_to_midas_proxy": ratio})
    raw.checks.append(chk)
    raw.assumptions.append(f"raw model fails the product-anchor check when off by >2x ({chk.name} passed={chk.passed}); use the "
                           f"proxy-calibrated estimate {v:.3g} [{lo:.3g}-{hi:.3g}] ug/mm^3 (paired ratio {ratio:.2f} to the Midas proxy x Midas 510(k) value)")
    raw.assumptions.append("D3MA homopolymer sorption/solubility are ASSUMED (no measured value)")
    raw.notes = (raw.notes + f" proxy-calibrated: {v:.3g} [{lo:.3g}-{hi:.3g}] ug/mm^3").strip()
    return raw


def predict_water_sorption(formulation: Formulation, db=None) -> ModelPrediction:
    """Raw rule-of-mixtures sorption (value/low/high) with the product-anchor check and a
    proxy-calibrated estimate (``inputs['proxy_calibrated_value'|'_low'|'_high']``)."""
    return _with_proxy(_raw_water_sorption(formulation, db), "sorption")


def predict_water_solubility(formulation: Formulation, db=None) -> ModelPrediction:
    return _with_proxy(_raw_water_solubility(formulation, db), "solubility")


def predict_water_sorption_proxy_calibrated(formulation: Formulation, db=None) -> ModelPrediction:
    """Proxy-calibrated sorption: (model / model[Midas proxy]) x Midas 510(k) 22-24 ug/mm^3."""
    raw = predict_water_sorption(formulation, db)
    v, lo, hi = raw.inputs["proxy_calibrated_value"], raw.inputs["proxy_calibrated_low"], raw.inputs["proxy_calibrated_high"]
    return raw.model_copy(update={"value": v, "low": lo, "high": hi, "model": "paired ratio to Midas-proxy prediction x Midas 510(k) sorption",
                                  "interval": "Midas 22-24 range x paired-ratio uncertainty 0.8-1.25 (ASSUMED)",
                                  "equation": "W_sp = W_model(f) / W_model(Midas proxy) x 22-24 ug/mm^3"})


def predict_degree_of_conversion(formulation: Formulation, post_cure_gain_pp: tuple[float, float, float] = (5.0, 0.0, 10.0), db=None) -> ModelPrediction:
    db = db or _db()
    wm, wl, wh, keys = _matrix_rule(formulation, "degree_of_conversion_homopolymer", db)
    g, gl, gh = post_cure_gain_pp
    v, lo, hi = min(wm + g, 100.0), min(wl + gl, 100.0), min(wh + gh, 100.0)
    return ModelPrediction(
        name=PropertyName.DEGREE_OF_CONVERSION_ESTIMATE, value=v, low=lo, high=hi, unit=Unit.PERCENT,
        interval="homopolymer DC range + post-cure gain range",
        model="volume-weighted homopolymer DC + heated post-cure gain",
        equation="DC = sum_i v_i DC_i + Delta_postcure",
        assumptions=[
            "copolymer DC approximated by the volume average of homopolymer DCs (real blends of stiff + flexible monomers usually exceed the average)",
            "heated (60 C) post-cure adds 0-10 percentage points (ASSUMED; Magnus reached >95 %, VarseoSmile 74.6 %, brief section 3.6)",
            "filler does not reduce conversion (light attenuation is handled by the cure-depth model)",
        ],
        citations=[CITES["Gajewski2012"], CITES["S45"]] + db.cite(keys),
        inputs={"dc_blend": wm},
    )


def predict_shrinkage(formulation: Formulation, db=None) -> ModelPrediction:
    db = db or _db()
    phi = C.filler_volume_fraction(formulation, db, "uncured")
    sm, sl, sh, keys = _matrix_rule(formulation, "shrinkage_volumetric_homopolymer", db, state="uncured")
    # stored as percent -> value in %
    return ModelPrediction(
        name=PropertyName.POLYMERIZATION_SHRINKAGE_ESTIMATE, value=(1 - phi) * sm, low=(1 - phi) * sl, high=(1 - phi) * sh,
        unit=Unit.PERCENT, interval="range of homopolymer shrinkage values",
        model="rule of mixtures on resin volume (fillers do not shrink)",
        equation="S = (1 - phi_paste) sum_i v_i S_i",
        assumptions=["shrinkage linear in resin content (Kleverlaan & Feilzer 2005)", "homopolymer shrinkage at homopolymer DC; blend DC differs"],
        citations=[CITES["Kleverlaan2005"]] + db.cite(keys),
        inputs={"phi_uncured": phi, "matrix_shrinkage_pct": sm},
    )


# ===========================================================================
# Convenience: evaluate everything for one formulation
# ===========================================================================
def evaluate_formulation(
    formulation: Formulation,
    dose_mj_cm2: Optional[float] = None,
    E_composite_gpa: Optional[tuple[float, float, float]] = None,
    db=None,
) -> dict[str, ModelPrediction]:
    """All property-model predictions for a formulation (keys: hardness_empirical,
    hardness_johnson, viscosity, cure_depth, water_sorption, water_solubility,
    degree_of_conversion, shrinkage).  E_composite_gpa (nominal, low, high) from the FFT
    homogenisation replaces the HS lower bound in the Johnson cross-check."""
    db = db or _db()
    phi = C.filler_volume_fraction(formulation, db, "cured")
    classes = filler_classes(formulation, db, "cured")
    fl = [(c.volume_fraction, c.E_gpa or 70.0, c.nu or 0.22) for c in classes]
    out = {
        "hardness_empirical": predict_vickers_empirical(phi),
        "hardness_johnson": predict_vickers_johnson(phi, fillers=fl, E_composite_gpa=E_composite_gpa, n_samples=300),
        "viscosity": predict_viscosity(formulation, db=db),
        "cure_depth": predict_cure_depth(formulation, dose_mj_cm2=dose_mj_cm2, db=db),
        "water_sorption": predict_water_sorption(formulation, db),
        "water_solubility": predict_water_solubility(formulation, db),
        "degree_of_conversion": predict_degree_of_conversion(formulation, db=db),
        "shrinkage": predict_shrinkage(formulation, db),
    }
    return out


def records_for(formulation_candidate: Candidate, predictions: Iterable[ModelPrediction]) -> list[PropertyRecord]:
    return [p.to_record(formulation_candidate) for p in predictions]
