"""Measured/computed property records and the material record.

`PropertyRecord` is the atom of the results database: one quantity, with
units, material context (composition, phase, orientation/interface),
conditions (temperature, rate, defect state), method, potential, uncertainty,
validation status and provenance.

Scale separation is explicit via `PropertyOrigin` and `PropertyScale`: an
atomistic ideal cleavage energy can never be confused with a measured K_Ic or
an R-curve, because they carry different `name`s, origins and scales.

Indentation follows the same rule.  ``*_MD`` indentation names hold only this
harness's MD/quasi-static contact values: they are model-specific (nm depths,
1e8-1e10 1/s, defect-free cells, an idealised tip and an empirical potential),
so their validation status is capped (:data:`STATUS_CEILING`): an experimental
comparison is kept as information but can never make them
``validated_against_experiment``.  Measured instrumented-indentation values
(ISO 14577 H_IT, E_IT) go in the experimental-only ``*_MEASURED`` names, and a
Vickers number obtained from a projected-area hardness is a DERIVED_MODEL
``VICKERS_HARDNESS_ESTIMATE`` built with the ISO 14577 conversion
(``records.vickers_estimate_from_indentation``), never with the unit table.

The validator enforces these rules on every record, including records built by
hand or loaded from JSON: a DERIVED_MODEL record never carries a rung of the
evidence ladder (:data:`DERIVED_MODEL_STATUSES`), so a derived HV of an MD
hardness cannot become ``validated_against_experiment``; and an EXPERIMENTAL
record (the only origin a ``*_MEASURED`` name accepts) may not name an engine
run, a potential or a simulated (rigid_sdf / atomistic) tip.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import Field, model_validator

from fracture_md.schema.common import (
    CheckResult,
    Citation,
    Conditions,
    ContactConditions,
    EngineRunProvenance,
    Provenance,
    StrictModel,
    Uncertainty,
    Validation,
    ValidationStatus,
    status_exceeds,
)
from fracture_md.schema.material import Candidate, Composition, InterfaceSpec, Orientation
from fracture_md.units import Quantity, Unit, compatible, h_it_to_hv


class PropertyName(str, Enum):
    # --- basic bulk
    LATTICE_PARAMETER = "lattice_parameter"
    DENSITY = "density"
    VOLUME_PER_ATOM = "volume_per_atom"
    COHESIVE_ENERGY = "cohesive_energy"
    ENERGY_PER_ATOM = "energy_per_atom"
    RESIDUAL_STRESS = "residual_stress"
    ENERGY_ABOVE_HULL = "energy_above_hull"
    BORN_STABILITY = "born_stability"  # min eigenvalue of C (GPa); >0 means stable
    # Cauchy stress of a periodic cell held at a stated (possibly strained) shape: voigt 6, GPa, positive =
    # tensile.  RESIDUAL_STRESS is the stress left after a relaxation; this is a single point at a given cell
    # (e.g. one DFT stress-strain sample, engines.dft_qe.single_point).
    CELL_STRESS = "cell_stress"
    # --- equation of state (E(V) fit; for a cubic crystal B_EOS = (C11 + 2 C12)/3 at the same volume)
    BULK_MODULUS_EOS = "bulk_modulus_eos"  # B0 of a Birch-Murnaghan (or stated) E(V) fit, GPa
    BULK_MODULUS_PRESSURE_DERIVATIVE = "bulk_modulus_pressure_derivative"  # B0' = dB/dP at P = 0, dimensionless
    # static (0 K) pressure at which a second phase becomes stable: enthalpy crossing H_B(P) = H_A(P), GPa
    # (e.g. zincblende -> rocksalt SiC, engines.dft_reference.enthalpy_crossing)
    PHASE_TRANSITION_PRESSURE = "phase_transition_pressure"
    # --- elasticity
    ELASTIC_STIFFNESS_VOIGT = "elastic_stiffness_voigt"  # 6x6, GPa
    ELASTIC_COMPLIANCE_VOIGT = "elastic_compliance_voigt"  # 6x6, 1/GPa (engineering shear strains)
    ELASTIC_STIFFNESS_TENSOR = "elastic_stiffness_tensor"  # 3x3x3x3, GPa
    CLAMPED_ION_STIFFNESS_VOIGT = "clamped_ion_stiffness_voigt"
    INTERNAL_STRAIN_TENSOR = "internal_strain_tensor"
    BULK_MODULUS_VOIGT = "bulk_modulus_voigt"
    BULK_MODULUS_REUSS = "bulk_modulus_reuss"
    BULK_MODULUS_HILL = "bulk_modulus_hill"
    SHEAR_MODULUS_VOIGT = "shear_modulus_voigt"
    SHEAR_MODULUS_REUSS = "shear_modulus_reuss"
    SHEAR_MODULUS_HILL = "shear_modulus_hill"
    YOUNGS_MODULUS_HILL = "youngs_modulus_hill"
    POISSON_RATIO_HILL = "poisson_ratio_hill"
    YOUNGS_MODULUS_DIRECTIONAL = "youngs_modulus_directional"
    SHEAR_MODULUS_DIRECTIONAL = "shear_modulus_directional"
    POISSON_RATIO_DIRECTIONAL = "poisson_ratio_directional"
    ZENER_ANISOTROPY = "zener_anisotropy"
    UNIVERSAL_ANISOTROPY_INDEX = "universal_anisotropy_index"
    PUGH_RATIO = "pugh_ratio"
    # --- strength
    IDEAL_TENSILE_STRENGTH = "ideal_tensile_strength"
    IDEAL_SHEAR_STRENGTH = "ideal_shear_strength"
    CRITICAL_STRAIN = "critical_strain"
    FLAW_LIMITED_STRENGTH = "flaw_limited_strength"  # future: GB/pore/processing flaws
    # --- surfaces, interfaces, cleavage
    SURFACE_ENERGY = "surface_energy"
    SURFACE_ENERGY_AVERAGE_PAIR = "surface_energy_average_pair"  # (γ_top+γ_bottom)/2 for polar cuts
    INTERFACE_ENERGY = "interface_energy"
    WORK_OF_SEPARATION = "work_of_separation"
    PEAK_NORMAL_TRACTION = "peak_normal_traction"
    OPENING_AT_PEAK_TRACTION = "opening_at_peak_traction"
    PEAK_TANGENTIAL_TRACTION = "peak_tangential_traction"
    UNSTABLE_STACKING_FAULT_ENERGY = "unstable_stacking_fault_energy"
    STACKING_FAULT_ENERGY = "stacking_fault_energy"
    TRACTION_SEPARATION_NORMAL = "traction_separation_normal"
    TRACTION_SEPARATION_TANGENTIAL = "traction_separation_tangential"
    COHESIVE_ENERGY_SURFACE_MIXED_MODE = "cohesive_energy_surface_mixed_mode"
    UBER_LENGTH_SCALE = "uber_length_scale"
    # --- fracture energies: atomistic (ideal) vs macroscopic
    G_IC_ATOMISTIC = "G_Ic_atomistic"  # ideal brittle, = work of separation (Griffith)
    G_IIC_ATOMISTIC = "G_IIc_atomistic"
    K_IC_GRIFFITH_ESTIMATE = "K_Ic_griffith_estimate"  # derived; NOT a measured toughness
    K_IC_MEASURED = "K_Ic_measured"  # experimental only
    R_CURVE_MEASURED = "R_curve_measured"
    FATIGUE_LIFE_MEASURED = "fatigue_life_measured"
    STRENGTH_DISTRIBUTION_MEASURED = "strength_distribution_measured"  # e.g. Weibull
    # --- secondary screening
    THERMAL_EXPANSION = "thermal_expansion"
    HARDNESS_ESTIMATE = "hardness_estimate"
    HARDNESS_MEASURED = "hardness_measured"
    MELTING_OR_DECOMPOSITION_TEMPERATURE = "melting_or_decomposition_temperature"
    OXIDATION_RESISTANCE = "oxidation_resistance"
    # --- composite / dental restorative properties (estimates vs measurements kept apart)
    FILLER_VOLUME_FRACTION = "filler_volume_fraction"
    FILLER_WEIGHT_FRACTION = "filler_weight_fraction"
    EFFECTIVE_YOUNGS_MODULUS = "effective_youngs_modulus"  # homogenised composite
    EFFECTIVE_POISSON_RATIO = "effective_poisson_ratio"
    EFFECTIVE_STIFFNESS_VOIGT = "effective_stiffness_voigt"
    VICKERS_HARDNESS_ESTIMATE = "vickers_hardness_estimate"  # model; unit HV or GPa
    VICKERS_HARDNESS_MEASURED = "vickers_hardness_measured"  # ISO 6507-1 / ISO 14577 test
    FLEXURAL_STRENGTH_ESTIMATE = "flexural_strength_estimate"
    FLEXURAL_STRENGTH_MEASURED = "flexural_strength_measured"  # ISO 4049 / ISO 10477 3-point
    BIAXIAL_FLEXURAL_STRENGTH_MEASURED = "biaxial_flexural_strength_measured"  # ISO 6872
    FLEXURAL_MODULUS_ESTIMATE = "flexural_modulus_estimate"
    FLEXURAL_MODULUS_MEASURED = "flexural_modulus_measured"
    FRACTURE_TOUGHNESS_ESTIMATE = "fracture_toughness_estimate"  # composite K_Ic from a stated model
    FRACTURE_TOUGHNESS_MEASURED = "fracture_toughness_measured"  # SEVNB/SENB/CNB test
    FRACTURE_ENERGY_SIMULATED = "fracture_energy_simulated"  # phase-field / CZM simulation output
    WATER_SORPTION_ESTIMATE = "water_sorption_estimate"
    WATER_SORPTION_MEASURED = "water_sorption_measured"  # ISO 4049
    WATER_SOLUBILITY_ESTIMATE = "water_solubility_estimate"
    WATER_SOLUBILITY_MEASURED = "water_solubility_measured"
    VISCOSITY_ESTIMATE = "viscosity_estimate"
    VISCOSITY_MEASURED = "viscosity_measured"
    CURE_DEPTH_ESTIMATE = "cure_depth_estimate"
    CURE_DEPTH_MEASURED = "cure_depth_measured"
    POLYMERIZATION_SHRINKAGE_ESTIMATE = "polymerization_shrinkage_estimate"
    DEGREE_OF_CONVERSION_ESTIMATE = "degree_of_conversion_estimate"
    BOND_STRENGTH_ESTIMATE = "bond_strength_estimate"  # per interface of the bonded assembly
    BOND_STRENGTH_MEASURED = "bond_strength_measured"  # uTBS ISO/TS 11405 or SBS ISO 29022
    # --- indentation.  *_MD: this harness's MD / quasi-static contact values (model-specific, status-capped);
    #     *_MEASURED: instrumented indentation experiments (ISO 14577-1) only.
    INDENTATION_HARDNESS_MD_ESTIMATED_CONTACT = "indentation_hardness_md_estimated_contact"  # P / A_E3 (estimated contact, B1), GPa
    INDENTATION_HARDNESS_MD_OLIVER_PHARR = "indentation_hardness_md_oliver_pharr"  # P_max / A(h_c), unloading fit
    INDENTATION_CONTACT_PRESSURE_MD = "indentation_contact_pressure_md"  # P / A at h when the plasticity gates fail
    INDENTATION_REDUCED_MODULUS_MD = "indentation_reduced_modulus_md"  # M of the elastic loading fit (Sneddon for the
    # tip's own shape, Hertz for a sphere; rigid tip, MD-cell box factor included), GPa
    INDENTATION_MAX_LOAD_MD = "indentation_max_load_md"  # P_max, nN
    POP_IN_LOAD_MD = "pop_in_load_md"  # load at the first plasticity event, nN
    POP_IN_DEPTH_MD = "pop_in_depth_md"  # depth at the first plasticity event, angstrom
    INDENTATION_HARDNESS_MEASURED = "indentation_hardness_measured"  # H_IT (ISO 14577-1, projected area)
    INDENTATION_MODULUS_MEASURED = "indentation_modulus_measured"  # E_IT (ISO 14577-1)


# Names that may only hold experimental measurements.
MEASURED_ONLY_NAMES: frozenset[PropertyName] = frozenset({
    PropertyName.K_IC_MEASURED,
    PropertyName.R_CURVE_MEASURED,
    PropertyName.FATIGUE_LIFE_MEASURED,
    PropertyName.STRENGTH_DISTRIBUTION_MEASURED,
    PropertyName.HARDNESS_MEASURED,
    PropertyName.VICKERS_HARDNESS_MEASURED,
    PropertyName.FLEXURAL_STRENGTH_MEASURED,
    PropertyName.BIAXIAL_FLEXURAL_STRENGTH_MEASURED,
    PropertyName.FLEXURAL_MODULUS_MEASURED,
    PropertyName.FRACTURE_TOUGHNESS_MEASURED,
    PropertyName.WATER_SORPTION_MEASURED,
    PropertyName.WATER_SOLUBILITY_MEASURED,
    PropertyName.VISCOSITY_MEASURED,
    PropertyName.CURE_DEPTH_MEASURED,
    PropertyName.BOND_STRENGTH_MEASURED,
    PropertyName.INDENTATION_HARDNESS_MEASURED,
    PropertyName.INDENTATION_MODULUS_MEASURED,
})
# Names that may only hold values derived from other records with a stated model (correlations such as
# Chen-Niu, conversions such as ISO 14577 HV = 0.0945 H_IT, Griffith).  An atomistic "hardness" is an
# *_MD indentation name, never HARDNESS_ESTIMATE.
DERIVED_ONLY_NAMES: frozenset[PropertyName] = frozenset({
    PropertyName.K_IC_GRIFFITH_ESTIMATE,
    PropertyName.HARDNESS_ESTIMATE,
    PropertyName.VICKERS_HARDNESS_ESTIMATE,
})
# MD / quasi-static indentation values of this harness (ATOMISTIC_COMPUTED only, contact conditions and the
# simulation engine required, validation status capped by STATUS_CEILING).
MD_INDENTATION_NAMES: frozenset[PropertyName] = frozenset({
    PropertyName.INDENTATION_HARDNESS_MD_ESTIMATED_CONTACT,
    PropertyName.INDENTATION_HARDNESS_MD_OLIVER_PHARR,
    PropertyName.INDENTATION_CONTACT_PRESSURE_MD,
    PropertyName.INDENTATION_REDUCED_MODULUS_MD,
    PropertyName.INDENTATION_MAX_LOAD_MD,
    PropertyName.POP_IN_LOAD_MD,
    PropertyName.POP_IN_DEPTH_MD,
})
# Hardnesses defined on the PROJECTED contact area: never stored in HV (the unit table converts the sloped
# Vickers area and would overstate HV by 7.9 %); a Vickers number derived from them must use ISO 14577.
PROJECTED_AREA_HARDNESS_NAMES: frozenset[PropertyName] = frozenset({
    PropertyName.INDENTATION_HARDNESS_MD_ESTIMATED_CONTACT,
    PropertyName.INDENTATION_HARDNESS_MD_OLIVER_PHARR,
    PropertyName.INDENTATION_CONTACT_PRESSURE_MD,
    PropertyName.INDENTATION_HARDNESS_MEASURED,
})
# Highest validation status a record of the name may carry.  MD indentation values are model-specific: an
# experimental or DFT comparison is attached as evidence but never raises them above this rung.  (Changing
# the policy, e.g. to NUMERICALLY_VERIFIED, is a one-line edit here.)
MD_INDENTATION_STATUS_CEILING = ValidationStatus.LITERATURE_MODEL_AGREEMENT
STATUS_CEILING: dict[PropertyName, ValidationStatus] = {n: MD_INDENTATION_STATUS_CEILING for n in MD_INDENTATION_NAMES}
# Statuses a DERIVED_MODEL record may carry (records.decide_status gives MODEL_ESTIMATE unless OUT_OF_DOMAIN or
# FAILED applies; a runner may leave it UNVALIDATED).  A derived value never claims a rung of the evidence ladder,
# so the ISO 14577 HV (or any other conversion) of a status-capped MD hardness cannot carry the status its input
# is denied: agreement of a model estimate with experiment is kept as evidence only.
DERIVED_MODEL_STATUSES: frozenset[ValidationStatus] = frozenset({
    ValidationStatus.UNVALIDATED,
    ValidationStatus.MODEL_ESTIMATE,
    ValidationStatus.OUT_OF_DOMAIN,
    ValidationStatus.FAILED,
})


class PropertyOrigin(str, Enum):
    ATOMISTIC_COMPUTED = "atomistic_computed"  # this harness, a potential or DFT
    DERIVED_MODEL = "derived_model"  # computed from other records via a stated model
    LITERATURE_COMPUTED = "literature_computed"
    EXPERIMENTAL = "experimental"


class PropertyScale(str, Enum):
    ATOMISTIC_IDEAL = "atomistic_ideal"  # defect-free, 0 K or MD cell
    ATOMISTIC_DEFECTED = "atomistic_defected"
    MESOSCALE = "mesoscale"
    MACROSCOPIC = "macroscopic"  # measured specimens / components


class MethodSpec(StrictModel):
    name: str  # e.g. "autodiff relaxed-ion elastic tensor"
    protocol_id: Optional[str] = None
    engine: Optional[str] = None
    description: str = ""
    assumptions: list[str] = Field(default_factory=list)


class Derivation(StrictModel):
    """For DERIVED_MODEL values: the model, its assumptions and the input records."""

    model: str  # e.g. "Griffith: K_Ic = sqrt(2 γ E')"
    equation: str = ""
    assumptions: list[str] = Field(default_factory=list)
    input_record_ids: list[str] = Field(default_factory=list)


class PropertyRecord(StrictModel):
    id: str
    name: PropertyName
    value: Quantity
    # Optional components for tensor/array values ("voigt 6x6", "3x3x3x3", "delta_A", ...)
    shape_description: str = ""
    frame: Literal["crystal", "lab", "not_applicable"] = "not_applicable"
    composition: Composition
    phase: str
    orientation: Optional[Orientation] = None
    interface: Optional[InterfaceSpec] = None
    conditions: Conditions = Field(default_factory=Conditions)
    origin: PropertyOrigin
    scale: PropertyScale
    method: MethodSpec
    potential_id: Optional[str] = None
    derivation: Optional[Derivation] = None
    uncertainty: list[Uncertainty] = Field(default_factory=list)
    validation: Validation = Field(default_factory=Validation)
    provenance: Provenance = Field(default_factory=Provenance)
    citations: list[Citation] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _scale_rules(self) -> "PropertyRecord":
        if self.name in MEASURED_ONLY_NAMES and self.origin != PropertyOrigin.EXPERIMENTAL:
            raise ValueError(f"{self.name.value} may only hold experimental measurements")
        if self.origin == PropertyOrigin.DERIVED_MODEL and self.derivation is None:
            raise ValueError("derived values must state their model in `derivation`")
        if self.name in DERIVED_ONLY_NAMES and self.origin != PropertyOrigin.DERIVED_MODEL:
            hint = (" (an MD/atomistic indentation hardness is an indentation_hardness_md_* record)"
                    if self.name in (PropertyName.HARDNESS_ESTIMATE, PropertyName.VICKERS_HARDNESS_ESTIMATE) else "")
            raise ValueError(f"{self.name.value} must be a derived model estimate{hint}")
        if self.origin == PropertyOrigin.ATOMISTIC_COMPUTED and self.potential_id is None:
            raise ValueError("atomistic values must name the potential (or DFT functional) used")
        self._unit_rules()
        if self.origin == PropertyOrigin.EXPERIMENTAL:
            self._experimental_rules()
        if self.name in MD_INDENTATION_NAMES:
            self._md_indentation_rules()
        ceiling = STATUS_CEILING.get(self.name)
        if ceiling is not None and status_exceeds(self.validation.status, ceiling):
            raise ValueError(f"{self.name.value} is a model-specific MD value: its validation status is capped at "
                             f"{ceiling.value} (got {self.validation.status.value}); experimental/DFT comparisons "
                             "are kept as evidence only")
        if self.origin == PropertyOrigin.DERIVED_MODEL and self.validation.status not in DERIVED_MODEL_STATUSES:
            raise ValueError(f"{self.name.value} is a derived model estimate: its validation status must be one of "
                             f"{sorted(s.value for s in DERIVED_MODEL_STATUSES)} (got {self.validation.status.value}); "
                             "experimental/DFT comparisons are kept as evidence only and never validate a model "
                             f"estimate (inputs {self.derivation.input_record_ids})")
        if self.name == PropertyName.VICKERS_HARDNESS_ESTIMATE and self.derivation is not None:
            area_inputs = [i for i in self.derivation.input_record_ids
                           if any(f":{n.value}:" in i for n in PROJECTED_AREA_HARDNESS_NAMES)]
            if area_inputs and "14577" not in self.derivation.model + " " + self.derivation.equation:
                raise ValueError("a Vickers number derived from a projected-area hardness must use the ISO 14577 "
                                 "conversion HV = 0.0945 H_IT[MPa] (records.vickers_estimate_from_indentation), "
                                 f"not the HV unit table; inputs {area_inputs}")
        return self

    def _unit_rules(self) -> None:
        canon = canonical_unit(self.name)
        if canon is not None and not compatible(self.value.unit, canon):
            raise ValueError(f"{self.name.value} must have the dimension of {canon.value}, got {self.value.unit.value!r}")
        if self.name in PROJECTED_AREA_HARDNESS_NAMES and self.value.unit == Unit.HV:
            raise ValueError(f"{self.name.value} is a projected-area hardness and cannot be stored in HV (the HV unit "
                             "is kgf/mm^2 on the sloped Vickers area); store GPa and derive a "
                             "vickers_hardness_estimate with the ISO 14577 conversion")

    def _experimental_rules(self) -> None:
        # `origin` is self-declared; these fields are not: a record that names a simulation engine run, a
        # potential or a simulated tip is not a measurement (e.g. an MD hardness stored as *_MEASURED).
        sim = []
        if self.provenance.engine_run is not None:
            sim.append(f"provenance.engine_run (engine {self.provenance.engine_run.engine})")
        if self.potential_id is not None:
            sim.append(f"potential_id {self.potential_id!r}")
        contact = self.conditions.contact
        if contact is not None and contact.tip_model != "physical":
            sim.append(f"conditions.contact.tip_model {contact.tip_model!r}")
        if sim:
            raise ValueError(f"{self.name.value} has origin experimental but carries simulation provenance "
                             f"({'; '.join(sim)}): a measured value names no engine run or potential and a physical "
                             "tip; this harness's indentation values are indentation_*_md records (status-capped), "
                             "published simulations are literature_computed")

    def _md_indentation_rules(self) -> None:
        n = self.name.value
        if self.origin != PropertyOrigin.ATOMISTIC_COMPUTED:
            raise ValueError(f"{n} holds only this harness's MD/quasi-static indentation values (origin "
                             "atomistic_computed); literature MD values belong in validation evidence and measured "
                             "values in indentation_*_measured")
        if self.scale not in (PropertyScale.ATOMISTIC_IDEAL, PropertyScale.ATOMISTIC_DEFECTED):
            raise ValueError(f"{n} must have an atomistic scale, got {self.scale.value}")
        c = self.conditions
        if c.contact is None:
            raise ValueError(f"{n} requires conditions.contact (indenter shape, tip model, radius, wall stiffness, "
                             "azimuth, speed, depth)")
        if c.contact.tip_model == "physical":
            raise ValueError(f"{n} is a simulated value: conditions.contact.tip_model must be rigid_sdf or atomistic")
        if c.loading not in ("quasistatic", "dynamic"):
            raise ValueError(f"{n} requires conditions.loading 'quasistatic' (0 K steps) or 'dynamic' (MD), "
                             f"got {c.loading!r}")
        if c.loading == "dynamic" and (c.strain_rate is None or c.contact.indentation_speed is None):
            raise ValueError(f"{n} from MD requires conditions.strain_rate (v / h_max) and "
                             "conditions.contact.indentation_speed")
        if self.provenance.engine_run is None:
            raise ValueError(f"{n} requires provenance.engine_run (jax or lammps, version, deck/potential hashes)")


# --------------------------------------------------------------------------
# Structured result payloads (arrays kept as Quantities with explicit units)
# --------------------------------------------------------------------------
class SampledCurve(StrictModel):
    """A sampled 1-D curve y(x) with explicit units (e.g. T_n(δ_n))."""

    x_name: str
    x: Quantity
    y_name: str
    y: Quantity
    extra: dict[str, Quantity] = Field(default_factory=dict)  # e.g. energy per area on the same grid

    @model_validator(mode="after")
    def _same_length(self) -> "SampledCurve":
        n = len(self.x.value) if isinstance(self.x.value, list) else 1
        m = len(self.y.value) if isinstance(self.y.value, list) else 1
        if n != m:
            raise ValueError(f"x and y must have the same length ({n} != {m})")
        for k, v in self.extra.items():
            if (len(v.value) if isinstance(v.value, list) else 1) != n:
                raise ValueError(f"extra[{k}] length mismatch")
        return self


class CohesiveLawFit(StrictModel):
    model: Literal["UBER", "exponential_softening", "linear_softening", "tabulated"]
    parameters: dict[str, Quantity]
    rms_residual: Optional[Quantity] = None
    domain: str = ""  # which part of the curve was fitted


class TractionSeparationLaw(StrictModel):
    """Atomistic traction–separation data for one plane/termination and mode."""

    mode: Literal["normal", "tangential", "mixed"]
    interface: InterfaceSpec
    protocol_mode: Literal["rigid", "relaxed", "relaxed_elastic_corrected"]
    curve: SampledCurve  # x = opening (angstrom), y = traction (GPa)
    energy_per_area: Quantity  # J/m^2, same grid as curve.x
    peak_traction: Quantity  # GPa
    opening_at_peak: Quantity  # angstrom
    work_of_separation: Quantity  # J/m^2  (E(δ_max) - E(0)) / A
    integrated_work: Quantity  # J/m^2  ∫ T dδ over the sampled range
    fits: list[CohesiveLawFit] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)


# Per-sample channels of an IndentationCurve (SampledCurve.extra) and their canonical units.  Unknown channel
# names are allowed; known ones are dimension-checked.
INDENTATION_CHANNEL_UNITS: dict[str, Unit] = {
    "time": Unit.PICOSECOND,
    "nominal_depth": Unit.ANGSTROM,  # prescribed tip displacement (x is the reported depth)
    "contact_depth": Unit.ANGSTROM,  # h_c = h - pen_fw (penetration-corrected)
    "penetration": Unit.ANGSTROM,  # force-weighted wall penetration pen_fw
    "load_raw": Unit.NANONEWTON,  # instantaneous (not block-averaged) load
    "lateral_load_x": Unit.NANONEWTON,
    "lateral_load_y": Unit.NANONEWTON,
    "fixed_layer_reaction": Unit.NANONEWTON,  # Newton-3 check of the load
    "contact_area": Unit.ANGSTROM2,
    "n_contact": Unit.DIMENSIONLESS,
    "mean_pressure": Unit.GPA,
    "tip_work": Unit.EV,
    "potential_energy": Unit.EV,
    "kinetic_energy": Unit.EV,
    "thermostat_heat": Unit.EV,  # cumulative heat ADDED to the sample by the bath since loading start (< 0 when the
    # bath cools); energy ledger E - E(0) - W - Q.  Harness: sum of the bath's KE change at each thermostat step
    # (CSVR: (alpha^2 - 1) KE_bath); LAMMPS import:
    # -(ecouple - ecouple(step 0)) (io.lammps_indent.LammpsIndentRun.bath_heat_eV)
    "temperature_core": Unit.KELVIN,
}


class IndentationSegment(StrictModel):
    """A contiguous stage of an indentation curve, as sample indices [start, stop)."""

    kind: Literal["approach", "load", "hold", "unload", "relax"]
    start: int
    stop: int


class IndentationCurve(StrictModel):
    """Load-depth record of one indentation run (MD or quasi-static), arrays with explicit units.

    ``curve.x`` is the depth (length), ``curve.y`` the indenter load (force, positive in compression);
    further per-sample channels go in ``curve.extra`` (:data:`INDENTATION_CHANNEL_UNITS`).  The curve is raw
    data: hardness values extracted from it are separate ``*_MD`` PropertyRecords.
    """

    curve_id: str
    mode: Literal["quasistatic", "md"]
    contact: ContactConditions
    temperature: Quantity  # K (thermostat target; 0 for quasi-static)
    curve: SampledCurve
    segments: list[IndentationSegment] = Field(default_factory=list)
    orientation: Optional[Orientation] = None  # surface plane and lab axes
    potential_id: str
    engine_run: EngineRunProvenance
    seed: Optional[int] = None
    depth_definition: str = "tip displacement below the mean height of the relaxed top surface layer"
    load_definition: str = "z component of the total force of the indenter on the sample"
    checks: list[CheckResult] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> "IndentationCurve":
        cv = self.curve
        if not compatible(cv.x.unit, Unit.ANGSTROM):
            raise ValueError(f"indentation curve x must be a depth (length), got {cv.x.unit.value!r}")
        if not compatible(cv.y.unit, Unit.NANONEWTON):
            raise ValueError(f"indentation curve y must be a load (force), got {cv.y.unit.value!r}")
        if not compatible(self.temperature.unit, Unit.KELVIN):
            raise ValueError(f"temperature must be in K, got {self.temperature.unit.value!r}")
        n = len(cv.x.value) if isinstance(cv.x.value, list) else 1
        if n < 2:
            raise ValueError("an indentation curve needs at least 2 samples")
        for k, v in cv.extra.items():
            like = INDENTATION_CHANNEL_UNITS.get(k)
            if like is not None and not compatible(v.unit, like):
                raise ValueError(f"channel {k!r} must have the dimension of {like.value}, got {v.unit.value!r}")
        if self.mode == "md":
            if "time" not in cv.extra:
                raise ValueError("an MD indentation curve needs a 'time' channel in curve.extra")
            if self.contact.indentation_speed is None:
                raise ValueError("an MD indentation curve needs contact.indentation_speed")
        prev = 0
        for sg in sorted(self.segments, key=lambda g: g.start):
            if not (0 <= sg.start < sg.stop <= n) or sg.start < prev:
                raise ValueError(f"segments must be non-overlapping index ranges within [0, {n}), got "
                                 f"{[(g.kind, g.start, g.stop) for g in self.segments]}")
            prev = sg.stop
        if self.segments and not any(g.kind == "load" for g in self.segments):
            raise ValueError("an indentation curve needs a 'load' segment")
        return self


class MaterialRecord(StrictModel):
    """Everything known about one candidate: the unit handed to PF-CZM and agents."""

    record_version: str = "0.1.0"
    candidate: Candidate
    properties: list[PropertyRecord] = Field(default_factory=list)
    traction_separation: list[TractionSeparationLaw] = Field(default_factory=list)
    indentation_curves: list[IndentationCurve] = Field(default_factory=list)
    potential_ids: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)

    @model_validator(mode="after")
    def _derived_vickers(self) -> "MaterialRecord":
        problems = check_vickers_conversions(self.properties)
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def get(self, name: PropertyName) -> list[PropertyRecord]:
        return [p for p in self.properties if p.name == name]


def check_vickers_conversions(records: list[PropertyRecord], rel_tol: float = 1.0e-9) -> list[str]:
    """Problems with Vickers estimates derived from projected-area hardness records of the same set.

    A ``vickers_hardness_estimate`` whose derivation names a projected-area hardness record present in
    ``records`` must equal the ISO 14577 conversion of that record (HV = 0.0945 H_IT[MPa]); a value
    converted with the HV unit table (7.9 % higher) is reported.
    """
    by_id = {r.id: r for r in records}
    problems = []
    for r in records:
        if r.name != PropertyName.VICKERS_HARDNESS_ESTIMATE or r.derivation is None:
            continue
        inputs = [by_id[i] for i in r.derivation.input_record_ids
                  if i in by_id and by_id[i].name in PROJECTED_AREA_HARDNESS_NAMES]
        if not inputs:
            continue
        if len(inputs) > 1:
            problems.append(f"{r.id}: derived from several projected-area hardness records {[i.id for i in inputs]}")
            continue
        src = inputs[0]
        if isinstance(src.value.value, list) or isinstance(r.value.value, list):
            continue
        want = h_it_to_hv(src.value).magnitude(Unit.HV)
        got = r.value.magnitude(Unit.HV)
        if abs(got - want) > rel_tol * max(abs(want), 1e-300):
            problems.append(f"{r.id}: {got:.6g} HV is not the ISO 14577 conversion of {src.id} "
                            f"({src.value.magnitude(Unit.GPA):.6g} GPa -> {want:.6g} HV)")
    return problems


# Canonical units of array-valued (or otherwise non-scalar-table) properties; used for the record-level
# dimension check and the pre-run check of indexed criteria such as "elastic_stiffness_voigt[0,0]".
ARRAY_PROPERTY_UNITS: dict[str, Unit] = {
    **{n: Unit.GPA for n in ("elastic_stiffness_voigt", "elastic_stiffness_tensor", "clamped_ion_stiffness_voigt",
                             "effective_stiffness_voigt", "born_stability", "bulk_modulus_voigt",
                             "bulk_modulus_reuss", "shear_modulus_voigt", "shear_modulus_reuss",
                             "youngs_modulus_directional", "shear_modulus_directional", "residual_stress",
                             "cell_stress",
                             "peak_tangential_traction", "flaw_limited_strength")},
    "elastic_compliance_voigt": Unit.PER_GPA,
    **{n: Unit.DIMENSIONLESS for n in ("poisson_ratio_directional", "zener_anisotropy", "universal_anisotropy_index",
                                       "pugh_ratio", "critical_strain", "effective_poisson_ratio")},
    **{n: Unit.J_PER_M2 for n in ("surface_energy_average_pair", "interface_energy", "unstable_stacking_fault_energy",
                                  "stacking_fault_energy", "G_IIc_atomistic", "cohesive_energy_surface_mixed_mode",
                                  "fracture_energy_simulated")},
    **{n: Unit.ANGSTROM for n in ("opening_at_peak_traction", "uber_length_scale")},
}


def canonical_unit(name: PropertyName) -> Optional[Unit]:
    """Canonical unit of a property (scalar table, else array table); None = no dimension rule."""
    return scalar_unit(name) or ARRAY_PROPERTY_UNITS.get(PropertyName(name).value)


def scalar_unit(name: PropertyName) -> Optional[Unit]:
    """Canonical unit for scalar properties (used by exporters and checks)."""
    table = {
        PropertyName.LATTICE_PARAMETER: Unit.ANGSTROM,
        PropertyName.DENSITY: Unit.G_PER_CM3,
        PropertyName.VOLUME_PER_ATOM: Unit.ANGSTROM3_PER_ATOM,
        PropertyName.COHESIVE_ENERGY: Unit.EV_PER_ATOM,
        PropertyName.ENERGY_PER_ATOM: Unit.EV_PER_ATOM,
        PropertyName.BULK_MODULUS_EOS: Unit.GPA,
        PropertyName.BULK_MODULUS_PRESSURE_DERIVATIVE: Unit.DIMENSIONLESS,
        PropertyName.PHASE_TRANSITION_PRESSURE: Unit.GPA,
        PropertyName.BULK_MODULUS_HILL: Unit.GPA,
        PropertyName.SHEAR_MODULUS_HILL: Unit.GPA,
        PropertyName.YOUNGS_MODULUS_HILL: Unit.GPA,
        PropertyName.POISSON_RATIO_HILL: Unit.DIMENSIONLESS,
        PropertyName.SURFACE_ENERGY: Unit.J_PER_M2,
        PropertyName.WORK_OF_SEPARATION: Unit.J_PER_M2,
        PropertyName.PEAK_NORMAL_TRACTION: Unit.GPA,
        PropertyName.IDEAL_TENSILE_STRENGTH: Unit.GPA,
        PropertyName.IDEAL_SHEAR_STRENGTH: Unit.GPA,
        PropertyName.G_IC_ATOMISTIC: Unit.J_PER_M2,
        PropertyName.K_IC_GRIFFITH_ESTIMATE: Unit.MPA_SQRT_M,
        PropertyName.K_IC_MEASURED: Unit.MPA_SQRT_M,
        PropertyName.VICKERS_HARDNESS_ESTIMATE: Unit.HV,
        PropertyName.VICKERS_HARDNESS_MEASURED: Unit.HV,
        PropertyName.FLEXURAL_STRENGTH_ESTIMATE: Unit.MPA,
        PropertyName.FLEXURAL_STRENGTH_MEASURED: Unit.MPA,
        PropertyName.FLEXURAL_MODULUS_ESTIMATE: Unit.GPA,
        PropertyName.EFFECTIVE_YOUNGS_MODULUS: Unit.GPA,
        PropertyName.FRACTURE_TOUGHNESS_ESTIMATE: Unit.MPA_SQRT_M,
        PropertyName.WATER_SORPTION_ESTIMATE: Unit.MICROGRAM_PER_MM3,
        PropertyName.WATER_SOLUBILITY_ESTIMATE: Unit.MICROGRAM_PER_MM3,
        PropertyName.VISCOSITY_ESTIMATE: Unit.PA_S,
        PropertyName.CURE_DEPTH_ESTIMATE: Unit.MICROMETER,
        PropertyName.BOND_STRENGTH_ESTIMATE: Unit.MPA,
        PropertyName.BOND_STRENGTH_MEASURED: Unit.MPA,
        PropertyName.HARDNESS_ESTIMATE: Unit.GPA,
        PropertyName.HARDNESS_MEASURED: Unit.GPA,
        PropertyName.INDENTATION_HARDNESS_MD_ESTIMATED_CONTACT: Unit.GPA,
        PropertyName.INDENTATION_HARDNESS_MD_OLIVER_PHARR: Unit.GPA,
        PropertyName.INDENTATION_CONTACT_PRESSURE_MD: Unit.GPA,
        PropertyName.INDENTATION_REDUCED_MODULUS_MD: Unit.GPA,
        PropertyName.INDENTATION_MAX_LOAD_MD: Unit.NANONEWTON,
        PropertyName.POP_IN_LOAD_MD: Unit.NANONEWTON,
        PropertyName.POP_IN_DEPTH_MD: Unit.ANGSTROM,
        PropertyName.INDENTATION_HARDNESS_MEASURED: Unit.GPA,
        PropertyName.INDENTATION_MODULUS_MEASURED: Unit.GPA,
    }
    return table.get(PropertyName(name))
