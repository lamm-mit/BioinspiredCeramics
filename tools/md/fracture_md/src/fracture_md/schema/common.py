"""Cross-cutting schema types: validation status, uncertainty, provenance, conditions.

Design rule: every reported number carries (a) units, (b) the conditions it was
computed or measured at, (c) how it was obtained, (d) uncertainty or convergence
evidence, and (e) a validation status.  Nothing in the physics core returns a
bare float to the outside world.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

from fracture_md.units import Quantity, Unit, UnitError, q, require_compatible

SCHEMA_VERSION = "0.1.0"


class StrictModel(BaseModel):
    """Base model: forbid unknown fields so schema drift is caught early."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
class ValidationStatus(str, Enum):
    """Ordered from weakest to strongest evidence (FAILED/OUT_OF_DOMAIN are flags)."""

    UNVALIDATED = "unvalidated"
    # Numerical checks passed (convergence, symmetry, FD==AD, integration).
    NUMERICALLY_VERIFIED = "numerically_verified"
    # Agrees with an independent trusted implementation of the *same model*
    # (e.g. JAX Tersoff vs matscipy Tersoff).  Says nothing about physics.
    REFERENCE_IMPLEMENTATION_AGREEMENT = "reference_implementation_agreement"
    # Compared against published values for the same model (e.g. Tersoff paper).
    LITERATURE_MODEL_AGREEMENT = "literature_model_agreement"
    # Compared against DFT of the same quantity/structure.
    VALIDATED_AGAINST_DFT = "validated_against_dft"
    # Compared against experiment.
    VALIDATED_AGAINST_EXPERIMENT = "validated_against_experiment"
    # Derived with an explicit model from other outputs (e.g. Griffith K_Ic).
    MODEL_ESTIMATE = "model_estimate"
    # Quantity lies outside the potential's validated domain.
    OUT_OF_DOMAIN = "out_of_domain"
    FAILED = "failed"


# The evidence ladder, weakest to strongest.  MODEL_ESTIMATE / OUT_OF_DOMAIN / FAILED are flags, not rungs.
EVIDENCE_LADDER: tuple[ValidationStatus, ...] = (
    ValidationStatus.UNVALIDATED,
    ValidationStatus.NUMERICALLY_VERIFIED,
    ValidationStatus.REFERENCE_IMPLEMENTATION_AGREEMENT,
    ValidationStatus.LITERATURE_MODEL_AGREEMENT,
    ValidationStatus.VALIDATED_AGAINST_DFT,
    ValidationStatus.VALIDATED_AGAINST_EXPERIMENT,
)


def status_exceeds(status: ValidationStatus, ceiling: ValidationStatus) -> bool:
    """True if `status` is a rung of the evidence ladder above `ceiling` (flags never exceed)."""
    if status not in EVIDENCE_LADDER or ceiling not in EVIDENCE_LADDER:
        return False
    return EVIDENCE_LADDER.index(status) > EVIDENCE_LADDER.index(ceiling)


class CheckResult(StrictModel):
    """Outcome of one automated check (symmetry, convergence, regression...)."""

    name: str
    passed: bool
    metric: Optional[float] = None
    tolerance: Optional[float] = None
    unit: Optional[Unit] = None
    details: str = ""


class ValidationEvidence(StrictModel):
    kind: Literal[
        "numerical_check",
        "reference_implementation",
        "literature_model_value",
        "dft_reference",
        "experimental_reference",
        "expert_judgement",
    ]
    description: str
    reference_value: Optional[Quantity] = None
    computed_value: Optional[Quantity] = None
    relative_difference: Optional[float] = None
    citation: Optional[str] = None
    passed: Optional[bool] = None


class Validation(StrictModel):
    status: ValidationStatus = ValidationStatus.UNVALIDATED
    evidence: list[ValidationEvidence] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------
class UncertaintyKind(str, Enum):
    NUMERICAL_CONVERGENCE = "numerical_convergence"  # vs strain step, size, vacuum...
    AUTODIFF_VS_FINITE_DIFFERENCE = "autodiff_vs_finite_difference"
    FIT_STANDARD_ERROR = "fit_standard_error"
    STATISTICAL_SAMPLING = "statistical_sampling"  # MD time averages, seeds
    ENSEMBLE = "ensemble"  # committee / ensemble of potentials
    INTER_POTENTIAL_SPREAD = "inter_potential_spread"  # model-form proxy
    REFERENCE_DISCREPANCY = "reference_discrepancy"  # vs DFT/experiment
    MEASUREMENT = "measurement"  # experimental error bar
    UNQUANTIFIED = "unquantified"


class Uncertainty(StrictModel):
    kind: UncertaintyKind
    # Absolute half-width (1-sigma unless `interval` says otherwise).
    value: Optional[float] = None
    unit: Optional[Unit] = None
    relative: Optional[float] = None
    interval: Literal["1sigma", "2sigma", "range", "max_abs_diff", "n/a"] = "n/a"
    samples: Optional[list[float]] = None
    description: str = ""


# --------------------------------------------------------------------------
# Conditions & provenance
# --------------------------------------------------------------------------
class DefectState(StrictModel):
    """Defect content of the simulated/measured region."""

    kind: Literal[
        "pristine",
        "point_defects",
        "pore",
        "grain_boundary",
        "phase_boundary",
        "stacking_fault",
        "twin_boundary",
        "dislocation",
        "crack",
        "polycrystal",
        "unknown",
    ] = "pristine"
    # Free-form but typed parameters, e.g. {"vacancy_fraction_Si": 0.01}.
    parameters: dict[str, float] = Field(default_factory=dict)
    description: str = ""


IndenterShape = Literal["berkovich", "sphere", "cone"]
# rigid_sdf: rigid frictionless repulsive wall on the tip's signed-distance function; atomistic: rigid (or
# flexible) tip made of atoms; physical: a real indenter (experimental records only).
TipModel = Literal["rigid_sdf", "atomistic", "physical"]
# Contact-area estimators of an indentation hardness (recipe_v0, protocol item 15):
#   nominal_ideal          E1: ideal sharp-tip area C0 h^2 (never plateau evidence)
#   rounded_area_function  E2: exact rounded-tip area function at the contact depth
#   dilated_hull           E3: convex hull of the projected contact atoms, dilated by a/(2 sqrt 2) (primary)
#   closed_discs           E4: union of atom discs after a morphological closing (check)
#   triangle_fit           E5: triangle fitted to the contact atoms (never plateau evidence)
#   oliver_pharr           A(h_c) with h_c from the Oliver-Pharr unloading fit
AreaEstimator = Literal["nominal_ideal", "rounded_area_function", "dilated_hull", "closed_discs", "triangle_fit",
                        "oliver_pharr"]


class ContactConditions(StrictModel):
    """Indenter and contact set-up behind an indentation value.

    Every ``*_MD`` indentation record carries one (the PropertyRecord validator enforces it), so a
    number can never be separated from the tip model, wall stiffness, azimuth, speed and depth it
    depends on.  Angles in degrees, lengths in angstrom, speed in m/s or angstrom/ps, wall stiffness
    in eV/angstrom^3 (any compatible unit is accepted; dimensions are checked).
    """

    indenter_shape: IndenterShape
    tip_model: TipModel
    face_angle: Optional[Quantity] = None  # Berkovich centre-line-to-face angle (65.27 deg, modified Berkovich)
    cone_half_angle: Optional[Quantity] = None  # cone only (70.296 deg = area-equivalent cone of the Berkovich)
    tip_radius: Quantity  # rounding radius of the pyramid/cone, or the sphere radius; 0 = sharp
    azimuth: Quantity = Field(default_factory=lambda: q(0.0, Unit.DEGREE))  # face-1 normal vs lab +x
    azimuth_reference: str = ""  # crystal direction of lab +x, e.g. "[110] of the (001) slab"
    wall_stiffness: Optional[Quantity] = None  # K of U = (K/3) sum max(0, -sdf)^3 (rigid_sdf tips only)
    tip_description: str = ""  # atomistic tips: material and tip-sample interaction
    indentation_speed: Optional[Quantity] = None  # MD loading speed; None = quasi-static
    max_depth: Quantity  # deepest nominal tip displacement of the run
    depth: Optional[Quantity] = None  # depth (or end of the averaging window) at which the value was taken
    depth_zero_rule: Literal["first_contact", "elastic_fit", "geometric", "not_applicable"] = "not_applicable"
    area_estimator: Optional[AreaEstimator] = None
    surface_termination: str = ""  # e.g. "C-terminated (001), unreconstructed"
    lateral_size: Optional[Quantity] = None  # [L_x, L_y] of the periodic cell
    thickness: Optional[Quantity] = None  # slab thickness including fixed/thermostat layers
    boundary: str = ""  # e.g. "bottom 10 A fixed; next 10 A CSVR 300 K (0.1 ps); NVE above; lateral periodic"

    @model_validator(mode="after")
    def _check(self) -> "ContactConditions":
        for name, like in (("face_angle", Unit.DEGREE), ("cone_half_angle", Unit.DEGREE), ("azimuth", Unit.DEGREE),
                           ("tip_radius", Unit.ANGSTROM), ("max_depth", Unit.ANGSTROM), ("depth", Unit.ANGSTROM),
                           ("lateral_size", Unit.ANGSTROM), ("thickness", Unit.ANGSTROM),
                           ("wall_stiffness", Unit.EV_PER_ANGSTROM3), ("indentation_speed", Unit.METER_PER_SECOND)):
            try:
                require_compatible(getattr(self, name), like, f"contact.{name}")
            except UnitError as exc:
                raise ValueError(str(exc)) from None
        if self.indenter_shape == "berkovich" and self.face_angle is None:
            raise ValueError("a Berkovich contact must state its face_angle (65.27 deg for the modified Berkovich)")
        if self.indenter_shape == "cone" and self.cone_half_angle is None:
            raise ValueError("a conical contact must state its cone_half_angle")
        for name in ("face_angle", "cone_half_angle"):
            a = getattr(self, name)
            if a is not None and not 0.0 < a.magnitude(Unit.DEGREE) < 90.0:
                raise ValueError(f"contact.{name} must lie in (0, 90) deg")
        r = self.tip_radius.magnitude(Unit.ANGSTROM)
        if r < 0.0 or (self.indenter_shape == "sphere" and r <= 0.0):
            raise ValueError("contact.tip_radius must be >= 0 (> 0 for a sphere)")
        if self.tip_model == "rigid_sdf" and (self.wall_stiffness is None or self.wall_stiffness.magnitude() <= 0.0):
            raise ValueError("a rigid_sdf tip must state its wall_stiffness K > 0 (eV/angstrom^3)")
        if self.tip_model == "atomistic" and not self.tip_description.strip():
            raise ValueError("an atomistic tip must be described in tip_description (material, interaction)")
        h_max = self.max_depth.magnitude(Unit.ANGSTROM)
        if not h_max > 0.0:
            raise ValueError("contact.max_depth must be > 0")
        if self.depth is not None and self.depth.magnitude(Unit.ANGSTROM) > h_max * (1.0 + 1e-9):
            raise ValueError("contact.depth exceeds contact.max_depth")
        if self.indentation_speed is not None and not self.indentation_speed.magnitude() > 0.0:
            raise ValueError("contact.indentation_speed must be > 0")
        if self.lateral_size is not None:
            ls = self.lateral_size.as_array()
            if ls.shape != (2,) or not (ls > 0).all():
                raise ValueError("contact.lateral_size must be [L_x, L_y] with positive entries")
        return self


class Conditions(StrictModel):
    """Thermodynamic / loading conditions for a value."""

    temperature: Quantity = Field(default_factory=lambda: q(0.0, Unit.KELVIN))
    # "static" means energy minimisation / 0 K lattice statics (no rate).
    loading: Literal["static", "quasistatic", "dynamic", "not_applicable"] = "static"
    strain_rate: Optional[Quantity] = None
    stress_state: str = ""  # e.g. "uniaxial strain, lateral lattice fixed to bulk"
    defect_state: DefectState = Field(default_factory=DefectState)
    environment: str = "vacuum"
    # Indentation / contact loading (required for *_MD indentation records; omitted from dumps when None so
    # that the ids and content hashes of every non-contact record are unchanged).
    contact: Optional[ContactConditions] = None

    @model_validator(mode="after")
    def _dimensions(self) -> "Conditions":
        try:
            require_compatible(self.temperature, Unit.KELVIN, "conditions.temperature")
            require_compatible(self.strain_rate, Unit.PER_SECOND, "conditions.strain_rate")
        except UnitError as exc:
            raise ValueError(str(exc)) from None
        return self

    @model_serializer(mode="wrap")
    def _omit_unset_contact(self, handler):
        d = handler(self)
        if self.contact is None and isinstance(d, dict):
            d.pop("contact", None)
        return d


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EngineRunProvenance(StrictModel):
    """The simulation engine behind a number and the hashes needed to reproduce it.

    The JAX harness is the verified reference for small cells and record building; LAMMPS is the
    production engine for large (indentation) cells, driven by a deck generator and read back by an
    importer.  Either way the record states the engine and its version, the potential card and its
    convention (e.g. the Erhart-Albe three-body indexing), the precision and, for LAMMPS, the sha256 of
    the input deck and of every potential file the deck reads.
    """

    engine: Literal["jax", "lammps", "ase_reference"]
    engine_version: str  # e.g. "jax 0.11.2, jax-md 0.2.29" or "LAMMPS 22 Jul 2025 update 6"
    potential_card_id: Optional[str] = None
    potential_convention: str = ""  # e.g. "three_body_index=ik", "pair_style vashishta, file D = D_paper/2"
    potential_files_sha256: dict[str, str] = Field(default_factory=dict)  # file name -> sha256
    deck_sha256: Optional[str] = None  # LAMMPS input deck (required for engine="lammps")
    deck_generator: str = ""  # import path (+ version) of the generator that wrote the deck
    recipe_sha256: Optional[str] = None  # frozen protocol recipe (e.g. recipe_v0.json)
    dtype: Literal["float32", "float64", "mixed", "unknown"] = "unknown"
    matmul_precision: Optional[str] = None  # JAX_DEFAULT_MATMUL_PRECISION (must be recorded for float32 JAX)
    xla_flags: Optional[str] = None
    accelerator: str = ""  # e.g. "cpu", "cuda (GB10)", "kokkos/cuda"
    mpi_ranks: Optional[int] = None
    command: str = ""  # command line (LAMMPS) or entry point (JAX)

    @field_validator("deck_sha256", "recipe_sha256")
    @classmethod
    def _hex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _SHA256.match(v):
            raise ValueError(f"expected a lowercase hex sha256 digest, got {v!r}")
        return v

    @model_validator(mode="after")
    def _check(self) -> "EngineRunProvenance":
        if not self.engine_version.strip():
            raise ValueError("engine_version must be stated")
        bad = [k for k, v in self.potential_files_sha256.items() if not _SHA256.match(v)]
        if bad:
            raise ValueError(f"potential_files_sha256 entries are not sha256 digests: {bad}")
        if self.engine == "lammps":
            if self.deck_sha256 is None:
                raise ValueError("a LAMMPS run must record deck_sha256 (sha256 of the input deck)")
            if not self.potential_files_sha256:
                raise ValueError("a LAMMPS run must record potential_files_sha256 (every pair file the deck reads; "
                                 "hash the pair_coeff block for inline coefficients)")
        if self.engine == "jax" and self.dtype == "float32" and not self.matmul_precision:
            raise ValueError("a float32 JAX run must record matmul_precision (JAX_DEFAULT_MATMUL_PRECISION)")
        if self.mpi_ranks is not None and self.mpi_ranks < 1:
            raise ValueError("mpi_ranks must be >= 1")
        return self


class Provenance(StrictModel):
    """Where a number came from.  Filled by fracture_md.provenance.collect()."""

    created_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = SCHEMA_VERSION
    code_version: str = "unknown"  # git describe of fracture_md
    code_dirty: Optional[bool] = None
    implementation_ids: dict[str, str] = Field(default_factory=dict)  # name -> content hash
    engine: str = ""
    engine_version: str = ""
    package_versions: dict[str, str] = Field(default_factory=dict)
    platform: str = ""
    device: str = ""
    float_precision: Literal["float32", "float64", "unknown"] = "unknown"
    potential_id: Optional[str] = None
    protocol_id: Optional[str] = None
    input_hashes: dict[str, str] = Field(default_factory=dict)
    random_seeds: dict[str, int] = Field(default_factory=dict)
    requested_by: str = "human"  # or an agent id
    parent_ids: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    # Simulation engine of the run (required for *_MD indentation records; omitted from dumps when None).
    engine_run: Optional[EngineRunProvenance] = None

    @model_serializer(mode="wrap")
    def _omit_unset_engine_run(self, handler):
        d = handler(self)
        if self.engine_run is None and isinstance(d, dict):
            d.pop("engine_run", None)
        return d


class Citation(StrictModel):
    key: str
    text: str
    doi: Optional[str] = None
    url: Optional[str] = None
