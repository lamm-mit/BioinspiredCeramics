"""Material, orientation, interface, microstructure and candidate representations."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import Field, model_validator

from fracture_md.schema.common import Citation, DefectState, StrictModel
from fracture_md.units import Quantity


class Composition(StrictModel):
    """Chemical composition.  `formula` is the reduced formula, e.g. 'SiC'."""

    formula: str
    # element -> amount (atoms per formula unit, or fraction for solid solutions)
    elements: dict[str, float]
    is_fraction: bool = False

    @classmethod
    def from_formula(cls, formula: str) -> "Composition":
        from pymatgen.core import Composition as PmgComposition

        c = PmgComposition(formula)
        return cls(
            formula=c.reduced_formula,
            elements={str(el): float(n) for el, n in c.get_el_amt_dict().items()},
        )


class Phase(StrictModel):
    """A crystalline (or amorphous) phase."""

    name: str  # e.g. "3C-SiC"
    prototype: Optional[str] = None  # e.g. "zincblende (B3)"
    space_group_symbol: Optional[str] = None  # e.g. "F-43m"
    space_group_number: Optional[int] = None
    polytype: Optional[str] = None
    crystal_system: Optional[str] = None
    external_ids: dict[str, str] = Field(default_factory=dict)  # {"materials_project": "mp-8062"}
    amorphous: bool = False


class Orientation(StrictModel):
    """Crystallographic orientation of a loading axis or a plane.

    Conventions:
      * `plane_miller` are (hkl) Miller indices of a plane in the conventional
        cell of `phase`; for hexagonal phases use 4-index (hkil) in `plane_miller_bravais`.
      * `x_direction`/`y_direction` are [uvw] lattice directions that lie in the
        plane and define the lab x/y axes of the simulation cell; lab z is the
        plane normal.  For loading along a direction, use `direction` only.
      * `rotation_crystal_to_lab` (row-major 3x3) maps Cartesian crystal-frame
        vectors to lab-frame vectors.  Tensors are reported in the frame named in
        the property record.
    """

    plane_miller: Optional[tuple[int, int, int]] = None
    plane_miller_bravais: Optional[tuple[int, int, int, int]] = None
    direction: Optional[tuple[int, int, int]] = None
    x_direction: Optional[tuple[int, int, int]] = None
    y_direction: Optional[tuple[int, int, int]] = None
    rotation_crystal_to_lab: Optional[list[list[float]]] = None
    description: str = ""


class Termination(StrictModel):
    """Which atomic plane a cleavage/interface cut passes through."""

    label: str  # e.g. "shuffle", "glide", "Si|C", "nonpolar"
    cut_fraction: Optional[float] = None  # fractional height of the cut in the oriented cell
    species_below: list[str] = Field(default_factory=list)  # species of the layer just below the cut
    species_above: list[str] = Field(default_factory=list)
    broken_bonds_per_area: Optional[Quantity] = None  # bonds per area, Unit.PER_ANGSTROM2
    polar: Optional[bool] = None
    stoichiometric_halves: Optional[bool] = None
    description: str = ""


class InterfaceKind(str, Enum):
    CLEAVAGE_PLANE = "cleavage_plane"  # bulk -> two free surfaces
    FREE_SURFACE = "free_surface"
    GRAIN_BOUNDARY = "grain_boundary"
    TWIN_BOUNDARY = "twin_boundary"
    STACKING_FAULT = "stacking_fault"
    PHASE_BOUNDARY = "phase_boundary"


class InterfaceSpec(StrictModel):
    kind: InterfaceKind
    phase_a: str
    phase_b: Optional[str] = None  # None for cleavage/free surface/GB of the same phase
    orientation_a: Orientation
    orientation_b: Optional[Orientation] = None
    termination: Optional[Termination] = None
    csl_sigma: Optional[int] = None
    misorientation_axis: Optional[tuple[int, int, int]] = None
    misorientation_angle: Optional[Quantity] = None
    coherency: Optional[Literal["coherent", "semicoherent", "incoherent"]] = None
    lattice_misfit: Optional[float] = None  # dimensionless
    description: str = ""


class Microstructure(StrictModel):
    """Mesoscale description (for microstructure candidates and PF-CZM geometry)."""

    description: str
    phase_fractions: dict[str, float] = Field(default_factory=dict)  # volume fractions
    grain_size: Optional[Quantity] = None
    grain_size_distribution: Optional[dict[str, float]] = None  # e.g. lognormal params
    grain_aspect_ratio: Optional[float] = None
    twin_spacing: Optional[Quantity] = None
    porosity: Optional[float] = None  # volume fraction
    texture: Optional[str] = None
    interfaces: list[InterfaceSpec] = Field(default_factory=list)
    defects: list[DefectState] = Field(default_factory=list)


class StructureRecord(StrictModel):
    """Atomistic structure, engine-agnostic (ASE conventions: cell rows are lattice vectors)."""

    symbols: list[str]
    cell_rows_angstrom: list[list[float]]
    positions_angstrom: list[list[float]]
    pbc: tuple[bool, bool, bool] = (True, True, True)
    source: str = ""  # e.g. "ase.build.bulk", "materials_project:mp-8062"
    sha256: Optional[str] = None


# --------------------------------------------------------------------------
# Manufacturability & novelty
# --------------------------------------------------------------------------
class EvidenceLevel(str, Enum):
    DEMONSTRATED = "demonstrated"  # made and characterised as specified
    ANALOGUE_DEMONSTRATED = "analogue_demonstrated"  # close analogue made
    PLAUSIBLE = "plausible"  # thermodynamics & known routes support it
    SPECULATIVE = "speculative"  # no direct support, not ruled out
    IMPLAUSIBLE = "implausible"  # contradicted by thermodynamics/processing evidence
    UNASSESSED = "unassessed"


class ProcessingRoute(StrictModel):
    name: str  # e.g. "chemical vapour deposition", "spark plasma sintering"
    precursors: list[str] = Field(default_factory=list)
    temperature_range: Optional[tuple[Quantity, Quantity]] = None
    pressure: Optional[str] = None
    atmosphere: Optional[str] = None
    notes: str = ""
    citations: list[Citation] = Field(default_factory=list)


class ManufacturabilityEvidence(StrictModel):
    level: EvidenceLevel = EvidenceLevel.UNASSESSED
    energy_above_hull: Optional[Quantity] = None  # eV/atom
    stability_source: Optional[str] = None  # e.g. "Materials Project r2024" or "this work (Tersoff)"
    competing_phases: list[str] = Field(default_factory=list)
    chemical_compatibility: str = ""  # interfacial reactions, oxidation, etc.
    routes: list[ProcessingRoute] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    notes: str = ""


class NoveltyStatus(str, Enum):
    KNOWN = "known"  # well documented material/microstructure
    KNOWN_CONCEPT_NEW_PARAMETERS = "known_concept_new_parameters"
    POSSIBLY_NOVEL = "possibly_novel"  # no prior art found in the searches recorded
    UNASSESSED = "unassessed"


class NoveltyAssessment(StrictModel):
    status: NoveltyStatus = NoveltyStatus.UNASSESSED
    claim: str = ""  # the specific thing claimed to be new
    searches: list[str] = Field(default_factory=list)  # queries/databases consulted
    prior_art: list[Citation] = Field(default_factory=list)
    notes: str = ""


class CandidateKind(str, Enum):
    COMPOSITION = "composition"
    PHASE = "phase"
    INTERFACE = "interface"
    PHASE_COMBINATION = "phase_combination"
    MICROSTRUCTURE = "microstructure"
    REFERENCE = "reference"  # a baseline, not a discovery candidate
    FORMULATION = "formulation"  # a composite recipe + processing route (e.g. dental resin)


class Candidate(StrictModel):
    """Anything the discovery loop can propose and evaluate."""

    id: str
    name: str
    kind: CandidateKind
    composition: Composition
    phases: list[Phase] = Field(default_factory=list)
    interfaces: list[InterfaceSpec] = Field(default_factory=list)
    microstructure: Optional[Microstructure] = None
    structure: Optional[StructureRecord] = None
    # Composite recipe (schema.formulation.Formulation) for FORMULATION candidates.
    formulation: Optional["Formulation"] = None
    hypothesis: str = ""  # why it might be strong and fracture resistant
    mechanism_tags: list[str] = Field(default_factory=list)  # e.g. ["crack_deflection", "twin_toughening"]
    novelty: NoveltyAssessment = Field(default_factory=NoveltyAssessment)
    manufacturability: ManufacturabilityEvidence = Field(default_factory=ManufacturabilityEvidence)
    proposed_by: str = "human"
    parent_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _phases_for_phase_candidates(self) -> "Candidate":
        if self.kind in (CandidateKind.PHASE, CandidateKind.REFERENCE) and not self.phases:
            raise ValueError("phase/reference candidates must name at least one phase")
        return self


from fracture_md.schema.formulation import Formulation  # noqa: E402  (forward ref)

Candidate.model_rebuild()
