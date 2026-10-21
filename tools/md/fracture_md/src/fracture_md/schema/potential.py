"""Interatomic-potential identity cards.

A result is only as trustworthy as the potential that produced it, so every
property record points at a :class:`PotentialCard` by `id`.  The card records
identity (functional form + parameter/weights hash), license, training or
fitting domain, and *per-property-class* validation evidence.  In particular,
agreement on equilibrium bulk properties is recorded separately from evidence
for bond-breaking (cleavage, ideal strength), because the former does not imply
the latter.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import Field

from fracture_md.schema.common import Citation, StrictModel, ValidationEvidence
from fracture_md.units import Quantity


class PotentialKind(str, Enum):
    ANALYTIC_PAIR = "analytic_pair"
    ANALYTIC_MANY_BODY = "analytic_many_body"  # pair + explicit three-body terms (Stillinger-Weber, Vashishta)
    BOND_ORDER = "bond_order"  # Tersoff/Brenner/Erhart-Albe
    MACHINE_LEARNED = "machine_learned"
    DFT = "dft"  # a DFT "engine" used as a potential
    TEST_FIXTURE = "test_fixture"  # toy model for numerical checks only


class PropertyClass(str, Enum):
    """Regimes in which a potential may or may not be trustworthy."""

    EQUILIBRIUM_BULK = "equilibrium_bulk"  # lattice, cohesive energy
    ELASTIC = "elastic"  # small-strain response
    LARGE_STRAIN = "large_strain"  # ideal strength, instabilities
    SURFACES = "surfaces"  # surface energies, reconstructions
    BOND_BREAKING = "bond_breaking"  # cleavage traction-separation, crack tips
    STACKING_FAULTS = "stacking_faults"  # GSFE, twins, polytypes
    INTERFACES = "interfaces"  # GB / phase-boundary energetics
    DEFECTS = "defects"  # vacancies, interstitials
    FINITE_TEMPERATURE = "finite_temperature"  # thermal expansion, MD
    PHASE_STABILITY = "phase_stability"
    # Dislocation nucleation/glide, amorphisation and phase transformation under contact loading
    # (indentation, scratching); distinct from the homogeneous ideal strength (LARGE_STRAIN).
    PLASTICITY = "plasticity"
    # Equation of state and bonding beyond ~10 GPa (indentation mean contact pressures are 30-100 GPa;
    # cutoff-function artefacts of bond-order potentials appear here first).
    HIGH_PRESSURE = "high_pressure"


class DomainAssessment(StrictModel):
    property_class: PropertyClass
    status: Literal["validated", "partially_validated", "unvalidated", "known_deficient"]
    evidence: list[ValidationEvidence] = Field(default_factory=list)
    notes: str = ""


class TrainingDomain(StrictModel):
    elements: list[str]
    description: str
    data_sources: list[str] = Field(default_factory=list)  # e.g. "MPtrj (PBE/PBE+U)"
    reference_method: Optional[str] = None  # e.g. "DFT-PBE", "fit to experiment + LDA"
    includes_surfaces: Optional[bool] = None
    includes_large_strain: Optional[bool] = None
    includes_bond_breaking: Optional[bool] = None


class PotentialCard(StrictModel):
    id: str  # stable identifier, e.g. "tersoff-1989-SiC@<hash8>"
    name: str
    kind: PotentialKind
    functional_form: str
    elements: list[str]
    cutoff: Quantity  # largest interaction cutoff
    parameters_sha256: str  # hash of canonical parameter JSON or weights file
    parameter_source: str  # where parameters/weights were obtained
    license: str  # license of the parameters/weights (not only the code)
    code_license: Optional[str] = None
    implementation: str  # python import path of the energy function
    implementation_sha256: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)
    training_domain: TrainingDomain
    domain_assessment: list[DomainAssessment] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    differentiable: bool = True
    precision: Literal["float32", "float64", "mixed"] = "float64"
    notes: str = ""

    def status_for(self, pc: PropertyClass) -> str:
        for d in self.domain_assessment:
            if d.property_class == pc:
                return d.status
        return "unvalidated"
