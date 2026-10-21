"""Photocurable composite-resin formulations (e.g. 3D-printable dental restoratives).

A formulation is a Candidate whose "structure" is a recipe plus a processing
route rather than a crystal.  Amounts are stored as weight fractions (the way a
lab weighs them); volume fractions are derived from component densities and
recorded alongside so that micromechanics never silently mixes the two.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import Field, model_validator

from fracture_md.schema.common import Citation, StrictModel
from fracture_md.units import Quantity


class ComponentRole(str, Enum):
    BASE_MONOMER = "base_monomer"  # e.g. UDMA, Bis-EMA, Bis-GMA
    DILUENT_MONOMER = "diluent_monomer"  # e.g. TEGDMA
    PHOTOINITIATOR = "photoinitiator"  # e.g. TPO, BAPO (385/405 nm)
    CO_INITIATOR = "co_initiator"
    INHIBITOR = "inhibitor"  # e.g. BHT, MEHQ
    UV_ABSORBER = "uv_absorber"  # controls cure depth / print resolution
    FILLER = "filler"  # e.g. silanised Ba glass, silica, zirconia
    COUPLING_AGENT = "coupling_agent"  # e.g. gamma-MPS, 8-MOTS (usually pre-applied to filler)
    PIGMENT = "pigment"
    OTHER = "other"


class Component(StrictModel):
    name: str
    role: ComponentRole
    weight_fraction: float = Field(ge=0.0, le=1.0)
    cas: Optional[str] = None
    density: Optional[Quantity] = None  # g/cm^3
    refractive_index: Optional[float] = None  # at ~589 nm unless noted
    # Fillers: size distribution (d50 and span) and surface treatment
    particle_d50: Optional[Quantity] = None  # um
    particle_size_note: str = ""
    surface_treatment: Optional[str] = None  # e.g. "3 wt% gamma-MPS"
    supplier_examples: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    notes: str = ""


class ProcessingRoute(StrictModel):
    method: Literal["DLP", "SLA", "LCD", "milling_block", "direct_placement"] = "DLP"
    wavelength_nm: Optional[float] = 405.0
    layer_thickness: Optional[Quantity] = None  # um
    print_temperature: Optional[Quantity] = None  # K
    exposure_dose: Optional[Quantity] = None  # mJ/cm^2 per layer
    post_cure: str = ""  # e.g. "N2 atmosphere, 405 nm, 60 C, 2 x 10 min"
    build_orientation: str = ""
    notes: str = ""


class Formulation(StrictModel):
    name: str
    components: list[Component]
    processing: ProcessingRoute = Field(default_factory=ProcessingRoute)
    filler_volume_fraction: Optional[float] = None  # derived; see composites.formulation
    notes: str = ""

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "Formulation":
        tot = sum(c.weight_fraction for c in self.components)
        if abs(tot - 1.0) > 1e-6:
            raise ValueError(f"component weight fractions must sum to 1 (got {tot:.6f})")
        return self

    def by_role(self, role: ComponentRole) -> list[Component]:
        return [c for c in self.components if c.role == role]
