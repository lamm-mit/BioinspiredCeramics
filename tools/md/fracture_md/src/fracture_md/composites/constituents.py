"""Typed constituent database for photocurable dental composites, and weight <-> volume
fraction conversion for :class:`fracture_md.schema.formulation.Formulation` objects.

Data live in ``data/dental/constituents.json``.  Each property is a :class:`Quantity` with an
optional ``[low, high]`` range, the conditions it refers to, citation keys (resolved in the
file's ``citations`` table) and a confidence label:

* ``high``    - seen in a primary source's table/full text or an official datasheet
* ``medium``  - primary abstract, manufacturer document, or a secondary source quoting a primary
* ``low``     - secondary/snippet level; use with care
* ``assumed`` - NOT a literature value: an engineering assumption stated as such (with a range)

Weight fractions are what a lab weighs; volume fractions drive micromechanics, rheology and
optics.  ``state="uncured"`` uses liquid monomer densities (paste: viscosity, cure depth);
``state="cured"`` uses cured-network densities (hardness, sorption, modulus).
"""

from __future__ import annotations

import re

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Mapping, Optional

from pydantic import Field

from fracture_md.schema.common import Citation, StrictModel
from fracture_md.schema.formulation import Component, ComponentRole, Formulation
from fracture_md.units import Quantity, Unit

REPO_ROOT = Path(__file__).resolve().parents[3]
CONSTITUENTS_PATH = REPO_ROOT / "data" / "dental" / "constituents.json"

Confidence = Literal["high", "medium", "low", "assumed"]
ConstituentKind = Literal["monomer", "photoinitiator", "inhibitor", "filler", "silane", "absorber"]


class ConstituentProperty(StrictModel):
    value: Quantity
    low: Optional[float] = None  # same unit as value
    high: Optional[float] = None
    conditions: str = ""
    citation_keys: list[str] = Field(default_factory=list)
    confidence: Confidence = "medium"
    notes: str = ""

    def magnitude(self, unit: Unit | str | None = None) -> float:
        return self.value.magnitude(unit)

    def range(self, unit: Unit | str | None = None) -> tuple[float, float]:
        """(low, high) in `unit`; falls back to (value, value) when no range is stored."""
        from fracture_md.units import convert

        u = Unit(unit) if unit is not None else self.value.unit
        lo = self.low if self.low is not None else self.value.value
        hi = self.high if self.high is not None else self.value.value
        return float(convert(lo, self.value.unit, u)), float(convert(hi, self.value.unit, u))


class Constituent(StrictModel):
    id: str
    name: str
    kind: ConstituentKind
    role: ComponentRole
    aliases: list[str] = Field(default_factory=list)
    cas: Optional[str] = None
    filler_type: Optional[str] = None  # e.g. "ba_glass", "silica", "zirconia", "zirconia_silica", "ybf3"
    properties: dict[str, ConstituentProperty] = Field(default_factory=dict)
    supplier_examples: list[str] = Field(default_factory=list)
    notes: str = ""

    def has(self, key: str) -> bool:
        return key in self.properties

    def prop(self, key: str) -> ConstituentProperty:
        if key not in self.properties:
            raise KeyError(f"constituent {self.id!r} has no property {key!r} (have {sorted(self.properties)})")
        return self.properties[key]

    def get(self, key: str, unit: Unit | str | None = None, default: Optional[float] = None) -> Optional[float]:
        if key not in self.properties:
            return default
        return self.properties[key].magnitude(unit)


class ConstituentDatabase(StrictModel):
    version: str = "0.1.0"
    description: str = ""
    citations: dict[str, Citation] = Field(default_factory=dict)
    constituents: dict[str, Constituent] = Field(default_factory=dict)

    def __getitem__(self, key: str) -> Constituent:
        return self.lookup(key)

    def lookup(self, name: str) -> Constituent:
        """Find a constituent by id, name or alias (case/format-insensitive)."""
        k = _norm(name)
        for c in self.constituents.values():
            if k == _norm(c.id) or k == _norm(c.name) or any(k == _norm(a) for a in c.aliases):
                return c
        raise KeyError(f"unknown constituent {name!r}")

    def find(self, name: str) -> Optional[Constituent]:
        try:
            return self.lookup(name)
        except KeyError:
            return None

    def of_kind(self, kind: ConstituentKind) -> list[Constituent]:
        return [c for c in self.constituents.values() if c.kind == kind]

    def cite(self, keys: list[str]) -> list[Citation]:
        return [self.citations[k] for k in keys if k in self.citations]


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _parse_prop(d: Mapping) -> ConstituentProperty:
    return ConstituentProperty(
        value=Quantity(value=d["value"], unit=Unit(d["unit"])),
        low=d.get("low"),
        high=d.get("high"),
        conditions=d.get("cond", d.get("conditions", "")),
        citation_keys=list(d.get("cite", d.get("citation_keys", []))),
        confidence=d.get("conf", d.get("confidence", "medium")),
        notes=d.get("notes", ""),
    )


@lru_cache(maxsize=4)
def _load(path: str) -> ConstituentDatabase:
    raw = json.loads(Path(path).read_text())
    cits = {k: Citation(key=k, **v) for k, v in raw.get("citations", {}).items()}
    cons = {}
    for cid, c in raw["constituents"].items():
        props = {k: _parse_prop(v) for k, v in c.get("properties", {}).items()}
        cons[cid] = Constituent(
            id=cid,
            name=c["name"],
            kind=c["kind"],
            role=ComponentRole(c["role"]),
            aliases=c.get("aliases", []),
            cas=c.get("cas"),
            filler_type=c.get("filler_type"),
            properties=props,
            supplier_examples=c.get("supplier_examples", []),
            notes=c.get("notes", ""),
        )
    db = ConstituentDatabase(
        version=raw.get("version", "0.1.0"), description=raw.get("description", ""), citations=cits, constituents=cons
    )
    missing = {k for c in cons.values() for p in c.properties.values() for k in p.citation_keys if k not in cits}
    if missing:
        raise ValueError(f"constituents.json cites unknown keys: {sorted(missing)}")
    return db


def load_constituents(path: Path | str | None = None) -> ConstituentDatabase:
    return _load(str(path or CONSTITUENTS_PATH))


# ---------------------------------------------------------------------------
# Weight <-> volume fractions
# ---------------------------------------------------------------------------
def weight_to_volume_fractions(weights: Mapping[str, float], densities: Mapping[str, float]) -> dict[str, float]:
    """phi_i = (w_i/rho_i) / sum_j (w_j/rho_j).  Densities in any common unit (g/cm^3)."""
    specific = {k: w / densities[k] for k, w in weights.items()}
    tot = sum(specific.values())
    if tot <= 0:
        raise ValueError("total specific volume must be positive")
    return {k: v / tot for k, v in specific.items()}


def volume_to_weight_fractions(volumes: Mapping[str, float], densities: Mapping[str, float]) -> dict[str, float]:
    """w_i = phi_i rho_i / sum_j phi_j rho_j."""
    mass = {k: v * densities[k] for k, v in volumes.items()}
    tot = sum(mass.values())
    if tot <= 0:
        raise ValueError("total mass must be positive")
    return {k: m / tot for k, m in mass.items()}


def two_phase_volume_fraction(filler_weight_fraction: float, rho_filler: float, rho_matrix: float) -> float:
    """Filler volume fraction of a two-phase composite from its filler weight fraction."""
    w = filler_weight_fraction
    return (w / rho_filler) / (w / rho_filler + (1.0 - w) / rho_matrix)


def two_phase_weight_fraction(filler_volume_fraction: float, rho_filler: float, rho_matrix: float) -> float:
    v = filler_volume_fraction
    return v * rho_filler / (v * rho_filler + (1.0 - v) * rho_matrix)


FILLER_ROLES = {ComponentRole.FILLER}


def component_density(
    comp: Component, db: Optional[ConstituentDatabase] = None, state: Literal["uncured", "cured"] = "cured"
) -> float:
    """Density (g/cm^3) of a formulation component.

    A component's own ``density`` (taken as the as-supplied/liquid density) wins over the
    database ``density``.  For monomers in the cured state the database ``density_cured`` is
    used if present, else density / (1 - homopolymer volumetric shrinkage), else the liquid
    density (stated approximation: blend conversion differs from homopolymer conversion).
    """
    db = db or load_constituents()
    c = db.find(comp.name)
    if comp.density is not None:
        base = comp.density.magnitude(Unit.G_PER_CM3)
    elif c is not None and c.has("density"):
        base = c.get("density", Unit.G_PER_CM3)
    else:
        raise KeyError(f"no density for component {comp.name!r}")
    if state == "cured" and comp.role in (ComponentRole.BASE_MONOMER, ComponentRole.DILUENT_MONOMER) and c is not None:
        if c.has("density_cured") and comp.density is None:
            return c.get("density_cured", Unit.G_PER_CM3)  # type: ignore[return-value]
        if c.has("shrinkage_volumetric_homopolymer"):
            s = c.get("shrinkage_volumetric_homopolymer", Unit.DIMENSIONLESS)
            return base / (1.0 - s)  # type: ignore[operator]
    return base  # type: ignore[return-value]


def formulation_volume_fractions(
    formulation: Formulation, db: Optional[ConstituentDatabase] = None, state: Literal["uncured", "cured"] = "cured"
) -> dict[str, float]:
    """Volume fraction of every component (keys = component names)."""
    db = db or load_constituents()
    w = {c.name: c.weight_fraction for c in formulation.components}
    rho = {c.name: component_density(c, db, state) for c in formulation.components}
    return weight_to_volume_fractions(w, rho)


SILANE_DENSITY = (1.2, 1.1, 1.3)  # g/cm^3, ASSUMED cured organosilane (gamma-MPS / 8-MOTS) layer density
_SILANE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*wt\s*%")


def silane_mass_fraction_on_filler(component: Component) -> float:
    """Silane mass as a fraction of the as-weighed (silanised) filler mass, parsed from
    ``surface_treatment`` (e.g. "3 wt% gamma-MPS" -> 0.03).  0 when no wt% is given."""
    t = component.surface_treatment or ""
    m = _SILANE_RE.search(t)
    return float(m.group(1)) / 100.0 if m else 0.0


def filler_volume_fraction(
    formulation: Formulation,
    db: Optional[ConstituentDatabase] = None,
    state: Literal["uncured", "cured"] = "cured",
    basis: Literal["inorganic", "as_weighed"] = "inorganic",
    silane_density: float = SILANE_DENSITY[0],
) -> float:
    """Filler volume fraction.

    basis="inorganic" (default; the ash basis of the HV, E and sorption calibrations): the
    silane mass on a pre-silanised filler (``surface_treatment`` "N wt% ...") is organic - it is
    moved to the organic phase at ``silane_density`` (~1.2 g/cm^3) and only the inorganic core
    counts as filler.  basis="as_weighed": the silanised powder is counted as filler at the glass
    density (the legacy behaviour; over-states phi by ~0.02-0.03 at 3 wt% silane)."""
    vf = formulation_volume_fractions(formulation, db, state)
    if basis == "as_weighed":
        return sum(vf[c.name] for c in formulation.components if c.role in FILLER_ROLES)
    db = db or load_constituents()
    v_fill = v_tot = 0.0
    for c in formulation.components:
        rho = component_density(c, db, state)
        if c.role in FILLER_ROLES:
            s = silane_mass_fraction_on_filler(c)
            v_core = c.weight_fraction * (1.0 - s) / rho
            v_fill += v_core
            v_tot += v_core + c.weight_fraction * s / silane_density
        else:
            v_tot += c.weight_fraction / rho
    return v_fill / v_tot if v_tot > 0 else 0.0


def filler_ash_weight_fraction(formulation: Formulation) -> float:
    """Inorganic (ash, 600-900 C burn-off) filler mass fraction: silane mass excluded."""
    return sum(c.weight_fraction * (1.0 - silane_mass_fraction_on_filler(c)) for c in formulation.components if c.role in FILLER_ROLES)


def filler_weight_fraction(formulation: Formulation) -> float:
    return sum(c.weight_fraction for c in formulation.components if c.role in FILLER_ROLES)


def with_volume_fraction(formulation: Formulation, db: Optional[ConstituentDatabase] = None) -> Formulation:
    """Copy of the formulation with ``filler_volume_fraction`` (cured state) filled in."""
    return formulation.model_copy(update={"filler_volume_fraction": filler_volume_fraction(formulation, db, "cured")})


def matrix_components(formulation: Formulation) -> list[Component]:
    return [c for c in formulation.components if c.role not in FILLER_ROLES]


def monomer_weight_fractions(formulation: Formulation) -> dict[str, float]:
    """Weight fractions of the (base + diluent) monomers normalised to the monomer blend."""
    mons = [c for c in formulation.components if c.role in (ComponentRole.BASE_MONOMER, ComponentRole.DILUENT_MONOMER)]
    tot = sum(c.weight_fraction for c in mons)
    if tot <= 0:
        raise ValueError("formulation has no monomers")
    return {c.name: c.weight_fraction / tot for c in mons}


def filler_weight_fractions(formulation: Formulation) -> dict[str, float]:
    """Weight fractions of each filler normalised to the total filler."""
    fl = [c for c in formulation.components if c.role in FILLER_ROLES]
    tot = sum(c.weight_fraction for c in fl)
    return {c.name: c.weight_fraction / tot for c in fl} if tot > 0 else {}


def make_component(
    name: str, weight_fraction: float, db: Optional[ConstituentDatabase] = None, **overrides
) -> Component:
    """Build a formulation Component from the database (role, CAS, density, n, d50)."""
    db = db or load_constituents()
    c = db.lookup(name)
    kw: dict = dict(name=c.name, role=c.role, weight_fraction=weight_fraction, cas=c.cas)
    if c.has("density"):
        kw["density"] = c.prop("density").value
    if c.has("refractive_index"):
        kw["refractive_index"] = c.get("refractive_index", Unit.DIMENSIONLESS)
    if c.has("d50"):
        kw["particle_d50"] = c.prop("d50").value
    kw["supplier_examples"] = list(c.supplier_examples)
    kw.update(overrides)
    return Component(**kw)
