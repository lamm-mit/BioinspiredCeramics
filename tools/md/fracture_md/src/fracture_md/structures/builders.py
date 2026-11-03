"""Reference structure builders."""

from __future__ import annotations

from ase import Atoms
from ase.build import bulk

from fracture_md.schema.material import Composition, Phase

# Experimental room-temperature lattice parameter of 3C-SiC (for starting
# geometries only; the relaxed value is always recomputed with the potential).
A_3C_SIC_EXP = 4.3596  # Å


def sic_3c_conventional(a: float = A_3C_SIC_EXP) -> Atoms:
    """8-atom cubic conventional cell of zincblende 3C-SiC (F-43m, #216)."""
    return bulk("SiC", "zincblende", a=a, cubic=True)


def sic_3c_primitive(a: float = A_3C_SIC_EXP) -> Atoms:
    """2-atom FCC primitive cell of 3C-SiC."""
    return bulk("SiC", "zincblende", a=a)


def sic_3c_phase() -> Phase:
    return Phase(
        name="3C-SiC",
        prototype="zincblende (B3)",
        space_group_symbol="F-43m",
        space_group_number=216,
        polytype="3C",
        crystal_system="cubic",
        external_ids={"materials_project": "mp-8062"},
    )


def sic_composition() -> Composition:
    return Composition(formula="SiC", elements={"Si": 1.0, "C": 1.0})
