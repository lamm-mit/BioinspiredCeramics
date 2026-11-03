"""Post-processing of atomistic snapshots on the host (numpy + scipy; no JAX).

* ``local_strain``: per-atom coordination (potential-aware bond rules), displacement, von Mises local shear
  strain and surface height maps, used by the indentation animation and its engine cross-checks.
"""

from fracture_md.analysis.local_strain import (
    ALL_PAIRS_2P35,
    LAMMPS_COORD_2P5,
    SIC_HETERONUCLEAR_2P4,
    VASHISHTA_SIC_BONDS,
    BondRule,
    ShearStrain,
    bond_rule_for,
    coordination,
    coordination_by_rule,
    displacement,
    surface_height_map,
    top_layer_atoms,
    von_mises_shear,
)

__all__ = [
    "ALL_PAIRS_2P35",
    "LAMMPS_COORD_2P5",
    "SIC_HETERONUCLEAR_2P4",
    "VASHISHTA_SIC_BONDS",
    "BondRule",
    "ShearStrain",
    "bond_rule_for",
    "coordination",
    "coordination_by_rule",
    "displacement",
    "surface_height_map",
    "top_layer_atoms",
    "von_mises_shear",
]
