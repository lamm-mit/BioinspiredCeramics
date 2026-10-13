"""Density-functional energy to cleave a material. A stand-in until the DFT tool is attached."""

from ..schema import MATERIAL, SURFACE_ENERGY, spec


def dft_cleavage(material):
    """Stand-in for a DFT calculation of the energy to cleave a bulk material.

    Cleaving creates two surfaces, so the work is twice the surface energy, taken here from a
    table. The real tool relaxes the bulk cell and the cleaved slab and returns the energy
    difference per area.
    """
    return {"cleavage_energy_J_m2": 2 * SURFACE_ENERGY[material],
            "method": "stand-in (table of surface energies), not a DFT result"}


SPEC = spec("dft_cleavage",
            "Density-functional energy to cleave a bulk material. Returns the cleavage energy in J/m^2.",
            {"material": MATERIAL | {"description": "the ceramic of the design"}})
