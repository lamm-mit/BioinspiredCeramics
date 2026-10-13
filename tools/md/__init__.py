"""Molecular dynamics of the interface between two materials. A stand-in until the MD tool is attached."""

from ..schema import MATERIAL, SURFACE_ENERGY, spec


def md_interface(material_a, material_b):
    """Stand-in for a molecular-dynamics separation of the interface between two materials at 300 K.

    It assumes perfect adhesion, so the interface opens inside the weaker material with work
    W = 2g of that material. The strength is the peak of the universal binding-energy relation
    T(d) = W d / l^2 exp(-d / l), which is W / (e l), with l = 0.05 nm. The real tool builds
    the interface, equilibrates it and pulls it apart.
    """
    w = 2 * min(SURFACE_ENERGY[material_a], SURFACE_ENERGY[material_b])
    return {"separation_energy_J_m2": w, "strength_MPa": round(w / (2.718281828 * 0.05e-9) / 1e6),
            "method": "stand-in (weaker-material cleavage and binding-energy relation), not an MD result"}


SPEC = spec("md_interface",
            "Molecular dynamics separation of the interface between the ceramic and the interlayer at 300 K. "
            "Returns the separation energy in J/m^2 and the strength in MPa.",
            {"material_a": MATERIAL | {"description": "the ceramic of the design"},
             "material_b": MATERIAL | {"description": "the interlayer of the design"}})
