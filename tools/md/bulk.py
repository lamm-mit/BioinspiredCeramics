"""Bulk elastic moduli of the ceramic of a design, from md_agent with the Pedone potential.

A cross-check of the layer modulus that the design takes from the literature. md_agent (Alex Wang,
md_agent/) relaxes the crystal and computes C_ij by finite strain at 0 K. Its modules import each
other by name, so its folder goes on the path.
"""

import sys
import time
from pathlib import Path

from ..schema import MATERIAL, spec

sys.path.insert(0, str(Path(__file__).parent / "md_agent"))
import ceramics  # noqa: E402


def md_bulk(material):
    """Young, bulk and shear moduli (Voigt-Reuss-Hill) and density of the ceramic, at 0 K.

    Alumina is the 30-atom hexagonal cell of corundum, which is enough for a perfect crystal since the
    Ewald sum runs over periodic images. A call takes a few seconds.
    """
    if material != "alumina":
        raise ValueError(f"md_bulk covers alumina only, not {material}. Silica glass needs a melt-quench of "
                         "about 10 minutes and silicon carbide needs MACE, which is not installed")
    start = time.perf_counter()
    s = ceramics.build_ceramic(ceramics.CeramicSpec(material="Al2O3", potential="classical", supercell=[1, 1, 1]))
    relaxed = ceramics.relax_cell(s)
    e = ceramics.elastic_constants(s)
    return {k: round(float(e[k]), 1) for k in ["youngs_modulus_GPa", "bulk_modulus_GPa", "shear_modulus_GPa"]} | {
        "density_g_cm3": round(float(relaxed["density_g_cm3"]), 3),
        "n_atoms": len(s.atoms),
        "seconds": round(time.perf_counter() - start, 1),
        "method": "md_agent, Pedone 2006 rigid-ion potential, perfect crystal relaxed at 0 K, "
                  "C_ij by finite strain of 0.5% with relaxed ions, Voigt-Reuss-Hill average"}


SPEC = spec("md_bulk",
            "Molecular-statics elastic moduli of the bulk ceramic with the Pedone potential, at 0 K. A cross-check of "
            "the layer modulus of the design. Returns the Young, bulk and shear moduli in GPa and the density.",
            {"material": MATERIAL | {"description": "the ceramic of the design"}})
