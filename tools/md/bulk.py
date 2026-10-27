"""Bulk elastic moduli of the ceramic of a design, from md_agent.

A cross-check of the layer modulus that the design takes from the literature. md_agent (Alex Wang,
md_agent/) relaxes the crystal and computes C_ij by finite strain at 0 K. Alumina uses the Pedone potential in the
pipeline's own environment. Silicon carbide needs MACE-MP-0, so it runs in the environment that has MACE
(SIMTOOLS_PYTHON, by default ~/simtools/env/bin/python). md_agent's modules import each other by name, so its folder
goes on the path.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..schema import CERAMIC, spec

sys.path.insert(0, str(Path(__file__).parent / "md_agent"))
import ceramics  # noqa: E402

# The md_agent crystal and potential of each ceramic.
SETUP = {"alumina": ("Al2O3", "classical", "Pedone 2006 rigid-ion potential"),
         "silicon carbide": ("SiC_3C", "mace", "MACE-MP-0 medium, float64")}


def md_bulk(material):
    """Young, bulk and shear moduli (Voigt-Reuss-Hill) and density of the ceramic, at 0 K, in a few seconds.

    Alumina is the 30-atom hexagonal cell of corundum, which is enough for a perfect crystal since the Ewald sum runs
    over periodic images. Silicon carbide is the 8-atom cubic cell.
    """
    if material not in SETUP:
        raise ValueError(f"md_bulk covers {list(SETUP)}, not {material}")
    key, potential, name = SETUP[material]
    if potential == "mace" and not ceramics.mace_available():
        python = os.environ.get("SIMTOOLS_PYTHON", str(Path.home() / "simtools/env/bin/python"))
        out = subprocess.run([python, "-m", "tools.md.bulk", material], cwd=Path(__file__).parent.parent.parent,
                             check=True, capture_output=True, text=True)
        return json.loads(out.stdout.strip().splitlines()[-1])
    os.environ.setdefault("MD_AGENT_MACE_DTYPE", "float64")
    start = time.perf_counter()
    s = ceramics.build_ceramic(ceramics.CeramicSpec(material=key, potential=potential, supercell=[1, 1, 1],
                                                    mace_model="medium"))
    relaxed = ceramics.relax_cell(s)
    e = ceramics.elastic_constants(s)
    return {k: round(float(e[k]), 1) for k in ["youngs_modulus_GPa", "bulk_modulus_GPa", "shear_modulus_GPa"]} | {
        "density_g_cm3": round(float(relaxed["density_g_cm3"]), 3),
        "n_atoms": len(s.atoms),
        "seconds": round(time.perf_counter() - start, 1),
        "method": f"md_agent, {name}, perfect crystal relaxed at 0 K, "
                  "C_ij by finite strain of 0.5% with relaxed ions, Voigt-Reuss-Hill average"}


SPEC = spec("md_bulk",
            "Molecular-statics elastic moduli of the bulk ceramic, at 0 K. A cross-check of the layer modulus of the "
            "design. Returns the Young, bulk and shear moduli in GPa and the density.",
            {"material": CERAMIC | {"description": "the ceramic of the design"}})

if __name__ == "__main__":
    print(json.dumps(md_bulk(sys.argv[1])))
