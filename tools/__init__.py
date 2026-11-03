"""The tools the engineer agent calls. TOOLS holds the JSON schema of each and RUN the function behind it.

Each tool is a folder with its function, its schema and a README. DFT runs Quantum ESPRESSO, MD runs
MACE-MP-0 with D3, and FEM runs jax-pfczm. DFT and MD keep each result in their results/ folder, since it
depends only on the materials.
"""

from . import dft, fem, md

TOOLS = [dft.SPEC, md.SPEC, fem.SPEC]
RUN = {"dft_cleavage": dft.dft_cleavage, "md_interface": md.md_interface, "fem_fracture": fem.fem_fracture}
