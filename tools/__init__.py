"""The tools the engineer agent calls. TOOLS holds the JSON schema of each and RUN the function behind it.

Each tool is a folder with its function, its schema and a README. All three functions are closed-form
stand-ins that let the pipeline run end to end without the solvers. To attach a solver, replace the
function body in its folder and keep the names of the inputs and outputs.
"""

from . import dft, fem, md

TOOLS = [dft.SPEC, md.SPEC, fem.SPEC]
RUN = {"dft_cleavage": dft.dft_cleavage, "md_interface": md.md_interface, "fem_fracture": fem.fem_fracture}
