"""The tools the engineer agent calls. TOOLS holds the JSON schema of each and RUN the function behind it.

Each tool is a folder with its function, its schema and a README. All three functions are closed-form
stand-ins that let the pipeline run end to end without the solvers. To attach a solver, replace the
function body in its folder and keep the names of the inputs and outputs.
"""

from pipeline.geometry import layered_geometry

from . import dft, fem, md
from .schema import MATERIAL, number, spec

TOOLS = [dft.SPEC, md.SPEC, fem.SPEC,
         spec("layered_geometry",
              "Geometry of the layered design. Writes one STL per material for a test bar to print and for a small "
              "notched cell to simulate, and a JSON description of both. Returns the file paths.",
              {"ceramic": MATERIAL, "interlayer": MATERIAL, "layer_thickness_um": number("from the design"),
               "interlayer_thickness_um": number("from the design")})]
RUN = {"dft_cleavage": dft.dft_cleavage, "md_interface": md.md_interface, "fem_fracture": fem.fem_fracture,
       "layered_geometry": layered_geometry}
