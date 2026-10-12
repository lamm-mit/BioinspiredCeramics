"""The tools the engineer agent calls. TOOLS holds the JSON schema of each and RUN the function behind it."""

from pipeline.geometry import layered_geometry

from . import fem, md
from .schema import MATERIAL, number, spec

TOOLS = [md.SPEC, fem.SPEC,
         spec("layered_geometry",
              "Geometry of the layered design. Writes one STL per material for a test bar to print and for a small "
              "notched cell to simulate, and a JSON description of both. Returns the file paths.",
              {"ceramic": MATERIAL, "interlayer": MATERIAL, "layer_thickness_um": number("from the design"),
               "interlayer_thickness_um": number("from the design")})]
RUN = {"md_interface": md.md_interface, "crack_fea": fem.crack_fea, "layered_geometry": layered_geometry}
