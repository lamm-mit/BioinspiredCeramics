"""The design the agents fill in, and the checks on it.

DESIGN lists the fields the designer takes from the corpus, and UNIT says what a passage must
contain to supply a numeric field. SCALE and RANGE tell the source check how to find a numeric
value in its source, and ASK is the question for the second reading of that source.
make_geometry builds the part of a complete design.
Everything specific to one design problem lives in this file and in the tools package.
"""

from tools.schema import MATERIAL, number

from .geometry import layered_geometry

DESIGN = {
    "ceramic": MATERIAL | {"description": "the layer material"},
    "interlayer": MATERIAL | {"description": "the material between layers, chosen so that cracks deflect along it"},
    "layer_modulus_GPa": number("Young's modulus of the ceramic, as stated in the evidence"),
    "layer_thickness_um": number("thickness of one ceramic layer, within the range of layers that the "
                                 "evidence reports as made or tested"),
    "interlayer_thickness_um": number("thickness of the interlayer between two ceramic layers, within the range "
                                      "that the evidence reports as made or tested"),
}
# A passage can supply a numeric field only if it holds a number with the field's unit. Converted
# papers often write the unit in LaTeX, as in $\\sim 35~\\mu m$ or $367$ GPa, so both spellings count.
MICRON = r"\d[\s$~}]*(?:\\mu\s*m|[µμu]m)\b"
# Most passages with a length in µm are about something else, so the interlayer thickness also
# needs a word for the interlayer in the same sentence as the number.
INTERLAYER = r"(?:interlayer|interphase|weak layer)"
UNIT = {"layer_modulus_GPa": r"\d[\s$~}]*GPa", "layer_thickness_um": MICRON,
        "interlayer_thickness_um": rf"(?is){INTERLAYER}[^.]{{0,120}}?{MICRON}|{MICRON}[^.]{{0,80}}?{INTERLAYER}"}
# The units a source may use for a numeric field, each with its factor to the field's own unit.
LENGTH = {"µm": 1, "μm": 1, "um": 1, "microns": 1, "micron": 1, "mm": 1e3, "nm": 1e-3}
SCALE = {"layer_modulus_GPa": {"GPa": 1, "MPa": 1e-3}, "layer_thickness_um": LENGTH, "interlayer_thickness_um": LENGTH}
# A numeric field that the designer may choose anywhere within the range its source reports.
RANGE = {"layer_thickness_um", "interlayer_thickness_um"}
# What the checker is asked about a numeric field when it reads the cited source a second time.
# The braces take values of the design, so the question names the material.
ASK = {"layer_modulus_GPa": "Young's modulus of {ceramic} itself. A composite or another material does not count.",
       "layer_thickness_um": "thickness of one {ceramic} layer, such as a printed layer or a layer of a laminate. "
                             "An interlayer, a pore, a grain, a coating or a layer of another material does not count.",
       "interlayer_thickness_um": "thickness of the {interlayer} interlayer or weak layer between two {ceramic} layers. "
                                  "A {ceramic} layer, a pore or a grain does not count."}


def design_errors(design):
    """What is wrong with a design as a whole, as field -> reason. The source check looks at one field at a time."""
    if design.get("interlayer") and design.get("interlayer") == design.get("ceramic"):
        return {"interlayer": "the interlayer is the same material as the ceramic. Choose a different, weaker material "
                              "that the evidence supports."}
    return {}


def make_geometry(design, folder):
    """The STL files and the JSON description of the part a complete design describes."""
    return layered_geometry(design["ceramic"], design["interlayer"], design["layer_thickness_um"],
                            design["interlayer_thickness_um"], folder)
