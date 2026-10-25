"""The design the agents fill in, and the checks on it.

DESIGN lists the fields the designer takes from the corpus. The first three state the bioinspired
hypothesis, and the rest make it a printable laminate. UNIT says what a passage must
contain to supply a numeric field. SCALE and RANGE tell the source check how to find a numeric
value in its source, and ASK is the question for the second reading of that source.
make_geometry builds the part of a complete design, and CRITERIA is what the judge checks a simulated
design against. cross_checks runs tools that test a design value without entering the simulation.
Everything specific to one design problem lives in this file and in the tools package.
"""

from tools.md.bulk import md_bulk
from tools.schema import MATERIAL, number

from .fabrication import PROCESSES
from .geometry import layered_geometry


def choice(options, description):
    return {"type": "string", "enum": options, "description": description}


DESIGN = {
    "biological_model": choice(["nacre", "conch shell", "bone", "Bouligand"], "the biological material the design copies"),
    "mechanism": choice(["crack deflection", "crack bridging", "compressive residual stress"],
                        "the toughening mechanism of that biological material, as the evidence states"),
    "process": choice(PROCESSES, "the additive manufacturing process that builds the ceramic layers, as the evidence "
                                 "shows for this ceramic"),
    "ceramic": MATERIAL | {"description": "the layer material"},
    "interlayer": MATERIAL | {"description": "the material between layers, chosen so that cracks deflect along it"},
    "layer_modulus_GPa": number("Young's modulus of the ceramic, as stated in the evidence"),
    "layer_thickness_um": number("thickness of one ceramic layer, within the range of layers that the "
                                 "evidence reports as made or tested with the chosen process"),
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
# What the judge decides for each candidate, from its design values and the tool outputs.
CRITERIA = ["A crack deflects along the interfaces.",
            "The design meets every requirement that the question states."]


def design_errors(design):
    """What is wrong with a design as a whole, as field -> reason. The source check looks at one field at a time."""
    # A material the tools do not know would fail at the first tool call, so it is caught here, where the designer can repair it.
    errors = {k: f"{v!r} is not one of the options the tools accept, which are {DESIGN[k]['enum']}."
              for k, v in design.items() if v not in DESIGN.get(k, {}).get("enum", [v])}
    if design.get("interlayer") and design.get("interlayer") == design.get("ceramic"):
        errors["interlayer"] = ("the interlayer is the same material as the ceramic. Choose a different, weaker material "
                                "that the evidence supports.")
    return errors


def make_geometry(design, folder):
    """The STL files and the JSON description of the part a complete design describes."""
    return layered_geometry(design["ceramic"], design["interlayer"], design["layer_thickness_um"],
                            design["interlayer_thickness_um"], folder)


def cross_checks(design):
    """The MD moduli of the bulk ceramic beside the layer modulus the design took from the literature."""
    try:
        return {"md_bulk": md_bulk(design["ceramic"]) | {"design_layer_modulus_GPa": design["layer_modulus_GPa"]}}
    except ValueError as e:  # md_bulk covers alumina only
        return {"md_bulk": {"error": str(e)}}
