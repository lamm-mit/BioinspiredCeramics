"""Separation of the interface between the ceramic and the interlayer, by molecular dynamics (interface.py).

The result depends only on the pair of materials, so each pair is computed once and kept in results/, which later
calls read. A new pair runs interface.py in the environment that has MACE (SIMTOOLS_PYTHON, by default
~/simtools/env/bin/python) and takes 16 to 101 min on the GPU of the Spark.
"""

import json
import os
import subprocess
from pathlib import Path

from ..schema import CERAMIC, INTERLAYER, spec

HERE = Path(__file__).parent


def md_interface(material_a, material_b):
    """Work of separation and strength of the ceramic `material_a` under the interlayer `material_b`, at the weaker of
    the interface and the middle of the interlayer, where a crack between two layers runs."""
    if material_a not in CERAMIC["enum"] or material_b not in INTERLAYER["enum"]:
        raise ValueError(f"md_interface takes a ceramic of {CERAMIC['enum']} and an interlayer of {INTERLAYER['enum']}, "
                         f"not {material_a} and {material_b}")
    cached = HERE / "results" / f"{material_a.replace(' ', '_')}__{material_b.replace(' ', '_')}.json"
    if not cached.exists():
        python = os.environ.get("SIMTOOLS_PYTHON", str(Path.home() / "simtools/env/bin/python"))
        subprocess.run([python, "-m", "tools.md.interface", material_a, material_b], cwd=HERE.parent.parent,
                       check=True, capture_output=True, text=True)
    return json.loads(cached.read_text())


SPEC = spec("md_interface",
            "Molecular dynamics separation of the ceramic and the interlayer, prepared at 300 K and pulled apart at "
            "0 K, at the interface or inside the interlayer, whichever is weaker. Returns the separation energy in "
            "J/m^2 and the strength in MPa.",
            {"material_a": CERAMIC | {"description": "the ceramic of the design"},
             "material_b": INTERLAYER | {"description": "the interlayer of the design"}})
