"""Mesh check of the seven toolpath patterns of patterns.py, at an energy ratio of 0.2. python -m pipeline.docs.figures.mesh ROWS [fixed]

The AT2 length is 2 pixels, so a finer grid also shortens it. With fixed, the length stays at the 2 pixels of the
60-row grid, 16.3 µm for these cells, and only the pixel shrinks. Each pattern gives its outcome, the applied K
when the outcome was decided and the energy balance, in runs/mesh/<ROWS>[_fixed].json.
"""

import csv
import json
import os
import sys
from pathlib import Path

rows, fixed = int(sys.argv[1]), sys.argv[2:] == ["fixed"]
os.environ["FEM_ROWS"] = str(rows)

from pipeline.docs.figures.patterns import DESIGN, LAYER_ENERGY, PATTERNS
from pipeline.geometry import layered_geometry
from tools.fem import cell, fem_fracture

if fixed:
    cell.LENGTH_PIXELS = 2.0 * rows / 60
tag = f"{rows}_fixed" if fixed else str(rows)
work = Path("runs") / "mesh"
work.mkdir(parents=True, exist_ok=True)
cache = work / f"{tag}.json"
results = json.loads(cache.read_text()) if cache.exists() else {}
for i, (name, pattern) in enumerate(PATTERNS.items()):
    if name in results:
        continue
    g = layered_geometry(DESIGN, pattern, work / f"{i}_{tag}")["geometry_json"]
    out = fem_fracture(DESIGN["layer_modulus_GPa"], DESIGN["layer_thickness_um"], LAYER_ENERGY, 0.2 * LAYER_ENERGY, 100, g)
    steps = list(csv.DictReader(open(out["load_path_csv"])))
    results[name] = {k: out[k] for k in ["outcome", "nearest_joint_um", "along_interface_left_um", "along_interface_right_um",
                                         "into_next_layer_um", "energy_balance", "converged", "pixel_um", "seconds", "damage_png"]}
    results[name] |= {"K_over_K_layer": float(steps[-1]["K_over_K_layer"]), "steps": len(steps), "length_um": cell.LENGTH_PIXELS * out["pixel_um"]}
    cache.write_text(json.dumps(results, indent=1))
    print(tag, name, out["outcome"], round(results[name]["K_over_K_layer"], 2), out["seconds"], "s", flush=True)
