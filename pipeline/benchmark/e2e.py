"""Run the whole agent graph on four questions per model set and check each run.

python -m pipeline.benchmark.e2e corpus_folder

A run passes six checks. Every design field is filled. Every field cites a source.
Every value passes the source check of the graph against the evidence it cites.
The geometry step wrote its STL files. fem_fracture returned a result. Its inputs equal the
design values, the geometry file and the outputs of dft_cleavage and md_interface for that
design, so nothing entered the simulation from anywhere else. The checks apply to the last
candidate of the run. Each row also holds the judge's verdict and the number of candidates.
Rows go to benchmark/results/e2e.jsonl.
"""

import json
import sys
import time
from pathlib import Path

from tools.dft import dft_cleavage
from tools.md import md_interface

from .. import corpus
from ..design import DESIGN
from ..graph import EMBEDDING, MODELS, build, source_errors

here = Path(__file__).parent

QUESTIONS = [
    "Design a nacre-inspired layered alumina whose interfaces deflect cracks. Choose an interlayer and a "
    "layer thickness, and check whether a crack deflects.",
    "Propose a laminated alumina with a polymer interlayer for crack deflection, with layers as thin as the "
    "printing literature supports, and check whether a crack deflects.",
    "Would a silica glass interlayer between alumina layers deflect cracks? Use the layer thickness of "
    "printed alumina filaments.",
    "Design a layered alumina with graphite interlayers, with layers as thin as tape casting allows, and "
    "check whether a crack deflects.",
]


def team(text, reader, designer_effort="medium", effort="none"):
    """One text model for the text agents, with its own effort for the designer, the checker and the judge, and one figure reader."""
    return {"planner": (text, effort), "designer": (text, designer_effort), "checker": (text, designer_effort),
            "engineer": (text, effort), "judge": (text, designer_effort), "reporter": (text, effort),
            "figure_reader": (reader, "none")}


CONFIGS = {"qwen3.6:35b for every agent": team("qwen3.6:35b", "qwen3.6:35b"),
           "gpt-oss:20b + qwen3.6:35b reader": MODELS,
           "gpt-oss:120b + qwen3.6:35b reader": team("gpt-oss:120b", "qwen3.6:35b", effort="low")}


def checks(s):
    design, results, made = s["design"], s.get("results", []), s.get("geometry", {})
    fea = [r for r in results if r["tool"] == "fem_fracture" and "error" not in r["output"]]
    geometry = bool(made) and all(Path(made[k]).exists() for k in made if k.endswith(("_stl", "_json")))
    faithful = False
    if fea and all(k in design for k in DESIGN):
        layer = dft_cleavage(design["ceramic"])
        interface = md_interface(design["ceramic"], design["interlayer"])
        expected = {"layer_modulus_GPa": design.get("layer_modulus_GPa"),
                    "layer_thickness_um": design.get("layer_thickness_um"),
                    "layer_fracture_energy_J_m2": layer["cleavage_energy_J_m2"],
                    "interface_fracture_energy_J_m2": interface["separation_energy_J_m2"],
                    "interface_strength_MPa": interface["strength_MPa"],
                    "geometry_json": made["geometry_json"]}
        faithful = all(fea[-1]["arguments"].get(k) == v for k, v in expected.items())
    return {"complete": all(k in design for k in DESIGN), "cited": all(s["sources"].get(k) for k in DESIGN),
            "grounded": not source_errors(design, s["sources"], s), "rejected": s.get("rejected", {}),
            "geometry": geometry, "simulated": bool(fea), "faithful": faithful, "design": design,
            "verdicts": [c["verdict"]["verdict"] for c in s.get("candidates", [])], "outcome": s.get("outcome"),
            "deflects": fea[-1]["output"]["crack_deflects"] if fea else None,
            "calls": [r["tool"] + (" failed" if "error" in r["output"] else "") for r in results]}


c = corpus.load(sys.argv[1], EMBEDDING)
for name, models in CONFIGS.items():
    app = build(models, c)
    for q in QUESTIONS:
        start = time.perf_counter()
        try:
            folder = Path("runs") / "e2e" / time.strftime("%Y%m%d-%H%M%S")
            row = checks(app.invoke({"question": q, "folder": str(folder)}, {"recursion_limit": 100}))
        except Exception as e:  # a failed run counts, with its error
            row = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        row |= {"config": name, "question": q, "seconds": round(time.perf_counter() - start)}
        with open(here / "results" / "e2e.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)
