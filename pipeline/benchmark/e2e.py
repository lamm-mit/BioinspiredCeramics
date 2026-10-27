"""Run the whole agent graph REPEATS times on four questions per model set and check each run.

python -m pipeline.benchmark.e2e corpus_folder

A run passes six checks. Every design field is filled. Every field cites a source.
Every value passes the source check of the graph against the evidence it cites.
The geometry step wrote its STL files. fem_fracture returned a result. Its inputs equal the
design values, the geometry file and the outputs of dft_cleavage and md_interface for that
design, so nothing entered the simulation from anywhere else. The checks apply to the last
candidate of the run. Each row also holds the judge's verdicts, the novelty of each candidate,
the best candidate, the size of its recipe and whether its print files exist.
Rows go to benchmark/results/e2e_pipeline.jsonl, and each run keeps its report and state in its
folder under runs/e2e/. CONFIGS holds the default models. The comparison of three model sets in
benchmark/results/e2e.jsonl ran in early October 2026 on the earlier graph, with these sets
"qwen3.6:35b for every agent" = team("qwen3.6:35b", "qwen3.6:35b") and
"gpt-oss:120b + qwen3.6:35b reader" = team("gpt-oss:120b", "qwen3.6:35b", effort="low").
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
from ..novelty import features

here = Path(__file__).parent

# Until 2026-10-25 the questions also asked for a silicon carbide laminate and for an unreported combination.
# From 2026-10-26 they ask for structures that a toolpath pattern carries.
QUESTIONS = [
    "Propose a bioinspired layered ceramic made by additive manufacturing whose interfaces deflect cracks. Choose the "
    "biological model and its mechanism, the ceramic, the interlayer, the print process and the layer thicknesses, and "
    "check whether a crack deflects.",
    "Design a nacre-inspired alumina with staggered tablets made by direct ink writing with a polymer interlayer, and "
    "check whether a crack deflects.",
    "Design a Bouligand alumina made by direct ink writing with weak interlayers, and check whether a crack deflects.",
    "Design a bioinspired alumina laminate whose Young's modulus is graded through the thickness, made by direct ink "
    "writing, and check whether a crack deflects.",
]


def team(text, reader, designer_effort="medium", effort="none"):
    """One text model for the text agents, with its own effort for the designer, the checker and the judge, and one figure reader."""
    return {"planner": (text, effort), "designer": (text, designer_effort), "architect": (text, designer_effort),
            "checker": (text, designer_effort),
            "engineer": (text, effort), "judge": (text, designer_effort), "fabricator": (text, designer_effort),
            "reporter": (text, effort),
            "figure_reader": (reader, "none")}


CONFIGS = {"gpt-oss:20b + qwen3.6:35b reader": MODELS}
REPEATS = 2


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
            "novelty": [round(c["novelty"]["score"], 3) for c in s.get("candidates", [])], "best": s.get("best"),
            "recipe_steps": len(s.get("recipe", {}).get("steps", [])),
            "cited_recipe_values": sum(not str(p.get("source", "")).startswith("assumption") for st in s.get("recipe", {}).get("steps", [])
                                       for p in st["parameters"].values()),
            "print_files": all(Path(p).exists() for p in s.get("fabrication", {}).get("files", {}).values()) if s.get("fabrication") else None,
            "deflects": fea[-1]["output"]["crack_deflects"] if fea else None,
            "crack": fea[-1]["output"].get("outcome") if fea else None,
            "patterns": [c["pattern"] for c in s.get("candidates", [])],
            "pattern_labels": [[x["label"].split(":")[0] for x in c["pattern_sources"].values()] for c in s.get("candidates", [])],
            "calls": [r["tool"] + (" failed" if "error" in r["output"] else "") for r in results],
            "proposals": [p["design"] | {"structure": p["structure"]} for p in s.get("proposals", [])],
            "built_structure": [any(all(c["design"].get(k) == v for k, v in p["design"].items()) and p["structure"] in features(c["pattern"])
                                    for p in s.get("proposals", [])) for c in s.get("candidates", [])],
            "took_proposal": [any(all(c["design"].get(k) == v for k, v in p["design"].items()) for p in s.get("proposals", []))
                              for c in s.get("candidates", [])]}


if __name__ == "__main__":
    c = corpus.load(sys.argv[1], EMBEDDING)
    (here / "results").mkdir(exist_ok=True)
    for name, models in CONFIGS.items():
        app = build(models, c)
        for repeat in range(REPEATS):
            for q in QUESTIONS:
                start = time.perf_counter()
                folder = Path("runs") / "e2e" / time.strftime("%Y%m%d-%H%M%S")
                try:
                    s = app.invoke({"question": q, "folder": str(folder)}, {"recursion_limit": 100})
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "state.json").write_text(json.dumps(s, indent=1, default=str))
                    (folder / "report.md").write_text(s["report"])
                    row = checks(s)
                except Exception as e:  # a failed run counts, with its error
                    row = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
                row |= {"config": name, "repeat": repeat, "question": q, "folder": str(folder),
                        "seconds": round(time.perf_counter() - start)}
                with open(here / "results" / "e2e_pipeline.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(row, flush=True)
