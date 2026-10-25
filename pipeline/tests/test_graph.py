# Tests for the agent graph, with the models and the search stubbed. python -m pipeline.tests.test_graph
import itertools
import json
import os
import shutil
from pathlib import Path

from pipeline import graph

os.environ.setdefault("FEM_ROWS", "30")  # the crack tool runs on a coarse grid, a few seconds per call

shutil.rmtree("runs_test/graph", ignore_errors=True)
folders = (f"runs_test/graph/{n}" for n in itertools.count(1))  # a fresh run folder per case

PASSAGES = [{"id": "p#6", "source": "p", "text": r"alumina layers of 210 µm, $E_{\rm A} \sim 367$  GPa"},
            {"id": "p#9", "source": "p", "text": "graphite interlayers 20 µm thick deflect cracks in alumina"}]
graph.cp.search = lambda corpus, q, k, figures=False, skip=(), pattern=None: [] if figures else [p for p in PASSAGES if p["id"] not in skip]
graph.cp.neighbors = lambda corpus, entities, k=3: []
graph.cross_checks = lambda design: {"md_bulk": {"youngs_modulus_GPa": 347.8, "method": "stub"}}
NOVELTY = {}  # interlayer -> the novelty score the stub returns, 0.9 when absent
graph.nv.novelty = lambda design, corpus, embed: {"statement": f"A {design['biological_model']}-inspired laminate.",
    "score": NOVELTY.get(design["interlayer"], 0.9), "combination": 1.0, "semantic": 0.8, "terms": [], "max_terms_together": 1,
    "records_with_max": 3, "closest_records": ["p#6"], "nearest_passage": "p#9", "nearest_similarity": 0.4, "method": "stub"}
# The fabricator cites F1 (p#6, which states 210) and F2 (p#9, which states 20), and one number neither states.
# A nozzle of 210 um keeps the G-code small.
RECIPE = {"steps": [{"stage": "feedstock", "instruction": "Mix the alumina powder into an aqueous ink.",
                     "parameters": {"particle_size_um": {"value": 20, "unit": "um", "source": "F2"},
                                    "binder_wt_percent": {"value": 2, "unit": "wt%", "source": "assumption: typical"}}}],
          "print_parameters": {"nozzle_diameter_um": {"value": 210, "unit": "um", "source": "F1: nozzle"},
                               "print_speed_mm_s": {"value": 7, "unit": "mm/s", "source": "F1"}},
          "cautions": ["Dry slowly."]}
REPORTED = []  # what the reporter was given
TAMPER = {}  # tool inputs the scripted engineer changes on its own, which the handoff file must flag


def fake_chat(model, messages, tools=None, effort=None):
    """A scripted engineer. It calls dft_cleavage and md_interface, then fem_fracture, from the design, the geometry and the results."""
    rec = {"model": model, "seconds": 0, "reply": {}}
    if not tools:
        REPORTED.append(messages[1]["content"])
        return {"content": "report"}, rec
    given = json.loads(messages[1]["content"])
    design = given["design"]
    done = [json.loads(m["content"]) for m in messages if m.get("role") == "tool"]
    if any("error" in d for d in done):
        return {"content": "A tool failed."}, rec
    call = lambda n, name, args: {"id": f"c{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args | {k: v for k, v in TAMPER.items() if k in args})}}
    if len(done) == 0:
        calls = [call(0, "dft_cleavage", {"material": design["ceramic"]}),
                 call(1, "md_interface", {"material_a": design["ceramic"], "material_b": design["interlayer"]})]
    elif len(done) == 2:
        calls = [call(2, "fem_fracture", {"layer_modulus_GPa": design["layer_modulus_GPa"], "layer_thickness_um": design["layer_thickness_um"],
                                          "layer_fracture_energy_J_m2": done[0]["cleavage_energy_J_m2"],
                                          "interface_fracture_energy_J_m2": done[1]["separation_energy_J_m2"],
                                          "interface_strength_MPa": done[1]["strength_MPa"],
                                          "geometry_json": given["geometry"]["geometry_json"]})]
    else:
        return {"content": "All tools ran."}, rec
    return {"role": "assistant", "content": "", "tool_calls": calls}, rec


graph.chat = fake_chat
MODELS = {a: ("m", "medium" if a == "designer" else "low") for a in graph.MODELS}
HYPOTHESIS = {"biological_model": "nacre", "mechanism": "crack deflection", "process": "direct ink writing"}
GOOD = HYPOTHESIS | {"ceramic": "alumina", "interlayer": "graphite", "layer_modulus_GPa": 367, "layer_thickness_um": 210,
                     "interlayer_thickness_um": 20}
SRC = {k: "p#9: nacre-like interlayers" for k in HYPOTHESIS} | {"ceramic": "question: asked", "interlayer": "p#9: deflects",
       "layer_modulus_GPa": "p#6: E_A", "layer_thickness_um": "p#6: layers", "interlayer_thickness_um": "p#9: 20 µm thick"}


asked = []
accept_all = lambda text: {"low": 0, "high": 1e9, "unit": ""}
ACCEPT = {"criteria": [{"criterion": "A crack deflects along the interfaces.", "met": True, "evidence": "crack_deflects is true"}],
          "change": {}, "summary": "The crack deflects."}
REVISE = {"criteria": [{"criterion": "A crack deflects along the interfaces.", "met": False, "evidence": "energy_ratio is 1.0"}],
          "change": {"interlayer": "choose a weaker interlayer", "not_a_field": "ignored"}, "summary": "The crack does not deflect."}
VERDICTS = []  # what the scripted judge answers, one per candidate. It accepts when the list is empty.
judged = []    # what the judge was shown


def run(script, truncate_first=False, checker=accept_all):
    """Run the graph with the designer answering from the script, one answer per call."""
    asked.clear()
    judged.clear()
    calls, answers = [], iter(script)
    def fake(model, system, text, images=(), effort=None):
        if system is graph.PLANNER:
            return {"queries": ["q"], "entities": []}, {"model": model, "seconds": 0, "reply": {}}
        if system is graph.CHECKER:
            asked.append(text.split("\n")[0])
            return checker(text), {"model": model, "seconds": 0, "reply": {}}
        if system is graph.FABRICATOR:
            return RECIPE, {"model": model, "seconds": 0, "reply": {}}
        if system is graph.JUDGE:
            judged.append(text)
            return (VERDICTS.pop(0) if VERDICTS else ACCEPT), {"model": model, "seconds": 0, "reply": {}}
        calls.append((effort, "rejected these fields" in text))
        if truncate_first and len(calls) == 1:
            raise ValueError("m returned no answer (finish: length, 8192 tokens)")
        return next(answers), {"model": model, "seconds": 0, "reply": {}}
    graph.ask_json = fake
    s = graph.build(MODELS, {}).invoke({"question": "Design a layered alumina.", "folder": next(folders)}, {"recursion_limit": 100})
    return s, calls


engineer = lambda s: any(t["agent"] == "engineer" for t in s["trace"])  # the stub engineer calls no tool
bad = 0
def check(name, ok, detail):
    global bad
    bad += not ok
    print("ok  " if ok else "FAIL", name, "|", detail)

s, calls = run([{"design": GOOD, "sources": SRC}])
check("clean answer, one designer call, tools run", s["design"] == GOOD and not s["rejected"] and len(calls) == 1 and engineer(s), (calls, engineer(s)))

s, calls = run([{"design": GOOD | {"layer_modulus_GPa": 380}, "sources": SRC}, {"design": {"layer_modulus_GPa": 367}, "sources": {"layer_modulus_GPa": "p#6: E_A"}}])
check("repair answers only the rejected field, the rest is kept", s["design"] == GOOD and not s["rejected"] and calls == [("medium", False), ("medium", True)] and engineer(s), (s["design"], calls))

s, calls = run([{"design": GOOD | {"layer_modulus_GPa": 380}, "sources": SRC}, {"design": {k: v for k, v in GOOD.items() if k != "layer_modulus_GPa"}, "sources": SRC}] * 2)
check("repair leaves the field out, run stops before the tools with a reason", "layer_modulus_GPa" in s["missing"] and "layer_modulus_GPa" in s["rejected"] and not engineer(s) and len(calls) == 4, (s["missing"], s["rejected"], calls, s["round"]))

s, calls = run([{"design": GOOD | {"layer_modulus_GPa": 380}, "sources": SRC}, {"sources": {}}] * 2)
check("repair reply with no design key keeps the passing fields", set(s["design"]) == set(GOOD) - {"layer_modulus_GPa"} and "layer_modulus_GPa" in s["rejected"], (s["design"], s["rejected"]))

s, calls = run([{"design": GOOD, "sources": SRC}], truncate_first=True)
check("no answer at medium effort, one retry at low effort", calls[:2] == [("medium", False), ("low", False)] and s["design"] == GOOD and any(t.get("retried") for t in s["trace"]), calls)

s, calls = run([{"design": GOOD, "sources": SRC | {"layer_thickness_um": "p#60: layers"}}, {"design": GOOD, "sources": SRC}])
check("wrong passage id repaired", s["design"] == GOOD and not s["rejected"] and len(calls) == 2, calls)
LABELS = {"ceramic": "question: asked", "interlayer": "E2: deflects", "layer_modulus_GPa": "E1: E_A", "layer_thickness_um": "[E1] layers",
          "interlayer_thickness_um": "E2: 20 µm thick"} | {k: "E2: nacre-like interlayers" for k in HYPOTHESIS}
s, calls = run([{"design": GOOD, "sources": LABELS}])
check("labels are stored as the ids they stand for", s["design"] == GOOD and s["sources"]["layer_modulus_GPa"] == "p#6: E_A" and s["sources"]["interlayer"] == "p#9: deflects" and s["sources"]["layer_thickness_um"] == "[p#6] layers", s["sources"])

prompts = []
def spy(script):
    answers = iter(script)
    def fake(model, system, text, images=(), effort=None):
        if system is graph.PLANNER:
            return {"queries": ["q"], "entities": []}, {"model": model, "seconds": 0, "reply": {}}
        if system is graph.CHECKER:
            return {"low": 0, "high": 1e9}, {"model": model, "seconds": 0, "reply": {}}
        if system is graph.FABRICATOR:
            return RECIPE, {"model": model, "seconds": 0, "reply": {}}
        if system is graph.JUDGE:
            return ACCEPT, {"model": model, "seconds": 0, "reply": {}}
        prompts.append(text)
        return next(answers), {"model": model, "seconds": 0, "reply": {}}
    graph.ask_json = fake
    return graph.build(MODELS, {}).invoke({"question": "Design a layered alumina.", "folder": next(folders)}, {"recursion_limit": 100})
s = spy([{"design": GOOD | {"layer_modulus_GPa": 380}, "sources": LABELS}, {"design": {"layer_modulus_GPa": 367}, "sources": {"layer_modulus_GPa": "E1: E_A"}}])
check("the designer reads labels, and the repair message names them", "[E1] alumina layers" in prompts[0] and "p#6" not in prompts[0] and "E1 gives 367 GPa, not 380" in prompts[1] and s["design"] == GOOD, prompts[1][-260:])

s, calls = run([{"design": GOOD, "sources": LABELS | {"layer_modulus_GPa": "E99: none"}}, {"design": GOOD, "sources": LABELS}])
check("a label out of range is rejected, then repaired", s["design"] == GOOD and len(calls) == 2, calls)
# the second reading of a cited source
alumina_layers = lambda text: ({"low": 20, "high": 20} if "weak layer" in text else {"low": 210, "high": 220} if "thickness" in text
                                else {"low": 367, "high": 367})
s, calls = run([{"design": GOOD | {"layer_thickness_um": 140}, "sources": SRC}, {"design": {"layer_thickness_um": 210}, "sources": {"layer_thickness_um": "p#6: A layers"}}],
               checker=lambda text: alumina_layers(text))
PASS140 = [{"id": "p#6", "source": "p", "text": r"ZTA/A layers of 140 µm/210 µm, $E_{\rm A} \sim 367$  GPa"}, PASSAGES[1]]
check("source check alone would pass a value from the wrong layer", True, "see next")
PASSAGES[:] = PASS140
s, calls = run([{"design": GOOD | {"layer_thickness_um": 140}, "sources": SRC}, {"design": {"layer_thickness_um": 210}, "sources": {"layer_thickness_um": "p#6: A layers"}}],
               checker=alumina_layers)
check("second reading rejects 140, the repair gives 210", s["design"] == GOOD and not s["rejected"] and len(calls) == 2 and calls[1][1], (s["design"], calls))
check("the question names the material, one reading per field and source", sorted(asked) == sorted({a for a in asked}) and any("alumina" in a and "thickness of one" in a for a in asked) and len(asked) == 3, asked)

s, calls = run([{"design": GOOD | {"layer_thickness_um": 140}, "sources": SRC}] * 4, checker=lambda text: {"low": None, "high": None} if "thickness of one" in text else alumina_layers(text))
check("second reading finds no value, the field is dropped and no tool runs", "layer_thickness_um" in s["missing"] and "second reading" in s["rejected"].get("layer_thickness_um", "") and not engineer(s), s["rejected"])

s, calls = run([{"design": GOOD | {"layer_modulus_GPa": 380}, "sources": SRC}, {"design": {"layer_modulus_GPa": 367}, "sources": {"layer_modulus_GPa": "p#6: E_A"}}], checker=alumina_layers)
check("a field that fails the source check is not sent to the checker", sum("modulus" in a for a in asked) == 1 and s["design"] == GOOD, asked)

s, calls = run([{"design": {k: v for k, v in GOOD.items() if k != "ceramic"}, "sources": SRC}] * 4, checker=alumina_layers)
check("no ceramic yet, so no question can be asked and nothing crashes", "ceramic" in s["missing"] and not asked, (s["missing"], asked))
# the geometry step and the handoff files
PASSAGES[:] = [{"id": "p#6", "source": "p", "text": r"alumina layers of 210 µm, $E_{\rm A} \sim 367$  GPa"}, PASSAGES[1]]
s, calls = run([{"design": GOOD, "sources": SRC}])
tools_run = [r["tool"] for r in s["results"]]
made = s["geometry"]
fem = next(r for r in s["results"] if r["tool"] == "fem_fracture")
check("the three tool calls run in order", tools_run == ["dft_cleavage", "md_interface", "fem_fracture"], tools_run)
check("the geometry files exist", all(Path(made[k]).exists() for k in made if k.endswith(("_stl", "_json"))), made.get("geometry_json"))
check("the crack tool received the geometry file", fem["arguments"]["geometry_json"] == made["geometry_json"] == fem["output"]["geometry_json"], fem["arguments"]["geometry_json"])
check("no error in any tool output", not any("error" in r["output"] for r in s["results"]), [r["output"].get("error") for r in s["results"]])
files = {p.stem: json.load(open(p)) for p in Path(s["candidate"], "handoff").glob("*.json")}
inputs = lambda tool: files[tool]["calls"][0]["inputs"]
check("one handoff file per tool, in the candidate folder", sorted(files) == ["dft", "fem", "md"] and Path(s["candidate"]).name == "candidate_1", sorted(files))
check("every call is done and holds its output", all(len(f["calls"]) == 1 and f["calls"][0]["status"] == "done" and f["calls"][0]["output"]["method"] for f in files.values()), [f["calls"][0]["status"] for f in files.values()])
check("every input has an origin", not [k for f in files.values() for k, i in f["calls"][0]["inputs"].items() if i["from"] == "unmatched"], {k: i["from"] for k, i in inputs("fem").items()})
check("a design input names its field and its source", inputs("md")["material_b"] == {"value": "graphite", "from": "design.interlayer", "source": "p#9: deflects"}, inputs("md")["material_b"])
check("a tool output is named as the origin, with the unit", inputs("fem")["layer_fracture_energy_J_m2"] == {"value": 3.0, "unit": "J/m^2", "from": "dft_cleavage.cleavage_energy_J_m2"}, inputs("fem")["layer_fracture_energy_J_m2"])
check("the geometry file is named as the origin", inputs("fem")["geometry_json"]["from"] == "geometry.geometry_json", inputs("fem")["geometry_json"])
check("the design file holds the values and their sources", json.load(open(Path(s["candidate"]) / "design.json")) == {"design": GOOD, "sources": s["sources"]}, s["candidate"])
TAMPER["layer_modulus_GPa"] = 999
s, calls = run([{"design": GOOD, "sources": SRC}])
TAMPER.clear()
check("an input the engineer made up is marked unmatched", next(r for r in s["results"] if r["tool"] == "fem_fracture")["inputs"]["layer_modulus_GPa"]["from"] == "unmatched", s["results"][-1]["inputs"]["layer_modulus_GPa"])
s, calls = run([{"design": {k: v for k, v in GOOD.items() if k != "interlayer_thickness_um"}, "sources": SRC}] * 4)
check("no interlayer thickness, so no geometry and no simulation", "interlayer_thickness_um" in s["missing"] and not s.get("results") and "geometry" not in s, s["missing"])
s, calls = run([{"design": GOOD | {"interlayer": "alumina"}, "sources": SRC}, {"design": {"interlayer": "graphite"}, "sources": {"interlayer": "p#9: deflects"}}])
check("an interlayer of the ceramic's own material is rejected, then repaired", s["design"] == GOOD and len(calls) == 2 and calls[1][1], (s["design"].get("interlayer"), calls))
s, calls = run([{"design": GOOD | {"interlayer": "vitrimer"}, "sources": SRC}, {"design": {"interlayer": "graphite"}, "sources": {"interlayer": "p#9: deflects"}}])
check("a material the tools do not accept is rejected, then repaired", s["design"] == GOOD and len(calls) == 2 and calls[1][1]
      and not any("error" in r["output"] for r in s["results"]), (s["design"].get("interlayer"), calls))
# the judge and the loop back to the designer
PASSAGES[:] = [{"id": "p#6", "source": "p", "text": r"alumina layers of 210 µm, $E_{\rm A} \sim 367$  GPa"},
               {"id": "p#9", "source": "p", "text": "graphite interlayers 20 µm thick deflect cracks in alumina"},
               {"id": "p#11", "source": "p", "text": "epoxy interlayers 5 µm thick deflect cracks in alumina"},
               {"id": "p#12", "source": "p", "text": "PMMA interlayers 8 µm thick deflect cracks in alumina"}]
GOOD2, SRC2 = GOOD | {"interlayer": "epoxy", "interlayer_thickness_um": 5}, SRC | {"interlayer": "p#11: deflects", "interlayer_thickness_um": "p#11: 5 µm thick"}
GOOD3, SRC3 = GOOD | {"interlayer": "PMMA", "interlayer_thickness_um": 8}, SRC | {"interlayer": "p#12: deflects", "interlayer_thickness_um": "p#12: 8 µm thick"}
folder = lambda s, n: Path(s["folder"]) / f"candidate_{n}"
whole = lambda s, n: all((folder(s, n) / f).exists() for f in ["design.json", "verdict.json", "geometry/geometry.json", "geometry/bar_ceramic.stl",
                                                              "geometry/cell_interlayer.stl", "handoff/dft.json", "handoff/md.json", "handoff/fem.json",
                                                              "novelty.json", "cross_checks.json"])
verdicts = lambda s: [c["verdict"]["verdict"] for c in s["candidates"]]

s, calls = run([{"design": GOOD, "sources": SRC}])
check("accepted at the first candidate", verdicts(s) == ["accept"] and whole(s, 1) and not folder(s, 2).exists() and len(judged) == 1 and "candidate_1 was accepted" in s["outcome"], s["outcome"])
check("a verdict on stand-in results is provisional", s["verdict"]["provisional"] and "provisionally" in s["outcome"] and json.load(open(folder(s, 1) / "verdict.json")) == s["verdict"] | {"reward": 0.9}, s["verdict"])
check("the judge reads the criteria, the design fields, the design and the tool outputs", "A crack deflects" in judged[0]
      and "Design fields:" in judged[0] and '"enum"' in judged[0] and '"crack_deflects": true' in judged[0] and '"interlayer": "graphite"' in judged[0], judged[0][:120])

VERDICTS[:] = [REVISE]
s, calls = run([{"design": GOOD, "sources": SRC}, {"design": GOOD2, "sources": SRC2}])
check("the judge of a later candidate sees the earlier ones and their verdicts", len(judged) == 2 and '"candidate": "candidate_1"' in judged[1]
      and "The crack does not deflect." in judged[1] and '"earlier_candidates": []' in judged[0], judged[1][-300:])
check("sent back once, accepted at the second candidate", verdicts(s) == ["revise", "accept"] and s["design"] == GOOD2 and whole(s, 1) and whole(s, 2) and "candidate_2 was accepted" in s["outcome"], (verdicts(s), s["outcome"]))
check("each candidate has its own handoff files", json.load(open(folder(s, 1) / "handoff/md.json"))["calls"][0]["inputs"]["material_b"]["value"] == "graphite"
      and json.load(open(folder(s, 2) / "handoff/md.json"))["calls"][0]["inputs"]["material_b"]["value"] == "epoxy" and len(json.load(open(folder(s, 2) / "handoff/md.json"))["calls"]) == 1, "md.json")
check("the verdict keeps only design fields to change", s["candidates"][0]["verdict"]["change"] == {"interlayer": "choose a weaker interlayer"}, s["candidates"][0]["verdict"]["change"])

prompts.clear()
VERDICTS[:] = []
graph_ask = graph.ask_json
def looped(script, verdicts):
    answers, marks = iter(script), list(verdicts)
    def fake(model, system, text, images=(), effort=None):
        rec = {"model": model, "seconds": 0, "reply": {}}
        if system is graph.PLANNER:
            return {"queries": ["q"], "entities": []}, rec
        if system is graph.CHECKER:
            return {"low": 0, "high": 1e9}, rec
        if system is graph.FABRICATOR:
            return RECIPE, rec
        if system is graph.JUDGE:
            return (marks.pop(0) if marks else ACCEPT), rec
        prompts.append(text)
        return next(answers), rec
    graph.ask_json = fake
    return graph.build(MODELS, {}).invoke({"question": "Design a layered alumina.", "folder": next(folders)}, {"recursion_limit": 100})

s = looped([{"design": GOOD, "sources": SRC}, {"design": GOOD2, "sources": SRC2}], [REVISE])
check("the designer is shown the earlier candidate and the judge's finding", len(prompts) == 2 and "Earlier candidates" not in prompts[0] and "Earlier candidates" in prompts[1]
      and "choose a weaker interlayer" in prompts[1] and '"interlayer": "graphite"' in prompts[1]
      and "even when the question states its value" in prompts[1], prompts[1][-200:])

s = looped([{"design": GOOD, "sources": SRC}, {"design": GOOD2, "sources": SRC2}, {"design": GOOD3, "sources": SRC3}], [REVISE, REVISE, REVISE, REVISE])
check("three candidates sent back, the run ends without a candidate", verdicts(s) == ["revise"] * 3 and whole(s, 3) and not folder(s, 4).exists()
      and "no candidate was accepted after 3" in s["outcome"] and s["report"].startswith("report") and not s.get("recipe"), (verdicts(s), s["outcome"]))

prompts.clear()
s = looped([{"design": GOOD, "sources": SRC}, {"design": GOOD, "sources": SRC}, {"design": {"interlayer": "epoxy", "interlayer_thickness_um": 5}, "sources": SRC2}], [REVISE])
check("a repeat of a judged candidate is rejected, then repaired", verdicts(s) == ["revise", "accept"] and s["design"] == GOOD2 and len(prompts) == 3
      and "repeats a candidate" in prompts[2] and "interlayer" in prompts[2].split("rejected these fields")[-1], (verdicts(s), s["design"].get("interlayer")))

s = looped([{"design": GOOD, "sources": SRC}, {"design": GOOD, "sources": SRC}, {"design": GOOD, "sources": SRC}] + [{"design": GOOD, "sources": SRC}] * 4, [REVISE])
check("a designer that only repeats ends with an incomplete design and no second simulation", verdicts(s) == ["revise"] and "interlayer" in s["missing"]
      and not folder(s, 2).exists() and "no candidate was accepted after 1" in s["outcome"] and "incomplete" in s["outcome"], (s["missing"], s["outcome"]))

check("a report on an incomplete last design still sees every method string", "stand-in" in REPORTED[-1] and '"crack_deflects": true' in REPORTED[-1]
      and '"results"' not in REPORTED[-1].split('"candidates"')[0], REPORTED[-1][:200])
graph.ask_json = graph_ask
TAMPER["material"] = "unobtainium"
s, calls = run([{"design": GOOD, "sources": SRC}])
TAMPER.clear()
check("a tool error blocks the verdict, and the model is not asked to judge", verdicts(s) == ["incomplete"] and not judged and "dft_cleavage" in s["verdict"]["summary"]
      and "did not complete" in s["outcome"] and json.load(open(folder(s, 1) / "handoff/dft.json"))["calls"][0]["status"] == "error", s["verdict"])
TAMPER["layer_modulus_GPa"] = 999
s, calls = run([{"design": GOOD, "sources": SRC}])
TAMPER.clear()
check("an input from nowhere blocks the verdict", verdicts(s) == ["incomplete"] and not judged and "fem_fracture.layer_modulus_GPa" in s["verdict"]["summary"], s["verdict"])
s, calls = run([{"design": {k: v for k, v in GOOD.items() if k != "ceramic"}, "sources": SRC}] * 4)
check("an incomplete first design is reported as not simulated", s["outcome"] == "no design was simulated, and the last design is incomplete" and not s.get("candidates"), s["outcome"])
sections = json.loads(REPORTED[-1])["sections"]
check("a report without candidates is told to write no recipe, novelty or simulation section", not any(w in " ".join(sections)
      for w in ["recipe", "novelty", "simulation", "hypothesis"]) and any("missing" in x for x in sections), sections)
# fabrication, novelty and reward
graph.ask_json = graph_ask
s, calls = run([{"design": GOOD, "sources": SRC}])
best = Path(s["folder"]) / "candidate_1"
files = s["fabrication"]["files"]
check("an accepted candidate gets a recipe, print files and a rendered part", s["best"] == "candidate_1" and (best / "recipe.json").exists()
      and (best / "recipe.md").exists() and all(Path(v).exists() for v in files.values()) and (best / "geometry/render.png").exists(), files)
check("the print files follow the chosen process", s["fabrication"]["process"] == "direct ink writing" and any(str(v).endswith(".gcode") for v in files.values()), list(files))
steps, printing = s["recipe"]["steps"][0]["parameters"], s["recipe"]["print_parameters"]
check("a recipe number keeps its passage only when that passage states it", steps["particle_size_um"]["source"] == "p#9"
      and printing["nozzle_diameter_um"]["source"] == "p#6" and printing["print_speed_mm_s"]["source"].startswith("assumption: p#6 does not state 7")
      and steps["binder_wt_percent"]["source"] == "assumption: typical", (steps, printing))
check("the recipe print parameters reach the print files", s["fabrication"]["parameters"]["nozzle_diameter_um"]["value"] == 210
      and s["fabrication"]["parameters"]["nozzle_diameter_um"]["source"] == "p#6", s["fabrication"]["parameters"].get("nozzle_diameter_um"))
report = s["report"]
links = [l.split("](")[1].rstrip(")") for l in report.splitlines() if l.startswith("![")]
check("the report appends candidates, figures, novelty, tool outputs, recipe and print files", all(h in report for h in [
      "## Candidates", "## Figures", "## Novelty of the best candidate", "## Simulation outputs of candidate_1", "## Recipe", "## Print files"])
      and len(links) == 3 and all((Path(s["folder"]) / l).exists() for l in links), links)
check("a report with an accepted candidate lists every section", len(json.loads(REPORTED[-1])["sections"]) == 9, json.loads(REPORTED[-1])["sections"])
check("the reporter is told the recipe, the novelty and the cross-checks", '"recipe"' in REPORTED[-1] and '"novelty"' in REPORTED[-1]
      and '"md_bulk"' in REPORTED[-1] and "| md_bulk | alumina | youngs_modulus_GPa = 347.8 |" in report and "with a novelty of 0.90" in s["outcome"], s["outcome"])

NOVELTY.update({"graphite": 0.2, "epoxy": 0.8})
prompts.clear()
s = looped([{"design": GOOD, "sources": SRC}, {"design": GOOD2, "sources": SRC2}], [])
check("an accepted candidate below the novelty target sends the designer back", verdicts(s) == ["accept", "accept"] and s["best"] == "candidate_2"
      and [c["reward"] for c in s["candidates"]] == [0.2, 0.8] and "below the target" in prompts[1] and (folder(s, 2) / "recipe.json").exists()
      and not (folder(s, 1) / "recipe.json").exists(), [c["reward"] for c in s["candidates"]])
NOVELTY.update({"epoxy": 0.3, "PMMA": 0.1})
s = looped([{"design": GOOD, "sources": SRC}, {"design": GOOD2, "sources": SRC2}, {"design": GOOD3, "sources": SRC3}], [])
check("with no candidate above the target, the run makes the one with the highest reward", len(s["candidates"]) == 3 and s["best"] == "candidate_2"
      and "below the target of 0.5" in s["outcome"], s["outcome"])
NOVELTY.clear()
s = looped([{"design": GOOD, "sources": SRC}, {"design": GOOD2, "sources": SRC2}], [REVISE])
check("a candidate the judge sends back has no reward", [c["reward"] for c in s["candidates"]] == [0.0, 0.9] and s["best"] == "candidate_2", [c["reward"] for c in s["candidates"]])
print("FAILED" if bad else "ALL PASS", bad)
