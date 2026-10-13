# Tests for the agent graph, with the models and the search stubbed. python -m pipeline.tests.test_graph
import itertools
import json
import shutil
from pathlib import Path

from pipeline import graph

shutil.rmtree("runs_test/graph", ignore_errors=True)
folders = (f"runs_test/graph/{n}" for n in itertools.count(1))  # a fresh run folder per case

PASSAGES = [{"id": "p#6", "source": "p", "text": r"alumina layers of 210 µm, $E_{\rm A} \sim 367$  GPa"},
            {"id": "p#9", "source": "p", "text": "graphite interlayers 20 µm thick deflect cracks in alumina"}]
graph.cp.search = lambda corpus, q, k, figures=False, skip=(), pattern=None: [] if figures else [p for p in PASSAGES if p["id"] not in skip]
graph.cp.neighbors = lambda corpus, entities, k=3: []
TAMPER = {}  # tool inputs the scripted engineer changes on its own, which the handoff file must flag


def fake_chat(model, messages, tools=None, effort=None):
    """A scripted engineer. It calls dft_cleavage and md_interface, then fem_fracture, from the design, the geometry and the results."""
    rec = {"model": model, "seconds": 0, "reply": {}}
    if not tools:
        return {"content": "report"}, rec
    given = json.loads(messages[1]["content"])
    design = given["design"]
    done = [json.loads(m["content"]) for m in messages if m.get("role") == "tool"]
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
GOOD = {"ceramic": "alumina", "interlayer": "graphite", "layer_modulus_GPa": 367, "layer_thickness_um": 210,
        "interlayer_thickness_um": 20}
SRC = {"ceramic": "question: asked", "interlayer": "p#9: deflects", "layer_modulus_GPa": "p#6: E_A", "layer_thickness_um": "p#6: layers",
       "interlayer_thickness_um": "p#9: 20 µm thick"}


asked = []
accept_all = lambda text: {"low": 0, "high": 1e9, "unit": ""}


def run(script, truncate_first=False, checker=accept_all):
    """Run the graph with the designer answering from the script, one answer per call."""
    asked.clear()
    calls, answers = [], iter(script)
    def fake(model, system, text, images=(), effort=None):
        if system is graph.PLANNER:
            return {"queries": ["q"], "entities": []}, {"model": model, "seconds": 0, "reply": {}}
        if system is graph.CHECKER:
            asked.append(text.split("\n")[0])
            return checker(text), {"model": model, "seconds": 0, "reply": {}}
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
          "interlayer_thickness_um": "E2: 20 µm thick"}
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
print("FAILED" if bad else "ALL PASS", bad)
