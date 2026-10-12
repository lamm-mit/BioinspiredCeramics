# Tests for the source check. python -m pipeline.tests.test_source_check
# The cases are number and unit formats that real runs tripped on.
import json
from pipeline.design import SCALE
from pipeline.graph import quantities, source_errors

um, gpa = SCALE["layer_thickness_um"], SCALE["layer_modulus_GPa"]
cases = [
    (r"ZTA/A/ZTA/A/ZTA of  $\sim 140~\mu m/210~\mu m/1040~\mu m/220~\mu m/110~\mu m$  after sintering", um, [140, 210, 1040, 220, 110]),
    ("ceramic layers ranged from 0.25 to 1.27 mm", um, [(250, 1270)]),
    (r"( $E_{\rm ZTA} \sim 333$  GPa and  $E_{\rm A} \sim 367$  GPa [28])", gpa, [333, 367]),
    ("printed with layer thicknesses of 50 µm, 75 µm, and 100 µm", um, [50, 75, 100]),
    ("nozzle diameter 0.41 mm", um, [410]),
    ("a total of 1,040 μm", um, [1040]),
    ("corresponds to approx. 40 µm layers after sintering", um, [40]),
    ("strength of 310.1±36.5 MPa at 95 mol% alumina", gpa, [0.3101]),
    ("see Fig. 12 and Table 3 for 2023 data", um, []),
    # cases from the review
    (r"modulus of $367 \pm 5$ GPa", gpa, [367]),
    (r"grain size of $\sim 2.7 \pm 1.5$ µm", um, [2.7]),
    ("a modulus of (367 ± 5) GPa", gpa, [367]),
    ("filaments of 315 ± 66 µm", um, [315]),
    ("layers of 50, 75, and 100 µm", um, [50, 75, 100]),
    ("pores of 100~200 μm", um, [(100, 200)]),
    ("a 200-µm-thick layer", um, [200]),
    (r"$367 \text{ GPa}$", gpa, [367]),
    ("about 150 microns", um, [150]),
    ("tested at 50,75,100 µm", um, [50, 75, 100]),
]
bad = 0
for text, units, want in cases:
    got = [round(lo, 4) if lo == hi else (round(lo, 4), round(hi, 4)) for lo, hi in quantities(text, units)]
    ok = got == want
    bad += not ok
    print("ok  " if ok else "FAIL", got, "<-", text[:60])

s = {"question": "Design a nacre-inspired layered alumina.",
     "passages": [{"id": "p#6", "text": r"layers of  $\sim 140~\mu m/210~\mu m$  after sintering, $E_{\rm A} \sim 367$  GPa"},
                  {"id": "p#25", "text": "The filaments were printed by direct ink writing."},
                  {"id": "p#3", "text": "other passage with 999 GPa"},
                  {"id": "q#5", "text": "ceramic layers ranged from 0.25 to 1.27 mm"},
                  {"id": "t#31", "text": "Property | AL98 | AL99\nYoung's Modulus, GPa | 325 | 360\nHardness, HV | 2100 | 2200"},
                  {"id": "u#1", "text": "filaments of 315 ± 66 µm were printed"},
                  {"id": "l#1", "text": "moduli of 333 GPa and 367 GPa were measured, and 333, 367 GPa again"},
                  {"id": "fig#1", "text": "caption", "image": "x.png"},
                  {"id": "fig#2", "text": "caption", "image": "y.png"}],
     "figure_notes": [{"id": "fig#1", "caption": "Printed filament", "reading": {"values": [{"quantity": "nozzle", "value": 0.41, "unit": "mm"}], "summary": "modulus is typically 380 GPa"}},
                      {"id": "fig#2", "caption": "No values", "reading": {"values": None, "summary": ""}}],
     "triples": [{"head": "Interlayer Paste", "relation": "is made of alumina and graphite", "tail": "Graphite", "source": "10.1016/j.x.2023.1"},
                 {"head": "AL98 Alumina", "relation": "has a Young's Modulus of 325–360 GPa", "tail": "Young's Modulus", "source": "10.1016/j.r.2020.2"}]}
gate = [
    ("value in passage", {"layer_modulus_GPa": 367}, {"layer_modulus_GPa": "p#6: E_A"}, []),
    ("value absent from cited passage", {"layer_modulus_GPa": 380}, {"layer_modulus_GPa": "p#25: fully dense Al2O3"}, ["layer_modulus_GPa"]),
    ("id does not exist", {"layer_thickness_um": 40}, {"layer_thickness_um": "10.1016_j.actamat.2023.103776#21: printed"}, ["layer_thickness_um"]),
    ("p#3 must not match p#31 or stand in for p#6", {"layer_modulus_GPa": 999}, {"layer_modulus_GPa": "p#31: x"}, ["layer_modulus_GPa"]),
    ("thickness inside a reported range", {"layer_thickness_um": 300}, {"layer_thickness_um": "q#5: layers 0.25 to 1.27 mm"}, []),
    ("thickness outside the range", {"layer_thickness_um": 100}, {"layer_thickness_um": "q#5: thin"}, ["layer_thickness_um"]),
    ("modulus between two values is not stated", {"layer_modulus_GPa": 350}, {"layer_modulus_GPa": "p#6"}, ["layer_modulus_GPa"]),
    ("figure value with unit conversion", {"layer_thickness_um": 410}, {"layer_thickness_um": "fig#1: nozzle 0.41 mm"}, []),
    ("figure summary does not count", {"layer_modulus_GPa": 380}, {"layer_modulus_GPa": "fig#1: typical"}, ["layer_modulus_GPa"]),
    ("choice from a triple, either DOI spelling", {"interlayer": "graphite"}, {"interlayer": "10.1016_j.x.2023.1: paste"}, []),
    ("choice from the question", {"ceramic": "alumina"}, {"ceramic": "question: asks for alumina"}, []),
    ("no source at all", {"ceramic": "alumina"}, {}, ["ceramic"]),
    ("number given as text", {"layer_modulus_GPa": "367 GPa"}, {"layer_modulus_GPa": "p#6: E"}, ["layer_modulus_GPa"]),
    ("two ids, value in the second", {"layer_modulus_GPa": 367}, {"layer_modulus_GPa": "p#25,p#6: both"}, []),
    # cases from the review
    ("value in a table row that names the unit", {"layer_modulus_GPa": 325}, {"layer_modulus_GPa": "t#31: table"}, []),
    ("table number of another row", {"layer_modulus_GPa": 2100}, {"layer_modulus_GPa": "t#31: table"}, ["layer_modulus_GPa"]),
    ("evidence id prefix before the id", {"layer_modulus_GPa": 367}, {"layer_modulus_GPa": "evidence id: p#6: E_A"}, []),
    ("capital Question", {"ceramic": "alumina"}, {"ceramic": "Question: asks for alumina"}, []),
    ("the word question inside a reason is not an id", {"ceramic": "alumina"}, {"ceramic": "a common choice for this question"}, ["ceramic"]),
    ("sources is null", {"ceramic": "alumina"}, None, ["ceramic"]),
    ("uncertainty does not widen a range", {"layer_thickness_um": 100}, {"layer_thickness_um": "u#1: filaments"}, ["layer_thickness_um"]),
    ("modulus inside a stated range", {"layer_modulus_GPa": 350}, {"layer_modulus_GPa": "10.1016/j.r.2020.2: AL98"}, []),
    ("modulus outside a stated range", {"layer_modulus_GPa": 380}, {"layer_modulus_GPa": "10.1016/j.r.2020.2: AL98"}, ["layer_modulus_GPa"]),
    ("modulus between two listed values is not stated", {"layer_modulus_GPa": 350}, {"layer_modulus_GPa": "l#1: two"}, ["layer_modulus_GPa"]),
]
for name, design, sources, want in gate:
    got = source_errors(design, sources or {}, s)
    ok = sorted(got) == want
    bad += not ok
    print("ok  " if ok else "FAIL", name, "->", got)
print("FAILED" if bad else "ALL PASS", bad)
