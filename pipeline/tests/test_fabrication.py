# Tests for the print files and figures. python -m pipeline.tests.test_fabrication
import json, math, shutil
from pathlib import Path
from PIL import Image
from pipeline.fabrication import DEFAULTS, PROCESSES, write_fabrication
from pipeline.geometry import layered_geometry
from pipeline.render import render_part

bad = 0
def check(name, ok, detail=""):
    global bad
    bad += not ok
    print("ok  " if ok else "FAIL", name, "|", str(detail)[:150])

shutil.rmtree("runs_test/fab", ignore_errors=True)
g = layered_geometry("alumina", "graphite", 210, 20, "runs_test/fab/geometry")["geometry_json"]
spec = json.load(open(g))
L, W, H = spec["bar"]["size_mm"]
n = spec["bar"]["ceramic_layers"]
check("process names are the three agreed strings", PROCESSES == ["direct ink writing", "vat photopolymerization", "binder jetting"])


def gcode(path):
    """Print moves as (x0, y0, x1, y1, z, E), travel moves as (x, y, z, has E), the tools in order, and the z of the last print move before each tool."""
    x = y = z = 0.0
    prints, travels, tools = [], [], []
    for line in Path(path).read_text().splitlines():
        w = line.split(";")[0].split()
        if w and w[0] in ("T0", "T1"):
            tools.append((w[0], prints[-1][4] if prints else None))
        if w and w[0] in ("G0", "G1"):
            a = {k[0]: float(k[1:]) for k in w[1:]}
            if w[0] == "G1":
                prints.append((x, y, a.get("X", x), a.get("Y", y), z, a.get("E", 0)))
            else:
                travels.append((a.get("X", x), a.get("Y", y), a.get("Z", z), "E" in a))
            x, y, z = a.get("X", x), a.get("Y", y), a.get("Z", z)
    return prints, travels, tools


r = write_fabrication(g, "direct ink writing", {}, "runs_test/fab/diw")
text = Path(r["files"]["gcode"]).read_text()
prints, travels, tools = gcode(r["files"]["gcode"])
tops = list(dict.fromkeys(p[4] for p in prints))
expected, z = [], 0.0
for i in range(2 * n - 1):
    t, k = (210, math.ceil(210 / 200)) if i % 2 == 0 else (20, 1)  # a 250 µm nozzle prints passes of at most 200 µm
    for _ in range(k):
        z += t / k / 1000
        expected.append(z)
check("pass heights follow the layer plan", len(tops) == len(expected) == r["printed_layers"] == 38
      and all(abs(a - b) < 1e-4 for a, b in zip(tops, expected)), (len(tops), tops[:4]))
check("top of the last pass is the bar height within 1 µm", abs(tops[-1] - H) < 1e-3, (tops[-1], H))
changes = [b for a, b in zip(tools, tools[1:]) if a[0] != b[0]]
check("tool changes are 2 x (ceramic layers - 1)", len(changes) == 2 * (n - 1) == r["tool_changes"], len(changes))
check("each switch to T1 follows the top of a ceramic layer",
      all(abs(zt - (j * 0.23 + 0.21)) < 1e-4 for j, (tool, zt) in enumerate(c for c in changes if c[0] == "T1")))
xs = [p[k] for p in prints for k in (0, 2)] + [t[0] for t in travels]
ys = [p[k] for p in prints for k in (1, 3)] + [t[1] for t in travels]
check("every move stays inside the bar footprint", 0 <= min(xs) and max(xs) <= L and 0 <= min(ys) and max(ys) <= W,
      (min(xs), max(xs), min(ys), max(ys)))
check("E is positive on every print move and absent on travel", all(p[5] > 0 for p in prints) and not any(t[3] for t in travels))
plunger = math.pi * (14.5 / 2) ** 2
volume = sum(spec["bar"]["volume_mm3"].values())
check("extruded volume matches the bar volume within 1 %", abs(sum(p[5] for p in prints) * plunger / volume - 1) < 0.01,
      sum(p[5] for p in prints) * plunger / volume)
check("header lists every parameter with its source", all(f";   {k} {x['value']} {x['unit'].replace('µ', 'u')} (default)" in text
                                                            for k, x in DEFAULTS["direct ink writing"].items()))
check("header says a thin interlayer is better coated", "dip or spray coating" in text and any("dip or spray" in x for x in r["notes"]))
check("header gives the estimated time", f"Estimated print time {r['estimated_print_time_min']:.1f} min" in text, r["estimated_print_time_min"])
check("G-code is ASCII and standard", text.isascii() and all(c in text for c in ["G21", "G90", "M83", "T0", "T1"]))

given = {"nozzle_diameter_um": {"value": 410, "unit": "µm", "source": "doi:10.1000/recipe"}}
r2 = write_fabrication(g, "direct ink writing", given, "runs_test/fab/diw_410")
first = [p for p in gcode(r2["files"]["gcode"])[0] if p[4] == 0.21]
roads = sorted({round(p[3], 4) for p in first})
check("a given parameter overrides the default and keeps its source",
      r2["parameters"]["nozzle_diameter_um"] == given["nozzle_diameter_um"]
      and r2["parameters"]["print_speed_mm_s"]["source"] == "default"
      and "nozzle_diameter_um 410 um (doi:10.1000/recipe)" in Path(r2["files"]["gcode"]).read_text(), r2["parameters"]["nozzle_diameter_um"])
check("the larger nozzle sets road spacing and one pass per layer", abs(roads[1] - roads[0] - 0.41) < 1e-6 and r2["printed_layers"] == 2 * n - 1,
      (roads[:3], r2["printed_layers"]))

rv = write_fabrication(g, "vat photopolymerization", {}, "runs_test/fab/vat")
job = json.load(open(rv["files"]["print_job_json"]))
zs = [x["z_mm"] for x in job["layers"]]
check("vat layers rise monotonically", all(b > a for a, b in zip(zs, zs[1:])), zs[:4])
check("vat layer thicknesses add up to the bar height", abs(sum(x["thickness_um"] for x in job["layers"]) / 1000 - H) < 1e-3
      and abs(zs[-1] - H) < 1e-6, zs[-1])
check("vat slices are at most 50 µm", max(x["thickness_um"] for x in job["layers"]) <= 50, rv["layers"])
check("base layers take the base exposure", [x["exposure_s"] for x in job["layers"][:4]] == [20, 20, 20, 5])
check("vat material changes are 2 x (ceramic layers - 1)", job["material_changes"] == rv["material_changes"] == 2 * (n - 1)
      and sum("resin" in x["material"] for x in job["layers"]) == n - 1)
mask = Image.open(rv["files"]["mask_png"])
check("mask has the stated size and footprint", list(mask.size) == job["mask"]["image_px"] and mask.mode == "1"
      and sum(v > 0 for v in mask.getdata()) == 900 * 80, (mask.size, mask.mode))

rb = write_fabrication(g, "binder jetting", {}, "runs_test/fab/bj")
job = json.load(open(rb["files"]["print_job_json"]))
check("binder jetting notes say the interlayer is infiltrated or coated between sessions",
      job["notes"] == rb["notes"] and any("infiltrated or coated between print sessions" in x for x in job["notes"]), job["notes"][0])
check("binder jetting has one session per ceramic layer", job["print_sessions"] == rb["print_sessions"] == n)
bound = [x for x in job["layers"] if x.get("printed", True)]
check("powder layers are at most 100 µm with 70 % binder saturation",
      all(x["thickness_um"] <= 100 and x["binder_saturation_percent"] == 70 for x in bound) and len(bound) == rb["printed_layers"] == 3 * n)
check("binder jetting layers add up to the bar height", abs(sum(x["thickness_um"] for x in job["layers"]) / 1000 - H) < 1e-3)

try:
    write_fabrication(g, "fused filament fabrication", {}, "runs_test/fab/none")
    check("unknown process raises ValueError", False)
except ValueError as e:
    check("unknown process raises ValueError", True, e)

png = render_part(g, "runs_test/fab/geometry.png")
pngs = [r["files"]["preview_png"], r2["files"]["preview_png"], rv["files"]["preview_png"], rv["files"]["mask_png"], rb["files"]["preview_png"], png]
check("all PNG files start with the PNG signature", all(Path(p).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in pngs), pngs)
check("results are JSON", all(json.dumps(x) for x in (r, r2, rv, rb)))
print("FAILED" if bad else "ALL PASS", bad)
