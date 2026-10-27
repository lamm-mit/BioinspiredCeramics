# Tests for the print files and figures. python -m pipeline.tests.test_fabrication
import json, math, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from pipeline.fabrication import DEFAULTS, PROCESSES, write_fabrication
from pipeline.geometry import layered_geometry
from pipeline.pattern import joint, ply_shift, roads
from pipeline.render import render_part

bad = 0
def check(name, ok, detail=""):
    global bad
    bad += not ok
    print("ok  " if ok else "FAIL", name, "|", str(detail)[:150])

shutil.rmtree("runs_test/fab", ignore_errors=True)
THICK = {"ceramic": "alumina", "interlayer": "graphite", "layer_thickness_um": 210, "interlayer_thickness_um": 20, "layer_modulus_GPa": 380}
g = layered_geometry(THICK, {}, "runs_test/fab/geometry")["geometry_json"]
spec = json.load(open(g))
L, W, H = spec["bar"]["size_mm"]
n = spec["bar"]["ceramic_layers"]
check("process names are the three agreed strings", PROCESSES == ["direct ink writing", "vat photopolymerization", "binder jetting"])


def gcode(path):
    """Print moves as (x0, y0, z0, x1, y1, z1, E, ply, block, run), travel moves as (x, y, z, has E), the tools in order
    with the z of the last print move before each, and the M165 mixes as (ply, {A, B, C}).

    A block is the moves after one "; ply" comment, and a run is a stretch of print moves between two travel moves.
    """
    x = y = z = 0.0
    ply, block, run, last = None, -1, 0, "G0"
    prints, travels, tools, mixes = [], [], [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("; ply "):
            ply, block = int(line.split()[2]) - 1, block + 1
        w = line.split(";")[0].split()
        if not w:
            continue
        if w[0] in ("T0", "T1"):
            tools.append((w[0], prints[-1][5] if prints else None))
        if w[0] == "M165":
            mixes.append((ply, {k[0]: float(k[1:]) for k in w[1:]}))
        if w[0] in ("G0", "G1"):
            a = {k[0]: float(k[1:]) for k in w[1:]}
            X, Y, Z = a.get("X", x), a.get("Y", y), a.get("Z", z)
            if w[0] == "G1":
                run += last == "G0"
                prints.append((x, y, z, X, Y, Z, a.get("E", 0), ply, block, run))
            else:
                travels.append((X, Y, Z, "E" in a))
            x, y, z, last = X, Y, Z, w[0]
    return prints, travels, tools, mixes


def inside(prints, travels, L, W):
    xs = [p[k] for p in prints for k in (0, 3)] + [t[0] for t in travels]
    ys = [p[k] for p in prints for k in (1, 4)] + [t[1] for t in travels]
    return 0 <= min(xs) and max(xs) <= L and 0 <= min(ys) and max(ys) <= W


# A flat sheet design with ceramic layers thicker than the largest pass.
r = write_fabrication(g, "direct ink writing", {}, "runs_test/fab/diw")
text = Path(r["files"]["gcode"]).read_text()
prints, travels, tools, _ = gcode(r["files"]["gcode"])
tops = list(dict.fromkeys(p[5] for p in prints))
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
check("every move stays inside the bar footprint", inside(prints, travels, L, W))
check("E is positive on every print move and absent on travel", all(p[6] > 0 for p in prints) and not any(t[3] for t in travels))
plunger = math.pi * (14.5 / 2) ** 2
# Road centers stay half a road width inside the footprint and a strip narrower than a road stays empty, so the
# roads hold a few percent less than the bar.
ratio = sum(p[6] for p in prints) * plunger / (L * W * H)
check("extruded volume is 95 to 100 % of the bar volume", 0.95 < ratio <= 1, ratio)
check("header lists every parameter with its source", all(f";   {k} {x['value']} {x['unit'].replace('µ', 'u')} (default)" in text
                                                            for k, x in DEFAULTS["direct ink writing"].items()))
check("header says a thin interlayer is better coated", "dip or spray coating" in text and any("dip or spray" in x for x in r["notes"]))
check("header gives the estimated time", f"Estimated print time {r['estimated_print_time_min']:.1f} min" in text, r["estimated_print_time_min"])
check("header gives the pattern and the ply angles", "; Toolpath pattern of straight roads" in text
      and "; Road angles of the plies in degrees from x, bottom first, " + " ".join(["0"] * n) in text)
check("G-code is ASCII and standard", text.isascii() and all(c in text for c in ["G21", "G90", "M83", "T0", "T1"]))

given = {"nozzle_diameter_um": {"value": 410, "unit": "µm", "source": "doi:10.1000/recipe"}}
r2 = write_fabrication(g, "direct ink writing", given, "runs_test/fab/diw_410")
first = sorted({round(p[4], 4) for p in gcode(r2["files"]["gcode"])[0] if p[7] == 0 and p[5] == 0.21})
check("a given parameter overrides the default and keeps its source",
      r2["parameters"]["nozzle_diameter_um"] == given["nozzle_diameter_um"]
      and r2["parameters"]["print_speed_mm_s"]["source"] == "default"
      and "nozzle_diameter_um 410 um (doi:10.1000/recipe)" in Path(r2["files"]["gcode"]).read_text(), r2["parameters"]["nozzle_diameter_um"])
check("the larger nozzle prints one pass per layer, and roads stay one road width apart",
      np.allclose(np.diff(first), spec["road_width_mm"], atol=2e-3) and r2["printed_layers"] == 2 * n - 1, (first[:3], r2["printed_layers"]))
check("a nozzle 49 % off the road width gets a note", any("differs from the 276 µm road width by 49 %" in x for x in r2["notes"])
      and not any("differs from" in x for x in r["notes"]), r2["notes"][-1])

# Every pattern on a design with a 126 µm road width.
D = {"ceramic": "alumina", "interlayer": "PMMA", "layer_thickness_um": 70, "interlayer_thickness_um": 35, "layer_modulus_GPa": 367}
PATTERNS = {"flat": {}, "nacre": {"break_length_h": 20, "interlayer_placement": "shell"},
            "Bouligand": {"ply_rotation_deg": 15, "interlayer_placement": "shell"},
            "crossed": {"road_angle_deg": -30, "ply_rotation_deg": 60, "ply_rotation_alternates": True, "interlayer_placement": "shell"},
            "rings": {"road_path": "rings", "ring_spacing_w": 8, "interlayer_placement": "shell"},
            "waves": {"road_path": "waves", "wave_amplitude_w": 1, "wave_length_w": 10, "interlayer_placement": "shell"},
            "wavy plies": {"ply_wave_amplitude_h": 0.5, "ply_wave_length_h": 30, "break_length_h": 20},
            "graded": {"modulus_profile": "linear", "outer_layer_modulus_GPa": 250}}
runs = {}
for name, pattern in PATTERNS.items():
    s = json.load(open(layered_geometry(D, pattern, f"runs_test/fab/{name}/geometry")["geometry_json"]))
    rp = write_fabrication(f"runs_test/fab/{name}/geometry/geometry.json", "direct ink writing", {}, f"runs_test/fab/{name}/diw")
    runs[name] = (s, rp, Path(rp["files"]["gcode"]).read_text(), *gcode(rp["files"]["gcode"]))
    print(f"     {name}: {Path(rp['files']['gcode']).stat().st_size / 1e6:.1f} MB, {rp['estimated_print_time_min']:.0f} min")
L, W, H = runs["flat"][0]["bar"]["size_mm"]
n = runs["flat"][0]["bar"]["ceramic_layers"]
check("every move of every pattern stays inside the bar footprint", all(inside(x[3], x[4], L, W) for x in runs.values()))
check("every G-code file is under 30 MB", all(len(x[2]) < 30e6 for x in runs.values()), {k: len(x[2]) // 10 ** 6 for k, x in runs.items()})
shell = [k for k, x in PATTERNS.items() if x.get("interlayer_placement") == "shell"]
check("shell placement prints with one coaxial head T0, sheet placement with T0 and T1",
      all({t for t, _ in runs[k][5]} == {"T0"} and "coaxial core-shell head" in runs[k][2] for k in shell)
      and all({t for t, _ in x[5]} == {"T0", "T1"} for k, x in runs.items() if k not in shell))

for name in ["Bouligand", "crossed"]:
    s, prints = runs[name][0], runs[name][3]
    off = [min(d % 180, 180 - d % 180) for p in prints if p[7] < 3 and math.hypot(p[3] - p[0], p[4] - p[1]) > 1
           for d in [math.degrees(math.atan2(p[4] - p[1], p[3] - p[0])) - s["plies"][p[7]]["angle_deg"]]]
    check(f"{name} roads of plies 1 to 3 run at the ply angles {[s['plies'][k]['angle_deg'] for k in range(3)]}",
          {p[7] for p in prints if p[7] < 3} == {0, 1, 2} and max(off) < 0.1, max(off))

s, rp, _, prints, travels, *_ = runs["nacre"]
ply0 = [p for p in prints if p[7] == 0]
crossing = [p for p in ply0 for f in np.linspace(0.05, 0.95, 10)
            if joint(s["pattern"], s["design"], 0, np.array([p[0] + f * (p[3] - p[0])]))[0]]
expected = len(roads(s["pattern"], s["design"], n, H, 0, (L, W))["ceramic"])
check("nacre extrudes nothing across a break", not crossing, crossing[:2])
check("nacre ply 1 has one extruded run per road", len({p[9] for p in ply0}) == expected, (len({p[9] for p in ply0}), expected))
check("a note says breaks this narrow may close", any("may close before sintering" in x for x in rp["notes"]))

s, prints = runs["wavy plies"][0], runs["wavy plies"][3]
k = n // 2
block = min(p[8] for p in prints if p[7] == k)
mid = [p for p in prints if p[8] == block]  # the ceramic roads of the middle ply
wave = [abs(p[5] - (k * 0.105 + 0.07 + ply_shift(s["pattern"], s["design"], H, p[3], p[5]))) for p in mid]
spread = lambda prints: max(np.ptp([p[5] for p in prints if p[8] == b]) for b in {p[8] for p in prints})
check("Z follows the ply wave within 2 µm on the middle ply", max(wave) < 2e-3 and np.ptp([p[5] for p in mid]) > 0.06,
      (max(wave), np.ptp([p[5] for p in mid])))
check("Z is constant within each pass of the flat pattern", spread(runs["flat"][3]) == 0, spread(runs["flat"][3]))

s, mixes = runs["graded"][0], runs["graded"][6]
outer = {k: m["B"] / (m["A"] + m["B"]) for k, m in mixes}
check("graded G-code sets one mix per ply, each summing to 1", len(mixes) == n and [k for k, _ in mixes] == list(range(n))
      and all(abs(sum(m.values()) - 1) < 1e-6 for _, m in mixes), mixes[:2])
check("the outer paste fraction is larger at ply 1 than at the middle ply and follows plies[k]",
      outer[0] > outer[n // 2] and all(abs(outer[k] - s["plies"][k]["outer_fraction"]) < 1e-3 for k in outer), (outer[0], outer[n // 2]))
check("graded header says the mix assumes linear mixing of the modulus",
      "mixes linearly with the volume fraction" in runs["graded"][2] and "; Ply moduli in GPa" in runs["graded"][2])
w, h, sh = 0.126, 0.07, 0.035
check("shell mixes give C as the shell share of the road cross-section", all(
      abs(m["C"] - (1 - (w - sh) * h / (w * (h if k == 0 else h + sh)))) < 1e-4 and m["B"] == 0 for k, m in runs["Bouligand"][6]))

rv = write_fabrication(g, "vat photopolymerization", {}, "runs_test/fab/vat")
L, W, H = spec["bar"]["size_mm"]
n = spec["bar"]["ceramic_layers"]
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
      and np.count_nonzero(np.array(mask)) == 900 * 80, (mask.size, mask.mode))

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
pngs = [r["files"]["preview_png"], r2["files"]["preview_png"], rv["files"]["preview_png"], rv["files"]["mask_png"],
        rb["files"]["preview_png"], png] + [x[1]["files"]["preview_png"] for x in runs.values()]
check("all PNG files start with the PNG signature", all(Path(p).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in pngs), pngs)
check("results are JSON", all(json.dumps(x) for x in [r, r2, rv, rb] + [x[1] for x in runs.values()]))
print("FAILED" if bad else "ALL PASS", bad)
