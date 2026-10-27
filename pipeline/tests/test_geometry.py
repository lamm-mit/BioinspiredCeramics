# Tests for the toolpath pattern and the geometry. python -m pipeline.tests.test_geometry
import json, math, shutil
import numpy as np
from pipeline.geometry import box, layered_geometry, read_stl, volume, voxel_surface, watertight
from pipeline.pattern import material, pattern_errors, plies, roads

bad = 0
def check(name, ok, detail=""):
    global bad
    bad += not ok
    print("ok  " if ok else "FAIL", name, "|", str(detail)[:150])

b = box(0, 0, 0, 2, 3, 4)
check("box has 12 triangles and volume 24", len(b) == 12 and math.isclose(volume(b), 24), volume(b))
check("box is closed", watertight(b))
check("box with one triangle removed is not closed", not watertight(b[:-1]))
check("box with one triangle turned inside out is not closed", not watertight(np.concatenate([b[:1, ::-1], b[1:]])))
check("two separate boxes are closed", watertight(np.concatenate([b, box(0, 0, 5, 2, 3, 6)])))
check("two boxes that share only an edge are closed, though the edge is non-manifold", watertight(np.concatenate([box(0, 0, 0, 1, 1, 1), box(1, 1, 0, 2, 2, 1)])))
v = voxel_surface(np.array([[[1], [0]], [[0], [1]]], bool), (0, 0, 0), 0.5)
check("two voxels touching along an edge give a closed surface of their volume", watertight(v) and math.isclose(volume(v), 2 * 0.5 ** 3), volume(v))

# Plies of 80 µm with 20 µm interlayers make a road width of 120 µm and a cell of 16 road widths, so the section
# holds whole periods of shell lines across it and of 1.6 mm tablets along it, and every boundary falls between pixels.
D = {"ceramic": "alumina", "interlayer": "PMMA", "layer_thickness_um": 80, "interlayer_thickness_um": 20, "layer_modulus_GPa": 367,
     "process": "direct ink writing"}
h, s = 0.08, 0.02
pitch = h + s
n, height = 5, 5 * pitch - s
rng = np.random.default_rng(0)
x, y, z = rng.uniform(0, 10, 20000), rng.uniform(0, 4, 20000), rng.uniform(0, height, 20000)
phase, ply, modulus = material({}, D, n, height, x, y, z)
check("the flat pattern is the analytic laminate, ceramic where z mod pitch < layer thickness",
      ((phase == 0) == (np.mod(z, pitch) < h)).all() and (ply == np.floor(z / pitch)).all() and (modulus == 367).all())

shutil.rmtree("runs_test/geometry", ignore_errors=True)
flat = h * n / height
for name, pattern, want in [("flat", {}, flat), ("shell lines across the section", {"road_angle_deg": 90, "interlayer_placement": "shell"}, flat * (1.2 * pitch - s) / (1.2 * pitch)),
                            ("tablets", {"break_length_h": 20}, flat * (20 * h - s) / (20 * h))]:
    r = layered_geometry(D, pattern, f"runs_test/geometry/{name.split()[0]}")
    spec = json.load(open(r["geometry_json"]))
    got = float((np.load(spec["cell"]["section"])["phase"] == 0).mean())
    check(f"ceramic fraction of the section, {name}", abs(got - want) < 1e-3 and spec["cell"]["ceramic_fraction"] == round(got, 4), (got, want))
    parts = {k: read_stl(f) for k, f in spec["cell"]["files"].items()}
    shape = [round(spec["cell"]["size_mm"][a] / spec["cell"]["voxel_mm"]) for a in range(3)]
    check(f"voxel STL files read back closed and fill the block, {name}", all(watertight(t) and volume(t) > 0 for t in parts.values())
          and math.isclose(sum(volume(t) for t in parts.values()), np.prod(shape) * spec["cell"]["voxel_mm"] ** 3, rel_tol=1e-4),
          {k: (len(t), round(volume(t), 5)) for k, t in parts.items()})
outline = read_stl(spec["bar"]["files"]["outline"])
check("bar outline is a closed box of whole plies", watertight(outline) and math.isclose(volume(outline), 45 * 4 * spec["bar"]["size_mm"][2], rel_tol=1e-5)
      and r["bar_ceramic_layers"] == int((3 + s) // pitch) == 30 and math.isclose(r["bar_height_mm"], 30 * pitch - s), r["bar_height_mm"])
check("cell has five plies and a notch half a ply deep in the bottom ply", spec["cell"]["ceramic_layers"] == 5 and spec["cell"]["notch"]["depth_mm"] == h / 2, spec["cell"]["notch"])
check("same design, pattern and folder give the same result", layered_geometry(D, {"break_length_h": 20}, "runs_test/geometry/tablets") == r and json.dumps(r))
paths = roads({"break_length_h": 20}, D, n, height, 0, (45, 4))["ceramic"]
got = sum(np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1).sum() for q in paths)
rows = len({round(float(q[0, 1]), 6) for q in paths})
want = rows * (45 - 1.2 * pitch) * (1 - s / (20 * h))
check("the roads of a tablet ply hold every tablet, as long as the roads less the breaks", math.isclose(got, want, rel_tol=2e-3),
      f"{got:.2f} of {want:.2f} mm in {len(paths)} segments of {rows} roads")

# Every road of the bottom two plies lies in ceramic, half a ply below the top where the nozzle runs.
d = D | {"layer_thickness_um": 70, "interlayer_thickness_um": 35}
PATTERNS = {"flat": {}, "nacre": {"break_length_h": 20, "interlayer_placement": "shell"},
            "Bouligand": {"ply_rotation_deg": 15, "interlayer_placement": "shell"},
            "crossed": {"road_angle_deg": -30, "ply_rotation_deg": 60, "ply_rotation_alternates": True, "interlayer_placement": "shell"},
            "rings": {"road_path": "rings", "ring_spacing_w": 8, "interlayer_placement": "shell"},
            "waves": {"road_path": "waves", "wave_amplitude_w": 1, "wave_length_w": 10, "interlayer_placement": "shell"},
            "wavy plies": {"ply_wave_amplitude_h": 0.5, "ply_wave_length_h": 30, "break_length_h": 20},
            "graded": {"modulus_profile": "linear", "outer_layer_modulus_GPa": 250}}
check("the example patterns pass their checks", all(pattern_errors(p, d) == {} for p in PATTERNS.values()),
      {k: pattern_errors(p, d) for k, p in PATTERNS.items() if pattern_errors(p, d)})
height = 9 * 0.105 - 0.035
for name, p in PATTERNS.items():
    points = np.concatenate([q for k in (0, 1) for q in roads(p, d, 9, height, k, (6, 4))["ceramic"]])
    phase = material(p, d, 9, height, points[:, 0], points[:, 1], points[:, 2] - 0.035)[0]
    check(f"roads of plies 0 and 1 lie in ceramic, {name}", len(points) > 100 and (phase == 0).all(), f"{(phase != 0).sum()} of {len(points)} points outside")

wrong = {"too short a wavelength": ({"road_path": "waves", "wave_amplitude_w": 1, "wave_length_w": 2, "interlayer_placement": "shell"}, d, "wave_length_w"),
         "too short a break": ({"break_length_h": 2}, d, "break_length_h"),
         "too steep a ply wave": ({"ply_wave_amplitude_h": 1, "ply_wave_length_h": 5}, d, "ply_wave_length_h"),
         "sheet placement with only a ply rotation": ({"ply_rotation_deg": 15}, d, "interlayer_placement"),
         "tablets by vat photopolymerization": ({"break_length_h": 20}, d | {"process": "vat photopolymerization"}, "interlayer_placement"),
         "an alternation without a rotation": ({"ply_rotation_alternates": True}, d, "ply_rotation_alternates"),
         "an outer modulus without a profile": ({"outer_layer_modulus_GPa": 300}, d, "outer_layer_modulus_GPa")}
for name, (p, design, key) in wrong.items():
    e = pattern_errors(p, design)
    check(f"pattern check catches {name}", list(e) == [key], e)

rows = plies(PATTERNS["graded"], d, 27)
E = [row["modulus_GPa"] for row in rows]
check("graded plies are symmetric, softest at the faces within one ply step of 250 GPa, and 367 GPa in the middle ply",
      E == E[::-1] and E[0] == min(E) and abs(E[0] - 250) <= (367 - 250) / 27 and E[13] == 367 and E[:14] == sorted(E[:14]), E[:3])
print("FAILED" if bad else "ALL PASS", bad)
