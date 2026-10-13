# Tests for the geometry. python -m pipeline.tests.test_geometry
import json, math, shutil
from pathlib import Path
from pipeline.geometry import box, layered_geometry, volume, watertight

bad = 0
def check(name, ok, detail=""):
    global bad
    bad += not ok
    print("ok  " if ok else "FAIL", name, "|", str(detail)[:150])

b = box(0, 0, 0, 2, 3, 4)
check("box has 12 triangles and volume 24", len(b) == 12 and math.isclose(volume(b), 24), volume(b))
check("box is closed", watertight(b))
check("box with one triangle removed is not closed", not watertight(b[:-1]))
flipped = [b[0][::-1]] + b[1:]
check("box with one triangle turned inside out is not closed", not watertight(flipped))
check("two separate boxes are closed", watertight(b + box(0, 0, 5, 2, 3, 6)))

shutil.rmtree("runs_test", ignore_errors=True)
r = layered_geometry("alumina", "graphite", 210, 20, "runs_test/a")
spec = json.load(open(r["geometry_json"]))
layer, gap = 0.21, 0.02
n = int((3 + gap) // (layer + gap))
check("layer count fills the bar height", r["bar_ceramic_layers"] == n == 13 and r["bar_height_mm"] <= 3, (r["bar_ceramic_layers"], r["bar_height_mm"]))
check("bar height is whole layers", math.isclose(r["bar_height_mm"], n * layer + (n - 1) * gap, rel_tol=1e-9), r["bar_height_mm"])
check("ceramic volume is analytic", math.isclose(spec["bar"]["volume_mm3"]["ceramic"], 45 * 4 * layer * n, rel_tol=1e-6), spec["bar"]["volume_mm3"])
check("interlayer volume is analytic", math.isclose(spec["bar"]["volume_mm3"]["interlayer"], 45 * 4 * gap * (n - 1), rel_tol=1e-6))
check("cell has five ceramic layers and a notch in the bottom layer", spec["cell"]["ceramic_layers"] == 5 and spec["cell"]["notch"]["depth_mm"] == layer / 2, spec["cell"]["notch"])
check("smallest feature is the interlayer", r["smallest_feature_um"] == 20)
check("all four STL files exist and start as STL", all(Path(r[k]).read_text().startswith("solid ") for k in r if k.endswith("_stl")))
tri = Path(r["bar_ceramic_stl"]).read_text().count("facet normal")
check("STL triangle count matches", tri + Path(r["bar_interlayer_stl"]).read_text().count("facet normal") == spec["bar"]["triangles"] == 12 * (2 * n - 1), tri)
normals = {l.strip() for l in Path(r["bar_ceramic_stl"]).read_text().splitlines() if "facet normal" in l}
check("facet normals are the six outward axis directions", normals == {f"facet normal {x} {y} {z}" for x, y, z in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]}, sorted(normals))
r2 = layered_geometry("alumina", "graphite", 210, 20, "runs_test/a")
check("same design and folder give the same files", r2 == r)
r3 = layered_geometry("alumina", "PMMA", 50, 2, "runs_test/b")
check("thin layers, 57 ceramic layers in a 3 mm bar", r3["bar_ceramic_layers"] == 57 and r3["geometry_json"] != r["geometry_json"], r3["bar_ceramic_layers"])
json.dumps(r)
print("FAILED" if bad else "ALL PASS", bad)
