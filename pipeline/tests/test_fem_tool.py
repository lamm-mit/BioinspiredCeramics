# Tests for the FEM tool, on a coarse grid. python -m pipeline.tests.test_fem_tool
import math, os
os.environ.setdefault("FEM_ROWS", "30")  # 30 pixel rows keep each simulation to a few seconds
import csv, json, shutil
from pathlib import Path
import numpy as np
from pipeline.geometry import layered_geometry
from tools.fem import fem_fracture, read_cell
from tools.fem import cell as fem_cell

bad = 0
def check(name, ok, detail=""):
    global bad
    bad += not ok
    print("ok  " if ok else "FAIL", name, "|", str(detail)[:150])

def build(name, pattern, t=210, s=20, interlayer="graphite"):
    design = {"ceramic": "alumina", "interlayer": interlayer, "layer_thickness_um": t, "interlayer_thickness_um": s,
              "layer_modulus_GPa": 367}
    return layered_geometry(design, pattern, f"runs_test/fem/{name}")["geometry_json"]

def runs(line):
    """Lengths of the runs of True in line that touch neither end."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], line.astype(int), [0]])))
    return [b - a for a, b in zip(edges[::2], edges[1::2]) if a > 0 and b < len(line)]

shutil.rmtree("runs_test/fem", ignore_errors=True)
geometry = build("flat", {})
CONTRACT = {"crack_deflects", "energy_ratio", "criterion", "cohesive_length_over_layer_thickness", "method"}

low = fem_fracture(367, 210, 3.0, 0.15, 2700, geometry)
along = max(low["along_interface_left_um"], low["along_interface_right_um"])
check("outputs hold the contract keys and are JSON", CONTRACT <= set(low) and json.loads(json.dumps(low)) == low, sorted(low))
check("an interlayer with 5% of the layer energy deflects the crack", low["crack_deflects"] and low["outcome"] == "deflected"
      and along > 210 and low["energy_ratio"] == 0.05, (low["outcome"], along))
check("the method names the model and is not a stand-in", "AT2" in low["method"] and "stand-in" not in low["method"], low["method"])
check("the method states the limits of a 2D section and of a modulus gradient", "Bouligand" in low["method"]
      and "fracture energy of the ceramic stays uniform" in low["method"], low["method"][-250:])
check("energy balance within 5% and every step converged", low["energy_balance"] < 0.05 and low["converged"], (low["energy_balance"], low["converged"]))
rows = list(csv.DictReader(open(low["load_path_csv"])))
check("the damage image and the load path are written", Path(low["damage_png"]).read_bytes()[:4] == b"\x89PNG"
      and len(rows) > 5 and float(rows[-1]["along_interface_left_um"]) == low["along_interface_left_um"], len(rows))
check("the load rises step by step", all(float(a["K_MPa_sqrt_m"]) < float(b["K_MPa_sqrt_m"]) for a, b in zip(rows, rows[1:])), rows[-1]["K_MPa_sqrt_m"])
check("a flat laminate has no joint in ply 1", low["nearest_joint_um"] is None, low["nearest_joint_um"])
sec = fem_cell.section(read_cell(geometry, 210), 30)
phase = sec["phase"].reshape(30, -1)
check("interlayers of 20 um become bands of 3 pixels of 38 um", {n for j in range(phase.shape[1]) for n in runs(phase[:, j] == 2)} == {3},
      sec["pixel_mm"] * 1000)

high = fem_fracture(367, 210, 3.0, 3.0, 2700, geometry)
check("an interlayer as tough as the layer lets the crack through", not high["crack_deflects"] and high["outcome"] == "penetrated"
      and high["into_next_layer_um"] > 0, (high["outcome"], high["into_next_layer_um"]))
check("the He-Hutchinson verdict is kept for comparison", low["he_hutchinson_deflects"] and not high["he_hutchinson_deflects"])
check("cohesive length is E G / s^2 over the layer thickness", math.isclose(high["cohesive_length_over_layer_thickness"],
      367e9 * 3.0 / 2.7e9 ** 2 / 210e-6, rel_tol=1e-9), high["cohesive_length_over_layer_thickness"])
check("the crack tool returns the geometry file it was given", high["geometry_json"] == geometry)

# Two cases of the validation table of tools/fem/README.md, on its 60-row grid.
coarse, os.environ["FEM_ROWS"] = os.environ["FEM_ROWS"], "60"
for ratio, outcome in [(0.15, "deflected"), (0.35, "penetrated")]:
    out = fem_fracture(367, 210, 3.0, 3.0 * ratio, 2700, geometry)
    check(f"at 60 rows the flat laminate is {outcome} at an energy ratio of {ratio}", out["outcome"] == outcome
          and out["crack_deflects"] == (outcome == "deflected"), (out["outcome"], out["seconds"]))
os.environ["FEM_ROWS"] = coarse

# Tablets 20 layer thicknesses long, with the breaks of ply 1 centered at mid-length of the 45 mm bar.
# The geometry step moves the notch away from that joint. Put back over it, the crack meets the first interlayer
# right under the joint and runs on up it through interlayer only. At this energy ratio the same layers without
# breaks let the crack through.
L, s = 20 * 0.07, 0.035
over = build("tablets_over_notch", {"break_length_h": 20, "break_shift": (-(22.5 - s / 2) / L) % 1,
                                    "interlayer_placement": "shell"}, 70, 35, "PMMA")
spec = json.load(open(over))
moved = spec["cell"]["notch"]["at_mm"]
spec["cell"]["notch"]["at_mm"] = spec["cell"]["size_mm"][0] / 2
Path(over).write_text(json.dumps(spec))
tablets = fem_fracture(367, 70, 3.0, 0.45, 2700, over)
check("the geometry step moves the notch off a joint of ply 1, a quarter tablet or more", abs(moved - spec["cell"]["size_mm"][0] / 2) >= L / 4 - 0.01,
      (moved, spec["cell"]["size_mm"][0] / 2))
flat = fem_fracture(367, 70, 3.0, 0.45, 2700, build("flat_70", {}, 70, 35, "PMMA"))
check("tablets with a joint over the notch send the crack through the joint", tablets["outcome"] == "through joint"
      and not tablets["crack_deflects"] and tablets["into_next_layer_um"] > 35, (tablets["outcome"], tablets["into_next_layer_um"]))
check("the same layers without breaks let the crack through", flat["outcome"] == "penetrated" and not flat["crack_deflects"], flat["outcome"])
check("the joint over the notch is at 0 um", tablets["nearest_joint_um"] == 0, tablets["nearest_joint_um"])
sec = fem_cell.section(read_cell(build("tablets", {"break_length_h": 20, "interlayer_placement": "shell"}, 70, 35, "PMMA"), 70), 30)
check("staggered tablets get the notch a quarter tablet, 350 um, from the joints of plies 0 and 1",
      abs(sec["nearest_joint_px"] * sec["pixel_mm"] * 1000 - 350) < 2 * sec["pixel_mm"] * 1000, sec["nearest_joint_px"] * sec["pixel_mm"])
phase = sec["phase"].reshape(30, -1)
check("joints of 35 um become at least 3 pixels of 16 um wide", min(n for i in range(30) for n in runs(phase[i] == 2)) >= 3)

# A modulus graded from 367 GPa at mid-thickness to 250 GPa at the faces. The bottom plies of the cell are near a face.
graded = build("graded", {"modulus_profile": "linear", "outer_layer_modulus_GPa": 250}, 70, 35, "PMMA")
seen, run = {}, fem_cell.pf.PhaseFieldSolver.run
def spy(self, mat, *args, **kwargs):
    seen["material"] = mat
    return run(self, mat, *args, **kwargs)
fem_cell.pf.PhaseFieldSolver.run = spy
out = fem_fracture(367, 70, 3.0, 0.45, 2700, graded)
fem_cell.pf.PhaseFieldSolver.run = run
sec = fem_cell.section(read_cell(graded, 70), 30)
E, ceramic = seen["material"].E, sec["phase"] == 0
E0, E2 = (np.unique(E[ceramic & (sec["ply"] == k)]) for k in (0, 2))
plies = json.load(open(graded))["plies"]
check("the solver gets the modulus of each ply over the layer modulus", len(E0) == 1 and len(E2) == 1 and E0[0] < E2[0]
      and math.isclose(E0[0], plies[0]["modulus_GPa"] / 367, rel_tol=1e-6) and math.isclose(E2[0], plies[2]["modulus_GPa"] / 367, rel_tol=1e-6),
      (E0 * 367, E2 * 367))
check("the graded cell runs to an outcome", out["outcome"] in {"deflected", "penetrated"} and out["converged"], out["outcome"])

for name, pattern in [("bouligand", {"ply_rotation_deg": 15, "interlayer_placement": "shell"}),
                      ("wavy", {"ply_wave_amplitude_h": 0.5, "ply_wave_length_h": 30, "break_length_h": 20})]:
    out = fem_fracture(367, 70, 3.0, 0.45, 2700, build(name, pattern, 70, 35, "PMMA"))
    check(f"the {name} cell runs to one of the three outcomes with its energy balanced within 5%", out["outcome"] in
          {"deflected", "through joint", "penetrated"} and out["energy_balance"] < 0.05 and out["converged"]
          and out["nearest_joint_um"] is not None, (out["outcome"], out["energy_balance"], out["nearest_joint_um"]))

def raises(name, error, **change):
    args = dict(layer_modulus_GPa=367, layer_thickness_um=210, layer_fracture_energy_J_m2=3.0,
                interface_fracture_energy_J_m2=0.3, interface_strength_MPa=2700, geometry_json=geometry) | change
    try:
        fem_fracture(**args)
        check(name, False, "no error")
    except error as e:
        check(name, True, f"{type(e).__name__}: {e}")

raises("a negative modulus is rejected", ValueError, layer_modulus_GPa=-1)
raises("a zero interface energy is rejected", ValueError, interface_fracture_energy_J_m2=0)
raises("NaN is rejected", ValueError, layer_fracture_energy_J_m2=float("nan"))
raises("a number written as text is rejected", ValueError, interface_strength_MPa="2700")
raises("a missing geometry file is rejected", FileNotFoundError, geometry_json="runs_test/fem/none.json")
raises("a layer thickness the geometry was not built with is rejected", ValueError, layer_thickness_um=100)
Path("runs_test/fem/no_cell.json").write_text(json.dumps({"units": "mm", "stack_axis": "z"}))
raises("a geometry without a cell is rejected", ValueError, geometry_json="runs_test/fem/no_cell.json")
old = json.load(open(geometry))
del old["cell"]["section"]
Path("runs_test/fem/no_section.json").write_text(json.dumps(old))
raises("a geometry without a section file is rejected", ValueError, geometry_json="runs_test/fem/no_section.json")
print("FAILED" if bad else "ALL PASS", bad)
