# Tests for the FEM tool, on a coarse grid. python -m pipeline.tests.test_fem_tool
import math, os
os.environ.setdefault("FEM_ROWS", "30")  # 30 pixel rows keep each simulation to a few seconds
import csv, json, shutil
from pathlib import Path
from pipeline.geometry import layered_geometry
from tools.fem import fem_fracture

bad = 0
def check(name, ok, detail=""):
    global bad
    bad += not ok
    print("ok  " if ok else "FAIL", name, "|", str(detail)[:150])

shutil.rmtree("runs_test/fem", ignore_errors=True)
geometry = layered_geometry("alumina", "graphite", 210, 20, "runs_test/fem")["geometry_json"]
CONTRACT = {"crack_deflects", "energy_ratio", "criterion", "cohesive_length_over_layer_thickness", "method"}

low = fem_fracture(367, 210, 3.0, 0.15, 2700, geometry)
along = max(low["along_interface_left_um"], low["along_interface_right_um"])
check("outputs hold the contract keys and are JSON", CONTRACT <= set(low) and json.loads(json.dumps(low)) == low, sorted(low))
check("an interlayer with 5% of the layer energy deflects the crack", low["crack_deflects"] and low["outcome"] == "deflected"
      and along > 210 and low["energy_ratio"] == 0.05, (low["outcome"], along))
check("the method names the model and is not a stand-in", "AT2" in low["method"] and "stand-in" not in low["method"], low["method"])
check("energy balance within 5% and every step converged", low["energy_balance"] < 0.05 and low["converged"], (low["energy_balance"], low["converged"]))
rows = list(csv.DictReader(open(low["load_path_csv"])))
check("the damage image and the load path are written", Path(low["damage_png"]).read_bytes()[:4] == b"\x89PNG"
      and len(rows) > 5 and float(rows[-1]["along_interface_left_um"]) == low["along_interface_left_um"], len(rows))
check("the load rises step by step", all(float(a["K_MPa_sqrt_m"]) < float(b["K_MPa_sqrt_m"]) for a, b in zip(rows, rows[1:])), rows[-1]["K_MPa_sqrt_m"])

high = fem_fracture(367, 210, 3.0, 3.0, 2700, geometry)
check("an interlayer as tough as the layer lets the crack through", not high["crack_deflects"] and high["outcome"] == "penetrated"
      and high["into_next_layer_um"] > 0, (high["outcome"], high["into_next_layer_um"]))
check("the He-Hutchinson verdict is kept for comparison", low["he_hutchinson_deflects"] and not high["he_hutchinson_deflects"])
check("cohesive length is E G / s^2 over the layer thickness", math.isclose(high["cohesive_length_over_layer_thickness"],
      367e9 * 3.0 / 2.7e9 ** 2 / 210e-6, rel_tol=1e-9), high["cohesive_length_over_layer_thickness"])
check("the crack tool returns the geometry file it was given", high["geometry_json"] == geometry)

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
print("FAILED" if bad else "ALL PASS", bad)
