# Tests for the bulk moduli of md_agent. PYTHONPATH=<folder with ase> python -m pipeline.tests.test_md_bulk
import json, math
from tools.md.bulk import md_bulk

bad = 0
def check(name, ok, detail=""):
    global bad
    bad += not ok
    print("ok  " if ok else "FAIL", name, "|", str(detail)[:150])

KEYS = {"youngs_modulus_GPa", "bulk_modulus_GPa", "shear_modulus_GPa", "density_g_cm3", "n_atoms", "seconds", "method"}
a = md_bulk("alumina")
print(a)
check("alumina returns the keys as JSON", set(a) == KEYS and json.loads(json.dumps(a)) == a, sorted(a))
check("alumina returns within 3 minutes", a["seconds"] < 180, a["seconds"])
check("E = 9 K G / (3 K + G)", math.isclose(a["youngs_modulus_GPa"], 9 * a["bulk_modulus_GPa"] * a["shear_modulus_GPa"]
      / (3 * a["bulk_modulus_GPa"] + a["shear_modulus_GPa"]), abs_tol=0.2), a)
check("alumina density within 3% of the X-ray 3.99 g/cm3", abs(a["density_g_cm3"] / 3.99 - 1) < 0.03, a["density_g_cm3"])
check("alumina bulk modulus within 25% of the measured 252 GPa", abs(a["bulk_modulus_GPa"] / 252 - 1) < 0.25, a["bulk_modulus_GPa"])
check("alumina Young modulus within 25% of the measured 400 GPa", abs(a["youngs_modulus_GPa"] / 400 - 1) < 0.25, a["youngs_modulus_GPa"])
check("the method names the potential", "Pedone" in a["method"], a["method"])

s = md_bulk("silicon carbide")
print(s)
check("silicon carbide runs MACE, in its own environment when this one lacks it", set(s) == KEYS and "MACE" in s["method"], s["method"])
check("silicon carbide density within 3% of the X-ray 3.21 g/cm3", abs(s["density_g_cm3"] / 3.21 - 1) < 0.03, s["density_g_cm3"])
check("silicon carbide bulk modulus within 25% of the measured 225 GPa", abs(s["bulk_modulus_GPa"] / 225 - 1) < 0.25, s["bulk_modulus_GPa"])
try:
    md_bulk("silica glass")
    check("silica glass is rejected", False, "no error")
except ValueError as e:
    check("silica glass is rejected with its name", "silica glass" in str(e), e)
print("FAILED" if bad else "ALL PASS", bad)
