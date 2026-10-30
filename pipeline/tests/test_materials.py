# Tests for the Materials Project resolver and the plane choice. python -m pipeline.tests.test_materials
# Reads the cache in tools/results/structures. A material missing from it is fetched, which needs MP_API_KEY.
from collections import Counter

import numpy as np
from ase.io import read

from tools.dft import HERE as DFT, dft_cleavage
from tools.materials import atoms, cleavage_slab, plane, resolve

bad = 0
def check(name, ok, detail=""):
    global bad
    bad += not ok
    print("ok  " if ok else "FAIL", name, "|", str(detail)[:150])

a, s = resolve("alumina"), resolve("silicon carbide")
check("alumina is corundum, mp-1143, on the hull", (a["material_id"], a["spacegroup"], a["energy_above_hull_eV_atom"]) == ("mp-1143", "R-3c", 0.0), a["material_id"])
check("silicon carbide is the cubic 3C polytype the tools were built on", (s["material_id"], s["spacegroup"]) == ("mp-8062", "F-43m"), s["material_id"])
check("a formula and a common name give the same entry", resolve("Al2O3")["material_id"] == a["material_id"])
c = atoms(a, conventional=True).cell.cellpar()
check("the conventional corundum cell is hexagonal with a within 1.5% of the measured 4.759 A", abs(c[0] / 4.759 - 1) < 0.015 and round(c[5]) == 120, c.round(3))

pa, ps = plane("alumina"), plane("silicon carbide")
check("MACE chooses (0001) of alumina, below (11-20), which cuts fewer bonds", pa["hkl"] == [0, 0, 1] and pa["candidates"][1]["hkl"] == [1, 1, 0], [r["hkl"] for r in pa["candidates"]])
check("MACE chooses (110) of silicon carbide", ps["hkl"] == [1, 1, 0], ps["hkl"])
for name, entry, p, energy in [("alumina", a, pa, 3.107), ("silicon carbide", s, ps, 5.815)]:
    check(f"the MACE energy of {name} is within 10% of PBE", abs(p["mace_cleavage_energy_J_m2"] / energy - 1) < 0.1, p["mace_cleavage_energy_J_m2"])
    folder = DFT / "results" / entry["material_id"]
    new, old = cleavage_slab(read(folder / "bulk/espresso.pwo", index=-1), p), read(folder / "slab/espresso.pwo", index=0)
    area = lambda x: np.linalg.norm(np.cross(x.cell[0], x.cell[1]))
    faces = lambda x: [Counter(np.array(x.get_chemical_symbols())[f(x.positions[:, 2]) ]) for f in
                       (lambda z: z > z.max() - 0.3, lambda z: z < z.min() + 0.3)]
    check(f"the {name} slab from the relaxed bulk is the one the DFT result was computed on",
          Counter(new.get_chemical_symbols()) == Counter(old.get_chemical_symbols()) and np.isclose(area(new), area(old), rtol=1e-4)
          and np.isclose(np.ptp(new.positions[:, 2]), np.ptp(old.positions[:, 2]), atol=0.05) and faces(new) == faces(old),
          (len(new), round(area(new), 3), faces(new)))
    r = dft_cleavage(name)
    check(f"dft_cleavage reads {name} from its mp-id", r["material_id"] == entry["material_id"] and r["cleavage_energy_J_m2"] == energy, r["cleavage_energy_J_m2"])
for bad_name in ["Xx9", "Al7O2"]:
    try:
        resolve(bad_name)
        check(f"{bad_name} is rejected", False, "no error")
    except ValueError as e:
        check(f"{bad_name} is rejected with a ValueError", True, e)
print("FAILED" if bad else "ALL PASS", bad)
