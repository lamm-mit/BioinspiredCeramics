"""Crystal structures of ceramics from the Materials Project, and the plane a ceramic cleaves on.

resolve() turns a common name or a formula into the Materials Project entry of the ceramic, the polymorph lowest above
the convex hull unless NAMES fixes another. Each entry is kept in results/structures/<mp-id>.json, and index.json maps
each query to its mp-id, so a material is fetched once and later runs need no network. The key is MP_API_KEY, from the
environment or from a .env file in the repository or the home folder.

plane() picks the plane and termination with the lowest relaxed cleavage energy from MACE-MP-0, among the nonpolar,
symmetric and stoichiometric slabs of every plane up to Miller index 1, and keeps it in results/structures/
<mp-id>_plane.json. A bond count does not do: it ranks (0001) of alumina below (11-20), while relaxation, which sinks
the surface Al by 0.74 Å in DFT, makes (0001) the lowest. MACE runs in its own environment (SIMTOOLS_PYTHON),
through python -m tools.materials <material>.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import dotenv_values
from pymatgen.core import Composition, Structure
from pymatgen.core.surface import SlabGenerator, get_symmetrically_distinct_miller_indices
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

HERE = Path(__file__).parent
STORE = HERE / "results" / "structures"
J_M2 = 16.02177  # J/m^2 per eV/Å^2
# Common names, with the space group of the polymorph meant where it is not the one lowest above the hull. The DFT and
# MD tools were built on cubic 3C silicon carbide.
NAMES = {"alumina": ("Al2O3", None), "silicon carbide": ("SiC", "F-43m"), "zirconia": ("ZrO2", None),
         "silicon nitride": ("Si3N4", None), "aluminum nitride": ("AlN", None), "boron nitride": ("BN", None),
         "titania": ("TiO2", None), "magnesia": ("MgO", None)}
FIELDS = ["material_id", "formula_pretty", "symmetry", "nsites", "energy_above_hull", "density", "bulk_modulus",
          "shear_modulus", "structure"]


def api_key():
    files = [dotenv_values(f) for f in (HERE.parent / ".env", Path.home() / ".env") if f.exists()]
    return os.environ.get("MP_API_KEY") or next((f["MP_API_KEY"] for f in files if f.get("MP_API_KEY")), None)


def resolve(material):
    """The Materials Project entry of a ceramic named by its common name or its formula, fetched at its first use."""
    index_file = STORE / "index.json"
    index = json.loads(index_file.read_text()) if index_file.exists() else {}
    if material not in index:
        from mp_api.client import MPRester

        formula, group = NAMES.get(material, (material, None))
        with MPRester(api_key(), mute_progress_bars=True) as mpr:
            docs = mpr.materials.summary.search(formula=formula, deprecated=False, fields=FIELDS)
            version = mpr.db_version
        # A formula the search cannot parse returns unrelated entries, so the reduced formula must match.
        reduced = Composition(formula).reduced_formula
        docs = [d for d in docs if d.formula_pretty == reduced and group in (None, d.symmetry.symbol)]
        if not docs:
            raise ValueError(f"the Materials Project has no {formula}" + (f" in space group {group}" if group else ""))
        d = min(docs, key=lambda d: (d.energy_above_hull, d.nsites))
        vrh = lambda m: None if m is None else round(m["vrh"], 1)
        entry = {"material_id": str(d.material_id), "formula": d.formula_pretty, "spacegroup": d.symmetry.symbol,
                 "energy_above_hull_eV_atom": d.energy_above_hull, "density_g_cm3": round(d.density, 3),
                 "bulk_modulus_GPa": vrh(d.bulk_modulus), "shear_modulus_GPa": vrh(d.shear_modulus),
                 "database_version": version, "retrieved": time.strftime("%Y-%m-%d"), "structure": d.structure.as_dict()}
        STORE.mkdir(parents=True, exist_ok=True)
        (STORE / f"{d.material_id}.json").write_text(json.dumps(entry, indent=1))
        index[material] = str(d.material_id)
        index_file.write_text(json.dumps(index, indent=1, sort_keys=True))
    return json.loads((STORE / f"{index[material]}.json").read_text())


def atoms(entry, conventional=False):
    """The structure of an entry as ASE Atoms, as the Materials Project stores it or as the conventional standard cell."""
    s = Structure.from_dict(entry["structure"])
    return AseAtomsAdaptor.get_atoms(SpacegroupAnalyzer(s).get_conventional_standard_structure() if conventional else s)


def slabs(bulk, hkls=None, thickness=10.0, vacuum=12.0):
    """The nonpolar, symmetric, stoichiometric slabs of a bulk cell, as (Miller indices, shift, ASE Atoms).

    Miller indices refer to the conventional standard cell, and the shift picks the termination. Each slab is at least
    thickness Å thick, with vacuum Å above it and its normal along z.
    """
    s = SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(bulk), symprec=0.1).get_conventional_standard_structure()
    out = []
    for hkl in hkls or get_symmetrically_distinct_miller_indices(s, 1):
        generator = SlabGenerator(s, hkl, thickness, vacuum, center_slab=True, primitive=True,
                                  max_normal_search=max(map(abs, hkl)))
        for slab in generator.get_slabs():
            if not slab.is_polar() and slab.is_symmetric() and slab.composition.reduced_formula == s.composition.reduced_formula:
                out.append((tuple(int(i) for i in hkl), round(float(slab.shift), 6),
                            AseAtomsAdaptor.get_atoms(slab.get_orthogonal_c_slab())))
    return out


def plane(material):
    """The plane, as Miller indices and shift, with the lowest relaxed cleavage energy from MACE-MP-0, and every
    candidate it was chosen from. Computed once per material, in the environment that has MACE."""
    path = STORE / f"{resolve(material)['material_id']}_plane.json"
    if not path.exists():
        python = os.environ.get("SIMTOOLS_PYTHON", str(Path.home() / "simtools/env/bin/python"))
        subprocess.run([python, "-m", "tools.materials", material], cwd=HERE.parent, check=True)
    return json.loads(path.read_text())


def cleavage_slab(bulk, choice):
    """The slab of the chosen plane cut from a bulk cell, the termination whose shift is nearest the chosen one."""
    hkl, shift = tuple(choice["hkl"]), choice["shift"]
    return min(slabs(bulk, [hkl]), key=lambda c: abs(c[1] - shift))[2]


def rank(material):
    """Relaxed and unrelaxed cleavage energies of every candidate slab from MACE-MP-0 medium, which writes plane()."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS
    from mace.calculators import mace_mp

    start = time.time()
    entry = resolve(material)
    calc = mace_mp(model="medium", default_dtype="float64", device="cuda")
    bulk = atoms(entry)
    bulk.calc = calc
    BFGS(FrechetCellFilter(bulk), logfile=None).run(fmax=0.005, steps=500)
    e_atom = bulk.get_potential_energy() / len(bulk)
    rows = []
    for hkl, shift, slab in slabs(bulk):
        slab.calc = calc
        area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
        energy = lambda: round(float((slab.get_potential_energy() - len(slab) * e_atom) / area * J_M2), 3)
        unrelaxed = energy()
        BFGS(slab, logfile=None).run(fmax=0.02, steps=500)
        rows.append({"hkl": hkl, "shift": shift, "atoms": len(slab), "unrelaxed_J_m2": unrelaxed, "relaxed_J_m2": energy()})
    best = min(rows, key=lambda r: r["relaxed_J_m2"])
    out = {"hkl": best["hkl"], "shift": best["shift"], "mace_cleavage_energy_J_m2": best["relaxed_J_m2"],
           "candidates": sorted(rows, key=lambda r: r["relaxed_J_m2"]),
           "method": "MACE-MP-0 medium, float64. Bulk cell relaxed with its cell, every nonpolar, symmetric, stoichiometric "
                     "slab up to Miller index 1 of the conventional cell, at least 10 Å thick with 12 Å of vacuum, "
                     "relaxed at fixed cell to 0.02 eV/Å, 0 K",
           "computed": time.strftime("%Y-%m-%d"), "wall_time_s": round(time.time() - start)}
    (STORE / f"{entry['material_id']}_plane.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    rank(sys.argv[1])
