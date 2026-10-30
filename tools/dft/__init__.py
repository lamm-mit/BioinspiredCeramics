"""Density-functional energy to cleave a ceramic, from Quantum ESPRESSO with PBE and the SSSP 1.3.0 efficiency library.

The energy depends only on the material, so each ceramic is computed once and kept in results/<mp-id>.json, which
later calls read. A new computation relaxes the Materials Project cell of the ceramic and then a slab cut from it on
the plane that tools.materials chose, and takes hours on 8 cores. fracture_md_qe/setup/setup_dft_env.sh installs
pw.x and the pseudopotentials.
"""

import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.io import read

from ..materials import atoms, cleavage_slab, plane, resolve
from ..schema import CERAMIC, spec

HERE = Path(__file__).parent
PSEUDO = HERE / "fracture_md_qe/weights/pseudos/SSSP_1.3.0_PBE_efficiency"
ENV = Path(os.environ.get("FMD_DFT_ENV", Path.home() / "miniforge3/envs/fmd-dft"))
CORES = os.environ.get("FMD_DFT_CORES", "12,13,14,15,16,17,18,19")
J_M2 = 16.02177  # J/m^2 per eV/Å^2
KSPACING = 0.3  # 1/Å, with 2π, enough for energy differences of insulators to a few mJ/m^2
VACUUM = 12.0  # Å between periodic images of the slab

def dft_cleavage(material):
    """Energy per area to split the ceramic on its plane into two relaxed surfaces, twice the surface energy, at 0 K.

    (E_slab - N e_bulk) / A, with e_bulk the energy per atom of the relaxed bulk cell, the slab cut from that cell
    with every atom relaxed, and A the area of one face. The plane is the one of lowest relaxed cleavage energy from
    MACE-MP-0 (tools.materials.plane). Raises ValueError for a material the Materials Project does not have.
    """
    entry = resolve(material)
    cached = HERE / "results" / f"{entry['material_id']}.json"
    if not cached.exists():
        cached.parent.mkdir(exist_ok=True)
        cached.write_text(json.dumps(compute(material, entry, cached.with_suffix("")), indent=1))
    return json.loads(cached.read_text())


def compute(material, entry, folder):
    start = time.time()
    choice = plane(material)
    relaxed = pw(atoms(entry), folder / "bulk", "vc-relax")[-1]
    e_atom = relaxed.get_potential_energy() / len(relaxed)
    slab = cleavage_slab(relaxed, choice)
    steps = pw(slab, folder / "slab", "relax")
    area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
    energy = lambda s: (s.get_potential_energy() - len(slab) * e_atom) / area * J_M2
    shutil.rmtree(folder / "bulk" / "tmp", ignore_errors=True)
    shutil.rmtree(folder / "slab" / "tmp", ignore_errors=True)
    head = (folder / "slab" / "espresso.pwo").read_text().split("\n")
    version = next(line.split()[2] for line in head if "Program PWSCF" in line)
    sssp = json.loads((PSEUDO.parent / "SSSP_1.3.0_PBE_efficiency.json").read_text())
    elements = sorted(set(slab.get_chemical_symbols()))
    plane_name = f"({''.join(map(str, choice['hkl']))}) of {entry['spacegroup']}, shift {choice['shift']:g}"
    return {"cleavage_energy_J_m2": round(energy(steps[-1]), 3),
            "unrelaxed_cleavage_energy_J_m2": round(energy(steps[0]), 3),
            "plane": plane_name, "slab_atoms": len(slab), "slab_thickness_A": round(np.ptp(slab.positions[:, 2]), 2),
            "lattice_A": [round(x, 4) for x in relaxed.cell.cellpar()[:3]],
            "material_id": entry["material_id"], "mace_cleavage_energy_J_m2": choice["mace_cleavage_energy_J_m2"],
            "method": f"Quantum ESPRESSO {version}, PBE, SSSP 1.3.0 efficiency "
                      f"({', '.join(sssp[e]['filename'] for e in elements)}), {cutoffs(elements)[0]:g}/"
                      f"{cutoffs(elements)[1]:g} Ry, k-spacing {KSPACING} 1/Å. Bulk cell of {entry['material_id']} "
                      f"(Materials Project {entry['database_version']}) vc-relaxed, {len(slab)}-atom slab on {plane_name}, "
                      f"the plane of lowest MACE-MP-0 cleavage energy, with {VACUUM:g} Å of vacuum, every atom relaxed, 0 K",
            "pseudopotential_md5": {e: sssp[e]["md5"] for e in elements},
            "computed": time.strftime("%Y-%m-%d"), "wall_time_s": round(time.time() - start)}


def cutoffs(elements):
    sssp = json.loads((PSEUDO.parent / "SSSP_1.3.0_PBE_efficiency.json").read_text())
    return max(sssp[e]["cutoff_wfc"] for e in elements), max(sssp[e]["cutoff_rho"] for e in elements)


def pw(atoms, folder, calculation):
    """Runs pw.x and returns every configuration it wrote, the first unrelaxed and the last relaxed."""
    elements = sorted(set(atoms.get_chemical_symbols()))
    sssp = json.loads((PSEUDO.parent / "SSSP_1.3.0_PBE_efficiency.json").read_text())
    ecut, ecutrho = cutoffs(elements)
    b = np.linalg.norm(atoms.cell.reciprocal(), axis=1) * 2 * math.pi
    kpts = [max(1, math.ceil(x / KSPACING)) for x in b]
    if calculation == "relax":
        kpts[2] = 1
    n = len(CORES.split(","))
    command = (f"{ENV}/bin/mpirun -np {n} --map-by pe-list={CORES}:ordered --bind-to core --mca pml ob1 "
               f"--mca btl self,sm -x OMP_NUM_THREADS=1 {ENV}/bin/pw.x -nk 4")
    data = {"control": {"calculation": calculation, "outdir": "./tmp", "disk_io": "low", "tprnfor": True,
                        "tstress": True, "etot_conv_thr": 1e-5 * len(atoms), "forc_conv_thr": 1e-3, "nstep": 150},
            "system": {"ecutwfc": ecut, "ecutrho": ecutrho, "occupations": "smearing", "smearing": "mv", "degauss": 0.01},
            # Local Thomas-Fermi mixing damps the charge sloshing along the vacuum of a slab.
            "electrons": {"conv_thr": 1e-9 * len(atoms), "mixing_mode": "local-TF", "mixing_beta": 0.3,
                          "electron_maxstep": 200},
            "ions": {"ion_dynamics": "bfgs"}, "cell": {"cell_dynamics": "bfgs", "press_conv_thr": 0.1}}
    atoms = atoms.copy()
    atoms.calc = Espresso(profile=EspressoProfile(command=command, pseudo_dir=str(PSEUDO)), directory=str(folder),
                          pseudopotentials={e: sssp[e]["filename"] for e in elements}, input_data=data, kpts=kpts)
    atoms.get_potential_energy()
    return read(folder / "espresso.pwo", index=":")


SPEC = spec("dft_cleavage",
            "Density-functional (PBE) energy to cleave the bulk ceramic into two relaxed surfaces, at 0 K. Returns the "
            "cleavage energy in J/m^2, twice the surface energy.",
            {"material": CERAMIC | {"description": "the ceramic of the design"}})
