"""Separation of the interface between a ceramic and an interlayer, by molecular dynamics with MACE-MP-0 and D3.

python -m tools.md.interface <ceramic> <interlayer> runs in an environment with mace-torch, torch-dftd and rdkit
(~/simtools/env on the Spark) and writes results/<ceramic>__<interlayer>.json, which md_interface reads.

The cell is a ceramic slab about 7 Å thick under a 1 nm interlayer. Graphite is two graphene sheets strained onto the
ceramic cell. Silica glass is melted on the ceramic and quenched. Epoxy and PMMA are short molecules packed on the
ceramic at their density. Each interlayer is annealed on the ceramic, held at 300 K and relaxed at 0 K. The cell is
then split in two places, at the interface and through the middle of the interlayer, and the weaker split is where
a crack runs.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.build import graphene, make_supercell, surface
from ase.constraints import FixAtoms, Hookean
from ase.md.langevin import Langevin
from ase.neighborlist import natural_cutoffs, neighbor_list
from ase.optimize import FIRE
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..materials import atoms, resolve

HERE = Path(__file__).parent
J_M2 = 16.02177  # J/m^2 per eV/Å^2
MPA = 160217.7  # MPa per eV/Å^3
THICKNESS = 10.0  # Å of interlayer
DENSITY = {"silica glass": 2.2, "epoxy": 1.2, "PMMA": 1.18}  # g/cm^3
# Cured epoxy as the adduct of bisphenol A diglycidyl ether with two methylamines, which carries the hydroxyl, ether
# and amine groups of the network. PMMA as a 5-mer with hydrogen ends.
SMILES = {"epoxy": "CNCC(O)COc1ccc(cc1)C(C)(C)c1ccc(cc1)OCC(O)CNC", "PMMA": "C" + "CC(C)(C(=O)OC)" * 5}


def cut(cell, between):
    """The cell shifted along its third axis so that the cut lies at the cell boundary, with 6 Å of vacuum on each side.
    The cut is the midpoint of the widest gap between two neighboring atomic planes that both hold only `between`."""
    cell = cell.copy()
    cell.wrap()
    h = cell.cell[2, 2]
    z = cell.positions[:, 2]
    planes = np.unique(np.round(z, 1))
    symbols = [set(np.array(cell.get_chemical_symbols())[np.abs(z - p) < 0.06]) for p in planes]
    gaps = np.diff(np.append(planes, planes[0] + h))
    ok = [i for i in range(len(planes)) if symbols[i] == symbols[(i + 1) % len(planes)] == {between}]
    i = max(ok, key=lambda i: gaps[i])
    cell.positions[:, 2] -= planes[i] + gaps[i] / 2
    cell.wrap()
    cell.center(vacuum=6.0, axis=2)
    return cell


def ceramic(name):
    """A slab of the ceramic about 7 Å thick, Al-terminated (0001) for alumina and (110) for silicon carbide, in an
    orthogonal cell whose in-plane unit is returned for matching a crystalline interlayer. The bulk cell is the
    Materials Project one (tools.materials), at its PBE lattice."""
    if name == "alumina":
        unit = make_supercell(atoms(resolve(name), conventional=True), [[1, 0, 0], [1, 2, 0], [0, 0, 1]])
        slab = cut(unit, "Al")
        # Keep the upper half, from the gap between the two Al sublayers nearest the middle.
        z = slab.positions[:, 2]
        al = np.unique(np.round(z[np.array(slab.get_chemical_symbols()) == "Al"], 1))
        pairs = [(a + b) / 2 for a, b in zip(al, al[1:]) if b - a < 0.7]
        slab = slab[z > min(pairs, key=lambda m: abs(m - z.mean()))]
    else:
        slab = surface(atoms(resolve(name), conventional=True), (1, 1, 0), 3, periodic=True)
    slab.positions[:, 2] -= slab.positions[:, 2].min()
    slab.set_tags(0)  # the interlayer carries tags above 0
    return slab, slab.cell.lengths()[:2]


def fit(unit, lattice, lo=12.0, hi=17.5):
    """Ceramic and interlayer repeats in x and y whose lengths agree best, with the ceramic cell between lo and hi Å.
    Returns the two repeats and the strain the interlayer takes to fit."""
    best = None
    for nx in range(1, 10):
        for ny in range(1, 10):
            L = np.array([nx, ny]) * unit
            if L.min() < lo or L.max() > hi:
                continue
            m = np.maximum(1, np.round(L / lattice)).astype(int)
            strain = L / (m * lattice) - 1
            if best is None or abs(strain).max() < abs(best[2]).max():
                best = ((nx, ny), tuple(m), strain)
    return best


def graphite(cell, m):
    """Two AB-stacked graphene sheets of m orthogonal cells, strained onto the cell, as tags 1 and 2."""
    sheet = make_supercell(graphene("C2", a=2.46, vacuum=0), [[1, 0, 0], [1, 2, 0], [0, 0, 1]]).repeat((*m, 1))
    upper = sheet.copy()
    upper.positions += [0, 2.46 / np.sqrt(3), 3.35]
    layers = sheet + upper
    layers.set_tags([1] * len(sheet) + [2] * len(upper))
    layers.set_cell([cell[0], cell[1], layers.cell[2]], scale_atoms=True)
    return layers


def molecules(name, cell, z0, rng):
    """Copies of the molecule at their density, rotated at random and placed in a layer three times the final thickness
    so that they fit, one tag per molecule."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES[name]))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    xyz = mol.GetConformer().GetPositions()
    xyz -= xyz.mean(0)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    n = round(DENSITY[name] * area * THICKNESS * 0.6022 / Descriptors.MolWt(mol))
    placed = Atoms(cell=cell, pbc=True)
    for tag in range(1, n + 1):
        for _ in range(5000):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            new = Atoms(symbols, xyz @ q.T + [*rng.uniform(0, 1, 2) @ cell[:2, :2], 0])
            new.positions[:, 2] += z0 + 2 - new.positions[:, 2].min() + rng.uniform(0, 3 * THICKNESS - np.ptp(new.positions[:, 2]))
            if new.positions[:, 2].max() > z0 + 3 * THICKNESS + 4:
                continue
            new.set_cell(cell)
            new.pbc = True
            new.set_tags(tag)
            both = placed + new
            d = both.get_all_distances(mic=True)[len(placed):, :len(placed)]
            # Other molecules and the periodic images of the same molecule stay 2 Å away.
            if d.min(initial=9) > 2.0 and min_image_self(new) > 2.0:
                placed = both
                break
        else:
            raise RuntimeError(f"could not place {name} molecule {tag} of {n}")
    return placed


def min_image_self(atoms):
    """Shortest distance between an atom and an atom of a periodic image of the same molecule in x or y."""
    p = atoms.positions
    shifts = [i * atoms.cell[0] + j * atoms.cell[1] for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
    return min(np.linalg.norm(p[:, None] - (p + s)[None], axis=2).min() for s in shifts)


def glass(cell, z0, rng):
    """Si and O at random in the layer, at the density of silica glass, with no two atoms within 1.6 Å. The melt and
    quench on the ceramic make the glass."""
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    n = round(DENSITY["silica glass"] * area * THICKNESS * 0.6022 / 60.08)
    pos = []
    while len(pos) < 3 * n:
        p = np.append(rng.uniform(0, 1, 2) @ cell[:2, :2], z0 + 2 + rng.uniform(0, THICKNESS))
        trial = Atoms("X" * (len(pos) + 1), pos + [p], cell=cell, pbc=True)
        if not pos or trial.get_distances(len(pos), range(len(pos)), mic=True).min() > 1.6:
            pos.append(p)
    layer = Atoms("Si" * n + "O" * 2 * n, pos, cell=cell, pbc=True)
    layer.set_tags(1)
    return layer


def build(name_c, name_i, calc, rng, log):
    slab, unit = ceramic(name_c)
    if name_i == "graphite":
        (nx, ny), m, strain = fit(unit, np.array([2.46, 2.46 * np.sqrt(3)]))
    else:  # an amorphous interlayer fits any cell, so the largest, in which a 20 Å molecule finds room
        nx, ny = (17.5 // unit).astype(int)
        m, strain = None, None
    slab = slab.repeat((nx, ny, 1))
    top = slab.positions[:, 2].max()
    cell = np.diag([*slab.cell.lengths()[:2], top + 3 * THICKNESS + 24])
    slab.set_cell(cell)
    if name_i == "graphite":
        layer = graphite(cell, m)
        layer.positions[:, 2] += top + 3.0
    elif name_i == "silica glass":
        layer = glass(cell, top, rng)
    else:
        layer = molecules(name_i, cell, top, rng)
    atoms = slab + layer
    atoms.pbc = True
    atoms.calc = calc
    bottom = np.flatnonzero(atoms.positions[:, 2] < 2.0)
    masses = atoms.get_masses()
    masses[np.array(atoms.get_chemical_symbols()) == "H"] = 2.0  # deuterium mass, so that 1 fs steps are stable
    atoms.set_masses(masses)
    interlayer = np.flatnonzero(atoms.get_tags() > 0)
    # A soft ceiling holds the interlayer down, the glass at its density while it melts and the molecules while it falls
    # from three times the thickness to the thickness.
    ceiling = lambda z: [Hookean(a1=int(i), a2=[0, 0, 1, -z], k=5.0) for i in interlayer]
    ceiling_z = top + 2 + (THICKNESS if name_i == "silica glass" else 3 * THICKNESS + 2)
    atoms.set_constraint([FixAtoms(bottom)] + ceiling(ceiling_z))
    FIRE(atoms, maxstep=0.1, logfile=None).run(fmax=1.0, steps=200)
    # A coupling time of 50 fs. At 500 fs the polymers packed at 500 K reached 1900 to 2600 K, and the glass ended its
    # quench to 300 K at 1100 K.
    md = Langevin(atoms, 1 * units.fs, temperature_K=300, friction=0.02 / units.fs, rng=rng)
    schedule = {"graphite": [(300, 0, 500)],
                "silica glass": [(3500, 0, 1000)] + [(t, 0, 200) for t in np.linspace(3200, 300, 10)],
                "epoxy": [(500, 1, 300) for _ in range(10)] + [(300, 0, 500)],
                "PMMA": [(500, 1, 300) for _ in range(10)] + [(300, 0, 500)]}[name_i]
    drop = (2 * THICKNESS + 2) / 10  # Å per lowering segment, ten of them down to the interlayer thickness
    for temperature, lower, steps in schedule:
        md.set_temperature(temperature_K=temperature)
        for _ in range(steps // 10):
            # The ceiling falls 0.07 Å every 10 fs. Dropped 2.2 Å at once, it stored up to 12 eV in each atom above it
            # and broke the molecules into 21 to 40 pieces.
            if lower:
                ceiling_z -= drop * 10 / steps
                atoms.set_constraint([FixAtoms(bottom)] + ceiling(ceiling_z))
            md.run(10)
        log(f"{name_i} on {name_c}: {temperature:.0f} K, ceiling {ceiling_z:.1f} A, {steps} steps")
    atoms.set_constraint(FixAtoms(bottom))
    FIRE(atoms, logfile=None).run(fmax=0.05, steps=1000)
    if name_i in SMILES and not intact(atoms):
        raise RuntimeError(f"{name_i} on {name_c}: a molecule broke or bonded to another during packing")
    return atoms, strain, len(interlayer)


def intact(atoms):
    """True when the heavy atoms of each molecule, one tag each, form one bonded piece apart from the other molecules.
    Protons may move, to the surface or to another molecule."""
    sub = atoms[(atoms.get_tags() > 0) & (np.array(atoms.get_chemical_symbols()) != "H")]
    i, j = neighbor_list("ij", sub, natural_cutoffs(sub, mult=1.2))
    n, piece = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(sub),) * 2))
    tags = sub.get_tags()
    return n == len(set(tags)) and all(len(set(tags[piece == p])) == 1 for p in range(n))


def settle(atoms, upper, calc):
    """Moves the `upper` atoms along z to the height of lowest energy, then relaxes every atom.

    FIRE stops on the largest force on one atom, so it leaves a weakly bound layer short of its equilibrium height. A
    residual force of 0.01 eV/Å on each of 200 atoms is a traction of 1.4 GPa over 230 Å^2. In one cell FIRE left
    graphene 4.8 Å above silicon carbide and 1.0 Å above its lowest energy.
    """
    def energy(d):
        a = atoms.copy()
        a.calc = calc
        a.positions[upper, 2] += d
        return a.get_potential_energy()
    atoms.positions[upper, 2] += minimize_scalar(energy, bracket=(0, 0.1), tol=1e-3).x
    FIRE(atoms, logfile=None).run(fmax=0.05, steps=1000)


def split(atoms, upper, calc):
    """Work to pull the `upper` atoms off the rest, rigid and with both parts relaxed, and the peak traction of the
    rigid opening, per area of the cell.

    Molecules that interpenetrate collide in a rigid opening, so the traction counts only until two atoms across the
    split first come closer than at contact, the opening returned as `overlap_at_A`.
    """
    area = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
    e0 = atoms.get_potential_energy()
    fixed = set(atoms.constraints[0].index)

    def part(mask, relax):
        p = atoms[mask]
        p.calc = calc
        p.set_constraint(FixAtoms([i for i, j in enumerate(np.flatnonzero(mask)) if j in fixed]))
        e = p.get_potential_energy()
        if relax:
            FIRE(p, logfile=None).run(fmax=0.05, steps=1000)
            e = p.get_potential_energy()
        return e

    rigid = (part(~upper, False) + part(upper, False) - e0) / area * J_M2
    relaxed = (part(~upper, True) + part(upper, True) - e0) / area * J_M2
    # The peak of a weak interface lies within half an ångström of contact, so the steps are finest there.
    openings = np.concatenate([np.arange(0, 1, 0.05), np.arange(1, 3.01, 0.25), [3.5, 4, 5, 6]])
    energies, closest = [], []
    for d in openings:
        a = atoms.copy()
        a.calc = calc
        a.positions[upper, 2] += d
        energies.append(a.get_potential_energy())
        closest.append(a.get_all_distances(mic=True)[np.ix_(upper, ~upper)].min())
    n = np.argmax(np.append(np.array(closest) < closest[0] - 0.05, True))
    traction = np.gradient(energies, openings)[1:max(n - 1, 2)] / area * MPA  # central differences only
    return {"separation_energy_J_m2": round(float(relaxed), 4), "rigid_separation_energy_J_m2": round(float(rigid), 4),
            "strength_MPa": round(float(traction.max())), "opening_at_peak_A": round(float(openings[1 + traction.argmax()]), 2),
            "overlap_at_A": round(float(openings[n]), 2) if n < len(openings) else None}


def separate(atoms, calc):
    """The splits at the interface and through the middle of the interlayer, and the weaker of the two.

    The glass network is cut atom by atom, at the interface and at the middle of the layer. Graphene sheets and
    molecules move whole, as the bond network after the MD joins them, by the height of their center. A molecule that
    bonded to the ceramic, and a proton it passed to the surface, stay with the ceramic.
    """
    tags = atoms.get_tags()
    z = atoms.positions[:, 2]
    if len(set(tags)) == 2:
        top = tags > 0
        inside = top & (z > np.median(z[top]))
    else:
        i, j = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=1.2))
        unit = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(atoms),) * 2))[1]
        top = ~np.isin(unit, unit[tags == 0])
        centers = {u: z[unit == u].mean() for u in set(unit[top])}
        inside = np.isin(unit, [u for u, h in centers.items() if h >= np.median(list(centers.values()))])
    for upper in (top, inside):
        settle(atoms, upper, calc)
    sites = {"interface": split(atoms, top, calc), "interlayer": split(atoms, inside, calc)}
    return sites, min(sites, key=lambda k: sites[k]["separation_energy_J_m2"])


def main(name_c, name_i):
    from mace.calculators import mace_mp

    start = time.time()
    out = HERE / "results" / f"{name_c.replace(' ', '_')}__{name_i.replace(' ', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    log = lambda s: print(f"[{time.time() - start:6.0f} s] {s}", flush=True)
    calc = mace_mp(model="medium", dispersion=True, dispersion_cutoff=15.0, default_dtype="float32", device="cuda",
                   enable_cueq=True)
    rng = np.random.default_rng(0)
    atoms, strain, n_layer = build(name_c, name_i, calc, rng, log)
    sites, weak = separate(atoms, calc)
    log(f"{sites}")
    anneal = {"graphite": "held 0.5 ps at 300 K", "silica glass": "melted 1 ps at 3500 K and quenched to 300 K in 2 ps",
              "epoxy": "packed by a ceiling falling 0.07 Å every 10 fs over 3 ps at 500 K and held 0.5 ps at 300 K",
              "PMMA": "packed by a ceiling falling 0.07 Å every 10 fs over 3 ps at 500 K and held 0.5 ps at 300 K"}[name_i]
    result = {"separation_energy_J_m2": sites[weak]["separation_energy_J_m2"], "strength_MPa": sites[weak]["strength_MPa"],
              "failure_site": weak, "interface": sites["interface"], "interlayer": sites["interlayer"],
              "atoms": len(atoms), "interlayer_atoms": n_layer, "area_A2": round(float(np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))), 1),
              "method": f"MACE-MP-0 medium with D3(BJ) dispersion to 15 Å, {len(atoms)} atoms. The interlayer was {anneal} on "
                        f"the ceramic slab, whose bottom 2 Å stay fixed, with a Langevin thermostat of 50 fs coupling time, then relaxed at 0 K, with the interlayer and its upper half "
                        f"each moved rigidly to their height of lowest energy. Separation energy: work to "
                        f"split the cell with both parts relaxed, at the {weak}, the weaker of the interface and the middle "
                        "of the interlayer. Strength: peak traction of the rigid opening at that site",
              "computed": time.strftime("%Y-%m-%d"), "wall_time_s": round(time.time() - start)}
    if name_i == "graphite":
        result["interlayer_strain"] = [round(float(s), 4) for s in strain]
    atoms.write(out.with_suffix(".xyz"))
    out.write_text(json.dumps(result, indent=1))
    log(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
