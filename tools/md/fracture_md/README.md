# fracture-md

fracture-md is an atomistic harness for the strength and cleavage of ceramics, written in JAX by Jaime Berkovich. This folder holds a copy of its Python package ([SOURCE.md](SOURCE.md), MIT licence) and [`slot.py`](slot.py), a function that returns its cleavage results in the units of the `md_interface` slot.

## What it computes

Everything is 0 K lattice statics of a perfect crystal. Each number comes as a record with its unit, method, numerical checks and a validation status. A number that the interatomic model is known to get wrong is marked `out_of_domain`.

| Calculation | What comes out |
|---|---|
| Cleavage of one plane, `physics.cleavage.run_cleavage` | traction-separation curve, work of separation, surface energy, peak traction and the opening at the peak, UBER fit. Rigid opening, and a displacement-controlled relaxed opening |
| Elastic tensor, `physics.elastic.run_elastic_tensor` | relaxed-ion and clamped-ion C_ij by exact autodiff, checked by finite differences; bulk, shear and Young moduli, anisotropy, directional moduli |
| Ideal strength, `physics.ideal_strength.run_ideal_strength` | stress-strain curve under uniaxial tension or shear, its peak and the instability that ends it |
| Slip, `physics.cleavage.run_tangential_slip` | energy and traction of one half sliding over the other on a plane, an upper bound of the unstable stacking-fault energy |
| Nanoindentation, interim | force against depth and a contact pressure. No hardness is written |

## Materials and interatomic models

| Model | Name in `get_potential` | Materials | Status |
|---|---|---|---|
| Erhart-Albe 2005 | `erhart_albe_2005_SiC` | SiC | elastic constants within the measured spread; bond breaking flagged |
| Tersoff 1989 | `tersoff_1989_SiC` | SiC | C11 12 % too stiff; bond breaking flagged |
| Vashishta 2007 | `vashishta_2007_SiC` | SiC | used for indentation; C44 43 % below DFT |
| MACE-MP-0 small, JAX port | `mace_mp0_small_jax` | 89 elements, tested on SiC only | optional install; C11 17 % and C44 39 % below DFT |

Of the six materials in [`../../schema.py`](../../schema.py), only silicon carbide has tested models. Alumina, silica glass and graphite could run only through MACE-MP-0, with a structure the user supplies, and nothing has been tested for them. Epoxy and PMMA are not covered.

## Validation

- The code. The JAX kernels of Tersoff and Erhart-Albe match matscipy and LAMMPS to 2e-10 eV/Å or better, Vashishta matches LAMMPS on 15 configurations to 6e-10 eV/Å, and the MACE port matches the PyTorch MACE to 5e-6 eV/Å. A second engine that shares no code (ASE and matscipy) reproduces the lattice constant to 6e-8 Å, the elastic constants to 1.4e-3 GPa and the work of separation to 1e-11 J/m² in all 54 cross-checks.
- The models against their own papers. Tersoff gives C11/C12/C44 = 437/118/257 GPa, as published (Pastewka et al. 2013), and relaxed surface energies of 1.85 J/m² on (111) and 2.40 J/m² on (110), the published values for this potential.
- The models against DFT. The surface energies of both classical models are 27-56 % below DFT. None of the four SiC models passes the DFT-PBE admission table of fracture-md (elastic constants, ideal shear strength, stacking faults, B3-B1 transition pressure), see [../../dft/fracture_md_qe/README.md](../../dft/fracture_md_qe/README.md).
- Bond breaking. Tersoff and Erhart-Albe switch each interaction off between 1.2 and 1.4 bond lengths, which is where cleavage breaks bonds. Their peak tractions and opening lengths are set by that switch, not by chemistry, and are marked `out_of_domain`. The work of separation is the only cleavage number of these models with a chance of being physical, and it is not validated against DFT or experiment.
- Every result is an ideal value for a perfect crystal at 0 K, an upper bound of real strength and toughness.

## Run times, measured here

Fresh virtual environment, Python 3.12, jax 0.11.2 on the CPU, numpy 2.5.3 and the pins of [pyproject.toml](pyproject.toml), four Cortex-X925 cores of a DGX Spark (cores 15-18, nice 10), 2026-10-21. Every number reproduces the fracture-md documentation (docs/physics_cleavage.md and docs/physics_bulk.md) to the digits printed there, and every numerical check passed.

| Calculation | Result | Wall time |
|---|---|---|
| bulk lattice relaxation, 3C-SiC | a0 = 4.35933 Å (Erhart-Albe), 4.32106 Å (Tersoff) | 2-5 s |
| rigid cleavage (111) shuffle, 108 atoms, Erhart-Albe | W_sep 4.349 J/m² rigid, 3.333 J/m² with relaxed surfaces; peak 129.9 GPa at 0.448 Å | 6 s |
| rigid cleavage (110), 144 atoms, Erhart-Albe | 5.326 and 4.573 J/m²; 139.5 GPa at 0.519 Å | 8 s |
| rigid cleavage (111) shuffle, Tersoff | 5.010 and 3.698 J/m²; 192.0 GPa at 0.453 Å | 6 s |
| rigid cleavage (110), Tersoff | 6.136 and 4.806 J/m²; 203.7 GPa at 0.526 Å | 7 s |
| elastic tensor, 64 atoms, Tersoff | C11/C12/C44 = 436.633/117.941/256.597 GPa | 15 s |
| elastic tensor, 64 atoms, Erhart-Albe | C11/C12/C44 = 383.780/144.414/239.747 GPa | 13 s |
| one call of `cleavage_for_slot`, child process included | as the cleavage rows | 16-17 s |

The first call compiles. Later calls reuse a cache in `~/.cache/fracture_md/jax` (or `$FRACTURE_MD_JAX_CACHE`), which saved only a few seconds here. Peak memory was 6.3 GB. The 85 fast tests of fracture-md for cleavage, slabs, the elastic tensor, ideal strength, the second engine and the DFT adapter pass against this copy in 93 s. Ideal-strength runs take 50-120 s per direction (fracture-md documentation, not measured here).

## Install

```bash
pip install -e tools/md/fracture_md
```

The jax and numpy pins are those of `requirements.txt` at the repository root. Run on the CPU with `JAX_PLATFORMS=cpu`, since the checks need float64. Importing `fracture_md` turns on float64 for the whole Python process, which is why `slot.py` runs it in a child process.

MACE-MP-0 is optional and needs its own environment, as in fracture-md: torch 2.14.0 (CPU wheel), mace-torch 0.3.16, and mace-jax installed from GitHub (ACEsuit/mace-jax at commit 19cf364a405fcf9f2f82590a98d61bc870e05881) with jax 0.11.2. Then

```bash
python -m fracture_md.potentials.mace_jax convert --model small
```

downloads the MACE-MP-0 small PyTorch checkpoint into `~/.cache/mace/` and writes the JAX weights to `tools/md/fracture_md/weights/mace-mp-0-small-jax/`, or under `$FRACTURE_MD_WEIGHTS`. The weights are MIT-licensed and are never committed. The fracture-md script `scripts/setup_mace_env.sh` does all of this in one step. The MACE path was not run in this copy.

The DFT pseudopotentials are described in [../../dft/fracture_md_qe/README.md](../../dft/fracture_md_qe/README.md).

## Example

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import fracture_md  # turns on float64 JAX for the whole process
from fracture_md.physics.cleavage import relax_bulk_lattice, run_cleavage
from fracture_md.potentials import get_potential
from fracture_md.schema.material import Candidate, CandidateKind
from fracture_md.schema.protocol import CleavageProtocol
from fracture_md.structures.builders import sic_3c_conventional, sic_3c_phase, sic_composition

model = get_potential("erhart_albe_2005_SiC")
cell, _ = relax_bulk_lattice(model, sic_3c_conventional())
sic = Candidate(id="3C-SiC", name="3C-SiC", kind=CandidateKind.REFERENCE, composition=sic_composition(),
                phases=[sic_3c_phase()], hypothesis="example")
out = run_cleavage(CleavageProtocol(plane_miller=(1, 1, 0), termination="nonpolar"), sic, model, structure=cell)
law = out.traction_separation[0]
print(law.work_of_separation.magnitude(), law.peak_traction.magnitude())  # 5.326 J/m2, 139.5 GPa
for r in out.properties:
    print(r.name.value, r.value.value, r.validation.status.value)
```

## How it relates to `md_interface`

`md_interface(material_a, material_b)` asks for the separation energy and the strength of the interface between the ceramic and the interlayer. fracture-md splits one material on one plane. It does not build an interface between two materials, so it can serve `md_interface` only when the interface fails inside a material it models, for example when both sides are silicon carbide. The interlayers of `schema.py` are not among those materials.

`cleavage_for_slot(material, plane="111", potential="erhart_albe_2005_SiC")` in [`slot.py`](slot.py) does that one case. It takes `"silicon carbide"` and raises `ValueError` for every other material. The plane is `"111"` (shuffle cut, the lowest work of separation of the planes fracture-md has computed, for both models) or `"110"`, and the potential `erhart_albe_2005_SiC` or `tersoff_1989_SiC`. It raises `RuntimeError` when the calculation fails or a numerical check does not pass.

| Output | Unit | Meaning |
|---|---|---|
| `separation_energy_J_m2` | J/m² | work of separation with relaxed surfaces, the Griffith energy of the plane |
| `strength_MPa` | MPa | peak traction of the rigid opening |
| `method` | | what produced the numbers, with their limits |
| `rigid_separation_energy_J_m2` | J/m² | work of separation of the rigid opening, surfaces not relaxed |
| `uber_strength_MPa`, `uber_length_nm` | MPa, nm | peak and length of the UBER fit to the rigid curve |
| `opening_at_peak_nm` | nm | opening at the peak traction |
| `strength_set_by_cutoff_function` | | true when the peak lies where the model switches bonds off |
| `separation_energy_status`, `strength_status` | | fracture-md validation status of each number |
| `n_atoms`, `wall_time_s` | | slab size and run time |
| `curve_json` | | file with the curve, the UBER fit and every fracture-md record |

Measured outputs:

| Plane | Potential | `separation_energy_J_m2` | `strength_MPa` | `uber_strength_MPa` |
|---|---|---|---|---|
| 111 | Erhart-Albe 2005 | 3.333 | 129937 | 75694 |
| 110 | Erhart-Albe 2005 | 4.573 | 139481 | 81053 |
| 111 | Tersoff 1989 | 3.698 | 191994 | 87386 |
| 110 | Tersoff 1989 | 4.806 | 203669 | 93263 |

For comparison, the stand-in of `md_interface` gives 3.6 J/m² and 26488 MPa with silicon carbide on both sides. With both models the strength is set by the switch-off of the bonds and is marked `out_of_domain`. It is a property of the model, not a strength of SiC. All these strengths are ideal strengths of a perfect crystal, far above any measured strength. One call takes about 16 s on four cores and returns within the call, as the tool contract asks. It runs on the CPU in a child Python, the Python of the calling process or `$FRACTURE_MD_PYTHON`.

`md_interface` in [`../__init__.py`](../__init__.py) is unchanged. With `pip install -e tools/md/fracture_md` in the pipeline environment it could read:

```python
from .fracture_md.slot import cleavage_for_slot


def md_interface(material_a, material_b):
    if material_a == material_b == "silicon carbide":
        return cleavage_for_slot("silicon carbide")  # about 16 s; the method string carries the limits
    ...  # the stand-in body, unchanged
```

The judge marks a verdict provisional only when a method says stand-in. The method of `cleavage_for_slot` does not, although its strength is marked out of domain, so wire it in only with that in mind. This folder has no `__init__.py` on purpose: a regular package of that name would hide the installed `fracture_md` whenever Python runs from `tools/md/`. The relative import above still works, since Python treats the folder as a namespace package.
