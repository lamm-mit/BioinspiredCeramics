# fracture-md Quantum ESPRESSO adapter

`fracture_md.engines.dft_qe` runs plane-wave DFT with Quantum ESPRESSO `pw.x` (PBE, SSSP 1.3.0 pseudopotentials) and returns fracture-md records whose provenance names everything needed to repeat the number: the pw.x build, the functional, the sha256 of every pseudopotential file, the cutoffs, the k-mesh, the smearing, the MPI layout and the time of every run. `fracture_md.engines.dft_reference` builds reference protocols on it. Both are part of the fracture-md package of Jaime Berkovich, copied in [`../../md/fracture_md/`](../../md/fracture_md/) (install it with `pip install -e tools/md/fracture_md`). This folder holds the scripts that install `pw.x` and the pseudopotentials, and the PBE reference data.

## What it computes

Everything is 0 K, static and spin-unpolarised.

| Function | What comes out |
|---|---|
| `dft_qe.single_point` | energy, forces and stress of a given cell |
| `dft_qe.relax_structure` | vc-relax to zero stress or to a target pressure, or a fixed-cell relaxation; lattice parameter, density, residual stress |
| `dft_qe.equation_of_state` | Birch-Murnaghan fit: V0, a0, B0, B0prime, with a check of the stress against dE/dV |
| `dft_qe.elastic_constants` | relaxed-ion C_ij from finite strains, clamped-ion C_ij, Kleinman parameter |
| `dft_qe.convergence_scan` | energy and stress against the cutoff or the k-mesh |
| `dft_reference` | ideal shear strength paths, (111) stacking-fault energies, B3 to B1 transition pressure from the enthalpy crossing |

## Validation

Against published PBE values from all-electron (ACWF) and VASP-PAW calculations, for Si, 3C-SiC and diamond: lattice parameters within 0.1 %, C11, C12 and C44 within 0.8 %, bulk moduli within 1.1 %. For 3C-SiC: a0 = 4.381 Å, C11/C12/C44 = 383.7/127.8/239.4 GPa, B = 213 GPa.

Reference results for 3C-SiC (PBE): ideal shear strength on (111) 27.8 GPa in the twinning sense, 47.6 GPa in the anti-twinning sense and 30.6 GPa along [1-10]; unstable stacking-fault energy 2.84 J/m² (glide set) and 3.79 J/m² (shuffle set); intrinsic stacking fault 7.6 mJ/m²; B3 to B1 transition at 65.3 GPa. Ideal shear strength on (111)[11-2] of diamond 92.2 GPa and of Si 7.39 GPa. The transition pressure agrees with a published PBE value (+0.5 %). The other strengths and fault energies passed every numerical check, and the published values they were set beside use other functionals or protocols.

These are PBE model values, not measurements. PBE makes these crystals about 1 % too large and a few % too soft compared with experiment. Only Si, C and SiC have been converged and validated. SSSP has pseudopotentials for the elements of alumina and silica, but no convergence or validation was done for them.

## What it needs

- `pw.x`: Quantum ESPRESSO 7.5 from conda-forge, in its own conda environment `fmd-dft` (Open MPI 5, OpenBLAS, FFTW, ScaLAPACK, ELPA). QE is GPL-2.0-or-later and is not part of this repository.
- The SSSP 1.3.0 PBE pseudopotentials, efficiency and precision (Prandini et al., npj Comput. Mater. 4, 72 (2018); library doi:10.24435/materialscloud:f3-ym, CC-BY-4.0 with the licence of each pseudopotential family).
- The fracture-md package, `pip install -e tools/md/fracture_md`.

[`setup/setup_dft_env.sh`](setup/setup_dft_env.sh) installs the first two. It creates the conda environment from the exact builds in [`setup/fmd-dft-linux-aarch64.lock`](setup/fmd-dft-linux-aarch64.lock) on linux-aarch64 (a DGX Spark), or from `qe=7.5` elsewhere, and then writes a lock for that platform. It downloads SSSP from the Materials Cloud Archive, checks every archive and every pseudopotential file against the published md5, and writes a manifest with the sha256 of each file ([`setup/sssp_manifest.py`](setup/sssp_manifest.py)). The pseudopotentials go to `tools/dft/fracture_md_qe/weights/pseudos/`, which is never committed. Running the script again only verifies.

```bash
bash tools/dft/fracture_md_qe/setup/setup_dft_env.sh
SMOKE=1 FMD_DFT_CORES=15,16,17,18 bash tools/dft/fracture_md_qe/setup/setup_dft_env.sh   # also runs a 2-atom Si SCF
export FMD_PSEUDO_ROOT=$PWD/tools/dft/fracture_md_qe/weights/pseudos
```

The adapter reads `FMD_PSEUDO_ROOT` (without it, it looks in `tools/md/fracture_md/weights/pseudos`), `FMD_PW_X` or `FMD_DFT_ENV` to find `pw.x` (default `~/miniconda3/envs/fmd-dft`), and `FMD_DFT_CORES`, the cores to pin one MPI rank each to. The scripts are copies of those of fracture-md with only the paths changed.

## Run times, measured here

2-atom 3C-SiC, 70 Ry, 8x8x8 k-points, four Cortex-X925 cores of a DGX Spark (cores 15-18, one MPI rank per core, k-point pools, nice 10), 2026-10-21:

| Run | Result | fracture-md documentation | Wall time |
|---|---|---|---|
| SCF at a = 4.3812 Å | stress 0.002 GPa, 11 SCF iterations | | 2.5 s |
| vc-relax from the measured a = 4.3596 Å | a0 = 4.3812 Å | 4.3812 Å | 7.3 s |
| elastic constants, strains of 0.5 and 1 % | C11/C12/C44 = 383.6/127.8/239.4 GPa, all checks passed | 383.7/127.8/239.4 GPa | 65 s |

The setup script, run from scratch, downloaded and verified the 206 pseudopotential files. Their sha256 sums are identical to those of the fracture-md install, and its smoke SCF took 0.6 s. The 10 fast tests of the adapter in fracture-md and its 2 tests that run `pw.x` pass against the copied package (the two in 22 s).

Larger cells cost more. On four slower Cortex-A725 cores at 70 Ry, fracture-md measured one SCF at 20 s for 8 atoms, 13 min for 64 atoms and 19 min for a 32-atom (001) slab with 15 Å of vacuum. A stacking-fault point with relaxation took 9-19 min, and the whole reference set below (SiC, Si and diamond) 18.5 core-hours.

## How it relates to `dft_cleavage`

`dft_cleavage(material)` asks for the energy to cleave the bulk ceramic. That energy is the energy of a slab with two free surfaces less the energy of the same atoms in the bulk, per area of the cut: (E_slab - N E_bulk) / A. The adapter can compute both energies with `single_point`, or with `relax_structure` for relaxed surfaces, but it has no routine yet that builds the slab, checks the vacuum and the thickness, and returns the number. fracture-md builds such slabs for its MD cleavage (`structures.slabs.build_cleavage_slab`). One SCF of a 20 to 40 atom SiC slab takes minutes to tens of minutes on four cores, and more with relaxation and the convergence checks, so `dft_cleavage` would submit it as a job and poll it, as [`../../README.md`](../../README.md) asks of slow solvers. Polar SiC surfaces also need a dipole correction and a rule for sharing the energy between the two faces. Until that exists, `dft_cleavage` in [`../__init__.py`](../__init__.py) stays the stand-in.

## Reference data

[`reference/`](reference/) holds the DFT-PBE results of fracture-md, copied unchanged from its `data/dft_reference/` and `data/dft/` at commit bffe3e1.

| File | Content |
|---|---|
| `pbe_literature_values.json` | the published PBE values (all-electron and VASP-PAW) for Si, 3C-SiC and diamond that validate the set-up, with sources and confidence |
| `sic_3c_elastic_pbe.json` | a0, C11/C12/C44 relaxed and clamped ion, and B of 3C-SiC |
| `sic_3c_ideal_shear_pbe.json` | ideal shear strength on (111) in both [11-2] senses and along [1-10], with the stress-strain curves |
| `sic_3c_gsfe_pbe.json` | (111) stacking-fault energies of the glide and shuffle sets, with a k-point and cutoff check |
| `sic_b3_b1_transition_pbe.json` | the zincblende to rocksalt enthalpy crossing and the equations of state of both phases |
| `si_c_ideal_shear_pbe.json` | ideal shear strength of Si and diamond on (111)[11-2] |
| `potential_comparison_sic.json` | DFT against Tersoff 1989, Erhart-Albe 2005, Vashishta 2007 and MACE-MP-0 small, with the admission criteria and the verdicts. No model passes |

Each file carries values with units, the numerical settings, the pseudopotential files with their sha256, a status, published values for comparison with their functional, source and confidence, and fracture-md records with the provenance of every pw.x run. The paths to raw pw.x outputs inside the files point to the machine where they were computed. The raw outputs are not in this repository.

## Example

```python
from fracture_md.engines import dft_qe as q

runner = q.QERunner.from_env(cores=[15, 16, 17, 18])  # one MPI rank per core
st = q.DFTSettings(ecutwfc_ry=70.0, kpts=(8, 8, 8))  # converged for SiC
cell = q.primitive_cell("3C-SiC", 4.3596)
rel = q.relax_structure(cell, settings=st, runner=runner, workdir="runs/dft/vcrelax")
el = q.elastic_constants(rel.artifacts["atoms"], settings=st, runner=runner, workdir="runs/dft/elastic")
for r in el.properties:
    print(r.name.value, r.value.value, r.validation.status.value)
```

On the four cores above, this example takes 80 s and prints C11/C12/C44 = 383.7/127.8/239.4 GPa.
