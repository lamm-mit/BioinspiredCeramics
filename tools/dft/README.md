# DFT

`dft_cleavage` gives the energy to cleave the ceramic of a design into two relaxed surfaces, from plane-wave DFT. The engineer calls it once per candidate, before `fem_fracture`, which takes its output as the fracture energy of a layer.

| Input | Unit | Origin |
|---|---|---|
| `material` | | the ceramic of the design, alumina or silicon carbide ([`../schema.py`](../schema.py)) |

| Output | Unit | Meaning |
|---|---|---|
| `cleavage_energy_J_m2` | J/m² | (E_slab - N e_bulk) / A with every atom of the slab relaxed, twice the surface energy |
| `unrelaxed_cleavage_energy_J_m2` | J/m² | the same for the slab as cut, before relaxation |
| `plane`, `slab_atoms`, `slab_thickness_A`, `lattice_A` | | the surface, the slab and the relaxed bulk lattice |
| `pseudopotential_md5`, `computed`, `wall_time_s` | | what was run, when, and how long it took |
| `method` | | code, functional, pseudopotentials, cutoffs, k-spacing and cell |

## Method

[`__init__.py`](__init__.py) runs Quantum ESPRESSO `pw.x` through ASE. The functional is PBE with the SSSP 1.3.0 efficiency pseudopotentials at their recommended cutoffs, the highest of the elements in the cell: 50/400 Ry for alumina and 45/360 Ry for silicon carbide. The k-mesh has a spacing of 0.3 Å⁻¹ with 2π, one point along the normal of a slab, and Marzari-Vanderbilt smearing of 0.01 Ry. The bulk cell is vc-relaxed. The slab is cut from the relaxed cell with 12 Å of vacuum, and every atom is relaxed at fixed cell to forces below 0.026 eV/Å. Local Thomas-Fermi mixing damps the charge sloshing along the vacuum.

| Ceramic | Plane | Slab |
|---|---|---|
| alumina | (0001), cut between the two Al sublayers of a buckled Al layer, which leaves the stoichiometric Al-terminated surface on both faces | the 30-atom hexagonal cell, 18 atomic planes, 12.5 Å |
| silicon carbide | (110) of 3C, whose planes hold one Si and one C each, so the cut is nonpolar | 16 atoms, 8 planes, 10.8 Å |

## Results

| Ceramic | Cleavage energy (J/m²) | Unrelaxed (J/m²) | Surface energy (J/m²) | PBE lattice (Å) | Time (h) |
|---|---|---|---|---|---|
| alumina | 3.107 | 6.978 | 1.55 | a 4.813, c 13.131 | 9.8 |
| silicon carbide | 5.815 | 7.369 | 2.91 | a 4.381 | 2.5 |

Relaxation more than halves the energy of alumina. The surface Al sinks 0.74 Å into the slab, from 0.85 Å above the oxygen plane below it to 0.11 Å, the contraction of about 85 % that earlier DFT found for this surface [Manassidis 1993; Siegel 2002]. The PBE lattice of alumina is 1.1 % above the measured one along both axes. Silicon carbide relaxes by 21 %, as the surface Si sinks 0.24 Å and the C stays. Its unrelaxed energy checks against a bond count. Each (110) cell of 13.6 Å² cuts two Si–C bonds of about 3.2 eV, a quarter of the measured cohesive energy per SiC, which gives 7.5 J/m² against the computed 7.37 J/m².

The energy depends only on the material, so each ceramic is computed once and kept in [`results/`](results/), with the pw.x input and output of the bulk and the slab. Later calls read the file. A material without a file is computed at its first call. The times above are on 8 cores of the Spark while MD runs and the pipeline shared the machine.

## Setup

`pw.x` comes from the conda environment that [fracture_md_qe/setup/setup_dft_env.sh](fracture_md_qe/setup/setup_dft_env.sh) creates, Quantum ESPRESSO 7.5 from conda-forge, and the pseudopotentials from the same script. On the Spark it was run with Miniforge.

```bash
CONDA=~/miniforge3/bin/conda FMD_DFT_ENV=~/miniforge3/envs/fmd-dft bash tools/dft/fracture_md_qe/setup/setup_dft_env.sh
```

`FMD_DFT_ENV` names the environment (default `~/miniforge3/envs/fmd-dft`) and `FMD_DFT_CORES` the cores, one MPI rank each (default 12 to 19).

## Tools in this folder

| Folder | Tool | Status |
|---|---|---|
| [fracture_md_qe/](fracture_md_qe/) | Quantum ESPRESSO adapter of fracture-md (PBE, SSSP 1.3.0): energies, stresses, lattice parameters, equations of state, elastic constants, ideal shear strengths, stacking faults, transition pressures, with the setup scripts and the PBE reference data for SiC, Si and diamond | validated against published PBE for Si, SiC and diamond. `dft_cleavage` uses its environment and pseudopotentials, and calls `pw.x` through ASE directly |
