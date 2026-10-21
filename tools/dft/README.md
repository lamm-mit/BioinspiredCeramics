# DFT

`dft_cleavage` gives the energy to cleave the bulk ceramic of a design. The engineer calls it once per candidate, before `fem_fracture`, which takes its output as the fracture energy of a layer.

| Input | Unit | Origin |
|---|---|---|
| `material` | | the ceramic of the design, one of the materials in [`../schema.py`](../schema.py) |

| Output | Unit | Meaning |
|---|---|---|
| `cleavage_energy_J_m2` | J/m² | energy per area to split the bulk material into two surfaces |
| `method` | | what produced the number |

The stand-in in [`__init__.py`](__init__.py) returns twice a tabulated surface energy. To attach the DFT tool, replace the body of `dft_cleavage` and keep the input and output names. [`../README.md`](../README.md) gives what a tool has to do and the handoff file it is called from.

## Tools in this folder

| Folder | Tool | Status |
|---|---|---|
| [fracture_md_qe/](fracture_md_qe/) | Quantum ESPRESSO adapter of fracture-md (PBE, SSSP 1.3.0): energies, stresses, lattice parameters, equations of state, elastic constants, ideal shear strengths, stacking faults, transition pressures, with the setup scripts and the PBE reference data for SiC, Si and diamond | validated against published PBE for Si, SiC and diamond. It has no cleavage routine yet, so `dft_cleavage` above stays the stand-in |
