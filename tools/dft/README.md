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
