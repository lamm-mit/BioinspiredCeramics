# FEM

`fem_fracture` runs a crack through the notched cell of a design and reports whether it deflects along the interfaces. The engineer calls it last, with the outputs of the other two tools.

| Input | Unit | Origin |
|---|---|---|
| `layer_modulus_GPa` | GPa | the design |
| `layer_thickness_um` | µm | the design |
| `layer_fracture_energy_J_m2` | J/m² | `cleavage_energy_J_m2` of `dft_cleavage` |
| `interface_fracture_energy_J_m2` | J/m² | `separation_energy_J_m2` of `md_interface` |
| `interface_strength_MPa` | MPa | `strength_MPa` of `md_interface` |
| `geometry_json` | | the description the geometry step wrote |

| Output | Unit | Meaning |
|---|---|---|
| `crack_deflects` | | true when the crack turns into an interface |
| `energy_ratio` | | interface fracture energy over layer fracture energy |
| `criterion` | | the rule behind `crack_deflects`, which the judge quotes |
| `cohesive_length_over_layer_thickness` | | E G / s² of the interface over the layer thickness, which sets the mesh size |
| `method` | | what produced the numbers |

`geometry_json` points to a file in millimetres with the bar and the cell, their sizes and layer counts, the STL file of each material, and the notch. The notch is described and not cut, as `{"plane": "x", "at_mm": ..., "from_z_mm": 0, "depth_mm": ...}`, since a phase-field model sets it as an initial crack. Layers stack along z, with a ceramic layer at the bottom and at the top.

The stand-in in [`__init__.py`](__init__.py) applies the He-Hutchinson criterion for elastically similar layers, so the crack deflects when the energy ratio is below 0.25 [He and Hutchinson 1989]. It computes with JAX and reports the device. On a unified-memory machine JAX must not reserve GPU memory up front, so the module sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`. To attach the FEM tool, replace the body of `fem_fracture` and keep the input and output names. [`../README.md`](../README.md) gives what a tool has to do and the handoff file it is called from.
