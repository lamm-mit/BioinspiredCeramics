# MD

`md_interface` gives the separation energy and the strength of the interface between the ceramic and the interlayer of a design. The engineer calls it once per candidate, before `fem_fracture`, which takes both outputs as the cohesive law of the interface.

| Input | Unit | Origin |
|---|---|---|
| `material_a` | | the ceramic of the design |
| `material_b` | | the interlayer of the design |

| Output | Unit | Meaning |
|---|---|---|
| `separation_energy_J_m2` | J/m² | work per area to pull the interface apart |
| `strength_MPa` | MPa | peak traction of that separation |
| `method` | | what produced the numbers |

The stand-in in [`__init__.py`](__init__.py) assumes perfect adhesion, so the interface opens inside the weaker material with work W = 2γ of that material. The strength is the peak of the universal binding-energy relation T(d) = W d / l² exp(-d / l), which is W / (e l), with l = 0.05 nm. The real tool builds the interface at 300 K, equilibrates it and pulls it apart. To attach it, replace the body of `md_interface` and keep the input and output names. [`../README.md`](../README.md) gives what a tool has to do and the handoff file it is called from.
