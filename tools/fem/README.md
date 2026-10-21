# FEM

`fem_fracture` grows a crack from the notch of the cell of a design and reports whether it deflects along the first interlayer or runs on into the next ceramic layer. The engineer calls it last, with the outputs of the other two tools. The solver is jax-pfczm, a phase-field fracture code in JAX, copied into [`jax_pfczm/`](jax_pfczm/SOURCE.md). Importing it turns on float64 in JAX for the whole process.

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
| `crack_deflects` | | true when the simulated crack runs along the first interlayer for more than one layer thickness on a side before it reaches the middle of the next layer |
| `energy_ratio` | | interface fracture energy over layer fracture energy |
| `criterion` | | the rule behind `crack_deflects`, which the judge quotes |
| `cohesive_length_over_layer_thickness` | | E G / s² of the interface over the layer thickness |
| `he_hutchinson_deflects` | | the closed-form verdict, `energy_ratio` < 0.25, for comparison |
| `outcome` | | `deflected` or `penetrated` |
| `along_interface_left_um`, `along_interface_right_um` | µm | how far the crack runs along the first interlayer on each side of the notch |
| `into_next_layer_um` | µm | how far the crack runs above the first interlayer |
| `interface_reached_K_MPa_sqrt_m` | MPa m^½ | applied stress intensity when the crack joined the first interlayer |
| `interface_broke_ahead_of_crack` | | true when part of the first interlayer broke before the crack reached it |
| `layer_K_MPa_sqrt_m` | MPa m^½ | toughness of the layer, sqrt(E G / (1 - ν²)) |
| `band_energy_ratio` | | the energy ratio that a crack along the middle of an interlayer band meets, see Limits |
| `pixel_um`, `interlayer_band_um` | µm | pixel size and thickness of the interlayer bands |
| `energy_balance` | | \|external work - elastic energy - dissipated energy\| / external work |
| `converged` | | true when every load step converged |
| `seconds` | s | time of the simulation |
| `damage_png`, `load_path_csv` | | paths of the two files below |
| `geometry_json`, `device`, `method` | | the file it was given, the JAX device, and what produced the numbers |

The lengths and the energy balance are those of the last load step.

## Model

- Section. The cell becomes a 2D plane-strain section in the x-z plane, on square pixels: 60 rows over the cell height and 241 columns over its width of four heights. The extra column puts the notch at the middle.
- Materials. Every pixel has the layer modulus and a Poisson ratio of 0.25, so the layers are elastically similar, as the He-Hutchinson criterion assumes. The tool has no interlayer modulus. Each interlayer is a band of pixels at its true height, 3 pixels thick or as thick as the interlayer if that is more. Band pixels have the interface fracture energy and the other pixels the layer fracture energy.
- Notch. A column of void pixels, one pixel wide, from z = 0 to the notch depth.
- Load. The boundary of the cell follows the mode-I field of a crack at the notch tip, a K-field, and K rises in steps of 0.1 sqrt(E G). This is the setting of the He-Hutchinson criterion: a crack under a remote mode-I load meets an interface at a right angle. The crack grows stably and meets the interlayer close to its threshold. Two other loads were tried. Pulling the edges of the cell apart makes the crack jump from the notch through several layers in one load step, which hides the order of deflection and penetration, and loads the weak bands along their whole length (see Validation). Pushing the notch faces apart crushed the pixels at the faces, since AT2 damages in compression as in tension.
- Model. AT2 [Bourdin 2000] with a length l of 2 pixels, the resolution jax-pfczm asks for, solved by alternate minimisation, which conserves energy.
- Outcome. A node with damage of 0.9 or more is broken, and the crack is the broken nodes joined to the notch. The run stops at the first load step in which the crack runs more than one layer thickness along the first interlayer on either side, which is a deflection, or reaches the middle of the next layer, which is a penetration. A step that does both counts as a penetration. A crack that does neither by K = 6 sqrt(E G) raises an error.

## Why the strength does not enter

The MD strength of an interface is of the order of GPa. With a modulus of a few hundred GPa and a fracture energy of about 1 J/m², the cohesive length E G / s² is tens of nanometres, while a pixel is 1 to 20 µm and a layer 10 to 210 µm. The cell is in the Griffith limit, where the energies decide the crack path and the strength does not. A PF-CZM model with the MD strength would need pixels smaller than the cohesive length, and it would need a strength for the ceramic too, which the tool is not given. AT2 uses its length as a numerical width of the crack. That length sets a model strength of (3/16) sqrt(3 E G / ((1 - ν²) l)), about 60 MPa for alumina layers of 210 µm with the stand-in energy, which is a numerical device and not a material property. The tool still reports `cohesive_length_over_layer_thickness`, so that the judge can see how far the cell is from the scale where strength would matter. When the cohesive length exceeds a tenth of the layer thickness, `method` says that the strength would matter and that AT2 leaves it out.

The simulation runs in scaled units, with pixel size, modulus and layer fracture energy equal to 1. The outcome therefore depends only on the energy ratio, the shape of the cell, which is the interlayer thickness over the layer thickness, and the pixel count. The modulus, the layer fracture energy and the size of the cell scale K, the energies and the lengths that the tool reports, and do not change `crack_deflects`. `interface_strength_MPa` changes only `cohesive_length_over_layer_thickness`.

## Validation

The energy ratio was swept for alumina layers of 210 µm with 20 µm interlayers, a modulus of 367 GPa and a layer fracture energy of 3 J/m², at three pixel counts. The thresholds of the criterion were set before the sweep and not changed after it.

| Energy ratio | Band energy ratio | 40 rows: 28 µm pixels, 85 µm bands | 60 rows: 19 µm pixels, 57 µm bands | 80 rows: 14 µm pixels, 42 µm bands |
|---|---|---|---|---|
| 0.05 | 0.075 | deflected | deflected | |
| 0.1 | 0.145 | deflected | deflected | deflected |
| 0.15 | 0.209 | deflected | deflected | deflected |
| 0.2 | 0.270 | penetrated | deflected | deflected |
| 0.25 | 0.327 | penetrated | penetrated | penetrated |
| 0.3 | 0.382 | penetrated | penetrated | penetrated |
| 0.35 | 0.434 | penetrated | penetrated | penetrated |
| 0.5 | 0.580 | penetrated | penetrated | |
| 1 | 1 | penetrated | penetrated | |

At 60 and 80 rows the outcome flips between energy ratios of 0.2 and 0.25, and He-Hutchinson puts the flip at 0.25. At 40 rows it flips between 0.15 and 0.2, with bands 85 µm thick, 40% of a layer. In terms of the band energy ratio, which is what a crack along a band meets, the flip lies between 0.27 and 0.33, a little above 0.25. Two effects of AT2 pull against each other. The diffuse band makes a crack along the interlayer cost more than the interface energy, which favours penetration. A weak band also has a lower strength, so at 60 and 80 rows it broke just ahead of the crack tip for energy ratios up to 0.1 or 0.15, which favours deflection. At 60 rows a deflected crack ran 217 to 499 µm along the interlayer on each side, and a crack that penetrated ran 9 µm along it, the width of the crack itself.

The energy balance closed within 1.6% at 60 rows and 2.4% at 80 rows. Every load step converged except one at 80 rows for an energy ratio of 0.15, which reached the limit of 200 iterations. A call took 30 to 55 s at 60 rows, 65 to 110 s at 80 rows and 10 to 20 s at 40 rows. Alumina layers of 10 µm with 2 µm interlayers of PMMA and of graphite, with the stand-in energies, deflected at 60 rows, in 79 and 33 s.

The outcome depends on the load. With the edges of the cell pulled apart instead of the K-field, at 60 rows, an energy ratio of 0.1 sent the crack straight through all five layers in one load step, and 0.02 deflected it along the first interlayer after the weak bands had softened along the whole cell, to a damage of 0.26. The K-field keeps the growth stable, as the He-Hutchinson comparison assumes.

## Limits

- Diffuse bands. AT2 spreads the damage of a crack over a few l, so a crack along a band also damages the tougher layers on both sides of it and costs more than the interface energy. For a band of half-thickness s, the 1D AT2 damage profile gives a band energy ratio r (cosh x + r sinh x) / (r cosh x + sinh x), with x = s / l and r the energy ratio. The tool reports it as `band_energy_ratio`. A band of 3 pixels with l = 2 pixels turns 0.05 into 0.075 and 0.25 into 0.33.
- Bands are thicker than the interlayer when the interlayer is under 3 pixels. At 60 rows a pixel is 19 µm for a design with 210 µm layers and 20 µm interlayers, so its interlayers become 57 µm bands and the ceramic layers between them are thinner by as much.
- AT2 has no elastic threshold and damages in compression as in tension, so weak bands soften a little wherever the stress reaches them.
- In AT2 the strength of a band is sqrt(energy ratio) times that of the layer, so a weak band can break just ahead of the crack tip before the crack reaches it, as in the Cook-Gordon mechanism. `interface_broke_ahead_of_crack` reports it. For real materials this depends on the ratio of the interface strength to the ceramic strength, which the model does not have.
- K rises in steps of 0.1 sqrt(E G). An unstable jump happens within one step, and the order of events inside that step is lost.
- One modulus for all pixels, 2D plane strain, quasi-static load, no residual stress and no plasticity of the interlayer.

## Run time and files

At 60 rows a call takes 30 to 80 s on six cores of the Arm CPU of a DGX Spark. The solver runs on the CPU even when JAX sees a GPU, which leaves the GPU to the served models. The first call compiles the solver for its grid in a few seconds, and later calls reuse it. `FEM_ROWS` sets the number of rows. 80 rows take 65 to 110 s, and the tests use 30 rows, a few seconds per call.

Each call writes two files next to `geometry_json` and replaces those of an earlier call.

- `fem_damage.png`, the damage field at the last load step, over the interlayer bands.
- `fem_load_path.csv`, one row per load step: the applied K, the top of the crack, how far it runs along the first interlayer on each side, the elastic energy, the dissipated energy and the external work per unit thickness, and the solver iterations.

[`cell.py`](cell.py) builds the pixels, the materials and the load, runs the solver and measures the crack. `python -m pipeline.tests.test_fem_tool` tests the tool. [`../README.md`](../README.md) gives what a tool has to do and the handoff file it is called from.
