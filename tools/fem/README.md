# FEM

`fem_fracture` grows a crack from the notch of the cell of a design, on a section of its toolpath pattern, and reports whether the crack deflects along the interlayer, runs through a joint of the pattern, or runs on into the ceramic of the next ply. The engineer calls it last, with the outputs of the other two tools. The solver is jax-pfczm, a phase-field fracture code in JAX, copied into [`jax_pfczm/`](jax_pfczm/SOURCE.md). Importing it turns on float64 in JAX for the whole process.

| Input | Unit | Origin |
|---|---|---|
| `layer_modulus_GPa` | GPa | the design |
| `layer_thickness_um` | µm | the design |
| `layer_fracture_energy_J_m2` | J/m² | `cleavage_energy_J_m2` of `dft_cleavage` |
| `interface_fracture_energy_J_m2` | J/m² | `separation_energy_J_m2` of `md_interface` |
| `interface_strength_MPa` | MPa | `strength_MPa` of `md_interface` |
| `geometry_json` | | the description the geometry step wrote, which names the section file |

| Output | Unit | Meaning |
|---|---|---|
| `crack_deflects` | | true only for `deflected`. A crack that runs straight up a joint has not deflected |
| `energy_ratio` | | interface fracture energy over layer fracture energy |
| `criterion` | | the rule behind `crack_deflects`, which the judge quotes |
| `cohesive_length_over_layer_thickness` | | E G / s² of the interface over the layer thickness |
| `he_hutchinson_deflects` | | the closed-form verdict, `energy_ratio` < 0.25, for comparison |
| `outcome` | | `deflected`, `through joint` or `penetrated`, see Outcome |
| `along_interface_left_um`, `along_interface_right_um` | µm | how far the crack runs through interlayer on each side of the notch |
| `into_next_layer_um` | µm | how far the crack runs above the base of ply 1, the ply above the notched one |
| `nearest_joint_um` | µm | distance from the notch to the nearest interlayer column of ply 1 at the middle height of its ceramic, or null when ply 1 has no joint |
| `interface_reached_K_MPa_sqrt_m` | MPa m^½ | applied stress intensity when the crack first reached interlayer |
| `interface_broke_ahead_of_crack` | | true when interlayer of ply 0, the first interlayer or a joint of the notched ply, broke before the crack reached interlayer |
| `layer_K_MPa_sqrt_m` | MPa m^½ | toughness of the layer, sqrt(E G / (1 - ν²)) |
| `band_energy_ratio` | | the energy ratio that a crack along the middle of an interlayer band meets, see Limits |
| `pixel_um`, `interlayer_band_um` | µm | pixel size, and the thickness of an interlayer band, the interlayer thickness or 3 pixels if that is more |
| `energy_balance` | | \|external work - elastic energy - dissipated energy\| / external work |
| `converged` | | true when every load step converged |
| `seconds` | s | time of the simulation |
| `damage_png`, `load_path_csv` | | paths of the two files below |
| `geometry_json`, `device`, `method` | | the file it was given, the JAX device, and what produced the numbers with its limits |

The lengths and the energy balance are those of the last load step.

## Model

- Section file. The geometry step writes `cell_section.npz` next to `geometry.json` and names it in `cell.section`. It samples the toolpath pattern on the x-z section of the cell at 2 µm pixels. Per pixel it holds `phase` (0 ceramic, 2 interlayer), `ply` (0 at the bottom) and `modulus_GPa`, with row 0 at z = 0 and column 0 at the left edge of the cell. Interlayer pixels lie between plies and in the joints of the pattern, the road ends of tablets and the shells around roads.
- Pixels. The FEM grid has 60 rows over the cell height and one more column than the cell width takes, so that the notch column sits on the middle. The cell is 4 to 8 heights wide, so the grid has 241 to 481 columns. Each FEM pixel takes the phase, ply and modulus of the file pixel at its center.
- Thin features. AT2 with l = 2 pixels needs a weak feature at least 3 pixels thick. A feature thinner than that grows in the file by the missing half-thickness on each side before sampling. The growth uses the interlayer thickness, or the feature's own thickness where the file draws it thinner, as in the bottom interlayer of wavy plies, which the wave compresses where it fades to the flat bottom face. A flat interlayer of 20 µm at 60 rows of 19 µm becomes a band of 3 pixels, as in the earlier version of the tool.
- Materials. Each pixel has the modulus of its ply over the layer modulus, in scaled units, and a Poisson ratio of 0.25. Interlayer pixels keep the modulus of the ply they sit in, since the tool has no interlayer modulus. Without a modulus gradient all pixels have one modulus, so the layers are elastically similar, as the He-Hutchinson criterion assumes. Interlayer pixels have the interface fracture energy and ceramic pixels the layer fracture energy.
- Notch. A column of void pixels, one pixel wide, from z = 0 to the notch depth, half a layer thickness, at the notch position of the geometry step. That is the middle of a flat laminate. In a patterned cell it is the column of the middle half of the cell farthest from a joint of ply 0 at the notch tip and of ply 1 at its middle, so that the crack meets the plies and the interlayer between them before a joint.
- Load. The boundary of the cell follows the mode-I field of a crack at the notch tip, a K-field, and K rises in steps of 0.1 sqrt(E G). This is the setting of the He-Hutchinson criterion, a crack under a remote mode-I load that meets an interface at a right angle. The crack grows stably and meets the interlayer close to its threshold. The field takes the modulus of the pixel ahead of the notch, so that under a modulus gradient K is that of the notched ply. Two other loads were tried. Pulling the edges of the cell apart makes the crack jump from the notch through several layers in one load step, which hides the order of deflection and penetration, and loads the weak bands along their whole length (see Validation). Pushing the notch faces apart crushed the pixels at the faces, since AT2 damages in compression as in tension.
- Model. AT2 [Bourdin 2000] with a length l of 2 pixels, the resolution jax-pfczm asks for, solved by alternate minimisation, which conserves energy.

## Outcome

A node with damage of 0.9 or more is broken, and the crack is the broken nodes joined to the notch. Ply 1 is the ply above the notched one. Its middle lies half a layer thickness above its base, taken in each pixel column, so it follows wavy plies. The run stops at the first load step that decides one of three outcomes.

- `deflected`. The crack runs through interlayer more than one layer thickness sideways from the notch before it reaches the middle of ply 1.
- `through joint`. The crack reaches the middle of ply 1 along a path of interlayer nodes from the interlayer of ply 0, which is a vertical joint of the pattern.
- `penetrated`. The crack reaches the middle of ply 1 by breaking the ceramic of ply 1.

A step that both runs sideways and reaches the middle counts by the middle. A crack that does neither by K = 6 sqrt(E G) raises an error. For a flat laminate the rule is the earlier one, since ply 1 has no joint.

## Why the strength does not enter

The MD strength of an interface is of the order of GPa. With a modulus of a few hundred GPa and a fracture energy of about 1 J/m², the cohesive length E G / s² is tens of nanometres, while a pixel is 1 to 20 µm and a layer 10 to 210 µm. The cell is in the Griffith limit, where the energies decide the crack path and the strength does not. A PF-CZM model with the MD strength would need pixels smaller than the cohesive length, and it would need a strength for the ceramic too, which the tool is not given. AT2 uses its length as a numerical width of the crack. That length sets a model strength of (3/16) sqrt(3 E G / ((1 - ν²) l)), about 60 MPa for alumina layers of 210 µm with the DFT cleavage energy of 3.1 J/m², which is a numerical device and not a material property. The tool still reports `cohesive_length_over_layer_thickness`, so that the judge can see how far the cell is from the scale where strength would matter. When the cohesive length exceeds a tenth of the layer thickness, `method` says that the strength would matter and that AT2 leaves it out.

The simulation runs in scaled units, with pixel size, the layer modulus and the layer fracture energy equal to 1. The outcome therefore depends only on the energy ratio, the pattern of the section with its modulus ratios, and the pixel count. The layer modulus, the layer fracture energy and the size of the cell scale K, the energies and the lengths that the tool reports, and do not change `crack_deflects`. `interface_strength_MPa` changes only `cohesive_length_over_layer_thickness`.

## Validation

The energy ratio was swept for flat alumina layers of 210 µm with 20 µm interlayers, a modulus of 367 GPa and a layer fracture energy of 3 J/m², at three pixel counts. The thresholds of the criterion were set before the sweep and not changed after it.

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

The table was made with interlayer bands drawn at their nominal heights. The tool now samples the section file, and on 2026-10-26 it gave the same flat-laminate results. At 30 rows the energy ratios 0.05 and 1 matched the earlier version to 1e-12 in the energy balance. At 60 rows 0.15 and 0.2 deflected and 0.25 and 0.35 penetrated, as in the table. `into_next_layer_um` is now measured from the base of ply 1 and so exceeds the earlier value by the growth of the band, one pixel at 30 rows.

The outcome depends on the load. With the edges of the cell pulled apart instead of the K-field, at 60 rows, an energy ratio of 0.1 sent the crack straight through all five layers in one load step, and 0.02 deflected it along the first interlayer after the weak bands had softened along the whole cell, to a damage of 0.26. The K-field keeps the growth stable, as the He-Hutchinson comparison assumes.

## Patterns

Each pattern was run for alumina layers of 70 µm with 35 µm PMMA interlayers, a modulus of 367 GPa and a layer fracture energy of 3 J/m², at 30 rows of 16 µm pixels, on 2026-10-26. These are checks that the tool runs on each pattern and that the outcome follows the geometry. They are not converged results, since 30 rows give the ceramic of a ply only 4.3 pixels.

| Pattern | Cell width (mm) | `nearest_joint_um` | 0.05 | 0.15 | 1 |
|---|---|---|---|---|---|
| flat laminate | 1.96 | null | deflected | penetrated | penetrated |
| modulus graded from 367 GPa at mid-thickness to 250 GPa at the faces, 254 to 288 GPa in the cell | 1.96 | null | deflected | penetrated | penetrated |
| tablets 20 layer thicknesses long, staggered by half a tablet, shells around roads | 2.8 | 588 | penetrated | penetrated | penetrated |
| the same tablets, shifted so that a joint of ply 1 lies over the notch | 2.8 | 0 | through joint | through joint | through joint |
| Bouligand stack turned by 15° per ply, shells around roads | 1.96 | 49 | through joint | through joint | penetrated |
| wavy plies, amplitude 0.5 and wavelength 30 layer thicknesses, with tablets | 3.92 | 604 | penetrated | penetrated | penetrated |

At an energy ratio of 0.05 the crack of every patterned cell ran 155 to 400 µm sideways in the same load step as it reached ply 1, so these outcomes count by the middle of ply 1. The staggered tablets penetrate where the flat laminate deflects. Their crack crosses ply 1 40 to 60 µm from the notch, beside the aligned joints of plies 0 and 2, which lie 65 to 100 µm from it. With a joint over the notch the crack meets the first interlayer under the joint and runs on up it through interlayer only. At an energy ratio of 1 the joint is as tough as the ceramic and the crack still runs straight up it, so `through joint` reports the path of the crack and says nothing about toughening (see Model scope).

## Model scope

- Diffuse bands. AT2 spreads the damage of a crack over a few l, so a crack along a band also damages the tougher layers on both sides of it and costs more than the interface energy. For a band of half-thickness s, the 1D AT2 damage profile gives a band energy ratio r (cosh x + r sinh x) / (r cosh x + sinh x), with x = s / l and r the energy ratio. The tool reports it as `band_energy_ratio`. A band of 3 pixels with l = 2 pixels turns 0.05 into 0.075 and 0.25 into 0.33.
- Bands and joints are thicker than the interlayer when the interlayer is under 3 pixels. At 60 rows a pixel is 19 µm for a design with 210 µm layers and 20 µm interlayers, so its interlayers become 57 µm bands and the ceramic between them is thinner by as much. Near a junction of a joint and a band that the file draws thinner than the interlayer, a feature can stay 2 pixels thick.
- The section file has 2 µm pixels and places each edge of a feature to within 1 µm. An interlayer thinner than 2 µm is missing from the file in places, and then from the simulation.
- 2D section. A crack between the rotated plies of a Bouligand stack twists out of the plane of the section, which raises the toughness [Suksangpanya 2018]. The section holds the rotated roads only as the interlayer shells it cuts, so the twist and its toughening are absent.
- Modulus gradient. Only the modulus of each ply varies, and every ceramic pixel keeps the layer fracture energy. The K-field at the boundary is that of a uniform material with the modulus of the notched ply.
- `through joint` does not count as a deflection. It says that the crack reached ply 1 through interlayer, and with a joint over the notch it does so even when the interlayer is as tough as the ceramic. The outcome then depends on where the notch falls against the joints, which `nearest_joint_um` reports.
- AT2 has no elastic threshold and damages in compression as in tension, so weak bands soften a little wherever the stress reaches them.
- In AT2 the strength of a band is sqrt(energy ratio) times that of the layer, so a weak band can break just ahead of the crack tip before the crack reaches it, as in the Cook-Gordon mechanism. `interface_broke_ahead_of_crack` reports it. For real materials this depends on the ratio of the interface strength to the ceramic strength, which the model does not have.
- K rises in steps of 0.1 sqrt(E G). An unstable jump happens within one step, and the order of events inside that step is lost.
- Interlayer pixels have the modulus of their ply, 2D plane strain, quasi-static load, no residual stress and no plasticity of the interlayer.

## Run time and files

At 60 rows a call took 30 to 80 s in the validation runs above, on six cores of the Arm CPU of a DGX Spark. On the lab's DGX Spark the same grid took 134 to 182 s per call when idle (2026-10-25) and a median of 190 s with the language models loaded (2026-10-23). XLA uses about 2 of its 20 cores, and pinning the process to one fast core cut the time to 74 to 104 s. On an Apple M3 Pro laptop (2026-10-26) the flat laminate of the validation took 25 to 47 s per call at 60 rows and 5 to 10 s at 30 rows, and the patterned cells above 5 to 34 s at 30 rows. A cell of 8 heights has twice the columns of one of 4 heights and runs longer. The solver runs on the CPU even when JAX sees a GPU, which leaves the GPU to the served models. The first call compiles the solver for its grid in a few seconds, and later calls reuse it. `FEM_ROWS` sets the number of rows. 80 rows take 65 to 110 s, and the tests use 30 rows, with two calls at 60 rows for the validation table, about 2 min in all on the laptop.

Each call writes two files next to `geometry_json` and replaces those of an earlier call.

- `fem_damage.png`, the damage field at the last load step over the section, with interlayer grey and, under a modulus gradient, stiffer ceramic darker.
- `fem_load_path.csv`, one row per load step, with the applied K, the top of the crack, how far it runs through interlayer on each side, how far above the base of ply 1, the elastic energy, the dissipated energy and the external work per unit thickness, and the solver iterations.

[`cell.py`](cell.py) samples the pixels, builds the materials and the load, runs the solver and measures the crack. `python -m pipeline.tests.test_fem_tool` tests the tool. [`../README.md`](../README.md) gives what a tool has to do and the handoff file it is called from.
