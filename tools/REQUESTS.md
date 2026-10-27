# Requests to the tool authors

All three slots of the engineer run a solver. `dft_cleavage` relaxes a slab of the ceramic with Quantum ESPRESSO, `md_interface` builds the ceramic under its interlayer and splits it with MACE-MP-0 and D3, and `fem_fracture` runs a phase-field simulation of the notched cell. The DFT and MD energies depend only on the materials, so the two ceramics and the eight pairs were computed once and are read from `results/`. This file lists, per tool, what would close the remaining gaps. Each row names the input, the output with its unit, and why the pipeline needs it. The FEM measurements are from 2026-10-25 on the idle Spark, and the DFT and MD runs from 2026-10-26 and 27 on the Spark shared with the language models.

## FEM, jax-pfczm (Jaime Berkovich)

One fixed case, alumina layers of 70 µm with graphite interlayers of 35 µm at 60 pixel rows, energy ratios 0.113 and 0.25, took the times below. The verdicts, the crack lengths and the energy balance (1.1% and 2.0%) were the same on every device.

| Device | Ratio 0.113, deflected, first call with compilation | Ratio 0.25, penetrated |
|---|---|---|
| CPU as the tool runs now, 20 cores | 182 s | 134 s |
| CPU with `XLA_FLAGS=--xla_cpu_multi_thread_eigen=true` | 195 s | 157 s |
| CPU pinned to one Cortex-X925 core | 104 s | 74 s |
| CPU pinned to the ten Cortex-X925 cores | 124 s | 89 s |
| GPU, at its reduced clock of 507 MHz | 148 s | 103 s |

A call does 252 to 367 staggered iterations, at 0.3 to 0.5 s each. One preconditioned CG iteration of the displacement solve costs 1.6 ms on 20 cores, 0.8 ms on one core and 1.2 ms on the GPU, for 29,000 unknowns. Float32 is no faster on the CPU (1.7 ms), so the cost is the overhead of many small operations and of the scatter-add in each operator product. The process used 1.9 of 20 cores. No setting gained the factor of 3 that would justify a change, so the tool still runs on the CPU with default threads.

| Need | Input | Output | Why the pipeline needs it |
|---|---|---|---|
| A faster displacement solve | the same grid, materials and tolerances | the same fields at a third of the time per staggered iteration or less, and the CG iteration count of each step in the returned history | FEM is the longest step of a design run, a median of 190 s per call in pipeline runs. A geometric multigrid preconditioner, or a sparse Cholesky factor reused across the staggered iterations of a load step, would cut the CG iterations. A stencil operator on the structured grid would remove the scatter-add |
| An interlayer modulus | `interlayer_modulus_GPa` and `interlayer_poisson_ratio`, from the design | `crack_deflects`, with the Dundurs parameters α and β and the He-Hutchinson threshold at that α | Graphite, epoxy and PMMA are 3 to 30 GPa against 367 GPa for alumina, so α is near 0.9, where the He-Hutchinson threshold differs from 0.25. `Material` already takes a modulus per pixel. The pipeline needs to know whether the K-field load and l = 2 pixels stay valid with that mismatch, and a sweep that shows it |
| Interlayers thinner than three pixels | `interlayer_thickness_um` below 3 pixels, for example 2 µm interlayers between 10 µm layers | `band_energy_ratio` equal to `energy_ratio`, dimensionless | A band of 3 pixels turns an energy ratio of 0.25 into 0.33, which moves the flip of the verdict. A sharp cohesive interface or local refinement would remove that shift |
| The interface strength | `interface_strength_MPa` from `md_interface` and a ceramic strength in MPa | `crack_deflects` and `interface_broke_ahead_of_crack` that depend on the ratio of the two strengths | In AT2 the band strength is the square root of the energy ratio times the layer strength, a numerical value. A crack that opens the interface ahead of its tip, as in the Cook-Gordon mechanism, depends on the real strength ratio |
| A 3D cell | the voxel block of the cell, phase, ply and modulus per voxel, which `pipeline.pattern.material()` gives at any point | `crack_deflects`, the crack twist and tilt angles in degrees, and the dissipated energy in J/m² | The tool solves the x-z section of the cell. A Bouligand crack twists out of that plane, about 12° in DIW alumina [Zhou 2023], and the ends of tablets in other planes are absent, so the toughening of rotated and staggered plies is underestimated |
| A rate-dependent interface | a loading rate in 1/s and a viscoelastic interlayer, its relaxed and unrelaxed moduli in GPa and relaxation time in s | `crack_deflects` and the dissipated energy in J/m² as functions of the loading rate | Natural composites dissipate energy in viscoelastic organic layers, and polymer interlayers behave the same way. The tool is quasi-static, so the pipeline cannot rank a design by its response to impact |

## DFT, Quantum ESPRESSO (`dft/__init__.py`, with the environment of the fracture-md adapter of Jaime Berkovich)

| Need | Input | Output | Why the pipeline needs it |
|---|---|---|---|
| Interface energies with PBE-D3 | each ceramic slab with one interlayer in a cell of at most 100 atoms | the separation energy at the interface in J/m² | MACE-MP-0 is fitted to PBE bulk and surface data, and its error on these interfaces is unknown. It gives 0.48 to 0.52 J/m² between graphite sheets, against 0.32 to 0.37 J/m² measured [Zacharia 2004; Liu 2012] |
| A job interface | the inputs above | a job id at once, then a status call that returns the result or the error | A new ceramic takes 2.5 h (silicon carbide, 16 atoms) to 9.8 h (alumina, 30 atoms) on 8 cores, and the graph waits for the call |
| Polar surfaces | silicon carbide (111) | the cleavage energy with a dipole correction, in J/m², and the rule that shares it between the two faces | The (111) planes cleave most easily in SiC, and without the correction the energy depends on the slab. The tool cleaves the nonpolar (110) |
| A hydroxylated alumina surface | alumina (0001) with one OH per surface Al | the cleavage energy and the slab for `md_interface` | The tool cleaves the bare Al-terminated surface, the stable one in vacuum. In air the surface is hydroxylated, and on the bare surface the MD epoxy passes protons to surface oxygens |

## MD interface (`md/interface.py`)

| Need | Input | Output | Why the pipeline needs it |
|---|---|---|---|
| A slower glass quench | quench rate 10¹² to 10¹³ K/s, 50 to 100 ps | the separation energy in J/m², with the spread over seeds | The 2 ps quench at about 10¹⁵ K/s leaves more under-coordinated atoms than a real glass, which can raise or lower the energy to cut it |
| A crosslinked epoxy and long PMMA chains | an epoxy network of bisphenol A diglycidyl ether and a diamine, PMMA chains of 50 or more monomers | the separation energy and the traction while a chain pulls out, in J/m² and MPa | The adduct and the 5-mer are not entangled or crosslinked, so the energy through the polymer is twice its surface energy and leaves out chain pull-out and scission |
| Seeds | the same pair, 3 to 5 random seeds | the mean and spread of the separation energy | Each pair is one cell of 250 to 480 atoms, so its uncertainty is unknown |
| A relaxed opening | the same cell | the traction while the cell opens in steps with every atom relaxed | The rigid opening gives an ideal strength of 0.4 to 47 GPa. Interpenetrating molecules collide in it, so the traction stops at the first overlap |

## md_agent (Alex Wang)

| Need | Input | Output | Why the pipeline needs it |
|---|---|---|---|
| A public function for bulk moduli | `material` and a potential | Young, bulk and shear moduli in GPa and density in g/cm³ | `md/bulk.py` calls `build_ceramic`, `relax_cell` and `elastic_constants` directly, so a change inside `ceramics.py` can break it |
| Moduli of the interlayers | silica glass, graphite, epoxy and PMMA | Young's modulus and Poisson's ratio, with the spread over seeds | FEM gives an interlayer the modulus of the ply it sits in. The FEM request for an interlayer modulus needs these numbers from the same models as the energies |

`md_bulk` gives alumina in 2.2 s from the 30-atom corundum cell with the Pedone potential, with density 4.06 g/cm³ against 3.99. Young's modulus is 348 GPa against about 400 GPa measured for dense polycrystalline α-alumina, 13% low. The bulk modulus is 274 GPa against 252 GPa, 9% high, as the md_agent validation reports. C44 is 114 GPa against 147 GPa measured [Gieske 1968], which accounts for most of the soft shear modulus of 135 GPa. Silicon carbide with MACE-MP-0 medium takes 30 s and gives a bulk modulus of 213 GPa against 225 GPa and a Young's modulus of 308 GPa against about 420 GPa.
