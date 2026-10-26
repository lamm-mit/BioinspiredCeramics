# Requests to the tool authors

Two of the three slots of the engineer still return stand-ins. `dft_cleavage` needs a cleavage energy and `md_interface` needs an interface between two materials, and no attached code computes either yet. `fem_fracture` runs a real phase-field simulation, but it takes 74 to 182 s per call on the DGX Spark and gives every pixel one modulus. This file lists, per tool, what would close those gaps. Each row names the input, the output with its unit, and why the pipeline needs it. The measurements are from 2026-10-25 on the idle Spark. A 0.7 GB embedding model of another session was loaded during the last three FEM runs.

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

## DFT, Quantum ESPRESSO adapter of fracture-md (Jaime Berkovich)

| Need | Input | Output | Why the pipeline needs it |
|---|---|---|---|
| A cleavage routine | `material` (alumina or silicon carbide) and a plane | `cleavage_energy_J_m2`, (E_slab - N E_bulk) / A with relaxed surfaces, in J/m², with the slab thickness and vacuum at which it converged | `dft_cleavage` returns twice a tabulated surface energy. Its number is the layer fracture energy of FEM, and the ratio of the interface energy to it decides the verdict |
| Alumina set up and checked | the SSSP pseudopotentials of Al and O | the cutoff and k-mesh, a and c in Å and C_ij in GPa against experiment (a = 4.759 Å, c = 12.991 Å) | Alumina is the ceramic of most designs. Only Si, C and SiC have been converged and validated |
| A job interface | the inputs above | a job id at once, then a status call that returns the result or the error | A slab SCF takes minutes to tens of minutes on four cores, and the graph waits for each call |
| Polar surfaces | silicon carbide (111) | the cleavage energy with a dipole correction, in J/m², and the rule that shares it between the two faces | The low-energy SiC planes are polar, and without the correction the energy depends on the slab |

## fracture-md (Jaime Berkovich)

| Need | Input | Output | Why the pipeline needs it |
|---|---|---|---|
| An interface between two materials | `material_a`, `material_b` and the two planes | `separation_energy_J_m2` in J/m² and `strength_MPa` in MPa | `md_interface` is a stand-in. `cleavage_for_slot` splits one material, so it fits only silicon carbide on both sides, which no design uses |
| A strength set by the chemistry | the same | `strength_MPa` with a status other than `out_of_domain` | The peak tractions of Tersoff and Erhart-Albe, 130 to 204 GPa, are set by their cutoff functions. FEM reports the cohesive length E G / s², which needs a physical s |
| Bulk moduli of silicon carbide | `material` = silicon carbide | Young, bulk and shear moduli in GPa (Voigt-Reuss-Hill) and density in g/cm³, as a function like `cleavage_for_slot` | `md_bulk` covers alumina only. md_agent needs MACE for SiC, which is not installed. fracture-md computes the SiC elastic tensor with Erhart-Albe in 13 s |

## md_agent (Alex Wang)

| Need | Input | Output | Why the pipeline needs it |
|---|---|---|---|
| An interface and its separation at 300 K | the ceramic, the interlayer and the ceramic surface | `separation_energy_J_m2` in J/m² and `strength_MPa` in MPa, by stepwise separation with relaxation or by steered MD | `md_interface` is a stand-in, and FEM takes both numbers as the interface of the cell |
| Graphite with dispersion | graphite, with MACE-MP-0 and a D3 correction | the separation energy between graphite layers in J/m² | A crack along a graphite interlayer opens between its layers. PBE without D3 almost does not bind them, so the energy would come out too low |
| Epoxy and PMMA | the polymer, with an all-atom force field and its cross terms with the ceramic surface | the separation energy of the polymer from alumina in J/m² | The coarse-grained polymer model is in reduced units and carries no chemistry |
| A public function for bulk moduli | `material` and a potential | Young, bulk and shear moduli in GPa and density in g/cm³ | `tools/md/bulk.py` calls `build_ceramic`, `relax_cell` and `elastic_constants` directly, so a change inside `ceramics.py` can break it |
| A silica glass protocol for elastic moduli | cell size and quench rate | E, K and G in GPa and density in g/cm³, with the spread over seeds | `md_bulk` does not cover silica glass. A 72-atom quartz cell held 1.5 ps at 6000 K did not melt, and the md_agent protocol of 16,000 steps would take about 10 min at the measured 39 ms per step |

`md_bulk` gives alumina in 2.2 s from the 30-atom corundum cell, with density 4.06 g/cm³ against 3.99. Young's modulus is 348 GPa against about 400 GPa measured for dense polycrystalline α-alumina, 13% low. The bulk modulus is 274 GPa against 252 GPa, 9% high, as the md_agent validation reports. C44 is 114 GPa against 147 GPa measured [Gieske 1968], which accounts for most of the soft shear modulus of 135 GPa.
