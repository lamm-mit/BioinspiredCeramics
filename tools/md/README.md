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

## Tools in this folder

Each folder holds one MD code with its own README. `md_interface` keeps the stand-in until one of them builds the interface between the ceramic and the interlayer.

| Folder | Tool | What it gives `md_interface` now |
|---|---|---|
| [md_agent/](md_agent/) | the MD agent of Alex Wang: bulk ceramics with the Pedone potential or MACE-MP-0, and coarse-grained proteins and polymers ([README](md_agent/README.md)) | nothing yet, since it builds no interfaces (below) |
| [fracture_md/](fracture_md/) | fracture-md of Jaime Berkovich, a JAX atomistic harness: cleavage of one plane, elastic tensor, ideal strength, slip ([README](fracture_md/README.md)) | the work of separation and the peak traction of a plane inside silicon carbide, from `cleavage_for_slot` in [`slot.py`](fracture_md/slot.py). It splits one material, so it fits only when both sides are silicon carbide, and its peak tractions are set by the cutoff of the interatomic model. It is not wired into `md_interface` |

## md_agent

[`md_agent/`](md_agent/) holds the MD code of Alex Wang, copied from his repository ([SOURCE.md](md_agent/SOURCE.md)). `md_interface` does not call it yet, and `__init__.py` does not import it, so its optional dependencies (OpenMM, ASE, MACE and PyTorch) are not needed to run the pipeline. It provides:

- Bulk ceramics, in `ceramics.py`. Periodic crystals from a library that includes α-alumina, 3C-SiC and α-quartz, or from a CIF file. It relaxes the lattice, computes the elastic tensor by finite strain, runs MD at a temperature with density, RDFs, coordination and diffusion, and makes glasses by melt-quench, which covers silica glass. The Pedone classical potential covers Si, Al, Mg and Ca oxides, and MACE-MP-0 covers the rest. Cells hold up to 800 atoms with the classical potential and 300 with MACE on a CPU, and runs last picoseconds.
- Coarse-grained chains, in `md_engine.py` and `md_analysis.py`. Proteins with the HPS model in real units, and generic bead-spring polymers in reduced Lennard-Jones units.
- An agent, `md_agent.py`, that sets up and runs either kind from a text description.

What `md_interface` needs that md_agent does not do yet:

1. Build the interface: a ceramic slab and an interlayer slab in one periodic cell, with the ceramic surface the design implies.
2. Equilibrate it at 300 K.
3. Pull it apart, by stepwise separation with relaxation or by steered MD, and return the work per area as `separation_energy_J_m2` and the peak traction as `strength_MPa`.

Caveats for the interlayers a design may use:

- Graphite. Its layers are held by dispersion. MACE-MP-0 is fitted to PBE, which has almost no dispersion, so without a D3 correction it underbinds graphite layers, and a separation inside the graphite would come out too weak.
- Epoxy and PMMA. They need an all-atom polymer force field, for the polymer and for its bond to the ceramic surface. The coarse-grained polymer model of md_agent is in reduced units and carries no chemistry, so it cannot give J/m² for a named polymer.
- Silicon carbide and alumina are in the library. Their surfaces and interfaces are not, and a separation energy needs both.
