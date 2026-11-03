# MD

`md_interface` gives the energy and the strength to separate the ceramic of a design from its interlayer, at the weaker of two places, the interface and the middle of the interlayer. A crack that turns along the interlayer runs through that place. The engineer calls it once per candidate, before `fem_fracture`, which takes both outputs as the cohesive law of the interlayer.

| Input | Unit | Origin |
|---|---|---|
| `material_a` | | the ceramic of the design, alumina or silicon carbide |
| `material_b` | | the interlayer of the design, silica glass, graphite, epoxy or PMMA |

| Output | Unit | Meaning |
|---|---|---|
| `separation_energy_J_m2` | J/m² | work per area to split the cell at the weaker place, with both parts relaxed |
| `strength_MPa` | MPa | peak traction of a rigid opening at that place |
| `failure_site` | | `interface` or `interlayer` |
| `interface`, `interlayer` | | the separation energy, rigid separation energy, strength and opening at the peak of each place, and `overlap_at_A`, the opening at which two atoms across the split first come closer than at contact, or null |
| `atoms`, `interlayer_atoms`, `area_A2` | | the cell |
| `interlayer_strain` | | in-plane strain of graphite to fit the ceramic cell |
| `method`, `computed`, `wall_time_s` | | the model and the protocol, when, and how long it took |

## Method

[`interface.py`](interface.py) uses MACE-MP-0 medium, a machine-learned potential fitted to PBE for 89 elements [Batatia 2023], with the D3(BJ) dispersion correction to 15 Å, through ASE on the GPU. PBE binds graphite layers and polymer chains almost only through dispersion, so D3 carries most of the interlayer cohesion.

1. The ceramic is a slab about 7 Å thick, Al-terminated (0001) alumina or (110) silicon carbide, the planes `dft_cleavage` cleaves, in an orthogonal cell. It is cut from the Materials Project bulk cell ([`../materials.py`](../materials.py)) at its PBE lattice, 1.0 % above the measured one for alumina and 0.1 % below it for silicon carbide. The eight results below were computed earlier from the measured lattices. Its bottom 2 Å stay fixed. Under graphite the cell is the one of 12 to 17.5 Å per side that strains graphene least. Under an amorphous interlayer it is the largest under 17.5 Å, 14.3 × 16.5 Å for alumina and 12.3 × 17.4 Å for silicon carbide, so that a 20 Å molecule finds room.
2. The interlayer, about 1 nm thick, goes on top.
   - Graphite is two AB-stacked graphene sheets, strained by at most 3.3 % to fit the ceramic cell.
   - Silica glass is Si and O at random at 2.2 g/cm³, melted for 1 ps at 3500 K on the ceramic and quenched to 300 K in 2 ps, a rate of about 10¹⁵ K/s.
   - Epoxy is the adduct of bisphenol A diglycidyl ether with two methylamines, which carries the hydroxyl, ether and amine groups of a cured network. PMMA is a 5-mer with hydrogen ends. Both are placed at random in a layer three times the final thickness and packed to about 1.2 g/cm³ by a ceiling that falls 0.07 Å every 10 fs over 3 ps at 500 K. The run stops with an error if the heavy atoms of a molecule break apart or bond to another molecule.
3. The cell is held 0.5 ps at 300 K with a Langevin thermostat of 50 fs coupling time and 1 fs steps, with hydrogen at the mass of deuterium so that the steps stay stable, and then relaxed to forces below 0.05 eV/Å. The interlayer, and then its upper half, are moved rigidly along the normal to the height of lowest energy, and the cell is relaxed again. FIRE stops on the largest force on one atom, so it leaves a weakly bound layer short of that height. In the silicon carbide cell it left graphene 4.8 Å above the surface, 1.0 Å above the minimum, with a net pull of 1.4 eV/Å on the sheet.
4. The cell is split twice, at the interface and through the middle of the interlayer. The glass network is cut atom by atom. Graphene sheets and molecules move whole, by the height of their centers, as the bond network after the MD joins them. On alumina, epoxy molecules pass protons to surface oxygens and bond to surface Al, and such a molecule stays with the ceramic, so the interface split runs between it and the free molecules. Each split gives the work with both parts relaxed, the work of a rigid split, and the traction of a rigid opening from 0 to 6 Å, in steps of 0.05 Å up to 1 Å. Molecules that interpenetrate collide in a rigid opening, so the traction counts only until two atoms across the split come closer than at contact.

The energies are those of a sharp, defect-free cut at 0 K. They are the intrinsic Griffith energies, like the cleavage energy of `dft_cleavage`, and leave out the plastic and viscous dissipation that makes the measured toughness of a polymer or a glass larger.

## Results

**To rerun.** The values below come from an earlier protocol, and all eight pairs must be recomputed with the current [`interface.py`](interface.py). That protocol dropped the ceiling 2.2 Å at once and coupled the thermostat over 500 fs. The polymer layers reached 1900 to 2600 K instead of 500 K, and their molecules broke into 21 to 40 pieces, so the epoxy and PMMA energies are those of fragments. The glass was still at about 1100 K at the end of its quench to 300 K. Graphite saw only the thermostat change and should move little. `md_interface` reads a stored result when one exists, so delete the JSON and XYZ files in [`results/`](results/) before the rerun. A polymer pair now runs 1.5 ps more MD, about 25 min more on the Spark.

The middle of the interlayer is the weaker place in seven of the eight pairs. Graphene on silicon carbide sits 3.8 Å above the surface, held by dispersion alone, and separates there at 0.22 J/m². On alumina it sits at 2.7 Å and holds at 0.56 J/m², so the crack runs between the sheets. The 0.48 to 0.52 J/m² between the sheets is about 45 % above the 0.32 to 0.37 J/m² measured for graphite [Zacharia 2004; Liu 2012], an overbinding of D3 on MACE. The polymers separate at 0.13 to 0.25 J/m², about half of graphite, and silica glass at 2.0 to 2.7 J/m², ten times more. The strengths of 0.9 to 2.7 GPa for graphite and the polymers and 37 to 44 GPa for silica glass are ideal strengths of a defect-free cut. The interface strength of silica glass on silicon carbide rests on one point, since atoms of the glass sit among the surface atoms and collide at an opening of 0.15 Å.

| Ceramic | Interlayer | Interface (J/m²) | Middle of interlayer (J/m²) | Strength at the weaker (GPa) | Time (min) |
|---|---|---|---|---|---|
| alumina | graphite | 0.56 | **0.48** | 2.7 | 19 |
| alumina | silica glass | 3.11 | **2.74** | 37 | 101 |
| alumina | epoxy | 0.36 | **0.25** | 2.6 | 58 |
| alumina | PMMA | 0.31 | **0.19** | 1.5 | 57 |
| silicon carbide | graphite | **0.22** | 0.52 | 1.2 | 16 |
| silicon carbide | silica glass | 3.23 | **2.04** | 44 | 63 |
| silicon carbide | epoxy | 0.20 | **0.13** | 0.85 | 48 |
| silicon carbide | PMMA | 0.27 | **0.17** | 0.86 | 47 |

The result depends only on the pair of materials, so each pair is computed once and kept in [`results/`](results/) with the final cell as an XYZ file. Later calls read the file. A pair without a file runs `interface.py` at its first call, in the environment named by `SIMTOOLS_PYTHON` (default `~/simtools/env/bin/python`). The times above are on the GPU of the Spark with two or three runs sharing it.

## Setup

The pipeline's own environment has no PyTorch. The MD tools run in a second one.

```bash
python3 -m venv ~/simtools/env
~/simtools/env/bin/pip install torch --index-url https://download.pytorch.org/whl/cu129
~/simtools/env/bin/pip install mace-torch torch-dftd ase rdkit cuequivariance-torch cuequivariance-ops-torch-cu12 jax==0.11.2
```

## md_bulk

[`bulk.py`](bulk.py) runs md_agent on the bulk ceramic and returns its Young, bulk and shear moduli (Voigt-Reuss-Hill) and its density. The pipeline reports them beside the layer modulus each design takes from the literature. Alumina uses the Pedone potential and takes about 3 s, with a Young's modulus of 348 GPa against about 400 GPa measured, mostly because its C44 is 114 GPa against 147 GPa. Silicon carbide uses MACE-MP-0 medium in the MD environment and takes about 30 s, with a bulk modulus of 213 GPa against 225 GPa measured and a Young's modulus of 308 GPa against about 420 GPa, since MACE-MP-0 is soft in shear.

## Tools in this folder

| Folder | Tool | Use in the pipeline |
|---|---|---|
| [md_agent/](md_agent/) | the MD agent of Alex Wang: bulk ceramics with the Pedone potential or MACE-MP-0, and coarse-grained proteins and polymers ([README](md_agent/README.md)) | `md_bulk` |
| [fracture_md/](fracture_md/) | fracture-md of Jaime Berkovich, a JAX atomistic harness: cleavage of one plane, elastic tensor, ideal strength, slip ([README](fracture_md/README.md)) | none. `cleavage_for_slot` in [`slot.py`](fracture_md/slot.py) splits one material, so it fits only silicon carbide on both sides, which no design uses |
