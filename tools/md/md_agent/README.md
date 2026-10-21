# md_agent

A tool-using LLM agent that takes a free-text description of a material, sets up a
molecular dynamics simulation, runs it, and returns molecular-level properties with
error bars. Three model families:

- **protein** and **polymer**: coarse-grained chains (`md_engine.py`, `md_analysis.py`)
- **ceramic**: atomistic periodic crystals and glasses (`ceramics.py`)

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...        # only for the real LLM policy
```

## GPUs and running inside a larger framework

Device selection is automatic, with environment overrides for schedulers
and multi-agent frameworks that assign devices:

| Variable | Effect |
|---|---|
| `MD_AGENT_GPU_INDEX` | GPU to use (OpenMM `DeviceIndex`, and `cuda:N` for MACE) |
| `MD_AGENT_OPENMM_PLATFORM` | Force `CUDA`, `OpenCL`, `CPU` or `Reference`; errors if unavailable |
| `MD_AGENT_DEVICE` | Force the torch device for MACE (e.g. `cpu`, `cuda:2`) |
| `MD_AGENT_MACE_DTYPE` | Override MACE precision (`float32` on GPU, `float64` on CPU by default) |

`CUDA_VISIBLE_DEVICES` is honoured as usual.

- **Protein/polymer (OpenMM backend):** tries CUDA, then OpenCL (mixed
  precision), then CPU, then Reference. A platform that is listed but fails
  to create a context (e.g. no driver) falls through to the next one. The
  platform actually used is reported in each run's `backend` field.
- **Ceramics, MACE:** runs on `cuda` when PyTorch sees a GPU. The atom limit
  rises from 300 (CPU) to 3,000 (GPU). The device and dtype are reported in
  `setup_ceramic`'s `potential_detail`.
- **CPU-only parts:** the numpy coarse-grained backend (reference
  implementation) and the classical ceramic potential (numpy Ewald). On a
  GPU node the classical potential still works but doesn't use the GPU.

**Testing status:** the fallback logic was tested on a CPU-only machine,
including a simulated "CUDA listed but unusable" case, a forced GPU index,
and a fake CUDA-capable torch. It has **not yet run on a real GPU**. Before
relying on it there, run `python3 md_engine.py` and `python3 ceramics.py`
on a GPU node and confirm `openmm/CUDA` / `cuda` appear in the outputs.

**Programmatic use:** a host framework doesn't need the CLI. Either call
`simulate_material(description, ...)`, or drive the tools directly: create
an `MDWorkspace(description)` and call `workspace.dispatch(tool_name, args)`
with the OpenAI-style function schemas in `TOOL_SCHEMAS`. Each workspace
holds its own systems, runs and step budgets, so give each agent its own.
Run concurrent agents as separate processes, not threads: the MACE model is
cached once per process and its calculator isn't thread-safe. Pin each
process to a GPU with `CUDA_VISIBLE_DEVICES` or `MD_AGENT_GPU_INDEX`.

## Usage

A tool-using agent that takes a free-text material description, sets up a
coarse-grained MD simulation, runs it, and returns molecular-level
properties with error bars:

```bash
python3 md_agent.py                                   # mock-policy demo, 2 examples, no key
python3 md_agent.py "4 chains of GGAGQGGYGGLGSQGAGRGGLGGQGAGAAAAAAAA at 2 mM, 150 mM NaCl" \
    --use-mock --out report.json
python3 md_agent.py "<description>"                  # real LLM policy (needs OPENAI_API_KEY)
python3 md_engine.py                                  # engine validation harness
python3 md_analysis.py                                # analysis checks vs analytic cases
```

**Models.** `protein`: HPS (Dignon et al. 2018). One bead per residue,
Ashbaugh-Hatch hydropathy pairs + Debye-Huckel salt screening, implicit
solvent, real units. `polymer`: generic bead-spring chain in reduced LJ
units, with solvent quality set by the attraction strength. **Backends:**
a dependency-free numpy engine (Verlet list + BAOAB Langevin) and an OpenMM
backend implementing the same Hamiltonian. `auto` uses OpenMM when it's
installed.

**Properties:** Rg, end-to-end distance, Ree²/Rg², Flory exponent ν,
asphericity κ², persistence length, intra/inter-chain contacts (with the
most frequent residue-pair types), largest-cluster fraction (aggregation),
COM diffusion (model time), potential energy. Equilibration diagnostics
(first- vs second-half drift, block SEM, kinetic temperature) drive the
agent's decision to extend a run.

**Guardrails.** The LLM never supplies numbers or sequences:
- Reported properties come from `compute_properties`, not from prose.
- A sequence must be verbatim in the user's description or the reference
  library (optionally tandem-repeated). An invented sequence is rejected
  with an error the LLM sees.
- Every number in the final narrative is checked against the tool outputs (grounding audit).
- Total MD steps, beads (5,000) and agent turns are all hard-capped.

**Validation done** (`python3 md_engine.py`):
- Finite-difference forces (~1e-7 relative error).
- numpy vs OpenMM energies and forces (~1e-15, i.e. the same Hamiltonian).
- Ideal-chain ⟨Ree²⟩ = (N−1)⟨b²⟩ and equipartition on both backends.
- Poor-solvent collapse.
- Salt screening of poly-Glu (Rg 3.6 nm at 5 mM vs 2.2 nm at 1 M; values vary run to run).

The mock agent end to end, plus the error paths (invented sequence,
budget, bad fields), have been exercised. The OpenAI tool-calling loop has
only been tested against a fake client, **not a live API**.

**Limits — read before using results for silk mechanics:**
- One bead per residue means no β-sheet nanocrystals and no H-bonds, so
  this does not predict fiber strength or toughness.
- HPS-KR is known to over-compact many IDPs. The synthetic MaSp repeat comes
  out compact (ν ≈ 0.32), so trust trends across sequences and conditions
  more than absolute values.
- Langevin dynamics are model time.
- Throughput in this container: about 0.6 ms/step for a 105-bead chain.
  100k steps (1 ns) takes about 1 min.


## Ceramics (`ceramics.py`)

```bash
python3 md_agent.py "Alpha-alumina at 300 K: lattice, density, elastic moduli" --use-mock
python3 md_agent.py "Silica glass made by melt-quench" --use-mock
python3 ceramics.py                                   # validation harness (~11 min here)
```

**Structures.** Built-in library of experimental room-temperature structures:
- MgO, CaO, α-Al₂O₃, α-quartz, Na₂O, rutile TiO₂, cubic ZrO₂, CeO₂,
  3C-SiC, wurtzite ZnO, SrTiO₃.
- Or a CIF file named in the description.

The LLM can't invent a structure. A CIF path must appear in the user's text.

**Potentials** (`potential="auto"` picks classical when it applies):
- `classical`: Pedone et al. 2006 rigid-ion potential (Ewald Coulomb +
  Morse + C/r¹²) for Si/Al/Mg/Ca oxides and their glasses. Implemented here
  in numpy with analytic forces and stress.
- `mace`: MACE-MP-0, a universal machine-learned potential (trained on PBE
  DFT) for everything else. Optional install; small model by default,
  `mace_model="medium"` is more accurate.

**Tools / properties:**
- 0 K relaxed lattice parameters and density.
- Elastic tensor C_ij by finite strain with internal relaxation, giving
  bulk/shear/Young's moduli, Poisson ratio, Pugh ratio and a mechanical
  stability check.
- NPT/NVT MD giving density and lattice parameters at temperature, partial
  RDFs, coordination-number distributions and per-species diffusion.
- Thermal expansion from runs at several temperatures.
- Melt-quench for glasses.
- Each result is compared with the experimental reference where the library
  has one.

**Validation** (`python3 ceramics.py`, 46 checks):
- All 11 library structures reproduce their X-ray density (within 3%) and
  cation coordination.
- Ewald sum: NaCl Madelung constant exact to 6 digits; independent of the
  splitting parameter; finite-difference forces (~4e-9 eV/Å) and stress
  (~4e-8 GPa).
- Classical vs experiment:

  | | a | c | Bulk modulus |
  |---|---|---|---|
  | MgO | +0.7% | | 168 vs 160 GPa |
  | CaO | −0.7% | | |
  | α-Al₂O₃ | −1.0% | +0.1% | 274 vs 252 GPa |
  | α-quartz | +0.2% | +0.4% | |

  Coordination is correct in all four. NPT MD at 300 K is stable at the
  target temperature.
- MACE-MP-0 (small) vs experiment:

  | | a | c | Bulk modulus |
  |---|---|---|---|
  | MgO | +1.0% | | 147 vs 160 GPa |
  | rutile TiO₂ | +1.1% | +0.4% | |
  | 3C-SiC | +0.5% | | |
  | Na₂O | +0.8% | | |

  This is the usual PBE pattern: lattice slightly too large, moduli
  slightly too soft.
- Silica glass by melt-quench (classical, 162 atoms, 6000 K melt): the melt
  check passed (every species moved), and Si is 98.6% 4-coordinated with
  Si–O 1.61 Å, the expected tetrahedral network. Density came out
  2.56 g/cm³ vs 2.20 for real silica glass, as expected from a tiny cell
  quenched at ~6e14 K/s, so treat simulated glass densities as qualitative.

**Limits:**
- Small cells (≤800 atoms classical, ≤300 MACE) and picosecond MD.
- Intrinsic lattice properties only: no grain boundaries, porosity or
  cracks, so this doesn't give the strength or toughness of a real part.
- Elastic constants are static 0 K values.
- The isotropic Berendsen barostat holds c/a fixed during MD.
- Glasses are quenched at ~1e14–1e15 K/s, many orders of magnitude faster
  than experiment, so glass density and structure differ from real glass.
- Perfect periodic crystals superheat. With this potential silica only melts
  at about 6000 K, and `melt_quench` refuses to call the result a glass
  unless every species actually moved during the melt.
- Na is not enabled in the classical potential. As transcribed it gave
  Na₂O 4% too small, and a transcription error can't be ruled out, so
  Na-containing materials use MACE.
- Speed in this container:
  - classical: about 0.02–0.08 s per step for 64–270 atoms;
  - MACE small: about 0.4 s per step for 64 atoms.
