# Tools

The engineer agent calls three tools for each candidate design, in this order. Each folder holds one tool, with its function, the JSON schema the model reads, a README, and in subfolders the solvers that can back it, each with its own README. DFT and MD are closed-form stand-ins for now, and FEM runs a phase-field simulation of the notched cell with [jax-pfczm](fem/jax_pfczm/SOURCE.md). A stand-in returns quantities of the right kind and order of magnitude, so the pipeline runs end to end, and its `method` string says that it is a stand-in.

| Tool | Folder | Inputs, with their origin | Outputs |
|---|---|---|---|
| `dft_cleavage` | [dft/](dft/) | `material`, the ceramic of the design | `cleavage_energy_J_m2` |
| `md_interface` | [md/](md/) | `material_a` and `material_b`, the ceramic and the interlayer of the design | `separation_energy_J_m2`, `strength_MPa` |
| `fem_fracture` | [fem/](fem/) | `layer_modulus_GPa` and `layer_thickness_um` from the design, `layer_fracture_energy_J_m2` from `dft_cleavage`, `interface_fracture_energy_J_m2` and `interface_strength_MPa` from `md_interface`, `geometry_json` from the geometry step | `crack_deflects`, `energy_ratio`, `criterion`, `cohesive_length_over_layer_thickness`, the measured crack path, and a damage image and a load-path file |

Every output also carries `method`. DFT gives the energy to cleave the bulk ceramic, which needs only a small cell. MD gives the energy and the strength of the interface between the ceramic and the interlayer, which needs a larger cell. The FEM model takes both energies and the interface strength, and turns the notched cell that the geometry step describes into pixels. At the size of the cell the fracture is controlled by energy, so the strength does not change its verdict ([fem/README.md](fem/README.md)).

The solvers in the subfolders: [fem/jax_pfczm/](fem/jax_pfczm/SOURCE.md) runs the FEM tool. [dft/fracture_md_qe/](dft/fracture_md_qe/) is the Quantum ESPRESSO adapter of fracture-md. It has no cleavage routine yet, so `dft_cleavage` stays a stand-in. [md/md_agent/](md/md_agent/) (Alex Wang) and [md/fracture_md/](md/fracture_md/) (fracture-md) are MD codes. Neither builds an interface between two materials yet, so `md_interface` stays a stand-in ([md/README.md](md/README.md)).

[`md/bulk.py`](md/bulk.py) runs md_agent on the bulk ceramic and returns its elastic moduli in about 3 s. It is a cross-check of the layer modulus the design takes from the literature, and enters the report beside the tool outputs, not the simulation. It covers alumina only, where it gives a Young's modulus of 348 GPa against about 400 GPa measured. [REQUESTS.md](REQUESTS.md) lists what each tool still needs from its authors so that the DFT and MD slots return real numbers.

`TOOLS` in [`__init__.py`](__init__.py) lists the schemas the engineer sees, and `RUN` maps each tool name to its function. [`schema.py`](schema.py) holds the six materials a design may use and the table of surface energies the stand-ins read.

## Handoff files

For each candidate the pipeline writes one JSON file per tool, `runs/<time>/candidate_<n>/handoff/<folder>.json`. The file is written with status `pending` before the tool runs and updated when it returns, so a slow or failed solver leaves a record of what it was asked. Each input carries its value, its unit and its origin, which is a design field with the evidence the designer cited, the geometry, or an earlier tool output. One input of `md.json` from a live run reads as follows.

```json
"material_b": {"value": "graphite", "from": "design.interlayer", "source": "10.1016_j.actamat.2023.118704#31"}
```

An input that equals no design value, geometry value or earlier output is marked `unmatched`. The judge then gives no verdict on that candidate, since a number entered the simulation from nowhere.

## What a tool has to do

- Take keyword arguments with the unit in each name, as in `layer_modulus_GPa`.
- Return a small JSON-serializable dictionary of named scalars. Fields, crack paths and load-displacement curves go to files, and the dictionary returns their paths.
- Return `method`, a short string that says what produced the numbers. While any method says stand-in, the judge marks its verdict provisional and the reporter labels the results as placeholders.
- Raise an exception on failure. The message goes back to the engineer, which can correct its arguments and call again, and the handoff file records the error.
- Return within the call. The graph waits for the function. A solver that runs longer than a few minutes should be submitted as a job and polled inside the function.

## Attaching a solver

Replace the function body in the tool's folder and keep the names of its inputs and outputs. A solver in the same Python environment is a direct call. A solver in another environment or on another machine is a subprocess or an HTTP request that sends the inputs as JSON and reads JSON back.

```python
def md_interface(material_a, material_b):
    out = subprocess.run(["python", "-m", "my_md.run"], input=json.dumps(locals()),
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)
```

A solver that needs another input from the literature, such as a layer count or an interface waviness, needs a new design field. Add it to `DESIGN` in [`pipeline/design.py`](../pipeline/design.py) with a description of the evidence that supplies it, add its unit pattern to `UNIT` for a numeric field, and add the property to the tool schema with the description "from the design". An input that another tool computes needs only the schema property, with a description that names that tool's output.

## Open points for the tool authors

1. The exact input names, units and allowed ranges of each tool.
2. The quantity DFT supplies. The chain above uses a cleavage energy. Elastic constants or an interface binding energy would change which tool feeds which.
3. The fracture formulation. The pipeline needs to know whether the solver takes a strength or a length scale, and whether interfaces are sharp or diffuse. For the AT2 phase-field model the length that matches a strength s is l = 27 E G / (256 s²) [Tanné 2018]. FEM takes energies only. It uses AT2 with l = 2 pixels and turns each interlayer into a diffuse band at least 3 pixels thick. An MD strength of a few GPa gives a cohesive length E G / s² of nanometres, far below a pixel, which is about 1 µm for 10 µm layers and 19 µm for 210 µm layers, so the cell is in the Griffith limit: the energy ratio and the shape of the cell decide the outcome, and the strength does not. The bands raise the effective energy ratio, which FEM reports as `band_energy_ratio`.
4. The geometry. The geometry step writes a 45 x 4 x 3 mm bend bar and a five-layer notched cell. A solver that needs other inputs, such as a notch length or a layer count, reads them from `geometry_json` or asks for a design field. FEM reads the cell size, the layer count, the notch and the two thicknesses from `geometry_json`.
5. The materials each tool supports. The design allows the six in `schema.py`.
6. The run time per call and the machine each tool runs on. FEM takes 134 to 182 s per call on the CPU of an idle DGX Spark at 60 pixel rows, and a median of 190 s with the language models loaded. Pinned to one fast core it takes 74 to 104 s. [REQUESTS.md](REQUESTS.md) gives the measurements.
7. The scalars each tool returns and the files it writes. FEM returns the outcome, how far the crack runs along the interlayer and into the next layer, the load at which it reached the interlayer and the energy balance, and writes `fem_damage.png` and `fem_load_path.csv` next to `geometry_json`.
