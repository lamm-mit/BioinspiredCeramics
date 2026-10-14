# Multi-agent pipeline

The agents turn a design question into a judged candidate design. They search a corpus of papers and a knowledge graph, choose each design value with a cited source, write the part as STL files, and call DFT, MD and FEM tools with those values. A judge then decides whether the candidate meets the criteria, and a candidate it sends back returns to the designer with the judge's findings, for up to three candidates per run. Every model is open-weight and served locally. The three tools are stand-ins until the solvers are attached, as [../tools/README.md](../tools/README.md) describes.

## Agents

```text
planner -> retriever -> figure_reader -> designer -> geometry -> engineer <-> tools -> judge -> reporter
              ^---- a field lacks evidence, once ----'   ^                                     |
                                                         '---- sent back, up to 3 candidates ---'
```

| Step | Default model | Reasoning | Job |
|---|---|---|---|
| planner | gpt-oss-20b | low | Writes search queries and knowledge-graph entities for the design fields |
| retriever | EmbeddingGemma | | Finds passages and figures by cosine similarity, passages that hold a number with a field's unit or name one of its options, and knowledge-graph triples |
| figure reader | Qwen3.6-35B-A3B | off | Reads values from each retrieved figure, with the place in the figure each was read |
| designer | gpt-oss-20b | medium | Proposes a candidate with a cited source for every value, or asks once for another search |
| checker | gpt-oss-20b | medium | Reads the cited source of each numeric value a second time and confirms that the value belongs to the field |
| geometry | none | | Writes a bend bar and a notched simulation cell as STL files, one per material |
| engineer | gpt-oss-20b | low | Calls the tools in order and passes each result to the next call |
| judge | gpt-oss-20b | medium | Decides each criterion from the design and the tool outputs, and names the fields to change |
| reporter | gpt-oss-20b | low | Writes the report with the outcome, the source of every value and the judge's findings |

Code checks surround the models. A design value must appear in the evidence it cites, in the field's unit and within a range the source states, and the designer gets one repair with the findings. A material the tools do not accept, or a repeat of a candidate the judge sent back, is rejected the same way. A design that still lacks a field goes to the reporter, so no tool runs on it. The verdict follows from the judge's answer on each criterion. A tool that failed, or a tool input that matches no design value, geometry value or earlier tool output, leaves the candidate without a verdict. A verdict on stand-in results is marked provisional.

## Output of a run

```text
runs/<time>/
  report.md  state.json
  candidate_1/
    design.json      the design values and the source of each
    geometry/        geometry.json, and bar and cell STL files for the ceramic and the interlayer
    handoff/         dft.json md.json fem.json, the parameters passed to each tool with their origins, and the outputs
    verdict.json     the judge's finding on each criterion
  candidate_2/       written only when the judge sends candidate_1 back
```

`state.json` holds every agent's input and output, and every model call with its latency and token counts.

## Running

Commands run from the repository root. On a DGX Spark the setup script builds the Python environment with GPU JAX, installs Ollama in the home folder without root, serves it with a 32K context, pulls the three default models (36 GB), generates the benchmark charts and runs the check.

```bash
bash pipeline/scripts/setup_spark.sh
```

```bash
.venv/bin/python -m pipeline.run "Design a nacre-inspired layered alumina whose interfaces deflect cracks. Choose an interlayer, its thickness and a layer thickness, and check whether a crack deflects." <corpus>/
```

`python -m pipeline.check` tests the server, each model, image input, one tool call, JAX on the GPU and the tools. The agents need only an OpenAI-compatible endpoint, so a laptop can run them against the Spark's server through an SSH tunnel.

```bash
ssh -N -L 11434:localhost:11434 <user>@<spark-host>
```

```bash
LLM_BASE_URL=http://localhost:11434/v1 python -m pipeline.run "<question>" <corpus>/
```

## Corpus

No corpus is part of this repository. `scripts/build_corpus.py` turns papers converted to markdown, for example by Marker, into `chunks.jsonl`, with one record per passage of up to about 2,000 characters and one per figure with its caption. It turns per-paper triple files, JSON lists of `node_1`, `node_2`, `edge` and `paper`, into `kg.jsonl`. The first load embeds every record and caches the vectors in the corpus folder.

```bash
python pipeline/scripts/build_corpus.py <markdown_papers>/ <corpus>/ <triples>/
```

## Changing the problem

`design.py` holds everything about the design problem. `DESIGN` lists the fields the designer fills, each with a description of the evidence it needs. `UNIT`, `SCALE` and `RANGE` tell the source check how to find a numeric value, and `ASK` is the checker's question for each numeric field. `make_geometry` builds the part, and `CRITERIA` lists what the judge decides. The tools live in [../tools/](../tools/). The prompts in `graph.py` do not name the problem, so a new problem needs new fields, criteria and tools, and no prompt changes.

| Field | What the designer supplies |
|---|---|
| `ceramic` | The layer material, one of the materials the tools accept |
| `interlayer` | The material between layers, chosen so that cracks deflect along it |
| `layer_modulus_GPa` | Young's modulus of the ceramic, as stated in the evidence |
| `layer_thickness_um` | The thickness of one ceramic layer, within the range of layers the evidence reports as made or tested |
| `interlayer_thickness_um` | The thickness of the interlayer, within the range the evidence reports |

## Tests

The tests need no model server. `test_graph` stubs the models and the search, and covers the repair loop, the second reading, the geometry and handoff files, and the judge loop.

```bash
python -m pipeline.tests.test_source_check
python -m pipeline.tests.test_geometry
python -m pipeline.tests.test_graph
```

## Benchmark

`benchmark/run.py` scores models on chart reading, figure reading, passage extraction and tool calling, `speed.py` measures load time, memory and decode speed, `e2e.py` runs the whole graph for three model sets, and `report.py` summarizes the result files. `make_charts.py` generates the 20 chart items. The figure and passage items are taken from published papers and are kept outside the repository, so those two suites run only when their item files are copied into `benchmark/data/`. The tool suite keeps the two-tool chain it was measured with, so its scores stay comparable.

## Status

On 2026-10-14 the default models ran four design questions three times each on an eight-paper corpus kept outside the repository. All 12 runs produced a complete design that passed the source check, with STL files and three handoff files, against 9 of 12 complete designs before the geometry step and the judge were added. The judge accepted 10 candidates at the first attempt. The other 2 runs chose an interlayer the tools do not accept, so the MD tool failed and no verdict was given. The design check now rejects such a choice before the tools run. The median run took 162 s, with a range of 120 to 430 s, on one DGX Spark whose GPU ran at a reduced clock of 507 MHz, so the times are upper bounds. A question that starts from a silicon carbide interlayer and asks to change it if cracks do not deflect then ran twice with the current code. The stand-in gave an energy ratio of 1.0 for that pair, and in both runs the judge sent the candidate back and proposed epoxy. In the first run the epoxy candidate was accepted, after 293 s. In the second the replacement design cited a paper without a passage that states its interlayer thickness. The source check rejected that value, and the run ended without a candidate after 989 s.

## Limitations

The tools are stand-ins. Every verdict is provisional, and no tool output is a prediction.

The judge is a 20-billion-parameter model reading two criteria. Before it was given the earlier candidates of the run, it misread a question that asked to replace a failed interlayer.

The checks confirm that a cited source states a numeric value. For a material they confirm only that the cited evidence exists. In the run above, the passage cited for epoxy mentions epoxy only as an infiltrant. The second reading can also pass a number that belongs to another quantity. In a notebook run the designer took a 0.5 µm pore size as the interlayer thickness, and the geometry was built from it. A person should check the source of each design value before a costly solver run.

Retrieval is cosine similarity over passage embeddings, a unit filter for numeric fields and a name match in the knowledge graph. It does not rerank passages or follow multi-hop paths.

The geometry is a flat laminate, so the loop changes materials and dimensions and never the architecture.

Models run at temperature 0, and two runs can still differ, because inference on the server is not bit-for-bit deterministic.
