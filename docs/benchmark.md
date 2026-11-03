# Benchmark

Twenty open-weight models were scored in 26 configurations on one DGX Spark on 1 and 2 October 2026. Image reading separates them, from 24 to 47 correct answers out of 50. Passage extraction and tool calling do not, since 20 of the 26 configurations score at least 45 out of 46. Turning thinking off left image accuracy unchanged within sampling error and cut the time per image by a factor of 1.9 to 7.5. With the chosen models the whole pipeline, with the novelty reward and the fabricator, made an accepted candidate with its recipe and print files in 4 of 8 runs on 2026-10-25, at a median of 21 min per run. The FEM tool places the switch from crack deflection to penetration next to the He-Hutchinson threshold.

## Setup

The benchmark ran on one NVIDIA DGX Spark, a GB10 Grace Blackwell with 20 Arm cores (10 Cortex-X925 and 10 Cortex-A725) and 128 GB of LPDDR5X memory shared by the CPU and the GPU at 273 GB/s. Its GPU ran at a reduced clock of 507 MHz of 3,003 MHz throughout, so every time below is an upper bound ([hardware.md](hardware.md)). The 26 configurations ran one after another on 1 and 2 October 2026, in about 18 hours, with the GPU at or below 58 °C.

| Layer | Setting |
|---|---|
| Model server | Ollama 0.35.0, `OLLAMA_CONTEXT_LENGTH=32768`, `OLLAMA_NUM_PARALLEL=4`, `OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_MAX_LOADED_MODELS=3` |
| Model files | The default Ollama tag of each model, 4-bit Q4_K_M or MXFP4 for gpt-oss, and a community GGUF for DeepSeek-V4-Flash ([models.md](models.md)) |
| Client | The OpenAI Python SDK 3.22.1 against Ollama's OpenAI-compatible endpoint, through the same `chat` and `ask_json` functions the agents use ([`../llm.py`](../pipeline/llm.py)) |
| Sampling | Temperature 0, at most 8,192 output tokens, each model's default reasoning unless the configuration names an effort |
| Concurrency | Four requests in flight per suite. A configuration's next suite starts when its last item returns |
| Scoring | Deterministic, against the stored answer and tolerance of each item, in [`../pipeline/benchmark/run.py`](../pipeline/benchmark/run.py). No model and no person judges an answer |

## Models

A model entered the benchmark when four conditions held. Its weights are open. Its license allows local use. It fits one Spark with its 32K context cache and decodes at a usable speed. It has a standard Ollama tag, with DeepSeek-V4-Flash as the one exception. Within those limits the set covers the image-capable open families available as Ollama tags in late September 2026, because the figure reader needs vision and no family was known to be best at reading plots, and the text-only reasoning models that could run the text agents. Qwen3-VL-235B (143 GB at 4 bits) and Mistral Medium 3.5 (at most about 3.4 tokens/s) were left out for size and speed.

Twenty models gave 26 configurations. Five models ran a second time with thinking off (`@none`), since thinking multiplies the time per image, and gpt-oss-120b ran a second time at low reasoning effort (`@low`). The 19 image-capable configurations took all four suites. The 7 text-only configurations took passage extraction and tool calling only, and Gemma 3 skipped tool calling because it has no tool interface. [models.md](models.md) lists every model with its size, license and capabilities.

## Why these suites

Each suite tests one job of one agent, and each stress inside a suite reproduces a failure seen in earlier pipeline runs. A suite therefore predicts how a model behaves in its seat in the pipeline, and the end-to-end runs test the seats together.

| Suite | Agent and job | Failure it reproduces, and how |
|---|---|---|
| Chart reading | Figure reader, reading a value off a plot | Misread log axes, so one chart type in five uses a log axis. Answers from memory of real materials, so every value is random and the sample names are neutral |
| Paper figures | Figure reader, on real published figures | Clean charts are easier than published ones, with panels, legends and overlapping markers. Each answer is stated in the paper text, so the ground truth is the authors' number |
| Passage extraction | Designer and checker, taking a value from a passage | Invented values, so 10 of 30 passages lack the quantity and a number there counts as invented. The wrong number from a sentence with several, as with a 140 µm layer of another material taken for a 210 µm alumina layer, so 19 of the 20 answerable passages hold competing numbers. Unit slips, so 5 need a conversion |
| Tool calling | Engineer, passing values between tools | A mis-copied argument enters a simulation without any error, so step two checks that every input equals the design value or the earlier tool output within 0.1% |
| Speed and memory | All agents together | The default set must fit beside the solvers in 128 GB and finish a run in minutes, so `speed.py` measures load time, memory, prefill and decode |

A reply cut off at the token limit counts as wrong, because thinking models exhausted the budget in pipeline runs and returned no answer. Only generated items and items from CC BY 4.0 papers are used, so the benchmark can be shared and rerun. The suites do not test whether a design is scientifically sound, how good a hypothesis is, or the judge and the fabricator, which only the end-to-end runs exercise. [benchmark_items.md](benchmark_items.md) lists all 96 items with their answers, tolerances and how many configurations answered each correctly.

## Suites

Each suite tests one job an agent does, through the same client the agents use.

| Suite | Items | Source | Scored as correct when | Runs from the repository alone |
|---|---|---|---|---|
| Chart reading | 20 | Synthetic charts with exact answers | The value is within 3% of the axis span, 12% on log axes, or 1% when the value is printed | yes |
| Paper figures | 30 | Figures from 19 CC BY papers | The value is within 5% when printed and 10% when read from a bar or curve, or the label matches | no |
| Passage extraction | 30 | Passages from 22 CC BY papers | The value is within 1% in the unit asked, or null when the passage does not state it | no |
| Tool calling | 16 | Eight designs, two steps each | Step one calls `md_interface` with the right materials. Step two calls `crack_fea` with all five inputs within 0.1% | yes |

The chart and tool suites run from the repository alone. `make_charts.py` regenerates every chart and answer, and `run.py` builds the 16 tool items from its list of designs. The figure and passage suites quote published papers, so their item files are kept outside the repository and `run.py` skips a suite whose file is absent. `speed.py` and `report.py` also read `passages.jsonl`, the first for its prompt and the second to find the 10 passages without an answer. The result files of all four suites are in [`../pipeline/benchmark/results/`](../pipeline/benchmark/results/). They hold item ids, verdicts, timings and the first 300 characters of each reply.

The chart suite covers five chart types that recur in materials papers. They are flexural stress-strain curves, bar charts with error bars, a property map with a log axis, traction-separation laws and rising crack-resistance curves. Every value is random, so knowledge of real materials cannot replace reading, and the property maps also use neutral sample names.

The figure suite holds 24 numeric and 6 categorical items. For each numeric item the paper text states the value and a reviewer confirmed that the value is readable in the figure. The model sees the image and the question and not the caption. Where a neighboring bar or point fell inside the tolerance band, the item was moved to a separated target.

The passage suite holds 20 items that state the quantity and 10 that do not. Nineteen of the 20 contain competing numbers for other samples or conditions, and 5 need a unit conversion. An answer to one of the 10 absent quantities counts as an invented value.

The tool suite gives the engineer prompt and a design, then checks the calls. Step one passes when every call is `md_interface` with a pair the design needs, because a model may request the two pairs together or one per turn. Step two supplies both `md_interface` results and checks that `crack_fea` receives them unchanged. It tests argument copying, which is where a wrong number would enter a simulation. The suite keeps the two-tool chain it was measured with, so its scores stay comparable. The pipeline now calls three tools, `dft_cleavage`, `md_interface` and `fem_fracture`, and the end-to-end runs test that chain.

## Protocol

Each model answers each item once at temperature 0, with four requests in flight and at most 8,192 output tokens. A reply with no JSON object, a reply cut off at the token limit, or a failed request counts as wrong. The time per item is the wall time of a suite divided by its item count. That number is comparable between models, because the server runs some architectures four requests at a time and others one at a time. Reasoning runs at each model's default unless the name carries an effort, as in `qwen3.6:35b@none` for thinking off or `gpt-oss:120b@low`. All models are the default Ollama tags, which are 4-bit (Q4_K_M, or MXFP4 for gpt-oss) unless noted.

`speed.py` measures each model separately. It records the cold load time, the memory the server reports (weights plus the cache for a 32K context and four slots), prefill speed on a prompt of about 2,300 tokens, and decode speed for 256 tokens with one request and with four concurrent requests.

With 20 to 30 items per suite, the 95% interval on a score spans about ±14 to ±20 percentage points, and ±19 to ±22 points for the 16 tool items. A difference between two models smaller than about 20 points is not significant.

## Results

### Image-capable models

Image reading separates the models. Passage extraction and tool calling sit at or near full marks for 20 of the 26 configurations.

| Model | Thinking | Charts | Paper figures | Images, total | Passages | Tools | Failed replies | Seconds per image |
|---|---|---|---|---|---|---|---|---|
| `qwen3.8:27b` | on | 20/20 | 27/30 | 47/50 | 30/30 | 16/16 | 0 | 28.9 |
| `gemma4:31b` | on | 19/20 | 27/30 | 46/50 | 30/30 | 16/16 | 2 | 38.1 |
| `qwen3.6:27b` | off | 18/20 | 27/30 | 45/50 | 30/30 | 16/16 | 0 | 20.4 |
| `gemma4:31b` | off | 19/20 | 25/30 | 44/50 | 30/30 | 16/16 | 0 | 5.7 |
| `muse-glimmer:30b` | off | 17/20 | 27/30 | 44/50 | 30/30 | 16/16 | 0 | 19.3 |
| `muse-glimmer:30b` | on | 18/20 | 26/30 | 44/50 | 30/30 | 16/16 | 0 | 37.0 |
| `qwen3.6:35b` | off | 17/20 | 26/30 | 43/50 | 30/30 | 16/16 | 0 | 12.2 |
| `qwen3.8:27b` | off | 19/20 | 24/30 | 43/50 | 29/30 | 16/16 | 0 | 12.5 |
| `qwen3-vl:30b` | on | 18/20 | 23/30 | 41/50 | 30/30 | 16/16 | 0 | 11.7 |
| `qwen3.6:27b` | on | 17/20 | 24/30 | 41/50 | 30/30 | 16/16 | 4 | 153.6 |
| `qwen3-vl:32b` | on | 15/20 | 25/30 | 40/50 | 30/30 | 16/16 | 3 | 148.6 |
| `qwen3.5:122b` | on | 16/20 | 24/30 | 40/50 | 28/30 | 16/16 | 9 | 201.6 |
| `ministral-3:14b` | none | 14/20 | 24/30 | 38/50 | 28/30 | 15/16 | 0 | 3.1 |
| `qwen3.6:35b` | on | 11/20 | 24/30 | 35/50 | 30/30 | 16/16 | 13 | 83.1 |
| `gemma4:26b` | on | 12/20 | 22/30 | 34/50 | 30/30 | 16/16 | 11 | 32.2 |
| `mistral-small3.2:24b` | none | 10/20 | 21/30 | 31/50 | 27/30 | 8/16 | 1 | 5.0 |
| `nemotron3:33b` | on | 16/20 | 15/30 | 31/50 | 30/30 | 9/16 | 0 | 24.8 |
| `gemma3:27b` | none | 12/20 | 18/30 | 30/50 | 28/30 | n/a | 0 | 4.8 |
| `llama4:scout` | none | 10/20 | 14/30 | 24/50 | 29/30 | 1/16 | 1 | 20.1 |

"None" marks a model that has no thinking mode. A failed reply is one with no JSON object or one cut off at the token limit.

The top eight configurations score 43 to 47 out of 50. At 50 items the 95% interval of each score is about 9 percentage points on either side, or 4 to 5 answers, so these eight are tied on accuracy and differ in time and memory.

Thinking costs time and can cost the answer. Five models ran with thinking on and off. Turning it off changed the image score by between 4 fewer and 8 more correct answers out of 50, and no change is significant at the 5% level (paired exact test, p from 0.06 to 1.0). It cut the time per image by a factor of 1.9 to 7.5. With thinking on, Qwen3.6-35B-A3B, Gemma 4 26B and Qwen3.5-122B lost 13, 11 and 9 replies because the reasoning used the whole 8,192-token budget. Qwen3.8-27B is the exception. It reasons briefly, lost no reply and reached the highest score.

Three of the four configurations on the speed and accuracy frontier run at thinking off or without a thinking mode. Only Qwen3.8-27B gains accuracy from thinking, at 29 s per image.

### Text-only models

| Model | Reasoning effort | Passages | Tools | Invented values | Seconds per item |
|---|---|---|---|---|---|
| `gpt-oss:20b` | default | 30/30 | 16/16 | 0/10 | 3.8 |
| `gpt-oss:120b` | default | 30/30 | 16/16 | 0/10 | 5.1 |
| `glm-4.7-flash` | default | 30/30 | 16/16 | 0/10 | 9.3 |
| `nemotron-3.5-lightning:30b` | default | 30/30 | 16/16 | 0/10 | 13.2 |
| `gpt-oss:120b` | low | 29/30 | 16/16 | 1/10 | 3.1 |
| `deepseek-v4-flash` | default | 29/30 | 16/16 | 1/10 | 11.2 |
| `nemotron-3-super:120b` | default | 29/30 | 16/16 | 1/10 | 17.6 |

Passage extraction and tool calling are solved by most models. Across all 26 configurations, 5 of 260 answers to an absent quantity were invented values, and no configuration invented more than one. Tool calling fails only for Llama 4 Scout (1/16), Mistral Small 3.2 (8/16), Nemotron 3 Nano Omni (9/16) and Ministral 3 (15/16). These two suites therefore act as a floor. They tell which models can do the job, and they cannot rank the models that pass.

### Speed and memory

`speed.py` ran on each model right after its accuracy run.

**Measurement condition.** During these runs the GPU clock of the test unit was held at 507 MHz of a 3,003 MHz maximum. Two models were also measured on the same unit at the normal clock. Qwen3.6-35B-A3B decoded 83.7 tokens/s and prefilled 2,267 tokens/s, and gpt-oss-120b decoded 43 tokens/s. That is 2.0 times the decode speed and 3.7 times the prefill speed listed below. The reduced clock slows compute-bound work, which is prefill and the decoding of mixture-of-experts models. Dense models decode at the memory-bandwidth bound in either state. Every model time in this document, including the seconds per item above, was taken at the reduced clock, so the times are upper bounds.

| Model | Loaded memory, GB | Cold load, s | Prefill, tokens/s | Decode, tokens/s | Decode with 4 requests, tokens/s |
|---|---|---|---|---|---|
| `nemotron-3.5-lightning:30b` | 26 | 12 | 703 | 49.3 | 44.2 |
| `gemma4:26b` |  | 17 | 859 | 46.6 | 56.0 |
| `qwen3.6:35b` | 22 | 12 | 614 | 42.2 | 28.4 |
| `nemotron3:33b` | 25 | 11 | 751 | 40.7 | 38.8 |
| `qwen3-vl:30b` | 22 | 11 | 821 | 35.1 | 34.2 |
| `gpt-oss:20b` | 13 | 10 | 1145 | 30.9 | 55.8 |
| `glm-4.7-flash` | 26 | 10 | 672 | 29.3 | 40.6 |
| `gpt-oss:120b` | 65 | 20 | 585 | 21.9 | 33.4 |
| `qwen3.6:27b` | 17 | 12 | 230 | 18.7 | 17.8 |
| `qwen3.8:27b` | 18 | 12 | 230 | 16.3 | 16.0 |
| `gemma4:31b` |  | 19 | 221 | 16.0 | 17.8 |
| `qwen3.5:122b` | 80 | 26 | 237 | 14.9 | 14.4 |
| `ministral-3:14b` | 30 | 10 | 559 | 14.6 | 26.6 |
| `nemotron-3-super:120b` | 86 | 29 | 221 | 12.5 | 12.1 |
| `llama4:scout` | 71 | 22 | 255 | 10.6 | 14.9 |
| `deepseek-v4-flash` | 103 | 37 | 101 | 9.5 | 9.1 |
| `mistral-small3.2:24b` | 36 | 13 | 335 | 8.6 | 16.3 |
| `muse-glimmer:30b` | 18 | 11 | 282 | 7.7 | 16.1 |
| `gemma3:27b` | 20 | 14 | 276 | 7.4 | 13.3 |
| `qwen3-vl:32b` | 29 | 12 | 226 | 6.4 | 6.3 |

Loaded memory is what the server reports with a 32K context and four slots. The server under-reports the two Gemma 4 models, so those cells are empty.

Three patterns hold across the table. Mixture-of-experts models with about 3B active parameters decode at 29 to 49 tokens/s, and dense models of 24 to 33B at 6 to 19 tokens/s. Four concurrent requests raise total throughput by a factor of 1.1 to 2.1 for gpt-oss, Gemma, Mistral, Muse-Glimmer, Llama 4 and GLM. They give no gain for the Qwen, Nemotron and DeepSeek models, and the server logs that it runs the Qwen mixture-of-experts architectures one request at a time. The context cache can exceed the weights, as Mistral Small 3.2 loads at 36 GB from a 15 GB download.

## Whole pipeline

### Toolpath patterns on the full corpus, 2026-10-26

The four questions of `e2e.py` ask since 2026-10-26 for structures that a toolpath pattern carries, open, nacre tablets, Bouligand plies and a graded modulus. They ran twice each on the full corpus with the stand-in DFT and MD tools, before the knowledge-graph proposer. All 8 runs simulated a candidate, 20 in all, and the judge accepted 5, all from the open and nacre questions. Six runs ended with a complete design whose tool inputs matched the design and the earlier tool outputs. In the other 2 the revision after the first candidate lacked a field. All 6 Bouligand candidates sent the crack through a joint, which the FEM tool does not count as deflection, since the 2D section cuts the rotated plies and their tablet ends. Their novelty was 0.41 to 0.56, against 0.19 to 0.30 for the other questions. The runs took 18 to 170 min, median 47 min, while DFT and MD runs shared the Spark.

### Before toolpath patterns, 2026-10-25

Four of 8 runs made a candidate, with its STL files, three handoff files, a recipe of 6 to 7 steps citing 15 to 26 values from passages that state them, and G-code for direct ink writing. The open question and the nacre question made a candidate in 4 of 4 runs, while the silicon carbide question and the unreported-combination question made none.

The runs used the eight-paper corpus kept outside the repository, the FEM tool at 60 pixel rows, the stand-in DFT and MD tools, a novelty target of 0.5 and at most 3 candidates per run. The 8 runs simulated 11 candidates, and the judge accepted 6. The silicon carbide runs ended before any simulation, because the eight papers state no layer or interlayer thickness for silicon carbide and the source check rejected every value the designer proposed. The judge sent back all 3 candidates of the unreported-combination question, since their sources report nacre-inspired alumina with graphite interlayers made by direct ink writing. The median run took 21 min, with a range of 6 to 41 min, and the 11 FEM calls took 164 to 411 s, with a median of 174 s.

Every accepted candidate on this corpus scored a novelty between 0.15 and 0.23, below the target of 0.5. The one candidate above the target, at 0.55, was sent back by the judge. The reward loop therefore spent its candidates without an accepted design above the target, and each run made its accepted candidate with the highest reward.

### Before the novelty reward and the fabricator, 2026-10-23

Four older questions ran three times each with the judge and the FEM tool, before the hypothesis fields, the novelty reward and the fabricator were added. The judge accepted 9 of 12 runs, and the other 3 ended with a design that still lacked a field. The runs used the eight-paper corpus kept outside the repository, the FEM tool at 60 pixel rows, and the stand-in DFT and MD tools. All 9 accepted runs wrote the STL files of the part and the three handoff files. The other 3 ended with a design that still lacked a field, which goes to the reporter without a simulation. The median run took 394 s, with a range of 215 to 943 s. The two accepted runs of the polymer question took 925 and 943 s, the longest of the set. Each FEM call took a median of 190 s ([hardware.md](hardware.md)). Every verdict is provisional, since two of the three tools are stand-ins.

A two-step question starts from a silicon carbide interlayer and asks to change it if cracks do not deflect. It was accepted in 1 of 4 runs across 2026-10-14 and 2026-10-23. In the accepted run the judge sent the silicon carbide candidate back, proposed epoxy, and accepted the epoxy candidate after 293 s. In one of the failed runs the replacement design cited a paper without a passage that states its interlayer thickness, the source check rejected that value, and the run ended without a candidate.

### Earlier runs, 2026-10-14

Four design questions ran three times each on the same corpus, with a stand-in FEM tool. All 12 runs produced a complete design, and the judge accepted 10 at the first attempt. The other 2 chose an interlayer the tools did not accept, so the MD tool failed and no verdict was given. The design check now rejects such a choice before the tools run. The median run took 162 s, with a range of 120 to 430 s.

### Model sets, early October 2026

`e2e.py` ran the whole agent graph on four design questions per model set, on the example corpus, with the server handling one request at a time. The graph then had no geometry step, no judge and no FEM tool, and its crack tool was a closed-form stand-in.

| Models | Runs | Complete design | Every field cited | Simulation ran | Exact tool inputs | Median time, s | Lowest free memory, GB |
|---|---|---|---|---|---|---|---|
| `gpt-oss:20b` for the text agents, `qwen3.6:35b` figure reader | 4 | 4 | 4 | 4 | 4 | 89 | 67 |
| `gpt-oss:120b` for the text agents, `qwen3.6:35b` figure reader | 4 | 4 | 4 | 4 | 4 | 148 | 22 |
| `qwen3.6:35b` for every agent | 4 | 3 | 3 | 3 | 3 | 185 | 84 |

Eleven of the 12 runs produced a complete design with a cited source for every field. In all eleven the crack tool received exactly the design values and the `md_interface` outputs, so no number entered a simulation from anywhere else. The twelfth run found no layer thickness. Its design stayed incomplete, no tool was called, and the report named the missing field.

Four runs per set cannot rank the sets on reliability. The check shows that the pipeline completes with each of them, and that the median time differs by a factor of 2 and the lowest free memory by a factor of 4. A previous pass with three questions gave 3 of 3 for the first and third set. In that pass Gemma 4 31B for every agent also completed 3 of 3, at a median of 351 s and with about 100 GB of memory in use.

## FEM tool validation

The sweep of crack outcome against energy ratio puts the switch from deflection to penetration between ratios of 0.2 and 0.25 at 60 and 80 pixel rows, next to the threshold of 0.25 for elastically similar layers [He and Hutchinson 1989], and between 0.15 and 0.2 on the coarsest grid of 40 rows.

Each point is one simulation of the notched cell, 24 in all, with alumina layers of 210 µm, interlayers of 20 µm, a modulus of 367 GPa and a layer fracture energy of 3 J/m². The thresholds of the deflection criterion were set before the sweep and not changed after it. On the 40-row grid the interlayer bands are 85 µm thick, 40% of a layer, and a diffuse band raises the energy that a crack along it meets. The energy balance closed within 1.6% at 60 rows and 2.4% at 80 rows. The pipeline runs the tool at 60 rows. [tools/fem/README.md](../tools/fem/README.md) gives the model and its limits.

## Reproducing

Commands run from the repository root.

```bash
python -m pipeline.benchmark.make_charts
python -m pipeline.benchmark.run qwen3.6:35b qwen3.6:35b@none gpt-oss:120b
python -m pipeline.benchmark.speed qwen3.6:35b gpt-oss:120b
python -m pipeline.benchmark.e2e <corpus>/
python -m pipeline.benchmark.report
```

`run.py` writes one row per item to `benchmark/results/<model>.jsonl`, with the answer, the verdict, the latency, the output tokens, the reasoning length and the start of the raw reply, and skips a model whose file exists. `e2e.py` runs the whole agent graph on four questions per model set and appends one row per run to `benchmark/results/e2e.jsonl`, which now holds the rows of the early October comparison. The rows of the runs of 2026-10-14 and 2026-10-23 are not in that file. `report.py` writes `benchmark/results/summary.md` and the two benchmark figures.
