# Models

Three open-weight models run the pipeline by default. gpt-oss-20b does every text job, Qwen3.6-35B-A3B reads the figures with thinking off, and EmbeddingGemma embeds the corpus. Together they are a 36 GB download. With this set the pipeline produced a design that the judge accepted in 9 of 12 runs on 2026-10-23, with a median of 394 s per run ([benchmark.md](benchmark.md)). Larger text models gave no measured gain in the model comparison of early October 2026.

## Default routing

| Agent | Model | Reasoning | Benchmark evidence for this choice |
|---|---|---|---|
| planner | `gpt-oss:20b` | low | Short structured output |
| figure reader | `qwen3.6:35b` | off | 43/50 image questions at 12 s per image. With thinking on it scored 35/50, lost 13 replies and took 83 s |
| designer | `gpt-oss:20b` | medium | The step that weighs evidence. 30/30 passages, and no invented value for the 10 passages without an answer |
| checker | `gpt-oss:20b` | medium | Reads the cited source of each numeric value again, the task of the passage suite, 30/30 |
| engineer | `gpt-oss:20b` | low | 16/16 tool steps, every argument copied exactly |
| judge | `gpt-oss:20b` | medium | Decides each criterion from the design and the tool outputs. No suite scores it alone. The end-to-end runs score it with the rest of the pipeline |
| fabricator | `gpt-oss:20b` | medium | Writes the synthesis and fabrication recipe of the accepted candidate. Added after the end-to-end runs of 2026-10-23 and not yet benchmarked |
| reporter | `gpt-oss:20b` | low | Restates the state |
| retriever | `embeddinggemma` | | Embeds 80,792 records of a 2,880-paper corpus in about 30 minutes |

The routing is the `MODELS` table in [`../graph.py`](../graph.py), one line per agent with a model and a reasoning effort, and `EMBEDDING` names the embedding model. The setup script pulls every model the table names.

## Alternatives

The end-to-end comparison of three model sets in [benchmark.md](benchmark.md) completed 4 of 4 designs with gpt-oss-20b or gpt-oss-120b for the text agents, and 3 of 4 with Qwen3.6-35B-A3B for every agent. The gpt-oss-20b set was the fastest, at a median of 89 s per run against 148 s and 185 s. That comparison predates the geometry step, the judge and the FEM tool.

| Need | Change | Evidence | Cost |
|---|---|---|---|
| One model to install | `qwen3.6:35b` for every agent, thinking on for the designer only | 3 of 4 designs complete, 46/46 text items | Twice the run time, 22 GB download |
| Highest figure accuracy | Figure reader `qwen3.8:27b` with thinking on | 47/50 image questions, no lost reply | 29 s per image against 12 s, and 17 GB more |
| Fastest figure reading | Figure reader `gemma4:31b` with thinking off | 44/50 at 5.7 s per image | About 100 GB in use with a 32K context and four slots. Run it alone and with one slot |
| Smallest footprint | `ministral-3:14b` for every agent | 38/50 image questions, 28/30 passages, 15/16 tool steps | 9 GB download, lower accuracy |
| The largest text model | `gpt-oss:120b` for the text agents | 4 of 4 designs complete, 46/46 text items | 1.7 times the run time, 65 GB download, 22 GB left free. Above the 40 GB limit for a second model in [hardware.md](hardware.md), so it replaces gpt-oss-20b and shares the machine with nothing else |

The top eight image configurations are tied within sampling error at 43 to 47 of 50, so time and memory decide among them. The text suites do not separate the text models, and the end-to-end check does not either, so the smallest and fastest one is the default.

Four models are not suited to the pipeline. Llama 4 Scout passed 1 of 16 tool steps, Mistral Small 3.2 passed 8 and Nemotron 3 Nano Omni 9. Gemma 3 has no tool calling.

## Every model tested

All models are the default Ollama tags at 4-bit (Q4_K_M) unless noted, served from one DGX Spark. Sizes are the download sizes. The license is the one the Ollama tag ships with.

| Model | Ollama tag | Parameters, total and active | Size | Image input | Tool calls | Thinking | License |
|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | `qwen3.6:35b` | 35.5B, 3B active | 22 GB | yes | yes | yes | Apache 2.0 |
| Qwen3.6-27B | `qwen3.6:27b` | 27.3B | 17 GB | yes | yes | yes | Apache 2.0 |
| Qwen3.8-27B | `qwen3.8:27b` | 27.3B | 17 GB | yes | yes | yes | Apache 2.0 |
| Qwen3.5-122B-A10B | `qwen3.5:122b` | 125B, 10B active | 81 GB | yes | yes | yes | Apache 2.0 |
| Qwen3-VL-30B-A3B | `qwen3-vl:30b` | 31.1B, 3B active | 19 GB | yes | yes | always | Apache 2.0 |
| Qwen3-VL-32B | `qwen3-vl:32b` | 33.4B | 20 GB | yes | yes | always | Apache 2.0 |
| Gemma 4 26B-A4B | `gemma4:26b` | 25.2B, 3.8B active | 18 GB | yes | yes | yes | Apache 2.0 |
| Gemma 4 31B | `gemma4:31b` | 30.7B | 20 GB | yes | yes | yes | Apache 2.0 |
| Gemma 3 27B | `gemma3:27b` | 27.4B | 17 GB | yes | no | no | Gemma Terms of Use |
| Muse-Glimmer-30B | `muse-glimmer:30b` | 27.9B | 18 GB | yes | yes | yes | Apache 2.0 |
| Nemotron 3 Nano Omni | `nemotron3:33b` | 33.0B, 3B active | 27 GB | yes | yes | yes | NVIDIA Open Model License |
| Llama 4 Scout | `llama4:scout` | 108.6B, 17B active | 67 GB | yes | yes | no | Llama 4 Community License |
| Mistral Small 3.2 | `mistral-small3.2:24b` | 24.0B | 15 GB | yes | yes | no | Apache 2.0 |
| Ministral 3 14B | `ministral-3:14b` | 13.9B | 9.1 GB | yes | yes | no | Apache 2.0 |
| gpt-oss-120b | `gpt-oss:120b` | 116.8B, 5.1B active, MXFP4 | 65 GB | no | yes | always | Apache 2.0 |
| gpt-oss-20b | `gpt-oss:20b` | 20.9B, 3.6B active, MXFP4 | 13 GB | no | yes | always | Apache 2.0 |
| Nemotron 3 Super | `nemotron-3-super:120b` | 123.6B, 12B active | 86 GB | no | yes | yes | NVIDIA Software and Model Evaluation License |
| Nemotron 3.5 Lightning | `nemotron-3.5-lightning:30b` | 32.9B, 3B active | 25 GB | no | yes | yes | NVIDIA Open Model License |
| GLM-4.7-Flash | `glm-4.7-flash` | 29.9B, 3B active | 19 GB | no | yes | yes | MIT |
| DeepSeek-V4-Flash | community GGUF | 284B, 13B active, about 3 bits per weight | 103 GB | no | yes | yes | MIT |
| EmbeddingGemma | `embeddinggemma` | 308M, BF16 | 0.6 GB | embedding model | | | Gemma Terms of Use |

"Always" means the model thinks on every request and the API cannot turn it off. DeepSeek-V4-Flash is no longer in the Ollama library, and the tested file is a community GGUF build from Hugging Face.

Models that do not fit one Spark at a usable speed were left out. Qwen3-VL-235B needs 143 GB at 4-bit. Mistral Medium 3.5 (128B, dense) fits in 80 GB but decodes at most about 3.4 tokens/s at 273 GB/s.

## Reasoning effort

The OpenAI-compatible API takes a `reasoning_effort` parameter, and each agent sets its own in `MODELS`. The models honor it differently.

| Family | `none` | `low` |
|---|---|---|
| Qwen3.6, Qwen3.8 | turns thinking off | still thinks at length |
| Gemma 4, Muse-Glimmer | turns thinking off | not tested |
| Qwen3-VL | ignored, the model always thinks | ignored |
| gpt-oss | not supported | short reasoning, about 5 times shorter than the default |

On one figure, Qwen3.6-35B-A3B took 12 s with `none` and 66 s with `low`. On a short calculation, gpt-oss-120b took 5.5 s with `low` and 30 s at the default.

## Three behaviors to know

gpt-oss answers with a tool call when a prompt contains tool schemas, even when the request offers no tools. The reply then has no text content. Only the engineer therefore receives the tool schemas, and every other agent sees the design fields alone.

A thinking model can spend its whole output budget on reasoning and return no answer. The pipeline allows 8,192 output tokens per call and raises a clear error when a reply holds no answer ([`../llm.py`](../llm.py)). The figure reader runs with thinking off, and the gpt-oss agents at low or medium effort.

The server under-reports the memory of the Gemma 4 models. It lists Gemma 4 31B at 11 GB, and with that model alone about 100 GB of the machine was in use. Loaded next to gpt-oss-120b it exhausted the memory. [hardware.md](hardware.md) describes the event.
