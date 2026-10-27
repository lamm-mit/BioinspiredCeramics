# Hardware

The pipeline runs on one NVIDIA DGX Spark with 128 GB of unified memory. The three default models take 36 GB of weights on its GPU. The FEM tool runs on its CPU at 164 to 411 s per call, median 174 s (n = 11), and a run with up to three candidates took a median of 21 min on 2026-10-25. Memory is the binding limit. On 2026-10-21 a 65 GB model of another job, loaded beside the pipeline's models, left 2 GB free and stalled the machine, so the pipeline should share the Spark with no model above about 40 GB. Decode speed is bounded by the 273 GB/s memory bandwidth, which favors mixture-of-experts models with few active parameters.

## Requirements

| Component | Runs on | Measured | Environment |
|---|---|---|---|
| Language models, gpt-oss-20b, Qwen3.6-35B-A3B and EmbeddingGemma | Spark GPU, 36 GB of weights | median 394 s per pipeline run, n = 12, 2026-10-23 | Ollama 0.35.0, 32K context, one request at a time ([`../scripts/setup_spark.sh`](../scripts/setup_spark.sh)) |
| FEM, `fem_fracture` with jax-pfczm (AT2 phase field) | Spark CPU | median 190 s and maximum 418 s per call at 60 pixel rows, with the default models loaded, 2026-10-23. 134 to 182 s on the idle Spark, and 74 to 104 s pinned to one fast core, 2026-10-25 ([REQUESTS.md](../../tools/REQUESTS.md)) | the pipeline environment with JAX 0.11.2 ([`../requirements.txt`](../requirements.txt)) |
| MD, `md_interface` with MACE-MP-0 medium and D3(BJ) | Spark GPU | one run per pair of materials, 16 to 19 min for graphite, 47 to 58 min for a polymer and 63 to 101 min for silica glass, with two or three runs sharing the GPU, 2026-10-27 | a second environment, `~/simtools/env`, with PyTorch, MACE and torch-dftd ([README](../../tools/md/README.md)) |
| MD, `md_bulk` with md_agent | Spark | alumina with the Pedone potential in 2.2 to 2.5 s, silicon carbide with MACE-MP-0 in about 30 s. md_agent's 33 validation checks in 107 s | ASE in the pipeline environment for alumina, the MD environment for silicon carbide ([requirements](../../tools/md/md_agent/requirements.txt)) |
| DFT, `dft_cleavage` with Quantum ESPRESSO 7.5 | Spark CPU, 8 MPI ranks | one run per ceramic, silicon carbide in 2.5 h and alumina in 9.8 h, 2026-10-27 | a separate conda environment, `fmd-dft` ([README](../../tools/dft/README.md)) |
| Tests | Apple M3 Pro laptop | FEM tests in 13 to 14 s, graph test in 243 s, seven test scripts with 181 checks | the pipeline environment, no model server |

The GPU of the Spark ran at a reduced clock of 507 MHz during the model benchmark and the runs of 2026-10-14, so the model times in these documents are upper bounds (Known issues, below).

`dft_cleavage` and `md_interface` depend only on the materials, so each ceramic and each pair is computed once and kept in the tool's `results/` folder. In a run they read those files and return at once. A new ceramic costs one DFT run of a few hours on the CPU, and a new interlayer one MD run of up to 2 h per ceramic on the GPU ([../../tools/README.md](../../tools/README.md)).

The tests and the agents also run on a laptop. The tests need no model server and use a coarse FEM grid of 30 pixel rows. The agents reach the Spark's model server through an SSH tunnel ([../README.md](../README.md)).

## Memory budget

| Item | Memory, GB |
|---|---|
| gpt-oss-20b | 13 |
| Qwen3.6-35B-A3B | 22 |
| EmbeddingGemma | 0.6 |
| Default models, total | 36 |
| Largest other model that may share the machine | 40 |
| Left for context caches, solvers and the system | about 52 |

The sizes are download sizes. With a 32K context and four slots the server reports 13.4 GB for gpt-oss-20b and 22.1 GB for Qwen3.6-35B-A3B. In the model comparison of early October 2026 the lowest free memory during full runs was 67 GB with the default models, and 22 GB with gpt-oss-120b in place of gpt-oss-20b.

On 2026-10-21 another job loaded a 120B model of 65 GB beside the pipeline's two models. The weights alone then came to 101 GB of 128 GB. Free memory fell to 2 GB, the kernel logged GPU out-of-memory errors, and the machine stalled for minutes. The machine has no swap, and the model server does not count its context checkpoints, so the free memory it implies is too high.

The operating rule follows from this event. Before a run, list what the model server has loaded with `ollama ps` and check for other jobs on the machine. Do not run the pipeline beside another model above about 40 GB. That keeps the loaded weights at 76 GB or less and leaves about 52 GB for the context caches, the solvers and the system.

## The reference machine

| Item | Value |
|---|---|
| System | NVIDIA DGX Spark, DGX OS 7.2.3 (Ubuntu 24.04.3) |
| Chip | GB10 Grace Blackwell, 20 Arm cores (10 Cortex-X925, 10 Cortex-A725) |
| Memory | 128 GB LPDDR5x shared by CPU and GPU, 256-bit, 273 GB/s |
| GPU software | Driver 580.82.09, CUDA 13.0, compute capability 12.1 |
| Storage | 4 TB NVMe |
| Model server | Ollama 0.35.0, 32K context. The benchmark used 4 parallel slots per model, and the setup script uses 1 |
| Python stack | JAX 0.11.2 with the `cuda13` plugin, LangGraph 1.2.12 |

NVIDIA lists up to 1 PFLOP of FP4 compute with sparsity, a 240 W power supply, and ConnectX-7 networking at 200 Gb/s that links two units. It states that one unit runs models up to 200 billion parameters and two linked units run 405 billion [1, 2]. The Spark launched at $3,999 in October 2025, and its list price has been $4,699 since February 2026 [3].

## Why this machine

**Memory for the models and the solvers together.** The default routing loads gpt-oss-20b, Qwen3.6-35B-A3B and EmbeddingGemma, 36 GB of weights, and leaves the rest of the 128 GB to the solvers. The benchmark loaded every candidate up to a 103 GB model on this one unit. A 32 GB consumer GPU holds one of the two default models at a time.

**The data-center CUDA stack.** JAX installs from the standard `jax[cuda13]` wheels on the Spark's Arm CPU and runs on its GPU [4]. A finite-element or molecular-dynamics code written in JAX on this machine moves to a cluster GPU unchanged. The FEM tool runs its solver on the CPU to leave the GPU to the models. Apple and AMD unified-memory machines offer similar capacity without CUDA. On Apple GPUs JAX runs only through the experimental jax-metal plugin [5].

**Local and open.** Weights, corpus and results stay on the machine. No request leaves it, and every default model is open-weight under a license that permits local use.

**Cost and power.** One unit costs about as much as 1,100 to 1,600 hours of an H100 in the cloud at $2.89 to $4.41 per hour [6]. It draws at most 240 W.

## What it costs in speed

Decoding one token reads every active weight once, so single-request speed is at most the bandwidth divided by the active weight bytes. At 4 bits per weight that bound is about 16 tokens/s for a dense 27B model and about 150 tokens/s for a model with 3B active parameters.

| Model | Kind | Weights read per token | Bound, tokens/s | Measured decode, tokens/s |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | mixture of experts, 3B active | about 1.8 GB | about 150 | 84 |
| gpt-oss-120b | mixture of experts, 5.1B active | about 2.7 GB | about 100 | 43 |
| Qwen3.6-27B | dense | about 16.5 GB | about 16 | 19 |

The two mixture-of-experts rows were measured at the normal GPU clock. The dense row was measured at the reduced clock described under known issues, where it still met its bound. The dense model therefore runs as fast as the memory allows, and no setting makes it faster on this machine. The two mixture-of-experts models reach about half of their bound and still decode 2 to 4 times faster than the dense model. [benchmark.md](benchmark.md) lists load time, memory and speed for all 20 models.

Two settings matter. A mixture-of-experts model reads only its active experts per token, which is the main lever. Concurrent requests help for some architectures. With four requests at once the server raised total throughput by a factor of 1.1 to 2.1 for gpt-oss, Gemma, Mistral, Muse-Glimmer, Llama 4 and GLM, and gave no gain for the Qwen, Nemotron and DeepSeek models.

Reasoning models spend most of their time thinking. With thinking on, the time per image question was 2 to 7 times longer at the same accuracy, and three models lost 9 to 13 answers of 96 because the reasoning used the whole token budget. The default routing therefore turns thinking off for the figure reader. It keeps gpt-oss at low effort for the planner, engineer and reporter, and at medium effort for the designer, checker, judge and fabricator.

## Alternatives

| System | Memory | Bandwidth | CUDA | Approximate price, September 2026 |
|---|---|---|---|---|
| DGX Spark (GB10) | 128 GB unified | 273 GB/s | Yes | $4,699 [3] |
| RTX 5090 | 32 GB | 1,792 GB/s | Yes | $1,999 list, $4,299 or more in stores [7] |
| RTX PRO 6000 Blackwell | 96 GB | 1,792 GB/s | Yes | $8,500 to $16,000, card only [8] |
| Mac Studio M5 Ultra | 96 to 512 GB unified | 1.2 TB/s | No | from $5,499 [9] |
| Ryzen AI Max+ 395 systems | 128 GB unified | 256 GB/s | No | about $2,000 to $3,500 [10] |
| H100 80 GB, cloud | 80 GB | 3.35 TB/s | Yes | $2.89 to $4.41 per hour [6] |

A discrete GPU fits when one mid-size model must answer fast and nothing else needs to stay resident. A Mac fits when the goal is the largest possible model and GPU JAX is not needed. The cloud fits short bursts and models above 200 billion parameters. The agents only need an OpenAI-compatible endpoint, so any of these can serve the models while the agents and the tools run elsewhere.

## Known issues on the Spark

JAX reserves 75% of GPU memory at its first operation by default. On unified memory that is 96 GB (90 GiB), which the model server also needs, and the allocation fails. [`tools/fem/__init__.py`](../../tools/fem/__init__.py) sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`, so JAX allocates on demand.

Memory can run out without warning. The machine has no swap, and the server does not count two things. It keeps context checkpoints in host memory, up to 32 per slot at 0.3 to 0.8 GB each, and it under-reports the Gemma 4 models. It lists Gemma 4 31B at 11 GB, and with that model alone loaded about 100 GB was in use. With gpt-oss-120b and Gemma 4 31B loaded together, available memory fell to 0.2 GB and the whole machine stalled for 25 minutes before it recovered. The setup script therefore serves one request at a time and loads no more models than the agents use. With gpt-oss-120b, Qwen3.6-35B-A3B and EmbeddingGemma loaded, 10 to 22 GB stayed available. The stall of 2026-10-21, described under Memory budget, had the same mechanism, with a model of another job.

The test unit also stopped once without any log entry. Two benchmark runs had loaded it for two hours, and a Gemma 4 model had been loaded next to another model four minutes earlier, so memory exhaustion is the likely cause. After the restart the GPU clock stayed at 507 MHz of 3,003 MHz, which halved the decode speed of mixture-of-experts models. That part is unexplained. The unit then ran the full benchmark for 18 hours without a fault, with the GPU at 58 °C or below. The two speeds given above at the normal clock were measured before the restart, and all other benchmark timings after it. The clock was still at 507 MHz during the end-to-end runs of 2026-10-14.

Ollama evicts the least recently used model when a new one does not fit. Reloading a model costs 10 to 37 s.

The GPU can fault on a single request. On 2026-10-26 the server process of Qwen3.6-35B-A3B hit three GPU page faults (Xid 31) within 30 s while it read one figure of the full corpus, with 90 GB of memory free. Ollama restarted the process, but the request failed. The figure reader now skips a figure the server fails on and records the error in the run's trace, so one figure no longer ends a run.

## Sources

1. NVIDIA DGX Spark product page, https://www.nvidia.com/en-us/products/workstations/dgx-spark/
2. NVIDIA DGX Spark hardware guide, https://docs.nvidia.com/dgx/dgx-spark/hardware.html
3. NVIDIA developer forum, price change announcement of 23 February 2026, https://forums.developer.nvidia.com/t/2-23-2026-price-change-announcement/361713
4. JAX installation guide, https://docs.jax.dev/en/latest/installation.html
5. Apple, Accelerated JAX on Mac, https://developer.apple.com/metal/jax/
6. Lambda and RunPod on-demand pricing, https://lambda.ai/pricing and https://www.runpod.io/pricing
7. NVIDIA GeForce RTX 50 series comparison, https://www.nvidia.com/en-us/geforce/graphics-cards/compare/, and Tom's Hardware on RTX 5090 retail prices, https://www.tomshardware.com/pc-components/gpus/nvidias-rtx-5090-vanishes-from-online-retail-in-the-us-third-party-sellers-now-demand-as-much-as-usd9-500-for-nvidias-fastest-gpu
8. NVIDIA RTX PRO 6000 Blackwell, https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/, and TechSpot on its price, https://www.techspot.com/news/113460-nvidia-raises-rtx-pro-6000-blackwell-price-staggering.html
9. Apple newsroom, Mac Studio with M5 Max and M5 Ultra, https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/
10. AMD Ryzen AI Max+ 395 product page, https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html
