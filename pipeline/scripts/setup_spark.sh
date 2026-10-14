#!/usr/bin/env bash
# Set up a DGX Spark for the agents. Run it on the Spark from the repository root.
# Builds the Python environment with GPU JAX, installs Ollama in the home folder (no root
# needed), serves it on localhost:11434, pulls the models in pipeline/graph.py, generates the
# benchmark charts and runs the check, which reads one of the charts.
# If an Ollama server already runs on port 11434, skip the lines that install and start it.
set -e
python3 -m venv .venv
.venv/bin/pip install -r pipeline/requirements.txt
models=$(.venv/bin/python -c "from pipeline.graph import *; print(*{m for m, e in MODELS.values()}, EMBEDDING)")
mkdir -p ~/ollama
curl -L https://github.com/ollama/ollama/releases/download/v0.35.0/ollama-linux-arm64.tar.zst | tar --zstd -x -C ~/ollama
# One request at a time, and no more loaded models than the agents use. The server does not
# count its context caches and the machine has no swap, so spare models can exhaust the memory.
OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=$(echo $models | wc -w) \
  OLLAMA_KEEP_ALIVE=30m setsid nohup ~/ollama/bin/ollama serve > ~/ollama/serve.log 2>&1 < /dev/null &
sleep 5
for model in $models; do
  ~/ollama/bin/ollama pull "$model"
done
.venv/bin/python -m pipeline.benchmark.make_charts
.venv/bin/python -m pipeline.check
