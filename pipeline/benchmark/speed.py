"""Measure cold-load time, memory, prefill speed and decode speed per model on an Ollama server.

python -m pipeline.benchmark.speed model [model ...]

Decode speed for one request comes from the server's own timing. For four concurrent requests
it is the tokens generated divided by the wall time, because some model architectures are
served one request at a time. Rows are appended to benchmark/results/speed.jsonl.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

base = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").removesuffix("/v1")
here = Path(__file__).parent
passages = [json.loads(line)["passage"] for line in open(here / "data" / "passages.jsonl")]
prompt = "Summarize these passages.\n\n" + "\n\n".join(passages[:12])  # the row records its token count


def generate(model, text, n, raw=False):
    return httpx.post(f"{base}/api/generate", timeout=1800, json={
        "model": model, "prompt": text, "raw": raw, "stream": False,
        "options": {"num_predict": n, "temperature": 0}}).json()


(here / "results").mkdir(exist_ok=True)
for model in sys.argv[1:]:
    httpx.post(f"{base}/api/generate", json={"model": model, "keep_alive": 0})  # unload, for a cold start
    cold = generate(model, "Hello", 1, raw=True)  # raw, so no chat template is cached for the next prompt
    loaded = {m["name"]: m for m in httpx.get(f"{base}/api/ps").json()["models"]}
    warm = generate(model, prompt, 256)
    start = time.perf_counter()
    with ThreadPoolExecutor(4) as pool:  # the prompt is cached by now, so this times decoding
        batch = list(pool.map(lambda _: generate(model, prompt, 256), range(4)))
    wall = time.perf_counter() - start
    row = {"model": model, "load_s": round(cold["load_duration"] / 1e9, 1),
           "memory_GB": round(loaded[model if ":" in model else model + ":latest"]["size"] / 1e9, 1),
           "prompt_tokens": warm["prompt_eval_count"],
           "prefill_tok_s": round(warm["prompt_eval_count"] / warm["prompt_eval_duration"] * 1e9),
           "decode_tok_s": round(warm["eval_count"] / warm["eval_duration"] * 1e9, 1),
           "decode_tok_s_4_parallel": round(sum(b["eval_count"] for b in batch) / wall, 1)}
    httpx.post(f"{base}/api/generate", json={"model": model, "keep_alive": 0})
    with open(here / "results" / "speed.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(row, flush=True)
