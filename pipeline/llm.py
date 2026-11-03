"""One client for every model. Ollama, vLLM and SGLang all serve this OpenAI-compatible API."""

import base64
import json
import os
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

client = OpenAI(base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
                api_key=os.getenv("LLM_API_KEY", "none"), timeout=1800)


def user(text, images=()):
    """A user message. Images travel inline as base64 data URLs."""
    if not images:
        return {"role": "user", "content": text}
    parts = [{"type": "text", "text": text}]
    for path in images:
        kind = "png" if Path(path).suffix == ".png" else "jpeg"
        data = base64.b64encode(Path(path).read_bytes()).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,{data}"}})
    return {"role": "user", "content": parts}


def chat(model, messages, tools=None, effort=None):
    """One completion. Returns the reply as a plain message dict and a trace record.

    effort is the reasoning effort of a thinking model. "none" turns thinking off where the
    model allows it, and "low" shortens it.
    """
    start = time.perf_counter()
    options = {"tools": tools} if tools else {}
    if effort:
        options["reasoning_effort"] = effort
    r = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=8192, **options)
    reply = r.choices[0].message.model_dump(exclude_none=True)
    record = {"model": model, "seconds": round(time.perf_counter() - start, 1),
              "tokens_in": r.usage.prompt_tokens, "tokens_out": r.usage.completion_tokens,
              "finish": r.choices[0].finish_reason, "reply": reply}
    return reply, record


def parse(content):
    """The first JSON object in a reply. Text around it is ignored, such as a code fence or a brace in K_{R}."""
    for start in (i for i, c in enumerate(content) if c == "{"):
        try:
            return json.JSONDecoder().raw_decode(content[start:])[0]
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no JSON object in the reply: {content[:200]!r}")


def ask_json(model, system, text, images=(), effort=None):
    """Ask for one JSON object. A thinking model returns its reasoning in a separate field."""
    reply, record = chat(model, [{"role": "system", "content": system}, user(text, images)], effort=effort)
    if not reply.get("content"):  # a thinking model can spend the whole token budget on reasoning
        raise ValueError(f"{model} returned no answer (finish: {record['finish']}, {record['tokens_out']} tokens)")
    return parse(reply["content"]), record


def embed(model, texts):
    """Unit-norm embeddings, so a dot product is a cosine similarity."""
    rows = []
    for i in range(0, len(texts), 64):
        rows += [d.embedding for d in client.embeddings.create(model=model, input=texts[i:i + 64]).data]
    v = np.array(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)
