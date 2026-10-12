"""Check that the model server, each model, image input, tool calling and JAX work.

python -m pipeline.check

Every check runs even when an earlier one fails, so one run lists everything to fix.
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from tools import RUN, TOOLS
from tools.fem import crack_fea
from tools.md import md_interface

from .graph import EMBEDDING, ENGINEER, MODELS
from .llm import ask_json, chat, client, embed, user


def check(name, f):
    start = time.perf_counter()
    try:
        out = f()
        print(f"ok    {name:26} {time.perf_counter() - start:6.1f} s   {out}")
    except Exception as e:
        print(f"FAIL  {name:26} {type(e).__name__}: {e}")


def missing_models():
    served = {m.id for m in client.models.list()}
    needed = {model for model, effort in MODELS.values()} | {EMBEDDING}
    missing = sorted(m for m in needed if m not in served and f"{m}:latest" not in served)
    if missing:
        raise LookupError(f"{client.base_url} does not serve {missing}, pull them first")
    return f"all {len(needed)} served by {client.base_url}"


def tool_call():
    """The engineer's first step on a small design, with the tool it asks for actually run."""
    design = {"ceramic": "alumina", "interlayer": "PMMA", "layer_modulus_GPa": 380, "layer_thickness_um": 10,
              "interlayer_thickness_um": 2}
    model, effort = MODELS["engineer"]
    reply, _ = chat(model, [{"role": "system", "content": ENGINEER}, user(json.dumps(design))], TOOLS, effort)
    call = reply["tool_calls"][0]["function"]
    return call["name"], call["arguments"], RUN[call["name"]](**json.loads(call["arguments"]))


def simulate():
    layer, interface = md_interface("alumina", "alumina"), md_interface("alumina", "PMMA")
    return crack_fea(380, 10, layer["separation_energy_J_m2"], interface["separation_energy_J_m2"],
                     interface["strength_MPa"], "none")


check("models", missing_models)
check("embedding", lambda: embed(EMBEDDING, ["alumina", "aluminium oxide"]).shape)
model, effort = MODELS["planner"]
check("JSON reply", lambda: ask_json(model, 'Return JSON {"answer": number}.', "What is 6 times 7?", effort=effort)[0])
model, effort = MODELS["figure_reader"]
check("image input (expect 132)", lambda: ask_json(model, 'Return JSON {"answer": number, "unit": string}.',
                                                   "What is the peak traction of this cohesive law?",
                                                   [Path(__file__).parent / "benchmark/data/charts/chart-14.png"], effort)[0])
check("tool call", tool_call)
check("JAX device", lambda: (str(jax.devices()[0]), float((jnp.ones((4096, 4096)) @ jnp.ones((4096, 4096)))[0, 0])))
check("simulation tools", simulate)
