"""Score models on four suites. python -m pipeline.benchmark.run model [model ...]

charts     read a value from a synthetic chart with an exact answer (vision)
figures    read a value or a label from a real paper figure (vision)
passages   extract a value from a real passage, or null when it is absent
tools      call md_interface, then crack_fea with the values it returned

Each model answers each item once at temperature 0, four requests at a time, through the
same client the agents use. A model name may carry a reasoning effort, as in qwen3.6:35b@none.
Rows go to benchmark/results/<model>.jsonl. Each row carries its suite's wall time per item,
which is comparable between models whether or not the server runs their requests in parallel.
A model with a results file is skipped, so an interrupted campaign resumes where it stopped.
"""

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from tools import TOOLS
from tools.md import md_interface

from ..graph import ENGINEER
from ..llm import ask_json, chat, client, user

here = Path(__file__).parent
VISION = ("Answer the question from the image. Give a label exactly as it is written in the figure. "
          'Return JSON {"answer": number or string, "unit": string}.')
EXTRACT = ("Extract the quantity the question asks for from the passage, in the unit asked. "
           'Return JSON {"value": number or null, "unit": string}. Use null when the passage does not state it.')
DESIGNS = [("alumina", "PMMA", 10, 380), ("alumina", "epoxy", 25, 390), ("alumina", "graphite", 50, 400),
           ("alumina", "silica glass", 5, 370), ("silicon carbide", "graphite", 100, 410),
           ("silicon carbide", "PMMA", 20, 420), ("silicon carbide", "epoxy", 40, 430),
           ("silicon carbide", "silica glass", 15, 450)]


def matches(answer, truth, tolerance):
    if truth is None:
        return answer is None or str(answer).lower() in ("null", "none", "")
    if isinstance(truth, str):
        return re.sub(r"\W", "", str(answer).lower()) == re.sub(r"\W", "", truth.lower())
    # Models sometimes write a number as text, such as "1,694 MPa".
    m = re.search(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", str(answer).replace(",", ""))
    return m is not None and abs(float(m.group()) - truth) <= max(tolerance, 1e-6) * abs(truth)


def tool_calls(reply):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in reply.get("tool_calls", [])]


def tools_item(model, effort, design, stage):
    ceramic, interlayer, thickness, modulus = design
    messages = [{"role": "system", "content": ENGINEER}, user(json.dumps(
        {"ceramic": ceramic, "interlayer": interlayer, "layer_modulus_GPa": modulus, "layer_thickness_um": thickness}))]
    pairs = [(ceramic, ceramic), (ceramic, interlayer)]
    if stage == 1:  # a model may request both pairs at once or one per turn, so one valid call passes
        reply, rec = chat(model, messages, TOOLS, effort)
        calls = tool_calls(reply)
        ok = bool(calls) and all(n == "md_interface" and {(a.get("material_a"), a.get("material_b")),
                                                          (a.get("material_b"), a.get("material_a"))} & set(pairs)
                                 for n, a in calls)
        return calls, ok, rec
    # Stage 2 starts after both md_interface calls have returned.
    outputs = [md_interface(a, b) for a, b in pairs]
    messages.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{k}", "type": "function", "function": {"name": "md_interface", "arguments": json.dumps(
            {"material_a": a, "material_b": b})}} for k, (a, b) in enumerate(pairs)]})
    messages += [{"role": "tool", "tool_call_id": f"call_{k}", "name": "md_interface", "content": json.dumps(o)}
                 for k, o in enumerate(outputs)]
    expected = {"layer_modulus_GPa": modulus, "layer_thickness_um": thickness,
                "layer_cleavage_energy_J_m2": outputs[0]["separation_energy_J_m2"],
                "interface_energy_J_m2": outputs[1]["separation_energy_J_m2"],
                "interface_strength_MPa": outputs[1]["strength_MPa"]}
    reply, rec = chat(model, messages, TOOLS, effort)
    calls = tool_calls(reply)
    ok = any(n == "crack_fea" and all(matches(a.get(k), v, 1e-3) for k, v in expected.items()) for n, a in calls)
    return calls, ok, rec


def score(name, suite, item):
    model, _, effort = name.partition("@")
    effort = effort or None
    start = time.perf_counter()
    row = {"model": name, "suite": suite, "id": item["id"]}
    try:
        if suite in ("charts", "figures"):
            out, rec = ask_json(model, VISION, item["question"], [here / "data" / item["image"]], effort)
            row["answer"] = out.get("answer")
        elif suite == "passages":
            out, rec = ask_json(model, EXTRACT, f"Passage: {item['passage']}\nQuestion: {item['question']} "
                                                f"Unit: {item['unit']}", (), effort)
            row["answer"] = out.get("value")
        else:
            row["answer"], ok, rec = tools_item(model, effort, item["design"], item["stage"])
        truth_ok = ok if suite == "tools" else matches(row["answer"], item["answer"], item["tolerance"])
        row |= {"correct": truth_ok and rec["finish"] != "length", "seconds": rec["seconds"],
                "tokens_out": rec["tokens_out"], "finish": rec["finish"],
                "reasoning_chars": len(rec["reply"].get("reasoning") or ""),
                "reply": (rec["reply"].get("content") or "")[:300]}
    except Exception as e:  # malformed JSON, truncation or a refused request counts as a wrong answer
        row |= {"correct": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                "seconds": round(time.perf_counter() - start, 1)}
    return row


items = {s: [json.loads(line) for line in open(here / "data" / f"{s}.jsonl")]
         for s in ("charts", "figures", "passages")}
items["tools"] = [{"id": f"tools-{d[0]}-{d[1]}-{stage}", "design": d, "stage": stage}
                  for d in DESIGNS for stage in (1, 2)]

for name in sys.argv[1:]:
    path = here / "results" / f"{name.replace(':', '_').replace('/', '_')}.jsonl"
    if path.exists():
        continue
    # Ollama lists what a model accepts, for example vision and tools.
    caps = httpx.post(str(client.base_url).replace("/v1/", "/api/show"),
                      json={"model": name.partition("@")[0]}).json()["capabilities"]
    suites = [s for s in items
              if (s not in ("charts", "figures") or "vision" in caps) and (s != "tools" or "tools" in caps)]
    rows, start = [], time.perf_counter()
    for s in suites:
        suite_start = time.perf_counter()
        with ThreadPoolExecutor(4) as pool:
            done = list(pool.map(lambda item: score(name, s, item), items[s]))
        per_item = round((time.perf_counter() - suite_start) / len(done), 1)
        rows += [r | {"suite_s_per_item": per_item} for r in done]
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{name}: {sum(r['correct'] for r in rows)}/{len(rows)} correct in {time.perf_counter() - start:.0f} s",
          flush=True)
