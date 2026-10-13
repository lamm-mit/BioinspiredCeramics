"""One JSON file per tool for each candidate: the parameters passed to the tool, where each came from, and what came back.

A file is written before its tool runs, with status "pending", and again when the tool returns.
A solver author can read the inputs their tool will receive, and a slow solver leaves a record
of what it was asked while it runs.
"""

import json
import math
from pathlib import Path

from tools import RUN, TOOLS

UNITS = {"GPa": "GPa", "MPa": "MPa", "um": "µm", "J_m2": "J/m^2"}  # a numeric input name ends in its unit
DESCRIPTION = {t["function"]["name"]: t["function"]["description"] for t in TOOLS}


def path(s, name):
    """The file of a tool, named after its folder in the tools package, such as handoff/md.json."""
    return Path(s["candidate"]) / "handoff" / f"{RUN[name].__module__.rsplit('.', 1)[-1]}.json"


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6)
    return isinstance(a, str) and a == b


def origin(name, value, s):
    """Where a tool input came from: a design field with its source, the geometry, or an earlier tool output.

    The engineer may only pass values on, so an input that equals none of them is marked unmatched.
    When several origins hold the same value, the one with the input's own name counts, or else all are named.
    """
    found = [{"from": f"design.{k}", "source": s["sources"].get(k)} for k, v in s["design"].items() if same(v, value)]
    found += [{"from": f"geometry.{k}"} for k, v in s["geometry"].items() if same(v, value)]
    found += [{"from": f"{r['tool']}.{k}"} for r in s["results"] for k, v in r["output"].items() if same(v, value)]
    found = [f for f in found if f["from"].endswith("." + name)] or found
    if not found:
        return {"from": "unmatched"}
    return {"from": " or ".join(dict.fromkeys(f["from"] for f in found))} | {k: v for k, v in found[0].items() if k == "source"}


def record(s, name, args):
    """Add a pending call to the tool's file, and return its inputs with their origins."""
    file = path(s, name)
    file.parent.mkdir(parents=True, exist_ok=True)
    content = json.loads(file.read_text()) if file.exists() else {
        "tool": name, "description": DESCRIPTION[name], "candidate": Path(s["candidate"]).name, "calls": []}
    inputs = {k: {"value": v} | {"unit": u for suffix, u in UNITS.items() if k.endswith("_" + suffix)} | origin(k, v, s)
              for k, v in args.items()}
    content["calls"].append({"inputs": inputs, "status": "pending", "output": None})
    file.write_text(json.dumps(content, indent=1, ensure_ascii=False))
    return inputs


def finish(s, name, output):
    """Store what the tool returned in its last call."""
    file = path(s, name)
    content = json.loads(file.read_text())
    content["calls"][-1] |= {"status": "error" if "error" in output else "done", "output": output}
    file.write_text(json.dumps(content, indent=1, ensure_ascii=False))
