"""Run the agents on one question.

python -m pipeline.run "question" corpus_folder

Prints each agent as it finishes and writes the report and the full state to runs/<time>/.
"""

import json
import sys
import time
from pathlib import Path

from . import corpus
from .graph import EMBEDDING, MODELS, build

question, folder = sys.argv[1], sys.argv[2]
app = build(MODELS, corpus.load(folder, EMBEDDING))

start = time.perf_counter()
for mode, chunk in app.stream({"question": question}, {"recursion_limit": 40}, stream_mode=["updates", "values"]):
    if mode == "updates":
        print(f"{time.perf_counter() - start:7.1f} s  {', '.join(chunk)}", flush=True)
    state = chunk

out = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
out.mkdir(parents=True)
(out / "state.json").write_text(json.dumps(state, indent=1, default=str))
(out / "report.md").write_text(state["report"])
print(f"\n{state['report']}\n\nReport and full state in {out}/")
