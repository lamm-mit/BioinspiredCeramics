"""Summarize the benchmark into results/summary.md, results/accuracy.png and results/speed_accuracy.png.

python -m pipeline.benchmark.report
"""

import json
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

here = Path(__file__).parent
results = here / "results"
SUITES = {"charts": "Chart reading", "figures": "Paper figures", "passages": "Passage extraction",
          "tools": "Tool calling"}


def read(name):
    return [json.loads(line) for line in open(results / name)] if (results / name).exists() else []


rows = [r for p in sorted(results.glob("*.jsonl")) if p.stem not in ("speed", "e2e") for r in read(p.name)]
speed = {r["model"]: r for r in read("speed.jsonl")}
e2e = read("e2e.jsonl")
absent = {i["id"] for i in map(json.loads, open(here / "data" / "passages.jsonl")) if i["answer"] is None}


def cell(model, suite):
    group = [r for r in rows if r["model"] == model and r["suite"] == suite]
    return (sum(r["correct"] for r in group), len(group)) if group else None


def rate(model, suites):
    cells = [c for s in suites if (c := cell(model, s))]
    return sum(k for k, n in cells) / sum(n for k, n in cells) if cells else None


models = sorted({r["model"] for r in rows},
                key=lambda m: (-(rate(m, ["charts", "figures"]) or -1), -(rate(m, ["passages", "tools"]) or -1)))


def seconds(model, suites):
    """Wall time per item over these suites, with four requests in flight."""
    group = [r for r in rows if r["model"] == model and r["suite"] in suites]
    return statistics.mean(r["suite_s_per_item"] for r in group) if group else None


lines = ["| Model | " + " | ".join(SUITES.values()) + " | Invented values | Failed replies | "
         "s per image item | s per text item |", "|---" * (len(SUITES) + 5) + "|"]
for m in models:
    cells = [f"{c[0]}/{c[1]} ({c[0] / c[1]:.0%})" if (c := cell(m, s)) else "n/a" for s in SUITES]
    invented = [r for r in rows if r["model"] == m and r["id"] in absent]
    made_up = sum(str(r.get("answer")).lower() not in ("none", "null", "") for r in invented)
    failed = sum("error" in r or r.get("finish") == "length" for r in rows if r["model"] == m)
    times = [f"{t:.1f}" if (t := seconds(m, group)) is not None else "n/a"
             for group in (("charts", "figures"), ("passages", "tools"))]
    lines.append(f"| {m} | " + " | ".join(cells) + f" | {made_up}/{len(invented)} | {failed} | "
                 + " | ".join(times) + " |")

lines += ["", "| Model | Load s | Memory GB | Prompt tokens | Prefill tok/s | Decode tok/s | "
          "Decode tok/s, 4 requests |", "|---|---|---|---|---|---|---|"]
lines += [f"| {m} | {s['load_s']} | {s['memory_GB']} | {s['prompt_tokens']} | {s['prefill_tok_s']} | "
          f"{s['decode_tok_s']} | {s['decode_tok_s_4_parallel']} |" for m, s in speed.items()]

lines += ["", "| Models | Runs | Complete design | Every field cited | Simulated | Exact tool inputs | Median s |",
          "|---|---|---|---|---|---|---|"]
for config in dict.fromkeys(r["config"] for r in e2e):
    runs = [r for r in e2e if r["config"] == config]
    counts = [sum(bool(r.get(k)) for r in runs) for k in ("complete", "cited", "simulated", "faithful")]
    lines.append(f"| {config} | {len(runs)} | " + " | ".join(map(str, counts))
                 + f" | {statistics.median(r['seconds'] for r in runs):.0f} |")
(results / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

def label(model):
    return model.replace("@none", ", thinking off").replace("@low", ", low effort")


# The heatmap uses one hue from light to dark, a count in every cell, and gray where a model
# cannot run a suite. Most scores are above 0.8, so the color scale is stretched toward the top.
grid = np.array([[(c[0] / c[1] if (c := cell(m, s)) else np.nan) for s in SUITES] for m in models])
cmap = LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#2a78d6", "#0e3d75"])
cmap.set_bad("#e6e6e3")
norm = PowerNorm(2.5, vmin=0, vmax=1)
fig, ax = plt.subplots(figsize=(7.8, 0.4 * len(models) + 1.6))
im = ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
for i, m in enumerate(models):
    for j, s in enumerate(SUITES):
        c = cell(m, s)
        ax.text(j, i, f"{c[0]}/{c[1]}" if c else "n/a", ha="center", va="center", fontsize=8.5,
                color="white" if c and norm(c[0] / c[1]) > 0.4 else "#0b0b0b")
ax.set_xticks(range(len(SUITES)), [f"{v}\n(n = {max((cell(m, s) or (0, 0))[1] for m in models)})"
                                   for s, v in SUITES.items()], fontsize=9)
ax.set_yticks(range(len(models)), [label(m) for m in models], fontsize=9)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-0.5, len(SUITES)), minor=True)
ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
ax.grid(which="minor", color="white", lw=2)
vision = sum(cell(m, "charts") is not None for m in models)
ax.axhline(vision - 0.5, color="#0b0b0b", lw=1)  # image-capable models above the line, text-only below
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Model Benchmark Results", fontsize=12, loc="left")
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Fraction correct", ticks=[0, 0.5, 0.7, 0.8, 0.9, 1])
fig.tight_layout()
fig.savefig(results / "accuracy.png", dpi=200)

# Image-question accuracy against the wall time per image question, with thinking time included.
# Only the models on the speed-accuracy frontier are labeled. The table names every point.
points = [(seconds(m, ("charts", "figures")), rate(m, ["charts", "figures"]), m) for m in models
          if rate(m, ["charts", "figures"]) is not None]
thinks = {m: statistics.median(r.get("reasoning_chars", 0) for r in rows
                               if r["model"] == m and r["suite"] in ("charts", "figures")) > 0 for _, _, m in points}
fig, ax = plt.subplots(figsize=(6.8, 4.4))
for on, color, name in ((True, "#8a8a86", "thinking on"),
                        (False, "#2a78d6", "thinking off, or a model without thinking")):
    group = [p for p in points if thinks[p[2]] == on]
    ax.scatter([p[0] for p in group], [p[1] for p in group], s=42, color=color, edgecolor="white", zorder=3,
               label=name)
for x, y, m in points:
    if not any(x2 < x and y2 >= y for x2, y2, _ in points):
        ax.annotate(label(m), (x, y), textcoords="offset points", xytext=(8, 7), ha="right", fontsize=7.5,
                    color="#0b0b0b")
ax.set(xscale="log", xlabel="Seconds per image question", ylabel="Image questions, fraction correct", ylim=(0.4, 1))
ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.margins(x=0.2)
ax.grid(True, which="major", lw=0.3, color="#d9d9d6")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Image Reading Accuracy and Time", fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(results / "speed_accuracy.png", dpi=200)
