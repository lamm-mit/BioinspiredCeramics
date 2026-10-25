"""The report figures of pipeline/docs from the result files. python -m pipeline.docs.figures.make_figures

e2e_outcomes.png and novelty_reward.png read benchmark/results/e2e_pipeline.jsonl. fem_validation.png
draws the validation table of tools/fem/README.md.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

here = Path(__file__).parent
rows = [json.loads(line) for line in open(here.parents[1] / "benchmark" / "results" / "e2e_pipeline.jsonl")]
BLUE, ORANGE, GRAY, INK, GRID = "#2a78d6", "#eb6834", "#b4b2a9", "#0b0b0b", "#d9d9d6"
QUESTIONS = sorted({r["question"] for r in rows}, key=[r["question"] for r in rows].index)
LABELS = ["Open bioinspired design", "Nacre alumina, polymer interlayer", "Bioinspired silicon carbide", "Unreported combination"]
REPEATS = sorted({r["repeat"] for r in rows})

# Outcome of every run, one cell per question and repeat.
fig, ax = plt.subplots(figsize=(6.6, 3.2))
for r in rows:
    x, y = REPEATS.index(r["repeat"]), QUESTIONS.index(r["question"])
    color = BLUE if r.get("best") else ORANGE if r.get("verdicts") else GRAY
    ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color=color, ec="white", lw=3))
    best = max(n for n, v in zip(r["novelty"], r["verdicts"]) if v == "accept") if r.get("best") else None
    text = f"{len(r.get('verdicts') or [])} candidates" + (f", novelty {best:.2f}" if best is not None else "")
    ax.text(x, y, f"{text}\n{r['seconds'] / 60:.0f} min", ha="center", va="center", fontsize=8.5,
            color="white" if color == BLUE else INK)
ax.set(xlim=(-0.5, len(REPEATS) - 0.5), ylim=(len(QUESTIONS) - 0.5, -0.5), xlabel="Repeat")
ax.set_xticks(range(len(REPEATS)), [str(k + 1) for k in REPEATS])
ax.set_yticks(range(len(QUESTIONS)), LABELS[:len(QUESTIONS)])
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(handles=[Patch(color=BLUE, label="Candidate accepted"), Patch(color=ORANGE, label="None accepted"),
                   Patch(color=GRAY, label="No complete design")],
          frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("End-to-End Run Outcomes", fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(here / "e2e_outcomes.png", dpi=200)

# Novelty of each candidate in each run, with the verdict and the target that sends the designer back.
fig, ax = plt.subplots(figsize=(6.6, 3.4))
MARK = {"accept": ("o", BLUE), "revise": ("s", ORANGE), "incomplete": ("X", GRAY)}
for r in rows:
    if not r.get("verdicts"):
        continue
    xs = range(1, len(r["verdicts"]) + 1)
    ax.plot(xs, r["novelty"], color=GRID, lw=1, zorder=1)
    for x, n, v in zip(xs, r["novelty"], r["verdicts"]):
        marker, color = MARK[v]
        ax.scatter(x, n, marker=marker, color=color, s=55, edgecolor="white", lw=1, zorder=3)
    if r.get("best"):
        k = int(r["best"].split("_")[1])
        ax.scatter(k, r["novelty"][k - 1], marker="o", s=170, facecolor="none", edgecolor=INK, lw=1.2, zorder=4)
ax.axhline(0.5, color="#52514e", ls="--", lw=1, zorder=0)
ax.text(3.08, 0.5, "target", fontsize=8, color="#52514e", va="center")
ax.set(xlim=(0.7, 3.4), ylim=(0, 1), xlabel="Candidate in the run", ylabel="Novelty against the corpus")
ax.set_xticks([1, 2, 3])
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(handles=[plt.Line2D([], [], marker=m, ls="", color=c, ms=7, label=label) for (m, c), label in
                   [(MARK["accept"], "Accepted"), (MARK["revise"], "Sent back"), (MARK["incomplete"], "No verdict")]]
          + [plt.Line2D([], [], marker="o", ls="", ms=10, mfc="none", mec=INK, label="Made")],
          frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("Novelty Reward Across Candidates", fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(here / "novelty_reward.png", dpi=200)

# tools/fem/README.md, Validation: outcome per energy ratio and grid, None where not run.
RATIOS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 1]
D, P = True, False
SWEEP = {"40 rows, 28 µm pixels": [D, D, D, P, P, P, P, P, P],
         "60 rows, 19 µm pixels": [D, D, D, D, P, P, P, P, P],
         "80 rows, 14 µm pixels": [None, D, D, D, P, P, P, None, None]}
fig, ax = plt.subplots(figsize=(6.4, 2.9))
ax.axvline(0.25, color="#52514e", ls="--", lw=1, zorder=1)
ax.text(0.25 * 1.04, 2.45, "He-Hutchinson threshold, 0.25", fontsize=8, color="#52514e", va="bottom")
for i, outcomes in enumerate(SWEEP.values()):
    for ratio, d in zip(RATIOS, outcomes):
        if d is not None:
            ax.scatter(ratio, i, s=70, marker="o" if d else "s", color=BLUE if d else ORANGE, edgecolor="white", lw=1.5, zorder=3)
ax.set_xscale("log")
ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1], ["0.05", "0.1", "0.2", "0.5", "1"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set(xlim=(0.04, 1.25), ylim=(2.7, -0.6), xlabel="Energy ratio, interface over layer fracture energy", ylabel="Grid resolution")
ax.set_yticks(range(3), list(SWEEP))
ax.tick_params(axis="y", length=0)
ax.grid(True, axis="x", which="major", lw=0.3, color=GRID)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="Deflected"),
                   plt.Line2D([], [], marker="s", ls="", color=ORANGE, ms=8, label="Penetrated")],
          frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("Crack Deflection Versus Energy Ratio", fontsize=12, loc="left")
fig.tight_layout()
fig.savefig(here / "fem_validation.png", dpi=200)
