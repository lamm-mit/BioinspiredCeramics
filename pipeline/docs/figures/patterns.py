"""patterns.png, seven toolpath patterns through the phase-field tool. python -m pipeline.docs.figures.patterns

Every cell is alumina with PMMA interlayers, 70 µm layers and 35 µm interlayers, a modulus of 367 GPa and a layer
fracture energy of 3 J/m². The interface fracture energy is set by the energy ratio, and the interface strength
of 100 MPa enters only the cohesive length. The inputs are illustrative. The first seven panels show the damage at an
energy ratio of 0.2 for each pattern, and the last panel the outcome over ratios from 0.1 to 0.5 for three of them.
Runs are cached in runs/patterns/results.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread

from pipeline.geometry import layered_geometry
from tools.fem import fem_fracture

here, work = Path(__file__).parent, Path("runs") / "patterns"
DESIGN = {"ceramic": "alumina", "interlayer": "PMMA", "layer_thickness_um": 70, "interlayer_thickness_um": 35, "layer_modulus_GPa": 367}
LAYER_ENERGY = 3.0
PATTERNS = {"Flat layers": {},
            "Nacre tablets": {"break_length_h": 20, "interlayer_placement": "shell"},
            "Bouligand plies, 15°": {"ply_rotation_deg": 15, "interlayer_placement": "shell"},
            "Crossed lamellae, ±30°": {"road_angle_deg": -30, "ply_rotation_deg": 60, "ply_rotation_alternates": True,
                                       "interlayer_placement": "shell"},
            "Osteon rings": {"road_path": "rings", "ring_spacing_w": 8, "interlayer_placement": "shell"},
            "Wavy plies": {"ply_wave_amplitude_h": 0.5, "ply_wave_length_h": 30},
            "Modulus graded to 200 GPa": {"modulus_profile": "linear", "outer_layer_modulus_GPa": 200}}
SWEPT, RATIOS = ["Flat layers", "Nacre tablets", "Bouligand plies, 15°"], [0.1, 0.2, 0.3, 0.4, 0.5]
MARK = {"deflected": ("o", "#2a78d6"), "through joint": ("D", "#1baa6b"), "penetrated": ("s", "#eb6834")}

if __name__ == "__main__":
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "results.json"
    results = json.loads(cache.read_text()) if cache.exists() else {}
    for name, pattern in PATTERNS.items():
        for ratio in RATIOS if name in SWEPT else [0.2]:
            key = f"{name} | {ratio}"
            if key not in results:
                folder = work / f"{list(PATTERNS).index(name)}_{ratio}"
                g = layered_geometry(DESIGN, pattern, folder)["geometry_json"]
                out = fem_fracture(DESIGN["layer_modulus_GPa"], DESIGN["layer_thickness_um"], LAYER_ENERGY, ratio * LAYER_ENERGY, 100, g)
                results[key] = {k: out.get(k) for k in ["outcome", "crack_deflects", "nearest_joint_um", "seconds", "damage_png", "pixel_um"]}
                cache.write_text(json.dumps(results, indent=1))
                print(key, results[key]["outcome"], results[key]["seconds"], "s", flush=True)

    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.08)
    for i, name in enumerate(PATTERNS):
        r = results[f"{name} | 0.2"]
        ax = fig.add_subplot(grid[i // 2, i % 2])
        ax.imshow(imread(r["damage_png"]))
        ax.set_title(f"{name}: {r['outcome']}", fontsize=10, loc="left")
        ax.axis("off")
    ax = fig.add_subplot(grid[3, 1].subgridspec(1, 2, width_ratios=[0.32, 1])[1])  # room for the row labels
    for y, name in enumerate(SWEPT):
        for ratio in RATIOS:
            marker, color = MARK[results[f"{name} | {ratio}"]["outcome"]]
            ax.scatter(ratio, y, marker=marker, color=color, s=70, edgecolor="white", lw=1.5, zorder=3)
    ax.axvline(0.25, color="#52514e", ls="--", lw=1, zorder=1)
    ax.text(0.255, len(SWEPT) - 0.55, "He-Hutchinson, 0.25", fontsize=8, color="#52514e")
    ax.set(xlim=(0.05, 0.55), ylim=(len(SWEPT) - 0.4, -0.6), xlabel="Energy ratio, interface over layer fracture energy")
    ax.set_yticks(range(len(SWEPT)), SWEPT)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[plt.Line2D([], [], marker=m, ls="", color=c, ms=8, label=k) for k, (m, c) in MARK.items()],
              frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Outcome Versus Energy Ratio", fontsize=11, loc="left")
    fig.suptitle("Crack Paths in Seven Toolpath Patterns", fontsize=14, x=0.125, ha="left", y=0.995)
    fig.savefig(here / "patterns.png", dpi=150, bbox_inches="tight")
