"""A figure of a part, from the section and the toolpaths its geometry.json describes.

The cell is drawn by its x-z section, the plane a crack model works on, and the bottom three plies by the
ceramic roads of their toolpaths over a window at mid-bar, so ply rotation, tablets, waves and rings show.
"""

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .pattern import describe, roads

COLOR = {"ceramic": "#b08a4a", "interlayer": "#2b2b2b", "mark": "#c0392b"}
# Ceramic of the lowest modulus is drawn pale, and of the highest in the ceramic color.
GRADED = LinearSegmentedColormap.from_list("graded", ["#f1e6cf", COLOR["ceramic"]])


def band(points, width):
    """The outline of a road of the given width along points, with square ends."""
    t = np.gradient(points, axis=0)
    n = np.column_stack([-t[:, 1], t[:, 0]]) / np.linalg.norm(t, axis=1, keepdims=True) * width / 2
    return np.concatenate([points + n, (points - n)[::-1]])


def render_part(geometry_json, out_png):
    """Draw the cell section and the roads of the bottom three plies of geometry_json into out_png, and return its path."""
    spec = json.loads(Path(geometry_json).read_text())
    design, pattern, bar, cell = spec["design"], spec["pattern"], spec["bar"], spec["cell"]
    w, s = spec["road_width_mm"], design["interlayer_thickness_um"] / 1000
    (x0, y0, _), (cw, _, ch) = cell["origin_mm"], cell["size_mm"]
    moduli = [p["modulus_GPa"] for p in spec["plies"]]
    norm = Normalize(min(moduli), max(moduli)) if max(moduli) > min(moduli) else None

    # Axes are placed in inches, and the figure is as tall as the section needs at equal aspect.
    H = min(2.1, 8.4 * ch / cw)
    tall = 5.95 + H
    fig = plt.figure(figsize=(10, tall))

    def place(x, y, dx, dy):
        ax = fig.add_axes([x / 10, y / tall, dx / 10, dy / tall])
        ax.spines[["top", "right"]].set_visible(False)
        return ax

    ax = place(0.9, 4.65, 8.4, H)
    sec = np.load(cell["section"])
    fill = GRADED(norm(sec["modulus_GPa"]))[..., :3] if norm else to_rgb(COLOR["ceramic"])
    ax.imshow(np.where((sec["phase"] == 0)[..., None], fill, to_rgb(COLOR["interlayer"])), origin="lower",
              extent=(x0, x0 + cw, 0, ch), interpolation="antialiased")
    notch = cell["notch"]
    ax.plot([x0 + notch["at_mm"]] * 2, [notch["from_z_mm"], notch["from_z_mm"] + notch["depth_mm"]], color=COLOR["mark"],
            linewidth=3, solid_capstyle="butt")
    ax.set(xlim=(x0, x0 + cw), ylim=(0, ch), aspect="equal", xlabel="x (mm)", ylabel="z (mm)")
    ax.set_title(f"Simulation Cell, Section at y = {y0:.2f} mm", fontsize=11)
    if norm:
        fig.colorbar(plt.cm.ScalarMappable(norm, GRADED), ax=ax, fraction=0.03, pad=0.02, label="modulus (GPa)")

    # A window of the bar width at mid-length. Each road is drawn at its ceramic width over the interlayer. Under
    # sheet placement neighboring roads touch, so their outlines show the road direction.
    length, width, height = bar["size_mm"]
    side = min(4.0, width)
    sheet = pattern["interlayer_placement"] == "sheet"
    for k in range(3):
        ax = place(0.9 + 3 * k, 1.2, 2.6, 2.6)
        # Roads depend on absolute position, so a footprint ending one road width past the window draws the same
        # roads inside it as the whole bar, in less than half the time for rings.
        paths = roads(pattern, design, bar["ceramic_layers"], height, k, (length / 2 + side / 2 + w, width))["ceramic"]
        color = GRADED(norm(moduli[k])) if norm else COLOR["ceramic"]
        ax.add_collection(PolyCollection([band(p[:, :2], w if sheet else w - s) for p in paths], facecolor=color,
                                         edgecolor=tuple(0.6 * c for c in to_rgb(color)) if sheet else "none", linewidth=0.3))
        ax.plot([x0, x0 + cw], [y0, y0], color=COLOR["mark"], linewidth=1.2, linestyle="--")
        ax.set_facecolor(COLOR["interlayer"])
        ax.set(xlim=(length / 2 - side / 2, length / 2 + side / 2), ylim=(width / 2 - side / 2, width / 2 + side / 2),
               aspect="equal", xlabel="x (mm)", ylabel="y (mm)" if k == 0 else None)
        ax.set_title(f"{['Bottom', 'Second', 'Third'][k]} Ply, {spec['plies'][k]['angle_deg']:g}°"
                     + (f", {moduli[k]:.0f} GPa" if norm else ""), fontsize=11)

    fig.legend(handles=[Patch(color=COLOR["ceramic"], label=f"{design['ceramic']}, {design['layer_thickness_um']:g} µm layers, "
                                                            f"{1000 * w:g} µm roads"),
                        Patch(color=COLOR["interlayer"], label=f"{design['interlayer']}, {design['interlayer_thickness_um']:g} µm interlayers"),
                        Patch(color=COLOR["mark"], label=f"notch, {1000 * notch['depth_mm']:g} µm deep"),
                        Line2D([], [], color=COLOR["mark"], linestyle="--", label="section plane")],
               loc="lower center", ncol=2, frameon=False, fontsize=10)
    words = describe(pattern)
    fig.suptitle("Printed Part Structure", fontsize=14, y=1 - 0.15 / tall, va="top")
    fig.text(0.5, 1 - 0.5 / tall, textwrap.fill(words[0].upper() + words[1:], 110), ha="center", va="top", fontsize=10.5)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return str(out_png)
