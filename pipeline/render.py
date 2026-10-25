"""A figure of a layered part, drawn from its STL files read back from disk.

The bar is drawn in an oblique view and the cell by its front face, the x-z section a crack model works on.
"""

import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

COLOR = {"ceramic": "#b08a4a", "interlayer": "#2b2b2b", "mark": "#c0392b"}


def triangles(path):
    """The triangles of an ASCII STL file, each as three (x, y, z) points in mm."""
    v = [tuple(map(float, m)) for m in re.findall(r"vertex (\S+) (\S+) (\S+)", Path(path).read_text())]
    return [v[i:i + 3] for i in range(0, len(v), 3)]


def render_part(geometry_json, out_png):
    """Draw the bar and the notched cell of geometry_json into out_png, and return its path."""
    spec = json.loads(Path(geometry_json).read_text())
    design, bar, cell = spec["design"], spec["bar"], spec["cell"]
    fig = plt.figure(figsize=(10, 5))

    # Each bar triangle that faces the viewer is drawn, lowest first, so a layer hides the top face of the
    # layer below. Depth y recedes at 32 degrees, scaled by 0.55. Edges take the face color to hide the mesh diagonals.
    ax = fig.add_axes([0.08, 0.6, 0.88, 0.26])
    cx, cz = 0.55 * math.cos(math.radians(32)), 0.55 * math.sin(math.radians(32))
    shade = {"front": 1.0, "side": 0.86, "top": 1.08}
    faces = []
    for role in ("ceramic", "interlayer"):
        rgb = matplotlib.colors.to_rgb(COLOR[role])
        for t in triangles(bar["files"][role]):
            u, v = [t[1][k] - t[0][k] for k in range(3)], [t[2][k] - t[0][k] for k in range(3)]
            n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            face = "front" if n[1] < 0 else "side" if n[0] > 0 else "top" if n[2] > 0 else None
            if face:
                color = tuple(min(1, c * shade[face]) for c in rgb)
                faces.append((sum(p[2] for p in t) / 3, [(p[0] + cx * p[1], p[2] + cz * p[1]) for p in t], color))
    faces.sort(key=lambda f: f[0])
    ax.add_collection(PolyCollection([f[1] for f in faces], facecolors=[f[2] for f in faces],
                                     edgecolors=[f[2] for f in faces], linewidth=0.3, antialiased=False))
    length, width, height = bar["size_mm"]
    ax.set(xlim=(-0.5, length + cx * width + 0.5), ylim=(-0.1 * height, height + cz * width + 0.1 * height), aspect="equal",
           xticks=range(0, int(length) + 1, 5), yticks=[0, round(height, 2)], xlabel="x (mm)", ylabel="z (mm)")
    ax.set_title(f"Print Bar, {length:g} x {width:g} x {height:g} mm", fontsize=11)

    ax = fig.add_axes([0.08, 0.17, 0.88, 0.3])
    for role in ("ceramic", "interlayer"):
        front = [[(p[0], p[2]) for p in t] for t in triangles(cell["files"][role]) if all(abs(p[1]) < 1e-9 for p in t)]
        ax.add_collection(PolyCollection(front, facecolor=COLOR[role], edgecolor=COLOR[role], linewidth=0.8))
    notch = cell["notch"]
    ax.plot([notch["at_mm"]] * 2, [notch["from_z_mm"], notch["from_z_mm"] + notch["depth_mm"]], color=COLOR["mark"],
            linewidth=2.5, solid_capstyle="butt")
    ax.set(xlim=(0, cell["size_mm"][0]), ylim=(0, cell["size_mm"][2]), aspect="equal", xlabel="x (mm)", ylabel="z (mm)")
    ax.set_title("Simulation Cell, Front Face", fontsize=11)
    for a in fig.axes:
        a.spines[["top", "right"]].set_visible(False)

    fig.legend(handles=[Patch(color=COLOR["ceramic"], label=f"{design['ceramic']}, {design['layer_thickness_um']:g} µm layers"),
                        Patch(color=COLOR["interlayer"], label=f"{design['interlayer']}, {design['interlayer_thickness_um']:g} µm interlayers"),
                        Patch(color=COLOR["mark"], label=f"notch, {1000 * notch['depth_mm']:g} µm deep")],
               loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Layered Ceramic Geometry", fontsize=14, y=0.97)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return str(out_png)
