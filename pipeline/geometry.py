"""A layered part as STL files and a JSON description, built from design values.

The description is the source. The STL files are generated from it, one per material, for a
test bar to print and for a small notched cell to simulate. Lengths in the files are in mm.
Layers stack along z, with a ceramic layer at the bottom and at the top.
"""

import json
from pathlib import Path


def box(x0, y0, z0, x1, y1, z1):
    """The 12 triangles of an axis-aligned box, each counter-clockwise seen from outside."""
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
    return [(v[a], v[b], v[c]) for a, b, c, d in quads] + [(v[a], v[c], v[d]) for a, b, c, d in quads]


def volume(triangles):
    """Enclosed volume, from the signed tetrahedra between each triangle and the origin."""
    return sum(a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0])
               for a, b, c in triangles) / 6


def watertight(triangles):
    """True when every edge of every triangle is matched by the same edge running the other way."""
    edges = {}
    for t in triangles:
        for k in range(3):
            edges[t[k], t[(k + 1) % 3]] = edges.get((t[k], t[(k + 1) % 3]), 0) + 1
    return all(n == 1 and edges.get((b, a)) == 1 for (a, b), n in edges.items())


def write_stl(path, name, triangles):
    """An ASCII STL with a unit normal per facet."""
    lines = [f"solid {name}"]
    for a, b, c in triangles:
        u, v = [b[k] - a[k] for k in range(3)], [c[k] - a[k] for k in range(3)]
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        size = sum(x * x for x in n) ** 0.5
        lines += ["facet normal " + " ".join(f"{x / size + 0.0:.6g}" for x in n), " outer loop"]  # + 0.0 turns -0 into 0
        lines += [f"  vertex {x:.6g} {y:.6g} {z:.6g}" for x, y, z in (a, b, c)] + [" endloop", "endfacet"]
    Path(path).write_text("\n".join(lines + [f"endsolid {name}"]) + "\n")


def stack(length, width, layers, layer, gap):
    """Triangles of the ceramic layers and of the interlayers of one stack, and its height."""
    pitch = layer + gap
    ceramic = [t for n in range(layers) for t in box(0, 0, n * pitch, length, width, n * pitch + layer)]
    between = [t for n in range(layers - 1) for t in box(0, 0, n * pitch + layer, length, width, (n + 1) * pitch)]
    return ceramic, between, layers * layer + (layers - 1) * gap


def layered_geometry(ceramic, interlayer, layer_thickness_um, interlayer_thickness_um, folder,
                     bar_mm=(45.0, 4.0, 3.0), cell_layers=5):
    """Write the print bar and the simulation cell of a layered design into folder, and return their paths.

    The bar is a bend bar of about bar_mm [ASTM C1161], lowered to a whole number of layers. The cell holds
    cell_layers ceramic layers around a notch that enters the bottom layer from below. The bar
    size, the cell size and the notch are defaults of this step. The design does not set them.
    """
    layer, gap = layer_thickness_um / 1000, interlayer_thickness_um / 1000
    design = {"ceramic": ceramic, "interlayer": interlayer, "layer_thickness_um": layer_thickness_um,
              "interlayer_thickness_um": interlayer_thickness_um, "bar_mm": list(bar_mm), "cell_layers": cell_layers}
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)

    layers = max(1, int((bar_mm[2] + gap) // (layer + gap)))
    parts = {"bar": stack(bar_mm[0], bar_mm[1], layers, layer, gap)}
    cell_height = cell_layers * layer + (cell_layers - 1) * gap
    parts["cell"] = stack(4 * cell_height, layer, cell_layers, layer, gap)
    spec = {"units": "mm", "stack_axis": "z", "design": design, "defaults": ["bar_mm", "cell_layers", "cell size", "notch"]}
    for name, (solid, between, height) in parts.items():
        files = {}
        for material, triangles in [("ceramic", solid), ("interlayer", between)]:
            assert watertight(triangles), f"{name} {material} mesh is not closed"
            files[material] = str(out / f"{name}_{material}.stl")
            write_stl(files[material], f"{name}_{material}", triangles)
        size = [bar_mm[0], bar_mm[1], height] if name == "bar" else [4 * cell_height, layer, height]
        spec[name] = {"size_mm": [round(x, 6) for x in size], "ceramic_layers": layers if name == "bar" else cell_layers,
                      "volume_mm3": {"ceramic": round(volume(solid), 9), "interlayer": round(volume(between), 9)},
                      "triangles": len(solid) + len(between), "files": files}
    # The notch is described and not cut, since a phase-field model sets it as an initial crack.
    spec["cell"]["notch"] = {"plane": "x", "at_mm": round(2 * cell_height, 6), "from_z_mm": 0, "depth_mm": round(layer / 2, 6)}
    (out / "geometry.json").write_text(json.dumps(spec, indent=1))
    return {"geometry_json": str(out / "geometry.json"), "bar_ceramic_stl": spec["bar"]["files"]["ceramic"],
            "bar_interlayer_stl": spec["bar"]["files"]["interlayer"], "cell_ceramic_stl": spec["cell"]["files"]["ceramic"],
            "cell_interlayer_stl": spec["cell"]["files"]["interlayer"], "bar_ceramic_layers": layers,
            "bar_height_mm": round(parts["bar"][2], 6), "smallest_feature_um": min(layer_thickness_um, interlayer_thickness_um),
            "method": "generated from the design. Bar size, cell size and notch are defaults of the geometry step"}
