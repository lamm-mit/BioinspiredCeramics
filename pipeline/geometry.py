"""The part a design and its toolpath pattern describe, as a JSON description, a section and STL files.

The part is a bend bar of about bar_mm [ASTM C1161], lowered to a whole number of plies, whose toolpaths
are in the print files. The simulation cell is its bottom cell_layers plies at mid-length, with a notch
entering the bottom ply from below. cell_section.npz samples pattern.material() on that cell's x-z section
at 2 µm pixels. The STL files hold the outline of the bar and, per material, a voxel block of the cell.
Lengths are in mm.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from .pattern import complete, material, plies, ply_angle, road_width

SECTION_UM = 2.0
# Voxels across the cell height in the STL block.
VOXEL_ROWS = 24


def box(x0, y0, z0, x1, y1, z1):
    """The 12 triangles of an axis-aligned box, each counter-clockwise seen from outside."""
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
    return np.array([(v[a], v[b], v[c]) for a, b, c, d in quads] + [(v[a], v[c], v[d]) for a, b, c, d in quads])


def volume(triangles):
    """Enclosed volume, from the signed tetrahedra between each triangle and the origin."""
    return float(np.linalg.det(np.asarray(triangles, float)).sum() / 6)


def watertight(triangles):
    """True when every directed edge is matched by as many edges running the other way, so the surface has no boundary."""
    t = np.asarray(triangles, np.float64)
    edges = np.concatenate([t[:, [0, 1, 2]], t[:, [1, 2, 0]]], axis=2).reshape(-1, 6)
    forward, back = (np.unique(e, axis=0, return_counts=True) for e in (edges, edges[:, [3, 4, 5, 0, 1, 2]]))
    return len(forward[0]) == len(back[0]) and (forward[0] == back[0]).all() and (forward[1] == back[1]).all()


STL = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def write_stl(path, name, triangles):
    """A binary STL with a unit normal per facet."""
    t = np.asarray(triangles, np.float32).reshape(-1, 3, 3)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    rows = np.zeros(len(t), STL)
    rows["normal"], rows["vertices"] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30), t
    Path(path).write_bytes(f"binary STL {name}".encode()[:80].ljust(80, b" ") + np.uint32(len(t)).tobytes() + rows.tobytes())


def read_stl(path):
    """The triangles of a binary STL as an (m, 3, 3) array."""
    data = Path(path).read_bytes()
    return np.frombuffer(data, STL, int(np.frombuffer(data, "<u4", 1, 80)[0]), 84)["vertices"]


def voxel_surface(mask, origin, size):
    """Triangles of the boundary of the voxels set in mask, each face counter-clockwise seen from outside."""
    m = np.pad(mask, 1).astype(np.int8)
    out = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3  # e_b x e_c = e_a
        step = np.diff(m, axis=a)
        for sign in (-1, 1):  # -1: the voxel below along a is set, so the face looks along +a
            p = np.argwhere(step == sign).astype(float)
            p[:, b] -= 1
            p[:, c] -= 1
            eb, ec = np.eye(3)[b], np.eye(3)[c]
            quad = np.stack([p, p + eb, p + eb + ec, p + ec], axis=1)
            if sign == 1:
                quad = quad[:, ::-1]
            out += [quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]]
    return np.asarray(origin) + np.concatenate(out) * size


def section_y(p, design, n, height, x, width):
    """y of the section plane: the middle of the ceramic road of the bottom ply nearest the mid-width.

    A plane along a road edge would lie in an interlayer shell over its whole length.
    """
    w, h = road_width(design), design["layer_thickness_um"] / 1000
    y = np.linspace(width / 2 - w, width / 2 + w, 401)
    ceramic = material(p, design, n, height, np.full_like(y, x), y, np.full_like(y, h / 2))[0] == 0
    if ceramic.all() or not ceramic.any():
        return width / 2
    # Distance of each sample to the nearest non-ceramic sample, largest at the middle of a road.
    gaps = np.flatnonzero(~ceramic)
    depth = np.min(np.abs(np.arange(len(y))[:, None] - gaps[None, :]), axis=1) * ceramic
    best = np.flatnonzero(depth == depth.max())
    return float(y[best[np.argmin(np.abs(y[best] - width / 2))]])


def layered_geometry(design, pattern, folder, bar_mm=(45.0, 4.0, 3.0), cell_layers=5):
    """Write the geometry of a design with its toolpath pattern into folder, and return the paths and a summary."""
    p = complete(pattern)
    h, s = design["layer_thickness_um"] / 1000, design["interlayer_thickness_um"] / 1000
    pitch, w = h + s, road_width(design)
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)

    n = max(1, int((bar_mm[2] + s) // pitch))
    height = n * pitch - s
    cell_height = cell_layers * pitch - s
    # The cell holds two periods of the pattern, so the crack meets its features, within 4 to 8 cell heights.
    periods = [p["break_length_h"] * h, p["ring_spacing_w"] * w * (p["road_path"] == "rings"),
               p["wave_length_w"] * w * (p["road_path"] == "waves"), p["ply_wave_length_h"] * h * bool(p["ply_wave_amplitude_h"])]
    if p["interlayer_placement"] == "shell":
        periods += [w / abs(math.sin(math.radians(ply_angle(p, k)))) for k in range(cell_layers)
                    if abs(math.sin(math.radians(ply_angle(p, k)))) > 0.05]
    cell_width = min(8 * cell_height, max(4 * cell_height, 2 * max(periods)))
    x0 = bar_mm[0] / 2 - cell_width / 2
    y0 = section_y(p, design, n, height, bar_mm[0] / 2, bar_mm[1])

    # The section, pixel centers from the bottom left of the cell.
    px = SECTION_UM / 1000
    xs = x0 + (np.arange(round(cell_width / px)) + 0.5) * px
    zs = (np.arange(round(cell_height / px)) + 0.5) * px
    X, Z = np.meshgrid(xs, zs)
    phase, ply, modulus = material(p, design, n, height, X, np.full_like(X, y0), Z)
    section = out / "cell_section.npz"
    np.savez_compressed(section, phase=phase.astype(np.int8), ply=ply.astype(np.int16),
                        modulus_GPa=modulus.astype(np.float32), pixel_um=SECTION_UM)
    # The notch sits where its crack meets the plies and the interlayer between them, and not a joint of the pattern.
    # That is the column of the middle half of the cell farthest from a weak pixel of ply 0 at the notch tip and of
    # ply 1 at its middle, the one nearest the middle among equals. A flat laminate keeps the notch at the middle.
    weak = phase == 2
    base = (ply >= 1).argmax(axis=0)
    joint = weak[int(h / 2 / px)] | weak[np.minimum(base + int(h / 2 / px), len(zs) - 1), np.arange(len(xs))]
    gap = ndimage.distance_transform_edt(~joint) if joint.any() else np.full(len(xs), np.inf)
    x = xs - x0
    inner = np.abs(x - cell_width / 2) <= cell_width / 4
    far = inner & (gap >= gap[inner].max() - 1)
    notch_mm = x[far][np.argmin(np.abs(x[far] - cell_width / 2))]

    # A voxel block of the cell, centered on the section plane, one STL per material.
    voxel = cell_height / VOXEL_ROWS
    depth = min(bar_mm[1], 2 * w)
    shape = [max(1, round(cell_width / voxel)), max(1, round(depth / voxel)), VOXEL_ROWS]
    origin = (x0, y0 - shape[1] * voxel / 2, 0.0)
    X, Y, Z = np.meshgrid(*[origin[a] + (np.arange(shape[a]) + 0.5) * voxel for a in range(3)], indexing="ij")
    block = material(p, design, n, height, X, Y, Z)[0]
    files = {}
    for name, code in [("ceramic", 0), ("interlayer", 2)]:
        triangles = voxel_surface(block == code, origin, voxel)
        assert watertight(triangles), f"cell {name} surface is not closed"
        files[name] = str(out / f"cell_{name}.stl")
        write_stl(files[name], f"cell_{name}", triangles)
    outline = box(0, 0, 0, bar_mm[0], bar_mm[1], height)
    write_stl(out / "bar_outline.stl", "bar_outline", outline)

    spec = {"units": "mm", "stack_axis": "z",
            "design": {k: design[k] for k in ["ceramic", "interlayer", "layer_thickness_um", "interlayer_thickness_um",
                                               "layer_modulus_GPa"]} | {"bar_mm": list(bar_mm), "cell_layers": cell_layers},
            "pattern": p, "road_width_mm": round(w, 6), "plies": plies(p, design, n),
            "bar": {"size_mm": [bar_mm[0], bar_mm[1], round(height, 6)], "ceramic_layers": n,
                    "files": {"outline": str(out / "bar_outline.stl")}},
            "cell": {"origin_mm": [round(x0, 6), round(y0, 6), 0.0], "size_mm": [round(cell_width, 6), round(depth, 6), round(cell_height, 6)],
                     "ceramic_layers": cell_layers, "section": str(section), "section_pixel_um": SECTION_UM,
                     "section_plane": f"x-z at y = {y0:.4f} mm", "voxel_mm": round(voxel, 6),
                     "ceramic_fraction": round(float((phase == 0).mean()), 4), "files": files,
                     # The notch is described and not cut, since a phase-field model sets it as an initial crack.
                     "notch": {"plane": "x", "at_mm": round(float(notch_mm), 6), "from_z_mm": 0, "depth_mm": round(h / 2, 6)}},
            "defaults": ["bar_mm", "cell_layers", "road width 1.2 ply pitches [Zhou 2023]", "cell placement at mid-length",
                         "section plane through the middle of a bottom-ply road", "notch"]}
    (out / "geometry.json").write_text(json.dumps(spec, indent=1))
    return {"geometry_json": str(out / "geometry.json"), "bar_outline_stl": spec["bar"]["files"]["outline"],
            "cell_ceramic_stl": files["ceramic"], "cell_interlayer_stl": files["interlayer"], "cell_section_npz": str(section),
            "bar_ceramic_layers": n, "bar_height_mm": round(height, 6), "road_width_um": round(w * 1000, 3),
            "cell_size_mm": spec["cell"]["size_mm"], "smallest_feature_um": min(design["layer_thickness_um"], design["interlayer_thickness_um"]),
            "method": "generated from the design and its toolpath pattern. Bar size, cell placement, road width and notch are defaults of the geometry step"}
