"""The notched cell of a design as a phase-field model, and the crack it grows.

The model is a 2D plane-strain section of the cell in the x-z plane, on square pixels sampled from the
section file of the geometry step. It runs in scaled units, with pixel size, the design's Young modulus and
the layer fracture energy equal to 1. AT2 is energy-controlled at this scale, so the solution in those units
depends only on the energy ratio, the pattern and the modulus ratios of the section, and the pixel count.
The tool scales it back to physical units.
"""

import os
import time

import jax
import numpy as np
from scipy import ndimage

from . import jax_pfczm as pf

# Pixel rows across the cell height. More rows resolve the interlayers better and run longer.
ROWS = 60
# An interlayer or a joint becomes a feature of weak pixels at least this many pixels thick.
BAND_PIXELS = 3
# AT2 length l in pixels. h / l = 0.5 is the resolution jax-pfczm asks for.
LENGTH_PIXELS = 2.0
POISSON = 0.25
# A node with damage at or above this value is broken.
BROKEN = 0.9
# Load steps of the applied stress intensity, in units of sqrt(E G) of the layer. The run stops
# once the crack has deflected, run through a joint or penetrated.
K_STEP, K_MAX = 0.1, 6.0
_solvers = {}


def rows():
    """FEM_ROWS overrides ROWS, for example to run tests on a coarse grid."""
    return int(os.environ.get("FEM_ROWS", ROWS))


def touch(mask):
    """Nodes with at least one of their pixels in mask, a (rows, columns) bool array of pixels."""
    p = np.pad(mask, 1)
    return p[:-1, :-1] | p[1:, :-1] | p[:-1, 1:] | p[1:, 1:]


def section(cell, n_rows):
    """Pixels of the x-z section of the cell, with phase, ply and modulus per pixel and what the analysis needs.

    Each pixel takes the phase, ply and modulus of the section file at its center. Phase 0 is ceramic,
    2 weak (interlayer or joint), 3 the notch, a column of void pixels one pixel wide at the column nearest the notch of the geometry
    step. One pixel column is added to the width so that a notch at the middle sits on a column. A weak feature of the file thinner
    than BAND_PIXELS pixels is first grown by the missing half-thickness on each side. above is the height
    of each node over the local base of ply 1, in pixels.
    """
    width_mm, _, height_mm = cell["size_mm"]
    layer_mm, gap_mm = cell["layer_thickness_um"] / 1000, cell["interlayer_thickness_um"] / 1000
    h = height_mm / n_rows
    nx = int(round(width_mm / h)) + 1
    fine = np.load(cell["section"])
    weak_fine, px = fine["phase"] == 2, float(fine["pixel_um"]) / 1000
    # Pixel centers in mm. Columns sit half a pixel to the left so that the notch column is centered on the notch.
    X, Z = np.meshgrid(np.arange(nx) * h, (np.arange(n_rows) + 0.5) * h)
    iz = np.minimum((Z / px).astype(int), weak_fine.shape[0] - 1)
    ix = np.minimum((X / px).astype(int), weak_fine.shape[1] - 1)
    # Weak features of the file thinner than BAND_PIXELS pixels grow by the missing half-thickness of the
    # interlayer, or of the feature itself where the file draws it thinner, as where the ply wave fades. The
    # file draws an interlayer at least floor(s / px) of its pixels thick. half is the half-thickness of the
    # feature nearest each pixel center, the largest depth of weak pixels within reach. It overestimates near
    # a junction of two features, where the interlayer's value takes over.
    reach = int(BAND_PIXELS * h / px) + 1
    half = ndimage.maximum_filter(ndimage.distance_transform_edt(weak_fine), size=reach) * px - px / 2
    _, (wz, wx) = ndimage.distance_transform_edt(~weak_fine, return_indices=True)
    wz, wx = wz[iz, ix], wx[iz, ix]
    grow = np.maximum(BAND_PIXELS * h - px * np.floor(gap_mm / px + 1e-9), BAND_PIXELS * h - 2 * half[wz, wx]) / 2
    # Distance from each pixel center to the square of the nearest weak pixel of the file.
    dz = np.maximum(np.abs(Z - (wz + 0.5) * px) - px / 2, 0)
    dx = np.maximum(np.abs(X - (wx + 0.5) * px) - px / 2, 0)
    weak = weak_fine[iz, ix] | (np.hypot(dz, dx) <= grow)
    if layer_mm - max(0, BAND_PIXELS * h - gap_mm) < 2 * h:
        raise ValueError(f"{n_rows} pixel rows do not resolve the layers, the interlayers and the notch of this cell")
    ply = fine["ply"][iz, ix]
    phase = np.where(weak, 2, 0)
    column = int(round(cell["notch"]["at_mm"] / h))
    depth = max(1, int(round(cell["notch"]["depth_mm"] / h)))
    phase[:depth, column] = 3
    layer_px = layer_mm / h
    base = (ply >= 1).argmax(axis=0)
    # Columns whose pixel at the middle of ply 1's ceramic is weak, a joint of the pattern.
    joints = np.flatnonzero(weak[(base + layer_px / 2).astype(int), np.arange(nx)])
    return {"grid": pf.Grid(nx, n_rows, 1.0), "phase": phase.ravel(), "ply": ply.ravel(),
            "modulus_GPa": fine["modulus_GPa"][iz, ix].ravel(), "notch_column": column, "notch_rows": depth,
            "pixel_mm": h, "band_pixels": max(BAND_PIXELS, gap_mm / h), "rows": n_rows, "layer_px": layer_px,
            "above": np.arange(n_rows + 1)[:, None] - np.append(base, base[-1]), "weak_nodes": touch(weak),
            "first_nodes": touch(weak & (ply == 0)),
            "nearest_joint_px": int(np.abs(joints - column).min()) if joints.size else None}


def material(sec, energy_ratio, layer_modulus_GPa):
    """Each pixel has the modulus of its ply over the layer modulus. Weak pixels have the interface fracture
    energy, the others the layer fracture energy, and the notch is void."""
    phase = sec["phase"]
    void = phase == 3
    return pf.Material(E=np.where(void, 1e-9, sec["modulus_GPa"] / layer_modulus_GPa),
                       nu=np.full(phase.size, POISSON), Gc=np.where(phase == 2, energy_ratio, 1.0),
                       ft=np.ones(phase.size), damageable=~void, phase=phase)


def kfield_bc(sec, E):
    """The cell boundary follows the mode-I crack-tip field of the notch, with stress intensity K = t.

    The field is centred on the notch tip, with the crack along -z, in a material of modulus E. Every edge
    node is held at it.
    """
    grid = sec["grid"]
    x, z = grid.nodes[:, 0], grid.nodes[:, 1]
    edge = (x == 0) | (z == 0) | np.isclose(x, grid.width) | np.isclose(z, grid.height)
    fixed = np.zeros((grid.nnode, 2), bool)
    fixed[edge] = True
    # Local frame of the crack: x1 = z - z_tip ahead of the tip, x2 = x_tip - x.
    ux, uy = pf.k_field(z[edge] - sec["notch_rows"], sec["notch_column"] + 0.5 - x[edge], 1.0, E, POISSON)

    def bc(t):
        u = np.zeros((grid.nnode, 2))
        u[edge, 0], u[edge, 1] = -t * uy, t * ux
        return u, fixed

    return bc


class Ramp:
    """Load steps for PhaseFieldSolver.run that end early once stop is set."""

    def __init__(self, values):
        self.values, self.stop = values, False

    def __len__(self):
        return len(self.values)

    def __getitem__(self, k):
        return self.values[k]

    def __iter__(self):
        for t in self.values:
            if self.stop:
                return
            yield t


def crack(sec, broken):
    """The broken nodes connected to the notch. broken is a (rows + 1, nx + 1) bool array of nodes."""
    labels, _ = ndimage.label(broken, structure=np.ones((3, 3)))
    i, n = sec["notch_column"], sec["notch_rows"]
    seeds = set(labels[: n + 1, i: i + 2].ravel()) - {0}
    return np.isin(labels, list(seeds)) if seeds else np.zeros_like(broken)


def measure(sec, d):
    """Where the crack is, in pixels. top is its highest node, left and right how far it runs through weak
    pixels on each side of the notch, and above how far above the local base of ply 1 it reaches. mid is true once it passes the middle of ply 1's
    ceramic, and joint once a path of weak nodes within the crack leads there from the weak pixels of ply 0.
    band_broken is true when any weak pixel of ply 0 is broken, joined to the crack or not."""
    broken = d.reshape(sec["grid"].ny + 1, sec["grid"].nx + 1) >= BROKEN
    c = crack(sec, broken)
    weak = c & sec["weak_nodes"]
    side = np.nonzero(weak)[1] - (sec["notch_column"] + 0.5)
    above = np.where(c, sec["above"], -1)
    upper = (above > sec["layer_px"] / 2) & (above < sec["layer_px"])
    paths, _ = ndimage.label(weak, structure=np.ones((3, 3)))
    rows_hit = np.nonzero(c.any(axis=1))[0]
    return {"top": float(rows_hit.max()) if rows_hit.size else float(sec["notch_rows"]),
            "left": float(max(0.0, -side.min())) if side.size else 0.0,
            "right": float(max(0.0, side.max())) if side.size else 0.0,
            "above": float(max(0, above.max())), "in_band": bool(weak.any()), "mid": bool(upper.any()),
            "joint": bool(np.intersect1d(paths[upper], paths[weak & sec["first_nodes"]]).any()),
            "band_broken": bool((broken & sec["first_nodes"]).any())}


def band_energy_ratio(energy_ratio, band_pixels):
    """Fracture energy of a straight AT2 crack along the middle of a band, over that of the layer.

    The AT2 damage profile spreads over a few l, so part of it lies in the tougher layers on both
    sides. For a band of half-thickness s and r = energy_ratio the 1D profile gives
    r (cosh x + r sinh x) / (r cosh x + sinh x) with x = s / l, which tends to r coth x for a weak band.
    """
    r, x = energy_ratio, band_pixels / 2 / LENGTH_PIXELS
    return float(r * (np.cosh(x) + r * np.sinh(x)) / (r * np.cosh(x) + np.sinh(x)))


def simulate(cell, energy_ratio, layer_modulus_GPa, n_rows=None):
    """Raise K step by step and follow the crack until it deflects, runs through a joint or penetrates.

    The crack deflects when it runs through weak pixels more than one layer thickness sideways from the
    notch before it reaches the middle of ply 1, measured from the local base of ply 1. When it reaches
    that middle first, or in the same load step, it ran through a joint if it got there from the weak
    pixels of ply 0 through weak pixels only, and penetrated if it broke the ceramic of ply 1. The K-field
    takes the modulus of the pixel ahead of the notch, so that K is that of the notched ply. The solver runs
    on the CPU, which leaves the GPU to the served models.
    """
    sec = section(cell, n_rows or rows())
    grid = sec["grid"]
    mat = material(sec, energy_ratio, layer_modulus_GPa)
    ramp = Ramp(np.arange(1, round(K_MAX / K_STEP) + 1) * K_STEP)
    track = []

    def follow(u, d, f):
        m = measure(sec, d)
        track.append(m)
        ramp.stop = m["mid"] or max(m["left"], m["right"]) > sec["layer_px"]
        return m["top"]

    start = time.perf_counter()
    try:
        device = jax.devices("cpu")[0]
    except RuntimeError:  # JAX was started without its CPU backend
        device = jax.devices()[0]
    with jax.default_device(device):
        key = (grid.nx, grid.ny)
        if key not in _solvers:
            _solvers[key] = pf.PhaseFieldSolver(grid, pf.PhaseFieldModel("at2", length=LENGTH_PIXELS), cg_tol=1e-8)
        tip = mat.E[sec["notch_rows"] * grid.nx + sec["notch_column"]]
        res = _solvers[key].run(mat, kfield_bc(sec, tip), ramp, tol_E=1e-6, tol_d=1e-3, max_stagger=200,
                                observers={"top": follow})
    if not ramp.stop:
        raise RuntimeError(f"the crack neither deflected nor reached the next ply up to K = {K_MAX} sqrt(E G) of the layer")
    last = track[-1]
    return {"section": sec, "result": res, "track": track,
            "outcome": ("through joint" if last["joint"] else "penetrated") if last["mid"] else "deflected",
            "reached": next((k for k, m in enumerate(track) if m["in_band"]), None),
            "band_broke": next((k for k, m in enumerate(track) if m["band_broken"]), None),
            "band_energy_ratio": band_energy_ratio(energy_ratio, sec["band_pixels"]),
            "device": str(device), "wall_s": time.perf_counter() - start}


def draw(sim, path, title):
    """A PNG of the damage at the last load step over the section, with lengths in mm.

    Weak pixels are grey and the notch black. Ceramic is white, shaded darker for stiffer plies when the
    modulus varies.
    """
    from matplotlib.figure import Figure

    sec = sim["section"]
    grid, h = sec["grid"], sec["pixel_mm"]
    extent = (-h / 2, (grid.nx - 0.5) * h, 0, grid.ny * h)  # x of the cell, as section() samples it
    fig = Figure(figsize=(8, 0.8 + 8 * grid.ny / grid.nx))
    ax = fig.subplots()
    phase = sec["phase"].reshape(grid.ny, grid.nx)
    E = sec["modulus_GPa"].reshape(grid.ny, grid.nx)
    lo, hi = E.min(), E.max()
    ceramic = 0.12 * (E - lo) / (hi - lo) if hi > lo else np.zeros_like(E)
    if hi > lo:
        title += f"\nDarker ceramic is stiffer, from {lo:.0f} to {hi:.0f} GPa"
    ax.imshow(np.where(phase == 2, 0.35, np.where(phase == 3, 1.0, ceramic)), origin="lower", extent=extent,
              cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    d = sim["result"]["d"].reshape(grid.ny + 1, grid.nx + 1)
    image = ax.imshow(np.ma.masked_less(d, 0.05), origin="lower", extent=extent, cmap="inferno_r", vmin=0, vmax=1,
                      alpha=0.9, interpolation="bilinear")
    fig.colorbar(image, ax=ax, label="damage", fraction=0.03, pad=0.01)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
