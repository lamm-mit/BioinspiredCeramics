"""The notched cell of a design as a phase-field model, and the crack it grows.

The model is a 2D plane-strain section of the cell in the x-z plane, on square pixels. It runs in
scaled units: pixel size 1, Young modulus 1 and layer fracture energy 1. AT2 is energy-controlled
at this scale, so the solution in those units depends only on the energy ratio, the shape of the
cell and the pixel count. The tool scales it back to physical units.
"""

import os
import time

import jax
import numpy as np
from scipy import ndimage

from . import jax_pfczm as pf

# Pixel rows across the cell height. More rows resolve the interlayers better and run longer.
ROWS = 60
# An interlayer becomes a band of weak pixels at least this many pixels thick.
BAND_PIXELS = 3
# AT2 length l in pixels. h / l = 0.5 is the resolution jax-pfczm asks for.
LENGTH_PIXELS = 2.0
POISSON = 0.25
# A node with damage at or above this value is broken.
BROKEN = 0.9
# Load steps of the applied stress intensity, in units of sqrt(E G) of the layer. The run stops
# once the crack has deflected or penetrated.
K_STEP, K_MAX = 0.1, 6.0
_solvers = {}


def rows():
    """FEM_ROWS overrides ROWS, for example to run tests on a coarse grid."""
    return int(os.environ.get("FEM_ROWS", ROWS))


def section(cell, n_rows):
    """Pixels of the x-z section of the cell: grid, phase per pixel and the indices the analysis needs.

    cell holds the cell of geometry_json and the layer and interlayer thicknesses. Phase 0 is ceramic,
    2 an interlayer band, 3 the notch, a column of void pixels one pixel wide. One pixel column is
    added to the width so that the notch sits on the middle of the cell.
    """
    width_mm, _, height_mm = cell["size_mm"]
    layer_mm = cell["layer_thickness_um"] / 1000
    gap_mm = cell["interlayer_thickness_um"] / 1000
    h = height_mm / n_rows
    nx = int(round(width_mm / h)) + 1
    grid = pf.Grid(nx, n_rows, 1.0)
    phase = np.zeros((n_rows, nx), int)
    band = max(BAND_PIXELS, int(round(gap_mm / h)))
    bands = []
    for k in range(1, cell["ceramic_layers"]):
        middle = (k * layer_mm + (k - 1) * gap_mm + gap_mm / 2) / h
        first = int(round(middle - band / 2))
        phase[first:first + band] = 2
        bands.append((first, first + band))
    notch = cell["notch"]
    column = int(round(notch["at_mm"] / h))
    depth = max(1, int(round(notch["depth_mm"] / h)))
    if len(bands) < 2 or depth >= bands[0][0] or bands[1][0] - bands[0][1] < 2:
        raise ValueError(f"{n_rows} pixel rows do not resolve the layers, the interlayers and the notch of this cell")
    phase[:depth, column] = 3
    return {"grid": grid, "phase": phase.ravel(), "bands": bands, "notch_column": column, "notch_rows": depth,
            "pixel_mm": h, "band_pixels": band, "rows": n_rows}


def material(sec, energy_ratio):
    """All pixels share one modulus. Band pixels have the interface fracture energy, the notch is void."""
    phase = sec["phase"]
    void = phase == 3
    return pf.Material(E=np.where(void, 1e-9, 1.0), nu=np.full(phase.size, POISSON),
                       Gc=np.where(phase == 2, energy_ratio, 1.0), ft=np.ones(phase.size), damageable=~void, phase=phase)


def kfield_bc(sec):
    """The cell boundary follows the mode-I crack-tip field of the notch, with stress intensity K = t.

    The field is centred on the notch tip, with the crack along -z. Every edge node is held at it.
    """
    grid = sec["grid"]
    x, z = grid.nodes[:, 0], grid.nodes[:, 1]
    edge = (x == 0) | (z == 0) | np.isclose(x, grid.width) | np.isclose(z, grid.height)
    fixed = np.zeros((grid.nnode, 2), bool)
    fixed[edge] = True
    # Local frame of the crack: x1 = z - z_tip ahead of the tip, x2 = x_tip - x.
    ux, uy = pf.k_field(z[edge] - sec["notch_rows"], sec["notch_column"] + 0.5 - x[edge], 1.0, 1.0, POISSON)

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
    """Where the crack is: its top in pixels, how far it runs along the first band on each side,
    and whether any part of that band is broken, joined to the crack or not."""
    broken = d.reshape(sec["grid"].ny + 1, sec["grid"].nx + 1) >= BROKEN
    c = crack(sec, broken)
    first, last = sec["bands"][0]
    rows_hit = np.nonzero(c.any(axis=1))[0]
    middle = sec["notch_column"] + 0.5
    cols = np.nonzero(c[first:last + 1].any(axis=0))[0]
    return {"top": float(rows_hit.max()) if rows_hit.size else float(sec["notch_rows"]),
            "left": float(max(0.0, middle - cols.min())) if cols.size else 0.0,
            "right": float(max(0.0, cols.max() - middle)) if cols.size else 0.0, "in_band": bool(cols.size),
            "band_broken": bool(broken[first:last + 1].any())}


def band_energy_ratio(energy_ratio, band_pixels):
    """Fracture energy of a straight AT2 crack along the middle of a band, over that of the layer.

    The AT2 damage profile spreads over a few l, so part of it lies in the tougher layers on both
    sides. For a band of half-thickness s and r = energy_ratio the 1D profile gives
    r (cosh x + r sinh x) / (r cosh x + sinh x) with x = s / l, which tends to r coth x for a weak band.
    """
    r, x = energy_ratio, band_pixels / 2 / LENGTH_PIXELS
    return float(r * (np.cosh(x) + r * np.sinh(x)) / (r * np.cosh(x) + np.sinh(x)))


def simulate(cell, energy_ratio, n_rows=None):
    """Raise K step by step and follow the crack until it deflects or penetrates.

    The crack deflects when it runs along the first band for more than one layer thickness on a
    side before it reaches the middle of the next ceramic layer. It penetrates when it reaches that
    middle first, or in the same load step. The solver runs on the CPU, which leaves the GPU to the
    served models.
    """
    sec = section(cell, n_rows or rows())
    grid = sec["grid"]
    layer_px = cell["layer_thickness_um"] / 1000 / sec["pixel_mm"]
    next_middle = (sec["bands"][0][1] + sec["bands"][1][0]) / 2
    ramp = Ramp(np.arange(1, round(K_MAX / K_STEP) + 1) * K_STEP)
    track = []

    def follow(u, d, f):
        m = measure(sec, d)
        track.append(m)
        ramp.stop = max(m["left"], m["right"]) > layer_px or m["top"] > next_middle
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
        res = _solvers[key].run(material(sec, energy_ratio), kfield_bc(sec), ramp, tol_E=1e-6, tol_d=1e-3,
                                max_stagger=200, observers={"top": follow})
    deflected = next((k for k, m in enumerate(track) if max(m["left"], m["right"]) > layer_px), None)
    penetrated = next((k for k, m in enumerate(track) if m["top"] > next_middle), None)
    if deflected is None and penetrated is None:
        raise RuntimeError(f"the crack neither deflected nor penetrated up to K = {K_MAX} sqrt(E G) of the layer")
    deflects = penetrated is None or (deflected is not None and deflected < penetrated)
    return {"section": sec, "result": res, "track": track, "deflects": deflects,
            "reached": next((k for k, m in enumerate(track) if m["in_band"]), None),
            "band_broke": next((k for k, m in enumerate(track) if m["band_broken"]), None),
            "deflected": deflected, "penetrated": penetrated, "layer_px": layer_px, "next_middle_px": next_middle,
            "band_energy_ratio": band_energy_ratio(energy_ratio, sec["band_pixels"]),
            "device": str(device), "wall_s": time.perf_counter() - start}


def draw(sim, path, title):
    """A PNG of the damage at the last load step over the interlayer bands, with lengths in mm."""
    from matplotlib.figure import Figure

    sec = sim["section"]
    grid, h = sec["grid"], sec["pixel_mm"]
    extent = (0, grid.nx * h, 0, grid.ny * h)
    fig = Figure(figsize=(8, 0.8 + 8 * grid.ny / grid.nx))
    ax = fig.subplots()
    phase = sec["phase"].reshape(grid.ny, grid.nx)
    ax.imshow(np.where(phase == 2, 0.3, np.where(phase == 3, 1.0, 0.0)), origin="lower", extent=extent, cmap="Greys",
              vmin=0, vmax=1, interpolation="nearest")
    d = sim["result"]["d"].reshape(grid.ny + 1, grid.nx + 1)
    image = ax.imshow(np.ma.masked_less(d, 0.05), origin="lower", extent=extent, cmap="inferno_r", vmin=0, vmax=1,
                      alpha=0.9, interpolation="bilinear")
    fig.colorbar(image, ax=ax, label="damage", fraction=0.03, pad=0.01)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
