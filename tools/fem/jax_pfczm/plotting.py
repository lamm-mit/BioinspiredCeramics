"""Plotting helpers (optional dependency: matplotlib; install with the ``plot`` extra)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .grid import Grid
from .models import Material


def plot_damage(grid: Grid, d: np.ndarray, mat: Optional[Material], path: str, title: str = "") -> None:
    """Save a PNG of the nodal damage ``d`` (nnode,) over the phase map (particles grey, interface light grey).

    Damage below 0.05 is transparent.  ``mat`` may be None (no phase background).  Draws on a standalone
    ``matplotlib.figure.Figure``: needs no display and leaves the caller's matplotlib backend unchanged.
    """
    from matplotlib.figure import Figure

    D = d.reshape(grid.ny + 1, grid.nx + 1)
    fig = Figure(figsize=(6, 6 * grid.ny / grid.nx + 0.6))
    ax = fig.subplots()
    ext = (grid.x0, grid.x0 + grid.width, grid.y0, grid.y0 + grid.height)
    if mat is not None and mat.phase is not None:
        ph = mat.phase.reshape(grid.ny, grid.nx)
        ax.imshow(np.where(ph == 1, 1.0, np.where(ph == 2, 0.5, 0.0)), origin="lower", extent=ext, cmap="Greys",
                  vmin=0, vmax=1.6, interpolation="nearest")
    ax.imshow(np.ma.masked_less(D, 0.05), origin="lower", extent=ext, cmap="inferno_r", vmin=0, vmax=1,
              alpha=0.9, interpolation="bilinear")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def save_damage_gif(grid: Grid, mat: Optional[Material], snaps, path: str, title: str = "", fps: int = 6,
                    load_label: str = "U") -> None:
    """Animated GIF of damage snapshots ``[(t, d), ...]`` (``run(..., snapshot_every=k)['snapshots']``).

    Uses matplotlib's PillowWriter (Pillow ships with matplotlib) on a standalone Figure (backend unchanged).
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.figure import Figure

    ext = (grid.x0, grid.x0 + grid.width, grid.y0, grid.y0 + grid.height)
    fig = Figure(figsize=(5, 5 * grid.ny / grid.nx + 0.5))
    ax = fig.subplots()
    if mat is not None and mat.phase is not None:
        ph = mat.phase.reshape(grid.ny, grid.nx)
        ax.imshow(np.where(ph == 1, 1.0, np.where(ph == 2, 0.5, 0.0)), origin="lower", extent=ext, cmap="Greys",
                  vmin=0, vmax=1.6, interpolation="nearest")
    im = ax.imshow(np.zeros((grid.ny + 1, grid.nx + 1)), origin="lower", extent=ext, cmap="inferno_r", vmin=0,
                   vmax=1, alpha=0.9)
    ttl = ax.set_title(title, fontsize=8)

    def upd(k):
        t, dd = snaps[k]
        im.set_data(np.ma.masked_less(dd.reshape(grid.ny + 1, grid.nx + 1), 0.05))
        ttl.set_text(f"{title}  {load_label} = {t:.3g}")
        return (im,)

    anim = FuncAnimation(fig, upd, frames=len(snaps), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
