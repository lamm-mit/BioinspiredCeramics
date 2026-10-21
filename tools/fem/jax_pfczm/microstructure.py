"""Pixel microstructures: random disk packings, rasterisation to a Material, notched (SENT) runs.

Phase labels written by :func:`rasterize_microstructure`: 0 matrix, 1 particle, 2 interface band
(matrix side of each particle), 3 void (notch; very compliant and non-damageable).
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Optional, Sequence

import numpy as np

from .bcs import crack_path, sent_bc
from .grid import Grid
from .models import Material, PhaseFieldModel
from .solver import PhaseFieldSolver


# ---------------------------------------------------------------------------
# Microstructures
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class PhaseProps:
    E: float
    nu: float
    Gc: float = 1.0
    ft: float = 1.0
    damageable: bool = True


def random_disks(width: float, height: float, area_fraction: float, diameters: Sequence[float],
                 fractions: Sequence[float], min_gap: float, seed: int = 0, exclude=None,
                 max_tries: int = 200000) -> list[tuple[float, float, float]]:
    """Random sequential addition of non-overlapping disks (largest first) to a target area fraction.

    ``fractions``: share of the target area fraction for each diameter class.
    ``exclude(x, y, r) -> bool`` rejects placements (e.g. overlapping the notch).
    Periodic images are not used (particles may be cut by the boundary).
    """
    rng = np.random.default_rng(seed)
    A = width * height
    disks: list[tuple[float, float, float]] = []
    order = np.argsort(diameters)[::-1]
    for k in order:
        r = diameters[k] / 2.0
        target = area_fraction * fractions[k] * A
        placed = 0.0
        tries = 0
        while placed < target and tries < max_tries:
            tries += 1
            x, y = rng.uniform(0, width), rng.uniform(0, height)
            if exclude is not None and exclude(x, y, r):
                continue
            ok = True
            for (xc, yc, rc) in disks:
                if (x - xc) ** 2 + (y - yc) ** 2 < (r + rc + min_gap) ** 2:
                    ok = False
                    break
            if ok:
                disks.append((x, y, r))
                # area inside the box (approximate by clipping on a small grid)
                placed += _disk_area_in_box(x, y, r, width, height)
    return disks


def _disk_area_in_box(x, y, r, W, H, n=24):
    g = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(g, g)
    inside = X**2 + Y**2 <= 1
    px, py = x + r * X[inside], y + r * Y[inside]
    frac = np.mean((px >= 0) & (px <= W) & (py >= 0) & (py <= H))
    return frac * math.pi * r * r


def rasterize_microstructure(grid: Grid, disks, matrix: PhaseProps, particle: PhaseProps,
                             interface: Optional[PhaseProps] = None, interface_thickness: float = 0.0,
                             void_mask: Optional[np.ndarray] = None) -> Material:
    """Phase labels: 0 matrix, 1 particle, 2 interface band (matrix side), 3 void (notch)."""
    c = grid.centers
    phase = np.zeros(grid.nel, int)
    dist_rel = np.full(grid.nel, np.inf)
    for (x, y, r) in disks:
        dd = np.hypot(c[:, 0] - x, c[:, 1] - y) - r
        dist_rel = np.minimum(dist_rel, dd)
    phase[dist_rel <= 0] = 1
    if interface is not None and interface_thickness > 0:
        phase[(dist_rel > 0) & (dist_rel <= interface_thickness)] = 2
    if void_mask is not None:
        phase[np.asarray(void_mask, bool)] = 3
    props = {0: matrix, 1: particle, 2: interface or matrix,
             3: PhaseProps(E=matrix.E * 1e-9, nu=matrix.nu, Gc=matrix.Gc, ft=matrix.ft, damageable=False)}
    mat = Material(E=np.zeros(grid.nel), nu=np.zeros(grid.nel), Gc=np.zeros(grid.nel), ft=np.zeros(grid.nel),
                   damageable=np.zeros(grid.nel, bool), phase=phase)
    for k, p in props.items():
        s = phase == k
        mat.E[s], mat.nu[s], mat.Gc[s], mat.ft[s], mat.damageable[s] = p.E, p.nu, p.Gc, p.ft, p.damageable
    return mat


def notch_void_mask(grid: Grid, a0: float, y_notch: float) -> np.ndarray:
    """One element row of void elements from x = x0 to x0 + a0 centred on y_notch."""
    c = grid.centers
    return (c[:, 0] < grid.x0 + a0) & (np.abs(c[:, 1] - y_notch) < 0.5 * grid.h + 1e-12)


def run_notched_microstructure(grid: Grid, mat: Material, model: PhaseFieldModel, U_max: float, n_steps: int,
                               *, a0: float, y_notch: float, snapshot_every: int = 0, verbose: bool = False,
                               tol_E: float = 1e-6, tol_d: float = 1e-3, max_stagger: int = 200) -> dict:
    """Clamped SENT of a (micro)structure; effective fracture energy = dissipated / ligament."""
    solver = PhaseFieldSolver(grid, model, cg_tol=1e-8)
    bc, top = sent_bc(grid)
    loads = np.linspace(0, U_max, n_steps + 1)[1:]
    t0 = time.time()
    res = solver.run(mat, bc, loads, snapshot_every=snapshot_every, verbose=verbose, tol_E=tol_E, tol_d=tol_d,
                     max_stagger=max_stagger, observers={"F": lambda u, d, f: f[top, 1].sum()})
    lig = grid.width - a0
    F = res["F"]
    xs, ys = crack_path(grid, res["d"], x_from=grid.x0 + a0, threshold=0.9)
    if xs.size > 1:
        path_len = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
        tort = path_len / max(xs[-1] - xs[0], 1e-300)
        spanned = float((xs[-1] - xs[0] + grid.h) / lig)
    else:
        tort, spanned = float("nan"), 0.0
    # where did the crack run?  phase of elements adjacent to damaged nodes (d > 0.9)
    dn = res["d"] > 0.9
    el_d = dn[grid.conn].mean(axis=1) >= 0.5
    phase = mat.phase if mat.phase is not None else np.zeros(grid.nel, int)
    frac = {int(k): float(np.mean(phase[el_d] == k)) for k in np.unique(phase)} if el_d.any() else {}
    broken = F[-1] < 0.05 * F.max()  # residual load < 5 % of peak (tail energy ~1 % of Gamma)
    return {"loads": loads, "F": F, "F_peak": float(F.max()), "U_at_peak": float(loads[int(np.argmax(F))]),
            "dissipated": res["dissipated"], "Gamma_final": float(res["dissipated"][-1]),
            "G_eff_projected": float(res["dissipated"][-1] / lig), "ligament": lig, "tortuosity": tort,
            "ligament_fraction_cracked": spanned, "damaged_element_phase_fractions": frac,
            "fully_broken": bool(broken), "energy_balance_rel_err": float(
                abs(res["external_work"][-1] - res["total"][-1]) / max(res["external_work"][-1], 1e-300)),
            "all_converged": bool(res["converged"].all()), "n_unconverged": int((~res["converged"]).sum()),
            "wall_s": time.time() - t0, "d": res["d"], "snapshots": res["snapshots"],
            "external_work": res["external_work"], "elastic": res["elastic"]}


def build_notched_disk_specimen(matrix: PhaseProps, particle: Optional[PhaseProps] = None, phi: float = 0.0,
                                interface: Optional[PhaseProps] = None, *, h: float = 1.0, width: float = 64.0,
                                height: float = 48.0, a0_frac: float = 0.2, diameters=(8.0, 4.0),
                                fractions=(0.7, 0.3), min_gap: Optional[float] = None,
                                interface_thickness: Optional[float] = None, seed: int = 1):
    """Edge-notched rectangle with a random disk microstructure (generic SENT specimen).

    Notch: one row of void elements from x = 0 to ``a0 = a0_frac * width`` at mid-height.  Disks
    (area fraction ``phi`` of the box, diameter classes ``diameters`` with area shares ``fractions``)
    are placed by :func:`random_disks` keeping ``min_gap`` (default 2h) from each other and from the notch.
    ``interface`` (optional) is a band of ``interface_thickness`` (default h) around each particle.
    Returns ``(grid, mat, info)``; ``info`` has a0, y_notch, disks, particle_area_fraction and
    interface_area_fraction (fractions of the non-void area).
    """
    grid = Grid(int(round(width / h)), int(round(height / h)), h)
    min_gap = 2.0 * h if min_gap is None else min_gap
    interface_thickness = h if interface_thickness is None else interface_thickness
    a0 = a0_frac * grid.width
    yn = grid.height / 2
    # notch void row: element row whose centre is nearest to yn
    jn = int(yn / h)
    y_notch = (jn + 0.5) * h
    void = notch_void_mask(grid, a0, y_notch)
    disks = []
    if particle is not None and phi > 0:
        excl = lambda x, y, r: (x < a0 + r + min_gap) and abs(y - y_notch) < r + min_gap + h
        disks = random_disks(grid.width, grid.height, phi, diameters, fractions, min_gap, seed=seed, exclude=excl)
    mat = rasterize_microstructure(grid, disks, matrix, particle if particle is not None else matrix,
                                   interface=interface if disks else None,
                                   interface_thickness=interface_thickness if disks else 0.0, void_mask=void)
    ph = mat.phase
    solid = ph != 3
    area_frac = float(np.mean(ph[solid] == 1))
    return grid, mat, {"a0": a0, "y_notch": y_notch, "disks": disks, "particle_area_fraction": area_frac,
                       "interface_area_fraction": float(np.mean(ph[solid] == 2))}
