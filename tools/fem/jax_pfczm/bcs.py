"""Standard boundary conditions (bar, SENT, K-field surfing) and crack post-processing.

A boundary condition is a callable ``bc(t) -> (uD, fixed)`` with ``uD`` float (nnode, 2) prescribed
displacements and ``fixed`` bool (nnode, 2) Dirichlet mask; the mask must not depend on ``t``.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid


# ---------------------------------------------------------------------------
# Standard boundary conditions / specimens
# ---------------------------------------------------------------------------
def bar_bc(grid: Grid):
    """Uniaxial bar along x: u_x = 0 on the left edge, u_x = U on the right edge, u_y = 0 at node (0,0)."""
    nodes = grid.nodes
    left = np.isclose(nodes[:, 0], grid.x0)
    right = np.isclose(nodes[:, 0], grid.x0 + grid.width)
    fixed = np.zeros((grid.nnode, 2), bool)
    fixed[left, 0] = True
    fixed[right, 0] = True
    fixed[0, 1] = True

    def bc(U):
        uD = np.zeros((grid.nnode, 2))
        uD[right, 0] = U
        return uD, fixed

    return bc, right


def sent_bc(grid: Grid):
    """Clamped single-edge-notched tension: bottom edge u = 0, top edge u_x = 0, u_y = U."""
    nodes = grid.nodes
    bot = np.isclose(nodes[:, 1], grid.y0)
    top = np.isclose(nodes[:, 1], grid.y0 + grid.height)
    fixed = np.zeros((grid.nnode, 2), bool)
    fixed[bot, :] = True
    fixed[top, :] = True

    def bc(U):
        uD = np.zeros((grid.nnode, 2))
        uD[top, 1] = U
        return uD, fixed

    return bc, top


def k_field(x, y, K, E, nu):
    """Mode-I plane-strain crack-tip (Williams) displacement field; crack along -x."""
    mu = E / (2 * (1 + nu))
    kappa = 3 - 4 * nu
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    f = K / (2 * mu) * np.sqrt(r / (2 * np.pi))
    return f * np.cos(th / 2) * (kappa - np.cos(th)), f * np.sin(th / 2) * (kappa - np.cos(th))


def surfing_bc(grid: Grid, K: float, E: float, nu: float, x_tip0: float, ramp: float = 1.0):
    """Surfing BC (Hossain et al., JMPS 71 (2014) 15-32) on the whole outer boundary.

    Load parameter t: for t <= ramp the K-field at x_tip0 is ramped (K*t/ramp);
    for t > ramp the full field is translated to x_tip0 + (t - ramp).
    """
    nodes = grid.nodes
    bnd = (np.isclose(nodes[:, 0], grid.x0) | np.isclose(nodes[:, 0], grid.x0 + grid.width)
           | np.isclose(nodes[:, 1], grid.y0) | np.isclose(nodes[:, 1], grid.y0 + grid.height))
    fixed = np.zeros((grid.nnode, 2), bool)
    fixed[bnd, :] = True

    def bc(t):
        s = min(t / ramp, 1.0) if ramp > 0 else 1.0
        xt = x_tip0 + max(t - ramp, 0.0)
        ux, uy = k_field(nodes[bnd, 0] - xt, nodes[bnd, 1], K * s, E, nu)
        uD = np.zeros((grid.nnode, 2))
        uD[bnd, 0], uD[bnd, 1] = ux, uy
        return uD, fixed

    return bc


def crack_tip_x(grid: Grid, d: np.ndarray, j_row: int, threshold: float = 0.95) -> float:
    """Largest x on node row j_row where d >= threshold (linear interpolation)."""
    row = d[grid.node_id(np.arange(grid.nx + 1), j_row)]
    idx = np.nonzero(row >= threshold)[0]
    if idx.size == 0:
        return grid.x0
    i = int(idx.max())
    x = grid.x0 + i * grid.h
    if i < grid.nx and row[i] > row[i + 1]:
        x += grid.h * (row[i] - threshold) / (row[i] - row[i + 1])
    return x


def crack_path(grid: Grid, d: np.ndarray, x_from: float = 0.0, threshold: float = 0.9):
    """Crack path y(x) = argmax_y d along each node column where max d >= threshold."""
    D = d.reshape(grid.ny + 1, grid.nx + 1)
    xs = grid.x0 + np.arange(grid.nx + 1) * grid.h
    jmax = np.argmax(D, axis=0)
    dmax = D.max(axis=0)
    sel = (dmax >= threshold) & (xs >= x_from)
    return xs[sel], grid.y0 + jmax[sel] * grid.h
