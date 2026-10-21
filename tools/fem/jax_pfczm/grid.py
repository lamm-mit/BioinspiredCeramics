"""Structured grid of square bilinear (Q4) elements and the reference-element operators.

Conventions (do not change without updating every module):

* Node (i, j), i = 0..nx, j = 0..ny  ->  node id ``j*(nx+1) + i`` (x fastest).
* Element (i, j), i = 0..nx-1, j = 0..ny-1  ->  element id ``j*nx + i``.
* Element connectivity is counter-clockwise from the lower-left corner:
  (i, j), (i+1, j), (i+1, j+1), (i, j+1).
* Displacement dofs are interleaved: node n has dofs ``2n`` (u_x) and ``2n+1`` (u_y).
* 2x2 Gauss quadrature; Gauss point q is the one nearest to local node q.
* Strains in engineering Voigt order ``[eps_xx, eps_yy, gamma_xy]`` with gamma_xy = 2 eps_xy.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

_G = 1.0 / math.sqrt(3.0)
_XI = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
_GPS = _XI * _G  # Gauss point q is nearest to local node q


@dataclasses.dataclass(frozen=True)
class Grid:
    """Structured grid of square Q4 elements.  Node (i, j) -> j*(nx+1)+i; element (i, j) -> j*nx+i."""

    nx: int
    ny: int
    h: float
    x0: float = 0.0
    y0: float = 0.0

    @property
    def nnode(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def nel(self) -> int:
        return self.nx * self.ny

    def node_id(self, i, j):
        return np.asarray(j) * (self.nx + 1) + np.asarray(i)

    @property
    def conn(self) -> np.ndarray:
        i, j = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="xy")
        i, j = i.ravel(), j.ravel()
        return np.stack([self.node_id(i, j), self.node_id(i + 1, j), self.node_id(i + 1, j + 1),
                         self.node_id(i, j + 1)], axis=1)

    @property
    def nodes(self) -> np.ndarray:
        i, j = np.meshgrid(np.arange(self.nx + 1), np.arange(self.ny + 1), indexing="xy")
        return np.stack([self.x0 + i.ravel() * self.h, self.y0 + j.ravel() * self.h], axis=1)

    @property
    def centers(self) -> np.ndarray:
        i, j = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="xy")
        return np.stack([self.x0 + (i.ravel() + 0.5) * self.h, self.y0 + (j.ravel() + 0.5) * self.h], axis=1)

    @property
    def width(self) -> float:
        return self.nx * self.h

    @property
    def height(self) -> float:
        return self.ny * self.h


def shape_gradients(h: float) -> np.ndarray:
    """dN_a/dx_k at each Gauss point: array (4 gp, 2 [d/dx, d/dy], 4 nodes)."""
    G = np.zeros((4, 2, 4))
    for q, (xi, eta) in enumerate(_GPS):
        for a, (xa, ya) in enumerate(_XI):
            G[q, 0, a] = 0.25 * xa * (1 + ya * eta) * 2.0 / h
            G[q, 1, a] = 0.25 * ya * (1 + xa * xi) * 2.0 / h
    return G


def bmatrix(G: np.ndarray) -> np.ndarray:
    """Strain-displacement matrices (4 gp, 3 [exx, eyy, gxy], 8 dofs [ux0, uy0, ux1, uy1, ...])."""
    B = np.zeros((4, 3, 8))
    for q in range(4):
        for a in range(4):
            B[q, 0, 2 * a] = G[q, 0, a]
            B[q, 1, 2 * a + 1] = G[q, 1, a]
            B[q, 2, 2 * a] = G[q, 1, a]
            B[q, 2, 2 * a + 1] = G[q, 0, a]
    return B


# Backwards-compatible private aliases (names used in the original implementation).
_shape_gradients = shape_gradients
_bmatrix = bmatrix
