"""4H-SiC (0001) slabs for spherical (pop-in) and Berkovich indentation (docs/MLIP_PLAN.md 7.4 item 6).

Geometry (hexagonal setting, c along lab z):

* stacking ABCB of Si-C bilayers (sites A = (0, 0), B = (1/3, 2/3), C = (2/3, 1/3) in units of the hexagonal a1, a2);
  bilayer k has Si at z = k d and C on the same site at z = k d + u c, d = c / 4, ideal u = 3/16 (the vertical Si-C
  bond, along +z = [0001]); the other three bonds of that C go to the Si of bilayer k + 1, 0.25 d higher.
  Sites: A and C between two B layers are hexagonal (h), the two B layers are cubic (k).
* orthorhombic cell for LAMMPS: lab x = a1 = [2-1-10] direction, lab y = [01-10] direction, Ly = sqrt(3) a, so each
  bilayer holds 2 Si + 2 C per (a x sqrt(3) a) rectangle;
* terminations with one dangling bond per surface atom (the low-energy, bulk-terminated cuts):
  ``top="Si"``: the (0001) Si face, top layer Si (its vertical bond up removed), bottom layer C;
  ``top="C"``: the (000-1) C face, built as the Si-face slab rotated by 180 deg about lab x' (z -> -z, y -> -y,
  a proper rotation), so the top layer is C and the bottom Si.
* ``top_site``: which bilayer of ABCB ends the slab: chooses whether the top Si (or C) layer sits on an h or a k site
  (the two inequivalent bulk-terminated 4H (0001) surfaces).

The lattice comes from a(T): give ``a`` and ``c`` of 4H at the run temperature (measured with the potential, e.g.
examples/indent/ace/in.lattice_ace.lmp), or :func:`ideal_lattice_from_3c` maps a cubic a(T) to the ideal 4H
(a = a3C / sqrt 2, c = 4 a3C / sqrt 3; real 4H has c/a = 3.2715 vs the ideal 3.2660).

Types for LAMMPS: 1 = C, 2 = Si (the decks' convention), via ``engines.lammps_indent.write_lammps_data``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

import numpy as np
from ase import Atoms

STACKING_4H = "ABCB"
SITES = {"A": (0.0, 0.0), "B": (1.0 / 3.0, 2.0 / 3.0), "C": (2.0 / 3.0, 1.0 / 3.0)}
U_IDEAL = 3.0 / 16.0
#: room-temperature 4H-SiC (literature, for reference only; the decks use the potential's own a(T), c(T))
A_4H_EXP_300K = 3.0805
C_4H_EXP_300K = 10.0848


def ideal_lattice_from_3c(a3c_A: float) -> tuple[float, float]:
    """(a, c) of the ideal 4H whose bonds equal those of 3C with cubic lattice parameter ``a3c_A``."""
    return a3c_A / math.sqrt(2.0), 4.0 * a3c_A / math.sqrt(3.0)


def site_kinds(stacking: str = STACKING_4H) -> list[str]:
    """'h' or 'k' for each layer of a periodic stacking (h: both neighbours on the same site)."""
    n = len(stacking)
    return ["h" if stacking[(i - 1) % n] == stacking[(i + 1) % n] else "k" for i in range(n)]


@dataclasses.dataclass
class Slab4H:
    atoms: Atoms
    info: dict[str, Any]

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)


def _layer_xy(site: str, a: float, nx: int, ny: int) -> np.ndarray:
    """In-plane positions of one hexagonal layer on ``site`` in the nx x ny orthorhombic supercell."""
    f0, f1 = SITES[site]
    a1 = np.array([a, 0.0])
    a2 = np.array([-0.5 * a, 0.5 * math.sqrt(3.0) * a])
    p0 = f0 * a1 + f1 * a2
    Lx, Ly = nx * a, ny * math.sqrt(3.0) * a
    base = np.array([[0.0, 0.0], [0.5 * a, 0.5 * math.sqrt(3.0) * a]])
    pts = []
    for i in range(nx):
        for j in range(ny):
            for b in base:
                pts.append(p0 + b + np.array([i * a, j * math.sqrt(3.0) * a]))
    xy = np.array(pts)
    xy[:, 0] %= Lx
    xy[:, 1] %= Ly
    xy[np.isclose(xy[:, 0], Lx, atol=1e-9), 0] = 0.0
    xy[np.isclose(xy[:, 1], Ly, atol=1e-9), 1] = 0.0
    return xy


def build_4h_0001_slab(a: float, c: float, *, nx: int, ny: int, n_bilayers: int, top: str = "Si",
                       top_site: str = "k", u: float = U_IDEAL, vacuum_A: float = 20.0,
                       bottom_gap_A: float = 0.0, stacking: str = STACKING_4H) -> Slab4H:
    """A 4H (0001) slab of ``n_bilayers`` Si-C bilayers on an nx x ny orthorhombic (a x sqrt(3) a) supercell.

    ``top``: "Si" ((0001) Si face) or "C" ((000-1) C face); ``top_site``: "h" or "k", the site of the top surface
    layer.  The bottom is the opposite polarity, one dangling bond per atom (held fixed in the decks).  Box:
    [0, Lx) x [0, Ly) x [0, Lz), the bottom layer at ``bottom_gap_A``, ``vacuum_A`` above the top layer.
    """
    if top not in ("Si", "C"):
        raise ValueError("top must be 'Si' or 'C'")
    if top_site not in ("h", "k"):
        raise ValueError("top_site must be 'h' or 'k'")
    if nx < 1 or ny < 1 or n_bilayers < 2:
        raise ValueError("nx, ny >= 1 and n_bilayers >= 2")
    if not (0.0 < u < 0.25):
        raise ValueError("u must lie in (0, 0.25)")
    if len(stacking) != 4:
        raise ValueError("a 4H stacking has 4 layers")
    d = c / 4.0
    kinds = site_kinds(stacking)
    # Si-face slab: units k = 0..n-1 are (C of bilayer k, Si of bilayer k+1); the top layer is Si of bilayer n.
    # For the C face the slab is rotated (z -> -z), so its top layer is the C of bilayer 0 (the Si-face bottom).
    # Choose the stacking offset so that the TOP layer of the final slab sits on a site of kind top_site.
    top_index = n_bilayers if top == "Si" else 0
    offs = [o for o in range(4) if kinds[(top_index + o) % 4] == top_site]
    off = offs[0]
    sym, pos = [], []
    for k in range(n_bilayers):
        sc = stacking[(k + off) % 4]
        ss = stacking[(k + 1 + off) % 4]
        for s, site, z in (("C", sc, k * d + u * c), ("Si", ss, (k + 1) * d)):
            xy = _layer_xy(site, a, nx, ny)
            sym += [s] * len(xy)
            pos += [[x, y, z] for x, y in xy]
    X = np.array(pos)
    Lx, Ly = nx * a, ny * math.sqrt(3.0) * a
    if top == "C":
        X[:, 2] = -X[:, 2]
        X[:, 1] = (-X[:, 1]) % Ly
        X[np.isclose(X[:, 1], Ly, atol=1e-9), 1] = 0.0
    X[:, 2] -= X[:, 2].min()
    X[:, 2] += bottom_gap_A
    zmax = float(X[:, 2].max())
    Lz = zmax + vacuum_A
    atoms = Atoms(symbols=sym, positions=X, cell=[Lx, Ly, Lz], pbc=[True, True, False])
    top_kind = kinds[(top_index + off) % 4]
    info = {
        "polytype": "4H", "stacking": stacking, "stacking_offset": off, "surface": "(0001) Si face" if top == "Si"
        else "(000-1) C face", "top": top, "top_site": top_kind, "bottom": "C" if top == "Si" else "Si",
        "a_A": a, "c_A": c, "u": u, "bilayer_spacing_A": d, "nx": nx, "ny": ny, "n_bilayers": n_bilayers,
        "lateral_size_A": [Lx, Ly], "thickness_A": zmax - bottom_gap_A, "box_z_A": Lz, "vacuum_A": vacuum_A,
        "n_atoms": len(atoms), "x_direction": "[2-1-10]", "y_direction": "[01-10]" if top == "Si" else "[0-110]",
        "z_direction": "[0001]" if top == "Si" else "[000-1]",
    }
    return Slab4H(atoms, info)


def slab_for_sizes(a: float, c: float, *, lateral_A: float, thickness_A: float, **kw: Any) -> Slab4H:
    """The smallest slab with both lateral sizes >= ``lateral_A`` and thickness (bottom to top layer) >=
    ``thickness_A`` (whole cells and bilayers)."""
    nx = int(math.ceil(lateral_A / a - 1e-9))
    ny = int(math.ceil(lateral_A / (math.sqrt(3.0) * a) - 1e-9))
    d = c / 4.0
    nb = max(2, int(math.ceil(thickness_A / d - 1e-9)))
    s = build_4h_0001_slab(a, c, nx=nx, ny=ny, n_bilayers=nb, **kw)
    while s.info["thickness_A"] < thickness_A - 1e-9:
        nb += 1
        s = build_4h_0001_slab(a, c, nx=nx, ny=ny, n_bilayers=nb, **kw)
    return s


def coordination(atoms: Atoms, cutoff_A: float) -> np.ndarray:
    """Neighbour counts within ``cutoff_A`` (lateral periodic images; open z)."""
    X = atoms.get_positions()
    L = np.diag(atoms.cell.array)
    out = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        dxyz = X - X[i]
        dxyz[:, 0] -= L[0] * np.round(dxyz[:, 0] / L[0])
        dxyz[:, 1] -= L[1] * np.round(dxyz[:, 1] / L[1])
        r = np.sqrt((dxyz ** 2).sum(1))
        out[i] = int(((r > 1e-8) & (r < cutoff_A)).sum())
    return out


def stacking_sequence(atoms: Atoms, a: float, species: str = "Si", tol: float = 0.05) -> str:
    """The site letters (A/B/C) of the ``species`` layers from bottom to top (layers by z; ``a`` = hexagonal a).
    Raises if a layer does not sit on one site within ``tol`` (fractional)."""
    X = atoms.get_positions()
    sym = np.array(atoms.get_chemical_symbols())
    sel = X[sym == species]
    M = np.array([[a, 0.0], [-0.5 * a, 0.5 * math.sqrt(3.0) * a]]).T  # columns a1, a2
    zs = np.sort(np.unique(np.round(sel[:, 2], 4)))
    out = []
    for z in zs:
        lay = sel[np.isclose(sel[:, 2], z, atol=1e-3)]
        f = np.linalg.solve(M, lay[:, :2].T).T % 1.0  # fractional hexagonal coordinates of every atom of the layer
        dist = {s: np.abs(((f - np.array(v)) + 0.5) % 1.0 - 0.5).max() for s, v in SITES.items()}
        best = min(dist, key=dist.get)
        if dist[best] > tol:
            raise ValueError(f"layer at z = {z:.4f} A is not on a single A/B/C site (deviation {dist[best]:.3f})")
        out.append(best)
    return "".join(out)


__all__ = ["A_4H_EXP_300K", "C_4H_EXP_300K", "STACKING_4H", "Slab4H", "build_4h_0001_slab", "coordination",
           "ideal_lattice_from_3c", "site_kinds", "slab_for_sizes", "stacking_sequence"]
