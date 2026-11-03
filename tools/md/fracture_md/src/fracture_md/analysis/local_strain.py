"""Per-atom structural fields of an indentation snapshot (numpy + scipy only, CPU).

* ``coordination``          number of neighbours of every atom within one cutoff for all pairs (the definition of
                            LAMMPS ``compute coord/atom``; used for engine cross-checks).
* ``coordination_by_rule``  bonded neighbours under a :class:`BondRule`: per species-pair cutoffs, and pairs that
                            the rule does not list never count. For SiC the rule is HETERONUCLEAR ONLY (Si-C closer
                            than 2.4 A): under the tip the homonuclear second neighbours (3.08 A at rest) are
                            compressed into a 2.35-2.5 A all-pair cutoff and would be reported as new bonds (the
                            round-2 G06 run counted 495 atoms with "new homonuclear bonds" at 1.4 nm, all of which
                            reverted on unloading). The heteronuclear 2.4 A rule is also the Gate I coordination of
                            the indentation protocol. ``bond_rule_for`` picks the rule from the potential name.
* ``displacement``          minimum-image displacement from frame 0 (A).
* ``ShearStrain``           local von Mises shear strain of Shimizu, Ogata & Li (Mater. Trans. 48 (2007) 2923), the
                            "atomic strain" of OVITO: per-atom deformation gradient F fitted by least squares to the
                            neighbour vectors of the REFERENCE frame within ``cutoff``, Green-Lagrange
                            eta = (F^T F - I)/2, eta_M = sqrt(eta_xy^2 + eta_xz^2 + eta_yz^2 + [(eta_xx-eta_yy)^2 +
                            (eta_yy-eta_zz)^2 + (eta_zz-eta_xx)^2]/6). Rotation invariant and 0 for rigid motion.
* ``surface_height_map``    top-view height map relative to the initial surface (per-bin max of all atoms, or
                            per-bin mean of a tracked atom set such as the frame-0 top layer).

Every function takes an optional ``centers`` index array: the field is then evaluated for those atoms only, with
neighbours taken from all atoms (a 4 A slice of a 1.5e5-atom slab costs ~2 % of the full evaluation).
Neighbour search: scipy cKDTree; non-periodic directions get a padded fake period so that images never
interact. Periodic directions need cutoff < L/2. Boxes are orthogonal (lo, hi bounds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

# 3C-SiC shells: first 1.89 A, second 3.08 A, third 3.61 A (a0 ~ 4.36 A, Vashishta / expt)
SIC_BOND_CUTOFF_A = 2.35  # all-pair cutoff between the 1st and 2nd shells (legacy display rule)
SIC_STRAIN_CUTOFF_A = 3.35  # between the 2nd (3.08) and 3rd (3.61) shells: 16 neighbours in bulk


# ============================================================================================ bond rules
@dataclass(frozen=True)
class BondRule:
    """Which neighbours count as bonds: unordered species pairs with their cutoffs (A).

    ``pairs`` lists (species_a, species_b, cutoff_A). A pair of species that is not listed counts only if
    ``default_A`` is set (then with that cutoff); with ``default_A=None`` it never counts. Distances are
    compared strictly (r < cutoff), like LAMMPS ``compute coord/atom``.
    """

    name: str
    pairs: tuple[tuple[str, str, float], ...] = ()
    default_A: float | None = None
    note: str = ""

    def cutoff(self, a: str, b: str) -> float | None:
        for p, q, c in self.pairs:
            if sorted((p, q)) == sorted((a, b)):
                return float(c)
        return self.default_A

    @property
    def max_cutoff_A(self) -> float:
        cs = [c for _, _, c in self.pairs] + ([self.default_A] if self.default_A is not None else [])
        if not cs:
            raise ValueError(f"bond rule {self.name!r} has no cutoff")
        return float(max(cs))

    @property
    def heteronuclear_only(self) -> bool:
        return self.default_A is None and all(p != q for p, q, _ in self.pairs)

    def label(self) -> str:
        """Plain-language description for a colour bar ("Si–C neighbours closer than 2.40 Å")."""
        if self.default_A is not None and not self.pairs:
            return f"neighbours closer than {self.default_A:.2f} Å"
        parts = [f"{max(p, q)}–{min(p, q)}" for p, q, _ in self.pairs]  # "Si–C"
        cs = {round(c, 3) for _, _, c in self.pairs}
        if len(cs) == 1:
            return f"{' and '.join(parts)} neighbours closer than {self.pairs[0][2]:.2f} Å"
        return ", ".join(f"{s} < {c:.2f} Å" for s, (_, _, c) in zip(parts, self.pairs))

    def to_dict(self) -> dict:
        return {"name": self.name, "pairs": [list(p) for p in self.pairs], "default_A": self.default_A,
                "heteronuclear_only": self.heteronuclear_only, "note": self.note}


SIC_HETERONUCLEAR_2P4 = BondRule(
    "SiC heteronuclear-only, Si-C < 2.4 A", (("C", "Si", 2.4),), None,
    note="counts Si-C bonds only; compressed homonuclear second neighbours (3.08 A at rest) are never bonds")
VASHISHTA_SIC_BONDS = BondRule(
    "Vashishta 2007 SiC: heteronuclear-only, Si-C < 2.4 A", (("C", "Si", 2.4),), None,
    note="Gate I coordination of the indentation protocol (heteronuclear 2.4 A)")
ALL_PAIRS_2P35 = BondRule("all pairs < 2.35 A (not potential-aware)", (), SIC_BOND_CUTOFF_A,
                          note="legacy display rule; miscounts compressed homonuclear second neighbours in SiC")
LAMMPS_COORD_2P5 = BondRule("all pairs < 2.5 A (LAMMPS coord/atom cross-check)", (), 2.5)

# potential-name keyword (lower case) -> rule; the first match wins
BOND_RULES_BY_POTENTIAL: tuple[tuple[str, BondRule], ...] = (
    ("vashishta", VASHISHTA_SIC_BONDS),
    ("tersoff", SIC_HETERONUCLEAR_2P4),
    ("erhart", SIC_HETERONUCLEAR_2P4),
    ("albe", SIC_HETERONUCLEAR_2P4),
)


def bond_rule_for(potential: str | None, species=None) -> BondRule:
    """The bond rule for a potential (by name) or, failing that, for the species present.

    Vashishta 2007 SiC -> heteronuclear-only Si-C < 2.4 A (the protocol's Gate I rule); Tersoff / Erhart-Albe SiC
    -> the same heteronuclear rule; any other Si + C system -> heteronuclear 2.4 A; anything else -> the all-pair
    2.35 A display rule, which is NOT potential-aware (its ``name`` says so; renders record the rule used).
    """
    name = (potential or "").lower()
    for key, rule in BOND_RULES_BY_POTENTIAL:
        if key in name:
            return rule
    if species is not None and set(map(str, np.unique(np.asarray(species)))) == {"C", "Si"}:
        return SIC_HETERONUCLEAR_2P4
    return ALL_PAIRS_2P35


# ============================================================================================ neighbours
def _wrap_for_tree(pos, lo, hi, pbc, cutoff):
    pos = np.asarray(pos, float)
    L = np.asarray(hi, float) - np.asarray(lo, float)
    x = pos - np.asarray(lo, float)
    box = np.empty(3)
    for d in range(3):
        if pbc[d]:
            if cutoff >= 0.5 * L[d]:
                raise ValueError(f"cutoff {cutoff} A must be < half the periodic box length {L[d]:.3f} A")
            x[:, d] = np.mod(x[:, d], L[d])
            box[d] = L[d]
        else:
            x[:, d] -= x[:, d].min()
            box[d] = x[:, d].max() + 2.0 * cutoff + 1.0
    x = np.where(x >= box, x - box, x)  # np.mod can round up to exactly L
    x = np.where(x < 0, 0.0, x)
    return x, box


def min_image(d, lo, hi, pbc):
    """Minimum-image vectors for periodic directions (orthogonal box)."""
    d = np.array(d, float, copy=True)
    L = np.asarray(hi, float) - np.asarray(lo, float)
    for k in range(3):
        if pbc[k]:
            d[..., k] -= L[k] * np.round(d[..., k] / L[k])
    return d


def neighbor_pairs(pos, lo, hi, pbc, cutoff) -> np.ndarray:
    """All unordered pairs (P, 2) with i < j and |r_ij| <= cutoff (minimum image)."""
    x, box = _wrap_for_tree(pos, lo, hi, pbc, cutoff)
    tree = cKDTree(x, boxsize=box)
    return tree.query_pairs(cutoff, output_type="ndarray")


def directed_pairs(pos, lo, hi, pbc, cutoff, centers=None) -> tuple[np.ndarray, np.ndarray]:
    """Directed neighbour pairs (i, j), i != j, |r_ij| <= cutoff, with i restricted to ``centers`` if given.

    Every unordered pair appears twice when ``centers`` is None (i -> j and j -> i).
    """
    if centers is None:
        p = neighbor_pairs(pos, lo, hi, pbc, cutoff)
        return np.concatenate([p[:, 0], p[:, 1]]), np.concatenate([p[:, 1], p[:, 0]])
    centers = np.asarray(centers, int)
    x, box = _wrap_for_tree(pos, lo, hi, pbc, cutoff)
    if len(centers) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    sdm = cKDTree(x[centers], boxsize=box).sparse_distance_matrix(cKDTree(x, boxsize=box), cutoff,
                                                                  output_type="ndarray")
    i = centers[sdm["i"]]
    j = sdm["j"].astype(int)
    keep = i != j
    order = np.lexsort((j[keep], i[keep]))  # deterministic order, independent of the tree traversal
    return i[keep][order], j[keep][order]


def coordination(pos, lo, hi, pbc, cutoff=SIC_BOND_CUTOFF_A, centers=None) -> np.ndarray:
    """Neighbours within ``cutoff`` for all pairs (LAMMPS ``compute coord/atom cutoff <c>`` with no type filter).

    Returns (N,) or, with ``centers``, (len(centers),) in the order of ``centers``.
    """
    n = len(pos)
    i, j = directed_pairs(pos, lo, hi, pbc, cutoff, centers)
    r = np.linalg.norm(min_image(np.asarray(pos, float)[j] - np.asarray(pos, float)[i], lo, hi, pbc), axis=-1)
    cnt = np.bincount(i[r < cutoff], minlength=n)
    return cnt if centers is None else cnt[np.asarray(centers, int)]


def coordination_by_rule(pos, species, lo, hi, pbc, rule: BondRule, centers=None) -> np.ndarray:
    """Bonded neighbours under ``rule`` (per species-pair cutoffs; unlisted pairs never bond unless default_A).

    Returns (N,) or, with ``centers``, (len(centers),) in the order of ``centers``.
    """
    pos = np.asarray(pos, float)
    sp = np.asarray(species).astype(str)
    n = len(pos)
    kinds, code = np.unique(sp, return_inverse=True)
    cut = np.zeros((len(kinds), len(kinds)))
    for a in range(len(kinds)):
        for b in range(len(kinds)):
            c = rule.cutoff(str(kinds[a]), str(kinds[b]))
            cut[a, b] = 0.0 if c is None else c
    rmax = float(cut.max())
    if rmax <= 0:
        cnt = np.zeros(n, int)
        return cnt if centers is None else cnt[np.asarray(centers, int)]
    i, j = directed_pairs(pos, lo, hi, pbc, rmax, centers)
    r = np.linalg.norm(min_image(pos[j] - pos[i], lo, hi, pbc), axis=-1)
    bonded = r < cut[code[i], code[j]]
    cnt = np.bincount(i[bonded], minlength=n)
    return cnt if centers is None else cnt[np.asarray(centers, int)]


def displacement(pos_t, pos_0, lo, hi, pbc) -> np.ndarray:
    return min_image(np.asarray(pos_t) - np.asarray(pos_0), lo, hi, pbc)


# ============================================================================================ local strain
class ShearStrain:
    """Von Mises local shear strain relative to a fixed reference frame (neighbours fixed at the reference).

    ``centers``: evaluate for these atoms only (neighbours from all atoms); ``__call__`` then returns
    (len(centers),) values in that order, else (N,). Atoms with fewer than ``min_neighbors`` neighbours or an
    ill-conditioned neighbour set (e.g. a thin slab without halo) return NaN.
    """

    def __init__(self, pos_ref, lo, hi, pbc, cutoff=SIC_STRAIN_CUTOFF_A, min_neighbors=3, cond_max=1e3,
                 centers=None):
        self.pbc = np.asarray(pbc, bool)
        self.pos_ref = np.asarray(pos_ref, float)
        n_all = len(self.pos_ref)
        self.centers = np.arange(n_all) if centers is None else np.asarray(centers, int)
        self.n = len(self.centers)
        i, self.j = directed_pairs(self.pos_ref, lo, hi, pbc, cutoff, None if centers is None else self.centers)
        d0 = min_image(self.pos_ref[self.j] - self.pos_ref[i], lo, hi, pbc)
        within = np.linalg.norm(d0, axis=-1) < cutoff
        i, self.j, self.d0 = i[within], self.j[within], d0[within]
        pos_in_centers = np.full(n_all, -1)
        pos_in_centers[self.centers] = np.arange(self.n)
        self.i = i  # full index of the centre atom of each pair
        self.ci = pos_in_centers[i]  # its row in the output
        Y = self._sum_outer(self.d0, self.d0)
        cnt = np.bincount(self.ci, minlength=self.n)
        ev = np.linalg.eigvalsh(Y)  # ascending
        ok = (cnt >= min_neighbors) & (ev[:, 0] > 0) & (ev[:, -1] <= cond_max * np.maximum(ev[:, 0], 1e-300))
        self.valid = ok
        Yinv = np.zeros_like(Y)
        Yinv[ok] = np.linalg.inv(Y[ok])
        self.Yinv = Yinv

    def _sum_outer(self, a, b):
        out = np.zeros((self.n, 3, 3))
        for r in range(3):
            for c in range(3):
                out[:, r, c] = np.bincount(self.ci, weights=a[:, r] * b[:, c], minlength=self.n)
        return out

    def deformation_gradient(self, pos_t, lo, hi) -> np.ndarray:
        P = np.asarray(pos_t, float)
        d = min_image(P[self.j] - P[self.i], lo, hi, self.pbc)
        return self._sum_outer(d, self.d0) @ self.Yinv

    def __call__(self, pos_t, lo, hi) -> np.ndarray:
        F = self.deformation_gradient(pos_t, lo, hi)
        E = 0.5 * (np.swapaxes(F, 1, 2) @ F - np.eye(3))
        vm = np.sqrt(E[:, 0, 1] ** 2 + E[:, 0, 2] ** 2 + E[:, 1, 2] ** 2
                     + ((E[:, 0, 0] - E[:, 1, 1]) ** 2 + (E[:, 1, 1] - E[:, 2, 2]) ** 2
                        + (E[:, 2, 2] - E[:, 0, 0]) ** 2) / 6.0)
        return np.where(self.valid, vm, np.nan)


def von_mises_shear(pos_t, pos_ref, lo, hi, pbc, cutoff=SIC_STRAIN_CUTOFF_A, centers=None) -> np.ndarray:
    return ShearStrain(pos_ref, lo, hi, pbc, cutoff, centers=centers)(pos_t, lo, hi)


# ============================================================================================ surface
def surface_height_map(pos, lo, hi, pbc, z0, bin_A=4.0, fill=True, reduce="max"):
    """Top-view height map (A) relative to z0 on a regular (x, y) grid.

    reduce="max": height of the highest atom in each bin (any atom set).
    reduce="mean": mean height of the atoms in each bin; use it with a TRACKED set of atoms, e.g. the atoms that
    formed the top layer at frame 0 (the renderer does this). The per-bin maximum over all atoms shows a
    spurious checkerboard when bins are about the in-plane atomic spacing (bins without a top-layer atom
    report the next layer down), which the tracked-layer mean avoids.
    Returns (H (ny, nx), x_edges, y_edges). Lateral periodic coordinates are wrapped into the box first.
    Empty bins are filled from the nearest filled bin when ``fill``.
    """
    pos = np.asarray(pos, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    L = hi - lo
    xy = pos[:, :2] - lo[:2]
    for k in range(2):
        if pbc[k]:
            xy[:, k] = np.mod(xy[:, k], L[k])
    nx = max(1, int(round(L[0] / bin_A)))
    ny = max(1, int(round(L[1] / bin_A)))
    ix = np.clip((xy[:, 0] / L[0] * nx).astype(int), 0, nx - 1)
    iy = np.clip((xy[:, 1] / L[1] * ny).astype(int), 0, ny - 1)
    flat = iy * nx + ix
    if reduce == "max":
        H = np.full(nx * ny, -np.inf)
        np.maximum.at(H, flat, pos[:, 2])
    elif reduce == "mean":
        cnt = np.bincount(flat, minlength=nx * ny).astype(float)
        sm = np.bincount(flat, weights=pos[:, 2], minlength=nx * ny)
        with np.errstate(invalid="ignore", divide="ignore"):
            H = np.where(cnt > 0, sm / cnt, -np.inf)
    else:
        raise ValueError("reduce must be 'max' or 'mean'")
    H = H.reshape(ny, nx)
    empty = ~np.isfinite(H)
    if fill and empty.any() and (~empty).any():
        from scipy.ndimage import distance_transform_edt

        _, (iy_n, ix_n) = distance_transform_edt(empty, return_indices=True)
        H = H[iy_n, ix_n]
    H = np.where(np.isfinite(H), H - z0, np.nan)
    xe = lo[0] + np.linspace(0, L[0], nx + 1)
    ye = lo[1] + np.linspace(0, L[1], ny + 1)
    return H, xe, ye


def top_layer_atoms(pos0, z0, tol_A=0.45) -> np.ndarray:
    """Indices of the atoms within tol_A of the initial surface height z0 (the top atomic layer)."""
    return np.flatnonzero(np.abs(np.asarray(pos0)[:, 2] - z0) <= tol_A)
