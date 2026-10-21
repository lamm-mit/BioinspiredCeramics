"""Oriented bulk cells, cleavage terminations and cleavage slabs.

Geometry conventions (lab frame, ASE rows = lattice vectors)
------------------------------------------------------------
* The cleavage plane (hkl) is given by Miller indices with respect to the
  **conventional** cell of the phase (``miller_cell``; by default the cell of
  the Atoms object passed in).  Its unit normal is ``n = G/|G|`` with
  ``G = h b1 + k b2 + l b3`` (reciprocal vectors without 2π).
* :func:`oriented_bulk_cell` returns a *primitive* 3-D periodic cell whose
  first two lattice vectors ``a1, a2`` span the (hkl) plane lattice (shortest,
  Gauss-reduced basis, 120° for hexagonal nets) and whose third vector ``a3``
  has a positive component ``d`` along the normal and the smallest possible
  in-plane tilt.  The cell is rotated so that ``a1 ∥ +x`` and ``n ∥ +z``:
  ``r_lab = R r_crystal`` with ``R`` stored in ``atoms.info``.
  ``a1`` is the shortest in-plane lattice vector; among symmetry-equivalent
  shortest vectors the one with the lexicographically largest Cartesian unit
  vector (rounded to 1e-4) is taken, so the lab frame depends on the lattice
  only — the conventional cell, its repeats and relaxed supercells give the
  same x axis.  Right-handedness ``(a1 × a2)·n > 0`` is obtained by choosing
  the SIGN of ``a2`` (``a1`` is never swapped); equal-length ``a2`` candidates
  prefer the obtuse setting.  Lab x is the crack-propagation direction used by
  the PF-CZM export (E'), lab y the crack front.  ``d`` is the
  stacking period of the plane family (e.g. one Si–C bilayer for (111) SiC).
* Atomic *layers* are sets of atoms with equal height z (tolerance
  ``layer_tol``).  A *termination* is a cut through the gap between two
  consecutive layers; its height is the gap midpoint, so no atom lies on it.
* :func:`build_cleavage_slab` stacks whole oriented cells, so each half-slab
  is an integer number of primitive cells (stoichiometric by construction) and
  **both outer surfaces are of the same termination type as the internal
  cut**.  The slab cell is ``[n1 a1, n2 a2, (0, 0, Lz)]`` with the third
  vector perpendicular to the plane, ``Lz = thickness + vacuum``.  The JAX
  path treats all three directions as periodic; the vacuum (checked by the
  cleavage protocols) guarantees that the top surface never interacts with
  the periodic image of the bottom surface.

Broken bonds are counted with a geometric bond cutoff (default: midway
between the first and second neighbour shells), independent of any
potential: they describe the *crystallography* of the cut, not its energy.

Termination labels
------------------
* zincblende/diamond {111}: the cut through the wide gap crossing 1 bond per
  surface atom is ``"shuffle"``; the cut through the narrow gap crossing 3
  bonds per surface atom is ``"glide"`` (detected from the bond counts
  {1, 3}, not from the Miller indices).
* planes whose atomic layers all have the bulk composition (e.g. zincblende
  (110)) are ``"nonpolar"``;
* otherwise (layers of a single species, e.g. zincblende (100)) ``"polar"``.
Repeated labels get suffixes ``_2, _3, ...`` in order of cut height.  The
special label ``"auto_min_broken_bonds"`` selects the cut with the fewest
broken bonds per area (ties: widest interlayer gap).
"""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction
from typing import Any, Iterator, Optional

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from fracture_md.schema.material import Orientation, Termination
from fracture_md.units import Unit, q

AUTO_TERMINATION = "auto_min_broken_bonds"


# ----------------------------------------------------------------------------
# small integer / lattice helpers
# ----------------------------------------------------------------------------
def _rationalize(vec: np.ndarray, max_den: int = 96, tol: float = 1e-6) -> Optional[tuple[int, ...]]:
    """Smallest coprime integer vector parallel to `vec` (None if irrational)."""
    v = np.asarray(vec, float)
    s = np.max(np.abs(v))
    if s == 0:
        return None
    v = v / s
    fr = [Fraction(x).limit_denominator(max_den) for x in v]
    den = 1
    for f in fr:
        den = den * f.denominator // math.gcd(den, f.denominator)
    ints = [int(f * den) for f in fr]
    g = 0
    for i in ints:
        g = math.gcd(g, abs(i))
    ints = [i // g for i in ints]
    w = np.asarray(ints, float)
    if np.linalg.norm(w / np.max(np.abs(w)) - v) > tol:
        return None
    return tuple(ints)


def conventional_lattice(atoms: Atoms, symprec: float = 1e-5) -> np.ndarray:
    """Conventional lattice vectors (rows, Å) in the *input* Cartesian frame.

    If ``atoms.cell`` is already an integer diagonal supercell of the
    conventional cell (the usual case: a conventional cell or an n1×n2×n3
    repeat of it), the input axes are kept and simply divided by the
    repeats.  Otherwise spglib's conventional setting is used (its axis labels
    may be permuted with respect to the input frame; the plane family is
    unchanged but the sign convention of polar directions may differ).
    """
    import spglib

    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if ds is None:  # pragma: no cover - triclinic P1 without symmetry info
        return np.asarray(atoms.cell.array, float)
    conv = (atoms.cell.array.T @ np.linalg.inv(ds.transformation_matrix)).T
    m = atoms.cell.array @ np.linalg.inv(conv)  # input rows in conventional basis
    mr = np.round(m)
    if np.allclose(m, mr, atol=1e-6):
        for perm in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
            mp = mr[:, perm]
            if np.allclose(mp, np.diag(np.diag(mp))) and np.all(np.abs(np.diag(mp)) >= 1):
                return atoms.cell.array / np.abs(np.diag(mp))[:, None]
    return conv


def primitive_lattice(atoms: Atoms, symprec: float = 1e-5) -> np.ndarray:
    """Primitive lattice vectors (rows) in the input frame, verified to be translations."""
    import spglib

    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers)
    res = spglib.standardize_cell(cell, to_primitive=True, no_idealize=True, symprec=symprec)
    if res is None:  # pragma: no cover
        return np.asarray(atoms.cell.array, float)
    P = np.asarray(res[0], float)
    # Verify: each primitive vector is a symmetry translation of the input crystal.
    pos = atoms.get_positions()
    inv = np.linalg.inv(atoms.cell.array)
    for t in P:
        f = (pos + t) @ inv
        f0 = pos @ inv
        d = f[:, None, :] - f0[None, :, :]
        d -= np.round(d)
        dist = np.linalg.norm(d @ atoms.cell.array, axis=-1)
        same = atoms.numbers[:, None] == atoms.numbers[None, :]
        if not np.all(np.min(np.where(same, dist, np.inf), axis=1) < 1e-4):
            raise RuntimeError("spglib primitive vector is not a lattice translation of the input")
    if np.linalg.det(P) < 0:
        P = P.copy()
        P[2] *= -1
    return P


def plane_normal(miller: tuple[int, int, int], miller_cell: np.ndarray) -> np.ndarray:
    """Unit normal (Cartesian, input frame) of plane (hkl) w.r.t. `miller_cell` (rows)."""
    B = np.linalg.inv(np.asarray(miller_cell, float)).T  # rows b_i with a_i . b_j = δ_ij
    g = np.asarray(miller, float) @ B
    if np.linalg.norm(g) == 0:
        raise ValueError("Miller indices must not all be zero")
    return g / np.linalg.norm(g)


_LEN_RTOL = 1e-4  # relative tolerance for "equal length" lattice vectors (robust to relaxation noise)


def _dir_key(v: np.ndarray) -> tuple[float, ...]:
    """Deterministic tie-break key of a direction: its Cartesian unit vector rounded to 1e-4."""
    u = np.asarray(v, float) / np.linalg.norm(v)
    return tuple(float(x) + 0.0 for x in np.round(u, 4))


def _plane_basis(P: np.ndarray, n: np.ndarray, search: int = 8):
    """In-plane primitive basis (v1, v2) and out-of-plane a3 of lattice P (rows) for normal n."""
    hp = _rationalize(P @ n)
    if hp is None:
        raise ValueError("plane normal is not a rational lattice plane of the primitive lattice")
    hp = np.asarray(hp, int)
    if hp @ (P @ n) < 0:
        hp = -hp
    Bp = np.linalg.inv(P).T
    gp = hp @ Bp
    a_min = abs(np.linalg.det(P)) * np.linalg.norm(gp)  # area of the primitive 2-D net
    d_period = 1.0 / np.linalg.norm(gp)

    for N in (search, 2 * search, 4 * search):
        rng = np.arange(-N, N + 1)
        m = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)
        m = m[np.any(m != 0, axis=1)]
        dots = m @ hp
        inplane = m[dots == 0]
        if len(inplane) < 2:
            continue
        cart = inplane @ P
        norms = np.linalg.norm(cart, axis=1)
        # a1: the shortest in-plane lattice vector; among symmetry-equivalent shortest vectors
        # the canonical one (largest lexicographic key of its Cartesian direction), so the lab
        # x axis depends on the lattice only, not on the primitive basis spglib returned
        # (conventional cell, its repeats and relaxed supercells give the same frame).
        shortest = np.nonzero(norms <= norms.min() * (1 + _LEN_RTOL))[0]
        k1 = max(shortest, key=lambda k: _dir_key(cart[k]))
        v1, m1 = cart[k1], inplane[k1]
        # a2: shortest partner completing the primitive 2-D net.  Its SIGN is chosen so that
        # (a1 x a2).n > 0 (right-handed with the normal); a1 is never swapped.  Among
        # equal-length partners prefer the obtuse (120 deg for hexagonal nets) one, then the
        # canonical direction.
        cands = []
        for mm, c, ln in zip(inplane, cart, norms):
            ar = np.cross(v1, c) @ n
            if abs(abs(ar) - a_min) < 1e-6 * a_min:
                if ar < 0:
                    mm, c = -mm, -c
                cands.append((ln, mm, c))
        if not cands:
            continue
        lmin2 = min(cc[0] for cc in cands)
        cands = [cc for cc in cands if cc[0] <= lmin2 * (1 + _LEN_RTOL)]
        _, m2, v2 = max(cands, key=lambda cc: (cc[2] @ v1 <= 1e-10 * lmin2 * lmin2, _dir_key(cc[2])))
        outs = m[dots == 1]
        if len(outs) == 0:
            continue
        # Pick a3 with hp.m = 1, then remove in-plane tilt with integer combinations of v1, v2
        # (canonical choice among equal-tilt candidates).
        m3 = outs[np.argmin(np.linalg.norm(outs @ P, axis=1))]
        a3 = m3 @ P
        par = a3 - (a3 @ n) * n
        c, *_ = np.linalg.lstsq(np.stack([v1, v2], 1), par, rcond=None)
        opts = []
        for f1 in range(math.floor(c[0]) - 1, math.ceil(c[0]) + 2):
            for f2 in range(math.floor(c[1]) - 1, math.ceil(c[1]) + 2):
                cand_m = m3 - f1 * m1 - f2 * m2
                cand = cand_m @ P
                opts.append((float(np.linalg.norm(cand - (cand @ n) * n)), cand_m, cand))
        tmin = min(o[0] for o in opts)
        opts = [o for o in opts if o[0] <= tmin + 1e-6 * max(d_period, 1.0)]
        _, m3, a3 = max(opts, key=lambda o: _dir_key(o[2]))
        M = np.stack([m1, m2, m3])
        if abs(round(abs(np.linalg.det(M))) - 1) > 1e-9:
            raise RuntimeError("oriented cell is not primitive (internal error)")
        return v1, v2, a3, d_period, tuple(int(x) for x in hp)
    raise ValueError(f"could not find a plane basis for normal {n} (increase search range)")


# ----------------------------------------------------------------------------
# oriented bulk cell
# ----------------------------------------------------------------------------
def oriented_bulk_cell(
    atoms_conventional: Atoms,
    miller: tuple[int, int, int],
    *,
    miller_cell: Optional[np.ndarray] = None,
    primitive: bool = True,
    symprec: float = 1e-5,
) -> Atoms:
    """3-D periodic cell with a1, a2 spanning (hkl) and the plane normal along +z.

    Parameters
    ----------
    atoms_conventional : bulk crystal (any periodic cell of it).
    miller : (h, k, l) with respect to ``miller_cell`` (default: the cell of
        `atoms_conventional`, i.e. conventional Miller indices).
    primitive : reduce to the primitive lattice first (smallest in-plane net).

    The returned Atoms carries ``info`` keys: ``miller``, ``rotation_crystal_to_lab``
    (3×3, r_lab = R r_crystal), ``normal_crystal``, ``x_direction_uvw`` /
    ``y_direction_uvw`` (integer directions in the miller_cell basis, or None),
    ``stacking_period`` (Å), ``in_plane_area`` (Å^2), ``primitive_miller``.
    """
    mc = np.asarray(atoms_conventional.cell.array if miller_cell is None else miller_cell, float)
    n = plane_normal(miller, mc)
    P = primitive_lattice(atoms_conventional, symprec) if primitive else np.asarray(atoms_conventional.cell.array, float)
    v1, v2, a3, d_period, hp = _plane_basis(P, n)

    x = v1 / np.linalg.norm(v1)
    z = n
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows: lab axes expressed in crystal frame

    C = np.stack([v1, v2, a3])
    Cinv = np.linalg.inv(C)
    pos = atoms_conventional.get_positions()
    f = (pos @ Cinv) % 1.0
    f[np.abs(f - 1.0) < 1e-9] = 0.0
    keep, fr, nums = [], [], []
    for i in range(len(atoms_conventional)):
        dup = False
        for fj, nj in zip(fr, nums):
            d = f[i] - fj
            d -= np.round(d)
            if np.linalg.norm(d @ C) < 1e-5:
                dup = True
                if nj != atoms_conventional.numbers[i]:
                    raise RuntimeError("two species map to the same site: cell is not a lattice of the crystal")
                break
        if not dup:
            keep.append(i)
            fr.append(f[i])
            nums.append(atoms_conventional.numbers[i])
    expected = len(atoms_conventional) * abs(np.linalg.det(C)) / abs(np.linalg.det(atoms_conventional.cell.array))
    if abs(expected - len(keep)) > 1e-6:
        raise RuntimeError(f"oriented cell has {len(keep)} atoms, expected {expected:.3f}")

    C_lab = C @ R.T
    C_lab[np.abs(C_lab) < 1e-12] = 0.0
    fr_arr = np.asarray(fr)
    # Sort by height then x, y for reproducible ordering.
    zc = fr_arr @ C_lab
    order = np.lexsort((np.round(zc[:, 1], 6), np.round(zc[:, 0], 6), np.round(zc[:, 2], 6)))
    out = Atoms(numbers=np.asarray(nums)[order], scaled_positions=fr_arr[order], cell=C_lab, pbc=True)
    mcinv = np.linalg.inv(mc)
    xd = _rationalize(v1 @ mcinv)
    yd = _rationalize(np.cross(n, v1) @ mcinv)
    out.info.update(
        miller=tuple(int(i) for i in miller),
        rotation_crystal_to_lab=R,
        normal_crystal=n,
        x_direction_uvw=xd,
        y_direction_uvw=yd,
        stacking_period=float(C_lab[2, 2]),
        in_plane_area=float(np.linalg.norm(np.cross(C_lab[0], C_lab[1]))),
        primitive_miller=hp,
        miller_cell=mc,
    )
    if abs(C_lab[2, 2] - d_period) > 1e-8:
        raise RuntimeError("stacking period mismatch (internal error)")
    return out


# ----------------------------------------------------------------------------
# layers and terminations
# ----------------------------------------------------------------------------
def _cluster_heights(z: np.ndarray, period: Optional[float], tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Group heights into layers.  Returns (layer_heights sorted, layer index per atom)."""
    z = np.asarray(z, float)
    order = np.argsort(z, kind="stable")
    labels = np.empty(len(z), int)
    heights: list[list[float]] = []
    for k, i in enumerate(order):
        if k > 0 and z[i] - heights[-1][-1] < tol:
            heights[-1].append(z[i])
        else:
            heights.append([z[i]])
        labels[i] = len(heights) - 1
    if period is not None and len(heights) > 1 and (heights[0][0] + period) - heights[-1][-1] < tol:
        # merge the last layer into the first (wrap-around)
        last = len(heights) - 1
        heights[0] = [h - period for h in heights[-1]] + heights[0]
        heights.pop()
        labels[labels == last] = 0
    hs = np.array([np.mean(h) for h in heights])
    return hs, labels


def _first_two_shells(atoms: Atoms, rmax: float = 8.0) -> tuple[float, float]:
    d = neighbor_list("d", atoms, rmax)
    d = np.sort(d)
    d1 = d[0]
    d2 = d[d > d1 * (1 + 1e-3)][0]
    return float(d1), float(d2)


def _composition(numbers) -> dict[str, int]:
    from ase.data import chemical_symbols

    out: dict[str, int] = {}
    for z in numbers:
        s = chemical_symbols[int(z)]
        out[s] = out.get(s, 0) + 1
    return out


def _same_ratio(a: dict[str, int], b: dict[str, int]) -> bool:
    if set(a) != set(b):
        return False
    ka = sorted(a)
    ra = np.array([a[k] for k in ka], float)
    rb = np.array([b[k] for k in ka], float)
    return bool(np.allclose(ra / ra.sum(), rb / rb.sum()))


def enumerate_terminations(
    oriented_cell: Atoms,
    bond_cutoff: Optional[float] = None,
    layer_tol: float = 0.02,
) -> list[dict[str, Any]]:
    """Every distinct cut height between consecutive atomic layers of an oriented cell.

    Returns one dict per interlayer gap (in order of increasing height) with
    keys: ``label``, ``index``, ``cut_z`` (Å, in the oriented cell),
    ``cut_fraction`` (cut_z / stacking period), ``gap_width`` (Å),
    ``species_below``/``species_above``, ``layer_below_composition``/
    ``layer_above_composition``, ``broken_bonds_per_cell``,
    ``broken_bonds_per_area`` (Å^-2), ``broken_bonds_per_surface_atom`` (per
    atom of the layer below; the layer above has the same count per cell),
    ``polar``, ``stoichiometric_halves``, ``bond_cutoff`` (Å), ``description``.
    """
    d = float(oriented_cell.cell.array[2, 2])
    if abs(oriented_cell.cell.array[0, 2]) > 1e-9 or abs(oriented_cell.cell.array[1, 2]) > 1e-9:
        raise ValueError("oriented cell must have a1, a2 in the xy plane (use oriented_bulk_cell)")
    area = float(np.linalg.norm(np.cross(oriented_cell.cell.array[0], oriented_cell.cell.array[1])))
    pos = oriented_cell.get_positions()
    z = pos[:, 2] % d
    hs, lab = _cluster_heights(z, d, layer_tol)
    nl = len(hs)
    if bond_cutoff is None:
        d1, d2 = _first_two_shells(oriented_cell)
        bond_cutoff = 0.5 * (d1 + d2)
    bulk_comp = _composition(oriented_cell.numbers)
    layer_comp = [_composition(oriented_cell.numbers[lab == k]) for k in range(nl)]
    polar = not all(_same_ratio(c, bulk_comp) for c in layer_comp)

    ii, jj, DD = neighbor_list("ijD", oriented_cell, bond_cutoff)
    zi = z[ii]
    up = DD[:, 2] > 1e-9
    out = []
    for k in range(nl):
        k2 = (k + 1) % nl
        z_lo = hs[k]
        z_hi = hs[k2] + (d if k2 == 0 else 0.0)
        gap = z_hi - z_lo
        zc = (0.5 * (z_lo + z_hi)) % d
        cross = np.floor((zi[up] + DD[up, 2] - zc) / d) - np.floor((zi[up] - zc) / d)
        nb = int(round(float(np.sum(cross))))
        n_below = int(np.sum(lab == k))
        n_above = int(np.sum(lab == k2))
        out.append(
            dict(
                index=k,
                cut_z=float(zc),
                cut_fraction=float(zc / d),
                gap_width=float(gap),
                layer_below=k,
                layer_above=k2,
                species_below=sorted(layer_comp[k]),
                species_above=sorted(layer_comp[k2]),
                layer_below_composition=layer_comp[k],
                layer_above_composition=layer_comp[k2],
                n_atoms_layer_below=n_below,
                n_atoms_layer_above=n_above,
                broken_bonds_per_cell=nb,
                broken_bonds_per_area=nb / area,
                broken_bonds_per_surface_atom=nb / n_below,
                polar=polar,
                stoichiometric_halves=True,
                bond_cutoff=float(bond_cutoff),
                cell_area=area,
                stacking_period=d,
            )
        )
    # ---- labels
    per_atom = {round(t["broken_bonds_per_surface_atom"], 6) for t in out}
    counts: dict[str, int] = {}
    for t in out:
        if per_atom == {1.0, 3.0}:
            base = "shuffle" if round(t["broken_bonds_per_surface_atom"]) == 1 else "glide"
        else:
            base = "polar" if polar else "nonpolar"
        counts[base] = counts.get(base, 0) + 1
        t["label"] = base if counts[base] == 1 else f"{base}_{counts[base]}"
        t["description"] = (
            f"cut between layer {t['layer_below']} ({'+'.join(t['species_below'])}) and layer "
            f"{t['layer_above']} ({'+'.join(t['species_above'])}); gap {t['gap_width']:.4f} Å; "
            f"{t['broken_bonds_per_cell']} bonds per {area:.4f} Å^2 cell "
            f"({t['broken_bonds_per_surface_atom']:.3g} per surface atom, bond cutoff {bond_cutoff:.3f} Å); "
            + ("polar layers (single-species planes): only γ_top+γ_bottom is defined without a chemical potential"
               if polar else "non-polar layers (bulk composition)")
        )
    return out


def select_termination(terminations: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if label == AUTO_TERMINATION:
        return min(terminations, key=lambda t: (round(t["broken_bonds_per_area"], 9), -t["gap_width"]))
    for t in terminations:
        if t["label"] == label:
            return t
    raise ValueError(f"termination {label!r} not found; available: {[t['label'] for t in terminations]}")


def termination_schema(t: dict[str, Any], label: Optional[str] = None) -> Termination:
    """Convert a termination dict into the schema object (``broken_bonds_per_area`` in 1/Å^2)."""
    return Termination(
        label=label or t["label"],
        cut_fraction=t["cut_fraction"],
        species_below=list(t["species_below"]),
        species_above=list(t["species_above"]),
        broken_bonds_per_area=q(t["broken_bonds_per_area"], Unit.PER_ANGSTROM2),
        polar=bool(t["polar"]),
        stoichiometric_halves=bool(t["stoichiometric_halves"]),
        description=t["description"],
    )


# ----------------------------------------------------------------------------
# cleavage slab
# ----------------------------------------------------------------------------
@dataclasses.dataclass
class CleavageSlab:
    """A slab with one internal cleavage cut.

    Iterating yields ``(atoms, upper_mask, cut_z, termination, area)`` so that
    ``atoms, upper, cut_z, term, area = build_cleavage_slab(...)`` works.
    """

    atoms: Atoms
    upper_mask: np.ndarray  # bool (N,), atoms above the cut
    cut_z: float  # Å, lab height of the cut plane (in the slab cell)
    termination: Termination
    area: float  # Å^2, in-plane cell area
    termination_info: dict[str, Any]
    orientation: Orientation
    oriented_cell: Atoms
    layer_index: np.ndarray  # (N,) 0 = bottom layer
    layer_z: np.ndarray  # heights of the layers (Å)
    n_layers: int  # actual number of atomic layers
    n_cells: int  # oriented cells stacked along the normal
    n_cells_lower: int
    in_plane_repeats: tuple[int, int]
    stacking_period: float
    vacuum_gap: float  # Å, empty space between top surface and bottom surface image
    half_thickness: tuple[float, float]  # Å, cut plane to outer layer (lower, upper)
    interaction_range: float  # cutoff + skin used for the in-plane repeats

    def __iter__(self) -> Iterator[Any]:
        return iter((self.atoms, self.upper_mask, self.cut_z, self.termination, self.area))


def auto_in_plane_repeats(oriented_cell: Atoms, interaction_range: float) -> tuple[int, int]:
    """Smallest (n1, n2) so that both in-plane perpendicular widths exceed 2*interaction_range."""
    c = oriented_cell.cell.array
    area = np.linalg.norm(np.cross(c[0], c[1]))
    w1 = area / np.linalg.norm(c[1])
    w2 = area / np.linalg.norm(c[0])
    return int(np.floor(2 * interaction_range / w1)) + 1, int(np.floor(2 * interaction_range / w2)) + 1


def _stack_oriented_cells(
    oc: Atoms,
    cut_z: float,
    n_cells: int,
    reps: tuple[int, int],
    z0: float,
    Lz: float,
) -> tuple[Atoms, np.ndarray]:
    """Stack ``n_cells`` whole oriented cells (``reps`` in-plane) starting at a termination cut.

    Every basis atom is translated by a multiple of a3 into the slice
    ``cut_z <= z < cut_z + d``; the cut is then put at height ``z0``, so the
    lowest layer is the one just above the cut and the highest layer the one
    just below it (the cut's ``species_below``).  The cell is
    ``[n1 a1, n2 a2, (0, 0, Lz)]``; atoms are wrapped and sorted by height,
    then x, then y.  Shared by the cleavage and indentation slab builders
    (the cleavage output is pinned by tests/test_slabs.py).
    """
    d = float(oc.cell.array[2, 2])
    # Translate every basis atom by a lattice vector (a multiple of a3) into the
    # slice cut_z <= z < cut_z + d, then put the cut at z = 0.
    cell = oc.cell.array
    pos = oc.get_positions()
    kshift = np.floor((pos[:, 2] - cut_z) / d)
    base_xyz = pos - kshift[:, None] * cell[2][None, :]
    base_xyz[:, 2] -= cut_z

    n1, n2 = reps
    # Cells in (k, i1, i2) order, basis atoms innermost; vectorised with the same
    # floating-point operations as the scalar loop (i1 a1 + i2 a2 + k a3, then
    # base + shift + z0), so the output is bitwise that of the original loop.
    grid = np.meshgrid(np.arange(n_cells), np.arange(n1), np.arange(n2), indexing="ij")
    k, i1, i2 = (g.reshape(-1, 1) for g in grid)
    shift = i1 * cell[0] + i2 * cell[1] + k * cell[2]
    positions = (base_xyz[None, :, :] + shift[:, None, :] + np.array([0.0, 0.0, z0])).reshape(-1, 3)
    numbers = np.tile(oc.numbers, n_cells * n1 * n2)
    slab_cell = np.array([n1 * cell[0], n2 * cell[1], [0.0, 0.0, Lz]])
    slab = Atoms(numbers=numbers, positions=positions, cell=slab_cell, pbc=True)
    slab.wrap()
    # Stable order: by height, then x, then y.
    p = slab.get_positions()
    order = np.lexsort((np.round(p[:, 1], 6), np.round(p[:, 0], 6), np.round(p[:, 2], 6)))
    return slab[order], slab_cell


def build_cleavage_slab(
    conventional_atoms: Atoms,
    miller: tuple[int, int, int],
    termination_label: str = AUTO_TERMINATION,
    n_layers: int = 12,
    in_plane_repeats: Optional[tuple[int, int]] = None,
    vacuum: float = 15.0,
    *,
    cutoff: float = 3.0,
    skin: float = 0.5,
    bond_cutoff: Optional[float] = None,
    miller_cell: Optional[np.ndarray] = None,
    layer_tol: float = 0.02,
    min_n_cells: int = 2,
) -> CleavageSlab:
    """Slab of whole oriented cells with the cleavage cut in its middle.

    ``n_layers`` counts atomic layers; it is rounded *up* to a whole number of
    oriented cells (``n_cells``), and the cut is placed after
    ``n_cells // 2`` cells so both halves are whole, stoichiometric cells and
    both outer surfaces have the same termination as the cut.  ``in_plane_repeats``
    defaults to the smallest repeats satisfying the minimum-image condition for
    ``cutoff + skin``.  Raises if a half-slab is thinner than ``cutoff + skin``
    (the outer surface would interact directly with the cut).  For rigid
    cleavage with a many-body potential the outer surfaces cannot influence
    E(δ) at all if each half is thicker than 2 × cutoff (the bond order of a
    bond depends on neighbours of its end atoms only).
    """
    oc = oriented_bulk_cell(conventional_atoms, miller, miller_cell=miller_cell)
    terms = enumerate_terminations(oc, bond_cutoff=bond_cutoff, layer_tol=layer_tol)
    t = select_termination(terms, termination_label)
    label = t["label"]
    d = oc.info["stacking_period"]
    layers_per_cell = len(terms)
    n_cells = max(min_n_cells, int(math.ceil(n_layers / layers_per_cell)))
    n_lower = n_cells // 2
    interaction_range = cutoff + skin
    reps = tuple(in_plane_repeats) if in_plane_repeats is not None else auto_in_plane_repeats(oc, interaction_range)

    n1, n2 = reps
    z0 = 0.5 * vacuum
    Lz = n_cells * d + vacuum
    slab, slab_cell = _stack_oriented_cells(oc, t["cut_z"], n_cells, reps, z0, Lz)
    p = slab.get_positions()

    cut_z = z0 + n_lower * d
    upper = p[:, 2] > cut_z
    hs, lab = _cluster_heights(p[:, 2], None, layer_tol)
    zmin, zmax = float(p[:, 2].min()), float(p[:, 2].max())
    half = (cut_z - zmin, zmax - cut_z)
    if min(half) < interaction_range:
        raise ValueError(
            f"half-slab thickness {min(half):.3f} Å < cutoff+skin {interaction_range:.3f} Å; increase n_layers"
        )
    area = float(np.linalg.norm(np.cross(slab_cell[0], slab_cell[1])))
    R = oc.info["rotation_crystal_to_lab"]
    orientation = Orientation(
        plane_miller=tuple(int(i) for i in miller),
        x_direction=oc.info["x_direction_uvw"],
        y_direction=oc.info["y_direction_uvw"],
        rotation_crystal_to_lab=np.asarray(R).tolist(),
        description=(f"lab z = ({miller[0]} {miller[1]} {miller[2]}) plane normal; lab x = a1 = canonical shortest "
                     "in-plane lattice vector (crack propagation direction for PF-CZM export); lab y = z × x "
                     "(crack front)"),
    )
    info = dict(t)
    info["available_labels"] = [tt["label"] for tt in terms]
    info["requested_label"] = termination_label
    return CleavageSlab(
        atoms=slab,
        upper_mask=upper,
        cut_z=float(cut_z),
        termination=termination_schema(t, label),
        area=area,
        termination_info=info,
        orientation=orientation,
        oriented_cell=oc,
        layer_index=lab,
        layer_z=hs,
        n_layers=len(hs),
        n_cells=n_cells,
        n_cells_lower=n_lower,
        in_plane_repeats=(int(n1), int(n2)),
        stacking_period=float(d),
        vacuum_gap=float(Lz - (zmax - zmin)),
        half_thickness=(float(half[0]), float(half[1])),
        interaction_range=float(interaction_range),
    )
