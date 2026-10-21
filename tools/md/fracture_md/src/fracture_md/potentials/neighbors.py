"""Neighbour lists for the JAX path.

We use JAX-MD's ``partition.neighbor_list`` in *Dense* format.  ``idx`` has
shape (N, M); entry ``idx[i, m] == N`` marks padding.  Every consumer
(``energy_fn(frac, box, species, idx, params)``) works in FRACTIONAL
coordinates, whichever builder produced ``idx``.

Two builders (``build_neighbors(method=...)``):

* ``all_pairs``: fractional coordinates, ``disable_cell_list=True``.  O(N^2)
  candidates (N = 1e5 would need ~40 GB int32 + 80 GB float64), any cell
  shape.  The reference path: ``method="auto"`` keeps every system with
  N <= ``ALL_PAIRS_MAX_N`` on it, so small-cell results are unchanged.
* ``cell_list``: a Cartesian cell list for ORTHORHOMBIC boxes
  (``space.periodic(L)``; JAX-MD picks floor(L_a / (cutoff + skin)) cells per
  axis).  O(N) memory, for 1e5-1e6 atom slabs.  JAX-MD's own fractional cell
  list is not used: it takes ONE cells-per-side count from the smallest width,
  which for a wide indentation slab means ~8x larger candidate buffers.
  The handle is still driven by fractional coordinates; it converts them.

Rules enforced here (each is a common source of silently wrong energies):

1. Minimum image: every periodic cell width must exceed
   ``2 * (cutoff + skin)``.  Otherwise an atom could see only one of two
   periodic images inside the cutoff.  ``build_neighbors`` raises instead of
   returning a wrong list; use ``structures.system.auto_supercell``.
2. Overflow: a Dense list has fixed capacity M (and a cell list a fixed cell
   capacity).  After any ``update`` the caller must check ``overflowed`` and
   reallocate.  As a cleavage gap opens the neighbour count can only fall, but
   sliding, compression under a tip or MD can raise it.  The error code is
   sticky, so inside ``lax.scan`` one check per chunk suffices.  Check ONLY the
   overflow bits and never call ``nbrs.error.throw()``: JAX-MD 0.2.29 sets
   MALFORMED_BOX for VALID boxes in fractional cell lists (inverted predicate,
   partition.py:1021).
3. Skin: a list built at configuration X is valid for configuration Y only if
   no atom moved more than ``skin/2`` (JAX-MD's ``dr_threshold`` semantics).
   Protocols that move a whole block by more than that (cleavage openings)
   must rebuild.  ``pairs_within`` (small N, brute force) and ``missing_pairs``
   (scipy cKDTree, large N) are independent completeness checks.
4. Wrapping (cell list): JAX-MD bins positions with an int cast, which
   truncates towards zero, so a negative coordinate lands in the wrong cell and
   in-cutoff pairs vanish with NO error flag (551 of 25,145 pairs lost at a
   -0.3 L shift of an 8,112-atom slab).  MD integrates unwrapped fractional
   coordinates, so the cell-list handle wraps ``frac % 1`` before EVERY build
   and update (the skin test uses the periodic displacement, so wrapping never
   triggers a rebuild by itself).
5. Cell count: JAX-MD silently falls back to O(N^2) candidates when a cell
   would be >= 1/3 of the box.  ``cell_list`` requires every
   L_a > 3 (cutoff + skin) and checks after each allocation that the cell list
   was really used.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, space

from fracture_md.structures.system import AtomicSystem, perpendicular_widths

NeighborMethod = Literal["auto", "all_pairs", "cell_list"]

# method="auto": systems up to this size stay on the all-pairs path (every existing test and runner).
ALL_PAIRS_MAX_N = 4096
# method="auto" on a box the cell list cannot take (non-orthorhombic or thinner than 3 (cutoff + skin)):
# all-pairs is still used up to this size (~5 GB of candidates), above it build_neighbors raises instead.
AUTO_ALL_PAIRS_FALLBACK_MAX_N = 16384
# |off-diagonal| of an orthorhombic box (Å)
ORTHORHOMBIC_TOL_A = 1e-9
# relative margin on L_a > 3 (cutoff + skin): JAX-MD compares against a float32 copy of the box
_THREE_CELLS_MARGIN = 1e-6
# Cell-list allocation above this many candidates (N x 27 x cell capacity) goes through one jitted
# forced rebuild instead of JAX-MD's eager allocate, which materialises the candidate arrays op by op
# (GB10, float64, rc + skin = 7.85 Å, 262k atoms: eager peak 8.87 GB = 34 kB/atom, jitted 3.39 GB).
EAGER_ALLOCATE_MAX_CANDIDATES = 2**25
# first Dense-capacity guess of the jitted allocation, in units of the cell capacity: the (cutoff + skin)
# sphere is 4.19 cell volumes at most (cells are >= cutoff + skin wide); an overflow doubles it
_DENSE_GUESS_PER_CELL_CAPACITY = 4.19

_OVERFLOW_BITS = int(partition.PartitionErrorCode.NEIGHBOR_LIST_OVERFLOW | partition.PartitionErrorCode.CELL_LIST_OVERFLOW)


class MinimumImageError(ValueError):
    pass


class CellListError(ValueError):
    """The box cannot take the Cartesian cell list (not orthorhombic, or fewer than 3 cells per axis)."""


class _CartesianCellList:
    """JAX-MD real-space cell list for an orthorhombic box, driven by FRACTIONAL coordinates.

    Same ``allocate(frac, extra_capacity=0)`` / ``update(frac, nbrs)`` interface as the JAX-MD
    ``NeighborListFns`` the all-pairs path holds, so callers that use ``handle.neighbor_fn`` directly
    (physics/cleavage.py) work unchanged with either method.  Positions are wrapped (rule 4).
    """

    def __init__(self, lengths: np.ndarray, cutoff: float, skin: float, capacity_multiplier: float):
        self.lengths = np.asarray(lengths, dtype=np.float64)
        L = jnp.asarray(self.lengths)
        disp, _ = space.periodic(L)
        self.fns = partition.neighbor_list(
            disp,
            L,
            r_cutoff=cutoff,
            dr_threshold=skin,
            capacity_multiplier=capacity_multiplier,
            format=partition.Dense,
            fractional_coordinates=False,
            disable_cell_list=False,
        )
        self.cutoff, self.skin, self.capacity_multiplier = float(cutoff), float(skin), float(capacity_multiplier)
        # JAX-MD's grid: floor(L / (cutoff + skin)) cells per axis from a float32 copy of the box
        l32 = self.lengths.astype(np.float32).astype(np.float64)
        self.cells_per_axis = np.floor(l32 / (self.cutoff + self.skin)).astype(np.int64)
        # JAX-MD estimates the cell capacity on a grid re-derived from that cell size (count_cell_filling),
        # which roundoff can make one cell coarser; mirrored so the capacity equals JAX-MD's own
        n_est = np.floor(l32 / (l32 / self.cells_per_axis))
        self._est_cells = n_est.astype(np.int64)
        self._est_cell_size = l32 / n_est
        self._template = None
        self._rebuild = jax.jit(lambda pos, nbrs: self.fns.update(pos, nbrs))

    def positions(self, frac: jnp.ndarray) -> jnp.ndarray:
        """Cartesian positions wrapped into [0, L) (works under jit; keeps the dtype of `frac`)."""
        frac = jnp.asarray(frac)
        return jnp.mod(frac, 1.0) * jnp.asarray(self.lengths, dtype=frac.dtype)

    def max_cell_count(self, positions) -> int:
        """Largest number of atoms in one cell (host, O(N)), as JAX-MD's capacity estimate bins wrapped positions."""
        x = np.asarray(positions, dtype=np.float64)
        ci = np.mod((x / self._est_cell_size).astype(np.int64), self._est_cells)
        return int(np.bincount(np.ravel_multi_index(ci.T, tuple(self._est_cells))).max())

    def allocate(self, frac: jnp.ndarray, extra_capacity: int = 0, **kw):
        _reject_box_kwarg(kw)
        pos = self.positions(frac)
        n = int(pos.shape[0])
        # JAX-MD's cell capacity rule (_estimate_cell_capacity) + extra_capacity
        cell_cap = int(self.max_cell_count(pos) * self.capacity_multiplier) + int(extra_capacity)
        if n * 27 * cell_cap <= EAGER_ALLOCATE_MAX_CANDIDATES:
            nbrs = self.fns.allocate(pos, extra_capacity=extra_capacity)
        else:
            nbrs = self._allocate_jitted(pos, cell_cap, int(extra_capacity))
        if nbrs.cell_list_fn is None:  # rule 5: JAX-MD fell back to all-pairs candidates
            raise CellListError(f"JAX-MD did not build a cell list for box lengths {self.lengths.round(3).tolist()} Å")
        return nbrs

    def _allocate_jitted(self, pos, cell_cap: int, extra_capacity: int, max_tries: int = 12):
        """Allocation as ONE jitted forced rebuild of a template list (same result as JAX-MD's allocate).

        The template carries the capacities (cell: JAX-MD's rule; Dense: a density guess) and a reference
        position half a box away, so the update's skin test always rebuilds.  An overflow raises the
        capacity and retries; the result is then cut to JAX-MD's Dense capacity
        int(occupancy x capacity_multiplier + extra_capacity).
        """
        n = int(pos.shape[0])
        if self._template is None:  # static fields (cell_list_fn, update_fn, format, cell_size) of a real list
            self._template = self.fns.allocate(pos[: min(n, 64)])
        L = jnp.asarray(self.lengths, dtype=pos.dtype)
        far = jnp.mod(pos + 0.5 * L, L)

        def cap(m, c):  # JAX-MD's limits: at most N - 1 neighbours and the 27 x c candidates
            return max(1, min(n - 1, 27 * c, m))

        def guess(c):
            return cap(int(_DENSE_GUESS_PER_CELL_CAPACITY * c) + extra_capacity, c)

        m = guess(cell_cap)
        pec = partition.PartitionErrorCode
        for _ in range(max_tries):
            tmpl = dataclasses.replace(self._template, idx=jnp.full((n, m), n, dtype=jnp.int32), reference_position=far,
                                       error=partition.PartitionError(jnp.zeros((), jnp.uint8)),
                                       cell_list_capacity=cell_cap, max_occupancy=m)
            nbrs = self._rebuild(pos, tmpl)
            code = int(nbrs.error.code)
            if code & pec.CELL_LIST_OVERFLOW:  # (not expected: the capacity comes from exact counts)
                cell_cap = int(cell_cap * 1.5) + 1
                m = max(m, guess(cell_cap))
                continue
            if code & pec.NEIGHBOR_LIST_OVERFLOW:
                m = cap(2 * m, cell_cap)
                continue
            target = cap(int(occupancy(nbrs.idx) * self.capacity_multiplier + extra_capacity), cell_cap)
            if target > m:  # headroom above the guess: rebuild once more at the final capacity
                m = target
                continue
            # valid entries are compacted to the front of each row, so cutting to `target` loses nothing
            return dataclasses.replace(nbrs, idx=nbrs.idx[:, :target], max_occupancy=target)
        raise RuntimeError(f"cell-list allocation did not converge in {max_tries} tries (N = {n})")

    def update(self, frac: jnp.ndarray, nbrs, **kw):
        _reject_box_kwarg(kw)
        return self.fns.update(self.positions(frac), nbrs)


def _reject_box_kwarg(kw: dict) -> None:
    if kw.get("box") is not None:
        raise ValueError("the Cartesian cell list is built for one fixed box; use NeighborHandle.update(frac, box) "
                         "or build_neighbors on the new box")
    if kw:
        raise TypeError(f"unexpected keyword arguments {sorted(kw)}")


@dataclasses.dataclass
class NeighborHandle:
    """Owns a neighbour function and its current list.

    ``neighbor_fn`` exposes ``allocate(frac, extra_capacity=0)`` and ``update(frac, nbrs)``
    for both methods; ``method`` is the builder actually used ("all_pairs" or "cell_list")
    and ``selection`` says why (for provenance).
    """

    neighbor_fn: object
    nbrs: object
    cutoff: float
    skin: float
    method: str = "all_pairs"
    box: Optional[np.ndarray] = None
    capacity_multiplier: float = 1.5
    selection: str = ""

    @property
    def idx(self) -> jnp.ndarray:
        return self.nbrs.idx  # type: ignore[attr-defined]

    @property
    def n_atoms(self) -> int:
        return int(self.idx.shape[0])

    @property
    def capacity(self) -> int:
        """Dense capacity M (columns of idx)."""
        return int(self.idx.shape[1])

    def update(self, frac: jnp.ndarray, box: jnp.ndarray | None = None) -> "NeighborHandle":
        """Host-side update (jitted); re-allocates at `frac` if the update overflowed.

        For the cell list a new box means a new cell grid, so the list is rebuilt from scratch
        (with the minimum-image and three-cell checks for the new box).
        """
        if box is not None and self.method == "cell_list":
            b = np.asarray(box, dtype=np.float64)
            if not np.array_equal(b, self.box):
                return _build(jnp.asarray(frac), b, self.cutoff, self.skin, self.capacity_multiplier, "cell_list", 0,
                              self.selection)
            box = None  # same box, same cell grid
        kw = {} if box is None else {"box": box}
        if box is None:
            new = _jitted_update(self.neighbor_fn)(frac, self.nbrs)
        else:
            new = _jitted_update_box(self.neighbor_fn)(frac, self.nbrs, box)
        if overflowed(new):
            new = self.neighbor_fn.allocate(frac, **kw)  # type: ignore[attr-defined]
        return dataclasses.replace(self, nbrs=new, box=self.box if box is None else np.asarray(box, dtype=np.float64))

    def reallocate(self, frac: jnp.ndarray, extra_capacity: int = 0) -> "NeighborHandle":
        """Fresh allocation at `frac` (host; new shapes, so consumers recompile).

        The rollback path after an in-scan overflow: re-allocate from the chunk-START configuration
        with ``extra_capacity`` headroom (added to the neighbour AND the cell capacity) and redo the chunk.
        """
        nbrs = self.neighbor_fn.allocate(frac, extra_capacity=extra_capacity)  # type: ignore[attr-defined]
        return dataclasses.replace(self, nbrs=nbrs)

    def describe(self) -> dict[str, Any]:
        """Builder, capacities and occupancy (JSON-able, for raw data / provenance)."""
        nb = self.nbrs
        out: dict[str, Any] = dict(
            method=self.method, selection=self.selection, n_atoms=self.n_atoms, cutoff_A=float(self.cutoff),
            skin_A=float(self.skin), capacity_multiplier=float(self.capacity_multiplier), capacity=self.capacity,
            occupancy=occupancy(self.idx), overflowed=overflowed(nb),
        )
        if self.method == "cell_list":
            fn = self.neighbor_fn
            out.update(box_lengths_A=fn.lengths.tolist(), cells_per_axis=fn.cells_per_axis.tolist(),  # type: ignore[attr-defined]
                       cell_capacity=int(nb.cell_list_capacity))  # type: ignore[attr-defined]
        return out


# ---------------------------------------------------------------------------
# Caches.  Creating a JAX-MD neighbour function compiles ~6 kernels; before
# these caches every build_neighbors call paid that (~160 ms each, ~95 s per
# test-suite run).  Neighbour functions are keyed on the method, exact box,
# cutoff, skin and capacity; updates are jitted once per neighbour function.
# ---------------------------------------------------------------------------
_NFN_CACHE: "dict[tuple, object]" = {}
_UPDATE_CACHE: "dict[int, tuple]" = {}
_NFN_CACHE_MAX = 512


def _neighbor_fn(box, cutoff: float, skin: float, capacity_multiplier: float, method: str = "all_pairs"):
    b = np.asarray(box, dtype=np.float64)
    key = (method, b.tobytes(), float(cutoff), float(skin), float(capacity_multiplier))
    fn = _NFN_CACHE.get(key)
    if fn is None:
        if method == "cell_list":
            fn = _CartesianCellList(np.diag(b), cutoff, skin, capacity_multiplier)
        elif method == "all_pairs":
            disp, _ = space.periodic_general(jnp.asarray(b), fractional_coordinates=True)
            fn = partition.neighbor_list(
                disp,
                jnp.asarray(b),
                r_cutoff=cutoff,
                dr_threshold=skin,
                capacity_multiplier=capacity_multiplier,
                format=partition.Dense,
                fractional_coordinates=True,
                disable_cell_list=True,
            )
        else:
            raise ValueError(f"unknown neighbour method {method!r}")
        if len(_NFN_CACHE) >= _NFN_CACHE_MAX:
            _NFN_CACHE.pop(next(iter(_NFN_CACHE)))
        _NFN_CACHE[key] = fn
    return fn


def _jitted_update(neighbor_fn):
    ent = _UPDATE_CACHE.get(id(neighbor_fn))
    if ent is None or ent[0] is not neighbor_fn:
        ent = (neighbor_fn, jax.jit(lambda pos, nbrs: neighbor_fn.update(pos, nbrs)),
               jax.jit(lambda pos, nbrs, box: neighbor_fn.update(pos, nbrs, box=box)))
        _UPDATE_CACHE[id(neighbor_fn)] = ent
    return ent[1]


def _jitted_update_box(neighbor_fn):
    _jitted_update(neighbor_fn)
    return _UPDATE_CACHE[id(neighbor_fn)][2]


def check_minimum_image(system: AtomicSystem, interaction_range: float) -> None:
    _check_minimum_image_box(system.box, interaction_range)


def _check_minimum_image_box(box, interaction_range: float) -> None:
    widths = perpendicular_widths(np.asarray(box))
    bad = [ax for ax in range(3) if widths[ax] <= 2.0 * interaction_range]
    if bad:
        raise MinimumImageError(
            f"cell widths {widths.round(3).tolist()} Å too small for interaction range "
            f"{interaction_range:.3f} Å along axes {bad}; replicate the cell "
            "(structures.system.auto_supercell) or add vacuum"
        )


def cell_list_obstacle(box, interaction_range: float) -> Optional[str]:
    """Why the Cartesian cell list cannot take `box` (None if it can).

    Requires an orthorhombic box (|off-diagonal| < ORTHORHOMBIC_TOL_A) and at least three cells of
    width >= interaction_range (= cutoff + skin) along every axis: L_a > 3 * interaction_range.
    """
    b = np.asarray(box, dtype=np.float64)
    off = np.abs(b - np.diag(np.diag(b))).max()
    if off >= ORTHORHOMBIC_TOL_A:
        return (f"box is not orthorhombic (max |off-diagonal| {off:.3g} Å); the Cartesian cell list needs an "
                "orthorhombic cell (orient the crystal so the box is diagonal)")
    L = np.diag(b)
    need = 3.0 * interaction_range * (1.0 + _THREE_CELLS_MARGIN)
    if np.any(L <= need):
        return (f"box lengths {L.round(3).tolist()} Å: every length must exceed 3 (cutoff + skin) = "
                f"{3.0 * interaction_range:.3f} Å (else JAX-MD silently falls back to O(N^2) candidates)")
    return None


def resolve_method(box, n_atoms: int, cutoff: float, skin: float, method: NeighborMethod = "auto") -> tuple[str, str]:
    """(builder, reason) that ``build_neighbors`` uses for this box, size and requested method.

    auto: all_pairs for N <= ALL_PAIRS_MAX_N; otherwise cell_list if the box takes it; otherwise
    all_pairs up to AUTO_ALL_PAIRS_FALLBACK_MAX_N atoms, else CellListError (never a silent O(N^2)
    build at 1e5-1e6 atoms).  An explicit "cell_list" on a box it cannot take raises CellListError.
    """
    rng = cutoff + skin
    if method == "all_pairs":
        return "all_pairs", "requested"
    if method == "cell_list":
        why = cell_list_obstacle(box, rng)
        if why is not None:
            raise CellListError(why)
        return "cell_list", "requested"
    if method != "auto":
        raise ValueError(f"unknown neighbour method {method!r}; use 'auto', 'all_pairs' or 'cell_list'")
    if n_atoms <= ALL_PAIRS_MAX_N:
        return "all_pairs", f"auto: N = {n_atoms} <= ALL_PAIRS_MAX_N = {ALL_PAIRS_MAX_N}"
    why = cell_list_obstacle(box, rng)
    if why is None:
        return "cell_list", f"auto: N = {n_atoms} > {ALL_PAIRS_MAX_N}, orthorhombic box"
    if n_atoms <= AUTO_ALL_PAIRS_FALLBACK_MAX_N:
        return "all_pairs", f"auto: N = {n_atoms} <= {AUTO_ALL_PAIRS_FALLBACK_MAX_N} and no cell list: {why}"
    raise CellListError(f"{n_atoms} atoms: {why}; an O(N^2) all-pairs build is refused above "
                        f"{AUTO_ALL_PAIRS_FALLBACK_MAX_N} atoms under method='auto' (pass method='all_pairs' to force it)")


def _build(frac, box: np.ndarray, cutoff: float, skin: float, capacity_multiplier: float, method: str,
           extra_capacity: int, selection: str) -> NeighborHandle:
    _check_minimum_image_box(box, cutoff + skin)
    if method == "cell_list":
        why = cell_list_obstacle(box, cutoff + skin)
        if why is not None:
            raise CellListError(why)
    nfn = _neighbor_fn(box, cutoff, skin, capacity_multiplier, method)
    nbrs = nfn.allocate(frac, extra_capacity=extra_capacity)  # type: ignore[attr-defined]
    return NeighborHandle(neighbor_fn=nfn, nbrs=nbrs, cutoff=cutoff, skin=skin, method=method,
                          box=np.array(box, dtype=np.float64), capacity_multiplier=capacity_multiplier,
                          selection=selection)


def build_neighbors(
    system: AtomicSystem,
    cutoff: float,
    skin: float = 0.5,
    capacity_multiplier: float = 1.5,
    *,
    method: NeighborMethod = "auto",
    extra_capacity: int = 0,
) -> NeighborHandle:
    """Allocate a Dense neighbour list for `system` (idx for fractional-coordinate energy kernels).

    ``method``: "auto" (all-pairs up to ALL_PAIRS_MAX_N atoms, else the Cartesian cell list; see
    ``resolve_method``), "all_pairs" or "cell_list" (orthorhombic boxes with every L_a > 3 (cutoff + skin)).
    ``extra_capacity`` adds headroom to the Dense capacity (and, for the cell list, to the cell capacity),
    e.g. for densification under an indenter.
    """
    box = np.asarray(system.box, dtype=np.float64)
    _check_minimum_image_box(box, cutoff + skin)
    chosen, why = resolve_method(box, system.n_atoms, cutoff, skin, method)
    return _build(system.frac, box, cutoff, skin, capacity_multiplier, chosen, extra_capacity, why)


def neighbor_update_fn(handle: NeighborHandle) -> Callable:
    """Pure, jittable ``update(frac, nbrs) -> nbrs`` for use INSIDE ``lax.scan``.

    Takes (possibly unwrapped) fractional coordinates; the cell list wraps them.  Shapes never change
    in here: an overflow only sets the (sticky) error bits, so check ``overflowed(nbrs)`` after the
    scan and, if set, discard the chunk and ``handle.reallocate`` from the chunk-start configuration.
    """
    nfn = handle.neighbor_fn

    def update(frac, nbrs):
        return nfn.update(frac, nbrs)  # type: ignore[attr-defined]

    return update


def _nbrs_of(x):
    return x.nbrs if isinstance(x, NeighborHandle) else x


def overflow_bits(nbrs) -> jnp.ndarray:
    """Neighbour- and cell-list overflow bits of the error code (jittable; nonzero = overflowed).

    Deliberately ignores MALFORMED_BOX and CELL_SIZE_TOO_SMALL (see rule 2 in the module docstring).
    """
    return _nbrs_of(nbrs).error.code & _OVERFLOW_BITS  # type: ignore[attr-defined]


def overflowed(nbrs) -> bool:
    """True if the list (a NeighborHandle or a JAX-MD NeighborList) overflowed its neighbour or cell capacity."""
    return bool(overflow_bits(nbrs))


def overflow_flags(nbrs) -> dict[str, bool]:
    """Every error bit, for logs.  Only the two overflow bits mean the list is incomplete."""
    code = int(_nbrs_of(nbrs).error.code)  # type: ignore[attr-defined]
    pec = partition.PartitionErrorCode
    return {
        "neighbor_list_overflow": bool(code & pec.NEIGHBOR_LIST_OVERFLOW),
        "cell_list_overflow": bool(code & pec.CELL_LIST_OVERFLOW),
        "cell_size_too_small": bool(code & pec.CELL_SIZE_TOO_SMALL),
        "malformed_box_bit": bool(code & pec.MALFORMED_BOX),
    }


def max_row_occupancy(idx: jnp.ndarray) -> jnp.ndarray:
    """Largest number of valid (non-padding) entries in any row of a Dense idx (jittable)."""
    return jnp.max(jnp.sum(idx < idx.shape[0], axis=1))


def occupancy(nbrs) -> int:
    """Max valid neighbours per row (within cutoff + skin); capacity headroom = capacity - occupancy.

    Accepts a NeighborHandle, a JAX-MD NeighborList or a Dense idx array.
    """
    idx = nbrs if isinstance(nbrs, (np.ndarray, jnp.ndarray)) else _nbrs_of(nbrs).idx
    return int(max_row_occupancy(jnp.asarray(idx)))


def all_pairs_idx(n_atoms: int) -> jnp.ndarray:
    """Dense 'neighbour list' containing every j != i (reference for cross-checks)."""
    idx = np.array([[j for j in range(n_atoms) if j != i] for i in range(n_atoms)], dtype=np.int32)
    return jnp.asarray(idx)


def neighbor_sets(idx: jnp.ndarray) -> list[set[int]]:
    idx = np.asarray(idx)
    n = idx.shape[0]
    return [set(int(j) for j in row if j < n) for row in idx]


def pairs_within(system: AtomicSystem, r: float) -> list[set[int]]:
    """Brute-force minimum-image neighbour sets within distance r (numpy, for tests)."""
    frac = np.asarray(system.frac)
    box = np.asarray(system.box)
    d = frac[None, :, :] - frac[:, None, :]
    d -= np.round(d)
    dist = np.linalg.norm(d @ box.T, axis=-1)
    n = frac.shape[0]
    return [set(int(j) for j in np.nonzero((dist[i] < r) & (np.arange(n) != i))[0]) for i in range(n)]


def missing_pairs(idx, frac, box, r: float, rows=None) -> int:
    """Number of ordered pairs (i, j), i in `rows` (default: all), closer than `r` but absent from `idx`.

    Independent O(N log N) completeness check for large lists (scipy cKDTree with a periodic box;
    orthorhombic boxes only).  Entries of `idx` beyond `r` are allowed (the list holds cutoff + skin).
    """
    from scipy.spatial import cKDTree

    b = np.asarray(box, dtype=np.float64)
    if np.abs(b - np.diag(np.diag(b))).max() >= ORTHORHOMBIC_TOL_A:
        raise ValueError("missing_pairs needs an orthorhombic box")
    L = np.diag(b)
    x = np.mod(np.asarray(frac, dtype=np.float64), 1.0) * L
    x = np.where(x >= L, 0.0, x)  # cKDTree requires [0, L)
    idx = np.asarray(idx)
    n = idx.shape[0]
    rows = np.arange(n) if rows is None else np.asarray(rows, dtype=np.int64)
    found = cKDTree(x, boxsize=L).query_ball_point(x[rows], r)
    ii = np.repeat(rows, [len(js) for js in found])
    jj = np.concatenate([np.asarray(js, dtype=np.int64) for js in found]) if len(found) else np.zeros(0, np.int64)
    truth = ii[ii != jj] * n + jj[ii != jj]
    sub = idx[rows].astype(np.int64)
    have = (rows[:, None] * n + sub)[sub < n]
    return int(np.count_nonzero(~np.isin(truth, have)))
