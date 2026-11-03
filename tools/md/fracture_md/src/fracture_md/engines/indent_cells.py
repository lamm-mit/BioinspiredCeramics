"""Atom counts of the Berkovich indentation cells at any depth (the lead's cell-size decision, 2026-10-06).

Cells follow the size rule of docs/INDENTATION.md section 4: lateral size L and elastic thickness t_el (above the fixed
layer) in units of the ideal Berkovich contact radius at the maximum depth, a = 2.7923 h_max
(:meth:`schema.protocol.IndenterSpec.equivalent_contact_radius_A`).  The tiers are ``schema.protocol.CELL_TIERS``:
production (L/a, t_el/a) = (8, 10), pilot (6, 8), scoping (4, 4) (an upper bound on H only).

Two counts are given for every cell:

* ``continuum``: density x volume, 8 / a0^3 x L^2 x (t_el + fixed), no rounding.  This is how docs/INDENTATION.md
  quotes its sizes (1.38M at h = 1 nm and 4.61M at 1.5 nm for the production cell at a0 = 4.3705 A).
* ``atomistic_deck``: what ``examples/indent/in.berk_atomistic.lmp`` (and its ACE variant) actually builds: a square
  (001) slab of nx x nx x nz conventional cells, nx = ceil(L / a0), nz = ceil((t_el + fixed) / a0), 8 nx^2 nz sample
  atoms (``engines.lammps_indent.atomistic_cells``), plus the rigid diamond tip of height max(h_max + 10, 20) A
  (``engines.lammps_indent.atomistic_atom_estimate``: C0 H^3 / 3 x 8 / 3.567^3).

Pure arithmetic: no LAMMPS, no slab is built.  The memory projection uses a measured slope (GB of free-memory drop
per 1e6 slab atoms) and an intercept; both are inputs, measured by outputs/indent_ready (staged measurement).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Union

#: ideal Berkovich (65.27 deg) projected-area constant C0 (A = C0 h^2) and the contact radius per unit depth
BERKOVICH_C0 = 3.0 * math.sqrt(3.0) * math.tan(math.radians(65.27)) ** 2
CONTACT_RADIUS_PER_DEPTH = math.sqrt(BERKOVICH_C0 / math.pi)  # 2.7923
#: lateral reach of the Berkovich corners per unit depth (2 tan(65.27 deg) = 4.34)
CORNER_REACH_PER_DEPTH = 2.0 * math.tan(math.radians(65.27))
TIERS: dict[str, tuple[float, float]] = {"production": (8.0, 10.0), "pilot": (6.0, 8.0), "scoping": (4.0, 4.0)}
DIAMOND_A = 3.567  # the rigid tip's diamond lattice (in.berk_atomistic.lmp)
VASHISHTA_A300K = 4.3705
DEFAULT_FIXED_A = 10.0

TierLike = Union[str, Sequence[float]]


def tier_ratios(tier: TierLike) -> tuple[float, float]:
    """(L/a, t_el/a) of a tier name or an explicit pair."""
    if isinstance(tier, str):
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; known: {sorted(TIERS)}")
        return TIERS[tier]
    lr, tr = (float(v) for v in tier)
    if lr <= 0 or tr <= 0:
        raise ValueError(f"tier ratios must be positive, got {tier!r}")
    return lr, tr


def tip_height_A(h_max_A: float) -> float:
    """Default height of the rigid atomistic tip (write_atomistic_deck): max(h_max + 10, 20) A."""
    return max(float(h_max_A) + 10.0, 20.0)


def tip_atoms_estimate(height_A: float) -> int:
    """Atoms of the rigid diamond Berkovich of this height (pyramid volume C0 H^3 / 3 x 8 / 3.567^3)."""
    return int(round(BERKOVICH_C0 * height_A ** 3 / 3.0 * 8.0 / DIAMOND_A ** 3))


def cell_atoms(h_max_A: float, tier: TierLike = "production", *, a0_A: float = VASHISHTA_A300K,
               fixed_thickness_A: float = DEFAULT_FIXED_A, tip_height: Optional[float] = None,
               tip_radius_A: float = 0.0, cutoff_A: float = 5.0) -> dict[str, Any]:
    """Size and atom counts of the (L/a, t_el/a) cell at ``h_max_A`` (A).

    Returns L, t_el, the total slab thickness, the continuum count, the atomistic-deck count (nx, nz, sample, tip,
    total) and the reach check of docs/INDENTATION.md section 4 (4.34 h + R + rc < L/2; ``reach_ok``).
    """
    h = float(h_max_A)
    if h <= 0:
        raise ValueError("h_max_A must be > 0")
    lr, tr = tier_ratios(tier)
    a = CONTACT_RADIUS_PER_DEPTH * h
    L = lr * a
    t_el = tr * a
    thick = t_el + float(fixed_thickness_A)
    dens = 8.0 / a0_A ** 3
    nx = int(math.ceil(L / a0_A - 1e-9))
    nz = int(math.ceil(thick / a0_A - 1e-9))
    H = tip_height_A(h) if tip_height is None else float(tip_height)
    sample = 8 * nx * nx * nz
    tip = tip_atoms_estimate(H)
    reach = CORNER_REACH_PER_DEPTH * h + float(tip_radius_A) + float(cutoff_A)
    return {
        "h_max_A": h, "tier": [lr, tr], "a_contact_A": a, "L_A": L, "t_el_A": t_el, "thickness_A": thick,
        "a0_A": a0_A, "fixed_thickness_A": float(fixed_thickness_A),
        "continuum_atoms": int(round(dens * L * L * thick)),
        "atomistic_deck": {"nx": nx, "nz": nz, "L_built_A": nx * a0_A, "thickness_built_A": nz * a0_A,
                           "sample": sample, "tip_height_A": H, "tip_estimate": tip, "total": sample + tip},
        "reach_A": reach, "reach_ok": bool(reach < 0.5 * L),
    }


def h_for_atoms(n_atoms: float, tier: TierLike = "production", *, a0_A: float = VASHISHTA_A300K,
                fixed_thickness_A: float = DEFAULT_FIXED_A, count: str = "continuum") -> float:
    """The largest h_max (A) whose cell has at most ``n_atoms`` atoms (``count``: "continuum" or "deck" = the
    atomistic deck's sample atoms).  Bisection on the monotone count."""
    if n_atoms <= 0:
        raise ValueError("n_atoms must be > 0")

    def n_of(h: float) -> int:
        c = cell_atoms(h, tier, a0_A=a0_A, fixed_thickness_A=fixed_thickness_A)
        return c["continuum_atoms"] if count == "continuum" else c["atomistic_deck"]["sample"]

    if count not in ("continuum", "deck"):
        raise ValueError("count must be 'continuum' or 'deck'")
    lo, hi = 1e-3, 1.0
    while n_of(hi) <= n_atoms:
        lo, hi = hi, hi * 2.0
        if hi > 1e5:
            raise ValueError("n_atoms too large")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if n_of(mid) <= n_atoms:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return lo


def memory_gb(n_slab_atoms: float, gb_per_1e6: float, intercept_gb: float = 0.0) -> float:
    """Projected free-memory drop (GB) of a run with ``n_slab_atoms``: intercept + slope x N / 1e6."""
    return float(intercept_gb) + float(gb_per_1e6) * float(n_slab_atoms) / 1.0e6


def largest_cell_within(limit_gb: float, gb_per_1e6: float, intercept_gb: float = 0.0,
                        tier: TierLike = "production", *, a0_A: float = VASHISHTA_A300K,
                        fixed_thickness_A: float = DEFAULT_FIXED_A) -> dict[str, Any]:
    """The largest cell of ``tier`` (by depth) whose projected memory stays at or below ``limit_gb``: the atom budget
    (limit - intercept) / slope x 1e6, the depth whose atomistic-deck slab (+ tip) fits it, and that cell."""
    if gb_per_1e6 <= 0:
        raise ValueError("gb_per_1e6 must be > 0")
    budget = (float(limit_gb) - float(intercept_gb)) / float(gb_per_1e6) * 1.0e6
    if budget <= 0:
        return {"limit_gb": limit_gb, "atom_budget": 0, "h_max_A": None, "cell": None}
    h = h_for_atoms(budget, tier, a0_A=a0_A, fixed_thickness_A=fixed_thickness_A, count="deck")
    # the tip atoms count too: step down until slab + tip fit the budget
    for _ in range(10000):
        c = cell_atoms(h, tier, a0_A=a0_A, fixed_thickness_A=fixed_thickness_A)
        if c["atomistic_deck"]["total"] <= budget:
            break
        h -= 0.01
    return {"limit_gb": float(limit_gb), "gb_per_1e6": float(gb_per_1e6), "intercept_gb": float(intercept_gb),
            "atom_budget": int(budget), "h_max_A": round(h, 2), "cell": c,
            "projected_gb": memory_gb(c["atomistic_deck"]["total"], gb_per_1e6, intercept_gb)}


def size_table(depths_A: Sequence[float] = (10.0, 15.0, 20.0, 25.0, 30.0),
               tiers: Sequence[str] = ("production", "pilot", "scoping"), *, a0_A: float = VASHISHTA_A300K,
               gb_per_1e6: Optional[float] = None, intercept_gb: float = 0.0,
               rate_atom_steps_per_s: Optional[float] = None, steps_per_run: Optional[float] = None) -> list[dict]:
    """Rows (tier, h_max, continuum and deck counts; optional memory and wall-time projections)."""
    rows = []
    for t in tiers:
        for h in depths_A:
            c = cell_atoms(h, t, a0_A=a0_A)
            r = {"tier": t, "h_max_A": h, "continuum_atoms": c["continuum_atoms"],
                 "deck_sample": c["atomistic_deck"]["sample"], "deck_total": c["atomistic_deck"]["total"],
                 "nx": c["atomistic_deck"]["nx"], "nz": c["atomistic_deck"]["nz"], "reach_ok": c["reach_ok"]}
            if gb_per_1e6 is not None:
                r["projected_gb"] = round(memory_gb(c["atomistic_deck"]["total"], gb_per_1e6, intercept_gb), 2)
            if rate_atom_steps_per_s and steps_per_run:
                r["wall_h"] = round(c["atomistic_deck"]["total"] * steps_per_run / rate_atom_steps_per_s / 3600.0, 1)
            rows.append(r)
    return rows


__all__ = ["BERKOVICH_C0", "CONTACT_RADIUS_PER_DEPTH", "TIERS", "cell_atoms", "h_for_atoms", "largest_cell_within",
           "memory_gb", "size_table", "tier_ratios", "tip_atoms_estimate", "tip_height_A"]
