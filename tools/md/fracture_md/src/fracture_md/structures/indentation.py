"""Indentation slabs: orthorhombic oriented cells, the protocol's size rule and region masks.

Geometry (lab frame, as in :mod:`fracture_md.structures.slabs`)
---------------------------------------------------------------
* Lab z is the (hkl) surface normal; the tip comes down along -z onto the TOP
  surface.  Lab x = a1 (the canonical shortest in-plane lattice vector), lab y
  = z × x.  Berkovich azimuth 0 puts the outward normal of tip face 0 along
  lab +x, i.e. along ``orientation.x_direction`` ([110] for the (001) slab).
* In-plane cell: orthorhombic.  Rectangular nets are used as they are; a
  hexagonal net (e.g. (111), 120°) is replaced by the in-plane supercell
  (a1, a1 + 2 a2) (:func:`orthorhombic_oriented_cell`), so the lateral minimum
  image of the tip axis, cell lists and rectangular analysis frames are exact.
* The slab is a stack of whole oriented cells cut at a termination
  (``slabs._stack_oriented_cells``, shared with the cleavage builder): the
  top layer is the cut's ``species_below`` (e.g. (001) "C" or "Si"; (111)
  "Si" = shuffle, 3-fold; "C" = glide, 1-fold); the frozen bottom is whatever
  the stacking gives.
* Cell ``[n1 a1, n2 a2', (0, 0, Lz)]``: the bottom layer sits
  ``bottom_margin_A`` above z = 0 (wrapped coordinates never place a slab atom
  near frac_z = 1), and the vacuum above the top layer is at least
  ``height_cap_A + 2 (cutoff + skin)`` so that the capped tip
  (physics/indenter.py) fits inside the cell at any depth and the top surface
  never sees the bottom image.

Size rule (recommended protocol item 8)
---------------------------------------
Sizes are in units of a = sqrt(C0/π) h_max = 2.7923 h_max (the contact radius
of the ideal Berkovich at the maximum depth): lateral size L = (L/a) a and
elastic thickness t_el = (t_el/a) a, measured above the fixed bottom layer
(the thermostatted bath is elastic material and counts in t_el).  Tiers in
:data:`SIZE_TIERS`: production (8, 10), pilot (6, 8), scoping (4, 4) and the
check cells (12, 10) / (8, 15).  With h_max given, the builder asserts the
protocol's lateral rule 4.34 h_max + R_tip + cutoff < L/2 (the Berkovich
contact corners sit at 2 tanθ h = 4.34 h).  Explicit ``lateral_size_A`` /
``elastic_thickness_A`` override the tier (small test cells).

Regions (protocol item 9, :func:`indentation_masks`)
----------------------------------------------------
fixed: bottom ``fixed_thickness_A`` (>= the cutoff; default 10 Å);
bath: the next ``bath_thickness_A`` (CSVR/Langevin boundary layer; default
10 Å); nve: everything above; analysis: NVE atoms within an optional lateral
radius of the tip axis; top_layer: the outermost atomic layer (depth
reference z_s0 = its mean height).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional, Union

import numpy as np
from ase import Atoms
from ase.build import make_supercell

from fracture_md.physics.indenter import BERKOVICH_C0, IndenterGeometry
from fracture_md.schema.material import Orientation, Termination
from fracture_md.structures.slabs import (
    _cluster_heights,
    _stack_oriented_cells,
    auto_in_plane_repeats,
    conventional_lattice,
    enumerate_terminations,
    oriented_bulk_cell,
    termination_schema,
)

CONTACT_RADIUS_PER_DEPTH = math.sqrt(BERKOVICH_C0 / math.pi)  # a / h_max = 2.7923 (ideal Berkovich)
SIZE_TIERS: dict[str, tuple[float, float]] = {  # (L/a, t_el/a), recommended protocol item 8
    "production": (8.0, 10.0),
    "pilot": (6.0, 8.0),
    "scoping": (4.0, 4.0),
    "check_lateral": (12.0, 10.0),
    "check_thickness": (8.0, 15.0),
}
DEFAULT_FIXED_THICKNESS_A = 10.0  # protocol item 9: bottom 1.0 nm fixed (>= the 7.35 Å Vashishta cutoff)
DEFAULT_BATH_THICKNESS_A = 10.0  # protocol item 9: next 1.0 nm thermostatted (CSVR)
DEFAULT_BOTTOM_MARGIN_A = 2.0
_ORTHO_TOL = 1e-9  # Å


# ----------------------------------------------------------------------------
# orthorhombic oriented cell
# ----------------------------------------------------------------------------
def orthorhombic_oriented_cell(oriented_cell: Atoms, tol: float = _ORTHO_TOL) -> tuple[Atoms, np.ndarray]:
    """Oriented cell with a rectangular in-plane net, and the integer in-plane transform P (2x2).

    Identity if a1 ⊥ a2 already ((001), (110) of zincblende).  Otherwise the
    smallest in-plane supercell with a1 kept and a2' ⊥ a1, right-handed:
    a2' = a1 + 2 a2 for a 120° hexagonal net ((111)), found among
    a2' = p a1 + q a2 with |p|, |q| <= 3.  In-plane components below ``tol``
    are set to exactly 0 (fractional coordinates kept).  ``info`` is copied,
    with ``in_plane_area`` updated and ``in_plane_transform`` = P added.
    """
    c = np.asarray(oriented_cell.cell.array, float)
    if abs(c[0, 2]) > tol or abs(c[1, 2]) > tol or abs(c[0, 1]) > tol:
        raise ValueError("oriented cell must have a1 along x and a2 in the xy plane (use oriented_bulk_cell)")
    scale = max(np.linalg.norm(c[0]), np.linalg.norm(c[1]))
    if abs(c[0] @ c[1]) <= tol * scale:
        P2 = np.eye(2, dtype=int)
        out = oriented_cell.copy()
    else:
        best = None
        for q_ in (1, 2, 3):
            for p_ in range(-3, 4):
                v = p_ * c[0] + q_ * c[1]
                if abs(v @ c[0]) <= 1e-6 * scale * scale and v[1] > 0:
                    if best is None or q_ < best[1]:
                        best = (p_, q_)
            if best is not None:
                break
        if best is None:
            raise ValueError("no orthogonal in-plane supercell with |p|, |q| <= 3 (not a rectangular-centred net)")
        P2 = np.array([[1, 0], list(best)], dtype=int)
        P = np.eye(3, dtype=int)
        P[:2, :2] = P2
        out = make_supercell(oriented_cell, P, wrap=True)
        out.info.update(oriented_cell.info)
    cell = np.asarray(out.cell.array, float).copy()
    cell[:2, 2] = 0.0
    for i, j in ((0, 1), (1, 0)):
        if abs(cell[i, j]) <= tol * scale:
            cell[i, j] = 0.0
    out.set_cell(cell, scale_atoms=True)
    out.info["in_plane_area"] = float(abs(cell[0, 0] * cell[1, 1] - cell[0, 1] * cell[1, 0]))
    out.info["in_plane_transform"] = P2
    return out, P2


def contact_extent_A(geometry: IndenterGeometry, depth_A: float) -> float:
    """Protocol bound on the lateral reach of the contact at a depth (Å).

    Berkovich / cone: corner distance of the sharp tip plus the rounding
    radius, (2 tanθ or tanα) h + R_tip (4.34 h + R_tip at 65.27°, protocol
    item 8); sphere: contact radius sqrt(2 R h - h²) (R beyond the equator).
    """
    per_depth = geometry.corner_radius_per_depth
    if per_depth is not None:
        return per_depth * depth_A + geometry.tip_radius_A
    R, h = geometry.tip_radius_A, min(depth_A, geometry.tip_radius_A)
    return math.sqrt(max(2.0 * R * h - h * h, 0.0))


def resolve_top_termination(terminations: list[dict[str, Any]], top: str) -> dict[str, Any]:
    """Termination dict for a cut label ("polar_2", "shuffle", ...) or a top-layer species ("C", "Si")."""
    for t in terminations:
        if t["label"] == top:
            return t
    hits = [t for t in terminations if t["species_below"] == [top]]
    if len(hits) == 1:
        return hits[0]
    avail = [(t["label"], "+".join(t["species_below"])) for t in terminations]
    if len(hits) > 1:
        raise ValueError(f"top layer species {top!r} is ambiguous; give a cut label from {avail}")
    raise ValueError(f"top termination {top!r} not found; available (label, top species): {avail}")


# ----------------------------------------------------------------------------
# slab
# ----------------------------------------------------------------------------
@dataclasses.dataclass
class IndentationSlab:
    """A thick slab for indentation (top surface free, bottom to be frozen)."""

    atoms: Atoms
    surface_z: float  # Å, mean height of the top atomic layer (depth reference before relaxation)
    top_z: float  # Å, highest atom
    bottom_z: float  # Å, lowest atom
    layer_index: np.ndarray  # (N,) 0 = bottom layer
    layer_z: np.ndarray  # heights of the layers (Å)
    orientation: Orientation
    top_termination: Termination
    termination_info: dict[str, Any]
    oriented_cell: Atoms  # the (orthorhombic) oriented cell that was stacked
    in_plane_transform: np.ndarray  # (2, 2) int, oriented_bulk_cell net -> orthorhombic net
    in_plane_repeats: tuple[int, int]
    n_cells: int  # oriented cells stacked along the normal
    stacking_period: float  # Å
    lattice_constant_A: Optional[float]  # conventional a used (a(T)), None if the input cell was used as is
    lateral_size_A: tuple[float, float]  # (Lx, Ly)
    thickness_A: float  # top_z - bottom_z
    elastic_thickness_A: float  # thickness above the fixed layer (t_el)
    vacuum_above_A: float  # cell top minus top_z
    vacuum_gap_A: float  # top surface to the periodic image of the bottom surface
    fixed_thickness_A: float
    bath_thickness_A: float
    cutoff: float
    skin: float
    sizing: dict[str, Any]  # tier, h_max, a, L/a, t_el/a actually built, corner rule

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def center_xy(self) -> np.ndarray:
        """Default lateral tip position: the middle of the cell (Å)."""
        c = self.atoms.cell.array
        return 0.5 * (c[0, :2] + c[1, :2])

    def apex_at_depth(self, depth_A: float, xy: Optional[Any] = None) -> np.ndarray:
        """Apex (lowest tip point) for a nominal depth below :attr:`surface_z` (Å)."""
        xy_ = self.center_xy if xy is None else np.asarray(xy, float)
        return np.array([xy_[0], xy_[1], self.surface_z - float(depth_A)])

    def masks(self, **kw) -> dict[str, np.ndarray]:
        """:func:`indentation_masks` with this slab's fixed/bath thicknesses unless overridden."""
        return indentation_masks(self, **kw)


def _scale_to_lattice_constant(conv: Atoms, lattice_constant_A: float) -> Atoms:
    lat = conventional_lattice(conv)
    lengths = np.linalg.norm(lat, axis=1)
    if not (np.allclose(lengths, lengths[0], rtol=1e-6)
            and np.allclose(lat @ lat.T, np.diag(lengths**2), atol=1e-6 * lengths[0] ** 2)):
        raise ValueError("lattice_constant_A needs a cubic conventional cell; pass a cell already at a(T) instead")
    out = conv.copy()
    out.set_cell(conv.cell.array * (lattice_constant_A / lengths[0]), scale_atoms=True)
    return out


def build_indentation_slab(
    conventional_atoms: Atoms,
    miller: tuple[int, int, int] = (0, 0, 1),
    top_termination: str = "C",
    *,
    cutoff: float,
    skin: float = 0.5,
    lattice_constant_A: Optional[float] = None,
    h_max_A: Optional[float] = None,
    tier: Union[str, tuple[float, float]] = "production",
    lateral_size_A: Optional[Union[float, tuple[float, float]]] = None,
    elastic_thickness_A: Optional[float] = None,
    fixed_thickness_A: float = DEFAULT_FIXED_THICKNESS_A,
    bath_thickness_A: float = DEFAULT_BATH_THICKNESS_A,
    vacuum_A: Optional[float] = None,
    geometry: IndenterGeometry = IndenterGeometry(),
    bottom_margin_A: float = DEFAULT_BOTTOM_MARGIN_A,
    miller_cell: Optional[np.ndarray] = None,
    bond_cutoff: Optional[float] = None,
    layer_tol: float = 0.02,
) -> IndentationSlab:
    """Thick orthorhombic slab with a chosen top termination, sized by the protocol rule.

    Parameters
    ----------
    conventional_atoms : bulk crystal; with ``lattice_constant_A`` a cubic
        conventional cell is first rescaled to a(T) (e.g. 4.3705 Å, Vashishta
        at 300 K: the 0 K lattice at 300 K biases the modulus by ~3 %).
    miller, top_termination : surface plane and its top layer, as a cut label
        or a top-layer species ((001): "C" or "Si"; (110): "nonpolar";
        (111): "Si" = shuffle, "C" = glide).
    cutoff, skin : interaction range of the potential (minimum image, fixed
        layer >= cutoff, vacuum rule).
    h_max_A, tier : size by (L/a, t_el/a) with a = 2.7923 h_max (:data:`SIZE_TIERS`
        name or an explicit pair); asserts 4.34 h_max + R_tip + cutoff < L/2.
    lateral_size_A, elastic_thickness_A : explicit minimum sizes (Å),
        overriding the tier; the built sizes are rounded UP to whole cells.
    fixed_thickness_A, bath_thickness_A : bottom frozen layer (>= cutoff) and
        the thermostatted layer above it; t_el must exceed the bath.
    vacuum_A : vacuum above the top layer; default and minimum
        ``geometry.height_cap_A + 2 (cutoff + skin)``.
    """
    if h_max_A is None and (lateral_size_A is None or elastic_thickness_A is None):
        raise ValueError("give h_max_A (tier sizing) or both lateral_size_A and elastic_thickness_A")
    if fixed_thickness_A < cutoff:
        raise ValueError(f"fixed_thickness_A {fixed_thickness_A:.3f} Å < cutoff {cutoff:.3f} Å: free atoms would see "
                         "the bottom surface")
    conv = conventional_atoms if lattice_constant_A is None else _scale_to_lattice_constant(conventional_atoms,
                                                                                            lattice_constant_A)
    # ---- sizes
    sizing: dict[str, Any] = {"tier": None, "h_max_A": h_max_A}
    L_req = t_req = None
    if h_max_A is not None:
        if isinstance(tier, str) and tier not in SIZE_TIERS:
            raise ValueError(f"unknown size tier {tier!r}; available: {sorted(SIZE_TIERS)}")
        ratios = SIZE_TIERS[tier] if isinstance(tier, str) else tuple(float(x) for x in tier)
        if h_max_A + cutoff + skin >= geometry.height_cap_A:
            raise ValueError(f"height_cap_A {geometry.height_cap_A:.2f} Å <= h_max + cutoff + skin: the capped tip "
                             "would be shorter than the indentation plus one interaction shell")
        a_c = CONTACT_RADIUS_PER_DEPTH * h_max_A
        L_req, t_req = ratios[0] * a_c, ratios[1] * a_c
        sizing.update(tier=tier if isinstance(tier, str) else list(ratios), contact_radius_A=a_c,
                      L_over_a_requested=ratios[0], t_el_over_a_requested=ratios[1])
    Lxy_req = (L_req, L_req) if lateral_size_A is None else (
        (float(lateral_size_A), float(lateral_size_A)) if np.isscalar(lateral_size_A) else tuple(map(float, lateral_size_A)))
    t_el_req = t_req if elastic_thickness_A is None else float(elastic_thickness_A)
    if t_el_req <= bath_thickness_A:  # type: ignore[operator]
        raise ValueError(f"elastic thickness {t_el_req:.3f} Å must exceed the bath layer {bath_thickness_A:.3f} Å")
    v_min = geometry.height_cap_A + 2.0 * (cutoff + skin)
    vacuum_above = v_min if vacuum_A is None else float(vacuum_A)
    if vacuum_above < v_min - 1e-9:
        raise ValueError(f"vacuum {vacuum_above:.3f} Å < height_cap_A + 2 (cutoff + skin) = {v_min:.3f} Å")

    # ---- oriented cell, termination, repeats
    oc0 = oriented_bulk_cell(conv, miller, miller_cell=miller_cell)
    oc, P2 = orthorhombic_oriented_cell(oc0)
    terms = enumerate_terminations(oc, bond_cutoff=bond_cutoff, layer_tol=layer_tol)
    t = resolve_top_termination(terms, top_termination)
    d = float(oc.info["stacking_period"])
    cell = oc.cell.array
    a1, a2 = float(cell[0, 0]), float(cell[1, 1])
    reps_min = auto_in_plane_repeats(oc, cutoff + skin)
    reps = (max(reps_min[0], int(math.ceil(Lxy_req[0] / a1 - 1e-9))),  # type: ignore[operator]
            max(reps_min[1], int(math.ceil(Lxy_req[1] / a2 - 1e-9))))  # type: ignore[operator]
    gap = float(t["gap_width"])
    thickness_req = fixed_thickness_A + t_el_req  # type: ignore[operator]
    n_cells = max(1, int(math.ceil((thickness_req + gap) / d - 1e-9)))

    # ---- stack: bottom layer at bottom_margin_A, vacuum_above over the top layer
    z0 = bottom_margin_A - 0.5 * gap
    Lz = n_cells * d + vacuum_above + bottom_margin_A - gap
    slab, slab_cell = _stack_oriented_cells(oc, t["cut_z"], n_cells, reps, z0, Lz)
    p = slab.get_positions()
    hs, lab = _cluster_heights(p[:, 2], None, layer_tol)
    zmin, zmax = float(p[:, 2].min()), float(p[:, 2].max())
    top = lab == len(hs) - 1
    Lx, Ly = float(slab_cell[0, 0]), float(slab_cell[1, 1])
    sy = np.array(slab.get_chemical_symbols())
    if sorted(set(sy[top])) != sorted(t["species_below"]):  # pragma: no cover - internal consistency
        raise RuntimeError("top layer species differ from the chosen termination (internal error)")

    elastic = (zmax - zmin) - fixed_thickness_A
    sizing.update(L_A=(Lx, Ly), elastic_thickness_A=elastic, n_atoms=len(slab))
    if h_max_A is not None:
        a_c = sizing["contact_radius_A"]
        corner = contact_extent_A(geometry, h_max_A) + cutoff
        sizing.update(L_over_a=min(Lx, Ly) / a_c, t_el_over_a=elastic / a_c, corner_rule_A=corner)
        if not corner < 0.5 * min(Lx, Ly):
            raise ValueError(f"lateral rule violated: contact extent + cutoff = {corner:.2f} Å >= L/2 = "
                             f"{0.5 * min(Lx, Ly):.2f} Å; use a larger tier or lateral_size_A")

    R = oc.info["rotation_crystal_to_lab"]
    orientation = Orientation(
        plane_miller=tuple(int(i) for i in miller),
        x_direction=oc.info["x_direction_uvw"],
        y_direction=oc.info["y_direction_uvw"],
        rotation_crystal_to_lab=np.asarray(R).tolist(),
        description=(f"lab z = ({miller[0]} {miller[1]} {miller[2]}) surface normal (tip moves along -z onto the top "
                     "surface); lab x = a1 = canonical shortest in-plane lattice vector (Berkovich azimuth 0: face-0 "
                     "normal along +x); lab y = z × x; orthorhombic in-plane cell "
                     f"(in-plane transform {P2.tolist()})"),
    )
    info = dict(t)
    info["available_labels"] = [tt["label"] for tt in terms]
    info["requested_top"] = top_termination
    return IndentationSlab(
        atoms=slab,
        surface_z=float(np.mean(p[top, 2])),
        top_z=zmax,
        bottom_z=zmin,
        layer_index=lab,
        layer_z=hs,
        orientation=orientation,
        top_termination=termination_schema(t, t["label"]),
        termination_info=info,
        oriented_cell=oc,
        in_plane_transform=P2,
        in_plane_repeats=(int(reps[0]), int(reps[1])),
        n_cells=n_cells,
        stacking_period=d,
        lattice_constant_A=None if lattice_constant_A is None else float(lattice_constant_A),
        lateral_size_A=(Lx, Ly),
        thickness_A=zmax - zmin,
        elastic_thickness_A=elastic,
        vacuum_above_A=float(Lz - zmax),
        vacuum_gap_A=float(Lz - (zmax - zmin)),
        fixed_thickness_A=float(fixed_thickness_A),
        bath_thickness_A=float(bath_thickness_A),
        cutoff=float(cutoff),
        skin=float(skin),
        sizing=sizing,
    )


# ----------------------------------------------------------------------------
# regions
# ----------------------------------------------------------------------------
PARK_GAP_A = 5.0  # MD equilibration: the tip is parked max(PARK_GAP_A, initial_gap_A) above the surface
VACUUM_MARGIN_A = 1.0  # room for the relaxed surface above the built top layer when the tip starts high


def protocol_vacuum_A(protocol: Any, cutoff: float) -> float:
    """Vacuum above the top layer of a protocol's slab (both engines, :func:`slab_for_protocol`).

    ``protocol.vacuum_A`` is the requested vacuum; it is raised to the slab builder's minimum
    ``height_cap_A + 2 (cutoff + skin)`` (no self-interaction across z, capped tip inside the cell) and, in MD, to
    ``height_cap_A + max(PARK_GAP_A, initial_gap_A) + VACUUM_MARGIN_A`` so that the capped tip fits below the cell top
    at its park and start heights.  (A larger initial gap thus enlarges the cell instead of failing after the
    preparation.)
    """
    cap = float(protocol.indenter.height_cap_A)
    start = max(PARK_GAP_A, float(protocol.initial_gap_A)) if protocol.mode == "md" else 0.0
    return max(float(protocol.vacuum_A), cap + 2.0 * (float(cutoff) + float(protocol.skin_A)),
               cap + start + VACUUM_MARGIN_A)


def slab_for_protocol(protocol: Any, conventional_atoms: Atoms, *, cutoff: float, vacuum_A: Optional[float] = None,
                      geometry: Optional[IndenterGeometry] = None) -> IndentationSlab:
    """The indentation slab of a ``schema.protocol.IndentationProtocol`` (one mapping for both engines).

    ``physics.indentation.run_indentation`` (JAX) and ``engines.lammps_indent`` (LAMMPS deck generator) both
    build their slab here, so the two engines indent identical cells.  ``geometry`` defaults to the protocol's
    indenter (``IndenterGeometry(**protocol.indenter.geometry_kwargs())``); ``vacuum_A`` defaults to
    :func:`protocol_vacuum_A`.
    """
    geom = geometry if geometry is not None else IndenterGeometry(**protocol.indenter.geometry_kwargs())
    if vacuum_A is None:  # (an explicit geometry may carry a taller cap than the protocol's indenter)
        vacuum_A = max(protocol_vacuum_A(protocol, cutoff), geom.height_cap_A + 2.0 * (cutoff + protocol.skin_A))
    return build_indentation_slab(
        conventional_atoms, tuple(protocol.plane_miller), protocol.top_termination, cutoff=cutoff,
        skin=protocol.skin_A, lattice_constant_A=protocol.lattice_parameter_A,
        lateral_size_A=tuple(protocol.lateral_size_A), elastic_thickness_A=protocol.thickness_A - protocol.fixed_thickness_A,
        fixed_thickness_A=protocol.fixed_thickness_A, bath_thickness_A=protocol.thermostat_thickness_A,
        vacuum_A=vacuum_A, geometry=geom)


def indentation_masks(
    slab: IndentationSlab,
    *,
    fixed_thickness_A: Optional[float] = None,
    bath_thickness_A: Optional[float] = None,
    analysis_radius_A: Optional[float] = None,
    tip_xy: Optional[Any] = None,
    positions: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Boolean region masks (N,) of an indentation slab.

    ``fixed``: atoms less than ``fixed_thickness_A`` above the bottom layer
    (must be >= the cutoff; the frozen region spans at least that much);
    ``bath``: the next ``bath_thickness_A`` (boundary thermostat);
    ``nve``: all other atoms; ``free`` = bath | nve; ``analysis``: NVE atoms
    within ``analysis_radius_A`` (lateral minimum image) of the tip axis at
    ``tip_xy`` (default: cell centre; all NVE atoms if no radius);
    ``top_layer``: the outermost atomic layer of the built slab.  Heights are
    taken from ``positions`` (default: the slab as built).
    """
    fixed_t = slab.fixed_thickness_A if fixed_thickness_A is None else float(fixed_thickness_A)
    bath_t = slab.bath_thickness_A if bath_thickness_A is None else float(bath_thickness_A)
    if fixed_t < slab.cutoff:
        raise ValueError(f"fixed_thickness_A {fixed_t:.3f} Å < cutoff {slab.cutoff:.3f} Å")
    p = slab.atoms.get_positions() if positions is None else np.asarray(positions, float)
    h = p[:, 2] - slab.bottom_z
    tol = 1e-6
    fixed = h < fixed_t - tol
    bath = ~fixed & (h < fixed_t + bath_t - tol)
    nve = ~fixed & ~bath
    if not nve.any():
        raise ValueError("no NVE atoms: slab thinner than fixed + bath layers")
    analysis = nve.copy()
    if analysis_radius_A is not None:
        c = slab.atoms.cell.array
        L = np.array([c[0, 0], c[1, 1]])
        xy0 = slab.center_xy if tip_xy is None else np.asarray(tip_xy, float)
        dxy = p[:, :2] - xy0
        dxy -= L * np.round(dxy / L)
        analysis &= np.hypot(dxy[:, 0], dxy[:, 1]) < analysis_radius_A
    top_layer = slab.layer_index == slab.layer_index.max()
    return {"fixed": fixed, "bath": bath, "nve": nve, "free": ~fixed, "analysis": analysis, "top_layer": top_layer}
