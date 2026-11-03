"""Cleavage reference case: opening curves, traction–separation laws, surface energies.

Physical set-up
---------------
A slab built by :func:`fracture_md.structures.slabs.build_cleavage_slab`
contains one internal cut (``upper_mask`` marks atoms above it) and two outer
free surfaces of the same termination bounding a vacuum.  All three cell
directions are periodic on the JAX path; the vacuum is checked so that the
top surface never comes within ``cutoff + skin`` of the periodic image of the
bottom surface (``vacuum_gap - δ_max > cutoff + skin``).  The in-plane lattice
is always held at the bulk value ("laterally constrained": the crack plane is
embedded in a large body), so these are uniaxial-strain separation curves.

Opening coordinate and signs
----------------------------
The opening δ (Å) is the displacement of the upper block along the plane
normal n (lab +z) *in addition to* the bulk interlayer spacing, so δ = 0 is
the perfect crystal.  E(δ) is the total energy (eV); the per-area energy is
Φ(δ) = [E(δ) − E(0)]/A (J/m²); the normal traction is T(δ) = (1/A) dE/dδ
(GPa), positive = cohesive (resisting opening).  W_sep = Φ(δ_max) for δ_max
beyond the interaction range; for a rigid cut W_sep = 2γ_unrelaxed.

Autodiff convention (critical)
------------------------------
JAX-MD's ``periodic_general(fractional_coordinates=True)`` has a custom JVP:
``jax.grad(E, argnums=0)(frac)`` is the *real-space* gradient dE/dR (eV/Å).
We therefore never differentiate through our own map δ -> frac + δ s (s =
inv(box) n, fractional displacement per Å): that would return (1/L_z) dE/dδ.
Instead dE/dδ = Σ_{i ∈ upper} (∂E/∂R_i)·n with ∂E/∂R from grad-w.r.t.-frac,
and finite differences displace atoms in real space.

Neighbour lists as the gap opens
--------------------------------
Policy ``rebuild_each_step`` allocates a fresh Dense list at every opening;
``static_with_skin`` keeps a Verlet list with skin and lets JAX-MD's
``update`` rebuild it once any atom moved more than skin/2 since the last
build (capacity overflow -> reallocation).  At every protocol opening the
energy is also evaluated with an all-pairs list (every j ≠ i; an independent
numpy brute-force list for > 256 atoms) and with the list built at δ = 0.
For a rigid *normal* opening all cross-gap distances increase monotonically,
so the δ = 0 list is a superset of the interacting pairs at every δ > 0 (as
long as the vacuum check holds); this is verified, and is what makes the fine
integration grid cheap.  Rigid *sliding* brings new pairs into range, and
the δ = 0 list then becomes wrong beyond the skin — that is detected.

Relaxed curves and snap-back
----------------------------
:func:`relaxed_opening_curve` is displacement controlled: the bottom
``grip_layers`` layers are fixed, the top ``grip_layers`` layers are displaced
rigidly by δ and all other atoms relax (continuation from the previous
opening, with the extra opening seeded at the cut plane).  The traction is
the grip reaction −(1/A) Σ_top F_z, which by the envelope theorem equals
d E_relaxed/dδ.  Before the peak the whole slab stretches elastically; after
the peak the elastic energy stored between the grips is released (elastic
snap-back), so the post-peak relaxed curve *depends on slab thickness* and is
not a material cohesive law.  An approximate correction removes the elastic
extension of the material between the grips:
δ_coh = δ − T · h_el / C'33, with h_el = (inner grip separation − cut gap
width) and C'33 the relaxed-ion uniaxial-strain modulus normal to the plane
(computed here from the oriented bulk cell).  Both curves are reported.

Cutoff artefact
---------------
Short-range bond-order potentials (Tersoff, Erhart–Albe) switch interactions
off between R1 and R2 (≈ 2.2–3.0 Å), i.e. exactly where cleavage bonds are
being broken.  :func:`cutoff_artifact_check` reports whether any cross-gap
pair lies inside a switching interval at the opening of peak traction; if so
the peak traction and its opening are set by the switching function rather
than by the bonding physics (Pastewka et al., PRB 78, 161402(R) (2008),
doi:10.1103/PhysRevB.78.161402).  W_sep is an end-state energy difference
and is not affected by this artefact.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import sys
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

from fracture_md.potentials.base import EnergyModel
from fracture_md.potentials.neighbors import NeighborHandle, all_pairs_idx, build_neighbors
from fracture_md.schema.common import CheckResult, Conditions, Uncertainty, UncertaintyKind
from fracture_md.structures.system import AtomicSystem, auto_supercell, perpendicular_widths
from fracture_md.units import EV_PER_A2_TO_J_PER_M2, EV_PER_A3_TO_GPA, Unit

ALL_PAIRS_MAX_ATOMS = 256


# ============================================================================
# small helpers
# ============================================================================
def _plane_normal_and_area(box: np.ndarray) -> tuple[np.ndarray, float]:
    b = np.asarray(box, float)
    c = np.cross(b[:, 0], b[:, 1])
    a = float(np.linalg.norm(c))
    return c / a, a


def _real_positions(frac, box) -> np.ndarray:
    return np.asarray(frac) @ np.asarray(box).T


def _valid_count(idx: np.ndarray) -> int:
    idx = np.asarray(idx)
    return int(np.max(np.sum(idx < idx.shape[0], axis=1))) if idx.size else 0


def _bruteforce_idx(frac: np.ndarray, box: np.ndarray, r: float) -> jnp.ndarray:
    """Independent numpy minimum-image Dense list (pairs with distance < r)."""
    frac = np.asarray(frac)
    box = np.asarray(box)
    n = frac.shape[0]
    rows = []
    for i in range(n):
        d = frac - frac[i]
        d -= np.round(d)
        dist = np.linalg.norm(d @ box.T, axis=1)
        rows.append([j for j in np.nonzero(dist < r)[0] if j != i])
    m = max(len(rr) for rr in rows)
    idx = np.full((n, m), n, dtype=np.int32)
    for i, rr in enumerate(rows):
        idx[i, : len(rr)] = rr
    return jnp.asarray(idx)


def _reference_idx(frac, box, n_atoms: int, cutoff: float) -> tuple[jnp.ndarray, str]:
    if n_atoms <= ALL_PAIRS_MAX_ATOMS:
        return all_pairs_idx(n_atoms), "all_pairs"
    return _bruteforce_idx(frac, box, cutoff + 1.0), "numpy_bruteforce(cutoff+1.0)"


def vacuum_gap(system: AtomicSystem, upper_mask: np.ndarray) -> float:
    """Empty space (Å, along the normal) between the top surface and the bottom surface's image."""
    n, _ = _plane_normal_and_area(system.box)
    z = _real_positions(system.frac, system.box) @ n
    up = np.asarray(upper_mask, bool)
    if z[up].min() <= z[~up].max():
        raise ValueError("upper block is not above the lower block in unwrapped coordinates; centre the slab in the cell")
    width = perpendicular_widths(np.asarray(system.box))[2]
    return float(width - (z.max() - z.min()))


def switching_regions(model: EnergyModel) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Per-species-pair switching interval [R1, R2] (Å) of a model, if it has one."""
    p = model.params
    ne = len(model.elements)
    if isinstance(p, dict) and "R1" in p and "R2" in p:
        return np.asarray(p["R1"], float), np.asarray(p["R2"], float)
    if isinstance(p, dict) and "r_on" in p and "r_cut" in p:
        return np.full((ne, ne), float(p["r_on"])), np.full((ne, ne), float(p["r_cut"]))
    return None


@functools.lru_cache(maxsize=None)
def _batch_kernel(energy_fn):
    """jit(vmap(value_and_grad)) over a batch of fractional coordinates; box/species/idx/params traced."""
    return jax.jit(jax.vmap(jax.value_and_grad(energy_fn, argnums=0), in_axes=(0, None, None, None, None)))


@functools.lru_cache(maxsize=None)
def _uniaxial_hessian_kernel(energy_fn):
    """jit(d²E/dε²) for the uniaxial strain box -> (I + ε e_z e_z^T) box (energy_fn static, rest traced)."""

    def e_eps(eps, frac, box, species, idx, p):
        P = jnp.zeros((3, 3), dtype=box.dtype).at[2, 2].set(1.0)
        return energy_fn(frac, (jnp.eye(3, dtype=box.dtype) + eps * P) @ box, species, idx, p)

    return jax.jit(jax.hessian(e_eps, argnums=0))


class _Fns:
    """Energy / real-space gradient for a fixed box, species and params.

    Uses the model's shared compiled kernels (``potentials.base.compiled_kernels``, box and
    species as traced arguments) and one cached vmapped batch kernel per energy function, so
    building a new _Fns (one per curve/slab) never recompiles for an already-seen shape.
    ``jax.grad`` w.r.t. the fractional coordinates of a JAX-MD energy is the REAL-SPACE gradient
    dE/dR (custom JVP); see the module docstring.
    """

    def __init__(self, model: EnergyModel, box, species, params=None):
        from fracture_md.potentials.base import compiled_kernels

        p = model.params if params is None else params
        self.box = box
        k = compiled_kernels(model.energy_fn)
        ef, bk = k["energy_and_forces"], _batch_kernel(model.energy_fn)
        ek = k["energy"]
        box_j = jnp.asarray(box)
        self.e = lambda fr, idx: ek(fr, box_j, species, idx, p)

        def eg(fr, idx):
            e, f = ef(fr, box_j, species, idx, p)
            return e, -f

        self.eg = eg
        self.batch = lambda frs, idx: bk(frs, box_j, species, idx, p)


class _NeighborManager:
    """Neighbour-list policy for block displacements, with diagnostics."""

    def __init__(self, system: AtomicSystem, cutoff: float, skin: float, policy: str, capacity_multiplier: float = 1.5):
        if policy not in ("rebuild_each_step", "static_with_skin"):
            raise ValueError(f"unknown neighbour policy {policy!r}")
        self.system, self.cutoff, self.skin, self.policy = system, cutoff, skin, policy
        self.cm = capacity_multiplier
        self.handle: NeighborHandle = build_neighbors(system, cutoff, skin=skin, capacity_multiplier=capacity_multiplier)
        self.initial_idx = self.handle.idx
        self.n_builds = 1
        self.n_overflow_reallocations = 0
        self.log: list[dict[str, Any]] = []

    def get(self, frac: jnp.ndarray) -> jnp.ndarray:
        rebuilt = False
        overflow = False
        if self.policy == "rebuild_each_step":
            # fresh allocation (not an update) with the same neighbour function: the box is fixed,
            # so the minimum-image check made at construction still holds
            nbrs = self.handle.neighbor_fn.allocate(frac)  # type: ignore[attr-defined]
            self.handle = dataclasses.replace(self.handle, nbrs=nbrs)
            self.n_builds += 1
            rebuilt = True
            overflow = bool(self.handle.nbrs.did_buffer_overflow)  # type: ignore[attr-defined]
        else:
            from fracture_md.potentials.neighbors import _jitted_update

            old_ref = np.asarray(self.handle.nbrs.reference_position)  # type: ignore[attr-defined]
            # jitted update, cached per neighbour function (the eager nbrs.update retraced every call)
            new = _jitted_update(self.handle.neighbor_fn)(frac, self.handle.nbrs)
            if bool(new.did_buffer_overflow):
                overflow = True
                self.n_overflow_reallocations += 1
                new = self.handle.neighbor_fn.allocate(frac)  # type: ignore[attr-defined]
            rebuilt = not np.array_equal(np.asarray(new.reference_position), old_ref)
            if rebuilt:
                self.n_builds += 1
            self.handle = dataclasses.replace(self.handle, nbrs=new)
        idx = self.handle.idx
        self.log.append(dict(rebuilt=rebuilt, overflow_detected=overflow, capacity=int(idx.shape[1]),
                             occupancy=_valid_count(np.asarray(idx)),
                             overflow_after=bool(self.handle.nbrs.did_buffer_overflow)))  # type: ignore[attr-defined]
        return idx

    def capacity_ok(self) -> bool:
        return all((not e["overflow_after"]) and e["occupancy"] <= e["capacity"] for e in self.log)


def _check(name, passed, metric=None, tol=None, unit=None, details="") -> CheckResult:
    return CheckResult(name=name, passed=bool(passed),
                       metric=None if metric is None or not np.isfinite(metric) else float(metric),
                       tolerance=None if tol is None else float(tol), unit=unit, details=details)


# ============================================================================
# rigid block scans
# ============================================================================
@dataclasses.dataclass
class OpeningCurve:
    """Rigid opening (or sliding) curve with numerical-check evidence.

    Arrays are on the protocol grid ``openings`` unless prefixed ``fine_``.
    Energies per area are relative to the first point (δ = 0).
    """

    mode: str  # "normal" or "tangential"
    openings: np.ndarray  # Å
    energy_eV: np.ndarray  # total energy
    energy_per_area: np.ndarray  # J/m^2
    traction: np.ndarray  # GPa, autodiff
    traction_fd_central: np.ndarray  # GPa, central FD of E (real-space displacement)
    area: float  # Å^2
    direction: np.ndarray  # unit displacement direction of the upper block (lab)
    nl_minus_reference_eV: np.ndarray
    initial_list_minus_reference_eV: np.ndarray
    reference_list: str
    neighbor_log: list[dict[str, Any]]
    checks: list[CheckResult]
    fine_openings: Optional[np.ndarray] = None
    fine_energy_per_area: Optional[np.ndarray] = None
    fine_traction: Optional[np.ndarray] = None
    fine_traction_fd_sampled: Optional[np.ndarray] = None
    peak_traction: float = float("nan")  # GPa (refined)
    opening_at_peak: float = float("nan")  # Å (refined)
    work_of_separation: float = float("nan")  # J/m^2, Φ(δ_max)
    integrated_work: float = float("nan")  # J/m^2, ∫T dδ (fine grid if available)
    vacuum_gap: float = float("nan")
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif f.name == "checks":
                v = [c.model_dump(mode="json") for c in v]
            out[f.name] = v
        return out


def _block_frac(frac0: jnp.ndarray, upper: np.ndarray, box, disp_real: np.ndarray) -> jnp.ndarray:
    s = np.linalg.inv(np.asarray(box)) @ np.asarray(disp_real, float)  # fractional displacement
    return frac0 + jnp.asarray(upper[:, None] * s[None, :])


def _block_derivative(g: jnp.ndarray, upper: np.ndarray, direction: np.ndarray) -> float:
    """dE/dδ for a rigid block displacement: Σ_upper (∂E/∂R_i)·direction (real-space gradient)."""
    return float(np.sum(np.asarray(g)[upper] @ direction))


def rigid_block_scan(
    model: EnergyModel,
    slab_system: AtomicSystem,
    upper_mask: np.ndarray,
    displacements: Sequence[float],
    direction: np.ndarray,
    *,
    base_offset: Optional[np.ndarray] = None,
    neighbor_policy: str = "rebuild_each_step",
    skin: float = 0.5,
    fd_step: float = 1e-4,
    check_reference: bool = True,
    mode: str = "normal",
    capacity_multiplier: float = 1.5,
) -> OpeningCurve:
    """Rigidly translate the upper block by s·direction (+ base_offset) and record E(s), dE/ds.

    Used by :func:`rigid_opening_curve` (direction = plane normal) and by the
    tangential-slip protocol (direction in-plane, base_offset = normal opening).
    """
    upper = np.asarray(upper_mask, bool)
    s_vals = np.asarray(displacements, float)
    box = slab_system.box
    direction = np.asarray(direction, float) / np.linalg.norm(direction)
    base = np.zeros(3) if base_offset is None else np.asarray(base_offset, float)
    species = model.species_for(slab_system)
    fns = _Fns(model, box, species)
    _, area = _plane_normal_and_area(box)
    frac0 = jnp.asarray(slab_system.frac)
    frac_of = lambda s: _block_frac(frac0, upper, box, base + s * direction)  # noqa: E731

    sys0 = slab_system.replace(frac=frac_of(s_vals[0]))
    mgr = _NeighborManager(sys0, model.cutoff, skin, neighbor_policy, capacity_multiplier=capacity_multiplier)
    ref_name = None
    E, dE, dE_fd, d_ref, d_init = [], [], [], [], []
    for s in s_vals:
        fr = frac_of(s)
        idx = mgr.get(fr)
        e, g = fns.eg(fr, idx)
        E.append(float(e))
        dE.append(_block_derivative(g, upper, direction))
        ep = float(fns.e(frac_of(s + fd_step), idx))
        em = float(fns.e(frac_of(s - fd_step), idx))
        dE_fd.append((ep - em) / (2 * fd_step))
        d_init.append(float(fns.e(fr, mgr.initial_idx)) - float(e))
        if check_reference:
            ridx, ref_name = _reference_idx(fr, box, slab_system.n_atoms, model.cutoff)
            d_ref.append(float(fns.e(fr, ridx)) - float(e))
    E = np.asarray(E)
    to_gpa = EV_PER_A3_TO_GPA / area
    curve = OpeningCurve(
        mode=mode,
        openings=s_vals,
        energy_eV=E,
        energy_per_area=(E - E[0]) / area * EV_PER_A2_TO_J_PER_M2,
        traction=np.asarray(dE) * to_gpa,
        traction_fd_central=np.asarray(dE_fd) * to_gpa,
        area=float(area),
        direction=direction,
        nl_minus_reference_eV=np.asarray(d_ref) if d_ref else np.full(len(s_vals), np.nan),
        initial_list_minus_reference_eV=np.asarray(d_init),
        reference_list=ref_name or "not_checked",
        neighbor_log=mgr.log,
        checks=[],
    )
    # ---- numerical checks common to all scans
    ck = curve.checks
    if check_reference:
        m = float(np.max(np.abs(curve.nl_minus_reference_eV)))
        ck.append(_check("neighbor_list_equals_reference_energy_every_step", m <= 1e-8, m, 1e-8, Unit.EV,
                         f"policy={neighbor_policy}; reference={ref_name}; checked at {len(s_vals)} displacements"))
    ck.append(_check("neighbor_capacity_no_overflow", mgr.capacity_ok(), float(mgr.n_overflow_reallocations), 0.0,
                     Unit.DIMENSIONLESS,
                     f"{mgr.n_builds} list builds, {mgr.n_overflow_reallocations} overflow reallocations; "
                     f"capacity {mgr.log[-1]['capacity']}, max occupancy {max(e['occupancy'] for e in mgr.log)}"))
    diff = np.abs(curve.traction - curve.traction_fd_central)
    tol = 1e-5 + 1e-6 * np.max(np.abs(curve.traction))
    ck.append(_check("traction_autodiff_equals_central_fd", float(diff.max()) <= tol, float(diff.max()), tol, Unit.GPA,
                     f"central FD with real-space block displacement h={fd_step} Å, same neighbour list"))
    curve.raw.update(neighbor_policy=neighbor_policy, n_list_builds=mgr.n_builds,
                     n_overflow_reallocations=mgr.n_overflow_reallocations, fd_step=fd_step)
    return curve


def _fine_scan(fns: _Fns, frac0, box, idx, upper, direction, grid, batch: int = 32):
    """E(s) and dE/ds on `grid` for the rigid block displacement s·direction (batched, fixed chunk shape).

    The last chunk is padded (repeating the last grid point) so every call has the same shape and
    the batch kernel compiles once; the stacked fractional coordinates are built vectorised.
    """
    grid = np.asarray(grid, float)
    n_g = len(grid)
    batch = max(1, min(batch, n_g))
    pad = (-n_g) % batch
    g_all = np.concatenate([grid, np.full(pad, grid[-1])])
    dfr = jnp.asarray(np.asarray(upper, bool)[:, None] * (np.linalg.inv(np.asarray(box)) @ np.asarray(direction, float))[None, :])
    fr0 = jnp.asarray(frac0)
    sel = jnp.asarray(np.asarray(upper, bool))
    dvec = jnp.asarray(np.asarray(direction, float))
    E, D = [], []
    for k in range(0, len(g_all), batch):
        sv = jnp.asarray(g_all[k : k + batch])
        frs = fr0[None, :, :] + sv[:, None, None] * dfr[None, :, :]
        e, g = fns.batch(frs, idx)
        E.append(np.asarray(e))
        D.append(np.asarray(jnp.sum(jnp.where(sel[None, :], g @ dvec, 0.0), axis=1)))
    return np.concatenate(E)[:n_g], np.concatenate(D)[:n_g]


def rigid_opening_curve(
    model: EnergyModel,
    slab_system: AtomicSystem,
    upper_mask: np.ndarray,
    openings: Sequence[float],
    neighbor_policy: str = "rebuild_each_step",
    *,
    skin: float = 0.5,
    fd_step: float = 1e-4,
    fine_step: Optional[float] = 0.005,
    check_reference: bool = True,
    integration_rtol: float = 1e-3,
) -> OpeningCurve:
    """Rigid normal opening of the upper block: E(δ), Φ(δ) (J/m²), T(δ) (GPa).

    T is computed by autodiff (real-space gradient projected on the normal,
    summed over the upper block) and by central finite differences of E with
    real-space displacements.  If ``fine_step`` is given, E and T are also
    evaluated on a fine grid (with the δ = 0 neighbour list, a verified
    superset for normal opening) to (i) check ∫T dδ = Φ(δ_max) by trapezoid
    integration of the autodiff tractions, (ii) compare T with finite
    differences of the *sampled* Φ, and (iii) locate the peak traction, which
    is then refined with Brent's method on the autodiff traction.

    Raises ValueError if the vacuum is too small for the largest opening.
    """
    openings = np.asarray(openings, float)
    if openings[0] != 0.0 or np.any(np.diff(openings) <= 0):
        raise ValueError("openings must start at 0 and increase strictly")
    n, area = _plane_normal_and_area(slab_system.box)
    upper = np.asarray(upper_mask, bool)
    vg = vacuum_gap(slab_system, upper)
    need = model.cutoff + skin
    eff = vg - openings[-1]
    if eff <= need:
        raise ValueError(f"vacuum gap {vg:.3f} Å minus δ_max {openings[-1]:.3f} Å = {eff:.3f} Å <= cutoff+skin {need:.3f} Å")
    curve = rigid_block_scan(model, slab_system, upper, openings, n, neighbor_policy=neighbor_policy, skin=skin,
                             fd_step=fd_step, check_reference=check_reference, mode="normal")
    curve.vacuum_gap = vg
    curve.checks.insert(0, _check("vacuum_gap_exceeds_interaction_range_at_max_opening", eff > need, eff, need, Unit.ANGSTROM,
                                  f"vacuum gap {vg:.3f} Å - δ_max {openings[-1]:.3f} Å > cutoff+skin {need:.3f} Å"))
    m0 = float(np.max(np.abs(curve.initial_list_minus_reference_eV)))
    curve.checks.append(_check("initial_list_valid_during_normal_opening", m0 <= 1e-8, m0, 1e-8, Unit.EV,
                               "the δ=0 list (radius cutoff+skin) is a superset of interacting pairs for normal "
                               "opening because cross-gap distances only increase; energy with it vs the managed list"))
    curve.work_of_separation = float(curve.energy_per_area[-1])
    curve.integrated_work = float(np.trapezoid(curve.traction, openings) / EV_PER_A3_TO_GPA * EV_PER_A2_TO_J_PER_M2)
    curve.raw["protocol_grid_trapezoid_J_m2"] = curve.integrated_work

    species = model.species_for(slab_system)
    fns = _Fns(model, slab_system.box, species)
    frac0 = jnp.asarray(slab_system.frac)
    box = slab_system.box
    frac_of = lambda s: _block_frac(frac0, upper, box, s * n)  # noqa: E731
    idx0 = build_neighbors(slab_system, model.cutoff, skin=skin).idx

    def T_of(s):
        _, g = fns.eg(frac_of(s), idx0)
        return _block_derivative(g, upper, n) / area * EV_PER_A3_TO_GPA

    if fine_step:
        nf = int(math.ceil(openings[-1] / fine_step)) + 1
        grid = np.linspace(0.0, openings[-1], nf)
        E_f, D_f = _fine_scan(fns, frac0, box, idx0, upper, n, grid)
        phi = (E_f - E_f[0]) / area * EV_PER_A2_TO_J_PER_M2
        T_f = D_f / area * EV_PER_A3_TO_GPA
        T_fd = np.gradient(E_f, grid, edge_order=2) / area * EV_PER_A3_TO_GPA
        curve.fine_openings, curve.fine_energy_per_area, curve.fine_traction = grid, phi, T_f
        curve.fine_traction_fd_sampled = T_fd
        integ = float(np.trapezoid(T_f, grid) / EV_PER_A3_TO_GPA * EV_PER_A2_TO_J_PER_M2)
        curve.integrated_work = integ
        rel = abs(integ - phi[-1]) / max(abs(phi[-1]), 1e-12)
        curve.checks.append(_check("integral_of_traction_equals_energy_change", rel <= integration_rtol, rel,
                                   integration_rtol, Unit.DIMENSIONLESS,
                                   f"trapezoid ∫T_autodiff dδ on {nf}-point grid (h={grid[1]:.4g} Å) = {integ:.8f} J/m^2 "
                                   f"vs Φ(δ_max) = {phi[-1]:.8f} J/m^2"))
        dfd = float(np.max(np.abs(T_fd - T_f)))
        tol_fd = 0.02 * float(np.max(np.abs(T_f)))
        curve.checks.append(_check("traction_autodiff_equals_fd_of_sampled_energy", dfd <= tol_fd, dfd, tol_fd, Unit.GPA,
                                   f"np.gradient (2nd order) of sampled E on the h={grid[1]:.4g} Å grid; residual "
                                   "dominated by the C1 (not C2) switching function kinks"))
        k = int(np.argmax(T_f))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, nf - 1)]
    else:
        k = int(np.argmax(curve.traction))
        lo, hi = openings[max(k - 1, 0)], openings[min(k + 1, len(openings) - 1)]
    if hi > lo:
        r = minimize_scalar(lambda s: -T_of(s), bounds=(lo, hi), method="bounded", options=dict(xatol=1e-6))
        curve.opening_at_peak, curve.peak_traction = float(r.x), float(-r.fun)
    else:  # pragma: no cover
        curve.opening_at_peak, curve.peak_traction = float(lo), T_of(lo)
    grid_peak = float(np.max(curve.traction))
    if grid_peak > curve.peak_traction:  # never report less than a sampled value
        kk = int(np.argmax(curve.traction))
        curve.opening_at_peak, curve.peak_traction = float(openings[kk]), grid_peak
    return curve


# ============================================================================
# UBER fit
# ============================================================================
def uber_energy(delta, W, l):
    x = np.asarray(delta) / l
    return W * (1.0 - (1.0 + x) * np.exp(-x))


def fit_uber(openings: Sequence[float], energy_per_area: Sequence[float], *, fit_range: Optional[tuple[float, float]] = None) -> dict[str, Any]:
    """Rose–Ferrante–Smith UBER fit Φ(δ) = W [1 − (1 + δ/l) e^{−δ/l}] (Rose, Smith, Ferrante, PRB 28, 1835 (1983)).

    `energy_per_area` in J/m² relative to δ = 0.  Returns W (J/m²), l (Å),
    σ_max = W/(l e) (GPa; the UBER peak traction, at δ = l), rms residual
    (J/m²), rms/W, max residual, R², and the fit covariance-based standard
    errors.  For short-cutoff potentials the curve reaches W at a finite δ
    (all bonds switched off), which UBER cannot represent; the residuals say
    how badly.
    """
    d = np.asarray(openings, float)
    e = np.asarray(energy_per_area, float) - float(energy_per_area[0])
    if fit_range is not None:
        sel = (d >= fit_range[0]) & (d <= fit_range[1])
        d, e = d[sel], e[sel]
    W0 = float(e[-1]) if e[-1] > 0 else float(np.max(e))
    slope = np.gradient(e, d)
    l0 = float(d[int(np.argmax(slope))]) or 0.3
    try:
        popt, pcov = curve_fit(uber_energy, d, e, p0=(W0, max(l0, 0.05)), bounds=([0.0, 1e-3], [np.inf, 10.0]), maxfev=20000)
        ok, msg = True, "converged"
    except (RuntimeError, ValueError) as err:  # pragma: no cover
        popt, pcov, ok, msg = np.array([W0, l0]), np.full((2, 2), np.nan), False, str(err)
    W, l = float(popt[0]), float(popt[1])
    res = uber_energy(d, W, l) - e
    rms = float(np.sqrt(np.mean(res**2)))
    ss = float(np.sum((e - e.mean()) ** 2))
    sigma_max = W / EV_PER_A2_TO_J_PER_M2 / (l * math.e) * EV_PER_A3_TO_GPA
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None)) if np.all(np.isfinite(pcov)) else np.full(2, np.nan)
    return dict(W_J_m2=W, l_A=l, sigma_max_GPa=float(sigma_max), rms_residual_J_m2=rms,
                relative_rms=rms / W if W > 0 else float("inf"), max_abs_residual_J_m2=float(np.max(np.abs(res))),
                r_squared=1.0 - float(np.sum(res**2)) / ss if ss > 0 else float("nan"),
                W_stderr_J_m2=float(perr[0]), l_stderr_A=float(perr[1]), success=ok, message=msg,
                n_points=int(len(d)), fit_range=[float(d[0]), float(d[-1])])


# ============================================================================
# cutoff-function artefact
# ============================================================================
def cross_gap_pairs(system: AtomicSystem, upper_mask: np.ndarray, rmax: float) -> list[tuple[int, int, float]]:
    """(i_lower, j_upper, distance) for all minimum-image pairs across the cut with r < rmax."""
    up = np.asarray(upper_mask, bool)
    frac = np.asarray(system.frac)
    box = np.asarray(system.box)
    lo_i, up_i = np.nonzero(~up)[0], np.nonzero(up)[0]
    d = frac[up_i][None, :, :] - frac[lo_i][:, None, :]
    d -= np.round(d)
    r = np.linalg.norm(d @ box.T, axis=-1)
    a, b = np.nonzero(r < rmax)
    return [(int(lo_i[x]), int(up_i[y]), float(r[x, y])) for x, y in zip(a, b)]


def cutoff_artifact_check(model: EnergyModel, system: AtomicSystem, upper_mask: np.ndarray, *,
                          opening: Optional[float] = None, label: str = "peak traction") -> dict[str, Any]:
    """Is any cross-gap pair inside the potential's switching interval [R1, R2]?

    `system` is the configuration at the opening of peak traction (or pass the
    δ = 0 slab and `opening` to shift the upper block rigidly).  Returns a dict
    with ``flagged`` (bool), ``n_pairs_in_switching_region``, counts of fully
    bonded (r ≤ R1) pairs, the distances, and a human-readable ``details``.
    """
    upper = np.asarray(upper_mask, bool)
    if opening is not None:
        n, _ = _plane_normal_and_area(system.box)
        system = system.replace(frac=_block_frac(jnp.asarray(system.frac), upper, system.box, opening * n))
    sw = switching_regions(model)
    if sw is None:
        return dict(flagged=False, applicable=False, n_pairs_in_switching_region=0, n_bonded_pairs=0,
                    details="model has no tabulated switching interval; check not applicable", distances_in_switch=[])
    R1, R2 = sw
    sp = np.asarray(model.species_for(system))
    pairs = cross_gap_pairs(system, upper, float(np.max(R2)) + 1e-9)
    in_sw, bonded = [], []
    for i, j, r in pairs:
        r1, r2 = R1[sp[i], sp[j]], R2[sp[i], sp[j]]
        if r1 < r < r2:
            in_sw.append((model.elements[sp[i]], model.elements[sp[j]], r, float(r1), float(r2)))
        elif r <= r1:
            bonded.append(r)
    flagged = len(in_sw) > 0
    if flagged:
        rs = [x[2] for x in in_sw]
        details = (f"peak traction set by the cutoff function: at the {label} opening, {len(in_sw)} cross-gap pairs "
                   f"(of {len(pairs)} within max R2) lie inside the switching interval (r = {min(rs):.3f}-{max(rs):.3f} Å; "
                   f"e.g. {in_sw[0][0]}-{in_sw[0][1]} R1={in_sw[0][3]:.3f}, R2={in_sw[0][4]:.3f} Å)")
    else:
        details = f"no cross-gap pair inside a switching interval at the {label} opening ({len(bonded)} fully bonded pairs)"
    return dict(flagged=flagged, applicable=True, n_pairs_in_switching_region=len(in_sw), n_bonded_pairs=len(bonded),
                n_pairs_considered=len(pairs), distances_in_switch=[x[2] for x in in_sw], details=details)


def _relax(model, system, *, fmax, newton_polish=False, **kw):
    """relax_positions, retried from its result with the Newton polish if L-BFGS stalls above fmax."""
    from fracture_md.physics.relax import relax_positions

    s_r, info = relax_positions(model, system, fmax=fmax, newton_polish=newton_polish, **kw)
    if not info.converged:
        s_r, info2 = relax_positions(model, s_r, fmax=fmax, newton_polish=True, **kw)
        info2.n_steps += info.n_steps
        info = info2
    return s_r, info


# ============================================================================
# uniaxial-strain modulus normal to the plane
# ============================================================================
def uniaxial_strain_modulus(model: EnergyModel, oriented_cell, *, strains=(1e-3, 2e-3), relax: bool = True,
                            fmax: float = 1e-7, skin: float = 0.5) -> dict[str, Any]:
    """C'33 = (1/V) d²E/dε² for uniaxial strain ε along the plane normal (lab z), lateral lattice fixed.

    `oriented_cell` is the periodic cell from ``oriented_bulk_cell`` (normal
    ∥ +z); it is replicated to satisfy the minimum-image condition.  The
    strain is applied as box -> (I + ε e_z e_zᵀ) box.  ``clamped`` is the exact
    autodiff second derivative at fixed fractional coordinates; ``relaxed``
    uses central differences of the energy with internal coordinates relaxed
    at each strain (Richardson-extrapolated from the two strain magnitudes).
    For cubic crystals C'33(100) = C11, C'33(110) = (C11+C12+2C44)/2,
    C'33(111) = (C11+2C12+4C44)/3.
    """
    reps = auto_supercell(oriented_cell, model.cutoff + skin)
    atoms = oriented_cell.repeat(reps)
    s0 = AtomicSystem.from_ase(atoms, elements=model.elements)
    sp = model.species_for(s0)
    nh = build_neighbors(s0, model.cutoff, skin=skin)
    P = jnp.asarray(np.diag([0.0, 0.0, 1.0]))
    V = s0.volume
    hess = _uniaxial_hessian_kernel(model.energy_fn)  # jitted, cached per energy function
    clamped = float(hess(jnp.asarray(0.0, dtype=jnp.asarray(s0.box).dtype), s0.frac, s0.box, sp, nh.idx,
                         model.params)) / V * EV_PER_A3_TO_GPA

    from fracture_md.physics.relax import relax_positions

    def e_rel(eps):
        s = s0.replace(box=(jnp.eye(3) + eps * P) @ s0.box)
        if not relax:
            return float(model.energy(s.frac, s.box, sp, nh.idx))
        s_r, info = relax_positions(model, s, fmax=fmax, skin=skin)
        return info.energy

    e0 = e_rel(0.0)
    c = []
    for h in strains:
        c.append((e_rel(h) - 2 * e0 + e_rel(-h)) / h**2 / V * EV_PER_A3_TO_GPA)
    if len(c) >= 2:
        h1, h2 = strains[0], strains[1]
        relaxed = (h2**2 * c[0] - h1**2 * c[1]) / (h2**2 - h1**2)
    else:
        relaxed = c[0]
    return dict(relaxed_GPa=float(relaxed), clamped_GPa=clamped, fd_values_GPa=[float(x) for x in c],
                strains=list(strains), n_atoms=len(atoms), supercell=list(reps))


# ============================================================================
# relaxed (displacement-controlled) opening
# ============================================================================
@dataclasses.dataclass
class RelaxedOpeningCurve:
    openings: np.ndarray  # grip displacement δ (Å)
    energy_eV: np.ndarray
    energy_per_area: np.ndarray  # J/m^2 relative to δ=0 relaxed state
    traction: np.ndarray  # GPa, grip reaction  −(1/A) Σ_top F_z
    traction_fd_sampled: np.ndarray  # GPa, np.gradient of relaxed E on the protocol grid
    cohesive_opening: Optional[np.ndarray]  # δ_coh = δ − T h_el / C'33 (Å) or None
    area: float
    grip_layers: int
    h_grips: float  # Å, inner separation of the grips at δ=0
    h_elastic: Optional[float]
    c33_prime_GPa: Optional[float]
    converged: np.ndarray
    fmax: np.ndarray
    checks: list[CheckResult]
    peak_traction: float = float("nan")
    opening_at_peak: float = float("nan")
    cohesive_opening_at_peak: float = float("nan")
    instability: Optional[dict[str, Any]] = None
    final_frac: Optional[np.ndarray] = None
    peak_frac: Optional[np.ndarray] = None
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name in ("final_frac", "peak_frac"):
                continue
            if isinstance(v, np.ndarray):
                v = v.tolist()
            elif f.name == "checks":
                v = [c.model_dump(mode="json") for c in v]
            out[f.name] = v
        return out


def layer_indices(system: AtomicSystem, tol: float = 0.02) -> np.ndarray:
    """Atomic-layer index per atom along the slab normal (0 = bottom)."""
    from fracture_md.structures.slabs import _cluster_heights

    n, _ = _plane_normal_and_area(system.box)
    z = _real_positions(system.frac, system.box) @ n
    _, lab = _cluster_heights(z, None, tol)
    return lab


def receptive_field(model: EnergyModel) -> float:
    """Distance (Å) beyond which an atom's position cannot change another atom's force.

    Uses ``model.receptive_field`` if the model defines it; otherwise the cutoff for analytic
    pair/bond-order potentials and 2 × cutoff for machine-learned (message-passing, e.g. MACE with
    2 interaction layers) cards — a conservative default for the grip thickness.
    """
    rf = getattr(model, "receptive_field", None)
    if rf is not None:
        return float(rf)
    from fracture_md.schema.potential import PotentialKind

    card = getattr(model, "card", None)
    if card is not None and card.kind == PotentialKind.MACHINE_LEARNED:
        return 2.0 * float(model.cutoff)
    return float(model.cutoff)


def relaxed_opening_curve(
    model: EnergyModel,
    slab_system: AtomicSystem,
    upper_mask: np.ndarray,
    openings: Sequence[float],
    grip_layers: int = 2,
    *,
    fmax: float = 1e-4,
    max_steps: int = 20000,
    skin: float = 0.5,
    layer_index: Optional[np.ndarray] = None,
    c33_prime_GPa: Optional[float] = None,
    gap_width: float = 0.0,
    envelope_points: int = 2,
    envelope_step: float = 2e-3,
    newton_polish: bool = False,
    refine_instability: Optional[float] = 0.01,
    min_grip_thickness: Optional[float] = None,
) -> RelaxedOpeningCurve:
    """Displacement-controlled relaxed opening with fixed bottom grip and rigid top grip.

    See the module docstring.  The in-plane lattice is held at the bulk value.
    The envelope-theorem check compares the grip reaction with a central
    difference of the *relaxed* energy (re-relaxed at δ ± h) at up to
    ``envelope_points`` pre-peak openings; after an elastic instability the
    sampled relaxed E(δ) has a slope discontinuity and is not differentiable.
    If the reaction drops by more than 30 % of the peak between two samples,
    the instability is bracketed by bisection (continuation from the last
    stable state) to within ``refine_instability`` Å; the extra points are
    merged into the returned grid (so ``openings`` may be longer than the
    input).  ``peak_traction`` is the maximum on the stable branch.

    ``grip_layers`` is a minimum: each grip is enlarged until the distance from
    its outer surface layer to the first free layer is at least
    ``min_grip_thickness`` (default: :func:`receptive_field` — the cutoff for
    pair/bond-order potentials, 2 × cutoff for message-passing MLIPs).  Thinner grips let
    the outer-surface forces leak into the reaction (T(0) ≠ 0), because pairs
    between free atoms and the missing crystal beyond the surface would cross
    the grip.  A check verifies T(δ=0) ≈ 0.
    """
    from fracture_md.physics.relax import relax_positions

    openings = np.asarray(openings, float)
    if openings[0] != 0.0 or np.any(np.diff(openings) <= 0):
        raise ValueError("openings must start at 0 and increase strictly")
    upper = np.asarray(upper_mask, bool)
    n, area = _plane_normal_and_area(slab_system.box)
    vg = vacuum_gap(slab_system, upper)
    if vg - openings[-1] <= model.cutoff + skin:
        raise ValueError("vacuum too small for the largest opening")
    lab = layer_indices(slab_system) if layer_index is None else np.asarray(layer_index)
    nl = int(lab.max()) + 1
    zl = np.array([np.mean((_real_positions(slab_system.frac, slab_system.box) @ n)[lab == k]) for k in range(nl)])
    tmin = receptive_field(model) if min_grip_thickness is None else float(min_grip_thickness)
    requested_grip = grip_layers
    g_bot = next((k for k in range(grip_layers, nl) if zl[k] - zl[0] >= tmin - 1e-9), nl)
    g_top = next((k for k in range(grip_layers, nl) if zl[nl - 1] - zl[nl - 1 - k] >= tmin - 1e-9), nl)
    grip_layers = max(g_bot, g_top)
    bottom = lab < grip_layers
    top = lab >= nl - grip_layers
    if np.any(top & ~upper) or np.any(bottom & upper):
        raise ValueError("grip layers straddle the cut; use fewer grip layers or a thicker slab")
    if np.all(upper[~(top | bottom)]) or not np.any(~upper & ~bottom):
        raise ValueError("no free atoms on one side of the cut")
    fixed = bottom | top
    box = slab_system.box
    binv = np.linalg.inv(np.asarray(box))
    x0 = _real_positions(slab_system.frac, box)
    z0 = x0 @ n
    h_grips = float(z0[top].min() - z0[bottom].max())
    species = model.species_for(slab_system)
    fns = _Fns(model, box, species)

    def relax_from(x_guess, x_top):
        xg = x_guess.copy()
        xg[top] = x_top
        xg[bottom] = x0[bottom]
        s = slab_system.replace(frac=jnp.asarray(xg @ binv.T))
        s_r, info = _relax(model, s, fmax=fmax, max_steps=max_steps, fixed=fixed, skin=skin,
                           newton_polish=newton_polish)
        x = _real_positions(s_r.frac, box)
        # undo in-plane wrapping relative to the guess so continuation stays unwrapped
        dfr = (x - xg) @ binv.T
        dfr -= np.round(dfr)
        x = xg + dfr @ np.asarray(box).T
        nh = build_neighbors(s_r, model.cutoff, skin=skin)
        e, g = fns.eg(s_r.frac, nh.idx)
        react = float(np.sum(np.asarray(g)[top] @ n))
        return x, float(e), react, info

    to_gpa = EV_PER_A3_TO_GPA / area
    pts: dict[float, dict[str, Any]] = {}

    def solve(d, x_start, d_start):
        guess = x_start + (d - d_start) * upper[:, None] * n[None, :]  # seed the extra opening at the cut plane
        x, e, react, info = relax_from(guess, x0[top] + d * n)
        pts[float(d)] = dict(x=x, E=e, T=react * to_gpa, conv=info.converged, fmax=info.fmax)
        return pts[float(d)]

    x_prev, d_prev = x0.copy(), 0.0
    for d in openings:
        r = solve(d, x_prev, d_prev)
        x_prev, d_prev = r["x"], float(d)

    def arrays():
        ds = np.array(sorted(pts))
        return ds, np.array([pts[d]["E"] for d in ds]), np.array([pts[d]["T"] for d in ds])

    # ---- instability (snap-back) detection: a drop of the reaction by > 30% of the peak between samples,
    # then bisection with continuation from the last stable state to localise it within `refine_instability`.
    ds, E, T = arrays()
    instab = None
    drops = np.nonzero(np.diff(T) < -0.3 * max(float(np.max(T)), 1e-12))[0]
    if len(drops):
        k = int(drops[0])
        d_lo, d_hi = float(ds[k]), float(ds[k + 1])
        T_ref = float(np.max(T[: k + 1]))
        n_bis = 0
        while refine_instability and d_hi - d_lo > refine_instability and n_bis < 12:
            mid = 0.5 * (d_lo + d_hi)
            r = solve(mid, pts[d_lo]["x"], d_lo)
            n_bis += 1
            if pts[d_lo]["T"] - r["T"] < 0.3 * T_ref:
                d_lo = mid
                T_ref = max(T_ref, r["T"])
            else:
                d_hi = mid
        released = (pts[d_lo]["E"] - pts[d_hi]["E"]) / area * EV_PER_A2_TO_J_PER_M2
        instab = dict(between_openings_A=[d_lo, d_hi], traction_before_GPa=float(pts[d_lo]["T"]),
                      traction_after_GPa=float(pts[d_hi]["T"]), n_bisection_steps=n_bis,
                      energy_jump_J_m2=float(released),
                      note="elastic snap-back under grip control: the elastic energy stored between the grips is "
                           "released; the softening branch is not sampled and post-instability points depend on slab "
                           "thickness")
    ds, E, T = arrays()
    conv = np.array([pts[d]["conv"] for d in ds])
    fm = np.array([pts[d]["fmax"] for d in ds])
    phi = (E - E[0]) / area * EV_PER_A2_TO_J_PER_M2
    T_fd = np.gradient(E, ds, edge_order=2 if len(ds) > 2 else 1) * to_gpa
    last_smooth = instab["between_openings_A"][0] if instab else float(ds[int(np.argmax(T))])
    smooth = ds <= last_smooth + 1e-12
    kp = int(np.argmax(np.where(smooth, T, -np.inf)))
    t0_tol = 1e-3 + 10 * fmax * to_gpa
    checks = [
        _check("relaxations_converged", bool(np.all(conv)), float(np.max(fm)), fmax, Unit.EV_PER_ANGSTROM,
               f"{int(conv.sum())}/{len(conv)} openings converged to fmax={fmax} eV/Å (free atoms)"),
        _check("grip_reaction_zero_at_zero_opening", abs(T[0]) <= t0_tol, abs(float(T[0])), t0_tol, Unit.GPA,
               f"grips of {grip_layers} layers (requested {requested_grip}; min thickness {tmin:.2f} Å); a nonzero "
               "value means surface forces leak into the grips or the bulk reference is not stress free"),
    ]
    # ---- envelope theorem: reaction == d E_relaxed / dδ (central difference with re-relaxation)
    env = []
    cand = [k for k in range(1, len(ds)) if smooth[k]]
    pick = sorted(set([cand[0], cand[-1]])) if cand else []
    for k in pick[:envelope_points]:
        d = float(ds[k])
        xk = pts[d]["x"]
        xp, ep, _, _ = relax_from(xk + envelope_step * upper[:, None] * n, x0[top] + (d + envelope_step) * n)
        xm, em, _, _ = relax_from(xk - envelope_step * upper[:, None] * n, x0[top] + (d - envelope_step) * n)
        fd = (ep - em) / (2 * envelope_step) * to_gpa
        env.append(dict(opening=d, reaction_GPa=float(T[k]), fd_GPa=float(fd)))
    if env:
        err = max(abs(v["reaction_GPa"] - v["fd_GPa"]) for v in env)
        tol = 1e-3 * max(abs(T[kp]), 1.0) + 50 * fmax * to_gpa
        checks.append(_check("grip_reaction_equals_derivative_of_relaxed_energy", err <= tol, err, tol, Unit.GPA,
                             "envelope theorem at pre-instability openings "
                             + ", ".join(f"δ={v['opening']:.3f} Å: {v['reaction_GPa']:.5f} vs {v['fd_GPa']:.5f} GPa" for v in env)))
    dev = float(np.max(np.abs(T_fd[smooth][1:-1] - T[smooth][1:-1]))) if np.sum(smooth) > 3 else float("nan")
    coh = None
    h_el = None
    if c33_prime_GPa is not None:
        h_el = h_grips - gap_width
        coh = ds - T * h_el / c33_prime_GPa
    # ∫T dδ over continuous segments only (the snap-back interval is excluded)
    seg = np.ones(len(ds) - 1, bool)
    if instab:
        seg &= ~((ds[:-1] >= instab["between_openings_A"][0] - 1e-12) & (ds[1:] <= instab["between_openings_A"][1] + 1e-12))
    integ = float(np.sum(0.5 * (T[1:] + T[:-1]) * np.diff(ds) * seg) / EV_PER_A3_TO_GPA * EV_PER_A2_TO_J_PER_M2)
    integ_coh = (float(np.sum(0.5 * (T[1:] + T[:-1]) * np.diff(coh) * seg) / EV_PER_A3_TO_GPA * EV_PER_A2_TO_J_PER_M2)
                 if coh is not None else float("nan"))
    out = RelaxedOpeningCurve(
        openings=ds, energy_eV=E, energy_per_area=phi, traction=T, traction_fd_sampled=T_fd,
        cohesive_opening=coh, area=float(area), grip_layers=grip_layers, h_grips=h_grips, h_elastic=h_el,
        c33_prime_GPa=c33_prime_GPa, converged=conv, fmax=fm, checks=checks,
        peak_traction=float(T[kp]), opening_at_peak=float(ds[kp]),
        cohesive_opening_at_peak=float(coh[kp]) if coh is not None else float("nan"),
        instability=instab,
        final_frac=np.asarray(pts[float(ds[-1])]["x"] @ binv.T), peak_frac=np.asarray(pts[float(ds[kp])]["x"] @ binv.T),
    )
    out.raw.update(envelope=env, sampled_fd_max_dev_pre_instability_GPa=dev, n_free=int(np.sum(~fixed)),
                   n_fixed_bottom=int(bottom.sum()), n_grip_top=int(top.sum()), vacuum_gap=vg,
                   integrated_work_continuous_segments_J_m2=integ, integrated_work_cohesive_J_m2=integ_coh,
                   protocol_openings_A=openings.tolist())
    return out


def relaxed_separation_energy(model: EnergyModel, slab_system: AtomicSystem, upper_mask: np.ndarray, separation: float,
                              *, fmax: float = 1e-4, skin: float = 0.5, max_steps: int = 20000) -> dict[str, Any]:
    """Relaxed work of separation: [E_relaxed(two separated slabs) − E_relaxed(intact slab)]/A.

    Both states are relaxed with no fixed atoms (the outer surfaces relax
    identically in both and cancel for thick slabs); the separated state starts
    from a rigid opening ``separation`` (> cutoff) so both new surfaces relax
    independently.  Only local relaxation from bulk-terminated positions is
    performed (no reconstruction search).
    """
    from fracture_md.physics.relax import relax_positions

    upper = np.asarray(upper_mask, bool)
    n, area = _plane_normal_and_area(slab_system.box)
    vg = vacuum_gap(slab_system, upper)
    if vg - separation <= model.cutoff + skin:
        raise ValueError("vacuum too small for the separation used for relaxed surfaces")
    s0, i0 = _relax(model, slab_system, fmax=fmax, skin=skin, max_steps=max_steps)
    sep_sys = slab_system.replace(frac=_block_frac(jnp.asarray(slab_system.frac), upper, slab_system.box, separation * n))
    s1, i1 = _relax(model, sep_sys, fmax=fmax, skin=skin, max_steps=max_steps)
    # the relaxed halves must still be separated beyond the cutoff
    pairs = cross_gap_pairs(s1, upper, model.cutoff)
    W = (i1.energy - i0.energy) / area * EV_PER_A2_TO_J_PER_M2
    # unrelaxed reference at the same separation
    sp = model.species_for(slab_system)
    e_un0 = float(model.energy(slab_system.frac, slab_system.box, sp, build_neighbors(slab_system, model.cutoff, skin).idx))
    e_un1 = float(model.energy(sep_sys.frac, sep_sys.box, sp, build_neighbors(sep_sys, model.cutoff, skin).idx))
    W_un = (e_un1 - e_un0) / area * EV_PER_A2_TO_J_PER_M2
    x0 = _real_positions(s0.frac, s0.box)
    xi = _real_positions(slab_system.frac, slab_system.box)
    d0 = xi - x0
    d0 -= np.round(d0 @ np.linalg.inv(np.asarray(s0.box)).T) @ np.asarray(s0.box).T
    checks = [
        _check("relaxed_states_converged", i0.converged and i1.converged, max(i0.fmax, i1.fmax), fmax, Unit.EV_PER_ANGSTROM),
        _check("relaxed_halves_remain_separated", len(pairs) == 0, float(len(pairs)), 0.0, Unit.DIMENSIONLESS,
               f"cross-gap pairs within cutoff after relaxation: {len(pairs)}"),
    ]
    return dict(W_relaxed_J_m2=float(W), W_unrelaxed_J_m2=float(W_un), E_intact_relaxed_eV=i0.energy,
                E_separated_relaxed_eV=i1.energy, relaxation_energy_J_m2=float(W_un - W), separation_A=float(separation),
                max_displacement_intact_A=float(np.max(np.linalg.norm(d0, axis=1))), checks=checks,
                separated_system=s1, intact_system=s0)


# ============================================================================
# protocol runners
# ============================================================================
def _source_hash(obj) -> str:
    try:
        from fracture_md.provenance import source_sha256

        return source_sha256(obj)
    except (OSError, TypeError):  # pragma: no cover
        return "unknown"


def relax_bulk_lattice(model: EnergyModel, atoms, *, skin: float = 0.5, fmax: float = 1e-6) -> tuple[Any, dict[str, Any]]:
    """Isotropically relax the lattice of a bulk crystal (exact for cubic phases) plus internal positions.

    Tries ``fracture_md.physics.relax.relax_cell`` (bulk module) first when it
    is importable and returns a stress-free cell; otherwise Brent-minimises
    E(s) over a uniform scale factor s of the minimum-image supercell with
    positions relaxed, and reports the residual stress (a deviatoric residual
    signals a non-cubic cell for which isotropic relaxation is insufficient).
    Returns (relaxed ASE Atoms in the original cell setting, info dict).
    """
    from fracture_md.physics.relax import relax_positions

    reps = auto_supercell(atoms, model.cutoff + skin)
    sup = atoms.repeat(reps)
    s0 = AtomicSystem.from_ase(sup, elements=model.elements)
    sp = model.species_for(s0)
    info: dict[str, Any] = dict(supercell=list(reps))
    try:  # pragma: no cover - depends on a concurrently written module
        from fracture_md.physics.relax import relax_cell  # type: ignore[attr-defined]

        res = relax_cell(model, s0)
        s_new = res[0] if isinstance(res, tuple) else res
        nh = build_neighbors(s_new, model.cutoff, skin=skin)
        st = np.asarray(model.stress(s_new.frac, s_new.box, sp, nh.idx)) * EV_PER_A3_TO_GPA
        if s_new.n_atoms == s0.n_atoms and np.max(np.abs(st)) < 1e-2:
            F = np.asarray(s_new.box) @ np.linalg.inv(np.asarray(s0.box))
            out = atoms.copy()
            out.set_cell(atoms.cell.array @ F.T, scale_atoms=True)
            info.update(method="fracture_md.physics.relax.relax_cell", max_abs_stress_GPa=float(np.max(np.abs(st))),
                        strain_matrix=F.tolist())
            return out, info
    except Exception as err:  # noqa: BLE001
        info["relax_cell_unavailable"] = f"{type(err).__name__}: {err}"

    nh = build_neighbors(s0, model.cutoff, skin=skin)
    frac = s0.frac
    box0 = s0.box
    L = float(np.min(perpendicular_widths(np.asarray(box0))))
    span = min(0.03, 0.4 * skin / L)
    scale = 1.0
    efn = jax.jit(lambda fr, b: model.energy_fn(fr, b, sp, nh.idx, model.params))
    for _ in range(3):
        r = minimize_scalar(lambda s: float(efn(frac, s * box0)), bounds=(scale - span, scale + span), method="bounded",
                            options=dict(xatol=1e-10))
        scale = float(r.x)
        s_rel, rinfo = relax_positions(model, s0.replace(frac=frac, box=scale * box0), fmax=fmax, skin=skin)
        frac = s_rel.frac
        nh = build_neighbors(s_rel, model.cutoff, skin=skin)
        efn = jax.jit(lambda fr, b, _idx=nh.idx: model.energy_fn(fr, b, sp, _idx, model.params))
        if abs(r.x - (scale - span)) > 1e-6 and abs(r.x - (scale + span)) > 1e-6:
            break
    st = np.asarray(model.stress(frac, scale * box0, sp, nh.idx)) * EV_PER_A3_TO_GPA
    dev = st - np.trace(st) / 3 * np.eye(3)
    out = atoms.copy()
    out.set_cell(atoms.cell.array * scale, scale_atoms=True)
    info.update(method="isotropic Brent scan of E(scale) with relaxed positions (fracture_md.physics.cleavage)",
                scale=scale, energy_per_atom_eV=float(efn(frac, scale * box0)) / s0.n_atoms,
                max_abs_stress_GPa=float(np.max(np.abs(st))), max_abs_deviatoric_stress_GPa=float(np.max(np.abs(dev))))
    return out, info


def _bulk_reference(model, candidate, structure, skin):
    """Return (bulk ASE Atoms (relaxed), miller_cell rows, info)."""
    from fracture_md.structures.slabs import conventional_lattice

    if structure is not None:
        atoms = structure.to_ase() if isinstance(structure, AtomicSystem) else structure.copy()
        return atoms, conventional_lattice(atoms), dict(method="structure supplied by caller (assumed relaxed)")
    if candidate.structure is not None:
        from ase import Atoms

        sr = candidate.structure
        atoms0 = Atoms(symbols=sr.symbols, positions=sr.positions_angstrom, cell=sr.cell_rows_angstrom, pbc=True)
    else:
        names = [p.name for p in candidate.phases]
        if any("3C-SiC" in nm for nm in names) or candidate.composition.formula == "SiC":
            from fracture_md.structures.builders import sic_3c_conventional

            atoms0 = sic_3c_conventional()
        else:
            raise ValueError("run_cleavage needs `structure` or candidate.structure for non-SiC candidates")
    mc0 = conventional_lattice(atoms0)
    atoms, info = relax_bulk_lattice(model, atoms0, skin=skin)
    F = atoms.cell.array.T @ np.linalg.inv(atoms0.cell.array.T)
    return atoms, (F @ mc0.T).T, info


def _interface(candidate, slab, phase_name):
    from fracture_md.schema.material import InterfaceKind, InterfaceSpec

    return InterfaceSpec(kind=InterfaceKind.CLEAVAGE_PLANE, phase_a=phase_name, orientation_a=slab.orientation,
                         termination=slab.termination,
                         description=f"cleavage on {slab.orientation.plane_miller} ({slab.termination.label})")


def _tsl(mode, interface, protocol_mode, x, T, phi, peak, at_peak, W, integ, fits, assumptions, checks, extra=None):
    from fracture_md.schema.properties import CohesiveLawFit, SampledCurve, TractionSeparationLaw
    from fracture_md.units import q

    ex = {"energy_per_area": q(np.asarray(phi).tolist(), Unit.J_PER_M2)}
    for k, v in (extra or {}).items():
        ex[k] = v
    return TractionSeparationLaw(
        mode=mode, interface=interface, protocol_mode=protocol_mode,
        curve=SampledCurve(x_name="opening" if mode == "normal" else "slip", x=q(np.asarray(x).tolist(), Unit.ANGSTROM),
                           y_name="normal_traction" if mode == "normal" else "tangential_traction",
                           y=q(np.asarray(T).tolist(), Unit.GPA), extra=ex),
        energy_per_area=q(np.asarray(phi).tolist(), Unit.J_PER_M2),
        peak_traction=q(float(peak), Unit.GPA), opening_at_peak=q(float(at_peak), Unit.ANGSTROM),
        work_of_separation=q(float(W), Unit.J_PER_M2), integrated_work=q(float(integ), Unit.J_PER_M2),
        fits=[CohesiveLawFit(**f) for f in fits], assumptions=assumptions, checks=checks,
    )


def _uber_fit_schema(fit):
    from fracture_md.units import q

    return dict(model="UBER",
                parameters={"W": q(fit["W_J_m2"], Unit.J_PER_M2), "l": q(fit["l_A"], Unit.ANGSTROM),
                            "sigma_max": q(fit["sigma_max_GPa"], Unit.GPA)},
                rms_residual=q(fit["rms_residual_J_m2"], Unit.J_PER_M2),
                domain=f"energy curve {fit['fit_range'][0]:.3g}-{fit['fit_range'][1]:.3g} Å; R^2={fit['r_squared']:.5f}, "
                       f"rms/W={fit['relative_rms']:.3g}")


def _convergence_checks(model, conv_atoms, miller_cell, protocol, base_slab, base_curve, skin, openings_sub):
    """Area (n+1 repeats), vacuum (+10 Å) and thickness (+1 cell per half) invariance of the rigid curve."""
    from fracture_md.structures.slabs import build_cleavage_slab

    def curve_for(**kw):
        args = dict(termination_label=base_slab.termination_info["label"], n_layers=protocol.n_layers,
                    in_plane_repeats=base_slab.in_plane_repeats, vacuum=protocol.vacuum_A)
        args.update(kw)
        sl = build_cleavage_slab(conv_atoms, protocol.plane_miller, args["termination_label"], args["n_layers"],
                                 args["in_plane_repeats"], args["vacuum"], cutoff=model.cutoff, skin=skin,
                                 miller_cell=miller_cell)
        sy = AtomicSystem.from_ase(sl.atoms, elements=model.elements)
        c = rigid_block_scan(model, sy, sl.upper_mask, openings_sub, _plane_normal_and_area(sy.box)[0],
                             skin=skin, check_reference=False)
        return sl, c

    sel = np.isin(base_curve.openings, openings_sub)
    phi_b, T_b = base_curve.energy_per_area[sel], base_curve.traction[sel]
    out, data = [], {}
    n1, n2 = base_slab.in_plane_repeats
    for name, kw, tol_phi in [
        ("area_normalisation_invariance", dict(in_plane_repeats=(n1 + 1, n2 + 1)), 1e-8),
        ("vacuum_convergence", dict(vacuum=protocol.vacuum_A + 10.0), 1e-8),
        ("slab_thickness_invariance", dict(n_layers=base_slab.n_layers + 2 * len(base_slab.termination_info["available_labels"])), 1e-8),
    ]:
        sl, c = curve_for(**kw)
        dphi = float(np.max(np.abs(c.energy_per_area - phi_b)))
        dT = float(np.max(np.abs(c.traction - T_b)))
        tolT = 1e-7
        details = (f"{kw}: {len(sl.atoms)} atoms vs {len(base_slab.atoms)}; max |ΔΦ| = {dphi:.2e} J/m^2, "
                   f"max |ΔT| = {dT:.2e} GPa at {len(openings_sub)} openings")
        if name == "slab_thickness_invariance":
            details += (f"; half thickness {base_slab.half_thickness[0]:.2f} Å (> 2*cutoff = {2 * model.cutoff:.2f} Å "
                        "required for exact invariance)")
        out.append(_check(name, dphi <= tol_phi and dT <= tolT, dphi, tol_phi, Unit.J_PER_M2, details))
        data[name] = dict(max_abs_dphi_J_m2=dphi, max_abs_dT_GPa=dT, n_atoms=len(sl.atoms), **{k: str(v) for k, v in kw.items()})
    return out, data


def run_cleavage(protocol, candidate, model: EnergyModel, structure=None, context=None):
    """Cleavage protocol runner (JAX-MD engine).  See module docstring for the physics.

    ``context.extra["cleavage"]`` may set: ``skin`` (0.5 Å), ``fine_step``
    (0.005 Å; None disables the fine grid), ``convergence_checks`` (True),
    ``relaxed_surfaces`` (True), ``bond_cutoff`` (None = auto).

    Returned objects: rigid TractionSeparationLaw (always; sampled on the fine
    grid), and for mode="relaxed" the grip-controlled "relaxed" law and the
    "relaxed_elastic_corrected" law.  ``ProtocolOutput.checks`` holds the
    numerical checks only; the physical-plausibility flag
    ``plausibility:peak_traction_not_set_by_cutoff_function`` is attached to
    the laws and the peak-traction records (and is expected to FAIL for
    Tersoff-type potentials).
    """
    from fracture_md.engines.base import ProtocolOutput, RunContext
    from fracture_md.export.pfczm import protocol_mode_tag
    from fracture_md.provenance import collect
    from fracture_md.records import make_record
    from fracture_md.schema.properties import Derivation, MethodSpec, PropertyName, PropertyOrigin
    from fracture_md.structures import slabs as slabs_mod
    from fracture_md.structures.slabs import build_cleavage_slab, oriented_bulk_cell

    ctx = context or RunContext()
    opts = dict(skin=0.5, fine_step=0.005, convergence_checks=True, relaxed_surfaces=True, bond_cutoff=None)
    opts.update(ctx.extra.get("cleavage", {}))
    skin = float(opts["skin"])
    card = model.card
    phase_name = candidate.phases[0].name if candidate.phases else candidate.name

    conv_atoms, miller_cell, bulk_info = _bulk_reference(model, candidate, structure, skin)
    slab = build_cleavage_slab(conv_atoms, protocol.plane_miller, protocol.termination, protocol.n_layers,
                               protocol.in_plane_repeats, protocol.vacuum_A, cutoff=model.cutoff, skin=skin,
                               bond_cutoff=opts["bond_cutoff"], miller_cell=miller_cell)
    system = AtomicSystem.from_ase(slab.atoms, elements=model.elements)
    openings = np.asarray(protocol.openings_A, float)
    rigid = rigid_opening_curve(model, system, slab.upper_mask, openings, protocol.neighbor_list_policy, skin=skin,
                                fine_step=opts["fine_step"])
    rigid_checks = list(rigid.checks)
    raw: dict[str, Any] = dict(bulk=bulk_info, slab=dict(
        n_atoms=len(slab.atoms), in_plane_repeats=slab.in_plane_repeats, n_layers=slab.n_layers, n_cells=slab.n_cells,
        area_A2=slab.area, vacuum_gap_A=slab.vacuum_gap, half_thickness_A=slab.half_thickness, cut_z_A=slab.cut_z,
        stacking_period_A=slab.stacking_period,
        termination={k: v for k, v in slab.termination_info.items()}))
    if opts["convergence_checks"]:
        sub = openings[:: max(1, len(openings) // 6)]
        cchecks, cdata = _convergence_checks(model, conv_atoms, miller_cell, protocol, slab, rigid, skin, sub)
        rigid_checks += cchecks
        raw["convergence"] = cdata
    uber = fit_uber(rigid.fine_openings if rigid.fine_openings is not None else openings,
                    rigid.fine_energy_per_area if rigid.fine_energy_per_area is not None else rigid.energy_per_area)
    art = cutoff_artifact_check(model, system, slab.upper_mask, opening=rigid.opening_at_peak, label="rigid peak traction")
    plaus = _check("plausibility:peak_traction_not_set_by_cutoff_function", not art["flagged"],
                   float(art["n_pairs_in_switching_region"]), 0.0, Unit.DIMENSIONLESS, art["details"])
    raw["rigid"] = rigid.to_dict()
    raw["uber_rigid"] = uber
    raw["cutoff_artifact_rigid"] = {k: v for k, v in art.items()}

    interface = _interface(candidate, slab, phase_name)
    term = slab.termination
    polar = bool(term.polar)
    symmetric = (not polar) and sorted(term.species_below) == sorted(term.species_above)
    lateral = "laterally constrained: in-plane lattice fixed at the relaxed bulk value"
    base_assumptions = [
        "0 K lattice statics; no thermal or zero-point effects",
        lateral,
        f"slab of {slab.n_layers} atomic layers ({slab.n_cells} oriented cells), {slab.in_plane_repeats} in-plane repeats, "
        f"vacuum gap {slab.vacuum_gap:.2f} Å, periodic in all directions",
        "bulk-terminated (unreconstructed) surfaces",
    ]
    if polar:
        base_assumptions.append(
            "polar termination: the two cleavage faces differ (" + "+".join(term.species_below) + " vs "
            + "+".join(term.species_above) + "); only γ_top + γ_bottom = W_sep is defined without a chemical-potential reference")
    rigid_assumptions = base_assumptions + [
        "rigid separation: upper block translated along the plane normal, no relaxation (unrelaxed UBER-type curve)"]
    if art["flagged"]:
        rigid_assumptions.append("FLAG: " + art["details"])
    if rigid.fine_openings is not None:
        # the law is sampled on the fine grid (exact ∫T dδ); neighbour-list and FD checks were made at the
        # protocol openings, and the δ=0 list used on the fine grid was verified to be a superset there
        rigid_assumptions.append(f"curve sampled on a uniform {len(rigid.fine_openings)}-point grid "
                                 f"(h = {rigid.fine_openings[1]:.4g} Å); protocol-grid values in raw['rigid']")
        tsls = [_tsl("normal", interface, "rigid", rigid.fine_openings, rigid.fine_traction, rigid.fine_energy_per_area,
                     rigid.peak_traction, rigid.opening_at_peak, rigid.work_of_separation, rigid.integrated_work,
                     [_uber_fit_schema(uber)], rigid_assumptions, rigid_checks + [plaus],
                     extra={"traction_fd_of_sampled_energy": _q(rigid.fine_traction_fd_sampled, Unit.GPA)})]
    else:
        tsls = [_tsl("normal", interface, "rigid", openings, rigid.traction, rigid.energy_per_area, rigid.peak_traction,
                     rigid.opening_at_peak, rigid.work_of_separation, rigid.integrated_work, [_uber_fit_schema(uber)],
                     rigid_assumptions, rigid_checks + [plaus],
                     extra={"traction_fd_central": _q(rigid.traction_fd_central, Unit.GPA)})]

    prov = collect(engine="jax_md", engine_version=_engine_version(), potential_id=card.id,
                   protocol_id=protocol.protocol_id(), requested_by=ctx.requested_by, parent_ids=list(ctx.parent_ids),
                   implementation_ids={**ctx.implementation_ids,
                                       "fracture_md.physics.cleavage": _source_hash(sys.modules[__name__]),
                                       "fracture_md.structures.slabs": _source_hash(slabs_mod)})
    pid = protocol.protocol_id()

    def method(name, desc, assumptions, mode=None):
        # `mode` -> machine-readable "[protocol_mode=...]" token used by the exporter to attach records to laws
        return MethodSpec(name=name + (" " + protocol_mode_tag(mode) if mode else ""), protocol_id=pid, engine="jax_md",
                          description=desc, assumptions=assumptions)

    cond_rigid = Conditions(stress_state="rigid normal separation, " + lateral)
    records = []
    m_rigid = method("rigid cleavage opening (autodiff tractions)",
                     "upper block translated rigidly along the plane normal; T = (1/A) dE/dδ by autodiff", rigid_assumptions,
                     "rigid")
    common = dict(candidate=candidate, card=card, phase=phase_name, interface=interface,
                  orientation=slab.orientation, provenance=prov)
    unc_W = []
    if "convergence" in raw:
        mx = max(v["max_abs_dphi_J_m2"] for v in raw["convergence"].values())
        unc_W = [Uncertainty(kind=UncertaintyKind.NUMERICAL_CONVERGENCE, value=mx, unit=Unit.J_PER_M2, interval="max_abs_diff",
                             description="max |ΔΦ| over area/vacuum/thickness variations")]
    rec_W_rigid = make_record(name=PropertyName.WORK_OF_SEPARATION, value=rigid.work_of_separation, unit=Unit.J_PER_M2,
                              method=m_rigid, conditions=cond_rigid, checks=rigid_checks, uncertainty=unc_W,
                              notes=f"unrelaxed W_sep = 2γ_unrelaxed = Φ(δ={openings[-1]} Å)", **common)
    records.append(rec_W_rigid)
    gname = PropertyName.SURFACE_ENERGY if symmetric else PropertyName.SURFACE_ENERGY_AVERAGE_PAIR
    gnote = ("γ = W_sep/2 (both faces identical by symmetry)" if symmetric else
             "(γ_top + γ_bottom)/2 = W_sep/2; individual polar-face energies need a chemical-potential reference "
             "and are not defined by a cleavage calculation")
    records.append(make_record(name=gname, value=rigid.work_of_separation / 2, unit=Unit.J_PER_M2, method=m_rigid,
                               conditions=cond_rigid, checks=rigid_checks, uncertainty=unc_W,
                               notes="unrelaxed; " + gnote, **common))
    peak_checks = rigid_checks + [plaus]
    records.append(make_record(name=PropertyName.PEAK_NORMAL_TRACTION, value=rigid.peak_traction, unit=Unit.GPA, method=m_rigid,
                               conditions=cond_rigid, checks=peak_checks, notes=art["details"], **common))
    records.append(make_record(name=PropertyName.OPENING_AT_PEAK_TRACTION, value=rigid.opening_at_peak, unit=Unit.ANGSTROM,
                               method=m_rigid, conditions=cond_rigid, checks=peak_checks, **common))
    records.append(make_record(name=PropertyName.UBER_LENGTH_SCALE, value=uber["l_A"], unit=Unit.ANGSTROM, method=m_rigid,
                               conditions=cond_rigid, checks=peak_checks,
                               uncertainty=[Uncertainty(kind=UncertaintyKind.FIT_STANDARD_ERROR, value=uber["l_stderr_A"],
                                                        unit=Unit.ANGSTROM, interval="1sigma",
                                                        description=f"UBER fit rms/W = {uber['relative_rms']:.3g}")],
                               notes=f"UBER fit W={uber['W_J_m2']:.4f} J/m^2, σ_max={uber['sigma_max_GPa']:.2f} GPa, "
                                     f"R^2={uber['r_squared']:.5f}, rms={uber['rms_residual_J_m2']:.4f} J/m^2", **common))

    W_for_griffith, W_rec_for_griffith, W_kind = rigid.work_of_separation, rec_W_rigid, "unrelaxed (rigid)"
    all_checks = list(rigid_checks)
    if opts["relaxed_surfaces"]:
        sep = max(float(openings[-1]), model.cutoff + 2.0)
        rel = relaxed_separation_energy(model, system, slab.upper_mask, sep, fmax=protocol.relax.fmax_eV_per_A, skin=skin,
                                        max_steps=protocol.relax.max_steps)
        raw["relaxed_surfaces"] = {k: v for k, v in rel.items() if not k.endswith("_system") and k != "checks"}
        rchecks = rel["checks"]
        all_checks += rchecks
        cond_rel = Conditions(stress_state="fully relaxed intact and separated slabs, " + lateral)
        m_rel = method("relaxed work of separation (energy minimisation)",
                       "E(two relaxed separated half-slabs) − E(relaxed intact slab), per area",
                       base_assumptions + ["local relaxation from bulk-terminated positions (no reconstruction search)"],
                       "relaxed_surfaces")
        rec_W_rel = make_record(name=PropertyName.WORK_OF_SEPARATION, value=rel["W_relaxed_J_m2"], unit=Unit.J_PER_M2,
                                method=m_rel, conditions=cond_rel, checks=rchecks, uncertainty=unc_W,
                                notes=f"relaxation lowers W_sep by {rel['relaxation_energy_J_m2']:.4f} J/m^2", **common)
        records.append(rec_W_rel)
        records.append(make_record(name=gname, value=rel["W_relaxed_J_m2"] / 2, unit=Unit.J_PER_M2, method=m_rel,
                                   conditions=cond_rel, checks=rchecks, uncertainty=unc_W, notes="relaxed; " + gnote, **common))
        W_for_griffith, W_rec_for_griffith, W_kind = rel["W_relaxed_J_m2"], rec_W_rel, "relaxed"

    artifacts: dict[str, Any] = dict(
        rigid_openings_A=openings, rigid_energy_per_area_J_m2=rigid.energy_per_area, rigid_traction_GPa=rigid.traction,
        rigid_fine_openings_A=rigid.fine_openings, rigid_fine_traction_GPa=rigid.fine_traction,
        rigid_fine_energy_per_area_J_m2=rigid.fine_energy_per_area, slab_atoms=slab.atoms, upper_mask=slab.upper_mask)

    if protocol.mode == "relaxed":
        oc = oriented_bulk_cell(conv_atoms, protocol.plane_miller, miller_cell=miller_cell)
        c33 = uniaxial_strain_modulus(model, oc, skin=skin)
        raw["c33_prime"] = c33
        relc = relaxed_opening_curve(model, system, slab.upper_mask, openings, protocol.grip_layers,
                                     fmax=protocol.relax.fmax_eV_per_A, max_steps=protocol.relax.max_steps, skin=skin,
                                     layer_index=slab.layer_index, c33_prime_GPa=c33["relaxed_GPa"],
                                     gap_width=slab.termination_info["gap_width"])
        raw["relaxed_curve"] = relc.to_dict()
        # cutoff artefact at the relaxed peak (actual relaxed positions)
        art_r = cutoff_artifact_check(model, system.replace(frac=jnp.asarray(relc.peak_frac % 1.0)), slab.upper_mask,
                                      label="relaxed peak traction")
        raw["cutoff_artifact_relaxed"] = art_r
        plaus_r = _check("plausibility:peak_traction_not_set_by_cutoff_function", not art_r["flagged"],
                         float(art_r["n_pairs_in_switching_region"]), 0.0, Unit.DIMENSIONLESS, art_r["details"])
        rel_assump = base_assumptions + [
            f"displacement control: bottom {relc.grip_layers} layers fixed, top {relc.grip_layers} layers displaced "
            f"(requested {protocol.grip_layers}; grips at least one receptive field = {receptive_field(model):.2f} Å thick) "
            "rigidly, all other atoms relaxed with continuation from the previous opening (extra opening seeded at the cut)",
            "traction = grip reaction −(1/A) Σ F_z (envelope theorem: = dE_relaxed/dδ)",
            "post-peak branch depends on slab thickness (elastic snap-back); not a material cohesive law without correction",
        ]
        if relc.instability:
            rel_assump.append(f"elastic instability between δ = {relc.instability['between_openings_A']} Å")
        if art_r["flagged"]:
            rel_assump.append("FLAG: " + art_r["details"])
        ro = relc.openings  # protocol openings plus bisection points around the instability
        uber_r = fit_uber(ro, relc.energy_per_area)
        raw["uber_relaxed"] = uber_r
        rchk = relc.checks
        all_checks += rchk
        if relc.instability:
            rel_assump.append(f"integrated_work excludes the snap-back interval; energy released at the instability "
                              f"{relc.instability['energy_jump_J_m2']:.4f} J/m^2")
        tsls.append(_tsl("normal", interface, "relaxed", ro, relc.traction, relc.energy_per_area, relc.peak_traction,
                         relc.opening_at_peak, relc.energy_per_area[-1], relc.raw["integrated_work_continuous_segments_J_m2"],
                         [_uber_fit_schema(uber_r)], rel_assump, rchk + [plaus_r]))
        corr_assump = rel_assump + [
            f"elastic-compliance correction δ_coh = δ − T·h_el/C'33 with h_el = {relc.h_elastic:.3f} Å (inner grip separation "
            f"minus cut gap) and relaxed-ion uniaxial-strain modulus C'33 = {c33['relaxed_GPa']:.2f} GPa (this work)",
            "approximation: all material between the grips except the cut spacing is treated as LINEAR-elastic bulk "
            "(the nonlinear softening of the strained slab near the peak is not removed, so δ_coh is overestimated there)",
            "the softening branch is unsampled under grip control (snap-back); the corrected law has a gap between the "
            "peak and the separated state"]
        coh = relc.cohesive_opening
        tsls.append(_tsl("normal", interface, "relaxed_elastic_corrected", coh, relc.traction, relc.energy_per_area,
                         relc.peak_traction, relc.cohesive_opening_at_peak, relc.energy_per_area[-1],
                         relc.raw["integrated_work_cohesive_J_m2"], [], corr_assump, rchk + [plaus_r],
                         extra={"grip_displacement": _q(ro, Unit.ANGSTROM)}))
        cond_relc = Conditions(stress_state="displacement-controlled grips, " + lateral)
        m_relc = method("relaxed displacement-controlled cleavage opening", "grip reaction traction", rel_assump, "relaxed")
        records.append(make_record(name=PropertyName.PEAK_NORMAL_TRACTION, value=relc.peak_traction, unit=Unit.GPA,
                                   method=m_relc, conditions=cond_relc, checks=rchk + [plaus_r],
                                   notes="maximum grip reaction on the stable branch; slab-thickness dependent if limited "
                                         "by an elastic instability. " + art_r["details"], **common))
        records.append(make_record(name=PropertyName.OPENING_AT_PEAK_TRACTION, value=relc.opening_at_peak, unit=Unit.ANGSTROM,
                                   method=m_relc, conditions=cond_relc, checks=rchk + [plaus_r],
                                   notes="grip displacement at peak (includes elastic stretch of the slab)", **common))
        m_relcc = method("relaxed cleavage opening, elastic-compliance corrected", "δ_coh = δ − T h_el / C'33", corr_assump,
                         "relaxed_elastic_corrected")
        records.append(make_record(name=PropertyName.OPENING_AT_PEAK_TRACTION, value=relc.cohesive_opening_at_peak,
                                   unit=Unit.ANGSTROM, method=m_relcc, conditions=cond_relc, checks=rchk + [plaus_r],
                                   notes="cohesive opening at peak after removing the bulk elastic extension", **common))
        artifacts.update(relaxed_openings_A=ro, relaxed_traction_GPa=relc.traction,
                         relaxed_energy_per_area_J_m2=relc.energy_per_area, relaxed_cohesive_opening_A=coh)

    griffith_assump = ["ideal brittle (Griffith) cleavage: G_Ic = W_sep = 2γ_s",
                       "no lattice trapping (the atomistic crack-tip barrier is ignored)",
                       "no plasticity, phase transformation or other dissipation", "straight crack on this plane and termination",
                       f"uses the {W_kind} work of separation"]
    records.append(make_record(
        name=PropertyName.G_IC_ATOMISTIC, value=W_for_griffith, unit=Unit.J_PER_M2,
        method=method("Griffith ideal brittle fracture energy [griffith_source="
                      + ("relaxed" if W_kind == "relaxed" else "rigid") + "]", "G_Ic = W_sep", griffith_assump),
        origin=PropertyOrigin.DERIVED_MODEL, conditions=Conditions(stress_state=lateral),
        derivation=Derivation(model="Griffith: G_Ic = 2 γ_s = W_sep", equation="G_Ic = [E_separated − E_intact]/A",
                              assumptions=griffith_assump, input_record_ids=[W_rec_for_griffith.id]),
        checks=[], notes="atomistic ideal value; NOT a measured or macroscopic toughness", **common))

    return ProtocolOutput(properties=records, traction_separation=tsls, checks=all_checks, structure=system,
                          artifacts=artifacts, raw=raw)


def _q(arr, unit):
    from fracture_md.units import q

    return q(np.asarray(arr, float).tolist(), unit)


def _engine_version() -> str:
    try:
        import importlib.metadata as md

        return f"jax-md {md.version('jax-md')} / jax {md.version('jax')}"
    except Exception:  # noqa: BLE001  # pragma: no cover
        return "unknown"


# ============================================================================
# tangential slip (γ-line) and mixed-mode surface
# ============================================================================
def slip_vector(slab, slip_direction: tuple[int, int, int], miller_cell: np.ndarray) -> tuple[np.ndarray, float]:
    """Lab-frame unit slip vector and the period (Å) of the in-plane lattice along [uvw]."""
    from fracture_md.structures.slabs import _rationalize

    R = np.asarray(slab.oriented_cell.info["rotation_crystal_to_lab"])
    v = np.asarray(slip_direction, float) @ np.asarray(miller_cell)
    v_lab = R @ v
    if abs(v_lab[2]) > 1e-8 * np.linalg.norm(v_lab):
        raise ValueError(f"slip direction {slip_direction} is not in the plane {slab.orientation.plane_miller}")
    c = slab.oriented_cell.cell.array
    coef = np.linalg.lstsq(np.stack([c[0], c[1]], 1)[:2], v_lab[:2], rcond=None)[0]
    ints = _rationalize(coef)
    if ints is None:
        raise ValueError("slip direction is not a rational in-plane lattice direction")
    period_vec = ints[0] * c[0] + ints[1] * c[1]
    return v_lab / np.linalg.norm(v_lab), float(np.linalg.norm(period_vec))


def rigid_mixed_mode_surface(model: EnergyModel, slab_system: AtomicSystem, upper_mask: np.ndarray, normal_openings,
                             slips, slip_unit: np.ndarray, *, skin: float = 0.5) -> dict[str, Any]:
    """Rigid energy surface Φ(δ_n, δ_t) (J/m²) with autodiff tractions T_n, T_t (GPa); list rebuilt at every point."""
    upper = np.asarray(upper_mask, bool)
    n, area = _plane_normal_and_area(slab_system.box)
    if vacuum_gap(slab_system, upper) - max(normal_openings) <= model.cutoff + skin:
        raise ValueError("vacuum too small")
    species = model.species_for(slab_system)
    fns = _Fns(model, slab_system.box, species)
    frac0 = jnp.asarray(slab_system.frac)
    E = np.zeros((len(normal_openings), len(slips)))
    Tn = np.zeros_like(E)
    Tt = np.zeros_like(E)
    for a, dn in enumerate(normal_openings):
        for b, dt in enumerate(slips):
            fr = _block_frac(frac0, upper, slab_system.box, dn * n + dt * slip_unit)
            idx = build_neighbors(slab_system.replace(frac=fr), model.cutoff, skin=skin).idx
            e, g = fns.eg(fr, idx)
            E[a, b] = float(e)
            Tn[a, b] = _block_derivative(g, upper, n)
            Tt[a, b] = _block_derivative(g, upper, slip_unit)
    phi = (E - E[0, 0]) / area * EV_PER_A2_TO_J_PER_M2
    return dict(normal_openings_A=np.asarray(normal_openings), slips_A=np.asarray(slips), energy_per_area_J_m2=phi,
                normal_traction_GPa=Tn / area * EV_PER_A3_TO_GPA, tangential_traction_GPa=Tt / area * EV_PER_A3_TO_GPA)


def _rigid_separation_from(model: EnergyModel, system: AtomicSystem, upper_mask: np.ndarray, base_offset: np.ndarray,
                           skin: float) -> Optional[float]:
    """Rigid W_sep (J/m^2) from the state `base_offset`: E(base + (cutoff + 1 Å) n) − E(base), or None if no room."""
    upper = np.asarray(upper_mask, bool)
    n, area = _plane_normal_and_area(system.box)
    sep = float(model.cutoff) + 1.0
    if vacuum_gap(system, upper) - (abs(float(np.asarray(base_offset) @ n)) + sep) <= model.cutoff + skin:
        return None
    fns = _Fns(model, system.box, model.species_for(system))
    frac0 = jnp.asarray(system.frac)
    es = []
    for extra in (0.0, sep):
        fr = _block_frac(frac0, upper, system.box, np.asarray(base_offset, float) + extra * n)
        es.append(float(fns.e(fr, build_neighbors(system.replace(frac=fr), model.cutoff, skin=skin).idx)))
    return (es[1] - es[0]) / area * EV_PER_A2_TO_J_PER_M2


def run_tangential_slip(protocol, candidate, model: EnergyModel, structure=None, context=None):
    """Rigid in-plane sliding of the upper block at fixed normal opening (γ-line).

    E(δ_t)/A and T_t = (1/A) dE/dδ_t along the slip direction over one lattice
    period; the neighbour list is managed by the protocol policy (rebuilt at
    every step: new pairs come into range while sliding).  The maximum is the
    *unrelaxed* unstable stacking energy (rigid γ-line); it is an ideal-shear
    quantity, NOT a mode-II fracture energy G_IIc.  mode="relaxed_normal"
    minimises E over the normal opening at each slip (upper block still rigid).
    ``context.extra["cleavage"]["mixed_mode_grid"] = (n_normal, n_slip, δn_max)``
    additionally returns the rigid Φ(δ_n, δ_t) surface in artifacts.
    """
    from fracture_md.engines.base import ProtocolOutput, RunContext
    from fracture_md.export.pfczm import protocol_mode_tag
    from fracture_md.provenance import collect
    from fracture_md.records import make_record
    from fracture_md.schema.properties import MethodSpec, PropertyName
    from fracture_md.structures.slabs import build_cleavage_slab

    ctx = context or RunContext()
    opts = dict(skin=0.5, bond_cutoff=None, mixed_mode_grid=None)
    opts.update(ctx.extra.get("cleavage", {}))
    skin = float(opts["skin"])
    card = model.card
    phase_name = candidate.phases[0].name if candidate.phases else candidate.name
    conv_atoms, miller_cell, bulk_info = _bulk_reference(model, candidate, structure, skin)
    slab = build_cleavage_slab(conv_atoms, protocol.plane_miller, protocol.termination, protocol.n_layers, None,
                               protocol.vacuum_A, cutoff=model.cutoff, skin=skin, bond_cutoff=opts["bond_cutoff"],
                               miller_cell=miller_cell)
    system = AtomicSystem.from_ase(slab.atoms, elements=model.elements)
    t_hat, period = slip_vector(slab, protocol.slip_direction, miller_cell)
    n, area = _plane_normal_and_area(system.box)
    slips = np.linspace(0.0, period, protocol.n_points)
    dn = float(protocol.normal_opening_A)
    if vacuum_gap(system, slab.upper_mask) - abs(dn) <= model.cutoff + skin:
        raise ValueError("vacuum too small for the normal opening")
    dn_opt = None
    separated_at: list[float] = []
    if protocol.mode == "relaxed_normal":
        species = model.species_for(system)
        fns = _Fns(model, system.box, species)
        frac0 = jnp.asarray(system.frac)
        upper = np.asarray(slab.upper_mask, bool)
        mgr = _NeighborManager(system, model.cutoff, skin, "rebuild_each_step")
        frac_at = lambda x, s_: _block_frac(frac0, upper, system.box, x * n + s_ * t_hat)  # noqa: E731
        dn_opt, E, T, Tn, d_ref, d_fd = [], [], [], [], [], []
        prev = dn
        h = 1e-4
        for s_ in slips:
            def e_of(x, _s=s_):
                fr = frac_at(x, _s)
                return float(fns.e(fr, mgr.get(fr)))
            lo, hi = prev - 0.3, prev + 0.3
            for _ in range(8):  # local minimum by continuation; move the bracket if the minimum sits on a bound
                r = minimize_scalar(e_of, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-6))
                if hi - r.x < 1e-3:
                    lo, hi = hi - 0.05, hi + 0.3
                elif r.x - lo < 1e-3:
                    lo, hi = lo - 0.3, lo + 0.05
                else:
                    break
            x = float(r.x)
            fr = frac_at(x, s_)
            idx = mgr.get(fr)
            e, g = fns.eg(fr, idx)
            if not cross_gap_pairs(system.replace(frac=fr), upper, model.cutoff):
                separated_at.append(float(s_))
            ridx, _ = _reference_idx(fr, system.box, system.n_atoms, model.cutoff)
            d_ref.append(abs(float(fns.e(fr, ridx)) - float(e)))
            tt = _block_derivative(g, upper, t_hat)
            fd = (float(fns.e(frac_at(x, s_ + h), idx)) - float(fns.e(frac_at(x, s_ - h), idx))) / (2 * h)
            d_fd.append(abs(tt - fd) / area * EV_PER_A3_TO_GPA)
            dn_opt.append(x)
            E.append(float(e))
            T.append(tt / area * EV_PER_A3_TO_GPA)
            Tn.append(_block_derivative(g, upper, n) / area * EV_PER_A3_TO_GPA)
            prev = x
        E, T = np.asarray(E), np.asarray(T)
        fd_tol = 1e-5 + 1e-6 * float(np.max(np.abs(T)))  # same relative rule as rigid_block_scan
        checks = [
            _check("neighbor_list_equals_reference_energy_every_step", max(d_ref) <= 1e-8, max(d_ref), 1e-8, Unit.EV),
            _check("traction_autodiff_equals_central_fd", max(d_fd) <= fd_tol, max(d_fd), fd_tol, Unit.GPA,
                   f"central FD with real-space block displacement h={h} Å, same neighbour list"),
            _check("normal_traction_zero_at_relaxed_opening", float(np.max(np.abs(Tn))) <= 1e-2, float(np.max(np.abs(Tn))),
                   1e-2, Unit.GPA, "1-D minimisation of E over the normal opening at each slip (upper block rigid)"),
            _check("plausibility:gamma_line_not_capped_by_separation", not separated_at, float(len(separated_at)), 0.0,
                   Unit.DIMENSIONLESS,
                   "no local minimum in contact at slips " + ", ".join(f"{v:.3f}" for v in separated_at) +
                   " Å: the blocks separate (E = W_sep) rather than slide" if separated_at else
                   "the blocks stay in contact along the whole path"),
        ]
        phi = (E - E[0]) / area * EV_PER_A2_TO_J_PER_M2
        curve = None
    else:
        curve = rigid_block_scan(model, system, slab.upper_mask, slips, t_hat, base_offset=dn * n,
                                 neighbor_policy=protocol_policy(protocol), skin=skin, mode="tangential")
        phi, T, checks = curve.energy_per_area, curve.traction, list(curve.checks)
        stale = float(np.max(np.abs(curve.initial_list_minus_reference_eV)))
        checks.append(_check("initial_list_staleness_detected_during_slip", True, stale, None, Unit.EV,
                             f"energy with the never-rebuilt δ=0 list deviates by up to {stale:.3e} eV from the reference "
                             "during sliding (new pairs enter range); the managed list is used for all reported values"))
    integ = float(np.trapezoid(T, slips) / EV_PER_A3_TO_GPA * EV_PER_A2_TO_J_PER_M2)
    kmax = int(np.argmax(np.abs(T)))
    k_e = int(np.argmax(phi))
    # Plausibility: the rigid γ-line maximum should not exceed the rigid work of separation from the
    # same reference state (slip 0, normal opening dn); if it does, rigid blocks would rather open
    # than slide and the γ-line maximum is not a physical unstable stacking energy.
    w_sep = _rigid_separation_from(model, system, slab.upper_mask, dn * n, skin)
    if w_sep is None:
        gl_check = _check("plausibility:gamma_line_max_below_work_of_separation", True, None, None, Unit.J_PER_M2,
                          "not evaluated: vacuum too small for a separating opening")
    else:
        gl_check = _check("plausibility:gamma_line_max_below_work_of_separation", float(phi[k_e]) <= w_sep,
                          float(phi[k_e]), w_sep, Unit.J_PER_M2,
                          f"max Φ(γ-line) = {float(phi[k_e]):.4f} J/m^2 vs rigid W_sep = {w_sep:.4f} J/m^2 from the same "
                          "reference state" + (": the rigid blocks would separate rather than slide; the γ-line maximum "
                                               "is not a physical γ_us" if float(phi[k_e]) > w_sep else ""))
    checks.append(gl_check)
    interior_min = [float(slips[k]) for k in range(1, len(slips) - 1) if phi[k] < phi[k - 1] and phi[k] < phi[k + 1]]
    from fracture_md.schema.material import Orientation

    orient = slab.orientation.model_copy(update={"direction": tuple(int(i) for i in protocol.slip_direction)})
    interface = _interface(candidate, slab, phase_name)
    assumptions = [
        "0 K; rigid in-plane translation of the upper block (no atomic relaxation): unrelaxed γ-line",
        f"normal opening {'optimised at each slip (local minimum by continuation)' if dn_opt is not None else f'fixed at {dn} Å'}",
        "ideal-shear (γ-line) quantity: NOT a mode-II fracture energy G_IIc",
        "laterally constrained: in-plane lattice fixed at bulk",
        f"slip along [{' '.join(str(i) for i in protocol.slip_direction)}], period {period:.4f} Å",
    ]
    if separated_at:
        assumptions.append(f"FLAG: blocks separated during sliding at slips {separated_at} Å (γ-line capped by W_sep)")
    if not gl_check.passed:
        assumptions.append("FLAG: " + gl_check.details)
    tsl = _tsl("tangential", interface, "rigid", slips, T, phi, float(np.abs(T[kmax])), float(slips[kmax]),
               float(phi[-1]), integ, [], assumptions, checks)
    prov = collect(engine="jax_md", engine_version=_engine_version(), potential_id=card.id,
                   protocol_id=protocol.protocol_id(), requested_by=ctx.requested_by, parent_ids=list(ctx.parent_ids),
                   implementation_ids={**ctx.implementation_ids,
                                       "fracture_md.physics.cleavage": _source_hash(sys.modules[__name__])})
    m = MethodSpec(name="rigid tangential slip (γ-line) " + protocol_mode_tag("rigid"), protocol_id=protocol.protocol_id(),
                   engine="jax_md",
                   description="upper block translated rigidly in-plane; T_t = (1/A) dE/dδ_t by autodiff",
                   assumptions=assumptions)
    common = dict(candidate=candidate, card=card, phase=phase_name, interface=interface, orientation=orient,
                  provenance=prov, method=m, conditions=Conditions(stress_state="rigid in-plane sliding, laterally constrained"),
                  checks=checks)
    records = [
        make_record(name=PropertyName.PEAK_TANGENTIAL_TRACTION, value=float(np.abs(T[kmax])), unit=Unit.GPA,
                    notes=f"at slip {slips[kmax]:.4f} Å; rigid (unrelaxed)", **common),
        make_record(name=PropertyName.UNSTABLE_STACKING_FAULT_ENERGY, value=float(phi[k_e]), unit=Unit.J_PER_M2,
                    notes=f"maximum of the rigid γ-line at slip {slips[k_e]:.4f} Å: unrelaxed upper bound of γ_us "
                          f"on the {slab.termination.label} cut; not G_IIc"
                          + ("; CAPPED BY SEPARATION (blocks separate instead of sliding)" if separated_at else "")
                          + ("; EXCEEDS THE RIGID W_sep (" + gl_check.details + ")" if not gl_check.passed else ""),
                    **common),
    ]
    artifacts = dict(slips_A=slips, energy_per_area_J_m2=phi, tangential_traction_GPa=T,
                     normal_openings_A=np.asarray(dn_opt) if dn_opt is not None else None)
    raw = dict(period_A=period, slip_unit_lab=t_hat.tolist(), interior_local_minima_A=interior_min, bulk=bulk_info,
               rigid_work_of_separation_J_m2=w_sep,
               separated_at_slips_A=separated_at,
               curve=curve.to_dict() if curve is not None else None, n_atoms=len(slab.atoms))
    if opts["mixed_mode_grid"]:
        nn, nt, dnmax = opts["mixed_mode_grid"]
        artifacts["mixed_mode"] = rigid_mixed_mode_surface(model, system, slab.upper_mask, np.linspace(0, dnmax, int(nn)),
                                                           np.linspace(0, period, int(nt)), t_hat, skin=skin)
    numerical = [c for c in checks if not c.name.startswith("plausibility:")]
    return ProtocolOutput(properties=records, traction_separation=[tsl], checks=numerical, structure=system,
                          artifacts=artifacts, raw=raw)


def protocol_policy(protocol) -> str:
    return getattr(protocol, "neighbor_list_policy", "rebuild_each_step")
