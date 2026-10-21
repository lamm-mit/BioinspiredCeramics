"""Rigid, frictionless indenters as exact signed-distance walls (Berkovich pyramid, cone, sphere).

The indenter is a rigid solid described by its exact signed distance function
(SDF, Å, negative inside the solid).  It acts on every atom with the LAMMPS
``fix indent`` force law

    U = (K/3) Σ_i δ_i³,   δ_i = max(0, -sdf(r_i)),   F_i = K δ_i² ∇sdf(r_i),

a purely repulsive wall (no adhesion, no friction) of stiffness K (eV/Å³).
The same functional is meant for the LAMMPS production deck (fix external), so
both engines indent with the same tip.  K = 100 eV/Å³ is the protocol default
(K = 10 biases the depth by ~7 %; report the force-weighted penetration
``pen_fw`` and use h_c = h - pen_fw).

Geometry (lab frame; the tip comes down along -z onto a surface with normal +z)
------------------------------------------------------------------------------
* ``apex`` (a params leaf, Å) is the LOWEST point of the (rounded) tip, so the
  depth is h = z_surface - apex_z.
* Berkovich: three-sided pyramid whose faces make ``half_angle_deg`` with the
  axis (65.27° modified Berkovich: ideal projected area C0 h² with
  C0 = 3√3 tan²θ = 24.494, equivalent cone 70.296°).  ``azimuth_deg`` is the
  in-plane angle of the outward normal of face 0 from lab +x.
* Cone: circular cone of half-angle ``half_angle_deg`` (default: the Berkovich
  equivalent cone, i.e. the same ideal area function).
* Sphere: ball of radius ``tip_radius_A`` (the ``fix indent sphere`` tip).
* Rounding: pyramid and cone are Minkowski-rounded with R = ``tip_radius_A``:
  the solid is the sharp solid K' (apex a' = apex + R e_z) dilated by a ball of
  radius R.  Faces stay exact planes, edges become cylinders of radius R and
  the apex a sphere of radius R.  The SDF is exact:
  sdf = dist(p, K') - R outside K' and max_i n_i·(p - a') - R inside it;
  dist(p, K') is the smallest distance to the FEASIBLE candidate nearest
  points (apex, three face projections, three edge rays), which is exact for a
  convex polyhedral cone.  R = 0 is the sharp tip.  Norms go through a safe
  norm so gradients stay finite at the apex, on edges and on the axis.
* Height cap: the solid is semi-infinite upwards; only atoms less than
  ``height_cap_A`` above the lowest point feel it.  Atoms at the top of the
  periodic cell (e.g. a bottom atom wrapped to frac_z = 0.999) are therefore
  never "inside" the tip.  :func:`z_guard_report` / :func:`check_z_guard` flag
  any atom sitting in the tip column above the cap (it would be invisible to
  the tip); :func:`check_indenter_cell` checks that the capped tip fits in the
  cell.  The indentation slab builder sizes the vacuum as
  ``height_cap_A + 2 (cutoff + skin)`` (structures/indentation.py).
* Lateral periodic images: the displacement from the apex is reduced by
  integer multiples of the in-plane lattice vectors (rounded fractional
  displacement).  This is always a lattice translation (so the energy is
  periodic) and the exact minimum image for the orthorhombic in-plane cells
  that structures/indentation.py builds.

JAX conventions
---------------
:func:`with_indenter` returns a composite energy function with the harness
signature ``energy_fn(frac, box, species, idx, params)`` and
``params = {"material": <material params>, "indenter": {"apex": (3,), "K": ()}}``.
The indenter term receives positions through ``jax_md.space.transform``, the
only form for which ``jax.grad(energy_fn, 0)`` is the real-space gradient.
Writing ``frac @ box.T`` instead returns the fractional gradient, i.e. forces
wrong by factors of the cell lengths, and ``exact_frac_functions`` does not
detect it (tests/test_indentation.py keeps that as a negative control).  Apex
and K are pytree leaves, so the tip moves and K changes without
recompilation; the geometry is static (a hashable dataclass) and part of the
compile-cache key.  Composite functions are cached at module level per
(material energy_fn, geometry), so ``compiled_kernels`` and the FIRE / MD
kernels compile once per material, geometry and shape.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from fracture_md.potentials.base import EnergyFn, EnergyModel

BERKOVICH_HALF_ANGLE_DEG = 65.27  # modified Berkovich face-to-axis angle
DEFAULT_TIP_RADIUS_A = 10.0
DEFAULT_STIFFNESS_EV_PER_A3 = 100.0  # recommended protocol item 4 (K = 10 biases the depth by ~7 %)
DEFAULT_HEIGHT_CAP_A = 60.0
PARKED_APEX_A = (0.0, 0.0, 1.0e4)  # default apex of a new composite: far above any slab, no contact
SHAPES = ("berkovich", "cone", "sphere")

_BIG = 1.0e30  # distance of infeasible candidates (representable in float32 as well)


def berkovich_area_coefficient(half_angle_deg: float = BERKOVICH_HALF_ANGLE_DEG) -> float:
    """C0 of the ideal projected area A = C0 h² of a sharp three-sided pyramid (24.494 at 65.27°).

    The cross-section at depth h is an equilateral triangle of inradius h tanθ,
    area 3√3 (h tanθ)².
    """
    t = math.tan(math.radians(half_angle_deg))
    return 3.0 * math.sqrt(3.0) * t * t


BERKOVICH_C0 = berkovich_area_coefficient()
EQUIVALENT_CONE_HALF_ANGLE_DEG = math.degrees(math.atan(math.sqrt(BERKOVICH_C0 / math.pi)))  # 70.296°


class IndenterGuardError(RuntimeError):
    """An atom is where the capped indenter cannot see it (above the height cap, inside the tip column)."""


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class IndenterGeometry:
    """Static tip geometry (hashable: part of the compile-cache key, never traced).

    ``half_angle_deg`` is the face-to-axis angle of the Berkovich pyramid
    (default 65.27°) or the half-angle of the cone (default 70.296°, the
    Berkovich equivalent cone); a sphere has none.  ``tip_radius_A`` is the
    Minkowski rounding radius (pyramid, cone; 0 = sharp) or the sphere radius.
    ``azimuth_deg`` is the in-plane angle of the outward normal of Berkovich
    face 0 from lab +x (record it with the crystal directions of lab x/y).
    ``height_cap_A`` is the height above the lowest point beyond which the tip
    does not act.  ``form`` names the force law; only "cubic"
    (U = K/3 Σ δ³, LAMMPS fix indent) exists.
    """

    shape: str = "berkovich"
    half_angle_deg: Optional[float] = None
    tip_radius_A: float = DEFAULT_TIP_RADIUS_A
    azimuth_deg: float = 0.0
    height_cap_A: float = DEFAULT_HEIGHT_CAP_A
    form: str = "cubic"

    def __post_init__(self) -> None:
        if self.shape not in SHAPES:
            raise ValueError(f"unknown indenter shape {self.shape!r}; available: {SHAPES}")
        if self.form != "cubic":
            raise ValueError(f"unknown indenter force law {self.form!r}; only 'cubic' (U = K/3 sum delta^3) exists")
        if self.shape == "sphere":
            if self.half_angle_deg is not None:
                raise ValueError("a spherical indenter has no half angle")
            if not self.tip_radius_A > 0.0:
                raise ValueError("a spherical indenter needs tip_radius_A > 0")
        else:
            if self.half_angle_deg is None:
                default = BERKOVICH_HALF_ANGLE_DEG if self.shape == "berkovich" else EQUIVALENT_CONE_HALF_ANGLE_DEG
                object.__setattr__(self, "half_angle_deg", default)
            object.__setattr__(self, "half_angle_deg", float(self.half_angle_deg))  # type: ignore[arg-type]
            if not 0.0 < self.half_angle_deg < 90.0:  # type: ignore[operator]
                raise ValueError(f"half_angle_deg must lie in (0, 90), got {self.half_angle_deg}")
            if self.tip_radius_A < 0.0:
                raise ValueError("tip_radius_A must be >= 0")
        if not self.height_cap_A > 0.0:
            raise ValueError("height_cap_A must be > 0")
        for f in ("tip_radius_A", "azimuth_deg", "height_cap_A"):
            object.__setattr__(self, f, float(getattr(self, f)))

    # ---------------------------------------------------------------- constructors
    @classmethod
    def berkovich(cls, tip_radius_A: float = DEFAULT_TIP_RADIUS_A, azimuth_deg: float = 0.0, *,
                  half_angle_deg: float = BERKOVICH_HALF_ANGLE_DEG,
                  height_cap_A: float = DEFAULT_HEIGHT_CAP_A) -> "IndenterGeometry":
        return cls("berkovich", half_angle_deg, tip_radius_A, azimuth_deg, height_cap_A)

    @classmethod
    def cone(cls, half_angle_deg: float = EQUIVALENT_CONE_HALF_ANGLE_DEG,
             tip_radius_A: float = DEFAULT_TIP_RADIUS_A, *,
             height_cap_A: float = DEFAULT_HEIGHT_CAP_A) -> "IndenterGeometry":
        return cls("cone", half_angle_deg, tip_radius_A, 0.0, height_cap_A)

    @classmethod
    def sphere(cls, radius_A: float, *, height_cap_A: float = DEFAULT_HEIGHT_CAP_A) -> "IndenterGeometry":
        return cls("sphere", None, radius_A, 0.0, height_cap_A)

    # ---------------------------------------------------------------- derived quantities
    @property
    def label(self) -> str:
        if self.shape == "sphere":
            return f"sphere(R{self.tip_radius_A:g}A)"
        az = f",az{self.azimuth_deg:g}deg" if self.shape == "berkovich" else ""
        return f"{self.shape}({self.half_angle_deg:g}deg,R{self.tip_radius_A:g}A{az})"

    @property
    def ideal_area_coefficient(self) -> Optional[float]:
        """C0 of the sharp-tip area function A = C0 h² (None for a sphere)."""
        if self.shape == "berkovich":
            return berkovich_area_coefficient(self.half_angle_deg)  # type: ignore[arg-type]
        if self.shape == "cone":
            return math.pi * math.tan(math.radians(self.half_angle_deg)) ** 2  # type: ignore[arg-type]
        return None

    @property
    def corner_radius_per_depth(self) -> Optional[float]:
        """Largest lateral distance of the sharp-tip contact perimeter per unit depth.

        Berkovich: the triangle corners sit at 2 tanθ h = 4.3376 h (65.27°); cone: tanα h.
        """
        if self.shape == "berkovich":
            return 2.0 * math.tan(math.radians(self.half_angle_deg))  # type: ignore[arg-type]
        if self.shape == "cone":
            return math.tan(math.radians(self.half_angle_deg))  # type: ignore[arg-type]
        return None

    def face_normals(self) -> np.ndarray:
        """(3, 3) outward unit normals of the Berkovich faces (rows), lab frame."""
        if self.shape != "berkovich":
            raise ValueError("face normals exist only for the Berkovich pyramid")
        th = math.radians(self.half_angle_deg)  # type: ignore[arg-type]
        phi = [math.radians(self.azimuth_deg) + 2.0 * math.pi * i / 3.0 for i in range(3)]
        return np.array([[math.cos(th) * math.cos(p), math.cos(th) * math.sin(p), -math.sin(th)] for p in phi])

    def edge_directions(self) -> np.ndarray:
        """(3, 3) unit vectors (rows, pointing up) of the pyramid edges between faces (0,1), (1,2), (2,0)."""
        n = self.face_normals()
        out = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            v = np.cross(n[i], n[j])
            v /= np.linalg.norm(v)
            out.append(v if v[2] > 0 else -v)
        return np.array(out)

    def area(self, h_c: Any) -> np.ndarray:
        """Exact projected area (Å²) at contact height ``h_c`` above the lowest point (:func:`tip_area_function`)."""
        return tip_area_function(h_c, self)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.update(label=self.label, ideal_area_coefficient=self.ideal_area_coefficient)
        return d


# ----------------------------------------------------------------------------
# signed distance (pure JAX; any leading batch shape)
# ----------------------------------------------------------------------------
def _dot(v: jnp.ndarray, u: Any) -> jnp.ndarray:
    """v · u over the last axis by elementwise products (no matmul: GPU float32 matmuls default to TF32)."""
    return v[..., 0] * u[0] + v[..., 1] * u[1] + v[..., 2] * u[2]


def _pick(c: jnp.ndarray, largest: bool = False) -> jnp.ndarray:
    """min (or max) over the last axis, differentiated through the selected entry only.

    ``jnp.min``/``jnp.max`` differentiate with an equality test between the
    reduced value and the operand; when XLA recomputes the operand in another
    fusion (different FMA contraction) the test can fail bitwise and the
    gradient becomes 0 or NaN (observed on CPU for batches of ~1e3 points).
    Selecting by a stop-gradient argmin/argmax and gathering avoids that; at
    ties the candidates share value and gradient (the distance is C1).
    """
    sel = jax.lax.stop_gradient(c)
    k = jnp.argmax(sel, axis=-1) if largest else jnp.argmin(sel, axis=-1)
    return jnp.take_along_axis(c, k[..., None], axis=-1)[..., 0]


def _safe_norm(v: jnp.ndarray) -> jnp.ndarray:
    """|v| along the last axis with a finite (zero) gradient at v = 0 (double-where)."""
    sq = jnp.sum(v * v, axis=-1)
    ok = sq > 1e-30
    return jnp.where(ok, jnp.sqrt(jnp.where(ok, sq, 1.0)), 0.0)


def _sdf_local(d: jnp.ndarray, geometry: IndenterGeometry) -> jnp.ndarray:
    """Exact SDF (Å) for displacements ``d`` (..., 3) from the lowest point of the tip."""
    R = geometry.tip_radius_A
    w = d - jnp.asarray([0.0, 0.0, R], dtype=d.dtype)  # from the eroded apex a' (sphere: from the centre)
    if geometry.shape == "sphere":
        return _safe_norm(w) - R
    if geometry.shape == "cone":
        a = math.radians(geometry.half_angle_deg)  # type: ignore[arg-type]
        rho = _safe_norm(w[..., :2])
        s = rho * math.cos(a) - w[..., 2] * math.sin(a)  # signed distance to the generator in the meridian plane
        t = rho * math.sin(a) + w[..., 2] * math.cos(a)  # foot position along the generator (apex if <= 0)
        return jnp.where(t > 0.0, s, _safe_norm(w)) - R

    nrm = geometry.face_normals()
    gram = nrm @ nrm.T
    n = jnp.asarray(nrm, dtype=d.dtype)
    e = jnp.asarray(geometry.edge_directions(), dtype=d.dtype)
    tol = 1e-9 if d.dtype == jnp.float64 else 1e-4  # Å, face-feasibility slack (value error O(tol^2))
    s = jnp.stack([_dot(w, n[i]) for i in range(3)], axis=-1)  # signed distances to the three face planes of K'
    inside = jnp.all(s <= 0.0, axis=-1)
    cands = [_safe_norm(w)]  # nearest point = eroded apex
    for i in range(3):  # nearest point inside face i: its projection satisfies the other two half-spaces
        j, k = (i + 1) % 3, (i + 2) % 3
        feasible = ((s[..., i] > 0.0) & (s[..., j] - gram[i, j] * s[..., i] <= tol)
                    & (s[..., k] - gram[i, k] * s[..., i] <= tol))
        cands.append(jnp.where(feasible, s[..., i], _BIG))
    for m in range(3):  # nearest point on edge ray m (t > 0; t <= 0 is the apex candidate)
        t = _dot(w, e[m])
        cands.append(jnp.where(t > 0.0, _safe_norm(w - t[..., None] * e[m]), _BIG))
    outside = _pick(jnp.stack(cands, axis=-1))
    return jnp.where(inside, _pick(s, largest=True), outside) - R


def _displacement(X: jnp.ndarray, apex: jnp.ndarray, box: Optional[jnp.ndarray]) -> jnp.ndarray:
    """X - apex reduced by in-plane lattice vectors (rounded fractional displacement; exact for orthorhombic)."""
    d = X - apex
    if box is None:
        return d
    binv = jax.lax.stop_gradient(jnp.linalg.inv(box))
    n1 = jnp.round(jax.lax.stop_gradient(_dot(d, binv[0])))  # fractional displacement along a1, a2
    n2 = jnp.round(jax.lax.stop_gradient(_dot(d, binv[1])))
    return d - n1[..., None] * box[:, 0] - n2[..., None] * box[:, 1]


def indenter_sdf(X: jnp.ndarray, apex: Any, geometry: IndenterGeometry, box: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Signed distance (Å) of Cartesian positions ``X`` (N, 3) from the tip surface (negative inside; no cap).

    ``apex`` (3,) is the lowest point of the tip; ``box`` (lattice vectors as
    columns) enables the lateral periodic images.
    """
    X = jnp.asarray(X)
    return _sdf_local(_displacement(X, jnp.asarray(apex, dtype=X.dtype), box), geometry)


def penetration(X: jnp.ndarray, apex: Any, geometry: IndenterGeometry, box: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """δ_i = max(0, -sdf) (Å) for atoms less than ``height_cap_A`` above the lowest point, 0 otherwise."""
    X = jnp.asarray(X)
    d = _displacement(X, jnp.asarray(apex, dtype=X.dtype), box)
    delta = jnp.maximum(-_sdf_local(d, geometry), 0.0)
    return jnp.where(d[..., 2] < geometry.height_cap_A, delta, 0.0)


def indenter_energy(X: jnp.ndarray, apex: Any, K: Any, geometry: IndenterGeometry,
                    box: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """U = (K/3) Σ δ³ (eV) for Cartesian positions ``X`` (Å), apex (3,) Å and K (eV/Å³)."""
    delta = penetration(X, apex, geometry, box)
    return (K / 3.0) * jnp.sum(delta**3)


def indenter_energy_frac(frac: jnp.ndarray, box: jnp.ndarray, indenter: dict, geometry: IndenterGeometry) -> jnp.ndarray:
    """Indenter energy in the harness convention (fractional coordinates, real-space gradient w.r.t. frac).

    ``indenter`` = {"apex": (3,), "K": ()}.  Positions go through
    ``space.transform`` (never ``frac @ box.T``: see the module docstring).
    """
    with jax.default_matmul_precision("highest"):  # float32 on a GPU: no TF32 in space.transform
        X = space.transform(box, frac)
        return indenter_energy(X, indenter["apex"], indenter["K"], geometry, box)


# ----------------------------------------------------------------------------
# composite energy (material + indenter)
# ----------------------------------------------------------------------------
# Module-level caches: one composite function object per (material energy_fn, geometry), so the
# identity-keyed kernel caches (compiled_kernels, relax._fire_kernel, MD chunk kernels) compile once.
_COMPOSITES: "dict[tuple[Any, IndenterGeometry], EnergyFn]" = {}
_COMPOSITE_PARTS: "dict[Any, tuple[Any, IndenterGeometry]]" = {}


def indented_energy_fn(energy_fn: EnergyFn, geometry: IndenterGeometry) -> EnergyFn:
    """Composite ``energy_fn(frac, box, species, idx, params)`` = material + indenter (cached).

    ``params = {"material": <params of energy_fn>, "indenter": {"apex": (3,), "K": ()}}``;
    species and idx pass through to the material unchanged (the tip is not an atom).
    """
    if energy_fn in _COMPOSITE_PARTS:
        raise ValueError("energy function already contains an indenter (nested indenters are not supported)")
    key = (energy_fn, geometry)
    fn = _COMPOSITES.get(key)
    if fn is None:

        def fn(frac, box, species, idx, params):
            return (energy_fn(frac, box, species, idx, params["material"])
                    + indenter_energy_frac(frac, box, params["indenter"], geometry))

        fn.__name__ = f"indented_{getattr(energy_fn, '__name__', 'energy')}"
        fn.__qualname__ = fn.__name__
        _COMPOSITES[key] = fn
        _COMPOSITE_PARTS[fn] = key
    return fn


def indenter_params(material: Any, apex_A: Any = PARKED_APEX_A, K_eV_per_A3: Optional[float] = None) -> dict:
    """Composite params pytree ``{"material": ..., "indenter": {"apex": (3,), "K": ()}}``.

    ``material`` is the material params pytree, a material EnergyModel, or a
    composite EnergyModel (its material params and, unless ``K_eV_per_A3`` is
    given, its K are reused).  K defaults to 100 eV/Å³.
    """
    K_default: Any = DEFAULT_STIFFNESS_EV_PER_A3
    if isinstance(material, EnergyModel):
        if indenter_geometry_of(material) is not None:
            K_default = material.params["indenter"]["K"]
            material = material.params["material"]
        else:
            material = material.params
    K = K_default if K_eV_per_A3 is None else K_eV_per_A3
    return {"material": material,
            "indenter": {"apex": jnp.asarray(apex_A, dtype=jnp.float64), "K": jnp.asarray(K, dtype=jnp.float64)}}


def set_indenter(params: dict, apex_A: Any = None, K_eV_per_A3: Any = None) -> dict:
    """Composite params with the apex and/or K replaced (pure dict update; usable inside jit / lax.scan)."""
    ind = dict(params["indenter"])
    if apex_A is not None:
        ind["apex"] = apex_A
    if K_eV_per_A3 is not None:
        ind["K"] = K_eV_per_A3
    return {**params, "indenter": ind}


def with_indenter(
    material: Union[EnergyModel, EnergyFn],
    geometry: IndenterGeometry = IndenterGeometry(),
    *,
    apex_A: Any = PARKED_APEX_A,
    K_eV_per_A3: float = DEFAULT_STIFFNESS_EV_PER_A3,
) -> Union[EnergyModel, EnergyFn]:
    """Material + indenter.

    Given an energy function, returns the cached composite energy function
    (:func:`indented_energy_fn`).  Given an EnergyModel, returns an EnergyModel
    with that composite function, ``params = indenter_params(model.params,
    apex_A, K)`` (the tip parked far above the slab by default), the same
    elements, cutoff (the indenter never enters the neighbour list) and card
    (the composite is not a new potential), and name ``"<model>+<geometry>"``.
    Apply it to the model a protocol runner receives (never re-fetch the
    potential by name), so gate perturbations of the material carry over.
    """
    if isinstance(material, EnergyModel):
        return EnergyModel(
            name=f"{material.name}+{geometry.label}",
            elements=material.elements,
            cutoff=material.cutoff,
            params=indenter_params(material.params, apex_A, K_eV_per_A3),
            energy_fn=indented_energy_fn(material.energy_fn, geometry),
            card=material.card,
        )
    if callable(material):
        return indented_energy_fn(material, geometry)
    raise TypeError(f"with_indenter expects an EnergyModel or an energy function, got {type(material).__name__}")


def indenter_geometry_of(model: Union[EnergyModel, EnergyFn]) -> Optional[IndenterGeometry]:
    """Geometry of a composite model / energy function made by :func:`with_indenter` (None otherwise)."""
    fn = model.energy_fn if isinstance(model, EnergyModel) else model
    part = _COMPOSITE_PARTS.get(fn)
    return None if part is None else part[1]


def material_energy_fn_of(model: Union[EnergyModel, EnergyFn]) -> Optional[EnergyFn]:
    """Material energy function inside a composite (None if ``model`` has no indenter)."""
    fn = model.energy_fn if isinstance(model, EnergyModel) else model
    part = _COMPOSITE_PARTS.get(fn)
    return None if part is None else part[0]


# ----------------------------------------------------------------------------
# loads, forces, contact and guards (jit-compiled once per geometry)
# ----------------------------------------------------------------------------
def _resolve(tip: Union[IndenterGeometry, EnergyModel], params: Optional[dict]) -> tuple[IndenterGeometry, dict]:
    """(geometry, indenter sub-dict) from a geometry or a composite model plus optional params."""
    if isinstance(tip, IndenterGeometry):
        if params is None:
            raise ValueError("params (composite params or the indenter sub-dict) are required with a geometry")
        geometry = tip
    elif isinstance(tip, EnergyModel):
        g = indenter_geometry_of(tip)
        if g is None:
            raise ValueError(f"model {tip.name!r} has no indenter (use with_indenter)")
        geometry = g
        params = tip.params if params is None else params
    else:
        raise TypeError(f"expected an IndenterGeometry or a composite EnergyModel, got {type(tip).__name__}")
    ind = params["indenter"] if "indenter" in params else params  # type: ignore[operator]
    return geometry, ind


def _diagnostics(frac, box, ind, geometry):
    with jax.default_matmul_precision("highest"):
        return _diagnostics_body(frac, box, ind, geometry)


def _diagnostics_body(frac, box, ind, geometry):
    X = space.transform(box, frac)
    delta = penetration(X, ind["apex"], geometry, box)
    s2 = jnp.sum(delta**2)
    s3 = jnp.sum(delta**3)
    load = -jax.grad(lambda a: indenter_energy(X, a, ind["K"], geometry, box))(jnp.asarray(ind["apex"], X.dtype))
    return {
        "energy_eV": (ind["K"] / 3.0) * s3,
        "load_eV_per_A": load,  # (3,) force of the material on the tip; +z = indentation load
        "n_contact": jnp.sum(delta > 0.0),
        "max_penetration_A": jnp.max(delta),
        # force-weighted penetration sum(|F_i| delta_i) / sum(|F_i|), |F_i| = K delta_i^2: h_c = h - pen_fw
        "pen_fw_A": jnp.where(s2 > 0.0, s3 / jnp.where(s2 > 0.0, s2, 1.0), 0.0),
    }


@functools.lru_cache(maxsize=None)
def _kernels(geometry: IndenterGeometry) -> dict[str, Any]:
    def energy(frac, box, ind):
        return indenter_energy_frac(frac, box, ind, geometry)

    def load(frac, box, ind):
        return -jax.grad(lambda a: energy(frac, box, {**ind, "apex": a}))(jnp.asarray(ind["apex"], frac.dtype))

    def local(frac, box, ind):  # (z above the lowest point, uncapped sdf) per atom
        with jax.default_matmul_precision("highest"):
            X = space.transform(box, frac)
            d = _displacement(X, jnp.asarray(ind["apex"], X.dtype), box)
            return d[..., 2], _sdf_local(d, geometry)

    return {
        "energy": jax.jit(energy),
        "forces": jax.jit(lambda frac, box, ind: -jax.grad(energy)(frac, box, ind)),
        "load": jax.jit(load),
        "local": jax.jit(local),
        "diagnostics": jax.jit(lambda frac, box, ind: _diagnostics(frac, box, ind, geometry)),
    }


def indenter_load(tip: Union[IndenterGeometry, EnergyModel], frac, box, params: Optional[dict] = None) -> jnp.ndarray:
    """Force of the material on the tip, -∂E/∂apex (3,), eV/Å; the +z component is the indentation load.

    By Newton's third law it equals -Σ_i F_ind,i (the indenter forces on the
    atoms); 1 eV/Å = 1.602177 nN.  ``tip`` is the geometry (then ``params`` is
    required: composite params or the indenter sub-dict) or a composite model.
    """
    geometry, ind = _resolve(tip, params)
    return _kernels(geometry)["load"](jnp.asarray(frac), jnp.asarray(box), ind)


def indenter_forces(tip: Union[IndenterGeometry, EnergyModel], frac, box, params: Optional[dict] = None) -> jnp.ndarray:
    """Real-space forces (N, 3) of the indenter alone on the atoms (eV/Å)."""
    geometry, ind = _resolve(tip, params)
    return _kernels(geometry)["forces"](jnp.asarray(frac), jnp.asarray(box), ind)


def indenter_diagnostics(tip: Union[IndenterGeometry, EnergyModel], frac, box, params: Optional[dict] = None) -> dict:
    """Jittable per-configuration tip diagnostics.

    Keys: ``energy_eV``, ``load_eV_per_A`` (3,), ``n_contact`` (atoms with
    δ > 0), ``max_penetration_A`` and ``pen_fw_A`` (force-weighted penetration
    Σδ³/Σδ², the depth offset of the soft wall: h_c = h - pen_fw).
    """
    geometry, ind = _resolve(tip, params)
    return _kernels(geometry)["diagnostics"](jnp.asarray(frac), jnp.asarray(box), ind)


def contact_atoms(tip: Union[IndenterGeometry, EnergyModel], frac, box, params: Optional[dict] = None,
                  tol_A: float = 0.0) -> np.ndarray:
    """Boolean mask (N,) of atoms with sdf < ``tol_A`` below the height cap (tol 0: atoms feeling the wall)."""
    geometry, ind = _resolve(tip, params)
    dz, sdf = _kernels(geometry)["local"](jnp.asarray(frac), jnp.asarray(box), ind)
    return np.asarray((sdf < tol_A) & (dz < geometry.height_cap_A))


def z_guard_report(tip: Union[IndenterGeometry, EnergyModel], frac, box, params: Optional[dict] = None) -> dict[str, Any]:
    """Atoms the capped tip cannot see.

    ``n_hidden`` counts atoms at or above the height cap that lie inside the
    (semi-infinite) tip column, i.e. inside the tip were it not capped; it
    must be 0.  Also returns ``n_above_cap``, ``max_height_above_apex_A``,
    ``height_cap_A`` and the indices ``hidden_atoms``.
    """
    geometry, ind = _resolve(tip, params)
    dz, sdf = (np.asarray(a) for a in _kernels(geometry)["local"](jnp.asarray(frac), jnp.asarray(box), ind))
    above = dz >= geometry.height_cap_A
    hidden = above & (sdf < 0.0)
    return {
        "n_hidden": int(hidden.sum()),
        "n_above_cap": int(above.sum()),
        "max_height_above_apex_A": float(dz.max()) if dz.size else float("-inf"),
        "height_cap_A": geometry.height_cap_A,
        "hidden_atoms": np.nonzero(hidden)[0],
    }


def check_z_guard(tip: Union[IndenterGeometry, EnergyModel], frac, box, params: Optional[dict] = None) -> dict[str, Any]:
    """:func:`z_guard_report`, raising :class:`IndenterGuardError` if any atom is hidden above the cap."""
    rep = z_guard_report(tip, frac, box, params)
    if rep["n_hidden"]:
        raise IndenterGuardError(
            f"{rep['n_hidden']} atom(s) inside the tip column at or above the height cap "
            f"({rep['height_cap_A']:.1f} Å above the apex; first indices {rep['hidden_atoms'][:8].tolist()}): "
            "the capped indenter cannot push them (atom entered from above / through the periodic top)"
        )
    return rep


def check_indenter_cell(box: Any, geometry: IndenterGeometry, apex_A: Any, *, tol_A: float = 1e-9) -> None:
    """Raise ValueError unless the cell suits the indenter.

    Requires a1, a2 in the xy plane and a3 along z, an orthorhombic in-plane
    cell (|off-diagonal| <= ``tol_A``; the lateral image of the tip axis is then
    the exact minimum image), and the capped tip inside the cell
    (apex_z + height_cap_A < Lz).
    """
    b = np.asarray(box, dtype=float)  # columns = lattice vectors
    off = np.abs(b - np.diag(np.diag(b)))
    if off.max() > tol_A:
        raise ValueError(f"indentation needs an orthorhombic cell with a3 along z (max |off-diagonal| "
                         f"{off.max():.3e} Å); use structures.indentation.orthorhombic_oriented_cell")
    apex = np.asarray(apex_A, dtype=float)
    if apex[2] + geometry.height_cap_A >= b[2, 2]:
        raise ValueError(f"capped tip reaches the cell top: apex_z {apex[2]:.3f} + cap {geometry.height_cap_A:.3f} "
                         f">= Lz {b[2, 2]:.3f} Å; add vacuum (structures.indentation sizes it)")


# ----------------------------------------------------------------------------
# rounded-tip area function (exact, numpy)
# ----------------------------------------------------------------------------
def tip_area_function(h_c: Any, geometry: IndenterGeometry) -> np.ndarray:
    """Exact projected area A(h_c) (Å²) of the part of the tip below height h_c above its lowest point.

    For the pyramid and the cone this is the horizontal cross-section at h_c.
    Sphere: π(2R h - h²) up to h = R, then πR² (the equator).  Rounded cone of half-angle α:
    πr² with r = (h - R) tanα + R/cosα above the sphere-cone tangency
    h = R(1 - sinα), r² = 2Rh - h² below it.  Rounded Berkovich: from the
    support function of the section, h_S(ψ) = a(ψ)(h - R) + R√(1 + a²) with
    a = 2 tanθ cosψ (ψ from a corner direction), or √(2Rh - h²) where the
    apex sphere is the outermost part; A = ½∮(h_S² - h_S'²) integrates in
    closed form.  For h ≥ R(1 - sinθ) (0.0917 R) the result is
    A = C0 (h - R)² + 6√3 tanθ/cosθ R (h - R) + R²[3√3 tan²θ + 3√B arctan(√3/√B)],
    B = 1 + 4tan²θ; A/(C0 h²) = 1.200 / 1.102 / 1.041 / 1.021 at h = R, 2R, 5R,
    10R (65.27°) and → 1 for R → 0.
    """
    z = np.asarray(h_c, dtype=float)
    R = geometry.tip_radius_A
    zp = np.maximum(z, 0.0)
    if geometry.shape == "sphere":
        r2 = np.where(zp < R, 2.0 * R * zp - zp * zp, R * R)
        return np.pi * r2
    th = math.radians(geometry.half_angle_deg)  # type: ignore[arg-type]
    cap2 = np.maximum(2.0 * R * zp - zp * zp, 0.0)  # squared radius of the apex-sphere section
    if geometry.shape == "cone":
        r = (zp - R) * math.tan(th) + R / math.cos(th)
        return np.where(zp >= R * (1.0 - math.sin(th)), np.pi * r * r, np.pi * cap2)

    A0 = 2.0 * math.tan(th)
    B = 1.0 + A0 * A0
    Z = zp - R
    if R > 0.0:
        # corner sector psi in [0, psi*] follows faces/edges; [psi*, pi/3] the apex sphere
        k = np.clip(1.0 - zp / R, -1.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            a_star = np.where(k > 0.0, k / np.sqrt(np.maximum(1.0 - k * k, 1e-300)), 0.0)
        psi = np.arccos(np.clip(a_star / A0, 0.0, 1.0))
        psi = np.minimum(psi, math.pi / 3.0)
    else:
        psi = np.full_like(zp, math.pi / 3.0)
    F = (0.5 * A0 * A0 * Z * Z * np.sin(2.0 * psi)
         + 2.0 * A0 * Z * R * np.sin(psi) * np.sqrt(1.0 + (A0 * np.cos(psi)) ** 2)
         + R * R * (0.5 * A0 * A0 * np.sin(2.0 * psi) + math.sqrt(B) * np.arctan(np.tan(psi) / math.sqrt(B))))
    area = 3.0 * (F + (math.pi / 3.0 - psi) * cap2)
    return np.where(z > 0.0, area, 0.0)
