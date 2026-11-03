"""Zero-temperature elastic constants: exact autodiff (relaxed-ion) and finite differences.

Physics and conventions
-----------------------
Homogeneous strain is applied to the cell at fixed fractional coordinates,

    box(e) = (I + eps(e)) @ box,     eps = [[e1, e6/2, e5/2],
                                            [e6/2, e2, e4/2],
                                            [e5/2, e4/2, e3]]

with Voigt order (xx, yy, zz, yz, xz, xy) and *engineering* shear strains
(e4 = 2 eps_yz, ...).  Internal (non-affine) displacements ``u`` are real-space
displacements added on top of the affine map.  To second order,

    E(u, e) = E0 + V σ·e + ½ e·H_ee·e + u·Λ·e + ½ u·K·u     (force-free: ∂E/∂u = 0)

    K   = ∂²E/∂u∂u        (3N x 3N, eV/Å²)      = jax.hessian w.r.t. frac
    Λ   = ∂²E/∂u∂e        (3N x 6,  eV/Å)       = jacfwd_e of grad_frac at fixed frac
    H_ee = ∂²E/∂e∂e       (6 x 6,   eV)

Under JAX-MD's ``periodic_general(fractional_coordinates=True)`` custom JVP a
tangent of ``frac`` *is* a real-space displacement, so derivatives with respect
to ``frac`` are real-space derivatives (verified against real-space finite
differences in tests).  We never differentiate through a user-side x -> frac map.

Minimising over u gives the relaxed-ion (Kleinman / Martin) result

    C_clamped = H_ee / V,      C_relaxed = (H_ee - Λᵀ K⁺ Λ) / V,

where K⁺ is the pseudo-inverse on the complement of the three rigid
translations (removed with an explicit projector).  u = Γ e with Γ = -K⁺Λ is
the internal-strain response.

Validity: the expansion is taken about an equilibrium configuration.  The
identification ``C = (1/V) ∂²E/∂e²`` with the stress-strain coefficients
dσ/de requires zero initial stress (otherwise stress-dependent terms appear
that depend on the strain measure); both the residual force and the residual
stress are therefore computed and reported.

Finite differences: central differences of the Cauchy stress
σ = (1/V) ∂E/∂e (exact cell derivative at the current configuration) for ±δ
along each Voigt component, with internal coordinates relaxed at every
strained cell (force-converged), for several δ to expose truncation (O(δ²))
and noise (O(tol/δ)) errors.
"""

from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fracture_md.potentials.base import EnergyModel
from fracture_md.potentials.neighbors import build_neighbors
from fracture_md.structures.symmetry import (
    VOIGT_LABELS,
    check_tensor_invariance,
    cubic_pattern_deviation,
    voigt_to_tensor_compliance,
    voigt_to_tensor_stiffness,
)
from fracture_md.structures.system import AtomicSystem
from fracture_md.units import EV_PER_A3_TO_GPA



def _zeros6():
    return jnp.zeros(6, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Strain parameterisation
# ---------------------------------------------------------------------------
def eps_from_voigt(e):
    """Symmetric strain tensor from a Voigt vector with engineering shears (JAX)."""
    return jnp.array(
        [
            [e[0], e[5] / 2.0, e[4] / 2.0],
            [e[5] / 2.0, e[1], e[3] / 2.0],
            [e[4] / 2.0, e[3] / 2.0, e[2]],
        ]
    )


def strained_box(box, e):
    """box(e) = (I + eps(e)) @ box  (lattice vectors are columns)."""
    return (jnp.eye(3, dtype=box.dtype) + eps_from_voigt(e)) @ box


# ---------------------------------------------------------------------------
# Jitted derivative primitives.  `efn` (the model's pure energy function) is a
# static argument so each model compiles once per system shape; box, frac,
# neighbour indices and parameters are traced arguments (no recompilation when
# the cell changes).
# ---------------------------------------------------------------------------
@partial(jax.jit, static_argnums=0)
def _energy(efn, frac, box, species, idx, params):
    return efn(frac, box, species, idx, params)


@partial(jax.jit, static_argnums=0)
def _grad_frac(efn, frac, box, species, idx, params):
    """Real-space gradient dE/dR (eV/Å) under JAX-MD's custom JVP."""
    return jax.grad(efn, argnums=0)(frac, box, species, idx, params)


@partial(jax.jit, static_argnums=0)
def _dE_de(efn, frac, box, species, idx, params):
    """dE/de at e=0 (eV); σ_Voigt = dE/de / V (engineering shear convention)."""
    return jax.grad(lambda e: efn(frac, strained_box(box, e), species, idx, params))(_zeros6())


@partial(jax.jit, static_argnums=0)
def _hess_ee(efn, frac, box, species, idx, params):
    return jax.hessian(lambda e: efn(frac, strained_box(box, e), species, idx, params))(_zeros6())


@partial(jax.jit, static_argnums=0)
def _mixed_ue(efn, frac, box, species, idx, params):
    """Λ[i, a, v] = ∂²E / ∂u_{ia} ∂e_v at fixed frac, shape (N, 3, 6)."""
    return jax.jacfwd(lambda e: jax.grad(efn, argnums=0)(frac, strained_box(box, e), species, idx, params))(
        _zeros6()
    )


@partial(jax.jit, static_argnums=0)
def _hvp_batch(efn, frac, box, species, idx, params, tangents):
    """Hessian-vector products K @ v for a batch of real-space tangents (k, N, 3)."""

    def one(v):
        return jax.jvp(lambda f: jax.grad(efn, argnums=0)(f, box, species, idx, params), (frac,), (v,))[1]

    return jax.vmap(one)(tangents)


def _hessian_batch_size(n_atoms: int, max_nbrs: int, budget: float = 2.0e7) -> int:
    per = max(1, n_atoms * max(1, max_nbrs) ** 2)
    return int(max(1, min(3 * n_atoms, budget // per)))


def position_hessian(model: EnergyModel, system: AtomicSystem, idx, params=None, species=None) -> np.ndarray:
    """Exact real-space Hessian K = ∂²E/∂R∂R, shape (3N, 3N), eV/Å², computed in batches."""
    p = model.params if params is None else params
    sp = model.species_for(system) if species is None else species
    n = system.n_atoms
    ndof = 3 * n
    bs = _hessian_batch_size(n, int(idx.shape[1]))
    eye = np.eye(ndof)
    cols = []
    for start in range(0, ndof, bs):
        block = eye[start : start + bs]
        k = block.shape[0]
        if k < bs:  # pad to a fixed batch shape (single compilation)
            block = np.vstack([block, np.zeros((bs - k, ndof))])
        out = _hvp_batch(model.energy_fn, system.frac, system.box, sp, idx, p, jnp.asarray(block.reshape(bs, n, 3)))
        cols.append(np.asarray(out).reshape(bs, ndof)[:k])
    H = np.vstack(cols)  # rows = K @ e_k = column k (K symmetric)
    return 0.5 * (H + H.T)


# ---------------------------------------------------------------------------
# Relaxed-ion correction
# ---------------------------------------------------------------------------
def translation_projector(n_atoms: int) -> np.ndarray:
    """P = I - T Tᵀ removing the three rigid translations from 3N-vectors."""
    T = np.zeros((3 * n_atoms, 3))
    for a in range(3):
        T[a::3, a] = 1.0 / np.sqrt(n_atoms)
    return np.eye(3 * n_atoms) - T @ T.T


@dataclasses.dataclass
class HessianSpectrum:
    eigenvalues: np.ndarray  # eV/Å², ascending, translations projected (3 exact zeros)
    n_zero_modes: int  # modes with |λ| < tol, *including* the 3 translations
    n_negative_modes: int  # λ < -tol: configuration is not a minimum
    min_nonzero: float  # smallest eigenvalue after removing the 3 translations
    tol: float


def relaxed_ion_correction(K: np.ndarray, Lam: np.ndarray, rel_tol: float = 1e-8):
    """Return (Λᵀ K⁺ Λ  [eV],  Γ = -K⁺Λ  [Å per unit strain],  spectrum, K⁺).

    K⁺ is built from the eigen-decomposition of P K P with P the translation
    projector; the three translational modes (and any further |λ| < tol modes)
    are excluded.
    """
    n = K.shape[0] // 3
    P = translation_projector(n)
    Kp = P @ K @ P
    Kp = 0.5 * (Kp + Kp.T)
    w, U = np.linalg.eigh(Kp)
    tol = rel_tol * max(1.0, float(np.max(np.abs(w))))
    # The three translation modes are exact zeros of P K P; identify them as the
    # modes with the largest overlap with the translation subspace.
    T = np.eye(3 * n) - P
    overlap = np.einsum("ik,ij,jk->k", U, T, U)
    trans = np.argsort(-overlap)[:3]
    keep = np.ones(w.size, bool)
    keep[trans] = False
    keep &= np.abs(w) > tol
    Kinv = (U[:, keep] / w[keep]) @ U[:, keep].T
    Lp = P @ Lam
    corr = Lp.T @ Kinv @ Lp
    Gamma = -Kinv @ Lp
    w_nt = np.delete(w, trans)
    spec = HessianSpectrum(
        eigenvalues=np.sort(w),
        n_zero_modes=int(np.sum(np.abs(w) <= tol)),
        n_negative_modes=int(np.sum(w_nt < -tol)),
        min_nonzero=float(np.min(w_nt)) if w_nt.size else float("nan"),
        tol=tol,
    )
    return 0.5 * (corr + corr.T), Gamma, spec, Kinv


# ---------------------------------------------------------------------------
# Autodiff elastic constants
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class AutodiffElasticResult:
    C_clamped: np.ndarray  # (6,6) GPa
    C_relaxed: np.ndarray  # (6,6) GPa
    internal_strain: np.ndarray  # Λ (N,3,6) eV/Å
    internal_displacement: np.ndarray  # Γ = -K⁺Λ (N,3,6) Å per unit Voigt strain
    hessian_eigenvalues: np.ndarray  # (3N,) eV/Å²
    spectrum: HessianSpectrum
    stress_GPa: np.ndarray  # (6,) residual Cauchy stress (Voigt)
    max_force: float  # eV/Å residual
    volume: float  # Å^3
    n_atoms: int
    energy: float  # eV

    def as_dict(self) -> dict[str, Any]:
        return {
            "C_clamped_GPa": self.C_clamped.tolist(),
            "C_relaxed_GPa": self.C_relaxed.tolist(),
            "residual_stress_GPa": self.stress_GPa.tolist(),
            "max_force_eV_per_A": self.max_force,
            "n_zero_modes": self.spectrum.n_zero_modes,
            "n_negative_modes": self.spectrum.n_negative_modes,
            "min_nonzero_hessian_eigenvalue_eV_per_A2": self.spectrum.min_nonzero,
            "volume_A3": self.volume,
            "n_atoms": self.n_atoms,
        }


def strain_derivatives(model: EnergyModel, system: AtomicSystem, idx, params=None):
    """(E, forces (N,3), dE/de (6), H_ee (6,6), Λ (N,3,6), K (3N,3N)) as numpy, eV/Å units."""
    p = model.params if params is None else params
    sp = model.species_for(system)
    efn = model.energy_fn
    args = (system.frac, system.box, sp, idx, p)
    e = float(_energy(efn, *args))
    g = np.asarray(_grad_frac(efn, *args))
    dEde = np.asarray(_dE_de(efn, *args))
    Hee = np.asarray(_hess_ee(efn, *args))
    Lam = np.asarray(_mixed_ue(efn, *args))
    K = position_hessian(model, system, idx, params=p, species=sp)
    return e, -g, dEde, 0.5 * (Hee + Hee.T), Lam, K


def elastic_constants_autodiff(
    model: EnergyModel, system: AtomicSystem, idx=None, *, skin: float = 0.5, params=None
) -> AutodiffElasticResult:
    """Clamped- and relaxed-ion 6x6 elastic constants (GPa) from exact second derivatives.

    ``system`` must be an equilibrium configuration (zero force) at zero stress
    for C to equal the stress-strain coefficients; residuals are returned.
    """
    if idx is None:
        idx = build_neighbors(system, model.cutoff, skin=skin).idx
    e, f, dEde, Hee, Lam, K = strain_derivatives(model, system, idx, params=params)
    V = system.volume
    corr, Gamma, spec, _ = relaxed_ion_correction(K, Lam.reshape(-1, 6))
    n = system.n_atoms
    return AutodiffElasticResult(
        C_clamped=Hee / V * EV_PER_A3_TO_GPA,
        C_relaxed=(Hee - corr) / V * EV_PER_A3_TO_GPA,
        internal_strain=Lam,
        internal_displacement=Gamma.reshape(n, 3, 6),
        hessian_eigenvalues=spec.eigenvalues,
        spectrum=spec,
        stress_GPa=dEde / V * EV_PER_A3_TO_GPA,
        max_force=float(np.max(np.linalg.norm(f, axis=1))),
        volume=V,
        n_atoms=n,
        energy=e,
    )


# ---------------------------------------------------------------------------
# Internal relaxation at fixed cell (quasi-Newton with a fixed Hessian)
# ---------------------------------------------------------------------------
def newton_relax_fixed_cell(
    model: EnergyModel,
    system: AtomicSystem,
    idx,
    *,
    fmax: float = 1e-8,
    K_pinv: Optional[np.ndarray] = None,
    max_iter: int = 40,
    max_step: float = 0.05,
    params=None,
    refresh_hessian_every: int = 8,
) -> tuple[AtomicSystem, float, int]:
    """Relax positions at fixed cell with (quasi-)Newton steps on real-space coordinates.

    ``K_pinv`` (translation-projected Hessian pseudo-inverse) may be supplied
    from a nearby configuration; the iteration then converges linearly with a
    ratio of O(|ΔK|/|K|) and the Hessian is refreshed every
    ``refresh_hessian_every`` iterations or when the force stops decreasing.
    Convergence is judged on the true forces only, so the converged state does
    not depend on the matrix used to reach it.  Returns (system, fmax, iters).
    Suitable for small periodic cells near a minimum (used for FD elastic
    constants and cell relaxation); use ``relax_positions`` far from minima.
    """
    p = model.params if params is None else params
    sp = model.species_for(system)
    efn = model.energy_fn
    binv = np.linalg.inv(np.asarray(system.box))
    frac = np.asarray(system.frac)
    n = system.n_atoms
    fm_prev = np.inf
    it = 0
    for it in range(max_iter + 1):
        g = np.asarray(_grad_frac(efn, jnp.asarray(frac), system.box, sp, idx, p))
        fm = float(np.max(np.linalg.norm(g, axis=1)))
        if fm <= fmax:
            break
        if it == max_iter:
            break
        if K_pinv is None or (it > 0 and it % refresh_hessian_every == 0) or fm > 0.5 * fm_prev:
            K = position_hessian(model, system.replace(frac=jnp.asarray(frac)), idx, params=p, species=sp)
            _, _, _, K_pinv = relaxed_ion_correction(K, np.zeros((3 * n, 6)))
        fm_prev = fm
        step = -(K_pinv @ g.ravel()).reshape(n, 3)
        smax = float(np.max(np.abs(step)))
        if smax > max_step:
            step *= max_step / smax
        frac = frac + step @ binv.T
    return system.replace(frac=jnp.asarray(frac % 1.0)), fm, it


# ---------------------------------------------------------------------------
# Finite-difference elastic constants
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class FiniteDifferenceElasticResult:
    C: np.ndarray  # (6,6) GPa at the primary magnitude
    primary_magnitude: float
    magnitudes: list[float]
    per_magnitude: dict[float, np.ndarray]  # δ -> (6,6) GPa
    relax_internal: bool
    max_residual_force: float  # eV/Å over all strained, relaxed configurations
    convergence_spread: float  # max |C(δ) - C(primary)| over δ, GPa
    stress0_GPa: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "C_GPa": self.C.tolist(),
            "primary_magnitude": self.primary_magnitude,
            "magnitudes": self.magnitudes,
            "per_magnitude_GPa": {str(k): v.tolist() for k, v in self.per_magnitude.items()},
            "relax_internal": self.relax_internal,
            "max_residual_force_eV_per_A": self.max_residual_force,
            "convergence_spread_GPa": self.convergence_spread,
        }


def cauchy_stress_voigt(model: EnergyModel, system: AtomicSystem, idx, params=None) -> np.ndarray:
    """Cauchy stress (Voigt, eV/Å^3) = (1/V) ∂E/∂e at the current configuration; positive = tensile."""
    p = model.params if params is None else params
    sp = model.species_for(system)
    return np.asarray(_dE_de(model.energy_fn, system.frac, system.box, sp, idx, p)) / system.volume


def elastic_constants_finite_difference(
    model: EnergyModel,
    system: AtomicSystem,
    strains: Sequence[float] = (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3),
    relax_internal: bool = True,
    *,
    fmax: float = 1e-9,
    idx=None,
    skin: float = 0.5,
    params=None,
    primary: Optional[float] = None,
) -> FiniteDifferenceElasticResult:
    """C_ij = [σ_i(+δ e_j) - σ_i(-δ e_j)] / (2δ) with internal relaxation at each strained cell.

    The neighbour list of the reference configuration is reused; this is
    exact as long as no pair distance changes by more than ``skin`` which is
    asserted (max |δ| * (cutoff + skin) < skin / 2).
    """
    p = model.params if params is None else params
    if idx is None:
        idx = build_neighbors(system, model.cutoff, skin=skin).idx
    strains = [float(s) for s in strains]
    if max(strains) * (model.cutoff + skin) >= skin / 2:
        raise ValueError("strain magnitude too large for a fixed neighbour list; reduce strains")
    n = system.n_atoms
    K_pinv = None
    if relax_internal:
        K = position_hessian(model, system, idx, params=p)
        _, _, _, K_pinv = relaxed_ion_correction(K, np.zeros((3 * n, 6)))
    sig0 = cauchy_stress_voigt(model, system, idx, params=p)
    per: dict[float, np.ndarray] = {}
    worst_f = 0.0
    for d in strains:
        C = np.zeros((6, 6))
        for j in range(6):
            sig = []
            for sgn in (+1.0, -1.0):
                e = np.zeros(6)
                e[j] = sgn * d
                s_e = system.replace(box=strained_box(system.box, jnp.asarray(e)))
                if relax_internal:
                    s_e, fm, _ = newton_relax_fixed_cell(model, s_e, idx, fmax=fmax, K_pinv=K_pinv, params=p)
                    worst_f = max(worst_f, fm)
                sig.append(cauchy_stress_voigt(model, s_e, idx, params=p))
            C[:, j] = (sig[0] - sig[1]) / (2.0 * d)
        per[d] = C * EV_PER_A3_TO_GPA
    if primary is None:
        primary = sorted(strains)[len(strains) // 2]
    Cp = per[primary]
    spread = max(float(np.max(np.abs(C - Cp))) for C in per.values())
    return FiniteDifferenceElasticResult(
        C=Cp,
        primary_magnitude=primary,
        magnitudes=strains,
        per_magnitude=per,
        relax_internal=relax_internal,
        max_residual_force=worst_f,
        convergence_spread=spread,
        stress0_GPa=sig0 * EV_PER_A3_TO_GPA,
    )


# ---------------------------------------------------------------------------
# Derived moduli
# ---------------------------------------------------------------------------
def _fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)


def directional_youngs(S_tensor: np.ndarray, n: np.ndarray) -> np.ndarray:
    """E(n) = 1 / (S_ijkl n_i n_j n_k n_l) for unit vectors n (..., 3)."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    return 1.0 / np.einsum("...i,...j,...k,...l,ijkl->...", n, n, n, n, S_tensor, optimize=True)


def directional_shear(S_tensor: np.ndarray, n: np.ndarray, m: np.ndarray) -> float:
    """G(n, m) = 1 / (4 S_ijkl n_i m_j n_k m_l): shear on the plane ⟂ n in direction m (n ⟂ m)."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    m = np.asarray(m, float) / np.linalg.norm(m)
    return float(1.0 / (4.0 * np.einsum("i,j,k,l,ijkl->", n, m, n, m, S_tensor)))


def directional_poisson(S_tensor: np.ndarray, n: np.ndarray, m: np.ndarray) -> float:
    """ν(n, m) = -S_ijkl n_i n_j m_k m_l / S_ijkl n_i n_j n_k n_l (load along n, lateral m)."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    m = np.asarray(m, float) / np.linalg.norm(m)
    return float(-np.einsum("i,j,k,l,ijkl->", n, n, m, m, S_tensor) / np.einsum("i,j,k,l,ijkl->", n, n, n, n, S_tensor))


def _youngs_extrema(S_tensor: np.ndarray, n_grid: int = 4000):
    from scipy.optimize import minimize

    pts = _fibonacci_sphere(n_grid)
    E = directional_youngs(S_tensor, pts)

    def refine(start, sign):
        th0 = np.arccos(np.clip(start[2], -1, 1))
        ph0 = np.arctan2(start[1], start[0])

        def f(x):
            v = np.array([np.sin(x[0]) * np.cos(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[0])])
            return sign * float(directional_youngs(S_tensor, v))

        r = minimize(f, [th0, ph0], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=2000))
        v = np.array([np.sin(r.x[0]) * np.cos(r.x[1]), np.sin(r.x[0]) * np.sin(r.x[1]), np.cos(r.x[0])])
        return sign * r.fun, v

    emin, nmin = refine(pts[np.argmin(E)], +1.0)
    emax, nmax = refine(pts[np.argmax(E)], -1.0)
    return float(emin), nmin, float(emax), nmax


HARDNESS_CHEN_NIU = (
    "Chen-Niu (2011) empirical hardness model: Hv = 2 (k^2 G)^0.585 - 3, k = G/B (Hill moduli, GPa)",
    "X.-Q. Chen, H. Niu, D. Li, Y. Li, Intermetallics 19, 1275-1281 (2011), doi:10.1016/j.intermet.2011.03.026",
)
HARDNESS_TIAN = (
    "Tian et al. (2012) variant: Hv = 0.92 k^1.137 G^0.708",
    "Y. Tian, B. Xu, Z. Zhao, Int. J. Refract. Met. Hard Mater. 33, 93-106 (2012), doi:10.1016/j.ijrmhm.2012.02.021",
)
# Leave-one-out score of the literature Chen-Niu model (published constants, not refitted) against the 14
# hardness-calibration targets (Vickers values chosen from measured records; DFT-PBE moduli): results.json
# summary.primary.chen_niu_lit of examples/16_hardness_calibration.py, pinned by tests/test_hardness_models.py.
# The calibrated alternative with an honest interval is fracture_md.discovery.hardness_models
# (vickers_hardness_estimate records; docs/HARDNESS_CALIBRATION.md).
CHEN_NIU_LITERATURE_SCORE = dict(n=14, MAPE_pct=35.0, bias_pct=26.5, pi90_factor=1.85, pi90_factor_conformal=2.58,
                                 source="data/hardness_calibration/results.json summary.primary.chen_niu_lit")


def chen_niu_score_uncertainty():
    """The measured error of the literature Chen-Niu model as an Uncertainty (reference_discrepancy; relative = the
    mean absolute leave-one-out error), attached to every ``hardness_estimate`` record."""
    from fracture_md.schema.common import Uncertainty, UncertaintyKind

    s = CHEN_NIU_LITERATURE_SCORE
    return Uncertainty(
        kind=UncertaintyKind.REFERENCE_DISCREPANCY, relative=s["MAPE_pct"] / 100.0, interval="n/a",
        description=("literature Chen-Niu scored against %d hardness-calibration targets (Vickers values chosen from "
                     "measured records, DFT-PBE moduli): mean absolute error %.0f %%, mean bias %+.0f %% (it "
                     "over-predicts), 90 %% prediction interval x/÷ %.2f around the bias-corrected value (conformal "
                     "x/÷ %.2f); relative = the mean absolute error (%s).  For a calibrated estimate use "
                     "fracture_md.discovery.hardness_models (docs/HARDNESS_CALIBRATION.md)")
        % (s["n"], s["MAPE_pct"], s["bias_pct"], s["pi90_factor"], s["pi90_factor_conformal"], s["source"]))


def analyze_elastic(C_voigt_GPa: np.ndarray, density_g_cm3: Optional[float] = None) -> dict[str, Any]:
    """Derived elastic properties of a 6x6 stiffness matrix (GPa).

    Returns a dict (all moduli in GPa, compliance in 1/GPa, velocities in m/s):
    compliance, Voigt/Reuss/Hill B and G, Hill E and ν, universal anisotropy
    A^U = 5 G_V/G_R + B_V/B_R - 6 (Ranganathan & Ostoja-Starzewski, PRL 101,
    055504 (2008)), Pugh ratio B/G, Zener ratio 2C44/(C11-C12) if the matrix
    has cubic form, directional E along <100>, <110>, <111> and its extrema
    over the unit sphere, representative directional G and ν, Born stability
    (min eigenvalue of C; cubic criteria), and an empirical hardness estimate.

    Assumptions: linear elasticity of a single crystal; VRH averages describe
    an untextured, fully dense polycrystal; the hardness correlation is an
    empirical fit (Chen et al. 2011) and not a computed plastic property.

    If C is not positive definite (Born-unstable) the extremum search of E(n)
    is skipped (E(n) has poles on the sphere): E_min / E_max are NaN.  A
    negative Chen-Niu value (outside the model's range, soft or ductile
    solids) is returned as NaN, with the raw value in
    ``hardness_chen_niu_raw`` and ``hardness_chen_niu_applicable`` False.
    """
    C = np.asarray(C_voigt_GPa, dtype=float)
    C = 0.5 * (C + C.T)
    S = np.linalg.inv(C)
    BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    GV = ((C[0, 0] + C[1, 1] + C[2, 2]) - (C[0, 1] + C[0, 2] + C[1, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0
    BR = 1.0 / ((S[0, 0] + S[1, 1] + S[2, 2]) + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    GR = 15.0 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2]) + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    BH, GH = 0.5 * (BV + BR), 0.5 * (GV + GR)
    EH = 9 * BH * GH / (3 * BH + GH)
    nuH = (3 * BH - 2 * GH) / (2 * (3 * BH + GH))
    AU = 5 * GV / GR + BV / BR - 6.0
    eig = np.linalg.eigvalsh(C)
    St = voigt_to_tensor_compliance(S)
    dirs = {"[100]": [1, 0, 0], "[110]": [1, 1, 0], "[111]": [1, 1, 1]}
    E_dir = {k: float(directional_youngs(St, np.asarray(v, float))) for k, v in dirs.items()}
    if eig[0] > 0:
        emin, nmin, emax, nmax = _youngs_extrema(St)
    else:  # Born-unstable: E(n) has poles; a Nelder-Mead search would run into them
        emin = emax = float("nan")
        nmin = nmax = np.full(3, np.nan)
    shear_pairs = {
        "(100)[010]": ([1, 0, 0], [0, 1, 0]),
        "(110)[1-10]": ([1, 1, 0], [1, -1, 0]),
        "(110)[001]": ([1, 1, 0], [0, 0, 1]),
        "(111)[1-10]": ([1, 1, 1], [1, -1, 0]),
    }
    G_dir = {k: directional_shear(St, n, m) for k, (n, m) in shear_pairs.items()}
    poisson_pairs = {
        "[100]/[010]": ([1, 0, 0], [0, 1, 0]),
        "[110]/[1-10]": ([1, 1, 0], [1, -1, 0]),
        "[110]/[001]": ([1, 1, 0], [0, 0, 1]),
        "[111]/[1-10]": ([1, 1, 1], [1, -1, 0]),
    }
    nu_dir = {k: directional_poisson(St, n, m) for k, (n, m) in poisson_pairs.items()}
    cubic_dev = cubic_pattern_deviation(C)
    is_cubic = cubic_dev < 1e-6 * max(1.0, float(np.max(np.abs(C))))
    out: dict[str, Any] = {
        "C_GPa": C.tolist(),
        "S_per_GPa": S.tolist(),
        "B_voigt": BV,
        "B_reuss": BR,
        "B_hill": BH,
        "G_voigt": GV,
        "G_reuss": GR,
        "G_hill": GH,
        "E_hill": EH,
        "nu_hill": nuH,
        "universal_anisotropy": AU,
        "pugh_ratio_B_over_G": BH / GH,
        "E_directional": E_dir,
        "E_min": emin,
        "E_min_direction": (nmin / np.max(np.abs(nmin))).tolist() if np.all(np.isfinite(nmin)) else nmin.tolist(),
        "E_max": emax,
        "E_max_direction": (nmax / np.max(np.abs(nmax))).tolist() if np.all(np.isfinite(nmax)) else nmax.tolist(),
        "G_directional": G_dir,
        "nu_directional": nu_dir,
        "C_eigenvalues": eig.tolist(),
        "born_min_eigenvalue": float(eig[0]),
        "born_stable": bool(eig[0] > 0),
        "cubic_pattern_deviation": cubic_dev,
        "is_cubic": bool(is_cubic),
    }
    if is_cubic:
        c11 = float(np.mean(np.diag(C)[:3]))
        c12 = float(np.mean([C[0, 1], C[0, 2], C[1, 2]]))
        c44 = float(np.mean(np.diag(C)[3:]))
        out.update(
            C11=c11,
            C12=c12,
            C44=c44,
            zener_ratio=2 * c44 / (c11 - c12),
            born_cubic={"C11-C12>0": c11 - c12 > 0, "C11+2C12>0": c11 + 2 * c12 > 0, "C44>0": c44 > 0},
            cauchy_pressure_C12_minus_C44=c12 - c44,
        )
    k = GH / BH
    hv_raw = float(2.0 * (k * k * GH) ** 0.585 - 3.0) if GH > 0 and BH > 0 else float("nan")
    hv_ok = bool(np.isfinite(hv_raw) and hv_raw >= 0.0)
    out["hardness_chen_niu_raw"] = hv_raw
    out["hardness_chen_niu_applicable"] = hv_ok
    out["hardness_chen_niu"] = hv_raw if hv_ok else float("nan")
    out["hardness_tian"] = float(0.92 * k**1.137 * GH**0.708) if GH > 0 and BH > 0 else float("nan")
    if density_g_cm3 is not None and GH > 0:
        rho = density_g_cm3 * 1e3  # kg/m^3
        vt = np.sqrt(GH * 1e9 / rho)
        vl = np.sqrt((BH + 4 * GH / 3) * 1e9 / rho)
        vm = (1.0 / 3.0 * (2.0 / vt**3 + 1.0 / vl**3)) ** (-1.0 / 3.0)
        out.update(v_transverse_m_s=float(vt), v_longitudinal_m_s=float(vl), v_mean_m_s=float(vm))
    return out


def hardness_applicability_check(an: dict[str, Any]):
    """'model_applicability' CheckResult for the Chen-Niu hardness estimate of an ``analyze_elastic`` dict.

    Fails when the correlation gives a negative (or undefined) hardness, i.e.
    the moduli lie outside the range the empirical fit covers.
    """
    from fracture_md.schema.common import CheckResult
    from fracture_md.units import Unit

    raw = an.get("hardness_chen_niu_raw", an.get("hardness_chen_niu"))
    ok = bool(an.get("hardness_chen_niu_applicable", raw is not None and np.isfinite(raw) and raw >= 0))
    return CheckResult(name="model_applicability", passed=ok, metric=None if raw is None or not np.isfinite(raw) else float(raw),
                       tolerance=0.0, unit=Unit.GPA,
                       details="Chen-Niu hardness Hv = 2 (k^2 G)^0.585 - 3 must be >= 0 (raw value; negative / undefined "
                               "= moduli outside the empirical fit, record not emitted)")


def kleinman_parameter(Gamma: np.ndarray, species: np.ndarray, a_conv: float) -> Optional[float]:
    """Kleinman internal-strain parameter ζ for a two-species zincblende crystal.

    Under an engineering shear strain e4 (eps_yz = e4/2) the two fcc sublattices
    displace relative to each other along x by Δu_x = Γ_x4 difference.  With
    the definition ζ = |Δu_x| / ((a/4) |e4|), ζ = 0 means purely affine
    (clamped-ion) deformation and ζ = 1 means the four nearest-neighbour bond
    lengths are conserved to first order.  Returns None unless exactly two
    species are present.
    """
    sp = np.asarray(species)
    kinds = np.unique(sp)
    if kinds.size != 2:
        return None
    ux = [float(np.mean(Gamma[sp == k, 0, 3])) for k in kinds]
    return abs(ux[1] - ux[0]) / (a_conv / 4.0)


# ---------------------------------------------------------------------------
# Checks on tensors
# ---------------------------------------------------------------------------
def tensor_checks(C: np.ndarray, rotations: Optional[np.ndarray], space_group_number: Optional[int], tol: float):
    """(symmetry C=Cᵀ, point-group invariance, cubic pattern) deviations in GPa."""
    res = {"asymmetry": float(np.max(np.abs(C - C.T)))}
    if rotations is not None:
        res["point_group_invariance"] = check_tensor_invariance(C, rotations)
    if space_group_number is not None and space_group_number >= 195:
        res["cubic_pattern"] = cubic_pattern_deviation(C)
    return res


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------
_MATSCIPY_PARAMS = {
    "tersoff_1989_SiC": "Tersoff_PRB_39_5566_Si_C",
    # matscipy 1.3.0 indexes the Erhart-Albe three-body terms by the i-j pair (the legacy model); the check runs
    # on the perfect crystal, where every neighbour pair is Si-C and the i-k (default) and i-j forms coincide.
    "erhart_albe_2005_SiC": "Erhart_PRB_71_035211_SiC",
    "erhart_albe_2005_SiC_matscipy_ij": "Erhart_PRB_71_035211_SiC",
}


def matscipy_reference_elastic(model: EnergyModel, system: AtomicSystem) -> Optional[np.ndarray]:
    """Relaxed-ion C (6x6 GPa) from matscipy's Manybody calculator (Born + non-affine + stress terms).

    Independent implementation of the same Tersoff model (Hessian and
    non-affine forces coded analytically in matscipy).  Returns None if
    matscipy is unavailable or has no equivalent parameter set.
    """
    name = _MATSCIPY_PARAMS.get(model.name)
    if name is None:
        return None
    try:
        from matscipy.calculators.manybody import Manybody
        from matscipy.calculators.manybody.explicit_forms import TersoffBrenner, tersoff_brenner
        from matscipy.elasticity import full_3x3x3x3_to_Voigt_6x6
    except ImportError:
        return None
    atoms = system.to_ase()
    atoms.calc = Manybody(**TersoffBrenner(getattr(tersoff_brenner, name)))
    C4 = atoms.calc.get_elastic_constants(
        atoms, cg_parameters={"x0": None, "rtol": 1e-12, "maxiter": None, "M": None, "callback": None, "atol": 1e-14}
    )
    return np.asarray(full_3x3x3x3_to_Voigt_6x6(C4)) * EV_PER_A3_TO_GPA


def replicate(system: AtomicSystem, reps: tuple[int, int, int]) -> AtomicSystem:
    """Periodic replication (ASE ordering: unit-cell block repeated per translation)."""
    return AtomicSystem.from_ase(system.to_ase().repeat(reps), elements=system.elements)


def run_elastic_tensor(protocol, candidate, model: EnergyModel, structure=None, context=None):
    """ElasticTensorProtocol runner (JAX engine).  See module docstring and docs/physics_bulk.md."""
    from fracture_md import provenance
    from fracture_md.engines.base import ProtocolOutput, RunContext
    from fracture_md.physics.relax import run_bulk_relax
    from fracture_md.records import make_record
    from fracture_md.schema.common import CheckResult, Conditions, Uncertainty, UncertaintyKind, ValidationEvidence
    from fracture_md.schema.material import Orientation
    from fracture_md.schema.properties import Derivation, MethodSpec, PropertyName, PropertyOrigin
    from fracture_md.schema.protocol import BulkRelaxProtocol
    from fracture_md.structures.symmetry import conventional_cell, lattice_parameters, space_group
    from fracture_md.structures.system import auto_supercell, minimum_image_ok
    from fracture_md.units import Unit

    context = context or RunContext()
    skin = 0.5
    out = ProtocolOutput()
    if structure is None:
        relax_out = run_bulk_relax(BulkRelaxProtocol(relax=protocol.relax), candidate, model, context=context)
        out = out.merge(relax_out)
        structure = relax_out.structure
    s = structure
    if not minimum_image_ok(np.asarray(s.box), model.cutoff + skin):
        s = replicate(s, auto_supercell(s.to_ase(), model.cutoff + skin))

    tol = protocol.symmetry_tolerance_GPa
    card = model.card
    checks: list[CheckResult] = []
    sg = space_group(s)
    rotations = sg.rotations_cart
    nh = build_neighbors(s, model.cutoff, skin=skin)

    # ---------------- autodiff
    ad = elastic_constants_autodiff(model, s, nh.idx)
    fmax_tol = max(10 * protocol.relax.fmax_eV_per_A, 1e-6)
    checks.append(CheckResult(name="equilibrium_zero_force", passed=ad.max_force < fmax_tol, metric=ad.max_force,
                              tolerance=fmax_tol, unit=Unit.EV_PER_ANGSTROM))
    st = float(np.max(np.abs(ad.stress_GPa)))
    checks.append(CheckResult(name="equilibrium_zero_stress", passed=st < 1e-3, metric=st, tolerance=1e-3, unit=Unit.GPA,
                              details="C = (1/V) d2E/de2 equals dσ/de only at zero stress"))
    checks.append(CheckResult(name="position_hessian_minimum", passed=ad.spectrum.n_negative_modes == 0
                              and ad.spectrum.n_zero_modes == 3, metric=ad.spectrum.min_nonzero,
                              unit=Unit.EV_PER_ANGSTROM2,
                              details=f"{ad.spectrum.n_zero_modes} zero modes (3 translations expected), "
                                      f"{ad.spectrum.n_negative_modes} negative modes"))
    C_main = ad.C_relaxed
    for label, C in [("relaxed", ad.C_relaxed), ("clamped", ad.C_clamped)]:
        tc = tensor_checks(C, rotations, sg.number, tol)
        checks.append(CheckResult(name=f"ad_{label}_major_symmetry", passed=tc["asymmetry"] < 1e-6, metric=tc["asymmetry"],
                                  tolerance=1e-6, unit=Unit.GPA))
        checks.append(CheckResult(name=f"ad_{label}_point_group_invariance", passed=tc["point_group_invariance"] < 1e-5,
                                  metric=tc["point_group_invariance"], tolerance=1e-5, unit=Unit.GPA,
                                  details=f"{len(rotations)} Cartesian operations of {sg.international} (#{sg.number})"))
        if "cubic_pattern" in tc:
            checks.append(CheckResult(name=f"ad_{label}_cubic_pattern", passed=tc["cubic_pattern"] < 1e-5,
                                      metric=tc["cubic_pattern"], tolerance=1e-5, unit=Unit.GPA))

    # ---------------- finite differences
    fd = fdc = None
    uncertainties: list[Uncertainty] = []
    if "finite_difference" in protocol.methods:
        fd = elastic_constants_finite_difference(model, s, protocol.strain_magnitudes, relax_internal=protocol.relax_internal,
                                                 idx=nh.idx, fmax=min(protocol.relax.fmax_eV_per_A, 1e-7) * 1e-2)
        fdc = elastic_constants_finite_difference(model, s, protocol.strain_magnitudes, relax_internal=False, idx=nh.idx)
        dfd = float(np.max(np.abs(fd.C - ad.C_relaxed)))
        dfdc = float(np.max(np.abs(fdc.C - ad.C_clamped)))
        ftol = protocol.fd_vs_autodiff_tolerance_GPa
        checks.append(CheckResult(name="autodiff_vs_fd_relaxed", passed=dfd < ftol, metric=dfd, tolerance=ftol, unit=Unit.GPA,
                                  details=f"FD at δ={fd.primary_magnitude:g}, internal relaxation to "
                                          f"fmax={fd.max_residual_force:.1e} eV/Å"))
        checks.append(CheckResult(name="autodiff_vs_fd_clamped", passed=dfdc < ftol, metric=dfdc, tolerance=ftol, unit=Unit.GPA))
        checks.append(CheckResult(name="fd_strain_magnitude_convergence", passed=fd.convergence_spread < ftol,
                                  metric=fd.convergence_spread, tolerance=ftol, unit=Unit.GPA,
                                  details=f"max |C(δ)-C(δ*)| over δ in {fd.magnitudes}"))
        tcf = tensor_checks(fd.C, rotations, sg.number, tol)
        checks.append(CheckResult(name="fd_major_symmetry", passed=tcf["asymmetry"] < tol, metric=tcf["asymmetry"], tolerance=tol,
                                  unit=Unit.GPA))
        uncertainties.append(Uncertainty(kind=UncertaintyKind.AUTODIFF_VS_FINITE_DIFFERENCE, value=dfd, unit=Unit.GPA,
                                         interval="max_abs_diff",
                                         description="max |C_AD - C_FD| over all 36 components (relaxed ion)"))
        uncertainties.append(Uncertainty(kind=UncertaintyKind.NUMERICAL_CONVERGENCE, value=fd.convergence_spread, unit=Unit.GPA,
                                         interval="max_abs_diff",
                                         description=f"FD stress-strain spread over strain magnitudes {fd.magnitudes}"))
        if "autodiff_relaxed_ion" not in protocol.methods:
            C_main = fd.C

    # ---------------- supercell-size invariance
    size_diff = None
    conv = conventional_cell(s)
    if conv is not None and s.n_atoms % conv.n_atoms == 0:
        reps = auto_supercell(conv.to_ase(), model.cutoff + skin)
        n_cur = s.n_atoms // conv.n_atoms
        reps_big = tuple(r + 1 for r in reps)
        if int(np.prod(reps_big)) == n_cur:
            reps_big = tuple(r + 2 for r in reps)
        big = replicate(conv, reps_big)
        nh_big = build_neighbors(big, model.cutoff, skin=skin)
        big, _, _ = newton_relax_fixed_cell(model, big, nh_big.idx, fmax=1e-9)
        ad_big = elastic_constants_autodiff(model, big, nh_big.idx)
        size_diff = float(np.max(np.abs(ad_big.C_relaxed - ad.C_relaxed)))
        checks.append(CheckResult(name="supercell_size_invariance", passed=size_diff < 1e-3, metric=size_diff, tolerance=1e-3,
                                  unit=Unit.GPA, details=f"{s.n_atoms} vs {big.n_atoms} atoms ({reps_big} conventional cells)"))
        uncertainties.append(Uncertainty(kind=UncertaintyKind.NUMERICAL_CONVERGENCE, value=size_diff, unit=Unit.GPA,
                                         interval="max_abs_diff", description=f"supercell {s.n_atoms} vs {big.n_atoms} atoms"))
        out.artifacts["C_relaxed_large_supercell_GPa"] = ad_big.C_relaxed

    # ---------------- reference implementation (matscipy)
    evidence: list[ValidationEvidence] = []
    C_ms = None
    if context.extra.get("matscipy_reference", True):
        try:
            C_ms = matscipy_reference_elastic(model, s)
        except Exception as exc:  # pragma: no cover - reference is optional
            out.raw["matscipy_error"] = repr(exc)
    if C_ms is not None:
        dms = float(np.max(np.abs(C_ms - ad.C_relaxed)))
        ok = dms < 0.05
        checks.append(CheckResult(name="matscipy_reference_relaxed_C", passed=ok, metric=dms, tolerance=0.05, unit=Unit.GPA,
                                  details="matscipy Manybody Born + non-affine + stress terms"))
        evidence.append(ValidationEvidence(kind="reference_implementation", passed=ok,
                                           description="Relaxed-ion C from matscipy 1.3 Manybody (independent analytic "
                                                       f"Hessian/non-affine implementation of the same model); max |ΔC| = {dms:.2e} GPa",
                                           citation="matscipy: Grigorev et al., JOSS 9, 5668 (2024), doi:10.21105/joss.05668"))
        out.artifacts["C_matscipy_GPa"] = C_ms

    # ---------------- analysis
    dens = None
    for rec in out.properties:
        if rec.name == PropertyName.DENSITY:
            dens = rec.value.magnitude(Unit.G_PER_CM3)
    if dens is None:
        from fracture_md.units import AMU_PER_A3_TO_G_PER_CM3

        dens = float(np.sum(np.asarray(s.masses_amu))) / s.volume * AMU_PER_A3_TO_G_PER_CM3
    an = analyze_elastic(C_main, density_g_cm3=dens)
    an_cl = analyze_elastic(ad.C_clamped)
    born_ok = an["born_stable"]
    born_check = CheckResult(name="born_stability", passed=born_ok, metric=an["born_min_eigenvalue"], tolerance=0.0,
                             unit=Unit.GPA, details="min eigenvalue of C_relaxed (positive definite = stable)")
    hv_check = hardness_applicability_check(an)
    out_checks = list(checks)
    out_checks.extend([born_check, hv_check])

    conv_a = None
    if conv is not None:
        lengths, _ = lattice_parameters(np.asarray(conv.box))
        conv_a = float(lengths[0])
    # Kleinman parameter (zincblende)
    zeta = None
    if sg.number == 216 and conv_a is not None:
        zeta = kleinman_parameter(ad.internal_displacement, np.asarray(s.species), conv_a)

    # ---------------- records
    prov = provenance.collect(engine="jax_md", potential_id=card.id, protocol_id=protocol.protocol_id(),
                              requested_by=context.requested_by, implementation_ids=dict(context.implementation_ids),
                              parent_ids=list(context.parent_ids))
    cond = Conditions(stress_state="zero stress, 0 K lattice statics")
    crystal = Orientation(description="crystal Cartesian frame (conventional cubic axes for cubic phases)")
    ad_method = MethodSpec(
        name="autodiff relaxed-ion elastic tensor", protocol_id=protocol.protocol_id(), engine="jax_md",
        description="C = (H_ee - Λᵀ K⁺ Λ)/V from exact JAX second derivatives; FD stress-strain cross-check "
                    "(internal relaxation at each strained cell by Newton steps with the exact Hessian; "
                    "RelaxSettings.optimizer applies to the preceding bulk relaxation only)",
        assumptions=["0 K, static", "zero initial stress and force", "linear elasticity (small strain)",
                     f"{s.n_atoms}-atom periodic cell", "translations removed from K⁺"],
    )
    rel_checks = [c for c in checks]
    # Quantities derived from C (VRH averages, directional moduli, anisotropy indices, Pugh ratio, hardness)
    # describe a mechanically meaningful solid only if C is positive definite: they carry the Born check.
    # C itself (and S, the tensor form, the Born indicator record) remain valid numbers either way.
    derived_checks = rel_checks + [born_check]
    recs = []

    def rec(name, value, unit, *, shape="", frame="crystal", orientation=crystal, chk=None, unc=(), evid=(), notes="",
            origin=PropertyOrigin.ATOMISTIC_COMPUTED, derivation=None, method=ad_method):
        r = make_record(name=name, value=value, unit=unit, candidate=candidate, method=method, card=card,
                        conditions=cond, orientation=orientation, uncertainty=list(unc),
                        checks=list(rel_checks if chk is None else chk), evidence=list(evid), frame=frame,
                        shape_description=shape, notes=notes, provenance=prov, origin=origin, derivation=derivation)
        recs.append(r)
        return r

    vo = "voigt 6x6, order " + ",".join(VOIGT_LABELS) + "; engineering shear strains"
    r_C = rec(PropertyName.ELASTIC_STIFFNESS_VOIGT, C_main, Unit.GPA, shape=vo, unc=uncertainties, evid=evidence,
              notes=f"C11={an.get('C11', C_main[0, 0]):.3f}, C12={an.get('C12', C_main[0, 1]):.3f}, "
                    f"C44={an.get('C44', C_main[3, 3]):.3f} GPa (relaxed ion)")
    rec(PropertyName.ELASTIC_COMPLIANCE_VOIGT, np.asarray(an["S_per_GPa"]), Unit.PER_GPA,
        shape=vo + " (compliance, S = C^-1)", unc=uncertainties)
    rec(PropertyName.CLAMPED_ION_STIFFNESS_VOIGT, ad.C_clamped, Unit.GPA, shape=vo + " (clamped ion, no internal relaxation)",
        chk=[c for c in checks if "relaxed" not in c.name or c.name.startswith("equilibrium")])
    rec(PropertyName.ELASTIC_STIFFNESS_TENSOR, voigt_to_tensor_stiffness(C_main), Unit.GPA, shape="3x3x3x3 C_ijkl",
        unc=uncertainties, evid=evidence)
    # distinct internal-displacement tensors (translation-equivalent atoms share Γ)
    G = ad.internal_displacement
    groups: list[np.ndarray] = []
    atom_group = np.zeros(s.n_atoms, int)
    for i in range(s.n_atoms):
        for gi, gv in enumerate(groups):
            if np.max(np.abs(G[i] - gv)) < 1e-7:
                atom_group[i] = gi
                break
        else:
            groups.append(G[i])
            atom_group[i] = len(groups) - 1
    rec(PropertyName.INTERNAL_STRAIN_TENSOR, np.asarray(groups), Unit.ANGSTROM,
        shape="(n_distinct, 3, 6): internal displacement per unit Voigt strain Γ = -K⁺Λ; distinct per-atom tensors "
              "(atom -> group map in artifacts['internal_displacement_group'])",
        notes=(f"Kleinman parameter ζ = {zeta:.4f} (ζ=|Δu_x|/((a/4)|e4|); 0 = affine, 1 = bond-length conserving)"
               if zeta is not None else ""))
    moduli = [
        (PropertyName.BULK_MODULUS_VOIGT, "B_voigt"), (PropertyName.BULK_MODULUS_REUSS, "B_reuss"),
        (PropertyName.BULK_MODULUS_HILL, "B_hill"), (PropertyName.SHEAR_MODULUS_VOIGT, "G_voigt"),
        (PropertyName.SHEAR_MODULUS_REUSS, "G_reuss"), (PropertyName.SHEAR_MODULUS_HILL, "G_hill"),
        (PropertyName.YOUNGS_MODULUS_HILL, "E_hill"),
    ]
    polycrystal = "isotropic untextured dense polycrystal (Voigt-Reuss-Hill average of the single-crystal C)"
    mod_recs = {}
    for pn, key in moduli:
        mod_recs[key] = rec(pn, float(an[key]), Unit.GPA, frame="not_applicable", orientation=None, notes=polycrystal,
                            chk=derived_checks)
    rec(PropertyName.POISSON_RATIO_HILL, float(an["nu_hill"]), Unit.DIMENSIONLESS, frame="not_applicable", orientation=None,
        notes=polycrystal, chk=derived_checks)
    rec(PropertyName.YOUNGS_MODULUS_DIRECTIONAL, [an["E_directional"][k] for k in ("[100]", "[110]", "[111]")], Unit.GPA,
        shape="E(n) for n = [100], [110], [111] (crystal frame)",
        orientation=Orientation(description="directions [100],[110],[111]"), chk=derived_checks)
    if born_ok:  # the extremum search is skipped (NaN) for a Born-unstable C
        rec(PropertyName.YOUNGS_MODULUS_DIRECTIONAL, [an["E_min"], an["E_max"]], Unit.GPA,
            shape="[E_min, E_max] over the unit sphere",
            orientation=Orientation(description="extrema of E(n) over the unit sphere"), chk=derived_checks,
            notes=f"E_min along {np.round(an['E_min_direction'], 4).tolist()}, E_max along {np.round(an['E_max_direction'], 4).tolist()}")
    rec(PropertyName.SHEAR_MODULUS_DIRECTIONAL, list(an["G_directional"].values()), Unit.GPA,
        shape="G(plane normal n, shear direction m) for " + ", ".join(an["G_directional"].keys()),
        orientation=Orientation(description="(n)[m] = " + ", ".join(an["G_directional"].keys())), chk=derived_checks)
    rec(PropertyName.POISSON_RATIO_DIRECTIONAL, list(an["nu_directional"].values()), Unit.DIMENSIONLESS,
        shape="ν(load n, lateral m) for " + ", ".join(an["nu_directional"].keys()),
        orientation=Orientation(description="n/m = " + ", ".join(an["nu_directional"].keys())), chk=derived_checks)
    if an["is_cubic"]:
        rec(PropertyName.ZENER_ANISOTROPY, float(an["zener_ratio"]), Unit.DIMENSIONLESS, frame="not_applicable",
            orientation=None, notes="A = 2 C44 / (C11 - C12); 1 = isotropic", chk=derived_checks)
    rec(PropertyName.UNIVERSAL_ANISOTROPY_INDEX, float(an["universal_anisotropy"]), Unit.DIMENSIONLESS,
        frame="not_applicable", orientation=None, notes="A^U = 5 G_V/G_R + B_V/B_R - 6 (Ranganathan & Ostoja-Starzewski 2008)",
        chk=derived_checks)
    rec(PropertyName.PUGH_RATIO, float(an["pugh_ratio_B_over_G"]), Unit.DIMENSIONLESS, frame="not_applicable", orientation=None,
        notes="B_H/G_H; > ~1.75 is conventionally read as ductile, < ~1.75 as brittle (heuristic, Pugh 1954)",
        chk=derived_checks)
    rec(PropertyName.BORN_STABILITY, float(an["born_min_eigenvalue"]), Unit.GPA, frame="crystal",
        shape="minimum eigenvalue of the 6x6 relaxed-ion C (>0: mechanically stable at zero stress)",
        notes=("cubic criteria " + str(an.get("born_cubic"))) if an["is_cubic"] else "")
    hv_deriv = Derivation(
        model=HARDNESS_CHEN_NIU[0], equation="Hv = 2 (k^2 G)^0.585 - 3,  k = G_H / B_H",
        assumptions=["empirical correlation fitted to measured Vickers hardness of mostly covalent/ionic solids",
                     "uses single-crystal VRH moduli of a defect-free lattice; ignores dislocation mobility, "
                     "microstructure, load/indentation-size effects",
                     "not a computed plastic property"],
        input_record_ids=[mod_recs["B_hill"].id, mod_recs["G_hill"].id],
    )
    # The hardness inherits the checks of its inputs (B_H, G_H: all numerical checks + Born stability), so it
    # is FAILED whenever they are, plus the applicability of the correlation itself.  A negative / undefined
    # Chen-Niu value is not emitted as a record (non-finite), only as the failed model_applicability check.
    if hv_check.passed:
        rec(PropertyName.HARDNESS_ESTIMATE, float(an["hardness_chen_niu"]), Unit.GPA, frame="not_applicable",
            orientation=None, origin=PropertyOrigin.DERIVED_MODEL, derivation=hv_deriv, chk=derived_checks + [hv_check],
            unc=[chen_niu_score_uncertainty(), Uncertainty(kind=UncertaintyKind.UNQUANTIFIED,
                             description=f"model-form: {HARDNESS_TIAN[0]} gives {an['hardness_tian']:.2f} GPa ({HARDNESS_TIAN[1]})")],
            notes=HARDNESS_CHEN_NIU[1],
            method=MethodSpec(name="Chen-Niu hardness model", protocol_id=protocol.protocol_id(), engine="jax_md",
                              description="empirical hardness from Hill B and G"))

    out.properties.extend(recs)
    out.checks.extend(out_checks)
    out.structure = s
    out.artifacts.update(
        C_relaxed_GPa=ad.C_relaxed, C_clamped_GPa=ad.C_clamped, S_per_GPa=np.asarray(an["S_per_GPa"]),
        C_tensor_GPa=voigt_to_tensor_stiffness(C_main), internal_strain_Lambda_eV_per_A=ad.internal_strain,
        internal_displacement_Gamma_A=ad.internal_displacement, internal_displacement_group=atom_group,
        hessian_eigenvalues_eV_per_A2=ad.hessian_eigenvalues, analysis=an, analysis_clamped=an_cl,
        rotations_cart=rotations,
    )
    if fd is not None:
        out.artifacts["C_fd_relaxed_GPa"] = fd.C
        out.artifacts["C_fd_clamped_GPa"] = fdc.C
        out.artifacts["C_fd_per_magnitude_GPa"] = {k: v for k, v in fd.per_magnitude.items()}
    out.raw.update(autodiff=ad.as_dict(), finite_difference=fd.as_dict() if fd else None,
                   space_group={"number": sg.number, "symbol": sg.international, "point_group": sg.point_group},
                   kleinman_zeta=zeta, supercell_size_diff_GPa=size_diff,
                   conventional_a_A=conv_a)
    return out
