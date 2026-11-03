"""Molecular dynamics on the JAX path: thin, pure, differentiable wrappers around ``jax_md.simulate``.

Units and time
--------------
Internal units are eV, Å and amu.  Newton's equation ``m d²x/dt² = F`` in
these units has the natural time unit

    τ = Å · sqrt(amu / eV) = sqrt(1.66053906660e-27 kg · 1e-20 m² / 1.602176634e-19 J)
      = 1.0180505e-14 s  ≈ 10.1805 fs

(:data:`TIME_UNIT_FS`, derived from the CODATA constants in ``units.py`` and
checked in the tests).  Time steps are accepted in fs and divided by τ;
momenta are in amu·Å/τ, so ``p²/2m`` is in eV.  Note: ``jax_md.simulate``
casts ``dt`` to float32 inside ``velocity_verlet`` (``f32(dt)``), so the
effective step is ``float32(dt/τ)`` (relative rounding ≲ 6e-8).  The time step
is a fixed constant of the discrete map, so this does not affect gradients
w.r.t. strain or potential parameters (see :func:`effective_timestep_fs`).

Why this module does NOT hand JAX-MD its own fractional shift function
----------------------------------------------------------------------
JAX-MD's ``periodic_general(fractional_coordinates=True)`` attaches a custom
JVP to ``space.transform`` so that the tangent of a fractional coordinate is
interpreted as a *real-space* displacement.  That is what makes
``grad(E)(frac)`` a real-space gradient, and it is self-consistent along a
trajectory as long as the box is a constant.  It is **not** consistent when
the box itself carries a tangent (strain as a differentiated loading
parameter): the shift ``S + box⁻¹ dR`` then propagates ``d(box⁻¹)·dR`` in the
wrong coordinate system, and the mixed derivative ∂σ/∂S also comes out wrong.
Measured on 64-atom 3C-SiC/Tersoff at 300 K (docs/AUTODIFF.md): the naive
JAX-MD NVE gradient d E_pot(t)/d ε_zz is off by 3-9 % for 20-100 fs
trajectories (6.1 %, 8.6 %, 3.0 % at 20, 50, 100 fs), and the stress-position
derivative d σ/d S by 27 % (σ_xx) up to ~9x (σ_xy) for a random direction.

We therefore integrate *true* fractional coordinates with plain ``jnp``
arithmetic (``S' = S + dR @ box⁻ᵀ``, unwrapped) and evaluate energy, forces and
virial through :func:`exact_frac_functions`, whose custom JVPs convert the
fractional tangent into the real-space tangent that JAX-MD's kernels expect
(``dX = dS @ boxᵀ``) and only use derivative paths that are exact under
JAX-MD's convention:

* ∂E/∂X (real gradient) and ∂²E/∂X² (real Hessian),
* ∂/∂box at fixed S of E and of ∂E/∂X,
* ∂/∂params of anything.

The resulting trajectories are pure functions of ``(box, params, momenta,
frac)``, differentiable in forward and reverse mode (first order), and their
derivatives agree with central finite differences to ~1e-8 for short runs.

Neighbour lists
---------------
``idx`` is an integer array fixed for the whole trajectory; it carries no
gradient and cannot be rebuilt inside a differentiated trajectory without
breaking purity.  Results are exact as long as ``idx`` is a superset of the
pairs inside the cutoff at every step.  :class:`MDTrajectory` records the
maximum displacement from the start and :func:`check_neighbor_list_validity`
enforces JAX-MD's ``skin/2`` criterion (with a conservative allowance for an
applied homogeneous strain relative to the configuration the list was built
on).
"""

from __future__ import annotations

import math
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import simulate

from fracture_md.potentials.base import EnergyModel
from fracture_md.structures.system import AtomicSystem
from fracture_md.units import AMU_KG, ANGSTROM_M, BOLTZMANN_EV_K, EV_J

# ----------------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------------
#: Natural time unit of the (eV, Å, amu) system, in seconds and femtoseconds.
TIME_UNIT_S: float = math.sqrt(AMU_KG * ANGSTROM_M**2 / EV_J)
TIME_UNIT_FS: float = TIME_UNIT_S * 1.0e15  # 10.18050...
KB_EV_PER_K: float = BOLTZMANN_EV_K

ENSEMBLES = ("nve", "nvt_nose_hoover", "nvt_langevin")


def fs_to_internal_time(t_fs):
    """Convert femtoseconds to the internal time unit τ = Å·sqrt(amu/eV)."""
    return t_fs / TIME_UNIT_FS


def internal_time_to_fs(t):
    return t * TIME_UNIT_FS


def effective_timestep_fs(dt_fs: float) -> float:
    """Time step actually used by jax_md.simulate (it casts dt/τ to float32)."""
    return float(np.float32(fs_to_internal_time(dt_fs))) * TIME_UNIT_FS


class NeighborListInvalidError(RuntimeError):
    """An atom moved far enough that the fixed neighbour list may miss in-cutoff pairs."""


# ----------------------------------------------------------------------------
# Energy / force / virial with mathematically exact fractional-coordinate JVPs
# ----------------------------------------------------------------------------
class ExactFracFunctions(NamedTuple):
    """Primal values identical to the model's energy_fn; JVPs w.r.t. TRUE fractional coordinates.

    All three take ``(frac, box, species, idx, params)``:

    * ``energy``  -> eV
    * ``force``   -> real-space force (N, 3), eV/Å   (= -∂E/∂X)
    * ``virial``  -> W = ∂E/∂ε at fixed frac, symmetric (3, 3), eV;  σ_config = W / V

    Their custom JVPs are exact to FIRST order in (frac, box, params); do not
    take second derivatives through them (e.g. Hessians of trajectory
    observables) — the rules' own primal derivatives go back through JAX-MD's
    convention.  Note the gradient of ``energy`` w.r.t. frac is the true
    fractional gradient ``∂E/∂X @ box`` (per atom), NOT the real-space gradient
    returned by ``jax.grad(model.energy_fn)``.
    """

    energy: Callable
    force: Callable
    virial: Callable


def exact_frac_functions(energy_fn: Callable) -> ExactFracFunctions:
    def e_jm(S, box, species, idx, p):
        return energy_fn(S, box, species, idx, p)

    g_real = jax.grad(e_jm, argnums=0)  # real-space gradient under JAX-MD's convention

    def virial_jm(S, box, species, idx, p):
        def e_eps(eps):
            return e_jm(S, (jnp.eye(3, dtype=box.dtype) + eps) @ box, species, idx, p)

        w = jax.grad(e_eps)(jnp.zeros((3, 3), dtype=box.dtype))
        return 0.5 * (w + w.T)

    @jax.custom_jvp
    def energy(S, box, species, idx, p):
        return e_jm(S, box, species, idx, p)

    @energy.defjvp
    def _energy_jvp(primals, tangents):
        S, box, species, idx, p = primals
        dS, dbox, _, _, dp = tangents
        f = lambda s_, b_, p_: e_jm(s_, b_, species, idx, p_)  # noqa: E731
        return jax.jvp(f, (S, box, p), (dS @ box.T, dbox, dp))

    @jax.custom_jvp
    def force(S, box, species, idx, p):
        return -g_real(S, box, species, idx, p)

    @force.defjvp
    def _force_jvp(primals, tangents):
        S, box, species, idx, p = primals
        dS, dbox, _, _, dp = tangents
        f = lambda s_, b_, p_: -g_real(s_, b_, species, idx, p_)  # noqa: E731
        return jax.jvp(f, (S, box, p), (dS @ box.T, dbox, dp))

    @jax.custom_jvp
    def virial(S, box, species, idx, p):
        return virial_jm(S, box, species, idx, p)

    @virial.defjvp
    def _virial_jvp(primals, tangents):
        S, box, species, idx, p = primals
        dS, dbox, _, _, dp = tangents
        # (box, params) part at fixed S: JAX-MD's derivatives are exact here.
        w, t_bp = jax.jvp(lambda b_, p_: virial_jm(S, b_, species, idx, p_), (box, p), (dbox, dp))

        # S part via symmetry of mixed partials:
        #   ∂W/∂S·dS = ∂/∂ε [ ∂E/∂X(S, (I+ε)box) · (dS @ ((I+ε)box)ᵀ) ]  at ε = 0,
        # which only needs ∂(∂E/∂X)/∂box at fixed S (exact).
        def h(eps):
            bb = (jnp.eye(3, dtype=box.dtype) + eps) @ box
            return jnp.vdot(g_real(S, bb, species, idx, p), dS @ bb.T)

        t_s = jax.grad(h)(jnp.zeros((3, 3), dtype=box.dtype))
        return w, t_bp + 0.5 * (t_s + t_s.T)

    return ExactFracFunctions(energy=energy, force=force, virial=virial)


# ----------------------------------------------------------------------------
# Initial conditions and simple observables
# ----------------------------------------------------------------------------
def kinetic_energy(momenta, masses):
    """Σ p²/2m (eV); masses (N,) or (N,1) in amu, momenta in amu·Å/τ."""
    m = jnp.reshape(masses, (-1, 1))
    return 0.5 * jnp.sum(momenta**2 / m)


def instantaneous_temperature(momenta, masses, n_constraints: int = 3):
    """2 KE / (k_B (3N - n_constraints)) in K."""
    dof = momenta.size - n_constraints
    return 2.0 * kinetic_energy(momenta, masses) / (dof * KB_EV_PER_K)


def nose_hoover_kT(temperature_K: float, n_atoms: int) -> float:
    """k_B T (3N - 3)/(3N) (eV): the kT handed to JAX-MD's Nosé–Hoover chain (which counts 3N dof).

    With zero net momentum the system has 3N - 3 degrees of freedom; this
    makes the chain's target kinetic energy (3N - 3) k_B T / 2, consistent with
    :func:`instantaneous_temperature` and :func:`initial_momenta`.
    """
    n = int(n_atoms)
    return KB_EV_PER_K * float(temperature_K) * (3 * n - 3) / (3 * n)


def initial_momenta(masses, temperature_K: float, seed: int = 0, key=None):
    """Maxwell–Boltzmann momenta (amu·Å/τ) from a fixed PRNG seed.

    Zero net momentum (mass-weighted centre-of-mass motion removed), then
    rescaled so the instantaneous temperature with 3N-3 degrees of freedom is
    exactly ``temperature_K``.  Deterministic for a given (seed, N, masses).
    """
    m = jnp.reshape(jnp.asarray(masses, dtype=jnp.float64), (-1, 1))
    n = m.shape[0]
    if key is None:
        key = jax.random.PRNGKey(seed)
    kT = KB_EV_PER_K * temperature_K
    p = jnp.sqrt(m * kT) * jax.random.normal(key, (n, 3), dtype=jnp.float64)
    p = p - m * (jnp.sum(p, axis=0, keepdims=True) / jnp.sum(m))
    if temperature_K <= 0.0:
        return jnp.zeros_like(p)
    t_now = instantaneous_temperature(p, m)
    return p * jnp.sqrt(temperature_K / t_now)


class MDTrajectory(NamedTuple):
    """Result of :func:`run_md` (a pytree of arrays; differentiable)."""

    frac: jnp.ndarray  # (N, 3) final UNWRAPPED fractional coordinates
    momentum: jnp.ndarray  # (N, 3) final momenta, amu·Å/τ
    box: jnp.ndarray  # (3, 3) box used (lattice vectors as columns)
    potential_energy: jnp.ndarray  # (n_steps+1,) eV, index 0 = initial state
    kinetic_energy: jnp.ndarray  # (n_steps+1,) eV
    max_displacement: jnp.ndarray  # scalar Å: max_t max_i |X_i(t) - X_i(0)| (gradient stopped)
    positions: Optional[jnp.ndarray] = None  # (n_steps+1, N, 3) frac if record_positions


def run_md(
    model: EnergyModel,
    system: AtomicSystem,
    idx: jnp.ndarray,
    *,
    dt_fs: float,
    n_steps: int,
    temperature_K: float = 300.0,
    seed: int = 0,
    ensemble: str = "nve",
    box: Optional[jnp.ndarray] = None,
    frac: Optional[jnp.ndarray] = None,
    params: Any = None,
    momenta: Optional[jnp.ndarray] = None,
    masses: Optional[jnp.ndarray] = None,
    key: Optional[jnp.ndarray] = None,
    langevin_gamma_per_ps: float = 1.0,
    nose_hoover_tau_fs: Optional[float] = None,
    remat: bool = False,
    remat_chunk: Optional[int] = None,
    record_positions: bool = False,
) -> MDTrajectory:
    """Integrate ``n_steps`` of MD with a fixed neighbour list; pure in (box, frac, params, momenta).

    Physical assumptions: classical Newtonian dynamics (velocity Verlet for
    NVE; Nosé–Hoover chain or BAOAB Langevin for NVT), fixed periodic cell
    ``box`` (default ``system.box``; pass a strained box to apply a
    homogeneous strain with atoms mapped affinely through fixed fractional
    coordinates), masses in amu (default: standard atomic masses).

    Initial momenta: ``momenta`` if given, else :func:`initial_momenta`
    (seeded, zero net momentum, exactly ``temperature_K``).  The thermostat
    key (Langevin) is ``key`` if given, else derived from ``seed``; a fixed key
    makes the stochastic trajectory a deterministic, differentiable function.

    ``remat=True`` wraps each step in ``jax.checkpoint`` (reverse-mode memory
    O(1) per step instead of all intermediates); ``remat_chunk=k`` additionally
    nests the scan in checkpointed chunks of k steps (memory ~ n/k + k states).
    Neither changes the numbers.

    Degrees of freedom: the initial momenta carry zero net momentum, so the
    system has 3N - 3 kinetic degrees of freedom (``instantaneous_temperature``
    uses 3N - 3).  ``jax_md.simulate.nvt_nose_hoover`` counts 3N and would
    thermostat to KE = 3N k_B T/2, i.e. a temperature 3N/(3N - 3) too high in
    our convention (+1.6 % for 64 atoms).  The Nosé–Hoover chain is therefore
    given ``kT_eff = k_B T (3N - 3)/(3N)`` (:func:`nose_hoover_kT`), so its
    target is KE = (3N - 3) k_B T/2.  Langevin is unchanged: it thermalises
    every degree of freedom including the centre of mass.
    """
    if ensemble not in ENSEMBLES:
        raise ValueError(f"unknown ensemble {ensemble!r}; choose from {ENSEMBLES}")
    n_steps = int(n_steps)
    box = system.box if box is None else box
    S0 = system.frac if frac is None else frac
    p = model.params if params is None else params
    species = model.species_for(system)
    m = (system.masses_amu if masses is None else jnp.asarray(masses))[:, None]
    fns = exact_frac_functions(model.energy_fn)

    k_mom, k_thermo = jax.random.split(jax.random.PRNGKey(seed))
    if key is not None:
        k_thermo = key
    p0 = initial_momenta(m, temperature_K, key=k_mom) if momenta is None else momenta

    binv = jnp.linalg.inv(box)
    dt = fs_to_internal_time(dt_fs)
    kT = KB_EV_PER_K * temperature_K

    def force_fn(S, **kw):
        return fns.force(S, box, species, idx, p)

    def shift_fn(S, dR, **kw):  # true fractional update, unwrapped (energy is periodic)
        return S + dR @ binv.T

    if ensemble == "nve":
        init, step = simulate.nve(force_fn, shift_fn, dt)
        state = init(k_mom, S0, kT, mass=m, momenta=p0)
    elif ensemble == "nvt_nose_hoover":
        tau = fs_to_internal_time(nose_hoover_tau_fs) if nose_hoover_tau_fs is not None else None
        init, step = simulate.nvt_nose_hoover(force_fn, shift_fn, dt, nose_hoover_kT(temperature_K, S0.shape[0]),
                                              tau=tau)
        state = init(k_mom, S0, mass=m, momenta=p0)
    else:
        gamma = langevin_gamma_per_ps * TIME_UNIT_FS * 1.0e-3  # 1/ps -> 1/τ
        init, step = simulate.nvt_langevin(force_fn, shift_fn, dt, kT, gamma=gamma)
        state = init(k_thermo, S0, mass=m, momenta=p0)

    def observe(st):
        pe = fns.energy(st.position, box, species, idx, p)
        ke = kinetic_energy(st.momentum, st.mass)
        disp = jax.lax.stop_gradient(jnp.max(jnp.linalg.norm((st.position - S0) @ box.T, axis=1)))
        return pe, ke, disp

    def body(st, _):
        st = step(st)
        pe, ke, disp = observe(st)
        out = (pe, ke, disp, st.position) if record_positions else (pe, ke, disp)
        return st, out

    if remat:
        body = jax.checkpoint(body)

    if remat_chunk is not None and remat_chunk > 1 and n_steps % remat_chunk == 0:
        k = int(remat_chunk)

        @jax.checkpoint
        def chunk(st, _):
            return jax.lax.scan(body, st, None, length=k)

        state_f, outs = jax.lax.scan(chunk, state, None, length=n_steps // k)
        outs = jax.tree_util.tree_map(lambda a: a.reshape((n_steps,) + a.shape[2:]), outs)
    else:
        state_f, outs = jax.lax.scan(body, state, None, length=n_steps)

    pe0, ke0, _ = observe(state)
    pe = jnp.concatenate([pe0[None], outs[0]])
    ke = jnp.concatenate([ke0[None], outs[1]])
    maxd = jnp.max(outs[2]) if n_steps > 0 else jnp.asarray(0.0)
    pos = jnp.concatenate([S0[None], outs[3]]) if record_positions else None
    return MDTrajectory(
        frac=state_f.position,
        momentum=state_f.momentum,
        box=box,
        potential_energy=pe,
        kinetic_energy=ke,
        max_displacement=maxd,
        positions=pos,
    )


# ----------------------------------------------------------------------------
# Derived observables and validity checks
# ----------------------------------------------------------------------------
def virial_stress(model: EnergyModel, system: AtomicSystem, idx, frac, momentum, box, params=None, masses=None):
    """Instantaneous virial stress tensor (eV/Å³), positive = tensile.

        σ = (1/V) [ ∂E/∂ε|_frac  −  Σ_i p_i ⊗ p_i / m_i ]

    i.e. the configurational (potential) part plus the kinetic part; the
    kinetic part is compressive.  Differentiable (exact first-order JVPs).
    """
    fns = exact_frac_functions(model.energy_fn)
    p = model.params if params is None else params
    m = (system.masses_amu if masses is None else jnp.asarray(masses))[:, None]
    species = model.species_for(system)
    w = fns.virial(frac, box, species, idx, p)
    kin = jnp.einsum("ia,ib->ab", momentum, momentum / m)
    return (w - kin) / jnp.abs(jnp.linalg.det(box))


def neighbor_list_margin(max_displacement: float, skin: float, cutoff: float, strain=None) -> float:
    """Remaining margin (Å) of the JAX-MD skin criterion; negative => list may be invalid.

    A Dense list built with radius (cutoff + skin) at configuration X0 contains
    every pair within ``cutoff`` at X as long as no atom moved more than
    ``skin/2`` from X0.  A homogeneous strain ε of the box relative to the
    build configuration changes pair distances by at most ‖ε‖₂ (cutoff + skin),
    which we charge half to each atom.
    """
    extra = 0.0
    if strain is not None:
        extra = 0.5 * float(np.linalg.norm(np.asarray(strain), 2)) * (cutoff + skin)
    return 0.5 * skin - extra - float(max_displacement)


def check_neighbor_list_validity(traj: MDTrajectory, skin: float, cutoff: float, strain=None) -> float:
    """Raise :class:`NeighborListInvalidError` if the fixed list may have become invalid; return margin."""
    margin = neighbor_list_margin(float(traj.max_displacement), skin, cutoff, strain)
    if margin <= 0.0:
        raise NeighborListInvalidError(
            f"max displacement {float(traj.max_displacement):.4f} Å exceeds skin/2 = {0.5 * skin:.4f} Å "
            f"(minus strain allowance); rebuild the list between trajectory segments or increase the skin"
        )
    return margin


def total_energy_series(traj: MDTrajectory) -> jnp.ndarray:
    return traj.potential_energy + traj.kinetic_energy
