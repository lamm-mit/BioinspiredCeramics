"""Energy minimisation of atomic positions (fixed cell) and of the cell (zero stress).

Position minimisers (``relax_positions(method=...)``):

* ``"lbfgs"`` (default): scipy L-BFGS on real-space positions with a jitted
  float64 energy-and-forces kernel, then Newton polishing with the exact
  autodiff Hessian.  Robust to tight tolerances (fmax ~ 1e-8 eV/Å), which
  elastic constants need.  Host-side loop.  ``"bfgs"`` and ``"cg"`` use the
  corresponding scipy methods with the same kernels.
* ``"fire"``: JAX-MD ``minimize.fire_descent`` inside ``lax.while_loop``
  (fully on-device; one jitted kernel per energy function and input shape).
  After every FIRE run the neighbour list is rebuilt at the result and the
  forces are re-evaluated with the fresh list; FIRE restarts until those
  forces are below fmax (then Newton polishing if still needed).

Performance: every energy / force / Hessian evaluation goes through kernels
compiled once per energy function (``potentials.base.compiled_kernels``, the
cached derivative primitives of ``physics.elastic``, and :func:`_fire_kernel`).
Nothing is re-jitted per call and ``model.energy_fn`` is never run eagerly.

Gradient convention reminder: with JAX-MD fractional coordinates,
``jax.grad(E, 0)(frac)`` is already the *real-space* gradient.  We therefore
never differentiate through a user-side map ``x -> frac``; we evaluate at
``frac = x @ inv(box).T`` and use ``grad(E)(frac)`` as dE/dx.

Neighbour lists: the list is rebuilt after every inner minimisation; the
minimisation restarts if the rebuilt list changes the forces beyond fmax.

Cell relaxation (:func:`relax_cell`) minimises the energy over the symmetric
strain of the cell (6 components, or one isotropic scale) *and* the internal
coordinates, i.e. it drives the Cauchy stress σ = (1/V) ∂E/∂ε to zero at zero
force.  Each outer iteration relaxes positions at fixed cell and then takes a
Newton step on the strain using the exact *relaxed-ion* strain Hessian
(H_ee - Λᵀ K⁺ Λ, from ``physics.elastic``), so convergence is quadratic.  The
loop stops once the residual stress is below ``stress_tol_GPa``; it then takes
``polish_steps`` (default 1) further Newton step(s), which is cheap and, the
convergence being quadratic, leaves the residual far below the tolerance
(values reached for 3C-SiC are listed in docs/physics_bulk.md).
``mode="isotropic"`` relaxes only the pressure; it is refused for non-cubic
crystals unless ``allow_noncubic=True`` and the deviatoric stress it leaves is
always reported.  :func:`run_bulk_relax` is the BulkRelaxProtocol runner.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import minimize, space
from scipy.optimize import minimize as scipy_minimize

from fracture_md.potentials.base import EnergyModel, compiled_kernels
from fracture_md.potentials.neighbors import NeighborHandle, build_neighbors
from fracture_md.structures.system import AtomicSystem


@dataclasses.dataclass
class RelaxInfo:
    converged: bool
    n_steps: int
    fmax: float  # eV/Å, max |F| over free atoms at the end
    energy: float  # eV
    method: str
    n_neighbor_rebuilds: int = 0
    message: str = ""
    # Filled by relax_cell only (optional, additive):
    stress_GPa: Optional[list] = None  # residual Cauchy stress, Voigt (xx,yy,zz,yz,xz,xy)
    max_stress_GPa: Optional[float] = None
    n_cell_steps: int = 0
    pressure_GPa: Optional[float] = None  # -tr(σ)/3 (positive = compressive)
    max_deviatoric_stress_GPa: Optional[float] = None  # max |σ - (tr σ / 3) I| over Voigt components
    zero_stress: Optional[bool] = None  # max |σ_i| < stress_tol_GPa (pressure *and* deviatoric)


def max_force(forces: np.ndarray, free: Optional[np.ndarray] = None) -> float:
    f = np.asarray(forces)
    if free is not None:
        f = f[np.asarray(free)]
    return float(np.max(np.linalg.norm(f, axis=1))) if f.size else 0.0


def deviatoric_voigt(sig) -> np.ndarray:
    """Deviatoric part σ - (tr σ / 3) I of a Voigt stress (xx, yy, zz, yz, xz, xy)."""
    d = np.array(sig, dtype=float)
    d[:3] -= np.mean(d[:3])
    return d


# RelaxSettings.optimizer -> relax_positions(method=...)
_OPTIMIZER_TO_METHOD = {"engine_default": "lbfgs", "lbfgs": "lbfgs", "bfgs": "bfgs", "cg": "cg", "fire": "fire"}
_SCIPY_METHODS = {"lbfgs": "L-BFGS-B", "bfgs": "BFGS", "cg": "CG"}


def position_method_for(optimizer: str) -> str:
    """Map a ``RelaxSettings.optimizer`` name to a :func:`relax_positions` method; unsupported names raise."""
    try:
        return _OPTIMIZER_TO_METHOD[optimizer]
    except KeyError:
        raise ValueError(f"optimizer {optimizer!r} is not supported by the JAX engine "
                         f"(supported: {sorted(_OPTIMIZER_TO_METHOD)})") from None


def relax_positions(
    model: EnergyModel,
    system: AtomicSystem,
    *,
    fmax: float = 1e-5,
    max_steps: int = 20000,
    fixed: Optional[np.ndarray] = None,
    skin: float = 0.5,
    method: str = "lbfgs",
    params=None,
    neighbors: Optional[NeighborHandle] = None,
    newton_polish: bool = True,
    max_newton_dof: int = 3000,
) -> tuple[AtomicSystem, RelaxInfo]:
    """Relax positions at fixed cell.  `fixed` is a boolean mask of frozen atoms.

    ``method`` is one of "lbfgs", "bfgs", "cg" (scipy, host loop) or "fire"
    (JAX-MD, on device); ``max_steps`` bounds the total number of energy /
    gradient evaluations (scipy) or FIRE steps.
    """
    if method not in ("lbfgs", "bfgs", "cg", "fire"):
        raise ValueError(f"unknown relax_positions method {method!r} (lbfgs, bfgs, cg, fire)")
    n = system.n_atoms
    free = np.ones(n, bool) if fixed is None else ~np.asarray(fixed, bool)
    species = model.species_for(system)
    box = jnp.asarray(system.box)
    binv = np.linalg.inv(np.asarray(box))
    p = model.params if params is None else params
    ef = compiled_kernels(model.energy_fn)["energy_and_forces"]  # compiled once per energy_fn (and shape)

    nh = neighbors if neighbors is not None else build_neighbors(system, model.cutoff, skin=skin)
    if method == "fire":
        return _relax_fire(model, system, species, p, nh, free, fmax, max_steps, skin, newton_polish, max_newton_dof)

    frac = np.asarray(system.frac)
    rebuilds = 0
    total_steps = 0
    x_ref = frac @ np.asarray(box).T  # real positions at last neighbour build
    fm_prev = np.inf
    label = f"{method}+newton" if newton_polish else method
    new_sys, e, fm = system, float("nan"), np.inf
    for _outer in range(20):
        x0 = frac @ np.asarray(box).T
        idx = nh.idx
        state = {"steps": 0}

        def fun(xf):
            x = x0.copy()
            x[free] = xf.reshape(-1, 3)
            fr = x @ binv.T
            e_, f_ = ef(jnp.asarray(fr), box, species, idx, p)
            state["steps"] += 1
            return float(e_), -np.asarray(f_)[free].ravel()

        opts = dict(maxiter=max(1, max_steps - total_steps), gtol=fmax / 10.0)
        if method == "lbfgs":
            opts.update(ftol=0.0, maxcor=30, maxls=50)
        res = scipy_minimize(fun, x0[free].ravel(), jac=True, method=_SCIPY_METHODS[method], options=opts)
        total_steps += state["steps"]
        x = x0.copy()
        x[free] = res.x.reshape(-1, 3)
        frac = (x @ binv.T) % 1.0
        moved = np.max(np.linalg.norm(x - x_ref, axis=1))
        # Always rebuild after an inner minimisation, then re-check the forces.
        new_sys = system.replace(frac=jnp.asarray(frac))
        nh = build_neighbors(new_sys, model.cutoff, skin=skin)
        rebuilds += 1
        x_ref = x
        e, f = ef(new_sys.frac, box, species, nh.idx, p)
        fm = max_force(np.asarray(f), free)
        if fm > fmax and newton_polish and 3 * int(free.sum()) <= max_newton_dof:
            # Energy-based line searches stall near fmax ~ 1e-7 eV/Å because the
            # energy change falls below float64 resolution of the total energy.
            # Newton steps with the exact autodiff Hessian converge on forces.
            new_sys, e, fm, k = _newton_polish(model, new_sys, species, p, nh, free, fmax)
            frac = np.asarray(new_sys.frac)
            total_steps += k
        if fm <= fmax:
            return new_sys, RelaxInfo(True, total_steps, fm, float(e), label, rebuilds, f"max displacement {moved:.3g} Å")
        if total_steps >= max_steps or fm >= 0.999 * fm_prev:
            break
        fm_prev = fm
    return new_sys, RelaxInfo(False, total_steps, fm, float(e), label, rebuilds,
                              f"not converged (fmax {fm:.2e} > {fmax:.2e} eV/Å after {total_steps} evaluations)")


def _newton_polish(model, system, species, p, nh, free, fmax, max_iter=12, max_step=0.05):
    """(Quasi-)Newton iterations on free real-space coordinates using the exact autodiff Hessian.

    The Hessian w.r.t. fractional coordinates is the real-space Hessian under
    JAX-MD's custom JVP (``physics.elastic.position_hessian``: batched
    Hessian-vector products, compiled once per energy function and shape).
    Zero modes (rigid translations when nothing is fixed) are removed with a
    pseudo-inverse.  The Hessian (the expensive part) is recomputed only when
    the max force fails to halve, otherwise its pseudo-inverse is reused (near
    a minimum the iteration then converges linearly with a tiny ratio);
    convergence is judged on the true forces only.
    """
    from fracture_md.physics.elastic import position_hessian  # local: elastic imports this module lazily

    kern = compiled_kernels(model.energy_fn)
    box = system.box
    binv = np.linalg.inv(np.asarray(box))
    idx = nh.idx
    frac = np.asarray(system.frac)
    freeidx = np.repeat(free, 3)
    k = 0
    hinv = None
    fm_prev = np.inf
    for k in range(1, max_iter + 1):
        g = -np.asarray(kern["forces"](jnp.asarray(frac), box, species, idx, p)).ravel()
        fm_k = max_force(-g.reshape(-1, 3), free)
        if fm_k <= fmax:
            k -= 1
            break
        if hinv is None or fm_k > 0.5 * fm_prev:
            h = position_hessian(model, system.replace(frac=jnp.asarray(frac)), idx, params=p, species=species)
            hinv = np.linalg.pinv(h[np.ix_(freeidx, freeidx)], rcond=1e-10, hermitian=True)
        fm_prev = fm_k
        step = -hinv @ g[freeidx]
        smax = np.max(np.abs(step))
        if smax > max_step:
            step *= max_step / smax
        dx = np.zeros(g.size)
        dx[freeidx] = step
        frac = (frac + dx.reshape(-1, 3) @ binv.T) % 1.0
    new_sys = system.replace(frac=jnp.asarray(frac))
    e, f = kern["energy_and_forces"](new_sys.frac, box, species, idx, p)
    return new_sys, float(e), max_force(np.asarray(f), free), k


@functools.lru_cache(maxsize=None)
def _fire_kernel(energy_fn, dt_start: float = 0.01, dt_max: float = 0.05):
    """Jitted FIRE minimisation (init + ``lax.while_loop``) for one energy function (cached).

    All arrays, fmax and the step budget are traced arguments, so the kernel
    compiles once per energy function and input shape.
    """
    grad = jax.grad(energy_fn, argnums=0)

    def run(frac, box, species, idx, p, free_j, mass, fmax, max_steps):
        _, shift = space.periodic_general(box, fractional_coordinates=True)

        def force_fn(fr, **kw):
            return -grad(fr, box, species, idx, p) * free_j

        init, apply = minimize.fire_descent(force_fn, shift, dt_start=dt_start, dt_max=dt_max)
        state = init(frac, mass=mass)

        def cond(c):
            st, i = c
            return (jnp.max(jnp.linalg.norm(st.force, axis=1)) > fmax) & (i < max_steps)

        def body(c):
            st, i = c
            return apply(st), i + 1

        state, steps = jax.lax.while_loop(cond, body, (state, jnp.asarray(0, dtype=jnp.int32)))
        return state.position % 1.0, steps

    return jax.jit(run)


def _relax_fire(model, system, species, p, nh, free, fmax, max_steps, skin, newton_polish, max_newton_dof):
    kern = compiled_kernels(model.energy_fn)
    fire = _fire_kernel(model.energy_fn)
    box = jnp.asarray(system.box)
    free_j = jnp.asarray(free, dtype=jnp.float64)[:, None]
    mass = jnp.asarray(system.masses_amu, dtype=jnp.float64)[:, None]
    s = system
    total, rebuilds = 0, 0
    fm, e = np.inf, float("nan")
    for _outer in range(20):
        budget = max(1, int(max_steps) - total)
        frac, steps = fire(jnp.asarray(s.frac), box, species, nh.idx, p, free_j, mass,
                           jnp.asarray(fmax, dtype=jnp.float64), jnp.asarray(budget, dtype=jnp.int32))
        total += int(steps)
        s = s.replace(frac=frac)
        # the list used inside FIRE may be stale if atoms moved > skin/2: rebuild and re-check the forces
        nh = build_neighbors(s, model.cutoff, skin=skin)
        rebuilds += 1
        e, f = kern["energy_and_forces"](s.frac, box, species, nh.idx, p)
        fm = max_force(np.asarray(f), free)
        if fm <= fmax or total >= max_steps:
            break
    label = "fire"
    if fm > fmax and newton_polish and 3 * int(free.sum()) <= max_newton_dof:
        s, e, fm, k = _newton_polish(model, s, species, p, nh, free, fmax)
        total += k
        label = "fire+newton"
    msg = ("forces re-checked with a neighbour list rebuilt at the result" if fm <= fmax else
           f"not converged (fmax {fm:.2e} > {fmax:.2e} eV/Å after {total} FIRE steps)")
    return s, RelaxInfo(bool(fm <= fmax), total, fm, float(e), label, rebuilds, msg)


# ---------------------------------------------------------------------------
# Cell relaxation (zero stress)
# ---------------------------------------------------------------------------
_ISOTROPIC_BASIS = np.array([[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]])


def relax_cell(
    model: EnergyModel,
    system: AtomicSystem,
    mode: str = "full",
    *,
    stress_tol_GPa: float = 1e-4,
    fmax: float = 1e-6,
    max_cell_steps: int = 60,
    skin: float = 0.5,
    max_strain_step: float = 0.02,
    params=None,
    max_newton_dof: int = 1500,
    allow_noncubic: bool = False,
    polish_steps: int = 1,
    position_method: str = "lbfgs",
    max_steps: int = 20000,
) -> tuple[AtomicSystem, RelaxInfo]:
    """Relax the cell to zero stress together with the internal coordinates.

    mode="full": all 6 symmetric strain components (triclinic freedom; rigid
      rotations are excluded by construction because ε is symmetric).
    mode="isotropic": a single scale factor, e = s (1,1,1,0,0,0); the loop is
      converged when the pressure |tr σ|/3 < tol.  Only for cubic crystals does
      the deviatoric stress vanish by symmetry (then this equals full
      relaxation), so the mode raises ``ValueError`` for a non-cubic input
      (spglib crystal system) unless ``allow_noncubic=True``.  In every case the
      deviatoric stress is returned (``max_deviatoric_stress_GPa``) and
      ``zero_stress`` says whether max |σ_i| < tol.

    Strain is applied as box -> (I + ε) box at fixed fractional coordinates;
    positions are then relaxed at fixed cell.  The strain step is Newton's
    step on E(e) with the exact relaxed-ion Hessian (falls back to a
    steepest-descent step with a nominal 100 GPa stiffness at any iteration
    where that Hessian is not positive definite, reported in the message),
    capped at ``max_strain_step`` per component.  After the tolerance is met,
    ``polish_steps`` extra Newton steps are taken (quadratic convergence: the
    final residual is typically orders of magnitude below the tolerance).
    Systems with more than ``max_newton_dof`` degrees of freedom use the
    clamped-ion Hessian (still converges, linearly) and ``relax_positions``
    (``position_method``, ``max_steps``) for the positions.

    Returns the relaxed system and a RelaxInfo whose ``stress_GPa`` holds the
    final Voigt stress.
    """
    from fracture_md.physics.elastic import (  # local import: elastic imports this module
        cauchy_stress_voigt,
        newton_relax_fixed_cell,
        relaxed_ion_correction,
        strain_derivatives,
        strained_box,
    )
    from fracture_md.units import EV_PER_A3_TO_GPA

    if mode not in ("full", "isotropic"):
        raise ValueError("mode must be 'full' or 'isotropic'")
    if mode == "isotropic" and not allow_noncubic:
        from fracture_md.structures.symmetry import space_group

        try:
            sg = space_group(system)
            cs, label = sg.crystal_system, f"{sg.international} (#{sg.number})"
        except RuntimeError:
            cs, label = "unknown", "spglib failed"
        if cs != "cubic":
            raise ValueError(f"isotropic cell relaxation of a non-cubic cell ({cs}: {label}) leaves deviatoric "
                             "stress; use mode='full' or pass allow_noncubic=True to force it")
    P = np.eye(6) if mode == "full" else _ISOTROPIC_BASIS
    p = model.params if params is None else params
    s = system
    ndof = 3 * s.n_atoms
    use_newton = ndof <= max_newton_dof
    nh = build_neighbors(s, model.cutoff, skin=skin)
    rebuilds, steps = 1, 0
    Kinv = None
    sd_iters: list[int] = []
    sig = np.full(6, np.nan)
    fm = np.inf
    e_tot = float("nan")
    converged = False
    polished = 0
    hr_min = float("nan")
    it = 0
    for it in range(max_cell_steps + polish_steps + 1):
        converged = False
        # --- positions at fixed cell
        g = -np.asarray(model.forces(s.frac, s.box, model.species_for(s), nh.idx, p))
        f0 = float(np.max(np.linalg.norm(g, axis=1)))
        if use_newton and f0 < 0.05:
            s, fm, k = newton_relax_fixed_cell(model, s, nh.idx, fmax=fmax, K_pinv=Kinv, params=p)
            steps += k
        else:
            fm = f0
        if fm > fmax:
            s, info = relax_positions(model, s, fmax=fmax, skin=skin, params=p, method=position_method,
                                      max_steps=max_steps)
            steps += info.n_steps
            fm = info.fmax
        nh = build_neighbors(s, model.cutoff, skin=skin)
        rebuilds += 1
        # --- stress and strain Hessian
        V = s.volume
        if use_newton:
            e_tot, _, dEde, Hee, Lam, K = strain_derivatives(model, s, nh.idx, params=p)
            corr, _, spec, Kinv = relaxed_ion_correction(K, Lam.reshape(-1, 6))
            H = Hee - corr
        else:
            from fracture_md.physics.elastic import _dE_de, _hess_ee

            sp = model.species_for(s)
            dEde = np.asarray(_dE_de(model.energy_fn, s.frac, s.box, sp, nh.idx, p))
            H = np.asarray(_hess_ee(model.energy_fn, s.frac, s.box, sp, nh.idx, p))
            e_tot = float(model.energy(s.frac, s.box, sp, nh.idx, p))
        sig = dEde / V * EV_PER_A3_TO_GPA
        crit = abs(np.mean(sig[:3])) if mode == "isotropic" else float(np.max(np.abs(sig)))
        gr = P.T @ dEde
        Hr = P.T @ H @ P
        Hr = 0.5 * (Hr + Hr.T)
        hr_min = float(np.min(np.linalg.eigvalsh(Hr)))
        if crit < stress_tol_GPa and fm <= fmax:
            converged = True
            # one (or polish_steps) extra Newton step: cheap, and quadratic convergence
            # takes the residual far below the tolerance; skipped at round-off level
            if polished >= polish_steps or crit < 1e-12 or hr_min <= 0:
                break
            polished += 1
        elif it >= max_cell_steps:
            break
        if hr_min > 0:
            d = -np.linalg.solve(Hr, gr)
        else:
            sd_iters.append(it)
            d = -gr / (V * 100.0 / EV_PER_A3_TO_GPA)
        e = P @ d
        emax = float(np.max(np.abs(e)))
        if emax > max_strain_step:
            e *= max_strain_step / emax
        s = s.replace(box=strained_box(s.box, jnp.asarray(e)))
        nh = build_neighbors(s, model.cutoff, skin=skin)
        rebuilds += 1
    cell_steps = it
    sig_final = cauchy_stress_voigt(model, s, nh.idx, params=p) * EV_PER_A3_TO_GPA
    dev = float(np.max(np.abs(deviatoric_voigt(sig_final))))
    msgs = []
    if sd_iters:
        msgs.append(f"steepest-descent fallback used at iteration(s) {sd_iters} (relaxed strain Hessian not "
                    "positive definite there)")
    if hr_min <= 0:
        msgs.append(f"relaxed strain Hessian NOT positive definite at the final state (min eigenvalue {hr_min:.3e} eV):"
                    " not a stable zero-stress minimum")
    zero = bool(np.max(np.abs(sig_final)) < stress_tol_GPa)
    if mode == "isotropic" and dev >= stress_tol_GPa:
        msgs.append(f"isotropic mode relaxed the pressure only: deviatoric stress max|σ - pI| = {dev:.3g} GPa remains "
                    "(NOT a zero-stress state)")
    if not converged:
        msgs.append(f"not converged after {cell_steps} cell steps")
    return s, RelaxInfo(
        converged=converged,
        n_steps=steps,
        fmax=float(fm),
        energy=float(e_tot),
        method=f"cell-{mode}: newton(strain, relaxed-ion Hessian) + "
        + ("newton(positions)" if use_newton else f"{position_method}(positions)"),
        n_neighbor_rebuilds=rebuilds,
        message="; ".join(msgs),
        stress_GPa=sig_final.tolist(),
        max_stress_GPa=float(np.max(np.abs(sig_final))),
        n_cell_steps=cell_steps,
        pressure_GPa=float(-np.mean(sig_final[:3])),
        max_deviatoric_stress_GPa=dev,
        zero_stress=zero,
    )


def isolated_atom_energy(model: EnergyModel, element: str, params=None) -> float:
    """Energy (eV) of one atom of ``element`` alone in a box larger than twice the cutoff.

    This is the model's own isolated-atom reference (for Tersoff-type bond
    order potentials it is exactly 0; for potentials with a non-zero
    one-body term, e.g. MLIPs, it is not).  No spin polarisation or other
    electronic effects beyond what the model encodes.
    """
    p = model.params if params is None else params
    L = 2.0 * (model.cutoff + 1.0) + 1.0
    frac = jnp.zeros((1, 3), dtype=jnp.float64) + 0.5
    box = jnp.eye(3, dtype=jnp.float64) * L
    species = jnp.asarray([model.elements.index(element)], dtype=jnp.int32)
    idx = jnp.full((1, 1), 1, dtype=jnp.int32)  # padding only: no neighbours
    return float(model.energy_fn(frac, box, species, idx, p))


def _initial_bulk_structure(candidate, model: EnergyModel, structure, skin: float, supercell=None) -> AtomicSystem:
    """Starting geometry: given AtomicSystem > candidate.structure > known phase builder; replicated for minimum image."""
    from ase import Atoms

    from fracture_md.structures.builders import sic_3c_conventional
    from fracture_md.structures.system import auto_supercell

    if isinstance(structure, AtomicSystem):
        atoms = structure.to_ase()
    elif candidate.structure is not None:
        r = candidate.structure
        atoms = Atoms(symbols=r.symbols, positions=r.positions_angstrom, cell=r.cell_rows_angstrom, pbc=r.pbc)
    elif candidate.phases and candidate.phases[0].name == "3C-SiC":
        atoms = sic_3c_conventional()
    else:
        raise ValueError(f"no structure for candidate {candidate.id!r}: pass `structure` or set candidate.structure")
    reps = tuple(supercell) if supercell is not None else auto_supercell(atoms, model.cutoff + skin)
    if tuple(reps) != (1, 1, 1):
        atoms = atoms.repeat(reps)
    return AtomicSystem.from_ase(atoms, elements=model.elements)


def run_bulk_relax(protocol, candidate, model: EnergyModel, structure=None, context=None):
    """BulkRelaxProtocol runner: zero-stress relaxation and basic bulk properties.

    Records: LATTICE_PARAMETER (conventional a for cubic; [a, b, c] otherwise),
    DENSITY, VOLUME_PER_ATOM, ENERGY_PER_ATOM (model zero), COHESIVE_ENERGY
    (E_isolated_atoms - E_bulk per atom, isolated atoms computed with the same
    model), RESIDUAL_STRESS (Voigt, GPa).  Checks: relaxation converged,
    residual stress < tol, fmax, space group (vs the candidate phase if given).

    ``protocol.relax.optimizer`` selects the position minimiser ("engine_default"
    = L-BFGS + Newton; "lbfgs", "bfgs", "cg", "fire"; see ``relax_positions``)
    and ``protocol.relax.max_steps`` bounds it; unsupported names raise.  In
    ``cell_mode="isotropic"`` only the pressure is relaxed: the input must be
    cubic unless ``context.extra["allow_noncubic_isotropic"]`` is true, and the
    residual_stress check always includes the deviatoric stress max|σ - pI|
    (if it exceeds the tolerance the check fails and ``stress_state`` says so).
    """
    from fracture_md import provenance
    from fracture_md.engines.base import ProtocolOutput, RunContext
    from fracture_md.records import make_record
    from fracture_md.schema.common import CheckResult, Conditions
    from fracture_md.schema.properties import MethodSpec, PropertyName
    from fracture_md.structures.symmetry import conventional_cell, lattice_parameters, space_group
    from fracture_md.units import AMU_PER_A3_TO_G_PER_CM3, EV_PER_A3_TO_GPA, Unit

    context = context or RunContext()
    skin = 0.5
    s0 = _initial_bulk_structure(candidate, model, structure, skin, protocol.supercell)
    fmax = protocol.relax.fmax_eV_per_A
    pmethod = position_method_for(protocol.relax.optimizer)
    if protocol.relax_cell:
        s, info = relax_cell(model, s0, mode=protocol.cell_mode, stress_tol_GPa=protocol.stress_tol_GPa, fmax=fmax,
                             skin=skin, position_method=pmethod, max_steps=protocol.relax.max_steps,
                             allow_noncubic=bool(context.extra.get("allow_noncubic_isotropic", False)))
    else:
        s, info = relax_positions(model, s0, fmax=fmax, max_steps=protocol.relax.max_steps, skin=skin, method=pmethod)
    nh = build_neighbors(s, model.cutoff, skin=skin)
    sp = model.species_for(s)
    e_tot = float(model.energy(s.frac, s.box, sp, nh.idx))
    f = np.asarray(model.forces(s.frac, s.box, sp, nh.idx))
    fm = max_force(f)
    from fracture_md.physics.elastic import cauchy_stress_voigt

    sig = cauchy_stress_voigt(model, s, nh.idx) * EV_PER_A3_TO_GPA
    n = s.n_atoms
    V = s.volume
    mass = float(np.sum(np.asarray(s.masses_amu)))
    rho = mass / V * AMU_PER_A3_TO_G_PER_CM3
    syms = s.symbols
    e_iso = {el: isolated_atom_energy(model, el) for el in sorted(set(syms))}
    e_coh = (sum(e_iso[x] for x in syms) - e_tot) / n

    sg = space_group(s)
    conv = conventional_cell(s)
    lengths, angles = lattice_parameters(np.asarray(conv.box if conv is not None else s.box))
    cubic = sg.crystal_system == "cubic"

    checks = [
        CheckResult(name="relaxation_converged", passed=bool(info.converged), details=info.method + " " + info.message),
        CheckResult(name="max_force", passed=fm <= max(fmax, 1e-12) * 1.0001, metric=fm, tolerance=fmax, unit=Unit.EV_PER_ANGSTROM),
    ]
    smax = float(np.max(np.abs(sig)))
    pres = abs(float(np.mean(sig[:3])))
    dev = float(np.max(np.abs(deviatoric_voigt(sig))))
    stress_ok = True
    if protocol.relax_cell:
        tol_s = protocol.stress_tol_GPa
        if protocol.cell_mode == "full":
            crit, det = smax, "max |σ_i|"
        else:
            # isotropic mode relaxes only the pressure; the deviatoric part must vanish too for "zero stress"
            crit = max(pres, dev)
            det = (f"max(|pressure|, max|σ - pI|) with |pressure| = {pres:.3e} GPa, max|σ - pI| = {dev:.3e} GPa "
                   "(isotropic mode relaxes the pressure only)")
        stress_ok = crit < tol_s
        checks.append(CheckResult(name="residual_stress", passed=stress_ok, metric=crit, tolerance=tol_s, unit=Unit.GPA,
                                  details=det))
    expected_sg = candidate.phases[0].space_group_number if candidate.phases else None
    if expected_sg is not None:
        checks.append(CheckResult(name="space_group", passed=sg.number == expected_sg, metric=float(sg.number),
                                  tolerance=0.0, details=f"spglib: {sg.international} (#{sg.number}); expected #{expected_sg}"))
    if cubic:
        dev = float(max(np.max(np.abs(lengths - lengths.mean())), 0.0))
        checks.append(CheckResult(name="cubic_cell_metric", passed=dev < 1e-6 and float(np.max(np.abs(angles - 90))) < 1e-6,
                                  metric=dev, tolerance=1e-6, unit=Unit.ANGSTROM,
                                  details=f"conventional a,b,c = {lengths.round(8).tolist()}, angles = {angles.round(6).tolist()}"))

    prov = provenance.collect(engine="jax_md", potential_id=model.card.id, protocol_id=protocol.protocol_id(),
                              requested_by=context.requested_by, implementation_ids=dict(context.implementation_ids),
                              parent_ids=list(context.parent_ids))
    method = MethodSpec(
        name="zero-stress bulk relaxation", protocol_id=protocol.protocol_id(), engine="jax_md",
        description=f"{info.method}; optimizer setting {protocol.relax.optimizer!r} -> positions {pmethod!r}"
                    + (f"; {info.message}" if info.message else ""),
        assumptions=["0 K lattice statics (no zero-point or thermal expansion)", "defect-free periodic crystal",
                     f"{n}-atom periodic cell"],
    )
    if not protocol.relax_cell:
        state = "fixed cell"
    elif stress_ok:
        state = "zero stress"
    elif protocol.cell_mode == "isotropic" and dev >= protocol.stress_tol_GPa:
        state = (f"NOT zero stress: hydrostatic pressure relaxed to {pres:.2e} GPa, deviatoric stress "
                 f"max|σ - pI| = {dev:.3g} GPa remains (isotropic cell relaxation of a non-cubic cell)")
    else:
        state = f"NOT zero stress: residual max|σ_i| = {smax:.3g} GPa exceeds the tolerance {protocol.stress_tol_GPa:g} GPa"
    cond = Conditions(stress_state=state)

    def rec(name, value, unit, shape="", notes=""):
        return make_record(name=name, value=value, unit=unit, candidate=candidate, method=method, card=model.card,
                           conditions=cond, checks=checks, frame="crystal" if shape else "not_applicable",
                           shape_description=shape, notes=notes, provenance=prov)

    if cubic:
        lat = rec(PropertyName.LATTICE_PARAMETER, float(lengths.mean()), Unit.ANGSTROM,
                  notes=f"conventional cubic a of {sg.international}")
    else:
        lat = rec(PropertyName.LATTICE_PARAMETER, lengths.tolist(), Unit.ANGSTROM, shape="conventional [a, b, c]",
                  notes=f"angles (alpha, beta, gamma) = {angles.round(6).tolist()} degree; {sg.international}")
    props = [
        lat,
        rec(PropertyName.DENSITY, rho, Unit.G_PER_CM3, notes="standard atomic masses (ASE) / relaxed volume"),
        rec(PropertyName.VOLUME_PER_ATOM, V / n, Unit.ANGSTROM3_PER_ATOM),
        rec(PropertyName.ENERGY_PER_ATOM, e_tot / n, Unit.EV_PER_ATOM, notes="relative to the potential's own energy zero"),
        rec(PropertyName.COHESIVE_ENERGY, e_coh, Unit.EV_PER_ATOM,
            notes="E_isolated_atoms - E_bulk per atom; isolated-atom energies from the same model "
                  f"({ {k: round(v, 10) for k, v in e_iso.items()} } eV); positive = bound"),
        rec(PropertyName.RESIDUAL_STRESS, sig.tolist(), Unit.GPA, shape="voigt 6 (xx,yy,zz,yz,xz,xy), positive = tensile"),
    ]
    out = ProtocolOutput(properties=props, checks=checks, structure=s)
    out.artifacts.update(stress_GPa=sig, conventional_cell=conv, lattice_lengths_A=lengths, lattice_angles_deg=angles,
                         isolated_atom_energies_eV=e_iso)
    out.raw.update(relax_info=dataclasses.asdict(info), space_group={"number": sg.number, "symbol": sg.international},
                   n_atoms=n, energy_eV=e_tot, cohesive_energy_eV_per_atom=e_coh, density_g_cm3=rho,
                   lattice_a_A=float(lengths.mean()) if cubic else lengths.tolist())
    return out
