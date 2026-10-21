"""Regression gates: the numerical checks agent-proposed code must pass before it is trusted.

Two kinds of implementation can be registered (see ``agents.registry``):

``potential_energy_fn``
    A module defining ``energy(frac, box, species, idx, params) -> eV`` and
    ``BASE_POTENTIAL`` (registry name of the trusted potential whose parameter
    pytree, element order and cutoff it uses).  The gate evaluates it on a
    fixed set of reference configurations of the base potential's material
    **plus configurations drawn at gate time from a fresh random seed**
    (recorded in the manifest): random supercell shapes and sizes (orthorhombic
    96-216 atoms, triclinic primitive-cell supercells of 54-96 atoms, a
    vacancy cell with an odd atom count), random strain, expansion and rattle.
    Every cell is replicated as needed for the minimum-image rule of the base
    potential's cutoff + 0.5 Å skin (:data:`GATE_SKIN_A`); for the short-range
    bond-order potentials (cutoff <= 3 Å) this changes nothing, for Vashishta
    (7.35 Å) the cells grow to 511-1728 atoms.
    Code that special-cases the fixed gate inputs (atom counts, cell shapes)
    cannot know these in advance.  The gate requires

    (a) it jit-compiles and returns a finite float64 scalar;
    (b) autodiff forces agree with central finite differences of the energy
        (atoms displaced in *real* space, frac + dR @ inv(box).T);
    (c) translation invariance; (d) rigid-rotation invariance of E and
        covariance of F; (e) invariance under permutation of like atoms;
    (f) zero net force (Newton's third law / translation invariance);
    (g) energy conservation along a short velocity-Verlet NVE trajectory, and
        second-order (dt²) convergence of the energy error when dt is halved
        (an energy discontinuity crossed during the run does not converge);
        plus a dimer energy scan from 1.5 Å to beyond the cutoff (so across
        the whole switching shell [R1, R2]) for every element pair, with a
        jump detector (ΔE between neighbouring grid points vs the trapezoid of
        the autodiff bond force), which catches energy discontinuities that a
        near-equilibrium NVE run never crosses.  (An NVE run on a lattice
        stretched into the cutoff shell is not usable: at +25 % the trusted
        Tersoff lattice is mechanically unstable and atoms move > 6 Å.);
    (h) regression against the trusted base implementation on every reference
        configuration: |ΔE|/atom, max |ΔF| and max |Δσ| below tolerance;
    (i) symmetry of the cell virial (∂E/∂h) hᵀ/V, i.e. no spurious torque
        from the cell derivative.

    (h) is what makes the gate a *regression* gate: an implementation can be
    internally consistent (b-f, i) and still compute a different model — the
    demo ``tersoff_hard_cutoff`` does exactly that (it is also caught by the
    discontinuity checks in (g)).

``protocol_runner``
    A module defining ``run(protocol, candidate, model, structure=None,
    context=None) -> ProtocolOutput`` for one protocol kind.  The gate re-runs
    every stored reference case for that kind (``data/reference_results/*.json``)
    and compares the produced property values (and traction-separation curves,
    resampled on the reference grid) with the stored ones, requires at least
    one runner check and all of the runner's own checks to pass, and applies
    kind-specific identities (cleavage: ∫T dδ = ΔE/A; elastic: symmetric, and
    positive definite when the reference is; indentation: the IndentationCurve is
    produced).  Indentation reference cases are small 0 K quasi-static runs
    (tests/data/make_indentation_reference_qs.py): an MD trajectory is chaotic and
    its numbers do not reproduce across platforms; production indentation records
    are built outside submit() (agents.interface).  Stored numbers alone can be
    replayed by a lookup table, so every case is also re-run as a **perturbed
    case** drawn from the gate's random seed -- a perturbed model
    E'(S, h) = s·E(S, λh) with a new card id (and a jittered protocol grid
    where the protocol has one) -- and compared with the *trusted built-in
    runner* on the same perturbed case.  Reference files that widen the
    default tolerances fail the gate.  No reference case -> the gate fails with
    an explicit "no reference" check rather than trusting untested code.

The fixed part of the gate is deterministic; the random part is reproducible
from the seed stored in the manifest (``gate.random_seed``; pass ``seed=`` to
re-run it).  Runs on CPU in float64.  The tolerances below are part of the gate
version: changing them bumps ``GATE_VERSION`` and is recorded in every manifest
the gate writes; the registry trusts only manifests gated with the current
version.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import secrets
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from fracture_md.physics.md import TIME_UNIT_FS
from fracture_md.schema.common import CheckResult
from fracture_md.units import BOLTZMANN_EV_K, EV_PER_A3_TO_GPA, Unit, UnitError, convert

# 0.2.0: random gate-time configurations, stretched-lattice NVE + dimer discontinuity scan, runner gate
#        compares against the trusted built-in runner on a perturbed case, >= 1 runner check, curve
#        comparison, reference files may not widen tolerances.
GATE_VERSION = "0.2.0"

# --------------------------------------------------------------------------
# Tolerances (float64, CPU).  Units in the names.
# --------------------------------------------------------------------------
TOL = {
    # (b) |F_autodiff - F_fd| (eV/Å); central FD, h = 1e-5 Å: truncation ~h²·E''' ~ 1e-8, round-off ~1e-9
    "fd_force_eV_per_A": 1.0e-6,
    "fd_step_A": 1.0e-5,
    # (c)-(e) total-energy invariances (eV) and force covariance (eV/Å); round-off is ~1e-11
    "invariance_energy_eV": 1.0e-8,
    "invariance_force_eV_per_A": 1.0e-8,
    # (f) |Σ_i F_i| (eV/Å)
    "net_force_eV_per_A": 1.0e-8,
    # (g) NVE, 600 K, 200 fs: max |E_tot(t) - E_tot(0)| per atom at dt = 0.5 fs (eV/atom), and the ratio of
    #     that error at dt and dt/2.  Velocity Verlet conserves a shadow energy, so for a smooth energy with
    #     consistent forces the error falls as dt² (ratio ~4); an energy discontinuity crossed during the run
    #     gives an O(1) error that does not shrink (ratio ~1).  Measured for the trusted kernels (64 atoms):
    #     Tersoff 1989 9.7e-5 -> 2.5e-5 eV/atom (ratio 3.8); Erhart-Albe 4.6e-5 -> 1.4e-5 (ratio 3.3);
    #     k_B T = 0.052 eV at 600 K.
    "nve_fluctuation_eV_per_atom": 5.0e-4,
    "nve_min_dt_halving_ratio": 2.5,
    # (g') dimer scan, grid step 2e-3 Å: |E(r+dr) - E(r) - ½(E'(r) + E'(r+dr)) dr| with the autodiff bond force.
    #      Smooth kernels: truncation dr³/12·|d³E/dr³| ~ 1e-6 eV; the C¹ kinks of the Tersoff sine cutoff at R1/R2
    #      give O(dr²·ΔE'') = 6.0e-5 eV (Tersoff Si-C at R1 = 2.204 Å; Erhart-Albe <= 1.4e-6).  The demo hard step
    #      gives 1.8-3.1 eV.  Any energy step larger than 1 meV is caught.
    "dimer_scan_step_A": 2.0e-3,
    "dimer_jump_eV": 1.0e-3,
    # (h) regression vs trusted base
    "regression_energy_eV_per_atom": 1.0e-8,
    "regression_force_eV_per_A": 1.0e-7,
    "regression_stress_GPa": 1.0e-6,
    # (i) antisymmetric part of the cell virial (GPa)
    "virial_asymmetry_GPa": 1.0e-6,
}
NVE_SETTINGS = dict(temperature_K=600.0, dt_fs=0.5, n_steps=400, seed=0)  # dt/2 run uses 2 x n_steps
# Neighbour-list skin used for every gate configuration; cells are replicated until every periodic width
# exceeds 2 (cutoff + GATE_SKIN_A) (structures.system.auto_supercell).  The 64-atom 2x2x2 3C cell satisfies this
# for cutoffs up to ~3.8 Å, so the configurations of the short-range potentials are those of gate 0.2.0 exactly.
GATE_SKIN_A = 0.5
# The fixed-list NVE run (g) uses a wider skin: the list is valid while no atom moves more than skin/2, and in the
# 512-atom cell a long-range base needs (Vashishta, softer in shear) the largest 600 K excursion over 200 fs is
# 0.26 Å (64-atom Tersoff cell: < 0.25 Å).  Entries beyond the cutoff contribute exact zeros, so the energies are
# those of the 0.5 Å list.
GATE_NVE_SKIN_A = 1.0
# TIME_UNIT_FS (sqrt(amu Å²/eV) in fs) is imported from physics.md, which derives it from CODATA constants.

# Default property-comparison tolerances for protocol-runner regression
# (value_new vs value_ref: |Δ| <= atol + rtol |ref|).  Reference files may override.
RUNNER_RTOL = 1.0e-5
RUNNER_ATOL_BY_UNIT = {
    "GPa": 1.0e-2,
    "J/m^2": 1.0e-4,
    "angstrom": 1.0e-6,
    "eV/atom": 1.0e-7,
    "eV": 1.0e-6,
    "g/cm^3": 1.0e-6,
    "angstrom^3/atom": 1.0e-5,
    "1": 1.0e-6,
}
# Cleavage identity ∫T dδ = [E(δ_max) - E(0)]/A, checked two ways:
#  * the runner's reported integrated_work vs work_of_separation (runners integrate on a fine grid; the
#    trusted JAX runner reaches ~1e-5 relative), and
#  * an independent trapezoid over the *reported* curve samples, whose discretisation error on the default
#    25-point grid is ~1.6 % for Tersoff (111) (peaked, only C1-smooth curve), hence the loose bound.
CLEAVAGE_INTEGRAL_RTOL = 1.0e-3
CLEAVAGE_TRAPEZOID_RTOL = 5.0e-2
# Traction-separation curves (new vs reference, new resampled on the reference opening grid):
# max |ΔT| <= CURVE_ATOL_GPA + CURVE_RTOL·max|T_ref| and max |Δ(E/A)| <= CURVE_ATOL_J_PER_M2 + CURVE_RTOL·max|E/A_ref|.
CURVE_RTOL = 1.0e-4
CURVE_ATOL_GPA = 1.0e-2
CURVE_ATOL_J_PER_M2 = 1.0e-4
# Perturbed runner case: E'(S, h) = s·E(S, λh), s and λ drawn from the gate seed in these ranges.  λ > 1 keeps
# every neighbour list built with the unperturbed cutoff a superset (effective cutoff = cutoff/λ).
PERTURB_ENERGY_SCALE = (1.02, 1.08)
PERTURB_LENGTH_SCALE = (1.01, 1.03)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_DIRS = (
    _REPO_ROOT / "data" / "reference_results",
    Path(__file__).resolve().parents[1] / "data" / "reference_results",
)


@dataclasses.dataclass
class GateReport:
    impl_id: str
    kind: str
    passed: bool
    checks: list[CheckResult]
    gate_version: str = GATE_VERSION
    duration_s: float = 0.0
    seed: Optional[int] = None

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        n = len(self.checks)
        head = f"{self.impl_id}: {'PASSED' if self.passed else 'FAILED'} ({n - len(self.failed)}/{n} checks)"
        return "\n".join([head] + [f"  FAIL {c.name}: {c.details}" for c in self.failed])

    def to_dict(self) -> dict:
        return {
            "impl_id": self.impl_id,
            "kind": self.kind,
            "passed": self.passed,
            "gate_version": self.gate_version,
            "duration_s": self.duration_s,
            "random_seed": self.seed,
            "checks": [c.model_dump(mode="json") for c in self.checks],
        }


def _check(name, passed, metric=None, tol=None, unit=None, details="") -> CheckResult:
    return CheckResult(
        name=name,
        passed=bool(passed),
        metric=None if metric is None or not np.isfinite(metric) else float(metric),
        tolerance=None if tol is None else float(tol),
        unit=unit,
        details=details if (metric is None or np.isfinite(metric)) else f"{details} (metric non-finite: {metric})",
    )


def _guard(name: str, fn: Callable[[], list[CheckResult]]) -> list[CheckResult]:
    """Run a check group; an exception becomes a failed check instead of aborting the gate."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - untrusted code may raise anything
        tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
        return [_check(name, False, details=f"raised {tb}")]


# ==========================================================================
# Reference configurations for potential gates
# ==========================================================================
@dataclasses.dataclass
class RefConfig:
    name: str
    system: Any  # AtomicSystem
    species: jnp.ndarray
    idx: jnp.ndarray
    description: str


def _replicate_for(atoms, base, periodic_axes=(0, 1, 2)):
    """``atoms`` repeated so that the minimum-image rule holds for the base cutoff + GATE_SKIN_A (no-op if it does)."""
    from fracture_md.structures.system import auto_supercell

    reps = auto_supercell(atoms, base.cutoff + GATE_SKIN_A, periodic_axes=periodic_axes)
    return atoms if reps == (1, 1, 1) else atoms.repeat(reps)


def _conventional_repeats(base, a0: float) -> int:
    """Scale factor for conventional-cell repeat counts: 1 for cutoff + skin <= a0 (Tersoff, Erhart-Albe)."""
    return max(1, int(np.ceil((base.cutoff + GATE_SKIN_A) / a0)))


def _equilibrium_a_3c(base) -> float:
    """Cubic lattice parameter minimising the base potential's energy (unrelaxed zincblende)."""
    from scipy.optimize import minimize_scalar

    from fracture_md.potentials import build_neighbors
    from fracture_md.structures.builders import sic_3c_conventional
    from fracture_md.structures.system import AtomicSystem

    # 2x2x2 for short cutoffs; enough cubes for the minimum image down to a = 4.2 Å otherwise
    n = max(2, int(np.floor(2.0 * (base.cutoff + GATE_SKIN_A) / 4.2)) + 1)
    s0 = AtomicSystem.from_ase(sic_3c_conventional(4.36).repeat((n, n, n)), elements=base.elements)
    nh = build_neighbors(s0, base.cutoff, skin=GATE_SKIN_A)
    sp = base.species_for(s0)
    f = jax.jit(lambda a: base.energy_fn(s0.frac, jnp.eye(3) * n * a, sp, nh.idx, base.params))
    r = minimize_scalar(lambda a: float(f(a)), bracket=(4.2, 4.33, 4.5), tol=1e-10)
    return float(r.x)


def reference_configurations(base) -> list[RefConfig]:
    """Deterministic reference configurations for a Si-C base potential.

    Names and cells below are those of short-range potentials; every cell is replicated for the minimum-image
    rule of the base cutoff + GATE_SKIN_A (e.g. 4x4x4 = 512 atoms for Vashishta) and the slab vacuum is at least
    cutoff + skin on each side.

    * ``eq_2x2x2``: perfect 3C-SiC 2x2x2 (64 atoms) at the base potential's a0.
    * ``rattled_sheared``: σ=0.05 Å rattle + random 1 % cell strain.
    * ``stretched_10/18/25``: isotropic expansion by 10/18/25 % plus a 0.03 Å
      rattle; nearest-neighbour Si-C distances then lie inside the smooth
      cutoff shell (R1 = 2.20 Å, R2 = 2.51 Å for Tersoff Si-C), i.e. the
      bond-breaking regime probed by cleavage and ideal strength.
    * ``slab_111``: (111) slab, 2x2 in-plane repeats (96 atoms), 5 Å vacuum
      on each side (free surfaces with under-coordinated atoms).
    """
    from ase.build import surface

    from fracture_md.potentials import build_neighbors
    from fracture_md.structures.builders import sic_3c_conventional
    from fracture_md.structures.system import AtomicSystem

    if not {"Si", "C"} <= set(base.elements):
        raise ValueError(
            f"potential gate reference configurations are defined for Si-C potentials; base has {base.elements}"
        )
    a0 = _equilibrium_a_3c(base)
    rng = np.random.default_rng(20260928)
    configs: list[tuple[str, Any, str]] = []

    at = sic_3c_conventional(a0).repeat((2, 2, 2))
    configs.append(("eq_2x2x2", at, f"perfect 3C-SiC 2x2x2 at a0 = {a0:.5f} Å"))

    at = sic_3c_conventional(a0).repeat((2, 2, 2))
    at.rattle(0.05, seed=11)
    e = 0.01 * rng.standard_normal((3, 3))
    at.set_cell(at.cell.array @ (np.eye(3) + 0.5 * (e + e.T)), scale_atoms=True)
    configs.append(("rattled_sheared", at, "rattle 0.05 Å + random symmetric 1 % strain"))

    for pct, seed in [(10, 12), (18, 13), (25, 14)]:
        at = sic_3c_conventional(a0 * (1.0 + pct / 100.0)).repeat((2, 2, 2))
        at.rattle(0.03, seed=seed)
        configs.append((f"stretched_{pct}", at, f"isotropic +{pct} % expansion + rattle 0.03 Å (cutoff shell)"))

    vac = max(5.0, base.cutoff + GATE_SKIN_A)
    slab = surface(sic_3c_conventional(a0), (1, 1, 1), 3, vacuum=vac, periodic=False).repeat((2, 2, 1))
    slab.rattle(0.02, seed=15)
    slab.pbc = True
    slab = _replicate_for(slab, base, periodic_axes=(0, 1))
    configs.append(("slab_111", slab, f"(111) slab, {len(slab)} atoms, {vac:g} Å vacuum each side, rattle 0.02 Å"))

    out = []
    for name, atoms, desc in configs:
        n0 = len(atoms)
        atoms = _replicate_for(atoms, base) if name != "slab_111" else atoms
        if len(atoms) != n0:
            desc = f"{desc} [replicated to {len(atoms)} atoms for the {base.cutoff:g} Å cutoff]"
        s = AtomicSystem.from_ase(atoms, elements=base.elements)
        nh = build_neighbors(s, base.cutoff, skin=GATE_SKIN_A)
        out.append(RefConfig(name, s, base.species_for(s), nh.idx, desc))
    return out


def new_gate_seed() -> int:
    """A fresh, unpredictable seed for the random part of a gate run (recorded in the manifest)."""
    return secrets.randbits(63)


def random_configurations(base, seed: int, a0: Optional[float] = None) -> list[RefConfig]:
    """Configurations drawn from `seed` at gate time (sizes, shapes and distortions unknown in advance).

    * ``random_ortho``: conventional supercell with random repeats (96-216 atoms), random expansion
      0-15 %, random symmetric strain (2 %) and rattle 0.02-0.08 Å.
    * ``random_triclinic``: primitive 2-atom cell repeated 3-4 times per axis (54-128 atoms), mapped by a
      random general (non-symmetric) deformation gradient -> triclinic cell; rattle.
    * ``random_vacancy``: 2x2x2 conventional cell with one random atom removed (63 atoms, odd N),
      expanded 5-25 % (bonds towards/inside the cutoff shell) and rattled.

    Every draw is checked against the minimum-image condition (cutoff + 0.5 Å skin) and redrawn if needed.
    For long-range bases the repeat counts are scaled before drawing (conventional cells by
    ceil((cutoff + skin)/a0), triclinic repeats start at the count needed for the cutoff), so the random
    streams and cells of the short-range potentials are unchanged.
    """
    from fracture_md.potentials import build_neighbors
    from fracture_md.potentials.neighbors import MinimumImageError
    from fracture_md.structures.builders import sic_3c_conventional, sic_3c_primitive
    from fracture_md.structures.system import AtomicSystem

    a0 = _equilibrium_a_3c(base) if a0 is None else a0
    rng = np.random.default_rng(int(seed))
    k = _conventional_repeats(base, a0)
    w_prim = a0 / np.sqrt(3.0)  # {111} spacing = perpendicular width of the fcc primitive cell
    need = 2.0 * (base.cutoff + GATE_SKIN_A) / w_prim
    lo_tri = 3 if need <= 3 else int(np.ceil(need)) + 1

    def ortho():
        reps = [(3, 2, 2), (2, 3, 2), (2, 2, 3), (3, 3, 2), (3, 2, 3), (2, 3, 3), (3, 3, 3)][rng.integers(7)]
        reps = tuple(k * r for r in reps)
        expand = rng.uniform(0.0, 0.15)
        at = sic_3c_conventional(a0 * (1 + expand)).repeat(reps)
        e = 0.02 * rng.standard_normal((3, 3))
        at.set_cell(at.cell.array @ (np.eye(3) + 0.5 * (e + e.T)), scale_atoms=True)
        sig = rng.uniform(0.02, 0.08)
        at.rattle(sig, seed=int(rng.integers(2**31)))
        return "random_ortho", at, f"{reps} conventional, +{100 * expand:.1f} % expansion, 2 % strain, rattle {sig:.3f} Å"

    def triclinic():
        reps = tuple(int(x) for x in rng.integers(lo_tri, lo_tri + 2, size=3))
        at = sic_3c_primitive(a0).repeat(reps)
        f = (1.0 + rng.uniform(0.02, 0.10)) * (np.eye(3) + 0.03 * rng.standard_normal((3, 3)))
        at.set_cell(at.cell.array @ f.T, scale_atoms=True)
        sig = rng.uniform(0.02, 0.06)
        at.rattle(sig, seed=int(rng.integers(2**31)))
        return "random_triclinic", at, f"primitive x {reps}, random deformation gradient, rattle {sig:.3f} Å"

    def vacancy():
        expand = rng.uniform(0.05, 0.25)
        at = sic_3c_conventional(a0 * (1 + expand)).repeat((2 * k, 2 * k, 2 * k))
        i = int(rng.integers(len(at)))
        del at[i]
        sig = rng.uniform(0.02, 0.05)
        at.rattle(sig, seed=int(rng.integers(2**31)))
        return ("random_vacancy", at, f"{2 * k}x{2 * k}x{2 * k} minus atom {i} (N={len(at)}), +{100 * expand:.1f} % "
                f"expansion, rattle {sig:.3f} Å")

    out = []
    for make in (ortho, triclinic, vacancy):
        for _attempt in range(20):
            name, atoms, desc = make()
            s = AtomicSystem.from_ase(atoms, elements=base.elements)
            try:
                nh = build_neighbors(s, base.cutoff, skin=GATE_SKIN_A)
            except MinimumImageError:
                continue
            out.append(RefConfig(name, s, base.species_for(s), nh.idx, f"{desc} [seed {seed}]"))
            break
    return out


def dimer_scan(energy_fn, params, base, pair: tuple[int, int], *, r_min: float = 1.5, box_A: float = 25.0,
               dr: Optional[float] = None) -> dict[str, Any]:
    """E(r) of an isolated dimer on a fine grid up to the cutoff, with an energy-force consistency jump detector.

    For consecutive grid points the energy change must equal the trapezoid of the autodiff bond force,
    ΔE_k = ½(E'(r_k) + E'(r_k+1)) dr, up to the O(dr³) truncation.  A discontinuity in E (step cutoff,
    branch switch) shows up as ΔE_k with no matching force.  Returns metrics in eV and Å.
    """
    from fracture_md.potentials import all_pairs_idx

    dr = TOL["dimer_scan_step_A"] if dr is None else dr
    r = np.arange(r_min, base.cutoff + 0.05, dr)
    box = jnp.eye(3) * box_A
    species = jnp.asarray(pair, dtype=jnp.int32)
    idx = all_pairs_idx(2)

    def e_and_dedr(rr):
        frac = jnp.stack([jnp.full(3, 0.3), jnp.full(3, 0.3) + jnp.array([rr, 0.0, 0.0]) / box_A])
        e = energy_fn(frac, box, species, idx, params)
        # grad w.r.t. frac is the REAL-space gradient under the kernels' (JAX-MD) convention, as in check (b);
        # differentiating through the user-side map r -> frac would be off by the box length.
        g = jax.grad(energy_fn, argnums=0)(frac, box, species, idx, params)
        return e, g[1, 0]

    ef = jax.jit(jax.vmap(e_and_dedr))
    e, de = (np.asarray(a) for a in ef(jnp.asarray(r)))
    jump = np.abs(np.diff(e) - 0.5 * (de[1:] + de[:-1]) * dr)
    k = int(np.nanargmax(jump)) if jump.size and np.all(np.isfinite(jump)) else 0
    return dict(max_jump_eV=float(np.max(jump)) if np.all(np.isfinite(jump)) else float("nan"),
                at_r_A=float(r[k]), n_points=int(r.size), r_range_A=(float(r[0]), float(r[-1])),
                energy_range_eV=(float(np.min(e)), float(np.max(e))))


# ==========================================================================
# NVE (velocity Verlet with lax.scan)
# ==========================================================================
def nve_energy_conservation(
    energy_fn,
    params,
    system,
    species,
    idx,
    *,
    temperature_K: float = 600.0,
    dt_fs: float = 0.5,
    n_steps: int = 400,
    seed: int = 0,
) -> dict[str, float]:
    """Velocity-Verlet NVE with a fixed neighbour list; returns conservation metrics.

    Units: eV, Å, amu, time unit sqrt(amu Å²/eV) = 10.1805 fs.  Initial
    velocities are Maxwell-Boltzmann at `temperature_K` (net momentum removed).
    The neighbour list is fixed, so the run is valid only if no atom moves more
    than skin/2 (reported as ``max_displacement_A``; the caller checks it).
    Positions are propagated in real space and mapped to fractional
    coordinates with inv(box); forces are -grad wrt frac (real-space gradient
    under JAX-MD's fractional-coordinate convention).
    """
    box = system.box
    m = system.masses_amu[:, None]
    mn = np.asarray(m)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((system.n_atoms, 3)) * np.sqrt(BOLTZMANN_EV_K * temperature_K / mn)
    v -= (mn * v).sum(0) / mn.sum()
    v = jnp.asarray(v)
    binv_t = jnp.linalg.inv(box).T
    dt = dt_fs / TIME_UNIT_FS

    def e_fn(fr):
        return energy_fn(fr, box, species, idx, params)

    grad = jax.grad(e_fn)

    def step(carry, _):
        fr, vel, acc = carry
        fr = fr + (vel * dt + 0.5 * acc * dt * dt) @ binv_t
        acc_new = -grad(fr) / m
        vel = vel + 0.5 * (acc + acc_new) * dt
        etot = e_fn(fr) + 0.5 * jnp.sum(m * vel * vel)
        return (fr, vel, acc_new), (etot, fr)

    @jax.jit
    def run(fr0, v0):
        a0 = -grad(fr0) / m
        e0 = e_fn(fr0) + 0.5 * jnp.sum(m * v0 * v0)
        _, (etot, frs) = jax.lax.scan(step, (fr0, v0, a0), None, length=n_steps)
        return e0, etot, frs

    e0, etot, frs = run(system.frac, v)
    etot = np.asarray(etot)
    dfr = np.asarray(frs) - np.asarray(system.frac)[None]
    dfr -= np.round(dfr)
    disp = float(np.max(np.linalg.norm(dfr @ np.asarray(box).T, axis=-1)))
    t = np.arange(1, n_steps + 1) * dt_fs
    dev = (etot - float(e0)) / system.n_atoms
    slope = np.polyfit(t, dev, 1)[0] if np.all(np.isfinite(dev)) else np.nan
    return dict(
        fluctuation_eV_per_atom=float(np.max(np.abs(dev))),
        drift_eV_per_atom=float(abs(slope) * (t[-1] - t[0])),
        max_displacement_A=disp,
        duration_fs=float(n_steps * dt_fs),
    )


# ==========================================================================
# Potential gate
# ==========================================================================
def run_potential_gate(module, *, base=None, configs: Optional[list[RefConfig]] = None,
                       seed: Optional[int] = None) -> list[CheckResult]:
    """All potential_energy_fn checks for an imported candidate module (no manifest side effects).

    `seed` draws the random gate-time configurations (:func:`random_configurations`); None draws a fresh
    seed.  Pass `configs` to replace the fixed *and* random sets (tests only).
    """
    from fracture_md.potentials import get_potential

    checks: list[CheckResult] = []
    efn = getattr(module, "energy", None)
    base_name = getattr(module, "BASE_POTENTIAL", None)
    ok = callable(efn) and isinstance(base_name, str)
    checks.append(_check(
        "defines_interface", ok,
        details="module defines energy(frac, box, species, idx, params) and BASE_POTENTIAL"
        if ok else f"energy callable: {callable(efn)}, BASE_POTENTIAL: {base_name!r}",
    ))
    if not ok:
        return checks
    if base is None:
        try:
            base = get_potential(base_name)
        except KeyError as exc:
            return checks + [_check("base_potential_resolves", False, details=str(exc))]
    checks.append(_check("base_potential_resolves", True, details=f"{base_name} -> {base.card.id}"))
    if configs is None:
        seed = new_gate_seed() if seed is None else int(seed)
        configs = reference_configurations(base)
        rand = random_configurations(base, seed)
        checks.append(_check("random_configurations_drawn", len(rand) == 3, len(rand), 3, None,
                             f"seed {seed}: " + "; ".join(f"{c.name}: N={c.system.n_atoms}, {c.description}"
                                                          for c in rand)))
        configs = configs + rand
    p = base.params
    e_new = jax.jit(efn)
    e_ref = jax.jit(base.energy_fn)
    g_new = jax.jit(jax.grad(efn, argnums=0))
    g_ref = jax.jit(jax.grad(base.energy_fn, argnums=0))

    def cell_grad(fn):
        return jax.jit(jax.grad(fn, argnums=1))

    gb_new, gb_ref = cell_grad(efn), cell_grad(base.energy_fn)

    def stress(gb, c):  # symmetric Cauchy stress, GPa, positive = tensile
        w = np.asarray(gb(c.system.frac, c.system.box, c.species, c.idx, p)) @ np.asarray(c.system.box).T
        return 0.5 * (w + w.T) / c.system.volume * EV_PER_A3_TO_GPA

    by_name = {c.name: c for c in configs}

    # (a) jit-compiles, finite float64 scalar
    def g_a():
        out = []
        for c in configs:
            e = e_new(c.system.frac, c.system.box, c.species, c.idx, p)
            good = e.shape == () and bool(jnp.isfinite(e)) and e.dtype == jnp.float64
            out.append(_check(f"jit_finite_scalar[{c.name}]", good, details=f"E = {float(e):.10f} eV, dtype {e.dtype}"))
        return out

    checks += _guard("jit_finite_scalar", g_a)
    if not all(c.passed for c in checks):
        return checks

    # (b) FD forces
    def g_b():
        out = []
        h = TOL["fd_step_A"]
        for name in ("rattled_sheared", "stretched_18", "slab_111", "random_triclinic", "random_vacancy"):
            if name not in by_name:
                continue
            c = by_name[name]
            s = c.system
            f = -np.asarray(g_new(s.frac, s.box, c.species, c.idx, p))
            binv_t = np.linalg.inv(np.asarray(s.box)).T
            errs = []
            for i in sorted({0, 5, s.n_atoms // 3, s.n_atoms // 2 + 1, s.n_atoms - 1}):
                for ax in range(3):
                    dr = np.zeros((s.n_atoms, 3))
                    dr[i, ax] = h
                    ds = jnp.asarray(dr @ binv_t)
                    ep = float(e_new(s.frac + ds, s.box, c.species, c.idx, p))
                    em = float(e_new(s.frac - ds, s.box, c.species, c.idx, p))
                    errs.append(abs(-(ep - em) / (2 * h) - f[i, ax]))
            err = max(errs)
            out.append(_check(f"force_fd_consistency[{name}]", err < TOL["fd_force_eV_per_A"], err,
                              TOL["fd_force_eV_per_A"], Unit.EV_PER_ANGSTROM,
                              f"{len(errs)} components, central FD h={h} Å in real space"))
        return out

    checks += _guard("force_fd_consistency", g_b)

    c_rs = by_name["rattled_sheared"]
    s_rs = c_rs.system
    e0 = float(e_new(s_rs.frac, s_rs.box, c_rs.species, c_rs.idx, p))
    f0 = -np.asarray(g_new(s_rs.frac, s_rs.box, c_rs.species, c_rs.idx, p))

    # (c) translation
    def g_c():
        shift = np.array([0.37, -1.21, 0.83])  # Å
        fr = (s_rs.frac + jnp.asarray(shift @ np.linalg.inv(np.asarray(s_rs.box)).T)) % 1.0
        de = abs(float(e_new(fr, s_rs.box, c_rs.species, c_rs.idx, p)) - e0)
        return [_check("translation_invariance", de < TOL["invariance_energy_eV"], de, TOL["invariance_energy_eV"],
                       Unit.EV, f"rigid shift {shift.tolist()} Å")]

    checks += _guard("translation_invariance", g_c)

    # (d) rotation: box -> R box at fixed frac rotates every position
    def g_d():
        ang = 0.7
        ax = np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5])
        kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        rot = np.eye(3) + np.sin(ang) * kx + (1 - np.cos(ang)) * kx @ kx
        box_r = jnp.asarray(rot @ np.asarray(s_rs.box))
        de = abs(float(e_new(s_rs.frac, box_r, c_rs.species, c_rs.idx, p)) - e0)
        fr_ = -np.asarray(g_new(s_rs.frac, box_r, c_rs.species, c_rs.idx, p))
        df = float(np.max(np.abs(fr_ - f0 @ rot.T)))
        return [
            _check("rotation_invariance_energy", de < TOL["invariance_energy_eV"], de, TOL["invariance_energy_eV"],
                   Unit.EV, "rigid rotation 0.7 rad about (1, 2, -0.5)"),
            _check("rotation_covariance_forces", df < TOL["invariance_force_eV_per_A"], df,
                   TOL["invariance_force_eV_per_A"], Unit.EV_PER_ANGSTROM, "F(R x) = R F(x)"),
        ]

    checks += _guard("rotation_invariance", g_d)

    # (e) permutation of atoms (neighbour list rebuilt for the permuted system)
    def g_e():
        from fracture_md.potentials import build_neighbors

        perm = np.random.default_rng(7).permutation(s_rs.n_atoms)
        sp_sys = s_rs.replace(frac=s_rs.frac[perm], species=s_rs.species[perm])
        nh = build_neighbors(sp_sys, base.cutoff)
        sp = base.species_for(sp_sys)
        de = abs(float(e_new(sp_sys.frac, sp_sys.box, sp, nh.idx, p)) - e0)
        fp = -np.asarray(g_new(sp_sys.frac, sp_sys.box, sp, nh.idx, p))
        df = float(np.max(np.abs(fp - f0[perm])))
        return [
            _check("permutation_invariance_energy", de < TOL["invariance_energy_eV"], de,
                   TOL["invariance_energy_eV"], Unit.EV, "random permutation of atom order"),
            _check("permutation_covariance_forces", df < TOL["invariance_force_eV_per_A"], df,
                   TOL["invariance_force_eV_per_A"], Unit.EV_PER_ANGSTROM, "F_perm = F[perm]"),
        ]

    checks += _guard("permutation_invariance", g_e)

    # (f) zero net force
    def g_f():
        out = []
        for c in configs:
            f = np.asarray(g_new(c.system.frac, c.system.box, c.species, c.idx, p))
            nf = float(np.linalg.norm(f.sum(0)))
            out.append(_check(f"zero_net_force[{c.name}]", nf < TOL["net_force_eV_per_A"], nf,
                              TOL["net_force_eV_per_A"], Unit.EV_PER_ANGSTROM, "|sum_i F_i|"))
        return out

    checks += _guard("zero_net_force", g_f)

    # (g) NVE
    def g_g():
        from fracture_md.potentials import build_neighbors
        from fracture_md.structures.system import perpendicular_widths

        c = by_name["eq_2x2x2"]
        s = c.system.replace(frac=(c.system.frac + jnp.asarray(
            np.random.default_rng(3).normal(0.0, 0.02, (c.system.n_atoms, 3)) @ np.linalg.inv(np.asarray(c.system.box)).T
        )) % 1.0)
        # as wide as the cell's minimum image allows, up to GATE_NVE_SKIN_A (1.0 Å for Tersoff/EA/Vashishta)
        room = 0.999 * (float(np.min(perpendicular_widths(np.asarray(s.box)))) / 2.0 - base.cutoff)
        skin = max(GATE_SKIN_A, min(GATE_NVE_SKIN_A, room))
        nh = build_neighbors(s, base.cutoff, skin=skin)
        r = nve_energy_conservation(efn, p, s, c.species, nh.idx, **NVE_SETTINGS)
        half = dict(NVE_SETTINGS, dt_fs=NVE_SETTINGS["dt_fs"] / 2, n_steps=2 * NVE_SETTINGS["n_steps"])
        r2 = nve_energy_conservation(efn, p, s, c.species, nh.idx, **half)
        ratio = r["fluctuation_eV_per_atom"] / max(r2["fluctuation_eV_per_atom"], 1e-300)
        disp = max(r["max_displacement_A"], r2["max_displacement_A"])
        ok_disp = disp < skin / 2
        desc = (f"velocity Verlet, {NVE_SETTINGS['temperature_K']:.0f} K, dt={NVE_SETTINGS['dt_fs']} fs, "
                f"{NVE_SETTINGS['n_steps']} steps, {s.n_atoms} atoms; max displacement {disp:.3f} Å "
                f"(fixed neighbour list valid below skin/2 = {skin / 2} Å)")
        return [
            _check("nve_neighbor_list_valid", ok_disp, disp, skin / 2, Unit.ANGSTROM, desc),
            _check("nve_energy_fluctuation", r["fluctuation_eV_per_atom"] < TOL["nve_fluctuation_eV_per_atom"],
                   r["fluctuation_eV_per_atom"], TOL["nve_fluctuation_eV_per_atom"], Unit.EV_PER_ATOM,
                   "max_t |E_tot(t) - E_tot(0)| / N; " + desc),
            _check("nve_second_order_convergence", ratio > TOL["nve_min_dt_halving_ratio"], ratio,
                   TOL["nve_min_dt_halving_ratio"], Unit.DIMENSIONLESS,
                   f"energy error at dt / at dt/2 = {r['fluctuation_eV_per_atom']:.3g} / "
                   f"{r2['fluctuation_eV_per_atom']:.3g} (expect ~4 for a smooth energy)"),
        ]

    checks += _guard("nve_energy_conservation", g_g)


    def g_g_dimer():
        out = []
        n_el = len(base.elements)
        for a in range(n_el):
            for b in range(a, n_el):
                lab = f"{base.elements[a]}-{base.elements[b]}"
                d = dimer_scan(efn, p, base, (a, b))
                out.append(_check(f"dimer_energy_continuity[{lab}]", d["max_jump_eV"] < TOL["dimer_jump_eV"],
                                  d["max_jump_eV"], TOL["dimer_jump_eV"], Unit.EV,
                                  f"max |ΔE - trapezoid(F)| over {d['n_points']} points r = "
                                  f"{d['r_range_A'][0]:.2f}-{d['r_range_A'][1]:.2f} Å (worst at r = "
                                  f"{d['at_r_A']:.3f} Å); a jump means an energy discontinuity"))
        return out

    checks += _guard("dimer_energy_continuity", g_g_dimer)

    # (h) regression vs trusted base implementation
    def g_h():
        out = []
        for c in configs:
            s = c.system
            args = (s.frac, s.box, c.species, c.idx, p)
            de = abs(float(e_new(*args)) - float(e_ref(*args))) / s.n_atoms
            df = float(np.max(np.abs(np.asarray(g_new(*args)) - np.asarray(g_ref(*args)))))
            ds = float(np.max(np.abs(stress(gb_new, c) - stress(gb_ref, c))))
            out += [
                _check(f"regression_energy[{c.name}]", de < TOL["regression_energy_eV_per_atom"], de,
                       TOL["regression_energy_eV_per_atom"], Unit.EV_PER_ATOM,
                       f"|E - E_base|/N vs {base.card.id}; {c.description}"),
                _check(f"regression_forces[{c.name}]", df < TOL["regression_force_eV_per_A"], df,
                       TOL["regression_force_eV_per_A"], Unit.EV_PER_ANGSTROM, "max |F - F_base|"),
                _check(f"regression_stress[{c.name}]", ds < TOL["regression_stress_GPa"], ds,
                       TOL["regression_stress_GPa"], Unit.GPA, "max |sigma - sigma_base| (cell derivative)"),
            ]
        return out

    checks += _guard("regression_vs_base", g_h)

    # (i) virial symmetry
    def g_i():
        out = []
        for name in ("rattled_sheared", "slab_111"):
            c = by_name[name]
            w = np.asarray(gb_new(c.system.frac, c.system.box, c.species, c.idx, p)) @ np.asarray(c.system.box).T
            asym = float(np.max(np.abs(w - w.T))) / c.system.volume * EV_PER_A3_TO_GPA
            out.append(_check(f"virial_symmetry[{name}]", asym < TOL["virial_asymmetry_GPa"], asym,
                              TOL["virial_asymmetry_GPa"], Unit.GPA, "max |W - W^T|/V, W = (dE/dh) h^T"))
        return out

    checks += _guard("virial_symmetry", g_i)
    return checks


# ==========================================================================
# Protocol-runner gate
# ==========================================================================
def _reference_dirs(reference_dir) -> list[Path]:
    if reference_dir is None:
        return [d for d in DEFAULT_REFERENCE_DIRS if d.is_dir()]
    dirs = [reference_dir] if isinstance(reference_dir, (str, Path)) else list(reference_dir)
    return [Path(d) for d in dirs if Path(d).is_dir()]


def _cases_from_file(path: Path) -> list[dict]:
    """Normalise the reference-file layouts we accept into a list of cases.

    Accepted layouts (JSON):
      * ``{"protocol": {...}, "candidate": {...}, "potential_id": str, "properties": [...], ...}``
      * ``{"cases": [<case as above>, ...], ...}``
      * an ExperimentResult-like dict with a ``"proposal"`` member carrying the
        protocol/candidate/potential_id.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = data["cases"] if isinstance(data, dict) and isinstance(data.get("cases"), list) else [data]
    out = []
    for i, c in enumerate(raw_cases):
        if not isinstance(c, dict):
            continue
        prop = c.get("proposal") if isinstance(c.get("proposal"), dict) else {}
        case = {
            "label": f"{path.stem}" + (f"#{i}" if len(raw_cases) > 1 else ""),
            "path": str(path),
            "protocol": c.get("protocol") or prop.get("protocol"),
            "candidate": c.get("candidate") or prop.get("candidate"),
            "potential_id": c.get("potential_id") or c.get("potential") or prop.get("potential_id"),
            "properties": c.get("properties", []),
            "traction_separation": c.get("traction_separation", []),
            "tolerances": {**((data.get("tolerances") or {}) if isinstance(data, dict) else {}),
                           **(c.get("tolerances") or {})},
            "protocol_kind": c.get("protocol_kind") or (c.get("protocol") or prop.get("protocol") or {}).get("kind"),
        }
        out.append(case)
    return out


def load_reference_cases(protocol_kind: str, reference_dir=None) -> list[dict]:
    cases = []
    for d in _reference_dirs(reference_dir):
        for path in sorted(d.glob("*.json")):
            try:
                cases += [c for c in _cases_from_file(path) if c["protocol_kind"] == protocol_kind]
            except (json.JSONDecodeError, OSError, AttributeError):
                continue
    return cases


def write_reference_result(path, *, protocol, candidate, potential_id: str, output, notes: str = "",
                           tolerances: Optional[dict] = None) -> Path:
    """Write a reference case in the layout the protocol-runner gate reads.

    `output` is a ProtocolOutput (or anything with .properties/.traction_separation).
    """
    from fracture_md.provenance import collect

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_version": "0.1.0",
        "protocol_kind": protocol.kind,
        "potential_id": potential_id,
        "protocol": protocol.model_dump(mode="json"),
        "candidate": candidate.model_dump(mode="json"),
        "properties": [r.model_dump(mode="json") for r in output.properties],
        "traction_separation": [t.model_dump(mode="json") for t in output.traction_separation],
        "checks": [c.model_dump(mode="json") for c in getattr(output, "checks", [])],
        "tolerances": tolerances or {},
        "notes": notes,
        "provenance": collect(potential_id=potential_id, protocol_id=protocol.protocol_id()).model_dump(mode="json"),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return path


def _value_of(rec) -> tuple[Any, Optional[str]]:
    """(value, unit string) from a PropertyRecord or its dict form."""
    if hasattr(rec, "value"):
        return rec.value.value, rec.value.unit.value
    v = rec.get("value")
    if isinstance(v, dict):
        return v.get("value"), v.get("unit")
    return v, rec.get("unit")


def _name_of(rec) -> str:
    n = rec.name if hasattr(rec, "name") else rec.get("name")
    return getattr(n, "value", n)


def compare_properties(new_records, ref_records, tolerances: Optional[dict] = None) -> list[CheckResult]:
    """Regression of produced records against reference records, matched by name and order."""
    tolerances = tolerances or {}
    out = []
    seen: dict[str, int] = {}
    new_by_name: dict[str, list] = {}
    for r in new_records:
        new_by_name.setdefault(_name_of(r), []).append(r)
    for ref in ref_records:
        name = _name_of(ref)
        k = seen.get(name, 0)
        seen[name] = k + 1
        label = f"regression[{name}]" if k == 0 else f"regression[{name}#{k}]"
        cands = new_by_name.get(name, [])
        if k >= len(cands):
            out.append(_check(label, False, details="property not produced by the runner"))
            continue
        vr, ur = _value_of(ref)
        vn, un = _value_of(cands[k])
        try:
            vn = convert(vn, un, ur) if un != ur else vn
            a_ref = np.asarray(vr, dtype=float)
            a_new = np.asarray(vn, dtype=float)
        except (UnitError, ValueError, TypeError) as exc:
            out.append(_check(label, False, details=f"cannot compare: {exc}"))
            continue
        if a_ref.shape != a_new.shape:
            out.append(_check(label, False, details=f"shape {a_new.shape} != reference {a_ref.shape}"))
            continue
        t = tolerances.get(name, {})
        rtol = float(t.get("rtol", tolerances.get("default_rtol", RUNNER_RTOL)))
        atol = float(t.get("atol", RUNNER_ATOL_BY_UNIT.get(ur, 1e-8)))
        err = np.abs(a_new - a_ref)
        allowed = atol + rtol * np.abs(a_ref)
        worst = float(np.max(err / allowed)) if err.size else 0.0
        out.append(_check(label, bool(np.all(err <= allowed)), float(np.max(err)) if err.size else 0.0, atol,
                          None, f"max |Δ| = {float(np.max(err)) if err.size else 0:.3g} {ur}; "
                          f"allowed atol={atol:g} + rtol={rtol:g}|ref| (worst ratio {worst:.3g})"))
    return out


def _as_tsl(t):
    from fracture_md.schema.properties import TractionSeparationLaw

    return t if isinstance(t, TractionSeparationLaw) else TractionSeparationLaw.model_validate(t)


def compare_traction_separation(new_tsls, ref_tsls, label: str = "") -> list[CheckResult]:
    """Regression of traction-separation curves, matched by order (and mode / protocol_mode).

    The new curve is linearly resampled onto the reference opening grid (which must be covered to within
    5 % of its range) and compared with |ΔT| <= CURVE_ATOL_GPA + CURVE_RTOL·max|T_ref| and
    |Δ(E/A)| <= CURVE_ATOL_J_PER_M2 + CURVE_RTOL·max|E/A_ref|.
    """
    out = []
    new_tsls, ref_tsls = list(new_tsls), list(ref_tsls)
    if len(new_tsls) != len(ref_tsls):
        out.append(_check(f"curve_count{label}", False, len(new_tsls), len(ref_tsls),
                          details=f"{len(new_tsls)} traction-separation laws produced, reference has {len(ref_tsls)}"))
    for i, (tn, tr) in enumerate(zip(new_tsls, ref_tsls)):
        lab = f"curve_regression{label}[{i}]"
        try:
            tn, tr = _as_tsl(tn), _as_tsl(tr)
            if (tn.mode, tn.protocol_mode) != (tr.mode, tr.protocol_mode):
                out.append(_check(lab, False, details=f"mode {(tn.mode, tn.protocol_mode)} != reference "
                                                      f"{(tr.mode, tr.protocol_mode)}"))
                continue
            xr = np.asarray(tr.curve.x.to(Unit.ANGSTROM).value, dtype=float)
            xn = np.asarray(tn.curve.x.to(Unit.ANGSTROM).value, dtype=float)
            order = np.argsort(xn)
            span = max(xr.max() - xr.min(), 1e-12)
            covered = xn.min() <= xr.min() + 0.05 * span and xn.max() >= xr.max() - 0.05 * span
            m = (xr >= xn.min()) & (xr <= xn.max())
            tr_y = np.asarray(tr.curve.y.to(Unit.GPA).value, dtype=float)
            tn_y = np.interp(xr[m], xn[order], np.asarray(tn.curve.y.to(Unit.GPA).value, dtype=float)[order])
            er = np.asarray(tr.energy_per_area.to(Unit.J_PER_M2).value, dtype=float)
            en = np.interp(xr[m], xn[order], np.asarray(tn.energy_per_area.to(Unit.J_PER_M2).value, dtype=float)[order])
            dt = float(np.max(np.abs(tn_y - tr_y[m]))) if m.any() else float("inf")
            de = float(np.max(np.abs(en - er[m]))) if m.any() else float("inf")
            tol_t = CURVE_ATOL_GPA + CURVE_RTOL * float(np.max(np.abs(tr_y)))
            tol_e = CURVE_ATOL_J_PER_M2 + CURVE_RTOL * float(np.max(np.abs(er)))
            out.append(_check(lab, bool(covered and dt <= tol_t and de <= tol_e), dt, tol_t, Unit.GPA,
                              f"{tr.mode}/{tr.protocol_mode}: max |ΔT| = {dt:.3g} GPa (tol {tol_t:.3g}), "
                              f"max |Δ(E/A)| = {de:.3g} J/m² (tol {tol_e:.3g}) on {int(m.sum())} reference openings; "
                              f"grid covered: {covered}"))
        except Exception as exc:  # noqa: BLE001
            out.append(_check(lab, False, details=f"cannot compare curves: {type(exc).__name__}: {exc}"))
    return out


_DEFAULT_TOLERANCE_KEYS = {"cleavage_integral_rtol": CLEAVAGE_INTEGRAL_RTOL, "cleavage_trapezoid_rtol":
                           CLEAVAGE_TRAPEZOID_RTOL, "default_rtol": RUNNER_RTOL}


def tolerance_widenings(tolerances: dict, ref_props=()) -> list[str]:
    """Entries of a reference file's `tolerances` that are looser than the gate defaults."""
    out = []
    units = {}
    for r in ref_props:
        try:
            units[_name_of(r)] = _value_of(r)[1]
        except Exception:  # noqa: BLE001
            pass
    for k, v in (tolerances or {}).items():
        if k in _DEFAULT_TOLERANCE_KEYS:
            try:
                if float(v) > _DEFAULT_TOLERANCE_KEYS[k]:
                    out.append(f"{k}={v} > default {_DEFAULT_TOLERANCE_KEYS[k]}")
            except (TypeError, ValueError):
                out.append(f"{k}={v!r} is not a number")
        elif isinstance(v, dict):
            if "rtol" in v and not float(v["rtol"]) <= RUNNER_RTOL:
                out.append(f"{k}.rtol={v['rtol']} > default {RUNNER_RTOL}")
            default_atol = RUNNER_ATOL_BY_UNIT.get(units.get(k), 1e-8)
            if "atol" in v and not float(v["atol"]) <= default_atol:
                out.append(f"{k}.atol={v['atol']} > default {default_atol}")
        else:
            out.append(f"unknown tolerance key {k!r}")
    return out


_PERTURBED_MODELS: dict = {}
_TRUSTED_OUTPUTS: dict = {}  # (trusted runner, perturbed protocol, candidate, perturbed card) -> ProtocolOutput


def _trusted_output(trusted_runner, protocol, candidate, model):
    """The trusted runner's output on a perturbed case (deterministic, so cached within the process)."""
    from fracture_md.engines.base import RunContext

    key = (id(trusted_runner), protocol.model_dump_json(), candidate.model_dump_json(), model.card.id)
    if key not in _TRUSTED_OUTPUTS:
        _TRUSTED_OUTPUTS[key] = trusted_runner(protocol, candidate, model, structure=None,
                                               context=RunContext(requested_by="gate"))
    return _TRUSTED_OUTPUTS[key]


def perturbed_case(protocol, model, rng: np.random.Generator):
    """(protocol', model', description) for the anti-lookup re-run of a reference case.

    model': E'(S, h) = s·E(S, λh) with s ∈ PERTURB_ENERGY_SCALE, λ ∈ PERTURB_LENGTH_SCALE and a new card id,
    i.e. a different but equally smooth "material" (a0' = a0/λ, energies ×s, stresses ×s/λ³...).
    protocol': grids jittered where the protocol has one (cleavage openings, ideal-strength strain step,
    elastic strain magnitudes, slip points), relaxation force tolerance halved otherwise.
    """
    s_e = float(rng.uniform(*PERTURB_ENERGY_SCALE))
    lam = float(rng.uniform(*PERTURB_LENGTH_SCALE))
    key = (model.card.id, s_e, lam)
    model_p = _PERTURBED_MODELS.get(key)
    if model_p is None:  # cached so that the jit-compiled kernels (keyed on energy_fn) are reused
        base_fn = model.energy_fn

        def energy_fn(frac, box, species, idx, params):
            return s_e * base_fn(frac, lam * box, species, idx, params)

        tag = hashlib.sha256(repr(key).encode()).hexdigest()[:10]
        card = model.card.model_copy(update=dict(id=f"{model.card.id}+gate-perturbed-{tag}",
                                                 name=f"{model.card.name} [gate-perturbed s={s_e:.4f}, λ={lam:.4f}]"))
        model_p = dataclasses.replace(model, name=f"{model.name}+gate-perturbed-{tag}", energy_fn=energy_fn,
                                      card=card)
        _PERTURBED_MODELS[key] = model_p
    card = model_p.card
    upd, what = {}, []
    kind = protocol.kind
    if kind == "cleavage" and getattr(protocol, "openings_A", None):
        o = np.asarray(protocol.openings_A, dtype=float)
        jit = float(rng.uniform(0.9, 0.97))
        upd["openings_A"] = sorted({0.0, *(float(round(x * jit, 6)) for x in o[o > 0])})
        what.append(f"openings ×{jit:.3f}")
    elif kind == "ideal_strength":
        f = float(rng.uniform(0.8, 0.95))
        upd["strain_step"] = float(protocol.strain_step * f)
        what.append(f"strain_step ×{f:.3f}")
    elif kind == "elastic_tensor":
        f = float(rng.uniform(0.8, 1.25))
        upd["strain_magnitudes"] = [float(x * f) for x in protocol.strain_magnitudes]
        what.append(f"strain magnitudes ×{f:.3f}")
    elif kind == "tangential_slip":
        n = int(protocol.n_points) + int(rng.integers(1, 6))
        upd["n_points"] = n
        what.append(f"n_points {protocol.n_points} -> {n}")
    if hasattr(protocol, "relax") and not what:
        upd["relax"] = protocol.relax.model_copy(update={"fmax_eV_per_A": protocol.relax.fmax_eV_per_A * 0.5})
        what.append("relax fmax ×0.5")
    protocol_p = protocol.model_copy(update=upd) if upd else protocol
    desc = f"model E' = {s_e:.5f}·E(S, {lam:.5f}·h), card {card.id}; protocol: {', '.join(what) or 'unchanged'}"
    return protocol_p, model_p, desc


def cleavage_integral_identity(tsl, rtol: float = CLEAVAGE_INTEGRAL_RTOL, label: str = "",
                               trapezoid_rtol: float = CLEAVAGE_TRAPEZOID_RTOL) -> list[CheckResult]:
    """∫T dδ vs [E(δ_max) - E(0)]/A for one traction-separation law (J/m²).

    Returns two checks: the runner's integrated_work vs work_of_separation
    (tight), and an independent trapezoid over the reported (x, T) samples vs
    the reported energy_per_area end points (loose: coarse-grid discretisation).
    """
    x = np.asarray(tsl.curve.x.to(Unit.ANGSTROM).value, dtype=float)
    t = np.asarray(tsl.curve.y.to(Unit.EV_PER_ANGSTROM3).value, dtype=float)
    ea = np.asarray(tsl.energy_per_area.to(Unit.J_PER_M2).value, dtype=float)
    trap = float(convert(np.trapezoid(t, x), Unit.EV_PER_ANGSTROM2, Unit.J_PER_M2))
    de = float(ea[-1] - ea[0])
    w_int = tsl.integrated_work.magnitude(Unit.J_PER_M2)
    w_sep = tsl.work_of_separation.magnitude(Unit.J_PER_M2)
    rel1 = abs(w_int - w_sep) / max(abs(w_sep), 1e-12)
    rel2 = abs(trap - de) / max(abs(de), 1e-12)
    return [
        _check(f"cleavage_integral_identity{label}", rel1 < rtol, rel1, rtol, Unit.DIMENSIONLESS,
               f"reported ∫T dδ = {w_int:.6f} J/m², W_sep = {w_sep:.6f} J/m²"),
        _check(f"cleavage_curve_trapezoid{label}", rel2 < trapezoid_rtol, rel2, trapezoid_rtol, Unit.DIMENSIONLESS,
               f"trapezoid over {x.size} reported samples = {trap:.6f} J/m², ΔE/A = {de:.6f} J/m²"),
    ]


def _kind_identities(kind: str, output, protocol, case_label: str, ref_props, tolerances) -> list[CheckResult]:
    from fracture_md.schema.properties import PropertyName

    out = []
    if kind in ("cleavage", "tangential_slip"):
        rtol = float(tolerances.get("cleavage_integral_rtol", CLEAVAGE_INTEGRAL_RTOL))
        trtol = float(tolerances.get("cleavage_trapezoid_rtol", CLEAVAGE_TRAPEZOID_RTOL))
        for i, tsl in enumerate(output.traction_separation):
            out += cleavage_integral_identity(tsl, rtol, f"[{case_label}:{i}]", trtol)
        if not output.traction_separation:
            out.append(_check(f"cleavage_curve_produced[{case_label}]", False, details="no traction-separation law"))
    if kind == "indentation":
        curves = list(getattr(output, "indentation_curves", None) or [])
        out.append(_check(f"indentation_curve_produced[{case_label}]", bool(curves), len(curves), 1, None,
                          "the runner returns the IndentationCurve of the run (load-depth record, checks)"))
    if kind == "elastic_tensor":
        tol = float(getattr(protocol, "symmetry_tolerance_GPa", 0.5))
        ref_pd = {}
        for r in ref_props:
            if _name_of(r) == PropertyName.ELASTIC_STIFFNESS_VOIGT.value:
                v, _u = _value_of(r)
                ref_pd[len(ref_pd)] = bool(np.all(np.linalg.eigvalsh(0.5 * (np.asarray(v) + np.asarray(v).T)) > 0))
        k = 0
        for r in output.properties:
            if r.name != PropertyName.ELASTIC_STIFFNESS_VOIGT:
                continue
            c = r.value.as_array(Unit.GPA)
            asym = float(np.max(np.abs(c - c.T)))
            out.append(_check(f"elastic_symmetry[{case_label}:{k}]", asym <= tol, asym, tol, Unit.GPA, "max |C - C^T|"))
            if ref_pd.get(k):
                lmin = float(np.min(np.linalg.eigvalsh(0.5 * (c + c.T))))
                out.append(_check(f"elastic_positive_definite[{case_label}:{k}]", lmin > 0, lmin, 0.0, Unit.GPA,
                                  "min eigenvalue of C (reference is positive definite)"))
            k += 1
    return out


def run_protocol_runner_gate(runner: Callable, protocol_kind: str, *, reference_dir=None,
                             seed: Optional[int] = None, trusted_runner: Optional[Callable] = None) -> list[CheckResult]:
    """Re-run stored reference cases with `runner` and compare (no manifest side effects).

    For every case: (1) the stored protocol vs the stored numbers (properties and curves); (2) a perturbed
    case drawn from `seed` (:func:`perturbed_case`) vs the trusted built-in runner for the kind
    (`trusted_runner`, default ``engines.dispatch.jax_md_runner(kind)``) on the same perturbed case.
    """
    from pydantic import TypeAdapter

    from fracture_md.engines.base import RunContext
    from fracture_md.potentials import get_potential
    from fracture_md.schema.material import Candidate
    from fracture_md.schema.protocol import Protocol

    seed = new_gate_seed() if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    if trusted_runner is None:
        from fracture_md.engines.dispatch import EngineUnavailable, jax_md_runner

        try:
            trusted_runner = jax_md_runner(protocol_kind)[0]
        except EngineUnavailable as exc:
            return [_check("trusted_runner_available", False,
                           details=f"no trusted built-in runner for {protocol_kind!r} to compare against: {exc}")]
    cases = load_reference_cases(protocol_kind, reference_dir)
    dirs = [str(d) for d in _reference_dirs(reference_dir)] or [str(d) for d in DEFAULT_REFERENCE_DIRS]
    if not cases:
        return [_check("reference_cases_available", False,
                       details=f"no reference results for protocol kind {protocol_kind!r} in {dirs}; "
                       "an untested runner cannot be trusted (see gates.write_reference_result)")]
    checks = [_check("reference_cases_available", True, len(cases),
                     details=f"{len(cases)} case(s): {[c['label'] for c in cases]}")]
    for case in cases:
        lab = case["label"]
        if not (case["protocol"] and case["candidate"] and case["potential_id"]):
            checks.append(_check(f"reference_case_complete[{lab}]", False,
                                 details="reference file lacks protocol, candidate or potential_id; cannot re-run"))
            continue

        widened = tolerance_widenings(case["tolerances"], case["properties"])
        checks.append(_check(f"reference_tolerances_not_widened[{lab}]", not widened, len(widened), 0,
                             details="reference file tolerances are at most the gate defaults" if not widened else
                             f"reference file widens tolerances: {widened}; looser tolerances need a gate "
                             "version change, not a data file"))
        pseed = int(rng.integers(2**63))

        def _run_checks(output, lab_):
            from fracture_md.engines.base import ProtocolOutput

            if not isinstance(output, ProtocolOutput):
                raise TypeError(f"runner returned {type(output).__name__}, expected ProtocolOutput")
            failed = [c.name for c in output.checks if not c.passed]
            return [_check(f"runner_checks_pass[{lab_}]", bool(output.checks) and not failed, len(failed), 0,
                           details=f"{len(output.checks)} runner checks (at least one required); failed: {failed}")]

        def _one(case=case, lab=lab):
            protocol = TypeAdapter(Protocol).validate_python(case["protocol"])
            candidate = Candidate.model_validate(case["candidate"])
            model = get_potential(case["potential_id"])
            t0 = time.perf_counter()
            output = runner(protocol, candidate, model, structure=None, context=RunContext(requested_by="gate"))
            dt = time.perf_counter() - t0
            res = [_check(f"runner_executes[{lab}]", True, dt, None, Unit.SECOND, f"ran in {dt:.1f} s")]
            res += _run_checks(output, lab)
            res += [c.model_copy(update={"name": f"{c.name}[{lab}]"})
                    for c in compare_properties(output.properties, case["properties"], case["tolerances"])]
            if case.get("traction_separation"):
                res += compare_traction_separation(output.traction_separation, case["traction_separation"],
                                                   f"[{lab}]")
            res += _kind_identities(protocol.kind, output, protocol, lab, case["properties"], case["tolerances"])
            return res

        def _perturbed(case=case, lab=lab, pseed=pseed):
            protocol = TypeAdapter(Protocol).validate_python(case["protocol"])
            candidate = Candidate.model_validate(case["candidate"])
            model = get_potential(case["potential_id"])
            prot_p, model_p, desc = perturbed_case(protocol, model, np.random.default_rng(pseed))
            plab = f"{lab}:perturbed"
            t0 = time.perf_counter()
            out_new = runner(prot_p, candidate, model_p, structure=None, context=RunContext(requested_by="gate"))
            t1 = time.perf_counter()
            out_ref = _trusted_output(trusted_runner, prot_p, candidate, model_p)
            t2 = time.perf_counter()
            res = [_check(f"runner_executes[{plab}]", True, t1 - t0, None, Unit.SECOND,
                          f"{desc}; candidate {t1 - t0:.1f} s, trusted built-in {t2 - t1:.1f} s [seed {pseed}]")]
            res += _run_checks(out_new, plab)
            ref_props = [r.model_dump(mode="json") for r in out_ref.properties]
            if not ref_props and not out_ref.traction_separation:
                res.append(_check(f"trusted_runner_produced_output[{plab}]", False,
                                  details="the trusted runner produced nothing to compare against"))
            res += [c.model_copy(update={"name": f"{c.name.replace('regression', 'trusted_runner_agreement', 1)}"
                                                 f"[{plab}]"})
                    for c in compare_properties(out_new.properties, ref_props)]
            if out_ref.traction_separation or out_new.traction_separation:
                res += compare_traction_separation(out_new.traction_separation, out_ref.traction_separation,
                                                   f"[{plab}]")
            return res

        checks += _guard(f"runner_executes[{lab}]", _one)
        checks += _guard(f"runner_executes[{lab}:perturbed]", _perturbed)
    return checks


# ==========================================================================
# Entry point
# ==========================================================================
def gate(registry, impl_id: str, *, reference_dir=None, record: bool = True, seed: Optional[int] = None) -> GateReport:
    """Run the regression gate for a registered implementation and record the outcome.

    This is the *only* code path that changes an implementation's status: to
    ``trusted`` if every check passes, to ``rejected`` otherwise.  All checks
    (passed and failed, with metrics and tolerances) are written to the
    manifest together with the gate version and environment.
    """
    from fracture_md.provenance import collect

    t0 = time.perf_counter()
    seed = new_gate_seed() if seed is None else int(seed)
    man = registry.get_manifest(impl_id)
    kind = man["kind"]
    checks: list[CheckResult] = []
    try:
        module = registry.load(impl_id)  # verifies the hash first
        checks.append(_check("source_hash_verified", True, details=f"sha256 {man['sha256']}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("source_hash_verified", False, details=f"{type(exc).__name__}: {exc}"))
        module = None
    if module is not None:
        if kind == "potential_energy_fn":
            decl = getattr(module, "BASE_POTENTIAL", None)
            if man.get("base_potential") and decl != man["base_potential"]:
                checks.append(_check("base_potential_matches_manifest", False,
                                     details=f"module {decl!r} != manifest {man['base_potential']!r}"))
            else:
                checks += _guard("potential_gate", lambda: run_potential_gate(module, seed=seed))
        elif kind == "protocol_runner":
            runner = getattr(module, "run", None) or getattr(module, "RUNNER", None)
            pk = man.get("protocol_kind")
            decl = getattr(module, "PROTOCOL_KIND", pk)
            if not callable(runner):
                checks.append(_check("defines_interface", False, details="module must define run(protocol, candidate, model, structure=None, context=None)"))
            elif decl != pk:
                checks.append(_check("protocol_kind_matches_manifest", False, details=f"module {decl!r} != manifest {pk!r}"))
            else:
                checks.append(_check("defines_interface", True, details=f"run() for protocol kind {pk!r}"))
                checks += _guard("protocol_runner_gate",
                                 lambda: run_protocol_runner_gate(runner, pk, reference_dir=reference_dir,
                                                                  seed=seed))
        else:  # pragma: no cover
            checks.append(_check("known_kind", False, details=f"unknown kind {kind!r}"))
    passed = bool(checks) and all(c.passed for c in checks)
    report = GateReport(impl_id=impl_id, kind=kind, passed=passed, checks=checks,
                        duration_s=time.perf_counter() - t0, seed=seed)
    if record:
        prov = collect()
        registry._record_gate(
            impl_id,
            passed=passed,
            checks=[c.model_dump(mode="json") for c in checks],
            gate_info={
                "gate_version": GATE_VERSION,
                "random_seed": seed,
                "run_utc": datetime.now(timezone.utc).isoformat(),
                "duration_s": round(report.duration_s, 2),
                "tolerances": TOL if kind == "potential_energy_fn" else
                {"rtol": RUNNER_RTOL, "atol_by_unit": RUNNER_ATOL_BY_UNIT, "cleavage_integral_rtol": CLEAVAGE_INTEGRAL_RTOL,
                 "cleavage_trapezoid_rtol": CLEAVAGE_TRAPEZOID_RTOL, "curve_rtol": CURVE_RTOL,
                 "curve_atol_GPa": CURVE_ATOL_GPA, "curve_atol_J_per_m2": CURVE_ATOL_J_PER_M2,
                 "perturb_energy_scale": PERTURB_ENERGY_SCALE, "perturb_length_scale": PERTURB_LENGTH_SCALE},
                "environment": {
                    "code_version": prov.code_version,
                    "platform": prov.platform,
                    "device": prov.device,
                    "float_precision": prov.float_precision,
                    "packages": {k: v for k, v in prov.package_versions.items() if k in ("jax", "jaxlib", "jax-md", "numpy")},
                },
            },
        )
    return report


def gate_source(source: Union[str, Path], *, kind: str = "potential_energy_fn", **kw) -> GateReport:
    """Gate a source file without registering it (dry run into a temporary store)."""
    import tempfile

    from fracture_md.agents.registry import ImplementationRegistry

    with tempfile.TemporaryDirectory() as td:
        reg = ImplementationRegistry(td)
        impl_id = reg.register(kind, "dry_run", source, author="dry-run", protocol_kind=kw.pop("protocol_kind", None))
        return gate(reg, impl_id, record=False, **kw)
