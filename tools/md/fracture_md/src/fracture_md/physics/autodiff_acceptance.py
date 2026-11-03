"""Autodiff acceptance case: differentiate a short MD trajectory and compare to finite differences.

Question answered: is a final observable O of a JAX-MD trajectory a
differentiable function of (a) a loading parameter and (b) a potential
parameter, and do the automatic derivatives equal central finite differences?

Parameters (names as in ``AutodiffAcceptanceProtocol.parameters``):

* ``applied_strain_<c>``, c ∈ {xx, yy, zz, yz, xz, xy}: homogeneous strain
  applied to the box at t = 0, ``box -> (I + ε) box``, atoms mapped affinely
  through FIXED fractional coordinates, initial momenta unchanged.  Base value
  0.  Shear components are engineering shear strains γ (ε_ij = γ/2),
  consistent with the Voigt convention used elsewhere.
* ``potential:<name>_scale``: multiplicative scale on the entire parameter
  table ``model.params[<name>]`` (e.g. Tersoff ``A``, ``B``, ``chi``, ``lam``).
  Base value 1.

Observables (all computed on every evaluation):

* ``final_potential_energy``  E_pot(t_N)                 [eV]
* ``final_virial_stress_zz``  σ_zz(t_N), configurational + kinetic, tensile positive  [GPa]
* ``mean_potential_energy``   (1/N) Σ_{n=1..N} E_pot(t_n) [eV]

Derivatives: reverse mode (``jax.jacrev`` through ``lax.scan``) and forward
mode (``jax.jvp``), both compared against central finite differences over a
sweep of step sizes h_k = fd_step · 10^{1-k}.  The FD reference is chosen
WITHOUT looking at the AD value: the consecutive pair (h_k, h_{k+1}) whose FD
values agree best is the "plateau", and the smaller-h member is the reference;
the pair's disagreement is reported as the FD self-consistency.

Why a SHORT trajectory: MD is chaotic.  Tangent vectors grow like e^{λt}
(finite-time Lyapunov estimate λ ≈ 0.006 fs⁻¹ over 1 ps for 300 K Tersoff
SiC, i.e. a 340x growth of a random position perturbation — see
:func:`finite_time_lyapunov` and examples/03), so the observable's
derivative is exact but its range of linearity shrinks like e^{-λt}.  FD
needs h ≪ e^{-λt}, while roundoff in f(θ ± h) is itself amplified by e^{λt};
beyond a few hundred fs no h satisfies both and AD–FD agreement is lost even
though AD is the exact derivative of the discrete (floating-point) map.  In
addition, the Tersoff cosine cutoff is only C¹, so every time a pair crosses
R1 or R2 the force Jacobian jumps; FD steps that straddle such events see a
kink.  See docs/AUTODIFF.md.

Implementation notes: trajectories are integrated by
:func:`fracture_md.physics.md.run_md`, which uses exact fractional-coordinate
JVPs (JAX-MD's own fractional custom JVP gives wrong strain derivatives
through a trajectory; measured).  The neighbour list is fixed; its validity is
checked for every evaluated trajectory (nominal and all FD points).
"""

from __future__ import annotations

import dataclasses
import re
import time
from typing import Any, Iterable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fracture_md.engines.base import ProtocolOutput, RunContext
from fracture_md.physics import md as md_mod
from fracture_md.physics.md import (
    KB_EV_PER_K,
    TIME_UNIT_FS,
    effective_timestep_fs,
    initial_momenta,
    neighbor_list_margin,
    run_md,
    virial_stress,
)
from fracture_md.potentials.base import EnergyModel
from fracture_md.potentials.neighbors import build_neighbors
from fracture_md.schema.common import CheckResult
from fracture_md.schema.material import Candidate
from fracture_md.schema.protocol import AutodiffAcceptanceProtocol
from fracture_md.structures.system import AtomicSystem, auto_supercell
from fracture_md.units import EV_PER_A3_TO_GPA, Unit

OBSERVABLES = ("final_potential_energy", "final_virial_stress_zz", "mean_potential_energy")
OBSERVABLE_UNITS = {
    "final_potential_energy": "eV",
    "final_virial_stress_zz": "GPa",
    "mean_potential_energy": "eV",
}
_VOIGT = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "yz": (1, 2), "xz": (0, 2), "xy": (0, 1)}
_POT_RE = re.compile(r"^potential:(?P<name>.+)_scale$")

# Absolute floor of the AD-vs-FD acceptance test |AD - FD| <= rtol·|FD| + atol, in observable units per unit
# parameter.  The FD reference of a derivative that is exactly zero is round-off ~ eps·|E|/h (E ~ 1e3 eV,
# h ~ 1e-6 -> ~1e-7 eV), so a pure relative test fails spuriously; real derivatives are 1-1e3 here.
AD_FD_ATOL: dict[str, float] = {
    "final_potential_energy": 1.0e-6,  # eV per unit parameter
    "mean_potential_energy": 1.0e-6,  # eV per unit parameter
    "final_virial_stress_zz": 1.0e-5,  # GPa per unit parameter
}
# Upper bound of the NVE conservation tolerance whatever the time step (eV/atom); 1e-3·(dt/fs)² reaches it at 2 fs.
NVE_TOL_CAP_EV_PER_ATOM = 4.0e-3


def ad_fd_agree(ad: float, fd: float, rtol: float, observable: str) -> bool:
    """Acceptance test |AD - FD| <= rtol·|FD| + atol(observable) (False for non-finite values)."""
    err = abs(float(ad) - float(fd))
    return bool(np.isfinite(err) and err <= rtol * abs(float(fd)) + AD_FD_ATOL[observable])


def nve_tolerance_eV_per_atom(timestep_fs: float) -> float:
    """NVE conservation tolerance min(1e-3·max(1, (dt/fs)²), NVE_TOL_CAP_EV_PER_ATOM) in eV/atom."""
    return min(1e-3 * max(1.0, (float(timestep_fs) / 1.0) ** 2), NVE_TOL_CAP_EV_PER_ATOM)


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # "strain" | "potential_scale"
    key: Any  # (i, j) for strain, table name for potential
    base: float

    @property
    def unit_label(self) -> str:
        return "strain (1)" if self.kind == "strain" else "scale factor (1)"


def parse_parameter(name: str, model: Optional[EnergyModel] = None) -> ParamSpec:
    if name.startswith("applied_strain_"):
        comp = name[len("applied_strain_"):]
        if comp not in _VOIGT:
            raise ValueError(f"unknown strain component in {name!r}; use one of {list(_VOIGT)}")
        return ParamSpec(name, "strain", _VOIGT[comp], 0.0)
    mt = _POT_RE.match(name)
    if mt:
        key = mt.group("name")
        if model is not None and key not in model.params:
            raise ValueError(f"potential {model.name} has no parameter table {key!r}; available {sorted(model.params)}")
        return ParamSpec(name, "potential_scale", key, 1.0)
    raise ValueError(f"unknown acceptance parameter {name!r} (expected applied_strain_<c> or potential:<name>_scale)")


def strain_tensor(specs: Sequence[ParamSpec], theta) -> jnp.ndarray:
    eps = jnp.zeros((3, 3), dtype=jnp.float64)
    for k, sp in enumerate(specs):
        if sp.kind != "strain":
            continue
        i, j = sp.key
        if i == j:
            eps = eps.at[i, i].add(theta[k])
        else:  # engineering shear strain γ -> ε_ij = ε_ji = γ/2
            eps = eps.at[i, j].add(0.5 * theta[k]).at[j, i].add(0.5 * theta[k])
    return eps


# ----------------------------------------------------------------------------
# Trajectory observable
# ----------------------------------------------------------------------------
@dataclasses.dataclass
class TrajectoryObservable:
    """``obs(params_dict) -> {observable: value}``; also ``.vector(theta)`` for jacobians.

    Holds everything that is NOT differentiated (structure, neighbour list,
    initial momenta, PRNG key, MD settings).  The initial momenta and the
    thermostat key are fixed at construction, so every evaluation — nominal,
    AD and each FD point — integrates the same stochastic realisation.
    """

    model: EnergyModel
    system: AtomicSystem
    idx: jnp.ndarray
    specs: tuple[ParamSpec, ...]
    dt_fs: float
    n_steps: int
    temperature_K: float
    seed: int = 0
    ensemble: str = "nve"
    remat: bool = False
    skin: float = 0.5
    md_kwargs: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        k_mom, _ = jax.random.split(jax.random.PRNGKey(self.seed))
        self.momenta0 = initial_momenta(self.system.masses_amu, self.temperature_K, key=k_mom)
        self._vec = jax.jit(self._vector_with_aux)

    @property
    def parameters(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def base_theta(self) -> jnp.ndarray:
        return jnp.asarray([s.base for s in self.specs], dtype=jnp.float64)

    def strain(self, theta) -> jnp.ndarray:
        return strain_tensor(self.specs, theta)

    def _params(self, theta):
        p = dict(self.model.params)
        for k, sp in enumerate(self.specs):
            if sp.kind == "potential_scale":
                p[sp.key] = p[sp.key] * theta[k]
        return p

    def trajectory(self, theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)
        box = (jnp.eye(3) + self.strain(theta)) @ self.system.box
        p = self._params(theta)
        tr = run_md(
            self.model, self.system, self.idx,
            dt_fs=self.dt_fs, n_steps=self.n_steps, temperature_K=self.temperature_K, seed=self.seed,
            ensemble=self.ensemble, box=box, params=p, momenta=self.momenta0, remat=self.remat,
            **self.md_kwargs,
        )
        return tr, box, p

    def _vector_with_aux(self, theta):
        tr, box, p = self.trajectory(theta)
        sig = virial_stress(self.model, self.system, self.idx, tr.frac, tr.momentum, box, params=p)
        vals = jnp.stack([
            tr.potential_energy[-1],
            sig[2, 2] * EV_PER_A3_TO_GPA,
            jnp.mean(tr.potential_energy[1:]),
        ])
        return vals, (tr.max_displacement, tr.potential_energy + tr.kinetic_energy)

    def vector(self, theta) -> jnp.ndarray:
        """Observables in the order of :data:`OBSERVABLES` (differentiable)."""
        return self._vector_with_aux(jnp.asarray(theta, dtype=jnp.float64))[0]

    def evaluate(self, theta):
        """Jitted (values, max_displacement, total-energy series) at theta."""
        v, (d, et) = self._vec(jnp.asarray(theta, dtype=jnp.float64))
        return np.asarray(v), float(d), np.asarray(et)

    def theta_from_dict(self, params_dict: dict[str, float]) -> jnp.ndarray:
        unknown = set(params_dict) - set(self.parameters)
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)}; this observable was built for {self.parameters}")
        return jnp.asarray([params_dict.get(s.name, s.base) for s in self.specs], dtype=jnp.float64)

    def __call__(self, params_dict: Optional[dict[str, float]] = None) -> dict[str, jnp.ndarray]:
        theta = self.theta_from_dict(params_dict or {})
        v = self.vector(theta)
        return {o: v[i] for i, o in enumerate(OBSERVABLES)}

    def strain_margin(self, theta) -> float:
        return neighbor_list_margin(0.0, self.skin, self.model.cutoff, np.asarray(self.strain(jnp.asarray(theta))))


def make_trajectory_observable(
    model: EnergyModel,
    system: AtomicSystem,
    idx=None,
    *,
    parameters: Sequence[str] = ("applied_strain_zz", "potential:A_scale"),
    dt_fs: float = 1.0,
    n_steps: int = 100,
    temperature_K: float = 300.0,
    seed: int = 0,
    ensemble: str = "nve",
    remat: bool = False,
    skin: float = 0.5,
    **md_kwargs,
) -> TrajectoryObservable:
    if idx is None:
        idx = build_neighbors(system, model.cutoff, skin=skin).idx
    specs = tuple(parse_parameter(p, model) for p in parameters)
    return TrajectoryObservable(model, system, idx, specs, float(dt_fs), int(n_steps), float(temperature_K),
                                int(seed), ensemble, bool(remat), float(skin), dict(md_kwargs))


def trajectory_observable(params_dict: dict[str, float], *, model: EnergyModel, system: AtomicSystem, idx=None,
                          **settings) -> dict[str, jnp.ndarray]:
    """Functional form: build the observable for the keys of ``params_dict`` and evaluate it.

    Returns final potential energy (eV), final virial σ_zz (GPa) and the
    time-averaged potential energy (eV).  Differentiable w.r.t. the values of
    ``params_dict`` (e.g. ``jax.grad(lambda d: trajectory_observable(d, ...)["final_potential_energy"])``).
    """
    obs = make_trajectory_observable(model, system, idx, parameters=list(params_dict), **settings)
    theta = jnp.stack([jnp.asarray(params_dict[k], dtype=jnp.float64) for k in obs.parameters])
    v = obs.vector(theta)
    return {o: v[i] for i, o in enumerate(OBSERVABLES)}


# ----------------------------------------------------------------------------
# Gradient check
# ----------------------------------------------------------------------------
def rel_err(a: float, b: float, floor: float = 1e-300) -> float:
    return float(abs(a - b) / max(abs(a), abs(b), floor))


def fd_steps_around(fd_step: float, n: int = 5) -> list[float]:
    """h_k = fd_step · 10^{1-k}, k = 0..n-1 (e.g. 1e-4 ... 1e-8 for fd_step = 1e-5)."""
    return [float(fd_step * 10.0 ** (1 - k)) for k in range(n)]


def select_plateau(values_by_step: dict[float, float]) -> tuple[float, float, float]:
    """(h_ref, fd_ref, spread): consecutive pair with the smallest mutual relative difference.

    Independent of the AD value (no circular selection).  Returns the smaller
    step of the pair and the pair's relative disagreement.
    """
    hs = sorted(values_by_step, reverse=True)
    if len(hs) == 1:
        return hs[0], values_by_step[hs[0]], float("nan")
    best = None
    for h1, h2 in zip(hs[:-1], hs[1:]):
        d = rel_err(values_by_step[h1], values_by_step[h2])
        if best is None or d < best[2]:
            best = (h2, values_by_step[h2], d)
    return best  # type: ignore[return-value]


@dataclasses.dataclass
class GradientEntry:
    observable: str
    parameter: str
    ad_reverse: float
    ad_forward: float
    fd: dict[float, float]
    fd_ref_step: float
    fd_ref: float
    fd_spread: float
    rel_err_reverse: float
    rel_err_forward: float
    rel_err_forward_vs_reverse: float

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["fd"] = {f"{h:.0e}": v for h, v in self.fd.items()}
        d["unit"] = f"{OBSERVABLE_UNITS[self.observable]} per unit parameter"
        return d


@dataclasses.dataclass
class GradientCheckReport:
    parameters: list[str]
    base_theta: list[float]
    values: dict[str, float]
    entries: list[GradientEntry]
    steps: list[float]
    min_neighbor_margin_A: float
    total_energy_series: np.ndarray
    timings_s: dict[str, float]

    def entry(self, observable: str, parameter: str) -> GradientEntry:
        for e in self.entries:
            if e.observable == observable and e.parameter == parameter:
                return e
        raise KeyError((observable, parameter))

    def as_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters,
            "base_theta": self.base_theta,
            "values": self.values,
            "value_units": OBSERVABLE_UNITS,
            "fd_steps": self.steps,
            "entries": [e.as_dict() for e in self.entries],
            "min_neighbor_margin_A": self.min_neighbor_margin_A,
            "timings_s": self.timings_s,
        }


def gradient_check(obs: TrajectoryObservable, steps: Sequence[float], theta0=None) -> GradientCheckReport:
    """Reverse-mode, forward-mode and central-FD derivatives of all observables w.r.t. all parameters."""
    theta0 = obs.base_theta if theta0 is None else jnp.asarray(theta0, dtype=jnp.float64)
    npar = theta0.shape[0]
    t0 = time.perf_counter()
    vals, disp0, et = obs.evaluate(theta0)
    t_primal = time.perf_counter() - t0

    t0 = time.perf_counter()
    J_rev = np.asarray(jax.jit(jax.jacrev(obs.vector))(theta0))  # (n_obs, n_par)
    t_rev = time.perf_counter() - t0

    t0 = time.perf_counter()
    jvp = jax.jit(lambda th, v: jax.jvp(obs.vector, (th,), (v,))[1])
    J_fwd = np.stack([np.asarray(jvp(theta0, jnp.eye(npar)[k])) for k in range(npar)], axis=1)
    t_fwd = time.perf_counter() - t0

    t0 = time.perf_counter()
    margins = [neighbor_list_margin(disp0, obs.skin, obs.model.cutoff, np.asarray(obs.strain(theta0)))]
    fd = np.zeros((len(steps), len(OBSERVABLES), npar))
    for k in range(npar):
        e = np.zeros(npar)
        e[k] = 1.0
        for ih, h in enumerate(steps):
            tp, tm = theta0 + h * e, theta0 - h * e
            vp, dp, _ = obs.evaluate(tp)
            vm, dm, _ = obs.evaluate(tm)
            fd[ih, :, k] = (vp - vm) / (2.0 * h)
            margins += [neighbor_list_margin(dp, obs.skin, obs.model.cutoff, np.asarray(obs.strain(tp))),
                        neighbor_list_margin(dm, obs.skin, obs.model.cutoff, np.asarray(obs.strain(tm)))]
    t_fd = time.perf_counter() - t0

    entries = []
    for io, o in enumerate(OBSERVABLES):
        for k, pname in enumerate(obs.parameters):
            by_h = {float(h): float(fd[ih, io, k]) for ih, h in enumerate(steps)}
            h_ref, v_ref, spread = select_plateau(by_h)
            r, f = float(J_rev[io, k]), float(J_fwd[io, k])
            entries.append(GradientEntry(o, pname, r, f, by_h, h_ref, v_ref, spread,
                                         rel_err(r, v_ref), rel_err(f, v_ref), rel_err(f, r)))
    return GradientCheckReport(
        parameters=obs.parameters,
        base_theta=[float(x) for x in np.asarray(theta0)],
        values={o: float(vals[i]) for i, o in enumerate(OBSERVABLES)},
        entries=entries,
        steps=[float(h) for h in steps],
        min_neighbor_margin_A=float(min(margins)),
        total_energy_series=et,
        timings_s={"primal": t_primal, "jacrev": t_rev, "jvp_all": t_fwd, "fd_all": t_fd},
    )


# ----------------------------------------------------------------------------
# Chaos: trajectory-length sweep and finite-time Lyapunov exponent
# ----------------------------------------------------------------------------
def finite_time_lyapunov(model: EnergyModel, system: AtomicSystem, idx, *, dt_fs: float, n_steps: int,
                         temperature_K: float = 300.0, seed: int = 0, tangent_seed: int = 123) -> dict[str, float]:
    """Growth of a random initial position perturbation along an NVE trajectory (forward-mode AD).

    δX(0): random real-space displacement with zero centre of mass, unit norm.
    Returns ‖δX(T)‖ / ‖δX(0)‖ and λ_T = ln(ratio) / T  [1/fs].  Exact tangent
    dynamics of the discrete map (no FD), so this is the sensitivity that FD of
    a trajectory observable has to resolve.
    """
    rng = np.random.default_rng(tangent_seed)
    dX = rng.standard_normal((system.n_atoms, 3))
    dX -= dX.mean(axis=0)
    dX /= np.linalg.norm(dX)
    box = system.box
    dS0 = jnp.asarray(dX) @ jnp.linalg.inv(box).T

    def final_frac(S0):
        return run_md(model, system, idx, dt_fs=dt_fs, n_steps=n_steps, temperature_K=temperature_K,
                      seed=seed, frac=S0).frac

    _, dS = jax.jit(lambda s, t: jax.jvp(final_frac, (s,), (t,)))(system.frac, dS0)
    ratio = float(jnp.linalg.norm(dS @ box.T))
    T = n_steps * dt_fs
    return {"n_steps": int(n_steps), "time_fs": float(T), "growth_ratio": ratio,
            "lyapunov_estimate_per_fs": float(np.log(ratio) / T) if T > 0 else 0.0}


def trajectory_length_sweep(
    model: EnergyModel,
    system: AtomicSystem,
    idx,
    *,
    lengths: Iterable[int] = (10, 50, 200, 1000),
    parameter: str = "applied_strain_zz",
    observable: str = "final_potential_energy",
    dt_fs: float = 1.0,
    temperature_K: float = 300.0,
    seed: int = 0,
    steps: Sequence[float] = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9),
    skin: float = 0.5,
    with_lyapunov: bool = True,
) -> list[dict[str, Any]]:
    """AD (forward mode, O(1) memory in trajectory length) vs FD as the trajectory lengthens.

    For each length reports: the AD derivative, FD at every step size, the
    plateau-selected FD reference and its relative error, the optimistic
    ``min_h |AD - FD(h)|`` (selected using AD, so only a lower bound), the FD
    self-consistency, and the tangent growth ratio.
    """
    io = OBSERVABLES.index(observable)
    rows = []
    for n in lengths:
        obs = make_trajectory_observable(model, system, idx, parameters=[parameter], dt_fs=dt_fs, n_steps=int(n),
                                         temperature_K=temperature_K, seed=seed, skin=skin)
        th0 = obs.base_theta
        _, disp, _ = obs.evaluate(th0)
        ad = float(jax.jit(lambda th: jax.jvp(obs.vector, (th,), (jnp.ones(1),))[1])(th0)[io])
        by_h = {}
        for h in steps:
            vp, _, _ = obs.evaluate(th0 + h)
            vm, _, _ = obs.evaluate(th0 - h)
            by_h[float(h)] = float((vp[io] - vm[io]) / (2 * h))
        h_ref, v_ref, spread = select_plateau(by_h)
        best_h = min(by_h, key=lambda h: abs(by_h[h] - ad))
        row = {
            "n_steps": int(n), "time_fs": float(n * dt_fs), "ad_forward": ad,
            "fd": {f"{h:.0e}": v for h, v in by_h.items()},
            "fd_ref_step": h_ref, "fd_ref": v_ref, "fd_spread": spread, "rel_err_vs_plateau": rel_err(ad, v_ref),
            "optimistic_best_step": best_h, "optimistic_min_rel_err": rel_err(ad, by_h[best_h]),
            "max_displacement_A": disp, "neighbor_margin_A": neighbor_list_margin(disp, skin, model.cutoff),
        }
        if with_lyapunov:
            row.update({k: v for k, v in finite_time_lyapunov(model, system, idx, dt_fs=dt_fs, n_steps=int(n),
                                                              temperature_K=temperature_K, seed=seed).items()
                        if k in ("growth_ratio", "lyapunov_estimate_per_fs")})
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# Physics sanity checks
# ----------------------------------------------------------------------------
def nve_energy_conservation(model: EnergyModel, system: AtomicSystem, idx, *, dts_fs=(0.25, 0.5, 1.0, 2.0),
                            total_time_fs: float = 250.0, temperature_K: float = 300.0,
                            seed: int = 0) -> list[dict[str, float]]:
    """Total-energy error of velocity Verlet vs time step (same initial state, same total time).

    Reports per-atom max |E(t) - E(0)|, standard deviation, and drift (slope of
    a linear fit, eV/atom/ps).  For a symplectic second-order integrator the
    fluctuation scales as dt² and there is no secular drift at short times;
    ``order_estimate`` = log2(std(2dt)/std(dt)) should be ≈ 2.
    """
    n = system.n_atoms
    rows = []
    for dt in dts_fs:
        ns = int(round(total_time_fs / dt))
        tr = jax.jit(lambda: run_md(model, system, idx, dt_fs=dt, n_steps=ns, temperature_K=temperature_K,
                                    seed=seed))()
        et = np.asarray(tr.potential_energy + tr.kinetic_energy)
        t = np.arange(ns + 1) * dt * 1e-3  # ps
        slope = float(np.polyfit(t, et / n, 1)[0])
        rows.append({"dt_fs": float(dt), "n_steps": ns, "max_abs_dev_eV_per_atom": float(np.max(np.abs(et - et[0])) / n),
                     "std_eV_per_atom": float(np.std(et) / n), "drift_eV_per_atom_per_ps": slope,
                     "max_displacement_A": float(tr.max_displacement)})
    rows.sort(key=lambda r: r["dt_fs"])
    for a, b in zip(rows[:-1], rows[1:]):
        if abs(b["dt_fs"] / a["dt_fs"] - 2.0) < 1e-9 and a["std_eV_per_atom"] > 0:
            b["order_estimate"] = float(np.log2(b["std_eV_per_atom"] / a["std_eV_per_atom"]))
    return rows


def time_reversibility(model: EnergyModel, system: AtomicSystem, idx, *, dt_fs: float = 1.0, n_steps: int = 100,
                       temperature_K: float = 300.0, seed: int = 0) -> dict[str, float]:
    """Integrate n steps, reverse the momenta, integrate n steps: should return to the start.

    Velocity Verlet is exactly time-reversible in exact arithmetic; the
    residual is floating-point roundoff amplified by the chaotic dynamics
    (grows ~e^{λt}).  Returns max position error (Å) and max momentum error.
    """
    box = system.box

    def fwd_back():
        tr1 = run_md(model, system, idx, dt_fs=dt_fs, n_steps=n_steps, temperature_K=temperature_K, seed=seed)
        tr2 = run_md(model, system, idx, dt_fs=dt_fs, n_steps=n_steps, frac=tr1.frac, momenta=-tr1.momentum)
        return tr2.frac, tr2.momentum

    k_mom, _ = jax.random.split(jax.random.PRNGKey(seed))
    p0 = initial_momenta(system.masses_amu, temperature_K, key=k_mom)
    S2, p2 = jax.jit(fwd_back)()
    dx = float(jnp.max(jnp.linalg.norm((S2 - system.frac) @ box.T, axis=1)))
    dp = float(jnp.max(jnp.abs(-p2 - p0)) / jnp.max(jnp.abs(p0)))
    return {"n_steps": int(n_steps), "dt_fs": float(dt_fs), "max_position_error_A": dx, "max_rel_momentum_error": dp}


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------
def default_structure(candidate: Candidate, model: EnergyModel, skin: float = 0.5) -> AtomicSystem:
    """Starting structure: candidate.structure if present, else relaxed-a0 3C-SiC; replicated for minimum image.

    For SiC the lattice parameter is minimised (isotropic) with the given
    potential so the MD starts from a stress-free perfect crystal.
    """
    from ase import Atoms
    from scipy.optimize import minimize_scalar

    from fracture_md.structures.builders import A_3C_SIC_EXP, sic_3c_conventional

    if candidate.structure is not None:
        rec = candidate.structure
        atoms = Atoms(symbols=rec.symbols, positions=rec.positions_angstrom, cell=rec.cell_rows_angstrom, pbc=rec.pbc)
    elif set(candidate.composition.elements) == {"Si", "C"}:
        a_exp = A_3C_SIC_EXP
        unit = sic_3c_conventional(a_exp)
        s0 = AtomicSystem.from_ase(unit.repeat(auto_supercell(unit, model.cutoff + skin)), elements=model.elements)
        nh = build_neighbors(s0, model.cutoff, skin=skin)
        sp = model.species_for(s0)
        ef = jax.jit(lambda a: model.energy(s0.frac, s0.box * (a / a_exp), sp, nh.idx))
        res = minimize_scalar(lambda a: float(ef(a)), bracket=(0.97 * a_exp, a_exp, 1.03 * a_exp), tol=1e-10)
        atoms = sic_3c_conventional(float(res.x))
    else:
        raise ValueError("autodiff acceptance needs a structure: pass `structure=` or set candidate.structure")
    reps = auto_supercell(atoms, model.cutoff + skin)
    return AtomicSystem.from_ase(atoms.repeat(reps), elements=model.elements)


def run_autodiff_acceptance(
    protocol: AutodiffAcceptanceProtocol,
    candidate: Candidate,
    model: EnergyModel,
    structure: Optional[AtomicSystem] = None,
    context: Optional[RunContext] = None,
) -> ProtocolOutput:
    """Run the acceptance case described by ``protocol``; see the module docstring.

    Checks (one per parameter for the protocol's observable, plus sanity):
      ``autodiff_vs_fd:<obs>:<param>``  pass iff |AD_rev - FD_ref| <= rtol·|FD_ref| + atol with
            rtol = protocol.rel_tolerance and atol = :data:`AD_FD_ATOL` (observable units per unit
            parameter; keeps an exactly-zero derivative from failing on FD round-off);
            metric = relative error (reported)
      ``forward_vs_reverse:<obs>:<param>``  metric = |jvp - jacrev| rel., tol = 1e-9
      ``neighbor_list_valid``  metric = min skin margin over all evaluated trajectories (Å), must be > 0
      ``nve_energy_conservation`` (NVE only)  metric = max |E(t)-E(0)| per atom,
            tol = min(1e-3 eV/atom·max(1, (dt/fs)²), :data:`NVE_TOL_CAP_EV_PER_ATOM`)
    No PropertyRecords are produced: a 0.1 ps trajectory from a perfect
    lattice is not an equilibrated ensemble average, so its energies are not
    material properties.
    """
    from fracture_md import provenance as prov

    context = context or RunContext()
    skin = float(context.extra.get("skin", 0.5))
    remat = bool(context.extra.get("remat", False))
    n_fd = int(context.extra.get("n_fd_steps", 5))
    system = structure if structure is not None else default_structure(candidate, model, skin)
    nh = build_neighbors(system, model.cutoff, skin=skin)
    mds = protocol.md
    obs = make_trajectory_observable(
        model, system, nh.idx, parameters=protocol.parameters, dt_fs=mds.timestep_fs, n_steps=mds.n_steps,
        temperature_K=mds.temperature_K, seed=mds.seed, ensemble=mds.ensemble, remat=remat, skin=skin,
    )
    steps = fd_steps_around(protocol.fd_step, n_fd)
    rep = gradient_check(obs, steps)

    checks: list[CheckResult] = []
    for e in rep.entries:
        if e.observable != protocol.observable:
            continue
        fd_str = ", ".join(f"h={h:.0e}: {v:.10g}" for h, v in e.fd.items())
        atol = AD_FD_ATOL[e.observable]
        abs_err = abs(e.ad_reverse - e.fd_ref)
        ok = ad_fd_agree(e.ad_reverse, e.fd_ref, protocol.rel_tolerance, e.observable)
        checks.append(CheckResult(
            name=f"autodiff_vs_fd:{e.observable}:{e.parameter}",
            passed=ok,
            metric=e.rel_err_reverse, tolerance=protocol.rel_tolerance, unit=Unit.DIMENSIONLESS,
            details=(f"reverse-mode AD {e.ad_reverse:.12g}, forward-mode AD {e.ad_forward:.12g}, FD reference "
                     f"{e.fd_ref:.12g} at h={e.fd_ref_step:.0e}; pass iff |AD - FD| = {abs_err:.3g} <= "
                     f"rtol·|FD| + atol (rtol {protocol.rel_tolerance:g}, atol {atol:g}) "
                     f"(plateau self-consistency {e.fd_spread:.2e}; "
                     f"forward-mode rel. err {e.rel_err_forward:.2e}) [{OBSERVABLE_UNITS[e.observable]} per unit "
                     f"parameter]; FD sweep: {fd_str}"),
        ))
        checks.append(CheckResult(
            name=f"forward_vs_reverse:{e.observable}:{e.parameter}",
            passed=bool(e.rel_err_forward_vs_reverse <= 1e-9), metric=e.rel_err_forward_vs_reverse, tolerance=1e-9,
            unit=Unit.DIMENSIONLESS, details="jax.jvp vs jax.jacrev through lax.scan (should agree to roundoff)",
        ))
    checks.append(CheckResult(
        name="neighbor_list_valid", passed=bool(rep.min_neighbor_margin_A > 0.0), metric=rep.min_neighbor_margin_A,
        tolerance=0.0, unit=Unit.ANGSTROM,
        details=f"skin {skin} Å; margin = skin/2 - strain allowance - max displacement, minimum over the nominal "
                f"and all {2 * len(steps) * len(obs.parameters)} FD trajectories",
    ))
    et = rep.total_energy_series
    econs = float(np.max(np.abs(et - et[0])) / system.n_atoms)
    if mds.ensemble == "nve":
        tol_e = nve_tolerance_eV_per_atom(mds.timestep_fs)
        checks.append(CheckResult(
            name="nve_energy_conservation", passed=bool(econs <= tol_e), metric=econs, tolerance=tol_e,
            unit=Unit.EV_PER_ATOM, details=f"max |E_tot(t) - E_tot(0)| / N over {mds.n_steps} steps of "
                                           f"{mds.timestep_fs} fs (nominal trajectory)",
        ))

    provenance = prov.collect(
        engine="jax_md", potential_id=model.card.id, protocol_id=protocol.protocol_id(),
        random_seeds={"md": int(mds.seed)}, requested_by=context.requested_by, parent_ids=list(context.parent_ids),
        implementation_ids={"fracture_md.physics.md": prov.source_sha256(md_mod),
                            "fracture_md.physics.autodiff_acceptance": prov.source_sha256(run_autodiff_acceptance),
                            **context.implementation_ids},
    )
    raw = {
        "gradient_check": rep.as_dict(),
        "observable": protocol.observable,
        "rel_tolerance": protocol.rel_tolerance,
        "md": {**mds.model_dump(), "n_atoms": system.n_atoms, "skin_A": skin, "remat": remat,
               "time_unit_fs": TIME_UNIT_FS, "effective_timestep_fs": effective_timestep_fs(mds.timestep_fs),
               "k_B_eV_per_K": KB_EV_PER_K},
        "nve_max_energy_deviation_eV_per_atom": econs,
        "notes": "Gradients are derivatives of the discrete, fixed-neighbour-list trajectory map. "
                 "Strain parameters: engineering shear for off-diagonal components. No PropertyRecords "
                 "(non-equilibrium 0.1 ps trajectory).",
        "provenance": provenance.model_dump(mode="json"),
    }
    return ProtocolOutput(
        checks=checks, structure=system,
        artifacts={"total_energy_eV": et},
        raw=raw,
    )
