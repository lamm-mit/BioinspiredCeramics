"""Indentation molecular dynamics: a rigid tip moving on a schedule into a slab with fixed, bath and NVE regions.

This is the JAX-harness indentation driver (synthesis step 7).  It is the verified reference engine for small
cells (<= ~1e5 atoms), cross-engine checks and records; production cells run in LAMMPS with the same tip
functional (recommended protocol items 1-2).  MD indentation numbers are model-specific: whatever is derived
from these trajectories carries a status capped at ``literature_model_agreement`` (records.STATUS_CEILING).

Physical model
--------------
* Material: any harness ``energy_fn(frac, box, species, idx, params)``.  Tip: the rigid, frictionless,
  purely repulsive signed-distance wall of :mod:`fracture_md.physics.indenter`,
  U = (K/3) sum_i max(0, -sdf_i)^3 (LAMMPS fix-indent force law), apex = the lowest point of the tip.
* Regions (per-atom masks, :func:`fracture_md.structures.indentation.indentation_masks`): ``fixed`` atoms
  never move (zero momentum, force and noise; their positions are bitwise constant); ``thermostat``
  atoms (the boundary bath above the fixed layer) are coupled to a heat bath; the rest is NVE.  The ``core``
  (logged as T_core_K) is the contact zone of recommended protocol item 10 and of the LAMMPS decks' ``c_tcore``:
  NVE atoms within ``core_radius_A`` (40 A) of the tip axis and ``core_depth_A`` (40 A) of the top of the slab,
  chosen once from the starting positions.  T_nve_K (every NVE atom) dilutes the tip's local heating by
  N_nve / N_core and is a separate channel.
* Displacement control: the apex follows a piecewise-linear schedule z(t) (:class:`TipSchedule`: approach,
  load at speed v, hold, unload, relax).  The knots are TRACED arrays, so any schedule with the same number
  of knots reuses the compiled kernel.
* Thermostats on the bath only: ``csvr`` (Bussi-Donadio-Parrinello stochastic velocity rescaling of the bath
  kinetic energy, relaxation time tau; the protocol default), ``langevin`` (per-atom friction 1/tau with
  matching noise) or ``none``.

Integrator (one step, BAOAB splitting; validated in outputs/indent_derisk/code-map/proto_inscan_md.py)
-----------------------------------------------------------------------------------------------------
    P += dt/2 F                                    B   (F is zero on fixed atoms)
    S += dt/2 (P/m) box^-T                         A   (true fractional coordinates, unwrapped, as physics/md.py)
    P_bath <- thermostat(P_bath);  q = dKE         O   (heat from the bath, exact tally)
    S += dt/2 (P/m) box^-T                         A
    nbrs <- nbrs.update(S)                         in-scan neighbour list update (lax.cond rebuild at skin/2)
    F  = -dE_mat/dX - dE_ind/dX (masked);  load = -dE_ind/d apex   at apex(t + dt)
    P += dt/2 F                                    B
    w = (load_z(t) + load_z(t+dt))/2 (z(t) - z(t+dt))   tip work: trapezoid of the load over the apex step

The discrete map sees the tip only through the apex heights z_n at the kicks.  Along the straight path from
(S_n, z_n) to (S_n+1, z_n+1) the trapezoid rule gives dE_pot = -(F_n + F_n+1)/2 . dX - (L_n + L_n+1)/2 dz +
O(dt^3), and velocity Verlet returns the first term as kinetic energy to O(dt^3), so with the trapezoidal tip
work above the energy ledger

    L(t) = KE + E_mat + E_ind - W(t) - Q(t) - E(0)

is conserved to O(dt^2), also across schedule knots (no velocity enters), and W is the trapezoidal integral
of the load-displacement curve (720-atom Vashishta slab at 300 K with a moving tip: max |L| ratio 4.02
between dt = 1 and 0.5 fs, with or without a CSVR bath).  The "energy jump of the tip at fixed S",
E_ind(S, z_n+1) - E_ind(S, z_n), is only first order: it misses (dt/2) int v^2 d2E_ind/dz^2 dt, which for a
stiff wall (K = 100 eV/A^3) was ~7 % of W on the same test (ratio 2.06).  The bath heat q is the
kinetic-energy change of the O step (exact).  W and Q are accumulated on the host in float64 with Kahan
compensation, also for float32 runs (the per-step tip work of a large system is far below the float32 ulp of
its total energy).  The ledger drift ratio between dt and dt/2 is a regression check (>= 2.5; ~4 expected).

Newton 3: every step also yields the fixed-layer reaction R (the pre-zero total force on the frozen atoms, the
LAMMPS decks' ``fix setforce`` sum Rx Ry Rz), the total momentum p of the mobile atoms and the thermostat's momentum
kick of the O step.  The material forces sum to zero and the tip forces to -load, so the force on the mobile atoms
is -(load + R), and velocity Verlet makes p(t) - p(0) = -int (load + R) dt + (bath kicks) an identity.  The host
accumulates both integrals over EVERY step (trapezoid, float64), so :meth:`IndentMDResult.momentum_balance` checks
the identity at any ``log_every_steps`` (round-off for a correct reaction channel).  The sampled load + reaction
(channel ``reaction_nN``) is the per-run Newton-3 check of physics/indentation.py: its mean over the contact samples
measures how much of the tip's impulse is still in the slab instead of in the fixed layer.

Neighbour lists, rollback and reallocation
------------------------------------------
The Dense list (potentials/neighbors.py; all-pairs for small cells, the Cartesian cell list for large
orthorhombic slabs) is updated INSIDE the scan at every step: a list held fixed over a chunk misses in-cutoff
pairs at 300 K (code-map M1).  Shapes cannot change inside a scan, so a capacity overflow only sets the sticky
error bits; the host then discards the chunk, re-allocates from the chunk-START configuration with more
headroom (``capacity_headroom``) and redoes it with the same random key.  Vashishta three-body slot overflow
(more than k3 partners inside r0) is handled the same way: the chunk kernel evaluates the potential through
``energy_fn.energy_and_status`` and ORs the flag; the host switches to a model with more slots
(``potentials.vashishta.with_three_body_slots``, same card) and redoes the chunk.  A re-allocation changes
array shapes and recompiles the kernel once; otherwise the kernel compiles exactly once per
(energy_fn, geometry, neighbour function, N, M, chunk length).  A rolled-back run reproduces a run that had
the final capacity from the start to round-off (1e-14 relative in the load through the redone chunks); the
two differ only in the summation order over differently padded lists, and that round-off later grows like
any perturbation of a chaotic trajectory (~10x per 50 fs in contact).

Checkpoint / restart
--------------------
At chunk boundaries (every ``checkpoint_every_s`` of wall time, default 30 min, or every
``checkpoint_every_chunks``) the full state is written atomically: positions, momenta, forces, PRNG key,
step, the neighbour list itself (idx, reference positions, error bits, capacities), the three-body slot count,
the Kahan accumulators and the per-step logs.  A restart restores the neighbour list rather than rebuilding
it, so a killed and resumed run is bitwise identical to an uninterrupted one on the CPU.

Frames
------
:class:`FrameWriter` writes animation frames at chunk boundaries (every ``every_ps`` of simulated time):
float32 positions of a slice through the tip axis plus the atoms near the top surface (chosen once from the
first frame, :func:`fracture_md.io.indent_traj.animation_subset_mask`) and per-atom fields (``contact``,
``penetration_A``, ``ke_eV``).  Frames are shards on disk (they survive a restart) and are assembled into one
``indentviz-traj/1`` file (fracture_md.io.indent_traj, read by the README renderer) with the dense load-depth
record.  The projected storage is checked against ``max_bytes`` before the run starts.

Units: eV, Angstrom, amu; time in fs (dt) and ps (logs, frames); momenta in amu A / tau
(tau = 10.1805 fs, physics/md.py); loads in eV/A in the logs (1 eV/A = 1.602176634 nN) and nN in frames.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, space

from fracture_md.physics.indenter import (
    DEFAULT_STIFFNESS_EV_PER_A3,
    IndenterGeometry,
    check_indenter_cell,
    check_z_guard,
    indenter_geometry_of,
    material_energy_fn_of,
    penetration,
)
from fracture_md.physics.md import KB_EV_PER_K, TIME_UNIT_FS
from fracture_md.potentials.base import EnergyModel
from fracture_md.potentials.neighbors import NeighborHandle, build_neighbors, overflow_bits
from fracture_md.structures.system import AtomicSystem

EV_PER_A_TO_NN = 1.602176634  # exact (e and the angstrom are SI-exact)
M_PER_S_TO_A_PER_FS = 1.0e-5  # 1 m/s = 1e10 A / 1e15 fs
THERMOSTATS = ("csvr", "langevin", "none")
DTYPES = ("float64", "float32")
FRAME_FIELDS = ("contact", "penetration_A", "ke_eV")
CHECKPOINT_FORMAT = "fracture-md-indent-md-checkpoint/1"
# Channels of the momentum balance (Newton 3 through the slab; the last three are running integrals over every step).
# A checkpoint without them cannot resume: the integrals would restart from zero mid-run.
_NEWTON3_LOGS = ("reaction_eV_per_A", "momentum_amu_A_per_tau", "impulse_load_reaction_amu_A_per_tau",
                 "impulse_load_amu_A_per_tau", "bath_momentum_kick_amu_A_per_tau")
_NEWTON3_INITIAL = ("load_eV_per_A", "reaction_eV_per_A", "momentum_amu_A_per_tau")
_CHECKPOINT_NAME = "indent_md_checkpoint.npz"
KNOT_TOL_FS = 1e-6  # a time within this of a schedule knot is on the knot (TipSchedule.segment)


# ============================================================================ configuration and schedule
@dataclasses.dataclass(frozen=True)
class IndentMDConfig:
    """Numerical settings of an indentation MD run (defaults follow the recommended protocol, recipe_v0).

    dt_fs: velocity-Verlet time step (1 fs is adequate for Vashishta, G09; Tersoff 1989 needs <= 0.5 fs).
    thermostat / thermostat_tau_ps: bath coupling ("csvr" default, tau 0.1 ps; "langevin": friction 1/tau).
    chunk_steps: MD steps per compiled lax.scan call; frames, checks and checkpoints happen at chunk ends.
    skin_A, capacity_multiplier, extra_capacity: neighbour list radius cutoff + skin and Dense capacity
    (occupancy x multiplier + extra); capacity_headroom is added to extra_capacity on every overflow rollback.
    three_body_headroom: extra slots above the measured partner count after a three-body overflow.
    dtype: "float64" (records, default) or "float32" (traced with matmul precision "highest"; ledger on the
    host in float64 anyway).  log_every_steps: decimation of the STORED per-step logs (the ledger and W, Q
    always use every step; the steps at the schedule knots and at chunk ends are always stored).
    checkpoint_every_s / checkpoint_every_chunks: checkpoint cadence (when a checkpoint directory is given).
    z_guard_every_chunks: cadence of the tip-column height-cap check.  core_radius_A / core_depth_A: the core
    region of T_core_K (:func:`core_region_mask`; the LAMMPS decks' 40 A ``rcore`` cylinder; ``core_depth_A``
    None: the whole NVE column).
    """

    dt_fs: float = 1.0
    temperature_K: float = 300.0
    thermostat: str = "csvr"
    thermostat_tau_ps: float = 0.1
    chunk_steps: int = 500
    skin_A: float = 1.0
    capacity_multiplier: float = 1.25
    extra_capacity: int = 8
    capacity_headroom: int = 16
    three_body_headroom: int = 4
    max_rollbacks_per_chunk: int = 4
    neighbor_method: str = "auto"
    dtype: str = "float64"
    seed: int = 0
    log_every_steps: int = 1
    checkpoint_every_s: float = 1800.0
    checkpoint_every_chunks: Optional[int] = None
    z_guard_every_chunks: int = 1
    core_radius_A: float = 40.0
    core_depth_A: Optional[float] = 40.0

    def __post_init__(self) -> None:
        if self.thermostat not in THERMOSTATS:
            raise ValueError(f"thermostat must be one of {THERMOSTATS}, got {self.thermostat!r}")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if not (self.dt_fs > 0 and self.thermostat_tau_ps > 0 and self.temperature_K >= 0):
            raise ValueError("dt_fs > 0, thermostat_tau_ps > 0 and temperature_K >= 0 required")
        if self.chunk_steps < 1 or self.log_every_steps < 1 or self.z_guard_every_chunks < 1:
            raise ValueError("chunk_steps, log_every_steps and z_guard_every_chunks must be >= 1")
        if self.skin_A < 0 or self.capacity_multiplier < 1.0 or self.extra_capacity < 0 or self.capacity_headroom < 1:
            raise ValueError("skin_A >= 0, capacity_multiplier >= 1, extra_capacity >= 0, capacity_headroom >= 1")
        if self.max_rollbacks_per_chunk < 0 or self.three_body_headroom < 0:
            raise ValueError("max_rollbacks_per_chunk and three_body_headroom must be >= 0")
        if self.checkpoint_every_chunks is not None and self.checkpoint_every_chunks < 1:
            raise ValueError("checkpoint_every_chunks must be >= 1 (or None)")
        if not self.core_radius_A > 0 or (self.core_depth_A is not None and not self.core_depth_A > 0):
            raise ValueError("core_radius_A > 0 and core_depth_A > 0 (or None) required")

    @property
    def chunk_fs(self) -> float:
        return self.chunk_steps * self.dt_fs

    @classmethod
    def from_protocol(cls, protocol: Any, **overrides: Any) -> "IndentMDConfig":
        """Settings of a ``schema.protocol.IndentationProtocol`` (mode "md"); ``overrides`` win.

        Maps timestep_fs, temperature_K, thermostat, thermostat_tau_ps, skin_A, neighbor_method, seed and
        log_every_steps; the chunk length defaults to the frame period (frames are written at chunk ends).
        """
        kw: dict[str, Any] = dict(dt_fs=protocol.timestep_fs, temperature_K=protocol.temperature_K,
                                  thermostat=protocol.thermostat, thermostat_tau_ps=protocol.thermostat_tau_ps,
                                  skin_A=protocol.skin_A, neighbor_method=protocol.neighbor_method,
                                  seed=protocol.seed, log_every_steps=protocol.log_every_steps)
        if protocol.frame_every_ps is not None:
            kw["chunk_steps"] = max(1, int(round(1e3 * protocol.frame_every_ps / protocol.timestep_fs)))
        kw.update(overrides)
        return cls(**kw)

    def physics_dict(self) -> dict[str, Any]:
        """The settings that define the trajectory (a restart must match them; cadences may change)."""
        d = dataclasses.asdict(self)
        for k in ("checkpoint_every_s", "checkpoint_every_chunks", "log_every_steps", "z_guard_every_chunks",
                  "max_rollbacks_per_chunk"):
            d.pop(k)
        return d

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TipSchedule:
    """Piecewise-linear apex path: z(t) through knots (t_fs, z_A) at a fixed lateral position.

    ``z_knots_A`` are absolute heights of the apex (the lowest point of the tip); the depth is
    ``surface_z0 - z``.  Beyond the last knot the apex stays at the last height.  ``labels`` names the
    K - 1 segments ("approach", "load", "hold", "unload", "relax", ...).  The knots are passed to the chunk
    kernel as traced arrays (no recompilation for a new schedule with the same number of knots).
    """

    t_knots_fs: tuple[float, ...]
    z_knots_A: tuple[float, ...]
    xy_A: tuple[float, float]
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        t = np.asarray(self.t_knots_fs, float)
        z = np.asarray(self.z_knots_A, float)
        object.__setattr__(self, "t_knots_fs", tuple(float(x) for x in t))
        object.__setattr__(self, "z_knots_A", tuple(float(x) for x in z))
        object.__setattr__(self, "xy_A", tuple(float(x) for x in self.xy_A))
        if t.ndim != 1 or t.shape != z.shape or len(t) < 2:
            raise ValueError("t_knots_fs and z_knots_A must be 1-D of equal length >= 2")
        if t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
            raise ValueError("t_knots_fs must start at 0 and increase strictly")
        if not np.all(np.isfinite(z)) or len(self.xy_A) != 2:
            raise ValueError("z_knots_A must be finite and xy_A a pair")
        labels = tuple(self.labels) if self.labels else tuple(f"segment{i}" for i in range(len(t) - 1))
        if len(labels) != len(t) - 1:
            raise ValueError("labels must name the len(t_knots_fs) - 1 segments")
        object.__setattr__(self, "labels", labels)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def constant(cls, apex_xyz: Sequence[float], duration_fs: float, label: str = "hold") -> "TipSchedule":
        """The apex held at ``apex_xyz`` (e.g. above the surface during equilibration)."""
        x, y, z = (float(v) for v in apex_xyz)
        return cls((0.0, float(duration_fs)), (z, z), (x, y), (label,))

    @classmethod
    def load_hold_unload(
        cls,
        apex_xy: Sequence[float],
        surface_z_A: float,
        *,
        max_depth_A: float,
        speed_m_per_s: float,
        initial_gap_A: float = 3.0,
        hold_ps: float = 10.0,
        unload: bool = True,
        final_gap_A: Optional[float] = None,
        unload_speed_m_per_s: Optional[float] = None,
        relax_ps: float = 0.0,
        pre_hold_ps: float = 0.0,
        dt_fs: Optional[float] = None,
    ) -> "TipSchedule":
        """Displacement-controlled load / hold / unload at constant speed (recommended protocol items 11-14).

        The apex starts ``initial_gap_A`` above the surface (after an optional ``pre_hold_ps``), moves down
        at ``speed_m_per_s`` to ``max_depth_A`` below it (approach + load), holds for ``hold_ps``, then (if
        ``unload``) moves up at the unloading speed (default: the loading speed) to ``final_gap_A`` above
        the surface (default: the initial gap; the load reaches zero on the way) and stays there for
        ``relax_ps``.  ``dt_fs``: round every segment to a whole number of time steps (at least one), keeping
        its heights, so every knot is on the step grid and a step samples the apex at exactly ``max_depth_A``
        (the speeds change by less than dt / segment duration; LAMMPS decks round the step counts the same way).
        """
        if not (max_depth_A > 0 and speed_m_per_s > 0 and initial_gap_A >= 0 and hold_ps >= 0 and relax_ps >= 0
                and pre_hold_ps >= 0):
            raise ValueError("max_depth_A, speed > 0; gaps, hold, relax, pre_hold >= 0")
        if dt_fs is not None and not dt_fs > 0:
            raise ValueError("dt_fs > 0 required")
        v = speed_m_per_s * M_PER_S_TO_A_PER_FS
        vu = (unload_speed_m_per_s or speed_m_per_s) * M_PER_S_TO_A_PER_FS
        z0 = surface_z_A + initial_gap_A
        zmin = surface_z_A - max_depth_A
        durations, z, lab = [], [z0], []
        if pre_hold_ps > 0:
            durations.append(1e3 * pre_hold_ps); z.append(z0); lab.append("pre_hold")
        if initial_gap_A > 0:
            durations.append(initial_gap_A / v); z.append(surface_z_A); lab.append("approach")
        durations.append(max_depth_A / v); z.append(zmin); lab.append("load")
        if hold_ps > 0:
            durations.append(1e3 * hold_ps); z.append(zmin); lab.append("hold")
        if unload:
            zf = surface_z_A + (initial_gap_A if final_gap_A is None else float(final_gap_A))
            durations.append((zf - zmin) / vu); z.append(zf); lab.append("unload")
            if relax_ps > 0:
                durations.append(1e3 * relax_ps); z.append(zf); lab.append("relax")
        if dt_fs is None:
            t = np.concatenate([[0.0], np.cumsum(durations)])
        else:  # whole steps, accumulated in integers (no drift of the knots off the grid)
            t = float(dt_fs) * np.concatenate([[0], np.cumsum([max(1, int(round(d / dt_fs))) for d in durations])])
        return cls(tuple(t), tuple(z), (float(apex_xy[0]), float(apex_xy[1])), tuple(lab))

    @classmethod
    def from_protocol(cls, protocol: Any, apex_xy: Sequence[float], surface_z_A: float) -> "TipSchedule":
        """Load / hold / unload / relax schedule of a ``schema.protocol.IndentationProtocol`` (mode "md"), with the
        knots on the protocol's time-step grid (``dt_fs = protocol.timestep_fs``)."""
        return cls.load_hold_unload(apex_xy, surface_z_A, max_depth_A=protocol.max_depth_A,
                                    speed_m_per_s=protocol.speed_m_per_s, initial_gap_A=protocol.initial_gap_A,
                                    hold_ps=protocol.hold_ps, unload=protocol.unload,
                                    relax_ps=protocol.relax_after_unload_ps if protocol.unload else 0.0,
                                    dt_fs=protocol.timestep_fs)

    # ------------------------------------------------------------------ queries
    @property
    def duration_fs(self) -> float:
        return self.t_knots_fs[-1]

    @property
    def max_apex_z_A(self) -> float:
        return max(self.z_knots_A)

    def n_steps(self, dt_fs: float, chunk_steps: int = 1) -> int:
        """Steps covering the schedule, rounded UP to whole chunks (the tail holds the last apex height)."""
        n = int(math.ceil(self.duration_fs / dt_fs - 1e-9))
        return int(math.ceil(n / chunk_steps)) * chunk_steps

    def apex_z(self, t_fs: Any) -> np.ndarray:
        return np.interp(np.asarray(t_fs, float), self.t_knots_fs, self.z_knots_A)

    def apex(self, t_fs: float) -> np.ndarray:
        return np.array([self.xy_A[0], self.xy_A[1], float(self.apex_z(t_fs))])

    def segment(self, t_fs: Any) -> np.ndarray:
        """Label of the segment at each time; a time ON a knot belongs to the segment that ends there.

        The sample at the load -> hold knot (the full maximum depth) is therefore "load", the convention of
        the LAMMPS importer (io/lammps_indent.py: step <= nload is "load"), so both engines evaluate the
        loading branch up to h_max.  Times within ``KNOT_TOL_FS`` of a knot count as on it (the logs store
        t_ps = step dt 1e-3, whose round-off can land either side); t = 0 is in the first segment and times
        beyond the end in the last.
        """
        t = np.asarray(t_fs, float) - KNOT_TOL_FS
        i = np.clip(np.searchsorted(self.t_knots_fs, t, side="left") - 1, 0, len(self.labels) - 1)
        return np.asarray(self.labels)[i]

    def knot_steps(self, dt_fs: float) -> np.ndarray:
        """MD steps (>= 1) at or next to each knot after t = 0: the steps just before and just after it (one
        step when the knot is on the step grid).  The driver always stores them, whatever ``log_every_steps``,
        so the deepest loading sample and the segment boundaries are in the logs."""
        s = np.asarray(self.t_knots_fs[1:], float) / float(dt_fs)
        tol = KNOT_TOL_FS / float(dt_fs)
        st = np.unique(np.concatenate([np.floor(s + tol), np.ceil(s - tol)])).astype(np.int64)
        return st[st >= 1]

    def arrays(self, dtype=jnp.float64) -> dict[str, jnp.ndarray]:
        return {"t_knots": jnp.asarray(self.t_knots_fs, dtype=jnp.float64),
                "z_knots": jnp.asarray(self.z_knots_A, dtype=dtype), "xy": jnp.asarray(self.xy_A, dtype=dtype)}

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ============================================================================ momenta and regions
def region_momenta(masses: Any, regions: Union[Mapping[str, Any], Sequence[Any]], temperature_K: float, *,
                   key: Optional[jnp.ndarray] = None, seed: int = 0, zero_momentum: bool = True) -> jnp.ndarray:
    """Maxwell-Boltzmann momenta (amu A / tau) per region; atoms in no region (fixed) get exactly zero.

    ``regions``: disjoint boolean masks (N,) (a dict or a sequence).  In each region the momenta are drawn
    at ``temperature_K``, the region's centre-of-mass momentum is removed (``zero_momentum``) and the
    region is rescaled to exactly ``temperature_K`` with 3 n - 3 degrees of freedom (3 n without the
    removal).  Deterministic for a given key (or seed).
    """
    m = np.asarray(masses, np.float64).reshape(-1)
    n = m.size
    masks = [np.asarray(r, bool).reshape(-1) for r in (regions.values() if isinstance(regions, Mapping) else regions)]
    if any(r.size != n for r in masks):
        raise ValueError("every region mask must have one entry per atom")
    if masks and np.any(np.sum(np.stack(masks), axis=0) > 1):
        raise ValueError("region masks must be disjoint")
    key = jax.random.PRNGKey(seed) if key is None else key
    p = np.zeros((n, 3))
    kT = KB_EV_PER_K * float(temperature_K)
    for r, k in zip(masks, jax.random.split(key, max(len(masks), 1))):
        cnt = int(r.sum())
        if cnt == 0 or temperature_K <= 0.0:
            continue
        mr = m[r][:, None]
        pr = np.sqrt(mr * kT) * np.asarray(jax.random.normal(k, (cnt, 3), dtype=jnp.float64))
        dof = 3 * cnt
        if zero_momentum and cnt > 1:
            pr = pr - mr * (pr.sum(axis=0, keepdims=True) / mr.sum())
            dof -= 3
        t_now = np.sum(pr**2 / mr) / (dof * KB_EV_PER_K)
        p[r] = pr * math.sqrt(temperature_K / t_now)
    return jnp.asarray(p)


def _region_masks(n: int, masks: Mapping[str, Any], thermostat_mask: Optional[Any]) -> dict[str, np.ndarray]:
    """fixed / thermostat / nve (disjoint, covering) plus optional analysis, from the slab masks."""
    fixed = np.asarray(masks.get("fixed", np.zeros(n, bool)), bool).reshape(-1)
    if thermostat_mask is None:
        thermostat_mask = masks.get("bath", masks.get("thermostat", np.zeros(n, bool)))
    thermo = np.asarray(thermostat_mask, bool).reshape(-1) & ~fixed
    if fixed.size != n or thermo.size != n:
        raise ValueError("region masks must have one entry per atom")
    out = {"fixed": fixed, "thermostat": thermo, "nve": ~fixed & ~thermo}
    if masks.get("analysis") is not None:
        out["analysis"] = np.asarray(masks["analysis"], bool).reshape(-1) & ~fixed
    return out


def core_region_mask(positions: Any, box: Any, nve: Any, tip_xy: Sequence[float], *, radius_A: float = 40.0,
                     depth_A: Optional[float] = 40.0) -> np.ndarray:
    """The core of T_core_K: NVE atoms in the contact-zone cylinder about the tip axis (recommended protocol item 10).

    An atom is in the core if it is in ``nve``, its lateral distance from ``tip_xy`` (minimum image in an
    orthorhombic ``box``, lattice vectors as columns) is below ``radius_A`` and (unless ``depth_A`` is None) it
    lies less than ``depth_A`` below the highest atom.  This is the LAMMPS decks' ``rcore`` cylinder (radius
    ``corer`` = 40 A about the tip axis from ``ztop0 - corer`` up; examples/indent/in.berk_sdf.lmp), chosen
    once from ``positions`` (Cartesian, A) instead of LAMMPS's per-step region membership.
    """
    X = np.asarray(positions, np.float64).reshape(-1, 3)
    L = np.diag(np.asarray(box, np.float64))[:2]
    dxy = X[:, :2] - np.asarray(tip_xy, np.float64)
    dxy -= L * np.round(dxy / L)
    core = np.asarray(nve, bool).reshape(-1) & (np.hypot(dxy[:, 0], dxy[:, 1]) < radius_A)
    if depth_A is not None:
        core &= X[:, 2] >= X[:, 2].max() - depth_A
    return core


# ============================================================================ the chunk kernel
class IndentMDState(NamedTuple):
    """Device state carried across chunks (a pytree)."""

    frac: jnp.ndarray  # (N, 3) unwrapped fractional coordinates
    momentum: jnp.ndarray  # (N, 3) amu A / tau (zero on fixed atoms)
    force: jnp.ndarray  # (N, 3) eV/A at (frac, apex(step)), zero on fixed atoms
    load: jnp.ndarray  # (3,) eV/A, force of the material on the tip at (frac, apex(step))
    key: jnp.ndarray  # PRNG key of the thermostat
    nbrs: Any  # JAX-MD NeighborList
    step: jnp.ndarray  # int64 scalar: MD steps done (t = step * dt)


# Compiled-kernel caches keyed on the identity of the material energy function and the neighbour function
# (both are cached objects in their own modules); the referenced objects are kept to make id() reuse
# impossible.  Bounded like potentials.neighbors._NFN_CACHE (oldest entries are dropped first).
_KERNELS: dict[tuple, tuple[tuple, Callable]] = {}
_AUX_KERNELS: dict[tuple, tuple[tuple, dict]] = {}
_KERNEL_CACHE_MAX = 64


def _cache_put(cache: dict, key: tuple, value: tuple) -> None:
    if len(cache) >= _KERNEL_CACHE_MAX:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _material_energy(material_fn: Callable, with_status: bool) -> Callable:
    """(frac, box, species, idx, params) -> (E, n3_max) with a real-space gradient w.r.t. frac."""
    if with_status:
        ens = material_fn.energy_and_status  # type: ignore[attr-defined]

        def f(S, box, species, idx, p):
            e, st = ens(S, box, species, idx, p)
            return e, st.max_partners.astype(jnp.int32)
    else:
        def f(S, box, species, idx, p):
            return material_fn(S, box, species, idx, p), jnp.zeros((), jnp.int32)
    return f


def _indenter_terms(S, apex, box, K, geometry: IndenterGeometry):
    """(E_ind, (sum delta^2, sum delta^3, n_contact, max delta)); positions through space.transform."""
    X = space.transform(box, S)
    delta = penetration(X, apex, geometry, box)
    s2 = jnp.sum(delta**2)
    s3 = jnp.sum(delta**3)
    return (K / 3.0) * s3, (s2, s3, jnp.sum(delta > 0.0).astype(jnp.int32), jnp.max(delta))


def _forces_and_terms(material_fn, with_status, geometry, S, box, species, idx, p, apex, K, free):
    """Masked total force, material and tip energies, load (3,), tip diagnostics, three-body count and the
    fixed-layer reaction (3,): the PRE-ZERO total force on the frozen atoms (LAMMPS ``fix setforce`` f_fb, the
    decks' Rx Ry Rz), whose z component balances the tip load in equilibrium (Newton 3 through the slab)."""
    fmat = _material_energy(material_fn, with_status)
    (e_mat, n3), g_mat = jax.value_and_grad(fmat, argnums=0, has_aux=True)(S, box, species, idx, p)
    (e_ind, aux), (g_ind, g_apex) = jax.value_and_grad(_indenter_terms, argnums=(0, 1), has_aux=True)(
        S, apex, box, K, geometry)
    F_all = -(g_mat + g_ind)
    F = jnp.where(free, F_all, 0.0)
    reaction = jnp.sum(jnp.where(free, 0.0, F_all), axis=0)
    return F, e_mat, e_ind, -g_apex, aux, n3, reaction


def _apex_at(consts: dict, step) -> jnp.ndarray:
    t = step.astype(jnp.float64) * consts["dt_fs"]
    z = jnp.interp(t, consts["t_knots"], consts["z_knots"].astype(jnp.float64)).astype(consts["xy"].dtype)
    return jnp.concatenate([consts["xy"], z[None]])


def make_chunk_kernel(
    material_fn: Callable,
    geometry: IndenterGeometry,
    n_steps: int,
    *,
    neighbor_fn: Any,
    thermostat: str = "csvr",
    n_regions: int = 2,
    with_status: Optional[bool] = None,
) -> Callable:
    """Jitted ``kernel(state, consts) -> (state, logs)`` advancing ``n_steps`` MD steps in one lax.scan.

    ``material_fn``: the material energy function (a composite made by ``with_indenter`` is not accepted:
    the tip term is evaluated separately so that the load and the tip diagnostics come out of the same
    pass).  ``neighbor_fn``: the handle's ``neighbor_fn`` (its ``update(frac, nbrs)`` runs inside the scan).
    ``with_status``: evaluate through ``material_fn.energy_and_status`` (Vashishta three-body slots;
    default: whenever the attribute exists).  ``consts`` (all traced, see :func:`run_indentation_md`):
    box, binv, species, params, K, mass (N, 1), free (N, 1) bool, thermo (N, 1) bool, regions (R, N) bool,
    kT, dt (internal units), dt_fs, c_thermo = exp(-dt/tau), dof_thermo, t_knots, z_knots, xy.

    Per-step logs (arrays of length n_steps): step, apex_z, load (3,), e_mat, e_ind, e_kin, w (tip work,
    trapezoid of the load over the apex step),
    q (bath heat), s2, s3 (sum delta^2, delta^3: pen_fw = s3/s2), n_contact, max_pen, ke_regions (R,),
    rebuilt, n3 (three-body partners), finite, reaction (3,: pre-zero total force on the fixed atoms, eV/A),
    momentum (3,: total momentum of the atoms after the step, amu A / tau; fixed atoms carry none) and dp_bath
    (3,: the thermostat's change of the bath momentum in the O step).  Velocity Verlet makes the momentum
    balance an identity: momentum(n) - momentum(n-1) = -(dt/2) [(load + reaction)(n-1) + (load + reaction)(n)]
    + dp_bath(n), because the material forces sum to zero and the tip forces to -load.  Cached per
    (material_fn, geometry, n_steps, neighbor_fn, thermostat, n_regions, with_status): repeated runs and chunks
    reuse the compiled kernel.
    """
    if material_energy_fn_of(material_fn) is not None:
        raise ValueError("make_chunk_kernel takes the MATERIAL energy function (the tip is added by the kernel)")
    if thermostat not in THERMOSTATS:
        raise ValueError(f"thermostat must be one of {THERMOSTATS}")
    if with_status is None:
        with_status = hasattr(material_fn, "energy_and_status")
    n_steps = int(n_steps)
    key = (id(material_fn), geometry, n_steps, id(neighbor_fn), thermostat, int(n_regions), bool(with_status))
    hit = _KERNELS.get(key)
    if hit is not None and hit[0][0] is material_fn and hit[0][1] is neighbor_fn:
        return hit[1]
    update = neighbor_fn.update

    def kernel(state: IndentMDState, c: dict):
        with jax.default_matmul_precision("highest"):  # float32: no TF32 in the fractional updates
            m = c["mass"]
            dt = c["dt"]
            binv_t = c["binv"].T
            nbr_dtype = state.nbrs.reference_position.dtype

            def ke_atoms(P):
                return 0.5 * jnp.sum(P * P / m, axis=1)

            def body(carry, _):
                S, P, F, load0, key, nbrs, step = carry
                apex0 = _apex_at(c, step)
                apex1 = _apex_at(c, step + 1)
                P = P + (0.5 * dt) * F  # B
                S = S + (0.5 * dt) * (P / m) @ binv_t  # A
                # ---- O: bath only
                if thermostat == "csvr":
                    key, k1, k2 = jax.random.split(key, 3)
                    ke_th = jnp.sum(jnp.where(c["thermo"][:, 0], ke_atoms(P), 0.0))
                    nf = c["dof_thermo"]
                    kt = 0.5 * nf * c["kT"]
                    cc = c["c_thermo"]
                    r1 = jax.random.normal(k1, (), dtype=P.dtype)
                    chi2 = 2.0 * jax.random.gamma(k2, 0.5 * (nf - 1.0), dtype=P.dtype)  # chi^2 with nf - 1 dof
                    safe = jnp.where(ke_th > 0.0, ke_th, 1.0)
                    ke_new = (ke_th + (1.0 - cc) * (kt * (chi2 + r1 * r1) / nf - ke_th)
                              + 2.0 * r1 * jnp.sqrt(jnp.maximum(ke_th * kt / nf * (1.0 - cc) * cc, 0.0)))
                    alpha = jnp.where(ke_th > 0.0, jnp.sqrt(jnp.maximum(ke_new, 0.0) / safe), 1.0)
                    dp_bath = (alpha - 1.0) * jnp.sum(jnp.where(c["thermo"], P, 0.0), axis=0)
                    P = jnp.where(c["thermo"], alpha * P, P)
                    q = (alpha * alpha - 1.0) * ke_th
                elif thermostat == "langevin":
                    key, k1 = jax.random.split(key)
                    cc = c["c_thermo"]
                    ke0 = jnp.sum(jnp.where(c["thermo"][:, 0], ke_atoms(P), 0.0))
                    xi = jax.random.normal(k1, P.shape, dtype=P.dtype)
                    P_new = jnp.where(c["thermo"], cc * P + jnp.sqrt(c["kT"] * (1.0 - cc * cc) * m) * xi, P)
                    dp_bath = jnp.sum(P_new - P, axis=0)
                    P = P_new
                    q = jnp.sum(jnp.where(c["thermo"][:, 0], ke_atoms(P), 0.0)) - ke0
                else:
                    q = jnp.zeros((), P.dtype)
                    dp_bath = jnp.zeros((3,), P.dtype)
                S = S + (0.5 * dt) * (P / m) @ binv_t  # A
                new = update(S.astype(nbr_dtype), nbrs)  # in-scan list update (rebuild inside when needed)
                rebuilt = jnp.any(new.reference_position != nbrs.reference_position)
                nbrs = new
                F, e_mat, e_ind, load, aux, n3, reaction = _forces_and_terms(
                    material_fn, with_status, geometry, S, c["box"], c["species"], nbrs.idx, c["params"], apex1,
                    c["K"], c["free"])
                P = P + (0.5 * dt) * F  # B
                # tip work on the material: trapezoid of the load over the apex step (the tip moves along z only)
                w = 0.5 * (load0[2] + load[2]) * (apex0[2] - apex1[2])
                ka = ke_atoms(P)
                ke = jnp.sum(ka)
                ke_reg = jnp.sum(jnp.where(c["regions"], ka[None, :], 0.0), axis=1)
                step = step + 1
                log = {
                    "step": step, "apex_z": apex1[2], "load": load, "e_mat": e_mat, "e_ind": e_ind, "e_kin": ke,
                    "w": w, "q": q, "s2": aux[0], "s3": aux[1], "n_contact": aux[2], "max_pen": aux[3],
                    "ke_regions": ke_reg, "rebuilt": rebuilt, "n3": n3,
                    "finite": jnp.isfinite(e_mat) & jnp.isfinite(ke) & jnp.isfinite(e_ind),
                    "reaction": reaction, "momentum": jnp.sum(P, axis=0), "dp_bath": dp_bath,
                }
                return (S, P, F, load, key, nbrs, step), log

            carry = (state.frac, state.momentum, state.force, state.load, state.key, state.nbrs, state.step)
            carry, logs = jax.lax.scan(body, carry, None, length=n_steps)
            return IndentMDState(*carry), logs

    fn = jax.jit(kernel)
    _cache_put(_KERNELS, key, ((material_fn, neighbor_fn), fn))
    return fn


def _aux_kernels(material_fn: Callable, geometry: IndenterGeometry, with_status: bool) -> dict[str, Callable]:
    """Jitted initial-force and frame-field kernels (cached per material function and geometry)."""
    key = (id(material_fn), geometry, bool(with_status))
    hit = _AUX_KERNELS.get(key)
    if hit is not None and hit[0][0] is material_fn:
        return hit[1]

    def init(S, box, species, idx, p, apex, K, free):
        with jax.default_matmul_precision("highest"):
            F, e_mat, e_ind, load, aux, n3, reaction = _forces_and_terms(material_fn, with_status, geometry, S, box,
                                                                         species, idx, p, apex, K, free)
            return {"force": F, "e_mat": e_mat, "e_ind": e_ind, "load": load, "s2": aux[0], "s3": aux[1],
                    "n_contact": aux[2], "max_pen": aux[3], "n3": n3, "reaction": reaction}

    def frame(S, box, apex):
        with jax.default_matmul_precision("highest"):  # float32 frames: TF32 in space.transform moved atoms ~0.04 A
            X = space.transform(box, S)
            return X, penetration(X, apex, geometry, box)

    fns = {"init": jax.jit(init), "frame": jax.jit(frame)}
    _cache_put(_AUX_KERNELS, key, ((material_fn,), fns))
    return fns


# ============================================================================ host-side helpers
class _Kahan:
    """Compensated float64 sum (W and Q ledgers)."""

    __slots__ = ("s", "c")

    def __init__(self, s: float = 0.0, c: float = 0.0):
        self.s, self.c = float(s), float(c)

    def add_series(self, x: np.ndarray) -> np.ndarray:
        """Add every entry in order; return the running sums after each addition."""
        out = np.empty(len(x))
        s, c = self.s, self.c
        for i, v in enumerate(np.asarray(x, np.float64).tolist()):
            y = v - c
            t = s + y
            c = (t - s) - y
            s = t
            out[i] = s
        self.s, self.c = s, c
        return out


def _cast_tree(tree: Any, dtype) -> Any:
    return jax.tree_util.tree_map(
        lambda a: jnp.asarray(a, dtype=dtype) if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else a, tree)


def _resolve_models(model: EnergyModel, geometry: Optional[IndenterGeometry],
                    K: Optional[float]) -> tuple[EnergyModel, IndenterGeometry, float]:
    """(material model, geometry, K) from a composite model or a material model plus a geometry."""
    g = indenter_geometry_of(model)
    if g is not None:
        if geometry is not None and geometry != g:
            raise ValueError(f"model carries indenter {g.label}, but geometry={geometry.label} was given")
        mat_fn = material_energy_fn_of(model)
        K_model = float(np.asarray(model.params["indenter"]["K"]))
        suffix = f"+{g.label}"
        name = model.name[: -len(suffix)] if model.name.endswith(suffix) else model.name
        material = dataclasses.replace(model, name=name, energy_fn=mat_fn, params=model.params["material"])
        return material, g, float(K if K is not None else K_model)
    if geometry is None:
        raise ValueError("pass a composite model (with_indenter) or a material model plus geometry=")
    return model, geometry, float(DEFAULT_STIFFNESS_EV_PER_A3 if K is None else K)


def _with_slots(material: EnergyModel, k3: int) -> EnergyModel:
    from fracture_md.potentials.vashishta import with_three_body_slots

    return with_three_body_slots(material, k3)


def _k3_of(material: EnergyModel) -> Optional[int]:
    return getattr(material.energy_fn, "k3", None) if hasattr(material.energy_fn, "energy_and_status") else None


def _nbrs_arrays(nbrs) -> dict[str, np.ndarray]:
    return {"nbr_idx": np.asarray(nbrs.idx), "nbr_reference_position": np.asarray(nbrs.reference_position),
            "nbr_error_code": np.asarray(nbrs.error.code), "nbr_max_occupancy": np.asarray(nbrs.max_occupancy),
            "nbr_cell_list_capacity": np.asarray(-1 if nbrs.cell_list_capacity is None else nbrs.cell_list_capacity)}


def _restore_nbrs(template, arrs: Mapping[str, np.ndarray]):
    cap = int(arrs["nbr_cell_list_capacity"])
    return dataclasses.replace(
        template, idx=jnp.asarray(arrs["nbr_idx"]), reference_position=jnp.asarray(arrs["nbr_reference_position"]),
        error=partition.PartitionError(jnp.asarray(arrs["nbr_error_code"])),
        max_occupancy=int(arrs["nbr_max_occupancy"]), cell_list_capacity=None if cap < 0 else cap)


def _system_digest(system: AtomicSystem, species: Any) -> str:
    h = hashlib.sha256()
    for a in (np.asarray(system.frac, np.float64), np.asarray(system.box, np.float64), np.asarray(species, np.int64)):
        h.update(a.tobytes())
    return h.hexdigest()


def _continuation_key(seed: int, frac: Any, momenta: Any) -> jnp.ndarray:
    """Thermostat key of a run started from given momenta: the fresh-start key folded with a digest of the state.

    Staged runs (equilibration stages, then production; physics.indentation) share ``cfg.seed``; with the bare
    fresh-start key they would replay the same bath noise stage after stage.  Each stage starts from different
    positions and momenta, so the fold makes the streams independent while a given start stays reproducible.
    """
    h = hashlib.sha256()
    for a in (np.asarray(frac, np.float64), np.asarray(momenta, np.float64)):
        h.update(np.ascontiguousarray(a).tobytes())
    tag = int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF
    return jax.random.fold_in(jax.random.split(jax.random.PRNGKey(seed))[1], tag)


def _save_npz_atomic(path: Path, arrays: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in arrays.items()})
    os.replace(tmp, path)


# ============================================================================ frames
class FrameWriter:
    """Animation frames at chunk boundaries -> one ``indentviz-traj/1`` file (fracture_md.io.indent_traj).

    ``path``: the final trajectory file; frames are stored as shards in ``<path>.frames/`` while the run
    goes on (so they survive a checkpoint restart) and assembled by :meth:`finalize`.  ``region``: "slice"
    (a slice |offset normal to lab x through the tip axis| <= ``slice_half_width_A`` + ``halo_A`` plus
    every atom within ``near_surface_A`` of the initial surface, chosen ONCE from the first frame) or "all".
    Positions are float32, wrapped laterally; per-atom fields: ``contact`` (1 where the atom feels the wall),
    ``penetration_A`` (delta) and ``ke_eV``.  ``max_bytes`` bounds the projected uncompressed frame storage
    (frames x atoms x (12 + 4 x fields) bytes) and is checked before the first frame.
    """

    def __init__(self, path: Union[str, Path], *, every_ps: float = 1.0, region: str = "slice",
                 slice_half_width_A: float = 2.0, halo_A: float = 3.5, near_surface_A: float = 6.0,
                 fields: Sequence[str] = FRAME_FIELDS, max_frames: Optional[int] = None, max_bytes: float = 2.0e8,
                 curve_every_steps: int = 10, meta: Optional[dict] = None, keep_shards: bool = False):
        if region not in ("slice", "all"):
            raise ValueError("region must be 'slice' or 'all'")
        unknown = set(fields) - set(FRAME_FIELDS)
        if unknown:
            raise ValueError(f"unknown frame fields {sorted(unknown)}; available {FRAME_FIELDS}")
        if not every_ps > 0 or curve_every_steps < 1:
            raise ValueError("every_ps > 0 and curve_every_steps >= 1 required")
        self.path = Path(path)
        self.shard_dir = self.path.with_name(self.path.name + ".frames")
        self.every_ps = float(every_ps)
        self.region = region
        self.slice_half_width_A, self.halo_A, self.near_surface_A = float(slice_half_width_A), float(halo_A), float(
            near_surface_A)
        self.fields = tuple(fields)
        self.max_frames = max_frames
        self.max_bytes = float(max_bytes)
        self.curve_every_steps = int(curve_every_steps)
        self.meta = dict(meta or {})
        self.keep_shards = keep_shards
        self.keep: Optional[np.ndarray] = None
        self.static: dict[str, Any] = {}
        self.n_frames = 0
        self.last_t_ps = -math.inf

    # ------------------------------------------------------------------ setup / resume
    @property
    def bytes_per_frame(self) -> int:
        n = 0 if self.keep is None else int(self.keep.sum())
        return n * (12 + 4 * len(self.fields))

    def projected_bytes(self, n_frames: int) -> int:
        return int(n_frames) * self.bytes_per_frame

    def setup(self, *, species_symbols: Sequence[str], box: np.ndarray, X0: np.ndarray, surface_z0_A: float,
              apex_xy: Sequence[float], geometry: IndenterGeometry, expected_frames: int, meta: dict,
              resume: bool = False) -> None:
        """Choose the atom subset once (or reload it on resume) and check the storage budget."""
        box = np.asarray(box, float)
        lo, hi = np.zeros(3), np.diag(box).copy()
        static_path = self.shard_dir / "static.npz"
        if resume and static_path.exists():
            with np.load(static_path, allow_pickle=False) as z:
                self.keep = np.asarray(z["keep"], bool)
        else:
            if self.region == "all":
                keep = np.ones(len(species_symbols), bool)
            else:
                from fracture_md.io.indent_traj import animation_subset_mask

                keep = animation_subset_mask(np.asarray(X0), lo, hi, (True, True, False), apex_xy, surface_z0_A,
                                             slice_half_width_A=self.slice_half_width_A, halo_A=self.halo_A,
                                             near_surface_A=self.near_surface_A)
            self.keep = keep
            if self.shard_dir.exists():
                shutil.rmtree(self.shard_dir)
            self.shard_dir.mkdir(parents=True)
            np.savez(static_path, keep=keep)
        n_exp = expected_frames if self.max_frames is None else min(expected_frames, self.max_frames)
        if self.projected_bytes(n_exp) > self.max_bytes:
            raise ValueError(f"frame storage {self.projected_bytes(n_exp) / 1e6:.1f} MB for {n_exp} frames x "
                             f"{int(self.keep.sum())} atoms exceeds max_bytes = {self.max_bytes / 1e6:.1f} MB; use "
                             "region='slice', a larger every_ps or max_frames")
        sub = self.region == "slice"
        self.static = dict(
            species=np.asarray(species_symbols)[self.keep], box_lo=lo, box_hi=hi, surface_z0_A=float(surface_z0_A),
            indenter=geometry.to_dict(), atom_ids=np.flatnonzero(self.keep),
            meta={**meta, **self.meta, "n_atoms_total": int(len(species_symbols)), "contact_field": "contact",
                  "frame_fields": list(self.fields),
                  **({"subset": f"slice |y - y_tip| <= {self.slice_half_width_A:g} + {self.halo_A:g} A halo "
                                f"through the tip axis plus atoms within {self.near_surface_A:g} A of the initial "
                                "surface (chosen at the first frame)"} if sub else {})},
        )

    def resume(self, t_ps: float) -> None:
        """Drop frames written after the checkpoint time ``t_ps`` (they are recomputed after a restart)."""
        shards = sorted(self.shard_dir.glob("frame_*.npz"))
        kept = []
        for f in shards:
            with np.load(f, allow_pickle=False) as z:
                t = float(z["time_ps"])
            if t > t_ps + 1e-9:
                f.unlink()
            else:
                kept.append(t)
        self.n_frames = len(kept)
        self.last_t_ps = max(kept) if kept else -math.inf

    # ------------------------------------------------------------------ writing
    def due(self, t_ps: float) -> bool:
        if self.max_frames is not None and self.n_frames >= self.max_frames:
            return False
        if self.n_frames == 0:
            return True
        return t_ps >= self.last_t_ps + self.every_ps - 1e-6

    def write(self, *, t_ps: float, step: int, X: np.ndarray, tip_xyz: Sequence[float], load_nN: float,
              fields: Mapping[str, np.ndarray]) -> Path:
        """Write one frame shard (``X``: Cartesian positions of ALL atoms; the subset is taken here)."""
        if self.keep is None:
            raise RuntimeError("FrameWriter.setup must be called before write")
        X = np.asarray(X, float)[self.keep]
        L = self.static["box_hi"][:2]
        X[:, :2] = np.mod(X[:, :2], L)
        X[:, :2] = np.where(X[:, :2] >= L, 0.0, X[:, :2])
        arrs = {"time_ps": np.float64(t_ps), "step": np.int64(step), "positions": X.astype(np.float32),
                "tip_xyz": np.asarray(tip_xyz, float), "load_nN": np.float64(load_nN)}
        for k in self.fields:
            arrs[f"field__{k}"] = np.asarray(fields[k], np.float32)[self.keep]
        path = self.shard_dir / f"frame_{self.n_frames:06d}.npz"
        np.savez_compressed(path, **arrs)
        self.n_frames += 1
        self.last_t_ps = float(t_ps)
        return path

    def storage_bytes(self) -> int:
        """Bytes currently on disk (shards and, after finalize, the trajectory file)."""
        n = sum(f.stat().st_size for f in self.shard_dir.glob("*.npz")) if self.shard_dir.exists() else 0
        return n + (self.path.stat().st_size if self.path.exists() else 0)

    def finalize(self, curve_t_ps: np.ndarray, curve_depth_A: np.ndarray, curve_load_nN: np.ndarray) -> Optional[Path]:
        """Assemble the shards into one ``indentviz-traj/1`` file with the dense load-depth record."""
        from fracture_md.io.indent_traj import IndentTrajWriter

        shards = sorted(self.shard_dir.glob("frame_*.npz"))
        if not shards:
            return None
        s = self.static
        w = IndentTrajWriter(self.path, species=s["species"], box_lo=s["box_lo"], box_hi=s["box_hi"],
                             pbc=(True, True, False), surface_z0_A=s["surface_z0_A"], indenter=s["indenter"],
                             meta=s["meta"], atom_ids=s["atom_ids"])
        for f in shards:
            with np.load(f, allow_pickle=False) as z:
                w.add_frame(float(z["time_ps"]), z["positions"], z["tip_xyz"], float(z["load_nN"]),
                            atom_fields={k: z[f"field__{k}"] for k in self.fields})
        w.set_curve(curve_t_ps, curve_depth_A, curve_load_nN)
        path = w.close()
        if not self.keep_shards:
            shutil.rmtree(self.shard_dir, ignore_errors=True)
        return path


# ============================================================================ result
@dataclasses.dataclass
class IndentMDResult:
    """Outcome of :func:`run_indentation_md`.

    ``logs``: per stored step (every ``log_every_steps``, plus the steps at the schedule knots and chunk ends)
    numpy arrays: step, t_ps, apex_z_A (the kernel's apex, in the run's dtype), depth_A (z_s0 minus the float64
    schedule's apex: the commanded depth, exact also in float32), load_eV_per_A (n, 3), load_nN (+z), e_mat_eV,
    e_ind_eV, e_kin_eV, e_total_eV, w_step_eV, q_step_eV, W_eV (cumulative tip work), Q_eV (cumulative heat
    ADDED by the bath, < 0 when it cools), ledger_eV (= e_total - W - Q - e_total(0)), pen_fw_A, n_contact,
    max_penetration_A, ke_<region>_eV and T_<region>_K for thermostat, nve, analysis (if given) and core
    (T_core_K: the contact-zone cylinder of :func:`core_region_mask`; the NVE region only if that is empty),
    rebuilt, three_body_max_partners, segment (a sample on a knot belongs to the segment ending there: the
    deepest sample is "load"), and the Newton-3 channels: reaction_eV_per_A (n, 3; the pre-zero total force on
    the fixed atoms, LAMMPS's fix-setforce sum) and reaction_nN (its z component; -load in equilibrium),
    momentum_amu_A_per_tau (n, 3; total momentum of the mobile atoms), and the running integrals from step 0 over
    every step (whatever ``log_every_steps``): impulse_load_reaction_amu_A_per_tau (n, 3; trapezoid of load +
    reaction), impulse_load_amu_A_per_tau (n, 3; trapezoid of the load: the tip impulse) and
    bath_momentum_kick_amu_A_per_tau (n, 3; sum of the thermostat's momentum kicks).  ``initial``: the same
    instantaneous quantities at step 0.
    """

    logs: dict[str, np.ndarray]
    initial: dict[str, Any]
    frac: np.ndarray
    momenta: np.ndarray
    force: np.ndarray
    completed: bool
    diagnostics: dict[str, Any]
    config: dict[str, Any]
    schedule: dict[str, Any]
    surface_z0_A: float
    trajectory_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None

    def ledger_summary(self) -> dict[str, float]:
        """max |ledger| (eV), final W and Q (eV), max |ledger| / max |W| and the drift per atom."""
        led = self.logs["ledger_eV"]
        W = self.logs["W_eV"]
        mx = float(np.max(np.abs(led))) if led.size else 0.0
        wmax = float(np.max(np.abs(W))) if W.size else 0.0
        return {"max_abs_ledger_eV": mx, "W_final_eV": float(W[-1]) if W.size else 0.0,
                "Q_final_eV": float(self.logs["Q_eV"][-1]) if W.size else 0.0,
                "max_abs_ledger_over_max_abs_W": mx / wmax if wmax > 0 else float("inf"),
                "max_abs_ledger_per_atom_eV": mx / self.frac.shape[0]}

    def momentum_balance(self) -> dict[str, Any]:
        """Newton 3 through the slab: the momentum identity of velocity Verlet at every logged sample.

        The force on the mobile atoms is -(load + reaction) (the material forces sum to zero, the tip forces to
        -load), so p(t) - p(0) + int_0^t (load + reaction) dt - (bath kicks up to t) = 0 for the total momentum p of
        the mobile atoms; the driver integrates over every step (trapezoid), so this holds at any
        ``log_every_steps``.  Returns (momenta and impulses in amu A / tau):

        * ``max_abs_residual`` (3,) and ``residual_over_tip_impulse``: the identity's residual, round-off for a
          correct and complete reaction channel (a missing fixed atom or a sign error gives O(1));
        * ``tip_impulse_z``: int load_z dt at the last sample, and ``max_abs_tip_impulse_z``;
        * ``max_abs_slab_momentum_z``: max |p_z(t) - p_z(0) - kicks_z(t)|, the part of the tip's impulse still in
          the slab rather than in the fixed layer, and ``slab_momentum_over_tip_impulse`` = that over
          ``max_abs_tip_impulse_z`` (the physical Newton-3 statement of recommended protocol item 15, 2 %: small for
          a slow tip at 0 K; at 300 K it includes the thermal momentum of the mobile atoms).
        """
        lg = self.logs
        if any(k not in lg for k in _NEWTON3_LOGS) or any(k not in self.initial for k in _NEWTON3_INITIAL):
            raise ValueError("this result has no momentum-balance channels (a run of an older driver)")
        p = np.asarray(lg["momentum_amu_A_per_tau"], np.float64)
        p0 = np.asarray(self.initial["momentum_amu_A_per_tau"], np.float64)
        J = np.asarray(lg["impulse_load_reaction_amu_A_per_tau"], np.float64)
        kick = np.asarray(lg["bath_momentum_kick_amu_A_per_tau"], np.float64)
        tip = np.asarray(lg["impulse_load_amu_A_per_tau"], np.float64)[:, 2]
        res = p - p0 + J - kick
        slab = p[:, 2] - p0[2] - kick[:, 2]
        tip_max = float(np.max(np.abs(tip))) if tip.size else 0.0
        res_max = np.max(np.abs(res), axis=0) if res.size else np.zeros(3)
        slab_max = float(np.max(np.abs(slab))) if slab.size else 0.0
        return {"max_abs_residual": res_max,
                "residual_over_tip_impulse": float(np.max(res_max)) / tip_max if tip_max > 0 else float("inf"),
                "tip_impulse_z": float(tip[-1]) if tip.size else 0.0, "max_abs_tip_impulse_z": tip_max,
                "max_abs_slab_momentum_z": slab_max,
                "slab_momentum_over_tip_impulse": slab_max / tip_max if tip_max > 0 else float("inf"),
                "n_samples": int(p.shape[0])}

    def summary(self) -> dict[str, Any]:
        lg = self.logs
        return {"completed": self.completed, "n_steps": int(lg["step"][-1]) if lg["step"].size else 0,
                "max_load_nN": float(np.max(lg["load_nN"])) if lg["load_nN"].size else 0.0,
                "max_depth_A": float(np.max(lg["depth_A"])) if lg["depth_A"].size else 0.0,
                "ledger": self.ledger_summary(), "diagnostics": self.diagnostics,
                "trajectory": None if self.trajectory_path is None else str(self.trajectory_path)}


# ============================================================================ driver
def run_indentation_md(
    model: EnergyModel,
    system: AtomicSystem,
    schedule: TipSchedule,
    *,
    masks: Mapping[str, Any],
    cfg: IndentMDConfig = IndentMDConfig(),
    geometry: Optional[IndenterGeometry] = None,
    K_eV_per_A3: Optional[float] = None,
    thermostat_mask: Optional[Any] = None,
    momenta: Optional[Any] = None,
    surface_z0_A: Optional[float] = None,
    neighbors: Optional[NeighborHandle] = None,
    frame_writer: Optional[FrameWriter] = None,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    restart: Union[bool, str] = "auto",
    stop_after_chunks: Optional[int] = None,
    max_wall_s: Optional[float] = None,
    progress: Optional[Callable[[dict], None]] = None,
    chunk_callback: Optional[Callable[[IndentMDState, NeighborHandle], None]] = None,
    meta: Optional[dict] = None,
) -> IndentMDResult:
    """Run displacement-controlled indentation MD (see the module docstring for the model and integrator).

    Parameters
    ----------
    model : a composite model (``physics.indenter.with_indenter``; geometry and K are taken from it) or a
        material model plus ``geometry`` (and ``K_eV_per_A3``, default 100).  Pass the model the protocol
        runner received, so gate perturbations of the material carry over.
    system : the slab (orthorhombic cell, a3 along z, vacuum above; structures.indentation).
    schedule : the apex path (:class:`TipSchedule`); the run covers it in whole chunks (a short final
        hold pads the last chunk).
    masks : region masks, at least ``fixed`` and ``bath`` (``slab.masks()``); optional ``analysis``,
        ``core`` (the T_core_K region; default :func:`core_region_mask` with ``cfg.core_radius_A`` and
        ``cfg.core_depth_A`` about the tip axis) and ``top_layer`` (the depth reference: z_s0 = its mean
        height unless ``surface_z0_A`` is given).
    thermostat_mask : override of the thermostatted atoms (e.g. every free atom during equilibration).
    momenta : initial momenta (amu A / tau); default :func:`region_momenta` at ``cfg.temperature_K`` for
        the thermostat and NVE regions (zero centre-of-mass momentum each), zero on fixed atoms.  With given
        momenta the thermostat key is ``cfg.seed``'s folded with a digest of the starting state, so the stages
        of a staged run (same seed) draw independent bath noise.
    neighbors : a pre-built handle (default: ``build_neighbors`` with the config's skin and capacity).
    frame_writer : optional :class:`FrameWriter` (frames at chunk boundaries).
    checkpoint_dir : directory for checkpoints (``indent_md_checkpoint.npz``); ``restart`` = "auto"
        resumes from an existing checkpoint there, True requires one, False ignores it.
    stop_after_chunks, max_wall_s : stop early after that many chunks of this call / that much wall time,
        writing a checkpoint first (a later call with the same arguments resumes).
    progress : called after every chunk with a small dict (step, t_ps, depth, load, T_core, ledger, rate).
    chunk_callback : called after every accepted chunk with the device state and the neighbour handle
        (analysis hooks, e.g. neighbour-list completeness checks); must not modify them.
    meta : extra provenance written into the trajectory file (e.g. run_id, potential plain name).
    """
    t_wall0 = time.monotonic()
    material, geometry, K = _resolve_models(model, geometry, K_eV_per_A3)
    dtype = jnp.float64 if cfg.dtype == "float64" else jnp.float32
    n = system.n_atoms
    box64 = np.asarray(system.box, np.float64)
    check_indenter_cell(box64, geometry, (schedule.xy_A[0], schedule.xy_A[1], schedule.max_apex_z_A))
    reg = _region_masks(n, masks, thermostat_mask)
    free = ~reg["fixed"]
    X0 = np.asarray(system.frac, np.float64) @ box64.T
    if masks.get("core") is not None:
        core = np.asarray(masks["core"], bool).reshape(-1) & free
        core_def = "masks['core']"
    else:
        core = core_region_mask(X0, box64, reg["nve"], schedule.xy_A, radius_A=cfg.core_radius_A,
                                depth_A=cfg.core_depth_A)
        core_def = (f"NVE atoms within {cfg.core_radius_A:g} A of the tip axis"
                    + ("" if cfg.core_depth_A is None else f" and {cfg.core_depth_A:g} A of the highest atom"))
    if core.any():
        reg["core"] = core
    else:  # e.g. an all-thermostatted equilibration stage: no NVE atoms, T_core_K falls back to T_nve_K
        core_def = f"empty ({core_def}); T_core_K = T_nve_K"
    region_names = [k for k in ("thermostat", "nve", "analysis", "core") if k in reg]
    region_arr = np.stack([reg[k] for k in region_names])
    species = material.species_for(system)
    mass64 = np.asarray(system.masses_amu, np.float64)
    if surface_z0_A is None:
        if masks.get("top_layer") is not None:
            surface_z0_A = float(np.mean(X0[np.asarray(masks["top_layer"], bool), 2]))
        else:
            from fracture_md.io.indent_traj import estimate_surface_z

            surface_z0_A = estimate_surface_z(X0)
    n_total = schedule.n_steps(cfg.dt_fs, cfg.chunk_steps)
    n_chunks_total = n_total // cfg.chunk_steps
    ckdir = None if checkpoint_dir is None else Path(checkpoint_dir)
    ckpath = None if ckdir is None else ckdir / _CHECKPOINT_NAME
    card_id = material.card.id
    digest = _system_digest(system, species)
    ident = {"config": cfg.physics_dict(), "schedule": schedule.to_dict(), "system_sha256": digest,
             "card_id": card_id, "geometry": geometry.to_dict(), "K_eV_per_A3": K,
             "regions": {k: hashlib.sha256(np.packbits(v).tobytes()).hexdigest() for k, v in reg.items()}}

    # ---------------------------------------------------------------- restart or fresh start
    ck = None
    if ckpath is not None and restart and ckpath.exists():
        with np.load(ckpath, allow_pickle=False) as z:
            ck = {k: np.asarray(z[k]) for k in z.files}
        if str(ck["format"]) != CHECKPOINT_FORMAT:
            raise ValueError(f"{ckpath}: not an indentation MD checkpoint")
        saved = json.loads(str(ck["ident_json"]))
        if saved != json.loads(json.dumps(ident, default=_json_default)):
            diff = sorted(k for k in ident if json.dumps(saved.get(k), default=_json_default, sort_keys=True)
                          != json.dumps(ident[k], default=_json_default, sort_keys=True))
            raise ValueError(f"checkpoint {ckpath} belongs to a different run (differs in {diff})")
    elif restart is True:
        raise FileNotFoundError(f"restart=True but no checkpoint at {ckpath}")

    extra_cap = int(cfg.extra_capacity) if ck is None else int(ck["extra_capacity"])
    if ck is not None and "k3" in ck and int(ck["k3"]) > 0 and _k3_of(material) != int(ck["k3"]):
        material = _with_slots(material, int(ck["k3"]))
    S0 = jnp.asarray(system.frac if ck is None else ck["frac"], dtype=dtype)
    if neighbors is None:
        neighbors = build_neighbors(system.replace(frac=jnp.asarray(S0, jnp.float64)), material.cutoff,
                                    skin=cfg.skin_A, capacity_multiplier=cfg.capacity_multiplier,
                                    method=cfg.neighbor_method, extra_capacity=extra_cap)  # type: ignore[arg-type]
    handle = neighbors
    if ck is not None:
        handle = dataclasses.replace(handle, nbrs=_restore_nbrs(handle.nbrs, ck))

    consts = {
        "box": jnp.asarray(box64, dtype), "binv": jnp.asarray(np.linalg.inv(box64), dtype), "species": species,
        "params": _cast_tree(material.params, dtype), "K": jnp.asarray(K, dtype),
        "mass": jnp.asarray(mass64[:, None], dtype), "free": jnp.asarray(free[:, None]),
        "thermo": jnp.asarray(reg["thermostat"][:, None]), "regions": jnp.asarray(region_arr),
        "kT": jnp.asarray(KB_EV_PER_K * cfg.temperature_K, dtype), "dt": jnp.asarray(cfg.dt_fs / TIME_UNIT_FS, dtype),
        "dt_fs": jnp.asarray(cfg.dt_fs, jnp.float64),
        "c_thermo": jnp.asarray(math.exp(-cfg.dt_fs / (1e3 * cfg.thermostat_tau_ps)), dtype),
        "dof_thermo": jnp.asarray(max(3.0 * float(reg["thermostat"].sum()), 2.0), dtype),
        **schedule.arrays(dtype),
    }
    if cfg.thermostat == "csvr" and reg["thermostat"].sum() == 0:
        raise ValueError("thermostat 'csvr' needs at least one thermostatted atom (masks['bath'])")

    with_status = hasattr(material.energy_fn, "energy_and_status")
    k3_history = [] if _k3_of(material) is None else [int(_k3_of(material))]  # type: ignore[arg-type]
    aux = _aux_kernels(material.energy_fn, geometry, with_status)

    def upgrade_slots(n3max: int) -> None:
        nonlocal material, aux
        new_k3 = int(n3max) + int(cfg.three_body_headroom)
        material = _with_slots(material, new_k3)
        consts["params"] = _cast_tree(material.params, dtype)
        aux = _aux_kernels(material.energy_fn, geometry, with_status)
        k3_history.append(new_k3)

    step_fs = cfg.dt_fs
    if ck is None:
        if momenta is None:
            key_m, key_t = jax.random.split(jax.random.PRNGKey(cfg.seed))
            P0 = region_momenta(mass64, [reg["thermostat"], reg["nve"]], cfg.temperature_K, key=key_m)
        else:
            P0 = jnp.where(jnp.asarray(free[:, None]), jnp.asarray(momenta, jnp.float64), 0.0)
            key_t = _continuation_key(cfg.seed, system.frac, P0)
        P0 = jnp.asarray(P0, dtype)
        apex0 = jnp.asarray(schedule.apex(0.0), dtype)
        for _ in range(4):
            init = aux["init"](S0, consts["box"], species, handle.idx, consts["params"], apex0, consts["K"],
                               consts["free"])
            if with_status and int(init["n3"]) > int(_k3_of(material)):  # type: ignore[arg-type]
                upgrade_slots(int(init["n3"]))
                continue
            break
        state = IndentMDState(S0, P0, init["force"], init["load"], key_t, handle.nbrs, jnp.asarray(0, jnp.int64))
        ke0 = float(0.5 * np.sum(np.asarray(P0, np.float64) ** 2 / mass64[:, None]))
        initial = {
            "e_mat_eV": float(init["e_mat"]), "e_ind_eV": float(init["e_ind"]), "e_kin_eV": ke0,
            "e_total_eV": float(init["e_mat"]) + float(init["e_ind"]) + ke0,
            "load_eV_per_A": np.asarray(init["load"], np.float64), "n_contact": int(init["n_contact"]),
            "apex_z_A": float(apex0[2]), "three_body_max_partners": int(init["n3"]),
            "reaction_eV_per_A": np.asarray(init["reaction"], np.float64),
            "momentum_amu_A_per_tau": np.sum(np.asarray(P0, np.float64), axis=0),
        }
        kW, kQ = _Kahan(), _Kahan()
        logs_acc: dict[str, list] = {}
        counters = {"rollbacks": [], "rebuilds": 0, "chunks_done": 0, "wall_s_previous": 0.0}
    else:
        state = IndentMDState(S0, jnp.asarray(ck["momentum"], dtype), jnp.asarray(ck["force"], dtype),
                              jnp.asarray(ck["load"], dtype), jnp.asarray(ck["key"]), handle.nbrs,
                              jnp.asarray(int(ck["step"]), jnp.int64))
        initial = json.loads(str(ck["initial_json"]))
        logs_acc = {k[len("log__"):]: [ck[k]] for k in ck if k.startswith("log__")}
        missing = sorted({k for k in _NEWTON3_LOGS if k not in logs_acc or not np.size(logs_acc[k][0])}
                         | {k for k in _NEWTON3_INITIAL if k not in initial})
        if missing:  # a checkpoint of a driver without the reaction channels: never a run with a hole in them
            raise ValueError(f"checkpoint {ckpath} has no {missing} (written before the fixed-layer reaction was "
                             "logged); restart the run from scratch (restart=False)")
        for k in _NEWTON3_INITIAL:
            initial[k] = np.asarray(initial[k], np.float64)
        kW, kQ = _Kahan(*ck["kahan_W"]), _Kahan(*ck["kahan_Q"])
        counters = json.loads(str(ck["counters_json"]))
        k3_history = list(counters.get("k3_history", k3_history))
    e_total0 = float(initial["e_total_eV"])
    # running integrals of the momentum balance, continued from the last stored sample (every chunk end is stored)
    if ck is None:
        newton = {"load": initial["load_eV_per_A"], "f": initial["load_eV_per_A"] + initial["reaction_eV_per_A"],
                  "J": np.zeros(3), "J_tip": np.zeros(3), "kick": np.zeros(3)}
    else:
        last = {k: np.asarray(logs_acc[k][-1], np.float64)[-1] for k in ("load_eV_per_A", *_NEWTON3_LOGS)}
        newton = {"load": last["load_eV_per_A"], "f": last["load_eV_per_A"] + last["reaction_eV_per_A"],
                  "J": last["impulse_load_reaction_amu_A_per_tau"], "J_tip": last["impulse_load_amu_A_per_tau"],
                  "kick": last["bath_momentum_kick_amu_A_per_tau"]}
    dt_int = cfg.dt_fs / TIME_UNIT_FS  # the time step in tau (impulses in amu A / tau)

    # ---------------------------------------------------------------- frames
    fw = frame_writer
    if fw is not None:
        sym = np.asarray(material.elements)[np.asarray(species)]
        X_start = np.asarray(aux["frame"](jnp.asarray(system.frac, dtype), consts["box"],
                                          jnp.asarray(schedule.apex(0.0), dtype))[0], np.float64)
        n_frames_exp = int(math.floor(n_total * cfg.dt_fs * 1e-3 / max(fw.every_ps, cfg.chunk_fs * 1e-3))) + 1
        fmeta = {"engine": "fracture-md JAX harness (physics/indent_md.py)", "potential": material.card.name,
                 "potential_id": material.name, "potential_card_id": card_id,
                 "temperature_K": cfg.temperature_K,
                 "tip_model": f"rigid SDF indenter {geometry.label}, K = {K:g} eV/A^3",
                 "dt_fs": cfg.dt_fs, "thermostat": f"{cfg.thermostat} (tau {cfg.thermostat_tau_ps:g} ps) on the bath",
                 "n_atoms_total": n, "config": cfg.to_dict(), "schedule": schedule.to_dict(), "K_eV_per_A3": K,
                 **({"tip_speed_m_s": _schedule_speed_m_s(schedule)} if _schedule_speed_m_s(schedule) else {}),
                 **(meta or {})}
        fw.setup(species_symbols=sym, box=box64, X0=X_start, surface_z0_A=surface_z0_A, apex_xy=schedule.xy_A,
                 geometry=geometry, expected_frames=n_frames_exp, meta=fmeta, resume=ck is not None)
        if ck is not None:
            fw.resume(int(ck["step"]) * step_fs * 1e-3)

    def write_frame(st: IndentMDState, load_z_eV_per_A: float) -> None:
        stp = int(st.step)
        t_ps = stp * step_fs * 1e-3
        if fw is None or not fw.due(t_ps):
            return
        apex = np.asarray(schedule.apex(stp * step_fs))
        X, delta = aux["frame"](st.frac, consts["box"], jnp.asarray(apex, dtype))
        P = np.asarray(st.momentum, np.float64)
        fields = {"contact": (np.asarray(delta) > 0).astype(np.float32), "penetration_A": np.asarray(delta),
                  "ke_eV": 0.5 * np.sum(P**2, axis=1) / mass64}
        fw.write(t_ps=t_ps, step=stp, X=np.asarray(X, np.float64), tip_xyz=apex,
                 load_nN=load_z_eV_per_A * EV_PER_A_TO_NN, fields=fields)

    if fw is not None and int(state.step) == 0:
        write_frame(state, float(initial["load_eV_per_A"][2]))

    # ---------------------------------------------------------------- chunk loop
    def get_kernel():
        return make_chunk_kernel(material.energy_fn, geometry, cfg.chunk_steps, neighbor_fn=handle.neighbor_fn,
                                 thermostat=cfg.thermostat, n_regions=len(region_names), with_status=with_status)

    kernel = get_kernel()
    cache0 = {id(kernel): kernel._cache_size()}
    kernels_used = {id(kernel): kernel}
    last_ckpt = time.monotonic()
    chunks_this_call = 0
    dof_reg = {k: 3.0 * float(reg[k].sum()) for k in region_names}
    knot_steps = schedule.knot_steps(cfg.dt_fs)
    completed = True
    t_loop0 = time.monotonic()
    steps_this_call = 0

    def save_checkpoint() -> None:
        if ckpath is None:
            return
        ckpath.parent.mkdir(parents=True, exist_ok=True)
        cnt = dict(counters, k3_history=k3_history,
                   wall_s_previous=counters["wall_s_previous"] + (time.monotonic() - t_wall0))
        arrs = {"format": np.array(CHECKPOINT_FORMAT), "ident_json": np.array(json.dumps(ident, default=_json_default)),
                "frac": np.asarray(state.frac), "momentum": np.asarray(state.momentum),
                "force": np.asarray(state.force), "load": np.asarray(state.load), "key": np.asarray(state.key),
                "step": np.asarray(int(state.step)),
                "extra_capacity": np.asarray(extra_cap), "k3": np.asarray(_k3_of(material) or 0),
                "kahan_W": np.array([kW.s, kW.c]), "kahan_Q": np.array([kQ.s, kQ.c]),
                "initial_json": np.array(json.dumps(initial, default=_json_default)),
                "counters_json": np.array(json.dumps(cnt, default=_json_default)), **_nbrs_arrays(state.nbrs)}
        for k, v in logs_acc.items():
            arrs[f"log__{k}"] = np.concatenate(v) if v else np.zeros(0)
        _save_npz_atomic(ckpath, arrs)

    while int(state.step) < n_total:
        start = state
        attempt = 0
        while True:
            new, lg = kernel(start, consts)
            ovf = bool(overflow_bits(new.nbrs))
            n3max = int(np.max(np.asarray(lg["n3"])))
            k3_ovf = with_status and n3max > int(_k3_of(material))  # type: ignore[arg-type]
            finite = bool(np.all(np.asarray(lg["finite"])))
            if not (ovf or k3_ovf):
                if not finite:
                    bad = int(np.asarray(lg["step"])[~np.asarray(lg["finite"])][0])
                    raise FloatingPointError(f"non-finite energy at step {bad} (t = {bad * step_fs * 1e-3:.4f} ps) "
                                             "without a capacity overflow: time step too large or atoms overlap")
                break
            if attempt >= cfg.max_rollbacks_per_chunk:
                raise RuntimeError(f"chunk at step {int(start.step)} still overflows after {attempt} rollbacks "
                                   f"(neighbour overflow {ovf}, three-body {k3_ovf}, max partners {n3max})")
            attempt += 1
            reason = []
            if ovf:
                extra_cap += int(cfg.capacity_headroom)
                handle = handle.reallocate(jnp.asarray(start.frac, jnp.float64), extra_capacity=extra_cap)
                start = start._replace(nbrs=handle.nbrs)
                reason.append(f"neighbour overflow -> capacity {handle.capacity} (extra {extra_cap})")
            if k3_ovf:
                upgrade_slots(n3max)
                reason.append(f"three-body slots -> {k3_history[-1]} (max partners {n3max})")
            counters["rollbacks"].append({"step": int(start.step), "reason": "; ".join(reason)})
            kernel = get_kernel()
            if id(kernel) not in kernels_used:
                kernels_used[id(kernel)] = kernel
                cache0.setdefault(id(kernel), kernel._cache_size())
        state = new
        chunks_this_call += 1
        counters["chunks_done"] += 1
        steps_this_call += cfg.chunk_steps

        # ---- host bookkeeping in float64
        h = {k: np.asarray(v) for k, v in lg.items()}
        W = kW.add_series(h["w"])
        Q = kQ.add_series(h["q"])
        e_mat, e_ind, ke = (h[k].astype(np.float64) for k in ("e_mat", "e_ind", "e_kin"))
        etot = e_mat + e_ind + ke
        s2, s3 = h["s2"].astype(np.float64), h["s3"].astype(np.float64)
        steps = h["step"].astype(np.int64)
        # momentum balance over every step of the chunk (trapezoids continue from the previous chunk's last step)
        load64, reac64 = h["load"].astype(np.float64), h["reaction"].astype(np.float64)
        f64 = load64 + reac64
        J = newton["J"] + np.cumsum(0.5 * dt_int * (np.vstack([newton["f"][None], f64[:-1]]) + f64), axis=0)
        J_tip = newton["J_tip"] + np.cumsum(0.5 * dt_int * (np.vstack([newton["load"][None], load64[:-1]]) + load64),
                                            axis=0)
        kick = newton["kick"] + np.cumsum(h["dp_bath"].astype(np.float64), axis=0)
        newton = {"load": load64[-1], "f": f64[-1], "J": J[-1], "J_tip": J_tip[-1], "kick": kick[-1]}
        rec = {
            "step": steps, "t_ps": steps * step_fs * 1e-3, "apex_z_A": h["apex_z"].astype(np.float64),
            # the commanded depth from the float64 schedule: a float32 apex_z is off by up to ~4e-6 A, which put the
            # deepest sample below max_depth_A half the time (and floor(2 h)/2 half an angstrom lower)
            "depth_A": surface_z0_A - schedule.apex_z(steps * step_fs), "load_eV_per_A": h["load"].astype(np.float64),
            "load_nN": h["load"][:, 2].astype(np.float64) * EV_PER_A_TO_NN, "e_mat_eV": e_mat, "e_ind_eV": e_ind,
            "e_kin_eV": ke, "e_total_eV": etot, "w_step_eV": h["w"].astype(np.float64),
            "q_step_eV": h["q"].astype(np.float64), "W_eV": W, "Q_eV": Q, "ledger_eV": etot - W - Q - e_total0,
            "pen_fw_A": np.where(s2 > 0, s3 / np.where(s2 > 0, s2, 1.0), 0.0),
            "n_contact": h["n_contact"].astype(np.int64), "max_penetration_A": h["max_pen"].astype(np.float64),
            "rebuilt": h["rebuilt"].astype(bool), "three_body_max_partners": h["n3"].astype(np.int64),
            "reaction_eV_per_A": reac64, "reaction_nN": reac64[:, 2] * EV_PER_A_TO_NN,
            "momentum_amu_A_per_tau": h["momentum"].astype(np.float64),
            "impulse_load_reaction_amu_A_per_tau": J, "impulse_load_amu_A_per_tau": J_tip,
            "bath_momentum_kick_amu_A_per_tau": kick,
        }
        for i, name in enumerate(region_names):
            kr = h["ke_regions"][:, i].astype(np.float64)
            rec[f"ke_{name}_eV"] = kr
            rec[f"T_{name}_K"] = 2.0 * kr / (max(dof_reg[name], 1.0) * KB_EV_PER_K)
        if "core" not in reg:
            rec["T_core_K"] = rec["T_nve_K"]
        counters["rebuilds"] += int(rec["rebuilt"].sum())
        sel = ((steps % cfg.log_every_steps) == 0) | np.isin(steps, knot_steps)  # knots: h_max, segment ends
        sel[-1] = True  # keep every chunk end (checkpoint / frame times)
        for k, v in rec.items():
            logs_acc.setdefault(k, []).append(v[sel])

        # ---- checks, frames, checkpoints
        if counters["chunks_done"] % cfg.z_guard_every_chunks == 0:  # in float64 (no TF32 on a float32 GPU run)
            apex_now = jnp.asarray(schedule.apex(int(state.step) * step_fs), jnp.float64)
            check_z_guard(geometry, jnp.asarray(state.frac, jnp.float64), jnp.asarray(box64),
                          {"apex": apex_now, "K": jnp.asarray(K, jnp.float64)})
        write_frame(state, float(rec["load_eV_per_A"][-1, 2]))
        if chunk_callback is not None:
            chunk_callback(state, handle)
        stop = (stop_after_chunks is not None and chunks_this_call >= stop_after_chunks) or \
            (max_wall_s is not None and time.monotonic() - t_wall0 >= max_wall_s)
        due = (cfg.checkpoint_every_chunks is not None and counters["chunks_done"] % cfg.checkpoint_every_chunks == 0) \
            or (time.monotonic() - last_ckpt >= cfg.checkpoint_every_s)
        if ckpath is not None and (due or stop or int(state.step) >= n_total):
            save_checkpoint()
            last_ckpt = time.monotonic()
        if progress is not None:
            el = time.monotonic() - t_loop0
            progress({"step": int(state.step), "n_steps": n_total, "t_ps": float(rec["t_ps"][-1]),
                      "segment": str(schedule.segment(float(rec["t_ps"][-1]) * 1e3)),
                      "depth_A": float(rec["depth_A"][-1]), "load_nN": float(rec["load_nN"][-1]),
                      "T_core_K": float(rec["T_core_K"][-1]), "ledger_eV": float(rec["ledger_eV"][-1]),
                      "W_eV": float(W[-1]), "Q_eV": float(Q[-1]), "n_contact": int(rec["n_contact"][-1]),
                      "capacity": handle.capacity, "rollbacks": len(counters["rollbacks"]),
                      "atom_steps_per_s": n * steps_this_call / el if el > 0 else float("nan"),
                      "frames": 0 if fw is None else fw.n_frames})
        if stop and int(state.step) < n_total:
            completed = False
            break

    # ---------------------------------------------------------------- assemble the result
    logs = {k: np.concatenate(v) if v else np.zeros(0) for k, v in logs_acc.items()}
    if logs.get("t_ps") is not None and logs["t_ps"].size:
        logs["segment"] = schedule.segment(logs["t_ps"] * 1e3)
    traj = None
    if fw is not None and completed:
        stride = max(1, fw.curve_every_steps // cfg.log_every_steps)
        traj = fw.finalize(logs["t_ps"][::stride], logs["depth_A"][::stride], logs["load_nN"][::stride])
    wall = time.monotonic() - t_wall0
    loop = time.monotonic() - t_loop0
    new_compiles = sum(k._cache_size() - cache0.get(i, 0) for i, k in kernels_used.items())
    diagnostics = {
        "n_atoms": n, "n_steps_total": n_total, "n_chunks_total": n_chunks_total, "chunk_steps": cfg.chunk_steps,
        "steps_done": int(state.step), "chunks_this_call": chunks_this_call, "rollbacks": counters["rollbacks"],
        "neighbor_rebuilds": counters["rebuilds"], "neighbor_list": handle.describe(), "extra_capacity": extra_cap,
        "three_body_slots_history": k3_history, "kernel_new_compiles": int(new_compiles),
        "kernel_cache_entries": [int(k._cache_size()) for k in kernels_used.values()],
        "wall_s": wall, "loop_wall_s": loop, "atom_steps_per_s": n * steps_this_call / loop if loop > 0 else None,
        "regions": {k: int(v.sum()) for k, v in reg.items()}, "core_region": core_def, "surface_z0_A": surface_z0_A,
        "precision": cfg.dtype, "backend": jax.default_backend(),
        "frames_written": 0 if fw is None else fw.n_frames,
        "frame_storage_bytes": 0 if fw is None else fw.storage_bytes(),
    }
    return IndentMDResult(
        logs=logs, initial=initial, frac=np.asarray(state.frac), momenta=np.asarray(state.momentum),
        force=np.asarray(state.force), completed=completed, diagnostics=diagnostics, config=cfg.to_dict(),
        schedule=schedule.to_dict(), surface_z0_A=float(surface_z0_A), trajectory_path=traj, checkpoint_path=ckpath,
    )


def _schedule_speed_m_s(schedule: TipSchedule) -> Optional[float]:
    """Speed of the first moving segment (m/s), for captions."""
    t, z = np.asarray(schedule.t_knots_fs), np.asarray(schedule.z_knots_A)
    for i in range(len(t) - 1):
        dz = z[i + 1] - z[i]
        if dz != 0.0:
            return float(abs(dz) / (t[i + 1] - t[i]) / M_PER_S_TO_A_PER_FS)
    return None


def _json_default(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return str(x)
