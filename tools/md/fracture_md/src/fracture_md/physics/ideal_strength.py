"""Ideal (theoretical) tensile strength under uniaxial stress, 0 K lattice statics.

Protocol
--------
Starting from the zero-stress relaxed crystal (box0), the cell is stretched
along a crystallographic direction n in steps of the *engineering* axial strain
ε = |F n| - 1 (F = box @ box0⁻¹).  At every step

  * the five other Cauchy-stress components (in a lab frame whose z axis is n)
    are driven to zero by quasi-Newton iterations on the corresponding lab
    strain components, using the relaxed-ion tangent stiffness (exact autodiff
    Hessians, see ``physics.elastic``) as the Jacobian — "uniaxial stress";
  * internal coordinates are relaxed at fixed cell (force-converged);
  * the neighbour list is rebuilt by brute-force minimum image with a fixed
    padded capacity (no stale lists at large strain, no recompilation).

Reported: the true (Cauchy) axial stress σ_nn(ε), the nominal (first
Piola-Kirchhoff) stress P = J σ_nn / λ, the peak Cauchy stress (= ideal
tensile strength, refined by a parabola through the three samples around the
maximum), the strain at the peak, and stability indicators per step:

* the lowest eigenvalue of the *combined* Hessian of the energy per volume
  with respect to the five free lab strains and the (non-translational)
  internal coordinates, the latter made dimensionless with ℓ = (V/N)^(1/3)
  so that every entry is in GPa (``min_combined_stiffness_GPa``).  This is the
  stability indicator: it is a smooth function of the load and passes through
  zero at the instability, whatever mixture of strain and internal relaxation
  the soft mode is (its eigenvector is reported);
* for diagnosis, the lowest non-translational eigenvalue of the position
  Hessian K (a Γ-point phonon instability of *this* supercell at fixed cell)
  and the lowest eigenvalue of the 5x5 lateral/shear block of the relaxed-ion
  tangent stiffness (the Schur complement H_ff - Λᵀ K⁺ Λ; it has a pole where
  K becomes singular on a strain-coupled mode, so its sign alone can flip
  from +large to -large and is not used as the criterion).

By Sylvester / Haynsworth inertia the combined Hessian has a negative
eigenvalue iff K or the relaxed lateral block has one (for nonsingular K).

Assumptions and limitations (reported with every record)
--------------------------------------------------------
* Small periodic cell: only instabilities commensurate with the simulation
  cell are captured; finite-wavevector phonon instabilities (Clatterbuck et
  al., PRL 91, 135501 (2003)) can make the true ideal strength lower than the
  peak stress found here.
* Homogeneous deformation of a defect-free crystal at 0 K: an upper bound to
  any measured strength (flaws, surfaces, temperature and rate are absent).
* For n along a symmetry axis ([001], [110], [111] of a cubic crystal) n stays
  an eigenvector of F, so σ_nn is the stress along the loading axis; for
  low-symmetry directions this holds only approximately.
* Short-cutoff bond-order potentials (Tersoff, Erhart-Albe) are known to be
  deficient at large strain (the cutoff function between R1 and R2 generates
  spurious stresses); their records are marked OUT_OF_DOMAIN by make_record.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from fracture_md.physics.elastic import (
    cauchy_stress_voigt,
    newton_relax_fixed_cell,
    relaxed_ion_correction,
    strain_derivatives,
)
from fracture_md.potentials.base import EnergyModel
from fracture_md.potentials.neighbors import MinimumImageError, check_minimum_image
from fracture_md.structures.symmetry import (
    rotate_voigt_stiffness,
    rotation_matrix_from_directions,
    strain_tensor_to_voigt,
    strain_voigt_to_tensor,
    stress_tensor_to_voigt,
    stress_voigt_to_tensor,
)
from fracture_md.structures.system import AtomicSystem
from fracture_md.units import EV_PER_A3_TO_GPA

_FREE = np.array([0, 1, 3, 4, 5])  # lab Voigt components relaxed to zero stress (all but zz)


def dense_neighbor_idx(system: AtomicSystem, r_list: float, capacity: int) -> np.ndarray:
    """Brute-force minimum-image Dense neighbour indices (N, capacity), padding = N.

    Raises if the cell violates the minimum-image condition for ``r_list`` or
    if more than ``capacity`` neighbours are found (caller must enlarge it).
    """
    check_minimum_image(system, r_list)
    frac = np.asarray(system.frac)
    box = np.asarray(system.box)
    d = frac[None, :, :] - frac[:, None, :]
    d -= np.round(d)
    dist = np.linalg.norm(d @ box.T, axis=-1)
    n = frac.shape[0]
    np.fill_diagonal(dist, np.inf)
    idx = np.full((n, capacity), n, dtype=np.int32)
    for i in range(n):
        js = np.nonzero(dist[i] < r_list)[0]
        if js.size > capacity:
            raise OverflowError(f"atom {i} has {js.size} neighbours > capacity {capacity}")
        idx[i, : js.size] = js
    return idx


@dataclasses.dataclass
class TensileStep:
    strain: float  # engineering axial strain |F n| - 1
    stretch: float  # λ = 1 + strain
    cauchy_axial_GPa: float
    nominal_axial_GPa: float  # J σ_nn / λ (force per reference area)
    energy_per_atom_eV: float
    lateral_residual_GPa: float  # max |σ_lab| over the 5 relaxed components
    fmax: float  # eV/Å after internal relaxation
    min_phonon_eig: float  # eV/Å², lowest non-translational eigenvalue of K
    min_lateral_stiffness_GPa: float  # lowest eigenvalue of the 5x5 relaxed tangent block
    lateral_strains: list  # lab Voigt strain of box relative to box0 (approx., small-rotation)
    converged: bool
    stable: bool = True  # combined (strain + internal) Hessian positive definite
    # Cutoff diagnostics for potentials with R1/R2 switching tables (Tersoff-like), NaN otherwise:
    max_bond_over_R1: float = float("nan")  # max r/R1 over reference nearest-neighbour bonds (>1: bond in cutoff zone)
    min_nonbond_over_R2: float = float("nan")  # min r/R2 over reference non-bonded pairs (<1: new spurious interaction)
    message: str = ""
    cell_repeats: list = dataclasses.field(default_factory=lambda: [1, 1, 1])  # replication of the input cell
    # lowest eigenvalue (GPa) of the combined Hessian of E/V w.r.t. (5 free lab strains, internal coords / ℓ)
    min_combined_stiffness_GPa: float = float("nan")
    soft_mode_strain: list = dataclasses.field(default_factory=list)  # lab (xx,yy,yz,xz,xy) part of that eigenvector
    soft_mode_internal_weight: float = float("nan")  # |internal part|^2 of the (unit) eigenvector


def uniaxial_tension_path(
    model: EnergyModel,
    system: AtomicSystem,
    direction_cart: np.ndarray,
    strains: np.ndarray,
    *,
    lateral_tol_GPa: float = 1e-3,
    fmax: float = 1e-6,
    max_iter: int = 40,
    skin: float = 0.5,
    stop_fraction: float = 0.7,
    steps_after_instability: int = 2,
    max_replications: int = 2,
) -> list[TensileStep]:
    """Quasi-static uniaxial-stress path; see module docstring.  ``strains`` must be increasing.

    If lateral contraction makes a cell width violate the minimum-image
    condition, the last converged state (and the reference cell) is replicated
    along the thin axes (at most ``max_replications`` times) and the step is
    redone; ``TensileStep.cell_repeats`` records the replication.  The path
    stops when a step fails to converge (or the cell still violates the
    minimum-image condition), when the axial stress has dropped below
    ``stop_fraction`` of its maximum, or ``steps_after_instability`` steps after
    the first unstable step (the symmetric branch beyond an instability is not
    physical; a few points are kept to show the trend).
    """
    s = system
    s_ref = system  # reference (unstrained) cell, replicated in lockstep with s
    box0 = np.asarray(system.box)
    V0 = system.volume
    Q = rotation_matrix_from_directions(direction_cart)  # rows: lab axes in crystal frame; lab z = n
    n = Q[2]
    r_list = model.cutoff + skin
    n_ref = dense_neighbor_idx(s, r_list, 4 * s.n_atoms)
    cutoff_tables = _cutoff_tables(model, s)
    cap = int(max(8, 2 * np.max(np.sum(n_ref < s.n_atoms, axis=1))))
    steps: list[TensileStep] = []
    Kinv = None
    peak = -np.inf
    first_unstable = None
    C_prev = None
    total_reps = np.ones(3, dtype=int)
    for eps_target in strains:
        s_prev = s
        st = None
        for attempt in range(max_replications + 1):
            try:
                st, s, Kinv, C_prev = _tension_step(model, s_prev, box0, V0, Q, n, float(eps_target), idx_cap=cap,
                                                    r_list=r_list, Kinv=Kinv, lateral_tol_GPa=lateral_tol_GPa,
                                                    fmax=fmax, max_iter=max_iter, C_prev=C_prev)
                break
            except MinimumImageError as exc:
                # Lateral (Poisson) contraction shrank a cell width below 2*(cutoff+skin).  Replicate the
                # last converged state along the thin axes and redo the step.  box0 is replicated the same
                # way, so the deformation gradient F = box @ inv(box0) and all per-atom / per-volume
                # quantities are unchanged; only the set of commensurate phonons grows.
                reps = _replication_for_margin(np.asarray(s_prev.box), r_list, margin=1.25)
                if attempt == max_replications or reps == (1, 1, 1):
                    steps.append(_failed_step(s_prev, box0, n, repr(exc)))
                    st = None
                    break
                s_prev = _replicate(s_prev, reps)
                s_ref = _replicate(s_ref, reps)
                box0 = box0 @ np.diag(np.asarray(reps, float))
                V0 = V0 * float(np.prod(reps))
                total_reps *= np.asarray(reps)
                if cutoff_tables is not None:
                    cutoff_tables = _cutoff_tables(model, s_ref)
                Kinv = None  # Hessian size changed
            except (OverflowError, np.linalg.LinAlgError) as exc:
                steps.append(_failed_step(s_prev, box0, n, repr(exc)))
                st = None
                break
        if st is None:
            break
        st.cell_repeats = [int(x) for x in total_reps]
        if cutoff_tables is not None:
            st.max_bond_over_R1, st.min_nonbond_over_R2 = _cutoff_diagnostics(s, *cutoff_tables)
        steps.append(st)
        peak = max(peak, st.cauchy_axial_GPa)
        if first_unstable is None and not st.stable:
            first_unstable = len(steps) - 1
        if not st.converged or (peak > 0 and st.cauchy_axial_GPa < stop_fraction * peak):
            break
        if first_unstable is not None and len(steps) - 1 - first_unstable >= steps_after_instability:
            break
    return steps


def _replication_for_margin(box: np.ndarray, r_list: float, margin: float = 1.25) -> tuple[int, int, int]:
    """Repeats so that every perpendicular width exceeds ``margin * 2 * r_list`` (1 where already satisfied)."""
    from fracture_md.structures.system import perpendicular_widths

    w = perpendicular_widths(box)
    return tuple(int(max(1, np.ceil(margin * 2.0 * r_list / wi))) for wi in w)  # type: ignore[return-value]


def _replicate(s: AtomicSystem, reps) -> AtomicSystem:
    """Periodic replication preserving the (possibly strained) cell; ASE ordering."""
    return AtomicSystem.from_ase(s.to_ase().repeat(tuple(int(r) for r in reps)), elements=s.elements)


def _pair_distances(s: AtomicSystem) -> np.ndarray:
    frac = np.asarray(s.frac)
    d = frac[None, :, :] - frac[:, None, :]
    d -= np.round(d)
    return np.linalg.norm(d @ np.asarray(s.box).T, axis=-1)


def _cutoff_tables(model: EnergyModel, s: AtomicSystem):
    """(R1_ij, R2_ij, bonded mask) for models exposing R1/R2 pair tables, else None."""
    p = model.params
    if not (isinstance(p, dict) and "R1" in p and "R2" in p):
        return None
    sp = np.asarray(model.species_for(s))
    R1 = np.asarray(p["R1"])[sp[:, None], sp[None, :]]
    R2 = np.asarray(p["R2"])[sp[:, None], sp[None, :]]
    r = _pair_distances(s)
    np.fill_diagonal(r, np.inf)
    bonded = r < R1
    return R1, R2, bonded


def _cutoff_diagnostics(s, R1, R2, bonded):
    r = _pair_distances(s)
    np.fill_diagonal(r, np.inf)
    nb = ~bonded
    np.fill_diagonal(nb, False)
    return float(np.max((r / R1)[bonded])) if bonded.any() else float("nan"), float(np.min((r / R2)[nb]))


def _failed_step(s, box0, n, msg) -> "TensileStep":
    F = np.asarray(s.box) @ np.linalg.inv(box0)
    lam = float(np.linalg.norm(F @ n))
    nan = float("nan")
    return TensileStep(strain=lam - 1.0, stretch=lam, cauchy_axial_GPa=nan, nominal_axial_GPa=nan, energy_per_atom_eV=nan,
                       lateral_residual_GPa=nan, fmax=nan, min_phonon_eig=nan, min_lateral_stiffness_GPa=nan,
                       lateral_strains=[nan] * 6, converged=False, stable=False, message=msg)


def _evaluate(model, s, Q, idx_cap, r_list, Kinv, fmax):
    """Relax positions at fixed cell; return the state needed for the lateral Newton iteration."""
    idx = jnp.asarray(dense_neighbor_idx(s, r_list, idx_cap))
    s, fm, _ = newton_relax_fixed_cell(model, s, idx, fmax=fmax, K_pinv=Kinv)
    e_tot, _, dEde, Hee, Lam, K = strain_derivatives(model, s, idx)
    corr, _, spec, Kinv = relaxed_ion_correction(K, Lam.reshape(-1, 6))
    V = s.volume
    sig_lab = stress_tensor_to_voigt(Q @ stress_voigt_to_tensor(dEde / V) @ Q.T) * EV_PER_A3_TO_GPA
    C_lab = rotate_voigt_stiffness((Hee - corr) / V * EV_PER_A3_TO_GPA, Q)
    return dict(s=s, fm=fm, e_tot=e_tot, spec=spec, Kinv=Kinv, sig_lab=sig_lab, C_lab=C_lab,
                r=sig_lab[_FREE], rmax=float(np.max(np.abs(sig_lab[_FREE]))), Hee=Hee, Lam=Lam, K=K)


def _lab_strain_map(Q: np.ndarray) -> np.ndarray:
    """6x6 M with e_crystal = M e_lab (Voigt, engineering shears) for ε_crystal = Qᵀ ε_lab Q."""
    M = np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6)
        e[j] = 1.0
        M[:, j] = strain_tensor_to_voigt(Q.T @ strain_voigt_to_tensor(e) @ Q)
    return M


def _nontranslation_basis(n_atoms: int) -> np.ndarray:
    """Orthonormal basis (3N, 3N-3) of real-space displacements orthogonal to the three rigid translations."""
    T = np.zeros((3 * n_atoms, 3))
    for a in range(3):
        T[a::3, a] = 1.0 / np.sqrt(n_atoms)
    q, _ = np.linalg.qr(np.hstack([T, np.eye(3 * n_atoms)[:, : 3 * n_atoms - 3]]))
    return q[:, 3:]


def combined_stability(Hee, Lam, K, V: float, n_atoms: int, Q: np.ndarray):
    """Lowest eigenpair of the combined Hessian of E/V over (5 free lab strains, internal coordinates / ℓ), GPa.

    A = (1/V) [[H_ff, ℓ Λ_fᵀ B], [ℓ Bᵀ Λ_f, ℓ² Bᵀ K B]] (converted to GPa) with ℓ = (V/N)^(1/3), H_ff and
    Λ_f the lab-frame free-strain blocks and B an orthonormal basis orthogonal to rigid translations.
    Returns (min eigenvalue, strain part of its unit eigenvector (5,), internal weight |v_u|^2).
    """
    M = _lab_strain_map(Q)
    H_lab = M.T @ np.asarray(Hee) @ M
    L_lab = np.asarray(Lam).reshape(-1, 6) @ M
    B = _nontranslation_basis(n_atoms)
    ell = (V / n_atoms) ** (1.0 / 3.0)
    Hf = H_lab[np.ix_(_FREE, _FREE)]
    Lf = B.T @ L_lab[:, _FREE] * ell
    Kb = B.T @ np.asarray(K) @ B * ell**2
    A = np.block([[Hf, Lf.T], [Lf, Kb]]) / V * EV_PER_A3_TO_GPA
    w, U = np.linalg.eigh(0.5 * (A + A.T))
    v = U[:, 0]
    return float(w[0]), v[:5].tolist(), float(np.sum(v[5:] ** 2))


def _apply_lab_strain(s, Q, de_lab):
    deps = Q.T @ strain_voigt_to_tensor(de_lab) @ Q
    return s.replace(box=jnp.asarray((np.eye(3) + deps) @ np.asarray(s.box)))


def _tension_step(model, s, box0, V0, Q, n, eps_target, *, idx_cap, r_list, Kinv, lateral_tol_GPa, fmax, max_iter,
                  C_prev=None):
    """One load step (see uniaxial_tension_path).

    Axial increment plus a linear (Poisson) predictor for the lateral strains
    from the previous tangent stiffness, then damped Newton iterations on the
    five lateral stresses (backtracking on max |σ_lateral|; the Jacobian is
    shifted to positive definiteness if necessary).
    """
    F = np.asarray(s.box) @ np.linalg.inv(box0)
    lam = float(np.linalg.norm(F @ n))
    d_ax = (1.0 + eps_target) / lam - 1.0
    de = np.zeros(6)
    de[2] = d_ax
    if C_prev is not None:
        de[_FREE] = -np.linalg.solve(C_prev[np.ix_(_FREE, _FREE)], C_prev[_FREE, 2] * d_ax)
        if np.max(np.abs(de[_FREE])) > abs(d_ax):  # predictor must not exceed the axial step (unstable tangent)
            de[_FREE] = 0.0
    st = _evaluate(model, _apply_lab_strain(s, Q, de), Q, idx_cap, r_list, Kinv, fmax)
    converged = False
    for _it in range(max_iter):
        if st["rmax"] < lateral_tol_GPa and st["fm"] <= fmax:
            converged = True
            break
        Cff = st["C_lab"][np.ix_(_FREE, _FREE)]
        Cff = 0.5 * (Cff + Cff.T)
        wmin = float(np.linalg.eigvalsh(Cff)[0])
        if wmin < 10.0:
            Cff = Cff + (10.0 - wmin) * np.eye(5)
        step = np.zeros(6)
        step[_FREE] = -np.linalg.solve(Cff, st["r"])
        if np.max(np.abs(step)) > 0.02:
            step *= 0.02 / np.max(np.abs(step))
        alpha = 1.0
        for _ls in range(6):
            trial = _evaluate(model, _apply_lab_strain(st["s"], Q, alpha * step), Q, idx_cap, r_list, st["Kinv"], fmax)
            if trial["rmax"] < st["rmax"]:
                break
            alpha *= 0.5
        st = trial
    s, fm, e_tot, spec, Kinv = st["s"], st["fm"], st["e_tot"], st["spec"], st["Kinv"]
    sig_lab, C_lab, r = st["sig_lab"], st["C_lab"], st["r"]
    w_comb, v_strain, w_int = combined_stability(st["Hee"], st["Lam"], st["K"], s.volume, s.n_atoms, Q)
    Cff = C_lab[np.ix_(_FREE, _FREE)]
    F = np.asarray(s.box) @ np.linalg.inv(box0)
    lam = float(np.linalg.norm(F @ n))
    J = s.volume / V0
    U = 0.5 * (F + F.T)  # symmetric by construction up to round-off
    e_lab = Q @ (U - np.eye(3)) @ Q.T
    w_lat = np.linalg.eigvalsh(0.5 * (Cff + Cff.T))
    st = TensileStep(
        strain=lam - 1.0,
        stretch=lam,
        cauchy_axial_GPa=float(sig_lab[2]),
        nominal_axial_GPa=float(J * sig_lab[2] / lam),
        energy_per_atom_eV=float(e_tot) / s.n_atoms,
        lateral_residual_GPa=float(np.max(np.abs(r))),
        fmax=float(fm),
        min_phonon_eig=float(spec.min_nonzero),
        min_lateral_stiffness_GPa=float(w_lat[0]),
        lateral_strains=[float(e_lab[0, 0]), float(e_lab[1, 1]), float(e_lab[2, 2]), float(2 * e_lab[1, 2]),
                         float(2 * e_lab[0, 2]), float(2 * e_lab[0, 1])],
        converged=converged,
        stable=bool(w_comb > 0),
        min_combined_stiffness_GPa=w_comb,
        soft_mode_strain=v_strain,
        soft_mode_internal_weight=w_int,
    )
    return st, s, Kinv, C_lab


def _parabolic_peak(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    k = int(np.argmax(y))
    if 0 < k < len(y) - 1:
        c = np.polyfit(x[k - 1 : k + 2], y[k - 1 : k + 2], 2)
        if c[0] < 0:
            xm = -c[1] / (2 * c[0])
            if x[k - 1] <= xm <= x[k + 1]:
                return float(np.polyval(c, xm)), float(xm), k
    return float(y[k]), float(x[k]), k


def direction_to_cartesian(system: AtomicSystem, uvw) -> np.ndarray:
    """Cartesian unit vector of lattice direction [uvw] in the crystal frame.

    For a cubic crystal whose cell is axis-aligned (the frame in which the
    elastic tensor is reported), [uvw] is taken directly in Cartesian
    coordinates.  Otherwise it is expressed in the lattice vectors of the
    spglib conventional cell (no rotation of the frame).
    """
    from fracture_md.structures.symmetry import conventional_cell, space_group

    box = np.asarray(system.box, dtype=float)
    if space_group(system).crystal_system == "cubic" and np.allclose(box, np.diag(np.diag(box)), atol=1e-8):
        v = np.asarray(uvw, dtype=float)
    else:
        conv = conventional_cell(system)
        L = np.asarray(conv.box if conv is not None else box)
        v = L @ np.asarray(uvw, dtype=float)
    return v / np.linalg.norm(v)


def run_ideal_strength(protocol, candidate, model: EnergyModel, structure=None, context=None):
    """IdealStrengthProtocol runner (tension, uniaxial stress).  See module docstring."""
    from fracture_md import provenance
    from fracture_md.engines.base import ProtocolOutput, RunContext
    from fracture_md.physics.elastic import analyze_elastic, elastic_constants_autodiff
    from fracture_md.physics.relax import run_bulk_relax
    from fracture_md.records import make_record
    from fracture_md.schema.common import CheckResult, Conditions, Uncertainty, UncertaintyKind
    from fracture_md.schema.material import Orientation
    from fracture_md.schema.properties import MethodSpec, PropertyName, SampledCurve
    from fracture_md.schema.protocol import BulkRelaxProtocol
    from fracture_md.units import Unit, q

    if protocol.loading != "tension":
        raise NotImplementedError("ideal shear strength is not implemented in this runner (loading='tension' only)")
    if protocol.lateral != "uniaxial_stress":
        raise NotImplementedError("only lateral='uniaxial_stress' is implemented")
    context = context or RunContext()
    out = ProtocolOutput()
    if structure is None:
        rel = run_bulk_relax(BulkRelaxProtocol(), candidate, model, context=context)
        out = out.merge(rel)
        out.properties = []  # keep only this protocol's records; relax checks stay in out.checks
        structure = rel.structure
    s0 = structure
    n = direction_to_cartesian(s0, protocol.direction)
    small = 1e-3
    grid = np.arange(protocol.strain_step, protocol.max_strain + 0.5 * protocol.strain_step, protocol.strain_step)
    strains = np.concatenate([[small], grid])
    fmax = protocol.relax.fmax_eV_per_A
    path = uniaxial_tension_path(model, s0, n, strains, fmax=fmax)
    good = [p for p in path if p.converged]  # failed / unconverged steps are excluded from the curve
    eps = np.array([0.0] + [p.strain for p in good])
    sig = np.array([0.0] + [p.cauchy_axial_GPa for p in good])
    nom = np.array([0.0] + [p.nominal_axial_GPa for p in good])
    epa = np.array([p.energy_per_atom_eV for p in good])

    # small-strain consistency with the elastic tensor: E(n) from relaxed-ion C
    ad = elastic_constants_autodiff(model, s0)
    Sfull = np.linalg.inv(ad.C_relaxed)
    Q = rotation_matrix_from_directions(n)
    E_n = 1.0 / np.linalg.inv(rotate_voigt_stiffness(ad.C_relaxed, Q))[2, 2]
    checks = []
    if good and good[0].strain < 0.01:
        slope = good[0].cauchy_axial_GPa / good[0].strain
        rel_slope = abs(slope - E_n) / E_n
        checks.append(CheckResult(name="small_strain_slope_equals_E_n", passed=rel_slope < 0.01, metric=rel_slope,
                                  tolerance=0.01, unit=Unit.DIMENSIONLESS,
                                  details=f"σ/ε at ε={good[0].strain:.1e}: {slope:.2f} GPa vs E(n) = {E_n:.2f} GPa from C"))

    # critical point = first of (a) the Cauchy-stress maximum, (b) the first unstable step
    # (combined strain + internal-coordinate Hessian not positive definite; see module docstring)
    k_peak = int(np.argmax(sig))  # index into eps/sig (0 = unstrained)
    k_unst = next((i + 1 for i, p in enumerate(good) if not p.stable), None)
    phonon_unstable = next((p.strain for p in good if p.min_phonon_eig < -1e-6), None)
    lateral_unstable = next((p.strain for p in good if p.min_lateral_stiffness_GPa <= 0), None)
    combined_unstable = next((p.strain for p in good if not p.stable), None)
    peak_inside = 0 < k_peak < len(sig) - 1
    soft_note = ""
    if k_unst is not None and k_unst <= k_peak:
        pu = good[k_unst - 1]
        lab = ("xx", "yy", "yz", "xz", "xy")
        vs = np.asarray(pu.soft_mode_strain, float)
        vs_n = vs / np.max(np.abs(vs)) if vs.size and np.max(np.abs(vs)) > 0 else vs
        soft_note = (f"; soft mode of the combined Hessian at ε={pu.strain:.4f} (eigenvalue "
                     f"{pu.min_combined_stiffness_GPa:.3g} GPa): lab strain ∝ "
                     + "{" + ", ".join(f"{k}: {x:+.3f}" for k, x in zip(lab, vs_n)) + "}"
                     + f", internal-coordinate weight {pu.soft_mode_internal_weight:.3f}")
        if pu.min_phonon_eig < -1e-6:
            mechanism = "phonon (cell-commensurate, fixed-cell) instability"
        elif pu.soft_mode_internal_weight > 1e-3:
            mechanism = "relaxed-ion lateral/shear instability (internal-strain coupled)"
        else:
            mechanism = "lateral/shear elastic instability"
        k_c = k_unst - 1  # last stable sample: conservative (lower-bound) estimate
        s_c, e_c = float(sig[k_c]), float(eps[k_c])
        unc_val = float(abs(sig[k_unst] - sig[k_c]))
        unc_desc = f"instability bracketed between ε = {eps[k_c]:.4f} (stable) and {eps[k_unst]:.4f} (unstable)"
        found = True
    else:
        mechanism = "stress maximum"
        s_c, e_c, _ = _parabolic_peak(eps, sig)
        k_c = k_peak
        unc_val = float(abs(s_c - sig[k_peak]))
        unc_desc = f"parabolic refinement vs raw maximum sample, strain step {protocol.strain_step}"
        found = peak_inside
    checks.append(CheckResult(name="critical_point_found", passed=bool(found), metric=float(eps[-1]),
                              details=f"{mechanism} within the sampled range (max strain reached {eps[-1]:.3f})"))
    upto = good[: max(1, k_c)]
    n_fail = sum(1 for p in path[: len(upto)] if not p.converged)
    checks.append(CheckResult(name="steps_converged_to_critical_point", passed=n_fail == 0 and len(good) >= len(upto),
                              metric=float(n_fail), tolerance=0.0,
                              details="lateral stresses and forces converged at every step up to the critical point"))
    # work identity: ΔE / V0 = ∫ P dε  (lateral stresses vanish, so they do no work); spline quadrature
    V0 = s0.volume
    e0 = ad.energy / s0.n_atoms
    dE = (np.concatenate([[e0], epa]) - e0) * s0.n_atoms / V0 * EV_PER_A3_TO_GPA  # GPa (= energy per reference volume)
    k_chk = max(1, k_c)
    if len(eps) >= 4:
        from scipy.interpolate import CubicSpline

        work_k = float(CubicSpline(eps, nom).integrate(0.0, eps[k_chk]))
    else:
        work_k = float(np.trapezoid(nom[: k_chk + 1], eps[: k_chk + 1]))
    rel_work = float(abs(work_k - dE[k_chk]) / max(abs(dE[k_chk]), 1e-12))
    checks.append(CheckResult(name="work_energy_identity", passed=rel_work < 0.02, metric=rel_work, tolerance=0.02,
                              unit=Unit.DIMENSIONLESS,
                              details=f"cubic-spline ∫P dε = {work_k:.4f} GPa vs ΔE/V0 = {dE[k_chk]:.4f} GPa "
                                      f"up to ε={eps[k_chk]:.3f}"))
    uvw = tuple(int(x) for x in protocol.direction)
    tag = "[" + "".join(str(x) for x in uvw) + "]"
    orient = Orientation(direction=uvw, description=f"uniaxial tension along {tag} (crystal frame)")
    cond = Conditions(stress_state=f"uniaxial Cauchy stress along {tag}; other 5 components relaxed to 0",
                      loading="quasistatic")
    method = MethodSpec(
        name="uniaxial-stress ideal tensile strength", protocol_id=protocol.protocol_id(), engine="jax_md",
        description="engineering-strain steps along n; lateral/shear stresses and internal coordinates relaxed each step",
        assumptions=["0 K lattice statics, homogeneous deformation of a perfect crystal",
                     f"{s0.n_atoms}-atom periodic cell (replicated automatically when lateral contraction violates "
                     "the minimum-image condition): only cell-commensurate instabilities are detected; "
                     "finite-wavevector phonon instabilities may lower the true ideal strength",
                     "strength = Cauchy (true) axial stress at the first of: stress maximum, or the last stable "
                     "sample before the combined (lateral strain + internal coordinate) Hessian loses positive "
                     "definiteness (cell-commensurate phonon or relaxed-ion lateral/shear instability)"],
    )
    unc = [Uncertainty(kind=UncertaintyKind.NUMERICAL_CONVERGENCE, value=unc_val, unit=Unit.GPA, interval="range",
                       description=unc_desc)]
    cut_bond = next((p.strain for p in good if p.max_bond_over_R1 > 1.0), None)
    cut_new = next((p.strain for p in good if p.min_nonbond_over_R2 < 1.0), None)
    cut_onset = min([x for x in (cut_bond, cut_new) if x is not None], default=None)
    cut_note = ("" if cut_onset is None else
                f"; POTENTIAL ARTIFACT WARNING: from ε={cut_onset:.3f} the result depends on the cutoff switching "
                f"function (nearest-neighbour bond beyond R1 at ε={cut_bond}, non-bonded pair inside R2 at ε={cut_new})")
    if cut_onset is not None and cut_onset <= e_c:
        checks.append(CheckResult(name="critical_point_before_cutoff_artifacts", passed=False, metric=float(cut_onset),
                                  details="critical point lies in the strain range where the cutoff function acts; "
                                          "the value reflects the cutoff choice rather than bond physics"))
    note = (f"limited by {mechanism} at ε ≈ {e_c:.3f}; stress maximum sample {sig[k_peak]:.3f} GPa at ε={eps[k_peak]:.3f}"
            f"{' (not bracketed)' if not peak_inside else ''}; first phonon instability at ε={phonon_unstable}, "
            f"first lateral instability at ε={lateral_unstable}, first combined-Hessian instability at "
            f"ε={combined_unstable}{soft_note}{cut_note}")
    if not found:
        note = ("NOT BRACKETED: no stress maximum or instability within the sampled strain range (stress still rising "
                f"at ε={eps[-1]:.3f}); the value is only a lower bound of this model's peak stress. " + note)
    prov = provenance.collect(engine="jax_md", potential_id=model.card.id, protocol_id=protocol.protocol_id(),
                              requested_by=context.requested_by, implementation_ids=dict(context.implementation_ids),
                              parent_ids=list(context.parent_ids))
    common = dict(candidate=candidate, method=method, card=model.card, conditions=cond, orientation=orient, checks=checks,
                  provenance=prov, frame="crystal", notes=note)
    recs = [
        make_record(name=PropertyName.IDEAL_TENSILE_STRENGTH, value=s_c, unit=Unit.GPA, uncertainty=unc, **common),
        make_record(name=PropertyName.CRITICAL_STRAIN, value=e_c, unit=Unit.DIMENSIONLESS,
                    shape_description="engineering axial strain at the critical point", **common),
    ]
    curve = SampledCurve(x_name="engineering axial strain", x=q(eps.tolist(), Unit.DIMENSIONLESS),
                         y_name="Cauchy axial stress", y=q(sig.tolist(), Unit.GPA),
                         extra={"nominal_stress": q(nom.tolist(), Unit.GPA), "energy_density": q(dE.tolist(), Unit.GPA)})
    out.properties.extend(recs)
    out.checks.extend(checks)
    out.structure = s0
    out.artifacts.update(stress_strain_curve=curve, steps=[dataclasses.asdict(p) for p in path], E_n_GPa=E_n,
                         direction_cartesian=n, S_per_GPa=Sfull)
    failed_msgs = [p.message for p in path if p.message]
    rep_strain = next((p.strain for p in good if p.cell_repeats != [1, 1, 1]), None)
    if rep_strain is not None:
        note += (f"; cell replicated {good[-1].cell_repeats} from ε={rep_strain:.3f} to keep the minimum-image "
                 "condition under lateral contraction")
        for r in recs:
            r.notes = note
    out.raw.update(path_termination=failed_msgs[-1] if failed_msgs else "completed", ideal_strength_GPa=s_c, critical_strain=e_c, mechanism=mechanism,
                   cell_repeats_final=(good[-1].cell_repeats if good else [1, 1, 1]), replication_strain=rep_strain,
                   phonon_instability_strain=phonon_unstable, lateral_instability_strain=lateral_unstable,
                   combined_instability_strain=combined_unstable,
                   cutoff_onset_strain=cut_onset, bond_beyond_R1_strain=cut_bond, nonbond_inside_R2_strain=cut_new,
                   E_n_GPa=float(E_n), stress_max_sample_GPa=float(sig[k_peak]), stress_max_strain=float(eps[k_peak]))
    return out
