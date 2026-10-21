"""DFT reference quantities against which interatomic potentials are judged (stage 2 of the DFT track).

Built on :mod:`fracture_md.engines.dft_qe` (pw.x through ASE); results, settings and comparisons are in
docs/DFT.md section 8 and data/dft_reference/.

* :func:`ideal_strength_path` - homogeneous (affine) shear or tension of a periodic cell.  Ions are relaxed at
  every strain and the five other Cauchy stress components are relaxed to a target (zero, or a hydrostatic
  pressure) by a quasi-Newton iteration on the deformation gradient (Jacobian from the elastic constants,
  Broyden updates, backtracking).  Same convention as the potential diagnostics
  (outputs/indent_derisk/potential-diagnostics/code/ideal_path.py): cell = cell0 @ T.T with T upper
  triangular in the lab frame (x = shear direction, z = plane normal); the controlled component is
  T_xz = engineering shear strain (T_zz - 1 for tension) and the reported stress is the Cauchy stress
  sigma_xz (sigma_zz).  The strength is the maximum of that stress along the path (parabolic refinement).
  It is an upper bound: phonon (non-affine) instabilities that need larger cells are not tested.
* :func:`gsfe_cell`, :func:`shifted_cell` - generalised stacking-fault energy of a {111} plane of a diamond or
  zincblende crystal by the tilted-cell method (one fault per periodic cell, no free surfaces), rigid or
  relaxed along the plane normal only (in-plane coordinates fixed).
* :func:`finite_strain_fit`, :func:`enthalpy_crossing` - static (0 K) transition pressure of two phases from
  Birch finite-strain fits of E(V) and H = E + PV.
* :func:`indentation_modulus_test`, :func:`ideal_shear_strength_test`, :func:`admission_verdict` - admission of an
  interatomic potential for contact / indentation work against the admission table
  (data/dft_reference/potential_comparison_sic.json, docs/DFT.md section 8.6).  Tests E and S were proposed on
  2026-10-05 in docs/POPIN_VALIDATION.md section 4.3 and accepted by the user as admission criteria on 2026-10-05.

The algorithms take an ``evaluate(atoms, label) -> Evaluation`` callable, so they are tested without pw.x
(tests/test_dft_reference.py); :func:`qe_evaluator` is the pw.x implementation.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from fracture_md.units import EV_PER_A2_TO_J_PER_M2, EV_PER_A3_TO_GPA

VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def voigt_index(pair) -> int:
    """Voigt index (0..5) of a tensor index pair, e.g. (0, 2) -> 4."""
    i, j = sorted(int(p) for p in pair)
    return VOIGT_PAIRS.index((i, j))


def voigt_matrix(s) -> np.ndarray:
    """Symmetric 3x3 tensor of a Voigt stress (xx, yy, zz, yz, xz, xy)."""
    s = np.asarray(s, float)
    return np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])


# ============================================================================
# geometry and elasticity helpers
# ============================================================================
def lab_rotation(x_dir, z_dir) -> np.ndarray:
    """Rotation whose rows are the lab axes (x, y = z cross x, z) in crystal coordinates (v_lab = R v_crystal)."""
    x = np.asarray(x_dir, float)
    z = np.asarray(z_dir, float)
    if abs(float(x @ z)) > 1e-10 * np.linalg.norm(x) * np.linalg.norm(z):
        raise ValueError("x_dir and z_dir must be orthogonal")
    y = np.cross(z, x)
    return np.array([x / np.linalg.norm(x), y / np.linalg.norm(y), z / np.linalg.norm(z)])


def oriented(atoms, x_dir, z_dir):
    """Copy of `atoms` (crystal axes = Cartesian axes) rotated so that x_dir is along lab x and z_dir along z."""
    R = lab_rotation(x_dir, z_dir)
    at = atoms.copy()
    at.set_cell(np.asarray(atoms.cell) @ R.T, scale_atoms=False)
    at.set_positions(atoms.get_positions() @ R.T)
    return at


def voigt_to_tensor(C) -> np.ndarray:
    """4th-rank stiffness tensor of a 6x6 Voigt matrix (engineering shear strains)."""
    C = np.asarray(C, float)
    t = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(VOIGT_PAIRS):
        for J, (k, l) in enumerate(VOIGT_PAIRS):
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, l), (l, k)):
                    t[a, b, c, d] = C[I, J]
    return t


def tensor_to_voigt(t) -> np.ndarray:
    return np.array([[t[i, j, k, l] for (k, l) in VOIGT_PAIRS] for (i, j) in VOIGT_PAIRS])


def rotate_stiffness(C, R) -> np.ndarray:
    """Voigt stiffness in the frame whose axes are the rows of R (C'_ijkl = R_ia R_jb R_kc R_ld C_abcd)."""
    R = np.asarray(R, float)
    return tensor_to_voigt(np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, voigt_to_tensor(C)))


def cubic_stiffness(c11: float, c12: float, c44: float) -> np.ndarray:
    C = np.zeros((6, 6))
    C[:3, :3] = c12
    C[np.arange(3), np.arange(3)] = c11
    C[np.arange(3, 6), np.arange(3, 6)] = c44
    return C


def relaxed_modulus(C_lab, pair) -> float:
    """d(sigma_p)/d(strain_p) with the other five stress components held at zero: 1 / S_pp (S = C^-1)."""
    S = np.linalg.inv(np.asarray(C_lab, float))
    k = voigt_index(pair)
    return float(1.0 / S[k, k])


def shear_modulus_111_cubic(c11: float, c12: float, c44: float) -> float:
    """Relaxed {111} shear modulus of a cubic crystal (any in-plane direction): 3 C44 (C11-C12) / (C11-C12+4 C44)."""
    return 3.0 * c44 * (c11 - c12) / (c11 - c12 + 4.0 * c44)


# ============================================================================
# ideal strength
# ============================================================================
@dataclasses.dataclass
class Evaluation:
    """One fixed-cell calculation (ions relaxed or not, as the evaluator decides)."""

    energy_eV: float
    stress_GPa: np.ndarray  # Voigt (xx, yy, zz, yz, xz, xy), Cauchy, positive = tensile
    scaled_positions: np.ndarray  # final (relaxed) fractional coordinates
    fmax_eV_A: float = 0.0
    converged: bool = True
    info: dict = dataclasses.field(default_factory=dict)


Evaluator = Callable[[Any, str], Evaluation]


def deformed(atoms0, T, scaled_positions=None):
    """Copy of atoms0 with cell rows cell0 @ T.T (x -> T x) and the given fractional coordinates."""
    at = atoms0.copy()
    at.set_cell(np.asarray(atoms0.cell) @ np.asarray(T, float).T, scale_atoms=True)
    if scaled_positions is not None:
        at.set_scaled_positions(np.asarray(scaled_positions, float))
    return at


def parabolic_peak(x, y) -> tuple[float, float, bool]:
    """Maximum of sampled y(x), refined by the parabola through the largest sample and its two neighbours.

    Returns (x_peak, y_peak, bracketed); bracketed is False when the largest sample is at either end."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    k = int(np.argmax(y))
    if 0 < k < len(x) - 1:
        c = np.polyfit(x[k - 1:k + 2], y[k - 1:k + 2], 2)
        if c[0] < 0:
            xm = -c[1] / (2.0 * c[0])
            if x[k - 1] <= xm <= x[k + 1]:
                return float(xm), float(np.polyval(c, xm)), True
        return float(x[k]), float(y[k]), True
    return float(x[k]), float(y[k]), False


def work_along_path(Ts, volumes, stresses_GPa) -> np.ndarray:
    """Cumulative work (eV per cell) of the Cauchy stress along a piecewise-linear deformation-gradient path.

    dW = V sigma : (dF F^-1), trapezoid in V sigma, F^-1 at the midpoint.  For a converged, consistent
    calculation W equals the energy change (the check that exposes Pulay stress or unrelaxed ions)."""
    W = [0.0]
    for k in range(1, len(Ts)):
        F0, F1 = np.asarray(Ts[k - 1], float), np.asarray(Ts[k], float)
        L = (F1 - F0) @ np.linalg.inv(0.5 * (F0 + F1))
        s0 = voigt_matrix(stresses_GPa[k - 1]) * volumes[k - 1]
        s1 = voigt_matrix(stresses_GPa[k]) * volumes[k]
        W.append(W[-1] + float(np.sum(0.5 * (s0 + s1) * L)) / EV_PER_A3_TO_GPA)
    return np.array(W)


def ideal_strength_path(evaluate: Evaluator, atoms0, *, C_lab, strains: Sequence[float], ctrl=(0, 2),
                        sign: int = 1, target_stress_GPa: Optional[Sequence[float]] = None,
                        relaxed: bool = True, tol_GPa: float = 0.1, max_iter: int = 10, max_step: float = 0.02,
                        drop_fraction: float = 0.8, n_drops: int = 2, refine: bool = True, case: str = "path",
                        log: Optional[Callable[[str], None]] = print) -> dict:
    """Stress-strain path of homogeneous shear (ctrl=(0, 2)) or tension (ctrl=(2, 2)) and its maximum.

    atoms0: reference cell in the lab frame (x = shear direction, z = plane normal), at the target stress.
    C_lab: 6x6 Voigt stiffness in that frame (GPa); its free-free block is the starting Jacobian of the
    lateral-stress iteration.  strains: increasing magnitudes of the controlled component; sign = +1 / -1
    selects the shear sense (the reported stress is sign * sigma_ctrl, positive in the loading sense).
    relaxed: iterate the other five T components until max |sigma_free - target| <= tol_GPa; otherwise keep
    them at identity.  The path stops when the stress falls below drop_fraction x the maximum, or after
    n_drops consecutive decreases.  refine: add the two half-step points around the largest sample.
    Returns a JSON-ready dict (rows, peak, small-strain modulus, work-energy check, stop reason)."""
    say = log or (lambda s: None)
    ctrl = tuple(int(c) for c in ctrl)
    if ctrl[0] > ctrl[1]:
        raise ValueError("ctrl must be an upper-triangular component, e.g. (0, 2) or (2, 2)")
    tension = ctrl[0] == ctrl[1]
    CV = voigt_index(ctrl)
    free = [p for p in VOIGT_PAIRS if p != ctrl]
    FV = [voigt_index(p) for p in free]
    target = np.zeros(6) if target_stress_GPa is None else np.asarray(target_stress_GPa, float)
    C_lab = np.asarray(C_lab, float)
    J0 = C_lab[np.ix_(FV, FV)]
    frac0 = atoms0.get_scaled_positions(wrap=False)
    n_atoms = len(atoms0)
    cache: dict = dict()
    n_eval = [0]

    def run(T, frac, label):
        key = tuple(np.round(np.asarray(T, float).ravel(), 12))
        if key not in cache:
            cache[key] = evaluate(deformed(atoms0, T, frac), label)
            n_eval[0] += 1
        return cache[key]

    def resid(ev):
        return np.asarray(ev.stress_GPa, float)[FV] - target[FV]

    def ctrl_value(s):
        return 1.0 + s if tension else sign * s

    def solve(T, frac, J, label):
        ev = run(T, frac, label + "_it0")
        r = resid(ev)
        it = 0
        while relaxed and np.max(np.abs(r)) > tol_GPa and it < max_iter:
            dx = -np.linalg.solve(J, r)
            big = float(np.max(np.abs(dx)))
            if big > max_step:
                dx *= max_step / big
            alpha, ok = 1.0, False
            for bt in range(3):
                Tt = T.copy()
                for k, p in enumerate(free):
                    Tt[p] += alpha * dx[k]
                trial = run(Tt, ev.scaled_positions, label + "_it%d%s" % (it + 1, "" if bt == 0 else "b%d" % bt))
                rt = resid(trial)
                s = alpha * dx
                J = J + np.outer((rt - r) - J @ s, s) / float(s @ s)  # Broyden (good) update, also from rejected trials
                if np.linalg.norm(rt) < np.linalg.norm(r):
                    ok = True
                    break
                alpha *= 0.5
            say("[%s]   newton %d: |r| %.4f -> %.4f GPa (step %.2e, alpha %.2f)" % (
                label, it + 1, float(np.max(np.abs(r))), float(np.max(np.abs(rt))), float(np.max(np.abs(s))), alpha))
            if ok:
                T, ev, r = Tt, trial, rt
            it += 1
        conv = bool(np.max(np.abs(r)) <= tol_GPa) if relaxed else True
        return T, ev, J, conv, it

    def row_of(s, T, ev, it, conv, kind):
        sig = np.asarray(ev.stress_GPa, float)
        tau = float(sig[CV] - target[CV]) * (1.0 if tension else float(sign))
        vol = float(abs(np.linalg.det(np.asarray(atoms0.cell) @ np.asarray(T).T)))
        return dict(strain=float(s), stress_GPa=tau, sigma_voigt_GPa=sig.tolist(), T=np.asarray(T).tolist(),
                    energy_eV=float(ev.energy_eV), energy_per_atom_eV=float(ev.energy_eV) / n_atoms,
                    volume_A3=vol, fmax_eV_A=float(ev.fmax_eV_A), converged_runs=bool(ev.converged),
                    lateral_residual_GPa=float(np.max(np.abs(resid(ev)))), lateral_converged=bool(conv),
                    newton_iterations=int(it), kind=kind, scaled_positions=np.asarray(ev.scaled_positions).tolist(),
                    info=ev.info)

    ref = run(np.eye(3), frac0, case + "/ref")
    rows = [row_of(0.0, np.eye(3), ref, 0, True, "reference")]
    say("[%s] reference: stress %s GPa" % (case, np.round(ref.stress_GPa, 3)))
    T = np.eye(3)
    frac = frac0
    J = J0.copy()
    hist: list = [(0.0, np.eye(3))]
    peak, drops, fails, stopped = -np.inf, 0, 0, "largest strain reached"
    for s in [float(x) for x in strains]:
        Tn = T.copy()
        Tn[ctrl] = ctrl_value(s)
        if relaxed and len(hist) >= 2:  # quadratic (3 previous points) or linear extrapolation of the free components
            pts = hist[-3:]
            deg = min(2, len(pts) - 1)
            for p in free:
                c = np.polyfit([h[0] for h in pts], [h[1][p] for h in pts], deg)
                Tn[p] = float(np.polyval(c, s))
        elif relaxed:  # linear elasticity: sigma_free = target at this controlled strain
            e_c = s if tension else sign * s
            e_f = np.linalg.solve(J0, (target[FV] - np.asarray(ref.stress_GPa)[FV]) - C_lab[FV, CV] * e_c)
            for k, p in enumerate(free):
                Tn[p] = (1.0 if p[0] == p[1] else 0.0) + e_f[k]
        Tn, ev, J, conv, it = solve(Tn, frac, J, "%s/s%.4f" % (case, s))
        row = row_of(s, Tn, ev, it, conv, "path")
        rows.append(row)
        hist.append((s, Tn.copy()))
        T, frac = Tn, ev.scaled_positions
        say("[%s] strain %.4f stress %8.3f GPa dE %.5f eV/atom lateral %.3f GPa it %d evals %d" % (
            case, s, row["stress_GPa"], row["energy_per_atom_eV"] - rows[0]["energy_per_atom_eV"],
            row["lateral_residual_GPa"], it, n_eval[0]))
        fails = 0 if conv else fails + 1
        if row["stress_GPa"] > peak:
            peak, drops = row["stress_GPa"], 0
        else:
            drops += 1
        if row["stress_GPa"] < drop_fraction * peak:
            stopped = "stress fell below %.2f x maximum" % drop_fraction
            break
        if drops >= n_drops:
            stopped = "%d consecutive decreases after the maximum" % n_drops
            break
        if fails >= 2:
            stopped = "lateral stress relaxation failed twice in a row"
            break
    path = [r for r in rows if r["kind"] == "path"]
    ys = [r["stress_GPa"] for r in path]
    k = int(np.argmax(ys)) if ys else 0
    if refine and 0 < k < len(path) - 1:
        for a, b in ((path[k - 1], path[k]), (path[k], path[k + 1])):
            s = 0.5 * (a["strain"] + b["strain"])
            Tm = 0.5 * (np.asarray(a["T"]) + np.asarray(b["T"]))
            Tm[ctrl] = ctrl_value(s)
            Tm, ev, J, conv, it = solve(Tm, np.asarray(a["scaled_positions"]), J, "%s/s%.4f" % (case, s))
            rows.append(row_of(s, Tm, ev, it, conv, "refine"))
            say("[%s] refine strain %.4f stress %8.3f GPa lateral %.3f GPa it %d" % (
                case, s, rows[-1]["stress_GPa"], rows[-1]["lateral_residual_GPa"], it))
    srt = sorted([r for r in rows if r["kind"] != "reference"], key=lambda r: r["strain"])
    # the peak is taken over samples that satisfy the lateral boundary condition (a sample whose lateral stresses
    # could not be relaxed, typically just past an instability, is not a point of the path)
    ok = [r for r in srt if r["lateral_converged"]] if relaxed else srt
    ok = ok if len(ok) >= 3 else srt
    x_pk, y_pk, bracketed = parabolic_peak([0.0] + [r["strain"] for r in ok], [0.0] + [r["stress_GPa"] for r in ok])
    # work-energy consistency along the path up to the peak strain
    chain = [rows[0]] + [r for r in ok if r["strain"] <= x_pk + 1e-12]
    W = work_along_path([r["T"] for r in chain], [r["volume_A3"] for r in chain], [r["sigma_voigt_GPa"] for r in chain])
    dE = np.array([r["energy_eV"] - rows[0]["energy_eV"] for r in chain])
    rel = float(abs(W[-1] - dE[-1]) / abs(dE[-1])) if len(chain) > 1 and abs(dE[-1]) > 0 else float("nan")
    first = ok[0] if ok else None
    small = float(first["stress_GPa"] / first["strain"]) if first and first["strain"] > 0 else None
    return dict(case=case, ctrl=list(ctrl), sign=int(sign), relaxed=bool(relaxed), tol_GPa=tol_GPa,
                target_stress_GPa=target.tolist(), n_atoms=n_atoms, n_evaluations=n_eval[0],
                peak_stress_GPa=y_pk, peak_strain=x_pk, peak_bracketed=bracketed,
                samples_excluded_from_peak=[r["strain"] for r in srt if r not in ok],
                small_strain_modulus_GPa=small, small_strain=first["strain"] if first else None,
                elastic_relaxed_modulus_GPa=relaxed_modulus(C_lab, ctrl) if relaxed else float(C_lab[CV, CV]),
                all_lateral_converged=all(r["lateral_converged"] for r in srt),
                all_runs_converged=all(r["converged_runs"] for r in rows),
                work_energy=dict(work_eV=W.tolist(), dE_eV=dE.tolist(), relative_mismatch_at_peak=rel,
                                 strains=[r["strain"] for r in chain]),
                stopped=stopped, rows=rows)


# ============================================================================
# generalised stacking faults on {111}
# ============================================================================
def gsfe_cell(prim, a: float, n_bilayers: int, cut: str):
    """Periodic {111} cell of a diamond / zincblende crystal with one chosen gap at the cell boundary.

    prim: 2-atom fcc primitive cell with cubic axes along x, y, z (dft_qe.primitive_cell; for zincblende the
    first atom sits at the fcc origin and the second at a/4 [111]).  Lab frame: x || [11-2], y || [-110],
    z || [111].  In-plane cell: the 1x1 {111} cell (a/2 [-110], a/2 [10-1]); c = n_bilayers a/sqrt(3) along z
    (n_bilayers a multiple of 3).  cut: 'shuffle' = widely spaced gap (a sqrt(3)/4, one bond per atom cut;
    first species below, second above) or 'glide' = narrowly spaced gap (a sqrt(3)/12, three bonds per atom;
    second species below, first above).  The gap is centred on z = 0 = c, so tilting c by a shift s displaces
    the block above that gap by +s relative to the block below (one fault per cell, no surfaces).
    Returns (atoms, info)."""
    from ase.build import make_supercell

    if n_bilayers % 3:
        raise ValueError("n_bilayers must be a multiple of 3 (ABC stacking period)")
    if cut not in ("shuffle", "glide"):
        raise ValueError(cut)
    R = lab_rotation([1, 1, -2], [1, 1, 1])
    p = oriented(prim, [1, 1, -2], [1, 1, 1])
    target = np.array([R @ (0.5 * a * np.array([-1.0, 1.0, 0.0])), R @ (0.5 * a * np.array([1.0, 0.0, -1.0])),
                       [0.0, 0.0, n_bilayers * a / math.sqrt(3.0)]])
    P = target @ np.linalg.inv(np.asarray(p.cell))
    Pi = np.round(P).astype(int)
    if not np.allclose(P, Pi, atol=1e-6):
        raise ValueError("cell is not a supercell of the primitive cell")
    sc = make_supercell(p, Pi)
    sc.set_cell(target, scale_atoms=False)
    sc.wrap()
    Lz = float(target[2, 2])
    sp0, sp1 = prim.get_chemical_symbols()[0], prim.get_chemical_symbols()[1]
    sym = np.array(sc.get_chemical_symbols())
    z = sc.positions[:, 2] % Lz
    order = np.argsort(z)
    levels: list = []
    for i in order:
        if levels and abs(z[i] - levels[-1][0]) < 1e-3:
            levels[-1][1].append(i)
        else:
            levels.append((z[i], [i]))
    wide, narrow = a * math.sqrt(3.0) / 4.0, a * math.sqrt(3.0) / 12.0
    for k in range(len(levels)):
        z_lo, idx_lo = levels[k]
        z_hi, idx_hi = levels[(k + 1) % len(levels)]
        if k == len(levels) - 1:
            z_hi = z_hi + Lz
        g = z_hi - z_lo
        below, above = sym[idx_lo[0]], sym[idx_hi[0]]
        is_wide = abs(g - wide) < 0.02 * a
        is_narrow = abs(g - narrow) < 0.02 * a
        if (cut == "shuffle" and is_wide and below == sp0 and above == sp1) or \
           (cut == "glide" and is_narrow and below == sp1 and above == sp0):
            at = sc.copy()
            pos = at.positions.copy()
            pos[:, 2] = (pos[:, 2] - 0.5 * (z_lo + z_hi)) % Lz
            at.set_positions(pos)
            area = float(np.linalg.norm(np.cross(target[0], target[1])))
            info = dict(cut=cut, gap_A=float(g), below=str(below), above=str(above), n_bilayers=int(n_bilayers),
                        n_atoms=len(at), area_A2=area, Lz_A=Lz, a_A=float(a), b_partial_A=a / math.sqrt(6.0),
                        b_full_A=a / math.sqrt(2.0), x_dir="[11-2]", y_dir="[-110]", z_dir="[111]")
            return at, info
    raise RuntimeError("no %s gap found" % cut)


def shifted_cell(atoms, shift_xy):
    """Copy with the third cell vector tilted by (sx, sy, 0) angstrom: the fault displacement of the gap at z = 0."""
    at = atoms.copy()
    cell = np.array(at.cell)
    cell[2, 0] += float(shift_xy[0])
    cell[2, 1] += float(shift_xy[1])
    at.set_cell(cell, scale_atoms=False)
    return at


def gamma_J_m2(e_eV: float, e_ref_eV: float, area_A2: float) -> float:
    return (e_eV - e_ref_eV) / area_A2 * EV_PER_A2_TO_J_PER_M2


# ============================================================================
# phase transition (static enthalpy crossing)
# ============================================================================
def finite_strain_fit(volumes, energies, order: int = 3) -> dict:
    """Birch (Eulerian) finite-strain fit: E is a polynomial of degree `order` in x = V^(-2/3) (3 = BM3, 4 = BM4).

    Volumes in angstrom^3 and energies in eV (per atom or per cell, consistently).  Returns V0, E0, B0 (GPa),
    B0', the RMS residual (meV) and callables energy(V) [eV] and pressure_GPa(V); vmin/vmax are the sampled range."""
    V = np.asarray(volumes, float)
    E = np.asarray(energies, float)
    p = np.poly1d(np.polyfit(V ** (-2.0 / 3.0), E, int(order)))
    dp, d2p = p.deriv(), p.deriv(2)

    def energy(v):
        return float(p(np.asarray(v, float) ** (-2.0 / 3.0)))

    def pressure_GPa(v):
        v = float(v)
        return float(dp(v ** (-2.0 / 3.0)) * (2.0 / 3.0) * v ** (-5.0 / 3.0)) * EV_PER_A3_TO_GPA

    def bulk_GPa(v):
        v = float(v)
        xv = v ** (-2.0 / 3.0)
        return float(v * (d2p(xv) * (4.0 / 9.0) * v ** (-10.0 / 3.0) + dp(xv) * (10.0 / 9.0) * v ** (-8.0 / 3.0))) \
            * EV_PER_A3_TO_GPA

    xs = V ** (-2.0 / 3.0)
    roots = [r.real for r in dp.roots if abs(r.imag) < 1e-12 and r.real > 0]
    roots = [r for r in roots if 0.8 * xs.min() <= r <= 1.2 * xs.max()]
    if not roots:
        raise ValueError("finite-strain fit has no minimum near the sampled volumes")
    x0 = min(roots, key=lambda r: abs(r - float(np.mean(xs))))
    V0 = float(x0 ** (-1.5))
    h = 1e-4 * V0
    Bp = (bulk_GPa(V0 + h) - bulk_GPa(V0 - h)) / (pressure_GPa(V0 + h) - pressure_GPa(V0 - h))
    resid = E - p(xs)
    return dict(order=int(order), coefficients=p.coeffs.tolist(), V0_A3=V0, E0_eV=energy(V0), B0_GPa=bulk_GPa(V0),
                B0_prime=float(Bp), rms_meV=float(np.sqrt(np.mean(resid ** 2)) * 1e3),
                max_residual_meV=float(np.max(np.abs(resid)) * 1e3), vmin_A3=float(V.min()), vmax_A3=float(V.max()),
                energy=energy, pressure_GPa=pressure_GPa)


def volume_at_pressure(fit: dict, P_GPa: float, margin: float = 0.03) -> float:
    """Volume where the fitted P(V) equals P_GPa (searched in the sampled range +- margin)."""
    from scipy.optimize import brentq

    lo, hi = fit["vmin_A3"] * (1.0 - margin), fit["vmax_A3"] * (1.0 + margin)
    f = lambda v: fit["pressure_GPa"](v) - P_GPa  # noqa: E731
    if f(lo) * f(hi) > 0:
        raise ValueError("pressure %.2f GPa outside the fitted range" % P_GPa)
    return float(brentq(f, lo, hi, xtol=1e-12))


def enthalpy(fit: dict, P_GPa: float) -> float:
    """H = E + P V (eV) at pressure P from a finite-strain fit."""
    v = volume_at_pressure(fit, P_GPa)
    return fit["energy"](v) + P_GPa / EV_PER_A3_TO_GPA * v


def pressure_range(fit: dict) -> tuple[float, float]:
    """Pressures (GPa) at the largest and smallest sampled volume."""
    return fit["pressure_GPa"](fit["vmax_A3"]), fit["pressure_GPa"](fit["vmin_A3"])


def enthalpy_crossing(fit_low: dict, fit_high: dict, p_min_GPa: float = 0.0, p_max_GPa: float = 200.0,
                      n_grid: int = 401) -> dict:
    """Pressure where H_high(P) = H_low(P) (the high-pressure phase becomes stable), with the volumes of both
    phases there and the volume collapse.  Only pressures inside both sampled ranges are searched."""
    from scipy.optimize import brentq

    lo = max(p_min_GPa, pressure_range(fit_low)[0], pressure_range(fit_high)[0])
    hi = min(p_max_GPa, pressure_range(fit_low)[1], pressure_range(fit_high)[1])
    if hi <= lo:
        raise ValueError("the two fits share no pressure range")
    Ps = np.linspace(lo, hi, n_grid)
    dH = np.array([enthalpy(fit_high, p) - enthalpy(fit_low, p) for p in Ps])
    idx = np.where((dH[:-1] > 0) & (dH[1:] <= 0))[0]
    out = dict(searched_GPa=[float(lo), float(hi)], dH_eV_at_ends=[float(dH[0]), float(dH[-1])])
    if len(idx) == 0:
        out.update(P_t_GPa=None)
        return out
    i = int(idx[0])
    Pt = float(brentq(lambda p: enthalpy(fit_high, p) - enthalpy(fit_low, p), Ps[i], Ps[i + 1], xtol=1e-10))
    vl, vh = volume_at_pressure(fit_low, Pt), volume_at_pressure(fit_high, Pt)
    out.update(P_t_GPa=Pt, V_low_A3=vl, V_high_A3=vh, volume_collapse=float((vl - vh) / vl),
               H_eV=float(enthalpy(fit_low, Pt)))
    return out


# ============================================================================
# pw.x evaluator and records
# ============================================================================
def qe_evaluator(settings, runner, workdir, *, kmesh, relax_ions: bool = True,
                 forc_conv_thr_ry_bohr: float = 5.0e-4, etot_conv_thr_ry: float = 1.0e-5, nstep: int = 60,
                 fix_mask: Optional[Sequence[bool]] = None, pseudo=None) -> Evaluator:
    """Evaluator running pw.x in workdir/<label>.

    Fixed-cell 'relax' (pw.x BFGS on the ions) when relax_ions and the cell has more than one atom, else 'scf'.
    fix_mask (3 bools, True = fixed) constrains every atom's Cartesian components, e.g. (True, True, False) =
    relaxation along z only.  kmesh is fixed for every call (the same reduced mesh along a path, as in
    dft_qe.elastic_constants).  A finished run with a byte-identical input is reused (restartable paths).
    evaluator.runs collects the PWResult objects for provenance.

    Ionic thresholds: forc_conv_thr 5e-4 Ry/bohr (0.013 eV/A) and etot_conv_thr 1e-5 Ry.  Tighter values made
    pw.x BFGS cycle for a 2-atom sheared SiC cell (energy changes of ~1e-6 Ry, below the egg-box noise of the
    ultrasoft/PAW augmentation, contradicted the forces; outputs/dft/stage2 notes in docs/DFT.md section 8).  With
    a Kleinman coupling of ~60 GPa/A and an optical force constant of ~20 eV/A^2 the residual force changes the
    stress by <= 0.05 GPa.  nstep bounds a stalled relaxation (the run is then flagged not converged)."""
    from fracture_md.engines import dft_qe as q

    pseudo = pseudo or q.load_pseudo_family(settings.pseudo_family)
    workdir = Path(workdir)
    runs: list = []
    relax = dict(forc_conv_thr_ry_bohr=float(forc_conv_thr_ry_bohr), etot_conv_thr_ry=float(etot_conv_thr_ry),
                 nstep=int(nstep))

    def evaluate(atoms, label):
        at = atoms.copy()
        if fix_mask is not None:
            from ase.constraints import FixCartesian

            at.set_constraint(FixCartesian(list(range(len(at))), mask=[bool(m) for m in fix_mask]))
        calc = "relax" if (relax_ions and len(at) > 1) else "scf"
        r = q.run_pw(at, settings, runner, workdir / label, calculation=calc, relax=relax, kmesh=kmesh,
                     label=label, pseudo=pseudo, timing_log=workdir / "timings.jsonl")
        runs.append(r)
        f = np.asarray(r.forces_eV_A, float)
        if fix_mask is not None and f.size:
            f = f * (~np.asarray(fix_mask, bool))[None, :]
        fmax = float(np.max(np.linalg.norm(f, axis=1))) if f.size else 0.0
        return Evaluation(energy_eV=float(r.energy_eV), stress_GPa=np.asarray(r.stress_GPa, float),
                          scaled_positions=r.atoms.get_scaled_positions(wrap=False), fmax_eV_A=fmax,
                          converged=bool(r.converged),
                          info=dict(label=label, calculation=calc, wall_s=round(float(r.wall_s), 2),
                                    n_kpoints=r.meta.get("n_kpoints"), scf_cycles=r.meta.get("n_scf_cycles"),
                                    reused=bool(r.meta.get("reused_previous_output", False))))

    evaluate.runs = runs  # type: ignore[attr-defined]
    return evaluate


def make_reference_record(name, value, unit, *, atoms, settings, runner, runs, label: str, method_name: str,
                          method_text: str, assumptions: Sequence[str] = (), checks=(), references=(),
                          notes: str = "", stress_state: str = "", uncertainty=(), orientation=None, shape: str = "",
                          candidate=None, requested_by: str = "human"):
    """PropertyRecord for a DFT reference quantity, with the provenance of every pw.x run behind it.

    Status rule (records.make_record): NUMERICALLY_VERIFIED when every check passes; a published value of the
    same functional (dft_qe.LiteratureValue in `references`) within tolerance gives LITERATURE_MODEL_AGREEMENT.
    Values from other functionals belong in the comparison tables, not in `references`."""
    from fracture_md.engines import dft_qe as q
    from fracture_md.schema.common import Conditions

    pseudo = q.load_pseudo_family(settings.pseudo_family)
    els = atoms.get_chemical_symbols()
    ver = runs[0].meta["qe_version"] if runs else "unknown"
    card = q.dft_card(settings, els, qe_version=ver, pseudo=pseudo)
    prov = q._provenance(card, runs, settings, runner, label, pseudo, els, requested_by)
    km = runs[0].kmesh if runs else settings.kmesh(atoms)
    method = q._method(method_name, q._settings_text(settings, km, pseudo) + "; " + method_text, list(assumptions),
                       label)
    all_checks = list(q._run_checks(runs)) + list(checks)
    b = q._Builder(candidate or q._auto_candidate(atoms), card, method, prov, Conditions(stress_state=stress_state),
                   references)
    return b(name, value, unit, checks=all_checks, notes=notes, uncertainty=list(uncertainty), orientation=orientation,
             shape=shape)


# ============================================================================
# admission of an interatomic potential (tests E and S; docs/DFT.md section 8.6)
# ============================================================================
#: tolerances of the two admission tests accepted by the user on 2026-10-05 (proposed the same day in
#: docs/POPIN_VALIDATION.md section 4.3): E, the indentation modulus of the indented face within 10 % of the value
#: from measured single-crystal C_ij; S, the easy-sense ideal shear strength within 15 % of same-protocol DFT
ADMISSION_E_REL_TOL = 0.10
ADMISSION_S_REL_TOL = 0.15
ADMISSION_ACCEPTED_BY_USER = "2026-10-05"
#: criteria that every admission verdict must contain (admission_verdict refuses a verdict without them)
ADMISSION_REQUIRED = ("E", "S")
#: faces whose indentation modulus test E evaluates by default (Miller indices of the surface normal)
ADMISSION_FACES = {"001": (0, 0, 1), "111": (1, 1, 1)}


def indentation_modulus_test(cij_potential, cij_measured, rel_tol: float = ADMISSION_E_REL_TOL,
                             faces: Optional[dict] = None) -> dict:
    """Test E of potential admission: the Vlassak-Nix indentation modulus M(hkl) of each indented face, from the
    potential's relaxed-ion cubic C_ij (C11, C12, C44 in GPa), within ``rel_tol`` of M(hkl) from the measured
    single-crystal C_ij (fracture_md.physics.popin.vlassak_nix_modulus_cubic_GPa for both).

    Returns ``M_GPa``, ``M_reference_GPa``, ``deviation`` (M / M_ref - 1) and ``passes_by_face`` per face, and
    ``passes``: True only when every face passes.  ``cij_potential`` None (no C_ij of the potential) gives
    ``passes`` None: not evaluated, never a pass."""
    from fracture_md.physics.popin import vlassak_nix_modulus_cubic_GPa

    faces = dict(ADMISSION_FACES if faces is None else faces)
    if not faces:
        raise ValueError("test E needs at least one face")
    ref = dict((f, vlassak_nix_modulus_cubic_GPa(*[float(c) for c in cij_measured], n)) for f, n in faces.items())
    if cij_potential is None or any(c is None for c in cij_potential):
        return dict(M_GPa=None, M_reference_GPa=ref, deviation=None, rel_tol=float(rel_tol),
                    passes_by_face=dict((f, None) for f in faces), passes=None)
    m = dict((f, vlassak_nix_modulus_cubic_GPa(*[float(c) for c in cij_potential], n)) for f, n in faces.items())
    dev = dict((f, m[f] / ref[f] - 1.0) for f in faces)
    by = dict((f, bool(abs(dev[f]) <= rel_tol)) for f in faces)
    return dict(M_GPa=m, M_reference_GPa=ref, deviation=dev, rel_tol=float(rel_tol), passes_by_face=by,
                passes=all(by.values()))


def ideal_shear_strength_test(tau_potential_GPa, tau_dft_GPa, rel_tol: float = ADMISSION_S_REL_TOL) -> dict:
    """Test S of potential admission: the potential's relaxed easy-sense ((111)[11-2] twinning-sense for
    zincblende) ideal shear strength within ``rel_tol`` of the DFT value computed with the same protocol.  A missing
    value on either side gives ``passes`` None (not evaluated)."""
    if tau_potential_GPa is None or tau_dft_GPa is None:
        return dict(value_GPa=tau_potential_GPa, reference_GPa=tau_dft_GPa, deviation=None, rel_tol=float(rel_tol),
                    passes=None)
    if float(tau_dft_GPa) <= 0.0:
        raise ValueError("the DFT reference strength must be positive")
    dev = float(tau_potential_GPa) / float(tau_dft_GPa) - 1.0
    return dict(value_GPa=float(tau_potential_GPa), reference_GPa=float(tau_dft_GPa), deviation=dev,
                rel_tol=float(rel_tol), passes=bool(abs(dev) <= rel_tol))


def admission_verdict(passes: dict, required: Sequence[str] = ADMISSION_REQUIRED) -> dict:
    """Overall admission of a potential from its per-criterion results (True pass, False fail, None not
    evaluated).  ``admitted`` is True only when every criterion passes, False when any fails, and None when none
    fails but some are not evaluated (undecided: never an admission).  The criteria in ``required`` (tests E and S,
    accepted by the user on 2026-10-05) must be present: a verdict without them raises KeyError."""
    missing = [k for k in required if k not in passes]
    if missing:
        raise KeyError("admission verdict without the required criteria %s" % ", ".join(missing))
    for k, v in passes.items():
        if v is not None and not isinstance(v, bool):
            raise TypeError("criterion %s: expected True, False or None, got %r" % (k, v))
    failed = [k for k, v in passes.items() if v is False]
    open_ = [k for k, v in passes.items() if v is None]
    admitted = False if failed else (None if open_ else True)
    return dict(admitted=admitted, failed=failed, not_evaluated=open_)
