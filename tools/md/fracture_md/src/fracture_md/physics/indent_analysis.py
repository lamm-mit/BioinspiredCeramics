"""Indentation analysis: windowed loads, depth zero, contact areas, plateau and plasticity gates, diagnostics.

Engine-agnostic post-processing (numpy + scipy; JAX only inside the Gate I quench backend) of indentation
runs of the JAX harness (physics/indent_md.py) and of imported LAMMPS runs (io/lammps_indent.py).  It
implements items 15-18 and 22 of the recommended MD Berkovich protocol (synthesis section 4, recipe_v0):

Load and depth
--------------
* Load: the z force of the indenter on the sample, reported as a centred running mean over
  ``window_ps`` (0.5 ps; the window must span >= 5 load autocorrelation times, tau_ac ~ 20 fs at 300 K)
  and as non-overlapping block means (:func:`windowed_load`, :func:`block_average`,
  :func:`integrated_autocorrelation_time`).
* Depth: tip displacement below z_s0, the mean height of the relaxed top layer.  With a rigid SDF wall the
  contact depth is h_c = h - pen_fw, pen_fw = sum(delta^3) / sum(delta^2) the force-weighted wall
  penetration (:func:`contact_depth`).  Two depth-zero rules are reported: first contact (windowed load
  above max(3 sigma_thermal, floor)) and an elastic fit h0 of the loading branch against the same depth h_c
  that E2 uses (Hertz for spheres; Sneddon for pyramids and cones with the tip's OWN profile, the sharp
  equivalent cone only for R = 0: a sharp-cone fit of a rounded tip absorbs the tip bluntness into h0, which
  E2's rounded area function then counts a second time; :func:`first_contact_depth`, :func:`sneddon_fit`,
  :func:`hertz_fit`).

Contact area (the estimators E1-E5 of the protocol; ``ESTIMATOR_NAMES`` maps them to the schema names)
------------------------------------------------------------------------------------------------------
* E1 nominal_ideal          C0 h^2 (never plateau evidence).
* E2 rounded_area_function  the exact (rounded-)tip area function at h_eff = h_c - h0.
* E3 dilated_hull           convex hull of the projected ESTIMATED-CONTACT atoms dilated by d = a/(2 sqrt 2)
                            (half the (001) surface nearest-neighbour spacing, 1.545 A at a(300 K) = 4.3705 A):
                            A_hull + P_hull d + pi d^2.  PRIMARY.  Same formula as the README renderer's
                            estimated-contact outline (viz.indentviz.contact_outline).
* E4 closed_discs           union of discs of radius d about the estimated-contact atoms after a morphological
                            closing with a 3 A disc, holes filled (check).
* E5 triangle_fit           3 sqrt3 (r_in + d)^2, r_in = mean over the three face normals of the largest
                            projection of an estimated-contact atom (never plateau evidence).
A sharp self-similar tip gives a depth-independent ELASTIC p_m (Sneddon: 24-26 GPa on E2 for Vashishta, G06),
so a flat p_m is not by itself evidence of plasticity; the E3/E4 spread is carried as a systematic.

Estimated contact (critic item B1; :func:`estimated_contact_mask`, ``CONTACT_FORCE_FRACTION``)
---------------------------------------------------------------------------------------------
E3-E5 use the ESTIMATED-contact atoms: the atoms touching the tip (sdf < 0, or an atomistic tip's engine flag)
whose tip force is at least ``CONTACT_FORCE_FRACTION`` = 1/2 of the mean tip force of the touching atoms (SDF wall:
force K delta^2, so K cancels; an atomistic tip needs a per-atom ``tip_force`` frame field, else every touching atom
is kept and ``analysis["contact"]["rules_used"]`` says so).  For a flat (plastic) pressure field this is the
half-cell rule: a rim atom counts when about half of its surface cell lies inside the contact.  The touching set
alone over-counts: rim atoms touch with a few per cent of the mean force, and the dilated hull counts each as a
full cell (+18 % in radius on the elastic Vashishta spheres).  Calibration (outputs/ready/r4-contact.md;
scripts/gate_e.py contact; 0 K AQS, Vashishta (001)-C, K = 100, four symmetry-distinct axis offsets against the
box-BEM contact radius at the same depth with the potential's own relaxed-ion C_ij):

* on exact continuum pressure fields sampled on the surface lattice (the estimator's own bias) the E3 radius is
  off by -4 % to +2 % (Hertz) and by +5 % (a = 12 A) to +2 % (a = 30 A) (flat, the plastic limit); the touching
  set over-counts by +8 % (a = 30 A) to +20 % (a = 10 A);
* R = 100 A, h = 1-2 A (a = 11-16 A, continuum p0 24-36 GPa: the model's contact is linear-elastic there, its load
  distribution matches the box BEM's): -1.1 % to -3.2 % (the estimator's Hertz bias); touching set +10 to +16 %;
* R = 30 A: +0.4 % to +4.9 % for h <= 2.25 A, +5.7 % to +9.8 % at h = 2.5-4 A, where the atomistic contact
  pressure is flat-topped (38-48 GPa at the centre, 0.6-0.7 of the linear continuum's at the same load) and
  reaches beyond the linear-elastic contact edge: part estimator flat-profile bias (+5-6 % at a = 9-12 A), part a
  genuinely wider model contact; the same at 300 K (two MD seeds: +1.5 / +5.3 / +8.2 / +9.1 % at h = 1-2 / 2-3 /
  3-4 / 4-4.6 A); touching set +15 to +19 %;
* sharp Berkovich, elastic box-BEM field of the G06 cell (h = 10.04 A): E3 is 1.56x the BEM contact area (+25 % in
  radius): the elastic contact is a pincushion and the hull bridges its concave sides (E4 0.84x);
  tests/test_indent_e3_elastic_berkovich.py;
* G06 (atomistic Berkovich, 1 nm): E3 2635 A^2 instead of 3155 A^2; like for like (E3 on the elastic field of its
  cell) 1.56x wider, p_m(E3) 0.63x (31.0 vs 49.0 GPa); its contact atoms fill 84 % of the hull (elastic 54 %):
  a fuller, pressure-limited model contact.  (Against the BEM's exact area it is 2.45x, which contains the 1.56x
  hull bias.)

It is called the ESTIMATED contact: no atomistic definition of contact is unique, and E3 p_m is a model-specific
contact pressure carrying the estimator systematic above.

Gates (recommended protocol items 16-18)
----------------------------------------
* Gate I, first irreversibility (diagnostic): the indenter is removed from a saved frame, a carved
  sub-volume is quenched at 0 K and compared with the identically carved and quenched pre-contact reference
  (:func:`carve_subvolume`, :func:`gate_i`).  The fixed shell is held at the REFERENCE positions in both
  quenches and the loaded interior starts from its positions minus the harmonic extension of the shell
  mismatch, so an elastic state quenches back to the reference minimum exactly (dPE ~ 0) and no spurious
  defects form at the shell (the shell mismatch is up to ~1 A for nm-deep contacts).  The recipe is FROZEN
  (critic item A5; ``GATE_I_*``, :func:`frozen_gate_i_recipe`, outputs/ready/frozen_gate_i.json):
  conjugate gradient to max |F| <= 1e-3 eV/A (harness :func:`jax_minimizer` or LAMMPS :func:`lammps_minimizer`;
  FIRE can relax a marginal onset defect away), a cylinder of radius and depth 1.5 a_c with a fixed 8 A shell,
  a_c = sqrt(A_tip(h) / pi) of the quenched frame, one :func:`gate_i` call per frame of a depth series
  (:func:`gate_i_contact_radius`; 2.7923 h for the ideal Berkovich).  Verdicts: ``reversible`` (no changed heteronuclear-only coordination, |dPE| < 0.1 eV, no
  atom displaced > 0.5 A); ``irreversible`` = not reversible (the G05 criterion, a diagnostic only);
  ``passes`` = >= 10 changed atoms AND dPE >= 1 eV (the Gate I verdict).  On the G05 R20 frames the frozen
  carve (8,430 atoms) gives the whole-cell (20,480 atoms) pass verdicts with both backends: 5.4 and 6.0 A do not
  pass, 6.6 A passes (21 atoms, 12.83 eV; whole cell 21 atoms, 12.66 eV).  The whole-cell CG quench alone
  leaves a sub-threshold defect at 6.0 A (5 atoms, 1.15 eV; LAMMPS CG 3 atoms, 1.55 eV) that every smaller
  carve and FIRE quench away, so ``irreversible`` at the onset is carve- and minimizer-dependent.
  ``defects_contained`` flags a carve too small for its defects (a changed atom within 5 A of the shell).
  Evaluation rule (critic N1 / N2; :func:`gate_i_evaluation`, :func:`gate_i_evaluation_rule`): a frame's verdict is
  "pass" or "fail" only when its quench AND the reference quench converged and its defects are contained; otherwise
  it is "not_evaluated" (``passes`` None), never a pass or a silent fail.  :func:`gate_i_frame` quenches an
  uncontained (converged) frame again in enlarged carves: radius and depth factors x 2^(1/3) per attempt (the volume
  doubles), up to 2 x the frozen factors (1.5 -> 1.89 -> 2.38 -> 3.0 a_c), recording the carve actually used; still
  uncontained at the cap, it is not evaluated.  The first-irreversibility depth (:func:`gate_i_first_irreversibility`)
  is NOT EVALUATED when a frame shallower than the first pass is not evaluated, and a not-evaluated frame is a
  not-evaluated Gate II evidence row that makes the Gate II onset NOT EVALUATED where it decides it.
* Gate II, developed plasticity (required for a hardness): residual impression at the contact centre
  >= max(2 A, 0.1 h_max), loop dissipation >= 10 %, h_f / h_max >= 0.3 (thresholds PROVISIONAL until they
  are frozen in recipe_v3 after the S1 scoping run; :class:`GateIIThresholds`, :func:`gate_ii`), evaluated
  after the full unload from h_max.  Its ONSET depth, which Gate III needs, comes from intermediate-depth
  evidence (protocol item 17: Gate I quenched residual impressions or unload branches, supplied as
  ``IndentationRunData.gate_ii_evidence``; :func:`gate_ii_onset`, :func:`gate_ii_evidence_from_gate_i`).
  A run without such evidence cannot produce a hardness: an onset placed at h_max always lies inside the
  Gate III window, and the analysis says so (``hardness.reason``).
* Gate III, plateau (LM-R10): p_m fitted linearly over the last 40 % of loading, window after the Gate II
  onset; |dp_m/dh| x window / p_m < 5 % on BOTH E3 and E4; H = mean E3 p_m over the window
  (:func:`plateau_metric`, :func:`gate_iii`).

Diagnostics
-----------
* Heteronuclear-only coordination (Si-C < 2.4 A; compressed homonuclear second neighbours are never bonds,
  analysis.local_strain.VASHISHTA_SIC_BONDS) and coordination classes (:func:`coordination_changes`).
* Species-balanced local stress sigma = (<S>_Si + <S>_C) / (<V>_Si + <V>_C) (the Vashishta per-atom virial
  carries a +-25 GPa isotropic Si/C offset) and the stoichiometric-layer force balance
  -sum S_zz / t = P (:func:`species_balanced_stress`, :func:`layer_force_balance`).
* Oliver-Pharr (epsilon 0.75, beta 1.05, fit to 20-95 % of the unloading curve): a DIAGNOSTIC, never the
  primary hardness (H_OP = 77 GPa for the elastic G06 contact; :func:`oliver_pharr`).
* Defect extent: the connected cluster of changed atoms at the contact, its distance to the fixed and
  thermostat layers and the gap to its periodic image (acceptance >= t_el/2 and >= L/3;
  :func:`defect_extent`).
* Newton-3: the indenter load against the fixed-layer reaction (:func:`newton3_balance`).
* The finite-cell (box) factor comes from :mod:`fracture_md.physics.contact_box`.

:func:`analyze_indentation` runs all of it on an :class:`IndentationRunData` (dense load-depth log plus a
lazy frame source) and returns a JSON-serialisable dict; physics/indentation.py turns that into an
``IndentationCurve`` and ``*_MD`` records with the status ceiling applied.

Units: A, ps, nN (loads), GPa (pressures), eV (energies).  1 eV/A = 1.602176634 nN; 1 nN/A^2 = 100 GPa.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg as sparse_cg
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull, QhullError, cKDTree

from fracture_md.analysis.local_strain import BondRule, bond_rule_for, coordination_by_rule, min_image

EV_PER_A_TO_NN = 1.602176634  # exact
GPA_PER_NN_PER_A2 = 100.0  # 1 nN / A^2 = 1e-9 N / 1e-20 m^2 = 1e11 Pa
NN_PER_BAR_A2 = 1.0e-6  # 1 bar A^2 = 1e5 Pa x 1e-20 m^2 = 1e-15 N
EV_PER_BAR_A3 = 1.0e5 * 1.0e-30 / 1.602176634e-19  # LAMMPS stress/atom (metal units) -> eV
BERKOVICH_FACE_ANGLE_DEG = 65.27
BERKOVICH_C0 = 3.0 * math.sqrt(3.0) * math.tan(math.radians(BERKOVICH_FACE_ANGLE_DEG)) ** 2  # 24.494
EQUIVALENT_CONE_HALF_ANGLE_DEG = math.degrees(math.atan(math.sqrt(BERKOVICH_C0 / math.pi)))  # 70.296
DEFAULT_WINDOW_PS = 0.5
DEFAULT_TAU_AC_PS = 0.020  # load autocorrelation time at 300 K (G06/G10: ~20 fs)
MIN_WINDOW_OVER_TAU = 5.0
CLOSING_RADIUS_A = 3.0  # E4 morphological closing
PLATEAU_TOL = 0.05  # LM-R10: |dp_m/dh| x window / p_m
PLATEAU_WINDOW_FRACTION = 0.4
EVALUATION_DEPTH_TOL_A = 0.05  # a loading branch this far short of a half-angstrom depth is evaluated there

# estimator code -> schema AreaEstimator name (schema.common)
ESTIMATOR_NAMES: dict[str, str] = {
    "E1": "nominal_ideal",
    "E2": "rounded_area_function",
    "E3": "dilated_hull",
    "E4": "closed_discs",
    "E5": "triangle_fit",
}
ESTIMATORS: tuple[str, ...] = tuple(ESTIMATOR_NAMES)
PRIMARY_ESTIMATOR = "E3"
CHECK_ESTIMATOR = "E4"
PLATEAU_EVIDENCE_ESTIMATORS = ("E3", "E4")  # E1, E2, E5 are reported but never used as plateau evidence
# B1 estimated contact: a touching atom counts if its tip force >= this fraction of the mean tip force of the touching
# atoms (the half-cell rule; calibrated on 0 K Vashishta AQS spheres against the box BEM, module docstring)
CONTACT_FORCE_FRACTION = 0.5
CONTACT_RULES = ("force_fraction", "touching")


def estimator_code(name: str) -> str:
    """'E3' for 'E3' or 'dilated_hull' (schema AreaEstimator names are accepted)."""
    if name in ESTIMATOR_NAMES:
        return name
    for k, v in ESTIMATOR_NAMES.items():
        if v == name:
            return k
    raise ValueError(f"unknown area estimator {name!r}; known: {ESTIMATOR_NAMES}")


def surface_dilation_A(lattice_constant_A: float) -> float:
    """E3/E4 dilation d = a / (2 sqrt 2): half the (001) surface nearest-neighbour spacing a/sqrt 2."""
    return float(lattice_constant_A) / (2.0 * math.sqrt(2.0))


def mean_pressure_GPa(load_nN: Any, area_A2: Any) -> np.ndarray:
    """p_m = P / A in GPa (NaN where the area is not positive and finite)."""
    P = np.asarray(load_nN, float)
    A = np.asarray(area_A2, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(A) & (A > 1.0), GPA_PER_NN_PER_A2 * P / A, np.nan)


# ============================================================================================ loads
def running_mean(x: Any, n: int) -> np.ndarray:
    """Centred moving average over ``n`` samples, renormalised at the ends (numpy convolve 'same')."""
    x = np.asarray(x, float)
    n = max(1, min(int(n), len(x)))  # a window longer than the series would lengthen the output (r4-gates R1b)
    k = np.ones(n) / n
    y = np.convolve(x, k, mode="same")
    w = np.convolve(np.ones_like(x), k, mode="same")
    return y / w


def integrated_autocorrelation_time(x: Any, dt: float, *, detrend_samples: Optional[int] = None,
                                    c: float = 5.0) -> float:
    """Integrated autocorrelation time of a stationary series (same unit as ``dt``).

    tau = dt (1/2 + sum_{k>=1} rho_k), summed with Sokal's automatic window (stop at the first k >= c tau_k).
    ``detrend_samples``: subtract a running mean of that many samples first (loads during loading have a
    deterministic trend).  A series sampled more coarsely than its correlation gives ~dt/2: sample finely
    (every step) to resolve tau_ac ~ 20 fs.
    """
    y = np.asarray(x, float)
    if len(y) < 8:  # before the detrending: an empty series would reach np.convolve (r4-gates R1)
        return float("nan")
    if detrend_samples:
        y = y - running_mean(y, detrend_samples)
    y = y - y.mean()
    n = len(y)
    if not np.any(y):
        return float("nan")
    f = np.fft.rfft(y, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf / acf[0]
    tau = 0.5
    for k in range(1, n):
        tau += acf[k]
        if k >= c * tau:
            break
    return float(max(tau, 0.5) * dt)


@dataclasses.dataclass
class WindowedLoad:
    """A centred running mean of the load and the checks on its window."""

    values: np.ndarray  # nN, same length as the input
    window_ps: float
    n_samples: int
    sample_dt_ps: float
    tau_ac_ps: float
    window_over_tau: float
    ok: bool  # window >= MIN_WINDOW_OVER_TAU tau_ac

    def summary(self) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if k != "values"}


def windowed_load(t_ps: Any, load_nN: Any, window_ps: float = DEFAULT_WINDOW_PS, *,
                  tau_ac_ps: Optional[float] = None, min_window_over_tau: float = MIN_WINDOW_OVER_TAU) -> WindowedLoad:
    """Centred running mean of the load over ``window_ps`` (recommended protocol item 15, 0.5 ps).

    The number of samples is round(window / median dt).  ``tau_ac_ps`` defaults to 20 fs (the measured load
    autocorrelation time at 300 K; pass a value from :func:`integrated_autocorrelation_time` on a finely
    sampled log); the window must cover at least ``min_window_over_tau`` autocorrelation times.
    """
    t = np.asarray(t_ps, float)
    P = np.asarray(load_nN, float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else window_ps
    n = max(1, int(round(window_ps / dt)))
    tau = DEFAULT_TAU_AC_PS if tau_ac_ps is None or not np.isfinite(tau_ac_ps) else float(tau_ac_ps)
    ratio = window_ps / tau if tau > 0 else float("inf")
    return WindowedLoad(running_mean(P, n), float(window_ps), n, dt, tau, float(ratio),
                        bool(ratio >= min_window_over_tau))


def block_average(t_ps: Any, x: Any, block_ps: float) -> dict[str, np.ndarray]:
    """Non-overlapping block means of ``x`` (blocks of ``block_ps``; the incomplete last block is dropped).

    Returns t_mid, mean, sem (standard error of the samples in each block) and n per block.
    """
    t = np.asarray(t_ps, float)
    x = np.asarray(x, float)
    if len(t) == 0:
        return {"t_mid": np.zeros(0), "mean": np.zeros(0), "sem": np.zeros(0), "n": np.zeros(0, int)}
    b = np.floor((t - t[0]) / block_ps + 1e-9).astype(int)
    nb = int(b.max())  # the last (possibly incomplete) block is dropped
    tm, mu, se, nn = [], [], [], []
    for k in range(nb):
        m = b == k
        if m.sum() == 0:
            continue
        v = x[m]
        tm.append(t[m].mean())
        mu.append(v.mean())
        se.append(v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else np.nan)
        nn.append(len(v))
    return {"t_mid": np.array(tm), "mean": np.array(mu), "sem": np.array(se), "n": np.array(nn, int)}


def in_contact_scatter(P: Any, Pw: Any, t_ps: Any, *, mask: Optional[Any] = None, slow_window_ps: float = 2.0,
                       min_load_nN: float = 20.0) -> dict[str, float]:
    """Scatter of the raw and windowed load about a slow (2 ps) running mean while in contact (G06 rule)."""
    P = np.asarray(P, float)
    Pw = np.asarray(Pw, float)
    t = np.asarray(t_ps, float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    slow = running_mean(P, max(1, int(round(slow_window_ps / dt))))
    m = P > min_load_nN
    if mask is not None:
        m &= np.asarray(mask, bool)
    if m.sum() <= 20:
        return {"raw_nN": 0.0, "windowed_nN": 0.0, "n": int(m.sum())}
    return {"raw_nN": float(np.std((P - slow)[m])), "windowed_nN": float(np.std((Pw - slow)[m])), "n": int(m.sum())}


def load_drop_events(h: Any, Pw: Any, *, rel_drop: float = 0.05, noise_nN: float = 0.0, min_peak_nN: float = 20.0,
                     t_ps: Optional[Any] = None) -> list[dict[str, float]]:
    """Pop-in candidates on a loading branch: the windowed load falls below its running maximum by more than
    max(rel_drop x max, 3 noise).  Consecutive samples form one event; the peak before it is reported."""
    h = np.asarray(h, float)
    Pw = np.asarray(Pw, float)
    runmax = np.maximum.accumulate(Pw)
    drop = runmax - Pw
    thr = np.maximum(rel_drop * runmax, 3.0 * noise_nN)
    cand = np.where((drop > thr) & (runmax > min_peak_nN))[0]
    out = []
    if len(cand):
        for e in np.split(cand, np.where(np.diff(cand) > 1)[0] + 1):
            ipk = int(np.argmax(Pw[:e[0] + 1]))
            ev = dict(h_peak_A=float(h[ipk]), P_peak_nN=float(Pw[ipk]), h_drop_A=float(h[e[0]]),
                      max_drop_nN=float(drop[e].max()), max_drop_frac=float((drop[e] / runmax[e]).max()))
            if t_ps is not None:
                ev["t_ps"] = float(np.asarray(t_ps, float)[e[0]])
            out.append(ev)
    return out


def newton3_balance(load_nN: Any, reaction_nN: Any, *, min_load_nN: float = 20.0) -> dict[str, float]:
    """Indenter load vs the fixed-layer reaction (pre-zero force sum on the frozen atoms; both +z on the tip
    side convention of LAMMPS: the reaction on the fixed layer is -P in equilibrium).  Returns the mean
    residual <P + R> / <P> over samples in contact (tolerance 2 %, recommended protocol item 15)."""
    P = np.asarray(load_nN, float)
    R = np.asarray(reaction_nN, float)
    m = P > min_load_nN
    if not m.any():
        return {"mean_residual_over_mean_P": float("nan"), "rms_residual_nN": float("nan"), "n": 0}
    res = P + R
    return {"mean_residual_over_mean_P": float(res[m].mean() / P[m].mean()),
            "rms_residual_nN": float(np.sqrt((res[m] ** 2).mean())), "n": int(m.sum())}


# ============================================================================================ depth
def contact_depth(h_A: Any, pen_fw_A: Any) -> np.ndarray:
    """h_c = h - pen_fw: the nominal depth corrected by the force-weighted wall penetration (item 4)."""
    return np.asarray(h_A, float) - np.asarray(pen_fw_A, float)


def force_weighted_penetration(delta_A: Any) -> float:
    """pen_fw = sum delta^3 / sum delta^2 over the penetrations delta = max(0, -sdf) (0 without contact)."""
    d = np.clip(np.asarray(delta_A, float), 0.0, None)
    s2 = float(np.sum(d * d))
    return float(np.sum(d ** 3) / s2) if s2 > 0 else 0.0


def first_contact_depth(h: Any, Pw: Any, *, noise_nN: float = 0.0, n_sigma: float = 3.0, floor_nN: float = 1.0,
                        mask: Optional[Any] = None) -> Optional[float]:
    """Depth-zero rule 1: the first depth at which the windowed load exceeds max(n_sigma x noise, floor)."""
    h = np.asarray(h, float)
    Pw = np.asarray(Pw, float)
    m = np.ones(len(h), bool) if mask is None else np.asarray(mask, bool)
    idx = np.where(m & (Pw > max(n_sigma * noise_nN, floor_nN)))[0]
    return float(h[idx[0]]) if len(idx) else None


def sneddon_cone_load(h: Any, M_GPa: float, h0_A: float, half_angle_deg: float = EQUIVALENT_CONE_HALF_ANGLE_DEG):
    """Sneddon rigid cone on an elastic half-space: P = (2/pi) M tan(alpha) (h - h0)^2 (nN; M in GPa)."""
    return (2.0 / math.pi) * M_GPa * math.tan(math.radians(half_angle_deg)) * np.clip(
        np.asarray(h, float) - h0_A, 0.0, None) ** 2 / GPA_PER_NN_PER_A2


def sneddon_cone_fit(h: Any, P: Any, *, Pw: Optional[Any] = None, load_range_nN: tuple[float, float] = (10.0, 150.0),
                     h_max_A: Optional[float] = None,
                     half_angle_deg: float = EQUIVALENT_CONE_HALF_ANGLE_DEG) -> dict[str, Any]:
    """Depth-zero rule 2 for SHARP tips: fit the early loading branch with the Sneddon cone (M, h0).

    Samples with the windowed load ``Pw`` (default ``P``) inside ``load_range_nN`` and depth below
    ``h_max_A`` (the first pop-in, if any) are fitted with the raw load (G06 analysis).  A rounded tip needs
    :func:`sneddon_fit` (this fit would absorb its bluntness into h0).
    """
    h = np.asarray(h, float)
    P = np.asarray(P, float)
    Pw = P if Pw is None else np.asarray(Pw, float)
    sel = (Pw > load_range_nN[0]) & (Pw < load_range_nN[1])
    if h_max_A is not None:
        sel &= h < h_max_A
    if sel.sum() < 3:
        return {"error": f"only {int(sel.sum())} samples in the fit range"}
    f = lambda hh, M, h0: sneddon_cone_load(hh, M, h0, half_angle_deg)  # noqa: E731
    try:
        popt, pcov = curve_fit(f, h[sel], P[sel], p0=[300.0, -1.0])
    except (RuntimeError, ValueError) as exc:
        return {"error": str(exc)}
    return {"M_GPa": float(popt[0]), "M_err_GPa": float(np.sqrt(pcov[0, 0])), "h0_A": float(popt[1]),
            "h0_err_A": float(np.sqrt(pcov[1, 1])), "n": int(sel.sum()),
            "h_range_A": [float(h[sel].min()), float(h[sel].max())], "half_angle_deg": float(half_angle_deg),
            "tip_radius_A": 0.0, "model": "Sneddon cone, P = (2/pi) M tan(alpha) (h - h0)^2"}


def equivalent_cone_half_angle_deg(geometry: Any = None, C0: float = BERKOVICH_C0) -> float:
    """Half-angle of the cone with the tip's ideal area function C0 h^2: atan(sqrt(C0 / pi)) (70.296 deg for the
    65.27 deg Berkovich); a cone's own half-angle.  ``geometry`` None: from ``C0``."""
    if geometry is not None:
        if geometry.shape == "sphere":
            raise ValueError("a sphere has no equivalent cone (use hertz_fit)")
        if geometry.shape == "cone":
            return float(geometry.half_angle_deg)
        C0 = float(geometry.ideal_area_coefficient)
    return math.degrees(math.atan(math.sqrt((C0 if C0 > 0.0 else BERKOVICH_C0) / math.pi)))


SNEDDON_NODES = 256  # Gauss-Legendre nodes in t = asin(r/a) (1e-6 relative against adaptive quadrature, tested)
SNEDDON_CONTACT_RADII = 2000  # contact radii tabulated per curve


def _is_sharp(geometry: Any) -> bool:
    return geometry is None or float(geometry.tip_radius_A) <= 0.0


@functools.lru_cache(maxsize=16)
def sneddon_profile_table(geometry: Any, h_top_A: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sneddon (1965) rigid axisymmetric punch on an elastic half-space for the tip's AREA-EQUIVALENT profile.

    The profile is r_eq(z) = sqrt(A(z) / pi) with A the exact tip area function (physics.indenter.
    tip_area_function): the rounded cone exactly, the rounded Berkovich through its area function (the usual
    equivalent-axisymmetric-indenter approximation; R = 0 gives the equivalent cone).  With x = sin t and
    f'(r) = dz/dr of the profile,

        h(a) = a int_0^{pi/2} f'(a sin t) dt,   P(a) / M = 2 a^2 int_0^{pi/2} sin^2 t f'(a sin t) dt,

    (a = contact radius; the cone gives Sneddon's (2/pi) M tan(alpha) h^2, the apex sphere Hertz).  Returns
    (h in A, P per unit modulus in nN/GPa, a in A), tabulated up to a depth above ``h_top_A`` (cached per
    geometry and depth).
    """
    from fracture_md.physics.indenter import tip_area_function

    if geometry.shape == "sphere":
        raise ValueError("use hertz_fit for a sphere")

    def area(z):
        return np.asarray(tip_area_function(z, geometry), float)

    a_max = math.sqrt(float(area(h_top_A)) / math.pi)  # h(a = r_eq(h_top)) > h_top: the contact sinks in
    z = 1.001 * h_top_A * np.linspace(0.0, 1.0, 20001) ** 2  # dense at the apex
    dz = 1e-6 * max(float(geometry.tip_radius_A), 1.0)
    zl = np.maximum(z - dz, 0.0)
    dAdz = (area(z + dz) - area(zl)) / (z + dz - zl)
    r = np.sqrt(area(z) / math.pi)
    slope = 2.0 * math.pi * r / dAdz  # dz/dr = 2 pi r / (dA/dz): 0 at the apex, cot(alpha_eq) far from it
    t, w = np.polynomial.legendre.leggauss(SNEDDON_NODES)
    t = (t + 1.0) * (math.pi / 4.0)
    w = w * (math.pi / 4.0)
    x = np.sin(t)
    a = a_max * (np.arange(1, SNEDDON_CONTACT_RADII + 1) / SNEDDON_CONTACT_RADII) ** 1.5
    S = np.interp(a[:, None] * x[None, :], r, slope)
    h = a * (S @ w)
    p = 2.0 * a * a * (S @ (w * x * x)) / GPA_PER_NN_PER_A2
    for v in (h, p, a):
        v.setflags(write=False)
    return h, p, a


@functools.lru_cache(maxsize=16)
def _sneddon_unit(geometry: Any, h_top_A: float) -> Callable[[np.ndarray], np.ndarray]:
    """P per unit modulus (nN/GPa) at depths x (A) below the tip contact: a log-log spline of the table."""
    h, p, _ = sneddon_profile_table(geometry, h_top_A)
    spl = CubicSpline(np.log(h), np.log(p))
    h1, p1, hn, pn = float(h[0]), float(p[0]), float(h[-1]), float(p[-1])

    def unit(x):
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        lo = (x > 0.0) & (x < h1)
        mid = (x >= h1) & (x <= hn)
        hi = x > hn
        out[lo] = p1 * (x[lo] / h1) ** 1.5  # the apex sphere (Hertz)
        out[mid] = np.exp(spl(np.log(x[mid])))
        out[hi] = pn * (x[hi] / hn) ** 2  # beyond the table (not reached by sneddon_fit): the cone scaling
        return out

    return unit


def _sneddon_h_top(x_max: float) -> float:
    return 10.0 * math.ceil((max(float(x_max), 0.0) + 10.0) / 10.0)  # >= 10 A of margin for h0, shared cache keys


def sneddon_tip_load(h: Any, M_GPa: float, h0_A: float, geometry: Any = None, *, C0: float = BERKOVICH_C0) -> np.ndarray:
    """Sneddon elastic load (nN; M in GPa) of the tip's own shape at depth h with zero h0: the equivalent cone
    (:func:`sneddon_cone_load`) for a sharp tip or no geometry, else :func:`sneddon_profile_table`."""
    if _is_sharp(geometry):
        return sneddon_cone_load(h, M_GPa, h0_A, equivalent_cone_half_angle_deg(geometry, C0))
    x = np.clip(np.asarray(h, float) - h0_A, 0.0, None)
    return M_GPa * _sneddon_unit(geometry, _sneddon_h_top(float(np.max(x, initial=0.0))))(x)


def sneddon_fit(h: Any, P: Any, *, geometry: Any = None, Pw: Optional[Any] = None,
                load_range_nN: tuple[float, float] = (10.0, 150.0), h_max_A: Optional[float] = None,
                C0: float = BERKOVICH_C0) -> dict[str, Any]:
    """Depth-zero rule 2 for pyramids and cones: fit the early loading branch with Sneddon's elastic solution for
    the tip's OWN shape, P = M p(h - h0) (M, h0; same sample selection as :func:`sneddon_cone_fit`).

    A sharp tip (tip_radius_A = 0, or no geometry: ideal area ``C0`` h^2) is the equivalent cone (analytic).  A
    rounded tip uses :func:`sneddon_profile_table`, so h0 is the surface zero, not the tip bluntness, and E2 =
    A_rounded(h_c - h0) and the modulus M are unbiased.  A sharp-cone fit of a rounded tip absorbs the apex
    offset into h0 (-0.55 A and M +2.8 % for the continuum R = 10 A cone) and E2 then counts the bluntness twice
    (+10 % area at 1 nm); it is kept as ``sharp_cone_fit`` (diagnostic).
    """
    if geometry is not None and geometry.shape == "sphere":
        raise ValueError("use hertz_fit for a sphere")
    sharp = sneddon_cone_fit(h, P, Pw=Pw, load_range_nN=load_range_nN, h_max_A=h_max_A,
                             half_angle_deg=equivalent_cone_half_angle_deg(geometry, C0))
    if _is_sharp(geometry) or "error" in sharp:
        return sharp
    h = np.asarray(h, float)
    P = np.asarray(P, float)
    Pw = P if Pw is None else np.asarray(Pw, float)
    sel = (Pw > load_range_nN[0]) & (Pw < load_range_nN[1])
    if h_max_A is not None:
        sel &= h < h_max_A
    unit = _sneddon_unit(geometry, _sneddon_h_top(float(h[sel].max() - min(sharp["h0_A"], 0.0))))
    f = lambda hh, M, h0: M * unit(hh - h0)  # noqa: E731
    try:
        popt, pcov = curve_fit(f, h[sel], P[sel], p0=[sharp["M_GPa"], sharp["h0_A"]])
    except (RuntimeError, ValueError) as exc:
        return {"error": str(exc), "sharp_cone_fit": sharp}
    return {"M_GPa": float(popt[0]), "M_err_GPa": float(np.sqrt(pcov[0, 0])), "h0_A": float(popt[1]),
            "h0_err_A": float(np.sqrt(pcov[1, 1])), "n": int(sel.sum()),
            "h_range_A": [float(h[sel].min()), float(h[sel].max())], "half_angle_deg": sharp["half_angle_deg"],
            "tip_radius_A": float(geometry.tip_radius_A),
            "model": f"Sneddon axisymmetric punch, area-equivalent profile of the {geometry.label} tip",
            "sharp_cone_fit": {k: sharp[k] for k in ("M_GPa", "h0_A", "model")}}


def hertz_load(h: Any, M_GPa: float, h0_A: float, R_A: float) -> np.ndarray:
    """Hertz sphere: P = (4/3) M sqrt(R) (h - h0)^1.5 (nN; M in GPa)."""
    return 4.0 / 3.0 * M_GPa * math.sqrt(R_A) * np.clip(np.asarray(h, float) - h0_A, 0.0, None) ** 1.5 / GPA_PER_NN_PER_A2


def hertz_fit(h: Any, P: Any, R_A: float, h_max_A: float, *, h0_A: Optional[float] = None,
              h0_grid: tuple[float, float, int] = (-1.5, 1.5, 3001)) -> dict[str, Any]:
    """Least-squares Hertz fit P = c (h - h0)^1.5 over h <= h_max_A with h0 on a 0.001 A grid (the estimator of
    verify/repro-lammps-reference and G10 box_bem.fit_hertz); returns M (GPa) and h0 (A)."""
    h = np.asarray(h, float)
    P = np.asarray(P, float)
    m = h <= h_max_A + 1e-12
    grid = [float(h0_A)] if h0_A is not None else np.linspace(*h0_grid)
    best = None
    for g in grid:
        x = np.clip(h[m] - g, 0.0, None) ** 1.5
        xx = float(x @ x)
        if xx <= 0:
            continue
        c = float(x @ P[m]) / xx
        s = float(((P[m] - c * x) ** 2).sum())
        if best is None or s < best[0]:
            best = (s, g, c)
    if best is None:
        return {"error": "no samples above h0"}
    M = best[2] * GPA_PER_NN_PER_A2 / (4.0 / 3.0 * math.sqrt(R_A))
    return {"M_GPa": float(M), "h0_A": float(best[1]), "rms_nN": float(math.sqrt(best[0] / max(1, m.sum()))),
            "n": int(m.sum()), "h_max_A": float(h_max_A), "R_A": float(R_A)}


# ============================================================================================ areas
def area_nominal(h_A: Any, C0: float = BERKOVICH_C0) -> np.ndarray:
    """E1: C0 max(h, 0)^2 (ideal sharp tip at the nominal depth)."""
    return C0 * np.clip(np.asarray(h_A, float), 0.0, None) ** 2


def area_rounded(h_eff_A: Any, geometry: Any = None, C0: float = BERKOVICH_C0) -> np.ndarray:
    """E2: the exact tip area function at h_eff = h_c - h0 (``geometry``: physics.indenter.IndenterGeometry;
    None = the sharp tip C0 h_eff^2)."""
    if geometry is None:
        return area_nominal(h_eff_A, C0)
    return np.asarray(geometry.area(np.asarray(h_eff_A, float)), float)


def area_dilated_hull(xy: Any, d_A: float) -> float:
    """E3: area of the convex hull of the projected contact atoms dilated by ``d_A`` (A_h + P_h d + pi d^2).

    Collinear points count as a segment of twice their extent as perimeter; one atom gives pi d^2; none, 0.
    Identical to viz.indentviz.contact_outline (the README estimated-contact outline).
    """
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) == 0:
        return 0.0
    A = Pr = 0.0
    if len(xy) >= 3:
        try:
            hull = ConvexHull(xy)
            A, Pr = float(hull.volume), float(hull.area)
        except QhullError:
            A = Pr = 0.0
    if A == 0.0 and len(xy) >= 2:
        ext = xy - xy.mean(axis=0)
        _, _, vt = np.linalg.svd(ext, full_matrices=False)
        s = ext @ vt[0]
        Pr = 2.0 * float(s.max() - s.min())
    return A + Pr * d_A + math.pi * d_A * d_A


def area_closed_discs(xy: Any, d_A: float, *, closing_radius_A: float = CLOSING_RADIUS_A,
                      grid_A: float = 0.2) -> float:
    """E4: union of discs of radius ``d_A`` about the contact atoms after a morphological closing.

    Rasterised on a ``grid_A`` grid: dilate the atom set by ``closing_radius_A``, fill holes, erode by
    ``closing_radius_A - d_A`` (exact Euclidean distance transform) -- the G06 analysis.
    """
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) == 0:
        return 0.0
    Rc = float(closing_radius_A)
    g = float(grid_A)
    lo = xy.min(0) - Rc - 1.0
    hi = xy.max(0) + Rc + 1.0
    gx = np.arange(lo[0], hi[0], g)
    gy = np.arange(lo[1], hi[1], g)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    dist, _ = cKDTree(xy).query(np.stack([GX.ravel(), GY.ravel()], 1), distance_upper_bound=Rc)
    dil = ndimage.binary_fill_holes((dist <= Rc).reshape(GX.shape))
    closed = ndimage.distance_transform_edt(dil, sampling=g) > (Rc - d_A)
    return float(closed.sum() * g * g)


def area_triangle_fit(xy: Any, d_A: float, face_azimuths_deg: Sequence[float] = (0.0, 120.0, 240.0)) -> tuple[float, list[float]]:
    """E5: 3 sqrt3 (r_in + d)^2 with r_in the mean over the face normals of the largest projection of a contact
    atom (a triangle of inradius r_in + d); returns (area, per-face projections)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) == 0:
        return 0.0, [0.0, 0.0, 0.0]
    phi = np.radians(np.asarray(face_azimuths_deg, float))
    proj = xy @ np.stack([np.cos(phi), np.sin(phi)], 1).T
    per_face = proj.max(0)
    rin = float(per_face.mean())
    return 3.0 * math.sqrt(3.0) * (rin + d_A) ** 2, [float(v) for v in per_face]


def contact_areas(xy: Any, h_A: float, *, h0_A: float = 0.0, h_c_A: Optional[float] = None, d_A: float,
                  C0: float = BERKOVICH_C0, geometry: Any = None,
                  face_azimuths_deg: Sequence[float] = (0.0, 120.0, 240.0),
                  closing_radius_A: float = CLOSING_RADIUS_A, grid_A: float = 0.2,
                  min_contact_atoms: int = 3) -> dict[str, Any]:
    """All five estimators for one frame (areas in A^2; E3-E5 NaN below ``min_contact_atoms`` atoms).

    ``xy``: projected contact-atom positions RELATIVE TO THE TIP AXIS (lateral minimum image applied);
    ``h_A``: nominal apex depth; ``h_c_A``: contact depth h - pen_fw (default h_A); ``h0_A``: elastic zero.
    """
    xy = np.asarray(xy, float).reshape(-1, 2)
    hc = h_A if h_c_A is None else h_c_A
    out: dict[str, Any] = {"n_contact": int(len(xy)), "E1": float(area_nominal(h_A, C0)),
                           "E2": float(area_rounded(max(hc - h0_A, 0.0), geometry, C0))}
    if len(xy) < min_contact_atoms:
        out.update(E3=float("nan"), E4=float("nan"), E5=float("nan"), tri_per_face_A=None)
        return out
    out["E3"] = area_dilated_hull(xy, d_A)
    out["E4"] = area_closed_discs(xy, d_A, closing_radius_A=closing_radius_A, grid_A=grid_A)
    out["E5"], out["tri_per_face_A"] = area_triangle_fit(xy, d_A, face_azimuths_deg)
    return out


def estimated_contact_mask(touching: Any, force: Optional[Any], *, fraction: float = CONTACT_FORCE_FRACTION,
                           ) -> np.ndarray:
    """The estimated-contact atoms (module docstring, critic item B1): the ``touching`` atoms whose tip force is at
    least ``fraction`` of the mean tip force of the touching atoms.

    ``force``: per-atom tip force magnitude (any positive scale: K delta^2 or delta^2 for an SDF wall, the summed
    WCA force for an atomistic tip), same length as ``touching``; None or no touching atom with a positive force:
    every touching atom is returned (the caller records that the rule could not be applied).  ``fraction`` 0 keeps
    every touching atom with a positive force.
    """
    t = np.asarray(touching, bool)
    if force is None or not t.any():
        return t.copy()
    f = np.asarray(force, float)
    if f.shape != t.shape:
        raise ValueError(f"force has shape {f.shape}, touching {t.shape}")
    ft = f[t]
    if not np.any(ft > 0.0):
        return t.copy()
    return t & (f >= float(fraction) * float(ft.mean())) & (f > 0.0)


def sdf_penetration(positions: Any, geometry: Any, apex: Sequence[float],
                    lateral_L: Optional[Sequence[float]] = None) -> np.ndarray:
    """Wall penetration delta = max(0, -sdf) (A) of every atom for an SDF tip whose lowest point is at ``apex``
    (x, y, z); the tip force of an atom is K delta^2 (U = (K/3) delta^3).  Lateral minimum image if ``lateral_L``."""
    from fracture_md.io.indent_traj import Indenter

    ind = Indenter.from_dict(geometry.to_dict())
    d = np.asarray(positions, float) - np.asarray(apex, float)
    if lateral_L is not None:
        L = np.asarray(lateral_L, float)
        d[:, :2] -= L * np.round(d[:, :2] / L)
    return np.clip(-np.asarray(ind.sdf(d), float), 0.0, None)


def projected_contact_xy(positions: Any, contact: Any, axis_xy: Sequence[float],
                         lateral_L: Optional[Sequence[float]] = None) -> np.ndarray:
    """(n, 2) positions of the contact atoms relative to the tip axis (lateral minimum image if L given)."""
    p = np.asarray(positions, float)[np.asarray(contact, bool), :2] - np.asarray(axis_xy, float)
    if lateral_L is not None:
        L = np.asarray(lateral_L, float)
        p -= L * np.round(p / L)
    return p


# ============================================================================================ plateau
def evaluation_depth(h_max_A: float, tol_A: float = EVALUATION_DEPTH_TOL_A) -> float:
    """The half-angstrom evaluation depth at h_max: floor(2 (h_max + tol)) / 2.

    A loading branch that stops a logging interval or a rounded step count short of a half-angstrom depth (a
    LAMMPS deck logs every ``logevery`` steps and moves for round((gap + hmax) / (v dt)) steps) is evaluated at
    that depth, not 0.5 A lower.  Protocol depths are half-angstrom values; a max depth just below one (e.g.
    7.46 A) is evaluated at the next one, beyond the deepest sample, by at most ``tol_A``.
    """
    return float(np.floor(2.0 * (float(h_max_A) + tol_A)) / 2.0)


def plateau_metric(h: Any, pm: Any, h_eval: float, *, window_fraction: float = PLATEAU_WINDOW_FRACTION,
                   h_onset: Optional[float] = None, tol: float = PLATEAU_TOL) -> dict[str, Any]:
    """LM-R10 plateau metric on loading samples: linear fit of p_m over [(1 - f) h_eval, h_eval].

    metric = |dp_m/dh| x f h_eval / mean(p_m); ``slope_ok`` = metric < tol; ``onset_before_window`` = a
    plasticity onset (Gate II depth, or first pop-in) at or before the window start; ``passes`` needs both.
    """
    h = np.asarray(h, float)
    pm = np.asarray(pm, float)
    lo = (1.0 - window_fraction) * h_eval
    sel = (h <= h_eval + 1e-6) & (h >= lo - 1e-6) & np.isfinite(pm)
    hh, pp = h[sel], pm[sel]
    if len(hh) < 3:
        return {"h_window_A": [float(lo), float(h_eval)], "n": int(len(hh))}
    A = np.vstack([hh, np.ones_like(hh)]).T
    coef, *_ = np.linalg.lstsq(A, pp, rcond=None)
    b = float(coef[0])
    resid = pp - A @ coef
    s2 = float((resid ** 2).sum()) / max(1, len(hh) - 2)
    sb = math.sqrt(s2 / float(((hh - hh.mean()) ** 2).sum()))
    W = window_fraction * h_eval
    mean = float(pp.mean())
    metric = abs(b) * W / mean
    onset_ok = h_onset is not None and h_onset <= lo + 1e-9
    return {"h_window_A": [float(lo), float(h_eval)], "n": int(len(hh)), "pm_mean_GPa": mean,
            "pm_sd_GPa": float(pp.std(ddof=1)), "slope_GPa_per_A": b, "slope_se": float(sb), "metric": float(metric),
            "metric_2se_upper": float((abs(b) + 2.0 * sb) * W / mean), "slope_ok": bool(metric < tol),
            "onset_before_window": bool(onset_ok), "passes": bool(metric < tol and onset_ok)}


def gate_iii(h: Any, pm_by_estimator: Mapping[str, Any], h_eval: float, *, h_onset: Optional[float],
             estimators: Sequence[str] = PLATEAU_EVIDENCE_ESTIMATORS, tol: float = PLATEAU_TOL,
             window_fraction: float = PLATEAU_WINDOW_FRACTION) -> dict[str, Any]:
    """Gate III (LM-R10 plateau): the metric must be < tol on BOTH E3 and E4 in a window that starts after the
    Gate II onset.  H = mean E3 p_m over the window; the E3-E4 difference is the estimator systematic."""
    per = {e: plateau_metric(h, pm_by_estimator[e], h_eval, window_fraction=window_fraction, h_onset=h_onset,
                             tol=tol) for e in estimators}
    ok = all(v.get("passes", False) for v in per.values())
    out: dict[str, Any] = {"h_eval_A": float(h_eval), "h_onset_A": h_onset, "tol": tol, "estimators": per,
                           "passes": bool(ok)}
    if all("pm_mean_GPa" in v for v in per.values()):
        vals = {e: per[e]["pm_mean_GPa"] for e in estimators}
        out["H_GPa"] = vals.get(PRIMARY_ESTIMATOR, next(iter(vals.values())))
        out["estimator_spread_GPa"] = float(max(vals.values()) - min(vals.values()))
        out["estimator_spread_rel"] = float(out["estimator_spread_GPa"] / out["H_GPa"])
    return out


# ============================================================================================ unloading
def loop_work(h: Any, P: Any, load_mask: Any, unload_mask: Any) -> dict[str, float]:
    """Loading and unloading work (trapezoid of P dh, eV) and the dissipated fraction (W_l - W_u) / W_l."""
    h = np.asarray(h, float)
    P = np.asarray(P, float)
    lm = np.asarray(load_mask, bool)
    um = np.asarray(unload_mask, bool)
    Wl = float(np.trapezoid(P[lm], h[lm])) / EV_PER_A_TO_NN if lm.sum() > 1 else float("nan")
    Wu = float(-np.trapezoid(P[um], h[um])) / EV_PER_A_TO_NN if um.sum() > 1 else float("nan")
    return {"W_load_eV": Wl, "W_unload_eV": Wu,
            "dissipated_fraction": (Wl - Wu) / Wl if np.isfinite(Wl) and np.isfinite(Wu) and Wl > 0 else float("nan")}


def unloading_final_depth(h_unl: Any, Pw_unl: Any, *, threshold_nN: float = 2.0) -> Optional[float]:
    """h_f: the depth at which the windowed unloading load first falls below ``threshold_nN``."""
    h = np.asarray(h_unl, float)
    Pw = np.asarray(Pw_unl, float)
    idx = np.where(Pw < threshold_nN)[0]
    return float(h[idx[0]]) if len(idx) else None


def oliver_pharr(h_unl: Any, P_unl: Any, *, P_max_nN: float, h_max_A: float,
                 area_fn: Callable[[float], float] = lambda hc: BERKOVICH_C0 * hc * hc,
                 epsilon: float = 0.75, beta: float = 1.05,
                 fit_range: tuple[float, float] = (0.2, 0.95)) -> dict[str, Any]:
    """Oliver-Pharr DIAGNOSTIC from the unloading branch: fit P = alpha (h - h_f)^m over fit_range x P_max,
    S = dP/dh at h_max, h_c = h_max - epsilon P_max / S, H_OP = P_max / A(h_c), E_r = sqrt(pi) S / (2 beta
    sqrt A).  Never the primary hardness: an elastic MD contact gives a meaningless H_OP (77 GPa for G06)."""
    h = np.asarray(h_unl, float)
    P = np.asarray(P_unl, float)
    s = (P > fit_range[0] * P_max_nN) & (P < fit_range[1] * P_max_nN)
    if s.sum() < 5:
        return {"error": f"only {int(s.sum())} unloading samples in the fit range"}
    f = lambda x, a, hf, mm: a * np.clip(x - hf, 1e-9, None) ** mm  # noqa: E731
    try:
        q, _ = curve_fit(f, h[s], P[s], p0=[P_max_nN / (h_max_A * 0.5) ** 1.5, h_max_A * 0.5, 1.5], maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        return {"error": str(exc)}
    S = float(q[0] * q[2] * (h_max_A - q[1]) ** (q[2] - 1.0))
    if not (np.isfinite(S) and S > 0.0):  # a diagnostic must never abort the analysis (r4-recipe R1)
        return {"error": f"unloading stiffness S = {S:g} nN/A is not positive", "fit_range": list(fit_range)}
    hc = h_max_A - epsilon * P_max_nN / S
    A = float(area_fn(hc))
    if not (np.isfinite(A) and A > 0.0):  # small or fast runs: h_c <= 0 gives A = 0 (r4-recipe R1)
        return {"error": f"A(h_c) = {A:g} A^2 at h_c = {hc:.4g} A: no Oliver-Pharr contact area",
                "fit_range": list(fit_range), "S_nN_per_A": S, "hc_A": float(hc)}
    return {"fit_range": list(fit_range), "alpha": float(q[0]), "hf_fit_A": float(q[1]), "m": float(q[2]),
            "S_nN_per_A": S, "hc_A": float(hc), "A_A2": A, "H_OP_GPa": float(GPA_PER_NN_PER_A2 * P_max_nN / A),
            "Er_GPa": float(math.sqrt(math.pi) * S / (2.0 * beta * math.sqrt(A)) * GPA_PER_NN_PER_A2),
            "epsilon": epsilon, "beta": beta, "role": "diagnostic only (never the primary hardness)"}


def height_map(positions: Any, axis_xy: Sequence[float], L_xy: Sequence[float], surface_z0: float,
               bin_A: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-atom height map (z_max - z_s0 per square bin of ``bin_A``) centred on the tip axis; returns (map indexed
    [ix, iy], x bin centres, y bin centres).  x wraps and bins with L_xy[0], y with L_xy[1] (rectangular cells);
    a remainder strip narrower than a bin joins the last bin.  Empty bins are -inf."""
    p = np.asarray(positions, float)
    idx, centres = [], []
    for k in (0, 1):
        L = float(L_xy[k])
        u = (p[:, k] - axis_xy[k] + L / 2) % L - L / 2
        nb = max(1, int(math.floor(L / bin_A)))
        idx.append(np.clip(((u + L / 2) / bin_A).astype(int), 0, nb - 1))
        centres.append((np.arange(nb) + 0.5) * bin_A - L / 2)
    H = np.full((len(centres[0]), len(centres[1])), -np.inf)
    np.maximum.at(H, (idx[0], idx[1]), p[:, 2])
    return H - surface_z0, centres[0], centres[1]


def _finite_stat(v: np.ndarray, fn: Callable[[np.ndarray], Any]) -> float:
    v = v[np.isfinite(v)]
    return float(fn(v)) if len(v) else float("nan")


def residual_impression(positions_final: Any, axis_xy: Sequence[float], L_xy: Sequence[float], surface_z0: float, *,
                        bin_A: float, contact_radius_A: Optional[float] = None, positions_initial: Any = None,
                        center_radius_A: float = 3.5) -> dict[str, Any]:
    """Residual impression from the final top-atom height map (G06 method): depth of the mean of the bins
    within ``center_radius_A`` of the axis below z_s0 (positive = a dent), deepest bin within 15 A, pile-up
    between 5 A and 1.6 a_c, and the far-field roughness (bins beyond 0.45 of the cell from the axis, i.e.
    (x / Lx)^2 + (y / Ly)^2 > 0.45^2).  Empty bins (-inf) are left out of every statistic and counted."""
    Hf, cx, cy = height_map(positions_final, axis_xy, L_xy, surface_z0, bin_A)
    CX, CY = np.meshgrid(cx, cy, indexing="ij")
    RR = np.hypot(CX, CY)
    ctr = RR < center_radius_A
    far = np.hypot(CX / float(L_xy[0]), CY / float(L_xy[1])) > 0.45
    out: dict[str, Any] = {"center_depth_A": -_finite_stat(Hf[ctr], np.mean),
                           "deepest_bin_A": -_finite_stat(Hf[RR < 15.0], np.min), "bin_A": float(bin_A),
                           "rough_rms_far_A": _finite_stat(Hf[far], np.std), "n_empty_bins": int(np.isinf(Hf).sum())}
    if contact_radius_A is not None:
        ring = (RR > 5.0) & (RR < 1.6 * contact_radius_A)
        out["max_pileup_A"] = _finite_stat(Hf[ring], np.max)
        out["contact_radius_A"] = float(contact_radius_A)
    if positions_initial is not None:
        H0, _, _ = height_map(positions_initial, axis_xy, L_xy, surface_z0, bin_A)
        out["initial_center_depth_A"] = -_finite_stat(H0[ctr], np.mean)
    return out


@dataclasses.dataclass(frozen=True)
class GateIIThresholds:
    """Gate II (developed plasticity) thresholds, recommended protocol item 17.  PROVISIONAL: frozen in
    recipe_v3 after the S1 scoping run (ceramics recover elastically; SR-X02)."""

    residual_min_A: float = 2.0  # residual impression >= max(residual_min_A, residual_min_fraction h_max)
    residual_min_fraction: float = 0.1
    dissipation_min: float = 0.10  # loop dissipation (W_load - W_unload) / W_load
    hf_ratio_min: float = 0.3  # h_f / h_max
    provisional: bool = True

    def residual_threshold_A(self, h_max_A: float) -> float:
        return max(self.residual_min_A, self.residual_min_fraction * h_max_A)


def gate_ii(*, residual_depth_A: Optional[float], dissipated_fraction: Optional[float],
            hf_over_hmax: Optional[float], h_max_A: float,
            thresholds: GateIIThresholds = GateIIThresholds()) -> dict[str, Any]:
    """Gate II: residual impression, dissipation and h_f/h_max must ALL pass (a missing input fails)."""
    thr_res = thresholds.residual_threshold_A(h_max_A)

    def ok(v, t):
        return v is not None and np.isfinite(v) and v >= t

    crit = {"residual_impression": {"value_A": residual_depth_A, "threshold_A": thr_res,
                                    "passed": bool(ok(residual_depth_A, thr_res))},
            "dissipation": {"value": dissipated_fraction, "threshold": thresholds.dissipation_min,
                            "passed": bool(ok(dissipated_fraction, thresholds.dissipation_min))},
            "hf_over_hmax": {"value": hf_over_hmax, "threshold": thresholds.hf_ratio_min,
                             "passed": bool(ok(hf_over_hmax, thresholds.hf_ratio_min))}}
    return {"passes": all(c["passed"] for c in crit.values()), "criteria": crit, "h_max_A": float(h_max_A),
            "provisional": thresholds.provisional, "thresholds": dataclasses.asdict(thresholds)}


NO_GATE_II_EVIDENCE = ("no intermediate-depth Gate II evidence (Gate I quenched residual impressions or unload branches, "
                       "IndentationRunData.gate_ii_evidence), so the Gate II onset depth is unknown")


def gate_ii_onset(evidence: Iterable[Mapping[str, Any]], *, gate_ii_at_hmax: Mapping[str, Any], h_max_A: float,
                  thresholds: GateIIThresholds = GateIIThresholds()) -> dict[str, Any]:
    """Gate II onset depth from intermediate-depth evidence (recommended protocol item 17).

    :func:`gate_ii` judges the state after the full unload from h_max, so it certifies developed plasticity AT
    h_max only; an onset placed there always lies inside the Gate III window (the last 40 % of loading), which
    must start after the onset.  The onset comes from evidence at depths d <= h_max.  Each ``evidence`` entry
    has ``h_A`` (d), ``residual_depth_A`` (residual impression at the contact centre after unloading from d,
    positive = a dent: a Gate I quenched sub-volume, :func:`gate_ii_evidence_from_gate_i`, or an unload branch)
    and optionally ``dissipated_fraction`` and ``hf_over_hmax`` (h_f / d) of an unload branch from d (default:
    the values of the full unload from h_max, as the protocol prescribes for a single-run depth scan) and a
    ``source``.  Each depth is judged with the Gate II thresholds at that depth.  The onset is the shallowest
    evidence depth from which every deeper evidence depth passes too, and there is none unless Gate II passes
    at h_max.  A depth passes only when every evidence entry at it passes.

    An entry with ``not_evaluated`` True (a Gate I frame whose quench did not converge or whose defects were not
    contained: critic N1 / N2) is neither a pass nor a fail: its row has ``passed`` None.  When such a depth decides
    the onset (the deepest depth, or the first depth above the passing run from h_max that is not a pass) the onset is
    NOT EVALUATED: ``h_onset_A`` None, ``not_evaluated`` True, a ``reason`` starting with "NOT EVALUATED" (the records'
    gate check, physics.indentation.gate_checks, then reads as NOT EVALUATED) and the bounds ``onset_lower_bound_A``
    / ``onset_upper_bound_A``.  Below the first failing depth it does not matter.  Returns ``h_onset_A`` (None without
    an onset), the per-depth verdicts and a ``reason``.
    """
    crit = gate_ii_at_hmax.get("criteria", {})
    diss_hmax = crit.get("dissipation", {}).get("value")
    hf_hmax = crit.get("hf_over_hmax", {}).get("value")

    def ok(v, t):
        return v is not None and np.isfinite(v) and v >= t

    rows = []
    for ev in sorted(evidence, key=lambda e: float(e["h_A"])):
        d = float(ev["h_A"])
        if d > h_max_A + 1e-6:
            continue
        res = ev.get("residual_depth_A")
        diss = ev.get("dissipated_fraction", diss_hmax)
        hf = ev.get("hf_over_hmax", hf_hmax)
        thr = thresholds.residual_threshold_A(d)
        row = {"h_A": d, "source": ev.get("source", "unspecified"), "residual_depth_A": res,
               "residual_threshold_A": thr, "dissipated_fraction": diss, "hf_over_hmax": hf}
        if ev.get("not_evaluated"):
            row.update(criteria={k: None for k in ("residual_impression", "dissipation", "hf_over_hmax")}, passed=None,
                       not_evaluated=True, reason=str(ev.get("reason") or "not evaluated"))
        else:
            c = {"residual_impression": bool(ok(res, thr)), "dissipation": bool(ok(diss, thresholds.dissipation_min)),
                 "hf_over_hmax": bool(ok(hf, thresholds.hf_ratio_min))}
            row.update(criteria=c, passed=all(c.values()))
        rows.append(row)
    n_ne = sum(1 for r in rows if r.get("not_evaluated"))
    out: dict[str, Any] = {"h_onset_A": None, "n_evidence": len(rows), "n_not_evaluated": n_ne, "per_depth": rows}
    if not gate_ii_at_hmax.get("passes"):
        out["reason"] = f"Gate II (developed plasticity) failed at h_max = {h_max_A:.3g} A"
        return out
    if not rows:
        out["reason"] = NO_GATE_II_EVIDENCE
        return out
    # one verdict per depth: fail if any entry fails, else not evaluated if any entry is, else pass
    groups: list[list[dict[str, Any]]] = []
    for r in rows:
        if groups and abs(groups[-1][0]["h_A"] - r["h_A"]) <= 1e-6:
            groups[-1].append(r)
        else:
            groups.append([r])

    def status(g):
        if any(r["passed"] is False for r in g):
            return "fail"
        return "not_evaluated" if any(r.get("not_evaluated") for r in g) else "pass"

    st = [status(g) for g in groups]
    depth = [g[0]["h_A"] for g in groups]
    k = len(groups)
    while k > 0 and st[k - 1] == "pass":
        k -= 1

    def ne_text(g):
        return "; ".join(f"{r['source']}: {r['reason']}" for r in g if r.get("not_evaluated"))

    if k == len(groups):
        if st[-1] == "fail":
            out["reason"] = f"the deepest Gate II evidence (h = {depth[-1]:.3g} A) fails"
        else:
            out.update(not_evaluated=True, onset_upper_bound_A=None,
                       reason=f"{GATE_I_NOT_EVALUATED}: the deepest Gate II evidence (h = {depth[-1]:.3g} A) is not "
                              f"evaluated ({ne_text(groups[-1])}), so the Gate II onset is unknown")
        return out
    if k > 0 and st[k - 1] == "not_evaluated":
        j = k - 1
        while j >= 0 and st[j] != "fail":
            j -= 1
        out.update(not_evaluated=True, onset_lower_bound_A=depth[j + 1], onset_upper_bound_A=depth[k],
                   reason=f"{GATE_I_NOT_EVALUATED}: Gate II passes from h = {depth[k]:.3g} A to h_max, but the evidence "
                          f"at h = {depth[k - 1]:.3g} A is not evaluated ({ne_text(groups[k - 1])}): the onset lies "
                          f"between {depth[j + 1]:.3g} and {depth[k]:.3g} A")
        return out
    srcs = list(dict.fromkeys(r["source"] for r in groups[k]))
    src = srcs[0] if len(srcs) == 1 else " + ".join(srcs)
    out.update(h_onset_A=depth[k], source=src,
               reason=f"Gate II passes from h = {depth[k]:.3g} A ({src}) to h_max")
    return out


def gate_ii_evidence_from_gate_i(gate_i_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Gate II evidence (:func:`gate_ii_onset`) from a :func:`gate_i` run: the residual impression at each quenched
    depth is the centre dent of the quenched sub-volume (``residual_depth_A`` = -center_dz_A, a 0 K unloaded state).
    Frames without a depth (``h_A``) or without a centre dent (``center_dz_A`` None: no top atom of the quenched
    reference within ``dent_radius_A`` of the axis, a property of the reference surface and not of the frame's quench)
    are skipped, evaluated or not: such a frame is no evidence even when evaluated, so its not-evaluated state cannot
    hide any.  A frame with a dent that is not evaluated (critic N1 / N2, :func:`gate_i_frame_status`: a quench did
    not converge, the defects were not contained, or the row does not show either) gives a ``not_evaluated`` entry
    with its ``reason`` and no residual: the dent of an unconverged or carve-limited quench is not evidence, and
    gate_ii_onset reports it."""
    out = []
    for fr in gate_i_result.get("frames", []):
        if fr.get("h_A") is None or fr.get("center_dz_A") is None:
            continue
        if gate_i_frame_status(fr) == "not_evaluated":
            out.append({"h_A": float(fr["h_A"]), "residual_depth_A": None, "source": "gate_i_quench",
                        "label": fr.get("label"), "not_evaluated": True,
                        "reason": "Gate I quench not evaluated: " + "; ".join(_gate_i_reasons(fr))})
            continue
        out.append({"h_A": float(fr["h_A"]), "residual_depth_A": -float(fr["center_dz_A"]), "source": "gate_i_quench",
                    "label": fr.get("label")})
    return out


def hardness_verdict(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Whether an analysis (:func:`analyze_indentation`) yields a hardness (Gates II and III) and, if not, why."""
    g2 = analysis.get("gate_ii") or {}
    g3 = analysis.get("gate_iii")
    onset = g2.get("onset") or {}
    if not g2.get("passes"):
        return {"available": False, "reason": onset.get("reason") or "Gate II (developed plasticity) failed at h_max"}
    if onset.get("h_onset_A") is None and onset.get("not_evaluated"):  # critic N1 / N2: a not-evaluated Gate I quench
        return {"available": False,
                "reason": f"{onset.get('reason')} (Gate II passed at h_max): the Gate III window (after the onset) "
                          "cannot be placed and no hardness can be produced"}
    if onset.get("h_onset_A") is None:
        return {"available": False,
                "reason": f"Gate II passed at h_max, but {onset.get('reason', NO_GATE_II_EVIDENCE)}: the Gate III window "
                          "(after the onset) cannot be placed and no hardness can be produced"}
    if not g3:
        return {"available": False, "reason": "no loading frames with contact areas (Gate III not evaluated)"}
    if g3.get("passes"):
        return {"available": True, "reason": f"Gates II (onset {onset['h_onset_A']:.3g} A) and III passed"}
    per = g3.get("estimators", {})
    if not all(v.get("onset_before_window") for v in per.values()):
        lo = next(iter(per.values()), {}).get("h_window_A", [float("nan")])[0]
        return {"available": False,
                "reason": f"the Gate II onset ({onset['h_onset_A']:.3g} A) lies inside the Gate III window starting at "
                          f"{lo:.3g} A: load deeper (h_max >= onset / {1.0 - PLATEAU_WINDOW_FRACTION:g})"}
    metrics = ", ".join(f"{e} {v.get('metric', float('nan')):.3f}" for e, v in per.items())
    return {"available": False, "reason": f"no LM-R10 plateau on E3 and E4 (metrics {metrics}, tolerance {g3['tol']:g})"}


# ============================================================================================ coordination
def coordination_changes(pos_ref: Any, pos: Any, species: Any, lo: Any, hi: Any, pbc: Sequence[bool], *,
                         rule: Optional[BondRule] = None, potential: Optional[str] = None, centers: Optional[Any] = None,
                         cn_ref: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Coordination under a potential-aware bond rule (default: heteronuclear-only Si-C < 2.4 A for SiC) in the
    reference and the current configuration, the changed atoms and the class histogram (bulk-origin atoms,
    i.e. 4-fold in the reference).  ``centers`` restricts the evaluation to those atoms (against all atoms)."""
    sp = np.asarray(species).astype(str)
    rule = rule or bond_rule_for(potential, sp)
    c = None if centers is None else np.asarray(centers, int)
    cn0 = coordination_by_rule(pos_ref, sp, lo, hi, pbc, rule, centers=c) if cn_ref is None else np.asarray(cn_ref)
    cn = coordination_by_rule(pos, sp, lo, hi, pbc, rule, centers=c)
    changed = cn != cn0
    bulk0 = cn0 == 4
    hist = {str(k): int(np.sum(bulk0 & (cn == k))) for k in range(0, 8)}
    hist["8+"] = int(np.sum(bulk0 & (cn >= 8)))
    return {"rule": rule.name, "cn_ref": cn0, "cn": cn, "changed": changed, "changed_bulk": changed & bulk0,
            "n_changed": int(changed.sum()), "n_changed_bulk": int((changed & bulk0).sum()),
            "bulk_to_5plus": int(np.sum(bulk0 & (cn >= 5))), "bulk_to_3minus": int(np.sum(bulk0 & (cn <= 3))),
            "hist_bulk_origin": hist}


# ============================================================================================ local stress
def species_balanced_stress(S: Any, V: Any, species: Any, members: Any, *, unit_to_GPa: float = 1.0e-4) -> np.ndarray:
    """Species-balanced coarse-grained stress sigma = (<S>_Si + <S>_C) / (<V>_Si + <V>_C) (GPa).

    ``S``: (N, 6) per-atom stress x volume (LAMMPS ``compute stress/atom`` in bar A^3: ``unit_to_GPa`` 1e-4);
    ``V``: (N,) per-atom volumes (A^3; e.g. Voronoi); ``members``: an (M, N) sparse/dense 0-1 matrix or a list
    of index arrays (one per coarse-graining point).  The Vashishta per-atom virial carries a +-25 GPa
    isotropic Si/C offset, so only the species-balanced mean is meaningful for non-stoichiometric groups; it
    is exact for any homogeneous state.  Returns (M, 6).
    """
    S = np.asarray(S, float)
    V = np.asarray(V, float)
    sp = np.asarray(species).astype(str)
    kinds = sorted(set(sp.tolist()))
    if isinstance(members, (list, tuple)):
        rows = np.concatenate([np.full(len(m), i) for i, m in enumerate(members)]) if members else np.zeros(0, int)
        cols = np.concatenate([np.asarray(m, int) for m in members]) if members else np.zeros(0, int)
        M = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(members), len(S)))
    else:
        M = csr_matrix(members)
    num = np.zeros((M.shape[0], S.shape[1]))
    den = np.zeros(M.shape[0])
    for k in kinds:
        Mk = M.multiply((sp == k)[None, :].astype(float)).tocsr()
        nk = np.asarray(Mk.sum(1)).ravel()
        nk_safe = np.maximum(nk, 1)
        num += (Mk @ S) / nk_safe[:, None]
        den += (Mk @ V) / nk_safe
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den[:, None] * unit_to_GPa


def layer_force_balance(z: Any, S_zz: Any, species: Any, *, z_top: float, depths_A: Sequence[tuple[float, float]],
                        unit_to_nN_A: float = NN_PER_BAR_A2) -> list[dict[str, Any]]:
    """Normal force carried by horizontal layers: P_layer = -(n/2)(<S_zz>_Si + <S_zz>_C) / t (nN).

    Each layer spans depths [d0, d1) below ``z_top`` (use lattice-constant-thick layers: they are
    stoichiometric in a (001) slab, where the balanced sum equals the plain sum).  ``S_zz``: per-atom stress x
    volume (LAMMPS bar A^3: ``unit_to_nN_A`` = 1e-6 nN per bar A^2).  In quasi-static balance every layer
    carries the indenter load (G05: 1-5 % at depths >= 16 A).
    """
    z = np.asarray(z, float)
    s = np.asarray(S_zz, float)
    sp = np.asarray(species).astype(str)
    kinds = sorted(set(sp.tolist()))
    out = []
    for d0, d1 in depths_A:
        m = (z_top - z >= d0) & (z_top - z < d1)
        n = int(m.sum())
        t = float(d1 - d0)
        counts = {k: int((m & (sp == k)).sum()) for k in kinds}
        if n == 0 or any(c == 0 for c in counts.values()):
            out.append({"depth_A": [float(d0), float(d1)], "n": n, "counts": counts, "P_nN": float("nan")})
            continue
        bal = sum(float(s[m & (sp == k)].mean()) for k in kinds) * n / len(kinds)
        out.append({"depth_A": [float(d0), float(d1)], "n": n, "counts": counts,
                    "P_nN": float(-bal / t * unit_to_nN_A), "P_plain_nN": float(-s[m].sum() / t * unit_to_nN_A)})
    return out


def force_balance_check(layers: Sequence[Mapping[str, Any]], load_nN: float, *, tol: float = 0.05,
                        tare: Optional[Sequence[Mapping[str, Any]]] = None) -> dict[str, Any]:
    """Layer loads against the indenter load: passes if the mean over layers is within ``tol`` (per-layer
    deviations, which carry the instantaneous thermal noise of a single frame, are reported).

    ``tare``: the same layers in a pre-contact frame (zero load), subtracted layer by layer.  A virial-only
    per-atom stress (LAMMPS ``stress/atom NULL virial``) omits the kinetic stress -n k_B T: at 300 K that is a
    ~0.4 GPa offset, ~20 nN over a 70 A cell, i.e. 3-15 % of a nm-contact load (G05 frames); the tare removes it
    together with any residual slab stress.
    """
    if tare is not None:
        layers = [dict(l, P_nN=l["P_nN"] - r["P_nN"]) for l, r in zip(layers, tare)]
    vals = np.array([l["P_nN"] for l in layers if np.isfinite(l["P_nN"])], float)
    if len(vals) == 0 or not load_nN:
        return {"passes": False, "reason": "no layers or zero load"}
    rel = vals / load_nN - 1.0
    mean_rel = float(vals.mean() / load_nN - 1.0)
    return {"passes": bool(abs(mean_rel) <= tol), "mean_rel_dev": mean_rel, "max_abs_rel_dev": float(np.abs(rel).max()),
            "layer_loads_nN": vals.tolist(), "load_nN": float(load_nN), "tol": tol, "tared": tare is not None}


# ============================================================================================ defect extent
def defect_extent(positions: Any, flagged: Any, *, axis_xy: Sequence[float], surface_z0: float, depth_A: float,
                  L_xy: Sequence[float], fixed_top_z: Optional[float] = None, bath_top_z: Optional[float] = None,
                  linkage_A: float = 3.5) -> dict[str, Any]:
    """Extent of the connected cluster of flagged (e.g. coordination-changed) atoms at the contact.

    Clusters link atoms closer than ``linkage_A`` (lateral minimum image); the cluster containing the flagged
    atom nearest to the contact point (axis, depth ``depth_A``) is measured: depth below z_s0, clearance to
    the thermostat and fixed layers, lateral extent and the gap to its own periodic image (L minus extent).
    """
    p = np.asarray(positions, float)
    idx = np.flatnonzero(np.asarray(flagged, bool))
    if len(idx) == 0:
        return {"n_flagged": 0}
    q = p[idx].copy()
    L = np.asarray(L_xy, float)
    d = q[:, :2] - np.asarray(axis_xy, float)
    d -= L * np.round(d / L)
    wrapped = np.column_stack([d % L, q[:, 2] - q[:, 2].min()])
    box = np.array([L[0], L[1], wrapped[:, 2].max() + 2.0 * linkage_A + 1.0])
    pairs = cKDTree(np.minimum(wrapped, np.nextafter(box, 0)), boxsize=box).query_pairs(linkage_A, output_type="ndarray")
    G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
    ncomp, lab = connected_components(G, directed=False)
    rxy = np.hypot(d[:, 0], d[:, 1])
    k = lab[np.argmin(rxy + np.abs((surface_z0 - q[:, 2]) - depth_A))]
    sel = lab == k
    dq = d[sel]
    zlow = float(q[sel, 2].min())
    out: dict[str, Any] = {
        "n_flagged": int(len(idx)), "n_main_cluster": int(sel.sum()), "n_clusters": int(ncomp),
        "depth_below_z_s0_A": float(surface_z0 - zlow),
        "x_extent_A": [float(dq[:, 0].min()), float(dq[:, 0].max())],
        "y_extent_A": [float(dq[:, 1].min()), float(dq[:, 1].max())],
        "r_max_A": float(np.hypot(dq[:, 0], dq[:, 1]).max()),
        "gap_to_periodic_image_x_A": float(L[0] - (dq[:, 0].max() - dq[:, 0].min())),
        "gap_to_periodic_image_y_A": float(L[1] - (dq[:, 1].max() - dq[:, 1].min())),
    }
    if fixed_top_z is not None:
        out["clearance_to_fixed_layer_A"] = float(zlow - fixed_top_z)
    if bath_top_z is not None:
        out["clearance_to_thermostat_layer_A"] = float(zlow - bath_top_z)
    return out


def defect_extent_check(extent: Mapping[str, Any], *, t_el_A: float, L_A: float) -> dict[str, Any]:
    """Run acceptance (recommended protocol item 22): the defect cluster stays >= t_el/2 from the fixed layer
    and >= L/3 from its periodic image."""
    if extent.get("n_flagged", 0) == 0:
        return {"passes": True, "reason": "no flagged atoms"}
    clr = extent.get("clearance_to_fixed_layer_A")
    gap = min(extent["gap_to_periodic_image_x_A"], extent["gap_to_periodic_image_y_A"])
    ok_fixed = clr is None or clr >= 0.5 * t_el_A
    ok_img = gap >= L_A / 3.0
    return {"passes": bool(ok_fixed and ok_img), "clearance_to_fixed_layer_A": clr, "required_fixed_A": 0.5 * t_el_A,
            "image_gap_A": float(gap), "required_image_gap_A": L_A / 3.0}


# ============================================================================================ Gate I
@dataclasses.dataclass(frozen=True)
class GateIThresholds:
    """Gate I (first irreversibility) settings, recommended protocol item 16 (G05 method)."""

    n_changed_min: int = 10  # changed heteronuclear-only coordination (interior atoms)
    dpe_min_eV: float = 1.0
    reversible_dpe_eV: float = 0.1  # |dPE| below this and nothing changed / displaced -> reversible
    displacement_A: float = 0.5  # residual non-affine displacement that counts an atom as moved
    surface_skin_A: float = 1.5  # atoms this close to the surface are excluded from the interior (G05)
    shell_margin_A: float = 2.0  # interior atoms stay this far from the fixed shell
    dent_radius_A: float = 6.0  # top atoms within this radius of the axis give the centre dent (G05)
    defect_margin_A: float = 5.0  # a changed/displaced atom closer than this to the shell: the carve is too small


# Gate I quench recipe, FROZEN (critic item A5; outputs/ready/frozen_gate_i.json): conjugate gradient to max |F| <=
# 1e-3 eV/A (harness :func:`jax_minimizer` or LAMMPS :func:`lammps_minimizer`), a cylinder of radius and depth
# 1.5 a_c around the axis with a fixed 8 A shell (>= the 7.35 A Vashishta cutoff), a_c = the equivalent radius of the
# tip cross-section at the depth of the quenched frame (one gate_i call per frame of a depth series, recipe_v0
# gates.gate_i.frames; :func:`gate_i_contact_radius`).  On G05 R20 (whole cell 20,480
# atoms) this ~8.5k-atom carve gives the whole-cell PASS verdicts at 5.4 / 6.0 / 6.6 A; the sub-threshold 6.0 A
# defect of the whole-cell CG quench (5 atoms, 1.15 eV) is carve- and minimizer-dependent and is a diagnostic only.
GATE_I_MINIMIZER = "cg"
GATE_I_FMAX_EV_PER_A = 1.0e-3
GATE_I_RADIUS_FACTOR = 1.5
GATE_I_DEPTH_FACTOR = 1.5
GATE_I_SHELL_A = 8.0


def gate_i_contact_radius(geometry: Any, h_A: float) -> float:
    """a_c of the frozen Gate I carve: sqrt(A(h) / pi) of the tip cross-section at the nominal depth ``h_A`` of the
    deepest quenched frame (``geometry``: physics.indenter.IndenterGeometry; 2.7923 h for the ideal Berkovich,
    sqrt(2 R h - h^2) for a sphere).  Geometric, so it is known before any contact analysis."""
    return float(math.sqrt(float(np.asarray(geometry.area(max(float(h_A), 0.0)))) / math.pi))


def frozen_gate_i_recipe(thresholds: Optional["GateIThresholds"] = None) -> dict[str, Any]:
    """The frozen Gate I settings as a JSON-ready dict (recipe; outputs/ready/frozen_gate_i.json)."""
    thr = thresholds or GateIThresholds()
    return {"minimizer": {"method": GATE_I_MINIMIZER, "fmax_eV_per_A": GATE_I_FMAX_EV_PER_A, "newton_polish": False,
                          "backends": ["fracture_md.physics.indent_analysis.jax_minimizer (scipy CG, harness)",
                                       "fracture_md.physics.indent_analysis.lammps_minimizer (LAMMPS min_style cg, "
                                       "min_modify norm max)"]},
            "carve": {"contact_radius": "a_c = sqrt(A_tip(h_deepest) / pi) (gate_i_contact_radius)",
                      "radius_factor": GATE_I_RADIUS_FACTOR, "depth_factor": GATE_I_DEPTH_FACTOR,
                      "shell_A": GATE_I_SHELL_A,
                      "lateral_rule": "no lateral carve when 2 (radius + shell) + cutoff >= min(L)",
                      "shell_positions": "reference (pre-contact) positions in both quenches; the loaded interior "
                                         "starts from its positions minus the harmonic extension of the shell mismatch"},
            "thresholds": dataclasses.asdict(thr),
            "verdict": "Gate I passes (first irreversibility) when n_changed >= n_changed_min AND dPE >= dpe_min_eV; "
                       "'irreversible' (any change) is a diagnostic only; defects_contained = False asks for a larger "
                       "carve"}


# Gate I evaluation rule (critic findings N1 and N2, on top of the frozen recipe above, which it does not change: the frozen
# carve stays the first attempt).  A frame is EVALUATED only when (N1) its quench and the reference quench in the same
# carve both converged (QuenchResult.converged: max |F| <= fmax) and (N2) its defects are contained (every changed or
# displaced interior atom >= GateIThresholds.defect_margin_A from the fixed shell, or from the MD fixed layer when the
# carve reaches it).  Otherwise its verdict is "not_evaluated": never a pass, never a fail; first_irreversibility and the
# Gate II evidence carry it as NOT EVALUATED.  A converged frame whose defects reach the shell is quenched again in an
# enlarged carve (gate_i_frame): radius and depth factors grow geometrically by GATE_I_CARVE_GROWTH = 2^(1/3) per
# attempt (each enlargement doubles the carved volume), the shell stays 8 A, up to GATE_I_CARVE_MAX_SCALE = 2 x the
# first attempt's factors (with the frozen recipe 1.5 -> 1.89 -> 2.38 -> 3.0 a_c: at most four attempts, 8x the frozen
# volume; 3.0 a_c is the pre-A5 carve).  The lateral rule of carve_subvolume still applies, so a grown carve can become
# the whole lateral slab; a carve that can no longer grow, or the cap, ends the series as not evaluated.
GATE_I_CARVE_GROWTH = 2.0 ** (1.0 / 3.0)
GATE_I_CARVE_MAX_SCALE = 2.0
GATE_I_NOT_EVALUATED = "NOT EVALUATED"  # = physics.indentation.NOT_EVALUATED (the records' details prefix)
GATE_I_VERDICTS = ("pass", "fail", "not_evaluated")


def gate_i_carve_schedule(radius_factor: float = GATE_I_RADIUS_FACTOR, depth_factor: float = GATE_I_DEPTH_FACTOR, *,
                          growth: float = GATE_I_CARVE_GROWTH,
                          max_scale: float = GATE_I_CARVE_MAX_SCALE) -> list[tuple[float, float]]:
    """(radius_factor, depth_factor) of the successive Gate I carves of one frame (N2): the first attempt as given
    (default the frozen recipe), then both factors times ``growth`` per attempt up to ``max_scale`` x the first attempt
    (the cap itself is the last attempt).  ``max_scale`` <= 1 or ``growth`` <= 1: the first attempt only."""
    rf, df = float(radius_factor), float(depth_factor)
    out = [(rf, df)]
    if not (growth > 1.0 and max_scale > 1.0):
        return out
    s = 1.0
    while s < max_scale * (1.0 - 1e-9):
        s = min(s * float(growth), float(max_scale))
        out.append((rf * s, df * s))
    return out


def gate_i_evaluation_rule(*, radius_factor: float = GATE_I_RADIUS_FACTOR, depth_factor: float = GATE_I_DEPTH_FACTOR,
                           growth: float = GATE_I_CARVE_GROWTH, max_scale: float = GATE_I_CARVE_MAX_SCALE,
                           thresholds: Optional["GateIThresholds"] = None) -> dict[str, Any]:
    """The Gate I evaluation rule (critic N1 / N2) as a JSON-ready dict, recorded by quench_s1.py next to the frozen
    recipe (:func:`frozen_gate_i_recipe`, which it leaves unchanged)."""
    thr = thresholds or GateIThresholds()
    return {"convergence": "a frame is evaluated only when its quench and the reference quench in the same carve both "
                           "converged (max |F| <= fmax); otherwise its verdict is not_evaluated, never a pass or a fail",
            "containment": f"defects_contained: every changed or displaced interior atom >= {thr.defect_margin_A:g} A "
                           "(GateIThresholds.defect_margin_A) from the fixed shell, or from the MD fixed layer when the "
                           "carve reaches it; a converged, uncontained frame is quenched again in an enlarged carve",
            "carve_growth": float(growth), "max_scale": float(max_scale),
            "schedule": [list(x) for x in gate_i_carve_schedule(radius_factor, depth_factor, growth=growth,
                                                                 max_scale=max_scale)],
            "growth_rule": "radius and depth factors x carve_growth per attempt (2^(1/3): the carved volume doubles), "
                           "shell unchanged, up to max_scale x the first attempt (the frozen recipe by default); the "
                           "lateral rule of carve_subvolume still applies; stop when evaluated, when a quench did not "
                           "converge (growth cannot fix it), when the carve can no longer grow, or at the cap",
            "verdicts": list(GATE_I_VERDICTS),
            "propagation": "first_irreversibility (gate_i_first_irreversibility) is NOT EVALUATED when a frame "
                           "shallower than the first passing frame is not evaluated; Gate II evidence of a not-evaluated "
                           "frame is a not_evaluated row (gate_ii_evidence_from_gate_i) that gate_ii_onset reports as "
                           "NOT EVALUATED when it decides the onset"}


def _reference_converged(row: Mapping[str, Any]) -> Optional[bool]:
    if "reference_converged" in row:
        return row["reference_converged"]
    ref = row.get("reference")
    return ref.get("converged") if isinstance(ref, Mapping) else None


def gate_i_evaluation(row: Mapping[str, Any], *, thresholds: Optional["GateIThresholds"] = None) -> dict[str, Any]:
    """Gate I verdict of one quenched frame under the evaluation rule (N1 / N2; :func:`gate_i_evaluation_rule`).

    ``row``: a :func:`gate_i` frame verdict (``converged``, ``reference_converged``, ``defects_contained``, ``passes``
    or ``passes_thresholds``), or a row of an older gate_i.json (reference convergence in ``row["reference"]``).  A
    convergence or containment record that is missing counts as not shown, so the frame is not evaluated.  Returns
    ``evaluated``, ``verdict`` ("pass" / "fail" / "not_evaluated"), ``passes`` (True / False, None when not evaluated),
    ``passes_thresholds`` (n_changed >= n_changed_min AND dPE >= dpe_min_eV, whatever the evaluation) and
    ``not_evaluated_reasons``."""
    thr = thresholds or GateIThresholds()
    reasons = []

    def quench(what, conv, fmax, steps):
        if conv is None:
            reasons.append(f"no convergence record of the {what} quench")
        elif not conv:
            fm = "?" if fmax is None else f"{float(fmax):.3g}"
            reasons.append(f"the {what} quench did not converge (max |F| {fm} eV/A after {steps} iterations)")

    ref = row.get("reference") if isinstance(row.get("reference"), Mapping) else {}
    quench("frame", row.get("converged"), row.get("fmax"), row.get("steps"))
    quench("reference", _reference_converged(row), row.get("reference_fmax", ref.get("fmax")),
           row.get("reference_steps", ref.get("steps")))
    cont = row.get("defects_contained")
    if cont is None:
        reasons.append("no defect-containment record (defects_contained)")
    elif not cont:
        m = row.get("defect_margin_A")
        reasons.append(f"defects within {float(m):.3g} A of the fixed boundary (< {thr.defect_margin_A:g} A): the carve "
                       "is too small" if m is not None else "defects not contained by the carve")
    thr_pass = row.get("passes_thresholds", row.get("passes"))
    thr_pass = None if thr_pass is None else bool(thr_pass)
    if thr_pass is None:
        reasons.append("no threshold verdict (passes)")
    ok = not reasons
    return {"evaluated": ok, "verdict": ("pass" if thr_pass else "fail") if ok else "not_evaluated",
            "passes": thr_pass if ok else None, "passes_thresholds": thr_pass, "not_evaluated_reasons": reasons}


def gate_i_frame_status(row: Mapping[str, Any]) -> str:
    """"pass", "fail" or "not_evaluated" of a Gate I frame row: its ``verdict`` when it has one (:func:`gate_i` output),
    else :func:`gate_i_evaluation` of the row (older gate_i.json files)."""
    v = row.get("verdict")
    if v in GATE_I_VERDICTS:
        return str(v)
    return gate_i_evaluation(row)["verdict"]


def _gate_i_reasons(row: Mapping[str, Any]) -> list[str]:
    r = row.get("not_evaluated_reasons")
    if r is not None and row.get("verdict") in GATE_I_VERDICTS:
        return list(r)
    return gate_i_evaluation(row)["not_evaluated_reasons"]


def gate_i_first_irreversibility(frames: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """First-irreversibility depth of a Gate I depth series (N1): the shallowest EVALUATED frame that passes, provided
    no shallower frame is not evaluated (that frame might have passed).

    Returns ``h_A`` (None when there is none or it is not determined), ``status`` ("evaluated" or "not_evaluated"),
    ``reason``, ``not_evaluated_A`` (depths of not-evaluated frames) and, when not determined, ``upper_bound_A`` (the
    shallowest evaluated passing frame, or None).  Frames without a depth (``h_A``) are ignored."""
    rows = sorted((f for f in frames if f.get("h_A") is not None), key=lambda f: float(f["h_A"]))
    ne: list[float] = []
    why: list[str] = []
    for f in rows:
        st = gate_i_frame_status(f)
        h = float(f["h_A"])
        if st == "pass":
            if ne:
                return {"h_A": None, "status": "not_evaluated", "upper_bound_A": h, "not_evaluated_A": ne,
                        "reason": f"{GATE_I_NOT_EVALUATED}: Gate I passes at h = {h:.3g} A, but the shallower frame(s) at "
                                  f"{', '.join(f'{x:.3g}' for x in ne)} A are not evaluated ({'; '.join(why)}): the first "
                                  f"irreversibility lies at or above {h:.3g} A"}
            return {"h_A": h, "status": "evaluated", "not_evaluated_A": [],
                    "reason": f"Gate I passes first at h = {h:.3g} A ({f.get('n_changed')} changed atoms, dPE "
                              f"{float(f.get('dPE_eV', float('nan'))):.3g} eV)"}
        if st == "not_evaluated":
            ne.append(h)
            why.append(f"{h:.3g} A: {'; '.join(_gate_i_reasons(f))}")
    if not rows:
        return {"h_A": None, "status": "not_evaluated", "upper_bound_A": None, "not_evaluated_A": [],
                "reason": f"{GATE_I_NOT_EVALUATED}: no quenched frame with a depth"}
    if ne:
        return {"h_A": None, "status": "not_evaluated", "upper_bound_A": None, "not_evaluated_A": ne,
                "reason": f"{GATE_I_NOT_EVALUATED}: no evaluated frame passes Gate I and the frame(s) at "
                          f"{', '.join(f'{x:.3g}' for x in ne)} A are not evaluated ({'; '.join(why)})"}
    return {"h_A": None, "status": "evaluated", "not_evaluated_A": [],
            "reason": f"no quenched frame passes Gate I (deepest {float(rows[-1]['h_A']):.3g} A)"}


@dataclasses.dataclass
class CarvedRegion:
    """A carved sub-volume for a Gate I quench (indices into the full system)."""

    indices: np.ndarray  # atoms kept
    fixed: np.ndarray  # (len(indices),) bool: the fixed shell
    interior: np.ndarray  # (len(indices),) bool: atoms compared between the quenches
    radius_A: float
    depth_A: float
    shell_A: float
    lateral_carved: bool  # False: the cylinder (+ shell) did not fit in the cell, full lateral periodic slab kept
    # depth below z_s0 of the shallowest ``extra_fixed`` (MD fixed-layer) atom kept, when it lies above the carve
    # depth: the carve's effective bottom (an enlarged carve can reach the MD fixed layer; N2).  Not in summary().
    floor_depth_A: Optional[float] = None

    @property
    def bottom_depth_A(self) -> float:
        """Depth below z_s0 of the fixed boundary under the interior: the carve depth, or the MD fixed layer above it."""
        return self.depth_A if self.floor_depth_A is None else min(self.depth_A, self.floor_depth_A)

    def summary(self) -> dict[str, Any]:
        return {"n_atoms": int(len(self.indices)), "n_fixed": int(self.fixed.sum()), "n_interior": int(self.interior.sum()),
                "radius_A": self.radius_A, "depth_A": self.depth_A, "shell_A": self.shell_A,
                "lateral_carved": self.lateral_carved}


def carve_subvolume(positions_ref: Any, *, axis_xy: Sequence[float], surface_z0: float, L_xy: Sequence[float],
                    contact_radius_A: float, radius_factor: float = GATE_I_RADIUS_FACTOR,
                    depth_factor: float = GATE_I_DEPTH_FACTOR, shell_A: float = GATE_I_SHELL_A, cutoff_A: float = 0.0,
                    extra_fixed: Optional[Any] = None, thresholds: GateIThresholds = GateIThresholds()) -> CarvedRegion:
    """Carve a cylinder of radius ``radius_factor`` a_c and depth ``depth_factor`` a_c around the contact with
    a fixed shell of ``shell_A`` (sides and bottom), from REFERENCE (pre-contact) positions.  The defaults are the
    frozen Gate I carve (``GATE_I_*``; a_c from :func:`gate_i_contact_radius`).

    If the cylinder plus shell would meet its periodic image (2 (R + shell) + cutoff > L) the lateral carve is
    dropped and the full periodic slab is kept (only the bottom shell is fixed): a small cell is quenched
    whole, as G05 did.  ``extra_fixed``: atoms that are fixed anyway (the MD fixed layer).
    """
    p = np.asarray(positions_ref, float)
    L = np.asarray(L_xy, float)
    d = p[:, :2] - np.asarray(axis_xy, float)
    d -= L * np.round(d / L)
    r = np.hypot(d[:, 0], d[:, 1])
    depth = surface_z0 - p[:, 2]
    Rc = radius_factor * contact_radius_A
    Dc = depth_factor * contact_radius_A
    lateral = 2.0 * (Rc + shell_A) + cutoff_A < float(L.min())
    keep = depth < Dc + shell_A
    if lateral:
        keep &= r < Rc + shell_A
    idx = np.flatnonzero(keep)
    fixed = depth[idx] >= Dc
    if lateral:
        fixed |= r[idx] >= Rc
    floor = None
    if extra_fixed is not None:
        ef = np.asarray(extra_fixed, bool)[idx]
        fixed |= ef
        if ef.any() and float(depth[idx][ef].min()) < Dc:
            floor = float(depth[idx][ef].min())
    m = thresholds.shell_margin_A
    interior = ~fixed & (depth[idx] > thresholds.surface_skin_A) & (depth[idx] < Dc - m)
    if lateral:
        interior &= r[idx] < Rc - m
    return CarvedRegion(idx, fixed, interior, float(Rc), float(Dc), float(shell_A), bool(lateral), floor)


def harmonic_extension(positions: Any, fixed: Any, values: Any, *, bond_A: float = 3.2,
                       L_xy: Optional[Sequence[float]] = None, maxiter: int = 5000) -> np.ndarray:
    """Harmonic extension of per-atom ``values`` (N, k) given on the ``fixed`` atoms to the free atoms: the
    solution of the graph Laplace equation on the neighbour graph (atoms closer than ``bond_A``; lateral
    periodic if ``L_xy``).  Free atoms not connected to any fixed atom get zero."""
    p = np.asarray(positions, float)
    fx = np.asarray(fixed, bool)
    v = np.asarray(values, float).reshape(len(p), -1)
    out = np.where(fx[:, None], v, 0.0)
    free = np.flatnonzero(~fx)
    if len(free) == 0 or not fx.any():
        return out
    if L_xy is not None:
        L = np.asarray(L_xy, float)
        w = np.column_stack([p[:, 0] % L[0], p[:, 1] % L[1], p[:, 2] - p[:, 2].min()])
        box = np.array([L[0], L[1], w[:, 2].max() + 2 * bond_A + 1.0])
        tree = cKDTree(np.minimum(w, np.nextafter(box, 0)), boxsize=box)
    else:
        tree = cKDTree(p)
    pr = tree.query_pairs(bond_A, output_type="ndarray")
    n = len(p)
    A = coo_matrix((np.ones(2 * len(pr)), (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    Lap = (diags(deg) - A).tocsr()
    Lff = Lap[free][:, free]
    rhs = -(Lap[free][:, np.flatnonzero(fx)] @ v[fx])
    # isolated free components (no path to a fixed atom) make Lff singular: regularise them to zero
    Lff = Lff + diags(np.where(deg[free] == 0, 1.0, 0.0) + 1e-12)
    for k in range(v.shape[1]):
        if len(free) <= 20000:
            out[free, k] = spsolve(Lff.tocsc(), rhs[:, k])
        else:
            out[free, k], _ = sparse_cg(Lff, rhs[:, k], maxiter=maxiter)
    return out


@dataclasses.dataclass
class QuenchResult:
    positions: np.ndarray  # (n, 3) quenched positions of the carved atoms
    energy_eV: float
    converged: bool
    steps: int
    fmax_eV_per_A: float
    wall_s: float = 0.0
    restarts: int = 0  # extra minimiser calls from the stopped state (restarting_minimizer); 0 = one plain quench


Minimizer = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], QuenchResult]


def restarting_minimizer(minimizer: Minimizer, restarts: int) -> Minimizer:
    """The re-quench of an unconverged Gate I quench (critic N1; ``quench_s1.py --restarts``): ``minimizer`` and, while
    its result has not converged, up to ``restarts`` more calls started from the positions where the previous one
    stopped.  Each call is a CG restart (the search direction is reset to steepest descent), which is how a CG stopped
    by a failed line search (LAMMPS "linesearch alpha is zero") or by its iteration cap gets going again.  The energy
    surface, the fixed atoms and the stopping criterion (max |F| <= fmax, no Newton polish) are the minimiser's, so a
    converged result meets the frozen recipe (gates/gate_i/minimizer) like a single call; it can end in another local
    minimum than an uninterrupted CG would have, which is why the count is recorded.  The result is the last call's
    (positions, energy, converged, fmax) with ``steps`` and ``wall_s`` summed over the calls and ``restarts`` = the
    number of extra calls made.  ``restarts`` <= 0: ``minimizer`` itself."""
    n_max = int(restarts)
    if n_max <= 0:
        return minimizer

    def run(positions, species, cell, fixed) -> QuenchResult:
        q = minimizer(positions, species, cell, fixed)
        steps, wall, n = int(q.steps), float(q.wall_s), 0
        while not q.converged and n < n_max:
            q = minimizer(np.asarray(q.positions, float), species, cell, fixed)
            n += 1
            steps += int(q.steps)
            wall += float(q.wall_s)
        return dataclasses.replace(q, steps=steps, wall_s=wall, restarts=n)

    return run


def jax_minimizer(model: Any, *, fmax: float = 1.0e-3, max_steps: int = 50000, method: str = "cg",
                  skin: float = 1.0) -> Minimizer:
    """Gate I quench backend on the JAX harness: ``relax_positions`` of the carved atoms at fixed cell with the
    shell frozen (no Newton polish).  Returns minimizer(positions, species, cell (3, 3) rows = lattice vectors
    with a vacuum along z, fixed mask) -> QuenchResult.

    ``method`` "cg" (default; scipy conjugate gradient, as the protocol's CG quench and LAMMPS min_style cg)
    or "fire"/"lbfgs".  The sub-threshold ``irreversible`` flag AT the onset depends on the minimizer: on the
    G05 R20 frame at h = 6.0 A, a whole-cell CG quench stops in a marginal defect state (5 changed atoms, dPE
    1.15 eV; LAMMPS CG: 3 atoms, 1.55 eV) while FIRE's inertia carries it back to the crystal (0 atoms, 0.002 eV);
    both agree away from the onset (5.4 A reversible; 6.6 A 21-23 atoms, 11-13 eV).  It depends on the carve as
    well: the frozen carve (1.5 a_c, 8 A shell) and every other carve smaller than the whole cell quench the 6.0 A
    defect back to the crystal.  The Gate I PASS verdict (>= 10 atoms and >= 1 eV) is the same in every case.
    The frozen recipe (CG, fmax 1e-3, no Newton polish; module constants ``GATE_I_*``) is pinned by
    tests/test_indent_analysis.py.
    """

    def run(positions, species, cell, fixed) -> QuenchResult:
        import time

        from ase import Atoms

        from fracture_md.physics.relax import relax_positions
        from fracture_md.structures.system import AtomicSystem

        t0 = time.monotonic()
        at = Atoms(symbols=list(np.asarray(species).astype(str)), positions=np.asarray(positions, float),
                   cell=np.asarray(cell, float), pbc=True)
        sysm = AtomicSystem.from_ase(at)
        s, info = relax_positions(model, sysm, fmax=fmax, max_steps=max_steps, fixed=np.asarray(fixed, bool),
                                  skin=skin, method=method, newton_polish=False)
        X = np.asarray(s.frac, float) @ np.asarray(s.box, float).T
        # unwrap to the input images (relax returns wrapped fractional coordinates)
        box = np.asarray(s.box, float)
        dfrac = (X - np.asarray(positions, float)) @ np.linalg.inv(box).T
        X = np.asarray(positions, float) + (dfrac - np.round(dfrac)) @ box.T
        return QuenchResult(X, float(info.energy), bool(info.converged), int(info.n_steps), float(info.fmax),
                            time.monotonic() - t0)

    return run


LAMMPS_QUENCH_DECK = """# Gate I quench (fracture_md.physics.indent_analysis.lammps_minimizer): CG of a carved sub-volume, shell fixed
units metal
atom_style atomic
atom_modify map array
boundary p p f
read_data quench.data
pair_style {pair_style}
pair_coeff * * {pfile} {elements}
neighbor 1.0 bin
neigh_modify delay 0 every 1 check yes
group fixed type {fixed_types}
fix hold fixed setforce 0.0 0.0 0.0
thermo_style custom step pe fmax fnorm
thermo 1000
min_style cg
min_modify norm max
minimize 0.0 {fmax!r} {max_iter} {max_eval}
print "QUENCH_DONE pe=$(pe:%.12f) fmax=$(fmax:%.6e) iter=$(step)"
write_dump all custom quench.dump id x y z fx fy fz modify sort id format float %.15g
"""


def lammps_minimizer(potential: str = "vashishta_2007_SiC", *, cores: Sequence[int] = (), fmax: float = 1.0e-3,
                     max_iter: int = 200000, env: Any = None, workdir: Any = None, keep: bool = False,
                     timeout_s: Optional[float] = None) -> Minimizer:
    """Gate I quench backend on LAMMPS (CPU MPI): the same objective as :func:`jax_minimizer` (``min_style cg`` with
    the max-norm force criterion ``fmax``, shell atoms held by ``fix setforce 0``), for carved sub-volumes of
    1e5-1e6 atoms where the harness scipy-CG host loop is slow.  The LAMMPS Vashishta kernel equals the harness one
    to 1e-11 eV/A (tests/test_vashishta.py), so both backends quench the same energy surface; their CG line
    searches differ, so a marginal onset state can end in a different local minimum (G05 R20 at 6.0 A: harness CG
    5 changed atoms / 1.15 eV, LAMMPS CG 3 atoms / 1.55 eV, both irreversible).

    ``cores``: explicit MPI ranks (``engines.lammps_indent.mpirun_prefix``: pe-list binding, never X925 cores);
    empty = one unbound rank.  ``env``: the LAMMPS conda env (default ``engines.lammps_indent.lammps_env()``).
    ``workdir``: run directory (default a temporary directory, removed unless ``keep``).  Returns
    minimizer(positions, species, cell, fixed) -> QuenchResult with the positions unwrapped to the input images.
    Its ``fmax_eV_per_A`` (and ``converged`` = fmax <= ``fmax``) is the max per-atom |F| of the dumped final forces,
    the ``min_modify norm max`` criterion and jax_minimizer's measure; LAMMPS's thermo ``fmax`` (in the log) is the
    max force COMPONENT, up to sqrt(3) smaller, which made a quench stopped by the iteration limit look converged.
    Pinned against jax_minimizer by tests/test_gate_i_lammps_minimizer.py.
    """
    from fracture_md.engines.lammps_indent import lammps_env, mpirun_prefix, write_potential_file

    elements = ("C", "Si")

    def run(positions, species, cell, fixed) -> QuenchResult:
        import os
        import shutil
        import subprocess
        import tempfile
        import time
        from pathlib import Path

        envp = lammps_env(env)
        if envp is None:
            raise FileNotFoundError("LAMMPS environment not found (set FRACTURE_MD_LAMMPS_ENV)")
        t0 = time.monotonic()
        X = np.asarray(positions, float)
        sp = np.asarray(species).astype(str)
        fx = np.asarray(fixed, bool)
        C = np.asarray(cell, float)
        if np.abs(C - np.diag(np.diag(C))).max() > 1e-9:
            raise ValueError("lammps_minimizer needs an orthorhombic cell (quench_cell)")
        Lx, Ly, Lz = np.diag(C)
        unknown = sorted(set(sp.tolist()) - set(elements))
        if unknown:
            raise ValueError(f"lammps_minimizer supports {elements}, not {unknown}")
        if X[:, 2].min() < 0.0 or X[:, 2].max() >= Lz:
            raise ValueError("carved atoms must lie inside [0, Lz) along z (gate_i shifts them into the cell)")
        # wrap x, y into the box (read_data would remap them); the unwrap below restores the input images
        W = X.copy()
        W[:, 0] = np.mod(W[:, 0], Lx)
        W[:, 1] = np.mod(W[:, 1], Ly)
        W[:, :2] = np.where(W[:, :2] >= np.array([Lx, Ly]), 0.0, W[:, :2])
        typ = np.where(sp == "C", 1, 2) + np.where(fx, 2, 0)  # 3, 4 = fixed C, Si
        rd = Path(workdir) if workdir is not None else Path(tempfile.mkdtemp(prefix="gate_i_quench_"))
        rd.mkdir(parents=True, exist_ok=True)
        try:
            lines = ["Gate I carved sub-volume (fracture_md.physics.indent_analysis.lammps_minimizer)", "",
                     f"{len(X)} atoms", "4 atom types", "", f"0.0 {float(Lx)!r} xlo xhi", f"0.0 {float(Ly)!r} ylo yhi",
                     f"0.0 {float(Lz)!r} zlo zhi", "", "Masses", "", "1 12.011", "2 28.0855", "3 12.011", "4 28.0855",
                     "", "Atoms # atomic", ""]
            lines += [f"{i + 1} {t} {x!r} {y!r} {z!r}" for i, (t, (x, y, z)) in enumerate(zip(typ.tolist(), W.tolist()))]
            (rd / "quench.data").write_text("\n".join(lines) + "\n")
            pfile = write_potential_file(potential, rd / "potential.vashishta")
            (rd / "in.quench").write_text(LAMMPS_QUENCH_DECK.format(
                pair_style="vashishta", pfile=pfile.name, elements="C Si C Si", fixed_types="3 4", fmax=float(fmax),
                max_iter=int(max_iter), max_eval=10 * int(max_iter)))
            cmd = mpirun_prefix(list(cores), env=envp) + [str(envp / "bin" / "lmp"), "-in", "in.quench", "-log",
                                                           "log.quench", "-screen", "none", "-nocite"]
            res = subprocess.run(cmd, cwd=rd, capture_output=True, text=True, timeout=timeout_s,
                                 env={**os.environ, "OMP_NUM_THREADS": "1"})
            log = (rd / "log.quench").read_text() if (rd / "log.quench").exists() else ""
            if res.returncode != 0 or "QUENCH_DONE" not in log:
                raise RuntimeError(f"LAMMPS quench failed (rc {res.returncode}): {res.stderr[-1500:]}\n{log[-2000:]}")
            done = next(ln for ln in log.splitlines() if ln.startswith("QUENCH_DONE"))
            kv = dict(item.split("=") for item in done.split()[1:])
            d = np.loadtxt(rd / "quench.dump", skiprows=9, ndmin=2)
            d = d[np.argsort(d[:, 0])]
            if d.shape[1] < 7 or len(d) != len(X) or not np.array_equal(d[:, 0], np.arange(1, len(X) + 1)):
                raise RuntimeError(f"LAMMPS quench dump does not hold atoms 1..{len(X)} with id x y z fx fy fz")
            Q = d[:, 1:4].copy()
            dxy = Q[:, :2] - X[:, :2]
            Q[:, :2] = X[:, :2] + dxy - np.array([Lx, Ly]) * np.round(dxy / np.array([Lx, Ly]))
            # max per-atom |F| (the norm-max stopping criterion; the shell's forces are zeroed by setforce), not the
            # thermo fmax of the QUENCH_DONE line, which is the max component
            fm = float(np.linalg.norm(d[:, 4:7], axis=1).max())
            return QuenchResult(Q, float(kv["pe"]), bool(fm <= fmax * (1 + 1e-9)), int(kv["iter"]), fm,
                                time.monotonic() - t0)
        finally:
            if workdir is None and not keep:
                shutil.rmtree(rd, ignore_errors=True)

    return run


def quench_cell(positions: Any, L_xy: Sequence[float], *, cutoff_A: float, skin_A: float = 1.0) -> np.ndarray:
    """An orthorhombic cell (rows) for a carved sub-volume: the lateral periods, and along z the atom extent
    plus 2 (cutoff + skin) + 2 A of vacuum (the harness treats all three directions as periodic)."""
    p = np.asarray(positions, float)
    Lz = float(p[:, 2].max() - p[:, 2].min()) + 2.0 * (cutoff_A + skin_A) + 2.0
    return np.diag([float(L_xy[0]), float(L_xy[1]), Lz])


def gate_i(frames: Sequence[Mapping[str, Any]], reference: Mapping[str, Any], *, species: Any, L_xy: Sequence[float],
           axis_xy: Sequence[float], surface_z0: float, contact_radius_A: float, minimizer: Minimizer,
           cutoff_A: float, rule: Optional[BondRule] = None, potential: Optional[str] = None,
           extra_fixed: Optional[Any] = None, radius_factor: float = GATE_I_RADIUS_FACTOR,
           depth_factor: float = GATE_I_DEPTH_FACTOR, shell_A: float = GATE_I_SHELL_A,
           thresholds: GateIThresholds = GateIThresholds(),
           progress: Optional[Callable[[dict], None]] = None) -> dict[str, Any]:
    """Gate I quench driver (recommended protocol item 16).

    ``reference`` and every entry of ``frames`` are dicts with ``positions`` (N, 3) of the FULL system (indenter
    removed, i.e. positions only) and a label (``h_A``); the reference is the pre-contact frame.  The
    sub-volume is carved once from the reference positions with the contact radius a_c of the DEEPEST frame
    (``contact_radius_A``), the reference is quenched with the shell at its own positions, and every frame is
    quenched with the shell at the SAME reference positions and its interior started from its positions
    minus the harmonic extension of the shell mismatch (module docstring).  Returns the carve summary, the
    reference quench and one verdict per frame (:func:`gate_i_compare`) with the convergence of both quenches and its
    evaluation (:func:`gate_i_evaluation`, critic N1 / N2): ``verdict`` "pass" / "fail" only when the frame AND the
    reference quench converged and the defects are contained, else "not_evaluated" with ``passes`` None (the raw
    threshold outcome stays in ``passes_thresholds``).  This call quenches ONE carve; :func:`gate_i_frame` re-quenches
    an uncontained frame in enlarged carves.  For a depth series (S1) call it once per
    frame with that frame's a_c (:func:`gate_i_contact_radius`): one call with every frame quenches all of them in the
    deepest frame's carve.  At S1 scale use :func:`lammps_minimizer` (frozen_gate_i.json: LAMMPS CG on 4 A725 cores
    ~2.3e-6 s per iteration and atom, 9 min for a 125k-atom G06 frame; the harness scipy CG took 47 min on a shared
    GB10).
    """
    sp = np.asarray(species).astype(str)
    X0 = np.asarray(reference["positions"], float)
    reg = carve_subvolume(X0, axis_xy=axis_xy, surface_z0=surface_z0, L_xy=L_xy, contact_radius_A=contact_radius_A,
                          radius_factor=radius_factor, depth_factor=depth_factor, shell_A=shell_A, cutoff_A=cutoff_A,
                          extra_fixed=extra_fixed, thresholds=thresholds)
    ix = reg.indices
    spc = sp[ix]
    cell = quench_cell(X0[ix], L_xy, cutoff_A=cutoff_A)
    shift = np.array([0.0, 0.0, cutoff_A + 2.0 - X0[ix, 2].min()])  # keep the carved atoms inside the cell
    ref_q = minimizer(X0[ix] + shift, spc, cell, reg.fixed)
    ref_pos = ref_q.positions - shift
    out: dict[str, Any] = {"carve": reg.summary(), "reference": {"label": reference.get("label"),
                                                                  "energy_eV": ref_q.energy_eV,
                                                                  "converged": ref_q.converged, "steps": ref_q.steps,
                                                                  "fmax": ref_q.fmax_eV_per_A, "wall_s": ref_q.wall_s,
                                                                  "restarts": getattr(ref_q, "restarts", 0)},
                           "thresholds": dataclasses.asdict(thresholds), "frames": []}
    lo = np.zeros(3)
    hi = np.array([L_xy[0], L_xy[1], 1.0])
    pbc = (True, True, False)
    rule = rule or bond_rule_for(potential, spc)
    cn_ref = coordination_by_rule(ref_pos, spc, lo, hi, pbc, rule)
    for fr in frames:
        X = np.asarray(fr["positions"], float)[ix]
        dX = min_image(X - X0[ix], lo, hi, pbc)
        u = harmonic_extension(X0[ix], reg.fixed, dX, L_xy=L_xy)
        start = X0[ix] + dX - u
        start[reg.fixed] = X0[ix][reg.fixed]
        q = minimizer(start + shift, spc, cell, reg.fixed)
        verdict = gate_i_compare(ref_pos, q.positions - shift, spc, ref_energy_eV=ref_q.energy_eV, energy_eV=q.energy_eV,
                                 interior=reg.interior, L_xy=L_xy, axis_xy=axis_xy, surface_z0=surface_z0, rule=rule,
                                 cn_ref=cn_ref, thresholds=thresholds, carve=reg)
        verdict.update(label=fr.get("label"), h_A=fr.get("h_A"), converged=q.converged, steps=q.steps,
                       fmax=q.fmax_eV_per_A, wall_s=q.wall_s, restarts=getattr(q, "restarts", 0),
                       reference_converged=ref_q.converged, reference_fmax=ref_q.fmax_eV_per_A,
                       reference_steps=ref_q.steps)
        verdict.update(gate_i_evaluation(verdict, thresholds=thresholds))
        out["frames"].append(verdict)
        if progress is not None:
            progress(verdict)
    return out


def _carve_identity(reg: CarvedRegion) -> tuple:
    return (reg.indices.tobytes(), reg.fixed.tobytes(), reg.interior.tobytes())


def gate_i_frame(frame: Mapping[str, Any], reference: Mapping[str, Any], *, species: Any, L_xy: Sequence[float],
                 axis_xy: Sequence[float], surface_z0: float, contact_radius_A: float, minimizer: Minimizer,
                 cutoff_A: float, rule: Optional[BondRule] = None, potential: Optional[str] = None,
                 extra_fixed: Optional[Any] = None, radius_factor: float = GATE_I_RADIUS_FACTOR,
                 depth_factor: float = GATE_I_DEPTH_FACTOR, shell_A: float = GATE_I_SHELL_A,
                 thresholds: GateIThresholds = GateIThresholds(), growth: float = GATE_I_CARVE_GROWTH,
                 max_scale: float = GATE_I_CARVE_MAX_SCALE,
                 progress: Optional[Callable[[dict], None]] = None) -> dict[str, Any]:
    """Gate I of ONE frame with the containment rule (critic N2): :func:`gate_i` in the carve of ``radius_factor`` /
    ``depth_factor`` (default the frozen recipe), and while the frame and reference quenches converged but the
    defects are not contained, again in the next carve of :func:`gate_i_carve_schedule` (factors x ``growth`` per
    attempt up to ``max_scale`` x the first; ``max_scale`` 1 = no enlargement).  It stops when the frame is evaluated,
    when a quench did not converge (N1: not evaluated, a larger carve cannot fix it), when the carve no longer changes
    (whole lateral slab down to the bottom) or at the cap (not evaluated: defects not contained).

    Returns the verdict row of the LAST attempt (:func:`gate_i` frame verdict with ``verdict`` / ``evaluated`` /
    ``passes``) plus ``carve`` (the summary of the carve actually used), ``reference`` (its reference quench),
    ``radius_factor`` / ``depth_factor`` / ``shell_A`` used and ``containment``: the rule, every attempt (factors,
    carve summary, both quenches, n_changed, dPE, margin, verdict) and why the series stopped.  ``progress`` is
    called with each attempt's record."""
    schedule = gate_i_carve_schedule(radius_factor, depth_factor, growth=growth, max_scale=max_scale)
    X0 = np.asarray(reference["positions"], float)
    attempts: list[dict[str, Any]] = []
    last: Optional[dict[str, Any]] = None
    seen = None
    stop = "evaluated"
    for k, (rf, df) in enumerate(schedule):
        reg = carve_subvolume(X0, axis_xy=axis_xy, surface_z0=surface_z0, L_xy=L_xy, contact_radius_A=contact_radius_A,
                              radius_factor=rf, depth_factor=df, shell_A=shell_A, cutoff_A=cutoff_A,
                              extra_fixed=extra_fixed, thresholds=thresholds)
        ident = _carve_identity(reg)
        if seen is not None and ident == seen:
            stop = "carve_cannot_grow"
            break
        seen = ident
        res = gate_i([frame], reference, species=species, L_xy=L_xy, axis_xy=axis_xy, surface_z0=surface_z0,
                     contact_radius_A=contact_radius_A, minimizer=minimizer, cutoff_A=cutoff_A, rule=rule,
                     potential=potential, extra_fixed=extra_fixed, radius_factor=rf, depth_factor=df, shell_A=shell_A,
                     thresholds=thresholds)
        (v,) = res["frames"]
        last = {**v, "carve": res["carve"], "reference": res["reference"], "radius_factor": float(rf),
                "depth_factor": float(df), "shell_A": float(shell_A)}
        att = {"attempt": k, "radius_factor": float(rf), "depth_factor": float(df), "carve": res["carve"],
               "bottom_depth_A": reg.bottom_depth_A, "reference": res["reference"],
               "frame": {key: v.get(key) for key in ("converged", "steps", "fmax", "wall_s", "restarts")},
               **{key: v.get(key) for key in ("n_changed", "dPE_eV", "n_displaced", "passes_thresholds",
                                              "defect_margin_A", "defects_contained", "verdict",
                                              "not_evaluated_reasons")}}
        attempts.append(att)
        if progress is not None:
            progress(att)
        if v["evaluated"]:
            stop = "evaluated"
            break
        if not (v.get("converged") and v.get("reference_converged")):
            stop = "not_converged"
            break
        stop = "cap"  # not contained: the next carve of the schedule, if any
    assert last is not None
    out = dict(last)
    why = {"evaluated": "evaluated", "not_converged": "a quench did not converge (a larger carve cannot fix that)",
           "carve_cannot_grow": "the enlarged carve is the same atoms as the previous one (whole lateral slab down to "
                                "the bottom): it cannot grow further",
           "cap": f"defects still not contained at the cap of the carve schedule (radius / depth factor "
                  f"{schedule[-1][0]:.4g} / {schedule[-1][1]:.4g} = {float(max_scale):g} x the first attempt)"}[stop]
    if stop in ("carve_cannot_grow", "cap") and not out["evaluated"]:
        out["not_evaluated_reasons"] = list(out["not_evaluated_reasons"]) + [why]
    out["containment"] = {"rule": {"growth": float(growth), "max_scale": float(max_scale),
                                   "schedule": [list(x) for x in schedule]},
                          "n_attempts": len(attempts), "enlarged": len(attempts) > 1, "stopped": stop, "reason": why,
                          "attempts": attempts}
    return out


def gate_i_compare(ref_positions: Any, positions: Any, species: Any, *, ref_energy_eV: float, energy_eV: float,
                   interior: Any, L_xy: Sequence[float], axis_xy: Sequence[float], surface_z0: float,
                   rule: Optional[BondRule] = None, cn_ref: Optional[np.ndarray] = None,
                   thresholds: GateIThresholds = GateIThresholds(),
                   carve: Optional[CarvedRegion] = None) -> dict[str, Any]:
    """Compare a quenched frame with the quenched reference (same atoms, same fixed shell).

    Counts interior atoms whose heteronuclear-only coordination changed, the energy difference dPE, interior
    atoms displaced by more than ``displacement_A`` (after removing the mean far-field shift of the interior),
    and the centre dent (mean z change of reference top atoms within ``dent_radius_A`` of the axis; negative
    = a dent).  ``reversible``: nothing changed, nothing displaced and |dPE| < reversible_dpe_eV;
    ``irreversible`` = not reversible; ``passes`` (Gate I) = n_changed >= n_changed_min AND dPE >= dpe_min_eV.
    With ``carve`` (the region of :func:`carve_subvolume`, indices matching the positions): ``defect_margin_A`` =
    the smallest distance of a changed or displaced interior atom from the fixed shell (reference positions), and
    ``defects_contained`` = margin >= ``thresholds.defect_margin_A`` (False: quench again with a larger carve).
    """
    sp = np.asarray(species).astype(str)
    lo = np.zeros(3)
    hi = np.array([L_xy[0], L_xy[1], 1.0])
    pbc = (True, True, False)
    rule = rule or bond_rule_for(None, sp)
    R0 = np.asarray(ref_positions, float)
    R1 = np.asarray(positions, float)
    inter = np.asarray(interior, bool)
    cn0 = coordination_by_rule(R0, sp, lo, hi, pbc, rule) if cn_ref is None else cn_ref
    cn1 = coordination_by_rule(R1, sp, lo, hi, pbc, rule)
    ch = inter & (cn1 != cn0)
    dX = min_image(R1 - R0, lo, hi, pbc)
    dXr = dX - dX[inter].mean(0) if inter.any() else dX
    disp = np.linalg.norm(dXr, axis=1)
    moved = inter & (disp > thresholds.displacement_A)
    d = R0[:, :2] - np.asarray(axis_xy, float)
    L = np.asarray(L_xy, float)
    d -= L * np.round(d / L)
    top = (R0[:, 2] > surface_z0 - 0.6) & (np.hypot(d[:, 0], d[:, 1]) < thresholds.dent_radius_A)
    dpe = float(energy_eV - ref_energy_eV)
    n_ch = int(ch.sum())
    reversible = n_ch == 0 and int(moved.sum()) == 0 and abs(dpe) < thresholds.reversible_dpe_eV
    hist = {int(k): int(v) for k, v in zip(*np.unique(cn1[ch], return_counts=True))}
    out = {"n_changed": n_ch, "changed_cn_hist": hist, "dPE_eV": dpe, "n_displaced": int(moved.sum()),
           "max_displacement_A": float(disp[inter].max()) if inter.any() else 0.0,
           "center_dz_A": float(dX[top, 2].mean()) if top.any() else None,
           "reversible": bool(reversible), "irreversible": bool(not reversible),
           "passes": bool(n_ch >= thresholds.n_changed_min and dpe >= thresholds.dpe_min_eV), "rule": rule.name}
    if carve is not None:
        flagged = ch | moved
        if flagged.any():
            rr = np.hypot(d[flagged, 0], d[flagged, 1])
            to_shell = carve.bottom_depth_A - (surface_z0 - R0[flagged, 2])
            if carve.lateral_carved:
                to_shell = np.minimum(to_shell, carve.radius_A - rr)
            margin = float(to_shell.min())
        else:
            margin = None
        out["defect_margin_A"] = margin
        out["defects_contained"] = bool(margin is None or margin >= thresholds.defect_margin_A)
    return out


# ============================================================================================ run analysis
@dataclasses.dataclass
class FrameData:
    """One saved configuration of an indentation run (full sample or a subset, same atoms in every frame)."""

    step: int
    time_ps: float
    positions: np.ndarray  # (N, 3) A
    contact: Optional[np.ndarray] = None  # (N,) bool engine contact flag (atomistic tip) or sdf < 0
    fields: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)  # per-atom columns (e.g. c_cn[1])
    full: bool = True  # positions are the whole analysed atom set (False: e.g. only the contact atoms)
    areas: bool = True  # compute contact areas for this frame (False: a full snapshot duplicating a contact frame)
    label: str = ""  # key-frame role ("max_load", "end_hold", "final"); default: from the segment of the frame


@dataclasses.dataclass
class IndentationRunData:
    """Engine-independent input of :func:`analyze_indentation`.

    Curve arrays are per logged sample: time (ps), apex depth h (A, below z_s0), load (nN, +z on the tip =
    compression of the sample), segment labels ("approach", "load", "hold", "unload", "relax", ...) and
    optional channels (pen_fw_A, reaction_nN = the fixed-layer force sum, T_core_K, ...).  ``frames`` is a
    callable returning an iterator of :class:`FrameData` (read lazily; large runs never sit in memory).
    ``gate_ii_evidence``: intermediate-depth Gate II evidence (:func:`gate_ii_onset`; e.g.
    :func:`gate_ii_evidence_from_gate_i`), without which no hardness can be produced.
    """

    t_ps: np.ndarray
    h_A: np.ndarray
    load_nN: np.ndarray
    segment: np.ndarray
    species: np.ndarray  # (N,) of the frame atoms
    surface_z0_A: float
    axis_xy: tuple[float, float]
    lateral_L_A: tuple[float, float]
    lattice_constant_A: float
    channels: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    frames: Optional[Callable[[], Iterator[FrameData]]] = None
    geometry: Any = None  # physics.indenter.IndenterGeometry of an SDF tip (E2, contact from sdf < 0)
    tip_model: str = "rigid_sdf"  # or "atomistic" (contact from the engine flag)
    ideal_area_coefficient: float = BERKOVICH_C0
    face_azimuths_deg: tuple[float, ...] = (0.0, 120.0, 240.0)
    fixed_top_z: Optional[float] = None  # top of the fixed layer (A, lab z)
    bath_top_z: Optional[float] = None  # top of the thermostat layer
    potential: str = ""
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    gate_ii_evidence: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def mask(self, *labels: str) -> np.ndarray:
        return np.isin(self.segment, labels)


@dataclasses.dataclass
class AnalysisConfig:
    """Settings of :func:`analyze_indentation` (defaults: recipe_v0 / recommended protocol items 15-18)."""

    window_ps: float = DEFAULT_WINDOW_PS
    tau_ac_ps: Optional[float] = None  # None: estimate from the log if sampled every <= 5 fs, else 20 fs
    h0_rule: str = "elastic_fit"  # depth zero used by E2 ("elastic_fit" or "first_contact")
    fit_load_range_nN: tuple[float, float] = (10.0, 150.0)  # Sneddon fit window (sharp tips)
    hertz_fit_h_max_A: float = 4.0  # sphere fit window
    first_contact_floor_nN: float = 1.0
    precontact_gap_A: float = 2.0  # samples with the tip >= this above the surface define the thermal load noise
    plateau_depths_A: Optional[Sequence[float]] = None  # h_eval list (default: 0.8, 1.0 x h_max and h_max)
    estimators: Sequence[str] = ESTIMATORS
    closing_radius_A: float = CLOSING_RADIUS_A
    grid_A: float = 0.2
    coordination_every: int = 0  # 0: coordination only on the key frames (first, max load, end of hold, final)
    gate_ii: GateIIThresholds = GateIIThresholds()
    unload_threshold_nN: float = 2.0
    op_fit_range: tuple[float, float] = (0.2, 0.95)
    defect_linkage_A: float = 3.5
    t_el_A: Optional[float] = None  # for the defect-extent acceptance
    frame_steps: Optional[Sequence[int]] = None  # analyse only these frames (default: all)
    contact_rule: str = "force_fraction"  # B1 estimated contact (CONTACT_RULES); "touching" = every atom touching the tip
    contact_force_fraction: float = CONTACT_FORCE_FRACTION


def _sphere_radius(run: IndentationRunData) -> Optional[float]:
    g = run.geometry
    return float(g.tip_radius_A) if g is not None and getattr(g, "shape", "") == "sphere" else None


TIP_FORCE_FIELD = "tip_force"  # optional per-atom frame field: tip force magnitude (eV/A) of an atomistic tip


def _frame_contact(run: IndentationRunData, fr: FrameData, apex_z: float) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """(touching atoms, per-atom tip force up to a constant or None) of a frame.

    Touching: the engine flag if the frame has one, else sdf < 0 of the run's SDF tip.  Force: an SDF tip gives
    delta^2 (K delta^2 / K) from the frame positions; an atomistic tip only through the ``tip_force`` field."""
    sdf_tip = run.geometry is not None and run.tip_model == "rigid_sdf"
    delta = None
    if sdf_tip:
        delta = sdf_penetration(fr.positions, run.geometry, (run.axis_xy[0], run.axis_xy[1], apex_z), run.lateral_L_A)
    if fr.contact is not None:
        touching = np.asarray(fr.contact, bool)
    elif delta is not None:
        touching = delta > 0.0
    else:
        raise ValueError("frames carry no contact flag and the run has no SDF geometry")
    if TIP_FORCE_FIELD in fr.fields:
        force = np.asarray(fr.fields[TIP_FORCE_FIELD], float)
    elif delta is not None:
        force = delta * delta
    else:
        force = None
    return touching, force


def analyze_indentation(run: IndentationRunData, cfg: AnalysisConfig = AnalysisConfig()) -> dict[str, Any]:
    """Full analysis of one indentation run (module docstring); returns a JSON-serialisable dict.

    Keys: load (windowing, noise, tau_ac, Newton-3), depth_zero, pop_in, frames_table (per frame: h, h_c,
    windowed P, n_contact, areas E1-E5, p_m E1-E5), plateau_LM_R10 (per h_eval and estimator), pm_1A_bins,
    hold, unload (h_f, loop work, Oliver-Pharr diagnostic), residual (height map), gate_ii (with its onset from
    ``run.gate_ii_evidence``), gate_iii, hardness (available + reason), coordination (key frames), defect_extent,
    temperature.
    """
    t, h, P = (np.asarray(x, float) for x in (run.t_ps, run.h_A, run.load_nN))
    iload = run.mask("approach", "load")
    # hold / unloading branches start at the last sample of the previous segment (the end of loading / hold)
    ihold = _with_previous(run.mask("hold"))
    iunl = _with_previous(run.mask("unload", "relax"))
    unl = _with_previous(run.mask("unload"))
    dt_log = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    tau = cfg.tau_ac_ps
    if tau is None and dt_log <= 0.005:
        tau = integrated_autocorrelation_time(P[iload & (P > 20.0)], dt_log, detrend_samples=max(3, int(0.5 / dt_log)))
    wl = windowed_load(t, P, cfg.window_ps, tau_ac_ps=tau)
    Pw = wl.values
    noise = in_contact_scatter(P, Pw, t, mask=iload)
    out: dict[str, Any] = {"n_samples": int(len(t)), "load": {"window": wl.summary(), "noise": noise}}
    if "reaction_nN" in run.channels:
        out["load"]["newton3"] = newton3_balance(P, run.channels["reaction_nN"])
    hl, Pl, Pwl = h[iload], P[iload], Pw[iload]
    h_max = float(hl.max())
    out["h_max_A"] = h_max
    out["P_at_hmax_windowed_nN"] = float(Pwl[-5:].mean())
    # ---- pop-in candidates and depth zero
    pops = load_drop_events(hl, Pwl, noise_nN=noise["windowed_nN"], t_ps=t[iload])
    out["pop_in"] = {"load_drop_events": pops, "first_A": pops[0]["h_peak_A"] if pops else None}
    first_pop = out["pop_in"]["first_A"]
    pre = iload & (h < -cfg.precontact_gap_A)  # tip at least this far above the surface: thermal noise only
    sigma_pre = float(np.std(Pw[pre])) if pre.sum() > 1 else 0.0
    # depth zero on the same depth variable as E2 (h_c = h - pen_fw with a soft SDF wall): a fit on the nominal h
    # would put the wall penetration into h0, and E2 = A(h_c - h0) would subtract it twice
    pen = run.channels.get("pen_fw_A")
    hcl = contact_depth(h, pen)[iload] if pen is not None else hl
    h_fc = first_contact_depth(hcl, Pwl, noise_nN=sigma_pre, floor_nN=cfg.first_contact_floor_nN)
    out["load"]["noise"]["precontact_windowed_sd_nN"] = sigma_pre
    R_sph = _sphere_radius(run)
    if R_sph is not None:
        fit = hertz_fit(hcl, Pl, R_sph, cfg.hertz_fit_h_max_A)
    else:
        # the tip's own (rounded) profile: a sharp-cone fit would absorb the bluntness into h0
        pre_pop = hl < first_pop if first_pop is not None else np.ones(len(hl), bool)
        fit = sneddon_fit(hcl[pre_pop], Pl[pre_pop], Pw=Pwl[pre_pop], geometry=run.geometry,
                          load_range_nN=cfg.fit_load_range_nN, C0=run.ideal_area_coefficient)
    fit["depth_variable"] = "h_c = h - pen_fw" if pen is not None else "h"
    h0_fit = fit.get("h0_A")
    h0 = h0_fit if cfg.h0_rule == "elastic_fit" and h0_fit is not None else (h_fc if h_fc is not None else 0.0)
    out["depth_zero"] = {"first_contact_A": h_fc, "elastic_fit": fit, "rule_used": cfg.h0_rule, "h0_A": float(h0)}
    # ---- per-frame areas
    d_dil = surface_dilation_A(run.lattice_constant_A)
    out["d_dilation_A"] = d_dil
    step_log = np.asarray(run.channels.get("step", np.arange(len(t))), float)
    rows: list[dict[str, Any]] = []
    key_frames: dict[str, FrameData] = {}
    first_frame: Optional[FrameData] = None
    last_frame: Optional[FrameData] = None
    cn_rows: list[dict[str, Any]] = []
    L = np.asarray(run.lateral_L_A, float)
    lo = np.zeros(3)
    hi = np.array([L[0], L[1], 1.0])
    pbc = (True, True, False)
    rule = bond_rule_for(run.potential, run.species)
    cn0 = None
    if cfg.contact_rule not in CONTACT_RULES:
        raise ValueError(f"unknown contact_rule {cfg.contact_rule!r}; known: {CONTACT_RULES}")
    rules_used: set[str] = set()
    if run.frames is not None:
        want = None if cfg.frame_steps is None else set(int(s) for s in cfg.frame_steps)
        for fr in run.frames():
            if want is not None and int(fr.step) not in want:
                continue
            i = int(np.argmin(np.abs(t - fr.time_ps)))
            hh = float(np.interp(fr.time_ps, t, h))
            seg = str(run.segment[i])
            if not fr.areas:
                rows_skipped = True
            else:
                rows_skipped = False
            apex_z = run.surface_z0_A - hh
            if not rows_skipped:
                touching, force = _frame_contact(run, fr, apex_z)
                if cfg.contact_rule == "force_fraction" and force is not None:
                    contact = estimated_contact_mask(touching, force, fraction=cfg.contact_force_fraction)
                    rule_used = "force_fraction"
                else:
                    contact = touching
                    rule_used = "touching" if cfg.contact_rule == "touching" else "touching (no per-atom tip force)"
                rules_used.add(rule_used)
                xy = projected_contact_xy(fr.positions, contact, run.axis_xy, L)
                hc = hh - float(np.interp(fr.time_ps, t, pen)) if pen is not None else hh
                ar = contact_areas(xy, hh, h0_A=h0, h_c_A=hc, d_A=d_dil, C0=run.ideal_area_coefficient,
                                   geometry=run.geometry, face_azimuths_deg=run.face_azimuths_deg,
                                   closing_radius_A=cfg.closing_radius_A, grid_A=cfg.grid_A)
                row = {"step": int(fr.step), "time_ps": float(fr.time_ps), "phase": seg, "h_A": hh, "h_c_A": hc,
                       "P_nN": float(Pw[i]), **{k: v for k, v in ar.items() if k != "tri_per_face_A"},
                       "n_touching": int(touching.sum()), "contact_rule": rule_used}
                for e in cfg.estimators:
                    row["pm_" + e] = float(mean_pressure_GPa(row["P_nN"], ar[e]))
                rows.append(row)
            if not fr.full:
                continue
            if first_frame is None:
                first_frame = fr
                cn0 = coordination_by_rule(fr.positions, run.species, lo, hi, pbc, rule)
            last_frame = fr
            if cfg.coordination_every and (len(rows) - 1) % cfg.coordination_every == 0:
                cn_rows.append(_coord_row(fr, hh, seg, cn0, run, rule, lo, hi, pbc))
            if fr.label:
                if fr.label != "initial":
                    key_frames[fr.label] = fr
            elif seg in ("load", "approach"):
                key_frames["max_load"] = fr
            elif seg == "hold":
                key_frames["end_hold"] = fr
    out["frames_table"] = rows
    out["contact"] = {"rule": cfg.contact_rule, "force_fraction": cfg.contact_force_fraction,
                      "rules_used": sorted(rules_used), "name": "estimated contact",
                      "note": "E3-E5 use the estimated-contact atoms (touching atoms with tip force >= fraction x mean); "
                              "'touching (no per-atom tip force)' frames keep every touching atom"}
    out["bond_rule"] = rule.name
    # ---- plateau
    lr = [r for r in rows if r["phase"] in ("load", "approach")]
    hL = np.array([r["h_A"] for r in lr])
    pm = {e: np.array([r["pm_" + e] for r in lr]) for e in cfg.estimators}
    evals = list(cfg.plateau_depths_A) if cfg.plateau_depths_A is not None else \
        sorted({round(0.8 * h_max, 1), evaluation_depth(h_max)})
    # ---- unloading / Gate II
    Pmax = float(P[ihold][-20:].mean()) if ihold.any() else float(Pwl[-1])
    hm = float(h[ihold][-1]) if ihold.any() else h_max
    out["hold"] = ({"P_start_nN": float(P[ihold][:20].mean()), "P_end_nN": float(P[ihold][-20:].mean()),
                    "rel_change": float((P[ihold][-20:].mean() - P[ihold][:20].mean()) / P[ihold][:20].mean())}
                   if ihold.sum() > 40 else None)
    hf = unloading_final_depth(h[unl], Pw[unl], threshold_nN=max(cfg.unload_threshold_nN, 2 * noise["windowed_nN"])) \
        if unl.any() else None
    loop = loop_work(h, P, iload, iunl) if iunl.any() else None
    area_fn = (lambda x: float(area_rounded(x, run.geometry, run.ideal_area_coefficient))) if run.geometry is not None \
        else (lambda x: run.ideal_area_coefficient * x * x)
    op = oliver_pharr(h[unl], P[unl], P_max_nN=Pmax, h_max_A=hm, area_fn=area_fn,
                      fit_range=cfg.op_fit_range) if unl.sum() > 5 else None
    out["unload"] = {"P_start_nN": Pmax, "h_start_A": hm, "h_f_A": hf,
                     "hf_over_hmax": (hf / h_max) if hf is not None else None, "loop": loop,
                     "oliver_pharr_diagnostic": op}
    residual = None
    last_phase = None if last_frame is None else str(run.segment[int(np.argmin(np.abs(t - last_frame.time_ps)))])
    if last_frame is not None and iunl.any() and last_phase in ("unload", "relax"):
        imax = [k for k, r in enumerate(rows) if r["phase"] in ("load", "approach")]
        a_c = math.sqrt(max(rows[imax[-1]]["E3"], 1.0) / math.pi) if imax and np.isfinite(rows[imax[-1]]["E3"]) else None
        residual = residual_impression(last_frame.positions, run.axis_xy, L, run.surface_z0_A,
                                       bin_A=run.lattice_constant_A, contact_radius_A=a_c,
                                       positions_initial=first_frame.positions if first_frame is not None else None)
    out["residual"] = residual
    out["gate_ii"] = gate_ii(residual_depth_A=None if residual is None else residual["center_depth_A"],
                             dissipated_fraction=None if loop is None else loop["dissipated_fraction"],
                             hf_over_hmax=out["unload"]["hf_over_hmax"], h_max_A=h_max, thresholds=cfg.gate_ii)
    # the full unload judges Gate II at h_max only; the onset depth that Gate III needs comes from intermediate-depth
    # evidence (placing it at h_max would put it inside every Gate III window: no hardness could ever pass)
    onset = gate_ii_onset(run.gate_ii_evidence, gate_ii_at_hmax=out["gate_ii"], h_max_A=h_max, thresholds=cfg.gate_ii)
    out["gate_ii"]["onset"] = onset
    h_onset = onset["h_onset_A"]
    if first_pop is not None and h_onset is None:
        out["pop_in"]["note"] = "a load drop is not Gate II (developed plasticity needs the residual state)"
    out["plateau_LM_R10"] = {f"h{he:g}": {e: plateau_metric(hL, pm[e], he, h_onset=h_onset) for e in cfg.estimators}
                             for he in evals}
    out["gate_iii"] = gate_iii(hL, pm, evaluation_depth(h_max), h_onset=h_onset) if len(hL) else None
    out["hardness"] = hardness_verdict(out)
    bins = np.arange(0.0, math.ceil(h_max) + 1.0, 1.0)

    def _finite_mean(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    out["pm_1A_bins"] = [dict(h_bin=[float(a), float(b)], P_nN=_finite_mean([r["P_nN"] for r in lr if a <= r["h_A"] < b]),
                              **{e: _finite_mean([r["pm_" + e] for r in lr if a <= r["h_A"] < b]) for e in cfg.estimators})
                         for a, b in zip(bins[:-1], bins[1:]) if any(a <= r["h_A"] < b for r in lr)]
    deepest = max(lr, key=lambda r: r["h_A"]) if lr else None
    out["deepest_loading_frame"] = None if deepest is None else {k: deepest[k] for k in deepest}
    # ---- coordination / defect extent on key frames
    if last_frame is not None:
        key_frames["final"] = last_frame
    if cn0 is not None:
        ext = {}
        for name, fr in key_frames.items():
            i = int(np.argmin(np.abs(t - fr.time_ps)))
            hh = float(np.interp(fr.time_ps, t, h))
            crow = _coord_row(fr, hh, str(run.segment[i]), cn0, run, rule, lo, hi, pbc, keep_mask=True)
            flagged = crow.pop("_changed_bulk")
            cn_rows.append({"frame": name, **crow})
            ext[name] = defect_extent(fr.positions, flagged, axis_xy=run.axis_xy, surface_z0=run.surface_z0_A,
                                      depth_A=max(hh, 0.0), L_xy=L, fixed_top_z=run.fixed_top_z,
                                      bath_top_z=run.bath_top_z, linkage_A=cfg.defect_linkage_A)
            if cfg.t_el_A is not None:
                ext[name]["check"] = defect_extent_check(ext[name], t_el_A=cfg.t_el_A, L_A=float(L.min()))
        out["defect_extent"] = ext
    out["coordination"] = cn_rows
    if "T_core_K" in run.channels:
        Tc = np.asarray(run.channels["T_core_K"], float)
        out["temperature"] = {"T_core_at_hmax_K": float(Tc[iload][-20:].mean()),
                              "T_core_max_1ps_K": float(running_mean(Tc, max(1, int(round(1.0 / dt_log)))).max())}
    return out


def _with_previous(mask: np.ndarray) -> np.ndarray:
    """``mask`` plus the sample just before its first True sample (segment boundaries are shared)."""
    m = np.asarray(mask, bool).copy()
    idx = np.flatnonzero(m)
    if len(idx) and idx[0] > 0:
        m[idx[0] - 1] = True
    return m


def _coord_row(fr: FrameData, hh: float, seg: str, cn0: np.ndarray, run: IndentationRunData, rule: BondRule, lo, hi, pbc,
               keep_mask: bool = False) -> dict[str, Any]:
    cc = coordination_changes(None, fr.positions, run.species, lo, hi, pbc, rule=rule, cn_ref=cn0)
    row = {"step": int(fr.step), "h_A": hh, "phase": seg, "n_changed_bulk": cc["n_changed_bulk"],
           "bulk_to_5plus": cc["bulk_to_5plus"], "bulk_to_3minus": cc["bulk_to_3minus"],
           "hist_bulk_origin": cc["hist_bulk_origin"]}
    if keep_mask:
        row["_changed_bulk"] = cc["changed_bulk"]
    return row


def json_ready(x: Any) -> Any:
    """Recursively convert numpy scalars/arrays and dataclasses into JSON-serialisable Python objects."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return json_ready(dataclasses.asdict(x))
    if isinstance(x, Mapping):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_ready(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_ready(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


__all__ = [
    "AnalysisConfig", "CarvedRegion", "ESTIMATORS", "ESTIMATOR_NAMES", "FrameData", "GateIIThresholds",
    "GateIThresholds", "IndentationRunData", "QuenchResult", "WindowedLoad", "analyze_indentation",
    "area_closed_discs", "area_dilated_hull", "area_nominal", "area_rounded", "area_triangle_fit", "block_average",
    "CONTACT_FORCE_FRACTION", "CONTACT_RULES", "TIP_FORCE_FIELD", "estimated_contact_mask", "lammps_minimizer",
    "sdf_penetration", "restarting_minimizer", "GATE_I_DEPTH_FACTOR", "GATE_I_FMAX_EV_PER_A", "GATE_I_MINIMIZER", "GATE_I_RADIUS_FACTOR",
    "GATE_I_SHELL_A", "frozen_gate_i_recipe", "gate_i_contact_radius", "GATE_I_CARVE_GROWTH", "GATE_I_CARVE_MAX_SCALE",
    "GATE_I_NOT_EVALUATED", "GATE_I_VERDICTS", "gate_i_carve_schedule", "gate_i_evaluation", "gate_i_evaluation_rule",
    "gate_i_first_irreversibility", "gate_i_frame", "gate_i_frame_status",
    "carve_subvolume", "contact_areas", "contact_depth", "coordination_changes", "defect_extent",
    "defect_extent_check", "equivalent_cone_half_angle_deg", "estimator_code", "evaluation_depth",
    "first_contact_depth", "force_balance_check", "force_weighted_penetration", "gate_i", "gate_i_compare", "gate_ii",
    "gate_ii_evidence_from_gate_i", "gate_ii_onset", "gate_iii", "hardness_verdict", "harmonic_extension",
    "height_map", "hertz_fit", "hertz_load", "in_contact_scatter", "integrated_autocorrelation_time", "jax_minimizer",
    "json_ready", "layer_force_balance", "load_drop_events", "loop_work", "mean_pressure_GPa", "newton3_balance",
    "oliver_pharr", "plateau_metric", "projected_contact_xy", "quench_cell", "residual_impression", "running_mean",
    "sneddon_cone_fit", "sneddon_cone_load", "sneddon_fit", "sneddon_profile_table", "sneddon_tip_load",
    "species_balanced_stress", "surface_dilation_A", "unloading_final_depth", "windowed_load",
]
