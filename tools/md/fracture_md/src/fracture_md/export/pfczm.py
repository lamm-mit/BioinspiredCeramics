"""Versioned, machine-readable PF-CZM property bundle.

This module is the hand-off from the atomistic harness to a (JAX) phase-field
cohesive-zone (PF-CZM) fracture solver.  A bundle contains

* the anisotropic elastic stiffness/compliance of the crystal (crystal frame,
  Voigt, GPa) plus an isotropic Voigt-Reuss-Hill equivalent,
* one or more atomistic traction-separation laws (opening in Å, traction in
  GPa, energy per area in J/m^2) with UBER fits and an explicit post-peak
  softening branch,
* DERIVED PF-CZM parameter sets (E', f_t, G_c, Irwin length, Griffith K
  estimate, recommended softening law and the Wu (2017) degradation-function
  coefficients), each carrying the model used to derive it,
* typed, explicitly-statused extension blocks (tangential, mixed-mode,
  defects, temperature, strain rate, uncertainty samples, grain boundaries,
  interfaces, microstructure) so that a downstream solver can test for the
  presence of optional physics instead of guessing,
* the full annotated :class:`PropertyRecord` list the bundle was built from,
  for audit, and an integrity block (sha256 of the canonical JSON content,
  of every inline array and of the optional NPZ sidecar).

Physical interpretation (read before using the numbers in a continuum model)
----------------------------------------------------------------------------
Atomistic cleavage of a perfect crystal gives *ideal* quantities: the peak
traction is an ideal (theoretical) cleavage strength of order E/10 and the
work of separation is the ideal Griffith energy 2γ.  Consequently the Irwin
length l_ch = E' G_c / f_t^2 built from them is of order a nanometre.  These
are the ideal-strength / ideal-cleavage limits of the PF-CZM parameters, NOT
the macroscopic tensile strength f_t or fracture energy G_f of a real ceramic
(which are governed by flaws, microstructure, R-curve toughening and
temperature).  The bundle labels every such number as atomistic_ideal and
derived where applicable; the Griffith K estimate is a model estimate and can
never be stored as, or mistaken for, a measured K_Ic.

Conventions
-----------
* Voigt order xx, yy, zz, yz, xz, xy.  C_voigt maps engineering strains to
  stresses; S_voigt = inv(C_voigt) therefore uses engineering shear strains
  (S_ijkl = S_mn / (2 or 4) for shear indices).
* Tensors are stored in the crystal Cartesian frame; `rotation_crystal_to_lab`
  (row-major) maps crystal-frame vectors to the solver's lab frame.
* Stress/traction positive = tensile.  Openings are normal (mode I) or
  tangential displacement jumps across the cut plane.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, Union

import numpy as np
from pydantic import Field, field_validator, model_validator

from fracture_md.records import decide_status, make_record
from fracture_md.schema.common import (
    CheckResult,
    Citation,
    Conditions,
    DefectState,
    Provenance,
    StrictModel,
    Uncertainty,
    UncertaintyKind,
    ValidationStatus,
)
from fracture_md.schema.material import (
    Candidate,
    Composition,
    InterfaceKind,
    InterfaceSpec,
    Microstructure,
    Phase,
)
from fracture_md.schema.potential import PotentialCard, PropertyClass
from fracture_md.schema.properties import (
    CohesiveLawFit,
    Derivation,
    MethodSpec,
    PropertyName,
    PropertyOrigin,
    PropertyRecord,
    PropertyScale,
    TractionSeparationLaw,
)
from fracture_md.units import Quantity, Unit, UnitError, conversion_factor, q

BUNDLE_FORMAT = "fracture-md.pfczm-bundle"
BUNDLE_VERSION = "0.2.0"

# 1 GPa * 1 Å = 1e9 Pa * 1e-10 m = 0.1 J/m^2 (exact).
GPA_ANGSTROM_TO_J_PER_M2 = 0.1
# Floors for relative differences (|x - ref| / max(|ref|, eps)): energies per area, tractions.
_EPS_W = 1e-6  # J/m^2
_EPS_T = 1e-6  # GPa

ExtensionStatus = Literal["not_computed", "computed", "partial"]

CITATIONS_WU = [
    Citation(
        key="Wu2017",
        text="J.-Y. Wu, A unified phase-field theory for the mechanics of damage and quasi-brittle "
        "failure, J. Mech. Phys. Solids 103 (2017) 72-99",
        doi="10.1016/j.jmps.2017.03.015",
    ),
    Citation(
        key="WuNguyen2018",
        text="J.-Y. Wu, V.P. Nguyen, A length scale insensitive phase-field damage model for brittle "
        "fracture, J. Mech. Phys. Solids 119 (2018) 20-42",
        doi="10.1016/j.jmps.2018.06.006",
    ),
    Citation(
        key="Cornelissen1986",
        text="H.A.W. Cornelissen, D.A. Hordijk, H.W. Reinhardt, Experimental determination of crack "
        "softening characteristics of normalweight and lightweight concrete, Heron 31(2) (1986) 45-56",
    ),
]

CITATION_BARNETT_LOTHE = Citation(
    key="BarnettLothe1973",
    text="D.M. Barnett, J. Lothe, Synthesis of the sextic and the integral formalism for dislocations, "
    "Greens function and surface waves in anisotropic elastic solids, Phys. Norv. 7 (1973) 13-19; "
    "energy release rate G = 1/2 K.L^-1.K, see T.C.T. Ting, Anisotropic Elasticity (OUP 1996)",
)

ATOMISTIC_SCALE_NOTE = (
    "Atomistic, defect-free, 0 K (unless conditions say otherwise) ideal-cleavage values: f_t is an "
    "ideal cleavage strength and G_c the ideal Griffith work of separation (2*gamma for a rigid cut). "
    "The resulting Irwin length is nanometric. These are ideal-strength / ideal-cleavage LIMITS of the "
    "PF-CZM parameters, NOT macroscopic tensile strength or fracture energy of a real ceramic."
)


# =============================================================================
# Errors / warnings
# =============================================================================
class BundleVersionError(ValueError):
    """Bundle format/major version cannot be read by this code."""


class BundleVersionWarning(UserWarning):
    """Bundle was written by a newer minor/patch version of the format."""


class BundleIntegrityError(ValueError):
    """A sha256 recorded in the integrity block does not match the content."""


class BundleConsistencyWarning(UserWarning):
    """A physics consistency check on a loaded bundle failed (e.g. ∫T dδ != W)."""


# =============================================================================
# Small helpers
# =============================================================================
_STATUS_RANK = {
    ValidationStatus.FAILED: 0,
    ValidationStatus.OUT_OF_DOMAIN: 1,
    ValidationStatus.UNVALIDATED: 2,
    ValidationStatus.MODEL_ESTIMATE: 3,
    ValidationStatus.NUMERICALLY_VERIFIED: 4,
    ValidationStatus.REFERENCE_IMPLEMENTATION_AGREEMENT: 5,
    ValidationStatus.LITERATURE_MODEL_AGREEMENT: 6,
    ValidationStatus.VALIDATED_AGAINST_DFT: 7,
    ValidationStatus.VALIDATED_AGAINST_EXPERIMENT: 8,
}


def worst_status(statuses: Iterable[ValidationStatus]) -> ValidationStatus:
    """Weakest evidence among `statuses` (FAILED < OUT_OF_DOMAIN < UNVALIDATED < ...)."""
    s = list(statuses)
    if not s:
        return ValidationStatus.UNVALIDATED
    return min(s, key=lambda x: _STATUS_RANK[ValidationStatus(x)])


def parse_version(v: str) -> tuple[int, int, int]:
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", v.strip())
    if not m:
        raise BundleVersionError(f"not a semantic version: {v!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _require_compatible(qty: Optional[Quantity], like: Unit, name: str) -> None:
    if qty is None:
        return
    try:
        conversion_factor(qty.unit, like)
    except UnitError as e:
        raise ValueError(f"{name}: unit {qty.unit.value!r} is not compatible with {like.value!r} ({e})") from e


def _as_list(qty: Quantity) -> list:
    return qty.value if isinstance(qty.value, list) else [qty.value]


def _finite(a: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def array_sha256(a: Any) -> str:
    """sha256 over shape + float64 C-order bytes (platform independent for IEEE little-endian)."""
    arr = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
    h = hashlib.sha256()
    h.update(("shape=" + ",".join(str(s) for s in arr.shape) + ";").encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def file_sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# Elasticity helpers (pure numpy; independent of other modules on purpose)
# =============================================================================
_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def _voigt_index(i: int, j: int) -> int:
    return _VOIGT_PAIRS.index((min(i, j), max(i, j)))


def stiffness_voigt_to_tensor(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    T = np.empty((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    T[i, j, k, l] = C[_voigt_index(i, j), _voigt_index(k, l)]
    return T


def tensor_to_stiffness_voigt(T: np.ndarray) -> np.ndarray:
    C = np.empty((6, 6))
    for m, (i, j) in enumerate(_VOIGT_PAIRS):
        for n, (k, l) in enumerate(_VOIGT_PAIRS):
            C[m, n] = T[i, j, k, l]
    return C


def compliance_voigt_to_tensor(S: np.ndarray) -> np.ndarray:
    """Engineering-shear Voigt compliance -> 4th-order tensor (factors 1, 1/2, 1/4)."""
    S = np.asarray(S, dtype=float)
    T = np.empty((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    m, n = _voigt_index(i, j), _voigt_index(k, l)
                    T[i, j, k, l] = S[m, n] * (1.0 if m < 3 else 0.5) * (1.0 if n < 3 else 0.5)
    return T


def rotate_tensor4(T: np.ndarray, R: np.ndarray) -> np.ndarray:
    """T'_ijkl = R_ia R_jb R_kc R_ld T_abcd  (R maps old-frame vectors to new-frame vectors)."""
    R = np.asarray(R, dtype=float)
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, T, optimize=True)


def vrh_moduli(C: np.ndarray, S: np.ndarray) -> dict[str, float]:
    """Voigt, Reuss and Hill bulk/shear moduli and Hill E, nu (units of C)."""
    C, S = np.asarray(C, float), np.asarray(S, float)
    BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2.0 * (C[0, 1] + C[1, 2] + C[0, 2])) / 9.0
    GV = (C[0, 0] + C[1, 1] + C[2, 2] - (C[0, 1] + C[1, 2] + C[0, 2]) + 3.0 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0
    BR = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2] + 2.0 * (S[0, 1] + S[1, 2] + S[0, 2]))
    GR = 15.0 / (4.0 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4.0 * (S[0, 1] + S[1, 2] + S[0, 2]) + 3.0 * (S[3, 3] + S[4, 4] + S[5, 5]))
    BH, GH = 0.5 * (BV + BR), 0.5 * (GV + GR)
    EH = 9.0 * BH * GH / (3.0 * BH + GH)
    nuH = (3.0 * BH - 2.0 * GH) / (2.0 * (3.0 * BH + GH))
    return dict(BV=BV, BR=BR, BH=BH, GV=GV, GR=GR, GH=GH, EH=EH, nuH=nuH)


def directional_youngs_from_compliance(S_voigt: np.ndarray, direction: Sequence[float]) -> float:
    """E(n) = 1 / (n_i n_j n_k n_l S_ijkl) in the units reciprocal to S."""
    n = np.asarray(direction, float)
    n = n / np.linalg.norm(n)
    St = compliance_voigt_to_tensor(S_voigt)
    return float(1.0 / np.einsum("i,j,k,l,ijkl->", n, n, n, n, St))


def _barnett_lothe_L(Ct: np.ndarray, m: np.ndarray, n: np.ndarray, n_quad: int = 256) -> np.ndarray:
    """Barnett-Lothe tensor L = -(1/π) ∫_0^π N3(ω) dω for the plane spanned by (m, n).

    N3 = (mn)(nn)^-1(nm) - (mm), (ab)_ik = a_j C_ijkl b_l, with (m, n) rotated by ω
    about t = m x n (the crack front).  The integrand is π-periodic and smooth, so
    the trapezoidal rule converges exponentially.  L is real, symmetric, positive
    definite and depends only on t.
    """
    w = np.arange(n_quad) * (np.pi / n_quad)
    mw = np.cos(w)[:, None] * m[None, :] + np.sin(w)[:, None] * n[None, :]
    nw = -np.sin(w)[:, None] * m[None, :] + np.cos(w)[:, None] * n[None, :]
    nn = np.einsum("wj,ijkl,wl->wik", nw, Ct, nw)
    nm = np.einsum("wj,ijkl,wl->wik", nw, Ct, mw)
    mn = np.einsum("wj,ijkl,wl->wik", mw, Ct, nw)
    mm = np.einsum("wj,ijkl,wl->wik", mw, Ct, mw)
    N3 = mn @ np.linalg.solve(nn, nm) - mm
    return -N3.mean(axis=0)


def plane_strain_modulus_anisotropic(
    C_voigt: np.ndarray,
    normal: Sequence[float],
    propagation: Optional[Sequence[float]] = None,
    n_angles: int = 36,
    n_quad: int = 256,
) -> dict[str, Any]:
    """Anisotropic mode-I plane-strain modulus E' for a crack on the plane with `normal`.

    Uses the Barnett-Lothe / Stroh result for a straight crack in a homogeneous
    anisotropic solid, G = (1/2) K·L⁻¹·K, so for pure mode I loading
    (K = K_I n):  E' := K_I²/G = 2 / (n·L⁻¹·n).  For an isotropic solid this
    reduces exactly to E/(1-ν²) (checked in tests).  E' depends on the crack
    front direction t = m × n (m = propagation direction in the plane).
    Assumptions: linear elasticity, straight crack front, plane strain, pure
    mode-I K (any mode coupling induced by anisotropy is neglected).

    Returns dict(value, min, max, mean, at_propagation (or None), propagation_used).
    Units follow C_voigt.
    """
    Ct = stiffness_voigt_to_tensor(C_voigt)
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    trial = np.array([0.3141, 0.5926, 0.5358])  # generic, not parallel to low-index planes
    a = np.cross(n, trial)
    a /= np.linalg.norm(a)
    b = np.cross(n, a)

    def eprime(mvec):
        mvec = mvec - np.dot(mvec, n) * n
        mvec = mvec / np.linalg.norm(mvec)
        L = _barnett_lothe_L(Ct, mvec, n, n_quad)
        return float(2.0 / (n @ np.linalg.solve(L, n)))

    vals = np.array([eprime(np.cos(t) * a + np.sin(t) * b) for t in np.linspace(0, np.pi, n_angles, endpoint=False)])
    at_prop = eprime(np.asarray(propagation, float)) if propagation is not None else None
    allv = np.append(vals, at_prop) if at_prop is not None else vals
    return dict(
        value=at_prop if at_prop is not None else float(vals.mean()),
        min=float(allv.min()),
        max=float(allv.max()),
        mean=float(vals.mean()),
        at_propagation=at_prop,
        propagation_used=None if propagation is None else list(map(float, propagation)),
    )


# =============================================================================
# Wu (2017) PF-CZM coefficients and normalized softening laws
# =============================================================================
# κ = -k0 G_f / f_t^2 (initial softening slope, normalized) and ŵ = w_c f_t / G_f
# (ultimate opening, normalized) of the target softening laws.
#   linear:        σ = f_t (1 - w/w_c), w_c = 2 G_f/f_t               -> κ = 1/2, ŵ = 2
#   exponential:   σ = f_t exp(-f_t w/G_f)                             -> κ = 1,   ŵ = ∞
#   Cornelissen:   σ/f_t = (1+(c1 x)^3) e^{-c2 x} - x (1+c1^3) e^{-c2}, x = w/w_c,
#                  c1 = 3, c2 = 6.93, w_c = 5.1361 G_f/f_t             -> κ = 1.3546, ŵ = 5.1361
# The Cornelissen κ and ŵ are those quoted for Wu's calibration in Korec et al.
# (Korec, Jirásek, Wong & Martínez-Pañeda, arXiv:2306.01903, Eq. 49).
_SOFTENING_TARGETS = {
    "linear": dict(p=2.0, kappa=0.5, w_hat=2.0),
    "exponential": dict(p=2.5, kappa=1.0, w_hat=math.inf),
    "cornelissen": dict(p=2.0, kappa=1.3546, w_hat=5.1361),
}
CORNELISSEN_C1, CORNELISSEN_C2, CORNELISSEN_WC_HAT = 3.0, 6.93, 5.1361


def wu_softening_constants(softening: str, xi: float = 2.0) -> dict[str, float]:
    """(p, a2, a3) of Wu's rational degradation function for a target softening law.

    Wu (2017) Eqs. for the optimal geometric function α(d) = ξ d + (1-ξ) d², ξ = 2
    (c_α = 4∫₀¹√α = π):

        a2 = (1/ξ) [ (4π ξ² / c_α) κ ]^{2/3} - (p + 1/2)
        a3 = 0                                                  if p > 2
        a3 = (1/a2) [ (1/ξ) (c_α ŵ / (2π))² - (1 + a2) ]        if p = 2

    with κ = -k0 G_f/f_t² and ŵ = w_c f_t/G_f of the target law.  Convention:
    Q(d) = a1 d + a1 a2 d² + a1 a2 a3 d³ (Wu 2017; Wu & Nguyen 2018).  Published
    values reproduced (tests): linear p=2, a2=-1/2, a3=0; exponential p=5/2,
    a2 = 2^{5/3} - 3 = 0.1748, a3=0; Cornelissen p=2, a2=1.3868, a3=0.6567.
    NOTE: some later papers write Q = a1 d (1 + a2 d + a3 d²), i.e. their a3 is
    our a2*a3 (0.9107 for Cornelissen); the convention-free polynomial
    coefficients are returned by :func:`wu_pfczm_coefficients` as Q_coefficients.
    """
    if softening not in _SOFTENING_TARGETS:
        raise ValueError(f"softening must be one of {sorted(_SOFTENING_TARGETS)}, got {softening!r}")
    t = _SOFTENING_TARGETS[softening]
    c_alpha = math.pi if xi == 2.0 else _c_alpha_integral(xi)
    p, kappa, w_hat = t["p"], t["kappa"], t["w_hat"]
    a2 = (1.0 / xi) * ((4.0 * math.pi * xi**2 / c_alpha) * kappa) ** (2.0 / 3.0) - (p + 0.5)
    if p > 2.0 or not math.isfinite(w_hat):
        a3 = 0.0
    else:
        a3 = ((1.0 / xi) * (c_alpha * w_hat / (2.0 * math.pi)) ** 2 - (1.0 + a2)) / a2 if a2 != 0 else 0.0
    a2 = 0.0 if abs(a2) < 1e-12 else a2
    a3 = 0.0 if abs(a3) < 1e-12 else a3
    return dict(p=p, a2=a2, a3=a3, xi=xi, c_alpha=c_alpha, kappa=kappa, w_hat=w_hat)


def _c_alpha_integral(xi: float, n: int = 20001) -> float:
    d = np.linspace(0.0, 1.0, n)
    return float(4.0 * np.trapezoid(np.sqrt(xi * d + (1 - xi) * d**2), d))


def _to_si(x: Any, unit: Unit) -> float:
    if isinstance(x, Quantity):
        return x.magnitude(unit)
    return float(x)


def wu_pfczm_coefficients(
    E: Union[float, Quantity],
    G_f: Union[float, Quantity],
    f_t: Union[float, Quantity],
    b: Union[float, Quantity],
    softening: str = "exponential",
) -> dict[str, Any]:
    """Coefficients of Wu's PF-CZM for given E0, G_f, f_t and length scale b.

    Model (J.-Y. Wu, JMPS 103 (2017) 72-99; Wu & Nguyen, JMPS 119 (2018) 20-42):

        α(d) = 2d - d²  (ξ = 2),        c_α = π
        ω(d) = (1-d)^p / [ (1-d)^p + a1 d (1 + a2 d + a2 a3 d²) ]
        a1   = 4 E0 G_f / (π b f_t²) = 4 l_ch / (π b),   l_ch = E0 G_f / f_t²

    Plain floats must be in one consistent unit system (e.g. SI: Pa, J/m², m);
    Quantities are converted to SI.  Returns a dict with p, a1, a2, a3, xi,
    c_alpha, l_ch, alpha (callable), omega (jittable callable), expression
    strings, Q_coefficients = (a1, a1 a2, a1 a2 a3) (convention free),
    b_max = 8 l_ch/(3π) and b_admissible.  The b ≤ 8 l_ch/(3π) (i.e. a1 ≥ 3/2)
    bound is quoted from Wu's work via Korec et al. (arXiv:2306.01903); it was
    not re-derived here.
    """
    E_ = _to_si(E, Unit.PA)
    G_ = _to_si(G_f, Unit.J_PER_M2)
    f_ = _to_si(f_t, Unit.PA)
    b_ = _to_si(b, Unit.METER)
    if min(E_, G_, f_, b_) <= 0:
        raise ValueError("E, G_f, f_t and b must be positive")
    c = wu_softening_constants(softening)
    l_ch = E_ * G_ / f_**2
    a1 = 4.0 * l_ch / (math.pi * b_)
    p, a2, a3 = c["p"], c["a2"], c["a3"]

    def alpha(d):
        return 2.0 * d - d * d

    def omega(d):
        import jax.numpy as jnp

        d = jnp.asarray(d)
        num = (1.0 - d) ** p
        return num / (num + a1 * d * (1.0 + a2 * d + a2 * a3 * d * d))

    b_max = 8.0 * l_ch / (3.0 * math.pi)
    return dict(
        softening=softening,
        p=p,
        a1=a1,
        a2=a2,
        a3=a3,
        xi=2.0,
        c_alpha=math.pi,
        l_ch=l_ch,
        b=b_,
        alpha=alpha,
        alpha_expr="alpha(d) = 2*d - d**2",
        omega=omega,
        omega_expr="omega(d) = (1-d)**p / ((1-d)**p + a1*d*(1 + a2*d + a2*a3*d**2))",
        Q_coefficients=(a1, a1 * a2, a1 * a2 * a3),
        b_max=b_max,
        b_admissible=b_ <= b_max,
        convention="Q(d) = a1 d + a1 a2 d^2 + a1 a2 a3 d^3 (Wu 2017; Wu & Nguyen 2018)",
        citations=[c.text + (f", doi:{c.doi}" if c.doi else "") for c in CITATIONS_WU],
    )


def softening_shape(softening: str, w_over_scale: np.ndarray) -> np.ndarray:
    """Normalized target softening law σ/f_t as a function of w f_t / G_f (numpy)."""
    w = np.asarray(w_over_scale, float)
    if softening == "linear":
        return np.clip(1.0 - w / 2.0, 0.0, None)
    if softening == "exponential":
        return np.exp(-w)
    if softening == "cornelissen":
        x = w / CORNELISSEN_WC_HAT
        c1, c2 = CORNELISSEN_C1, CORNELISSEN_C2
        y = (1.0 + (c1 * x) ** 3) * np.exp(-c2 * x) - x * (1.0 + c1**3) * np.exp(-c2)
        return np.where(x < 1.0, y, 0.0)
    raise ValueError(softening)


# =============================================================================
# Pydantic models
# =============================================================================
class BundleMaterial(StrictModel):
    candidate_id: str
    candidate_name: str
    composition: Composition
    phase_name: str
    phase: Optional[Phase] = None
    density: Optional[Quantity] = None  # g/cm^3
    density_record_id: Optional[str] = None

    @model_validator(mode="after")
    def _units(self) -> "BundleMaterial":
        _require_compatible(self.density, Unit.G_PER_CM3, "density")
        return self


class IsotropicEquivalent(StrictModel):
    """Isotropic (polycrystal-average) moduli; the PF-CZM default when no texture is known."""

    method: str = "Voigt-Reuss-Hill average of C_voigt (Hill 1952)"
    bulk_modulus: Quantity  # GPa
    shear_modulus: Quantity  # GPa
    youngs_modulus: Quantity  # GPa
    poisson_ratio: Quantity  # dimensionless
    voigt_reuss_bounds: dict[str, Quantity] = Field(default_factory=dict)  # B_V, B_R, G_V, G_R
    source: Literal["records", "computed_from_C"] = "computed_from_C"
    source_record_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _units(self) -> "IsotropicEquivalent":
        for k in ("bulk_modulus", "shear_modulus", "youngs_modulus"):
            _require_compatible(getattr(self, k), Unit.GPA, k)
        _require_compatible(self.poisson_ratio, Unit.DIMENSIONLESS, "poisson_ratio")
        return self


class DirectionalModulus(StrictModel):
    direction: tuple[float, float, float]  # crystal Cartesian (= [uvw] for cubic)
    label: str = ""  # e.g. "[111]"
    youngs_modulus: Quantity  # GPa
    source: Literal["computed_from_S", "record"] = "computed_from_S"
    record_id: Optional[str] = None


class ElasticBlock(StrictModel):
    frame: Literal["crystal"] = "crystal"
    rotation_crystal_to_lab: list[list[float]] = Field(
        default_factory=lambda: np.eye(3).tolist()
    )
    C_voigt: Quantity  # 6x6, GPa
    S_voigt: Quantity  # 6x6, 1/GPa, engineering shear strains
    voigt_convention: str = (
        "Voigt order xx,yy,zz,yz,xz,xy; sigma_m = C_mn eps_n with engineering shear strains "
        "(eps_4 = 2 eps_yz, ...); S = inv(C) (compliance with engineering shear strains); "
        "C_ijkl = C_mn; S_ijkl = S_mn * (1 | 1/2 | 1/4)."
    )
    elastic_type: str = "relaxed-ion (internal coordinates relaxed), 0 K static"
    isotropic_equivalent: IsotropicEquivalent
    directional_youngs: list[DirectionalModulus] = Field(default_factory=list)
    uncertainty: list[Uncertainty] = Field(default_factory=list)
    validation_status: ValidationStatus
    validation_notes: str = ""
    checks: list[CheckResult] = Field(default_factory=list)
    potential_id: Optional[str] = None
    conditions: Conditions = Field(default_factory=Conditions)
    source_record_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ElasticBlock":
        _require_compatible(self.C_voigt, Unit.GPA, "C_voigt")
        _require_compatible(self.S_voigt, Unit.PER_GPA, "S_voigt")
        C = self.C_voigt.as_array(Unit.GPA)
        S = self.S_voigt.as_array(Unit.PER_GPA)
        if C.shape != (6, 6) or S.shape != (6, 6):
            raise ValueError(f"C_voigt and S_voigt must be 6x6 (got {C.shape}, {S.shape})")
        _finite(C, "C_voigt")
        _finite(S, "S_voigt")
        scale = max(np.abs(C).max(), 1e-30)
        if np.abs(C - C.T).max() > 1e-8 * scale:
            raise ValueError("C_voigt must be symmetric (symmetrize before export and record the asymmetry)")
        if np.abs(S @ C - np.eye(6)).max() > 1e-6:
            raise ValueError("S_voigt is not the inverse of C_voigt")
        R = np.asarray(self.rotation_crystal_to_lab, float)
        if R.shape != (3, 3) or np.abs(R @ R.T - np.eye(3)).max() > 1e-8 or np.linalg.det(R) < 0:
            raise ValueError("rotation_crystal_to_lab must be a proper 3x3 rotation matrix")
        return self

    def to_tensor(self, unit: Unit = Unit.GPA) -> np.ndarray:
        """Stiffness C_ijkl (3x3x3x3) in the crystal frame, in `unit`."""
        return stiffness_voigt_to_tensor(self.C_voigt.as_array(unit))

    def to_lab_tensor(self, unit: Unit = Unit.GPA) -> np.ndarray:
        """Stiffness rotated into the lab frame with `rotation_crystal_to_lab`."""
        return rotate_tensor4(self.to_tensor(unit), np.asarray(self.rotation_crystal_to_lab))


class SofteningBranch(StrictModel):
    """Post-peak part of the sampled curve: what a cohesive softening law must represent."""

    opening_after_peak: Quantity  # angstrom, δ - δ_peak (starts at 0)
    traction: Quantity  # GPa
    G_post_peak: Quantity  # J/m^2, ∫_{δ_peak}^{δ_max} T dδ
    G_pre_peak: Quantity  # J/m^2, ∫_0^{δ_peak} T dδ  (elastic-like part of the interface response)
    notes: str = (
        "In PF-CZM the pre-peak (elastic) response is carried by the bulk; the cohesive law "
        "softens from f_t. Using G_c = W_sep (total) is Griffith-consistent; using G_post_peak "
        "instead removes the interface's pre-peak energy. Both are reported."
    )


class UBERConsistency(StrictModel):
    """Diagnostics of the UBER form Φ(δ) = W[1-(1+δ/l)e^{-δ/l}], T = W δ/l² e^{-δ/l}, T_max = W/(e l)."""

    W_fit: Quantity  # J/m^2
    l_fit: Quantity  # angstrom
    peak_traction_fit: Quantity  # GPa
    rel_diff_W: float
    rel_diff_peak_traction: float
    rel_diff_opening_at_peak: float
    rms_residual_energy: Quantity  # J/m^2
    tolerance: float
    within_tolerance: bool
    notes: str = "Informational; UBER is a fit model, not a validity criterion for the data."


class CohesiveLawBlock(StrictModel):
    id: str
    plane_miller: Optional[tuple[int, int, int]] = None
    plane_miller_bravais: Optional[tuple[int, int, int, int]] = None
    termination: str = ""
    interface: InterfaceSpec
    mode: Literal["normal", "tangential", "mixed"]
    protocol_mode: Literal["rigid", "relaxed", "relaxed_elastic_corrected"]
    opening: Quantity  # angstrom, strictly increasing
    traction: Quantity  # GPa
    energy_per_area: Quantity  # J/m^2, E(δ)/A - E(0)/A on the same grid
    peak_traction: Quantity  # GPa
    opening_at_peak: Quantity  # angstrom
    work_of_separation: Quantity  # J/m^2
    integrated_work: Quantity  # J/m^2, ∫T dδ as reported by the producer
    fits: list[CohesiveLawFit] = Field(default_factory=list)
    uber_consistency: Optional[UBERConsistency] = None  # normal/mixed laws only
    softening_branch: Optional[SofteningBranch] = None  # normal/mixed laws only (None for tangential γ-lines)
    assumptions: list[str] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    validation_status: ValidationStatus
    validation_notes: str = ""
    uncertainty: list[Uncertainty] = Field(default_factory=list)
    potential_id: Optional[str] = None
    conditions: Conditions = Field(default_factory=Conditions)
    source_record_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "CohesiveLawBlock":
        for name, like in (
            ("opening", Unit.ANGSTROM),
            ("traction", Unit.GPA),
            ("energy_per_area", Unit.J_PER_M2),
            ("peak_traction", Unit.GPA),
            ("opening_at_peak", Unit.ANGSTROM),
            ("work_of_separation", Unit.J_PER_M2),
            ("integrated_work", Unit.J_PER_M2),
        ):
            _require_compatible(getattr(self, name), like, name)
        x = np.asarray(_as_list(self.opening), float)
        y = np.asarray(_as_list(self.traction), float)
        e = np.asarray(_as_list(self.energy_per_area), float)
        if x.ndim != 1 or len(x) < 3:
            raise ValueError(f"cohesive law {self.id}: need at least 3 openings")
        if not (len(x) == len(y) == len(e)):
            raise ValueError(f"cohesive law {self.id}: opening/traction/energy lengths differ")
        for a, nm in ((x, "opening"), (y, "traction"), (e, "energy_per_area")):
            _finite(a, nm)
        if np.any(np.diff(x) <= 0):
            raise ValueError(f"cohesive law {self.id}: openings must be strictly increasing (monotonic)")
        return self

    # --------------------------------------------------------------- helpers
    def arrays(self, units: str = "native") -> dict[str, np.ndarray]:
        if units == "SI":
            return dict(
                opening=self.opening.as_array(Unit.METER),
                traction=self.traction.as_array(Unit.PA),
                energy_per_area=self.energy_per_area.as_array(Unit.J_PER_M2),
            )
        return dict(
            opening=self.opening.as_array(Unit.ANGSTROM),
            traction=self.traction.as_array(Unit.GPA),
            energy_per_area=self.energy_per_area.as_array(Unit.J_PER_M2),
        )

    def trapezoid_work(self) -> float:
        """∫ T dδ over the sampled range, J/m^2."""
        a = self.arrays()
        return float(np.trapezoid(a["traction"], a["opening"]) * GPA_ANGSTROM_TO_J_PER_M2)

    def uber_parameters(self) -> Optional[tuple[float, float]]:
        """(W [J/m^2], l [Å]) of the first parseable UBER fit, or None."""
        for f in self.fits:
            if f.model != "UBER":
                continue
            p = _parse_uber(f)
            if p is not None:
                return p
        return None


class WuCoefficients(StrictModel):
    """Wu (2017) PF-CZM coefficients for this parameter set; a1 = a1_times_b / b."""

    softening: Literal["linear", "exponential", "cornelissen"]
    p: float
    a2: float
    a3: float
    xi: float = 2.0
    c_alpha: float = math.pi
    a1_times_b: Quantity  # nm; a1 = 4 l_ch(E0) / (π b)
    E0_used: Quantity  # GPa, the Young's modulus entering l_ch in Wu's a1
    E0_choice: str = "isotropic Hill Young's modulus (Wu's E0 is the bulk Young's modulus)"
    b_max: Quantity  # nm, 8 l_ch(E0)/(3π): largest admissible regularization length (a1 >= 3/2)
    geometric_function: str = "alpha(d) = 2 d - d^2"
    degradation_function: str = "omega(d) = (1-d)^p / ((1-d)^p + a1 d (1 + a2 d + a2 a3 d^2))"
    convention: str = "Q(d) = a1 d + a1 a2 d^2 + a1 a2 a3 d^3 (Wu 2017; Wu & Nguyen 2018)"
    citations: list[Citation] = Field(default_factory=lambda: list(CITATIONS_WU))


class SofteningFit(StrictModel):
    softening: Literal["linear", "exponential", "cornelissen"]
    rms_normalized: float  # RMS of (T/f_t)_atomistic - shape(w f_t/G_post) over the post-peak samples


class PFCZMParameterSet(StrictModel):
    """DERIVED PF-CZM inputs for one cohesive law.  Every number here is a model estimate."""

    id: str
    cohesive_law_id: str
    origin: Literal["derived_model"] = "derived_model"
    scale: Literal["atomistic_ideal"] = "atomistic_ideal"
    interpretation: str = ATOMISTIC_SCALE_NOTE
    # --- elastic
    E_prime: Quantity  # GPa, the value to use
    E_prime_method: Literal["anisotropic_barnett_lothe_plane_strain", "isotropic_hill_plane_strain"]
    E_prime_isotropic_hill: Quantity  # GPa, E_H / (1 - ν_H²)
    E_prime_anisotropic: Optional[Quantity] = None  # GPa
    E_prime_anisotropic_range: Optional[tuple[Quantity, Quantity]] = None  # min/max over crack-front directions
    E_prime_anisotropic_note: str = ""
    # --- crack geometry E' refers to (crystal Cartesian unit vectors; lab = the law's orientation frame)
    crack_plane_normal_crystal: Optional[tuple[float, float, float]] = None
    crack_propagation_direction_crystal: Optional[tuple[float, float, float]] = None
    crack_front_direction_crystal: Optional[tuple[float, float, float]] = None
    crack_propagation_direction_uvw: Optional[tuple[int, int, int]] = None  # [uvw] w.r.t. the conventional cell
    crack_front_direction_uvw: Optional[tuple[int, int, int]] = None
    crack_frame_convention: str = (
        "lab frame of the cohesive law's orientation: plane normal = lab +z, crack propagation = lab +x "
        "(a1 of the oriented cell, the canonical shortest in-plane lattice vector), crack front = lab +y; "
        "E_prime_anisotropic is the Barnett-Lothe plane-strain modulus for this normal/propagation pair"
    )
    # --- strength and energy
    tensile_strength: Quantity  # GPa (= atomistic peak traction; ideal cleavage strength)
    fracture_energy: Quantity  # J/m^2 (= work of separation; ideal Griffith energy)
    G_post_peak: Quantity  # J/m^2 alternative (softening-branch area only)
    irwin_length: Quantity  # nm, l_ch = E' G_c / f_t^2
    irwin_length_E0: Quantity  # nm, E0 G_c / f_t^2 (the length entering Wu's a1)
    # --- toughness estimate (never a measured toughness)
    K_Ic_griffith_estimate: Quantity  # MPa*m^0.5, sqrt(E' G_Ic), G_Ic = K_Ic_griffith_energy
    K_Ic_origin: Literal["derived_model"] = "derived_model"
    K_Ic_is_measured_toughness: Literal[False] = False
    K_Ic_model: str = (
        "Griffith, plane strain, ideal brittle: K = sqrt(E' G_Ic) with G_Ic = 2γ_relaxed (relaxed work of "
        "separation, preferred) when available, else the law's W_sep; K_Ic_rigid_upper_bound = sqrt(E' W_sep,rigid) "
        "uses the unrelaxed (rigid) work of separation and is an upper bound"
    )
    K_Ic_griffith_energy: Optional[Quantity] = None  # J/m^2, the G_Ic entering K_Ic_griffith_estimate
    K_Ic_griffith_energy_source: Literal["relaxed_2gamma", "law_work_of_separation"] = "law_work_of_separation"
    G_IC_record_id: Optional[str] = None  # the runner's G_Ic_atomistic record on the same interface (if any)
    K_Ic_rigid_upper_bound: Optional[Quantity] = None  # MPa*m^0.5, sqrt(E' W_sep,rigid)
    W_sep_rigid: Optional[Quantity] = None  # J/m^2, rigid (unrelaxed) work of separation used for the bound
    K_Ic_record_id: Optional[str] = None
    # --- softening
    recommended_softening: Literal["linear", "exponential", "cornelissen"]
    softening_rationale: str = ""
    softening_fits: list[SofteningFit] = Field(default_factory=list)
    wu: WuCoefficients
    # --- bookkeeping
    derivation: Derivation
    validation_status: ValidationStatus
    input_statuses: dict[str, ValidationStatus] = Field(default_factory=dict)
    validation_notes: str = ""

    @model_validator(mode="after")
    def _check(self) -> "PFCZMParameterSet":
        for k in ("E_prime", "E_prime_isotropic_hill", "E_prime_anisotropic", "tensile_strength"):
            _require_compatible(getattr(self, k), Unit.GPA, k)
        for k in ("fracture_energy", "G_post_peak", "K_Ic_griffith_energy", "W_sep_rigid"):
            _require_compatible(getattr(self, k), Unit.J_PER_M2, k)
        if self.K_Ic_rigid_upper_bound is not None and self.K_Ic_rigid_upper_bound.unit != Unit.MPA_SQRT_M:
            raise ValueError("K_Ic_rigid_upper_bound must be in MPa*m^0.5")
        for k in ("irwin_length", "irwin_length_E0"):
            _require_compatible(getattr(self, k), Unit.NANOMETER, k)
        if self.K_Ic_griffith_estimate.unit != Unit.MPA_SQRT_M:
            raise ValueError("K_Ic_griffith_estimate must be in MPa*m^0.5")
        allowed = {ValidationStatus.MODEL_ESTIMATE, ValidationStatus.OUT_OF_DOMAIN, ValidationStatus.FAILED}
        if self.validation_status not in allowed:
            raise ValueError(
                "derived PF-CZM parameters can only be model_estimate, out_of_domain or failed "
                f"(got {self.validation_status.value}); they are never validated measurements"
            )
        return self


# ------------------------------------------------------------------ extensions
class _ExtBase(StrictModel):
    status: ExtensionStatus = "not_computed"
    notes: str = ""

    def _payload_len(self) -> int:  # overridden
        return 0

    @model_validator(mode="after")
    def _status_matches_content(self):
        n = self._payload_len()
        if self.status == "not_computed" and n:
            raise ValueError(f"{type(self).__name__}: status 'not_computed' but content present")
        if self.status != "not_computed" and not n:
            raise ValueError(f"{type(self).__name__}: status {self.status!r} but no content")
        return self


class TangentialExtension(_ExtBase):
    """Tangential (mode II/III) traction-separation laws, e.g. γ-lines at fixed opening."""

    laws: list[CohesiveLawBlock] = Field(default_factory=list)

    def _payload_len(self) -> int:
        return len(self.laws)


class MixedModeSurface(StrictModel):
    """Coupled cohesive potential Φ(δn, δt) sampled on a tensor-product grid.

    energy_per_area[i][j] = Φ(normal_opening[i], tangential_opening[j]); tractions
    follow as T_n = ∂Φ/∂δn and T_t = ∂Φ/∂δt (same sign convention as mode I).
    """

    id: str
    interface: InterfaceSpec
    slip_direction: Optional[tuple[int, int, int]] = None
    normal_opening: Quantity  # angstrom, 1-D strictly increasing
    tangential_opening: Quantity  # angstrom, 1-D strictly increasing
    energy_per_area: Quantity  # J/m^2, shape (n_normal, n_tangential)
    protocol_mode: Literal["rigid", "relaxed", "relaxed_elastic_corrected"] = "rigid"
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    source_record_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "MixedModeSurface":
        _require_compatible(self.normal_opening, Unit.ANGSTROM, "normal_opening")
        _require_compatible(self.tangential_opening, Unit.ANGSTROM, "tangential_opening")
        _require_compatible(self.energy_per_area, Unit.J_PER_M2, "energy_per_area")
        a = np.asarray(_as_list(self.normal_opening), float)
        b = np.asarray(_as_list(self.tangential_opening), float)
        E = np.asarray(self.energy_per_area.value, float)
        if np.any(np.diff(a) <= 0) or np.any(np.diff(b) <= 0):
            raise ValueError("mixed-mode grid axes must be strictly increasing")
        if E.shape != (len(a), len(b)):
            raise ValueError(f"energy_per_area shape {E.shape} != ({len(a)}, {len(b)})")
        return self


class MixedModeExtension(_ExtBase):
    surfaces: list[MixedModeSurface] = Field(default_factory=list)

    def _payload_len(self) -> int:
        return len(self.surfaces)


class DefectPropertySet(StrictModel):
    """Properties computed for a non-pristine DefectState (vacancies, pores, dislocations...)."""

    defect_state: DefectState
    elastic: Optional[ElasticBlock] = None
    cohesive_laws: list[CohesiveLawBlock] = Field(default_factory=list)
    property_record_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class DefectExtension(_ExtBase):
    sets: list[DefectPropertySet] = Field(default_factory=list)

    def _payload_len(self) -> int:
        return len(self.sets)


class ParameterOverride(StrictModel):
    """Replace one bundle quantity at a given condition (T or rate).

    `target` is a property name (PropertyName value, e.g. "peak_normal_traction")
    or a bundle path (e.g. "pfczm_parameters[<id>].tensile_strength").
    """

    target: str
    value: Quantity
    record_id: Optional[str] = None
    cohesive_law_id: Optional[str] = None


class TemperaturePoint(StrictModel):
    temperature: Quantity  # K
    overrides: list[ParameterOverride] = Field(default_factory=list)

    @model_validator(mode="after")
    def _u(self) -> "TemperaturePoint":
        _require_compatible(self.temperature, Unit.KELVIN, "temperature")
        return self


class TemperatureExtension(_ExtBase):
    points: list[TemperaturePoint] = Field(default_factory=list)

    def _payload_len(self) -> int:
        return len(self.points)


class StrainRatePoint(StrictModel):
    strain_rate: Quantity  # 1/s
    overrides: list[ParameterOverride] = Field(default_factory=list)

    @model_validator(mode="after")
    def _u(self) -> "StrainRatePoint":
        _require_compatible(self.strain_rate, Unit.PER_SECOND, "strain_rate")
        return self


class StrainRateExtension(_ExtBase):
    points: list[StrainRatePoint] = Field(default_factory=list)

    def _payload_len(self) -> int:
        return len(self.points)


class UncertaintySamples(StrictModel):
    """Samples of one bundle quantity (ensemble of potentials, seeds, bootstrap...)."""

    target: str  # property name or bundle path, as in ParameterOverride.target
    samples: Quantity  # 1-D list
    kind: UncertaintyKind
    distribution: Optional[str] = None  # e.g. "empirical", "normal", "weibull"
    distribution_parameters: dict[str, float] = Field(default_factory=dict)
    record_id: Optional[str] = None
    description: str = ""


class UncertaintyExtension(_ExtBase):
    samples: list[UncertaintySamples] = Field(default_factory=list)

    def _payload_len(self) -> int:
        return len(self.samples)


class InterfaceCohesiveSet(StrictModel):
    interface: InterfaceSpec
    cohesive_laws: list[CohesiveLawBlock] = Field(default_factory=list)
    interface_energy: Optional[Quantity] = None  # J/m^2
    property_record_ids: list[str] = Field(default_factory=list)


class InterfaceExtension(_ExtBase):
    interfaces: list[InterfaceCohesiveSet] = Field(default_factory=list)

    def _payload_len(self) -> int:
        return len(self.interfaces)


class MicrostructureExtension(_ExtBase):
    """status 'computed' here means 'specified' (the microstructure is an input description)."""

    microstructure: Optional[Microstructure] = None

    def _payload_len(self) -> int:
        return int(self.microstructure is not None)


class ExtensionBlocks(StrictModel):
    """Optional physics. A solver must check `.status` before using a block."""

    tangential: TangentialExtension = Field(default_factory=TangentialExtension)
    mixed_mode: MixedModeExtension = Field(default_factory=MixedModeExtension)
    defects: DefectExtension = Field(default_factory=DefectExtension)
    temperature: TemperatureExtension = Field(default_factory=TemperatureExtension)
    strain_rate: StrainRateExtension = Field(default_factory=StrainRateExtension)
    uncertainty: UncertaintyExtension = Field(default_factory=UncertaintyExtension)
    grain_boundaries: InterfaceExtension = Field(default_factory=InterfaceExtension)
    interfaces: InterfaceExtension = Field(default_factory=InterfaceExtension)
    microstructure: MicrostructureExtension = Field(default_factory=MicrostructureExtension)

    def summary(self) -> dict[str, str]:
        return {k: getattr(self, k).status for k in type(self).model_fields}


# ------------------------------------------------------------------ integrity
class ArrayManifestEntry(StrictModel):
    key: str  # sidecar key
    json_path: str  # where the same numbers live inline
    shape: list[int]
    unit: Unit
    sha256: str  # array_sha256 of the float64 values


class IntegrityBlock(StrictModel):
    algorithm: Literal["sha256"] = "sha256"
    content_sha256: Optional[str] = None  # canonical JSON of the bundle without `integrity`
    arrays: list[ArrayManifestEntry] = Field(default_factory=list)
    sidecar_file: Optional[str] = None  # file name, relative to the JSON file
    sidecar_sha256: Optional[str] = None
    notes: str = (
        "content_sha256 = sha256(canonical JSON (sorted keys, compact separators) of the bundle "
        "with the integrity block removed); array sha256 = sha256(b'shape=<dims>;' + float64 LE bytes)."
    )


class PFCZMBundle(StrictModel):
    format: Literal["fracture-md.pfczm-bundle"] = BUNDLE_FORMAT
    version: str = BUNDLE_VERSION
    bundle_id: str
    created_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    material: BundleMaterial
    potentials: list[PotentialCard] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
    elastic: Optional[ElasticBlock] = None
    cohesive_laws: list[CohesiveLawBlock] = Field(default_factory=list)
    pfczm_parameters: list[PFCZMParameterSet] = Field(default_factory=list)
    extensions: ExtensionBlocks = Field(default_factory=ExtensionBlocks)
    property_records: list[PropertyRecord] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    integrity: IntegrityBlock = Field(default_factory=IntegrityBlock)

    @field_validator("version")
    @classmethod
    def _semver(cls, v: str) -> str:
        parse_version(v)
        return v

    @model_validator(mode="after")
    def _refs(self) -> "PFCZMBundle":
        ids = [law.id for law in self.all_cohesive_laws()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate cohesive law ids: {ids}")
        normal = {law.id for law in self.cohesive_laws}
        for ps in self.pfczm_parameters:
            if ps.cohesive_law_id not in normal:
                raise ValueError(f"parameter set {ps.id} refers to unknown cohesive law {ps.cohesive_law_id}")
        for r in self.property_records:
            if r.name == PropertyName.K_IC_GRIFFITH_ESTIMATE and r.origin != PropertyOrigin.DERIVED_MODEL:
                raise ValueError("K_Ic_griffith_estimate records must be derived model estimates")
        by_id = {r.id: r for r in self.property_records}
        for ps in self.pfczm_parameters:
            if ps.K_Ic_record_id is None:
                continue
            r = by_id.get(ps.K_Ic_record_id)
            if r is None or r.name != PropertyName.K_IC_GRIFFITH_ESTIMATE or r.origin != PropertyOrigin.DERIVED_MODEL:
                raise ValueError(f"parameter set {ps.id}: K_Ic_record_id {ps.K_Ic_record_id!r} must resolve to a "
                                 "K_Ic_griffith_estimate property record with origin derived_model")
        return self

    # --------------------------------------------------------------- access
    def all_cohesive_laws(self) -> list[CohesiveLawBlock]:
        out = list(self.cohesive_laws) + list(self.extensions.tangential.laws)
        for blk in (self.extensions.grain_boundaries, self.extensions.interfaces):
            for s in blk.interfaces:
                out.extend(s.cohesive_laws)
        for s in self.extensions.defects.sets:
            out.extend(s.cohesive_laws)
        return out

    def law(self, law_id: str) -> CohesiveLawBlock:
        for law in self.all_cohesive_laws():
            if law.id == law_id:
                return law
        raise KeyError(law_id)

    def parameters_for(self, law_id: str) -> PFCZMParameterSet:
        for ps in self.pfczm_parameters:
            if ps.cohesive_law_id == law_id:
                return ps
        raise KeyError(law_id)

    def content_dict(self) -> dict:
        d = self.model_dump(mode="json")
        d.pop("integrity", None)
        return d

    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.content_dict()).encode()).hexdigest()

    def to_jax(self, units: str = "SI") -> dict:
        return bundle_to_jax(self, units=units)


# =============================================================================
# Builders
# =============================================================================
def _record_matrix(rec: PropertyRecord, unit: Unit) -> np.ndarray:
    a = rec.value.as_array(unit)
    if a.shape == (3, 3, 3, 3):
        a = tensor_to_stiffness_voigt(a)
    return a


def _pick_primary(records: list[PropertyRecord]) -> tuple[PropertyRecord, list[PropertyRecord]]:
    """Choose the strongest-evidence record; prefer autodiff over finite differences on ties."""

    def key(r: PropertyRecord):
        name = (r.method.name + " " + r.method.description).lower()
        return (_STATUS_RANK[r.validation.status], 1 if "autodiff" in name else 0)

    srt = sorted(records, key=key, reverse=True)
    return srt[0], srt[1:]


def _rotation_from_record(rec: PropertyRecord) -> Optional[np.ndarray]:
    if rec.orientation is not None and rec.orientation.rotation_crystal_to_lab is not None:
        return np.asarray(rec.orientation.rotation_crystal_to_lab, float)
    return None


def elastic_block_from_records(
    records: Sequence[PropertyRecord],
    extra_directions: Sequence[Sequence[float]] = (),
) -> tuple[Optional[ElasticBlock], list[str]]:
    """Build the ElasticBlock from bulk-runner records.

    Uses the `elastic_stiffness_voigt` record with the strongest validation
    status (autodiff preferred on ties); other stiffness records provide an
    inter-method spread (Uncertainty, max |ΔC_ij|).  C is symmetrized (the
    asymmetry is recorded as a check).  S is taken from an
    `elastic_compliance_voigt` record if present (cross-checked against
    inv(C)), else computed.  Hill moduli are taken from records if present and
    cross-checked against a VRH average of C.  Returns (block or None, notes).
    """
    notes: list[str] = []
    pristine = [r for r in records if r.conditions.defect_state.kind == "pristine"]
    stiff = [r for r in pristine if r.name == PropertyName.ELASTIC_STIFFNESS_VOIGT]
    if not stiff:  # fall back to a 3x3x3x3 tensor record
        stiff = [r for r in pristine if r.name == PropertyName.ELASTIC_STIFFNESS_TENSOR]
    if not stiff:
        notes.append("no elastic_stiffness_voigt record: elastic block and PF-CZM parameter sets not built")
        return None, notes
    usable = [r for r in stiff if r.validation.status != ValidationStatus.FAILED] or stiff
    primary, others = _pick_primary(usable)
    C = _record_matrix(primary, Unit.GPA)
    if C.shape != (6, 6):
        raise ValueError(f"stiffness record {primary.id} has shape {C.shape}")
    if primary.frame == "lab":
        R = _rotation_from_record(primary)
        if R is None:
            raise ValueError(f"stiffness record {primary.id} is in the lab frame without a rotation")
        C = tensor_to_stiffness_voigt(rotate_tensor4(stiffness_voigt_to_tensor(C), R.T))
        notes.append(f"rotated {primary.id} from lab to crystal frame")
    checks: list[CheckResult] = []
    asym = float(np.abs(C - C.T).max())
    checks.append(
        CheckResult(name="export_C_symmetry", passed=asym <= 0.5, metric=asym, tolerance=0.5, unit=Unit.GPA,
                    details="max |C_ij - C_ji| before symmetrization")
    )
    C = 0.5 * (C + C.T)
    eig = np.linalg.eigvalsh(C)
    checks.append(
        CheckResult(name="export_born_stability", passed=bool(eig.min() > 0), metric=float(eig.min()), tolerance=0.0,
                    unit=Unit.GPA, details="min eigenvalue of C_voigt (> 0 required for a stable crystal)")
    )
    S = np.linalg.inv(C)
    source_ids = [primary.id]
    comp = [r for r in records if r.name == PropertyName.ELASTIC_COMPLIANCE_VOIGT and r.conditions.defect_state.kind == "pristine"]
    if comp:
        Sr = comp[0].value.as_array(Unit.PER_GPA)
        d = float(np.abs(Sr - S).max() / max(np.abs(S).max(), 1e-30))
        checks.append(CheckResult(name="export_S_record_vs_inv_C", passed=d < 1e-3, metric=d, tolerance=1e-3,
                                  details=f"record {comp[0].id} vs inv(C) (relative max abs diff); inv(C) is exported"))
        source_ids.append(comp[0].id)

    uncertainty = list(primary.uncertainty)
    # other independent stiffness records (e.g. finite differences) -> inter-method spread
    others = [o for o in others if _record_matrix(o, Unit.GPA).shape == (6, 6)
              and float(np.abs(_record_matrix(o, Unit.GPA) - C).max()) > 1e-12]
    if others:
        diffs = [float(np.abs(_record_matrix(o, Unit.GPA) - C).max()) for o in others]
        if diffs:
            names = " ".join(o.method.name.lower() for o in others) + primary.method.name.lower()
            kind = (UncertaintyKind.AUTODIFF_VS_FINITE_DIFFERENCE
                    if ("finite" in names and "autodiff" in names) else UncertaintyKind.NUMERICAL_CONVERGENCE)
            uncertainty.append(Uncertainty(kind=kind, value=max(diffs), unit=Unit.GPA, interval="max_abs_diff",
                                           description="max |ΔC_ij| between the exported stiffness and the other "
                                           f"stiffness records ({', '.join(o.id for o in others)})"))
            source_ids.extend(o.id for o in others)

    m = vrh_moduli(C, S)
    iso_src: Literal["records", "computed_from_C"] = "computed_from_C"
    iso_ids: list[str] = []
    vals = dict(B=m["BH"], G=m["GH"], E=m["EH"], nu=m["nuH"])
    rec_map = {
        "B": PropertyName.BULK_MODULUS_HILL,
        "G": PropertyName.SHEAR_MODULUS_HILL,
        "E": PropertyName.YOUNGS_MODULUS_HILL,
        "nu": PropertyName.POISSON_RATIO_HILL,
    }
    found = {}
    for k, pn in rec_map.items():
        rr = [r for r in records if r.name == pn and r.conditions.defect_state.kind == "pristine"]
        if rr:
            unit = Unit.DIMENSIONLESS if k == "nu" else Unit.GPA
            found[k] = (rr[0], rr[0].value.magnitude(unit))
    if len(found) == 4:
        iso_src = "records"
        for k, (r, v) in found.items():
            rel = abs(v - vals[k]) / max(abs(vals[k]), 1e-30)
            checks.append(CheckResult(name=f"export_hill_{k}_record_vs_C", passed=rel < 1e-3, metric=rel, tolerance=1e-3,
                                      details=f"record {r.id} vs VRH of exported C (relative)"))
            iso_ids.append(r.id)
            vals[k] = v
    elif found:
        notes.append("incomplete Hill-modulus records; isotropic equivalent computed from C")

    iso = IsotropicEquivalent(
        bulk_modulus=q(vals["B"], Unit.GPA),
        shear_modulus=q(vals["G"], Unit.GPA),
        youngs_modulus=q(vals["E"], Unit.GPA),
        poisson_ratio=q(vals["nu"], Unit.DIMENSIONLESS),
        voigt_reuss_bounds=dict(B_V=q(m["BV"], Unit.GPA), B_R=q(m["BR"], Unit.GPA),
                                G_V=q(m["GV"], Unit.GPA), G_R=q(m["GR"], Unit.GPA)),
        source=iso_src,
        source_record_ids=iso_ids,
    )

    dirs: list[tuple[str, np.ndarray]] = [("[100]", np.array([1.0, 0, 0])), ("[110]", np.array([1.0, 1, 0])),
                                          ("[111]", np.array([1.0, 1, 1]))]
    for d in extra_directions:
        dd = np.asarray(d, float)
        dirs.append(("[" + " ".join(f"{x:g}" for x in dd) + "]", dd))
    directional = []
    seen = set()
    for label, d in dirs:
        u = tuple(np.round(d / np.linalg.norm(d), 10))
        if u in seen:
            continue
        seen.add(u)
        directional.append(DirectionalModulus(direction=tuple(float(x) for x in d), label=label,
                                              youngs_modulus=q(directional_youngs_from_compliance(S, d), Unit.GPA)))
    for r in records:
        if r.name != PropertyName.YOUNGS_MODULUS_DIRECTIONAL or isinstance(r.value.value, list):
            continue
        if r.orientation is None or r.orientation.direction is None:
            continue
        d = np.asarray(r.orientation.direction, float)
        Ec = directional_youngs_from_compliance(S, d)
        Er = r.value.magnitude(Unit.GPA)
        rel = abs(Er - Ec) / Ec
        checks.append(CheckResult(name=f"export_directional_E_{r.orientation.direction}", passed=rel < 1e-3,
                                  metric=rel, tolerance=1e-3, details=f"record {r.id} vs 1/(nnnn:S)"))
        directional.append(DirectionalModulus(direction=tuple(float(x) for x in d), label=str(r.orientation.direction),
                                              youngs_modulus=q(Er, Unit.GPA), source="record", record_id=r.id))
        source_ids.append(r.id)

    status = primary.validation.status
    if any(not c.passed for c in checks):
        status = ValidationStatus.FAILED
    block = ElasticBlock(
        C_voigt=q(C, Unit.GPA),
        S_voigt=q(S, Unit.PER_GPA),
        isotropic_equivalent=iso,
        directional_youngs=directional,
        uncertainty=uncertainty,
        validation_status=status,
        validation_notes=primary.validation.notes,
        checks=checks,
        potential_id=primary.potential_id,
        conditions=primary.conditions,
        source_record_ids=source_ids,
        elastic_type=primary.method.name or "relaxed-ion, 0 K static",
    )
    return block, notes


# ------------------------------------------------------------------ cohesive
_UBER_W_KEYS = ("w", "work_of_separation", "w_sep", "g", "g_c", "e_b", "binding_energy", "energy", "phi_inf", "gamma2")
_UBER_L_KEYS = ("l", "lambda", "length_scale", "l_uber", "l_c", "critical_length", "scaling_length")


def _parse_uber(fit: CohesiveLawFit) -> Optional[tuple[float, float]]:
    P = {k.lower(): v for k, v in fit.parameters.items()}
    W = next((P[k] for k in _UBER_W_KEYS if k in P), None)
    L = next((P[k] for k in _UBER_L_KEYS if k in P), None)
    if W is None or L is None:
        return None
    try:
        return W.magnitude(Unit.J_PER_M2), L.magnitude(Unit.ANGSTROM)
    except (UnitError, TypeError):
        return None


def _exp_like(x):
    """np.exp for numpy/Python inputs, jnp.exp otherwise (keeps the UBER helpers jittable)."""
    if isinstance(x, (np.ndarray, float, int, np.floating)):
        return np.exp
    import jax.numpy as jnp

    return jnp.exp


def uber_energy(delta, W, l):
    """UBER: Φ(δ) = W [1 - (1 + δ/l) exp(-δ/l)]  (works for numpy and jax arrays)."""
    x = delta / l
    return W * (1.0 - (1.0 + x) * _exp_like(delta)(-x))


def uber_traction(delta, W, l):
    """UBER traction T = dΦ/dδ = W δ / l² exp(-δ/l); peak W/(e l) at δ = l."""
    return W * delta / l**2 * _exp_like(delta)(-delta / l)


def fit_uber(opening_A: np.ndarray, energy_J_m2: np.ndarray) -> tuple[float, float, float]:
    """Least-squares UBER fit of Φ(δ) (J/m^2) -> (W J/m^2, l Å, rms J/m^2)."""
    from scipy.optimize import least_squares

    x = np.asarray(opening_A, float)
    e = np.asarray(energy_J_m2, float) - float(energy_J_m2[0])
    W0 = max(float(e[-1]), 1e-6)
    # initial l from the energy inflection ~ opening where Φ = (1 - 2/e) W
    target = (1.0 - 2.0 / math.e) * W0
    l0 = float(np.interp(target, np.maximum.accumulate(e), x)) if e[-1] > 0 else 0.3
    l0 = max(l0, 1e-3)

    def res(p):
        return uber_energy(x, p[0], p[1]) - e

    sol = least_squares(res, x0=[W0, l0], bounds=([1e-12, 1e-6], [np.inf, np.inf]), x_scale=[W0, l0])
    rms = float(np.sqrt(np.mean(sol.fun**2)))
    return float(sol.x[0]), float(sol.x[1]), rms


def _same_interface(a: Optional[InterfaceSpec], b: Optional[InterfaceSpec]) -> bool:
    if a is None or b is None:
        return False
    if a.model_dump(mode="json") == b.model_dump(mode="json"):
        return True
    ta = a.termination.label if a.termination else None
    tb = b.termination.label if b.termination else None
    return (
        a.kind == b.kind
        and a.orientation_a.plane_miller == b.orientation_a.plane_miller
        and a.orientation_a.plane_miller is not None
        and ta == tb
    )


_LAW_RECORD_NAMES = {
    PropertyName.WORK_OF_SEPARATION,
    PropertyName.PEAK_NORMAL_TRACTION,
    PropertyName.OPENING_AT_PEAK_TRACTION,
    PropertyName.PEAK_TANGENTIAL_TRACTION,
    PropertyName.TRACTION_SEPARATION_NORMAL,
    PropertyName.TRACTION_SEPARATION_TANGENTIAL,
    PropertyName.UBER_LENGTH_SCALE,
    PropertyName.SURFACE_ENERGY,
    PropertyName.SURFACE_ENERGY_AVERAGE_PAIR,
    PropertyName.G_IC_ATOMISTIC,
    PropertyName.UNSTABLE_STACKING_FAULT_ENERGY,
}


_PROTOCOL_MODE_TAG = re.compile(r"\[protocol_mode=([a-z_]+)\]")
_GRIFFITH_SOURCE_TAG = re.compile(r"\[griffith_source=([a-z_]+)\]")


def protocol_mode_tag(mode: str) -> str:
    """Machine-readable token the runners put in ``MethodSpec.name`` (e.g. ``[protocol_mode=rigid]``)."""
    return f"[protocol_mode={mode}]"


def _protocol_mode_of(r: PropertyRecord) -> Optional[str]:
    """Protocol mode of a record from the structured ``[protocol_mode=...]`` token in ``method.name``.

    Modes: rigid, relaxed, relaxed_elastic_corrected (the TractionSeparationLaw modes) and
    relaxed_surfaces (energy-minimised W_sep, not a sampled law).  Records without the token
    (derived records such as G_Ic_atomistic, or legacy records) return None and are attached to
    every law on the same interface.  Legacy fallback: the method NAME only (never assumptions or
    notes, which mention e.g. "displaced rigidly" for relaxed grip-controlled runs).
    """
    m = _PROTOCOL_MODE_TAG.search(r.method.name)
    if m:
        return m.group(1)
    t = r.method.name.lower()
    if "corrected" in t:
        return "relaxed_elastic_corrected"
    if "unrelaxed" in t or t.startswith("rigid"):
        return "rigid"
    if "displacement-controlled" in t:
        return "relaxed"
    return None


def _records_for_law(law: TractionSeparationLaw, records: Sequence[PropertyRecord]) -> list[PropertyRecord]:
    """Records on the same interface whose method text names the same protocol mode (or none)."""
    cands = [r for r in records if r.name in _LAW_RECORD_NAMES and _same_interface(r.interface, law.interface)]
    return [r for r in cands if _protocol_mode_of(r) in (None, law.protocol_mode)]


def _law_id(law: TractionSeparationLaw) -> str:
    o = law.interface.orientation_a
    plane = "".join(str(i) for i in o.plane_miller) if o.plane_miller else (
        "".join(str(i) for i in o.plane_miller_bravais) if o.plane_miller_bravais else "plane")
    term = law.interface.termination.label if law.interface.termination else "default"
    h = hashlib.sha256(canonical_json(law.interface.model_dump(mode="json")).encode()).hexdigest()[:8]
    return _slug(f"{law.interface.kind.value}-{plane}-{term}-{law.mode}-{law.protocol_mode}-{h}")


def cohesive_law_from_tsl(
    law: TractionSeparationLaw,
    *,
    records: Sequence[PropertyRecord] = (),
    cards: Sequence[PotentialCard] = (),
    law_id: Optional[str] = None,
    consistency_rtol: float = 0.05,
    uber_rtol: float = 0.10,
    periodicity_atol: float = 1e-4,
) -> CohesiveLawBlock:
    """Convert a cleavage-runner TractionSeparationLaw into a CohesiveLawBlock.

    Normal/mixed laws get: an UBER fit (unless one is supplied), UBER-consistency
    diagnostics, the post-peak softening branch and an export-side integration check
    (∫T dδ over the samples vs W_sep, relative tolerance `consistency_rtol`, guarded by
    max(|W|, eps)).  Tangential laws (γ-lines over one lattice period, W ≈ 0) get no
    UBER fit and no softening branch; instead a periodicity check |Φ(end) − Φ(0)| <=
    `periodicity_atol` (J/m^2) and an absolute integral check |∫T dδ − (Φ(end) − Φ(0))| <=
    max(`consistency_rtol`·max|Φ|, `periodicity_atol`).  Every law gets a validation status.  The status is the weakest of (a) the matching
    PropertyRecords' statuses and (b) the records.decide_status rule applied to
    the law's own checks with property class bond_breaking (normal/mixed) or
    stacking_faults (tangential), so a known-deficient potential yields
    OUT_OF_DOMAIN exactly as for the records.
    """
    x = law.curve.x.as_array(Unit.ANGSTROM)
    y = law.curve.y.as_array(Unit.GPA)
    if isinstance(law.energy_per_area.value, list):
        e = law.energy_per_area.as_array(Unit.J_PER_M2)
    elif "energy_per_area" in law.curve.extra:
        e = law.curve.extra["energy_per_area"].as_array(Unit.J_PER_M2)
    else:
        from scipy.integrate import cumulative_trapezoid

        e = cumulative_trapezoid(y, x, initial=0.0) * GPA_ANGSTROM_TO_J_PER_M2
    matched = _records_for_law(law, records)
    pot_ids = {r.potential_id for r in matched if r.potential_id}
    card = None
    if len(pot_ids) == 1:
        pid = next(iter(pot_ids))
        card = next((c for c in cards if c.id == pid), None)
    elif len(cards) == 1:
        card = cards[0]
    W = law.work_of_separation.magnitude(Unit.J_PER_M2)
    trap = float(np.trapezoid(y, x) * GPA_ANGSTROM_TO_J_PER_M2)
    checks = list(law.checks)
    fits = list(law.fits)
    uc, sb = None, None
    if law.mode == "tangential":
        # γ-line over one lattice period: Φ(end) ≈ Φ(0) (periodicity) and W ≈ 0, so relative checks
        # against W are meaningless; no UBER fit and no softening branch (not a separation law).
        amp = float(np.max(np.abs(e - e[0])))
        dphi_end = float(e[-1] - e[0])
        checks.append(CheckResult(
            name="export_gamma_line_periodicity", passed=abs(dphi_end) <= periodicity_atol, metric=abs(dphi_end),
            tolerance=periodicity_atol, unit=Unit.J_PER_M2,
            details=f"|Φ(δ_end) − Φ(0)| over the sampled slip range {x[0]:.4g}-{x[-1]:.4g} Å (one lattice period); "
                    f"γ-line amplitude max|Φ| = {amp:.6g} J/m^2"))
        tol_i = max(consistency_rtol * amp, periodicity_atol)
        checks.append(CheckResult(
            name="export_integrated_traction_vs_energy_change", passed=abs(trap - dphi_end) <= tol_i,
            metric=abs(trap - dphi_end), tolerance=tol_i, unit=Unit.J_PER_M2,
            details=f"trapezoid ∫T dδ = {trap:.6g} J/m^2 vs Φ(end) − Φ(0) = {dphi_end:.6g} J/m^2 (absolute; "
                    f"tolerance max({consistency_rtol:g} max|Φ|, {periodicity_atol:g} J/m^2))"))
    else:
        rel = abs(trap - W) / max(abs(W), _EPS_W)
        checks.append(
            CheckResult(name="export_integrated_traction_vs_work_of_separation", passed=rel <= consistency_rtol,
                        metric=rel, tolerance=consistency_rtol,
                        details=f"trapezoid ∫T dδ = {trap:.6g} J/m^2 vs W_sep = {W:.6g} J/m^2 over the sampled range")
        )
        uber = next((p for p in (_parse_uber(f) for f in fits if f.model == "UBER") if p is not None), None)
        if uber is None:
            Wf, lf, rms = fit_uber(x, e)
            fits.append(CohesiveLawFit(
                model="UBER",
                parameters={"work_of_separation": q(Wf, Unit.J_PER_M2), "length_scale": q(lf, Unit.ANGSTROM),
                            "peak_traction": q(Wf / (math.e * lf) / GPA_ANGSTROM_TO_J_PER_M2, Unit.GPA)},
                rms_residual=q(rms, Unit.J_PER_M2),
                domain="full sampled range; least-squares on E(δ)/A with Φ = W[1-(1+δ/l)exp(-δ/l)] "
                "(fitted by fracture_md.export)",
            ))
        else:
            Wf, lf = uber
            rms = float(np.sqrt(np.mean((uber_energy(x, Wf, lf) - (e - e[0])) ** 2)))
        Tpk0 = law.peak_traction.magnitude(Unit.GPA)
        dpk0 = law.opening_at_peak.magnitude(Unit.ANGSTROM)
        Tpk_fit = Wf / (math.e * lf) / GPA_ANGSTROM_TO_J_PER_M2
        rW = abs(Wf - W) / max(abs(W), _EPS_W)
        rT = abs(Tpk_fit - Tpk0) / max(abs(Tpk0), _EPS_T)
        rD = abs(lf - dpk0) / max(abs(dpk0), 1e-12)
        uc = UBERConsistency(
            W_fit=q(Wf, Unit.J_PER_M2), l_fit=q(lf, Unit.ANGSTROM), peak_traction_fit=q(Tpk_fit, Unit.GPA),
            rel_diff_W=rW, rel_diff_peak_traction=rT, rel_diff_opening_at_peak=rD,
            rms_residual_energy=q(rms, Unit.J_PER_M2), tolerance=uber_rtol,
            within_tolerance=bool(max(rW, rT, rD) <= uber_rtol),
        )
        ip = int(np.argmax(y))
        sb = SofteningBranch(
            opening_after_peak=q(x[ip:] - x[ip], Unit.ANGSTROM),
            traction=q(y[ip:], Unit.GPA),
            G_post_peak=q(float(np.trapezoid(y[ip:], x[ip:]) * GPA_ANGSTROM_TO_J_PER_M2), Unit.J_PER_M2),
            G_pre_peak=q(float(np.trapezoid(y[: ip + 1], x[: ip + 1]) * GPA_ANGSTROM_TO_J_PER_M2), Unit.J_PER_M2),
        )
    Tpk = law.peak_traction.magnitude(Unit.GPA)
    dpk = law.opening_at_peak.magnitude(Unit.ANGSTROM)

    pclass = PropertyClass.STACKING_FAULTS if law.mode == "tangential" else PropertyClass.BOND_BREAKING
    own = decide_status(card, pclass, checks, [])
    statuses = [own] + [r.validation.status for r in matched
                        if r.name in (PropertyName.TRACTION_SEPARATION_NORMAL, PropertyName.TRACTION_SEPARATION_TANGENTIAL,
                                      PropertyName.PEAK_NORMAL_TRACTION, PropertyName.PEAK_TANGENTIAL_TRACTION)]
    status = worst_status(statuses)
    vnotes = ""
    if card is not None and card.status_for(pclass) == "known_deficient":
        vnotes = (f"Potential {card.id} is known_deficient for {pclass.value}; the numerical checks verify "
                  "the calculation, not the physics of bond breaking. ")
    failed = [c for c in checks if not c.passed]
    if failed:
        vnotes += "Failed checks: " + "; ".join(f"{c.name} ({c.details[:200]})" if c.details else c.name
                                                for c in failed) + ". "
    if any(c.name.startswith("export_") for c in failed):
        # The export's own consistency check (∫T dδ vs W) failed: the curve is internally
        # inconsistent, which is a data failure, not a model-domain issue -> FAILED even when the
        # potential is also out of domain (records.decide_status would report OUT_OF_DOMAIN first).
        status = ValidationStatus.FAILED
    unc = [u for r in matched for u in r.uncertainty]
    o = law.interface.orientation_a
    return CohesiveLawBlock(
        id=law_id or _law_id(law),
        plane_miller=o.plane_miller,
        plane_miller_bravais=o.plane_miller_bravais,
        termination=law.interface.termination.label if law.interface.termination else "",
        interface=law.interface,
        mode=law.mode,
        protocol_mode=law.protocol_mode,
        opening=q(x, Unit.ANGSTROM),
        traction=q(y, Unit.GPA),
        energy_per_area=q(e, Unit.J_PER_M2),
        peak_traction=q(Tpk, Unit.GPA),
        opening_at_peak=q(dpk, Unit.ANGSTROM),
        work_of_separation=q(W, Unit.J_PER_M2),
        integrated_work=q(law.integrated_work.magnitude(Unit.J_PER_M2), Unit.J_PER_M2),
        fits=fits,
        uber_consistency=uc,
        softening_branch=sb,
        assumptions=list(law.assumptions),
        checks=checks,
        validation_status=status,
        validation_notes=vnotes,
        uncertainty=unc,
        potential_id=card.id if card else (next(iter(pot_ids)) if pot_ids else None),
        conditions=matched[0].conditions if matched else Conditions(),
        source_record_ids=[r.id for r in matched],
    )


# ------------------------------------------------------------------ PF-CZM
def _crack_geometry(law: CohesiveLawBlock, crystal_system: Optional[str]) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Plane normal and propagation direction in crystal Cartesian coordinates."""
    o = law.interface.orientation_a
    if o.rotation_crystal_to_lab is not None:
        R = np.asarray(o.rotation_crystal_to_lab, float)
        return R.T @ np.array([0.0, 0, 1]), R.T @ np.array([1.0, 0, 0]), "from orientation.rotation_crystal_to_lab"
    if crystal_system == "cubic" and o.plane_miller is not None:
        n = np.asarray(o.plane_miller, float)
        m = np.asarray(o.x_direction, float) if o.x_direction is not None else None
        return n, m, "cubic: plane normal ∥ (hkl), propagation ∥ x_direction [uvw]" if m is not None else \
            "cubic: plane normal ∥ (hkl); propagation direction not specified (mean over in-plane directions)"
    return None, None, "no rotation and non-cubic (or unknown) crystal system: anisotropic E' not computed"


def _unit(v: Optional[np.ndarray]) -> Optional[tuple[float, float, float]]:
    if v is None:
        return None
    v = np.asarray(v, float)
    return tuple(float(x) for x in v / np.linalg.norm(v))  # type: ignore[return-value]


def law_frame(law: CohesiveLawBlock, crystal_system: Optional[str]) -> dict[str, Optional[np.ndarray]]:
    """Crystal-frame unit normal, propagation (lab x) and front (lab y) directions and R of a law.

    R = rotation_crystal_to_lab (rows = lab axes in crystal coordinates).  Entries are None when
    the orientation does not determine them (no rotation and non-cubic, or no x_direction).
    """
    n, m, _ = _crack_geometry(law, crystal_system)
    o = law.interface.orientation_a
    R = np.asarray(o.rotation_crystal_to_lab, float) if o.rotation_crystal_to_lab is not None else None
    n_u = None if n is None else np.asarray(_unit(n))
    m_u = None if m is None else np.asarray(_unit(m))
    t_u = None if (n_u is None or m_u is None) else np.cross(n_u, m_u)
    if R is None and n_u is not None and m_u is not None:
        R = np.stack([m_u, t_u, n_u])
    return dict(normal=n_u, propagation=m_u, front=t_u, rotation=R)


def _griffith_inputs(law: CohesiveLawBlock, records: Sequence[PropertyRecord]) -> dict[str, Any]:
    """Relaxed 2γ (preferred Griffith energy), rigid W_sep and the G_Ic record for this law's interface."""
    same = [r for r in records if _same_interface(r.interface, law.interface)]
    gic = [r for r in same if r.name == PropertyName.G_IC_ATOMISTIC]
    out: dict[str, Any] = dict(G_relaxed=None, G_relaxed_record_id=None, W_rigid=None, W_rigid_record_id=None,
                               G_IC_record_id=gic[0].id if gic else None)
    for r in gic:
        m = _GRIFFITH_SOURCE_TAG.search(r.method.name)
        if m and m.group(1) == "relaxed":
            out.update(G_relaxed=r.value.magnitude(Unit.J_PER_M2), G_relaxed_record_id=r.id, G_IC_record_id=r.id)
            break
    ws = [r for r in same if r.name == PropertyName.WORK_OF_SEPARATION]
    if out["G_relaxed"] is None:
        rel = [r for r in ws if _protocol_mode_of(r) == "relaxed_surfaces"]
        if rel:
            out.update(G_relaxed=rel[0].value.magnitude(Unit.J_PER_M2), G_relaxed_record_id=rel[0].id)
    if law.protocol_mode == "rigid":
        out.update(W_rigid=law.work_of_separation.magnitude(Unit.J_PER_M2))
    else:
        rig = [r for r in ws if _protocol_mode_of(r) == "rigid"]
        if rig:
            out.update(W_rigid=rig[0].value.magnitude(Unit.J_PER_M2), W_rigid_record_id=rig[0].id)
    return out


def _choose_softening(law: CohesiveLawBlock, f_t_GPa: float) -> tuple[str, list[SofteningFit], str]:
    sb = law.softening_branch
    w = sb.opening_after_peak.as_array(Unit.ANGSTROM)
    t = sb.traction.as_array(Unit.GPA)
    G = sb.G_post_peak.magnitude(Unit.J_PER_M2)
    if len(w) < 4 or G <= 0:
        return "exponential", [], "too few post-peak samples; exponential chosen by default (no fit)"
    scale = G / (f_t_GPa * GPA_ANGSTROM_TO_J_PER_M2)  # Å, = G/f_t
    # Compare only where the branch still carries work (cumulative post-peak work <= 99 %),
    # so that a long, flat zero tail does not dominate the RMS.
    from scipy.integrate import cumulative_trapezoid

    cum = cumulative_trapezoid(t, w, initial=0.0) * GPA_ANGSTROM_TO_J_PER_M2
    sel = cum <= 0.99 * G
    sel[: min(4, len(sel))] = True
    fits = []
    for s in ("linear", "exponential", "cornelissen"):
        r = float(np.sqrt(np.mean((t[sel] / f_t_GPa - softening_shape(s, w[sel] / scale)) ** 2)))
        fits.append(SofteningFit(softening=s, rms_normalized=r))
    best = min(fits, key=lambda f: f.rms_normalized)
    txt = ("Post-peak atomistic branch normalized as T/f_t vs (δ-δ_peak) f_t/G_post_peak and compared with the "
           "target shapes at equal area (samples up to 99 % of G_post_peak); RMS: " + ", ".join(f"{f.softening}={f.rms_normalized:.3f}" for f in fits)
           + f". Lowest RMS: {best.softening}.")
    return best.softening, fits, txt


def derive_pfczm_parameters(
    law: CohesiveLawBlock,
    elastic: ElasticBlock,
    *,
    candidate: Candidate,
    card: Optional[PotentialCard] = None,
    crystal_system: Optional[str] = None,
    E_prime_choice: Literal["anisotropic", "isotropic_hill"] = "anisotropic",
    softening: Optional[str] = None,
    records: Sequence[PropertyRecord] = (),
) -> tuple[PFCZMParameterSet, PropertyRecord]:
    """Derive the PF-CZM parameter set (and a K_Ic_griffith_estimate record) for one normal law.

    Model (all DERIVED, atomistic-ideal scale):
      E'   : anisotropic Barnett-Lothe plane-strain modulus for this crack plane
             (if computable) or the isotropic Hill E/(1-ν²); `E_prime_method` says which.
      f_t  : atomistic peak normal traction (ideal cleavage strength).
      G_c  : atomistic work of separation W_sep (Griffith: G_c = 2γ for a rigid cut).
      l_ch : E' G_c / f_t²  (Irwin length, nm).
      K    : sqrt(E' G_Ic)  (Griffith plane-strain estimate, MPa m^0.5) — not a toughness measurement.
             G_Ic = relaxed 2γ (the runner's G_Ic_atomistic record or relaxed W_sep record on the same
             interface, found in `records`) when available — the Griffith energy of the equilibrium
             surfaces — else the law's W_sep.  K_Ic_rigid_upper_bound = sqrt(E' W_sep,rigid).
      Crack geometry: plane normal = lab z, propagation = lab x of the law's orientation; both are
             stored (crystal frame and [uvw]) so it is explicit which direction E' refers to.
      Wu coefficients: E0 = Hill Young's modulus (Wu's definition), G_f = G_c, f_t.
    """
    if law.mode == "tangential":
        raise ValueError("PF-CZM mode-I parameters need a normal (or mixed) cohesive law")
    iso = elastic.isotropic_equivalent
    E_h = iso.youngs_modulus.magnitude(Unit.GPA)
    nu_h = iso.poisson_ratio.magnitude(Unit.DIMENSIONLESS)
    Ep_iso = E_h / (1.0 - nu_h**2)
    n, m, geom_note = _crack_geometry(law, crystal_system)
    Ep_an = None
    rng = None
    if n is not None:
        res = plane_strain_modulus_anisotropic(elastic.C_voigt.as_array(Unit.GPA), n, m)
        Ep_an = res["value"]
        rng = (q(res["min"], Unit.GPA), q(res["max"], Unit.GPA))
        geom_note += (f"; E' over in-plane crack-propagation directions ranges {res['min']:.1f}-{res['max']:.1f} GPa"
                      + ("" if m is not None else f"; mean {res['mean']:.1f} GPa used"))
    if E_prime_choice == "anisotropic" and Ep_an is not None:
        Ep, method = Ep_an, "anisotropic_barnett_lothe_plane_strain"
    else:
        Ep, method = Ep_iso, "isotropic_hill_plane_strain"

    f_t = law.peak_traction.magnitude(Unit.GPA)
    G_c = law.work_of_separation.magnitude(Unit.J_PER_M2)
    G_post = law.softening_branch.G_post_peak.magnitude(Unit.J_PER_M2) if law.softening_branch else float("nan")
    gi = _griffith_inputs(law, records)
    if gi["G_relaxed"] is not None:
        G_K, G_K_src = float(gi["G_relaxed"]), "relaxed_2gamma"
    else:
        G_K, G_K_src = G_c, "law_work_of_separation"
    # SI arithmetic
    Ep_Pa, f_Pa, E0_Pa = Ep * 1e9, f_t * 1e9, E_h * 1e9
    l_ch_m = Ep_Pa * G_c / f_Pa**2
    l_ch0_m = E0_Pa * G_c / f_Pa**2
    K_SI = math.sqrt(Ep_Pa * G_K)  # Pa m^0.5
    K = K_SI / 1e6  # MPa m^0.5
    K_rigid = math.sqrt(Ep_Pa * gi["W_rigid"]) / 1e6 if gi["W_rigid"] is not None and gi["W_rigid"] > 0 else None
    o = law.interface.orientation_a
    fr = law_frame(law, crystal_system)

    soft_fits: list[SofteningFit] = []
    if softening is None:
        softening, soft_fits, rationale = _choose_softening(law, f_t)
    else:
        rationale = "softening law chosen by the caller"
    wu = wu_pfczm_coefficients(E0_Pa, G_c, f_Pa, 1.0, softening)  # b = 1 m -> a1 = a1*b in m
    wu_block = WuCoefficients(
        softening=softening, p=wu["p"], a2=wu["a2"], a3=wu["a3"],
        a1_times_b=q(wu["a1"] * 1e9, Unit.NANOMETER),
        E0_used=q(E_h, Unit.GPA),
        b_max=q(wu["b_max"] * 1e9, Unit.NANOMETER),
    )

    elastic_status = elastic.validation_status
    input_statuses = {f"cohesive_law:{law.id}": law.validation_status, "elastic": elastic_status}
    worst = worst_status([law.validation_status, elastic_status])
    if worst == ValidationStatus.FAILED:
        status = ValidationStatus.FAILED
    elif worst == ValidationStatus.OUT_OF_DOMAIN:
        status = ValidationStatus.OUT_OF_DOMAIN
    else:
        status = ValidationStatus.MODEL_ESTIMATE
    vnotes = ("Derived model estimate; inherits the weakest input status. " +
              ("An input is out of the potential's validated domain. " if status == ValidationStatus.OUT_OF_DOMAIN else "") +
              ("An input failed a check. " if status == ValidationStatus.FAILED else "") +
              (f"Cohesive law: {law.validation_notes}" if law.validation_notes else ""))
    input_ids = list(dict.fromkeys(law.source_record_ids + elastic.source_record_ids
                                   + [i for i in (gi["G_IC_record_id"], gi["G_relaxed_record_id"],
                                                  gi["W_rigid_record_id"]) if i]))
    assumptions = [
        "Griffith / ideal brittle cleavage: G_c = W_sep of the cohesive law (area under T(δ)); K uses G_Ic = 2γ_relaxed "
        "when available (equilibrium surfaces), the rigid W_sep only as an upper bound; no plasticity, no lattice trapping",
        "plane strain, linear elastic bulk with the 0 K relaxed-ion stiffness",
        "f_t = atomistic peak traction of a defect-free crystal (ideal cleavage strength)",
        "no flaw population, grain boundaries, R-curve, temperature or rate effects",
    ]
    derivation = Derivation(
        model="Wu PF-CZM inputs from atomistic cleavage (ideal limits); Griffith plane-strain K estimate",
        equation="E' = " + ("2/(n.L^-1.n) (Barnett-Lothe)" if method.startswith("anisotropic") else "E_H/(1-nu_H^2)")
        + "; f_t = T_peak; G_c = W_sep; l_ch = E' G_c / f_t^2; K = sqrt(E' G_Ic), G_Ic = "
        + ("2γ_relaxed" if G_K_src == "relaxed_2gamma" else "W_sep (law)"),
        assumptions=assumptions,
        input_record_ids=input_ids,
    )
    k_rec = make_record(
        name=PropertyName.K_IC_GRIFFITH_ESTIMATE,
        value=K,
        unit=Unit.MPA_SQRT_M,
        candidate=candidate,
        method=MethodSpec(
            name=f"Griffith plane-strain estimate K = sqrt(E' G_Ic), G_Ic = {G_K_src} ({law.protocol_mode} law, {method})",
            engine="fracture_md.export.pfczm",
            description=f"K = sqrt(E' G_Ic) with G_Ic = {G_K:.6g} J/m^2 ({G_K_src}); model estimate at the "
                        "ideal-cleavage limit" + (f"; rigid upper bound sqrt(E' W_sep,rigid) = {K_rigid:.6g} MPa m^0.5"
                                                  if K_rigid is not None else ""),
            assumptions=assumptions,
        ),
        card=card,
        origin=PropertyOrigin.DERIVED_MODEL,
        scale=PropertyScale.ATOMISTIC_IDEAL,
        conditions=law.conditions,
        interface=law.interface,
        derivation=derivation,
        notes=ATOMISTIC_SCALE_NOTE,
        validation_notes=f"inputs: {', '.join(f'{k}={v.value}' for k, v in input_statuses.items())}",
    )
    ps = PFCZMParameterSet(
        id=f"pfczm-{law.id}",
        cohesive_law_id=law.id,
        E_prime=q(Ep, Unit.GPA),
        E_prime_method=method,  # type: ignore[arg-type]
        E_prime_isotropic_hill=q(Ep_iso, Unit.GPA),
        E_prime_anisotropic=q(Ep_an, Unit.GPA) if Ep_an is not None else None,
        E_prime_anisotropic_range=rng,
        E_prime_anisotropic_note=geom_note,
        crack_plane_normal_crystal=_unit(fr["normal"]),
        crack_propagation_direction_crystal=_unit(fr["propagation"]),
        crack_front_direction_crystal=_unit(fr["front"]),
        crack_propagation_direction_uvw=o.x_direction,
        crack_front_direction_uvw=o.y_direction,
        tensile_strength=q(f_t, Unit.GPA),
        fracture_energy=q(G_c, Unit.J_PER_M2),
        G_post_peak=q(G_post, Unit.J_PER_M2),
        irwin_length=q(l_ch_m * 1e9, Unit.NANOMETER),
        irwin_length_E0=q(l_ch0_m * 1e9, Unit.NANOMETER),
        K_Ic_griffith_estimate=q(K, Unit.MPA_SQRT_M),
        K_Ic_griffith_energy=q(G_K, Unit.J_PER_M2),
        K_Ic_griffith_energy_source=G_K_src,  # type: ignore[arg-type]
        G_IC_record_id=gi["G_IC_record_id"],
        K_Ic_rigid_upper_bound=q(K_rigid, Unit.MPA_SQRT_M) if K_rigid is not None else None,
        W_sep_rigid=q(gi["W_rigid"], Unit.J_PER_M2) if gi["W_rigid"] is not None else None,
        K_Ic_record_id=k_rec.id,
        recommended_softening=softening,  # type: ignore[arg-type]
        softening_rationale=rationale,
        softening_fits=soft_fits,
        wu=wu_block,
        derivation=derivation,
        validation_status=status,
        input_statuses=input_statuses,
        validation_notes=vnotes,
    )
    return ps, k_rec


# ------------------------------------------------------------------ extensions from records
def _extensions_from_records(records: Sequence[PropertyRecord], candidate: Candidate) -> ExtensionBlocks:
    ext = ExtensionBlocks()
    upd: dict[str, Any] = {}
    # defects: group non-pristine records by defect state
    groups: dict[str, tuple[DefectState, list[str]]] = {}
    for r in records:
        ds = r.conditions.defect_state
        if ds.kind != "pristine":
            k = canonical_json(ds.model_dump(mode="json"))
            groups.setdefault(k, (ds, []))[1].append(r.id)
    if groups:
        upd["defects"] = DefectExtension(status="partial", sets=[
            DefectPropertySet(defect_state=ds, property_record_ids=ids,
                              notes="records only; no defect-specific elastic/cohesive blocks were built")
            for ds, ids in groups.values()],
            notes="populated automatically from records with a non-pristine defect_state")
    # temperature / strain rate overrides
    tpts: dict[float, list[ParameterOverride]] = {}
    rpts: dict[float, list[ParameterOverride]] = {}
    for r in records:
        T = r.conditions.temperature.magnitude(Unit.KELVIN)
        if T > 0:
            tpts.setdefault(T, []).append(ParameterOverride(target=r.name.value, value=r.value, record_id=r.id))
        if r.conditions.strain_rate is not None:
            rate = r.conditions.strain_rate.magnitude(Unit.PER_SECOND)
            rpts.setdefault(rate, []).append(ParameterOverride(target=r.name.value, value=r.value, record_id=r.id))
    if tpts:
        upd["temperature"] = TemperatureExtension(status="partial", points=[
            TemperaturePoint(temperature=q(T, Unit.KELVIN), overrides=o) for T, o in sorted(tpts.items())],
            notes="record values at T > 0 K; not yet propagated into the PF-CZM parameter sets")
    if rpts:
        upd["strain_rate"] = StrainRateExtension(status="partial", points=[
            StrainRatePoint(strain_rate=q(v, Unit.PER_SECOND), overrides=o) for v, o in sorted(rpts.items())])
    # uncertainty samples
    samples = []
    for r in records:
        for u in r.uncertainty:
            if u.samples:
                samples.append(UncertaintySamples(
                    target=r.name.value, record_id=r.id, kind=u.kind, distribution="empirical",
                    samples=Quantity(value=list(u.samples), unit=u.unit or r.value.unit),
                    description=u.description))
    if samples:
        upd["uncertainty"] = UncertaintyExtension(status="partial", samples=samples)
    if candidate.microstructure is not None:
        upd["microstructure"] = MicrostructureExtension(status="computed", microstructure=candidate.microstructure,
                                                        notes="specified by the candidate (input), not computed")
    return ext.model_copy(update=upd)


def build_bundle(
    candidate: Candidate,
    cards: Sequence[PotentialCard],
    property_records: Sequence[PropertyRecord],
    traction_laws: Sequence[TractionSeparationLaw] = (),
    provenance: Optional[Provenance] = None,
    *,
    bundle_id: Optional[str] = None,
    extensions: Optional[ExtensionBlocks] = None,
    E_prime_choice: Literal["anisotropic", "isotropic_hill"] = "anisotropic",
    softening: Optional[str] = None,
    consistency_rtol: float = 0.05,
    notes: Sequence[str] = (),
) -> PFCZMBundle:
    """Assemble a PF-CZM bundle from bulk-runner records and cleavage-runner laws.

    Tolerant of missing pieces: without a stiffness record there is no elastic
    block and no parameter sets; without traction laws there are no cohesive
    laws; extension blocks default to "not_computed" (defects/temperature/
    strain-rate/uncertainty/microstructure are populated automatically when
    the records or candidate carry that information).  `extensions`, when
    given, overrides the automatically built blocks field by field (only the
    fields whose status is not "not_computed").
    """
    build_notes: list[str] = list(notes)
    # De-duplicate records by id (the same protocol run twice yields identical deterministic ids).
    records: list[PropertyRecord] = []
    seen_rec: dict[str, PropertyRecord] = {}
    for r in property_records:
        if r.id in seen_rec:
            if seen_rec[r.id].value != r.value:
                build_notes.append(f"record id {r.id} supplied twice with different values; first kept")
            continue
        seen_rec[r.id] = r
        records.append(r)
    phase = candidate.phases[0] if candidate.phases else None
    phase_name = phase.name if phase else candidate.name
    crystal_system = phase.crystal_system if phase else None

    dens = [r for r in records if r.name == PropertyName.DENSITY and r.conditions.defect_state.kind == "pristine"]
    material = BundleMaterial(
        candidate_id=candidate.id,
        candidate_name=candidate.name,
        composition=candidate.composition,
        phase_name=phase_name,
        phase=phase,
        density=dens[0].value.to(Unit.G_PER_CM3) if dens else None,
        density_record_id=dens[0].id if dens else None,
    )
    if not dens:
        build_notes.append("no density record")

    # normals of the cleavage planes (cubic) -> directional E along crack normals
    extra_dirs = []
    if crystal_system == "cubic":
        for law in traction_laws:
            pm = law.interface.orientation_a.plane_miller
            if pm is not None:
                extra_dirs.append(pm)
    elastic, enotes = elastic_block_from_records(records, extra_dirs)
    build_notes.extend(enotes)

    normal_blocks: list[CohesiveLawBlock] = []
    tangential_blocks: list[CohesiveLawBlock] = []
    gb_sets: dict[str, InterfaceCohesiveSet] = {}
    if_sets: dict[str, InterfaceCohesiveSet] = {}
    seen_laws: dict[str, TractionSeparationLaw] = {}
    for law in traction_laws:
        lid = _law_id(law)
        if lid in seen_laws:
            if seen_laws[lid] == law:
                build_notes.append(f"traction law {lid} supplied twice (identical); exported once")
                continue
            k = 2
            while f"{lid}-{k}" in seen_laws:
                k += 1
            build_notes.append(f"traction law {lid} supplied twice with different data; second exported as {lid}-{k}")
            lid = f"{lid}-{k}"
        seen_laws[lid] = law
        try:
            blk = cohesive_law_from_tsl(law, records=records, cards=cards, consistency_rtol=consistency_rtol,
                                        law_id=lid)
        except ValueError as exc:  # pydantic ValidationError is a ValueError
            # e.g. an elastically corrected relaxed curve with snap-back (non-monotonic opening):
            # it cannot be a single-valued T(δ) and is not exported as a cohesive law.
            build_notes.append(f"traction law {_law_id(law)} not exported: {str(exc).splitlines()[0]} "
                               f"({str(exc).splitlines()[-1].strip() if len(str(exc).splitlines()) > 1 else ''})")
            continue
        kind = law.interface.kind
        if kind in (InterfaceKind.CLEAVAGE_PLANE, InterfaceKind.FREE_SURFACE, InterfaceKind.STACKING_FAULT):
            (tangential_blocks if law.mode == "tangential" else normal_blocks).append(blk)
        else:
            target = gb_sets if kind in (InterfaceKind.GRAIN_BOUNDARY, InterfaceKind.TWIN_BOUNDARY) else if_sets
            k = canonical_json(law.interface.model_dump(mode="json"))
            target.setdefault(k, InterfaceCohesiveSet(interface=law.interface)).cohesive_laws.append(blk)

    params: list[PFCZMParameterSet] = []
    derived_records: list[PropertyRecord] = []
    if elastic is not None:
        for blk in normal_blocks:
            card = next((c for c in cards if c.id == blk.potential_id), None)
            ps, krec = derive_pfczm_parameters(blk, elastic, candidate=candidate, card=card,
                                               crystal_system=crystal_system, E_prime_choice=E_prime_choice,
                                               softening=softening, records=records)
            params.append(ps)
            derived_records.append(krec)
    elif normal_blocks:
        build_notes.append("cohesive laws present but no elastic block: PF-CZM parameter sets not derived")

    ext = _extensions_from_records(records, candidate)
    upd: dict[str, Any] = {}
    if tangential_blocks:
        upd["tangential"] = TangentialExtension(status="computed", laws=tangential_blocks)
    if gb_sets:
        upd["grain_boundaries"] = InterfaceExtension(status="computed", interfaces=list(gb_sets.values()))
    if if_sets:
        upd["interfaces"] = InterfaceExtension(status="computed", interfaces=list(if_sets.values()))
    ext = ext.model_copy(update=upd)
    if extensions is not None:
        ext = ext.model_copy(update={k: getattr(extensions, k) for k in type(extensions).model_fields
                                     if getattr(extensions, k).status != "not_computed"})
    # re-validate the assembled extension block
    ext = ExtensionBlocks.model_validate(ext.model_dump())

    used_ids = {r.potential_id for r in records if r.potential_id}
    pots = [c for c in cards if c.id in used_ids] or list(cards)

    bundle_checks: list[CheckResult] = []
    if elastic is not None:
        bundle_checks.extend(elastic.checks)
    for blk in normal_blocks + tangential_blocks:
        bundle_checks.extend(c for c in blk.checks if c.name.startswith("export_"))

    if provenance is None:
        from fracture_md.provenance import collect

        provenance = collect(engine="fracture_md.export.pfczm", requested_by="human",
                             parent_ids=[r.id for r in records])
    created = datetime.now(timezone.utc)
    if bundle_id is None:
        h = hashlib.sha256(canonical_json([candidate.id, sorted(r.id for r in records),
                                           [b.id for b in normal_blocks], created.isoformat()]).encode()).hexdigest()
        bundle_id = f"pfczm-bundle:{candidate.id}:{h[:12]}"

    assumptions = [
        ATOMISTIC_SCALE_NOTE,
        "Elastic constants: 0 K static relaxed-ion values of the stated potential, crystal frame.",
        "Cohesive laws: opening of a periodic slab on a crystallographic plane; the in-plane lattice is held at "
        "the bulk value. Relaxed post-peak branches are slab-size dependent (see each law's assumptions).",
        "Validation status of every block is inherited from the potential's per-property-class domain "
        "assessment; OUT_OF_DOMAIN means the numbers are numerically correct for the model but the model "
        "is known to be deficient for that physics.",
    ]
    return PFCZMBundle(
        bundle_id=bundle_id,
        created_utc=created,
        material=material,
        potentials=pots,
        provenance=provenance,
        elastic=elastic,
        cohesive_laws=normal_blocks,
        pfczm_parameters=params,
        extensions=ext,
        property_records=records + derived_records,
        checks=bundle_checks,
        assumptions=assumptions,
        notes=build_notes,
    )


# =============================================================================
# Serialization, integrity, loading
# =============================================================================
def _inline_arrays(bundle: PFCZMBundle) -> list[tuple[str, str, np.ndarray, Unit]]:
    """(sidecar key, json path, array, unit) for every array the sidecar mirrors."""
    out = []
    if bundle.elastic is not None:
        out.append(("elastic__C_voigt", "elastic.C_voigt", bundle.elastic.C_voigt.as_array(), bundle.elastic.C_voigt.unit))
        out.append(("elastic__S_voigt", "elastic.S_voigt", bundle.elastic.S_voigt.as_array(), bundle.elastic.S_voigt.unit))
    for law in bundle.all_cohesive_laws():
        for f in ("opening", "traction", "energy_per_area"):
            qq = getattr(law, f)
            out.append((f"law__{law.id}__{f}", f"law[{law.id}].{f}", qq.as_array(), qq.unit))
    for s in bundle.extensions.mixed_mode.surfaces:
        for f in ("normal_opening", "tangential_opening", "energy_per_area"):
            qq = getattr(s, f)
            out.append((f"mixed__{s.id}__{f}", f"mixed_mode[{s.id}].{f}", qq.as_array(), qq.unit))
    return out


def _manifest(bundle: PFCZMBundle) -> list[ArrayManifestEntry]:
    return [ArrayManifestEntry(key=k, json_path=p, shape=list(a.shape), unit=u, sha256=array_sha256(a))
            for k, p, a, u in _inline_arrays(bundle)]


def write_bundle(path: Union[str, Path], bundle: PFCZMBundle, sidecar: bool = True) -> PFCZMBundle:
    """Write `bundle` as JSON (and an NPZ sidecar mirroring all arrays).

    The integrity block is (re)computed: per-array sha256, sidecar file sha256
    and the content sha256.  Returns the bundle with the updated integrity.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(bundle)
    sidecar_name, sidecar_hash = None, None
    if sidecar:
        npz = path.with_suffix(".npz")
        arrays = {k: np.asarray(a, dtype="<f8") for k, _, a, _ in _inline_arrays(bundle)}
        arrays["__units__"] = np.asarray(json.dumps({m.key: m.unit.value for m in manifest}))
        with open(npz, "wb") as f:
            np.savez(f, **arrays)
        sidecar_name, sidecar_hash = npz.name, file_sha256(npz)
    integ = IntegrityBlock(content_sha256=bundle.content_sha256(), arrays=manifest,
                           sidecar_file=sidecar_name, sidecar_sha256=sidecar_hash)
    out = bundle.model_copy(update={"integrity": integ})
    d = out.model_dump(mode="json")
    path.write_text(json.dumps(d, indent=1, sort_keys=False, ensure_ascii=False, allow_nan=False) + "\n")
    return out


def check_bundle(bundle: PFCZMBundle, rtol: float = 0.05, periodicity_atol: float = 1e-4) -> list[CheckResult]:
    """Physics/consistency checks run on load (never raise; the caller decides)."""
    res: list[CheckResult] = []
    for law in bundle.all_cohesive_laws():
        W = law.work_of_separation.magnitude(Unit.J_PER_M2)
        trap = law.trapezoid_work()
        e = law.arrays()["energy_per_area"]
        if law.mode == "tangential":
            # periodic γ-line: W ≈ 0, so absolute checks (see cohesive_law_from_tsl)
            amp = float(np.max(np.abs(e - e[0])))
            d_end = float(e[-1] - e[0])
            tol_i = max(rtol * amp, periodicity_atol)
            res.append(CheckResult(name=f"gamma_line_periodicity[{law.id}]", passed=abs(d_end) <= periodicity_atol,
                                   metric=abs(d_end), tolerance=periodicity_atol, unit=Unit.J_PER_M2,
                                   details="|Φ(end) - Φ(0)| (one lattice period)"))
            res.append(CheckResult(name=f"integral_T_vs_energy_change[{law.id}]", passed=abs(trap - d_end) <= tol_i,
                                   metric=abs(trap - d_end), tolerance=tol_i, unit=Unit.J_PER_M2,
                                   details=f"∫T dδ = {trap:.6g} vs Φ(end) - Φ(0) = {d_end:.6g} J/m^2"))
            continue
        rel = abs(trap - W) / max(abs(W), _EPS_W)
        res.append(CheckResult(name=f"integral_T_vs_W[{law.id}]", passed=rel <= rtol, metric=rel, tolerance=rtol,
                               details=f"∫T dδ = {trap:.6g} J/m^2 vs W = {W:.6g} J/m^2"))
        rel_e = abs((e[-1] - e[0]) - W) / max(abs(W), _EPS_W)
        res.append(CheckResult(name=f"energy_curve_end_vs_W[{law.id}]", passed=rel_e <= rtol, metric=rel_e,
                               tolerance=rtol, details="(E(δ_max) - E(0))/A vs W"))
    if bundle.elastic is not None:
        C = bundle.elastic.C_voigt.as_array(Unit.GPA)
        mn = float(np.linalg.eigvalsh(C).min())
        res.append(CheckResult(name="elastic_positive_definite", passed=mn > 0, metric=mn, tolerance=0.0, unit=Unit.GPA))
    for ps in bundle.pfczm_parameters:
        law = bundle.law(ps.cohesive_law_id)
        Ep = ps.E_prime.magnitude(Unit.PA)
        G = ps.fracture_energy.magnitude(Unit.J_PER_M2)
        f = ps.tensile_strength.magnitude(Unit.PA)
        l_ok = abs(ps.irwin_length.magnitude(Unit.METER) - Ep * G / f**2) <= 1e-9 * Ep * G / f**2
        GK = ps.K_Ic_griffith_energy.magnitude(Unit.J_PER_M2) if ps.K_Ic_griffith_energy is not None else G
        k_ok = abs(ps.K_Ic_griffith_estimate.magnitude() * 1e6 - math.sqrt(Ep * GK)) <= 1e-9 * math.sqrt(Ep * GK)
        if ps.K_Ic_rigid_upper_bound is not None and ps.W_sep_rigid is not None:
            Kb = math.sqrt(Ep * ps.W_sep_rigid.magnitude(Unit.J_PER_M2))
            k_ok = k_ok and abs(ps.K_Ic_rigid_upper_bound.magnitude(Unit.MPA_SQRT_M) * 1e6 - Kb) <= 1e-9 * Kb
        same = (abs(ps.tensile_strength.magnitude(Unit.GPA) - law.peak_traction.magnitude(Unit.GPA)) <= 1e-12 * f
                and abs(G - law.work_of_separation.magnitude(Unit.J_PER_M2)) <= 1e-12 * max(G, 1))
        res.append(CheckResult(name=f"pfczm_derivation_consistent[{ps.id}]", passed=bool(l_ok and k_ok and same),
                               details="l_ch = E'G_c/f^2, K = sqrt(E'G_Ic), K_rigid = sqrt(E'W_rigid), f_t/G_c "
                                       "equal to the law's values"))
    return res


def load_bundle(
    path: Union[str, Path],
    *,
    verify_integrity: bool = True,
    require_integrity: bool = True,
    consistency_rtol: float = 0.05,
) -> PFCZMBundle:
    """Load and validate a bundle.

    * format must be BUNDLE_FORMAT; major version must equal ours
      (BundleVersionError otherwise); a newer minor/patch warns
      (BundleVersionWarning) and fails only if it uses fields we do not know.
    * pydantic validation: units, 6x6 shapes, symmetric C, S = inv(C),
      strictly increasing openings, extension-status consistency, derived-K labelling.
    * integrity: content sha256, every inline array sha256, the sidecar file
      sha256 and sidecar arrays == inline arrays (BundleIntegrityError).  With
      ``require_integrity=True`` (the default, for solver use) a bundle whose
      integrity block was stripped (no content sha256) is rejected; pass
      ``require_integrity=False`` to read hand-edited/partial bundles, or
      ``verify_integrity=False`` to skip all hash checks.
    * physics: ∫T dδ ≈ W and E(δ_max)-E(0) ≈ W within `consistency_rtol`
      (BundleConsistencyWarning), derivation identities of the parameter sets.
    """
    path = Path(path)
    raw = json.loads(path.read_text())
    fmt = raw.get("format")
    if fmt != BUNDLE_FORMAT:
        raise BundleVersionError(f"not a PF-CZM bundle: format={fmt!r}")
    ver = raw.get("version")
    if ver is None:
        raise BundleVersionError("bundle has no version")
    major, minor, patch = parse_version(ver)
    ours = parse_version(BUNDLE_VERSION)
    if major != ours[0]:
        raise BundleVersionError(f"bundle major version {major} != supported {ours[0]} (file {ver}, code {BUNDLE_VERSION})")
    newer = (minor, patch) > ours[1:]
    if newer:
        warnings.warn(f"bundle version {ver} is newer than this reader ({BUNDLE_VERSION}); unknown optional "
                      "fields will be rejected", BundleVersionWarning, stacklevel=2)
    try:
        bundle = PFCZMBundle.model_validate(raw)
    except Exception as e:
        if newer and "extra" in str(e).lower():
            raise BundleVersionError(f"bundle {ver} uses fields unknown to reader {BUNDLE_VERSION}: {e}") from e
        raise

    if verify_integrity:
        integ = bundle.integrity
        if integ.content_sha256 is None and require_integrity:
            raise BundleIntegrityError("bundle has no integrity block (content_sha256 is null): it was not written by "
                                       "write_bundle or the block was stripped; pass require_integrity=False to accept it")
        if integ.content_sha256 is not None:
            got = bundle.content_sha256()
            if got != integ.content_sha256:
                raise BundleIntegrityError(f"content sha256 mismatch: recorded {integ.content_sha256}, computed {got}")
        current = {m.key: m for m in _manifest(bundle)}
        for m in integ.arrays:
            c = current.get(m.key)
            if c is None or c.sha256 != m.sha256 or c.shape != m.shape or c.unit != m.unit:
                raise BundleIntegrityError(f"inline array {m.key} does not match its recorded sha256/shape/unit")
        if integ.sidecar_file:
            npz = path.parent / integ.sidecar_file
            if not npz.exists():
                raise BundleIntegrityError(f"sidecar {npz} referenced by the bundle is missing")
            got = file_sha256(npz)
            if got != integ.sidecar_sha256:
                raise BundleIntegrityError(f"sidecar sha256 mismatch: recorded {integ.sidecar_sha256}, computed {got}")
            with np.load(npz, allow_pickle=False) as z:
                for m in integ.arrays:
                    if m.key not in z.files or array_sha256(z[m.key]) != m.sha256:
                        raise BundleIntegrityError(f"sidecar array {m.key} differs from the inline array")

    for c in check_bundle(bundle, rtol=consistency_rtol):
        if not c.passed:
            warnings.warn(f"bundle consistency check failed: {c.name}: metric={c.metric} tol={c.tolerance} "
                          f"{c.details}", BundleConsistencyWarning, stacklevel=2)
    return bundle


def load_sidecar(path: Union[str, Path]) -> dict[str, np.ndarray]:
    """Arrays of a bundle's NPZ sidecar (keys as in integrity.arrays; units in '__units__')."""
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def json_schema() -> dict:
    s = PFCZMBundle.model_json_schema(mode="validation")
    s = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"urn:fracture-md:pfczm-bundle:{BUNDLE_VERSION}",
        "title": "fracture-md PF-CZM property bundle",
        "description": f"format={BUNDLE_FORMAT}, version={BUNDLE_VERSION}. Generated from "
        "fracture_md.export.pfczm.PFCZMBundle by write_json_schema(); do not edit by hand.",
        **{k: v for k, v in s.items() if k not in ("title", "description")},
    }
    return s


def write_json_schema(path: Union[str, Path]) -> Path:
    """Write the JSON Schema of PFCZMBundle (deterministic: sorted keys, indent 2)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_schema(), indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return path


# =============================================================================
# JAX interface
# =============================================================================
_UNIT_SYSTEMS = {
    # quantity kind -> unit, per system
    "SI": dict(stress=Unit.PA, length=Unit.METER, energy_area=Unit.J_PER_M2, compliance=Unit.PER_PA,
               density=Unit.KG_PER_M3),
    "metal": dict(stress=Unit.EV_PER_ANGSTROM3, length=Unit.ANGSTROM, energy_area=Unit.EV_PER_ANGSTROM2,
                  compliance=None, density=Unit.G_PER_CM3),
}
# 1 Pa m^0.5 in eV/Å^2.5: 1 eV/Å^3 = 1.602176634e11 Pa, 1 Å^0.5 = 1e-5 m^0.5.
_PA_SQRT_M_PER_EV_A25 = 1.602176634e11 * 1e-5


def bundle_to_jax(bundle: PFCZMBundle, units: str = "SI") -> dict:
    """Pytree (nested dict) of jnp float64 arrays for a JAX PF-CZM solver.

    units="SI": C in Pa, S in 1/Pa, lengths/openings in m, tractions and
    strengths in Pa, energies per area in J/m^2, density kg/m^3, Irwin length
    in m, K in Pa*m^0.5.  units="metal": eV/Å^3, Å, eV/Å^2, g/cm^3, K in
    eV/Å^2.5 (S in Å^3/eV).  Keys are stable; cohesive laws and parameter
    sets are keyed by their ids.  Every law carries its crack geometry as
    crystal-frame unit vectors ``normal_crystal`` (plane normal = lab z),
    ``propagation_crystal`` (lab x, the direction E' refers to) and the 3x3
    ``rotation_crystal_to_lab`` (NaN where the orientation does not fix them);
    G_post_peak is NaN for tangential laws, K_Ic_rigid_upper_bound NaN when no
    rigid W_sep was available.
    """
    import jax.numpy as jnp

    if units not in _UNIT_SYSTEMS:
        raise ValueError(f"units must be one of {sorted(_UNIT_SYSTEMS)}")
    U = _UNIT_SYSTEMS[units]
    f64 = lambda a: jnp.asarray(np.asarray(a, dtype=np.float64))  # noqa: E731
    out: dict[str, Any] = {"units": units}
    if bundle.elastic is not None:
        el = bundle.elastic
        C = el.C_voigt.as_array(U["stress"])
        S = np.linalg.inv(C) if U["compliance"] is None else el.S_voigt.as_array(U["compliance"])
        iso = el.isotropic_equivalent
        out["elastic"] = dict(
            C_voigt=f64(C), S_voigt=f64(S), C_tensor=f64(stiffness_voigt_to_tensor(C)),
            C_tensor_lab=f64(rotate_tensor4(stiffness_voigt_to_tensor(C), np.asarray(el.rotation_crystal_to_lab))),
            rotation_crystal_to_lab=f64(el.rotation_crystal_to_lab),
            bulk_modulus=f64(iso.bulk_modulus.magnitude(U["stress"])),
            shear_modulus=f64(iso.shear_modulus.magnitude(U["stress"])),
            youngs_modulus=f64(iso.youngs_modulus.magnitude(U["stress"])),
            poisson_ratio=f64(iso.poisson_ratio.magnitude(Unit.DIMENSIONLESS)),
        )
    if bundle.material.density is not None:
        out["density"] = f64(bundle.material.density.magnitude(U["density"]))
    laws = {}
    cs = bundle.material.phase.crystal_system if bundle.material.phase else None
    nan3 = np.full(3, np.nan)
    for law in bundle.all_cohesive_laws():
        fr = law_frame(law, cs)
        d = dict(
            opening=f64(law.opening.as_array(U["length"])),
            traction=f64(law.traction.as_array(U["stress"])),
            energy_per_area=f64(law.energy_per_area.as_array(U["energy_area"])),
            peak_traction=f64(law.peak_traction.magnitude(U["stress"])),
            opening_at_peak=f64(law.opening_at_peak.magnitude(U["length"])),
            work_of_separation=f64(law.work_of_separation.magnitude(U["energy_area"])),
            G_post_peak=f64(law.softening_branch.G_post_peak.magnitude(U["energy_area"])
                            if law.softening_branch is not None else np.nan),
            # crystal-frame unit vectors (NaN when undetermined): plane normal (= lab z), propagation (= lab x)
            normal_crystal=f64(fr["normal"] if fr["normal"] is not None else nan3),
            propagation_crystal=f64(fr["propagation"] if fr["propagation"] is not None else nan3),
            rotation_crystal_to_lab=f64(fr["rotation"] if fr["rotation"] is not None else np.full((3, 3), np.nan)),
        )
        up = law.uber_parameters()
        if up is not None:
            d["uber_W"] = f64(q(up[0], Unit.J_PER_M2).magnitude(U["energy_area"]))
            d["uber_l"] = f64(q(up[1], Unit.ANGSTROM).magnitude(U["length"]))
        laws[law.id] = d
    out["cohesive_laws"] = laws
    params = {}
    for ps in bundle.pfczm_parameters:
        K_SI = ps.K_Ic_griffith_estimate.magnitude(Unit.MPA_SQRT_M) * 1e6
        params[ps.cohesive_law_id] = dict(
            E_prime=f64(ps.E_prime.magnitude(U["stress"])),
            E_prime_isotropic_hill=f64(ps.E_prime_isotropic_hill.magnitude(U["stress"])),
            f_t=f64(ps.tensile_strength.magnitude(U["stress"])),
            G_c=f64(ps.fracture_energy.magnitude(U["energy_area"])),
            G_post_peak=f64(ps.G_post_peak.magnitude(U["energy_area"])),
            l_ch=f64(ps.irwin_length.magnitude(U["length"])),
            K_Ic_griffith_estimate=f64(K_SI if units == "SI" else K_SI / _PA_SQRT_M_PER_EV_A25),
            K_Ic_rigid_upper_bound=f64(np.nan if ps.K_Ic_rigid_upper_bound is None else
                                       ps.K_Ic_rigid_upper_bound.magnitude(Unit.MPA_SQRT_M) * 1e6
                                       / (1.0 if units == "SI" else _PA_SQRT_M_PER_EV_A25)),
            wu_p=f64(ps.wu.p), wu_a2=f64(ps.wu.a2), wu_a3=f64(ps.wu.a3),
            wu_a1_times_b=f64(ps.wu.a1_times_b.magnitude(U["length"])),
        )
    out["pfczm"] = params
    return out


def traction_interpolant(
    law: CohesiveLawBlock, units: str = "SI", beyond: Literal["zero", "hold"] = "zero"
) -> Callable:
    """Jittable, differentiable T(δ) by linear interpolation of the sampled law.

    SI: δ in m -> T in Pa ("metal": Å -> eV/Å^3).  For δ below the first
    sample the first traction is returned; above the last sample the traction
    is 0 ("zero", fully separated) or held at the last value ("hold").  The
    derivative (jax.grad) is the piecewise-constant segment slope.
    """
    import jax.numpy as jnp

    U = _UNIT_SYSTEMS[units]
    x = jnp.asarray(law.opening.as_array(U["length"]))
    y = jnp.asarray(law.traction.as_array(U["stress"]))
    right = 0.0 if beyond == "zero" else float(y[-1])

    def T(delta):
        return jnp.interp(delta, x, y, left=y[0], right=right)

    return T


def uber_traction_interpolant(law: CohesiveLawBlock, units: str = "SI") -> Callable:
    """Analytic UBER traction T(δ) = W δ/l² exp(-δ/l) from the law's UBER fit (jittable)."""
    import jax.numpy as jnp

    up = law.uber_parameters()
    if up is None:
        raise ValueError(f"cohesive law {law.id} has no UBER fit")
    U = _UNIT_SYSTEMS[units]
    W = q(up[0], Unit.J_PER_M2).magnitude(U["energy_area"])
    lval = q(up[1], Unit.ANGSTROM).magnitude(U["length"])

    def T(delta):
        delta = jnp.asarray(delta)
        return W * delta / lval**2 * jnp.exp(-delta / lval)

    return T


# =============================================================================
# SYNTHETIC inputs (tests, documentation and the example's fallback path ONLY)
# =============================================================================
def synthetic_sic_inputs(
    C11: float = 437.0,
    C12: float = 118.0,
    C44: float = 254.0,
    W: float = 4.0,
    l_uber: float = 0.3,
    planes: Sequence[tuple[tuple[int, int, int], str, tuple[int, int, int], tuple[int, int, int], float, float]] = (),
    n_open: int = 241,
    max_open: float = 6.0,
) -> dict[str, Any]:
    """SYNTHETIC but realistic 3C-SiC-like inputs shaped exactly like the runners' outputs.

    NOT computed physics: the stiffness is a cubic matrix with C11/C12/C44 of the
    order produced by Tersoff-type SiC potentials, and each traction law is an
    exact UBER curve Φ = W[1-(1+δ/l)e^{-δ/l}] sampled on [0, max_open] Å.  All
    records are built with records.make_record against the real Tersoff (1989)
    PotentialCard so validation statuses follow the production rules, and each
    is labelled SYNTHETIC in method name, notes and provenance.

    Returns dict(candidate, cards, records, laws).
    """
    from fracture_md.potentials import tersoff_1989_sic
    from fracture_md.schema.material import CandidateKind, Orientation, Termination
    from fracture_md.schema.properties import SampledCurve
    from fracture_md.structures.builders import sic_3c_phase, sic_composition

    card = tersoff_1989_sic().card
    cand = Candidate(id="3C-SiC-synthetic", name="3C-SiC (synthetic export fixture)", kind=CandidateKind.REFERENCE,
                     composition=sic_composition(), phases=[sic_3c_phase()])
    prov = Provenance(engine="synthetic", extra={"synthetic": True}, potential_id=card.id)
    tag = "SYNTHETIC export fixture (not a simulation result)"
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    C[range(3), range(3)] = C11
    C[3, 3] = C[4, 4] = C[5, 5] = C44
    S = np.linalg.inv(C)
    m = vrh_moduli(C, S)
    ok = [CheckResult(name="symmetry", passed=True, metric=0.0, tolerance=0.5, unit=Unit.GPA)]
    method = MethodSpec(name="autodiff relaxed-ion elastic tensor (SYNTHETIC)", description=tag)
    fd = MethodSpec(name="finite_difference stress-strain (SYNTHETIC)", description=tag)
    recs = [
        make_record(name=PropertyName.ELASTIC_STIFFNESS_VOIGT, value=C, unit=Unit.GPA, candidate=cand, method=method,
                    card=card, checks=ok, frame="crystal", shape_description="voigt 6x6", notes=tag, provenance=prov),
        make_record(name=PropertyName.ELASTIC_STIFFNESS_VOIGT, value=C + 0.3 * np.eye(6), unit=Unit.GPA, candidate=cand,
                    method=fd, card=card, checks=ok, frame="crystal", shape_description="voigt 6x6", notes=tag,
                    provenance=prov),
        make_record(name=PropertyName.ELASTIC_COMPLIANCE_VOIGT, value=S, unit=Unit.PER_GPA, candidate=cand,
                    method=method, card=card, checks=ok, frame="crystal", notes=tag, provenance=prov),
        make_record(name=PropertyName.BULK_MODULUS_HILL, value=m["BH"], unit=Unit.GPA, candidate=cand, method=method,
                    card=card, checks=ok, notes=tag, provenance=prov),
        make_record(name=PropertyName.SHEAR_MODULUS_HILL, value=m["GH"], unit=Unit.GPA, candidate=cand, method=method,
                    card=card, checks=ok, notes=tag, provenance=prov),
        make_record(name=PropertyName.YOUNGS_MODULUS_HILL, value=m["EH"], unit=Unit.GPA, candidate=cand, method=method,
                    card=card, checks=ok, notes=tag, provenance=prov),
        make_record(name=PropertyName.POISSON_RATIO_HILL, value=m["nuH"], unit=Unit.DIMENSIONLESS, candidate=cand,
                    method=method, card=card, checks=ok, notes=tag, provenance=prov),
    ]
    a0 = 4.32106  # Tersoff-1989 relaxed lattice parameter (foundation-verified)
    rho = 4 * (28.0855 + 12.011) / a0**3 * 1.66053906660
    recs.append(make_record(name=PropertyName.DENSITY, value=rho, unit=Unit.G_PER_CM3, candidate=cand,
                            method=MethodSpec(name="relaxed cell density (SYNTHETIC)", description=tag), card=card,
                            checks=ok, notes=tag, provenance=prov))
    if not planes:
        planes = (((1, 1, 1), "shuffle-type (1 bond/atom)", (1, -1, 0), (1, 1, -2), W, l_uber),
                  ((1, 1, 0), "nonpolar", (0, 0, 1), (1, -1, 0), 1.15 * W, 0.9 * l_uber))
    laws = []
    x = np.linspace(0.0, max_open, n_open)
    for plane, term, xd, yd, Wp, lp in planes:
        iface = InterfaceSpec(kind=InterfaceKind.CLEAVAGE_PLANE, phase_a="3C-SiC",
                              orientation_a=Orientation(plane_miller=plane, x_direction=xd, y_direction=yd),
                              termination=Termination(label=term))
        e = uber_energy(x, Wp, lp)
        t = uber_traction(x, Wp, lp) / GPA_ANGSTROM_TO_J_PER_M2
        ip = int(np.argmax(t))
        law = TractionSeparationLaw(
            mode="normal", interface=iface, protocol_mode="rigid",
            curve=SampledCurve(x_name="opening", x=q(x, Unit.ANGSTROM), y_name="normal traction", y=q(t, Unit.GPA)),
            energy_per_area=q(e, Unit.J_PER_M2), peak_traction=q(float(t[ip]), Unit.GPA),
            opening_at_peak=q(float(x[ip]), Unit.ANGSTROM), work_of_separation=q(float(e[-1]), Unit.J_PER_M2),
            integrated_work=q(float(np.trapezoid(t, x) * GPA_ANGSTROM_TO_J_PER_M2), Unit.J_PER_M2),
            assumptions=[tag, "rigid opening, in-plane lattice fixed at bulk"],
            checks=[CheckResult(name="integration_identity", passed=True, metric=0.0, tolerance=1e-3)],
        )
        laws.append(law)
        cm = MethodSpec(name="rigid cleavage opening (SYNTHETIC UBER) " + protocol_mode_tag("rigid"), description=tag)
        for pn, val, unit in ((PropertyName.WORK_OF_SEPARATION, float(e[-1]), Unit.J_PER_M2),
                              (PropertyName.PEAK_NORMAL_TRACTION, float(t[ip]), Unit.GPA),
                              (PropertyName.OPENING_AT_PEAK_TRACTION, float(x[ip]), Unit.ANGSTROM)):
            recs.append(make_record(name=pn, value=val, unit=unit, candidate=cand, method=cm, card=card,
                                    interface=iface, orientation=iface.orientation_a, checks=ok, notes=tag,
                                    provenance=prov))
    return dict(candidate=cand, cards=[card], records=recs, laws=laws)
