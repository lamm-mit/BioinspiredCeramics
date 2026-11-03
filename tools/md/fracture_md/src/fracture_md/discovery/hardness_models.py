"""Hardness models: intrinsic (size-free) properties -> measured hardness, calibrated and leave-one-out tested.

Why this module exists.  An MD Berkovich hardness at 1-3 nm in a perfect crystal at 1e9 /s cannot reproduce a
laboratory hardness (docs/INDENTATION.md).  Measured hardness enters the project only through models that map
intrinsic properties (elastic constants, ideal shear strength) to hardness, whose free constants are CALIBRATED on
known materials and whose error is measured by LEAVE-ONE-OUT prediction of materials left out of the fit
(data/hardness_calibration/, docs/HARDNESS_CALIBRATION.md).

Layout (the core is harness-agnostic: numpy + scipy only; the PropertyRecord wrapper at the end imports the
harness lazily):

1. Elastic descriptors: 6x6 stiffness from independent constants, Voigt-Reuss-Hill B, G, E, nu, the Pugh ratio
   k = G/B, and the Vlassak-Nix indentation modulus M of a surface (Stroh half-space compliance).
2. Johnson's expanding-cavity relation (mean pressure of a cone in an elastic-perfectly plastic solid).
3. Hardness models (:data:`MODELS`): Teter, Chen-Niu, Tian (literature constants and recalibrated), an ideal-shear
   model H = c tau_ideal, and Johnson's cavity model with a yield stress estimated from tau_ideal or from G.
4. Calibration (weighted least squares on relative residuals), leave-one-out cross-validation, metrics (MAE, MAPE,
   Spearman rank correlation, max error) and 90 % prediction intervals.
5. Dataset adapter for data/hardness_calibration/materials.json (+ the DFT and MACE elastic snapshots).
6. PropertyRecord wrapper (``vickers_hardness_estimate``, DERIVED_MODEL, status MODEL_ESTIMATE or OUT_OF_DOMAIN).

Conventions.  Every hardness here is on the Vickers scale (load / SLOPED contact area, GPa) unless a name says
H_IT.  A projected-area hardness (ISO 14577 H_IT, Oliver-Pharr, a cavity-model mean pressure) is converted with
HV = 0.0945 H_IT[MPa], i.e. HV[GPa] = 0.92673 H_IT[GPa] (:data:`HV_GPA_PER_H_IT_GPA`).  Moduli and strengths are in
GPa.  Nothing computed here is a measurement.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

# ISO 14577-1: HV = 0.0945 H_IT[MPa]; HV number (kgf/mm^2) -> GPa with 9.80665e-3.
ISO14577_HV_PER_MPA = 0.0945
KGF_MM2_TO_GPA = 9.80665e-3
HV_GPA_PER_H_IT_GPA = ISO14577_HV_PER_MPA * 1000.0 * KGF_MM2_TO_GPA  # 0.926728...
# Cone equivalent to the Vickers / Berkovich pyramid (same area-depth function): semi-angle 70.3 deg, so the face is
# inclined beta = 19.7 deg to the surface.
VICKERS_EQUIVALENT_CONE_BETA_DEG = 19.7

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "hardness_calibration"
MATERIALS_FILE = DATA_DIR / "materials.json"
DFT_FILE = DATA_DIR / "dft_pbe_elastic.json"
MACE_FILE = DATA_DIR / "mace_mp0_small_elastic.json"


# ============================================================================================ 1. elasticity
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def stiffness_matrix(constants: Mapping[str, float], symmetry: str) -> np.ndarray:
    """6x6 Voigt stiffness (GPa) from the independent constants of a crystal class.

    ``symmetry``: ``cubic`` (C11, C12, C44), ``hexagonal`` (C11, C12, C13, C33, C44; C66 = (C11-C12)/2),
    ``trigonal`` (class -3m / 32 / 3m: hexagonal set plus C14, with C24 = -C14 and C56 = C14) or
    ``tetragonal`` (C11, C12, C13, C33, C44, C66).  Missing optional constants (C14) count as zero.
    """
    c = {k.upper(): float(v) for k, v in constants.items() if v is not None}
    s = symmetry.lower()
    C = np.zeros((6, 6))
    if s == "cubic":
        C[:3, :3] = c["C12"]
        np.fill_diagonal(C[:3, :3], c["C11"])
        C[3, 3] = C[4, 4] = C[5, 5] = c["C44"]
        return C
    if s in ("hexagonal", "trigonal", "tetragonal"):
        C[0, 0] = C[1, 1] = c["C11"]
        C[0, 1] = C[1, 0] = c["C12"]
        C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = c["C13"]
        C[2, 2] = c["C33"]
        C[3, 3] = C[4, 4] = c["C44"]
        C[5, 5] = c["C66"] if s == "tetragonal" else 0.5 * (c["C11"] - c["C12"])
        if s == "trigonal":
            c14 = c.get("C14", 0.0)
            C[0, 3] = C[3, 0] = c14
            C[1, 3] = C[3, 1] = -c14
            C[4, 5] = C[5, 4] = c14
        return C
    raise ValueError(f"unsupported symmetry {symmetry!r} (cubic, hexagonal, trigonal, tetragonal)")


def voigt_to_tensor(C: np.ndarray) -> np.ndarray:
    """4th-order stiffness tensor C_ijkl from a 6x6 Voigt matrix."""
    C = np.asarray(C, float)
    T = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(_VOIGT):
        for b, (k, l) in enumerate(_VOIGT):
            v = C[a, b]
            for (p, q) in ((i, j), (j, i)):
                for (r, s) in ((k, l), (l, k)):
                    T[p, q, r, s] = v
    return T


def vrh_moduli(C: np.ndarray) -> dict[str, float]:
    """Voigt, Reuss and Hill bulk and shear moduli, Hill E and nu, Pugh ratio k = G_H/B_H (GPa).

    Assumes an untextured, fully dense polycrystal of the single crystal.  ``min_eigenvalue`` > 0 is Born
    stability at zero stress.
    """
    C = np.asarray(C, float)
    C = 0.5 * (C + C.T)
    S = np.linalg.inv(C)
    BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2.0 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    GV = ((C[0, 0] + C[1, 1] + C[2, 2]) - (C[0, 1] + C[0, 2] + C[1, 2])
          + 3.0 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0
    BR = 1.0 / ((S[0, 0] + S[1, 1] + S[2, 2]) + 2.0 * (S[0, 1] + S[0, 2] + S[1, 2]))
    GR = 15.0 / (4.0 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4.0 * (S[0, 1] + S[0, 2] + S[1, 2])
                 + 3.0 * (S[3, 3] + S[4, 4] + S[5, 5]))
    B, G = 0.5 * (BV + BR), 0.5 * (GV + GR)
    E, nu = isotropic_E_nu(B, G)
    return {"B_V": float(BV), "B_R": float(BR), "G_V": float(GV), "G_R": float(GR), "B": float(B), "G": float(G),
            "E": E, "nu": nu, "k": float(G / B), "min_eigenvalue": float(np.linalg.eigvalsh(C)[0])}


def isotropic_E_nu(B: float, G: float) -> tuple[float, float]:
    """Young's modulus and Poisson ratio of an isotropic solid from B and G."""
    E = 9.0 * B * G / (3.0 * B + G)
    nu = (3.0 * B - 2.0 * G) / (2.0 * (3.0 * B + G))
    return float(E), float(nu)


def _rotation_to_normal(normal: Sequence[float]) -> np.ndarray:
    """Rows: lab axes (x', y', z' = normal) in crystal Cartesian coordinates."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    trial = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = trial - np.dot(trial, n) * n
    x /= np.linalg.norm(x)
    return np.array([x, np.cross(n, x), n])


def _stroh_g33(Ct: np.ndarray, phi: float, check: bool = True) -> float:
    """Surface compliance G33(phi) of an anisotropic half-space z < 0 with free surface z = 0: the normal surface
    displacement of a normal traction exp(i k m.x) is G33 / |k| (isotropic: G33 = 2 (1 - nu^2) / E)."""
    m = np.array([math.cos(phi), math.sin(phi), 0.0])
    n = np.array([0.0, 0.0, 1.0])
    Q = np.einsum("j,ijkl,l->ik", m, Ct, m)
    R = np.einsum("j,ijkl,l->ik", m, Ct, n)
    T = np.einsum("j,ijkl,l->ik", n, Ct, n)
    Ti = np.linalg.inv(T)
    N = np.zeros((6, 6))
    N[:3, 3:] = np.eye(3)
    N[3:, :3] = -Ti @ Q
    N[3:, 3:] = -Ti @ (R + R.T)
    w, V = np.linalg.eig(N)
    sel = np.argsort(w.imag)[:3]  # decaying into z < 0
    A = V[:3, sel]
    Bm = np.stack([(R.T + w[s] * T) @ V[:3, s] for s in sel], axis=1)
    if check and np.linalg.cond(V) > 1e8:  # degenerate Stroh eigenvalues (exactly isotropic): eigenvectors unreliable
        return float("nan")
    return float((-1j * A @ np.linalg.inv(Bm))[2, 2].real)


def vlassak_nix_modulus(C: np.ndarray, normal: Sequence[float] = (0.0, 0.0, 1.0), n_phi: int = 180) -> float:
    """Vlassak-Nix indentation modulus M = 2 / <G33(phi)>_phi (GPa) of the surface with unit ``normal``.

    ``C`` is the 6x6 Voigt stiffness in the crystal Cartesian frame (cubic: cube axes; hexagonal/trigonal in the
    IEEE frame: z || c).  For an isotropic solid M = E / (1 - nu^2).  Same definition as
    ``physics.contact_box.LayerMaterial.M_VN`` (Vlassak & Nix, J. Mech. Phys. Solids 42, 1223 (1994),
    doi:10.1016/0022-5096(94)90033-7); G33(phi) has period pi, so the trapezoid rule over [0, pi) converges
    exponentially.
    """
    Ct = voigt_to_tensor(C)
    Rm = _rotation_to_normal(normal)
    if not np.allclose(Rm, np.eye(3)):
        Ct = np.einsum("ia,jb,kc,ld,abcd->ijkl", Rm, Rm, Rm, Rm, Ct)
    phis = np.linspace(0.0, math.pi, n_phi, endpoint=False)
    g = np.array([_stroh_g33(Ct, p) for p in phis])
    if not np.all(np.isfinite(g)):
        # Degenerate (isotropic) case: break the degeneracy by a 1e-9 relative shear perturbation (M changes by
        # ~1e-10 relative; M is continuous in C).
        Cp = np.asarray(C, float).copy()
        Cp[3, 3] *= 1.0 + 1e-9
        Ct = voigt_to_tensor(Cp)
        if not np.allclose(Rm, np.eye(3)):
            Ct = np.einsum("ia,jb,kc,ld,abcd->ijkl", Rm, Rm, Rm, Rm, Ct)
        g = np.array([_stroh_g33(Ct, p, check=False) for p in phis])
    return float(2.0 / np.mean(g))


@dataclass(frozen=True)
class ElasticDescriptors:
    """Isotropic (Hill) moduli of one material plus where they came from.  GPa; k and nu dimensionless."""

    B: float
    G: float
    E: float
    nu: float
    k: float
    basis: str  # e.g. "single-crystal C_ij, McSkimin1972" or "polycrystal B, G, Nino2010"
    source_kind: str  # "exp_single_crystal", "exp_polycrystal", "dft_pbe", "mace_mp0_small", ...
    confidence: str = "high"
    M: Optional[float] = None  # Vlassak-Nix modulus of `M_surface`
    M_surface: Optional[str] = None
    C_voigt: Optional[tuple[tuple[float, ...], ...]] = None
    notes: str = ""

    @classmethod
    def from_stiffness(cls, C: np.ndarray, *, basis: str, source_kind: str, confidence: str = "high",
                       normal: Optional[Sequence[float]] = (0.0, 0.0, 1.0), surface_label: str = "",
                       notes: str = "") -> "ElasticDescriptors":
        C = np.asarray(C, float)
        m = vrh_moduli(C)
        M = vlassak_nix_modulus(C, normal) if normal is not None else None
        label = surface_label or (None if normal is None else str(tuple(normal)))
        return cls(B=m["B"], G=m["G"], E=m["E"], nu=m["nu"], k=m["k"], basis=basis, source_kind=source_kind,
                   confidence=confidence, M=M, M_surface=label,
                   C_voigt=tuple(tuple(float(x) for x in row) for row in C), notes=notes)

    @classmethod
    def from_moduli(cls, B: float, G: float, *, basis: str, source_kind: str, confidence: str = "high",
                    notes: str = "") -> "ElasticDescriptors":
        E, nu = isotropic_E_nu(B, G)
        return cls(B=float(B), G=float(G), E=E, nu=nu, k=float(G) / float(B), basis=basis, source_kind=source_kind,
                   confidence=confidence, notes=notes)

    def as_dict(self) -> dict[str, Any]:
        return {"B_GPa": self.B, "G_GPa": self.G, "E_GPa": self.E, "nu": self.nu, "k_pugh": self.k,
                "M_VN_GPa": self.M, "M_surface": self.M_surface, "basis": self.basis, "source_kind": self.source_kind,
                "confidence": self.confidence, "notes": self.notes}


# ============================================================================================ 2. Johnson
def johnson_cavity_mean_pressure(E: Any, nu: Any, Y: Any, beta_deg: float = VICKERS_EQUIVALENT_CONE_BETA_DEG,
                                 cap_fully_plastic: bool = True) -> dict[str, np.ndarray]:
    """Mean contact pressure p_m (projected area, GPa) of a cone pressed into an elastic-perfectly plastic solid.

    Expanding-cavity model (Johnson, J. Mech. Phys. Solids 18, 115 (1970); Contact Mechanics, CUP 1985, Sec. 6.5):
    a hemispherical core under hydrostatic pressure p_m, a plastic shell a < r < c and an elastic exterior; the
    volume displaced by the cone is taken up by the radial expansion of the shell, which gives

        (c/a)^3 = [E tan(beta) / Y + 4 (1 - 2 nu)] / [6 (1 - nu)],     p_m / Y = (2/3) [1 + ln (c/a)^3].

    For nu = 1/2 this is the commonly quoted p_m/Y = (2/3)[1 + ln(E tan(beta) / (3 Y))].  The model holds only in
    the elastic-plastic range: when (c/a)^3 <= 1 (no plastic shell) the contact is taken as elastic, p_m = E*
    tan(beta) / 2 with E* = E / (1 - nu^2) (Love's cone); with ``cap_fully_plastic`` p_m is capped at the
    rigid-plastic constraint 3 Y (Tabor).  Returns p_m, the regime parameter x = E tan(beta) / Y, (c/a)^3 and a
    regime label per entry.  Other textbook variants differ by a constant inside the bracket (e.g. Sakai 2020,
    "Indentation Contact Mechanics", eq. 5.10a has 2 + ln(...)); a fitted yield-stress factor absorbs it.
    """
    E, nu, Y = (np.asarray(v, float) for v in (E, nu, Y))
    tb = math.tan(math.radians(beta_deg))
    x = E * tb / Y
    ratio = (x + 4.0 * (1.0 - 2.0 * nu)) / (6.0 * (1.0 - nu))
    with np.errstate(divide="ignore", invalid="ignore"):
        pm_ecm = Y * (2.0 / 3.0) * (1.0 + np.log(np.where(ratio > 0, ratio, np.nan)))
    pm_el = 0.5 * E / (1.0 - nu ** 2) * tb
    elastic = ratio <= 1.0
    pm = np.where(elastic, pm_el, pm_ecm)
    plastic_cap = np.zeros_like(pm, dtype=bool)
    if cap_fully_plastic:
        plastic_cap = pm > 3.0 * Y
        pm = np.where(plastic_cap, 3.0 * Y, pm)
    regime = np.where(elastic, "elastic", np.where(plastic_cap, "fully_plastic_cap", "elastic_plastic"))
    return {"p_m": pm, "x": x, "c_over_a_cubed": ratio, "regime": regime}


def yield_from_ideal_shear(tau_ideal: Any, alpha: float = 1.0, criterion: str = "von_mises") -> np.ndarray:
    """Uniaxial yield-stress estimate Y = alpha * f * tau_ideal, f = sqrt(3) (von Mises) or 2 (Tresca).

    ASSUMPTION: the flow stress under the indenter is a fixed fraction alpha of the stress at which the perfect
    lattice becomes unstable in shear, the same alpha for every material (alpha = 1: yield at the ideal strength,
    an upper bound).  Neither defects nor thermal activation nor pressure enter except through alpha.
    """
    f = math.sqrt(3.0) if criterion == "von_mises" else 2.0
    return float(alpha) * f * np.asarray(tau_ideal, float)


# ============================================================================================ 3. models
def _arr(X: Mapping[str, Any], key: str) -> np.ndarray:
    return np.asarray(X[key], float)


@dataclass(frozen=True)
class HardnessModel:
    """One hardness model H = f(descriptors; params), H on the Vickers scale in GPa.

    ``form``: ``fixed`` (no free constant: literature or physical constants), ``linear`` (H = F(X) @ params, fitted
    by weighted linear least squares on relative residuals) or ``scalar`` (one nonlinear constant, fitted by a
    bounded 1-D search on log(param)).
    """

    name: str
    label: str
    equation: str
    citation: str
    requires: tuple[str, ...]
    param_names: tuple[str, ...]
    params: tuple[float, ...]  # literature / physical / starting constants
    form: str  # "fixed" | "linear" | "scalar"
    features: Optional[Callable[[Mapping[str, Any]], np.ndarray]] = None
    predict_fn: Optional[Callable[[Mapping[str, Any], Sequence[float]], np.ndarray]] = None
    bounds: tuple[float, float] = (1e-3, 1e3)
    assumptions: tuple[str, ...] = ()
    family: str = ""

    @property
    def n_free(self) -> int:
        return 0 if self.form == "fixed" else len(self.param_names)

    def predict(self, X: Mapping[str, Any], params: Optional[Sequence[float]] = None) -> np.ndarray:
        p = tuple(self.params if params is None else params)
        if self.features is not None:
            return self.features(X) @ np.asarray(p, float)
        return np.asarray(self.predict_fn(X, p), float)


def _chen_niu_fixed(X, p):
    G, k = _arr(X, "G"), _arr(X, "k")
    return p[0] * (k * k * G) ** p[2] + p[1]


def _tian_fixed(X, p):
    G, k = _arr(X, "G"), _arr(X, "k")
    return p[0] * k ** p[1] * G ** p[2]


def _johnson_tau(X, p):
    Y = yield_from_ideal_shear(_arr(X, "tau"), p[0])
    return HV_GPA_PER_H_IT_GPA * johnson_cavity_mean_pressure(_arr(X, "E"), _arr(X, "nu"), Y)["p_m"]


def _johnson_G(X, p):
    Y = p[0] * _arr(X, "G")
    return HV_GPA_PER_H_IT_GPA * johnson_cavity_mean_pressure(_arr(X, "E"), _arr(X, "nu"), Y)["p_m"]


_TETER = "D.M. Teter, MRS Bull. 23(1), 22-27 (1998), doi:10.1557/S0883769400031420"
_CHEN = "X.-Q. Chen, H. Niu, D. Li, Y. Li, Intermetallics 19, 1275-1281 (2011), doi:10.1016/j.intermet.2011.03.026"
_TIAN = "Y. Tian, B. Xu, Z. Zhao, Int. J. Refract. Met. Hard Mater. 33, 93-106 (2012), doi:10.1016/j.ijrmhm.2012.02.021"
_JOHNSON = ("K.L. Johnson, J. Mech. Phys. Solids 18, 115-126 (1970), doi:10.1016/0022-5096(70)90029-3; "
            "Contact Mechanics (CUP 1985) Sec. 6.5")
_ISO = "HV = 0.0945 H_IT[MPa] (ISO 14577-1): cavity-model mean pressure (projected area) -> Vickers scale"
_ELASTIC_ASSUMPTIONS = (
    "empirical correlation of hardness with isotropic (Hill) moduli of a dense, untextured polycrystal",
    "ignores dislocation mobility, phase transformations, microstructure, cracking and the indentation size effect",
)

MODELS: dict[str, HardnessModel] = {m.name: m for m in (
    HardnessModel("teter_lit", "Teter (literature)", "H = 0.151 G", _TETER, ("G",), ("c",), (0.151,), "fixed",
                  features=lambda X: _arr(X, "G")[:, None], assumptions=_ELASTIC_ASSUMPTIONS, family="teter"),
    HardnessModel("teter", "Teter (recalibrated)", "H = c G", _TETER, ("G",), ("c",), (0.151,), "linear",
                  features=lambda X: _arr(X, "G")[:, None], assumptions=_ELASTIC_ASSUMPTIONS, family="teter"),
    HardnessModel("chen_niu_lit", "Chen-Niu (literature)", "H = 2 (k^2 G)^0.585 - 3", _CHEN, ("G", "k"),
                  ("a", "b", "m"), (2.0, -3.0, 0.585), "fixed", predict_fn=_chen_niu_fixed,
                  assumptions=_ELASTIC_ASSUMPTIONS, family="chen_niu"),
    HardnessModel("chen_niu", "Chen-Niu (recalibrated)", "H = a (k^2 G)^0.585 + b", _CHEN, ("G", "k"), ("a", "b"),
                  (2.0, -3.0), "linear",
                  features=lambda X: np.column_stack([(_arr(X, "k") ** 2 * _arr(X, "G")) ** 0.585,
                                                      np.ones(len(_arr(X, "G")))]),
                  assumptions=_ELASTIC_ASSUMPTIONS + ("exponent 0.585 kept at the published value",),
                  family="chen_niu"),
    HardnessModel("tian_lit", "Tian (literature)", "H = 0.92 k^1.137 G^0.708", _TIAN, ("G", "k"), ("A", "p", "q"),
                  (0.92, 1.137, 0.708), "fixed", predict_fn=_tian_fixed, assumptions=_ELASTIC_ASSUMPTIONS,
                  family="tian"),
    HardnessModel("tian", "Tian (recalibrated)", "H = A k^1.137 G^0.708", _TIAN, ("G", "k"), ("A",), (0.92,),
                  "linear", features=lambda X: (_arr(X, "k") ** 1.137 * _arr(X, "G") ** 0.708)[:, None],
                  assumptions=_ELASTIC_ASSUMPTIONS + ("exponents kept at the published values",), family="tian"),
    HardnessModel("ideal_shear", "Ideal shear (calibrated)", "H = c tau_ideal",
                  "this work (ideal shear strengths from DFT literature, data/hardness_calibration/materials.json)",
                  ("tau",), ("c",), (1.0,), "linear", features=lambda X: _arr(X, "tau")[:, None],
                  assumptions=("hardness proportional to the lowest DFT ideal shear strength (the stress at which "
                               "the perfect lattice becomes unstable in shear), one constant for all materials",
                               "tau_ideal is the lowest published value over the slip systems computed; functionals "
                               "(LDA, PW91, PBE) and the systems compared differ between sources"),
                  family="ideal_shear"),
    HardnessModel("johnson_tau_ideal", "Johnson cavity, Y = sqrt(3) tau_ideal",
                  "HV = 0.927 p_m(E, nu, Y),  Y = sqrt(3) tau_ideal", _JOHNSON + "; " + _ISO, ("tau", "E", "nu"),
                  ("alpha",), (1.0,), "fixed", predict_fn=_johnson_tau,
                  assumptions=("elastic-perfectly plastic isotropic solid (Hill E, nu), von Mises yield at the ideal "
                               "shear strength (alpha = 1, an upper bound), cone of the Vickers-equivalent angle",),
                  family="johnson_tau"),
    HardnessModel("johnson_tau", "Johnson cavity, Y = alpha sqrt(3) tau_ideal",
                  "HV = 0.927 p_m(E, nu, Y),  Y = alpha sqrt(3) tau_ideal", _JOHNSON + "; " + _ISO,
                  ("tau", "E", "nu"), ("alpha",), (0.5,), "scalar", predict_fn=_johnson_tau, bounds=(0.02, 2.0),
                  assumptions=("elastic-perfectly plastic isotropic solid (Hill E, nu), no work hardening",
                               "yield stress Y = alpha sqrt(3) tau_ideal (von Mises), one alpha for all materials",
                               "cone of the Vickers-equivalent semi-angle 70.3 deg (beta = 19.7 deg)",
                               "cavity-model mean pressure (projected area) converted with ISO 14577"),
                  family="johnson_tau"),
    HardnessModel("johnson_G", "Johnson cavity, Y = alpha G", "HV = 0.927 p_m(E, nu, Y),  Y = alpha G",
                  _JOHNSON + "; " + _ISO, ("G", "E", "nu"), ("alpha",), (0.1,), "scalar", predict_fn=_johnson_G,
                  bounds=(1e-3, 2.0),
                  assumptions=("elastic-perfectly plastic isotropic solid (Hill E, nu), no work hardening",
                               "yield stress Y = alpha G, one alpha for all materials",
                               "cone of the Vickers-equivalent semi-angle 70.3 deg (beta = 19.7 deg)",
                               "cavity-model mean pressure (projected area) converted with ISO 14577"),
                  family="johnson_G"),
)}


# ============================================================================================ 4. calibration
def fit_model(model: HardnessModel, X: Mapping[str, Any], y: Sequence[float],
              w: Optional[Sequence[float]] = None) -> tuple[float, ...]:
    """Free constants minimising sum_i w_i ((H_i,pred - y_i) / y_i)^2 (relative residuals; w = 1 / sigma_rel^2).

    Relative residuals treat an 8 GPa and a 90 GPa material alike.  A ``fixed`` model returns its constants.
    """
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    if model.form == "fixed":
        return tuple(model.params)
    if model.form == "linear":
        F = model.features(X) / y[:, None]
        sw = np.sqrt(w)
        sol, *_ = np.linalg.lstsq(F * sw[:, None], sw, rcond=None)
        return tuple(float(v) for v in sol)
    if model.form == "scalar":
        from scipy.optimize import minimize_scalar

        def obj(lp: float) -> float:
            pred = model.predict(X, (math.exp(lp),))
            r = (pred - y) / y
            return float(np.sum(w * np.where(np.isfinite(r), r, 1e6) ** 2))

        lo, hi = (math.log(b) for b in model.bounds)
        grid = np.linspace(lo, hi, 121)
        vals = [obj(g) for g in grid]
        i = int(np.argmin(vals))
        a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        res = minimize_scalar(obj, bounds=(a, b), method="bounded", options={"xatol": 1e-10})
        return (float(math.exp(res.x)),)
    raise ValueError(f"unknown model form {model.form!r}")


def _rank(a: Sequence[float]) -> np.ndarray:
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    rx, ry = _rank(x), _rank(y)
    if len(rx) < 2 or np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def prediction_interval(log_residuals: Sequence[float], n_params: int = 0, level: float = 0.90) -> dict[str, Any]:
    """Multiplicative prediction interval from out-of-sample residuals q_i = ln(y_i / yhat_i).

    * ``parametric``: yhat * exp(qbar +/- t_{(1+level)/2, n-1} s_q sqrt(1 + 1/n)) (normal log residuals);
      ``factor`` = exp(t s_q sqrt(1 + 1/n)) is the symmetric half-width around the bias-corrected centre.
    * ``conformal`` (jackknife): yhat * exp(+/- |q|_(j)), j = ceil((n + 1) level); undefined (None) when j > n,
      i.e. for fewer than 9 materials at 90 %.

    The residuals must be out-of-sample (leave-one-out or a fixed literature model); `n_params` is recorded only.
    """
    from scipy.stats import t as student_t

    q = np.asarray(log_residuals, float)
    q = q[np.isfinite(q)]
    n = len(q)
    out: dict[str, Any] = {"level": level, "n": n, "n_params": n_params}
    if n < 3:
        out.update(parametric=None, conformal=None, note="fewer than 3 out-of-sample residuals")
        return out
    qbar, s = float(np.mean(q)), float(np.std(q, ddof=1))
    tq = float(student_t.ppf(0.5 + level / 2.0, n - 1))
    half = tq * s * math.sqrt(1.0 + 1.0 / n)
    out["parametric"] = {"bias_log": qbar, "s_log": s, "t": tq, "factor": math.exp(half),
                         "lower_ratio": math.exp(qbar - half), "upper_ratio": math.exp(qbar + half)}
    j = math.ceil((n + 1) * level)
    if j <= n:
        qa = float(np.sort(np.abs(q))[j - 1])
        out["conformal"] = {"rank": j, "abs_log": qa, "factor": math.exp(qa)}
    else:
        out["conformal"] = None
    return out


def regression_metrics(y: Sequence[float], yhat: Sequence[float], ids: Optional[Sequence[str]] = None,
                       n_params: int = 0, level: float = 0.90) -> dict[str, Any]:
    """MAE, RMSE, MAPE, mean signed % error, max |error| (and which material), Spearman rho, prediction interval."""
    y, yh = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(y) & np.isfinite(yh)
    ids = list(ids) if ids is not None else [str(i) for i in range(len(y))]
    ids_ok = [i for i, m in zip(ids, ok) if m]
    y, yh = y[ok], yh[ok]
    n = len(y)
    if n == 0:
        return {"n": 0}
    err = yh - y
    rel = err / y
    imax = int(np.argmax(np.abs(err)))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.log(y / yh)
    return {
        "n": n,
        "MAE_GPa": float(np.mean(np.abs(err))),
        "RMSE_GPa": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_pct": float(100.0 * np.mean(np.abs(rel))),
        "bias_pct": float(100.0 * np.mean(rel)),
        "max_abs_error_GPa": float(abs(err[imax])),
        "max_abs_error_material": ids_ok[imax],
        "max_abs_pct_error": float(100.0 * np.max(np.abs(rel))),
        "max_abs_pct_error_material": ids_ok[int(np.argmax(np.abs(rel)))],
        "spearman_rho": spearman_rho(y, yh),
        "n_nonpositive_predictions": int(np.sum(yh <= 0)),
        "prediction_interval": prediction_interval(q, n_params=n_params, level=level),
    }


@dataclass
class CalibrationResult:
    model: str
    ids: list[str]
    y: np.ndarray
    fit_mask: np.ndarray  # materials used in the fits (others are predicted by the full fit only)
    params_full: tuple[float, ...]
    yhat_full: np.ndarray  # in-sample (fit-set) / out-of-sample (excluded) predictions of the full fit
    yhat_loo: np.ndarray  # leave-one-out predictions (fit-set members) or full-fit predictions (excluded)
    params_loo: list[tuple[float, ...]]
    metrics_loo: dict[str, Any]
    metrics_loo_fit_set: dict[str, Any]
    metrics_in_sample: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "params_full": list(self.params_full),
            "param_names": list(MODELS[self.model].param_names) if self.model in MODELS else None,
            "param_range_over_loo_fits": ([[float(min(p[i] for p in self.params_loo)),
                                            float(max(p[i] for p in self.params_loo))]
                                           for i in range(len(self.params_full))] if self.params_loo else None),
            "materials": [{"id": i, "target_GPa": float(t), "in_fit_set": bool(f), "pred_full_GPa": float(pf),
                           "pred_loo_GPa": float(pl), "rel_err_loo_pct": float(100.0 * (pl - t) / t)}
                          for i, t, f, pf, pl in zip(self.ids, self.y, self.fit_mask, self.yhat_full, self.yhat_loo)],
            "metrics_loo": self.metrics_loo,
            "metrics_loo_fit_set": self.metrics_loo_fit_set,
            "metrics_in_sample": self.metrics_in_sample,
            **self.extra,
        }


def leave_one_out(model: HardnessModel, X: Mapping[str, Any], y: Sequence[float], ids: Sequence[str],
                  w: Optional[Sequence[float]] = None, fit_mask: Optional[Sequence[bool]] = None,
                  level: float = 0.90) -> CalibrationResult:
    """Calibrate on the fit set and predict each fit-set material from a fit WITHOUT it (leave-one-out).

    Materials outside ``fit_mask`` are never fitted on; they are predicted by the full fit (also out-of-sample).
    A ``fixed`` model has nothing to fit: its predictions are its leave-one-out predictions (but its published
    constants were fitted by its authors on data that may include these materials).
    Metrics: ``metrics_loo`` over all materials (out-of-sample predictions), ``metrics_loo_fit_set`` over the fit
    set only, ``metrics_in_sample`` of the full fit on the fit set.
    """
    y = np.asarray(y, float)
    n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, float)
    fm = np.ones(n, bool) if fit_mask is None else np.asarray(fit_mask, bool)
    Xa = {k: np.asarray(v, float) for k, v in X.items()}

    def sub(mask):
        return {k: v[mask] for k, v in Xa.items()}

    p_full = fit_model(model, sub(fm), y[fm], w[fm])
    yhat_full = model.predict(Xa, p_full)
    yhat_loo = yhat_full.copy()
    params_loo: list[tuple[float, ...]] = []
    if model.form != "fixed":
        for i in np.where(fm)[0]:
            m = fm.copy()
            m[i] = False
            p = fit_model(model, sub(m), y[m], w[m])
            params_loo.append(p)
            yhat_loo[i] = model.predict(sub(np.arange(n) == i), p)[0]
    ids = list(ids)
    k = model.n_free
    return CalibrationResult(
        model=model.name, ids=ids, y=y, fit_mask=fm, params_full=p_full, yhat_full=yhat_full, yhat_loo=yhat_loo,
        params_loo=params_loo,
        metrics_loo=regression_metrics(y, yhat_loo, ids, k, level),
        metrics_loo_fit_set=regression_metrics(y[fm], yhat_loo[fm], [i for i, f in zip(ids, fm) if f], k, level),
        metrics_in_sample=regression_metrics(y[fm], yhat_full[fm], [i for i, f in zip(ids, fm) if f], k, level),
    )


def nested_leave_one_out(candidates: Sequence[str], assemble_fn: Callable[[str, Sequence[str]], Any],
                         ids: Sequence[str], rank_fn: Callable[[list], Sequence[str]],
                         level: float = 0.90) -> dict[str, Any]:
    """Leave-one-out with the model SELECTION repeated inside every fold (nested cross-validation).

    Picking the best of several calibrated models by their own leave-one-out error makes that error optimistic:
    the choice has seen every held-out material.  Here each material is held out in turn.  On the others, every
    candidate is calibrated and scored by plain leave-one-out; ``rank_fn`` ranks the candidates from those scores
    alone (a list of (model name, metrics_loo) -> model names, best first); the top model is fitted on the others
    and predicts the held-out material.  When the held-out material lacks a descriptor of that model (e.g. no ideal
    shear strength) the next ranked candidate it has is used, flagged ``fallback``.

    ``assemble_fn(model_name, ids)`` returns the CalibrationSet of those materials for that model (materials without
    the model's descriptors dropped).  The metrics of the returned predictions are the honest error of the whole
    procedure "calibrate every candidate, pick one by the rule, predict a new material"; ``models_used`` counts how
    often each model was picked.
    """
    rows = []
    for i in ids:
        train = [j for j in ids if j != i]
        scores = []
        for mn in candidates:
            cs = assemble_fn(mn, train)
            if len(cs.ids) < 4:
                continue
            r = leave_one_out(MODELS[mn], cs.X, cs.y, cs.ids, cs.w, level=level)
            scores.append((mn, r.metrics_loo))
        ranked = list(rank_fn(scores))
        chosen, fallback, pred, yi = None, False, None, None
        for k, mn in enumerate(ranked):
            ci = assemble_fn(mn, [i])
            if not ci.ids:
                continue
            cs = assemble_fn(mn, train)
            p = fit_model(MODELS[mn], cs.X, cs.y, cs.w)
            pred = float(MODELS[mn].predict(ci.X, p)[0])
            yi = float(ci.y[0])
            chosen, fallback = mn, k > 0
            break
        rows.append(dict(id=i, selected_on_training=ranked[0] if ranked else None, model_used=chosen,
                         fallback=fallback, target_GPa=yi, pred_GPa=pred,
                         rel_err_pct=None if pred is None else 100.0 * (pred - yi) / yi))
    ok = [r for r in rows if r["pred_GPa"] is not None]
    met = regression_metrics([r["target_GPa"] for r in ok], [r["pred_GPa"] for r in ok], [r["id"] for r in ok],
                             level=level)
    counts: dict[str, int] = dict()
    for r in ok:
        counts[r["model_used"]] = counts.get(r["model_used"], 0) + 1
    return dict(materials=rows, metrics=met, models_used=counts, n_not_predicted=len(rows) - len(ok))


# ============================================================================================ 5. dataset
CONFIDENCE_SIGMA_FLOOR = {"high": 0.05, "medium": 0.10, "low": 0.20}
_CONF_ORDER = {"high": 2, "medium": 1, "low": 0}
# Test protocol behind a calibration target (materials.json calibration_target.protocol):
#   plateau                         Vickers at or above the load where HV stops changing, or a load-independent fit
#   low_load                        Vickers at loads inside the indentation size effect (e.g. 0.5-2 N)
#   compromise                      the compiler's compromise between a low-load value and cited values
#   load_not_stated                 a Vickers microhardness whose load the source does not state
#   review                          a review or handbook value whose load and specimen are not stated
#   converted_from_nanoindentation  a Berkovich H_IT converted with ISO 14577 (no Vickers value found)
# Only 'plateau' is a load-plateau value in the sense of docs/HARDNESS_CALIBRATION.md; the others are weaker and get
# a 20 % floor on sigma_rel in the 'sigma_protocol' weighting.
PROTOCOL_SIGMA_FLOOR = dict(plateau=0.0, low_load=0.20, compromise=0.20, load_not_stated=0.20, review=0.20,
                            converted_from_nanoindentation=0.20)
TARGET_PROTOCOLS = tuple(PROTOCOL_SIGMA_FLOOR)


def target_sigma_rel(value: float, lo: Optional[float], hi: Optional[float], confidence: str) -> float:
    """Relative standard uncertainty of a calibration target: max(half the record range / value, floor(confidence))
    with floors 5 % (high), 10 % (medium), 20 % (low).  The weight of the target is 1 / sigma_rel^2."""
    half = 0.0 if lo is None or hi is None else 0.5 * (float(hi) - float(lo)) / float(value)
    return float(max(half, CONFIDENCE_SIGMA_FLOOR.get(confidence, 0.20)))


@dataclass
class Target:
    value: float  # Vickers scale, GPa
    lo: Optional[float]
    hi: Optional[float]
    confidence: str
    basis: str
    kind: str  # "vickers_plateau", "vickers_compilation", "nano_hit_as_hv"
    raw_value: Optional[float] = None  # e.g. the H_IT before the ISO conversion
    protocol: Optional[str] = None  # test protocol behind the target (TARGET_PROTOCOLS); None = not labelled

    @property
    def sigma_rel(self) -> float:
        return target_sigma_rel(self.value, self.lo, self.hi, self.confidence)

    @property
    def sigma_rel_protocol(self) -> float:
        """sigma_rel with a protocol floor (PROTOCOL_SIGMA_FLOOR): a target that is not a load-plateau value gets
        at least 20 %, whatever the spread of its records."""
        floor = 0.0 if self.protocol is None else PROTOCOL_SIGMA_FLOOR.get(self.protocol, 0.20)
        return float(max(self.sigma_rel, floor))

    def as_dict(self) -> dict[str, Any]:
        return dict(value_GPa=self.value, range_GPa=[self.lo, self.hi], confidence=self.confidence,
                    sigma_rel=self.sigma_rel, sigma_rel_protocol=self.sigma_rel_protocol, basis=self.basis,
                    kind=self.kind, raw_value_GPa=self.raw_value, protocol=self.protocol)


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_materials(path: Path | str = MATERIALS_FILE) -> dict[str, dict[str, Any]]:
    """Material entries of the stage-1 dataset keyed by id."""
    return {m["id"]: m for m in load_json(path)["materials"]}


def vickers_plateau_target(m: Mapping[str, Any]) -> Target:
    t = m["calibration_target"]
    lo, hi = t.get("range_GPa") or (None, None)
    return Target(float(t["HV_GPa"]), lo, hi, t["confidence"], t["basis"], "vickers_plateau",
                  protocol=t.get("protocol"))


def vickers_record_target(m: Mapping[str, Any], source: str, value_GPa: float, note: str = "") -> Target:
    """A target taken from one Vickers record of the dataset (the record must exist: source and value match)."""
    for h in m["hardness_vickers"]:
        if h.get("source") == source and h.get("value_GPa") is not None and abs(h["value_GPa"] - value_GPa) < 1e-6:
            base = vickers_plateau_target(m)
            lo = min(base.lo, value_GPa) if base.lo is not None else value_GPa
            hi = max(base.hi, value_GPa) if base.hi is not None else value_GPa
            return Target(float(value_GPa), lo, hi, h["confidence"],
                          f"record {source} {h.get('location', '')}: {h.get('specimen', '')} {note}".strip(),
                          "vickers_compilation", protocol="review")
    raise KeyError(f"{m['id']}: no Vickers record from {source} with value {value_GPa} GPa")


def nano_target(m: Mapping[str, Any]) -> Optional[Target]:
    """Nanoindentation target on the Vickers scale: median of the H_IT records (confidence >= medium when any),
    converted with HV = 0.92673 H_IT (ISO 14577); range = min..max of the records used."""
    recs = [h for h in m.get("hardness_nanoindentation", []) if h.get("value_GPa") is not None
            and h.get("area_definition") == "projected"]
    if not recs:
        return None
    good = [h for h in recs if h.get("confidence") in ("high", "medium")]
    use = good or recs
    vals = np.array([h["value_GPa"] for h in use], float)
    conf = min((h.get("confidence", "low") for h in use), key=lambda c: _CONF_ORDER.get(c, 0))
    h_it = float(np.median(vals))
    basis = "; ".join(f"{h['value_GPa']} GPa {h['source']} ({h.get('specimen', '')[:60]})" for h in use)
    return Target(HV_GPA_PER_H_IT_GPA * h_it, HV_GPA_PER_H_IT_GPA * float(vals.min()),
                  HV_GPA_PER_H_IT_GPA * float(vals.max()), conf, "median H_IT of " + basis, "nano_hit_as_hv",
                  raw_value=h_it)


def _surface_normal(symmetry: str) -> tuple[tuple[float, float, float], str]:
    return ((0.0, 0.0, 1.0), "(001)") if symmetry == "cubic" else ((0.0, 0.0, 1.0), "(0001)")


def experimental_descriptors(m: Mapping[str, Any]) -> Optional[ElasticDescriptors]:
    """Experimental moduli: the highest-confidence of the single-crystal C_ij sets and the measured polycrystal
    (B and G both given); ties go to the single crystal, then to the order in the file.  None if neither exists."""
    cands: list[tuple[int, int, ElasticDescriptors]] = []
    for i, e in enumerate(m.get("elastic_experimental", [])):
        sym = e["symmetry"]
        C = stiffness_matrix(e["C_GPa"], sym)
        normal, lab = _surface_normal(sym)
        d = ElasticDescriptors.from_stiffness(C, basis=f"single-crystal C_ij, {e['source']}",
                                              source_kind="exp_single_crystal", confidence=e["confidence"],
                                              normal=normal, surface_label=lab, notes=e.get("specimen", ""))
        cands.append((_CONF_ORDER.get(e["confidence"], 0), 1, d))
    for e in m.get("elastic_polycrystal_measured", []):
        if e.get("B_GPa") is None or e.get("G_GPa") is None:
            continue
        d = ElasticDescriptors.from_moduli(e["B_GPa"], e["G_GPa"], basis=f"polycrystal B, G, {e['source']}",
                                           source_kind="exp_polycrystal", confidence=e["confidence"],
                                           notes=e.get("specimen", "") or "")
        cands.append((_CONF_ORDER.get(e["confidence"], 0), 0, d))
    if not cands:
        return None
    best = max(c[0] * 2 + c[1] for c in cands)
    return next(c[2] for c in cands if c[0] * 2 + c[1] == best)


def dft_descriptors(entry: Mapping[str, Any], material: Mapping[str, Any]) -> Optional[ElasticDescriptors]:
    """DFT-PBE (Materials Project 2025-09-25) moduli: VRH from the unrounded raw tensor (rotation invariant), M from
    the IEEE-frame tensor.  A failed MP fit is kept with confidence 'low'."""
    if not entry or not entry.get("found", True) or entry.get("C_raw_GPa") is None:
        return None
    Craw = np.asarray(entry["C_raw_GPa"], float)
    Cieee = np.asarray(entry["C_ieee_GPa"], float)
    sym = material.get("elastic_dft_pbe", {}).get("symmetry", "cubic")
    normal, lab = _surface_normal(sym)
    m = vrh_moduli(Craw)
    conf = material.get("elastic_dft_pbe", {}).get("confidence", "high")
    state = entry.get("state", "")
    return ElasticDescriptors(B=m["B"], G=m["G"], E=m["E"], nu=m["nu"], k=m["k"],
                              basis=f"DFT-PBE, Materials Project {entry.get('mp_id')} (state {state})",
                              source_kind="dft_pbe", confidence=conf, M=vlassak_nix_modulus(Cieee, normal),
                              M_surface=lab, C_voigt=tuple(tuple(float(x) for x in r) for r in Cieee),
                              notes="; ".join(entry.get("warnings", []) or []))


def mace_descriptors(entry: Mapping[str, Any]) -> Optional[ElasticDescriptors]:
    """MACE-MP-0 small moduli from data/hardness_calibration/mace_mp0_small_elastic.json."""
    if not entry:
        return None
    C = np.asarray(entry["C_voigt_GPa"], float)
    normal, lab = _surface_normal(entry["symmetry"])
    return ElasticDescriptors.from_stiffness(C, basis="MACE-MP-0 small (scripts/hardness_calibration/mace_elastic.py)",
                                             source_kind="mace_mp0_small", confidence="medium", normal=normal,
                                             surface_label=lab)


GGA_TAGS = ("PBE", "PW91", "GGA")


INHOUSE_PBE_DIR = Path(__file__).resolve().parents[3] / "data" / "dft_reference"
# calibration id -> (file in data/dft_reference/, key of its "cases" or None = lowest over every case)
_INHOUSE_PBE_FILES = (("3C-SiC", "sic_3c_ideal_shear_pbe.json", None), ("Si", "si_c_ideal_shear_pbe.json", "Si"),
                      ("diamond", "si_c_ideal_shear_pbe.json", "C"))


def inhouse_pbe_ideal_shear(directory: Path | str = INHOUSE_PBE_DIR) -> dict[str, dict[str, Any]]:
    """Ideal shear strengths computed in this project with DFT-PBE (data/dft_reference/, docs/DFT.md section 8),
    keyed by calibration id: the lowest computed case (3C-SiC: twin, anti-twin and [1-10] senses; Si and diamond:
    the (111)[11-2] twinning sense only).  Same code, functional, pseudopotential family and protocol for every
    material.  Empty when the files are absent."""
    out: dict[str, dict[str, Any]] = dict()
    d = Path(directory)
    for mid, fname, key in _INHOUSE_PBE_FILES:
        f = d / fname
        if not f.exists():
            continue
        cases = load_json(f).get("cases", dict())
        if key is not None:
            cases = dict(only=cases[key]) if key in cases else dict()
        vals = [(c["ideal_shear_strength_GPa"], name, c) for name, c in cases.items()
                if c.get("ideal_shear_strength_GPa") is not None]
        if not vals:
            continue
        v, name, c = min(vals, key=lambda t: t[0])
        out[mid] = dict(tau_GPa=float(v), functional="PBE (this project: QE 7.5, SSSP 1.3.0)",
                        source="data/dft_reference/" + fname, slip_system=c.get("description"),
                        confidence="computed (DFT-PBE model value)",
                        minimum_over=("twinning sense only" if key is not None else ", ".join(sorted(cases))))
    return out


def ideal_shear_value(m: Mapping[str, Any], rule: str = "lowest_any",
                      inhouse: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Optional[dict[str, Any]]:
    """Lowest DFT ideal shear strength among the records: ``lowest_any`` (any functional, as
    ``ideal_shear_dft_lowest``), ``gga_only`` (PBE / PW91 records only; LDA-only materials return None) or
    ``inhouse_pbe`` (the project's own PBE value, :func:`inhouse_pbe_ideal_shear`, where one exists: 3C-SiC, Si,
    diamond; else ``lowest_any``).  ``inhouse`` overrides the in-house table (tests)."""
    if rule == "inhouse_pbe":
        tab = inhouse_pbe_ideal_shear() if inhouse is None else inhouse
        if m["id"] in tab:
            return dict(tab[m["id"]])
        rule = "lowest_any"
    recs = [r for r in m.get("ideal_shear_dft", []) if r.get("value_GPa") is not None]
    if rule == "gga_only":
        recs = [r for r in recs if any(t in r.get("functional", "") for t in GGA_TAGS)
                and "LDA" not in r.get("functional", "")]
    elif rule != "lowest_any":
        raise ValueError(f"unknown ideal-shear rule {rule!r}")
    if not recs:
        return None
    r = min(recs, key=lambda r: r["value_GPa"])
    return {"tau_GPa": float(r["value_GPa"]), "functional": r.get("functional"), "source": r.get("source"),
            "slip_system": r.get("slip_system"), "confidence": r.get("confidence"),
            "minimum_over": r.get("minimum_over")}


@dataclass
class CalibrationSet:
    """Aligned arrays for one configuration (materials lacking a required descriptor are dropped and listed)."""

    ids: list[str]
    y: np.ndarray
    w: np.ndarray
    X: dict[str, np.ndarray]
    targets: dict[str, Target]
    dropped: dict[str, str]
    descriptor_basis: dict[str, str]


def assemble(ids: Sequence[str], targets: Mapping[str, Optional[Target]],
             elastic: Mapping[str, Optional[ElasticDescriptors]], tau: Mapping[str, Optional[dict[str, Any]]],
             requires: Sequence[str], weighting: str = "sigma") -> CalibrationSet:
    """Build the arrays a model needs.  ``weighting``: ``sigma`` (w = 1/sigma_rel^2), ``sigma_protocol``
    (w = 1/sigma_rel_protocol^2: targets that are not load-plateau values get sigma_rel >= 20 %) or ``equal``."""
    keep, dropped, basis = [], {}, {}
    X: dict[str, list[float]] = {k: [] for k in requires}
    for i in ids:
        t = targets.get(i)
        if t is None:
            dropped[i] = "no target"
            continue
        el = elastic.get(i)
        ta = tau.get(i)
        need_el = any(r in ("B", "G", "E", "nu", "k", "M") for r in requires)
        if need_el and el is None:
            dropped[i] = "no elastic descriptors from this source"
            continue
        if "tau" in requires and ta is None:
            dropped[i] = "no ideal shear strength under this rule"
            continue
        for r in requires:
            X[r].append(float(ta["tau_GPa"]) if r == "tau" else float(getattr(el, r)))
        keep.append(i)
        parts = []
        if need_el:
            parts.append(el.basis)
        if "tau" in requires:
            parts.append(f"tau {ta['tau_GPa']} GPa {ta['source']} ({ta['functional']})")
        basis[i] = "; ".join(parts)
    y = np.array([targets[i].value for i in keep], float)
    if weighting == "sigma":
        w = np.array([1.0 / targets[i].sigma_rel ** 2 for i in keep])
    elif weighting == "sigma_protocol":
        w = np.array([1.0 / targets[i].sigma_rel_protocol ** 2 for i in keep])
    else:
        w = np.ones(len(keep))
    return CalibrationSet(ids=keep, y=y, w=w, X={k: np.asarray(v, float) for k, v in X.items()},
                          targets={i: targets[i] for i in keep}, dropped=dropped, descriptor_basis=basis)


# ============================================================================================ 6. records
_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def _composition_dict(formula: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for el, n in _FORMULA_RE.findall(formula):
        out[el] = out.get(el, 0.0) + (float(n) if n else 1.0)
    return out


def reference_candidate(material_id: str, formula: str, phase_name: Optional[str] = None):
    """A REFERENCE Candidate for a calibration material (no pymatgen needed)."""
    from fracture_md.schema.material import Candidate, CandidateKind, Composition, Phase

    return Candidate(id=f"hardness-cal:{material_id}", name=material_id, kind=CandidateKind.REFERENCE,
                     composition=Composition(formula=formula, elements=_composition_dict(formula)),
                     phases=[Phase(name=phase_name or material_id)])


def hardness_estimate_record(*, value_GPa: float, candidate, model: HardnessModel, params: Sequence[float],
                             descriptors: Mapping[str, float], descriptor_basis: str,
                             interval_GPa: Optional[tuple[float, float]] = None, interval_note: str = "",
                             calibration_ref: str = "data/hardness_calibration/results.json",
                             descriptor_range: Optional[Mapping[str, tuple[float, float]]] = None,
                             input_record_ids: Sequence[str] = (), notes: str = "",
                             input_hashes: Optional[Mapping[str, str]] = None, requested_by: str = "human",
                             provenance=None):
    """A ``vickers_hardness_estimate`` PropertyRecord (DERIVED_MODEL, Vickers scale, GPa) of a calibrated model.

    Status: MODEL_ESTIMATE; OUT_OF_DOMAIN when a descriptor lies outside the range spanned by the calibration
    materials (``descriptor_range``), since the prediction interval was measured only inside it.  The 90 %
    prediction interval goes in ``uncertainty`` (kind reference_discrepancy, interval 'range').  Conditions are
    those of the calibration targets (room temperature, quasi-static Vickers test on a real specimen whose defect
    content is not modelled), not of the 0 K descriptors.
    """
    from fracture_md.records import make_record
    from fracture_md.schema.common import (Conditions, DefectState, Provenance, Uncertainty, UncertaintyKind,
                                           ValidationStatus)
    from fracture_md.schema.properties import (Derivation, MethodSpec, PropertyName, PropertyOrigin, PropertyRecord,
                                               PropertyScale)
    from fracture_md.units import Quantity, Unit

    outside = []
    for k, (lo, hi) in (descriptor_range or {}).items():
        v = descriptors.get(k)
        if v is not None and not (lo <= v <= hi):
            outside.append(f"{k} = {v:.4g} outside the calibration range [{lo:.4g}, {hi:.4g}]")
    pstr = ", ".join(f"{n} = {v:.6g}" for n, v in zip(model.param_names, params))
    unc = []
    if interval_GPa is not None:
        lo, hi = interval_GPa
        unc.append(Uncertainty(kind=UncertaintyKind.REFERENCE_DISCREPANCY, value=0.5 * (hi - lo), unit=Unit.GPA,
                               interval="range",
                               description=f"90 % prediction interval [{lo:.3g}, {hi:.3g}] GPa from leave-one-out "
                                           f"residuals against the calibration targets, Vickers values chosen from "
                                           f"measured records ({interval_note})".strip()))
    unc.append(Uncertainty(kind=UncertaintyKind.UNQUANTIFIED,
                           description="model form: the interval covers the scatter of the calibration materials "
                                       "only; mechanisms absent from them (phase transformation, amorphisation, "
                                       "metallic bonding, microstructure) are not covered"))
    cond = Conditions(temperature=Quantity(value=295.0, unit=Unit.KELVIN), loading="quasistatic",
                      stress_state="Vickers indentation at the load plateau (calibration targets)",
                      defect_state=DefectState(kind="unknown", description="real specimens (single crystals or dense "
                                               "polycrystals) as in the calibration targets; defects not modelled"),
                      environment="laboratory air")
    if provenance is None:
        impl = dict()
        impl["fracture_md.discovery.hardness_models"] = file_sha256(Path(__file__))
        try:  # the standard harness provenance (git version, package versions, platform)
            from fracture_md.provenance import collect

            provenance = collect(engine="fracture_md.discovery.hardness_models", implementation_ids=impl,
                                 input_hashes=dict(input_hashes or dict()), requested_by=requested_by,
                                 float_precision="float64")
        except ImportError:  # pragma: no cover - collect() needs JAX; keep the record buildable without it
            provenance = Provenance(engine="fracture_md.discovery.hardness_models", implementation_ids=impl,
                                    input_hashes=dict(input_hashes or dict()), requested_by=requested_by,
                                    float_precision="float64")
    rec = make_record(
        name=PropertyName.VICKERS_HARDNESS_ESTIMATE, value=float(value_GPa), unit=Unit.GPA, candidate=candidate,
        method=MethodSpec(name=f"{model.label} hardness model", description=model.equation,
                          assumptions=list(model.assumptions)),
        origin=PropertyOrigin.DERIVED_MODEL, scale=PropertyScale.MACROSCOPIC, uncertainty=unc, conditions=cond,
        provenance=provenance,
        derivation=Derivation(model=f"{model.label}: {model.equation} ({pstr})", equation=model.equation,
                              assumptions=list(model.assumptions) + [
                                  f"constants calibrated against Vickers hardness targets chosen from measured records "
                                  f"(only some on the load plateau; materials.json calibration_target.protocol; "
                                  f"{calibration_ref})",
                                  f"descriptors: {descriptor_basis}"],
                              input_record_ids=list(input_record_ids)),
        notes=(notes.rstrip(". ") + ". " if notes else "") + "Model estimate of a plateau-load Vickers hardness "
              f"(model citation: {model.citation}). Not a measurement.",
    )
    if outside:
        d = rec.model_dump(mode="json")
        d["validation"]["status"] = ValidationStatus.OUT_OF_DOMAIN.value
        d["validation"]["notes"] = ((d["validation"]["notes"] + " ") if d["validation"]["notes"] else "") + (
            "Extrapolation: " + "; ".join(outside) + ".")
        rec = PropertyRecord.model_validate(d)
    return rec


__all__ = [
    "HV_GPA_PER_H_IT_GPA", "VICKERS_EQUIVALENT_CONE_BETA_DEG", "MODELS", "HardnessModel", "ElasticDescriptors",
    "Target", "CalibrationSet", "CalibrationResult", "stiffness_matrix", "voigt_to_tensor", "vrh_moduli",
    "isotropic_E_nu", "vlassak_nix_modulus", "johnson_cavity_mean_pressure", "yield_from_ideal_shear", "fit_model",
    "leave_one_out", "regression_metrics", "prediction_interval", "spearman_rho", "target_sigma_rel",
    "load_materials", "vickers_plateau_target", "vickers_record_target", "nano_target", "experimental_descriptors",
    "dft_descriptors", "mace_descriptors", "ideal_shear_value", "assemble", "reference_candidate",
    "hardness_estimate_record", "file_sha256", "load_json", "nested_leave_one_out", "inhouse_pbe_ideal_shear",
    "PROTOCOL_SIGMA_FLOOR", "TARGET_PROTOCOLS",
]
