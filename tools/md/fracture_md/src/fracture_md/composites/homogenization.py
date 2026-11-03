"""Effective elastic stiffness of particle-filled resins: analytic bounds/estimates and an FFT solver.

Analytic (isotropic phases, spherical inclusions)
    * Voigt / Reuss bounds, multi-phase Hashin-Shtrikman(-Walpole) bounds
    * dilute Eshelby and multi-phase Mori-Tanaka estimates (spheres)
    * exact effective stiffness of a laminate (any number of layers)

Numerical
    * FFT-based Lippmann-Schwinger homogenisation on a periodic voxel grid,
      small strain, isotropic phases, float64, jit-compiled (CPU or GPU):
      - ``scheme="cg"`` (default): Galerkin/projection formulation solved by
        conjugate gradients (Zeman et al. 2010; de Geus et al. 2017): find the
        compatible zero-mean fluctuation e with  G[C:(E + e)] = 0, where G is
        the orthogonal projection onto compatible strain fields
        (continuous-frequency Green operator with lambda0=0, mu0=1/2).
      - ``scheme="basic"``: Moulinec-Suquet (1994/1998) fixed-point scheme
        with reference medium (lambda0, mu0) = mean of the extreme phases;
        used as an independent cross-check of the CG solver.
      Six macroscopic unit strains give the full 6x6 effective stiffness in
      Voigt order (xx, yy, zz, yz, xz, xy) with ENGINEERING shear strains (the
      harness-wide convention of physics/elastic.py).  For even grids the
      Nyquist component of every wave-vector is set to zero so that the
      discrete operator stays an exact real projection.

Two-scale treatment: nano-fillers (< ~0.1 um) are first homogenised into an
effective "nanofilled matrix" by Mori-Tanaka (:func:`nanofilled_matrix`);
micron fillers are resolved in the RVE (see composites/microstructure.py).

Design-agent entry point: :func:`effective_modulus` (spec dict, Formulation or
Candidate) -> (E [GPa], nu, uncertainty dict, method string).

All moduli here are in GPa.  See docs/dental/HOMOGENIZATION.md for the
verification record and data/dental/homogenization_results.json for the
production numbers.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fracture_md.composites.microstructure import (
    MATRIX_PHASE,
    SphereFamily,
    VoxelMicrostructure,
    laminate,
    pack_spheres,
    single_sphere,
    voxelize,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PATH = REPO_ROOT / "data" / "dental" / "homogenization_results.json"
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


# ---------------------------------------------------------------------------
# Isotropic elasticity helpers
# ---------------------------------------------------------------------------
def KG_from_E_nu(E: float, nu: float) -> tuple[float, float]:
    return E / (3.0 * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def E_nu_from_KG(K: float, G: float) -> tuple[float, float]:
    return 9.0 * K * G / (3.0 * K + G), (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))


def isotropic_stiffness_voigt(K: float, G: float) -> np.ndarray:
    lam = K - 2.0 * G / 3.0
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] = lam + 2.0 * G
    C[np.arange(3, 6), np.arange(3, 6)] = G
    return C


@dataclass(frozen=True)
class IsotropicPhase:
    name: str
    E: float  # GPa
    nu: float

    @property
    def K(self) -> float:
        return KG_from_E_nu(self.E, self.nu)[0]

    @property
    def G(self) -> float:
        return KG_from_E_nu(self.E, self.nu)[1]

    @classmethod
    def from_KG(cls, name: str, K: float, G: float) -> "IsotropicPhase":
        E, nu = E_nu_from_KG(K, G)
        return cls(name, float(E), float(nu))

    def stiffness(self) -> np.ndarray:
        return isotropic_stiffness_voigt(self.K, self.G)


# ---------------------------------------------------------------------------
# Constituent library (GPa; min / nominal / max).  Keys in `sources` refer to
# docs/dental/RESEARCH_BRIEF.md.  Values flagged "assumed" are engineering
# ranges, NOT literature values, and are propagated as uncertainty.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Constituent:
    key: str
    description: str
    E: tuple[float, float, float]
    nu: tuple[float, float, float]
    density_g_cm3: Optional[float]
    sources: tuple[str, ...]
    status: str  # "literature" | "assumed_range" | "mixed"
    notes: str = ""

    def phase(self, which: str = "nominal") -> IsotropicPhase:
        i = {"min": 0, "nominal": 1, "max": 2}[which]
        return IsotropicPhase(self.key, self.E[i], self.nu[1] if which != "nu_min" else self.nu[0])


CONSTITUENTS: dict[str, Constituent] = {
    "resin": Constituent(
        key="resin",
        description="cured UDMA/Bis-EMA/TEGDMA dimethacrylate network (unfilled)",
        E=(2.0, 2.5, 3.5),
        nu=(0.30, 0.35, 0.40),
        density_g_cm3=1.2,
        sources=("RESEARCH_BRIEF §3.7 'unfilled polymer (about 2-3 GPa)'", "[T4] homopolymers E 1.13-1.43 GPa (lower bound context)",
                 "dental-homogenization brief: E ~2.5-3.5 GPa, nu ~0.35"),
        status="assumed_range",
        notes="E range spans the brief's 2-3 GPa and the task's 2.5-3.5 GPa; nu 0.30-0.40 is an assumption (no source "
              "in the brief); density 1.2 g/cm^3 is the brief's own assumption (§4.4).",
    ),
    "ba_glass": Constituent(
        key="ba_glass",
        description="silanised Ba-Al-borosilicate dental glass (e.g. Schott GM27884 class)",
        E=(60.0, 70.0, 80.0),
        nu=(0.20, 0.22, 0.25),
        density_g_cm3=2.80,
        sources=("[T12] Schott GM27884 datasheet: density 2.80 g/cm^3, n 1.53",
                 "RESEARCH_BRIEF §6: Ba-aluminoborosilicate E NOT found; measure or estimate via Makishima-Mackenzie [T40]",
                 "[T14] silica glass E 72.2 GPa used as the anchor of the range"),
        status="assumed_range",
        notes="E and nu are NOT verified literature values for this glass; the wide 60-80 GPa range brackets silica "
              "(72.2 GPa) and is propagated. Replace with a measured / Makishima-Mackenzie value when available.",
    ),
    "silica": Constituent(
        key="silica",
        description="amorphous silica (fumed/colloidal nano-silica or silica glass)",
        E=(70.0, 72.2, 74.0),
        nu=(0.16, 0.168, 0.18),
        density_g_cm3=2.20,
        sources=("[T14] Pabst & Gregorova 2013: silica glass E 72.2 GPa, nu 0.168",),
        status="mixed",
        notes="nominal values from [T14]; +/- range assumed. Density 2.20 g/cm^3 is a handbook value for vitreous "
              "silica (not from the brief). Porous/partially sintered nanoclusters are softer than dense silica.",
    ),
    "zirconia": Constituent(
        key="zirconia",
        description="tetragonal zirconia (t-ZrO2 / Y-TZP) particles or nanoclusters",
        E=(200.0, 205.0, 210.0),
        nu=(0.30, 0.31, 0.32),
        density_g_cm3=6.05,
        sources=("[T15] Kisi & Howard 1998 (tabulated by Pabst 2004): recommended E 205 GPa, nu 0.31",),
        status="mixed",
        notes="E, nu nominal from [T15]; ranges assumed. Density 6.05 g/cm^3 is a handbook value for 3Y-TZP (not from "
              "the brief).",
    ),
}


def _lookup_constituent(name: str) -> Constituent:
    key = name.lower()
    if key in CONSTITUENTS:
        return CONSTITUENTS[key]
    if "zirc" in key or "zro2" in key:
        return CONSTITUENTS["zirconia"]
    if "silica" in key or "sio2" in key or "aerosil" in key or "fumed" in key:
        return CONSTITUENTS["silica"]
    if "glass" in key or "barium" in key or "ba-" in key or "ba_" in key:
        return CONSTITUENTS["ba_glass"]
    raise KeyError(f"no constituent data for filler '{name}' (known: {sorted(CONSTITUENTS)})")


# ---------------------------------------------------------------------------
# Analytic bounds and estimates (isotropic phases)
# ---------------------------------------------------------------------------
def _arrays(phases: Sequence[IsotropicPhase], fractions: Sequence[float]):
    K = np.array([p.K for p in phases], dtype=float)
    G = np.array([p.G for p in phases], dtype=float)
    c = np.array(fractions, dtype=float)
    if abs(c.sum() - 1.0) > 1e-9:
        raise ValueError(f"volume fractions must sum to 1 (got {c.sum():.6f})")
    return K, G, c


def voigt_reuss_bounds(phases, fractions) -> dict[str, tuple[float, float]]:
    K, G, c = _arrays(phases, fractions)
    return {"voigt": (float(c @ K), float(c @ G)), "reuss": (float(1.0 / (c @ (1.0 / K))), float(1.0 / (c @ (1.0 / G))))}


def hashin_shtrikman_bounds(phases, fractions) -> dict[str, tuple[float, float]]:
    """Multi-phase HS(-Walpole) bounds on (K, G) for a statistically isotropic mixture."""
    K, G, c = _arrays(phases, fractions)

    def k_hs(gs):
        return float(1.0 / np.sum(c / (K + 4.0 * gs / 3.0)) - 4.0 * gs / 3.0)

    def zeta(ks, gs):
        return gs / 6.0 * (9.0 * ks + 8.0 * gs) / (ks + 2.0 * gs)

    def g_hs(z):
        return float(1.0 / np.sum(c / (G + z)) - z)

    lo = (k_hs(G.min()), g_hs(zeta(K.min(), G.min())))
    hi = (k_hs(G.max()), g_hs(zeta(K.max(), G.max())))
    return {"lower": lo, "upper": hi}


def _eshelby_sphere_factors(Km: float, Gm: float) -> tuple[float, float]:
    alpha = 3.0 * Km / (3.0 * Km + 4.0 * Gm)
    beta = 6.0 * (Km + 2.0 * Gm) / (5.0 * (3.0 * Km + 4.0 * Gm))
    return alpha, beta


def mori_tanaka_spheres(matrix: IsotropicPhase, inclusions: Sequence[IsotropicPhase],
                        fractions: Sequence[float]) -> tuple[float, float]:
    """Multi-phase Mori-Tanaka (spheres).  ``fractions`` = inclusion volume fractions (matrix = remainder)."""
    c = np.array(fractions, dtype=float)
    c0 = 1.0 - c.sum()
    if c0 < 0:
        raise ValueError("inclusion fractions exceed 1")
    Km, Gm = matrix.K, matrix.G
    a, b = _eshelby_sphere_factors(Km, Gm)
    Kr = np.array([p.K for p in inclusions])
    Gr = np.array([p.G for p in inclusions])
    AK = 1.0 / (1.0 + a * (Kr - Km) / Km)
    AG = 1.0 / (1.0 + b * (Gr - Gm) / Gm)
    K = (c0 * Km + np.sum(c * Kr * AK)) / (c0 + np.sum(c * AK))
    G = (c0 * Gm + np.sum(c * Gr * AG)) / (c0 + np.sum(c * AG))
    return float(K), float(G)


def dilute_eshelby_spheres(matrix: IsotropicPhase, inclusions: Sequence[IsotropicPhase],
                           fractions: Sequence[float]) -> tuple[float, float]:
    """Dilute (non-interacting) Eshelby estimate, first order in the inclusion fractions."""
    c = np.array(fractions, dtype=float)
    Km, Gm = matrix.K, matrix.G
    a, b = _eshelby_sphere_factors(Km, Gm)
    Kr = np.array([p.K for p in inclusions])
    Gr = np.array([p.G for p in inclusions])
    K = Km + np.sum(c * (Kr - Km) / (1.0 + a * (Kr - Km) / Km))
    G = Gm + np.sum(c * (Gr - Gm) / (1.0 + b * (Gr - Gm) / Gm))
    return float(K), float(G)


def nanofilled_matrix(resin: IsotropicPhase, nano: Sequence[IsotropicPhase], fractions_in_matrix: Sequence[float],
                      name: str = MATRIX_PHASE) -> IsotropicPhase:
    """Effective isotropic phase for resin + nano-filler (Mori-Tanaka, spheres) - first scale of the two-scale model."""
    if len(nano) == 0 or sum(fractions_in_matrix) == 0:
        return IsotropicPhase(name, resin.E, resin.nu)
    K, G = mori_tanaka_spheres(resin, nano, fractions_in_matrix)
    return IsotropicPhase.from_KG(name, K, G)


def laminate_stiffness(C_layers: Sequence[np.ndarray], fractions: Sequence[float], axis: int = 0) -> np.ndarray:
    """Exact effective 6x6 stiffness (Voigt, engineering shear) of a laminate with layer normal ``axis``.

    Traction components (normal index set) are uniform, in-plane strains are
    uniform; the result follows from averaging the partitioned constitutive law.
    """
    nset = {0: [0, 5, 4], 1: [1, 5, 3], 2: [2, 3, 4]}[axis]
    tset = [i for i in range(6) if i not in nset]
    f = np.array(fractions, dtype=float)
    f = f / f.sum()
    inv_nn = sum(fk * np.linalg.inv(C[np.ix_(nset, nset)]) for fk, C in zip(f, C_layers))
    inv_nn_nt = sum(fk * np.linalg.solve(C[np.ix_(nset, nset)], C[np.ix_(nset, tset)]) for fk, C in zip(f, C_layers))
    tn_inv_nn = sum(fk * C[np.ix_(tset, nset)] @ np.linalg.inv(C[np.ix_(nset, nset)]) for fk, C in zip(f, C_layers))
    schur = sum(fk * (C[np.ix_(tset, tset)] - C[np.ix_(tset, nset)] @ np.linalg.solve(C[np.ix_(nset, nset)], C[np.ix_(nset, tset)]))
                for fk, C in zip(f, C_layers))
    Cnn = np.linalg.inv(inv_nn)
    out = np.zeros((6, 6))
    out[np.ix_(nset, nset)] = Cnn
    out[np.ix_(nset, tset)] = Cnn @ inv_nn_nt
    out[np.ix_(tset, nset)] = tn_inv_nn @ Cnn
    out[np.ix_(tset, tset)] = schur + tn_inv_nn @ Cnn @ inv_nn_nt
    return out


# ---------------------------------------------------------------------------
# Anisotropy / isotropic projection of a 6x6 Voigt stiffness
# ---------------------------------------------------------------------------
def _mandel(C: np.ndarray) -> np.ndarray:
    s = np.array([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])
    return C * s[:, None] * s[None, :]


def isotropy_metrics(C: np.ndarray) -> dict[str, float]:
    """Voigt/Reuss/Hill isotropic projections, universal anisotropy index and tensorial deviation from isotropy."""
    C = 0.5 * (C + C.T)
    S = np.linalg.inv(C)
    KV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    GV = (C[0, 0] + C[1, 1] + C[2, 2] - (C[0, 1] + C[0, 2] + C[1, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15.0
    KR = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    GR = 15.0 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2]) + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    KH, GH = 0.5 * (KV + KR), 0.5 * (GV + GR)
    EH, nuH = E_nu_from_KG(KH, GH)
    Ciso = isotropic_stiffness_voigt(KV, GV)
    dev = float(np.linalg.norm(_mandel(C) - _mandel(Ciso)) / np.linalg.norm(_mandel(C)))
    Edir = 1.0 / np.diag(S)[:3]
    Gdir = 1.0 / np.diag(S)[3:]
    return {
        "K_voigt": float(KV), "G_voigt": float(GV), "K_reuss": float(KR), "G_reuss": float(GR),
        "K_hill": float(KH), "G_hill": float(GH), "E_hill": float(EH), "nu_hill": float(nuH),
        "universal_anisotropy_index": float(5.0 * GV / GR + KV / KR - 6.0),
        "relative_deviation_from_isotropy": dev,
        "E_axis_spread_rel": float((Edir.max() - Edir.min()) / Edir.mean()),
        "G_axis_spread_rel": float((Gdir.max() - Gdir.min()) / Gdir.mean()),
        "zener_cubic_proxy": float(2 * np.mean(np.diag(C)[3:]) / (np.mean(np.diag(C)[:3]) - np.mean([C[0, 1], C[0, 2], C[1, 2]]))),
    }


# ---------------------------------------------------------------------------
# FFT solver
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _unit_wavevectors(n: int) -> np.ndarray:
    k = np.fft.fftfreq(n, d=1.0 / n)
    kr = np.fft.rfftfreq(n, d=1.0 / n)
    if n % 2 == 0:  # zero the Nyquist component so the discrete operator is an exact real projection
        k = np.where(np.abs(k) == n // 2, 0.0, k)
        kr = np.where(np.abs(kr) == n // 2, 0.0, kr)
    kx, ky, kz = np.meshgrid(k, k, kr, indexing="ij")
    xi = np.stack([kx, ky, kz])
    nrm = np.sqrt(np.sum(xi * xi, axis=0))
    nrm = np.where(nrm > 0, nrm, 1.0)
    return xi / nrm


def _gamma_apply(tau, nvec, c1, c2, n):
    """(Gamma0 * tau) for an isotropic reference: c1 = 1/(2 mu0), c2 = (lam0+mu0)/(mu0 (lam0+2 mu0))."""
    th = jnp.fft.rfftn(tau, axes=(2, 3, 4))
    t = jnp.einsum("ij...,j...->i...", th, nvec)
    s = jnp.einsum("i...,i...->...", nvec, t)
    out = c1 * (nvec[:, None] * t[None, :] + t[:, None] * nvec[None, :]) - c2 * (nvec[:, None] * nvec[None, :]) * s
    return jnp.fft.irfftn(out, s=(n, n, n), axes=(2, 3, 4))


def _stress(eps, lam, mu):
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    eye = jnp.eye(3, dtype=eps.dtype)[:, :, None, None, None]
    return lam * tr * eye + 2.0 * mu * eps


@partial(jax.jit, static_argnames=("n", "max_iter"))
def _solve_cg(lam, mu, E, nvec, tol, *, n, max_iter):
    Ef = jnp.broadcast_to(E[:, :, None, None, None], (3, 3, n, n, n))
    G = lambda tau: _gamma_apply(tau, nvec, 1.0, 1.0, n)  # noqa: E731  (projection: lam0=0, mu0=1/2)
    A = lambda x: G(_stress(x, lam, mu))  # noqa: E731
    b = -G(_stress(Ef, lam, mu))
    ref = jnp.sqrt(jnp.sum(_stress(Ef, lam, mu) ** 2))
    rs0 = jnp.sum(b * b)
    x0 = jnp.zeros_like(Ef)

    def cond(st):
        _, _, _, rs, k = st
        return (jnp.sqrt(rs) > tol * ref) & (k < max_iter)

    def body(st):
        x, r, p, rs, k = st
        Ap = A(p)
        alpha = rs / jnp.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = jnp.sum(r * r)
        p = r + (rs_new / rs) * p
        return x, r, p, rs_new, k + 1

    x, r, _, rs, k = jax.lax.while_loop(cond, body, (x0, b, b, rs0, 0))
    eps = Ef + x
    sig = _stress(eps, lam, mu)
    eq = jnp.sqrt(jnp.sum(G(sig) ** 2)) / ref  # independent equilibrium residual of the final field
    return jnp.mean(sig, axis=(2, 3, 4)), k, jnp.sqrt(rs) / ref, eq


@partial(jax.jit, static_argnames=("n", "max_iter"))
def _solve_basic(lam, mu, E, nvec, tol, lam0, mu0, *, n, max_iter):
    Ef = jnp.broadcast_to(E[:, :, None, None, None], (3, 3, n, n, n))
    c1 = 1.0 / (2.0 * mu0)
    c2 = (lam0 + mu0) / (mu0 * (lam0 + 2.0 * mu0))
    ref = jnp.sqrt(jnp.sum(Ef * Ef))

    def cond(st):
        _, err, k = st
        return (err > tol) & (k < max_iter)

    def body(st):
        eps, _, k = st
        sig = _stress(eps, lam, mu)
        new = eps - _gamma_apply(sig, nvec, c1, c2, n)
        err = jnp.sqrt(jnp.sum((new - eps) ** 2)) / ref
        return new, err, k + 1

    eps, err, k = jax.lax.while_loop(cond, body, (Ef, jnp.inf, 0))
    sig = _stress(eps, lam, mu)
    Gp = _gamma_apply(sig, nvec, 1.0, 1.0, n)
    eq = jnp.sqrt(jnp.sum(Gp**2)) / jnp.sqrt(jnp.sum(_stress(Ef, lam, mu) ** 2))
    return jnp.mean(sig, axis=(2, 3, 4)), k, err, eq


@dataclass
class FFTResult:
    C_eff: np.ndarray  # 6x6 GPa, Voigt, engineering shear (symmetrised)
    C_raw: np.ndarray  # unsymmetrised
    n: int
    scheme: str
    tol: float
    iterations: list[int]
    residuals: list[float]
    equilibrium_residuals: list[float]
    converged: bool
    symmetry_error: float
    volume_fractions: dict[str, float]
    wall_time_s: float
    device: str
    metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "n": self.n, "scheme": self.scheme, "tol": self.tol, "iterations": self.iterations,
            "residuals": self.residuals, "equilibrium_residuals": self.equilibrium_residuals,
            "converged": self.converged, "symmetry_error": self.symmetry_error,
            "volume_fractions": self.volume_fractions, "wall_time_s": self.wall_time_s, "device": self.device,
            "C_eff_GPa": np.round(self.C_eff, 6).tolist(), "metrics": self.metrics,
        }


def fft_homogenize(
    micro: VoxelMicrostructure | np.ndarray,
    phases: Mapping[str, IsotropicPhase] | Sequence[IsotropicPhase],
    *,
    scheme: str = "cg",
    tol: float = 1e-8,
    max_iter: int = 2000,
    load_cases: Sequence[int] = (0, 1, 2, 3, 4, 5),
) -> FFTResult:
    """Full 6x6 effective stiffness of a periodic voxel microstructure (GPa).

    ``phases`` maps phase names of ``micro.phase_names`` to IsotropicPhase, or
    is a sequence indexed by label.  Columns not in ``load_cases`` are filled
    from the symmetric counterpart when possible (else zero).
    """
    if isinstance(micro, VoxelMicrostructure):
        labels = micro.labels
        names = micro.phase_names
        vf = dict(micro.volume_fractions)
    else:
        labels = np.asarray(micro)
        names = tuple(str(i) for i in range(int(labels.max()) + 1))
        vf = {nm: float(np.mean(labels == i)) for i, nm in enumerate(names)}
    if isinstance(phases, Mapping):
        plist = [phases[nm] for nm in names]
    else:
        plist = list(phases)
    n = int(labels.shape[0])
    if labels.shape != (n, n, n):
        raise ValueError("labels must be an n^3 array")
    lam_ph = np.array([p.K - 2.0 * p.G / 3.0 for p in plist])
    mu_ph = np.array([p.G for p in plist])
    lam = jnp.asarray(lam_ph[labels])
    mu = jnp.asarray(mu_ph[labels])
    nvec = jnp.asarray(_unit_wavevectors(n))
    t0 = time.time()
    C = np.zeros((6, 6))
    its, res, eqs = [], [], []
    present = np.unique(labels)
    for j in load_cases:
        E = np.zeros((3, 3))
        a, b = VOIGT_PAIRS[j]
        if a == b:
            E[a, a] = 1.0
        else:
            E[a, b] = E[b, a] = 0.5  # engineering shear strain gamma = 1
        if scheme == "cg":
            sbar, k, r, eq = _solve_cg(lam, mu, jnp.asarray(E), nvec, tol, n=n, max_iter=max_iter)
        elif scheme == "basic":
            lam0 = 0.5 * (lam_ph[present].min() + lam_ph[present].max())
            mu0 = 0.5 * (mu_ph[present].min() + mu_ph[present].max())
            sbar, k, r, eq = _solve_basic(lam, mu, jnp.asarray(E), nvec, tol, lam0, mu0, n=n, max_iter=max_iter)
        else:
            raise ValueError(f"unknown scheme {scheme}")
        sbar = np.asarray(sbar)
        C[:, j] = [sbar[p, q] for p, q in VOIGT_PAIRS]
        its.append(int(k))
        res.append(float(r))
        eqs.append(float(eq))
    missing = [j for j in range(6) if j not in load_cases]
    for j in missing:
        C[:, j] = 0.0
        for i in load_cases:
            C[i, j] = C[j, i]
    wall = time.time() - t0
    sym_err = float(np.linalg.norm(C - C.T) / np.linalg.norm(C)) if not missing else float("nan")
    Cs = 0.5 * (C + C.T) if not missing else C
    conv = all(k < max_iter for k in its)
    dev = jax.devices()[0]
    res_obj = FFTResult(
        C_eff=Cs, C_raw=C, n=n, scheme=scheme, tol=tol, iterations=its, residuals=res, equilibrium_residuals=eqs,
        converged=conv, symmetry_error=sym_err, volume_fractions=vf, wall_time_s=wall,
        device=f"{dev.platform}:{getattr(dev, 'device_kind', '')}",
    )
    if not missing:
        res_obj.metrics = isotropy_metrics(Cs)
    return res_obj


# ---------------------------------------------------------------------------
# Design-agent API
# ---------------------------------------------------------------------------
NANO_D50_UM = 0.1  # fillers finer than this are treated at the nano scale (nanofilled matrix)


def _normalise_spec(spec: Any) -> dict:
    """Accept a spec dict, a schema Formulation or a Candidate(kind=FORMULATION)."""
    from fracture_md.schema.formulation import ComponentRole, Formulation
    from fracture_md.schema.material import Candidate

    if isinstance(spec, Candidate):
        if spec.formulation is None:
            raise ValueError("candidate has no formulation")
        spec = spec.formulation
    if isinstance(spec, Formulation):
        vols, fillers, notes = [], [], []
        resin_mass = 0.0
        resin_vol = 0.0
        for c in spec.components:
            rho = c.density.magnitude("g/cm^3") if c.density is not None else None
            if c.role == ComponentRole.FILLER:
                cons = _lookup_constituent(c.name + " " + (c.notes or ""))
                if rho is None:
                    rho = cons.density_g_cm3
                    notes.append(f"{c.name}: density not given, used {rho} g/cm^3 ({cons.key} default)")
                d50 = c.particle_d50.magnitude("um") if c.particle_d50 is not None else None
                fillers.append({"material": cons.key, "volume": c.weight_fraction / rho, "d50_um": d50, "name": c.name})
            else:
                resin_mass += c.weight_fraction
                if rho is None:
                    rho = CONSTITUENTS["resin"].density_g_cm3
                resin_vol += c.weight_fraction / rho
        vtot = resin_vol + sum(f["volume"] for f in fillers)
        out = {"fillers": [], "matrix": "resin", "notes": notes + [
            "volume fractions derived from weight fractions and densities; organic components without a density use "
            f"{CONSTITUENTS['resin'].density_g_cm3} g/cm^3 (assumed)"]}
        for f in fillers:
            out["fillers"].append({"material": f["material"], "volume_fraction": f["volume"] / vtot, "d50_um": f["d50_um"],
                                   "name": f["name"]})
        return out
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a dict, Formulation or Candidate")
    return dict(spec)


def _mt_two_scale(matrix: IsotropicPhase, micro: list[tuple[IsotropicPhase, float]],
                  nano: list[tuple[IsotropicPhase, float]]) -> tuple[float, float, float, IsotropicPhase]:
    """Two-scale MT: nano fillers into the matrix, then micro fillers.  Returns (K, G, phi_micro, matrix_eff)."""
    phi_micro = sum(f for _, f in micro)
    phi_nano = sum(f for _, f in nano)
    if phi_micro + phi_nano >= 1.0:
        raise ValueError("total filler volume fraction must be < 1")
    frac_in_matrix = [f / (1.0 - phi_micro) for _, f in nano]
    m_eff = nanofilled_matrix(matrix, [p for p, _ in nano], frac_in_matrix)
    if micro:
        K, G = mori_tanaka_spheres(m_eff, [p for p, _ in micro], [f for _, f in micro])
    else:
        K, G = m_eff.K, m_eff.G
    return K, G, phi_micro, m_eff


def _load_fft_correction(path: Path = RESULTS_PATH) -> Optional[dict]:
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data.get("fft_over_mori_tanaka_correction")
    except (OSError, ValueError):
        return None


def effective_modulus(spec: Any, *, use_fft_correction: bool = True) -> tuple[float, float, dict, str]:
    """Effective isotropic Young's modulus (GPa) and Poisson ratio of a particle-filled resin.

    ``spec`` is a dict such as::

        {"fillers": [{"material": "ba_glass", "volume_fraction": 0.45, "d50_um": 0.7},
                     {"material": "zirconia", "volume_fraction": 0.08, "d50_um": 0.3},
                     {"material": "silica", "volume_fraction": 0.05, "d50_um": 0.04}],
         "matrix": "resin",               # or {"E": 2.8, "nu": 0.35} (GPa)
         "matrix_range": {"E": [2.0, 3.5], "nu": [0.3, 0.4]}}   # optional override

    a schema ``Formulation`` (weight fractions converted with densities) or a
    ``Candidate`` holding one.  Material keys: ``ba_glass``, ``silica``,
    ``zirconia`` (or free text containing glass / silica / zirconia), or a dict
    ``{"E": .., "nu": .., "E_range": [lo, hi]}``.

    Model: two-scale Mori-Tanaka (fillers with d50 < 0.1 um go into a
    nanofilled matrix first), multiplied by the FFT/MT correction factor
    tabulated vs micro-filler volume fraction in
    data/dental/homogenization_results.json (full-field FFT on random sphere
    packings; if the file is absent the pure MT value is returned).
    Uncertainty: constituent min/nominal/max propagated over all corners, the
    FFT-correction spread, and the HS bounds for reference.
    Returns ``(E_GPa, nu, uncertainty, method)``.
    """
    s = _normalise_spec(spec)
    mat = s.get("matrix", "resin")
    if isinstance(mat, Mapping):
        m_nom = (float(mat["E"]), float(mat["nu"]))
        m_rng = s.get("matrix_range", {"E": [mat["E"], mat["E"]], "nu": [mat["nu"], mat["nu"]]})
    else:
        cons = CONSTITUENTS[mat]
        m_nom = (cons.E[1], cons.nu[1])
        m_rng = s.get("matrix_range", {"E": [cons.E[0], cons.E[2]], "nu": [cons.nu[0], cons.nu[2]]})
    fil = []  # (E_triple, nu_triple, fraction, is_nano, key)
    for f in s.get("fillers", []):
        vf = float(f["volume_fraction"])
        if vf <= 0:
            continue
        m = f["material"]
        if isinstance(m, Mapping):
            Er = m.get("E_range", [m["E"], m["E"]])
            nr = m.get("nu_range", [m["nu"], m["nu"]])
            Et, nt, key = (Er[0], m["E"], Er[1]), (nr[0], m["nu"], nr[1]), "custom"
        else:
            c = _lookup_constituent(m)
            Et, nt, key = c.E, c.nu, c.key
        d50 = f.get("d50_um")
        nano = bool(f.get("nano", d50 is not None and d50 < NANO_D50_UM))
        fil.append((Et, nt, vf, nano, key))
    phi_total = sum(x[2] for x in fil)

    def evaluate(Em, num, Efs, nufs):
        matrix = IsotropicPhase("resin", Em, num)
        micro = [(IsotropicPhase(x[4], Ef, nuf), x[2]) for x, Ef, nuf in zip(fil, Efs, nufs) if not x[3]]
        nano = [(IsotropicPhase(x[4], Ef, nuf), x[2]) for x, Ef, nuf in zip(fil, Efs, nufs) if x[3]]
        K, G, phi_micro, _ = _mt_two_scale(matrix, micro, nano)
        return K, G, phi_micro

    K, G, phi_micro = evaluate(m_nom[0], m_nom[1], [x[0][1] for x in fil], [x[1][1] for x in fil])
    E_mt, nu_mt = E_nu_from_KG(K, G)
    # constituent corners (monotone in E; nu handled by enumeration)
    E_corners = []
    nu_corners = []
    choices = [(m_rng["E"][0], m_rng["E"][1]), (m_rng["nu"][0], m_rng["nu"][1])]
    for x in fil:
        choices.append((x[0][0], x[0][2]))
        choices.append((x[1][0], x[1][2]))
    for combo in itertools.product(*choices):
        Em, num = combo[0], combo[1]
        Efs = combo[2::2]
        nufs = combo[3::2]
        k_, g_, _ = evaluate(Em, num, Efs, nufs)
        e_, n_ = E_nu_from_KG(k_, g_)
        E_corners.append(e_)
        nu_corners.append(n_)
    # HS bounds (single-scale, all phases, nominal)
    phases = [IsotropicPhase("resin", *m_nom)] + [IsotropicPhase(x[4], x[0][1], x[1][1]) for x in fil]
    fr = [1.0 - phi_total] + [x[2] for x in fil]
    hs = hashin_shtrikman_bounds(phases, fr)
    E_hs_lo = E_nu_from_KG(*hs["lower"])[0]
    E_hs_hi = E_nu_from_KG(*hs["upper"])[0]

    corr = _load_fft_correction() if use_fft_correction else None
    r_nom, r_lo, r_hi, method = 1.0, 1.0, 1.0, "two-scale Mori-Tanaka (spheres); no FFT correction available"
    if corr is not None and phi_micro > 0:
        # anchor r(0) = 1 (dilute limit: FFT -> MT), verified by the dilute-sphere check
        ph = np.concatenate([[0.0], np.array(corr["phi_micro"], dtype=float)])
        rn = np.concatenate([[1.0], np.array(corr["ratio_nominal"], dtype=float)])
        rl = np.concatenate([[1.0], np.array(corr["ratio_min"], dtype=float)])
        rh = np.concatenate([[1.0], np.array(corr["ratio_max"], dtype=float)])
        r_nom = float(np.interp(phi_micro, ph, rn))
        r_lo = float(np.interp(phi_micro, ph, rl))
        r_hi = float(np.interp(phi_micro, ph, rh))
        extrap = phi_micro > ph.max() + 1e-9
        method = ("two-scale Mori-Tanaka x FFT/MT correction (full-field FFT-Galerkin CG on random periodic sphere "
                  f"packings, phi_micro {ph.min():.2f}-{ph.max():.2f}"
                  + ("; phi outside table -> end value used" if extrap else "") + ")")
    E_nom = E_mt * r_nom
    nu_nom = nu_mt
    unc = {
        "E_min": float(min(E_corners) * min(r_lo, r_nom)),
        "E_max": float(max(E_corners) * max(r_hi, r_nom)),
        "nu_min": float(min(nu_corners)),
        "nu_max": float(max(nu_corners)),
        "E_mori_tanaka_nominal": float(E_mt),
        "E_constituent_range_MT": [float(min(E_corners)), float(max(E_corners))],
        "fft_correction": {"nominal": r_nom, "min": r_lo, "max": r_hi},
        "E_hashin_shtrikman_nominal": [float(E_hs_lo), float(E_hs_hi)],
        "phi_total": float(phi_total),
        "phi_micro": float(phi_micro),
        "interval": "range (min/max over constituent corners x FFT-correction spread); not a statistical CI",
        "notes": list(s.get("notes", [])) + [
            "Ba-glass E/nu and resin nu are assumed ranges (not verified literature values); see CONSTITUENTS.",
            "Estimate is the static Young's modulus of a dry, well-bonded, void-free composite; measured flexural "
            "moduli after water storage are typically lower.",
        ],
    }
    return float(E_nom), float(nu_nom), unc, method


def effective_modulus_records(candidate, spec: Any = None, *, checks: Sequence = ()) -> list:
    """PropertyRecords (EFFECTIVE_YOUNGS_MODULUS, EFFECTIVE_POISSON_RATIO) for a formulation candidate."""
    from fracture_md.records import make_record
    from fracture_md.schema.common import Uncertainty, UncertaintyKind
    from fracture_md.schema.properties import Derivation, MethodSpec, PropertyName, PropertyOrigin, PropertyScale
    from fracture_md.units import Unit

    E, nu, unc, method = effective_modulus(spec if spec is not None else candidate)
    assumptions = [
        "isotropic linear-elastic phases; perfect bonding (no interphase) unless stated",
        "spherical particles, statistically isotropic dispersion; nano-fillers (d50 < 0.1 um) homogenised first",
        "constituent moduli from composites.homogenization.CONSTITUENTS (Ba-glass E and resin nu are assumed ranges)",
    ]
    ms = MethodSpec(name="composite homogenisation (two-scale MT x FFT correction)", engine="fracture_md.composites",
                    description=method, assumptions=assumptions)
    der = Derivation(model=method, equation="E = E_MT(two-scale) * r_FFT(phi_micro)", assumptions=assumptions)
    recs = [
        make_record(name=PropertyName.EFFECTIVE_YOUNGS_MODULUS, value=E, unit=Unit.GPA, candidate=candidate, method=ms,
                    origin=PropertyOrigin.DERIVED_MODEL, scale=PropertyScale.MESOSCALE, derivation=der, checks=checks,
                    uncertainty=[Uncertainty(kind=UncertaintyKind.UNQUANTIFIED, interval="range", unit=Unit.GPA,
                                             samples=[unc["E_min"], unc["E_max"]],
                                             description="constituent min/max corners x FFT-correction spread")],
                    notes=json.dumps({k: unc[k] for k in ("E_mori_tanaka_nominal", "E_hashin_shtrikman_nominal",
                                                          "phi_total", "phi_micro")})),
        make_record(name=PropertyName.EFFECTIVE_POISSON_RATIO, value=nu, unit=Unit.DIMENSIONLESS, candidate=candidate,
                    method=ms, origin=PropertyOrigin.DERIVED_MODEL, scale=PropertyScale.MESOSCALE, derivation=der,
                    checks=checks,
                    uncertainty=[Uncertainty(kind=UncertaintyKind.UNQUANTIFIED, interval="range",
                                             samples=[unc["nu_min"], unc["nu_max"]],
                                             description="constituent min/max corners")]),
    ]
    return recs


# ---------------------------------------------------------------------------
# Verification suite and production driver
#   python -m fracture_md.composites.homogenization [--quick] [--n 96] [--out PATH]
# (GPU: docker exec -e FRACTURE_MD_ALLOW_GPU=1 ... fracture-md-gpu python -m ...)
# ---------------------------------------------------------------------------
LITERATURE_MODULI = [
    # (label, filler vol fraction, E_measured GPa [lo, hi], source key, notes) - all values quoted in RESEARCH_BRIEF.md
    ("OnX Tough 2 (printed, wet 36 C)", 0.22, [2.3, 2.3], "[S17]", "about 35 wt% / about 22 vol% measured; 3-pt flexural, wet"),
    ("printed crown resins (Tayeb 2025)", 0.20, [2.9, 3.2], "[S15]", "about 33 wt% ash, vol% estimated about 20 by the brief; ISO 4049 Ef"),
    ("SprintRay Ceramic Crown (indep.)", None, [5.35, 5.35], "[C12]", "filler vol% not published"),
    ("'Magnus' experimental printable", 0.53, [7.4, 7.4], "[S17]", "about 70 wt% / 53 vol%; wet 36 C, 3-pt"),
    ("SprintRay Midas Restore (Mfr)", 0.56, [10.5, 10.5], "[S3]", ">75 wt%; 56 vol% is the brief's estimate with assumed densities (§4.4); method not stated"),
    ("Filtek Supreme Ultra (direct)", 0.633, [11.35, 11.35], "[C17]", "78.5 wt% / 63.3 vol% nanoclusters"),
    ("Lava Ultimate (milled RNC)", 0.65, [12.77, 12.77], "[C14]", "about 80 wt% / 65 vol% silica+zirconia nanoclusters; industrially cured block"),
    ("VITA Enamic (PICN)", 0.75, [30.0, 30.0], "[C15]", "interpenetrating ceramic network - outside the particulate model (expected to exceed it)"),
]


def _phase_set(which: str, nano_matrix: Optional[tuple[float, str]] = None) -> dict[str, IsotropicPhase]:
    i = {"min": 0, "nominal": 1, "max": 2}[which]
    resin = IsotropicPhase("resin", CONSTITUENTS["resin"].E[i], CONSTITUENTS["resin"].nu[1])
    out = {k: IsotropicPhase(k, CONSTITUENTS[k].E[i], CONSTITUENTS[k].nu[1]) for k in ("ba_glass", "silica", "zirconia")}
    if nano_matrix is not None:
        frac, key = nano_matrix
        out[MATRIX_PHASE] = nanofilled_matrix(resin, [out[key]], [frac])
    else:
        out[MATRIX_PHASE] = IsotropicPhase(MATRIX_PHASE, resin.E, resin.nu)
    return out


def _mt_for_micro(v: VoxelMicrostructure, ph: Mapping[str, IsotropicPhase]) -> tuple[float, float]:
    incl = [k for k in v.phase_names if k != MATRIX_PHASE]
    return E_nu_from_KG(*mori_tanaka_spheres(ph[MATRIX_PHASE], [ph[k] for k in incl], [v.volume_fractions[k] for k in incl]))


def _hs_for_micro(v: VoxelMicrostructure, ph: Mapping[str, IsotropicPhase]) -> tuple[float, float, dict]:
    hs = hashin_shtrikman_bounds([ph[k] for k in v.phase_names], [v.volume_fractions[k] for k in v.phase_names])
    return E_nu_from_KG(*hs["lower"])[0], E_nu_from_KG(*hs["upper"])[0], hs


def _within_hs(res: FFTResult, hs: dict, slack: float = 1e-6) -> bool:
    m = res.metrics
    (klo, glo), (khi, ghi) = hs["lower"], hs["upper"]
    return (klo * (1 - slack) <= m["K_hill"] <= khi * (1 + slack)) and (glo * (1 - slack) <= m["G_hill"] <= ghi * (1 + slack))


def _packing(fams, seed, gap, n_particles):
    p = pack_spheres(fams, n_particles=n_particles, seed=seed, min_gap=gap)
    if not p.converged and gap > 0:
        p2 = pack_spheres(fams, n_particles=n_particles, seed=seed, min_gap=0.0)
        p2.notes.append(f"gap {gap} infeasible (max rel. overlap {p.max_relative_overlap:.3g}); repacked with gap 0")
        p = p2
    return p


def sphere_convergence_check(conv: Sequence[Mapping]) -> tuple[bool, float, str]:
    """Resolution convergence of E_FFT/E_MT for the single sphere: the change per unit h must not grow
    (at least first-order convergence) and the last change must be < 1%."""
    ratios = [c["ratio_fft_mt"] for c in conv]
    hs = [1.0 / c["n"] for c in conv]
    s1 = abs(ratios[-2] - ratios[-3]) / (hs[-3] - hs[-2])
    s2 = abs(ratios[-1] - ratios[-2]) / (hs[-2] - hs[-1])
    d2 = abs(ratios[-1] - ratios[-2])
    ok = bool(s2 <= 1.1 * s1 and d2 < 0.01)
    det = ("E_FFT/E_MT (voxel phi) at n=" + ",".join(str(c["n"]) for c in conv) + ": " + ", ".join(f"{x:.4f}" for x in ratios)
           + f"; |d ratio|/dh = {s1:.3f}, {s2:.3f} (non-increasing => >= first-order convergence); last change {d2:.4f}. "
           "SC array is cubic, so the converged ratio is not exactly 1.")
    return ok, d2, det


def run_verification(n_small: int = 32, n_list: Sequence[int] = (32, 64, 96), quick: bool = False) -> dict:
    """Analytic/benchmark checks of the solvers.  Returns a JSON-serialisable dict of checks."""
    m = IsotropicPhase("resin", 2.5, 0.35)
    g = IsotropicPhase("ba_glass", 70.0, 0.22)
    z = IsotropicPhase("zirconia", 205.0, 0.31)
    checks = []

    def add(name, passed, metric, tol, details, data=None):
        checks.append({"name": name, "passed": bool(passed), "metric": float(metric), "tolerance": float(tol),
                       "details": details, **({"data": data} if data else {})})

    # 1 homogeneous
    r = fft_homogenize(laminate(n_small, [0.5, 0.5]), [g, g])
    err = float(np.abs(r.C_eff - g.stiffness()).max() / np.abs(g.stiffness()).max())
    add("homogeneous_returns_itself", err < 1e-12, err, 1e-12, f"max rel. error, iterations {r.iterations}")
    # 2 laminates (2- and 3-phase, all axes, both schemes)
    worst = 0.0
    for axis in (0, 1, 2):
        for fr, ph in (([0.375, 0.625], [m, g]), ([0.25, 0.5, 0.25], [m, g, z])):
            lab = laminate(n_small, fr, axis=axis)
            exact = laminate_stiffness([p.stiffness() for p in ph], [lab.volume_fractions[k] for k in lab.phase_names], axis)
            for scheme in ("cg",) + (() if quick else ("basic",)):
                rr = fft_homogenize(lab, ph, scheme=scheme, tol=1e-11, max_iter=20000)
                worst = max(worst, float(np.abs(rr.C_eff - exact).max() / np.abs(exact).max()))
    add("laminate_exact", worst < 1e-8, worst, 1e-8, "FFT vs exact laminate stiffness, 2-/3-phase, axes x/y/z, CG and basic schemes")
    # 3 analytic identities
    K, G = mori_tanaka_spheres(m, [g], [0.4])
    hs = hashin_shtrikman_bounds([m, g], [0.6, 0.4])
    e = max(abs(K - hs["lower"][0]) / K, abs(G - hs["lower"][1]) / G)
    add("mori_tanaka_equals_HS_lower_two_phase", e < 1e-12, e, 1e-12, "MT (stiff spheres in soft matrix) == HS lower bound")
    Kd, Gd = dilute_eshelby_spheres(m, [g], [1e-4])
    Km, Gm = mori_tanaka_spheres(m, [g], [1e-4])
    e = abs((Kd - m.K) - (Km - m.K)) / (Km - m.K)
    add("mori_tanaka_to_dilute_limit", e < 1e-3, e, 1e-3, "MT and dilute Eshelby agree to first order at c=1e-4")
    # 4 dilute sphere FFT vs Eshelby
    nd = n_list[-1]
    s = single_sphere(nd, 0.02)
    rr = fft_homogenize(s, [m, g], tol=1e-8)
    phi = s.volume_fractions["inclusion"]
    Kd, Gd = dilute_eshelby_spheres(m, [g], [phi])
    Km, Gm = mori_tanaka_spheres(m, [g], [phi])
    rel_K = (rr.metrics["K_voigt"] - m.K) / (Km - m.K) - 1.0
    rel_G = (rr.metrics["G_voigt"] - m.G) / (Gm - m.G) - 1.0
    add("dilute_sphere_matches_MT", max(abs(rel_K), abs(rel_G)) < 0.08, max(abs(rel_K), abs(rel_G)), 0.08,
        f"single sphere phi={phi:.4f}, n={nd}: relative error of the stiffening (K,G) vs MT; dilute Eshelby K,G = {Kd:.5f},{Gd:.5f}",
        {"K_fft": rr.metrics["K_voigt"], "G_fft": rr.metrics["G_voigt"], "K_mt": Km, "G_mt": Gm, "K_dilute": Kd, "G_dilute": Gd})
    # 5 resolution convergence, single sphere phi=0.2 (SC array)
    conv = []
    for n in n_list:
        s = single_sphere(n, 0.2)
        rr = fft_homogenize(s, [m, g], tol=1e-8)
        phi = s.volume_fractions["inclusion"]
        Emt = E_nu_from_KG(*mori_tanaka_spheres(m, [g], [phi]))[0]
        hs = hashin_shtrikman_bounds([m, g], [1 - phi, phi])
        conv.append({"n": n, "phi_voxel": phi, "E_hill": rr.metrics["E_hill"], "E_MT": Emt, "ratio_fft_mt": rr.metrics["E_hill"] / Emt,
                     "C11": rr.C_eff[0, 0], "C12": rr.C_eff[0, 1], "C44": rr.C_eff[3, 3], "iterations": rr.iterations,
                     "within_HS": _within_hs(rr, hs), "symmetry_error": rr.symmetry_error,
                     "cubic_anisotropy_zener": rr.metrics["zener_cubic_proxy"]})
    ok, metric, det = sphere_convergence_check(conv)
    add("single_sphere_resolution_convergence", ok, metric, 0.01, det, {"table": conv})
    add("single_sphere_within_HS", all(c["within_HS"] for c in conv), 0.0, 0.0, "all resolutions inside HS bounds")
    # 6 CG vs basic scheme
    s = single_sphere(n_small, 0.2)
    rc = fft_homogenize(s, [m, g], tol=1e-10)
    rb = fft_homogenize(s, [m, g], scheme="basic", tol=1e-11, max_iter=20000)
    e = float(np.abs(rc.C_eff - rb.C_eff).max() / np.abs(rc.C_eff).max())
    add("cg_vs_basic_scheme", e < 1e-6, e, 1e-6, f"iterations CG {rc.iterations} vs basic {rb.iterations}")
    add("stiffness_major_symmetry", rc.symmetry_error < 1e-8, rc.symmetry_error, 1e-8, "||C-C^T||/||C|| of the unsymmetrised CG result")
    # 7 random packing: isotropy, HS, seed scatter
    ph = _phase_set("nominal")
    iso = []
    for seed in ((0,) if quick else (0, 1, 2)):
        p = _packing([SphereFamily("glass", "ba_glass", 0.7, 0.45, 0.15)], seed, 0.025, 40 if quick else 60)
        v = voxelize(p, n_list[1] if not quick else n_small)
        rr = fft_homogenize(v, ph, tol=1e-6)
        elo, ehi, hsd = _hs_for_micro(v, ph)
        iso.append({"seed": seed, "phi_voxel": 1 - v.volume_fractions[MATRIX_PHASE], "E_hill": rr.metrics["E_hill"],
                    "nu_hill": rr.metrics["nu_hill"],
                    "A_U": rr.metrics["universal_anisotropy_index"], "dev_iso": rr.metrics["relative_deviation_from_isotropy"],
                    "E_axis_spread_rel": rr.metrics["E_axis_spread_rel"], "within_HS": _within_hs(rr, hsd),
                    "E_HS": [elo, ehi], "n": v.n, "n_particles": p.n_particles})
    worst_dev = max(x["dev_iso"] for x in iso)
    add("random_packing_isotropy", worst_dev < 0.08, worst_dev, 0.08,
        "relative tensorial deviation from the isotropic projection, phi=0.45 glass, " + ", ".join(
            f"seed {x['seed']}: A_U={x['A_U']:.4f}, dev={x['dev_iso']:.4f}, E-axis spread={x['E_axis_spread_rel']:.3f}" for x in iso),
        {"table": iso})
    add("random_packing_within_HS", all(x["within_HS"] for x in iso), 0.0, 0.0, "K_hill, G_hill inside HS bounds for every seed")
    return {"checks": checks, "all_passed": all(c["passed"] for c in checks)}


def _case_micro(case: dict, n: int, seed: int, gap: float):
    fams = case["families"]
    p = _packing(fams, seed, gap, case.get("n_particles", 60))
    return voxelize(p, n)


def _run_case(case: dict, n: int, seed: int, gap: float, which_list=("nominal",)):
    v = _case_micro(case, n, seed, gap)
    out = {"seed": seed, "n": n, "voxel_volume_fractions": v.volume_fractions, "packing": v.packing.summary() if v.packing else None,
           "runs": {}}
    nano = case.get("nano")  # (fraction_in_matrix, key)
    for which in which_list:
        ph = _phase_set(which, nano)
        rr = fft_homogenize(v, ph, tol=1e-6)
        Emt, numt = _mt_for_micro(v, ph)
        elo, ehi, hsd = _hs_for_micro(v, ph)
        out["runs"][which] = {"E_fft": rr.metrics["E_hill"], "nu_fft": rr.metrics["nu_hill"], "E_MT": Emt, "nu_MT": numt,
                              "E_HS_lower": elo, "E_HS_upper": ehi, "within_HS": _within_hs(rr, hsd),
                              "ratio_fft_mt": rr.metrics["E_hill"] / Emt, "A_U": rr.metrics["universal_anisotropy_index"],
                              "dev_iso": rr.metrics["relative_deviation_from_isotropy"], "iterations": rr.iterations,
                              "max_residual": max(rr.residuals), "symmetry_error": rr.symmetry_error,
                              "wall_time_s": rr.wall_time_s, "device": rr.device,
                              "matrix_E_eff": ph[MATRIX_PHASE].E, "C_eff_GPa": np.round(rr.C_eff, 5).tolist()}
    return out


def production_cases(quick: bool = False) -> list[dict]:
    cases = []
    phis = (0.30, 0.45) if quick else (0.30, 0.40, 0.50, 0.55, 0.60)
    for phi in phis:
        cases.append({"id": f"glass_mono_phi{phi:.2f}", "group": "glass_mono", "phi_total": phi, "phi_zirconia": 0.0,
                      "families": [SphereFamily("glass", "ba_glass", 0.7, phi, 0.15)], "n_particles": 60, "minmax_fft": True})
    for phi in ((0.50,) if quick else (0.40, 0.50, 0.60, 0.65)):
        phi_nano = 0.15 * phi
        phi_micro = phi - phi_nano
        cases.append({"id": f"glass_plus_nanosilica_phi{phi:.2f}", "group": "two_scale_bimodal", "phi_total": phi, "phi_zirconia": 0.0,
                      "families": [SphereFamily("glass", "ba_glass", 0.7, phi_micro, 0.15)], "n_particles": 60,
                      "nano": (phi_nano / (1.0 - phi_micro), "silica"), "minmax_fft": False,
                      "description": "0.7 um glass resolved + 40 nm silica (15% of filler volume) in a MT nanofilled matrix"})
    for fz in ((0.10,) if quick else (0.05, 0.10, 0.15)):
        for phi in ((0.50,) if quick else (0.40, 0.50, 0.55, 0.60)):
            cases.append({"id": f"glass_zr{int(round(fz * 100)):02d}_phi{phi:.2f}", "group": f"glass_zirconia_{int(round(fz * 100))}",
                          "phi_total": phi, "phi_zirconia": fz,
                          "families": [SphereFamily("glass", "ba_glass", 0.7, phi - fz, 0.15),
                                       SphereFamily("zr", "zirconia", 0.3, fz, 0.15)],
                          "n_particles": 120, "minmax_fft": abs(fz - 0.10) < 1e-9})
    return cases


def richardson_observed_order(table: Sequence[Mapping]) -> dict:
    """Observed convergence order p and extrapolated value from the last three (n, E_fft) points (h = 1/n)."""
    t = sorted(table, key=lambda r: r["n"])[-3:]
    if len(t) < 3:  # two points: first order
        (nb, eb), (nc, ec) = [(r["n"], r["E_fft"]) for r in t]
        b, c = 1.0 / nb, 1.0 / nc
        return {"order": 1.0, "E0": float(ec - (eb - ec) * c / (b - c))}
    (na, ea), (nb, eb), (nc, ec) = [(r["n"], r["E_fft"]) for r in t]
    a, b, c = 1.0 / na, 1.0 / nb, 1.0 / nc
    R = (ea - eb) / (eb - ec) if eb != ec else np.inf
    f = lambda p: (a**p - b**p) / (b**p - c**p) - R  # noqa: E731
    lo, hi = 0.2, 4.0
    if not np.isfinite(R) or f(lo) * f(hi) > 0:
        p = 1.0
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
        p = 0.5 * (lo + hi)
    E0 = ec - (eb - ec) * c**p / (b**p - c**p)
    return {"order": float(p), "E0": float(E0)}


def correction_table(result: dict, n_prod: int) -> dict:
    """FFT/MT ratio vs micro-filler fraction, corrected for grid resolution (first-order Richardson).

    The relative discretisation error err(phi) = E(n_prod)/E(h->0) - 1 measured in the resolution study is
    interpolated linearly through (0, 0) and the study cases, and every production ratio is divided by
    (1 + err).  The band spans [corrected - 2 seed std, uncorrected n_prod value + 2 seed std].
    """
    res, res1 = [], []
    for r in result.get("resolution_study", []):
        if r.get("rel_error_at_production_n") is None:
            continue
        ex = richardson_observed_order(r["table"])
        r["observed_order"] = ex["order"]
        r["E_extrapolated_observed_order"] = ex["E0"]
        e_prod = [t["E_fft"] for t in r["table"] if t["n"] == n_prod][0]
        res.append((r["phi_micro"], e_prod / ex["E0"] - 1.0))  # nominal: observed-order extrapolation
        res1.append((r["phi_micro"], r["rel_error_at_production_n"]))  # first order (larger correction)
    pts = sorted([(0.0, 0.0)] + res)
    pts1 = sorted([(0.0, 0.0)] + res1)
    xp = np.array([p[0] for p in pts])
    fp = np.array([p[1] for p in pts])
    xp1 = np.array([p[0] for p in pts1])
    fp1 = np.array([p[1] for p in pts1])
    seed_rel = max([0.0] + [s["rel_std"] for s in result.get("seed_scatter", [])])
    by_phi: dict[float, list[tuple[float, float]]] = {}
    excluded = []
    for row in result["cases"]:
        notes = " ".join((row.get("packing") or {}).get("notes", []))
        if "infeasible" in notes or not (row.get("packing") or {}).get("converged", True):
            excluded.append(row["id"])  # contact-dominated packing (gap could not be enforced)
            continue
        key = round(row["phi_micro_voxel"], 2)
        err = float(np.interp(row["phi_micro_voxel"], xp, fp))
        err1 = float(np.interp(row["phi_micro_voxel"], xp1, fp1))
        for run in row["runs"].values():
            if "ratio_fft_mt" in run:
                r_ = run["ratio_fft_mt"]
                by_phi.setdefault(key, []).append((r_ / (1.0 + err), r_ / (1.0 + err1), r_))
    ks = sorted(by_phi)
    return {
        "phi_micro": ks,
        "ratio_nominal": [float(np.mean([c for c, _, _ in by_phi[k]])) for k in ks],
        "ratio_min": [float(max(1.0, min(c1 for _, c1, _ in by_phi[k]) * (1 - 2 * seed_rel))) for k in ks],
        "ratio_max": [float(max(r for _, _, r in by_phi[k]) * (1 + 2 * seed_rel)) for k in ks],
        "ratio_uncorrected_n": [float(np.mean([r for _, _, r in by_phi[k]])) for k in ks],
        "resolution_error_points_observed_order": [list(p) for p in pts],
        "resolution_error_points_first_order": [list(p) for p in pts1],
        "seed_rel_std_max": seed_rel,
        "excluded_cases": excluded,
        "n_production": n_prod,
        "notes": "E_FFT(Hill)/E_MT at the voxel volume fractions, pooled over cases and constituent sets per phi bin "
                 "(0.01). Nominal = resolution-corrected with observed-order Richardson extrapolation from the 64/96/128 "
                 "study (error interpolated linearly in phi through 0 at phi=0); min uses the larger first-order "
                 "correction and 2 seed std; max = uncorrected n_prod value + 2 seed std; min >= 1 because MT is the "
                 "HS lower bound here.",
    }


def literature_sanity_check(result: dict) -> dict:
    """Compare predicted E (glass-only, nominal and range) with published composite moduli (RESEARCH_BRIEF.md)."""
    lit = []
    for label, phi, (lo, hi), src, note in LITERATURE_MODULI:
        row = {"material": label, "phi": phi, "E_measured_GPa": [lo, hi], "source": src, "notes": note}
        if phi is not None:
            E_mt, _, unc, _ = effective_modulus({"fillers": [{"material": "ba_glass", "volume_fraction": phi, "d50_um": 0.7}]},
                                                use_fft_correction=False)
            lo_mt, hi_mt = unc["E_constituent_range_MT"]
            _, _, unc_c, _ = effective_modulus({"fillers": [{"material": "ba_glass", "volume_fraction": phi, "d50_um": 0.7}]})
            rn, rl, rh = (unc_c["fft_correction"][k] for k in ("nominal", "min", "max"))
            row.update({"E_MT_nominal": E_mt, "E_MT_range": [lo_mt, hi_mt], "E_fft_corrected_nominal": E_mt * rn,
                        "E_fft_corrected_range": [lo_mt * rl, hi_mt * rh], "E_HS": unc["E_hashin_shtrikman_nominal"],
                        "measured_within_predicted_range": bool(lo_mt * rl <= hi and lo <= hi_mt * rh)})
        lit.append(row)
    return {
        "rows": lit,
        "notes": "Measured values are mostly 3-point flexural moduli (often after water storage), which are lower than the "
                 "static Young's modulus of a dry, perfectly bonded composite predicted here; vol% for several products are "
                 "estimates. Glass-only constituents assumed for every row (nanocluster/zirconia products differ).",
    }


def refresh_derived(path: Path = RESULTS_PATH, n_prod: Optional[int] = None) -> dict:
    """Recompute the correction table and literature comparison of an existing results file."""
    with open(path) as fh:
        result = json.load(fh)
    phi_by_id = {c["id"]: c["phi_total"] - (0.15 * c["phi_total"] if c["group"] == "two_scale_bimodal" else 0.0)
                 for c in result["cases"]}
    for r in result.get("resolution_study", []):
        r.setdefault("phi_micro", phi_by_id.get(r["id"]))
    result["fft_over_mori_tanaka_correction"] = correction_table(result, n_prod or result["meta"]["grid_n"])
    result["literature_sanity_check"] = literature_sanity_check(result)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    return result


def run_production(n: int = 96, gap: float = 0.025, quick: bool = False, out: Optional[Path] = None) -> dict:
    t0 = time.time()
    dev = jax.devices()[0]
    result: dict[str, Any] = {
        "meta": {
            "description": "Effective stiffness of particle-filled dimethacrylate resins: FFT (Galerkin-CG) on random periodic "
                           "sphere packings vs Mori-Tanaka / Hashin-Shtrikman, with constituent uncertainty.",
            "generated_by": "python -m fracture_md.composites.homogenization",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": f"{dev.platform}:{getattr(dev, 'device_kind', '')}", "jax_version": jax.__version__,
            "float_precision": "float64" if jax.config.jax_enable_x64 else "float32",
            "grid_n": n, "min_surface_gap_um": gap, "quick": quick,
            "units": {"E": "GPa", "nu": "1", "phi": "volume fraction (1)", "length": "um"},
            "constituents": {k: {"E_GPa_min_nom_max": list(c.E), "nu_min_nom_max": list(c.nu), "density_g_cm3": c.density_g_cm3,
                                 "status": c.status, "sources": list(c.sources), "notes": c.notes} for k, c in CONSTITUENTS.items()},
            "conventions": "Voigt order xx,yy,zz,yz,xz,xy with engineering shear; E, nu from the Hill average of the FFT C_eff",
            "assumptions": [
                "linear elastic isotropic phases, perfect bonding (no interphase) unless stated, no porosity",
                "spheres with log-normal diameters (cv 0.15); glass d50 0.7 um, zirconia d50 0.3 um",
                f"minimum surface gap {gap} um enforced in packing (repacked with 0 gap if infeasible - flagged)",
                "nanofillers homogenised into the matrix by Mori-Tanaka (two-scale)",
                "min/max runs use all-min or all-max constituent E simultaneously (E_eff is monotone in phase moduli)",
            ],
        }
    }
    vn = (48, 64, 96) if not quick else (16, 24, 32)
    result["verification"] = run_verification(n_small=32 if not quick else 16, n_list=vn, quick=quick)
    print("verification all_passed:", result["verification"]["all_passed"], f"({time.time() - t0:.0f} s)", flush=True)

    cases = production_cases(quick)
    table = []
    for case in cases:
        which = ("min", "nominal", "max") if case.get("minmax_fft") else ("nominal",)
        rc = _run_case(case, n if not quick else 32, 0, gap, which)
        row = {k: case[k] for k in ("id", "group", "phi_total", "phi_zirconia")}
        row["description"] = case.get("description", "")
        row["phi_micro_voxel"] = 1.0 - rc["voxel_volume_fractions"][MATRIX_PHASE]
        row.update(rc)
        nom = rc["runs"]["nominal"]
        if "min" not in rc["runs"]:  # scale MT min/max by the nominal FFT/MT ratio
            v = None
            for w in ("min", "max"):
                ph = _phase_set(w, case.get("nano"))
                incl = [f.phase for f in case["families"]]
                vfs = [rc["voxel_volume_fractions"][k] for k in dict.fromkeys(incl)]
                Emt = E_nu_from_KG(*mori_tanaka_spheres(ph[MATRIX_PHASE], [ph[k] for k in dict.fromkeys(incl)], vfs))[0]
                row["runs"][w] = {"E_MT": Emt, "E_fft_scaled": Emt * nom["ratio_fft_mt"], "note": "MT x nominal FFT/MT ratio"}
        table.append(row)
        print(f"{case['id']}: phi_vox={row['phi_micro_voxel']:.3f} E_fft={nom['E_fft']:.3f} E_MT={nom['E_MT']:.3f} "
              f"ratio={nom['ratio_fft_mt']:.3f} HS=[{nom['E_HS_lower']:.2f},{nom['E_HS_upper']:.2f}] its={nom['iterations']} "
              f"({time.time() - t0:.0f} s)", flush=True)
    result["cases"] = table

    # statistical (seed) scatter and resolution study on two representative cases
    rep = [c for c in cases if c["id"] in ("glass_mono_phi0.50", "glass_zr10_phi0.55", "glass_mono_phi0.45", "glass_zr10_phi0.50")]
    seeds_out, res_out = [], []
    for case in rep:
        vals = []
        for seed in ((0, 1) if quick else (0, 1, 2)):
            rc = _run_case(case, n if not quick else 32, seed, gap)
            vals.append(rc["runs"]["nominal"]["E_fft"])
        seeds_out.append({"id": case["id"], "E_fft_by_seed": vals, "mean": float(np.mean(vals)),
                          "rel_std": float(np.std(vals, ddof=1) / np.mean(vals))})
        rs = []
        for nn in ((24, 32) if quick else (64, 96, 128)):
            rc = _run_case(case, nn, 0, gap)
            rs.append({"n": nn, "E_fft": rc["runs"]["nominal"]["E_fft"], "ratio_fft_mt": rc["runs"]["nominal"]["ratio_fft_mt"]})
        # Richardson-type extrapolation assuming first-order convergence in h = 1/n
        e1, e2 = rs[-2]["E_fft"], rs[-1]["E_fft"]
        h1, h2 = 1.0 / rs[-2]["n"], 1.0 / rs[-1]["n"]
        e0 = e2 - (e1 - e2) * h2 / (h1 - h2)
        res_out.append({"id": case["id"], "phi_micro": float(sum(f.volume_fraction for f in case["families"])),
                        "table": rs, "E_extrapolated_h0": e0,
                        "rel_error_at_production_n": float([r["E_fft"] for r in rs if r["n"] == (n if not quick else 32)][0] / e0 - 1.0)
                        if any(r["n"] == (n if not quick else 32) for r in rs) else None})
        print("seed/resolution", case["id"], seeds_out[-1], res_out[-1], flush=True)
    result["seed_scatter"] = seeds_out
    result["resolution_study"] = res_out

    # interphase sensitivity (soft 50 nm shell, E = 0.5 E_resin) on glass phi 0.45-0.50
    if not quick:
        case = [c for c in cases if c["id"] == "glass_mono_phi0.50"][0]
        p = _packing(case["families"], 0, 0.06, 60)
        rows = []
        for t_shell, e_fac in ((0.0, None), (0.05, 0.5), (0.05, 2.0)):
            v = voxelize(p, n, shell_thickness=t_shell)
            ph = dict(_phase_set("nominal"))
            if t_shell > 0:
                ph["interphase"] = IsotropicPhase("interphase", e_fac * CONSTITUENTS["resin"].E[1], 0.35)
            rr = fft_homogenize(v, ph, tol=1e-6)
            rows.append({"shell_um": t_shell, "shell_E_over_resin": e_fac, "E_fft": rr.metrics["E_hill"],
                         "voxel_volume_fractions": v.volume_fractions})
        result["interphase_sensitivity"] = {"case": "glass 0.7 um, phi 0.50, gap 0.06 um (shell must fit in gap)", "rows": rows}
        print("interphase", rows, flush=True)

    result["fft_over_mori_tanaka_correction"] = correction_table(result, n if not quick else 32)

    result["literature_sanity_check"] = literature_sanity_check(result)
    result["meta"]["wall_time_s"] = time.time() - t0
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(result, fh, indent=1, default=float)
        print("wrote", out, flush=True)
    return result


if __name__ == "__main__":  # pragma: no cover - production driver
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--gap", type=float, default=0.025)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=str, default=str(RESULTS_PATH))
    a = ap.parse_args()
    run_production(n=a.n, gap=a.gap, quick=a.quick, out=Path(a.out))
