"""Finite-cell (box) factor of an MD indentation cell: periodic, finite-thickness, anisotropic elastic contact.

An MD indentation cell is a laterally periodic slab (period L) of elastic thickness t_el on a frozen bottom
layer.  At nm contact radii it is STIFFER than the half-space of contact mechanics (periodic images, rigid
base): G10 measured +17-22 % on the apparent sphere modulus, G06/G08 estimate +31 % on the Berkovich p_m at
1 nm in a (4.6, 2.5) cell.  This module is the harness port of gapfill/G10/bem/box_bem.py:

* the exact surface Green's function of a cubic ANISOTROPIC layer (x = [100], y = [010], z = [001] of the MD
  cells; x = [110] cells are the same cubic layer rotated by 45 deg, which only rotates the kernel) of
  thickness t bonded to a rigid base, laterally periodic, in Fourier space: the 6x6 Stroh state-vector ODE
  d/dz [U; S] = A(k, phi) [U; S] integrated with a matrix exponential (U(-t) = 0, S(0) = -p e3); the G = 0
  term is the uniaxial-strain compression t / C33 per unit mean pressure; for |G| t > 30 the half-space
  Stroh compliance G33(phi) / |G| is used (error ~exp(-60));
* a Galerkin (pixel-averaged) discretisation on an N x N FFT grid with alias sums;
* frictionless, non-adhesive rigid indenters (exact sphere, paraboloid, cone, sharp Berkovich pyramid) with a
  self-consistent contact area: Polonsky-Keer conjugate gradient under load control.

h (the approach) is the rigid-indenter displacement relative to the rigid base, i.e. exactly the MD depth
h = z_s0 - apex_z with the bottom layer fixed.  Units: stiffness in GPa, lengths in A, load in GPa A^2
(1 GPa A^2 = 0.01 nN) unless a name says nN.

Validation (G10): the half-space kernel reproduces Sneddon's exact sphere plus the leading periodic-image
term to 0.1-0.8 % (the rest is the (a/L)^3 image term); for the Vashishta L = 61 A, t_el = 40.3 A cell at
0 K it gives B = 1.233 / 1.295 / 1.350 over Hertz windows h <= 2 / 3 / 4 A (tests/test_indent_analysis.py).
The box factor is an ELASTIC continuum estimate (an estimate, not a correction applied to hardness values);
records carry it as information.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.linalg import expm

KT_HALF_SPACE = 30.0  # |G| t above which the half-space compliance is used
PERIODIC_IMAGE_ALPHA = -3.9002649  # 4 zeta(1/2) beta(1/2): square-lattice 1/r sum with neutralising background
GPA_A2_PER_NN = 100.0  # 1 nN = 100 GPa A^2
BERKOVICH_FACE_ANGLE_DEG = 65.27


# ============================================================================================ elasticity
def cubic_stiffness(c11: float, c12: float, c44: float) -> np.ndarray:
    """Cubic stiffness tensor C_ijkl (GPa) in the cube frame."""
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    if i == j == k == l:
                        C[i, j, k, l] = c11
                    elif i == j and k == l:
                        C[i, j, k, l] = c12
                    elif (i == k and j == l) or (i == l and j == k):
                        C[i, j, k, l] = c44
    return C


def isotropic_stiffness(*, M: Optional[float] = None, E: Optional[float] = None, nu: float = 0.25) -> np.ndarray:
    """Isotropic stiffness from the indentation modulus M = E / (1 - nu^2) or from E."""
    if E is None:
        if M is None:
            raise ValueError("give M or E")
        E = M * (1.0 - nu * nu)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return cubic_stiffness(lam + 2 * mu, lam, mu)


def _qrt(C: np.ndarray, phi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.array([math.cos(phi), math.sin(phi), 0.0])
    n = np.array([0.0, 0.0, 1.0])
    return (np.einsum("j,ijkl,l->ik", m, C, m), np.einsum("j,ijkl,l->ik", m, C, n),
            np.einsum("j,ijkl,l->ik", n, C, n))


def half_space_compliance(C: np.ndarray, phi: float) -> np.ndarray:
    """Stroh surface compliance matrix G(phi): u(k) = G t(k) / |k| (G33 = 2 / M_isotropic)."""
    Q, R, T = _qrt(C, phi)
    Ti = np.linalg.inv(T)
    Mc = np.zeros((6, 6))
    Mc[:3, 3:] = np.eye(3)
    Mc[3:, :3] = -Ti @ Q
    Mc[3:, 3:] = -Ti @ (R + R.T)
    w, V = np.linalg.eig(Mc)
    sel = np.argsort(w.imag)[:3]
    A = V[:3, sel]
    B = np.stack([(R.T + w[s] * T) @ V[:3, s] for s in sel], axis=1)
    return -1j * A @ np.linalg.inv(B)


def layer_compliance_33(C: np.ndarray, k: float, phi: float, t: float) -> float:
    """-u_z(0) / p for a normal pressure p exp(i k m.x) on a layer -t < z < 0 bonded to a rigid base."""
    Q, R, T = _qrt(C, phi)
    Ti = np.linalg.inv(T)
    if k == 0.0:
        return float(t * Ti[2, 2])
    A = np.zeros((6, 6), complex)
    A[:3, :3] = -1j * k * Ti @ R.T
    A[:3, 3:] = Ti
    A[3:, :3] = k * k * (Q - R @ Ti @ R.T)
    A[3:, 3:] = -1j * k * R @ Ti
    E = expm(A * t)
    Sb = np.linalg.solve(E[3:, 3:], np.array([0, 0, -1.0], complex))
    U0 = E[:3, 3:] @ Sb
    return float(-U0[2].real)


class LayerMaterial:
    """An elastic material for the contact kernels: stiffness C, the tabulated half-space G33(phi) and the
    Vlassak-Nix indentation modulus M_VN = 2 / <G33>_phi."""

    def __init__(self, C: np.ndarray, n_phi: int = 721):
        self.C = np.asarray(C, float)
        phis = np.linspace(0.0, 2.0 * math.pi, n_phi)
        g = np.array([half_space_compliance(self.C, p)[2, 2].real for p in phis[:-1]])
        g = np.append(g, g[0])
        self._spline = CubicSpline(phis, g, bc_type="periodic")
        self.M_VN = float(2.0 / np.mean(g[:-1]))

    @classmethod
    def cubic(cls, c11: float, c12: float, c44: float, rotation_deg: float = 0.0) -> "LayerMaterial":
        """Cubic crystal, (001) surface; ``rotation_deg`` rotates the cube axes about z (45 deg: lab x = [110])."""
        C = cubic_stiffness(c11, c12, c44)
        if rotation_deg:
            c, s = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            C = np.einsum("ia,jb,kc,ld,abcd->ijkl", Rz, Rz, Rz, Rz, C)
        return cls(C)

    def g33_half_space(self, Gx: np.ndarray, Gy: np.ndarray) -> np.ndarray:
        k = np.hypot(Gx, Gy)
        with np.errstate(divide="ignore"):
            return self._spline(np.mod(np.arctan2(Gy, Gx), 2.0 * math.pi)) / k


# ============================================================================================ kernel + contact
def build_kernel(mat: LayerMaterial, L: float, t: float, N: int, n_alias: int = 4) -> np.ndarray:
    """Galerkin kernel K_d(q) on the N x N FFT grid of a periodic L x L cell and a layer of thickness ``t``
    (``t = inf``: half-space kernel with the G = 0 term set to 0, i.e. displacements relative to the mean)."""
    hpx = L / N
    fq = np.fft.fftfreq(N, d=1.0 / N)
    qx, qy = np.meshgrid(2 * math.pi * fq / L, 2 * math.pi * fq / L, indexing="ij")
    Kd = np.zeros((N, N))
    gstep = 2 * math.pi * N / L
    cache: dict[tuple[float, float], float] = {}
    for mx in range(-n_alias, n_alias + 1):
        for my in range(-n_alias, n_alias + 1):
            Gx = qx + mx * gstep
            Gy = qy + my * gstep
            k = np.hypot(Gx, Gy)
            S = np.sinc(Gx * hpx / (2 * math.pi)) * np.sinc(Gy * hpx / (2 * math.pi))
            g = mat.g33_half_space(Gx, Gy)
            if np.isfinite(t):
                for idx in zip(*np.nonzero(k * t <= KT_HALF_SPACE)):
                    gx, gy = float(Gx[idx]), float(Gy[idx])
                    key = (round(abs(gx), 9), round(abs(gy), 9))
                    if key not in cache:
                        cache[key] = layer_compliance_33(mat.C, math.hypot(gx, gy), math.atan2(gy, gx), t)
                    g[idx] = cache[key]
            else:
                g[k == 0] = 0.0
            Kd += g * S * S
    return Kd


def _conv(Kd: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft2(Kd * np.fft.fft2(p)))


def polonsky_keer(Kd: np.ndarray, gap0: np.ndarray, load: float, pixel_area: float, *,
                  p_init: Optional[np.ndarray] = None, rtol: float = 1e-9, maxit: int = 5000) -> tuple[np.ndarray, float, int]:
    """Frictionless rigid indenter on the periodic layer, load control (Polonsky-Keer conjugate gradient).

    ``gap0``: initial gap of the indenter profile (A); ``load`` in GPa A^2.  Returns the pressure field (GPa),
    the approach (A) and the iteration count.
    """
    if p_init is None:
        p = np.where(gap0 < np.percentile(gap0, 2), 1.0, 0.0)
    else:
        p = p_init.copy()
    p *= load / (p.sum() * pixel_area)
    dflag = 0.0
    G_old = 1.0
    tdir = np.zeros_like(p)
    it = 0
    for it in range(maxit):
        g = gap0 + _conv(Kd, p)
        S = p > 0
        g = g - g[S].mean()
        Gn = float(np.sum(g[S] ** 2))
        if Gn == 0.0 and not np.any((~S) & (g < 0)):
            break
        tn = np.zeros_like(p)
        tn[S] = g[S] + dflag * (Gn / G_old) * tdir[S]
        tdir = tn
        G_old = Gn
        r = _conv(Kd, tdir)
        r = r - r[S].mean()
        tau = float(np.sum(g[S] * tdir[S]) / np.sum(r[S] * tdir[S]))
        p_old = p.copy()
        p = p - tau * tdir
        p[p < 0] = 0.0
        Iol = (p == 0) & (g < 0)
        dflag = 0.0 if np.any(Iol) else 1.0
        p[Iol] -= tau * g[Iol]
        p *= load / (p.sum() * pixel_area)
        err = float(np.sum(np.abs(p - p_old)) / np.sum(p))
        if it > 3 and err < rtol and not np.any(Iol):
            break
    u = _conv(Kd, p)
    S = p > 0
    return p, float(np.mean((u + gap0)[S])), it


def _grid(N: int, L: float) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(N) + 0.5) * (L / N) - 0.5 * L
    return np.meshgrid(x, x, indexing="ij")


def sphere_gap(N: int, L: float, R: float, exact: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gap of a sphere (exact profile, or the paraboloid r^2 / 2R) centred in the cell; returns (gap, X, Y)."""
    X, Y = _grid(N, L)
    r2 = X ** 2 + Y ** 2
    if exact:
        g = np.where(r2 < R * R, R - np.sqrt(np.clip(R * R - r2, 0, None)), R + (np.sqrt(r2) - R) * 50)
    else:
        g = r2 / (2 * R)
    return g, X, Y


def cone_gap(N: int, L: float, half_angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gap of a sharp cone of half angle alpha: r / tan(alpha)."""
    X, Y = _grid(N, L)
    return np.hypot(X, Y) / math.tan(math.radians(half_angle_deg)), X, Y


def berkovich_gap(N: int, L: float, face_angle_deg: float = BERKOVICH_FACE_ANGLE_DEG,
                  azimuth_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gap of a sharp three-sided pyramid whose faces make ``face_angle_deg`` with the axis:
    max_i (p . u_i) / tan(theta), u_i the in-plane outward face normals at azimuth + 0/120/240 deg."""
    X, Y = _grid(N, L)
    th = math.radians(face_angle_deg)
    g = np.full(X.shape, -np.inf)
    for i in range(3):
        ph = math.radians(azimuth_deg) + 2 * math.pi * i / 3
        g = np.maximum(g, (X * math.cos(ph) + Y * math.sin(ph)) / math.tan(th))
    return g, X, Y


def load_curve(Kd: np.ndarray, gap0: np.ndarray, X: np.ndarray, Y: np.ndarray, L: float,
               loads: Sequence[float]) -> list[dict[str, float]]:
    """Contact solutions for increasing loads (GPa A^2): approach h, contact area, equivalent radius, p0."""
    N = gap0.shape[0]
    A_px = (L / N) ** 2
    out = []
    p = None
    for W in loads:
        p, d, it = polonsky_keer(Kd, gap0, float(W), A_px, p_init=p)
        S = p > 0
        area = float(S.sum() * A_px)
        out.append(dict(P=float(W), P_nN=float(W) / GPA_A2_PER_NN, h=d, area=area, a_eq=math.sqrt(area / math.pi),
                        p0=float(p.max()), p_mean=float(W) / area if area > 0 else float("nan"), iters=int(it)))
    return out


# ============================================================================================ references
def hertz_load(M: float, R: float, h: Any) -> np.ndarray:
    """Hertz P = (4/3) M sqrt(R) h^1.5 (GPa A^2)."""
    return 4.0 / 3.0 * M * math.sqrt(R) * np.clip(np.asarray(h, float), 0, None) ** 1.5


def sneddon_sphere(M: float, R: float, a: Any) -> tuple[np.ndarray, np.ndarray]:
    """Exact rigid sphere on an isotropic half-space (Sneddon 1965): (h(a), P(a))."""
    a = np.asarray(a, float)
    lg = np.log((R + a) / (R - a))
    return 0.5 * a * lg, 0.5 * M * ((a * a + R * R) * lg - 2 * a * R)


def sneddon_cone_load(M: float, half_angle_deg: float, h: Any) -> np.ndarray:
    """Rigid cone on an isotropic half-space: P = (2/pi) M tan(alpha) h^2 (GPa A^2)."""
    return (2.0 / math.pi) * M * math.tan(math.radians(half_angle_deg)) * np.clip(np.asarray(h, float), 0, None) ** 2


def fit_hertz_M(h: Any, P: Any, h_max: float, R: float, h0_fixed: Optional[float] = None) -> tuple[float, float]:
    """Windowed Hertz fit P = c (h - h0)^1.5 over h <= h_max, h0 on a 0.001 A grid in [-1.5, 1.5] (the MD
    estimator); returns (M, h0) in the units of P / A^2."""
    h = np.asarray(h, float)
    P = np.asarray(P, float)
    m = h <= h_max + 1e-12
    grid = [h0_fixed] if h0_fixed is not None else np.linspace(-1.5, 1.5, 3001)
    best = None
    for h0 in grid:
        x = np.clip(h[m] - h0, 0, None) ** 1.5
        c = float(x @ P[m]) / float(x @ x)
        s = float(((P[m] - c * x) ** 2).sum())
        if best is None or s < best[0]:
            best = (s, h0, c)
    return float(best[2] / (4.0 / 3.0 * math.sqrt(R))), float(best[1])


def default_grid(L: float, pixel_A: float = 0.25) -> int:
    """FFT grid size: the power of two giving a pixel of at most ``pixel_A``."""
    return int(2 ** math.ceil(math.log2(L / pixel_A)))


# ============================================================================================ box factors
def sphere_box_factors(mat: LayerMaterial, *, L: float, t_el: float, R: float,
                       windows: Sequence[float] = (2.0, 3.0, 4.0), N: Optional[int] = None,
                       h_top: Optional[float] = None) -> dict[str, Any]:
    """Box factor of a rigid exact sphere in the periodic layer (the G10 run_box recipe).

    For each Hertz window h <= w the MD estimator (free h0) is applied to the box curve and to Sneddon's exact
    sphere on the half-space with M_VN:  S_w = M_fit(Sneddon) / M_VN (paraboloid vs exact-sphere shape),
    B_w = M_fit(box) / M_fit(Sneddon) (pure box factor), M_pred_w = M_VN S_w B_w (continuum prediction of the
    MD apparent modulus).  The load curve is sampled at the approaches of a Hertz curve with 1.6 M_VN and
    interpolated on a 0.01 A grid (PCHIP on P^(2/3)).
    """
    N = N or default_grid(L)
    w_max = max(windows)
    top = h_top if h_top is not None else w_max + 0.6
    hs = np.concatenate([np.arange(0.05, 1.0, 0.1), np.arange(1.0, top + 0.3 + 1e-9, 0.2)])
    Kd = build_kernel(mat, L, t_el, N)
    gap0, X, Y = sphere_gap(N, L, R, exact=True)
    curve = load_curve(Kd, gap0, X, Y, L, hertz_load(1.6 * mat.M_VN, R, hs))
    hb = np.array([c["h"] for c in curve])
    Pb = np.array([c["P"] for c in curve])
    hgrid = np.round(np.arange(0.0, w_max + 1e-4, 0.01), 4)
    f = PchipInterpolator(np.concatenate([[0.0], hb]), np.concatenate([[0.0], Pb ** (2.0 / 3.0)]))
    hg = hgrid[hgrid <= min(w_max, hb.max()) + 1e-9]
    Pg = f(hg) ** 1.5
    a_s = np.linspace(1e-3, min(16.0, 0.95 * R), 4000)
    hS, PS = sneddon_sphere(mat.M_VN, R, a_s)
    fS = PchipInterpolator(hS, PS)
    PgS = np.concatenate([[0.0], fS(hgrid[1:])])
    fits = {}
    for w in windows:
        Mb, h0b = fit_hertz_M(hg, Pg, w, R)
        MS, h0S = fit_hertz_M(hgrid, PgS, w, R)
        fits[f"h<={w:g}"] = dict(M_box=Mb, h0_box=h0b, M_sneddon=MS, S=MS / mat.M_VN, B=Mb / MS, M_pred=Mb,
                                 M_over_MVN=Mb / mat.M_VN)
    return dict(indenter=dict(shape="sphere", R_A=R), L_A=L, t_el_A=t_el, N=N, pixel_A=L / N, M_VN_GPa=mat.M_VN,
                fits=fits, curve=curve)


def sharp_tip_box_factors(mat: LayerMaterial, *, L: float, t_el: float, depths_A: Sequence[float],
                          shape: str = "berkovich", face_angle_deg: float = BERKOVICH_FACE_ANGLE_DEG,
                          cone_half_angle_deg: Optional[float] = None, azimuth_deg: float = 0.0,
                          N: Optional[int] = None) -> dict[str, Any]:
    """Box factors of a sharp Berkovich pyramid (or cone) in the periodic layer against the same tip on the
    half-space (N x N periodic kernel with t = inf, a 4x larger period; elastic, frictionless).

    Per depth: F_P = P_box / P_hs (load at the same depth: the bias of any geometric-area p_m, E1/E2) and
    F_pm = (P / A)_box / (P / A)_hs with the self-consistent elastic contact areas (the bias of a p_m on the
    estimated contact, E3/E4).  For a cone the half-space reference is Sneddon's closed form with M_VN.
    """
    N = N or default_grid(L, 0.3)
    if shape == "berkovich":
        gap0, X, Y = berkovich_gap(N, L, face_angle_deg, azimuth_deg)
        C0 = 3 * math.sqrt(3) * math.tan(math.radians(face_angle_deg)) ** 2
        alpha = math.degrees(math.atan(math.sqrt(C0 / math.pi)))
    elif shape == "cone":
        alpha = float(cone_half_angle_deg or 70.296)
        gap0, X, Y = cone_gap(N, L, alpha)
    else:
        raise ValueError("shape must be 'berkovich' or 'cone'")
    hs = np.asarray(depths_A, float)
    P_guess = sneddon_cone_load(mat.M_VN, alpha, hs)

    def solve(Kd, g0, Xg, Yg, Lc, image_alpha):
        # load-controlled solutions on a load ladder that is extended until the deepest target depth is passed;
        # image_alpha: periodic half-space kernels (G = 0 dropped) under-report the approach by
        # alpha P / (pi M L) (alpha = 4 zeta(1/2) beta(1/2); G10 validate_bem), which is added back.
        scale = 1.0
        for _ in range(8):
            cur = load_curve(Kd, g0, Xg, Yg, Lc, scale * P_guess)
            for c in cur:
                c["h"] = c["h"] - image_alpha * c["P"] / (math.pi * mat.M_VN * Lc)
            if max(c["h"] for c in cur) >= hs.max():
                return cur
            scale *= 1.25
        raise RuntimeError("contact solutions did not reach the requested depths")

    box = solve(build_kernel(mat, L, t_el, N), gap0, X, Y, L, 0.0)
    # half-space reference: the same tip and pixel size in a 4x larger periodic cell, image term removed
    L4, N4 = 4.0 * L, 4 * N
    if shape == "berkovich":
        g4, X4, Y4 = berkovich_gap(N4, L4, face_angle_deg, azimuth_deg)
    else:
        g4, X4, Y4 = cone_gap(N4, L4, alpha)
    hs_ref = solve(build_kernel(mat, L4, np.inf, N4, n_alias=2), g4, X4, Y4, L4, PERIODIC_IMAGE_ALPHA)

    def at(curve, key, h):
        hh = np.array([0.0] + [c["h"] for c in curve])
        vv = np.array([0.0] + [c[key] for c in curve])
        o = np.argsort(hh)
        # P and A scale as h^2 for a sharp tip: interpolate sqrt(v) linearly in h
        return float(np.interp(h, hh[o], np.sqrt(vv[o])) ** 2)

    rows = []
    for h in hs:
        Pb, Ab = at(box, "P", h), at(box, "area", h)
        Ph, Ah = at(hs_ref, "P", h), at(hs_ref, "area", h)
        rows.append(dict(h_A=float(h), P_box_nN=Pb / GPA_A2_PER_NN, P_hs_nN=Ph / GPA_A2_PER_NN, A_box_A2=Ab, A_hs_A2=Ah,
                         P_sneddon_cone_nN=float(sneddon_cone_load(mat.M_VN, alpha, h)) / GPA_A2_PER_NN,
                         F_P=Pb / Ph, F_pm=(Pb / Ab) / (Ph / Ah), pm_hs_true_GPa=Ph / Ah,
                         a_eq_hs_A=math.sqrt(Ah / math.pi), L_over_a=L / math.sqrt(Ah / math.pi),
                         t_over_a=t_el / math.sqrt(Ah / math.pi)))
    return dict(indenter=dict(shape=shape, face_angle_deg=face_angle_deg, equivalent_cone_deg=alpha,
                              azimuth_deg=azimuth_deg), L_A=L, t_el_A=t_el, N=N, pixel_A=L / N, M_VN_GPa=mat.M_VN,
                rows=rows, note="elastic continuum estimate of the finite-cell bias (not applied as a correction)")


def box_factor_report(*, cij_GPa: Sequence[float], L_A: float, t_el_A: float, indenter: dict[str, Any],
                      depths_A: Sequence[float] = (), windows: Sequence[float] = (2.0, 3.0, 4.0),
                      rotation_deg: float = 0.0, N: Optional[int] = None) -> dict[str, Any]:
    """The box-factor report of an indentation cell (analysis/records attach it as information).

    ``cij_GPa`` = (C11, C12, C44) of the potential at the run temperature; ``indenter`` = {"shape": "sphere",
    "R_A": R} or {"shape": "berkovich"|"cone", ...}; ``rotation_deg`` = angle of the cube axes about z in the
    lab frame (45 for the harness (001) slab whose lab x is [110]).
    """
    mat = LayerMaterial.cubic(*cij_GPa, rotation_deg=rotation_deg)
    shape = indenter.get("shape", "berkovich")
    if shape == "sphere":
        rep = sphere_box_factors(mat, L=L_A, t_el=t_el_A, R=float(indenter["R_A"]), windows=windows, N=N)
    else:
        rep = sharp_tip_box_factors(mat, L=L_A, t_el=t_el_A, depths_A=depths_A, shape=shape,
                                    face_angle_deg=float(indenter.get("face_angle_deg", BERKOVICH_FACE_ANGLE_DEG)),
                                    cone_half_angle_deg=indenter.get("cone_half_angle_deg"),
                                    azimuth_deg=float(indenter.get("azimuth_deg", 0.0)), N=N)
    rep["cij_GPa"] = list(cij_GPa)
    rep["rotation_deg"] = rotation_deg
    rep.pop("curve", None)
    return rep


__all__ = [
    "LayerMaterial", "berkovich_gap", "box_factor_report", "build_kernel", "cone_gap", "cubic_stiffness",
    "default_grid", "fit_hertz_M", "half_space_compliance", "hertz_load", "isotropic_stiffness",
    "layer_compliance_33", "load_curve", "polonsky_keer", "sharp_tip_box_factors", "sneddon_cone_load",
    "sneddon_sphere", "sphere_box_factors", "sphere_gap",
]
