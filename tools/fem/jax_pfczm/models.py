"""Material fields, phase-field model definitions and mesh-dependence helpers.

Models (energy per unit thickness)::

    E(u, d) = sum_gp w [ (g(d) + k_res) psi0(eps) ] + G_c/c_w * integral( w(d)/L + L |grad d|^2 )

* ``at2``  (Bourdin-Francfort-Marigo 2000; Miehe et al. 2010):
  g = (1-d)^2, w(d) = d^2, c_w = 2, length L = l.
  Homogeneous uniaxial peak stress sigma_c = (3/16) sqrt(3 E' G_c / l).
* ``pfczm`` (J.-Y. Wu, JMPS 103 (2017) 72-99; Wu & Nguyen, JMPS 119 (2018) 20-42):
  g = (1-d)^p / ((1-d)^p + a1 d (1 + a2 d + a2 a3 d^2)), w(d) = 2d - d^2, c_w = pi,
  length L = b, a1 = 4 l_ch / (pi b), l_ch = E G_f / f_t^2.
  linear softening p=2, a2=-1/2, a3=0; exponential p=2.5, a2=2^(5/3)-3, a3=0;
  Cornelissen p=2, a2=1.3868, a3=0.6567.  Peak stress = f_t independent of b.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal, Optional

import jax.numpy as jnp
import numpy as np

from .grid import Grid

ModelKind = Literal["at2", "pfczm"]
Softening = Literal["linear", "exponential", "cornelissen"]
Driving = Literal["isotropic", "principal_stress"]


@dataclasses.dataclass
class Material:
    """Per-element (pixel) properties.  Arrays of shape (nel,), element order of :class:`Grid`.

    ``damageable`` False means the element never degrades (g = 1 there, no crack energy) and nodes whose
    every adjacent element is non-damageable are held at d = 0.  ``phase`` is an informational integer label.
    """

    E: np.ndarray
    nu: np.ndarray
    Gc: np.ndarray
    ft: np.ndarray
    damageable: np.ndarray
    phase: Optional[np.ndarray] = None  # integer labels, informational

    @classmethod
    def homogeneous(cls, grid: Grid, E: float, nu: float, Gc: float, ft: float = 1.0) -> "Material":
        n = grid.nel
        return cls(E=np.full(n, E, float), nu=np.full(n, nu, float), Gc=np.full(n, Gc, float),
                   ft=np.full(n, ft, float), damageable=np.ones(n, bool), phase=np.zeros(n, int))

    def copy(self) -> "Material":
        return Material(E=self.E.copy(), nu=self.nu.copy(), Gc=self.Gc.copy(), ft=self.ft.copy(),
                        damageable=self.damageable.copy(),
                        phase=None if self.phase is None else self.phase.copy())


def softening_parameters(softening: Softening) -> tuple[float, float, float]:
    """(p, a2, a3) of Wu's rational degradation function (Wu & Nguyen 2018, Table 1)."""
    if softening == "linear":
        return 2.0, -0.5, 0.0
    if softening == "exponential":
        return 2.5, 2.0 ** (5.0 / 3.0) - 3.0, 0.0
    if softening == "cornelissen":
        return 2.0, 1.3868, 0.6567
    raise ValueError(softening)


@dataclasses.dataclass(frozen=True)
class PhaseFieldModel:
    """Model choice.  ``length`` is l (AT2) or b (PF-CZM), in the same length unit as the grid."""

    kind: ModelKind = "pfczm"
    length: float = 1.0  # l (AT2) or b (PF-CZM)
    softening: Softening = "linear"
    driving: Optional[Driving] = None  # default: isotropic for AT2, principal_stress for PF-CZM
    k_res: float = 1e-8  # residual stiffness
    # Wu (2017) floors the history at Y0 = f_t^2/(2E) because his damage equation is solved without
    # bounds.  With the bound-constrained solver used here the elastic stage (no damage while
    # Y < Y0) follows from d >= 0 alone, and the floor would make the scheme non-variational
    # (extra dissipation next to the band), so it is off by default.
    history_floor: bool = False
    # "bound": variational alternate minimisation, driving force = current Y(u), irreversibility by the
    #          bound d >= d_n only (Bourdin et al. 2000; energy balance exact up to load-step error).
    # "history": Miehe et al. (2010) history field H = max_t Y (plus the bound); not variational.
    irreversibility: Literal["bound", "history"] = "bound"

    @property
    def driving_force(self) -> Driving:
        if self.driving is not None:
            return self.driving
        return "isotropic" if self.kind == "at2" else "principal_stress"

    @property
    def c_w(self) -> float:
        return 2.0 if self.kind == "at2" else math.pi


def element_parameters(grid: Grid, mat: Material, model: PhaseFieldModel) -> dict:
    """Per-element constants used by the solver (jnp arrays, shape (nel,)).

    lam, mu: plane-strain Lame constants; c1 = G_c/(c_w L), c2 = G_c L/c_w (zeroed where not damageable);
    a1 = 4 l_ch/(pi b) (PF-CZM, else 0); Y0 = f_t^2/(2E) (PF-CZM, zeroed where not damageable); mask = damageable.
    """
    E, nu = np.asarray(mat.E, float), np.asarray(mat.nu, float)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    L = model.length
    cw = model.c_w
    c1 = mat.Gc / (cw * L)
    c2 = mat.Gc * L / cw
    if model.kind == "pfczm":
        lch = E * mat.Gc / mat.ft**2
        a1 = 4.0 * lch / (math.pi * L)
        Y0 = mat.ft**2 / (2.0 * E)
    else:
        a1 = np.zeros_like(E)
        Y0 = np.zeros_like(E)
    m = np.asarray(mat.damageable, float)
    return dict(lam=jnp.asarray(lam), mu=jnp.asarray(mu), E=jnp.asarray(E), c1=jnp.asarray(c1 * m),
                c2=jnp.asarray(c2 * m), a1=jnp.asarray(a1), Y0=jnp.asarray(Y0 * m), mask=jnp.asarray(m))


_element_parameters = element_parameters  # name used in the original implementation


def check_pfczm_length(mat: Material, b: float) -> dict:
    """Wu's PF-CZM requires b small compared with l_ch (a1 >= ~3/2 for linear softening, i.e. b <= 8 l_ch/(3 pi))."""
    lch = mat.E * mat.Gc / mat.ft**2
    dm = np.asarray(mat.damageable, bool)
    lmin = float(lch[dm].min()) if dm.any() else float("inf")
    return {"l_ch_min": lmin, "b": b, "b_over_lch_max": b / lmin, "ok": b <= 8.0 * lmin / (3.0 * math.pi)}


def discrete_toughness_factor(kind: ModelKind, h_over_length: float, n_half: int = 400) -> float:
    """G_eff/G_c of a fully developed, grid-aligned crack for this discretisation.

    Minimises the 1-D discrete functional (nodal quadrature of w(d), linear FE
    gradient) with d = 1 at the crack node, per unit crack length, in units of
    G_c.  Continuum value 1.  AT2 has a closed form; PF-CZM is minimised with
    bounds 0 <= d <= 1 (compact support).  Bourdin et al. (2008) quote the AT2
    leading-order correction G_eff = G_c (1 + h/(2l)) for linear elements.
    ``n_half`` is unused (kept for signature compatibility).
    """
    s = float(h_over_length)
    if kind == "at2":
        r = 1.0 + 0.5 * s * s - math.sqrt((1.0 + 0.5 * s * s) ** 2 - 1.0)
        return (s / 2.0) * (1 + r * r) / (1 - r * r) + (1.0 / s) * (1 - r) / (1 + r)
    from scipy.optimize import minimize

    L, h = 1.0, s  # length = 1
    n = max(int(math.ceil(4.0 / h)) + 5, 20)
    n = min(n, 20000)

    def f(x):
        dd = np.concatenate([[1.0], x])
        wv = 2 * dd - dd**2
        loc = h * (0.5 * wv[0] + np.sum(wv[1:]))  # half-line: crack node weight h/2
        dif = np.diff(dd)
        grad = np.sum(dif**2) / h
        val = (loc / L + L * grad) / math.pi
        gl = np.zeros_like(dd)
        gl[0] = 0.5 * h * (2 - 2 * dd[0])
        gl[1:] = h * (2 - 2 * dd[1:])
        gg = np.zeros_like(dd)
        gg[:-1] -= 2 * dif / h
        gg[1:] += 2 * dif / h
        g = (gl / L + L * gg) / math.pi
        return val, g[1:]

    x0 = np.clip(1 - np.sin(np.arange(1, n + 1) * h), 0, 1)
    x0[np.arange(1, n + 1) * h > math.pi / 2] = 0.0
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(0, 1)] * n,
                   options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-12})
    return 2.0 * float(res.fun)  # both halves
