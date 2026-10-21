"""2D plane-strain phase-field fracture (AT2 and Wu's PF-CZM) on a structured grid, in JAX.

Discretisation
--------------
* Square bilinear (Q4) elements of size ``h`` on an ``nx x ny`` grid; nodal
  displacement ``u`` (2 dof/node) and nodal damage ``d``.
* Material parameters are piecewise constant per element (pixel): E, nu, G_c,
  f_t and a ``damageable`` flag (non-damageable pixels, e.g. filler particles,
  never degrade; nodes interior to them are fixed at d = 0).
* 2x2 Gauss quadrature.  The *pointwise* terms (degradation g(d), crack
  geometric function w(d)) are evaluated with the nodal damage of the node
  nearest to each Gauss point (a nodal/lumped quadrature), the gradient term
  |grad d|^2 with the consistent Q4 interpolation.  This makes the local part of
  the damage functional separable per node (diagonal Hessian) and is used
  *identically* in the displacement and damage subproblems, so the discrete
  energy is a single functional E_h(u, d).

Models (energy per unit thickness)
----------------------------------
E(u, d) = sum_gp w [ (g(d)+k) psi0(eps) ] + G_c/c_w * integral( w(d)/l + l |grad d|^2 )

* ``at2``  (Bourdin-Francfort-Marigo 2000; Miehe et al. 2010):
  g = (1-d)^2, w(d) = d^2, c_w = 2, length ``l``.
  Homogeneous uniaxial peak stress sigma_c = (3/16) sqrt(3 E' G_c / l).
* ``pfczm`` (J.-Y. Wu, JMPS 103 (2017) 72-99; Wu & Nguyen, JMPS 119 (2018) 20-42):
  g = (1-d)^p / ((1-d)^p + a1 d (1 + a2 d + a2 a3 d^2)), w(d) = 2d - d^2, c_w = pi,
  length ``b``, a1 = 4 l_ch / (pi b), l_ch = E G_f / f_t^2.
  linear softening p=2, a2=-1/2, a3=0; exponential p=2.5, a2=2^(5/3)-3, a3=0;
  Cornelissen p=2, a2=1.3868, a3=0.6567.  Peak stress = f_t independent of b.

Driving force / irreversibility
-------------------------------
* ``isotropic``: history H = max_t psi0 (variational; used for AT2).
* ``principal_stress`` (Wu 2017): H = max(Y0, max_t <sigma1_eff>^2 / (2 E)),
  Y0 = f_t^2 / (2E).  Stiffness is degraded isotropically ("hybrid" formulation).
* Damage is additionally bound-constrained d_n <= d <= 1 (irreversibility) and
  solved by a projected Newton method (active set + PCG + projected Armijo
  line search).  Displacement: matrix-free Jacobi-PCG with Dirichlet masking.
* Staggered alternate minimisation per load step until the relative change of
  the total energy and max |delta d| are below tolerances.

Units: any consistent set.  Recommended (dental application): length um,
stress MPa  ->  G_c in MPa*um = J/m^2, energies per thickness in MPa*um^2.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from typing import Callable, Literal, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import fracture_md  # noqa: F401  (float64)

ModelKind = Literal["at2", "pfczm"]
Softening = Literal["linear", "exponential", "cornelissen"]
Driving = Literal["isotropic", "principal_stress"]

_G = 1.0 / math.sqrt(3.0)
_XI = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
_GPS = _XI * _G  # Gauss point q is nearest to local node q


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Grid:
    """Structured grid of square Q4 elements.  Node (i, j) -> j*(nx+1)+i; element (i, j) -> j*nx+i."""

    nx: int
    ny: int
    h: float
    x0: float = 0.0
    y0: float = 0.0

    @property
    def nnode(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def nel(self) -> int:
        return self.nx * self.ny

    def node_id(self, i, j):
        return np.asarray(j) * (self.nx + 1) + np.asarray(i)

    @property
    def conn(self) -> np.ndarray:
        i, j = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="xy")
        i, j = i.ravel(), j.ravel()
        return np.stack([self.node_id(i, j), self.node_id(i + 1, j), self.node_id(i + 1, j + 1),
                         self.node_id(i, j + 1)], axis=1)

    @property
    def nodes(self) -> np.ndarray:
        i, j = np.meshgrid(np.arange(self.nx + 1), np.arange(self.ny + 1), indexing="xy")
        return np.stack([self.x0 + i.ravel() * self.h, self.y0 + j.ravel() * self.h], axis=1)

    @property
    def centers(self) -> np.ndarray:
        i, j = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="xy")
        return np.stack([self.x0 + (i.ravel() + 0.5) * self.h, self.y0 + (j.ravel() + 0.5) * self.h], axis=1)

    @property
    def width(self) -> float:
        return self.nx * self.h

    @property
    def height(self) -> float:
        return self.ny * self.h


def _shape_gradients(h: float) -> np.ndarray:
    """dN_a/dx_k at each Gauss point: (4 gp, 2, 4 nodes)."""
    G = np.zeros((4, 2, 4))
    for q, (xi, eta) in enumerate(_GPS):
        for a, (xa, ya) in enumerate(_XI):
            G[q, 0, a] = 0.25 * xa * (1 + ya * eta) * 2.0 / h
            G[q, 1, a] = 0.25 * ya * (1 + xa * xi) * 2.0 / h
    return G


def _bmatrix(G: np.ndarray) -> np.ndarray:
    """Strain-displacement matrices (4 gp, 3 [exx, eyy, gxy], 8 dofs)."""
    B = np.zeros((4, 3, 8))
    for q in range(4):
        for a in range(4):
            B[q, 0, 2 * a] = G[q, 0, a]
            B[q, 1, 2 * a + 1] = G[q, 1, a]
            B[q, 2, 2 * a] = G[q, 1, a]
            B[q, 2, 2 * a + 1] = G[q, 0, a]
    return B


# ---------------------------------------------------------------------------
# Material and model
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Material:
    """Per-element (pixel) properties.  Arrays of shape (nel,)."""

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


def _element_parameters(grid: Grid, mat: Material, model: PhaseFieldModel) -> dict:
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


def check_pfczm_length(mat: Material, b: float) -> dict:
    """Wu's PF-CZM requires b small compared with l_ch (a1 >= ~3/2 for linear softening, i.e. b <= 8 l_ch/(3 pi))."""
    lch = mat.E * mat.Gc / mat.ft**2
    dm = np.asarray(mat.damageable, bool)
    lmin = float(lch[dm].min()) if dm.any() else float("inf")
    return {"l_ch_min": lmin, "b": b, "b_over_lch_max": b / lmin, "ok": b <= 8.0 * lmin / (3.0 * math.pi)}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def _pcg(A: Callable, b, x0, Minv, tol, maxiter):
    r0 = b - A(x0)
    z0 = Minv * r0
    bnorm = jnp.maximum(jnp.linalg.norm(b), 1e-300)

    def cond(s):
        x, r, z, p, rz, k = s
        return (jnp.linalg.norm(r) > tol * bnorm) & (k < maxiter)

    def body(s):
        x, r, z, p, rz, k = s
        Ap = A(p)
        pAp = jnp.vdot(p, Ap)
        alpha = jnp.where(pAp > 0, rz / jnp.where(pAp > 0, pAp, 1.0), 0.0)
        x = x + alpha * p
        r = r - alpha * Ap
        z = Minv * r
        rz_new = jnp.vdot(r, z)
        beta = rz_new / jnp.where(rz != 0, rz, 1.0)
        p = z + beta * p
        return x, r, z, p, rz_new, k + 1

    x, r, z, p, rz, k = lax.while_loop(cond, body, (x0, r0, z0, z0, jnp.vdot(r0, z0), 0))
    return x, k, jnp.linalg.norm(r) / bnorm


class PhaseFieldSolver:
    """Builds jitted energy/solve functions for one (grid, model).  Material arrays are runtime inputs."""

    def __init__(self, grid: Grid, model: PhaseFieldModel, *, cg_tol: float = 1e-9, cg_maxiter: int = 20000,
                 dmg_cg_tol: float = 1e-10, newton_maxiter: int = 50, newton_tol: float = 1e-9):
        self.grid, self.model = grid, model
        self.cg_tol, self.cg_maxiter = cg_tol, cg_maxiter
        self.dmg_cg_tol, self.newton_maxiter, self.newton_tol = dmg_cg_tol, newton_maxiter, newton_tol
        conn = grid.conn
        self.conn = jnp.asarray(conn)
        dofs = np.stack([2 * conn, 2 * conn + 1], axis=2).reshape(-1, 8)
        self.dofs = jnp.asarray(dofs)
        G = _shape_gradients(grid.h)
        self.G = jnp.asarray(G)
        self.B = jnp.asarray(_bmatrix(G))
        self.w = grid.h**2 / 4.0
        p, a2, a3 = softening_parameters(model.softening)
        self._p, self._a2, self._a3 = p, a2, a3
        self._build()

    # ----- constitutive pieces ------------------------------------------------
    def degradation(self, d, a1):
        if self.model.kind == "at2":
            return (1.0 - d) ** 2
        p, a2, a3 = self._p, self._a2, self._a3
        one = jnp.maximum(1.0 - d, 0.0)
        num = one**p
        return num / (num + a1 * d * (1.0 + a2 * d + a2 * a3 * d * d))

    def geometric(self, d):
        return d * d if self.model.kind == "at2" else 2.0 * d - d * d

    def strains(self, u):  # u flat (2 nnode,)
        ue = u[self.dofs]  # (nel, 8)
        return jnp.einsum("qij,ej->eqi", self.B, ue)

    @staticmethod
    def psi0(eps, lam, mu):
        exx, eyy, gxy = eps[..., 0], eps[..., 1], eps[..., 2]
        tr = exx + eyy
        return 0.5 * lam[:, None] * tr**2 + mu[:, None] * (exx**2 + eyy**2 + 0.5 * gxy**2)

    @staticmethod
    def stress(eps, lam, mu):
        exx, eyy, gxy = eps[..., 0], eps[..., 1], eps[..., 2]
        tr = exx + eyy
        return jnp.stack([lam[:, None] * tr + 2 * mu[:, None] * exx, lam[:, None] * tr + 2 * mu[:, None] * eyy,
                          mu[:, None] * gxy], axis=-1)

    def driving(self, u, P):
        eps = self.strains(u)
        if self.model.driving_force == "isotropic":
            return self.psi0(eps, P["lam"], P["mu"])
        s = self.stress(eps, P["lam"], P["mu"])
        c = 0.5 * (s[..., 0] + s[..., 1])
        r = jnp.sqrt((0.5 * (s[..., 0] - s[..., 1])) ** 2 + s[..., 2] ** 2)
        s1 = jnp.maximum(c + r, 0.0)
        return s1**2 / (2.0 * P["E"][:, None])

    def gp_g(self, d, P):
        dg = d[self.conn]
        g = self.degradation(dg, P["a1"][:, None])
        return jnp.where(P["mask"][:, None] > 0, g, 1.0) + self.model.k_res

    # ----- energies -----------------------------------------------------------
    def elastic_energy(self, u, d, P):
        return self.w * jnp.sum(self.gp_g(d, P) * self.psi0(self.strains(u), P["lam"], P["mu"]))

    def _local_damage(self, d, H, P):
        dg = d[self.conn]
        g = self.degradation(dg, P["a1"][:, None])
        return self.w * jnp.sum(P["mask"][:, None] * g * H + P["c1"][:, None] * self.geometric(dg))

    def _gradient_damage(self, d, P):
        de = d[self.conn]
        gd = jnp.einsum("qkj,ej->eqk", self.G, de)
        return self.w * jnp.sum(P["c2"][:, None] * jnp.sum(gd**2, axis=-1))

    def crack_energy(self, d, P):
        """Dissipated (fracture) energy  sum G_c/c_w (w(d)/l + l|grad d|^2)."""
        dg = d[self.conn]
        return self.w * jnp.sum(P["c1"][:, None] * self.geometric(dg)) + self._gradient_damage(d, P)

    def damage_energy(self, d, H, P):
        return self._local_damage(d, H, P) + self._gradient_damage(d, P)

    # ----- build jitted kernels -----------------------------------------------
    def _build(self):
        nn = self.grid.nnode

        def kdiag(d, P):
            B = self.B
            g = self.gp_g(d, P)  # (nel,4)
            lam, mu = P["lam"][:, None, None], P["mu"][:, None, None]
            b0, b1, b2 = B[:, 0, :][None], B[:, 1, :][None], B[:, 2, :][None]  # (1,4,8)
            de = (lam + 2 * mu) * (b0**2 + b1**2) + 2 * lam * b0 * b1 + mu * b2**2  # (nel,4,8)
            de = self.w * jnp.sum(g[:, :, None] * de, axis=1)  # (nel,8)
            return jnp.zeros(2 * nn).at[self.dofs.reshape(-1)].add(de.reshape(-1))

        def solve_u(u0, uD, fixed, d, P):
            free = 1.0 - fixed
            Kmv = jax.grad(lambda x: self.elastic_energy(x, d, P))
            b = -free * Kmv(uD * fixed)
            A = lambda v: free * Kmv(free * v)
            diag = kdiag(d, P)
            Minv = free / jnp.where(diag > 0, diag, 1.0)
            x, k, res = _pcg(A, b, free * u0, Minv, self.cg_tol, self.cg_maxiter)
            u = uD * fixed + free * x
            return u, k, res

        def internal_force(u, d, P):
            return jax.grad(lambda x: self.elastic_energy(x, d, P))(u)

        # Laplacian diagonal (per node) for the damage preconditioner
        def lap_diag(P):
            Gsq = jnp.sum(self.G**2, axis=1)  # (4 gp, 4 nodes)
            de = self.w * P["c2"][:, None] * jnp.sum(Gsq, axis=0)[None, :]  # (nel,4)
            return jnp.zeros(nn).at[self.conn.reshape(-1)].add(de.reshape(-1))

        def newton_step(d, H, lb, ub, P):
            E0, g = jax.value_and_grad(self.damage_energy)(d, H, P)
            ones = jnp.ones_like(d)
            hloc = jax.jvp(lambda x: jax.grad(self._local_damage)(x, H, P), (d,), (ones,))[1]
            ld = lap_diag(P)
            delta = 1e-3 * ld + 1e-12 * (jnp.max(ld) + 1e-300)
            hmod = jnp.maximum(hloc, delta)
            tol_b = 1e-12
            active = ((d <= lb + tol_b) & (g > 0)) | ((d >= ub - tol_b) & (g < 0)) | (ub - lb <= tol_b)
            free = jnp.where(active, 0.0, 1.0)
            lap = jax.grad(lambda x: self._gradient_damage(x, P))
            # gradient energy is quadratic & homogeneous -> grad at v = K_lap v
            A = lambda v: free * (lap(free * v) + hmod * free * v)
            Minv = free / (ld + hmod)
            p, kcg, _ = _pcg(A, -free * g, jnp.zeros_like(d), Minv, self.dmg_cg_tol, 5000)

            def trial(t):
                return jnp.clip(d + t * p, lb, ub)

            def ls_cond(s):
                t, Et, it = s
                dt = trial(t)
                return (Et > E0 + 1e-4 * jnp.vdot(g, dt - d)) & (it < 40)

            def ls_body(s):
                t, Et, it = s
                t = 0.5 * t
                return t, self.damage_energy(trial(t), H, P), it + 1

            t, Et, it = lax.while_loop(ls_cond, ls_body, (1.0, self.damage_energy(trial(1.0), H, P), 0))
            dn = trial(t)
            dn = jnp.where(Et <= E0, dn, d)
            pg = jnp.where(active, 0.0, g)
            return dn, jnp.max(jnp.abs(pg)) / (jnp.max(ld) + 1e-300), jnp.max(jnp.abs(dn - d)), kcg

        self._solve_u = jax.jit(solve_u)
        self._fint = jax.jit(internal_force)
        self._newton = jax.jit(newton_step)
        self._driving = jax.jit(self.driving)
        self._Eel = jax.jit(self.elastic_energy)
        self._Gam = jax.jit(self.crack_energy)

    def solve_damage(self, d, H, lb, ub, P):
        it = 0
        for it in range(1, self.newton_maxiter + 1):
            d, pgn, dd, _ = self._newton(d, H, lb, ub, P)
            if float(pgn) < self.newton_tol or float(dd) < 1e-12:
                break
        return d, it

    # ----- quasi-static driver ------------------------------------------------
    def run(
        self,
        mat: Material,
        bc: Callable[[float], tuple[np.ndarray, np.ndarray]],
        loads: Sequence[float],
        *,
        d_fixed_one: Optional[np.ndarray] = None,
        tol_E: float = 1e-7,
        tol_d: float = 1e-4,
        max_stagger: int = 300,
        snapshot_every: int = 0,
        observers: Optional[dict[str, Callable]] = None,
        verbose: bool = False,
    ) -> dict:
        """Displacement-controlled quasi-static run.

        ``bc(t) -> (uD (nnode,2), fixed (nnode,2) bool)``; the fixed mask must not change.
        ``d_fixed_one``: nodal bool mask of pre-cracked nodes (d = 1 imposed).
        ``observers``: name -> f(u (nnode,2) np, d np, fint (nnode,2) np) -> float, recorded per step.
        Returns histories (loads, reactions, energies, external work, iterations) and final fields.
        """
        grid = self.grid
        P = _element_parameters(grid, mat, self.model)
        nn = grid.nnode
        # nodes whose every adjacent element is non-damageable are fixed at 0
        dm = np.asarray(mat.damageable, bool)
        touched = np.zeros(nn, bool)
        touched[grid.conn[dm].ravel()] = True
        ub = jnp.asarray(np.where(touched, 1.0, 0.0))
        lb0 = np.zeros(nn)
        if d_fixed_one is not None:
            lb0[np.asarray(d_fixed_one, bool)] = 1.0
            ub = jnp.maximum(ub, jnp.asarray(lb0))
        d = jnp.asarray(lb0)
        # initial damage profile around pre-cracks (no load)
        H = jnp.broadcast_to(P["Y0"][:, None], (grid.nel, 4)) * (1.0 if self.model.history_floor else 0.0)
        H_floor = H
        if d_fixed_one is not None and np.any(d_fixed_one):
            d, _ = self.solve_damage(d, H, jnp.asarray(lb0), ub, P)
        u = jnp.zeros(2 * nn)
        uD0, fixed0 = bc(float(loads[0]))
        fixed = jnp.asarray(np.asarray(fixed0, float).reshape(-1))
        hist = {k: [] for k in ["load", "elastic", "dissipated", "total", "external_work", "n_stagger",
                                "converged", "cg_iters_max", "max_d", "wall_s"]}
        obs_hist = {k: [] for k in (observers or {})}
        snaps = []
        Wext = 0.0
        prev_uD = None
        prev_f = None
        Gam0 = float(self._Gam(d, P))
        t_start = time.time()
        for step, t in enumerate(loads):
            uDn, fx = bc(float(t))
            if not np.array_equal(np.asarray(fx, bool), np.asarray(fixed0, bool)):
                raise ValueError("fixed-dof mask must be constant over the load path")
            uD = jnp.asarray(np.asarray(uDn, float).reshape(-1))
            # warm start: scale previous solution to new boundary values
            u = u * (1.0 - fixed) + uD * fixed
            d_n = d
            H_n = H
            lb = jnp.maximum(d_n, jnp.asarray(lb0))
            Eprev = None
            conv = False
            cgmax = 0
            k = 0
            for k in range(1, max_stagger + 1):
                u, kcg, _ = self._solve_u(u, uD, fixed, d, P)
                cgmax = max(cgmax, int(kcg))
                Yc = self._driving(u, P)
                H = jnp.maximum(H_n, Yc) if self.model.irreversibility == "history" else jnp.maximum(Yc, H_floor)
                d_old = d
                d, _ = self.solve_damage(d, H, lb, ub, P)
                Etot = float(self._Eel(u, d, P) + self._Gam(d, P))
                dd = float(jnp.max(jnp.abs(d - d_old)))
                if Eprev is not None:
                    dE = abs(Etot - Eprev) / max(abs(Etot), 1e-300)
                    if dE < tol_E and dd < tol_d:
                        conv = True
                        break
                elif dd < 1e-14:
                    conv = True
                    break
                Eprev = Etot
            # final equilibrium with the converged damage
            u, kcg, _ = self._solve_u(u, uD, fixed, d, P)
            f = self._fint(u, d, P)
            if prev_uD is not None:
                Wext += float(0.5 * jnp.vdot((f + prev_f) * fixed, (uD - prev_uD) * fixed))
            else:
                Wext += float(0.5 * jnp.vdot(f * fixed, uD * fixed))  # linear ramp from zero
            prev_uD, prev_f = uD, f
            Eel = float(self._Eel(u, d, P))
            Gam = float(self._Gam(d, P)) - Gam0
            hist["load"].append(float(t))
            hist["elastic"].append(Eel)
            hist["dissipated"].append(Gam)
            hist["total"].append(Eel + Gam)
            hist["external_work"].append(Wext)
            hist["n_stagger"].append(k)
            hist["converged"].append(conv)
            hist["cg_iters_max"].append(cgmax)
            hist["max_d"].append(float(jnp.max(d)))
            hist["wall_s"].append(time.time() - t_start)
            if observers:
                un, dn_, fn = np.asarray(u).reshape(nn, 2), np.asarray(d), np.asarray(f).reshape(nn, 2)
                for name, fun in observers.items():
                    obs_hist[name].append(float(fun(un, dn_, fn)))
            if snapshot_every and (step % snapshot_every == 0 or step == len(loads) - 1):
                snaps.append((float(t), np.asarray(d).copy()))
            if verbose:
                print(f"step {step:4d} t={t:.5g} stag={k:3d} conv={conv} maxd={hist['max_d'][-1]:.3f} "
                      f"Eel={Eel:.5g} Gam={Gam:.5g} W={Wext:.5g} cg={cgmax}", flush=True)
        out = {k: np.asarray(v) for k, v in hist.items()}
        out.update({k: np.asarray(v) for k, v in obs_hist.items()})
        out["u"] = np.asarray(u).reshape(nn, 2)
        out["d"] = np.asarray(d)
        out["H"] = np.asarray(H)
        out["snapshots"] = snaps
        out["initial_crack_energy"] = Gam0
        return out


# ---------------------------------------------------------------------------
# Discrete 1-D crack profile: exact mesh correction of the dissipated energy
# ---------------------------------------------------------------------------
def discrete_toughness_factor(kind: ModelKind, h_over_length: float, n_half: int = 400) -> float:
    """G_eff/G_c of a fully developed, grid-aligned crack for this discretisation.

    Minimises the 1-D discrete functional (nodal quadrature of w(d), linear FE
    gradient) with d = 1 at the crack node, per unit crack length, in units of
    G_c.  Continuum value 1.  AT2 has a closed form; PF-CZM is minimised with
    bounds 0 <= d <= 1 (compact support).  Bourdin et al. (2008) quote the AT2
    leading-order correction G_eff = G_c (1 + h/(2l)) for linear elements.
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


# ---------------------------------------------------------------------------
# Standard boundary conditions / specimens
# ---------------------------------------------------------------------------
def bar_bc(grid: Grid):
    """Uniaxial bar along x: u_x = 0 on the left edge, u_x = U on the right edge, u_y = 0 at node (0,0)."""
    nodes = grid.nodes
    left = np.isclose(nodes[:, 0], grid.x0)
    right = np.isclose(nodes[:, 0], grid.x0 + grid.width)
    fixed = np.zeros((grid.nnode, 2), bool)
    fixed[left, 0] = True
    fixed[right, 0] = True
    fixed[0, 1] = True

    def bc(U):
        uD = np.zeros((grid.nnode, 2))
        uD[right, 0] = U
        return uD, fixed

    return bc, right


def sent_bc(grid: Grid):
    """Clamped single-edge-notched tension: bottom edge u = 0, top edge u_x = 0, u_y = U."""
    nodes = grid.nodes
    bot = np.isclose(nodes[:, 1], grid.y0)
    top = np.isclose(nodes[:, 1], grid.y0 + grid.height)
    fixed = np.zeros((grid.nnode, 2), bool)
    fixed[bot, :] = True
    fixed[top, :] = True

    def bc(U):
        uD = np.zeros((grid.nnode, 2))
        uD[top, 1] = U
        return uD, fixed

    return bc, top


def k_field(x, y, K, E, nu):
    """Mode-I plane-strain crack-tip (Williams) displacement field; crack along -x."""
    mu = E / (2 * (1 + nu))
    kappa = 3 - 4 * nu
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    f = K / (2 * mu) * np.sqrt(r / (2 * np.pi))
    return f * np.cos(th / 2) * (kappa - np.cos(th)), f * np.sin(th / 2) * (kappa - np.cos(th))


def surfing_bc(grid: Grid, K: float, E: float, nu: float, x_tip0: float, ramp: float = 1.0):
    """Surfing BC (Hossain et al., JMPS 71 (2014) 15-32) on the whole outer boundary.

    Load parameter t: for t <= ramp the K-field at x_tip0 is ramped (K*t/ramp);
    for t > ramp the full field is translated to x_tip0 + (t - ramp).
    """
    nodes = grid.nodes
    bnd = (np.isclose(nodes[:, 0], grid.x0) | np.isclose(nodes[:, 0], grid.x0 + grid.width)
           | np.isclose(nodes[:, 1], grid.y0) | np.isclose(nodes[:, 1], grid.y0 + grid.height))
    fixed = np.zeros((grid.nnode, 2), bool)
    fixed[bnd, :] = True

    def bc(t):
        s = min(t / ramp, 1.0) if ramp > 0 else 1.0
        xt = x_tip0 + max(t - ramp, 0.0)
        ux, uy = k_field(nodes[bnd, 0] - xt, nodes[bnd, 1], K * s, E, nu)
        uD = np.zeros((grid.nnode, 2))
        uD[bnd, 0], uD[bnd, 1] = ux, uy
        return uD, fixed

    return bc


def crack_tip_x(grid: Grid, d: np.ndarray, j_row: int, threshold: float = 0.95) -> float:
    """Largest x on node row j_row where d >= threshold (linear interpolation)."""
    row = d[grid.node_id(np.arange(grid.nx + 1), j_row)]
    idx = np.nonzero(row >= threshold)[0]
    if idx.size == 0:
        return grid.x0
    i = int(idx.max())
    x = grid.x0 + i * grid.h
    if i < grid.nx and row[i] > row[i + 1]:
        x += grid.h * (row[i] - threshold) / (row[i] - row[i + 1])
    return x


def crack_path(grid: Grid, d: np.ndarray, x_from: float = 0.0, threshold: float = 0.9):
    """Crack path y(x) = argmax_y d along each node column where max d >= threshold."""
    D = d.reshape(grid.ny + 1, grid.nx + 1)
    xs = grid.x0 + np.arange(grid.nx + 1) * grid.h
    jmax = np.argmax(D, axis=0)
    dmax = D.max(axis=0)
    sel = (dmax >= threshold) & (xs >= x_from)
    return xs[sel], grid.y0 + jmax[sel] * grid.h


# ---------------------------------------------------------------------------
# Verification cases
# ---------------------------------------------------------------------------
def verify_at2_bar(nx: int = 20, l: float = 0.1, E: float = 1.0, nu: float = 0.2, Gc: float = 1.0,
                   n_steps: int = 80) -> dict:
    """Homogeneous AT2 bar: sigma(eps) vs the analytic homogeneous solution; peak vs (3/16) sqrt(3E'Gc/l)."""
    L = 1.0
    grid = Grid(nx, 1, L / nx)
    mat = Material.homogeneous(grid, E, nu, Gc)
    model = PhaseFieldModel("at2", length=l)
    solver = PhaseFieldSolver(grid, model)
    Ep = E / (1 - nu**2)
    eps_c = math.sqrt(Gc / (3 * Ep * l))
    loads = np.linspace(0, 2.0 * eps_c * L, n_steps + 1)[1:]
    bc, right = bar_bc(grid)
    res = solver.run(mat, bc, loads, observers={"F": lambda u, d, f: f[right, 0].sum()})
    sig = res["F"] / grid.height
    eps = loads / L
    x = Ep * eps**2 * l / Gc
    sig_an = Ep * eps / (1 + x) ** 2
    sc_an = 3.0 / 16.0 * math.sqrt(3 * Ep * Gc / l)
    pre = eps <= eps_c * (1 + 1e-9)  # homogeneous branch is stable up to the peak (localises after)
    err_curve = float(np.max(np.abs(sig[pre] - sig_an[pre])) / sc_an)
    eb = float(np.max(np.abs(res["external_work"][pre] - res["total"][pre])) / max(res["external_work"][pre][-1], 1e-300))
    return {"case": "at2_homogeneous_bar", "l": l, "h": grid.h, "E_plane_strain": Ep,
            "sigma_peak": float(sig.max()), "sigma_peak_analytic": sc_an,
            "rel_err_peak": float(abs(sig.max() - sc_an) / sc_an), "max_curve_err_over_sigma_c": err_curve,
            "energy_balance_rel_err": eb, "all_converged": bool(res["converged"].all()),
            "eps": eps.tolist(), "sigma": sig.tolist(), "sigma_analytic": sig_an.tolist()}


def verify_pfczm_bar(b: float, h: float, softening: Softening = "linear", E: float = 1.0, Gf: float = 1.0,
                     ft: float = 1.0, L: float = 1.0, weak: float = 0.02, n_steps: int = 120) -> dict:
    """PF-CZM bar (nu = 0) with a slightly weakened central column (f_t*(1-weak)).

    Checks: peak stress = f_t (independent of b); dissipated energy -> G_f * area * discrete factor;
    softening branch vs the target traction-separation law; energy balance W_ext = E_el + Gamma.
    """
    nx = int(round(L / h))
    grid = Grid(nx, 1, L / nx)
    mat = Material.homogeneous(grid, E, 0.0, Gf, ft)
    mat.ft[nx // 2] *= (1 - weak)
    model = PhaseFieldModel("pfczm", length=b, softening=softening)
    solver = PhaseFieldSolver(grid, model)
    wc = 2 * Gf / ft if softening == "linear" else 12.0 * Gf / ft
    u_pk = ft * (1 - weak) * L / E
    loads = np.concatenate([np.linspace(0, 0.9 * u_pk, 10)[1:], np.linspace(0.9 * u_pk, 1.1 * u_pk, 41)[1:],
                            np.linspace(1.1 * u_pk, u_pk + 1.05 * wc, n_steps)[1:]])
    bc, right = bar_bc(grid)
    res = solver.run(mat, bc, loads, observers={"F": lambda u, d, f: f[right, 0].sum()})
    sig = res["F"] / grid.height
    area = grid.height
    fac = discrete_toughness_factor("pfczm", grid.h / b)
    Gam = res["dissipated"][-1]
    ip = int(np.argmax(sig))
    wopen = loads - sig * L / E
    post = np.arange(len(sig)) > ip
    if softening == "linear":
        target = np.maximum(ft * (1 - wopen / wc), 0.0)
    else:
        target = ft * np.exp(-ft * wopen / Gf)
    tsl_err = float(np.max(np.abs(sig[post] - target[post])) / ft) if post.any() else float("nan")
    eb = float(np.max(np.abs(res["external_work"] - res["total"])) / max(res["external_work"][-1], 1e-300))
    return {"case": f"pfczm_bar_{softening}", "b": b, "h": grid.h, "sigma_peak": float(sig.max()),
            "ft_weak_column": ft * (1 - weak), "rel_err_peak": float(abs(sig.max() - ft * (1 - weak)) / ft),
            "dissipated": float(Gam), "Gf_area": Gf * area, "discrete_factor": fac,
            "rel_err_dissipated_vs_Gf": float(abs(Gam / (Gf * area) - 1.0)),
            "rel_err_dissipated_vs_Gf_discrete": float(abs(Gam / (Gf * area * fac) - 1.0)),
            "tsl_max_err_over_ft": tsl_err, "energy_balance_rel_err": eb,
            "all_converged": bool(res["converged"].all()), "final_max_d": float(res["max_d"][-1]),
            "opening": wopen.tolist(), "sigma": sig.tolist()}


def verify_surfing(kind: ModelKind = "at2", h: float = 0.5, length: float = 1.0, Lx: float = 30.0,
                   Ly: float = 12.0, E: float = 1.0, nu: float = 0.3, Gc: float = 1.0, ft: float = 1.0,
                   G_applied_over_Gc: float = 2.0, a0: float = 6.0, advance: float = 12.0, dt: float = 0.5,
                   n_ramp: int = 8, softening: Softening = "linear", driving: Optional[Driving] = None,
                   verbose: bool = False) -> dict:
    """Edge-notched strip under surfing (translating K-field) BCs: stable steady crack growth.

    Measures dGamma/da over the steady part (-> G_eff = G_c * discrete factor), the
    crack-path deviation from the notch plane and the energy balance.
    """
    nx, ny = int(round(Lx / h)), int(round(Ly / h))
    if ny % 2:
        ny += 1
    grid = Grid(nx, ny, h, 0.0, -ny * h / 2)
    mat = Material.homogeneous(grid, E, nu, Gc, ft)
    model = PhaseFieldModel(kind, length=length, softening=softening, driving=driving)
    solver = PhaseFieldSolver(grid, model, cg_tol=1e-10)
    Ep = E / (1 - nu**2)
    fac = discrete_toughness_factor(kind, h / length)
    K = math.sqrt(G_applied_over_Gc * Gc * fac * Ep)
    jmid = ny // 2
    nodes = grid.nodes
    pre = (np.isclose(nodes[:, 1], 0.0)) & (nodes[:, 0] <= a0 + 1e-9)
    bc = surfing_bc(grid, K, E, nu, a0, ramp=1.0)
    loads = np.concatenate([np.linspace(0, 1.0, n_ramp + 1)[1:], 1.0 + np.arange(dt, advance + 1e-9, dt)])
    res = solver.run(mat, bc, loads, d_fixed_one=pre, verbose=verbose,
                     observers={"a": lambda u, d, f: crack_tip_x(grid, d, jmid)})
    a = res["a"]
    Gam = res["dissipated"]
    steady = loads >= 1.0 + advance / 2.0
    if steady.sum() >= 3 and np.ptp(a[steady]) > 2 * h:
        slope = float(np.polyfit(a[steady], Gam[steady], 1)[0])
    else:
        slope = float("nan")
    xs, ys = crack_path(grid, res["d"], x_from=a0 + length)
    dev = float(np.max(np.abs(ys))) if ys.size else float("nan")
    eb = float(abs(res["external_work"][-1] - res["total"][-1]) / max(res["external_work"][-1], 1e-300))
    return {"case": f"surfing_{kind}", "h": h, "length": length, "h_over_length": h / length,
            "G_eff_measured": slope, "discrete_factor": fac, "G_eff_expected": Gc * fac,
            "rel_err_vs_discrete": float(abs(slope / (Gc * fac) - 1.0)),
            "bourdin_factor_1_plus_h_over_2l": 1 + h / (2 * length) if kind == "at2" else None,
            "crack_advance": float(a[-1] - a0), "max_path_deviation": dev,
            "energy_balance_rel_err": eb, "all_converged": bool(res["converged"].all()),
            "a": a.tolist(), "Gamma": Gam.tolist(), "loads": loads.tolist(), "d": res["d"], "grid": grid}


# ---------------------------------------------------------------------------
# Microstructures
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class PhaseProps:
    E: float
    nu: float
    Gc: float = 1.0
    ft: float = 1.0
    damageable: bool = True


def random_disks(width: float, height: float, area_fraction: float, diameters: Sequence[float],
                 fractions: Sequence[float], min_gap: float, seed: int = 0, exclude=None,
                 max_tries: int = 200000) -> list[tuple[float, float, float]]:
    """Random sequential addition of non-overlapping disks (largest first) to a target area fraction.

    ``fractions``: share of the target area fraction for each diameter class.
    ``exclude(x, y, r) -> bool`` rejects placements (e.g. overlapping the notch).
    Periodic images are not used (particles may be cut by the boundary).
    """
    rng = np.random.default_rng(seed)
    A = width * height
    disks: list[tuple[float, float, float]] = []
    order = np.argsort(diameters)[::-1]
    for k in order:
        r = diameters[k] / 2.0
        target = area_fraction * fractions[k] * A
        placed = 0.0
        tries = 0
        while placed < target and tries < max_tries:
            tries += 1
            x, y = rng.uniform(0, width), rng.uniform(0, height)
            if exclude is not None and exclude(x, y, r):
                continue
            ok = True
            for (xc, yc, rc) in disks:
                if (x - xc) ** 2 + (y - yc) ** 2 < (r + rc + min_gap) ** 2:
                    ok = False
                    break
            if ok:
                disks.append((x, y, r))
                # area inside the box (approximate by clipping on a small grid)
                placed += _disk_area_in_box(x, y, r, width, height)
    return disks


def _disk_area_in_box(x, y, r, W, H, n=24):
    g = (np.arange(n) + 0.5) / n * 2 - 1
    X, Y = np.meshgrid(g, g)
    inside = X**2 + Y**2 <= 1
    px, py = x + r * X[inside], y + r * Y[inside]
    frac = np.mean((px >= 0) & (px <= W) & (py >= 0) & (py <= H))
    return frac * math.pi * r * r


def rasterize_microstructure(grid: Grid, disks, matrix: PhaseProps, particle: PhaseProps,
                             interface: Optional[PhaseProps] = None, interface_thickness: float = 0.0,
                             void_mask: Optional[np.ndarray] = None) -> Material:
    """Phase labels: 0 matrix, 1 particle, 2 interface band (matrix side), 3 void (notch)."""
    c = grid.centers
    phase = np.zeros(grid.nel, int)
    dist_rel = np.full(grid.nel, np.inf)
    for (x, y, r) in disks:
        dd = np.hypot(c[:, 0] - x, c[:, 1] - y) - r
        dist_rel = np.minimum(dist_rel, dd)
    phase[dist_rel <= 0] = 1
    if interface is not None and interface_thickness > 0:
        phase[(dist_rel > 0) & (dist_rel <= interface_thickness)] = 2
    if void_mask is not None:
        phase[np.asarray(void_mask, bool)] = 3
    props = {0: matrix, 1: particle, 2: interface or matrix,
             3: PhaseProps(E=matrix.E * 1e-9, nu=matrix.nu, Gc=matrix.Gc, ft=matrix.ft, damageable=False)}
    mat = Material(E=np.zeros(grid.nel), nu=np.zeros(grid.nel), Gc=np.zeros(grid.nel), ft=np.zeros(grid.nel),
                   damageable=np.zeros(grid.nel, bool), phase=phase)
    for k, p in props.items():
        s = phase == k
        mat.E[s], mat.nu[s], mat.Gc[s], mat.ft[s], mat.damageable[s] = p.E, p.nu, p.Gc, p.ft, p.damageable
    return mat


def notch_void_mask(grid: Grid, a0: float, y_notch: float) -> np.ndarray:
    """One element row of void elements from x = x0 to x0 + a0 centred on y_notch."""
    c = grid.centers
    return (c[:, 0] < grid.x0 + a0) & (np.abs(c[:, 1] - y_notch) < 0.5 * grid.h + 1e-12)


def run_notched_microstructure(grid: Grid, mat: Material, model: PhaseFieldModel, U_max: float, n_steps: int,
                               *, a0: float, y_notch: float, snapshot_every: int = 0, verbose: bool = False,
                               tol_E: float = 1e-6, tol_d: float = 1e-3, max_stagger: int = 200) -> dict:
    """Clamped SENT of a (micro)structure; effective fracture energy = dissipated / ligament."""
    solver = PhaseFieldSolver(grid, model, cg_tol=1e-8)
    bc, top = sent_bc(grid)
    loads = np.linspace(0, U_max, n_steps + 1)[1:]
    t0 = time.time()
    res = solver.run(mat, bc, loads, snapshot_every=snapshot_every, verbose=verbose, tol_E=tol_E, tol_d=tol_d,
                     max_stagger=max_stagger, observers={"F": lambda u, d, f: f[top, 1].sum()})
    lig = grid.width - a0
    F = res["F"]
    xs, ys = crack_path(grid, res["d"], x_from=grid.x0 + a0, threshold=0.9)
    if xs.size > 1:
        path_len = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
        tort = path_len / max(xs[-1] - xs[0], 1e-300)
        spanned = float((xs[-1] - xs[0] + grid.h) / lig)
    else:
        tort, spanned = float("nan"), 0.0
    # where did the crack run?  phase of elements adjacent to damaged nodes (d > 0.9)
    dn = res["d"] > 0.9
    el_d = dn[grid.conn].mean(axis=1) >= 0.5
    phase = mat.phase if mat.phase is not None else np.zeros(grid.nel, int)
    frac = {int(k): float(np.mean(phase[el_d] == k)) for k in np.unique(phase)} if el_d.any() else {}
    broken = F[-1] < 0.05 * F.max()  # residual load < 5 % of peak (tail energy ~1 % of Gamma)
    return {"loads": loads, "F": F, "F_peak": float(F.max()), "U_at_peak": float(loads[int(np.argmax(F))]),
            "dissipated": res["dissipated"], "Gamma_final": float(res["dissipated"][-1]),
            "G_eff_projected": float(res["dissipated"][-1] / lig), "ligament": lig, "tortuosity": tort,
            "ligament_fraction_cracked": spanned, "damaged_element_phase_fractions": frac,
            "fully_broken": bool(broken), "energy_balance_rel_err": float(
                abs(res["external_work"][-1] - res["total"][-1]) / max(res["external_work"][-1], 1e-300)),
            "all_converged": bool(res["converged"].all()), "n_unconverged": int((~res["converged"]).sum()),
            "wall_s": time.time() - t0, "d": res["d"], "snapshots": res["snapshots"],
            "external_work": res["external_work"], "elastic": res["elastic"]}


def plot_damage(grid: Grid, d: np.ndarray, mat: Optional[Material], path: str, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = d.reshape(grid.ny + 1, grid.nx + 1)
    fig, ax = plt.subplots(figsize=(6, 6 * grid.ny / grid.nx + 0.6))
    ext = (grid.x0, grid.x0 + grid.width, grid.y0, grid.y0 + grid.height)
    if mat is not None and mat.phase is not None:
        ph = mat.phase.reshape(grid.ny, grid.nx)
        ax.imshow(np.where(ph == 1, 1.0, np.where(ph == 2, 0.5, 0.0)), origin="lower", extent=ext, cmap="Greys",
                  vmin=0, vmax=1.6, interpolation="nearest")
    ax.imshow(np.ma.masked_less(D, 0.05), origin="lower", extent=ext, cmap="inferno_r", vmin=0, vmax=1,
              alpha=0.9, interpolation="bilinear")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dental application: notched random microstructures (units: um, MPa -> J/m^2)
# ---------------------------------------------------------------------------
def dental_phase_props() -> dict:
    """Constituent PhaseProps (um/MPa units) from composites.toughening.CONSTITUENTS / INTERFACES."""
    from fracture_md.composites.toughening import CONSTITUENTS as C, INTERFACES as I, g_from_k

    def props(key):
        c = C[key]
        return PhaseProps(E=c["E_GPa"] * 1e3, nu=c["nu"], Gc=g_from_k(c["K_Ic"], c["E_GPa"], c["nu"]), ft=c["ft_MPa"])

    m = props("matrix_dimethacrylate")
    out = {"matrix": m, "glass": props("glass_filler"), "zirconia": props("zirconia_filler")}
    for k, v in I.items():
        out[f"interface_{k}"] = PhaseProps(E=m.E, nu=m.nu, Gc=m.Gc * v["Gc_ratio"], ft=m.ft * v["ft_ratio"])
    return out


def build_dental_specimen(filler: Optional[str], phi: float, interface: str = "strong", *, h: float = 1.0,
                          width: float = 64.0, height: float = 48.0, a0_frac: float = 0.2,
                          diameters=(8.0, 4.0), fractions=(0.7, 0.3), min_gap: Optional[float] = None,
                          interface_thickness: Optional[float] = None, seed: int = 1):
    grid = Grid(int(round(width / h)), int(round(height / h)), h)
    min_gap = 2.0 * h if min_gap is None else min_gap
    interface_thickness = h if interface_thickness is None else interface_thickness
    P = dental_phase_props()
    a0 = a0_frac * grid.width
    yn = grid.height / 2
    # notch void row: element row whose centre is nearest to yn
    jn = int(yn / h)
    y_notch = (jn + 0.5) * h
    void = notch_void_mask(grid, a0, y_notch)
    disks = []
    if filler is not None and phi > 0:
        excl = lambda x, y, r: (x < a0 + r + min_gap) and abs(y - y_notch) < r + min_gap + h
        disks = random_disks(grid.width, grid.height, phi, diameters, fractions, min_gap, seed=seed, exclude=excl)
    mat = rasterize_microstructure(grid, disks, P["matrix"], P.get(filler, P["matrix"]),
                                   interface=P[f"interface_{interface}"] if disks else None,
                                   interface_thickness=interface_thickness if disks else 0.0, void_mask=void)
    ph = mat.phase
    solid = ph != 3
    area_frac = float(np.mean(ph[solid] == 1))
    return grid, mat, {"a0": a0, "y_notch": y_notch, "disks": disks, "particle_area_fraction": area_frac,
                       "interface_area_fraction": float(np.mean(ph[solid] == 2))}


def _save_gif(grid, mat, snaps, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ext = (grid.x0, grid.x0 + grid.width, grid.y0, grid.y0 + grid.height)
    fig, ax = plt.subplots(figsize=(5, 5 * grid.ny / grid.nx + 0.5))
    ph = mat.phase.reshape(grid.ny, grid.nx)
    ax.imshow(np.where(ph == 1, 1.0, np.where(ph == 2, 0.5, 0.0)), origin="lower", extent=ext, cmap="Greys",
              vmin=0, vmax=1.6, interpolation="nearest")
    im = ax.imshow(np.zeros((grid.ny + 1, grid.nx + 1)), origin="lower", extent=ext, cmap="inferno_r", vmin=0,
                   vmax=1, alpha=0.9)
    ttl = ax.set_title(title, fontsize=8)

    def upd(k):
        t, dd = snaps[k]
        im.set_data(np.ma.masked_less(dd.reshape(grid.ny + 1, grid.nx + 1), 0.05))
        ttl.set_text(f"{title}  U = {t:.2f} um")
        return (im,)

    anim = FuncAnimation(fig, upd, frames=len(snaps), blit=False)
    anim.save(path, writer=PillowWriter(fps=6))
    plt.close(fig)


DENTAL_CASES = [
    ("matrix", None, 0.0, "strong"),
    ("glass_40_strong", "glass", 0.40, "strong"),
    ("glass_50_strong", "glass", 0.50, "strong"),
    ("glass_50_weak", "glass", 0.50, "weak"),
    ("zirconia_40_strong", "zirconia", 0.40, "strong"),
]


def run_dental_case(name: str, filler, phi, interface, *, h=1.0, width=64.0, height=48.0, b_over_h=2.0,
                    U_max=14.0, n_steps=70, seed=1, fig_dir: Optional[str] = None, gif: bool = False,
                    driving: Driving = "isotropic", verbose: bool = False) -> dict:
    """PF-CZM (linear softening) SENT of a random microstructure.

    Default driving force is the variational ``isotropic`` one (energy balance verified); Wu's
    ``principal_stress`` driving gives nearly the same dissipated energy but is not variational.
    """
    grid, mat, info = build_dental_specimen(filler, phi, interface, h=h, width=width, height=height, seed=seed)
    model = PhaseFieldModel("pfczm", length=b_over_h * h, softening="linear", driving=driving)
    lc = check_pfczm_length(mat, model.length)
    res = run_notched_microstructure(grid, mat, model, U_max, n_steps, a0=info["a0"], y_notch=info["y_notch"],
                                     snapshot_every=max(n_steps // 40, 1) if gif else 0, verbose=verbose)
    figs = {}
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        p = os.path.join(fig_dir, f"fracture_{name}.png")
        plot_damage(grid, res["d"], mat, p, f"{name}: final damage (PF-CZM, b={model.length} um)")
        figs["png"] = p
        if gif and res["snapshots"]:
            g = os.path.join(fig_dir, f"fracture_{name}.gif")
            _save_gif(grid, mat, res["snapshots"], g, name)
            figs["gif"] = g
    return {"case": name, "filler": filler, "phi_target": phi, "interface": interface, "h_um": h, "b_um": model.length,
            "model": {"kind": model.kind, "softening": model.softening, "driving": model.driving_force,
                      "irreversibility": model.irreversibility},
            "width_um": grid.width, "height_um": grid.height, "a0_um": info["a0"], "seed": seed,
            "particle_area_fraction": info["particle_area_fraction"],
            "interface_area_fraction": info["interface_area_fraction"], "n_particles": len(info["disks"]),
            "pfczm_length_check": lc, "F_peak_N_per_m": res["F_peak"],  # MPa*um = N/m per unit thickness
            "U_at_peak_um": res["U_at_peak"], "Gamma_final_J_per_m": res["Gamma_final"] * 1e-6,
            "G_eff_projected_J_m2": res["G_eff_projected"], "tortuosity": res["tortuosity"],
            "ligament_fraction_cracked": res["ligament_fraction_cracked"],
            "damaged_element_phase_fractions": res["damaged_element_phase_fractions"],
            "fully_broken": res["fully_broken"], "energy_balance_rel_err": res["energy_balance_rel_err"],
            "all_converged": res["all_converged"], "n_unconverged_steps": res["n_unconverged"],
            "wall_s": res["wall_s"], "figures": figs,
            "F_U_curve": {"U_um": res["loads"].tolist(), "F_N_per_m": res["F"].tolist()}}


def run_verification_suite(level: str = "production") -> dict:
    """All solver verification cases (analytic / benchmark) with pass/fail checks."""
    prod = level == "production"
    out = {"level": level, "cases": [], "checks": []}

    def chk(name, ok, metric, tol, details):
        out["checks"].append({"name": name, "passed": bool(ok), "metric": float(metric), "tolerance": tol,
                              "details": details})

    r = verify_at2_bar()
    out["cases"].append({k: v for k, v in r.items() if not isinstance(v, list)})
    chk("AT2 bar peak stress = (3/16)sqrt(3E'Gc/l)", r["rel_err_peak"] < 1e-3, r["rel_err_peak"], 1e-3,
        f"sigma_peak {r['sigma_peak']:.6f} vs analytic {r['sigma_peak_analytic']:.6f}")
    chk("AT2 bar pre-peak sigma(eps) = analytic homogeneous solution", r["max_curve_err_over_sigma_c"] < 1e-3,
        r["max_curve_err_over_sigma_c"], 1e-3, "max |sigma - sigma_an| / sigma_c")
    chk("AT2 bar energy balance (pre-peak)", r["energy_balance_rel_err"] < 5e-3, r["energy_balance_rel_err"], 5e-3,
        "|W_ext - (E_el + Gamma)| / W_ext")
    for soft in ("linear", "exponential"):
        for b in ((0.1, 0.05) if prod else (0.1,)):
            r = verify_pfczm_bar(b=b, h=0.0125 if prod else 0.025, softening=soft, n_steps=80 if prod else 50)
            out["cases"].append({k: v for k, v in r.items() if not isinstance(v, list)})
            chk(f"PF-CZM {soft} bar peak = f_t (b={b})", abs(r["sigma_peak"] - 1.0) < 0.025,
                abs(r["sigma_peak"] - 1.0), 0.025, "peak stress / f_t - 1 (f_t = 1; one column 2% weaker)")
            chk(f"PF-CZM {soft} bar dissipated = G_f A (b={b})", r["rel_err_dissipated_vs_Gf"] < 0.08,
                r["rel_err_dissipated_vs_Gf"], 0.08, "includes diffuse pre-localisation damage")
            chk(f"PF-CZM {soft} bar softening law (b={b})", r["tsl_max_err_over_ft"] < 0.08,
                r["tsl_max_err_over_ft"], 0.08, "max |sigma - target(w)| / f_t on the softening branch")
            chk(f"PF-CZM {soft} bar energy balance (b={b})", r["energy_balance_rel_err"] < 0.01,
                r["energy_balance_rel_err"], 0.01, "|W_ext - (E_el + Gamma)| / W_ext")
    hs = (0.5, 0.25) if prod else (0.5,)
    conv = []
    for h in hs:
        r = verify_surfing("at2", h=h, length=1.0, Lx=30, Ly=12, a0=6, advance=12, dt=0.5)
        rr = {k: v for k, v in r.items() if not isinstance(v, (list, np.ndarray, Grid))}
        out["cases"].append(rr)
        conv.append(rr)
        chk(f"AT2 surfing SEN crack path straight (h/l={h})", r["max_path_deviation"] <= h + 1e-9,
            r["max_path_deviation"], h, "max |y_crack| (notch plane y=0)")
        chk(f"AT2 surfing energy balance (h/l={h})", r["energy_balance_rel_err"] < 0.02,
            r["energy_balance_rel_err"], 0.02, "|W_ext - (E_el + Gamma)| / W_ext at end")
        b_f = 1 + h / 2
        chk(f"AT2 surfing dGamma/da = G_c (1 + h/2l) (h/l={h})", abs(r["G_eff_measured"] / b_f - 1) < 0.05,
            abs(r["G_eff_measured"] / b_f - 1), 0.05, f"measured {r['G_eff_measured']:.4f}, Bourdin et al. 2008 {b_f:.4f}")
    out["at2_mesh_convergence"] = [{"h_over_l": c["h_over_length"], "G_eff_over_Gc": c["G_eff_measured"],
                                    "bourdin_1_plus_h_over_2l": c["bourdin_factor_1_plus_h_over_2l"]} for c in conv]
    if len(conv) >= 2:
        e = [c["G_eff_measured"] - 1.0 for c in conv]
        chk("AT2 G_eff -> G_c under mesh refinement (monotone)", all(e[i + 1] < e[i] for i in range(len(e) - 1)),
            e[-1], None, f"G_eff/G_c - 1 = {[round(x, 4) for x in e]}")
    for drv in (("principal_stress", "isotropic") if prod else ()):
        r = verify_surfing("pfczm", h=0.5, length=1.0, Lx=36, Ly=16, ft=0.5, a0=6, advance=14, dt=0.5,
                           driving=drv)
        rr = {k: v for k, v in r.items() if not isinstance(v, (list, np.ndarray, Grid))}
        rr["driving"] = drv
        out["cases"].append(rr)
        chk(f"PF-CZM surfing crack path straight ({drv})", r["max_path_deviation"] <= 0.5 + 1e-9,
            r["max_path_deviation"], 0.5, "max |y_crack|")
        chk(f"PF-CZM surfing dGamma/da = G_f ({drv})", r["rel_err_vs_discrete"] < 0.2, r["rel_err_vs_discrete"], 0.2,
            f"measured {r['G_eff_measured']:.4f} vs {r['G_eff_expected']:.4f}")
    out["all_passed"] = all(c["passed"] for c in out["checks"])
    return out


def assemble_results(data_dir: str = "data/dental", out_name: str = "fracture_results.json") -> dict:
    """Collect verification + microstructure cases, compute toughening ratios, compare with analytic models,
    and write data/dental/fracture_results.json (with PropertyRecords)."""
    import glob
    import json

    from fracture_md.composites import toughening as tg
    from fracture_md.records import make_record
    from fracture_md.schema.common import CheckResult, Conditions, Uncertainty, UncertaintyKind
    from fracture_md.schema.properties import Derivation, MethodSpec, PropertyName, PropertyOrigin, PropertyScale
    from fracture_md.units import Unit, q

    ver = {}
    for lvl in ("production", "quick"):
        pth = os.path.join(data_dir, f"fracture_verification_{lvl}.json")
        if os.path.exists(pth):
            with open(pth) as fh:
                ver = json.load(fh)
            break
    cases = {}
    for pth in sorted(glob.glob(os.path.join(data_dir, "fracture_case_*.json"))):
        with open(pth) as fh:
            c = json.load(fh)
        tag = os.path.basename(pth)[len("fracture_case_"):-5]
        Fc = c["F_U_curve"]["F_N_per_m"]
        c["fully_broken"] = bool(Fc[-1] < 0.05 * max(Fc))
        cases[tag] = c
    ref = cases.get("matrix")
    C = tg.CONSTITUENTS
    Gm = tg.g_from_k(C["matrix_dimethacrylate"]["K_Ic"], C["matrix_dimethacrylate"]["E_GPa"], C["matrix_dimethacrylate"]["nu"])
    rows, records = [], []
    cand = tg._default_candidate()
    for tag, c in cases.items():
        row = {k: c[k] for k in ("case", "filler", "phi_target", "interface", "particle_area_fraction",
                                 "interface_area_fraction", "n_particles", "h_um", "b_um", "G_eff_projected_J_m2",
                                 "F_peak_N_per_m", "tortuosity", "damaged_element_phase_fractions", "fully_broken",
                                 "energy_balance_rel_err", "all_converged", "wall_s", "figures") if k in c}
        row["tag"] = tag
        row["driving"] = c.get("model", {}).get("driving")
        if ref is not None:
            row["toughening_ratio_G"] = c["G_eff_projected_J_m2"] / ref["G_eff_projected_J_m2"]
            row["peak_load_ratio"] = c["F_peak_N_per_m"] / ref["F_peak_N_per_m"]
        phi = c["particle_area_fraction"]
        if c["filler"]:
            kind = f"{c['filler']}_filler"
            Ec, nuc = tg.mori_tanaka_spheres(C["matrix_dimethacrylate"]["E_GPa"], C["matrix_dimethacrylate"]["nu"],
                                             C[kind]["E_GPa"], C[kind]["nu"], phi)
            spec = {"fillers": [{"kind": kind, "volume_fraction": phi, "d50_um": 6.0}], "interface": c["interface"]}
            an = tg.predict_toughness(spec, use_phase_field=False)
            row["analytic_K_models_MPa_sqrt_m"] = an["models"]
            row["analytic_G_ratio_models"] = {k: tg.g_from_k(v, Ec, nuc) / Gm for k, v in an["models"].items()}
            row["analytic_crack_path_mode"] = an["crack_path_mode_over_particles"]
        else:
            Ec, nuc = C["matrix_dimethacrylate"]["E_GPa"], C["matrix_dimethacrylate"]["nu"]
        row["E_composite_MoriTanaka_GPa"] = Ec
        if "toughening_ratio_G" in row:
            row["K_c_estimate_MPa_sqrt_m"] = tg.k_from_g(row["toughening_ratio_G"] * Gm, Ec, nuc)
        rows.append(row)
        chks = [CheckResult(name="staggered scheme converged at every load step", passed=bool(c["all_converged"])),
                CheckResult(name="energy balance W_ext = E_el + Gamma", passed=c["energy_balance_rel_err"] < 0.02,
                            metric=c["energy_balance_rel_err"], tolerance=0.02),
                CheckResult(name="specimen fully separated", passed=bool(c["fully_broken"])),
                CheckResult(name="solver verification suite passed", passed=bool(ver.get("all_passed", False)),
                            details=f"{sum(x['passed'] for x in ver.get('checks', []))}/{len(ver.get('checks', []))} checks")]
        if row["driving"] == "principal_stress":
            chks = [x for x in chks if not x.name.startswith("energy balance")]
        rec = make_record(
            name=PropertyName.FRACTURE_ENERGY_SIMULATED, value=c["G_eff_projected_J_m2"], unit=Unit.J_PER_M2,
            candidate=cand, phase=tag,
            method=MethodSpec(name="2D plane-strain PF-CZM (Wu 2017) SENT of a random microstructure",
                              engine="fracture_md.composites.phase_field (JAX)",
                              description=f"dissipated energy / ligament; h={c['h_um']} um, b={c['b_um']} um, "
                                          f"driving={row['driving']}, linear softening",
                              assumptions=["2D (disks = cylinders), plane strain, quasi-static, no plasticity",
                                           "constituent inputs from composites.toughening.CONSTITUENTS/INTERFACES",
                                           "specimen (64x48 um) << matrix l_ch (~0.28 mm): strength-dominated, "
                                           "diffuse damage included in the dissipated energy"]),
            origin=PropertyOrigin.DERIVED_MODEL, scale=PropertyScale.MESOSCALE,
            conditions=Conditions(temperature=q(310.15, Unit.KELVIN), loading="quasistatic",
                                  stress_state="clamped single-edge-notched tension, mode I"),
            checks=chks,
            uncertainty=[Uncertainty(kind=UncertaintyKind.UNQUANTIFIED,
                                     description="single microstructure realisation; mesh h=b/2; constituent ranges not propagated")],
            derivation=Derivation(model="phase-field fracture energy", equation="G_eff = Gamma_final / (W - a0)",
                                  assumptions=["Gamma = sum G_c/(pi b) (w(d) + b^2 |grad d|^2)"]),
            notes=json.dumps({"toughening_ratio_G": row.get("toughening_ratio_G"), "tortuosity": row.get("tortuosity")}),
        )
        records.append(json.loads(rec.model_dump_json()))
    # seed statistics: base rows get mean/std over realisations; the ratio used downstream is the mean
    base = {r["tag"]: r for r in rows if "_" not in r["tag"].replace(r["case"], "")}
    for r in rows:
        r["is_base_case"] = r["tag"] in base
    for tag, r in base.items():
        vals = [x["G_eff_projected_J_m2"] for x in rows
                if x["case"] == r["case"] and (x["tag"] == tag or x["tag"].startswith(tag + "_seed"))]
        r["seed_stats"] = {"n": len(vals), "G_eff_values_J_m2": vals, "mean": float(np.mean(vals)),
                           "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
        if ref is not None and "toughening_ratio_G" in r:
            r["toughening_ratio_G_seed1"] = r["toughening_ratio_G"]
            r["toughening_ratio_G"] = float(np.mean(vals)) / ref["G_eff_projected_J_m2"]
            if r["seed_stats"]["std"] is not None:
                r["toughening_ratio_G_std"] = r["seed_stats"]["std"] / ref["G_eff_projected_J_m2"]
            Ec, nuc = r["E_composite_MoriTanaka_GPa"], None
            ph = r["particle_area_fraction"]
            if r["filler"]:
                kind = f"{r['filler']}_filler"
                Ec, nuc = tg.mori_tanaka_spheres(C["matrix_dimethacrylate"]["E_GPa"], C["matrix_dimethacrylate"]["nu"],
                                                 C[kind]["E_GPa"], C[kind]["nu"], ph)
            else:
                nuc = C["matrix_dimethacrylate"]["nu"]
            r["K_c_estimate_MPa_sqrt_m"] = tg.k_from_g(r["toughening_ratio_G"] * Gm, Ec, nuc)
    rows = sorted(rows, key=lambda x: (not x["is_base_case"], x["tag"]))
    # design-agent predictions and the SprintRay biaxial comparison
    specs = {
        "glass50_strong": {"fillers": [{"kind": "glass_filler", "volume_fraction": 0.5, "d50_um": 1.0}], "interface": "strong"},
        "glass45_zirconia10_strong": {"fillers": [{"kind": "glass_filler", "volume_fraction": 0.45, "d50_um": 1.0},
                                                  {"kind": "zirconia_filler", "volume_fraction": 0.10, "d50_um": 0.4}],
                                      "interface": "strong"},
        "glass50_weak": {"fillers": [{"kind": "glass_filler", "volume_fraction": 0.5, "d50_um": 1.0}], "interface": "weak"},
    }
    with open(os.path.join(data_dir, "_pf_tmp.json"), "w") as fh:
        json.dump({"microstructure_study": {"cases": [r for r in rows if r["is_base_case"]]}}, fh, default=float)
    preds = {}
    for k, sp in specs.items():
        t = tg.predict_toughness(sp, results_path=os.path.join(data_dir, "_pf_tmp.json"))
        f = tg.predict_flexural_strength(sp, toughness=t)
        preds[k] = {"spec": sp, "K_Ic": {x: t[x] for x in ("value", "range", "unit", "method", "models")},
                    "flexural_3pt": {x: f[x] for x in ("value", "range", "unit", "test")},
                    "biaxial_equivalent": {"value": f["biaxial_equivalent"], "range": f["biaxial_range"],
                                           "test": f["biaxial_test"]}}
        records.append(json.loads(tg.toughness_record(t).model_dump_json()))
        records.append(json.loads(tg.flexural_record(f).model_dump_json()))
    os.remove(os.path.join(data_dir, "_pf_tmp.json"))
    sprint = {"Midas_Restore_biaxial_mean_MPa": 180.0,
              "equivalent_ISO4049_3pt_MPa": {str(m): tg.biaxial_to_3pt(180.0, m) for m in (8, 10, 15)},
              "note": "Weibull surface-flaw scaling only (same material, same flaw population); the 510(k) 3-point "
                      "value 120-150 MPa (ISO 10477) is below this, consistent with extra edge/ageing flaws in bars."}
    out = {"meta": {"module": "fracture_md.composites.phase_field / toughening", "units": "J/m^2, MPa, MPa*m^0.5, um",
                    "verification_all_passed": ver.get("all_passed"), "verification_device": ver.get("device")},
           "verification": ver, "microstructure_study": {"reference_case": "matrix", "cases": rows},
           "predictions": preds, "sprintray_biaxial_comparison": sprint, "records": records}
    with open(os.path.join(data_dir, out_name), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    return out


def main(argv=None):  # pragma: no cover - CLI
    import argparse
    import json

    ap = argparse.ArgumentParser(description="phase-field verification and dental microstructure study")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--study", nargs="*", default=None, help="case names (default: all)")
    ap.add_argument("--level", default="production")
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--n-steps", type=int, default=70)
    ap.add_argument("--out", default="data/dental")
    ap.add_argument("--fig-dir", default="docs/dental/figures")
    ap.add_argument("--gif", default="glass_50_strong")
    ap.add_argument("--driving", default="isotropic")
    ap.add_argument("--tag", default="")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    dev = str(jax.devices()[0])
    if a.verify:
        v = run_verification_suite(a.level)
        v["device"] = dev
        with open(os.path.join(a.out, f"fracture_verification_{a.level}.json"), "w") as fh:
            json.dump(v, fh, indent=1, default=float)
        print(json.dumps(v["checks"], indent=1, default=float))
    if a.study is not None:
        names = a.study or [c[0] for c in DENTAL_CASES]
        for (name, filler, phi, iface) in DENTAL_CASES:
            if name not in names:
                continue
            r = run_dental_case(name, filler, phi, iface, h=a.h, n_steps=a.n_steps, fig_dir=a.fig_dir,
                                gif=(name == a.gif and not a.tag), verbose=a.verbose, driving=a.driving, seed=a.seed)
            r["device"] = dev
            with open(os.path.join(a.out, f"fracture_case_{name}{a.tag}.json"), "w") as fh:
                json.dump(r, fh, indent=1, default=float)
            print(name, {k: r[k] for k in ("G_eff_projected_J_m2", "F_peak_N_per_m", "tortuosity", "fully_broken",
                                           "energy_balance_rel_err", "all_converged", "wall_s")}, flush=True)
    if a.assemble:
        o = assemble_results(a.out)
        for c in o["microstructure_study"]["cases"]:
            print(c["tag"], c.get("toughening_ratio_G"), c.get("K_c_estimate_MPa_sqrt_m"), c.get("tortuosity"))


if __name__ == "__main__":  # pragma: no cover
    main()
