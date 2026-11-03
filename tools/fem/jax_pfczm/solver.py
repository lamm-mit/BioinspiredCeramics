"""Staggered (alternate-minimisation) quasi-static solver for 2D plane-strain phase-field fracture.

Discretisation
--------------
* Square bilinear (Q4) elements of size ``h`` on an ``nx x ny`` grid; nodal
  displacement ``u`` (2 dof/node, interleaved) and nodal damage ``d``.
* Material parameters are piecewise constant per element (pixel): E, nu, G_c,
  f_t and a ``damageable`` flag (non-damageable pixels never degrade; nodes
  interior to them are fixed at d = 0).
* 2x2 Gauss quadrature.  The *pointwise* terms (degradation g(d), crack
  geometric function w(d)) are evaluated with the nodal damage of the node
  nearest to each Gauss point (a nodal/lumped quadrature), the gradient term
  |grad d|^2 with the consistent Q4 interpolation.  This makes the local part of
  the damage functional separable per node (diagonal Hessian) and is used
  *identically* in the displacement and damage subproblems, so the discrete
  energy is a single functional E_h(u, d).

Driving force / irreversibility
-------------------------------
* ``isotropic``: Y = psi0(eps) (variational; default for AT2).
* ``principal_stress`` (Wu 2017): Y = <sigma1_eff>^2 / (2 E).  Stiffness is degraded
  isotropically ("hybrid" formulation; not variational).
* Damage is bound-constrained d_n <= d <= 1 (irreversibility) and solved by a
  projected Newton method (active set + PCG + projected Armijo line search).
  Displacement: matrix-free Jacobi-PCG with Dirichlet masking.
* Staggered alternate minimisation per load step until the relative change of
  the total energy and max |delta d| are below tolerances.

Units: any consistent set (e.g. length um and stress MPa -> G_c in MPa*um = J/m^2,
energies per unit thickness in MPa*um^2 = 1e-6 J/m, forces per thickness in MPa*um = N/m).
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .grid import Grid, _bmatrix, _shape_gradients
from .models import Material, PhaseFieldModel, _element_parameters, softening_parameters


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
