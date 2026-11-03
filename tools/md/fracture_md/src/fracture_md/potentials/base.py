"""Common interface for energy models on the JAX path.

An :class:`EnergyModel` is a *pure* JAX function

    energy(frac, box, species, idx, params) -> scalar eV

plus metadata.  ``params`` is a pytree of float arrays, so observables can be
differentiated with respect to potential parameters as well as positions and
the cell.  ``species`` are indices into ``model.elements``.

Gradient conventions (JAX-MD, fractional coordinates):
    jax.grad(energy, 0)  -> real-space gradient dE/dR  [eV/Å]   (forces = minus this)
    jax.grad(energy, 1)  -> dE/d(box)                   [eV/Å]   (exact)
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from fracture_md.schema.potential import PotentialCard
from fracture_md.structures.system import AtomicSystem

EnergyFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]


def params_sha256(params: Any) -> str:
    """Canonical hash of a parameter pytree (float64 values rounded to 15 sig. digits)."""

    def _canon(x):
        if isinstance(x, dict):
            return {k: _canon(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple)):
            return [_canon(v) for v in x]
        a = np.asarray(x)
        if a.dtype.kind in "fc":
            return [float(f"{v:.15e}") for v in a.ravel()] + [list(a.shape)]
        if a.dtype.kind in "iub":
            return a.ravel().tolist() + [list(a.shape)]
        return str(x)

    payload = json.dumps(_canon(params), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class EnergyModel:
    name: str
    elements: tuple[str, ...]
    cutoff: float  # maximum interaction range (Å); neighbour lists use cutoff + skin
    params: Any
    energy_fn: EnergyFn
    card: PotentialCard

    # ---------------------------------------------------------------- helpers
    def species_for(self, system: AtomicSystem) -> jnp.ndarray:
        """Map a system's species indices onto this model's element ordering."""
        lut = []
        for el in system.elements:
            if el not in self.elements:
                raise ValueError(f"element {el} not supported by potential {self.name} ({self.elements})")
            lut.append(self.elements.index(el))
        return jnp.asarray(lut, dtype=jnp.int32)[system.species]

    # The three methods below dispatch to kernels compiled once per energy_fn
    # (and per input shape).  Calling energy_fn eagerly instead costs seconds
    # per call because JAX then executes thousands of tiny ops one by one.
    def energy(self, frac, box, species, idx, params=None) -> jnp.ndarray:
        return compiled_kernels(self.energy_fn)["energy"](frac, box, species, idx, self.params if params is None else params)

    def forces(self, frac, box, species, idx, params=None) -> jnp.ndarray:
        """Real-space forces (eV/Å)."""
        return compiled_kernels(self.energy_fn)["forces"](frac, box, species, idx, self.params if params is None else params)

    def stress(self, frac, box, species, idx, params=None) -> jnp.ndarray:
        """Cauchy stress (eV/Å^3), sign convention: positive = tensile.

        σ = (1/V) ∂E/∂ε at fixed fractional coordinates, evaluated with the
        symmetric small strain ε applied as box -> (I + ε) box.
        """
        return compiled_kernels(self.energy_fn)["stress"](frac, box, species, idx, self.params if params is None else params)


def _stress_kernel(energy_fn: EnergyFn):
    def stress(frac, box, species, idx, p):
        def e_of_eps(eps):
            eps_sym = 0.5 * (eps + eps.T)
            return energy_fn(frac, (jnp.eye(3) + eps_sym) @ box, species, idx, p)

        vol = jnp.abs(jnp.linalg.det(box))
        return jax.grad(e_of_eps)(jnp.zeros((3, 3), dtype=box.dtype)) / vol

    return stress


@functools.lru_cache(maxsize=None)
def compiled_kernels(energy_fn: EnergyFn) -> dict[str, Callable]:
    """jit-compiled energy / forces / stress for one energy function (cached)."""
    return {
        "energy": jax.jit(energy_fn),
        "forces": jax.jit(lambda *a: -jax.grad(energy_fn, argnums=0)(*a)),
        "energy_and_forces": jax.jit(lambda *a: _neg_grad(jax.value_and_grad(energy_fn, argnums=0)(*a))),
        "stress": jax.jit(_stress_kernel(energy_fn)),
    }


def _neg_grad(vg):
    e, g = vg
    return e, -g
