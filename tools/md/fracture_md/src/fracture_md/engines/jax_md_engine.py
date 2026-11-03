"""JAX-MD engine: dispatches engine-independent protocols to JAX physics runners."""

from __future__ import annotations

import importlib.metadata as md
from typing import Any, Optional

from fracture_md.engines.base import ProtocolOutput, RunContext, resolve
from fracture_md.potentials import get_potential
from fracture_md.schema.material import Candidate

# protocol kind -> "module:function"; imported lazily so that a broken runner
# only disables its own protocol.
RUNNERS: dict[str, str] = {
    "bulk_relax": "fracture_md.physics.relax:run_bulk_relax",
    "elastic_tensor": "fracture_md.physics.elastic:run_elastic_tensor",
    "cleavage": "fracture_md.physics.cleavage:run_cleavage",
    "tangential_slip": "fracture_md.physics.cleavage:run_tangential_slip",
    "ideal_strength": "fracture_md.physics.ideal_strength:run_ideal_strength",
    "autodiff_acceptance": "fracture_md.physics.autodiff_acceptance:run_autodiff_acceptance",
    "indentation": "fracture_md.physics.indentation:run_indentation",
}


class JaxMDEngine:
    name = "jax_md"

    def __init__(self) -> None:
        try:
            self.version = f"jax-md {md.version('jax-md')} / jax {md.version('jax')}"
        except md.PackageNotFoundError:  # pragma: no cover
            self.version = "unknown"

    def capabilities(self) -> set[str]:
        caps = set()
        for kind, spec in RUNNERS.items():
            try:
                resolve(spec)
                caps.add(kind)
            except (ImportError, AttributeError):
                pass
        return caps

    def run(
        self,
        protocol: Any,
        candidate: Candidate,
        potential: str,
        structure: Any = None,
        context: Optional[RunContext] = None,
    ) -> ProtocolOutput:
        kind = protocol.kind
        if kind not in RUNNERS:
            raise NotImplementedError(f"JAX-MD engine has no runner for protocol kind {kind!r}")
        runner = resolve(RUNNERS[kind])
        model = get_potential(potential)
        return runner(protocol, candidate, model, structure=structure, context=context or RunContext())
