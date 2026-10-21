"""Agent-ready experiment interface, versioned implementation registry and regression gates.

* :func:`submit` -- validate an :class:`~fracture_md.schema.experiment.ExperimentProposal`,
  apply the domain and implementation policies, run it and return an
  :class:`~fracture_md.schema.experiment.ExperimentResult` (persisted under ``runs/``).
* :class:`ImplementationRegistry` -- content-addressed store of agent-proposed code.
* :func:`gate` -- the numerical regression gate that decides ``trusted``/``rejected``.

See docs/AGENT_INTERFACE.md.
"""

from fracture_md.agents.gates import GATE_VERSION, GateReport, gate, gate_source, write_reference_result
from fracture_md.agents.interface import (
    DOMAIN_CLASSES,
    evaluate_criteria,
    make_proposal,
    resolve_model,
    submit,
)
from fracture_md.agents.registry import (
    ImplementationRegistry,
    RegistryError,
    TamperError,
)

__all__ = [
    "DOMAIN_CLASSES",
    "GATE_VERSION",
    "GateReport",
    "ImplementationRegistry",
    "RegistryError",
    "TamperError",
    "evaluate_criteria",
    "gate",
    "gate_source",
    "make_proposal",
    "resolve_model",
    "submit",
    "write_reference_result",
]
