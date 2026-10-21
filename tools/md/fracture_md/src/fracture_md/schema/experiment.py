"""Agent-facing experiment interface.

An agent (or a human) submits an :class:`ExperimentProposal` as JSON and
receives an :class:`ExperimentResult` as JSON.  The proposal names a candidate,
a protocol, a potential and an implementation version; the harness refuses to
run implementations that have not passed their regression gate, and it flags
results whose property class lies outside the potential's validated domain.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from fracture_md.schema.common import CheckResult, Provenance, StrictModel
from fracture_md.schema.material import Candidate
from fracture_md.schema.properties import (
    IndentationCurve,
    PropertyRecord,
    TractionSeparationLaw,
    check_vickers_conversions,
)
from fracture_md.schema.protocol import Protocol


class SuccessCriterion(StrictModel):
    """A falsifiable expectation attached to a proposal (agents must pre-register)."""

    property_name: str
    comparator: Literal[">", ">=", "<", "<=", "within"]
    threshold: float
    unit: str
    tolerance: Optional[float] = None
    rationale: str = ""


class ExperimentProposal(StrictModel):
    proposal_version: str = "0.1.0"
    proposal_id: str
    submitted_by: str  # agent id or "human:<name>"
    hypothesis: str
    candidate: Candidate
    protocol: Protocol
    potential_id: str
    engine: Literal["jax_md", "ase_reference", "auto"] = "auto"
    # Versioned implementation of the protocol (content hash).  None = current trusted version.
    implementation_id: Optional[str] = None
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    max_wall_time_s: float = 600.0
    allow_out_of_domain: bool = False
    parent_result_ids: list[str] = Field(default_factory=list)
    notes: str = ""


class ExperimentStatus(str, Enum):
    COMPLETED = "completed"
    REJECTED = "rejected"  # failed pre-run validation (domain, implementation gate, schema)
    FAILED = "failed"  # ran but a hard physics/numerics check failed
    PARTIAL = "partial"  # ran, some checks failed or were skipped


class CriterionOutcome(StrictModel):
    criterion: SuccessCriterion
    observed: Optional[float] = None
    satisfied: Optional[bool] = None
    details: str = ""


class ExperimentResult(StrictModel):
    result_version: str = "0.1.0"
    result_id: str
    proposal_id: str
    status: ExperimentStatus
    properties: list[PropertyRecord] = Field(default_factory=list)
    traction_separation: list[TractionSeparationLaw] = Field(default_factory=list)
    indentation_curves: list[IndentationCurve] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    criteria: list[CriterionOutcome] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)  # name -> relative path
    wall_time_s: Optional[float] = None
    provenance: Provenance = Field(default_factory=Provenance)
    raw: dict[str, Any] = Field(default_factory=dict)  # engine-specific extras (never used for ranking)

    @model_validator(mode="after")
    def _derived_vickers(self) -> "ExperimentResult":
        problems = check_vickers_conversions(self.properties)
        if problems:
            raise ValueError("; ".join(problems))
        return self
