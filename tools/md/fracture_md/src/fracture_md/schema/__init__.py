"""Typed, unit-explicit, engine-independent schema for fracture_md."""

from fracture_md.schema.common import (
    EVIDENCE_LADDER,
    SCHEMA_VERSION,
    AreaEstimator,
    CheckResult,
    Citation,
    Conditions,
    ContactConditions,
    DefectState,
    EngineRunProvenance,
    Provenance,
    Uncertainty,
    UncertaintyKind,
    Validation,
    ValidationEvidence,
    ValidationStatus,
    status_exceeds,
)
from fracture_md.schema.experiment import (
    CriterionOutcome,
    ExperimentProposal,
    ExperimentResult,
    ExperimentStatus,
    SuccessCriterion,
)
from fracture_md.schema.formulation import Component, ComponentRole, Formulation, ProcessingRoute as PrintProcess
from fracture_md.schema.material import (
    Candidate,
    CandidateKind,
    Composition,
    EvidenceLevel,
    InterfaceKind,
    InterfaceSpec,
    ManufacturabilityEvidence,
    Microstructure,
    NoveltyAssessment,
    NoveltyStatus,
    Orientation,
    Phase,
    ProcessingRoute,
    StructureRecord,
    Termination,
)
from fracture_md.schema.potential import (
    DomainAssessment,
    PotentialCard,
    PotentialKind,
    PropertyClass,
    TrainingDomain,
)
from fracture_md.schema.properties import (
    DERIVED_MODEL_STATUSES,
    DERIVED_ONLY_NAMES,
    MD_INDENTATION_NAMES,
    MEASURED_ONLY_NAMES,
    PROJECTED_AREA_HARDNESS_NAMES,
    STATUS_CEILING,
    CohesiveLawFit,
    Derivation,
    IndentationCurve,
    IndentationSegment,
    MaterialRecord,
    MethodSpec,
    PropertyName,
    PropertyOrigin,
    PropertyRecord,
    PropertyScale,
    SampledCurve,
    TractionSeparationLaw,
    canonical_unit,
    check_vickers_conversions,
)
from fracture_md.schema.protocol import (
    AutodiffAcceptanceProtocol,
    BulkRelaxProtocol,
    CleavageProtocol,
    ElasticTensorProtocol,
    IdealStrengthProtocol,
    IndentationProtocol,
    IndenterSpec,
    MDSettings,
    Protocol,
    RelaxSettings,
    TangentialSlipProtocol,
)

__all__ = [name for name in dir() if not name.startswith("_")]
