"""Helpers that turn raw numbers into fully annotated PropertyRecords.

All physics runners build records through :func:`make_record` so that the
validation-status rule is applied uniformly:

    1. potential's domain for ANY property class the value depends on is
       ``known_deficient``  -> OUT_OF_DOMAIN   (numerical checks are still attached as evidence)
    2. any attached check failed                  -> FAILED
    3. experimental agreement evidence (passed)   -> VALIDATED_AGAINST_EXPERIMENT
    4. DFT agreement evidence (passed)            -> VALIDATED_AGAINST_DFT
    5. literature value for the same model        -> LITERATURE_MODEL_AGREEMENT
    6. reference-implementation agreement         -> REFERENCE_IMPLEMENTATION_AGREEMENT
    7. numerical checks attached and all passed   -> NUMERICALLY_VERIFIED
    8. otherwise                                  -> UNVALIDATED
    Derived-model values are always MODEL_ESTIMATE (unless 1 or 2 applies).

Names with a status ceiling (``schema.properties.STATUS_CEILING``: the ``*_MD``
indentation values, capped at LITERATURE_MODEL_AGREEMENT) skip rungs 3-7 that
lie above the ceiling: an experimental comparison stays attached as evidence
but cannot turn a model-specific MD hardness into a validated one.  A value
may depend on several property classes (:data:`PROPERTY_CLASSES`; indentation:
plasticity, high pressure, large strain and bond breaking).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Optional, Union

from fracture_md.schema.common import (
    CheckResult,
    Citation,
    Conditions,
    Provenance,
    Uncertainty,
    UncertaintyKind,
    Validation,
    ValidationEvidence,
    ValidationStatus,
    status_exceeds,
)
from fracture_md.schema.material import Candidate, InterfaceSpec, Orientation
from fracture_md.schema.potential import PotentialCard, PropertyClass
from fracture_md.schema.properties import (
    MD_INDENTATION_NAMES,
    STATUS_CEILING,
    Derivation,
    MethodSpec,
    PropertyName,
    PropertyOrigin,
    PropertyRecord,
    PropertyScale,
)
from fracture_md.units import ISO14577_HV_PER_MPA, Quantity, Unit, compatible, convert, h_it_to_hv

# Which potential-domain class governs each property (used for OUT_OF_DOMAIN).
PROPERTY_CLASS: dict[PropertyName, PropertyClass] = {
    PropertyName.LATTICE_PARAMETER: PropertyClass.EQUILIBRIUM_BULK,
    PropertyName.DENSITY: PropertyClass.EQUILIBRIUM_BULK,
    PropertyName.VOLUME_PER_ATOM: PropertyClass.EQUILIBRIUM_BULK,
    PropertyName.COHESIVE_ENERGY: PropertyClass.EQUILIBRIUM_BULK,
    PropertyName.ENERGY_PER_ATOM: PropertyClass.EQUILIBRIUM_BULK,
    PropertyName.RESIDUAL_STRESS: PropertyClass.EQUILIBRIUM_BULK,
    PropertyName.BORN_STABILITY: PropertyClass.ELASTIC,
    PropertyName.PHASE_TRANSITION_PRESSURE: PropertyClass.HIGH_PRESSURE,
    PropertyName.IDEAL_TENSILE_STRENGTH: PropertyClass.LARGE_STRAIN,
    PropertyName.IDEAL_SHEAR_STRENGTH: PropertyClass.LARGE_STRAIN,
    PropertyName.CRITICAL_STRAIN: PropertyClass.LARGE_STRAIN,
    PropertyName.SURFACE_ENERGY: PropertyClass.SURFACES,
    PropertyName.SURFACE_ENERGY_AVERAGE_PAIR: PropertyClass.SURFACES,
    PropertyName.WORK_OF_SEPARATION: PropertyClass.SURFACES,
    PropertyName.PEAK_NORMAL_TRACTION: PropertyClass.BOND_BREAKING,
    PropertyName.OPENING_AT_PEAK_TRACTION: PropertyClass.BOND_BREAKING,
    PropertyName.TRACTION_SEPARATION_NORMAL: PropertyClass.BOND_BREAKING,
    PropertyName.TRACTION_SEPARATION_TANGENTIAL: PropertyClass.STACKING_FAULTS,
    PropertyName.PEAK_TANGENTIAL_TRACTION: PropertyClass.STACKING_FAULTS,
    PropertyName.UNSTABLE_STACKING_FAULT_ENERGY: PropertyClass.STACKING_FAULTS,
    PropertyName.STACKING_FAULT_ENERGY: PropertyClass.STACKING_FAULTS,
    PropertyName.UBER_LENGTH_SCALE: PropertyClass.BOND_BREAKING,
    PropertyName.G_IC_ATOMISTIC: PropertyClass.SURFACES,
    PropertyName.INTERFACE_ENERGY: PropertyClass.INTERFACES,
    PropertyName.THERMAL_EXPANSION: PropertyClass.FINITE_TEMPERATURE,
}
_ELASTIC_PREFIXES = ("elastic_", "clamped_ion", "internal_strain", "bulk_modulus", "shear_modulus",
                     "youngs_modulus", "poisson_ratio", "zener", "universal_anisotropy", "pugh")

# Regimes an MD indentation value depends on (first = primary class).  A potential that is known_deficient
# in any of them (e.g. Tersoff 1989 / Erhart-Albe: compressive cutoff wall, cutoff-controlled bond breaking)
# yields OUT_OF_DOMAIN indentation records; protocol kind "indentation" uses the same set (agents.interface).
INDENTATION_DOMAIN_CLASSES: tuple[PropertyClass, ...] = (
    PropertyClass.PLASTICITY,
    PropertyClass.HIGH_PRESSURE,
    PropertyClass.LARGE_STRAIN,
    PropertyClass.BOND_BREAKING,
)
# All property classes a value depends on (multi-class; single-class names come from PROPERTY_CLASS).
PROPERTY_CLASSES: dict[PropertyName, tuple[PropertyClass, ...]] = {
    **{n: (c,) for n, c in PROPERTY_CLASS.items()},
    **{n: INDENTATION_DOMAIN_CLASSES for n in MD_INDENTATION_NAMES},
}


def property_class_for(name: PropertyName) -> Optional[PropertyClass]:
    """Primary property class of a name (see :func:`property_classes_for` for all of them)."""
    if name in PROPERTY_CLASS:
        return PROPERTY_CLASS[name]
    if name in PROPERTY_CLASSES:
        return PROPERTY_CLASSES[name][0]
    if name.value.startswith(_ELASTIC_PREFIXES):
        return PropertyClass.ELASTIC
    return None


def property_classes_for(name: PropertyName) -> tuple[PropertyClass, ...]:
    """Every potential property class the validity of `name` depends on (may be empty)."""
    if name in PROPERTY_CLASSES:
        return PROPERTY_CLASSES[name]
    pc = property_class_for(name)
    return (pc,) if pc is not None else ()


def _as_classes(property_class: Any) -> tuple[PropertyClass, ...]:
    if property_class is None:
        return ()
    if isinstance(property_class, str):  # PropertyClass members are str
        return (PropertyClass(property_class),)
    return tuple(PropertyClass(p) for p in property_class)


# Evidence kinds that raise the status, strongest first.
_EVIDENCE_RUNGS: tuple[tuple[str, ValidationStatus], ...] = (
    ("experimental_reference", ValidationStatus.VALIDATED_AGAINST_EXPERIMENT),
    ("dft_reference", ValidationStatus.VALIDATED_AGAINST_DFT),
    ("literature_model_value", ValidationStatus.LITERATURE_MODEL_AGREEMENT),
    ("reference_implementation", ValidationStatus.REFERENCE_IMPLEMENTATION_AGREEMENT),
)


def status_ceiling(name: Optional[PropertyName]) -> Optional[ValidationStatus]:
    """Highest validation status a record of `name` may carry (None = no cap)."""
    return STATUS_CEILING.get(PropertyName(name)) if name is not None else None


def decide_status(
    card: Optional[PotentialCard],
    property_class: Union[None, PropertyClass, Iterable[PropertyClass]],
    checks: Iterable[CheckResult],
    evidence: Iterable[ValidationEvidence],
    derived: bool = False,
    name: Optional[PropertyName] = None,
) -> ValidationStatus:
    """Validation status from the card's domain, the checks and the evidence (module docstring).

    `property_class` may be one class or several (OUT_OF_DOMAIN if the card marks ANY known_deficient);
    `name` applies the status ceiling of that property name.
    """
    checks = list(checks)
    evidence = list(evidence)
    classes = _as_classes(property_class)
    if card is not None and any(card.status_for(pc) == "known_deficient" for pc in classes):
        return ValidationStatus.OUT_OF_DOMAIN
    if any(not c.passed for c in checks):
        return ValidationStatus.FAILED
    if derived:
        return ValidationStatus.MODEL_ESTIMATE
    ceiling = status_ceiling(name)
    kinds = {e.kind for e in evidence if e.passed}
    for kind, status in _EVIDENCE_RUNGS:
        if kind in kinds and (ceiling is None or not status_exceeds(status, ceiling)):
            return status
    if checks and (ceiling is None or not status_exceeds(ValidationStatus.NUMERICALLY_VERIFIED, ceiling)):
        return ValidationStatus.NUMERICALLY_VERIFIED
    return ValidationStatus.UNVALIDATED


def capped_evidence_kinds(name: Optional[PropertyName], evidence: Iterable[ValidationEvidence]) -> list[str]:
    """Passed evidence kinds of `evidence` that the ceiling of `name` keeps from raising the status."""
    ceiling = status_ceiling(name)
    if ceiling is None:
        return []
    rung = dict(_EVIDENCE_RUNGS)
    return sorted({e.kind for e in evidence if e.passed and e.kind in rung and status_exceeds(rung[e.kind], ceiling)})


def _record_id(candidate_id: str, name: PropertyName, method: str, extra: Any) -> str:
    h = hashlib.sha256(json.dumps([candidate_id, name.value, method, extra], sort_keys=True, default=str).encode())
    return f"{candidate_id}:{name.value}:{h.hexdigest()[:12]}"


def make_record(
    *,
    name: PropertyName,
    value: Any,
    unit: Unit,
    candidate: Candidate,
    method: MethodSpec,
    card: Optional[PotentialCard] = None,
    phase: Optional[str] = None,
    origin: PropertyOrigin = PropertyOrigin.ATOMISTIC_COMPUTED,
    scale: PropertyScale = PropertyScale.ATOMISTIC_IDEAL,
    conditions: Optional[Conditions] = None,
    orientation: Optional[Orientation] = None,
    interface: Optional[InterfaceSpec] = None,
    uncertainty: Iterable[Uncertainty] = (),
    checks: Iterable[CheckResult] = (),
    evidence: Iterable[ValidationEvidence] = (),
    derivation: Optional[Derivation] = None,
    frame: str = "not_applicable",
    shape_description: str = "",
    notes: str = "",
    provenance: Optional[Provenance] = None,
    validation_notes: str = "",
    property_classes: Optional[Iterable[PropertyClass]] = None,
    citations: Iterable[Citation] = (),
) -> PropertyRecord:
    """Build a PropertyRecord with the uniform status rule (module docstring).

    `property_classes` overrides the classes of `name` (e.g. a value derived from an indentation
    hardness inherits the indentation classes of its input).
    """
    checks, evidence = list(checks), list(evidence)
    pclasses = property_classes_for(name) if property_classes is None else _as_classes(property_classes)
    status = decide_status(card, pclasses, checks, evidence, derived=origin == PropertyOrigin.DERIVED_MODEL,
                           name=name)
    vnotes = validation_notes
    if status == ValidationStatus.OUT_OF_DOMAIN and card is not None:
        bad = [pc.value for pc in pclasses if card.status_for(pc) == "known_deficient"]
        vnotes = (vnotes + " " if vnotes else "") + (
            f"Potential {card.id} is known_deficient for {', '.join(bad) or '?'}; "
            "numerical checks below verify the calculation, not the physics."
        )
    capped = capped_evidence_kinds(name, evidence) if status not in (
        ValidationStatus.OUT_OF_DOMAIN, ValidationStatus.FAILED, ValidationStatus.MODEL_ESTIMATE) else []
    if capped:
        vnotes = (vnotes + " " if vnotes else "") + (
            f"Status capped at {status_ceiling(name).value}: {name.value} is a model-specific MD value, so the "
            f"attached {', '.join(capped)} evidence is information only (different observable, depth, rate and "
            "tip; not a validation)."
        )
    phase_name = phase or (candidate.phases[0].name if candidate.phases else candidate.name)
    rid = _record_id(
        candidate.id,
        name,
        method.name,
        [card.id if card else None, orientation.model_dump() if orientation else None,
         interface.model_dump(mode="json") if interface else None, (conditions or Conditions()).model_dump(mode="json")],
    )
    return PropertyRecord(
        id=rid,
        name=name,
        value=value if isinstance(value, Quantity) else Quantity(value=value, unit=unit),
        shape_description=shape_description,
        frame=frame,  # type: ignore[arg-type]
        composition=candidate.composition,
        phase=phase_name,
        orientation=orientation,
        interface=interface,
        conditions=conditions or Conditions(),
        origin=origin,
        scale=scale,
        method=method,
        potential_id=card.id if card is not None else None,
        derivation=derivation,
        uncertainty=list(uncertainty),
        validation=Validation(status=status, evidence=evidence, checks=checks, notes=vnotes),
        provenance=provenance or Provenance(),
        citations=list(citations),
        notes=notes,
    )


# --------------------------------------------------------------------------
# Hardness conversions (the only sanctioned projected-area -> Vickers path)
# --------------------------------------------------------------------------
ISO14577_CITATION = Citation(key="ISO14577-1", text="ISO 14577-1 Instrumented indentation test; HV = 0.0945 H_IT")
_HV_SOURCE_NAMES = frozenset({
    PropertyName.INDENTATION_HARDNESS_MD_ESTIMATED_CONTACT,
    PropertyName.INDENTATION_HARDNESS_MD_OLIVER_PHARR,
    PropertyName.INDENTATION_HARDNESS_MEASURED,
})


def vickers_estimate_from_indentation(
    record: PropertyRecord,
    *,
    candidate: Candidate,
    card: Optional[PotentialCard] = None,
    method: Optional[MethodSpec] = None,
    notes: str = "",
) -> PropertyRecord:
    """Vickers number from a projected-area indentation hardness, as a DERIVED_MODEL record.

    HV = 0.0945 H_IT[MPa] (ISO 14577-1; 39.1 GPa -> 3695 HV), never the HV unit table (3987 HV, +7.9 %).
    The result is a ``vickers_hardness_estimate`` (status MODEL_ESTIMATE), whose derivation names the
    input record; it inherits the input's checks and, through `card`, its indentation property classes,
    so a converted Tersoff/Erhart-Albe MD hardness stays OUT_OF_DOMAIN.  A contact pressure measured
    before the plasticity gates pass (``indentation_contact_pressure_md``) is not a hardness and is
    refused.
    """
    if record.name not in _HV_SOURCE_NAMES:
        raise ValueError(f"{record.name.value} cannot be converted to a Vickers number; allowed inputs: "
                         f"{sorted(n.value for n in _HV_SOURCE_NAMES)} (a contact pressure is not a hardness)")
    if isinstance(record.value.value, list):
        raise ValueError("vickers_estimate_from_indentation expects a scalar hardness")
    hv = h_it_to_hv(record.value)
    md = record.name in MD_INDENTATION_NAMES
    assumptions = [
        "ISO 14577-1 relation HV = 0.0945 H_IT: H_IT uses the projected contact area, HV the sloped Vickers area; "
        "assumes geometrically similar Vickers and Berkovich impressions",
        "no indentation-size, rate or elastic-recovery difference between the source test and a Vickers test",
    ]
    if md:
        assumptions.append(
            "the input is a model-specific MD value (nm depth, 1e8-1e10 1/s, defect-free cell, idealised rigid tip, "
            "empirical potential): the converted number is not a prediction of a laboratory HV")
    unc = [Uncertainty(kind=UncertaintyKind.UNQUANTIFIED,
                       description="conversion model and indentation-size effect between the source test and HV")]
    for u in record.uncertainty:
        if u.value is not None and u.unit is not None and compatible(u.unit, Unit.MPA) and u.unit != Unit.HV:
            unc.append(u.model_copy(update={"value": convert(u.value, u.unit, Unit.MPA) * ISO14577_HV_PER_MPA,
                                             "unit": Unit.HV,
                                             "samples": None if u.samples is None else
                                             [convert(x, u.unit, Unit.MPA) * ISO14577_HV_PER_MPA for x in u.samples],
                                             "description": (u.description + " (converted with HV = 0.0945 H_IT)")
                                             .strip()}))
        elif u.value is None and u.relative is not None:
            unc.append(u)
    return make_record(
        name=PropertyName.VICKERS_HARDNESS_ESTIMATE,
        value=hv.value,
        unit=Unit.HV,
        candidate=candidate,
        method=method or MethodSpec(name="ISO 14577 HV conversion of an indentation hardness",
                                    engine="fracture_md.records",
                                    description="HV = 0.0945 H_IT[MPa]", assumptions=assumptions),
        card=card,
        phase=record.phase,
        origin=PropertyOrigin.DERIVED_MODEL,
        scale=record.scale,
        conditions=record.conditions,
        orientation=record.orientation,
        uncertainty=unc,
        checks=record.validation.checks,
        derivation=Derivation(model="ISO 14577-1 conversion of a projected-area indentation hardness to HV",
                              equation="HV = 0.0945 H_IT[MPa]", assumptions=assumptions,
                              input_record_ids=[record.id]),
        property_classes=property_classes_for(record.name),
        citations=[ISO14577_CITATION],
        notes=notes or (f"converted from {record.name.value} {record.value.magnitude(Unit.GPA):.4g} GPa"
                        + ("; MD estimate, not validated" if md else "")),
        provenance=record.provenance,
    )
