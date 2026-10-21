"""Agent-ready experiment interface: ExperimentProposal (JSON) -> ExperimentResult (JSON).

``submit(proposal)`` is the single entry point an agent (or a human) uses to
run a computational experiment.  It never raises (bad input, corrupt or
missing implementation files, optional dependencies that are not installed,
exceptions -- including ``SystemExit`` -- from registered code); everything is
reported in the returned :class:`ExperimentResult`, which is also persisted as
``runs/<result_id>/{proposal.json,result.json}``.

Pipeline and status semantics
-----------------------------
1. **Schema validation** (pydantic).  Invalid -> ``REJECTED``.
2. **Pre-run policy checks**, all collected before deciding:

   * success criteria are well formed (known property name, known unit with
     the right dimension -- also for indexed array properties such as
     ``elastic_stiffness_voigt[0,0]`` --, finite threshold, ``within`` has a
     finite non-negative tolerance);
   * protocol parameters that would only fail deep inside the run are checked
     up front (``autodiff_acceptance``: parameter names, finite positive
     ``fd_step`` and ``rel_tolerance``; ``indentation``: the indenter's lateral
     reach plus the potential cutoff must stay below half the cell, and the
     cell must be small enough for an in-process run -- production-size cells
     go through the LAMMPS deck generator);
   * the potential resolves: a built-in id/name (``potentials.get_potential``)
     or ``"impl:<name>@<sha12>"`` naming a *trusted* registered
     ``potential_energy_fn`` implementation;
   * **domain policy**: the protocol kind maps to potential property classes
     (:data:`DOMAIN_CLASSES`).  If the potential card marks any of them
     ``known_deficient`` and ``allow_out_of_domain`` is False -> ``REJECTED``.
     With the flag the run proceeds with warnings, and ``records.make_record``
     marks the affected records ``OUT_OF_DOMAIN``;
   * **implementation gate**: ``implementation_id`` None -> the built-in trusted
     runner; otherwise it must be a registered ``protocol_runner`` for the
     same protocol kind that is trusted in the sense of
     :meth:`ImplementationRegistry.trust_problems` (status ``trusted`` plus a
     complete, passing gate record of the current ``GATE_VERSION``);
   * an engine exists for the protocol kind.

   Any failure -> ``REJECTED`` (the physics is not run).
3. **Run** the runner.  An exception (or ``SystemExit``) -> ``FAILED``
   (message in ``errors``, traceback in ``runs/<id>/traceback.txt``).
4. **Post-run**: the validation status of every computed record is re-derived
   with :func:`fracture_md.records.decide_status` from the potential card and
   the record's own checks/evidence; a runner cannot claim more than its
   evidence supports (including the status ceiling of model-specific ``*_MD``
   indentation values), and records of a ``known_deficient`` property class are
   forced to ``OUT_OF_DOMAIN`` (a warning lists every change).  Then success
   criteria are evaluated against the produced
   PropertyRecords (values converted to the criterion's unit).  ``COMPLETED``
   iff every numerical check of the run passed (runner checks, checks attached
   to traction-separation laws, no record with validation status FAILED) and
   every criterion could be evaluated; otherwise ``PARTIAL``.  A criterion that
   is evaluated and *not* satisfied does not change the status: a falsified
   hypothesis is a valid scientific outcome and is reported in ``criteria``.

``max_wall_time_s`` is not enforced preemptively (runners are synchronous,
in-process JAX code); exceeding it adds a warning.

Indentation
-----------
``submit()`` runs the JAX harness indentation runner (``physics.indentation.run_indentation``) on small reference
cells only (:data:`INDENTATION_MAX_ATOMS_IN_PROCESS`).  Production indentation records are built OUTSIDE
``submit()``: production cells run in LAMMPS (the deck generator ``engines.lammps_indent``), and each run is
imported and analysed by ``physics.indentation.records_from_lammps_run``, which applies the same record rules (the
``*_MD`` status ceiling, the LAMMPS engine cap at ``numerically_verified``, the per-run Newton-3 and energy-ledger
checks, NOT EVALUATED checks for what a deck does not log).  A registered indentation ``protocol_runner`` is gated
on the small quasi-static reference case (``agents.gates``), never on a production run.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
from pydantic import ValidationError

from fracture_md.agents.registry import ImplementationRegistry, RegistryError, TamperError
from fracture_md.engines.dispatch import EngineUnavailable, select_engine
from fracture_md.engines.base import ProtocolOutput, RunContext
from fracture_md.schema.common import ValidationStatus
from fracture_md.schema.experiment import (
    CriterionOutcome,
    ExperimentProposal,
    ExperimentResult,
    ExperimentStatus,
    SuccessCriterion,
)
from fracture_md.schema.material import Candidate, CandidateKind
from fracture_md.records import INDENTATION_DOMAIN_CLASSES
from fracture_md.schema.potential import PropertyClass
from fracture_md.schema.properties import ARRAY_PROPERTY_UNITS, PropertyName, PropertyRecord, scalar_unit
from fracture_md.units import Unit, UnitError, conversion_factor, convert

# ARRAY_PROPERTY_UNITS now lives in schema.properties (it also drives the record-level dimension check) and is
# re-exported here for the pre-run check of indexed criteria such as "elastic_stiffness_voigt[0,0]".
# Protocol kind -> potential property classes whose validity the result depends on.
DOMAIN_CLASSES: dict[str, frozenset[PropertyClass]] = {
    "bulk_relax": frozenset({PropertyClass.EQUILIBRIUM_BULK}),
    "elastic_tensor": frozenset({PropertyClass.ELASTIC}),
    "cleavage": frozenset({PropertyClass.SURFACES, PropertyClass.BOND_BREAKING}),
    "tangential_slip": frozenset({PropertyClass.STACKING_FAULTS}),
    "ideal_strength": frozenset({PropertyClass.LARGE_STRAIN}),
    "autodiff_acceptance": frozenset(),
    "indentation": frozenset(INDENTATION_DOMAIN_CLASSES),
}

# In-process (submit) indentation runs are for small reference cells; production-size cells are run with the
# LAMMPS deck generator and imported.  Estimated with a generic 0.1 atoms/A^3 (3C-SiC: 0.096-0.099).
INDENTATION_MAX_ATOMS_IN_PROCESS = 3_000_000
_INDENTATION_DENSITY_ESTIMATE_PER_A3 = 0.1

IMPL_PREFIX = "impl:"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CRIT_RE = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\[(?P<index>\d+(?:\s*,\s*\d+)*)\])?")


def default_registry() -> ImplementationRegistry:
    """The repository's committed store (``<repo>/implementations``), else ``./implementations``."""
    repo_store = _REPO_ROOT / "implementations"
    return ImplementationRegistry(repo_store if repo_store.is_dir() else "implementations")


# --------------------------------------------------------------------------
# Convenience constructors
# --------------------------------------------------------------------------
def reference_candidate_3c_sic() -> Candidate:
    """The 3C-SiC baseline candidate (a reference, not a discovery claim)."""
    from fracture_md.structures.builders import sic_3c_phase, sic_composition

    return Candidate(
        id="3C-SiC",
        name="3C-SiC (zincblende silicon carbide)",
        kind=CandidateKind.REFERENCE,
        composition=sic_composition(),
        phases=[sic_3c_phase()],
        hypothesis="Baseline: covalent, stiff, brittle reference ceramic.",
    )


def make_proposal(
    *,
    proposal_id: str,
    protocol: Any,
    potential_id: str,
    submitted_by: str = "human",
    hypothesis: str = "",
    candidate: Optional[Candidate] = None,
    **kw: Any,
) -> ExperimentProposal:
    """Build a validated ExperimentProposal (defaults to the 3C-SiC reference candidate)."""
    return ExperimentProposal(
        proposal_id=proposal_id,
        submitted_by=submitted_by,
        hypothesis=hypothesis,
        candidate=candidate or reference_candidate_3c_sic(),
        protocol=protocol,
        potential_id=potential_id,
        **kw,
    )


# --------------------------------------------------------------------------
# Potential resolution
# --------------------------------------------------------------------------
def resolve_model(potential_id: str, registry: Optional[ImplementationRegistry] = None):
    """Return (EnergyModel, implementation_ids) for a built-in id or ``impl:<id>``.

    Raises KeyError / RegistryError / TamperError with a readable message.
    """
    from fracture_md.potentials import get_potential

    if potential_id.startswith(IMPL_PREFIX):
        impl_id = potential_id[len(IMPL_PREFIX):]
        reg = registry or default_registry()
        model = reg.energy_model(impl_id, require_trusted=True)
        return model, {"potential": model.card.id, "potential_energy_fn": impl_id}
    model = get_potential(potential_id)
    return model, {"potential": model.card.id, "potential_energy_fn": model.card.implementation}


def domain_status(card, kind: str) -> dict[str, str]:
    """{property_class: status} for the classes a protocol kind depends on."""
    return {pc.value: card.status_for(pc) for pc in sorted(DOMAIN_CLASSES.get(kind, ()), key=lambda p: p.value)}


# --------------------------------------------------------------------------
# Success criteria
# --------------------------------------------------------------------------
def _parse_criterion_name(s: str) -> tuple[str, Optional[tuple[int, ...]]]:
    m = _CRIT_RE.fullmatch(s.strip())
    if not m:
        raise ValueError(f"malformed property_name {s!r} (use '<property>' or '<property>[i,j]')")
    idx = tuple(int(x) for x in m.group("index").split(",")) if m.group("index") else None
    return m.group("name"), idx


def validate_criteria(criteria: Iterable[SuccessCriterion]) -> list[str]:
    """Pre-run problems with the success criteria (empty list = OK)."""
    problems = []
    valid_names = {p.value for p in PropertyName}
    for i, c in enumerate(criteria):
        try:
            name, _ = _parse_criterion_name(c.property_name)
        except ValueError as exc:
            problems.append(f"criterion {i}: {exc}")
            continue
        if name not in valid_names:
            problems.append(f"criterion {i}: unknown property {name!r}")
            continue
        try:
            unit = Unit(c.unit)
        except ValueError:
            problems.append(f"criterion {i}: unknown unit {c.unit!r}; valid units: {[u.value for u in Unit]}")
            continue
        canon = scalar_unit(PropertyName(name)) or ARRAY_PROPERTY_UNITS.get(name)
        if canon is not None:
            try:
                conversion_factor(canon, unit)
            except UnitError:
                problems.append(f"criterion {i}: unit {unit.value!r} is incompatible with {name} ({canon.value})")
        if not np.isfinite(c.threshold):
            problems.append(f"criterion {i}: threshold must be finite, got {c.threshold!r}")
        if c.comparator == "within" and c.tolerance is None:
            problems.append(f"criterion {i}: comparator 'within' requires a tolerance")
        if c.tolerance is not None and not (np.isfinite(c.tolerance) and c.tolerance >= 0):
            problems.append(f"criterion {i}: tolerance must be finite and >= 0, got {c.tolerance!r}")
    return problems


def validate_protocol(protocol, model=None) -> list[str]:
    """Pre-run problems with protocol parameters that would otherwise fail deep inside a run."""
    problems = []
    if getattr(protocol, "kind", None) == "autodiff_acceptance":
        from fracture_md.physics.autodiff_acceptance import parse_parameter

        for name in protocol.parameters:
            try:
                parse_parameter(name, model)
            except Exception as exc:  # noqa: BLE001 - ValueError for bad names; anything else is also a rejection
                problems.append(f"protocol.parameters: {exc}")
        if not protocol.parameters:
            problems.append("protocol.parameters: at least one parameter is required")
        if not (np.isfinite(protocol.fd_step) and 0.0 < protocol.fd_step <= 1.0e-2):
            problems.append(f"protocol.fd_step must be finite and in (0, 1e-2], got {protocol.fd_step!r}")
        if not (np.isfinite(protocol.rel_tolerance) and 0.0 < protocol.rel_tolerance < 1.0):
            problems.append(f"protocol.rel_tolerance must be finite and in (0, 1), got {protocol.rel_tolerance!r}")
    if getattr(protocol, "kind", None) == "indentation":
        problems += _validate_indentation(protocol, model)
    return problems


def _validate_indentation(protocol, model=None) -> list[str]:
    """Indentation pre-checks that need the potential (cutoff) or concern in-process feasibility."""
    problems = []
    lx, ly = protocol.lateral_size_A
    reach = protocol.indenter.lateral_extent_A(protocol.max_depth_A)
    if model is not None:
        rc = float(model.card.cutoff.magnitude(Unit.ANGSTROM))
        if not reach + rc < min(lx, ly) / 2.0:
            problems.append(f"protocol.lateral_size_A: indenter reach {reach:.1f} A + potential cutoff {rc:.2f} A "
                            f"must stay below min(lateral_size_A)/2 = {min(lx, ly) / 2.0:g} A (contact region and "
                            "its periodic image must not interact)")
        if protocol.fixed_thickness_A < rc:
            problems.append(f"protocol.fixed_thickness_A = {protocol.fixed_thickness_A:g} A is thinner than the "
                            f"potential cutoff {rc:.2f} A (mobile atoms would see a free bottom surface)")
    n_est = protocol.estimated_atoms(_INDENTATION_DENSITY_ESTIMATE_PER_A3)
    if n_est > INDENTATION_MAX_ATOMS_IN_PROCESS:
        problems.append(f"protocol: ~{n_est:.3g} atoms exceeds the in-process limit {INDENTATION_MAX_ATOMS_IN_PROCESS:.3g}; "
                        "run production-size indentation cells with the LAMMPS deck generator (engines.lammps_indent) "
                        "and build their records outside submit() with physics.indentation.records_from_lammps_run")
    return problems


def _compare(obs: float, c: SuccessCriterion) -> bool:
    if c.comparator == ">":
        return obs > c.threshold
    if c.comparator == ">=":
        return obs >= c.threshold
    if c.comparator == "<":
        return obs < c.threshold
    if c.comparator == "<=":
        return obs <= c.threshold
    return abs(obs - c.threshold) <= float(c.tolerance or 0.0)


def evaluate_criteria(criteria: Iterable[SuccessCriterion], records: Iterable[PropertyRecord]) -> list[CriterionOutcome]:
    """Evaluate pre-registered criteria against PropertyRecords.

    Records are matched by ``PropertyName`` value; an optional index selects a
    component of an array value (``"elastic_stiffness_voigt[0,0]"`` = C11).
    The record value is converted to the criterion's unit (dimension-checked).
    If several records match (e.g. several terminations), the criterion must
    hold for *all* of them; ``observed`` is the first value and ``details``
    lists every value.  ``satisfied`` is None when the criterion could not be
    evaluated (property not produced, unit mismatch, non-scalar value).
    """
    records = list(records)
    out = []
    for c in criteria:
        try:
            name, index = _parse_criterion_name(c.property_name)
            unit = Unit(c.unit)
        except ValueError as exc:
            out.append(CriterionOutcome(criterion=c, satisfied=None, details=str(exc)))
            continue
        matches = [r for r in records if r.name.value == name]
        if not matches:
            out.append(CriterionOutcome(criterion=c, satisfied=None,
                                        details=f"no {name!r} record was produced"))
            continue
        values, notes = [], []
        for r in matches:
            try:
                arr = np.asarray(convert(r.value.value, r.value.unit, unit), dtype=float)
            except UnitError as exc:
                notes.append(f"{r.id}: {exc}")
                continue
            if index is not None:
                try:
                    arr = arr[index]
                except (IndexError, TypeError):
                    notes.append(f"{r.id}: index {index} out of range for shape {arr.shape}")
                    continue
            if arr.ndim != 0:
                notes.append(f"{r.id}: value has shape {arr.shape}; give an index, e.g. {name}[0,0]")
                continue
            values.append((r, float(arr)))
        if not values:
            out.append(CriterionOutcome(criterion=c, satisfied=None, details="; ".join(notes)))
            continue
        sats = [_compare(v, c) for _, v in values]
        status = [f"{v:.6g} {unit.value} [{r.validation.status.value}]" for r, v in values]
        details = f"{len(values)} record(s): {', '.join(status)}"
        if any(r.validation.status == ValidationStatus.OUT_OF_DOMAIN for r, _ in values):
            details += "; WARNING: evaluated on OUT_OF_DOMAIN values"
        if notes:
            details += "; skipped: " + "; ".join(notes)
        out.append(CriterionOutcome(criterion=c, observed=values[0][1], satisfied=all(sats), details=details))
    return out


# --------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------
def _canonical_json(obj: Any) -> str:
    """Deterministic JSON text for hashing; never raises (mixed-type keys, NaN, odd objects)."""
    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except (TypeError, ValueError):
        pass
    try:  # e.g. {1: "a", "b": 2}: keys of mixed type cannot be sorted -> stringify keys first
        return json.dumps(_jsonable(obj), sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        return repr(obj)


def _sha256(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8", "replace")).hexdigest()


def _new_result_id(proposal_id: str, payload: Any) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(proposal_id or "invalid"))[:60].lstrip(".") or "invalid"
    return f"{safe}__{ts}__{_sha256([payload, ts])[:8]}"


def _jsonable(x: Any, depth: int = 0) -> Any:
    """Best-effort conversion of runner `raw` extras to JSON (never used for ranking)."""
    if depth > 6:
        return repr(x)[:200]
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if np.isfinite(x) else repr(x)
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    if isinstance(x, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in x]
    try:
        a = np.asarray(x)
        if a.dtype.kind in "biuf" and a.size <= 10000:
            return _jsonable(a.tolist(), depth + 1)
    except Exception:  # noqa: BLE001
        pass
    return repr(x)[:200]


def _structure_record(v: Any):
    """StructureRecord for an AtomicSystem or ase.Atoms, else None."""
    if hasattr(v, "to_record"):
        return v.to_record(source="protocol run artifact")
    try:
        from ase import Atoms

        from fracture_md.structures.system import AtomicSystem

        if isinstance(v, Atoms):
            return AtomicSystem.from_ase(v).to_record(source="protocol run artifact (ase.Atoms)")
    except Exception:  # noqa: BLE001
        return None
    return None


def _save_artifacts(run_dir: Path, output: ProtocolOutput) -> tuple[dict[str, str], list[str]]:
    """Persist runner artifacts: numeric arrays -> artifacts.npz, structures -> <name>.structure.json,
    JSON-able containers -> artifacts.json.  Returns ({name: relative path}, warnings)."""
    arts, warns = {}, []
    arrays, blobs = {}, {}
    items = dict(output.artifacts or {})
    if output.structure is not None:
        items.setdefault("final_structure", output.structure)
    for k, v in items.items():
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(k))
        if isinstance(v, (str, Path)) and (run_dir / str(v)).exists():
            arts[k] = str(v)  # runner already wrote a file relative to the run dir
            continue
        try:
            rec = _structure_record(v)
            if rec is not None:
                fn = f"{safe}.structure.json"
                (run_dir / fn).write_text(rec.model_dump_json(indent=1), encoding="utf-8")
                arts[k] = fn
                continue
        except Exception as exc:  # noqa: BLE001
            warns.append(f"artifact {k!r}: structure not persisted ({exc})")
            continue
        if isinstance(v, (dict, list, tuple)) or hasattr(v, "model_dump"):
            blobs[k] = _jsonable(v)
            continue
        try:
            a = np.asarray(v)
            if a.dtype.kind in "biuf":
                arrays[safe] = a
                continue
        except Exception:  # noqa: BLE001
            pass
        warns.append(f"artifact {k!r} of type {type(v).__name__} was not persisted")
    if arrays:
        np.savez_compressed(run_dir / "artifacts.npz", **arrays)
        arts.update({k: f"artifacts.npz:{k}" for k in arrays})
    if blobs:
        (run_dir / "artifacts.json").write_text(json.dumps(blobs, indent=1, default=str) + "\n", encoding="utf-8")
        arts.update({k: f"artifacts.json:{k}" for k in blobs})
    return arts, warns


def _persist(runs_dir: Optional[Union[str, Path]], result: ExperimentResult, proposal_payload: Any) -> Optional[Path]:
    if runs_dir is None:
        return None
    d = Path(runs_dir) / result.result_id
    d.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(proposal_payload, indent=1, default=str)
    except (TypeError, ValueError):
        text = json.dumps(_jsonable(proposal_payload), indent=1, default=str)
    (d / "proposal.json").write_text(text + "\n", encoding="utf-8")
    (d / "result.json").write_text(result.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return d


# --------------------------------------------------------------------------
# submit
# --------------------------------------------------------------------------
def _load_proposal(proposal) -> tuple[Optional[ExperimentProposal], Any, list[str]]:
    """(validated proposal or None, JSON payload to persist, errors)."""
    if isinstance(proposal, ExperimentProposal):
        return proposal, proposal.model_dump(mode="json"), []
    payload = proposal
    if isinstance(proposal, (str, Path)) and not (isinstance(proposal, str) and proposal.lstrip().startswith("{")):
        try:
            payload = json.loads(Path(proposal).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, {"source": str(proposal)}, [f"cannot read proposal {proposal}: {exc}"]
    elif isinstance(proposal, str):
        try:
            payload = json.loads(proposal)
        except json.JSONDecodeError as exc:
            return None, {"source": proposal[:2000]}, [f"proposal is not valid JSON: {exc}"]
    try:
        return ExperimentProposal.model_validate(payload), payload, []
    except ValidationError as exc:
        errs = [f"schema: {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:25]]
        return None, payload, errs


def _rederive_statuses(records: list[PropertyRecord], card) -> tuple[list[PropertyRecord], list[str]]:
    """Re-derive each computed record's validation status with records.decide_status.

    Records produced by a runner are only as trustworthy as their evidence.  For
    atomistic/derived records the status is recomputed from the potential card
    and the record's own checks/evidence; it replaces the runner's claim when
    the re-derived status is OUT_OF_DOMAIN or FAILED, or when the runner claimed
    a stronger status than its evidence supports.  A runner may always be *more*
    conservative (UNVALIDATED / MODEL_ESTIMATE / OUT_OF_DOMAIN / FAILED kept).
    A runner simulates and never measures: an EXPERIMENTAL or LITERATURE_COMPUTED
    record in its output is marked FAILED.
    """
    from fracture_md.records import decide_status, property_classes_for
    from fracture_md.schema.properties import PropertyOrigin

    conservative = {ValidationStatus.UNVALIDATED, ValidationStatus.MODEL_ESTIMATE, ValidationStatus.OUT_OF_DOMAIN,
                    ValidationStatus.FAILED}
    out, changes = [], []
    for r in records:
        if r.origin in (PropertyOrigin.EXPERIMENTAL, PropertyOrigin.LITERATURE_COMPUTED):
            # a runner simulates: it cannot produce a measurement or a published value
            have = r.validation.status
            note = (f"record rejected by submit(): a runner produced origin {r.origin.value}; measured and "
                    "literature values belong in validation evidence, not in runner output")
            val = r.validation.model_copy(update={"status": ValidationStatus.FAILED,
                                                  "notes": (r.validation.notes + " " if r.validation.notes else "")
                                                  + note})
            out.append(r.model_copy(update={"validation": val}))
            changes.append(f"{r.name.value} ({r.id}): {have.value} -> failed (runner-produced {r.origin.value})")
            continue
        if r.origin not in (PropertyOrigin.ATOMISTIC_COMPUTED, PropertyOrigin.DERIVED_MODEL):
            out.append(r)
            continue
        want = decide_status(card, property_classes_for(r.name), r.validation.checks, r.validation.evidence,
                             derived=r.origin == PropertyOrigin.DERIVED_MODEL, name=r.name)
        have = r.validation.status
        change = have != want and (want in (ValidationStatus.OUT_OF_DOMAIN, ValidationStatus.FAILED)
                                   or have not in conservative)
        if change:
            note = (f"status re-derived by submit(): runner reported {have.value}, evidence/card give {want.value}")
            val = r.validation.model_copy(update={"status": want,
                                                  "notes": (r.validation.notes + " " if r.validation.notes else "")
                                                  + note})
            r = r.model_copy(update={"validation": val})
            changes.append(f"{r.name.value} ({r.id}): {have.value} -> {want.value}")
        out.append(r)
    return out, changes


def _check_label(c) -> str:
    """A failed check's name for the warnings; a required check that could not be evaluated says so
    (physics.indentation.not_evaluated_check: ``details`` starts with "NOT EVALUATED")."""
    return f"{c.name} (not evaluated)" if str(c.details).startswith("NOT EVALUATED") else c.name


def submit(
    proposal: Union[ExperimentProposal, dict, str, Path],
    *,
    runs_dir: Optional[Union[str, Path]] = "runs",
    registry: Optional[ImplementationRegistry] = None,
    structure: Any = None,
) -> ExperimentResult:
    """Validate, police, run and record one experiment.  Never raises.

    Indentation: only small reference cells run here; production indentation records are built outside submit(),
    from LAMMPS runs, by ``physics.indentation.records_from_lammps_run`` (module docstring).

    `proposal` may be a model, a dict, a JSON string or a path to a JSON file.
    `runs_dir=None` disables persistence.  `structure` optionally passes a
    pre-relaxed AtomicSystem to the runner (not part of the JSON contract).
    Any unexpected internal error is reported as a FAILED result (with the
    traceback in ``errors``) instead of propagating.
    """
    t0 = time.perf_counter()
    try:
        return _submit(proposal, runs_dir=runs_dir, registry=registry, structure=structure, t0=t0)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - last line of defence: submit() never raises
        tb = traceback.format_exc()
        rid = _new_result_id("internal_error", repr(proposal)[:2000])
        res = ExperimentResult(result_id=rid, proposal_id="unknown", status=ExperimentStatus.FAILED,
                               errors=[f"internal error in submit(): {type(exc).__name__}: {exc}", tb[-4000:]],
                               wall_time_s=time.perf_counter() - t0)
        try:
            _persist(runs_dir, res, {"source": repr(proposal)[:2000]})
        except Exception:  # noqa: BLE001
            pass
        return res


def _submit(proposal, *, runs_dir, registry, structure, t0) -> ExperimentResult:
    from fracture_md.provenance import collect

    reg = registry or default_registry()
    prop, payload, errors = _load_proposal(proposal)
    raw_pid = prop.proposal_id if prop else (payload.get("proposal_id", "invalid") if isinstance(payload, dict)
                                             else "invalid")
    pid = raw_pid if isinstance(raw_pid, str) else _canonical_json(raw_pid)[:200]
    raw_by = prop.submitted_by if prop else (payload.get("submitted_by", "unknown") if isinstance(payload, dict)
                                             else "unknown")
    requested_by = raw_by if isinstance(raw_by, str) else _canonical_json(raw_by)[:200]
    result_id = _new_result_id(pid, payload)
    base_prov = dict(requested_by=requested_by,
                     parent_ids=list(prop.parent_result_ids) if prop else [],
                     input_hashes={"proposal_sha256": _sha256(payload)},
                     extra={"result_id": result_id, "proposal_id": pid})

    def finish(status, *, warnings=(), errs=(), **kw) -> ExperimentResult:
        prov = kw.pop("provenance", None) or collect(**base_prov)
        res = ExperimentResult(result_id=result_id, proposal_id=pid, status=status, warnings=list(warnings),
                               errors=list(errs), wall_time_s=time.perf_counter() - t0, provenance=prov, **kw)
        if runs_dir is not None:
            res = res.model_copy(update={"artifacts": {**res.artifacts, "proposal": "proposal.json", "result": "result.json"}})
        _persist(runs_dir, res, payload)
        return res

    if prop is None:
        return finish(ExperimentStatus.REJECTED, errs=errors)

    kind = prop.protocol.kind
    warnings: list[str] = []
    reasons: list[str] = list(validate_criteria(prop.success_criteria))

    # ---- potential (any exception -> REJECTED: optional dependency missing, corrupt store, import errors, ...)
    model, impl_ids = None, {}
    try:
        model, impl_ids = resolve_model(prop.potential_id, reg)
    except KeyError as exc:
        reasons.append(f"potential {prop.potential_id!r}: {exc.args[0] if exc.args else exc}")
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        reasons.append(f"potential {prop.potential_id!r}: {type(exc).__name__}: {exc}")
    reasons += validate_protocol(prop.protocol, model)

    # ---- domain policy
    if model is not None:
        ds = domain_status(model.card, kind)
        deficient = [pc for pc, st in ds.items() if st == "known_deficient"]
        for pc, st in ds.items():
            if st in ("unvalidated",):
                warnings.append(f"potential {model.card.id} is unvalidated for {pc} (protocol {kind})")
        if deficient:
            msg = (f"potential {model.card.id} is known_deficient for {deficient} which protocol {kind!r} "
                   "depends on")
            if prop.allow_out_of_domain:
                warnings.append(msg + "; running because allow_out_of_domain=True -- affected records are "
                                "marked OUT_OF_DOMAIN and must not be used as physical predictions")
            else:
                reasons.append(msg + "; set allow_out_of_domain=true to run anyway (results will be marked "
                               "OUT_OF_DOMAIN) or choose a potential validated for this regime")
        if model.card.kind.value == "test_fixture":
            warnings.append(f"potential {model.card.id} is a numerical test fixture, not a physical model")

    # ---- engine / implementation gate
    selection = None
    if prop.implementation_id is not None:
        if prop.engine == "ase_reference":
            reasons.append("implementation_id refers to a JAX protocol runner; it cannot run on engine 'ase_reference'")
        else:
            iid = prop.implementation_id
            try:
                man = reg.get_manifest(iid)
                untrusted = reg.trust_problems(iid, man)
                if man["kind"] != "protocol_runner":
                    reasons.append(f"implementation {iid} is a {man['kind']}, not a protocol_runner "
                                   "(potential implementations go in potential_id as 'impl:<id>')")
                elif man.get("protocol_kind") != kind:
                    reasons.append(f"implementation {iid} runs {man.get('protocol_kind')!r}, not {kind!r}")
                elif untrusted:
                    reasons.append(f"implementation {iid} has {'; '.join(untrusted)}"
                                   if untrusted[0].startswith("status '") else
                                   f"implementation {iid} is not trusted: {'; '.join(untrusted)}")
                else:
                    mod = reg.load(iid)
                    runner = getattr(mod, "run", None) or getattr(mod, "RUNNER", None)
                    if not callable(runner):
                        reasons.append(f"implementation {iid} defines no run()")
                    else:
                        from fracture_md.engines.dispatch import EngineSelection, _jax_md_version

                        selection = EngineSelection("jax_md", _jax_md_version(), iid, runner=runner)
                        impl_ids["protocol_runner"] = iid
            except (RegistryError, TamperError) as exc:
                reasons.append(f"implementation {iid!r}: {exc}")
            except (Exception, SystemExit) as exc:  # noqa: BLE001 - e.g. the module raises at import
                reasons.append(f"implementation {iid!r}: {type(exc).__name__}: {exc}")
    else:
        try:
            selection = select_engine(prop.engine, kind,
                                      need_energy_model=prop.potential_id.startswith(IMPL_PREFIX))
            impl_ids["protocol_runner"] = selection.runner_id
        except EngineUnavailable as exc:
            reasons.append(f"no engine can run {kind!r}: {exc}")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"engine selection for {kind!r} failed: {type(exc).__name__}: {exc}")

    base_prov.update(implementation_ids={str(k): str(v) for k, v in impl_ids.items()},
                     potential_id=model.card.id if model is not None else prop.potential_id,
                     protocol_id=prop.protocol.protocol_id(),
                     engine=selection.name if selection else "",
                     engine_version=selection.version if selection else "")
    base_prov["extra"].update(engine_requested=prop.engine, allow_out_of_domain=prop.allow_out_of_domain)

    if reasons:
        return finish(ExperimentStatus.REJECTED, warnings=warnings, errs=reasons,
                      criteria=[CriterionOutcome(criterion=c, details="not evaluated (rejected)")
                                for c in prop.success_criteria])

    # ---- run
    run_dir = Path(runs_dir) / result_id if runs_dir is not None else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(requested_by=requested_by, output_dir=str(run_dir) if run_dir else None,
                     implementation_ids=dict(impl_ids), parent_ids=list(prop.parent_result_ids),
                     extra={"proposal_id": prop.proposal_id, "result_id": result_id})

    def failed_result(stage: str, exc: BaseException) -> ExperimentResult:
        tb = traceback.format_exc()
        arts = {}
        if run_dir is not None:
            (run_dir / "traceback.txt").write_text(tb, encoding="utf-8")
            arts["traceback"] = "traceback.txt"
        return finish(ExperimentStatus.FAILED, warnings=warnings, artifacts=arts,
                      errs=[f"{stage}{type(exc).__name__}: {exc}"],
                      criteria=[CriterionOutcome(criterion=c, details="not evaluated (run failed)")
                                for c in prop.success_criteria])

    try:
        output = selection.run(prop.protocol, prop.candidate, model, prop.potential_id, structure=structure, context=ctx)
        if not isinstance(output, ProtocolOutput):
            raise TypeError(f"runner returned {type(output).__name__}, expected ProtocolOutput")
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - physics failures are reported, not raised
        return failed_result("", exc)

    # ---- post-run (a malformed ProtocolOutput is a failed run, not a crash)
    try:
        props, changed = _rederive_statuses(list(output.properties), model.card)
        if changed:
            warnings.append(f"{len(changed)} record status(es) re-derived from card and evidence: {changed[:20]}")
        failed = [f"check {_check_label(c)}" for c in output.checks if not c.passed]
        for i, tsl in enumerate(output.traction_separation):
            failed += [f"traction_separation[{i}] check {_check_label(c)}" for c in tsl.checks if not c.passed]
        for i, cur in enumerate(output.indentation_curves):
            failed += [f"indentation_curves[{i}] check {_check_label(c)}" for c in cur.checks if not c.passed]
        failed += [f"record {r.name.value} ({r.id}) has validation status FAILED" for r in props
                   if r.validation.status == ValidationStatus.FAILED]
        ood = sorted({r.name.value for r in props if r.validation.status == ValidationStatus.OUT_OF_DOMAIN})
        if ood:
            warnings.append(f"{len(ood)} property name(s) are OUT_OF_DOMAIN for {model.card.id}: {ood}")
        if not output.checks and not any(t.checks for t in output.traction_separation) and \
                not any(c.checks for c in output.indentation_curves):
            warnings.append("the runner attached no numerical checks; COMPLETED here means only that it ran")
        if failed:
            warnings.append(f"{len(failed)} numerical check(s) failed: {failed[:20]}")
        criteria = evaluate_criteria(prop.success_criteria, props)
        unevaluated = [o.criterion.property_name for o in criteria if o.satisfied is None]
        if unevaluated:
            warnings.append(f"criteria could not be evaluated: {unevaluated}")
        elapsed = time.perf_counter() - t0
        if elapsed > prop.max_wall_time_s:
            warnings.append(f"wall time {elapsed:.1f} s exceeded max_wall_time_s={prop.max_wall_time_s} "
                            "(not enforced preemptively)")
        arts: dict[str, str] = {}
        if run_dir is not None:
            arts, aw = _save_artifacts(run_dir, output)
            warnings += aw
        status = ExperimentStatus.PARTIAL if (failed or unevaluated) else ExperimentStatus.COMPLETED
        raw = {"runner_raw": _jsonable(output.raw), "domain_status": domain_status(model.card, kind)}
        return finish(status, warnings=warnings, properties=props,
                      traction_separation=output.traction_separation,
                      indentation_curves=list(output.indentation_curves), checks=output.checks,
                      criteria=criteria, artifacts=arts, raw=raw)
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        return failed_result("post-run processing of the runner output failed: ", exc)
