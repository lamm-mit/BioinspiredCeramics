"""The frozen indentation recipe (recipe_v2) and the canonical values: loader, validator, sha256 (critic item A3).

Synthesis step 1 asks for two inputs that are fixed before any pilot run, so that no analysis choice can be tuned
after the fact (SR-X10):

* ``data/indentation/canonical_values.json``: one value per quantity (Berkovich geometry, the experimental anchor
  and C_ij, the Vashishta 2007 / Erhart-Albe / Tersoff 1989 numbers, the calibrations of this project), each with its
  source and a verification label (reproduced, literature-verified, measured, computed, estimate, definition).
* ``data/indentation/recipe_v2.json`` (the default; it supersedes ``recipe_v1.json``, which did not freeze how a
  Gate I quench becomes a verdict, critic findings N1 / N2, and ``recipe_v0.json``, which named a hardness record
  that no longer exists, review finding T10): the protocol (tip, K, R_tip, azimuth, surface, preparation, cell tiers,
  boundary layers, dt, speed, hold / unload, output), the analysis (load windows and block averaging, depth zero,
  the estimated-contact area estimators E1-E5, the evaluation-depth rule), the gates (frozen Gate I with
  its evaluation rule ``gates/gate_i/evaluation``, Gate II thresholds marked ``provisional_until_S1``, Gate III,
  GATE-E), the per-run checks (Newton 3 with its minimum load, the energy ledger and its floor, CPU use), the
  pre-registered seeds and the record rules (``records/names``: the ``PropertyName`` of every record the runner
  emits, bound to the schema by :func:`code_consistency`).
  Physics-derived parameters name their canonical value (``canonical_refs``); the validator checks that each still
  agrees.

The recipe's identity is the sha256 of its file bytes (:func:`recipe_sha256`).  It is pinned here
(:data:`FROZEN_RECIPE_SHA256`): a frozen recipe whose bytes changed does not load, so a change needs a new
``recipe_v<N+1>.json`` and a new pin.  Both engine paths record the hash in ``EngineRunProvenance.recipe_sha256``:
the JAX runner (``physics.indentation.run_indentation``, option ``ctx.extra["indentation"]["recipe"]``) and the
LAMMPS path (the deck generator ``engines.lammps_indent`` copies the recipe into every run directory and states it
in ``manifest.json``; ``physics.indentation.records_from_lammps_run(recipe=...)`` hashes the recipe the records are
built under and checks the deck's recipe protocol section against it, :func:`deck_recipe_check`).

:func:`code_consistency` compares every frozen value with the code default or constant that implements it
(``schema.protocol``, ``physics.indent_analysis``, ``physics.indentation``, ``engines.lammps_indent``,
``scripts/gate_e.py``, the LAMMPS deck texts, ...): a code change that alters a frozen value shows up as a mismatch
until a new recipe is written.  :func:`protocol_deviations` lists where an ``IndentationProtocol`` departs from the
recipe (sanctioned sensitivity values are marked as such); runners and the deck generator store the list next to the
hash.

The preparation (``protocol/preparation``: z pre-expansion ``zscale``, Langevin of every mobile atom from
``start_temperature_K`` for 60 % of the equilibration, then the bath alone) is applied by the engines and is not a
field of ``IndentationProtocol``.  Each engine states what it applies (:func:`preparation_summary`; for a LAMMPS deck
:func:`lammps_deck_preparation` reads it from the deck text and its ``-var`` values), and ``protocol_deviations(...,
preparation=)`` compares that with the recipe.  Departures the recipe itself declares acceptable
(``protocol/preparation/known_deviations``) are listed as sanctioned, with that pointer as their ``basis``
(:data:`KNOWN_DEVIATIONS`).  :func:`deck_recipe_check` fails a LAMMPS run whose deck manifest lists an unsanctioned
deviation, or states no deviations or no preparation.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_DIR = REPO_ROOT / "data" / "indentation"
CANONICAL_VALUES_PATH = RECIPE_DIR / "canonical_values.json"
GATE_E_SCRIPT = REPO_ROOT / "scripts" / "gate_e.py"

DEFAULT_RECIPE = "recipe_v2"
# sha256 of the frozen recipe files' bytes.  A frozen recipe is immutable: never edit this pin to follow an edited
# file; write recipe_v<N+1>.json and add its pin.  recipe_v1 is superseded by recipe_v2 (the Gate I evaluation rule of
# critic N1 / N2, gates/gate_i/evaluation, bound below); recipe_v0 by recipe_v1 (its records/names/hardness
# named indentation_hardness_md_true_contact, which the schema no longer has; review finding T10) and stays pinned so
# that what was built under it remains reproducible.
FROZEN_RECIPE_SHA256: dict[str, str] = {
    "recipe_v0": "559a21a46a3392f3eccdf7ad9bd780274036ff0fb8f75149c8847b631c3ef0d8",
    "recipe_v1": "cae77a8baa1902969da576053eef17312816939429e8411091c4d0671820b24c",
    "recipe_v2": "5fbbe6e8efc313f3b5783be7ef90ffd4795359924ddc2c8946f4fa333bfdf4fb",
}
RECIPE_SCHEMA = "fracture-md indentation recipe v1"
CANONICAL_SCHEMA_PREFIX = "fracture-md canonical values v1"
PROVISIONAL = "provisional_until_S1"
SECTION_STATUSES = ("frozen", PROVISIONAL)
VERIFICATION_LABELS = ("reproduced", "literature-verified", "measured", "computed", "estimate", "definition")
REQUIRED_SECTIONS = ("decisions", "scope", "protocol", "analysis", "gates", "per_run_checks", "statistics", "records",
                     "canonical_refs")
# The items critic A3 / synthesis step 1 require the recipe to freeze (each must resolve in every recipe version).
REQUIRED_POINTERS = (
    "protocol/potential/name", "protocol/potential/file_sha256",
    "protocol/tip/model", "protocol/tip/shape", "protocol/tip/face_angle_deg", "protocol/tip/tip_radius_A",
    "protocol/wall_stiffness/K_eV_per_A3", "protocol/orientation/azimuth_deg", "protocol/surface/termination",
    "protocol/surface/lattice_constant_A", "protocol/preparation/equilibration_ps", "protocol/preparation/zscale",
    "protocol/cell/tiers", "protocol/boundary_layers/fixed_thickness_A", "protocol/boundary_layers/bath_thickness_A",
    "protocol/boundary_layers/bath", "protocol/temperature_K", "protocol/time_step/dt_fs",
    "protocol/loading/speed_m_per_s", "protocol/hold_unload/hold_ps", "protocol/hold_unload/relax_after_unload_ps",
    "analysis/load/running_mean_window_ps", "analysis/load/block_average_ps", "analysis/load/min_window_over_tau_ac",
    "analysis/contact_area/estimators", "analysis/contact_area/primary", "analysis/contact_area/contact_force_fraction",
    "analysis/evaluation_depth/rule", "analysis/evaluation_depth/tolerance_A",
    "gates/gate_i/minimizer", "gates/gate_i/carve", "gates/gate_i/thresholds",
    "gates/gate_ii/status", "gates/gate_iii/tolerance", "gates/gate_e/tolerance",
    "per_run_checks/newton3/tolerance", "per_run_checks/newton3/min_load_nN_md",
    "per_run_checks/energy_ledger/tolerance_over_W", "statistics/seeds/pilot", "records/status_ceiling_md",
)
GATE_II_THRESHOLD_KEYS = ("residual_min_A", "residual_min_fraction", "dissipation_min", "hf_ratio_min")
# records/names/<key> of the recipe -> the schema.properties.PropertyName member it must equal (code_bindings).
RECORD_NAME_MEMBERS: dict[str, str] = {
    "max_load": "INDENTATION_MAX_LOAD_MD",
    "contact_pressure": "INDENTATION_CONTACT_PRESSURE_MD",
    "reduced_modulus": "INDENTATION_REDUCED_MODULUS_MD",
    "hardness": "INDENTATION_HARDNESS_MD_ESTIMATED_CONTACT",
    "pop_in_load": "POP_IN_LOAD_MD",
    "pop_in_depth": "POP_IN_DEPTH_MD",
}
DECK_RECIPE_CHECK = "indentation_recipe_protocol_matches_deck"

# The preparation an engine applies (preparation_summary keys, as deviation fields "preparation.<key>") and the
# recipe value each is compared with.
PREPARATION_FIELDS: dict[str, str] = {
    "zscale": "protocol/preparation/zscale",
    "langevin_start_temperature_K": "protocol/preparation/langevin/start_temperature_K",
    "langevin_temperature_K": "protocol/preparation/langevin/temperature_K",
    "langevin_tau_ps": "protocol/preparation/langevin/tau_ps",
    "langevin_fraction": "protocol/preparation/langevin/fraction_of_equilibration",
    "bath_only_fraction": "protocol/preparation/bath_only/fraction_of_equilibration",
}
JAX_RUNNER_PREPARATION = "jax_runner"  # preparation_summary source of physics.indentation.prepare_indentation_slab
# sha256 of recipe_v0's protocol section (section_sha256): the key of its known deviations below, so a later recipe
# with the same protocol section (recipe_v1, which only corrects the record names; recipe_v2, which adds the Gate I
# evaluation rule; the post-S1 recipe_v3 if it only freezes the Gate II thresholds) inherits them.
RECIPE_V0_PROTOCOL_SHA256 = "a5d42838189f6b92fa8a87da71b5333ea804f4bd5eb24faf63d19e1cf2107c6d"
# The departures a recipe's protocol declares acceptable (free text in protocol/preparation/known_deviations), as
# entries protocol_deviations can match: preparation source, field, value (fractions to one step) and, where the
# text limits it, the largest slab thickness.  A match is listed as sanctioned, with the text's pointer as its basis.
KNOWN_DEVIATIONS: dict[str, tuple[Mapping[str, Any], ...]] = {
    RECIPE_V0_PROTOCOL_SHA256: (
        # "in.berk_atomistic.lmp (write_atomistic_deck) splits equilibration_ps 2/3 Langevin + 1/3 bath"
        {"source": "in.berk_atomistic.lmp", "field": "preparation.langevin_fraction", "value": 2.0 / 3.0,
         "basis": "protocol/preparation/known_deviations/0"},
        {"source": "in.berk_atomistic.lmp", "field": "preparation.bath_only_fraction", "value": 1.0 / 3.0,
         "basis": "protocol/preparation/known_deviations/0"},
        # "the harness runner (prepare_indentation_slab) has no z pre-expansion: harmless below ~15 nm thickness"
        {"source": JAX_RUNNER_PREPARATION, "field": "preparation.zscale", "value": 1.0, "max_thickness_A": 150.0,
         "basis": "protocol/preparation/known_deviations/1"},
    ),
}

RecipeLike = Union[str, Path, "IndentationRecipe"]


class RecipeError(ValueError):
    """A recipe or canonical-values file that is malformed, inconsistent or not the frozen version it claims."""


# ============================================================================================ JSON helpers
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recipe_sha256(path: Union[str, Path]) -> str:
    """sha256 hex digest of a recipe file's bytes: the value recorded as EngineRunProvenance.recipe_sha256 (the same
    digest as ``engines.base.sha256_file``)."""
    return sha256_bytes(Path(path).read_bytes())


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise RecipeError(f"duplicate key {k!r} (one value per quantity / parameter)")
        out[k] = v
    return out


def _parse_json(data: bytes, what: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except RecipeError as exc:
        raise RecipeError(f"{what}: {exc}") from None
    except ValueError as exc:
        raise RecipeError(f"{what}: not valid JSON ({exc})") from None


def canonical_json(obj: Any) -> str:
    """Key-sorted compact JSON (the input of :func:`section_sha256`)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def section_sha256(doc: Mapping[str, Any], section: str) -> str:
    """sha256 of the canonical JSON of one top-level section (e.g. "protocol"): equal for two recipe versions whose
    section is identical, whatever the formatting or the other sections."""
    return sha256_bytes(canonical_json(doc[section]).encode("ascii"))


_MISSING = object()  # resolve_pointer default: raise KeyError
_ABSENT = object()  # a "not found" marker callers compare against


def resolve_pointer(doc: Any, pointer: str, default: Any = _MISSING) -> Any:
    """The value at ``pointer`` ("a/b/c"; integer parts index lists) of a JSON document."""
    cur = doc
    for part in pointer.split("/"):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and re.fullmatch(r"\d+", part) and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            if default is _MISSING:
                raise KeyError(pointer)
            return default
    return cur


def _same(a: Any, b: Any, rel: float = 1e-12) -> bool:
    """Equality of JSON-like values: numbers to ``rel``, sequences elementwise (lists == tuples), mappings by key."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-15)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same(a[k], b[k], rel) for k in a)
    return a == b


# ============================================================================================ canonical values
@dataclasses.dataclass(frozen=True)
class CanonicalValues:
    """data/indentation/canonical_values.json, validated: ``values[key]`` = {value, unit, quantity, source,
    verification, ...}."""

    path: Path
    sha256: str
    doc: Mapping[str, Any] = dataclasses.field(repr=False, compare=False, hash=False)

    @property
    def values(self) -> Mapping[str, Mapping[str, Any]]:
        return self.doc["values"]

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def value(self, key: str) -> Any:
        return copy.deepcopy(self.values[key]["value"])


def validate_canonical_values(doc: Any) -> list[str]:
    """Problems of a canonical-values document (empty list: valid)."""
    from fracture_md.units import Unit

    if not isinstance(doc, Mapping):
        return ["the canonical-values document must be a JSON object"]
    errs: list[str] = []
    if not str(doc.get("schema", "")).startswith(CANONICAL_SCHEMA_PREFIX):
        errs.append(f"schema must start with {CANONICAL_SCHEMA_PREFIX!r}")
    values = doc.get("values")
    if not isinstance(values, Mapping) or not values:
        return errs + ["'values' must be a non-empty object"]
    units = {u.value for u in Unit} | set(doc.get("extra_units", []))
    for key, e in values.items():
        if not re.fullmatch(r"[a-z0-9_]+(\.[A-Za-z0-9_]+)+", key):
            errs.append(f"{key}: keys are dotted lower-case names (group.quantity)")
        if not isinstance(e, Mapping):
            errs.append(f"{key}: entry must be an object")
            continue
        for f in ("value", "unit", "quantity", "source", "verification"):
            if f not in e:
                errs.append(f"{key}: missing {f!r}")
        if e.get("verification") not in VERIFICATION_LABELS:
            errs.append(f"{key}: verification {e.get('verification')!r} not in {VERIFICATION_LABELS}")
        if e.get("unit") is not None and e.get("unit") not in units:
            errs.append(f"{key}: unit {e.get('unit')!r} is neither a fracture_md.units.Unit nor an extra unit")
        if isinstance(e.get("value"), (int, float)) and not isinstance(e.get("value"), bool) and e.get("unit") is None:
            errs.append(f"{key}: a numeric value needs a unit ('1' for dimensionless)")
        if not str(e.get("source", "")).strip():
            errs.append(f"{key}: empty source")
        for alt in e.get("alternatives", []):
            if not isinstance(alt, Mapping) or "value" not in alt or "label" not in alt:
                errs.append(f"{key}: each alternative needs a value and a label")
    return errs


@functools.lru_cache(maxsize=8)
def _canonical_cached(sha: str, data: bytes, path: str) -> CanonicalValues:
    doc = _parse_json(data, path)
    errs = validate_canonical_values(doc)
    if errs:
        raise RecipeError(f"{path}: invalid canonical values:\n  " + "\n  ".join(errs))
    return CanonicalValues(path=Path(path), sha256=sha, doc=doc)


def load_canonical_values(path: Optional[Union[str, Path]] = None) -> CanonicalValues:
    """Load and validate the canonical values (default ``data/indentation/canonical_values.json``)."""
    p = Path(path) if path is not None else CANONICAL_VALUES_PATH
    data = p.read_bytes()
    return _canonical_cached(sha256_bytes(data), data, str(p))


# ============================================================================================ recipe
@dataclasses.dataclass(frozen=True)
class IndentationRecipe:
    """A validated indentation recipe: its id, version, file, sha256 (of the file bytes) and parsed content."""

    recipe_id: str
    version: int
    path: Path
    sha256: str
    _doc: Mapping[str, Any] = dataclasses.field(repr=False, compare=False, hash=False)
    canonical_sha256: str = ""

    @property
    def doc(self) -> dict[str, Any]:
        """A copy of the parsed recipe (the cached original is never handed out)."""
        return copy.deepcopy(dict(self._doc))

    def get(self, pointer: str, default: Any = _MISSING) -> Any:
        """A copy of the value at ``pointer`` ("protocol/tip/tip_radius_A"); KeyError (or ``default``) if absent."""
        v = resolve_pointer(self._doc, pointer, _ABSENT)
        if v is _ABSENT:
            if default is _MISSING:
                raise KeyError(pointer)
            return default
        return copy.deepcopy(v)

    @property
    def provisional(self) -> tuple[str, ...]:
        """Pointers of the values that are placeholders until the S1 scoping run (the Gate II thresholds in v0)."""
        return tuple(self._doc.get(PROVISIONAL, ()))

    @property
    def frozen(self) -> bool:
        """True when this is a pinned recipe (FROZEN_RECIPE_SHA256) with the pinned bytes."""
        return FROZEN_RECIPE_SHA256.get(self.recipe_id) == self.sha256

    def section_sha256(self, section: str) -> str:
        return section_sha256(self._doc, section)

    def summary(self) -> dict[str, Any]:
        """JSON-ready identity of the recipe (raw outputs, run manifests)."""
        return {"recipe_id": self.recipe_id, "version": self.version, "sha256": self.sha256, "file": self.path.name,
                "frozen": self.frozen, "protocol_sha256": self.section_sha256("protocol"),
                PROVISIONAL: list(self.provisional), "canonical_values_sha256": self.canonical_sha256}


def _check_canonical_refs(doc: Mapping[str, Any], canonical: CanonicalValues) -> list[str]:
    errs: list[str] = []
    refs = doc.get("canonical_refs", {})
    if not isinstance(refs, Mapping):
        return ["canonical_refs must be an object {pointer: canonical key}"]
    for ptr, key in refs.items():
        try:
            val = resolve_pointer(doc, ptr)
        except KeyError:
            errs.append(f"canonical_refs: {ptr} does not resolve in the recipe")
            continue
        if key not in canonical:
            errs.append(f"canonical_refs: {ptr} -> {key!r} is not a canonical value")
            continue
        if not _same(val, canonical.value(key), rel=1e-9):
            errs.append(f"canonical_refs: {ptr} = {val!r} disagrees with {key} = {canonical.value(key)!r}")
    return errs


def runs_needed(cv: float, half_width: float = 0.05, confidence: float = 0.95) -> int:
    """Smallest n with t(1 - (1 - confidence)/2, n - 1) cv / sqrt(n) <= half_width (the recipe's runs-needed rule)."""
    from scipy.stats import t as student_t

    q = 1.0 - (1.0 - confidence) / 2.0
    n = 2
    while student_t.ppf(q, n - 1) * cv / math.sqrt(n) > half_width:
        n += 1
    return n


def _semantic_checks(doc: Mapping[str, Any]) -> list[str]:
    errs: list[str] = []

    def get(p, d=None):
        return resolve_pointer(doc, p, d)

    # provisional values: listed pointers resolve, and their sections say so (and only those sections)
    prov = doc.get(PROVISIONAL, [])
    if not isinstance(prov, list):
        errs.append(f"{PROVISIONAL} must be a list of pointers")
        prov = []
    for ptr in prov:
        if get(ptr, _ABSENT) is _ABSENT:
            errs.append(f"{PROVISIONAL}: {ptr} does not resolve")
        elif get(ptr.rsplit("/", 1)[0] + "/status") != PROVISIONAL:
            errs.append(f"{PROVISIONAL}: the section of {ptr} must have status {PROVISIONAL!r}")
    for name, sec in (doc.get("gates") or {}).items():
        st = sec.get("status") if isinstance(sec, Mapping) else None
        if isinstance(sec, Mapping) and st not in SECTION_STATUSES:
            errs.append(f"gates/{name}/status {st!r} not in {SECTION_STATUSES}")
        if st == PROVISIONAL and not any(p.startswith(f"gates/{name}/") for p in prov):
            errs.append(f"gates/{name} is {PROVISIONAL} but none of its values is listed in {PROVISIONAL}")
    g2 = get("gates/gate_ii", {})
    for k in GATE_II_THRESHOLD_KEYS:
        v = g2.get(k) if isinstance(g2, Mapping) else None
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not v > 0:
            errs.append(f"gates/gate_ii/{k} must be a positive number")
    for k in ("residual_min_fraction", "dissipation_min", "hf_ratio_min"):
        v = g2.get(k) if isinstance(g2, Mapping) else None
        if isinstance(v, (int, float)) and not v < 1:
            errs.append(f"gates/gate_ii/{k} must be < 1")
    # estimators
    est = get("analysis/contact_area/estimators", {})
    prim, chk = get("analysis/contact_area/primary"), get("analysis/contact_area/check")
    ev = get("analysis/contact_area/plateau_evidence", [])
    if not isinstance(est, Mapping) or not est:
        errs.append("analysis/contact_area/estimators must name the estimators")
    else:
        if prim not in est or chk not in est or not set(ev) <= set(est) or prim not in ev:
            errs.append("analysis/contact_area: primary and check must be estimators and the primary plateau evidence")
        flagged = sorted(k for k, v in est.items() if isinstance(v, Mapping) and v.get("plateau_evidence"))
        if flagged != sorted(ev):
            errs.append("analysis/contact_area: plateau_evidence must list exactly the estimators flagged as evidence")
        if sorted(ev) != sorted(get("gates/gate_iii/estimators", [])):
            errs.append("gates/gate_iii/estimators must equal analysis/contact_area/plateau_evidence")
    f = get("analysis/contact_area/contact_force_fraction")
    if not isinstance(f, (int, float)) or not 0.0 <= f <= 1.0:
        errs.append("analysis/contact_area/contact_force_fraction must lie in [0, 1]")
    # cell tiers: production >= pilot >= scoping
    tiers = get("protocol/cell/tiers", {})
    try:
        order = [tiers[n] for n in ("production", "pilot", "scoping")]
        if any(not (len(t) == 2 and all(x > 0 for x in t)) for t in order) or \
                any(a[i] < b[i] for a, b in zip(order, order[1:]) for i in range(2)):
            errs.append("protocol/cell/tiers: (L/a, t_el/a) > 0 and production >= pilot >= scoping")
    except (KeyError, TypeError):
        errs.append("protocol/cell/tiers must give production, pilot and scoping (L/a, t_el/a)")
    # wall stiffness, time step, Newton 3, ledger
    K, Kmin = get("protocol/wall_stiffness/K_eV_per_A3"), get("protocol/wall_stiffness/K_min_eV_per_A3")
    if not (isinstance(K, (int, float)) and isinstance(Kmin, (int, float)) and K >= Kmin > 0):
        errs.append("protocol/wall_stiffness: K_eV_per_A3 >= K_min_eV_per_A3 > 0")
    for p, lo, hi in (("protocol/time_step/dt_fs", 0.0, 2.0), ("per_run_checks/newton3/tolerance", 0.0, 1.0),
                      ("per_run_checks/energy_ledger/tolerance_over_W", 0.0, 1.0),
                      ("per_run_checks/newton3/min_load_nN_md", 0.0, math.inf),
                      ("analysis/evaluation_depth/tolerance_A", -1e-300, 0.5)):
        v = get(p)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo < v <= hi:
            errs.append(f"{p} must lie in ({lo:g}, {hi:g}]")
    w, b, r = (get(f"analysis/load/{k}") for k in ("running_mean_window_ps", "tau_ac_default_ps", "min_window_over_tau_ac"))
    if all(isinstance(x, (int, float)) for x in (w, b, r)) and not w >= r * b:
        errs.append("analysis/load: the load window must span min_window_over_tau_ac default autocorrelation times")
    # seeds: integers, pre-registered, distinct; one lateral offset per pilot seed
    seeds = get("statistics/seeds", {})
    allseeds = list(seeds.get("s1", [])) + list(seeds.get("pilot", [])) if isinstance(seeds, Mapping) else []
    if not allseeds or any(not isinstance(s, int) or isinstance(s, bool) for s in allseeds) or len(set(allseeds)) != len(allseeds):
        errs.append("statistics/seeds: s1 and pilot seeds must be distinct integers")
    elif len(seeds.get("pilot_lateral_offsets_in_a_over_3", [])) != len(seeds.get("pilot", [])):
        errs.append("statistics/seeds: one lateral offset per pilot seed")
    rn = get("statistics/runs_needed/table", {})
    hw = get("statistics/ci95_half_width_target", 0.05)
    for cv, n in (rn.items() if isinstance(rn, Mapping) else []):
        if runs_needed(float(cv), hw) != n:
            errs.append(f"statistics/runs_needed/table: CV {cv} needs {runs_needed(float(cv), hw)} runs, not {n}")
    # decision G01: the status ceiling
    if get("records/status_ceiling_md") != "literature_model_agreement" or \
            "validated_against_experiment" not in get("scope/status_ceiling/never", []):
        errs.append("records/status_ceiling_md must be literature_model_agreement and scope/status_ceiling/never must "
                    "include validated_against_experiment (decision G01)")
    return errs


def validate_recipe(doc: Any, canonical: Optional[CanonicalValues] = None) -> list[str]:
    """Problems of a recipe document (empty list: valid): schema, the required sections and items, provisional
    values, estimator and gate consistency, seeds, the runs-needed table and the canonical references."""
    if not isinstance(doc, Mapping):
        return ["the recipe must be a JSON object"]
    errs: list[str] = []
    if doc.get("schema") != RECIPE_SCHEMA:
        errs.append(f"schema must be {RECIPE_SCHEMA!r}")
    rid, ver = doc.get("recipe_id"), doc.get("version")
    if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", rid):
        errs.append("recipe_id must be a non-empty identifier")
    if not isinstance(ver, int) or isinstance(ver, bool) or ver < 0:
        errs.append("version must be a non-negative integer")
    missing = [s for s in REQUIRED_SECTIONS if not isinstance(doc.get(s), Mapping)]
    if missing:
        return errs + [f"missing sections {missing}"]
    errs += [f"required item {p} is missing" for p in REQUIRED_POINTERS if resolve_pointer(doc, p, _ABSENT) is _ABSENT]
    if errs:
        return errs
    errs += _semantic_checks(doc)
    errs += _check_canonical_refs(doc, canonical if canonical is not None else load_canonical_values())
    return errs


@functools.lru_cache(maxsize=16)
def _recipe_cached(sha: str, data: bytes, path: str, canonical: CanonicalValues, verify_pin: bool) -> IndentationRecipe:
    doc = _parse_json(data, path)
    errs = validate_recipe(doc, canonical)
    if errs:
        raise RecipeError(f"{path}: invalid recipe:\n  " + "\n  ".join(errs))
    rid = str(doc["recipe_id"])
    pin = FROZEN_RECIPE_SHA256.get(rid)
    if verify_pin and pin is not None and pin != sha:
        raise RecipeError(f"{path}: {rid} is a frozen recipe (sha256 {pin[:12]}...) but this file hashes to "
                          f"{sha[:12]}...; a frozen recipe is never edited: write recipe_v<N+1>.json with a new id and pin")
    return IndentationRecipe(recipe_id=rid, version=int(doc["version"]), path=Path(path), sha256=sha, _doc=doc,
                             canonical_sha256=canonical.sha256)


def recipe_path(recipe: Union[str, Path]) -> Path:
    """The file of a recipe given by name (e.g. "recipe_v2": data/indentation/recipe_v2.json) or by path."""
    s = str(recipe)
    if isinstance(recipe, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", s) and not s.endswith(".json"):
        return RECIPE_DIR / f"{s}.json"
    return Path(recipe)


def load_recipe(recipe: RecipeLike = DEFAULT_RECIPE, *, verify_pin: bool = True) -> IndentationRecipe:
    """Load and validate a recipe: a name ("recipe_v2", the default), a path, or an :class:`IndentationRecipe`
    (returned as is).  Raises :class:`RecipeError` when the file is invalid, when a canonical reference disagrees with
    data/indentation/canonical_values.json, or (``verify_pin``) when a file claiming a frozen recipe id does not have
    the pinned bytes."""
    if isinstance(recipe, IndentationRecipe):
        return recipe
    p = recipe_path(recipe)
    data = p.read_bytes()
    return _recipe_cached(sha256_bytes(data), data, str(p), load_canonical_values(), bool(verify_pin))


def resolve_recipe(recipe: Optional[RecipeLike]) -> Optional[IndentationRecipe]:
    """None -> None (no recipe recorded: an explicit opt-out); anything else -> :func:`load_recipe`."""
    return None if recipe is None else load_recipe(recipe)


# ============================================================================================ preparation
def preparation_summary(*, source: str, zscale: float, langevin_steps: int, bath_steps: int,
                        start_temperature_K: float, temperature_K: float, langevin_tau_ps: float,
                        thickness_A: Optional[float] = None) -> dict[str, Any]:
    """The preparation an engine applies, in the recipe's terms (the ``preparation`` argument of
    :func:`protocol_deviations` and :func:`manifest_entry`).

    ``source``: the LAMMPS deck file name or :data:`JAX_RUNNER_PREPARATION`.  ``zscale``: the z pre-expansion factor of
    the mobile atoms after the minimisation (1 = none).  Langevin stage of every mobile atom: ``langevin_steps`` steps,
    velocities drawn at ``start_temperature_K``, target ``temperature_K``, time constant ``langevin_tau_ps``; then the
    boundary thermostat alone for ``bath_steps`` steps.  The two stages' shares of the equilibration are compared with
    the recipe to one step (step-count rounding).  ``thickness_A``: the slab thickness, when known (a known deviation
    may be limited to thin slabs)."""
    nl, nb = int(langevin_steps), int(bath_steps)
    if nl < 0 or nb < 0:
        raise ValueError("langevin_steps and bath_steps must be >= 0")
    n = nl + nb
    return {"source": str(source), "zscale": float(zscale), "langevin_start_temperature_K": float(start_temperature_K),
            "langevin_temperature_K": float(temperature_K), "langevin_tau_ps": float(langevin_tau_ps),
            "langevin_steps": nl, "bath_steps": nb, "langevin_fraction": nl / n if n else 0.0,
            "bath_only_fraction": nb / n if n else 0.0,
            "thickness_A": None if thickness_A is None else float(thickness_A)}


_LMP_VARIABLE = re.compile(r"^[ \t]*variable[ \t]+(\w+)[ \t]+index[ \t]+(\S+)", re.M)
_LMP_LANGEVIN = re.compile(r"fix[ \t]+lang[ \t]+mobile[ \t]+langevin[ \t]+\$\{(\w+)\}[ \t]+\$\{(\w+)\}[ \t]+\$\{(\w+)\}")
_LMP_START_T = re.compile(r"velocity[ \t]+mobile[ \t]+create[ \t]+\$\(([0-9.eE+-]+)\*v_(\w+)\)")
_LMP_RUN = re.compile(r"\brun[ \t]+(?:\$\{(\w+)\}|\$\((\d+)\*ceil\(v_(\w+)/(\d+)\)\))")


def lammps_deck_preparation(deck: Union[str, Path], variables: Mapping[str, Any], *,
                            thickness_A: Optional[float] = None) -> Optional[dict[str, Any]]:
    """:func:`preparation_summary` of a LAMMPS indentation deck (in.berk_sdf.lmp, in.berk_atomistic.lmp) run with the
    ``-var`` values ``variables`` (deck defaults for the others), read from the deck text: ``zscale``; the
    ``velocity mobile create $(f*v_T)`` start temperature; the ``fix lang mobile langevin ${T} ${T} ${tau}`` target and
    time constant; the steps of the first ``run`` after that fix (Langevin) and of the first after ``unfix lang`` (bath
    only), of the form ``${n}`` or ``$(k*ceil(v_n/m))``.  None for a quasi-static deck (``mode`` qs).  Raises
    :class:`RecipeError` when the deck has no such preparation (a changed deck must be read anew here)."""
    path = Path(deck)
    text = "\n".join(ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#"))
    defaults = dict(_LMP_VARIABLE.findall(text))

    def val(name: str) -> float:
        s = variables.get(name, defaults.get(name))
        if s is None:
            raise RecipeError(f"{path.name}: no value for the deck variable {name!r}")
        return float(s)

    if str(variables.get("mode", defaults.get("mode", "md"))) == "qs":
        return None
    lang, start = _LMP_LANGEVIN.search(text), _LMP_START_T.search(text)
    unfix = text.find("unfix lang", lang.end()) if lang else -1
    r1 = _LMP_RUN.search(text, lang.end()) if lang else None
    r2 = _LMP_RUN.search(text, unfix) if unfix >= 0 else None
    if not (lang and start and r1 and r2 and r1.start() < unfix) or "zscale" not in defaults:
        raise RecipeError(f"{path.name}: cannot read the preparation (zscale, velocity create, fix lang, the Langevin "
                          "and bath-only runs) from the deck text")

    def steps(m: Any) -> int:
        if m.group(1):
            return int(val(m.group(1)))
        return int(m.group(2)) * math.ceil(val(m.group(3)) / int(m.group(4)))

    return preparation_summary(source=path.name, zscale=val("zscale"), langevin_steps=steps(r1), bath_steps=steps(r2),
                               start_temperature_K=float(start.group(1)) * val(start.group(2)),
                               temperature_K=val(lang.group(2)), langevin_tau_ps=val(lang.group(3)),
                               thickness_A=thickness_A)


# ============================================================================================ code consistency
def _protocol_default(model: Any, field: str) -> Any:
    info = model.model_fields[field]
    return info.default_factory() if info.default_factory is not None else info.default


@functools.lru_cache(maxsize=1)
def _gate_e_module() -> Any:
    import sys

    name = "_fracture_md_gate_e_script"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, GATE_E_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses in the script look their module up while it executes
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def code_bindings() -> list[tuple[str, str, Callable[[], Any]]]:
    """(recipe pointer, code location, getter) for every frozen value that the code implements."""
    import dataclasses as dc
    import inspect
    from types import SimpleNamespace

    from fracture_md.engines import lammps_indent as LI
    from fracture_md.io.lammps_indent import LammpsLogInfo
    from fracture_md.physics import indent_analysis as IA
    from fracture_md.physics import indentation as IND
    from fracture_md.physics.indent_md import IndentMDConfig
    from fracture_md.potentials.vashishta import SIC_VASHISHTA_2007_FILE_SHA256
    from fracture_md.schema import properties as SP
    from fracture_md.schema import protocol as SPR
    from fracture_md.structures import indentation as SI
    from fracture_md.units import ISO14577_HV_PER_MPA

    P, I = SPR.IndentationProtocol, SPR.IndenterSpec

    def dflt(model, f):
        return lambda: _protocol_default(model, f)

    def sig(fn, name):
        return lambda: inspect.signature(fn).parameters[name].default

    def cfg(name):
        return lambda: getattr(IA.AnalysisConfig(), name)

    def gi(name):
        return lambda: dc.asdict(IA.GateIThresholds())[name]

    def g2(name):
        return lambda: getattr(IA.GateIIThresholds(), name)

    def ge(name):
        return lambda: getattr(_gate_e_module(), name)

    def deck_prep(kind, key):
        """The preparation a generated MD deck applies at the recipe's defaults (deck_variables -> deck text): what
        the generator writes, not a constant next to it."""
        def get():
            a = _gate_e_module().VASHISHTA_A_300K
            if kind == "sdf":
                v = LI.deck_variables(LI.tier_protocol(10.0, "pilot", lattice_parameter_A=a),
                                      SimpleNamespace(center_xy=(0.0, 0.0), lattice_constant_A=a))
                return lammps_deck_preparation(LI.EXAMPLES_DIR / LI.SDF_DECK, v)[key]
            ind = I(tip_model="atomistic", tip_radius_A=0.0, atomistic_tip="rigid diamond, WCA")
            v = LI.atomistic_deck_variables(LI.tier_protocol(10.0, "scoping", lattice_parameter_A=a, indenter=ind))
            return lammps_deck_preparation(LI.EXAMPLES_DIR / LI.ATOMISTIC_DECK, v)[key]
        return get

    def deck_default(deck, var):
        def get():
            m = re.search(rf"^[ \t]*variable[ \t]+{var}[ \t]+index[ \t]+(\S+)", (LI.EXAMPLES_DIR / deck).read_text(), re.M)
            if m is None:
                raise RecipeError(f"{deck} defines no variable {var}")
            return float(m.group(1))
        return get

    b: list[tuple[str, str, Callable[[], Any]]] = [
        # potential
        ("protocol/potential/file_sha256", "potentials.vashishta.SIC_VASHISHTA_2007_FILE_SHA256",
         lambda: SIC_VASHISHTA_2007_FILE_SHA256),
        ("protocol/potential/cutoff_A", "scripts/gate_e.py VASHISHTA_CUTOFF_A", ge("VASHISHTA_CUTOFF_A")),
        ("protocol/potential/name", "engines.lammps_indent.PAIR_STYLES", lambda: next(iter(LI.PAIR_STYLES))),
        # tip and wall
        ("protocol/tip/model", "schema.protocol.IndenterSpec.tip_model", dflt(I, "tip_model")),
        ("protocol/tip/shape", "schema.protocol.IndenterSpec.shape", dflt(I, "shape")),
        ("protocol/tip/face_angle_deg", "schema.protocol.BERKOVICH_FACE_ANGLE_DEG", lambda: SPR.BERKOVICH_FACE_ANGLE_DEG),
        ("protocol/tip/face_angle_deg", "schema.protocol.IndenterSpec.face_angle_deg", dflt(I, "face_angle_deg")),
        ("protocol/tip/face_angle_deg", "physics.indent_analysis.BERKOVICH_FACE_ANGLE_DEG",
         lambda: IA.BERKOVICH_FACE_ANGLE_DEG),
        ("protocol/tip/C0", "physics.indent_analysis.BERKOVICH_C0 (to 5 significant digits)",
         lambda: float(f"{IA.BERKOVICH_C0:.5g}")),
        ("protocol/tip/tip_radius_A", "schema.protocol.IndenterSpec.tip_radius_A", dflt(I, "tip_radius_A")),
        ("protocol/tip/height_cap_default_A", "schema.protocol.IndenterSpec.height_cap_A", dflt(I, "height_cap_A")),
        ("protocol/tip/atomistic_tip/eps_eV", "engines.lammps_indent.ATOMISTIC_WCA_EPS_EV", lambda: LI.ATOMISTIC_WCA_EPS_EV),
        ("protocol/tip/atomistic_tip/sigma_A", "engines.lammps_indent.ATOMISTIC_WCA_SIGMA_A",
         lambda: LI.ATOMISTIC_WCA_SIGMA_A),
        ("protocol/wall_stiffness/K_eV_per_A3", "schema.protocol.IndenterSpec.stiffness_eV_per_A3",
         dflt(I, "stiffness_eV_per_A3")),
        ("protocol/wall_stiffness/K_min_eV_per_A3", "schema.protocol.MIN_WALL_STIFFNESS_EV_PER_A3",
         lambda: SPR.MIN_WALL_STIFFNESS_EV_PER_A3),
        ("protocol/orientation/azimuth_deg", "schema.protocol.IndenterSpec.azimuth_deg", dflt(I, "azimuth_deg")),
        ("protocol/orientation/surface_miller", "schema.protocol.IndentationProtocol.plane_miller", dflt(P, "plane_miller")),
        ("protocol/surface/termination", "schema.protocol.IndentationProtocol.top_termination",
         dflt(P, "top_termination")),
        ("protocol/surface/lattice_constant_A", "scripts/gate_e.py VASHISHTA_A_300K", ge("VASHISHTA_A_300K")),
        # preparation, cell, boundary layers
        ("protocol/preparation/relax_fmax_eV_per_A", "schema.protocol.IndentationProtocol.relax.fmax_eV_per_A",
         lambda: _protocol_default(P, "relax").fmax_eV_per_A),
        ("protocol/preparation/equilibration_ps", "schema.protocol.IndentationProtocol.equilibration_ps",
         dflt(P, "equilibration_ps")),
        # the preparation the engines apply (critic T2): the z pre-expansion and the Langevin / bath-only split.  The
        # atomistic deck's 2/3 + 1/3 split and the JAX runner's missing pre-expansion are the recipe's own
        # known_deviations (KNOWN_DEVIATIONS), reported by protocol_deviations rather than bound here.
        ("protocol/preparation/zscale", "engines.lammps_indent.deck_variables -> in.berk_sdf.lmp zscale",
         deck_prep("sdf", "zscale")),
        ("protocol/preparation/zscale", "engines.lammps_indent.atomistic_deck_variables -> in.berk_atomistic.lmp zscale",
         deck_prep("atomistic", "zscale")),
        ("protocol/preparation/langevin/start_temperature_K",
         "in.berk_sdf.lmp velocity mobile create (with engines.lammps_indent.deck_variables T)",
         deck_prep("sdf", "langevin_start_temperature_K")),
        ("protocol/preparation/langevin/start_temperature_K",
         "in.berk_atomistic.lmp velocity mobile create (with engines.lammps_indent.atomistic_deck_variables T)",
         deck_prep("atomistic", "langevin_start_temperature_K")),
        ("protocol/preparation/langevin/start_temperature_K",
         "physics.indentation.PREP_START_TEMPERATURE_FACTOR x IndentationProtocol.temperature_K",
         lambda: IND.PREP_START_TEMPERATURE_FACTOR * _protocol_default(P, "temperature_K")),
        ("protocol/preparation/langevin/temperature_K", "in.berk_sdf.lmp fix lang target (deck_variables)",
         deck_prep("sdf", "langevin_temperature_K")),
        ("protocol/preparation/langevin/temperature_K", "in.berk_atomistic.lmp fix lang target (atomistic_deck_variables)",
         deck_prep("atomistic", "langevin_temperature_K")),
        ("protocol/preparation/langevin/tau_ps", "in.berk_sdf.lmp fix lang time constant (deck_variables)",
         deck_prep("sdf", "langevin_tau_ps")),
        ("protocol/preparation/langevin/tau_ps", "in.berk_atomistic.lmp fix lang time constant (atomistic_deck_variables)",
         deck_prep("atomistic", "langevin_tau_ps")),
        ("protocol/preparation/langevin/tau_ps",
         "schema.protocol.IndentationProtocol.thermostat_tau_ps (the JAX runner's Langevin stage)",
         dflt(P, "thermostat_tau_ps")),
        ("protocol/preparation/langevin/fraction_of_equilibration",
         "engines.lammps_indent.deck_variables nlang -> in.berk_sdf.lmp runs", deck_prep("sdf", "langevin_fraction")),
        ("protocol/preparation/langevin/fraction_of_equilibration", "physics.indentation.PREP_LANGEVIN_FRACTION",
         lambda: IND.PREP_LANGEVIN_FRACTION),
        ("protocol/preparation/bath_only/fraction_of_equilibration",
         "engines.lammps_indent.deck_variables nbath -> in.berk_sdf.lmp runs", deck_prep("sdf", "bath_only_fraction")),
        ("protocol/preparation/bath_only/fraction_of_equilibration", "1 - physics.indentation.PREP_LANGEVIN_FRACTION",
         lambda: 1.0 - IND.PREP_LANGEVIN_FRACTION),
        ("protocol/cell/tiers", "schema.protocol.CELL_TIERS", lambda: {n: [lr, tr] for n, lr, tr in SPR.CELL_TIERS}),
        ("protocol/cell/a_over_h", "schema.protocol.IndenterSpec().equivalent_contact_radius_A(1) (5 significant digits)",
         lambda: float(f"{I().equivalent_contact_radius_A(1.0):.5g}")),
        ("protocol/boundary_layers/fixed_thickness_A", "schema.protocol.IndentationProtocol.fixed_thickness_A",
         dflt(P, "fixed_thickness_A")),
        ("protocol/boundary_layers/bath_thickness_A", "schema.protocol.IndentationProtocol.thermostat_thickness_A",
         dflt(P, "thermostat_thickness_A")),
        ("protocol/boundary_layers/bath", "schema.protocol.IndentationProtocol.thermostat", dflt(P, "thermostat")),
        ("protocol/boundary_layers/bath", "physics.indent_md.IndentMDConfig.thermostat",
         lambda: IndentMDConfig.__dataclass_fields__["thermostat"].default),
        ("protocol/boundary_layers/bath_tau_ps", "schema.protocol.IndentationProtocol.thermostat_tau_ps",
         dflt(P, "thermostat_tau_ps")),
        ("protocol/temperature_K", "schema.protocol.IndentationProtocol.temperature_K", dflt(P, "temperature_K")),
        ("protocol/loading/speed_m_per_s", "schema.protocol.IndentationProtocol.speed_m_per_s", dflt(P, "speed_m_per_s")),
        ("protocol/loading/initial_gap_A", "schema.protocol.IndentationProtocol.initial_gap_A", dflt(P, "initial_gap_A")),
        ("protocol/loading/park_gap_A", "structures.indentation.PARK_GAP_A", lambda: SI.PARK_GAP_A),
        ("protocol/time_step/dt_fs", "schema.protocol.IndentationProtocol.timestep_fs", dflt(P, "timestep_fs")),
        ("protocol/time_step/dt_fs", "physics.indent_md.IndentMDConfig.dt_fs",
         lambda: IndentMDConfig.__dataclass_fields__["dt_fs"].default),
        ("protocol/hold_unload/hold_ps", "schema.protocol.IndentationProtocol.hold_ps", dflt(P, "hold_ps")),
        ("protocol/hold_unload/relax_after_unload_ps", "schema.protocol.IndentationProtocol.relax_after_unload_ps",
         dflt(P, "relax_after_unload_ps")),
        ("protocol/hold_unload/unload_clearance_A", "engines.lammps_indent.DEFAULT_UNLOAD_CLEARANCE_A",
         lambda: LI.DEFAULT_UNLOAD_CLEARANCE_A),
        ("protocol/hold_unload/park_after_unload_A", "in.berk_sdf.lmp variable park", deck_default(LI.SDF_DECK, "park")),
        ("protocol/hold_unload/park_after_unload_A", "in.berk_atomistic.lmp variable park",
         deck_default(LI.ATOMISTIC_DECK, "park")),
        ("protocol/output/log_every_steps", "schema.protocol.IndentationProtocol.log_every_steps",
         dflt(P, "log_every_steps")),
        ("protocol/output/frame_every_ps", "schema.protocol.IndentationProtocol.frame_every_ps", dflt(P, "frame_every_ps")),
        ("protocol/output/slice_half_width_A", "schema.protocol.IndentationProtocol.slice_half_width_A",
         dflt(P, "slice_half_width_A")),
        ("protocol/output/full_dump_every_A", "schema.protocol.IndentationProtocol.full_dump_every_A",
         dflt(P, "full_dump_every_A")),
        # analysis
        ("analysis/load/running_mean_window_ps", "physics.indent_analysis.DEFAULT_WINDOW_PS", lambda: IA.DEFAULT_WINDOW_PS),
        ("analysis/load/running_mean_window_ps", "physics.indent_analysis.AnalysisConfig.window_ps", cfg("window_ps")),
        ("analysis/load/block_average_ps", "schema.protocol.IndentationProtocol.block_average_ps",
         dflt(P, "block_average_ps")),
        ("analysis/load/min_window_over_tau_ac", "physics.indent_analysis.MIN_WINDOW_OVER_TAU",
         lambda: IA.MIN_WINDOW_OVER_TAU),
        ("analysis/load/tau_ac_default_ps", "physics.indent_analysis.DEFAULT_TAU_AC_PS", lambda: IA.DEFAULT_TAU_AC_PS),
        ("analysis/load/precontact_gap_A", "physics.indent_analysis.AnalysisConfig.precontact_gap_A",
         cfg("precontact_gap_A")),
        ("analysis/depth/h0_rule", "physics.indent_analysis.AnalysisConfig.h0_rule", cfg("h0_rule")),
        ("analysis/depth/first_contact_n_sigma", "physics.indent_analysis.first_contact_depth(n_sigma)",
         sig(IA.first_contact_depth, "n_sigma")),
        ("analysis/depth/first_contact_floor_nN", "physics.indent_analysis.AnalysisConfig.first_contact_floor_nN",
         cfg("first_contact_floor_nN")),
        ("analysis/depth/sneddon_fit_load_range_nN", "physics.indent_analysis.AnalysisConfig.fit_load_range_nN",
         cfg("fit_load_range_nN")),
        ("analysis/depth/hertz_fit_h_max_A", "physics.indent_analysis.AnalysisConfig.hertz_fit_h_max_A",
         cfg("hertz_fit_h_max_A")),
        ("analysis/contact_area/contact_rule", "physics.indent_analysis.AnalysisConfig.contact_rule", cfg("contact_rule")),
        ("analysis/contact_area/contact_force_fraction", "physics.indent_analysis.CONTACT_FORCE_FRACTION",
         lambda: IA.CONTACT_FORCE_FRACTION),
        ("analysis/contact_area/contact_force_fraction", "physics.indent_analysis.AnalysisConfig.contact_force_fraction",
         cfg("contact_force_fraction")),
        ("analysis/contact_area/closing_radius_A", "physics.indent_analysis.CLOSING_RADIUS_A", lambda: IA.CLOSING_RADIUS_A),
        ("analysis/contact_area/primary", "physics.indent_analysis.PRIMARY_ESTIMATOR", lambda: IA.PRIMARY_ESTIMATOR),
        ("analysis/contact_area/check", "physics.indent_analysis.CHECK_ESTIMATOR", lambda: IA.CHECK_ESTIMATOR),
        ("analysis/contact_area/plateau_evidence", "physics.indent_analysis.PLATEAU_EVIDENCE_ESTIMATORS",
         lambda: list(IA.PLATEAU_EVIDENCE_ESTIMATORS)),
        ("analysis/contact_area/estimators", "physics.indent_analysis.ESTIMATOR_NAMES (code -> schema name)",
         lambda: dict(IA.ESTIMATOR_NAMES)),
        ("analysis/contact_area/estimators/E3/name", "schema.protocol.IndentationProtocol.primary_area_estimator",
         dflt(P, "primary_area_estimator")),
        ("analysis/evaluation_depth/tolerance_A", "physics.indent_analysis.EVALUATION_DEPTH_TOL_A",
         lambda: IA.EVALUATION_DEPTH_TOL_A),
        ("analysis/unload/zero_load_threshold_nN", "physics.indent_analysis.AnalysisConfig.unload_threshold_nN",
         cfg("unload_threshold_nN")),
        ("analysis/oliver_pharr/epsilon", "physics.indent_analysis.oliver_pharr(epsilon)", sig(IA.oliver_pharr, "epsilon")),
        ("analysis/oliver_pharr/beta", "physics.indent_analysis.oliver_pharr(beta)", sig(IA.oliver_pharr, "beta")),
        ("analysis/oliver_pharr/fit_range", "physics.indent_analysis.AnalysisConfig.op_fit_range", cfg("op_fit_range")),
        # gates
        ("gates/gate_i/minimizer/method", "physics.indent_analysis.GATE_I_MINIMIZER", lambda: IA.GATE_I_MINIMIZER),
        ("gates/gate_i/minimizer/fmax_eV_per_A", "physics.indent_analysis.GATE_I_FMAX_EV_PER_A",
         lambda: IA.GATE_I_FMAX_EV_PER_A),
        ("gates/gate_i/minimizer/newton_polish", "physics.indent_analysis.frozen_gate_i_recipe()",
         lambda: IA.frozen_gate_i_recipe()["minimizer"]["newton_polish"]),
        ("gates/gate_i/minimizer/backends", "physics.indent_analysis.frozen_gate_i_recipe()",
         lambda: IA.frozen_gate_i_recipe()["minimizer"]["backends"]),
        ("gates/gate_i/carve/radius_factor", "physics.indent_analysis.GATE_I_RADIUS_FACTOR", lambda: IA.GATE_I_RADIUS_FACTOR),
        ("gates/gate_i/carve/depth_factor", "physics.indent_analysis.GATE_I_DEPTH_FACTOR", lambda: IA.GATE_I_DEPTH_FACTOR),
        ("gates/gate_i/carve/shell_A", "physics.indent_analysis.GATE_I_SHELL_A", lambda: IA.GATE_I_SHELL_A),
        ("gates/gate_i/carve/contact_radius", "physics.indent_analysis.frozen_gate_i_recipe()",
         lambda: IA.frozen_gate_i_recipe()["carve"]["contact_radius"]),
        ("gates/gate_i/carve/lateral_rule", "physics.indent_analysis.frozen_gate_i_recipe()",
         lambda: IA.frozen_gate_i_recipe()["carve"]["lateral_rule"]),
        ("gates/gate_i/thresholds", "physics.indent_analysis.GateIThresholds()",
         lambda: dc.asdict(IA.GateIThresholds())),
        ("gates/gate_i/verdict", "physics.indent_analysis.frozen_gate_i_recipe()",
         lambda: IA.frozen_gate_i_recipe()["verdict"]),
        # the Gate I evaluation rule (critic N1 / N2; recipe_v2): convergence, containment, enlarged carves
        ("gates/gate_i/evaluation", "physics.indent_analysis.gate_i_evaluation_rule()",
         lambda: IA.gate_i_evaluation_rule()),
        ("gates/gate_i/evaluation/carve_growth", "physics.indent_analysis.GATE_I_CARVE_GROWTH",
         lambda: IA.GATE_I_CARVE_GROWTH),
        ("gates/gate_i/evaluation/max_scale", "physics.indent_analysis.GATE_I_CARVE_MAX_SCALE",
         lambda: IA.GATE_I_CARVE_MAX_SCALE),
        ("gates/gate_ii/status", "physics.indent_analysis.GateIIThresholds().provisional",
         lambda: PROVISIONAL if IA.GateIIThresholds().provisional else "frozen"),
        ("gates/gate_ii/residual_min_A", "physics.indent_analysis.GateIIThresholds", g2("residual_min_A")),
        ("gates/gate_ii/residual_min_fraction", "physics.indent_analysis.GateIIThresholds", g2("residual_min_fraction")),
        ("gates/gate_ii/dissipation_min", "physics.indent_analysis.GateIIThresholds", g2("dissipation_min")),
        ("gates/gate_ii/hf_ratio_min", "physics.indent_analysis.GateIIThresholds", g2("hf_ratio_min")),
        ("gates/gate_iii/tolerance", "physics.indent_analysis.PLATEAU_TOL", lambda: IA.PLATEAU_TOL),
        ("gates/gate_iii/window_fraction", "physics.indent_analysis.PLATEAU_WINDOW_FRACTION",
         lambda: IA.PLATEAU_WINDOW_FRACTION),
        ("gates/gate_iii/estimators", "physics.indent_analysis.gate_iii(estimators)",
         lambda: list(inspect.signature(IA.gate_iii).parameters["estimators"].default)),
        ("gates/gate_e/sphere_R_A", "scripts/gate_e.py SPHERE_R_A", ge("SPHERE_R_A")),
        ("gates/gate_e/K_eV_per_A3", "scripts/gate_e.py WALL_K", ge("WALL_K")),
        ("gates/gate_e/tolerance", "scripts/gate_e.py GATE_TOL", ge("GATE_TOL")),
        ("gates/gate_e/gated_windows_A", "scripts/gate_e.py GATED_WINDOWS_A", ge("GATED_WINDOWS_A")),
        ("gates/gate_e/reported_windows_A", "scripts/gate_e.py REPORTED_WINDOWS_A", ge("REPORTED_WINDOWS_A")),
        ("gates/gate_e/aqs_far_field_tol", "scripts/gate_e.py FAR_FIELD_TOL", ge("FAR_FIELD_TOL")),
        ("gates/gate_e/aqs_far_field_r_A", "scripts/gate_e.py FAR_FIELD_R_A", ge("FAR_FIELD_R_A")),
        ("gates/gate_e/aqs_force_balance_tol", "scripts/gate_e.py FORCE_BALANCE_TOL", ge("FORCE_BALANCE_TOL")),
        ("gates/gate_e/contact_radius_tol", "scripts/gate_e.py CONTACT_RADIUS_TOL", ge("CONTACT_RADIUS_TOL")),
        ("gates/gate_e/cij_300K_GPa", "scripts/gate_e.py VASHISHTA_CIJ_300K", ge("VASHISHTA_CIJ_300K")),
        # per-run checks
        ("per_run_checks/newton3/tolerance", "physics.indentation.NEWTON3_TOL", lambda: IND.NEWTON3_TOL),
        ("per_run_checks/newton3/min_load_nN_md", "physics.indent_analysis.newton3_balance(min_load_nN)",
         sig(IA.newton3_balance, "min_load_nN")),
        ("per_run_checks/newton3/qs_min_load_fraction", "physics.indentation.QS_NEWTON3_MIN_LOAD_FRACTION",
         lambda: IND.QS_NEWTON3_MIN_LOAD_FRACTION),
        ("per_run_checks/energy_ledger/tolerance_over_W", "physics.indentation.LEDGER_TOL", lambda: IND.LEDGER_TOL),
        ("per_run_checks/energy_ledger/floor_eV_per_atom", "physics.indentation.LEDGER_FLOOR_EV_PER_ATOM",
         lambda: IND.LEDGER_FLOOR_EV_PER_ATOM),
        ("per_run_checks/energy_ledger/floor_linear_max_atoms", "physics.indentation.LEDGER_FLOOR_LINEAR_MAX_ATOMS",
         lambda: IND.LEDGER_FLOOR_LINEAR_MAX_ATOMS),
        ("analysis/contact_area/e3_radius_bias", "physics.indentation.E3_RADIUS_BIAS",
         lambda: {k: list(v) for k, v in IND.E3_RADIUS_BIAS.items()}),
        ("per_run_checks/equilibration_temperature_tol", "physics.indentation.EQUILIBRATION_T_TOL",
         lambda: IND.EQUILIBRATION_T_TOL),
        ("per_run_checks/lammps/min_cpu_percent", "physics.indentation.MIN_CPU_PERCENT", lambda: IND.MIN_CPU_PERCENT),
        ("per_run_checks/lammps/min_cpu_percent", "io.lammps_indent.LammpsLogInfo.cpu_use_check(min_percent)",
         sig(LammpsLogInfo.cpu_use_check, "min_percent")),
        ("per_run_checks/lammps/cpu_segment_min_steps", "io.lammps_indent.LammpsLogInfo.cpu_use_check(min_steps)",
         sig(LammpsLogInfo.cpu_use_check, "min_steps")),
        ("per_run_checks/lammps/cpu_segment_min_loop_s", "io.lammps_indent.LammpsLogInfo.cpu_use_check(min_loop_s)",
         sig(LammpsLogInfo.cpu_use_check, "min_loop_s")),
        # records: every name the recipe governs is the PropertyName the runner emits (review finding T10)
        *((f"records/names/{key}", f"schema.properties.PropertyName.{member}",
           (lambda m=member: SP.PropertyName[m].value)) for key, member in RECORD_NAME_MEMBERS.items()),
        ("records/status_ceiling_md", "schema.properties.MD_INDENTATION_STATUS_CEILING",
         lambda: SP.MD_INDENTATION_STATUS_CEILING.value),
        ("records/lammps_status_cap", "physics.indentation.ENGINE_STATUS_CAP['lammps']",
         lambda: IND.ENGINE_STATUS_CAP["lammps"].value),
        ("records/hv_per_mpa", "units.ISO14577_HV_PER_MPA", lambda: ISO14577_HV_PER_MPA),
    ]
    return b


def code_consistency(recipe: RecipeLike = DEFAULT_RECIPE) -> list[str]:
    """Mismatches between the recipe's frozen values and the code that implements them (empty list: consistent).

    Each entry names the recipe pointer, the code location and both values.  A provisional value (Gate II) is
    compared too: until S1 the code must carry the placeholder the recipe states."""
    rec = load_recipe(recipe)
    out: list[str] = []
    for ptr, where, getter in code_bindings():
        try:
            code = getter()
        except Exception as exc:  # noqa: BLE001 - reported as a mismatch, never hidden
            out.append(f"{ptr}: could not evaluate {where} ({type(exc).__name__}: {exc})")
            continue
        val = rec.get(ptr, _ABSENT)
        if ptr == "analysis/contact_area/estimators" and isinstance(val, Mapping):
            val = {k: v.get("name") for k, v in val.items()}
        if val is _ABSENT:
            out.append(f"{ptr}: not in {rec.recipe_id} (code {where} = {code!r})")
        elif not _same(val, code):
            out.append(f"{ptr} = {val!r} in {rec.recipe_id}, but {where} = {code!r}")
    return out


# ============================================================================================ protocol deviations
def _close(a: Any, b: Any, tol: float) -> bool:
    """Numbers within ``tol`` (absolute) or :func:`_same`; anything else :func:`_same`."""
    if tol > 0 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (a, b)):
        return abs(float(a) - float(b)) <= tol + 1e-12 * max(abs(float(a)), abs(float(b)))
    return _same(a, b)


def _preparation_deviations(rec: IndentationRecipe, preparation: Mapping[str, Any]) -> list[dict[str, Any]]:
    known = KNOWN_DEVIATIONS.get(rec.section_sha256("protocol"), ())
    n = int(preparation.get("langevin_steps", 0) or 0) + int(preparation.get("bath_steps", 0) or 0)
    thick = preparation.get("thickness_A")
    out = []
    for key, ptr in PREPARATION_FIELDS.items():
        field, val, ref = f"preparation.{key}", preparation.get(key), rec.get(ptr)
        tol = 1.0 / n if key.endswith("_fraction") and n else 0.0  # one step of step-count rounding
        if _close(val, ref, tol):
            continue
        row: dict[str, Any] = {"field": field, "value": val, "recipe": ref, "sanctioned": False}
        for k in known:
            if k["field"] == field and k["source"] == preparation.get("source") and _close(val, k["value"], tol) \
                    and (k.get("max_thickness_A") is None or (thick is not None and thick <= k["max_thickness_A"])):
                row.update(sanctioned=True, basis=k["basis"])
                break
        out.append(row)
    return out


def protocol_deviations(protocol: Any, recipe: RecipeLike = DEFAULT_RECIPE, *,
                        preparation: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
    """Where an ``IndentationProtocol`` departs from the recipe's protocol (stored next to the recipe hash; a LAMMPS
    deck's list is judged by :func:`deck_recipe_check`).  Each entry: field, value, recipe value and ``sanctioned`` =
    True for the recipe's sensitivity / check values (K = 1000, R_tip = 5 A, azimuth 30 deg, 20 and 10 m/s, dt =
    0.5 fs, the atomistic tip).

    ``preparation``: what the engine applies (:func:`preparation_summary`, :func:`lammps_deck_preparation`), compared
    field by field with ``protocol/preparation`` (fields ``preparation.<key>``; the Langevin and bath-only shares to one
    step).  A departure the recipe declares in its known_deviations (:data:`KNOWN_DEVIATIONS`) is sanctioned and names
    that pointer as its ``basis``.  Without ``preparation`` the preparation is not compared."""
    rec = load_recipe(recipe)
    g = rec.get
    ind = protocol.indenter
    rows: list[tuple[str, Any, Any, Sequence[Any]]] = [
        ("indenter.tip_model", ind.tip_model, g("protocol/tip/model"), ("atomistic",)),
        ("indenter.shape", ind.shape, g("protocol/tip/shape"), ()),
        ("indenter.face_angle_deg", ind.face_angle_deg, g("protocol/tip/face_angle_deg"), ()),
        ("indenter.azimuth_deg", ind.azimuth_deg, g("protocol/orientation/azimuth_deg"),
         (g("protocol/orientation/azimuth_check_deg"),)),
        ("plane_miller", list(protocol.plane_miller), g("protocol/orientation/surface_miller"), ()),
        ("top_termination", protocol.top_termination, g("protocol/surface/termination"), ()),
        ("mode", protocol.mode, "md", ()),
        ("fixed_thickness_A", protocol.fixed_thickness_A, g("protocol/boundary_layers/fixed_thickness_A"), ()),
        ("thermostat_thickness_A", protocol.thermostat_thickness_A, g("protocol/boundary_layers/bath_thickness_A"), ()),
        ("thermostat", protocol.thermostat, g("protocol/boundary_layers/bath"), ()),
        ("thermostat_tau_ps", protocol.thermostat_tau_ps, g("protocol/boundary_layers/bath_tau_ps"), ()),
        ("temperature_K", protocol.temperature_K, g("protocol/temperature_K"), ()),
        ("lattice_parameter_A", protocol.lattice_parameter_A, g("protocol/surface/lattice_constant_A"), ()),
        ("speed_m_per_s", protocol.speed_m_per_s, g("protocol/loading/speed_m_per_s"),
         tuple(g("protocol/loading/speed_checks_m_per_s"))),
        ("initial_gap_A", protocol.initial_gap_A, g("protocol/loading/initial_gap_A"), ()),
        ("timestep_fs", protocol.timestep_fs, g("protocol/time_step/dt_fs"), (g("protocol/time_step/plateau_check/dt_fs"),)),
        ("hold_ps", protocol.hold_ps, g("protocol/hold_unload/hold_ps"), ()),
        ("unload", protocol.unload, True, ()),
        ("relax_after_unload_ps", protocol.relax_after_unload_ps, g("protocol/hold_unload/relax_after_unload_ps"), ()),
        ("equilibration_ps", protocol.equilibration_ps, g("protocol/preparation/equilibration_ps"), ()),
        ("relax.fmax_eV_per_A", protocol.relax.fmax_eV_per_A, g("protocol/preparation/relax_fmax_eV_per_A"), ()),
        ("block_average_ps", protocol.block_average_ps, g("analysis/load/block_average_ps"), ()),
        ("primary_area_estimator", protocol.primary_area_estimator, g("analysis/contact_area/estimators/E3/name"), ()),
        ("size_tier", protocol.size_tier(), sorted(g("protocol/cell/tiers")), ()),
    ]
    if ind.tip_model == "rigid_sdf":
        rows += [("indenter.tip_radius_A", ind.tip_radius_A, g("protocol/tip/tip_radius_A"),
                  (g("protocol/tip/tip_radius_check_A"),)),
                 ("indenter.stiffness_eV_per_A3", ind.stiffness_eV_per_A3, g("protocol/wall_stiffness/K_eV_per_A3"),
                  (g("protocol/wall_stiffness/K_check_eV_per_A3"),))]
    out = []
    for field, val, ref, sanctioned in rows:
        ok = val in ref if field == "size_tier" else _same(val, ref)
        if not ok:
            out.append({"field": field, "value": val, "recipe": ref,
                        "sanctioned": any(_same(val, s) for s in sanctioned)})
    if preparation is not None:
        out += _preparation_deviations(rec, preparation)
    return out


def manifest_entry(recipe: RecipeLike = DEFAULT_RECIPE, protocol: Any = None, *,
                   preparation: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """The ``recipe`` block of a generated run's manifest.json: identity (:meth:`IndentationRecipe.summary`), the
    ``preparation`` the deck applies (:func:`lammps_deck_preparation`) and, with a protocol, its
    :func:`protocol_deviations` (preparation included).  :func:`deck_recipe_check` needs both the deviations and the
    preparation."""
    rec = load_recipe(recipe)
    out = rec.summary()
    if preparation is not None:
        out["preparation"] = dict(preparation)
    if protocol is not None:
        out["protocol_deviations"] = protocol_deviations(protocol, rec, preparation=preparation)
    return out


def deck_recipe_check(deck_entry: Optional[Mapping[str, Any]], recipe: Optional[RecipeLike]) -> Any:
    """CheckResult ``indentation_recipe_protocol_matches_deck``: was the LAMMPS run made under the protocol of the
    recipe its records are built under?  ``deck_entry``: the deck manifest's ``recipe`` block (:func:`manifest_entry`).

    Passes only when (1) the deck's recipe is the records' recipe, or has an identical protocol section (e.g. a later
    version that only freezes the Gate II thresholds); (2) the deck lists its ``protocol_deviations`` and every one is
    sanctioned (a sensitivity / check value, or a known deviation of the recipe); and (3) the deck states its
    ``preparation``.  Fails otherwise (re-run under the recipe, or analyse with the deck's recipe).  None when either
    side has no recipe (legacy decks, an explicit opt-out)."""
    from fracture_md.schema.common import CheckResult

    if not deck_entry or recipe is None:
        return None
    rec = load_recipe(recipe)
    d_sha, d_proto = deck_entry.get("sha256"), deck_entry.get("protocol_sha256")
    d_id = deck_entry.get("recipe_id", "?")
    if d_sha == rec.sha256:
        same, parts = True, [f"deck generated and records built under {rec.recipe_id} (sha256 {rec.sha256[:12]}...)"]
    else:
        same = d_proto is not None and d_proto == rec.section_sha256("protocol")
        parts = [f"deck generated under {d_id} (sha256 {str(d_sha)[:12]}...), records built under {rec.recipe_id} "
                 f"(sha256 {rec.sha256[:12]}...): protocol sections "
                 + ("identical" if same else "DIFFER (or the deck states no protocol digest)")]
    devs = deck_entry.get("protocol_deviations")
    prep = deck_entry.get("preparation")
    listed = [d for d in devs if isinstance(d, Mapping)] if isinstance(devs, list) else []
    if isinstance(prep, Mapping):  # the stated preparation, judged again under the records' recipe
        fields = {d.get("field") for d in listed}
        listed += [d for d in _preparation_deviations(rec, prep) if d["field"] not in fields]
    bad = [d for d in listed if d.get("sanctioned") is not True]
    if isinstance(devs, list) and any(not isinstance(d, Mapping) for d in devs):
        bad.append({"field": "?", "value": "a malformed protocol_deviations entry", "recipe": None})
    if not isinstance(devs, list):
        parts.append("the deck lists no protocol_deviations: its protocol was NOT compared with the recipe")
    if not isinstance(prep, Mapping):
        parts.append("the deck states no preparation (z pre-expansion, Langevin split): NOT compared with the recipe")
    if bad:
        parts.append(f"{len(bad)} UNSANCTIONED deviation(s) from the recipe protocol: " + "; ".join(
            f"{d.get('field')} = {d.get('value')!r} (recipe {d.get('recipe')!r})" for d in bad))
    ok = [str(d.get("field")) for d in listed if d.get("sanctioned") is True]
    parts.append(f"{len(ok)} sanctioned deviation(s)" + (f": {', '.join(ok)}" if ok else ""))
    passed = bool(same) and isinstance(devs, list) and isinstance(prep, Mapping) and not bad
    return CheckResult(name=DECK_RECIPE_CHECK, passed=passed, details="; ".join(parts))


__all__ = [
    "CANONICAL_VALUES_PATH", "CanonicalValues", "DECK_RECIPE_CHECK", "DEFAULT_RECIPE", "FROZEN_RECIPE_SHA256",
    "IndentationRecipe", "JAX_RUNNER_PREPARATION", "KNOWN_DEVIATIONS", "PREPARATION_FIELDS", "PROVISIONAL",
    "RECIPE_DIR", "RECIPE_V0_PROTOCOL_SHA256", "RECORD_NAME_MEMBERS", "REQUIRED_POINTERS", "RecipeError",
    "canonical_json", "code_bindings", "code_consistency", "deck_recipe_check", "lammps_deck_preparation",
    "load_canonical_values", "load_recipe", "manifest_entry", "preparation_summary", "protocol_deviations",
    "recipe_path", "recipe_sha256", "resolve_pointer", "resolve_recipe", "runs_needed", "section_sha256",
    "sha256_bytes", "validate_canonical_values", "validate_recipe",
]
