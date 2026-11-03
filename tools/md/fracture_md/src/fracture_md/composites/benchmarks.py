"""Typed benchmark entries (SprintRay Midas Restore, Ceramic Crown, competitors, ISO limits,
proposed targets) loaded from ``data/dental/targets.json``, and a comparison of model
predictions against them with explicit test-method-mismatch flags.

Evidence levels are kept apart on purpose: a manufacturer claim ("82 HV, load not stated"),
a 510(k) bench-test summary and an independent peer-reviewed measurement are different
kinds of evidence and are never merged.

Usage::

    from fracture_md.composites.benchmarks import load_benchmarks, compare, format_table
    bm = load_benchmarks()
    rows = compare(prediction_records)          # list[ComparisonRow]
    print(format_table(rows))
"""

from __future__ import annotations

import json
import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from pydantic import Field

from fracture_md.schema.common import StrictModel
from fracture_md.schema.properties import PropertyName, PropertyRecord
from fracture_md.units import Unit, convert

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGETS_PATH = REPO_ROOT / "data" / "dental" / "targets.json"


class EvidenceLevel(str, Enum):
    MANUFACTURER_CLAIM = "manufacturer_claim"  # "Mfr"
    FDA_510K = "fda_510k"  # "FDA" (510(k) summary bench test)
    MANUFACTURER_OR_FDA = "manufacturer_or_fda"  # "Mfr/FDA"
    INDEPENDENT = "independent_study"  # "Indep" (peer-reviewed third party)
    ISO_REQUIREMENT = "iso_requirement"  # "ISO"
    PROPOSED_TARGET = "proposed_target"  # "Target" (brief author's recommendation, not literature)


_EVIDENCE_MAP = {
    "Mfr": EvidenceLevel.MANUFACTURER_CLAIM,
    "FDA": EvidenceLevel.FDA_510K,
    "Mfr/FDA": EvidenceLevel.MANUFACTURER_OR_FDA,
    "Indep": EvidenceLevel.INDEPENDENT,
    "ISO": EvidenceLevel.ISO_REQUIREMENT,
    "Target": EvidenceLevel.PROPOSED_TARGET,
}

# Test-method families.  Values in different families are not directly comparable:
# flexural_3pt, flexural_biaxial, vickers, instrumented_indentation, shore_d, fracture_toughness,
# water_sorption, water_solubility, viscosity, bond_shear, bond_tensile, modulus, filler_content,
# cure, other.


class BenchmarkEntry(StrictModel):
    """One published (or required/targeted) value with its evidence level and test method."""

    group: str  # section of targets.json (sprintray_reference, competitors, ...)
    product: str
    property: str  # raw key from targets.json
    value: Optional[float] = None  # point value (mean) in `unit`, if numeric
    low: Optional[float] = None  # range lower bound in `unit`
    high: Optional[float] = None
    unit: Optional[Unit] = None
    raw_value: Any = None
    raw_unit: str = ""
    qualitative: Optional[str] = None  # for string-valued claims
    hv_load: Optional[str] = None  # e.g. "HV0.2" when the hardness load is known
    test_method: str = ""
    test_family: str = "other"
    condition: Optional[str] = None
    evidence: EvidenceLevel
    source_url: Optional[str] = None
    notes: str = ""

    @property
    def is_numeric(self) -> bool:
        return self.value is not None or (self.low is not None and self.high is not None)

    def central(self) -> Optional[float]:
        if self.value is not None:
            return self.value
        if self.low is not None and self.high is not None:
            return 0.5 * (self.low + self.high)
        return None


_UNIT_MAP = {
    "MPa": Unit.MPA,
    "GPa": Unit.GPA,
    "HV": Unit.HV,
    "ug/mm3": Unit.MICROGRAM_PER_MM3,
    "Pa.s": Unit.PA_S,
    "MPa.m^0.5": Unit.MPA_SQRT_M,
    "wt%": Unit.WT_PERCENT,
    "vol%": Unit.VOL_PERCENT,
    "nm": Unit.NANOMETER,
    "mm": Unit.MILLIMETER,
    "s": Unit.SECOND,
    "J/m2": Unit.J_PER_M2,
    "%": Unit.PERCENT,
}


def _family(prop: str, method: str) -> str:
    p, m = prop.lower(), method.lower()
    if "biaxial" in p or "piston-on-three-balls" in m or (
        "iso 6872" in m and "3-point" not in m and "modified" not in m and "flexural" in p
    ):
        return "flexural_biaxial"
    if "flexural_strength" in p:
        return "flexural_3pt"
    if "shore" in p:
        return "shore_d"
    if "nanoindentation" in p:
        return "instrumented_indentation"
    if "vickers" in p or "hardness" in p:
        return "vickers"
    if "toughness" in p:
        return "fracture_toughness"
    if "sorption" in p:
        return "water_sorption"
    if "solubility" in p:
        return "water_solubility"
    if "viscosity" in p:
        return "viscosity"
    if "shear_bond" in p or "micro_shear" in p:
        return "bond_shear"
    if "tensile_bond" in p or "microtensile" in p:
        return "bond_tensile"
    if "modulus" in p:
        return "modulus"
    if "filler" in p:
        return "filler_content"
    if "cure" in p:
        return "cure"
    return "other"


def _parse_entry(group: str, e: dict) -> BenchmarkEntry:
    raw_v, raw_u = e.get("value"), str(e.get("unit") or "")
    unit: Optional[Unit] = None
    hv_load = None
    if raw_u.startswith("HV"):
        unit = Unit.HV
        hv_load = raw_u if raw_u != "HV" else None
    elif raw_u in _UNIT_MAP:
        unit = _UNIT_MAP[raw_u]
    value = low = high = None
    qualitative = None
    if isinstance(raw_v, (int, float)) and not isinstance(raw_v, bool):
        value = float(raw_v)
    elif isinstance(raw_v, list) and len(raw_v) == 2 and all(isinstance(x, (int, float)) for x in raw_v):
        low, high = float(min(raw_v)), float(max(raw_v))
    elif isinstance(raw_v, str):
        qualitative = raw_v
        m = re.match(r"^\s*[<>]=?\s*([0-9.]+)", raw_v)
        if m and unit is not None:
            value = float(m.group(1))
    method = str(e.get("test_standard_or_method") or "")
    if unit in (Unit.HV, Unit.GPA) and "vickers" in e["property"] and hv_load is None:
        m = re.search(r"HV\s?([0-9.]+)", method)
        if m:
            hv_load = f"HV{m.group(1)}"
    if unit == Unit.GPA and "vickers" in e["property"]:
        # Hardness quoted in GPa (e.g. VITA Enamic 2.5 GPa, HV3): store in HV for comparability.
        f = convert(1.0, Unit.GPA, Unit.HV)
        value = value * f if value is not None else None
        low = low * f if low is not None else None
        high = high * f if high is not None else None
        unit = Unit.HV
    if raw_u == "wt%, vol%":  # Lava Ultimate [80, 65] "wt%, vol%": two different quantities
        low = high = value = None
        qualitative = f"{raw_v[0]} wt% / {raw_v[1]} vol%"
        unit = None
    return BenchmarkEntry(
        group=group,
        product=e["product"],
        property=e["property"],
        value=value,
        low=low,
        high=high,
        unit=unit,
        raw_value=raw_v,
        raw_unit=raw_u,
        qualitative=qualitative,
        hv_load=hv_load,
        test_method=method,
        test_family=_family(e["property"], method),
        condition=e.get("condition"),
        evidence=_EVIDENCE_MAP.get(str(e.get("evidence")), EvidenceLevel.MANUFACTURER_CLAIM),
        source_url=e.get("source_url"),
        notes=str(e.get("note", "") or ""),
    )


class BenchmarkSet(StrictModel):
    source_path: str
    entries: list[BenchmarkEntry] = Field(default_factory=list)

    def for_product(self, product_substring: str) -> list[BenchmarkEntry]:
        s = product_substring.lower()
        return [e for e in self.entries if s in e.product.lower()]

    def find(self, product_substring: str, property_substring: str) -> list[BenchmarkEntry]:
        p = property_substring.lower()
        return [e for e in self.for_product(product_substring) if p in e.property.lower()]

    def group(self, name: str) -> list[BenchmarkEntry]:
        return [e for e in self.entries if e.group == name]


@lru_cache(maxsize=4)
def _load_cached(path: str) -> BenchmarkSet:
    d = json.loads(Path(path).read_text())
    entries = []
    for group, items in d.items():
        if group == "meta" or not isinstance(items, list):
            continue
        for e in items:
            entries.append(_parse_entry(group, e))
    return BenchmarkSet(source_path=str(path), entries=entries)


def load_benchmarks(path: Path | str | None = None) -> BenchmarkSet:
    return _load_cached(str(path or TARGETS_PATH))


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
# PropertyName -> (benchmark property keys, test family of the prediction, higher_is_better, unit)
_PRED_MAP: dict[PropertyName, tuple[tuple[str, ...], str, bool, Unit]] = {
    PropertyName.VICKERS_HARDNESS_ESTIMATE: (("vickers_hardness",), "vickers", True, Unit.HV),
    PropertyName.VICKERS_HARDNESS_MEASURED: (("vickers_hardness",), "vickers", True, Unit.HV),
    PropertyName.FLEXURAL_STRENGTH_ESTIMATE: (("flexural_strength", "biaxial_flexural_strength"), "flexural_3pt", True, Unit.MPA),
    PropertyName.FLEXURAL_STRENGTH_MEASURED: (("flexural_strength", "biaxial_flexural_strength"), "flexural_3pt", True, Unit.MPA),
    PropertyName.BIAXIAL_FLEXURAL_STRENGTH_MEASURED: (("biaxial_flexural_strength", "flexural_strength"), "flexural_biaxial", True, Unit.MPA),
    PropertyName.FLEXURAL_MODULUS_ESTIMATE: (("flexural_modulus", "elastic_modulus"), "modulus", True, Unit.GPA),
    PropertyName.FLEXURAL_MODULUS_MEASURED: (("flexural_modulus", "elastic_modulus"), "modulus", True, Unit.GPA),
    PropertyName.EFFECTIVE_YOUNGS_MODULUS: (("flexural_modulus", "elastic_modulus"), "modulus", True, Unit.GPA),
    PropertyName.FRACTURE_TOUGHNESS_ESTIMATE: (("fracture_toughness",), "fracture_toughness", True, Unit.MPA_SQRT_M),
    PropertyName.FRACTURE_TOUGHNESS_MEASURED: (("fracture_toughness",), "fracture_toughness", True, Unit.MPA_SQRT_M),
    PropertyName.WATER_SORPTION_ESTIMATE: (("water_sorption",), "water_sorption", False, Unit.MICROGRAM_PER_MM3),
    PropertyName.WATER_SORPTION_MEASURED: (("water_sorption",), "water_sorption", False, Unit.MICROGRAM_PER_MM3),
    PropertyName.WATER_SOLUBILITY_ESTIMATE: (("water_solubility",), "water_solubility", False, Unit.MICROGRAM_PER_MM3),
    PropertyName.WATER_SOLUBILITY_MEASURED: (("water_solubility",), "water_solubility", False, Unit.MICROGRAM_PER_MM3),
    PropertyName.VISCOSITY_ESTIMATE: (("viscosity_uncured",), "viscosity", False, Unit.PA_S),
    PropertyName.BOND_STRENGTH_ESTIMATE: (("shear_bond", "micro_shear", "tensile_bond", "microtensile"), "bond_tensile_assembly", True, Unit.MPA),
    PropertyName.BOND_STRENGTH_MEASURED: (("shear_bond", "micro_shear"), "bond_shear", True, Unit.MPA),
}

DEFAULT_COMPARATORS = ("SprintRay Midas Restore", "SprintRay Ceramic Crown")


class ComparisonRow(StrictModel):
    prediction_record_id: str
    property_name: PropertyName
    predicted: float
    predicted_low: Optional[float] = None
    predicted_high: Optional[float] = None
    unit: Unit
    predicted_method: str = ""
    predicted_test_family: str = ""
    benchmark_product: str
    benchmark_property: str
    benchmark_value: Optional[float] = None
    benchmark_low: Optional[float] = None
    benchmark_high: Optional[float] = None
    benchmark_evidence: EvidenceLevel
    benchmark_method: str = ""
    benchmark_test_family: str = ""
    benchmark_condition: Optional[str] = None
    higher_is_better: bool
    verdict: Literal["beats", "below", "overlaps", "not_comparable"]
    margin: Optional[float] = None  # predicted central - benchmark central (sign: + = better)
    method_mismatch: bool = False
    mismatch_notes: list[str] = Field(default_factory=list)


def record_range(rec: PropertyRecord) -> tuple[float, Optional[float], Optional[float]]:
    """(value, low, high) of a record; low/high from a 'range' uncertainty (samples=[lo, hi])."""
    v = rec.value.magnitude()
    lo = hi = None
    for u in rec.uncertainty:
        if u.interval == "range" and u.samples and len(u.samples) == 2:
            lo, hi = min(u.samples), max(u.samples)
            break
    if lo is None:
        for u in rec.uncertainty:
            if u.value is not None:
                lo, hi = v - u.value, v + u.value
                break
    return v, lo, hi


def _temperature_c(text: str) -> Optional[float]:
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°|deg)?\s*C\b", text or "")
    return float(m.group(1)) if m else None


def compare(
    prediction_records: Iterable[PropertyRecord],
    benchmarks: Optional[BenchmarkSet] = None,
    products: Iterable[str] = DEFAULT_COMPARATORS,
    prediction_test_family: Optional[dict[PropertyName, str]] = None,
    prediction_hv_load: str = "HV0.2",
) -> list[ComparisonRow]:
    """Tabulate predicted vs benchmark values.

    For each prediction record whose name is known, every numeric benchmark of the named
    products with a matching property key is compared.  ``prediction_test_family`` overrides
    the test family a prediction represents (e.g. ``{FLEXURAL_STRENGTH_ESTIMATE:
    "flexural_biaxial"}`` for a Weibull-converted biaxial strength).  Flags: test family
    (3-point vs biaxial -> verdict ``not_comparable``), unknown or different hardness load,
    conditioning, and the benchmark's evidence level.  Verdicts use ranges: ``beats`` only if
    the whole predicted range is better than the whole benchmark range.
    """
    bm = benchmarks or load_benchmarks()
    fam_override = prediction_test_family or {}
    rows: list[ComparisonRow] = []
    for rec in prediction_records:
        if rec.name not in _PRED_MAP:
            continue
        keys, fam, hib, unit = _PRED_MAP[rec.name]
        fam = fam_override.get(rec.name, fam)
        try:
            factor = convert(1.0, rec.value.unit, unit)
        except Exception:
            continue
        v, lo, hi = record_range(rec)
        v = v * factor
        lo = lo * factor if lo is not None else None
        hi = hi * factor if hi is not None else None
        for prod in products:
            for e in bm.for_product(prod):
                if not any(k in e.property for k in keys):
                    continue
                if not e.is_numeric or e.unit is None:
                    continue
                try:
                    bf = convert(1.0, e.unit, unit)
                except Exception:
                    continue
                bc = e.central() * bf  # type: ignore[operator]
                blo = e.low * bf if e.low is not None else None
                bhi = e.high * bf if e.high is not None else None
                notes: list[str] = []
                mismatch = False
                bond_pair = fam.startswith("bond") or e.test_family.startswith("bond")
                if bond_pair and fam != e.test_family:
                    mismatch = True
                    notes.append(f"bond tests not comparable: prediction = {fam}, benchmark = {e.test_family} "
                                 "(whole-assembly tensile vs single-interface shear; different stress state and failure locus)")
                elif e.test_family != fam:
                    mismatch = True
                    notes.append(f"test method differs: prediction = {fam}, benchmark = {e.test_family} ({e.test_method})")
                if fam == "vickers":
                    if e.hv_load is None:
                        mismatch = True
                        notes.append("benchmark hardness load not stated (cannot compare rigorously)")
                    elif e.hv_load != prediction_hv_load:
                        mismatch = True
                        notes.append(f"hardness load differs ({prediction_hv_load} vs {e.hv_load}): indentation-size effect")
                if fam == "modulus":
                    pred_flex = rec.name in (PropertyName.FLEXURAL_MODULUS_ESTIMATE, PropertyName.FLEXURAL_MODULUS_MEASURED)
                    bench_flex = "flexural" in e.property
                    if pred_flex != bench_flex:
                        mismatch = True
                        notes.append(f"modulus method differs: prediction = {'flexural' if pred_flex else 'elastic'}, "
                                     f"benchmark = {'flexural' if bench_flex else 'elastic'} modulus")
                if fam == "flexural_3pt" and any(k in (e.test_method or "") for k in ("10477", "1.2", "12 mm", "10 mm span")):
                    notes.append("3-point bar geometry/standard differs from ISO 4049 2x2x25 mm bars (ISO 10477 or short bars): size effect")
                pred_env = (rec.conditions.environment or "").strip().lower() if rec.conditions else ""
                if e.condition and e.condition.strip().lower() not in pred_env:
                    mismatch = True
                    notes.append(f"conditioning differs: prediction '{pred_env or 'unstated'}' vs benchmark '{e.condition}'")
                t_b = _temperature_c(" ".join(x for x in (e.condition, e.test_method, e.property) if x))
                t_p = _temperature_c(pred_env)
                if t_p is None and rec.conditions and rec.conditions.temperature and rec.conditions.temperature.value > 0:
                    t_p = rec.conditions.temperature.magnitude(Unit.KELVIN) - 273.15
                if fam == "viscosity" and t_b is not None and t_p is not None and abs(t_b - t_p) > 2.0:
                    mismatch = True
                    notes.append(f"test temperature differs: prediction {t_p:.0f} C vs benchmark {t_b:.0f} C")
                if e.evidence in (EvidenceLevel.MANUFACTURER_CLAIM, EvidenceLevel.MANUFACTURER_OR_FDA):
                    notes.append("benchmark is a manufacturer claim (not independently verified)")
                elif e.evidence == EvidenceLevel.FDA_510K:
                    notes.append("benchmark is a 510(k) summary bench value")
                if e.condition:
                    notes.append(f"benchmark conditioning: {e.condition}")
                if {e.test_family, fam} == {"flexural_biaxial", "flexural_3pt"}:
                    notes.append(
                        "biaxial (ISO 6872 disc) and 3-point bar strengths differ for the same material "
                        "(different effective volume/stress state); convert by Weibull scaling first"
                    )
                p_lo = lo if lo is not None else v
                p_hi = hi if hi is not None else v
                b_lo = blo if blo is not None else bc
                b_hi = bhi if bhi is not None else bc
                if {e.test_family, fam} == {"flexural_biaxial", "flexural_3pt"} or (bond_pair and fam != e.test_family):
                    verdict = "not_comparable"
                elif hib:
                    verdict = "beats" if p_lo > b_hi else ("below" if p_hi < b_lo else "overlaps")
                else:
                    verdict = "beats" if p_hi < b_lo else ("below" if p_lo > b_hi else "overlaps")
                margin = (v - bc) if hib else (bc - v)
                rows.append(
                    ComparisonRow(
                        prediction_record_id=rec.id,
                        property_name=rec.name,
                        predicted=v,
                        predicted_low=lo,
                        predicted_high=hi,
                        unit=unit,
                        predicted_method=rec.method.name,
                        predicted_test_family=fam,
                        benchmark_product=e.product,
                        benchmark_property=e.property,
                        benchmark_value=bc,
                        benchmark_low=blo,
                        benchmark_high=bhi,
                        benchmark_evidence=e.evidence,
                        benchmark_method=e.test_method,
                        benchmark_test_family=e.test_family,
                        benchmark_condition=e.condition,
                        higher_is_better=hib,
                        verdict=verdict,
                        margin=margin,
                        method_mismatch=mismatch,
                        mismatch_notes=notes,
                    )
                )
    return rows


def format_table(rows: Iterable[ComparisonRow]) -> str:
    """Markdown table of a comparison."""
    lines = [
        "| property | predicted [range] | benchmark (product; evidence; key) | benchmark value | verdict | method mismatch / notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        rng = (
            f" [{r.predicted_low:.3g}-{r.predicted_high:.3g}]"
            if r.predicted_low is not None and r.predicted_high is not None
            else ""
        )
        bv = (
            f"{r.benchmark_low:.3g}-{r.benchmark_high:.3g}"
            if r.benchmark_low is not None and r.benchmark_high is not None
            else f"{r.benchmark_value:.3g}"
        )
        flag = ("YES: " if r.method_mismatch else "no; ") + "; ".join(r.mismatch_notes)
        lines.append(
            f"| {r.property_name.value} | {r.predicted:.3g}{rng} {r.unit.value} | {r.benchmark_product} "
            f"({r.benchmark_evidence.value}; {r.benchmark_property}) | {bv} {r.unit.value} | {r.verdict} | {flag} |"
        )
    return "\n".join(lines)
