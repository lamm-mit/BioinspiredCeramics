"""Generic multi-objective candidate ranking with uncertainty and validation status.

Design rules (shared by the ceramic and dental workstreams):

* **No single score.**  Objectives are declared in named *groups* (e.g. ``strength``,
  ``fracture``, ``durability``).  Pareto tiers are computed per group and over the union
  of all objectives; nothing is ever summed or weighted.
* **Uncertainty.**  Every objective value carries a ``[low, high]`` range.  Two dominance
  modes are offered: ``"median"`` (classical Pareto on central values) and ``"interval"``
  (A dominates B only where A's pessimistic end beats B's optimistic end, i.e. the ranges
  do not overlap - robust / "certain" dominance).  Pairwise ``probability_better`` treats
  each range as a stated interval of a normal (or log-normal) distribution.
* **Validation status propagates as the weakest link.**  A candidate's status is the
  weakest status among the objective values it is ranked on (FAILED < OUT_OF_DOMAIN <
  UNVALIDATED < MODEL_ESTIMATE < ... < VALIDATED_AGAINST_EXPERIMENT).
* **Explicit tiers for what cannot be ranked.**  A candidate that violates a constraint is
  put in ``rejected`` (with reasons); a candidate missing a required objective, or whose
  value is non-finite / FAILED, is put in ``not_evaluable``.  Neither is silently dropped
  or silently ranked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping, Optional, Sequence

from fracture_md.schema.common import ValidationStatus

STATUS_RANK = {
    ValidationStatus.FAILED: 0,
    ValidationStatus.OUT_OF_DOMAIN: 1,
    ValidationStatus.UNVALIDATED: 2,
    ValidationStatus.MODEL_ESTIMATE: 3,
    ValidationStatus.NUMERICALLY_VERIFIED: 4,
    ValidationStatus.REFERENCE_IMPLEMENTATION_AGREEMENT: 5,
    ValidationStatus.LITERATURE_MODEL_AGREEMENT: 6,
    ValidationStatus.VALIDATED_AGAINST_DFT: 7,
    ValidationStatus.VALIDATED_AGAINST_EXPERIMENT: 8,
}

DominanceMode = Literal["median", "interval"]


def weakest_status(statuses: Iterable[ValidationStatus | str]) -> ValidationStatus:
    """Weakest evidence level among ``statuses`` (UNVALIDATED if empty)."""
    s = [ValidationStatus(x) for x in statuses]
    if not s:
        return ValidationStatus.UNVALIDATED
    return min(s, key=lambda x: STATUS_RANK[x])


@dataclass(frozen=True)
class Objective:
    """One ranked quantity.  ``direction`` = "max" or "min"; ``group`` names its objective group.
    ``required`` objectives make a candidate ``not_evaluable`` when missing."""

    name: str
    group: str
    direction: Literal["max", "min"] = "max"
    unit: str = ""
    required: bool = True
    log_scale: bool = False  # treat ranges as log-normal in probability_better
    description: str = ""


@dataclass
class ObjectiveValue:
    value: float
    low: Optional[float] = None
    high: Optional[float] = None
    status: ValidationStatus = ValidationStatus.MODEL_ESTIMATE
    interval_z: float = 1.645  # half-width of [low, high] in standard deviations (1.645: 5-95 %, 1.96: 95 %)
    note: str = ""

    def lo(self) -> float:
        return self.value if self.low is None else min(self.low, self.value)

    def hi(self) -> float:
        return self.value if self.high is None else max(self.high, self.value)

    def finite(self) -> bool:
        return all(math.isfinite(x) for x in (self.value, self.lo(), self.hi()))


@dataclass
class CandidateEvaluation:
    candidate_id: str
    objectives: dict[str, ObjectiveValue] = field(default_factory=dict)
    constraint_violations: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class RankingResult:
    objectives: list[Objective]
    mode: DominanceMode
    tiers_overall: dict[str, int]  # candidate id -> Pareto tier (0 = non-dominated) over all objectives
    tiers_by_group: dict[str, dict[str, int]]  # group -> candidate id -> tier
    rejected: dict[str, list[str]]  # candidate id -> reasons
    not_evaluable: dict[str, list[str]]  # candidate id -> reasons
    status: dict[str, ValidationStatus]  # weakest status over the ranked objectives (all candidates)

    def tier_members(self, tier: int = 0, group: Optional[str] = None) -> list[str]:
        t = self.tiers_overall if group is None else self.tiers_by_group[group]
        return sorted(k for k, v in t.items() if v == tier)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "objectives": [o.__dict__ for o in self.objectives],
            "tiers_overall": self.tiers_overall,
            "tiers_by_group": self.tiers_by_group,
            "rejected": self.rejected,
            "not_evaluable": self.not_evaluable,
            "weakest_status": {k: v.value for k, v in self.status.items()},
        }


def _better_or_equal(a: ObjectiveValue, b: ObjectiveValue, o: Objective, mode: DominanceMode) -> tuple[bool, bool]:
    """(a >= b, a > b) for objective o in the given mode."""
    if mode == "median":
        x, y = a.value, b.value
        if o.direction == "min":
            x, y = -x, -y
        return x >= y, x > y
    # interval: a is at least as good if its pessimistic end is not worse than b's optimistic end
    if o.direction == "max":
        ge = a.lo() >= b.hi()
        gt = a.lo() > b.hi()
    else:
        ge = a.hi() <= b.lo()
        gt = a.hi() < b.lo()
    return ge, gt


def dominates(a: CandidateEvaluation, b: CandidateEvaluation, objectives: Sequence[Objective],
              mode: DominanceMode = "median") -> bool:
    """Pareto dominance of a over b on ``objectives`` (both must carry all of them).

    ``interval`` mode: a dominates b only if a is certainly at least as good on every objective
    (non-overlapping or touching ranges) and certainly better on at least one."""
    any_strict = False
    for o in objectives:
        ge, gt = _better_or_equal(a.objectives[o.name], b.objectives[o.name], o, mode)
        if not ge:
            if mode == "interval":
                # overlapping ranges on this objective: no certain dominance
                return False
            return False
        any_strict = any_strict or gt
    return any_strict


def pareto_tiers(cands: Sequence[CandidateEvaluation], objectives: Sequence[Objective],
                 mode: DominanceMode = "median") -> dict[str, int]:
    """Non-dominated sorting (tier 0 = Pareto front).  O(n^2) per tier; fine for n <~ 10^3."""
    remaining = list(cands)
    tiers: dict[str, int] = {}
    t = 0
    while remaining:
        front = [c for c in remaining if not any(dominates(o, c, objectives, mode) for o in remaining if o is not c)]
        if not front:  # cannot happen for a strict partial order; guard anyway
            front = list(remaining)
        for c in front:
            tiers[c.candidate_id] = t
        ids = {id(c) for c in front}
        remaining = [c for c in remaining if id(c) not in ids]
        t += 1
    return tiers


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def probability_better(a: ObjectiveValue, threshold: float | ObjectiveValue, direction: Literal["max", "min"] = "max",
                       log_scale: bool = False) -> float:
    """P(a better than threshold) treating [low, high] as a symmetric +/- interval_z sigma interval
    (normal, or log-normal if ``log_scale``).  The threshold may itself be uncertain (independent)."""
    def mu_sd(v: ObjectiveValue) -> tuple[float, float]:
        lo, hi, c = v.lo(), v.hi(), v.value
        if log_scale:
            lo, hi, c = (math.log(max(x, 1e-300)) for x in (lo, hi, c))
        return c, max((hi - lo) / (2.0 * v.interval_z), 0.0)

    ma, sa = mu_sd(a)
    if isinstance(threshold, ObjectiveValue):
        mb, sb = mu_sd(threshold)
    else:
        mb, sb = (math.log(max(threshold, 1e-300)) if log_scale else float(threshold)), 0.0
    s = math.hypot(sa, sb)
    d = (ma - mb) if direction == "max" else (mb - ma)
    if s == 0.0:
        return 1.0 if d > 0 else (0.5 if d == 0 else 0.0)
    return _normal_cdf(d / s)


def rank_candidates(cands: Sequence[CandidateEvaluation], objectives: Sequence[Objective],
                    mode: DominanceMode = "median") -> RankingResult:
    """Sort candidates into rejected / not_evaluable / Pareto tiers (overall and per group).

    Only *required* objectives enter the dominance test; optional objectives are reported but a
    candidate lacking one is still ranked on the rest.  (A group with no required objectives is
    ranked on the optional objectives that every ranked candidate carries.)"""
    rejected: dict[str, list[str]] = {}
    not_eval: dict[str, list[str]] = {}
    ok: list[CandidateEvaluation] = []
    status: dict[str, ValidationStatus] = {}
    for c in cands:
        st = [v.status for v in c.objectives.values()]
        status[c.candidate_id] = weakest_status(st) if st else ValidationStatus.UNVALIDATED
        if c.constraint_violations:
            rejected[c.candidate_id] = list(c.constraint_violations)
            continue
        reasons = []
        for o in objectives:
            v = c.objectives.get(o.name)
            if v is None:
                if o.required:
                    reasons.append(f"missing required objective {o.name!r}")
                continue
            if not v.finite():
                reasons.append(f"non-finite value for {o.name!r}")
            elif v.status == ValidationStatus.FAILED:
                reasons.append(f"{o.name!r} has validation status FAILED")
        if reasons:
            not_eval[c.candidate_id] = reasons
        else:
            ok.append(c)
    groups: dict[str, list[Objective]] = {}
    for o in objectives:
        groups.setdefault(o.group, []).append(o)

    def usable(objs: Sequence[Objective]) -> list[Objective]:
        req = [o for o in objs if o.required]
        if req:
            return req
        return [o for o in objs if all(o.name in c.objectives for c in ok)]

    tiers_overall = pareto_tiers(ok, usable(objectives), mode) if ok else {}
    tiers_group = {g: (pareto_tiers(ok, usable(objs), mode) if ok and usable(objs) else {}) for g, objs in groups.items()}
    for c in ok:  # status over ranked objectives only
        status[c.candidate_id] = weakest_status(c.objectives[o.name].status for o in objectives if o.name in c.objectives)
    return RankingResult(objectives=list(objectives), mode=mode, tiers_overall=tiers_overall, tiers_by_group=tiers_group,
                         rejected=rejected, not_evaluable=not_eval, status=status)


def count_wins(c: CandidateEvaluation, benchmarks: Mapping[str, float], objectives: Sequence[Objective],
               margin_rule: Literal["median", "range"] = "range") -> dict[str, bool]:
    """Per-objective 'beats benchmark' flags.  ``range``: the whole predicted range beats the benchmark
    value (margin exceeds the uncertainty); ``median``: the central value does."""
    out = {}
    for o in objectives:
        if o.name not in benchmarks or o.name not in c.objectives:
            continue
        v, b = c.objectives[o.name], benchmarks[o.name]
        if margin_rule == "median":
            out[o.name] = v.value > b if o.direction == "max" else v.value < b
        else:
            out[o.name] = v.lo() > b if o.direction == "max" else v.hi() < b
    return out
