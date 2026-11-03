"""Indentation runner and record builder (protocol kind "indentation").

``run_indentation(protocol, candidate, model, structure=None, context=None) -> ProtocolOutput`` is the
``RUNNERS["indentation"]`` entry of the JAX engine.  It builds the slab (structures/indentation.py), relaxes
and equilibrates it, indents it with the rigid SDF tip (physics/indenter.py) -- MD with
physics/indent_md.py (``protocol.mode == "md"``) or 0 K quasi-static FIRE steps (``"quasistatic"``) --,
analyses the run with physics/indent_analysis.py and returns

* one ``IndentationCurve`` (x = depth, y = 0.5 ps windowed load, channels: time, raw load, contact depth,
  penetration, work, temperatures, ...), and
* ``*_MD`` records: ``indentation_max_load_md``, ``indentation_contact_pressure_md`` (primary estimator E3
  over the plateau window at h_max, with the E3-E4 spread and the E3 estimator's regime-dependent systematic,
  :data:`E3_RADIUS_BIAS`), ``indentation_reduced_modulus_md`` (elastic
  loading fit), ``pop_in_*_md`` when a load drop occurs, and ``indentation_hardness_md_estimated_contact`` ONLY
  when Gate II (developed plasticity, with its onset depth from intermediate-depth evidence:
  ``IndentationRunData.gate_ii_evidence``, e.g. indent_analysis.gate_ii_evidence_from_gate_i) and Gate III
  (LM-R10 plateau on E3 and E4, window after that onset) both pass AND the B1 resolution of the estimated contact is
  signed off (:data:`HARDNESS_B1_SIGNED_OFF`, :func:`hardness_block_reason`); otherwise the
  deliverable is a contact pressure at h, never a hardness (SR-X01).  Oliver-Pharr stays a diagnostic in
  ``raw`` and is never a record.

Every record goes through ``records.make_record``: the potential card's domain (Tersoff 1989 / Erhart-Albe
are known_deficient for indentation -> OUT_OF_DOMAIN), the attached checks (FAILED if any fails) and the
status ceiling of the ``*_MD`` names (``literature_model_agreement``: an experimental comparison is kept as
evidence and never makes an MD hardness ``validated_against_experiment``).  Records built from an imported
LAMMPS run (:func:`records_from_lammps_run`) are further capped at ``numerically_verified``
(:data:`ENGINE_STATUS_CAP`).  G19 (tests/test_indentation_crossengine.py) verifies the LAMMPS SDF deck
against the harness in the quasi-static elastic regime (load agreement <= 1e-6 relative); LAMMPS MD
indentation is checked only statistically, so LAMMPS records stay capped at ``numerically_verified`` (the
``*_MD`` ceiling applies to both engines).  Evidence above the cap stays attached to the record as information.

Required checks (never silently omitted)
----------------------------------------
Every record carries the per-run checks of :func:`analysis_checks`: Newton 3 (the tip load against the
fixed-layer reaction), the load window and, for MD runs, the energy ledger (LAMMPS runs: also dangerous builds and
the CPU use, NOT APPLICABLE on a KOKKOS GPU run); on top of these the contact-pressure
record carries the plateau-window check and the hardness record the Gate II / Gate II onset / Gate III checks.  A
check that cannot be evaluated (no reaction or ledger channel, no load sample above the Newton-3 minimum load, fewer
than 3 frames in the plateau window, no Gate II evidence, ...) is emitted as an explicit NOT EVALUATED check
(:func:`not_evaluated_check`: ``passed`` False, ``metric`` None, ``details`` starting with "NOT EVALUATED"), so a
record that needs it is FAILED under the uniform status rule and never reaches ``numerically_verified`` on a check
that was not made.  Records that cannot be produced at all (no E3 contact area, no hardness without Gates II and
III, no elastic fit) are listed with their reasons and gate checks by :func:`records_not_available`
(``ProtocolOutput.raw["records_not_available"]`` and the IndentationCurve notes).

Production indentation records are built OUTSIDE ``agents.submit()``: production cells run in LAMMPS (the deck
generator ``engines.lammps_indent``) and are imported with :func:`records_from_lammps_run`; ``submit()`` runs the
JAX harness runner on small reference cells only (``agents.interface.INDENTATION_MAX_ATOMS_IN_PROCESS``).

Recipe (critic item A3): both paths record the sha256 of the frozen recipe the records are built under
(physics.indent_recipe, default ``indent_recipe.DEFAULT_RECIPE``) in ``EngineRunProvenance.recipe_sha256`` of the
curve and of every record, and its identity plus the protocol's deviations from it in ``raw["recipe"]``; an imported
run whose deck manifest names a recipe with a different protocol section fails the check
``indentation_recipe_protocol_matches_deck``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from fracture_md.engines.base import ProtocolOutput, RunContext
from fracture_md.physics import indent_recipe
from fracture_md.physics.indent_analysis import (
    ESTIMATOR_NAMES,
    EV_PER_A_TO_NN,
    PRIMARY_ESTIMATOR,
    AnalysisConfig,
    FrameData,
    IndentationRunData,
    analyze_indentation,
    evaluation_depth,
    json_ready,
)
from fracture_md.schema.common import (
    CheckResult,
    Conditions,
    ContactConditions,
    EngineRunProvenance,
    Provenance,
    Uncertainty,
    UncertaintyKind,
    ValidationEvidence,
    ValidationStatus,
    status_exceeds,
)
from fracture_md.schema.material import Candidate, Orientation
from fracture_md.schema.properties import (
    IndentationCurve,
    IndentationSegment,
    MethodSpec,
    PropertyName,
    PropertyRecord,
    PropertyScale,
    SampledCurve,
)
from fracture_md.structures.indentation import PARK_GAP_A, VACUUM_MARGIN_A, protocol_vacuum_A  # re-exported
from fracture_md.units import Unit, q

# Highest status of a record by the engine that produced its numbers (in addition to the name's ceiling).
# G19 (tests/test_indentation_crossengine.py) verifies the LAMMPS SDF deck against the harness in the quasi-static
# elastic regime (load agreement <= 1e-6 relative); LAMMPS MD indentation is checked only statistically, so LAMMPS
# records stay capped at numerically_verified (the *_MD ceiling applies to both engines).
ENGINE_STATUS_CAP: dict[str, ValidationStatus] = {"lammps": ValidationStatus.NUMERICALLY_VERIFIED}
_EVIDENCE_RUNG = {"experimental_reference": ValidationStatus.VALIDATED_AGAINST_EXPERIMENT,
                  "dft_reference": ValidationStatus.VALIDATED_AGAINST_DFT,
                  "literature_model_value": ValidationStatus.LITERATURE_MODEL_AGREEMENT,
                  "reference_implementation": ValidationStatus.REFERENCE_IMPLEMENTATION_AGREEMENT}
_SEGMENT_KIND = {"pre_hold": "approach", "approach": "approach", "load": "load", "hold": "hold", "unload": "unload",
                 "relax": "relax"}
NEWTON3_TOL = 0.02  # |<P + R>| / <P> (recommended protocol item 15)
# Quasi-static (0 K) runs: Newton 3 over the samples above this fraction of the maximum load.  There is no thermal
# load noise at 0 K (the MD minimum load of indent_analysis.newton3_balance, 20 nN, guards against it), and P + R is
# minus the residual force on the mobile atoms, bounded by the relaxation tolerance.
QS_NEWTON3_MIN_LOAD_FRACTION = 0.2
LEDGER_TOL = 1.0e-3  # |ledger| / W at h_max (item 12)
# Thermal floor of the energy ledger (item 12: "< 1e-3 W at h_max, with an absolute floor").  Velocity Verlet conserves
# a shadow energy that differs from E by O(dt^2) per atom, proportional to k_B T: |L(h_max)| may reach
# LEDGER_FLOOR_EV_PER_ATOM x N x (T / 300 K) x (dt / 1 fs)^2 whatever W is.  3e-5 eV/atom at 300 K and 1 fs: LAMMPS SDF
# deck (tests/test_lammps_indent.py, the same constant, ~4.6k atoms) max |L| 0.8-1.7e-5 eV/atom; harness 576-atom
# Vashishta test cell (tests/test_indentation_md.py, critic probe, 3 seeds x 2 baths) |L(h_max)| 1.7-2.5e-5 eV/atom =
# 0.68-1.01e-3 W, 4x smaller at dt/2.  The floor is zero at 0 K, where the 1e-3 W criterion applies as it stands.
LEDGER_FLOOR_EV_PER_ATOM = 3.0e-5
# Review T7: the ledger's thermal error is a sum of N weakly correlated per-atom terms and grows like sqrt(N), not N.
# Reduced S1 (r4-s1-deck, 300 K, 1 fs; outputs/ready/r4-fixer-0/ledger_probe.json): SDF deck 34,848 atoms |L(h_max)|
# 0.094 eV (max |L| over loading 0.133 eV = 3.8e-6 eV/atom), atomistic deck 36,864 sample atoms 0.128 eV (0.179 eV),
# i.e. 5-9e-4 eV x sqrt(N), against 4-6e-4 eV x sqrt(N) on the 576-atom cell and up to 1.1e-3 on the ~4.6k-atom deck.
# An N-linear floor was 4.4x 1e-3 W on the reduced S1 (a 7x ledger regression passed) and ~2.6x at S1 size.  The
# floor is therefore linear up to the largest calibration cell, LEDGER_FLOOR_LINEAR_MAX_ATOMS = N_c, and grows like
# sqrt(N N_c) beyond it (continuous at N_c): 0.40 eV on the reduced S1 (3-4x its |L(h_max)|, 1.7x / 1.0x 1e-3 W, so
# a 5x ledger fails), 4.2 eV at 3.84M atoms (~0.1x 1e-3 W there: the 1e-3 W criterion governs).
LEDGER_FLOOR_LINEAR_MAX_ATOMS = 5000
NOT_EVALUATED = "NOT EVALUATED"  # details prefix of a required check that could not be evaluated
NOT_APPLICABLE = "NOT APPLICABLE"  # details prefix of a check whose rule does not apply to the run (passed)
EQUILIBRATION_T_TOL = 0.05  # |<T> / T_target - 1| of the all-atom Langevin stage (at least 3 canonical sigma)
# The preparation (recipe_v0 protocol/preparation; bound by physics.indent_recipe.code_bindings): Langevin on every
# mobile atom, velocities drawn at PREP_START_TEMPERATURE_FACTOR x T, for PREP_LANGEVIN_FRACTION of equilibration_ps,
# then the bath alone.  No z pre-expansion (recipe_v0 known deviation: harmless below ~15 nm of slab).
PREP_LANGEVIN_FRACTION = 0.6
PREP_START_TEMPERATURE_FACTOR = 2.0
MIN_CPU_PERCENT = 97.0  # LAMMPS segments (item 2)
# Review C3: the >= 97 % CPU-use rule judges CPU MPI runs (a shared core slows an MPI run 2-3x, G07).  On the KOKKOS
# GPU backend the host process mostly waits on the GPU and LAMMPS reports a low '% CPU use' whatever the core sharing
# (S1 atomistic deck, 3.74M atoms, dedicated host core: 8-82 %; other KOKKOS runs 7-33 %), so the rule does not apply
# to GPU runs; examples/indent/s1/run_s1.sh checks at launch that the GPU is otherwise idle.
KOKKOS_GPU_ACCELERATOR = "kokkos/gpu"
# Critic item B1 (r4-contact request C2): E3 is an ESTIMATED contact with a stated systematic (MD_ASSUMPTIONS), not a
# true contact.  No hardness record is written until a reviewer signs off the B1 resolution; the hardness is then
# listed by records_not_available with this reason.  After the sign-off a hardness also needs the force-fraction rule
# at every analysed frame (hardness_block_reason).  Signing off is a one-line edit here, recorded in docs/INDENTATION.md.
HARDNESS_B1_SIGNED_OFF = False
B1_NOT_SIGNED_OFF = ("B1: the E3 contact is an ESTIMATED contact with a stated systematic; a hardness record needs "
                     "reviewer sign-off of the B1 resolution")
# Review P2: the E3 estimator's own radius bias a_E3 / a_true - 1 (force fraction 1/2, dilated hull) on EXACT
# continuum contact-pressure fields, cell-integrated on the (001) surface lattice (cell force = per-atom tip force) and
# averaged over axis offsets, against the field's own contact area, by contact regime (lowest, highest; contact radius
# a = 10-50 A).  B1 calibrated only the axisymmetric fields; a Berkovich contact in a linear-elastic solid is a
# pincushion (it sinks in at the face centres) and the convex hull bridges its concave sides.  Sources:
# outputs/indent_derisk/r4-contact/b1_synthetic_bias.json (Hertz, flat disc), outputs/ready/review/physics/ and
# review/verify/P2/ (Sneddon cone, box BEM sharp and R = 10 A Berkovich, N 256-1024, L 160-320 A), and
# outputs/ready/r4-fixer-0/e3_bias_probe*.{py,json} (box BEM with the Vashishta 300 K C_ij, half-space, lab x = [110];
# uniform-pressure disc and nominal triangle); tests/test_indentation_e3_systematic.py re-derives every row on the box
# BEM at N = 256.  The regime of a frame is not identified, so a record carries the whole range
# (:func:`e3_systematic_uncertainty`).
E3_RADIUS_BIAS: dict[str, tuple[float, float]] = {
    "hertz": (-0.05, 0.0),  # a = 10-50 A: -1 to -2 % at 10-12 A, -3 to -4 % at 15-50 A
    "flat_disc": (0.01, 0.07),  # saturated / plastic axisymmetric contact: +6 % at a = 10 A, +2 % at 30 A, +1 % at 50 A
    "flat_berkovich_triangle": (0.02, 0.12),  # plastic, no sink-in: +11 % at a = 10 A, +3-4 % at 30 A, +2 % at 50 A
    "elastic_cone": (-0.12, -0.06),  # Sneddon p ~ arccosh(a / r): -6 % at a = 12 A, -9 % at 50 A; a >> a/sqrt 2:
    #                                  1/cosh(1/2) - 1 = -11.3 % (the kept cells are those with arccosh(a / r) >= 1/2)
    "elastic_berkovich_R10": (0.04, 0.18),  # box BEM, R = 10 A: +5-6 / +14-15 / +17-18 % at a = 11 / 21 / 31 A (-> sharp)
    "elastic_berkovich_sharp": (0.21, 0.28),  # box BEM: +22 to +27 % at a = 14-40 A (area x1.5-1.6)
}
E3_RADIUS_BIAS_RANGE: tuple[float, float] = (min(v[0] for v in E3_RADIUS_BIAS.values()),
                                             max(v[1] for v in E3_RADIUS_BIAS.values()))
_E3_REGIME_TEXT = {
    "hertz": "Hertz", "flat_disc": "flat disc",
    "flat_berkovich_triangle": "flat nominal Berkovich triangle, no sink-in",
    "elastic_cone": "elastic cone", "elastic_berkovich_R10": "elastic R = 10 A Berkovich",
    "elastic_berkovich_sharp": "elastic sharp Berkovich: the pincushion contact sinks in at the face centres and the "
                               "convex hull bridges it, area x1.5-1.6, p_m(E3) ~37 % low",
}


def _pct(x: float) -> str:
    return "0" if round(100.0 * x) == 0 else f"{100.0 * x:+.0f}"


def e3_bias_statement() -> str:
    """The E3 systematic as MD_ASSUMPTIONS states it, generated from :data:`E3_RADIUS_BIAS` (review P2)."""
    lo, hi = E3_RADIUS_BIAS_RANGE
    per = ", ".join(f"{_pct(a)} to {_pct(b)} % ({_E3_REGIME_TEXT[k]})" for k, (a, b) in E3_RADIUS_BIAS.items())
    return (f"Its own radius bias on exact continuum contact fields depends on the contact regime "
            f"(physics.indentation.E3_RADIUS_BIAS): {per}. Overall {_pct(lo)} % to {_pct(hi)} % in radius: p_m on the "
            f"true contact is {(1 + lo) ** 2:.2f}-{(1 + hi) ** 2:.2f}x p_m(E3), carried as a systematic uncertainty of "
            "the record and not corrected; the bias changes through the elastic-to-plastic transition, so the slope of "
            "an E3 plateau is not free of it")


MD_ASSUMPTIONS = (
    "model-specific MD value: nm depth, 1e8-1e10 1/s indentation strain rate, defect-free periodic cell, rigid "
    "frictionless tip, empirical potential; not a prediction of a laboratory hardness",
    "contact area from the primary estimator E3: hull, dilated by a/(2 sqrt 2), of the ESTIMATED-contact atoms (the "
    "atoms touching the tip with >= 1/2 of their mean tip force; indent_analysis.estimated_contact_mask), not the true "
    "contact. Calibrated on 0 K Vashishta sphere contacts (B1): its radius is within 5 % of the elastic-continuum "
    "contact radius where the model's contact is linear-elastic, and up to 10 % larger where the model's contact "
    "pressure saturates (flat-topped at ~30-48 GPa). " + e3_bias_statement() + ". On the 1 nm atomistic Berkovich "
    "(G06) the estimated contact is, like for like (E3 on the elastic box-BEM field of the same cell at the same "
    "depth), 1.56x the elastic one and its p_m 0.63x (against that field's exact area it is 2.45x, which contains the "
    "1.56x hull bias): a fuller, pressure-limited contact; E3 p_m is therefore a model-specific contact pressure",
    "single trajectory unless stated: seed-to-seed scatter not sampled",
)


# ============================================================================================ curve
def _segments(labels: Sequence[str]) -> list[IndentationSegment]:
    out: list[IndentationSegment] = []
    lab = [_SEGMENT_KIND.get(str(x), None) for x in labels]
    start = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[start]:
            if lab[start] is not None:
                if out and out[-1].kind == lab[start] and out[-1].stop == start:
                    out[-1] = IndentationSegment(kind=out[-1].kind, start=out[-1].start, stop=i)
                else:
                    out.append(IndentationSegment(kind=lab[start], start=start, stop=i))
            start = i
    return out


def indentation_curve(run: IndentationRunData, analysis: Mapping[str, Any], *, mode: str, contact: ContactConditions,
                      temperature_K: float, potential_id: str, engine_run: EngineRunProvenance,
                      orientation: Optional[Orientation] = None, seed: Optional[int] = None, curve_id: str = "",
                      checks: Sequence[CheckResult] = (), assumptions: Sequence[str] = (), notes: str = "",
                      max_samples: int = 20000) -> IndentationCurve:
    """The IndentationCurve of a run: x = depth h (A), y = the windowed load (nN, window from the analysis),
    channels time (ps), load_raw, contact_depth / penetration (SDF tips), fixed_layer_reaction, lateral loads,
    tip work, temperatures.  Longer logs are decimated evenly (every k-th sample) to ``max_samples``."""
    from fracture_md.physics.indent_analysis import running_mean

    n = len(run.t_ps)
    k = max(1, int(math.ceil(n / max_samples)))
    sl = slice(None, None, k)
    Pw = running_mean(run.load_nN, analysis["load"]["window"]["n_samples"])
    ex: dict[str, Any] = {"load_raw": q(np.asarray(run.load_nN)[sl].tolist(), Unit.NANONEWTON)}
    if mode == "md":
        ex["time"] = q(np.asarray(run.t_ps)[sl].tolist(), Unit.PICOSECOND)
    ch = run.channels
    if "pen_fw_A" in ch:
        ex["penetration"] = q(np.asarray(ch["pen_fw_A"])[sl].tolist(), Unit.ANGSTROM)
        ex["contact_depth"] = q((np.asarray(run.h_A) - np.asarray(ch["pen_fw_A"]))[sl].tolist(), Unit.ANGSTROM)
    for key, name, unit in (("reaction_nN", "fixed_layer_reaction", Unit.NANONEWTON),
                            ("lateral_load_x_nN", "lateral_load_x", Unit.NANONEWTON),
                            ("lateral_load_y_nN", "lateral_load_y", Unit.NANONEWTON),
                            ("W_eV", "tip_work", Unit.EV), ("Q_eV", "thermostat_heat", Unit.EV),
                            ("T_core_K", "temperature_core", Unit.KELVIN), ("n_contact", "n_contact", Unit.DIMENSIONLESS)):
        if key in ch:
            ex[name] = q(np.asarray(ch[key], float)[sl].tolist(), unit)
    labels = np.asarray(run.segment)[sl]
    curve = SampledCurve(x_name="depth", x=q(np.asarray(run.h_A)[sl].tolist(), Unit.ANGSTROM), y_name="load",
                         y=q(Pw[sl].tolist(), Unit.NANONEWTON), extra=ex)
    return IndentationCurve(
        curve_id=curve_id or f"{potential_id}:{engine_run.engine}:{contact.indenter_shape}:{time.strftime('%Y%m%dT%H%M%S')}",
        mode=mode, contact=contact, temperature=q(float(temperature_K), Unit.KELVIN), curve=curve,
        segments=_segments(labels), orientation=orientation, potential_id=potential_id, engine_run=engine_run,
        seed=seed, checks=list(checks),
        assumptions=list(assumptions) + [f"y = load averaged over a centred {analysis['load']['window']['window_ps']:g} ps "
                                         "window; load_raw = instantaneous"],
        notes=notes)


# ============================================================================================ checks
def not_evaluated_check(name: str, reason: str, *, tolerance: Optional[float] = None,
                        unit: Optional[Unit] = None) -> CheckResult:
    """A required check that could not be evaluated: ``passed`` False (a record that needs it is FAILED under the
    uniform status rule, never ``numerically_verified``), ``metric`` None, ``details`` "NOT EVALUATED: <reason>"."""
    return CheckResult(name=name, passed=False, metric=None, tolerance=tolerance, unit=unit,
                       details=f"{NOT_EVALUATED}: {reason}")


def is_not_evaluated(check: CheckResult) -> bool:
    """True for a check made by :func:`not_evaluated_check`."""
    return (not check.passed) and check.metric is None and check.details.startswith(NOT_EVALUATED)


def _newton3_min_load_nN() -> float:
    """The minimum load of indent_analysis.newton3_balance as analyze_indentation calls it (its default)."""
    import inspect

    from fracture_md.physics.indent_analysis import newton3_balance

    return float(inspect.signature(newton3_balance).parameters["min_load_nN"].default)


def ledger_floor_atoms(n_atoms: float) -> float:
    """The effective atom count of the ledger floor: N up to :data:`LEDGER_FLOOR_LINEAR_MAX_ATOMS` (N_c), sqrt(N N_c)
    above it (the thermal error grows like sqrt(N); review T7)."""
    n = max(float(n_atoms), 0.0)
    nc = float(LEDGER_FLOOR_LINEAR_MAX_ATOMS)
    return n if n <= nc else math.sqrt(n * nc)


def ledger_floor_eV(n_atoms: int, temperature_K: float, dt_fs: float) -> float:
    """Thermal floor of the energy ledger: f N_eff (T / 300 K) (dt / 1 fs)^2 eV, f = :data:`LEDGER_FLOOR_EV_PER_ATOM`
    and N_eff = :func:`ledger_floor_atoms` (N up to :data:`LEDGER_FLOOR_LINEAR_MAX_ATOMS`, sqrt(N N_c) above).
    ``n_atoms``: the atoms the ledger sums over (for a LAMMPS deck the sample, :func:`lammps_ledger_atoms`)."""
    return (LEDGER_FLOOR_EV_PER_ATOM * ledger_floor_atoms(n_atoms) * max(float(temperature_K), 0.0) / 300.0
            * (float(dt_fs) / 1.0) ** 2)


def ledger_at_hmax(h_A: Any, ledger_eV: Any, W_eV: Any, *, segment: Optional[Any] = None,
                   floor_eV: float = 0.0) -> dict[str, float]:
    """The energy ledger at the deepest sample of the loading branch (segments approach / load when given), the
    input of the ledger check of :func:`analysis_checks`: ledger_over_W_at_hmax = |L| / |W| there, ledger_at_hmax_eV,
    W_at_hmax_eV, h_max_A, floor_eV and max_abs_ledger_eV over the run."""
    h = np.asarray(h_A, float)
    L = np.asarray(ledger_eV, float)
    W = np.asarray(W_eV, float)
    sel = np.ones(len(h), bool) if segment is None else np.isin(np.asarray(segment).astype(str), ("approach", "load"))
    if not sel.any():
        sel = np.ones(len(h), bool)
    i = int(np.flatnonzero(sel)[np.argmax(h[sel])])
    return {"ledger_over_W_at_hmax": abs(float(L[i])) / max(abs(float(W[i])), 1e-30), "ledger_at_hmax_eV": float(L[i]),
            "W_at_hmax_eV": float(W[i]), "h_max_A": float(h[i]), "floor_eV": float(floor_eV),
            "max_abs_ledger_eV": float(np.max(np.abs(L))) if L.size else 0.0}


def analysis_checks(analysis: Mapping[str, Any], *, engine: str, ledger: Optional[Mapping[str, float]] = None,
                    lammps: Optional[Mapping[str, Any]] = None, mode: str = "md") -> list[CheckResult]:
    """Per-run numerical checks (recommended protocol items 12, 15, 22) as CheckResults: Newton-3 of the load,
    load window >= 5 tau_ac, defect extent (fixed layer and periodic image), energy ledger (MD runs of either
    engine), dangerous neighbour builds and CPU use (LAMMPS).

    Newton 3 and (``mode`` "md") the ledger are required: when they cannot be evaluated (no reaction channel, no load
    sample above the Newton-3 minimum load, no ledger logged) they are emitted as NOT EVALUATED checks
    (:func:`not_evaluated_check`), never omitted.  ``ledger``: :func:`ledger_at_hmax` (plus optional summary keys).
    A quasi-static run (``mode`` "quasistatic") has no integrator and no ledger check.  ``lammps``: the run's
    provenance dict (io.lammps_indent.LammpsIndentRun.provenance_dict) with its ``accelerator``
    (:func:`lammps_accelerator`); on a KOKKOS GPU run the CPU-use rule does not apply and ``lammps_cpu_use`` is a
    passed NOT APPLICABLE check stating the reported value (review C3)."""
    out: list[CheckResult] = []
    n3 = analysis["load"].get("newton3")
    n3_name, n3_what = "indentation_newton3", "|<P_tip + R_fixed>| / <P_tip> over the samples in contact"
    if n3 is None:
        out.append(not_evaluated_check(n3_name, "the run logs no fixed-layer reaction (reaction_nN channel)",
                                       tolerance=NEWTON3_TOL))
    else:
        min_load = float(n3.get("min_load_nN", _newton3_min_load_nN()))
        v = n3.get("mean_residual_over_mean_P", float("nan"))
        if n3.get("n", 0) <= 0 or v is None or not np.isfinite(v):
            pmax = float(analysis.get("P_at_hmax_windowed_nN", float("nan")))
            out.append(not_evaluated_check(
                n3_name, f"no load sample above the Newton-3 minimum load {min_load:g} nN (windowed load at h_max "
                         f"{pmax:.3g} nN): the run is too small for the check", tolerance=NEWTON3_TOL))
        else:
            v = abs(float(v))
            out.append(CheckResult(name=n3_name, passed=bool(v < NEWTON3_TOL), metric=float(v), tolerance=NEWTON3_TOL,
                                   details=f"{n3_what} (P > {min_load:.3g} nN, {int(n3['n'])} samples)"))
    w = analysis["load"]["window"]
    out.append(CheckResult(name="indentation_load_window", passed=bool(w["ok"]), metric=float(w["window_over_tau"]),
                           tolerance=5.0, details=f"window {w['window_ps']:g} ps over tau_ac {w['tau_ac_ps']:.3g} ps (>= 5)"))
    for name, ext in (analysis.get("defect_extent") or {}).items():
        chk = ext.get("check")
        if chk is not None and ext.get("n_flagged", 0) > 0:
            out.append(CheckResult(name=f"indentation_defect_extent_{name}", passed=bool(chk["passes"]),
                                   metric=float(chk["image_gap_A"]), tolerance=float(chk["required_image_gap_A"]),
                                   details=f"defect cluster clearance to the fixed layer {chk['clearance_to_fixed_layer_A']} A "
                                           f"(>= {chk['required_fixed_A']:.1f}); gap to its periodic image "
                                           f"{chk['image_gap_A']:.1f} A (>= L/3)"))
    if mode not in ("md", "quasistatic"):
        raise ValueError(f"mode must be 'md' or 'quasistatic', got {mode!r}")
    if mode == "md" and ledger is None:
        out.append(not_evaluated_check("indentation_energy_ledger", "the run logs no energy ledger (KE + E_mat + "
                                       "E_ind - W - Q - E0) and tip work", tolerance=LEDGER_TOL))
    elif ledger is not None:
        r = float(ledger.get("ledger_over_W_at_hmax", ledger.get("max_abs_ledger_over_max_abs_W", float("nan"))))
        floor = float(ledger.get("floor_eV", 0.0))
        L_h = abs(float(ledger.get("ledger_at_hmax_eV", np.inf)))
        ok = r < LEDGER_TOL or L_h <= floor
        out.append(CheckResult(name="indentation_energy_ledger", passed=bool(ok), metric=r, tolerance=LEDGER_TOL,
                               details=f"|KE + E_mat + E_ind - W - Q - E0| / W at h_max = {r:.3g} (|L| = {L_h:.3g} eV), "
                                       f"or |L| below the thermal floor {floor:.3g} eV (LEDGER_FLOOR_EV_PER_ATOM x N_eff x "
                                       f"T/300 K x (dt/fs)^2, N_eff = N up to {LEDGER_FLOOR_LINEAR_MAX_ATOMS} atoms, "
                                       f"sqrt(N x {LEDGER_FLOOR_LINEAR_MAX_ATOMS}) above)"))
    if lammps is not None:
        db = lammps.get("dangerous_builds") or []
        out.append(CheckResult(name="lammps_dangerous_builds", passed=bool(sum(db) == 0), metric=float(sum(db)),
                               tolerance=0.0, details="sum of 'Dangerous builds' over the runs of the log"))
        seg = lammps.get("cpu_use_check") or {}
        cpu = lammps.get("cpu_use_percent") or []
        acc = str(lammps.get("accelerator") or "cpu")
        if not lammps_cpu_check_applies(acc):  # review C3: a KOKKOS GPU run (lammps_accelerator)
            judged = seg.get("min_judged_percent")
            rep = judged if judged is not None else (min(cpu) if cpu else None)
            out.append(CheckResult(
                name="lammps_cpu_use", passed=True, metric=None if rep is None else float(rep), tolerance=None,
                details=f"{NOT_APPLICABLE}: {acc} run; the host process waits on the GPU, so LAMMPS's '% CPU use' "
                        f"({'n/a' if rep is None else f'{float(rep):.3g} %'} at the lowest judged segment) does not "
                        f"measure core sharing; the >= {MIN_CPU_PERCENT:g} % rule judges CPU MPI runs (the GPU is "
                        "checked idle at launch by examples/indent/s1/run_s1.sh)"))
        elif seg.get("n_judged"):  # the recipe's rule (r4-recipe R2): every run segment of >= 100 steps and >= 1 s
            out.append(CheckResult(name="lammps_cpu_use", passed=bool(seg["passes"]),
                                   metric=float(seg["min_judged_percent"]), tolerance=MIN_CPU_PERCENT,
                                   details=f"per run segment of >= 100 steps and >= 1 s: {seg['reason']}"))
        elif cpu:  # logs without 'Loop time' segments (synthetic test logs): the minimum over the runs
            out.append(CheckResult(name="lammps_cpu_use", passed=bool(min(cpu) >= MIN_CPU_PERCENT), metric=float(min(cpu)),
                                   tolerance=MIN_CPU_PERCENT, details="minimum '% CPU use' over the runs (shared cores slow MPI)"))
    return out


# ============================================================================================ records
def _cap_evidence(evidence: Sequence[ValidationEvidence], engine: str) -> tuple[list[ValidationEvidence], list[ValidationEvidence]]:
    cap = ENGINE_STATUS_CAP.get(engine)
    if cap is None:
        return list(evidence), []
    keep, drop = [], []
    for e in evidence:
        rung = _EVIDENCE_RUNG.get(e.kind)
        (drop if e.passed and rung is not None and status_exceeds(rung, cap) else keep).append(e)
    return keep, drop


def make_md_record(*, name: PropertyName, value: float, unit: Unit, candidate: Candidate, card: Any, method: MethodSpec,
                   conditions: Conditions, provenance: Provenance, orientation: Optional[Orientation] = None,
                   checks: Sequence[CheckResult] = (), evidence: Sequence[ValidationEvidence] = (),
                   uncertainty: Sequence[Uncertainty] = (), notes: str = "", scale: PropertyScale = PropertyScale.ATOMISTIC_IDEAL) -> PropertyRecord:
    """make_record for an ``*_MD`` indentation value with the engine cap applied (module docstring)."""
    from fracture_md.records import make_record

    engine = provenance.engine_run.engine if provenance.engine_run is not None else "jax"
    keep, dropped = _cap_evidence(evidence, engine)
    rec = make_record(name=name, value=float(value), unit=unit, candidate=candidate, method=method, card=card,
                      conditions=conditions, orientation=orientation, checks=checks, evidence=keep,
                      uncertainty=uncertainty, notes=notes, provenance=provenance, scale=scale)
    if dropped:
        v = rec.validation
        rec = rec.model_copy(update={"validation": v.model_copy(update={
            "evidence": list(v.evidence) + dropped,
            "notes": (v.notes + " " if v.notes else "") + (
                f"Status capped at {ENGINE_STATUS_CAP[engine].value} for {engine} runs (the cross-engine "
                "indentation check G19 covers only the quasi-static elastic regime): attached "
                f"{', '.join(sorted({e.kind for e in dropped}))} evidence is information only.")})})
    cap = ENGINE_STATUS_CAP.get(engine)
    if cap is not None and status_exceeds(rec.validation.status, cap):  # pragma: no cover - guarded above
        raise AssertionError(f"{name.value} from {engine} exceeds {cap.value}")
    return rec


MIN_PLATEAU_FRAMES = 3  # contact frames in the plateau window for a window-averaged p_m (indent_analysis.plateau_metric)


def _finite(x: Any) -> bool:
    """A finite number (json_ready analyses carry None for NaN)."""
    try:
        return x is not None and bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def plateau_window_check(analysis: Mapping[str, Any]) -> CheckResult:
    """The contact-pressure record's required check: p_m (E3) is a mean over >= 3 contact frames of the LM-R10 window
    at the evaluation depth (metric = the number of frames).  With fewer frames the record falls back to the deepest
    loading frame and the check is NOT EVALUATED.  The LM-R10 slope metric itself gates the hardness (Gate III), not
    the contact pressure."""
    h_eval = evaluation_depth(float(analysis["h_max_A"]))
    plat = (analysis.get("plateau_LM_R10") or {}).get(f"h{h_eval:g}", {})
    e3 = plat.get(PRIMARY_ESTIMATOR, {})
    win = e3.get("h_window_A", [float("nan"), h_eval])
    if "pm_mean_GPa" in e3:
        return CheckResult(name="indentation_plateau_window", passed=True, metric=float(e3["n"]),
                           tolerance=float(MIN_PLATEAU_FRAMES),
                           details=f"p_m (E3) averaged over {int(e3['n'])} contact frames in h in [{win[0]:.3g}, "
                                   f"{win[1]:.3g}] A (>= {MIN_PLATEAU_FRAMES}); LM-R10 metric {e3['metric']:.3f}")
    return not_evaluated_check("indentation_plateau_window",
                               f"{int(e3.get('n', 0))} contact frame(s) with an E3 area in the plateau window h in "
                               f"[{win[0]:.3g}, {win[1]:.3g}] A (>= {MIN_PLATEAU_FRAMES} needed)",
                               tolerance=float(MIN_PLATEAU_FRAMES))


def gate_checks(analysis: Mapping[str, Any]) -> list[CheckResult]:
    """Gate II (developed plasticity at h_max), its onset depth (intermediate-depth evidence) and Gate III (LM-R10
    plateau on E3 and E4 after the onset) as CheckResults: the required checks of the hardness record, which exists
    only when all three pass.  A gate whose inputs are missing is NOT EVALUATED (an unload-less run has no residual
    impression; without Gate II evidence the onset is unknown and the Gate III window cannot be placed)."""
    from fracture_md.physics.indent_analysis import NO_GATE_II_EVIDENCE

    g2 = analysis.get("gate_ii") or {}
    crit = g2.get("criteria") or {}
    prov = " (PROVISIONAL thresholds)" if g2.get("provisional") else ""
    missing = [k for k, c in crit.items() if c.get("value", c.get("value_A")) is None]
    out: list[CheckResult] = []
    if not crit:
        out.append(not_evaluated_check("indentation_gate_ii", "no Gate II verdict in the analysis"))
    elif missing:
        out.append(not_evaluated_check("indentation_gate_ii", f"no {', '.join(missing)} (needs the unloading branch "
                                                              "and the final relaxed state)"))
    else:
        out.append(CheckResult(name="indentation_gate_ii", passed=bool(g2.get("passes")),
                               details="developed plasticity at h_max: " + "; ".join(
                                   f"{k} {c.get('value', c.get('value_A')):.3g} >= "
                                   f"{c.get('threshold', c.get('threshold_A')):.3g}: {'pass' if c['passed'] else 'fail'}"
                                   for k, c in crit.items()) + prov))
    onset = g2.get("onset") or {}
    if not g2.get("passes"):
        out.append(not_evaluated_check("indentation_gate_ii_onset", "Gate II does not pass at h_max"))
    elif onset.get("h_onset_A") is not None:
        out.append(CheckResult(name="indentation_gate_ii_onset", passed=True, metric=float(onset["h_onset_A"]),
                               unit=Unit.ANGSTROM, details=str(onset.get("reason", ""))))
    elif int(onset.get("n_evidence", 0)) == 0:
        out.append(not_evaluated_check("indentation_gate_ii_onset", NO_GATE_II_EVIDENCE))
    elif onset.get("not_evaluated"):  # critic N1 / N2: a not-evaluated Gate I quench decides the onset
        out.append(not_evaluated_check("indentation_gate_ii_onset",
                                       str(onset.get("reason", "")).removeprefix(f"{NOT_EVALUATED}: ")))
    else:
        out.append(CheckResult(name="indentation_gate_ii_onset", passed=False, details=str(onset.get("reason", ""))))
    g3 = analysis.get("gate_iii")
    if not g3:
        out.append(not_evaluated_check("indentation_gate_iii", "no loading frames with contact areas"))
    elif onset.get("h_onset_A") is None:
        out.append(not_evaluated_check("indentation_gate_iii", "the window must start after the Gate II onset, which "
                                                               "is unknown"))
    else:
        per = g3.get("estimators", {})
        metrics = [float(v["metric"]) for v in per.values() if "metric" in v]
        if len(metrics) < len(per) or not per:
            out.append(not_evaluated_check("indentation_gate_iii", "fewer than 3 frames in the plateau window of "
                                                                   f"{', '.join(per) or 'E3/E4'}", tolerance=g3.get("tol")))
        else:
            per_txt = ", ".join("%s %.3f" % (e, float(v["metric"])) for e, v in per.items())
            before = all(v.get("onset_before_window") for v in per.values())
            out.append(CheckResult(name="indentation_gate_iii", passed=bool(g3.get("passes")), metric=max(metrics),
                                   tolerance=float(g3["tol"]),
                                   details=f"LM-R10 metric on {per_txt} at h_eval {float(g3['h_eval_A']):.3g} A; "
                                           f"onset before the window: {before}"))
    return out


def hardness_block_reason(analysis: Mapping[str, Any]) -> Optional[str]:
    """Why a hardness record may not be written even when Gates II and III pass (critic item B1), or None.

    Without the B1 sign-off (:data:`HARDNESS_B1_SIGNED_OFF`) no hardness is written.  After it, every analysed frame
    must have used the force-fraction rule (``analysis["contact"]["rules_used"] == ["force_fraction"]``): an atomistic
    tip without a per-atom tip force keeps every touching atom, the over-counting set B1 rejected."""
    if not HARDNESS_B1_SIGNED_OFF:
        return B1_NOT_SIGNED_OFF
    used = list((analysis.get("contact") or {}).get("rules_used") or [])
    if used != ["force_fraction"]:
        return (f"the estimated contact used the rule(s) {used or 'none'}, not the force-fraction rule at every frame "
                "(an atomistic tip needs a per-atom tip force): no hardness on the touching-atom set (B1)")
    return None


def contact_rule_note(analysis: Mapping[str, Any]) -> str:
    """The estimated-contact definition of an analysis (``analysis["contact"]``) for the record notes (B1, C2.3)."""
    c = analysis.get("contact") or {}
    if not c:
        return ""
    return (f"Estimated contact (E3-E5): rule {c.get('rule')}, force fraction {c.get('force_fraction')}, rules used "
            f"{', '.join(map(str, c.get('rules_used') or [])) or 'none'}.")


def records_not_available(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ``*_MD`` records an analysis cannot produce, each with its reason and, where a gate decides, the gate
    checks (JSON-ready): the contact pressure when E3 is undefined at every usable frame (fewer than 3 contact atoms
    for the dilated hull, or no frames), the hardness without Gates II and III or before the B1 sign-off
    (:func:`hardness_block_reason`; ``gates_reason`` holds the gate part alone), the reduced modulus without an
    elastic fit, the pop-in records without a load drop.  Runners put the list in ``raw["records_not_available"]``
    and in the IndentationCurve notes, so a missing record is never silent."""
    out: list[dict[str, Any]] = []
    h_eval = evaluation_depth(float(analysis["h_max_A"]))
    e3 = ((analysis.get("plateau_LM_R10") or {}).get(f"h{h_eval:g}", {})).get(PRIMARY_ESTIMATOR, {})
    fr = analysis.get("deepest_loading_frame")
    if "pm_mean_GPa" not in e3 and not (fr and _finite(fr.get("pm_E3"))):
        if not fr:
            why = "no loading frame with contact areas (no frames, or none in contact)"
        else:
            n_c = fr.get("n_contact", fr.get("n_touching"))
            why = (f"the primary contact area E3 (dilated hull of the contact atoms) is undefined at the deepest loading "
                   f"frame (h = {float(fr['h_A']):.3g} A, {n_c} contact atom(s); the hull needs >= 3) and the plateau "
                   f"window has {int(e3.get('n', 0))} frame(s) with an E3 area")
        out.append({"name": PropertyName.INDENTATION_CONTACT_PRESSURE_MD.value, "reason": why,
                    "checks": [not_evaluated_check("indentation_contact_area_e3", why).model_dump(mode="json")]})
    g3 = analysis.get("gate_iii") or {}
    blocked = hardness_block_reason(analysis)
    if not ((analysis.get("gate_ii") or {}).get("passes") and g3.get("passes")):
        hv = analysis.get("hardness") or {}
        why = str(hv.get("reason", "Gates II and III did not both pass"))
        out.append({"name": PropertyName.INDENTATION_HARDNESS_MD_ESTIMATED_CONTACT.value,
                    "reason": why if blocked is None else f"{why}; {blocked}", "gates_reason": why,
                    "checks": [c.model_dump(mode="json") for c in gate_checks(analysis)]})
    elif blocked is not None:
        out.append({"name": PropertyName.INDENTATION_HARDNESS_MD_ESTIMATED_CONTACT.value,
                    "reason": f"Gates II and III passed; {blocked}", "gates_reason": "",
                    "checks": [c.model_dump(mode="json") for c in gate_checks(analysis)]})
    fit = (analysis.get("depth_zero") or {}).get("elastic_fit") or {}
    if "M_GPa" not in fit:
        out.append({"name": PropertyName.INDENTATION_REDUCED_MODULUS_MD.value,
                    "reason": f"no elastic loading fit: {fit.get('reason', 'too few loading samples in the fit window')}"})
    if not (analysis.get("pop_in") or {}).get("load_drop_events"):
        for nm in (PropertyName.POP_IN_DEPTH_MD, PropertyName.POP_IN_LOAD_MD):
            out.append({"name": nm.value, "reason": "no windowed load drop > 5 % on the loading branch"})
    return out


def _not_available_note(gaps: Sequence[Mapping[str, Any]]) -> str:
    return ("Records not produced: " + "; ".join(f"{g['name']} ({g['reason']})" for g in gaps) + ".") if gaps else ""


def _hardness_note(analysis: Mapping[str, Any]) -> str:
    onset = (analysis.get("gate_ii") or {}).get("onset") or {}
    h_on = onset.get("h_onset_A")
    at = f"; onset {h_on:.3g} A from {onset.get('source', 'unspecified')}" if h_on is not None else ""
    return (f"Gates II (developed plasticity, PROVISIONAL thresholds{at}) and III (LM-R10 plateau on E3 and E4, window "
            "after the onset) passed")


def e3_systematic_uncertainty(pm_GPa: float) -> Uncertainty:
    """The E3 estimator's regime-dependent systematic on a contact pressure p_m(E3) (review P2), as an Uncertainty.

    With the radius bias b in :data:`E3_RADIUS_BIAS_RANGE` (a_E3 = (1 + b) a_true), p_m on the true contact is
    (1 + b)^2 p_m(E3): 0.79-1.64x.  The contact regime of a frame (elastic pincushion, cone-like, flat) is not
    identified, so the whole range is carried; ``value`` is the larger one-sided deviation (``relative`` the same as a
    fraction) and the description gives both bounds.  The |E3 - E4| spread does not replace it: on a cone-like field
    E3 and E4 are both 7-10 % low in radius."""
    lo, hi = E3_RADIUS_BIAS_RANGE
    f_lo, f_hi = (1.0 + lo) ** 2, (1.0 + hi) ** 2
    rel = max(f_hi - 1.0, 1.0 - f_lo)
    pm = float(pm_GPa)
    return Uncertainty(kind=UncertaintyKind.NUMERICAL_CONVERGENCE, value=abs(rel * pm), unit=Unit.GPA, relative=rel,
                       interval="range",
                       description=f"E3 contact-area estimator bias (review P2, E3_RADIUS_BIAS): radius bias "
                                   f"{100 * lo:+.0f} % (elastic cone) to {100 * hi:+.0f} % (elastic Berkovich, "
                                   f"pincushion contact) on exact continuum contact fields, regime not identified; p_m on "
                                   f"the true contact = {f_lo:.2f}-{f_hi:.2f} x p_m(E3) = {f_lo * pm:.3g}-{f_hi * pm:.3g} "
                                   "GPa; not corrected")


def indentation_records(analysis: Mapping[str, Any], *, candidate: Candidate, card: Any,
                        conditions_for: Any, engine_run: EngineRunProvenance, provenance: Provenance,
                        method: MethodSpec, orientation: Optional[Orientation] = None,
                        checks: Sequence[CheckResult] = (), evidence: Sequence[ValidationEvidence] = (),
                        tip_model: str = "rigid_sdf", scale: PropertyScale = PropertyScale.ATOMISTIC_IDEAL) -> list[PropertyRecord]:
    """``*_MD`` records from an analysis (module docstring).

    ``conditions_for(depth_A, area_estimator, depth_zero_rule)`` returns the record Conditions (with
    ContactConditions) for a value taken at that depth; e.g. ``IndentationProtocol.conditions`` wrapped.
    ``evidence`` is attached to the hardness / contact-pressure records (status ceiling and engine cap apply).
    ``checks`` (the per-run checks, :func:`analysis_checks`) go on every record; the contact-pressure record adds
    :func:`plateau_window_check` and the hardness record :func:`gate_checks` (module docstring: required checks).
    What cannot be produced is listed by :func:`records_not_available`.
    """
    prov = provenance.model_copy(update={"engine_run": engine_run})
    h_max = float(analysis["h_max_A"])
    dz = analysis["depth_zero"]
    dz_rule = "elastic_fit" if dz["rule_used"] == "elastic_fit" and dz["elastic_fit"].get("h0_A") is not None else "first_contact"
    recs: list[PropertyRecord] = []
    common = dict(candidate=candidate, card=card, method=method, provenance=prov, orientation=orientation, scale=scale)
    run_checks = list(checks)
    cp_checks = run_checks + [plateau_window_check(analysis)]
    recs.append(make_md_record(
        name=PropertyName.INDENTATION_MAX_LOAD_MD, value=float(analysis["P_at_hmax_windowed_nN"]), unit=Unit.NANONEWTON,
        conditions=conditions_for(h_max, None, dz_rule), uncertainty=[Uncertainty(
            kind=UncertaintyKind.STATISTICAL_SAMPLING, value=float(analysis["load"]["noise"]["windowed_nN"]),
            unit=Unit.NANONEWTON, interval="1sigma", description="scatter of the windowed load about a 2 ps mean in contact")],
        notes=f"windowed load at h_max = {h_max:.3g} A", checks=run_checks, **common))
    # contact pressure on the primary estimator over the plateau window at h_max
    h_eval = evaluation_depth(h_max)
    plat = analysis["plateau_LM_R10"].get(f"h{h_eval:g}", {})
    e3 = plat.get(PRIMARY_ESTIMATOR, {})
    e4 = plat.get("E4", {})
    hv = analysis.get("hardness") or {}
    gates_ok = bool(analysis["gate_ii"]["passes"] and (analysis.get("gate_iii") or {}).get("passes"))
    no_hv = (hv.get("reason") or "Gates II and III did not both pass") if not gates_ok else hardness_block_reason(analysis)
    if "pm_mean_GPa" in e3:
        spread = abs(e3["pm_mean_GPa"] - e4.get("pm_mean_GPa", e3["pm_mean_GPa"]))
        unc = [Uncertainty(kind=UncertaintyKind.NUMERICAL_CONVERGENCE, value=float(spread), unit=Unit.GPA, interval="range",
                           description="contact-area estimator systematic |E3 - E4| over the window"),
               Uncertainty(kind=UncertaintyKind.FIT_STANDARD_ERROR, value=float(e3["pm_sd_GPa"]), unit=Unit.GPA,
                           interval="1sigma", description="scatter of p_m(E3) over the window"),
               e3_systematic_uncertainty(float(e3["pm_mean_GPa"]))]
        gates = f"Gate II {'passed' if analysis['gate_ii']['passes'] else 'failed'}, Gate III " \
                f"{'passed' if (analysis.get('gate_iii') or {}).get('passes') else 'failed'}"
        if no_hv:
            gates += f"; no hardness: {no_hv}"
        recs.append(make_md_record(
            name=PropertyName.INDENTATION_CONTACT_PRESSURE_MD, value=float(e3["pm_mean_GPa"]), unit=Unit.GPA,
            conditions=conditions_for(h_eval, ESTIMATOR_NAMES[PRIMARY_ESTIMATOR], dz_rule), evidence=evidence,
            uncertainty=unc, notes=(f"mean p_m (E3) over h in [{e3['h_window_A'][0]:.3g}, {e3['h_window_A'][1]:.3g}] A; "
                                    f"LM-R10 metric E3 {e3['metric']:.3f}, E4 {e4.get('metric', float('nan')):.3f}; {gates}. "
                                    "A contact pressure, not a hardness. " + contact_rule_note(analysis)).strip(),
            checks=cp_checks, **common))
    elif analysis.get("deepest_loading_frame") and _finite(analysis["deepest_loading_frame"].get("pm_E3")):
        fr = analysis["deepest_loading_frame"]
        spread = abs(fr["pm_E3"] - fr["pm_E4"]) if _finite(fr.get("pm_E4")) else None
        recs.append(make_md_record(
            name=PropertyName.INDENTATION_CONTACT_PRESSURE_MD, value=float(fr["pm_E3"]), unit=Unit.GPA,
            conditions=conditions_for(fr["h_A"], ESTIMATOR_NAMES[PRIMARY_ESTIMATOR], dz_rule), evidence=evidence,
            uncertainty=([] if spread is None else [Uncertainty(
                kind=UncertaintyKind.NUMERICAL_CONVERGENCE, value=float(spread), unit=Unit.GPA, interval="range",
                description="contact-area estimator systematic |E3 - E4| at this frame")])
            + [e3_systematic_uncertainty(float(fr["pm_E3"]))],
            notes=f"p_m (E3) of the deepest loading frame (h = {fr['h_A']:.3g} A); fewer than 3 frames in the plateau "
                  "window, so no plateau metric. A contact pressure, not a hardness."
                  + (f" No hardness: {no_hv}." if no_hv else "") + (" " + contact_rule_note(analysis)).rstrip(),
            checks=cp_checks, **common))
    # hardness only after Gates II and III
    g3 = analysis.get("gate_iii") or {}
    if analysis["gate_ii"]["passes"] and g3.get("passes") and hardness_block_reason(analysis) is None:
        recs.append(make_md_record(
            name=PropertyName.INDENTATION_HARDNESS_MD_ESTIMATED_CONTACT, value=float(g3["H_GPa"]), unit=Unit.GPA,
            conditions=conditions_for(h_eval, ESTIMATOR_NAMES[PRIMARY_ESTIMATOR], dz_rule), evidence=evidence,
            uncertainty=[Uncertainty(kind=UncertaintyKind.NUMERICAL_CONVERGENCE, value=float(g3["estimator_spread_GPa"]),
                                     unit=Unit.GPA, interval="range", description="E3-E4 estimator spread"),
                         e3_systematic_uncertainty(float(g3["H_GPa"]))],
            notes=(_hardness_note(analysis) + ". " + contact_rule_note(analysis)).strip(),
            checks=run_checks + gate_checks(analysis), **common))
    fit = dz["elastic_fit"]
    if "M_GPa" in fit:
        recs.append(make_md_record(
            name=PropertyName.INDENTATION_REDUCED_MODULUS_MD, value=float(fit["M_GPa"]), unit=Unit.GPA,
            conditions=conditions_for(float(fit.get("h_range_A", [0, fit.get("h_max_A", h_max)])[-1]), None, "elastic_fit"),
            uncertainty=[Uncertainty(kind=UncertaintyKind.FIT_STANDARD_ERROR, value=float(fit.get("M_err_GPa", float("nan"))),
                                     unit=Unit.GPA, interval="1sigma", description="elastic loading fit")]
            if "M_err_GPa" in fit else [],
            notes=f"{fit.get('model', 'Hertz sphere')} fit of the early loading branch against "
                  f"{fit.get('depth_variable', 'h')}; includes the finite-cell stiffening of the MD cell (box factor)"
                  + ("" if str(fit.get("depth_variable", "h")).startswith("h_c") else " and the rigid-tip wall compliance")
                  + "; not a Vlassak-Nix modulus",
            checks=run_checks, **common))
    pops = analysis["pop_in"]["load_drop_events"]
    if pops:
        ev = pops[0]
        for nm, val, unit in ((PropertyName.POP_IN_DEPTH_MD, ev["h_peak_A"], Unit.ANGSTROM),
                              (PropertyName.POP_IN_LOAD_MD, ev["P_peak_nN"], Unit.NANONEWTON)):
            recs.append(make_md_record(name=nm, value=float(val), unit=unit, conditions=conditions_for(ev["h_peak_A"], None, dz_rule),
                                       notes=(analysis["pop_in"].get("rule") or "first windowed load drop > 5 %")
                                       + f" (drop {ev['max_drop_frac']:.3f})",
                                       checks=run_checks, **common))
    return recs


# ============================================================================================ LAMMPS runs
# Boundary of the decks whose metadata line does not state it (G06 in.g06_berk.lmp, in.indent_sphere.lmp):
# bottom 3 A fixed, next 7 A Langevin 300 K (damping 0.1 ps).  The SDF deck and in.berk_atomistic.lmp print
# fixt / batht / bath / tau / T in their META line.
_LEGACY_BOUNDARY = {"fixt": 3.0, "batht": 7.0, "bath": "langevin", "tau": 0.1, "T": 300.0}


def lammps_log_accelerator(path: Any) -> str:
    """The accelerator a LAMMPS log reports (review C3), read up to the first run summary ("Loop time of").

    * ``"kokkos/gpu"`` (:data:`KOKKOS_GPU_ACCELERATOR`): the KOKKOS banner ("KOKKOS mode ... is enabled", which
      ``-k on`` prints at the top of the log) followed by "will use up to N GPU(s) per node" with N >= 1;
    * ``"kokkos/cpu"``: KOKKOS on host threads (no GPU line);
    * ``"gpu-package"``: the GPU package ("package gpu" or its "- Using acceleration for" banner);
    * ``"cpu"`` otherwise (including a missing log)."""
    from fracture_md.io.indent_traj import open_text

    if path is None or not Path(path).is_file():
        return "cpu"
    kokkos, gpus, gpu_pkg = False, 0, False
    with open_text(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("Loop time of "):
                break
            if s.startswith("KOKKOS mode") and "enabled" in s:
                kokkos = True
            m = re.search(r"will use up to (\d+) GPU\(s\) per node", s)
            if m:
                gpus = max(gpus, int(m.group(1)))
            if s.startswith("package gpu") or s.startswith("- Using acceleration for"):
                gpu_pkg = True
    if kokkos:
        return KOKKOS_GPU_ACCELERATOR if gpus > 0 else "kokkos/cpu"
    return "gpu-package" if gpu_pkg else "cpu"


def lammps_accelerator(run: Any) -> str:
    """The accelerator of an imported LAMMPS run: ``run.log.accelerator`` when the importer states it
    (io.lammps_indent), else :func:`lammps_log_accelerator` of the run's log.  It goes into the EngineRunProvenance
    and decides whether the CPU-use rule applies (:func:`lammps_cpu_check_applies`)."""
    acc = getattr(getattr(run, "log", None), "accelerator", None)
    if acc:
        return str(acc)
    return lammps_log_accelerator((getattr(run, "files", None) or {}).get("log"))


def lammps_cpu_check_applies(accelerator: str) -> bool:
    """False for a GPU run (KOKKOS GPU backend or GPU package): the >= :data:`MIN_CPU_PERCENT` CPU-use rule judges
    CPU MPI runs only (review C3)."""
    a = str(accelerator or "").lower()
    return not any(k in a for k in ("gpu", "cuda", "hip"))


def lammps_ledger_atoms(run: Any, rd: Any = None) -> tuple[int, str]:
    """N of the ledger's thermal floor for an imported LAMMPS run, and its source (review C6).

    The deck's ledger (etotal + ecouple - W - E0) sums over every sample atom: META ``nsample`` (in.berk_sdf.lmp:
    count(all); in.berk_atomistic.lmp: count(sample), without the tip).  The generator-2 analysis dump is a subset of
    the sample (the analysis set, ``nan`` in the META line, ~1/10 of the sample at S1), so the dumped atoms count only
    for decks that do not state ``nsample`` (they dump the whole sample).  ``rd``: the run's IndentationRunData (its
    ``species``), else ``run.species()``."""
    try:
        ns = float(run.meta.get("nsample", 0) or 0)
    except (TypeError, ValueError):
        ns = 0.0
    if ns > 0:
        return int(ns), "META nsample"
    return int(len(rd.species if rd is not None else run.species())), "dumped atoms"


def lammps_run_manifest(run: Any) -> Optional[dict[str, Any]]:
    """The deck generator's ``manifest.json`` next to an imported run (engines.lammps_indent.write_indentation_deck:
    protocol, slab orientation and termination, variables), or None (hand-written decks, G06, sphere, or a
    manifest of another tag)."""
    p = Path(run.run_dir) / "manifest.json"
    if not p.is_file():
        return None
    try:
        man = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return man if isinstance(man, dict) and str(man.get("tag")) == str(run.tag) else None


def lammps_lattice_constant(run: Any, lattice_constant_A: Optional[float] = None) -> tuple[float, str]:
    """Conventional lattice constant of the sample of an imported run and where it came from.

    Order: the argument; the log's ``lattice custom`` line (G06, sphere and atomistic decks build the sample
    in LAMMPS); an ``a0`` key of the META line; the generator manifest (``slab.lattice_constant_A``, then
    ``protocol.lattice_parameter_A``: the SDF deck reads the harness slab with ``read_data`` and has no lattice
    line).  Raises ValueError when none states it (pass ``lattice_constant_A``; it sets the E3 dilation).
    """
    if lattice_constant_A is not None:
        return float(lattice_constant_A), "argument"
    if run.lattice_constant_A is not None:
        return float(run.lattice_constant_A), "log 'lattice custom' line"
    if isinstance(run.meta.get("a0"), float) and run.meta["a0"] > 0.0:  # the SDF deck prints a0=0 when unknown
        return float(run.meta["a0"]), f"{run.preset} META a0"
    man = lammps_run_manifest(run)
    if man is not None:
        for sec, key in (("slab", "lattice_constant_A"), ("protocol", "lattice_parameter_A")):
            v = (man.get(sec) or {}).get(key)
            if v is not None:
                return float(v), f"manifest.json {sec}.{key}"
    raise ValueError(f"lattice constant of LAMMPS run {run.tag!r} unknown (no 'lattice custom' line in the log, no a0 in "
                     "the META line, no lattice constant in manifest.json): pass lattice_constant_A")


def _lammps_boundary(meta: Mapping[str, Any]) -> str:
    b = {k: meta.get(k, v) for k, v in _LEGACY_BOUNDARY.items()}
    md = str(meta.get("mode", "md")) != "qs"
    return (f"bottom {float(b['fixt']):g} A fixed; "
            + (f"next {float(b['batht']):g} A {b['bath']} {float(b['T']):g} K (tau {float(b['tau']):g} ps); NVE above; "
               if md else "")
            + "lateral periodic; free top surface")


def lammps_contact_conditions(run: Any, *, depth_A: float, area_estimator: Optional[str] = None,
                              depth_zero_rule: str = "not_applicable", termination: Optional[str] = None,
                              azimuth_reference: Optional[str] = None) -> ContactConditions:
    """ContactConditions of an imported LAMMPS run (io.lammps_indent.LammpsIndentRun), from what the run states.

    * ``sdf-berkovich`` (examples/indent/in.berk_sdf.lmp): shape, angle, tip radius R, azimuth az and wall
      stiffness K from SDF_META; the sample is the harness slab (lab x = the slab x of structures.indentation,
      named from the generator's manifest.json when present).
    * ``sphere`` (fix indent sphere): R and K from INDENT_META; ``g06-berkovich`` (rigid atomistic diamond tip):
      sharp, faces 65.27 deg, outward face normals at az, az + 120, az + 240 deg of the LAMMPS lattice x = [100]
      (``az`` from BERK_META: the atomistic-tip deck states it; G06, which does not, is 90).
    * boundary: fixed / thermostat thickness, bath style, temperature and tau from the META line (fixt, batht,
      bath, T, tau), else the legacy decks' 3 A fixed + 7 A Langevin 300 K (0.1 ps).
    ``termination`` / ``azimuth_reference`` override the defaults.
    """
    m = run.meta
    man = lammps_run_manifest(run)
    zmin0 = float(m.get("zmin0", 0.0))
    top = float(m.get("zsurfL", m.get("zs0")))
    h_max = float(m["hmax"]) if "hmax" in m else float(m["dh"]) * float(m["nqs"])  # qs META: steps of dh
    if run.preset == "sdf-berkovich":
        geom = run.geometry()
        shape = geom.shape
        tip: dict[str, Any] = dict(
            indenter_shape=shape, tip_model="rigid_sdf",
            face_angle=q(float(geom.half_angle_deg), Unit.DEGREE) if shape == "berkovich" else None,
            cone_half_angle=q(float(geom.half_angle_deg), Unit.DEGREE) if shape == "cone" else None,
            tip_radius=q(float(m["R"]), Unit.ANGSTROM), azimuth=q(float(m["az"]), Unit.DEGREE),
            wall_stiffness=q(float(m["K"]), Unit.EV_PER_ANGSTROM3))
        ori = ((man or {}).get("slab") or {}).get("orientation") or {}
        az_ref = (f"lab x = {tuple(ori['x_direction'])} (harness slab x)" if ori.get("x_direction") is not None
                  else "lab x = harness slab x (structures.indentation)")
        prot = (man or {}).get("protocol") or {}
        term = (f"{tuple(prot['plane_miller'])} {prot['top_termination']}-terminated"
                if prot.get("plane_miller") is not None and prot.get("top_termination") else
                "harness slab (termination not recorded: no deck-generator manifest)")
    elif run.preset == "sphere":
        tip = dict(indenter_shape="sphere", tip_model=run.tip_model(), tip_radius=q(float(m["R"]), Unit.ANGSTROM),
                   azimuth=q(0.0, Unit.DEGREE), wall_stiffness=q(float(m["K"]), Unit.EV_PER_ANGSTROM3))
        az_ref, term = "[100] (LAMMPS lattice x)", "(001) C-terminated, unreconstructed"
    else:  # g06-berkovich: the rigid atomistic diamond tip of in.g06_berk.lmp / in.berk_atomistic.lmp
        tip = dict(indenter_shape="berkovich", tip_model=run.tip_model(), face_angle=q(65.27, Unit.DEGREE),
                   tip_radius=q(0.0, Unit.ANGSTROM), azimuth=q(float(run.geometry().azimuth_deg), Unit.DEGREE),
                   wall_stiffness=None)
        az_ref, term = "[100] (LAMMPS lattice x)", "(001) C-terminated, unreconstructed"
    md = str(m.get("mode", "md")) != "qs"
    return ContactConditions(
        **tip, azimuth_reference=azimuth_reference if azimuth_reference is not None else az_ref,
        tip_description=run.tip_description(),
        indentation_speed=q(run.speed_m_per_s, Unit.METER_PER_SECOND) if md else None,
        max_depth=q(h_max, Unit.ANGSTROM), depth=q(min(depth_A, h_max), Unit.ANGSTROM),
        depth_zero_rule=depth_zero_rule, area_estimator=area_estimator,
        surface_termination=termination if termination is not None else term,
        lateral_size=q([float(m["lx"]), float(m["ly"])], Unit.ANGSTROM), thickness=q(top - zmin0, Unit.ANGSTROM),
        boundary=_lammps_boundary(m))


def records_from_lammps_run(run: Any, *, candidate: Candidate, potential: str = "vashishta_2007_SiC",
                            cfg: AnalysisConfig = AnalysisConfig(), temperature_K: Optional[float] = None,
                            lattice_constant_A: Optional[float] = None,
                            evidence: Sequence[ValidationEvidence] = (), analysis: Optional[Mapping[str, Any]] = None,
                            gate_ii_evidence: Sequence[Mapping[str, Any]] = (),
                            requested_by: str = "human",
                            recipe: Any = indent_recipe.DEFAULT_RECIPE) -> ProtocolOutput:
    """Analyse an imported LAMMPS indentation run and build its IndentationCurve and ``*_MD`` records.

    The EngineRunProvenance carries the deck, potential-file and pair-block sha256; the input files' hashes
    (and the generator manifest's, when present) go in Provenance.input_hashes; the harness card must match the
    potential file (checked for Vashishta by parsing the file: io.lammps_indent.card_for_lammps_potential).
    ``lattice_constant_A``: see :func:`lammps_lattice_constant` (needed for SDF-deck runs without a manifest).
    ``temperature_K``: default the META line's T (300 K for the legacy decks).  ``gate_ii_evidence``: intermediate-
    depth Gate II evidence (indent_analysis.gate_ii_onset; e.g. gate_ii_evidence_from_gate_i of Gate I quenches of
    the run's full dumps), without which no hardness record can be produced.  Status <= numerically_verified.

    Per-run checks (:func:`analysis_checks`): Newton 3 from the deck's fixed-layer reaction (Rz), and for MD runs the
    energy ledger from the deck's ``ledger_eV`` column (etotal + ecouple - W - E0: the CSVR / Langevin bath tally
    ecouple included) at the deepest loading sample, against 1e-3 W or the thermal floor :func:`ledger_floor_eV` of
    the sample atoms the ledger sums (:func:`lammps_ledger_atoms`: META ``nsample``; the dumped atoms for decks that do
    not state it, since the generator-2 analysis dump is a subset) at the META temperature and time step.  Decks that
    log no ledger (G06, sphere) get a NOT EVALUATED ledger check: their records are FAILED, never
    numerically_verified.  The run's accelerator (:func:`lammps_accelerator`, e.g. ``"kokkos/gpu"``) goes into the
    EngineRunProvenance and ``raw["lammps_run"]``; on a GPU run the CPU-use rule does not apply (:func:`analysis_checks`).

    ``recipe``: the frozen recipe the records are built under (name, path or physics.indent_recipe.IndentationRecipe;
    default indent_recipe.DEFAULT_RECIPE; None records none): its sha256 goes into the EngineRunProvenance, and when
    the run's manifest.json names the recipe its deck was generated under, the two protocol sections must agree
    (:func:`physics.indent_recipe.deck_recipe_check`, a check of every record).
    """
    from fracture_md.engines.base import sha256_file
    from fracture_md.io.lammps_indent import card_for_lammps_potential
    from fracture_md.potentials import get_potential
    from fracture_md.provenance import collect

    rec = indent_recipe.resolve_recipe(recipe)
    card = get_potential(potential).card
    pots = run.potential_files()
    file_card = card_for_lammps_potential(pots[0], run.log.style_for_file(pots[0])) if pots else None
    a_lat, a_src = lammps_lattice_constant(run, lattice_constant_A)
    rd = run.run_data(potential=potential, lattice_constant_A=a_lat)
    rd.gate_ii_evidence = [dict(e) for e in gate_ii_evidence]
    an = dict(analysis) if analysis is not None else analyze_indentation(rd, cfg)
    acc = lammps_accelerator(run)
    lp = {**run.provenance_dict(), "accelerator": acc}
    er = run.engine_run(potential_card_id=card.id,
                        potential_convention=f"{'; '.join(run.log.pair_style[-1:])}; file parsed as {file_card}",
                        recipe=None if rec is None else rec.path, accelerator=acc)
    if rec is not None and er.recipe_sha256 != rec.sha256:
        raise RuntimeError(f"{rec.path} changed while the records were built (sha256 {er.recipe_sha256} != {rec.sha256})")
    md = str(run.meta.get("mode", "md")) != "qs"
    T = float(temperature_K) if temperature_K is not None else float(run.meta.get("T", _LEGACY_BOUNDARY["T"]))
    ledger = None
    if md and "ledger_eV" in rd.channels and "W_eV" in rd.channels:
        dt_fs = float(run.timestep_ps) * 1e3
        n_led, n_src = lammps_ledger_atoms(run, rd)
        ledger = ledger_at_hmax(rd.h_A, rd.channels["ledger_eV"], rd.channels["W_eV"], segment=rd.segment,
                                floor_eV=ledger_floor_eV(n_led, T, dt_fs))
        ledger.update(n_atoms=n_led, n_atoms_source=n_src, temperature_K=T, dt_fs=dt_fs, source="deck ledger_eV column")
    checks = analysis_checks(an, engine="lammps", ledger=ledger, lammps=lp, mode="md" if md else "quasistatic")
    if file_card is not None:
        checks.append(CheckResult(name="lammps_potential_file_matches_card", passed=bool(file_card == card.id),
                                  details=f"{pots[0].name} parsed into {file_card}; record card {card.id}"))
    man = lammps_run_manifest(run)
    deck_recipe = (man or {}).get("recipe")
    recipe_check = indent_recipe.deck_recipe_check(deck_recipe, rec)
    if recipe_check is not None:
        checks.append(recipe_check)
    hashes = dict(run.sha256)
    if man is not None:
        hashes["manifest"] = sha256_file(Path(run.run_dir) / "manifest.json")
    prov = collect(requested_by=requested_by, potential_id=card.id, input_hashes=hashes,
                   extra={"lammps_run": lp, "lattice_constant": {"value_A": a_lat, "source": a_src}})
    v = run.speed_m_per_s
    h_max = float(run.meta["hmax"])

    def conditions_for(depth, est, rule):
        return Conditions(temperature=q(T, Unit.KELVIN), loading="dynamic",
                          strain_rate=q(v / (h_max * 1e-10), Unit.PER_SECOND),
                          stress_state="contact loading by " + ("a rigid indenter" if run.tip_model() == "rigid_sdf"
                                                                 else "an atomistic tip"),
                          contact=lammps_contact_conditions(run, depth_A=depth, area_estimator=est, depth_zero_rule=rule))

    ori = ((man or {}).get("slab") or {}).get("orientation")
    if isinstance(ori, Mapping) and ori:  # the harness slab written by the SDF deck generator (e.g. (001), lab x = [110])
        orientation: Optional[Orientation] = Orientation.model_validate(ori)
    elif run.preset == "sdf-berkovich":
        orientation = None  # a harness slab of unrecorded orientation: never guess the cube axes
    else:  # G06 / sphere / atomistic-tip decks build the sample with LAMMPS lattice commands (the atomistic deck's
        # manifest states it as text only: "(001) C-terminated")
        orientation = Orientation(plane_miller=(0, 0, 1), x_direction=(1, 0, 0), y_direction=(0, 1, 0),
                                  description="LAMMPS lattice custom cube axes")
    method = MethodSpec(name="MD indentation (LAMMPS run, imported and analysed by the harness)", engine="lammps",
                        description=f"{run.preset} deck, {run.log.version}, {run.log.mpi_tasks} MPI ranks"
                                    + ("" if acc == "cpu" else f", {acc}"),
                        assumptions=list(MD_ASSUMPTIONS))
    recs = indentation_records(an, candidate=candidate, card=card, conditions_for=conditions_for, engine_run=er,
                               provenance=prov, method=method, orientation=orientation, checks=checks, evidence=evidence,
                               tip_model=run.tip_model())
    contact = lammps_contact_conditions(run, depth_A=h_max)
    gaps = records_not_available(an)
    curve = indentation_curve(rd, an, mode="md", contact=contact, temperature_K=T, potential_id=card.id,
                              engine_run=er, orientation=orientation, curve_id=f"lammps:{run.tag}", checks=checks,
                              assumptions=list(MD_ASSUMPTIONS), notes=_not_available_note(gaps))
    return ProtocolOutput(properties=recs, checks=checks, indentation_curves=[curve],
                          raw={"indentation_analysis": json_ready(an), "lammps_run": lp,
                               "lattice_constant": {"value_A": a_lat, "source": a_src},
                               "ledger": None if ledger is None else json_ready(ledger),
                               "records_not_available": gaps,
                               "recipe": {"records": None if rec is None else rec.summary(), "deck": deck_recipe}})


# ============================================================================================ JAX harness runner
RUNNER_FRAMES_DIR = "runner_frames"  # <output_dir>/checkpoint/<this>: the runner's frames of a resumable MD run


def _bulk(model: Any, candidate: Candidate, structure: Any, skin: float, a_T: Optional[float]):
    from fracture_md.physics.cleavage import _bulk_reference  # relaxed conventional cell (shared with cleavage)

    atoms, _, info = _bulk_reference(model, candidate, structure, skin)
    return atoms, info


def harness_engine_run(potential_card_id: str, *, md_dtype: str = "float64",
                       command: str = "fracture_md.physics.indentation:run_indentation",
                       recipe_sha256: Optional[str] = None) -> EngineRunProvenance:
    """EngineRunProvenance of a harness indentation run with the precision of its trajectory.

    ``engines.base.jax_engine_run`` reports the process-wide x64 flag (always on in this package); the MD kernel
    runs in ``IndentMDConfig.dtype`` (float32 is traced with matmul precision "highest", physics.indent_md), so a
    float32 trajectory is recorded as float32 with that precision.  ``recipe_sha256``: the frozen recipe of the run
    (physics.indent_recipe).
    """
    from fracture_md.engines.base import jax_engine_run

    er = jax_engine_run(potential_card_id=potential_card_id, command=command, recipe_sha256=recipe_sha256)
    if md_dtype == er.dtype:
        return er
    upd: dict[str, Any] = {"dtype": md_dtype}
    if md_dtype == "float32":
        upd["matmul_precision"] = "highest"
    return EngineRunProvenance.model_validate({**er.model_dump(), **upd})


def relaxation_checks(prep_relax: Mapping[str, Any], *, fmax: float,
                      quasistatic_rows: Optional[Sequence[Mapping[str, Any]]] = None) -> list[CheckResult]:
    """Convergence of the FIRE relaxations behind a harness run as CheckResults: the slab preparation (bottom
    fixed) and, for a quasi-static run, every depth step (metric = the largest residual max |F_i|, tolerance =
    the protocol's ``relax.fmax_eV_per_A``).  A load from an unrelaxed configuration fails the record."""
    out = [CheckResult(name="indentation_preparation_relaxed", passed=bool(prep_relax["converged"]),
                       metric=float(prep_relax["fmax"]), tolerance=float(fmax),
                       details=f"FIRE relaxation of the slab (bottom fixed): {int(prep_relax['steps'])} steps, "
                               f"max |F_i| {float(prep_relax['fmax']):.3g} eV/A")]
    if quasistatic_rows:
        bad = [float(r["h_A"]) for r in quasistatic_rows if not r["converged"]]
        worst = max(float(r["fmax"]) for r in quasistatic_rows)
        out.append(CheckResult(name="indentation_quasistatic_converged", passed=not bad, metric=worst,
                               tolerance=float(fmax),
                               details=f"FIRE converged at {len(quasistatic_rows) - len(bad)}/{len(quasistatic_rows)} "
                                       "depth steps" + (f"; not at h = {', '.join(f'{x:g}' for x in bad)} A" if bad else "")))
    return out


def equilibration_checks(prep: Mapping[str, Any]) -> list[CheckResult]:
    """The bath reached its target temperature (``indentation_equilibration_temperature``): the mean temperature of
    the thermostatted atoms over the second half of equilibration stage 1 (Langevin on every free atom, a
    canonical sample) within max(EQUILIBRATION_T_TOL, 3 sqrt(2 / (3 n_free))) of the target (3 sigma of one
    instantaneous canonical sample: never fails a correct small test cell, catches a 1.45x target error on any
    cell).  A very short ``equilibration_ps`` can fail it legitimately (equilibrate longer).  [] when the run has no
    equilibration (0 K, quasi-static)."""
    eq = (prep or {}).get("equilibration") or {}
    if "stage1_T_thermostatted_mean_K" not in eq or not eq.get("target_K"):
        return []
    dev = abs(float(eq["stage1_T_thermostatted_mean_K"]) / float(eq["target_K"]) - 1.0)
    tol = max(EQUILIBRATION_T_TOL, 3.0 * math.sqrt(2.0 / (3.0 * max(int(eq["n_free"]), 1))))
    return [CheckResult(name="indentation_equilibration_temperature", passed=bool(dev <= tol), metric=float(dev),
                        tolerance=float(tol),
                        details=f"stage-1 (Langevin, {int(eq['n_free'])} free atoms) second-half mean "
                                f"{float(eq['stage1_T_thermostatted_mean_K']):.1f} K vs target {float(eq['target_K']):g} K")]


def _second_half_mean(logs: Mapping[str, Any], key: str) -> float:
    v = np.asarray(logs.get(key, logs["T_core_K"]), float)
    return float(np.mean(v[len(v) // 2:]))


def prepare_indentation_slab(model_c: Any, slab: Any, *, fixed: np.ndarray, fmax: float, dt_fs: float,
                             temperature_K: float, equilibration_ps: float, cfg: Any, seed: int = 0,
                             park_gap_A: float = PARK_GAP_A,
                             relax_max_steps: int = 50000) -> tuple[Any, Optional[np.ndarray], dict[str, Any]]:
    """Recommended protocol item 7: FIRE relaxation with the bottom frozen (at most ``relax_max_steps``); then (MD)
    Langevin on every mobile atom started at 2 T for 60 % of ``equilibration_ps`` and the boundary thermostat alone
    for the rest.  Returns (system, momenta or None, info); ``info["relax"]["converged"]`` must be checked
    (:func:`relaxation_checks`), and so must the stage temperatures in ``info["equilibration"]``
    (:func:`equilibration_checks`)."""
    from fracture_md.physics.indent_md import TipSchedule, region_momenta, run_indentation_md
    from fracture_md.physics.relax import relax_positions
    from fracture_md.structures.system import AtomicSystem

    s0 = AtomicSystem.from_ase(slab.atoms)
    s, info = relax_positions(model_c, s0, fixed=fixed, method="fire", fmax=fmax, newton_polish=False,
                              skin=cfg.skin_A, max_steps=int(relax_max_steps))
    out: dict[str, Any] = {"relax": dict(converged=info.converged, steps=info.n_steps, fmax=info.fmax, energy_eV=info.energy)}
    if equilibration_ps <= 0 or temperature_K <= 0:
        return s, None, out
    masks = slab.masks()
    xy = slab.center_xy
    park = (float(xy[0]), float(xy[1]), float(slab.surface_z + park_gap_A))  # tip above the surface, cap inside the cell
    t1 = PREP_LANGEVIN_FRACTION * equilibration_ps * 1e3
    t2 = (1.0 - PREP_LANGEVIN_FRACTION) * equilibration_ps * 1e3
    p0 = region_momenta(np.asarray(s.masses_amu), {"free": masks["free"]}, PREP_START_TEMPERATURE_FACTOR * temperature_K,
                        seed=seed)
    c1 = dataclasses.replace(cfg, thermostat="langevin", chunk_steps=max(1, min(cfg.chunk_steps, int(t1 / dt_fs))))
    r1 = run_indentation_md(model_c, s, TipSchedule.constant(park, t1, "equilibrate"), masks=masks, cfg=c1,
                            thermostat_mask=masks["free"], momenta=p0)
    s1 = s.replace(frac=r1.frac)
    c2 = dataclasses.replace(cfg, chunk_steps=max(1, min(cfg.chunk_steps, int(t2 / dt_fs))))
    r2 = run_indentation_md(model_c, s1, TipSchedule.constant(park, t2, "equilibrate"), masks=masks, cfg=c2,
                            momenta=r1.momenta)
    out["equilibration"] = {"stage1_T_thermostatted_final_K": float(r1.logs.get("T_thermostat_K", r1.logs["T_core_K"])[-1]),
                            "stage2_T_core_final_K": float(r2.logs["T_core_K"][-1]),
                            "stage2_load_max_nN": float(np.max(np.abs(r2.logs["load_nN"]))),
                            "stage1_T_thermostatted_mean_K": _second_half_mean(r1.logs, "T_thermostat_K"),
                            "stage2_T_core_mean_K": _second_half_mean(r2.logs, "T_core_K"),
                            "stage2_T_bath_mean_K": _second_half_mean(r2.logs, "T_thermostat_K"),
                            "target_K": float(temperature_K), "n_free": int(np.count_nonzero(masks["free"]))}
    return s1.replace(frac=r2.frac), np.asarray(r2.momenta), out


def harness_preparation(protocol: Any, slab: Any, *, langevin_tau_ps: Optional[float] = None) -> Optional[dict[str, Any]]:
    """The preparation :func:`prepare_indentation_slab` applies for an MD ``protocol`` on ``slab``, in the recipe's
    terms (physics.indent_recipe.preparation_summary; compared with the recipe by protocol_deviations): no z
    pre-expansion, Langevin from PREP_START_TEMPERATURE_FACTOR x T for PREP_LANGEVIN_FRACTION of the equilibration
    (time constant ``langevin_tau_ps``, default the protocol's bath tau: the runner's IndentMDConfig), then the bath
    alone.  None for a quasi-static protocol or one without a thermal preparation."""
    if protocol.mode != "md" or protocol.equilibration_ps <= 0 or protocol.temperature_K <= 0:
        return None
    n = protocol.equilibration_ps * 1e3 / protocol.timestep_fs
    return indent_recipe.preparation_summary(
        source=indent_recipe.JAX_RUNNER_PREPARATION, zscale=1.0, langevin_steps=int(round(PREP_LANGEVIN_FRACTION * n)),
        bath_steps=int(round((1.0 - PREP_LANGEVIN_FRACTION) * n)),
        start_temperature_K=PREP_START_TEMPERATURE_FACTOR * protocol.temperature_K, temperature_K=protocol.temperature_K,
        langevin_tau_ps=protocol.thermostat_tau_ps if langevin_tau_ps is None else float(langevin_tau_ps),
        thickness_A=float(slab.thickness_A))


def quasistatic_newton3(load_nN: Any, reaction_nN: Any, segment: Optional[Any] = None, *,
                        min_load_fraction: float = QS_NEWTON3_MIN_LOAD_FRACTION) -> dict[str, Any]:
    """Newton 3 of a quasi-static run (indent_analysis.newton3_balance over the relaxed states of the loading branch
    whose load exceeds ``min_load_fraction`` x the maximum load; 0 K, so no thermal-noise minimum load)."""
    from fracture_md.physics.indent_analysis import newton3_balance

    P = np.asarray(load_nN, float)
    R = np.asarray(reaction_nN, float)
    sel = np.ones(len(P), bool) if segment is None else np.isin(np.asarray(segment).astype(str), ("approach", "load"))
    pmax = float(P[sel].max()) if sel.any() else 0.0
    min_load = max(min_load_fraction * pmax, 0.0)
    out = newton3_balance(P[sel], R[sel], min_load_nN=min_load) if pmax > 0 else \
        {"mean_residual_over_mean_P": float("nan"), "rms_residual_nN": float("nan"), "n": 0}
    return {**out, "min_load_nN": float(min_load),
            "rule": f"quasi-static loading states with P > {min_load_fraction:g} max P (0 K: no thermal noise)"}


def quasistatic_indentation(model_c: Any, system: Any, *, fixed: np.ndarray, apex_xy: Sequence[float], surface_z0: float,
                            depths_A: Sequence[float], fmax: float = 1e-3, max_steps: int = 20000,
                            skin: float = 1.0) -> dict[str, Any]:
    """0 K quasi-static indentation: FIRE relaxation at each apex depth (the apex is a params leaf, so the FIRE
    kernel compiles once).  Returns per-step depth, load (nN), energy, pen_fw, contact atoms (positions),
    convergence (``converged``, ``fire_steps``, residual ``fmax``: see :func:`relaxation_checks`), the fixed-layer
    reaction ``reaction_nN`` (z component of the total force on the ``fixed`` atoms, material + tip, at the relaxed
    state: -load at equilibrium, Newton 3) and the final system."""
    import jax.numpy as jnp

    from fracture_md.physics.indenter import contact_atoms, indenter_diagnostics, indenter_params
    from fracture_md.physics.relax import relax_positions
    from fracture_md.potentials import build_neighbors

    fixed = np.asarray(fixed, bool)
    s = system
    rows: list[dict[str, Any]] = []
    frames = []
    X_at: list[np.ndarray] = []
    for i, h in enumerate(depths_A):
        p = indenter_params(model_c, (float(apex_xy[0]), float(apex_xy[1]), surface_z0 - float(h)))
        s, info = relax_positions(model_c, s, fixed=fixed, method="fire", fmax=fmax, max_steps=max_steps, params=p,
                                  newton_polish=False, skin=skin)
        d = indenter_diagnostics(model_c, s.frac, s.box, p)
        X = np.asarray(s.frac, float) @ np.asarray(s.box, float).T
        load = float(np.asarray(d["load_eV_per_A"])[2]) * EV_PER_A_TO_NN
        c = np.asarray(contact_atoms(model_c, jnp.asarray(s.frac), s.box, p), bool)
        sw = s.replace(frac=jnp.asarray(np.mod(np.asarray(s.frac, float), 1.0)))  # lattice-vector shifts only
        nb = build_neighbors(sw, model_c.cutoff, skin=skin)
        F = np.asarray(model_c.forces(sw.frac, sw.box, model_c.species_for(sw), nb.idx, p), float)
        reaction = float(F[fixed, 2].sum()) * EV_PER_A_TO_NN
        rows.append(dict(step=i, h_A=float(h), load_nN=load, energy_eV=float(info.energy), pen_fw_A=float(d["pen_fw_A"]),
                         n_contact=int(c.sum()), converged=bool(info.converged), fire_steps=int(info.n_steps),
                         fmax=float(info.fmax), reaction_nN=reaction))
        frames.append(FrameData(step=i, time_ps=float(i), positions=X[c], contact=np.ones(int(c.sum()), bool), full=False))
        X_at.append(X)
    return {"rows": rows, "frames": frames, "system": s, "X_at": X_at}


# ------------------------------------------------------------------ frames of a resumable MD run
def _key_frame_label(sched: Any, step: int, chunk_steps: int, dt_fs: float) -> Optional[str]:
    """"max_load" / "end_hold" for the LAST chunk end of the loading / hold segment (the next chunk end lies in
    another segment), the convention of analyze_indentation for unlabelled frames.  A pure function of the
    schedule, so a resumed run labels the same chunks as an uninterrupted one."""
    seg = str(sched.segment(step * dt_fs))
    if seg not in ("load", "hold") or str(sched.segment((step + chunk_steps) * dt_fs)) == seg:
        return None
    return "max_load" if seg == "load" else "end_hold"


def _save_frame_shard(path: Path, fr: FrameData) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    arrs = {"step": np.int64(fr.step), "time_ps": np.float64(fr.time_ps), "positions": np.asarray(fr.positions, float),
            "full": np.bool_(fr.full), "areas": np.bool_(fr.areas), "label": np.array(fr.label)}
    if fr.contact is not None:
        arrs["contact"] = np.asarray(fr.contact, bool)
    np.savez(tmp, **arrs)
    tmp.replace(path)


def _load_frame_shard(path: Path) -> FrameData:
    with np.load(path, allow_pickle=False) as z:
        return FrameData(int(z["step"]), float(z["time_ps"]), np.asarray(z["positions"]),
                         np.asarray(z["contact"]) if "contact" in z.files else None, {}, bool(z["full"]),
                         bool(z["areas"]), str(z["label"]))


def _restore_runner_frames(shard_dir: Path, contact: dict[int, FrameData], keys: dict[str, FrameData], *,
                           n_chunks: int, chunk_steps: int, expected_keys: Iterable[str]) -> int:
    """Complete the frames of a resumed run from the shards of the earlier call(s); returns how many were restored.

    Raises RuntimeError when a chunk's contact frame (or an expected key frame) is in neither place: the analysis
    would otherwise run on the post-restart part of the loading branch only."""
    n = 0
    for k in range(1, n_chunks + 1):
        st = k * chunk_steps
        f = shard_dir / f"contact_{st:012d}.npz"
        if st not in contact and f.exists():
            contact[st] = _load_frame_shard(f)
            n += 1
    for lab in expected_keys:
        f = shard_dir / f"key_{lab}.npz"
        if lab not in keys and f.exists():
            keys[lab] = _load_frame_shard(f)
            n += 1
    missing = [k * chunk_steps for k in range(1, n_chunks + 1) if k * chunk_steps not in contact]
    missing_keys = sorted(set(expected_keys) - set(keys))
    if missing or missing_keys:
        raise RuntimeError(f"resumed indentation run: the contact frames of {len(missing)} chunk(s) (first at step "
                           f"{missing[0] if missing else '-'}) and key frames {missing_keys} are neither recomputed nor "
                           f"stored in {shard_dir} (a checkpoint written by another runner version?); rerun with "
                           "ctx.extra['indentation']['restart'] = False")
    return n


def run_indentation(protocol: Any, candidate: Candidate, model: Any, structure: Any = None,
                    context: Optional[RunContext] = None) -> ProtocolOutput:
    """RUNNERS["indentation"]: build, prepare and indent a slab on the JAX harness; analyse; curve + records.

    ``context.extra["indentation"]`` options: ``md_config`` (IndentMDConfig overrides; ``dtype`` is recorded in
    engine_run), ``vacuum_A`` (overrides :func:`protocol_vacuum_A`), ``run_id``, ``restart`` and ``recipe`` (the frozen
    recipe of the run: name, path or IndentationRecipe, default indent_recipe.DEFAULT_RECIPE; None records none; its
    sha256 goes into every EngineRunProvenance, its identity and the protocol's deviations from it into
    ``raw["recipe"]``).  With an ``output_dir`` the MD run writes a checkpoint to ``<output_dir>/checkpoint``;
    ``restart`` = False (default) starts afresh every call (an existing checkpoint is overwritten), "auto" resumes a
    killed run of the same protocol from it and True requires it.  The runner keeps its per-chunk contact frames and
    key frames next to the checkpoint (``runner_frames``), so a resumed run is analysed on every frame of its loading
    branch.
    The preparation relaxation and every quasi-static step use ``protocol.relax`` (fmax, max_steps); their
    convergence is a check of every record (:func:`relaxation_checks`), and so is the equilibration temperature
    in MD (:func:`equilibration_checks`).  Newton 3 uses the fixed-layer reaction (MD: the driver's ``reaction_nN``
    channel, momentum balance in ``raw["run"]["momentum_balance"]``; quasi-static: the force on the fixed atoms at
    each relaxed depth, :func:`quasistatic_newton3`); MD runs also check the energy ledger (:func:`ledger_at_hmax`,
    :func:`ledger_floor_eV`).  ``raw["records_not_available"]`` lists what could not be produced and why.
    """
    from fracture_md.physics.indent_md import FrameWriter, IndentMDConfig, TipSchedule, run_indentation_md
    from fracture_md.physics.indenter import (IndenterGeometry, check_indenter_cell, contact_atoms, set_indenter,
                                              with_indenter)
    from fracture_md.provenance import collect
    from fracture_md.structures.indentation import slab_for_protocol
    from fracture_md.structures.system import AtomicSystem

    ctx = context or RunContext()
    t_wall = time.monotonic()
    opts = dict(ctx.extra.get("indentation", {}))
    geom = IndenterGeometry(**protocol.indenter.geometry_kwargs())
    if protocol.indenter.tip_model != "rigid_sdf":
        raise NotImplementedError("the JAX harness runner indents with the rigid SDF tip only (atomistic tips: LAMMPS)")
    rec = indent_recipe.resolve_recipe(opts.get("recipe", indent_recipe.DEFAULT_RECIPE))  # fail before any work
    restart = opts.get("restart", False)
    if restart not in (False, True, "auto"):
        raise ValueError(f"ctx.extra['indentation']['restart'] must be False, True or 'auto', got {restart!r}")
    K = float(protocol.indenter.stiffness_eV_per_A3)
    md = protocol.mode == "md"
    cfg = IndentMDConfig.from_protocol(protocol, **opts.get("md_config", {})) if md else None
    conv, bulk_info = _bulk(model, candidate, structure, protocol.skin_A, protocol.lattice_parameter_A)
    vacuum = float(opts["vacuum_A"]) if opts.get("vacuum_A") is not None else protocol_vacuum_A(protocol, model.cutoff)
    slab = slab_for_protocol(protocol, conv, cutoff=model.cutoff, vacuum_A=vacuum, geometry=geom)
    masks = slab.masks()
    park_gap = max(PARK_GAP_A, protocol.initial_gap_A)  # the park height; the MD start (initial gap) is below it
    if md:  # the capped tip must fit below the cell top at its park and start heights: fail before the preparation
        check_indenter_cell(np.asarray(slab.atoms.cell.array, float).T, geom, (*slab.center_xy, slab.surface_z + park_gap))
    mc = with_indenter(model, geom, K_eV_per_A3=K)
    out_dir = Path(ctx.output_dir) if ctx.output_dir else None
    if md:
        s, momenta, prep = prepare_indentation_slab(mc, slab, fixed=masks["fixed"], fmax=protocol.relax.fmax_eV_per_A,
                                                    dt_fs=protocol.timestep_fs, temperature_K=protocol.temperature_K,
                                                    equilibration_ps=protocol.equilibration_ps, cfg=cfg, seed=protocol.seed,
                                                    park_gap_A=park_gap, relax_max_steps=protocol.relax.max_steps)
    else:
        from fracture_md.physics.relax import relax_positions

        s, info = relax_positions(mc, AtomicSystem.from_ase(slab.atoms), fixed=masks["fixed"], method="fire",
                                  fmax=protocol.relax.fmax_eV_per_A, newton_polish=False, skin=protocol.skin_A,
                                  max_steps=protocol.relax.max_steps)
        momenta, prep = None, {"relax": dict(converged=info.converged, steps=info.n_steps, fmax=info.fmax)}
    X0 = np.asarray(s.frac, float) @ np.asarray(s.box, float).T
    z_s0 = float(np.mean(X0[masks["top_layer"], 2]))
    xy = tuple(float(v) for v in slab.center_xy)
    box = np.asarray(s.box, float)
    L = (float(box[0, 0]), float(box[1, 1]))
    species = np.asarray(s.symbols)
    a_lat = protocol.lattice_parameter_A or float(np.linalg.norm(conv.cell.array[0]))
    engine_run = harness_engine_run(model.card.id, md_dtype=cfg.dtype if md else "float64",
                                    recipe_sha256=None if rec is None else rec.sha256)
    if md:
        sched = TipSchedule.from_protocol(protocol, xy, z_s0)
        cs = cfg.chunk_steps
        contact_by_step: dict[int, FrameData] = {}
        key_frames: dict[str, FrameData] = {}
        ck_dir = (out_dir / "checkpoint") if out_dir is not None else None
        shard_dir = (ck_dir / RUNNER_FRAMES_DIR) if ck_dir is not None else None
        if shard_dir is not None:
            shard_dir.mkdir(parents=True, exist_ok=True)

        def on_chunk(state, handle):
            step = int(state.step) if hasattr(state, "step") else 0
            if shard_dir is not None and step == cs:  # the first chunk of a run from scratch: earlier frames are stale
                import shutil

                shutil.rmtree(shard_dir, ignore_errors=True)
                shard_dir.mkdir(parents=True)
            t_ps = step * cfg.dt_fs * 1e-3
            apex = sched.apex(step * cfg.dt_fs)
            p = set_indenter(mc.params, apex_A=apex)
            c = np.asarray(contact_atoms(mc, state.frac, box, p), bool)
            X = np.asarray(state.frac, float) @ box.T
            fr = FrameData(step, t_ps, X[c], np.ones(int(c.sum()), bool), {}, False)
            contact_by_step[step] = fr
            if shard_dir is not None:
                _save_frame_shard(shard_dir / f"contact_{step:012d}.npz", fr)
            lab = _key_frame_label(sched, step, cs, cfg.dt_fs)
            if lab is not None:
                key_frames[lab] = FrameData(step, t_ps, X, c, {}, True, False, lab)
                if shard_dir is not None:
                    _save_frame_shard(shard_dir / f"key_{lab}.npz", key_frames[lab])

        fw = FrameWriter(out_dir / "indentation_traj.npz", every_ps=protocol.frame_every_ps) \
            if out_dir is not None and protocol.frame_every_ps else None
        r = run_indentation_md(mc, s, sched, masks=masks, cfg=cfg, momenta=momenta, surface_z0_A=z_s0, frame_writer=fw,
                               checkpoint_dir=ck_dir, restart=restart, chunk_callback=on_chunk,
                               meta={"run_id": opts.get("run_id", "harness")})
        n_chunks = int(r.diagnostics["n_chunks_total"])
        resumed = int(r.diagnostics["chunks_this_call"]) < n_chunks
        n_restored = 0
        if resumed:  # chunks before the checkpoint were not recomputed: their frames come from the shards
            exp_keys = {lab for k in range(1, n_chunks + 1)
                        if (lab := _key_frame_label(sched, k * cs, cs, cfg.dt_fs)) is not None}
            n_restored = _restore_runner_frames(shard_dir, contact_by_step, key_frames, n_chunks=n_chunks,
                                                chunk_steps=cs, expected_keys=exp_keys) if shard_dir is not None else 0
        traj = r.trajectory_path
        if traj is None and resumed and fw is not None and fw.path.exists():
            traj = fw.path  # the checkpoint was complete: the trajectory was assembled by the call that finished it
        Xf = np.asarray(r.frac, float) @ box.T
        full_frames: dict[str, FrameData] = {"initial": FrameData(0, 0.0, X0, None, {}, True, False, "initial"),
                                             **key_frames}
        full_frames["final"] = FrameData(int(r.logs["step"][-1]), float(r.logs["t_ps"][-1]), Xf, None, {}, True, False, "final")
        lg = r.logs
        ch = {k: np.asarray(lg[k]) for k in ("pen_fw_A", "T_core_K", "W_eV", "Q_eV", "ledger_eV", "n_contact", "step",
                                             "reaction_nN") if k in lg}
        frames_all = sorted(list(contact_by_step.values()) + list(full_frames.values()), key=lambda f: (f.time_ps, not f.full))
        rd = IndentationRunData(t_ps=np.asarray(lg["t_ps"]), h_A=np.asarray(lg["depth_A"]), load_nN=np.asarray(lg["load_nN"]),
                                segment=np.asarray(lg["segment"]).astype(str), species=species, surface_z0_A=z_s0,
                                axis_xy=xy, lateral_L_A=L, lattice_constant_A=a_lat, channels=ch,
                                frames=lambda: iter(frames_all), geometry=geom, tip_model="rigid_sdf",
                                ideal_area_coefficient=geom.ideal_area_coefficient or 0.0,
                                face_azimuths_deg=tuple(geom.azimuth_deg + 120.0 * i for i in range(3)),
                                fixed_top_z=float(X0[masks["fixed"], 2].max()), bath_top_z=float(X0[masks["bath"], 2].max()),
                                potential=model.name)
        cfg_an = AnalysisConfig(window_ps=protocol.block_average_ps, t_el_A=slab.elastic_thickness_A)
        an = analyze_indentation(rd, cfg_an)
        ledger = {**r.ledger_summary(),
                  **ledger_at_hmax(lg["depth_A"], lg["ledger_eV"], lg["W_eV"], segment=rd.segment,
                                   floor_eV=ledger_floor_eV(s.n_atoms, protocol.temperature_K, cfg.dt_fs))}
        try:
            momentum = r.momentum_balance()
        except ValueError:  # pragma: no cover - a resumed checkpoint of an older driver is refused earlier
            momentum = None
        raw_md = {"summary": r.summary(), "ledger": ledger, "trajectory": None if traj is None else str(traj),
                  "momentum_balance": momentum, "resumed": resumed, "frames_restored_from_shards": n_restored,
                  "restart": restart}
        temperature = protocol.temperature_K
    else:
        depths = list(np.arange(protocol.depth_step_A, protocol.max_depth_A + 1e-9, protocol.depth_step_A))
        unl = list(reversed(depths[:-1])) + [0.0] if protocol.unload else []
        qs = quasistatic_indentation(mc, s, fixed=masks["fixed"], apex_xy=xy, surface_z0=z_s0, depths_A=depths + unl,
                                     fmax=protocol.relax.fmax_eV_per_A, max_steps=protocol.relax.max_steps,
                                     skin=protocol.skin_A)
        rows = qs["rows"]
        seg = np.array(["load"] * len(depths) + ["unload"] * len(unl))
        Xf = np.asarray(qs["system"].frac, float) @ box.T
        imax = len(depths) - 1
        frames_all = [FrameData(0, 0.0, X0, None, {}, True, False, "initial")] + qs["frames"] + \
            [FrameData(imax, float(imax), qs["X_at"][imax], None, {}, True, False, "max_load")] + \
            ([FrameData(len(rows), float(len(rows) - 1), Xf, None, {}, True, False, "final")] if protocol.unload else [])
        rd = IndentationRunData(t_ps=np.arange(len(rows), dtype=float), h_A=np.array([x["h_A"] for x in rows]),
                                load_nN=np.array([x["load_nN"] for x in rows]), segment=seg, species=species,
                                surface_z0_A=z_s0, axis_xy=xy, lateral_L_A=L, lattice_constant_A=a_lat,
                                channels={"pen_fw_A": np.array([x["pen_fw_A"] for x in rows]),
                                          "n_contact": np.array([x["n_contact"] for x in rows]),
                                          "reaction_nN": np.array([x["reaction_nN"] for x in rows])},
                                frames=lambda: iter(frames_all), geometry=geom, tip_model="rigid_sdf",
                                ideal_area_coefficient=geom.ideal_area_coefficient or 0.0,
                                face_azimuths_deg=tuple(geom.azimuth_deg + 120.0 * i for i in range(3)),
                                fixed_top_z=float(X0[masks["fixed"], 2].max()), potential=model.name)
        an = analyze_indentation(rd, AnalysisConfig(window_ps=1.0, tau_ac_ps=0.1, t_el_A=slab.elastic_thickness_A,
                                                    fit_load_range_nN=(0.0, np.inf)))
        an["load"]["newton3"] = quasistatic_newton3(rd.load_nN, rd.channels["reaction_nN"], rd.segment)
        ledger = None
        raw_md = {"quasistatic_rows": rows}
        temperature = 0.0
    checks = analysis_checks(an, engine="jax", ledger=ledger, mode="md" if md else "quasistatic") + relaxation_checks(
        prep["relax"], fmax=protocol.relax.fmax_eV_per_A, quasistatic_rows=None if md else raw_md["quasistatic_rows"]) \
        + equilibration_checks(prep)
    prov = collect(requested_by=ctx.requested_by, potential_id=model.card.id, protocol_id=protocol.protocol_id(),
                   parent_ids=list(ctx.parent_ids), random_seeds={"md": int(protocol.seed)})
    orientation = slab.orientation
    method = MethodSpec(name=f"{'MD' if md else 'quasi-static'} indentation (JAX harness)", engine="jax_md",
                        protocol_id=protocol.protocol_id(), assumptions=list(MD_ASSUMPTIONS))

    def conditions_for(depth, est, rule):
        return protocol.conditions(depth=q(min(depth, protocol.max_depth_A), Unit.ANGSTROM), area_estimator=est,
                                   depth_zero_rule=rule, azimuth_reference=f"lab x = {orientation.x_direction}")

    recs = indentation_records(an, candidate=candidate, card=model.card, conditions_for=conditions_for,
                               engine_run=engine_run, provenance=prov, method=method, orientation=orientation,
                               checks=checks, evidence=list(ctx.extra.get("indentation_evidence", [])))
    contact = protocol.contact_conditions(depth=q(protocol.max_depth_A, Unit.ANGSTROM),
                                          azimuth_reference=f"lab x = {orientation.x_direction}")
    gaps = records_not_available(an)
    curve = indentation_curve(rd, an, mode="md" if md else "quasistatic", contact=contact, temperature_K=temperature,
                              potential_id=model.card.id, engine_run=engine_run, orientation=orientation,
                              seed=protocol.seed if md else None, curve_id=f"{candidate.id}:{protocol.protocol_id()}",
                              checks=checks, assumptions=list(MD_ASSUMPTIONS), notes=_not_available_note(gaps))
    raw = {"indentation_analysis": json_ready(an), "slab": {"n_atoms": slab.n_atoms, "sizing": json_ready(slab.sizing),
                                                           "surface_z0_A": z_s0, "vacuum_above_A": slab.vacuum_above_A},
           "preparation": json_ready(prep), "bulk": json_ready(bulk_info), "run": json_ready(raw_md),
           "records_not_available": gaps, "wall_s": time.monotonic() - t_wall,
           "recipe": None if rec is None else indent_recipe.manifest_entry(rec, protocol, preparation=harness_preparation(
               protocol, slab, langevin_tau_ps=cfg.thermostat_tau_ps if cfg is not None else None))}
    return ProtocolOutput(properties=recs, checks=checks, indentation_curves=[curve], structure=None,
                          artifacts={"trajectory": raw_md.get("trajectory")} if md else {}, raw=raw)


__all__ = [
    "E3_RADIUS_BIAS", "E3_RADIUS_BIAS_RANGE", "ENGINE_STATUS_CAP", "KOKKOS_GPU_ACCELERATOR", "LEDGER_FLOOR_EV_PER_ATOM",
    "LEDGER_FLOOR_LINEAR_MAX_ATOMS", "LEDGER_TOL", "NEWTON3_TOL", "NOT_APPLICABLE", "NOT_EVALUATED", "PARK_GAP_A",
    "PREP_LANGEVIN_FRACTION", "PREP_START_TEMPERATURE_FACTOR", "harness_preparation",
    "VACUUM_MARGIN_A", "analysis_checks", "e3_bias_statement", "e3_systematic_uncertainty", "equilibration_checks", "gate_checks",
    "harness_engine_run", "indentation_curve", "indentation_records", "is_not_evaluated", "lammps_accelerator",
    "lammps_contact_conditions", "lammps_cpu_check_applies", "lammps_lattice_constant", "lammps_ledger_atoms",
    "lammps_log_accelerator", "lammps_run_manifest", "ledger_at_hmax", "ledger_floor_atoms", "ledger_floor_eV",
    "make_md_record",
    "not_evaluated_check", "plateau_window_check", "prepare_indentation_slab", "protocol_vacuum_A",
    "quasistatic_indentation", "quasistatic_newton3", "records_from_lammps_run", "records_not_available",
    "relaxation_checks", "run_indentation",
]
