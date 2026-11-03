"""ASE / matscipy reference engine (``engine = "ase_reference"``).

A second, independent simulation backend.  It consumes the same
engine-independent protocol objects as the JAX-MD engine and returns the same
:class:`~fracture_md.engines.base.ProtocolOutput` schema (same PropertyRecord
names, units, shape descriptions and condition strings), but computes every
number with ASE calculators and ASE optimisers instead of JAX:

* energies, forces and stresses come from an ASE ``Calculator``: matscipy's
  analytic ``Manybody(TersoffBrenner(...))`` for Tersoff (1989) and for the
  legacy i-j Erhart-Albe variant; ``ase.calculators.tersoff.Tersoff`` (pure
  python, LAMMPS ``pair_style tersoff`` semantics) for the published i-k
  Erhart-Albe (2005) potential, because matscipy 1.3.0 implements only the i-j
  three-body indexing (docs/POTENTIALS.md); optionally mace-torch's
  ``MACECalculator``; a LAMMPS hook is provided;
* relaxations use ASE FIRE / BFGS / LBFGS, with ``FrechetCellFilter`` for the
  cell;
* elastic constants are finite differences of the calculator stress under
  ± strain per Voigt component, with internal coordinates relaxed at every
  strained cell (no Hessians, no autodiff);
* rigid cleavage tractions are central finite differences of the calculator
  energy, T(δ) = [E(δ+h) − E(δ−h)] / (2hA); the calculator's analytic forces
  give an internal consistency check.

Shared code is limited to geometry and post-processing: slab construction
(``structures.slabs`` — so both engines see the *same* slab), symmetry
analysis (``structures.symmetry``), the elastic post-processing
``physics.elastic.analyze_elastic`` (VRH, directional moduli) and the UBER fit
``physics.cleavage.fit_uber``.  Potential cards (validation domain) come from
``fracture_md.potentials`` — they are metadata, the JAX energy function is
never evaluated here.

Cross-engine comparison: put the other engine's output in
``context.extra["reference_output"]`` (a ProtocolOutput or a list of
PropertyRecords).  Every record whose :func:`record_key` matches gets a
``ValidationEvidence(kind="reference_implementation")`` and a
``cross_engine_agreement`` check (tolerances :data:`DEFAULT_TOLERANCES`,
overridable with ``context.extra["reference_tolerances"] = {name: tol}``).
A disagreement is a failed check (at least one implementation is wrong), so
the record becomes FAILED unless the potential is out of domain.

See docs/ENGINES.md for the engine contract and the measured agreement.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata as md
import importlib.util
import math
import os
import warnings
from typing import Any, Callable, Optional, Sequence

import numpy as np

from fracture_md.engines.base import ProtocolOutput, RunContext
from fracture_md.schema.common import CheckResult, Conditions, Uncertainty, UncertaintyKind, ValidationEvidence
from fracture_md.schema.properties import PropertyName, PropertyRecord
from fracture_md.units import AMU_PER_A3_TO_G_PER_CM3, EV_PER_A2_TO_J_PER_M2, EV_PER_A3_TO_GPA, Quantity, Unit

ENGINE_NAME = "ase_reference"
SUPPORTED_KINDS = ("bulk_relax", "elastic_tensor", "cleavage")


class CalculatorUnavailable(ImportError):
    """The backend of a registered calculator is not installed / configured."""


# ============================================================================
# calculator registry
# ============================================================================
@dataclasses.dataclass
class CalculatorSpec:
    """How to build an ASE calculator for one potential id, plus the metadata the runners need.

    ``card_potential`` names the ``fracture_md.potentials`` registry entry whose
    PotentialCard (validation domain, citation) describes the *same model*;
    records made with this calculator carry that card.  ``switching_A`` maps a
    sorted element pair to the cutoff-function interval (R1, R2) in Å (for the
    cutoff-artefact plausibility flag); ``cutoff_A`` is the interaction range
    used for slab construction (in-plane repeats, vacuum checks).
    """

    name: str
    backend: str  # "matscipy", "mace-torch", "lammps", ...
    factory: Callable[[], Any]
    description: str
    card_potential: Optional[str] = None
    cutoff_A: Optional[float] = None
    switching_A: Optional[dict[tuple[str, str], tuple[float, float]]] = None
    aliases: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()  # importable modules
    availability_fn: Optional[Callable[[], tuple[bool, str]]] = None

    def availability(self) -> tuple[bool, str]:
        if self.availability_fn is not None:
            return self.availability_fn()
        missing = [m for m in self.requires if importlib.util.find_spec(m) is None]
        if missing:
            return False, f"missing python module(s) {missing}"
        return True, "ok"

    def make(self):
        ok, why = self.availability()
        if not ok:
            raise CalculatorUnavailable(f"calculator {self.name!r} ({self.backend}) unavailable: {why}")
        return self.factory()


CALCULATORS: dict[str, CalculatorSpec] = {}


def register_calculator(spec: CalculatorSpec, *, overwrite: bool = False) -> CalculatorSpec:
    if spec.name in CALCULATORS and not overwrite:
        raise ValueError(f"calculator {spec.name!r} already registered")
    CALCULATORS[spec.name] = spec
    return spec


def resolve_calculator(potential: str) -> CalculatorSpec:
    """CalculatorSpec for a potential id: registry name, alias, or potential-card id (``name@hash`` accepted).

    A full card id (``name@hash``) is matched exactly first: two models can share the card-id prefix
    (``erhart-albe-2005-SiC@48d0ddf5d1`` i-k and the legacy ``erhart-albe-2005-SiC@504bf563e0`` i-j), and a
    record must be re-run with the model that produced it.  A bare prefix resolves to the registered default.
    """
    if potential in CALCULATORS:
        return CALCULATORS[potential]
    if "@" in potential:
        for spec in CALCULATORS.values():
            card = _card(spec)
            if card is not None and card.id == potential:
                return spec
    base = potential.split("@")[0]
    for spec in CALCULATORS.values():
        if base == spec.name or base in spec.aliases:
            return spec
    for spec in CALCULATORS.values():  # card id, e.g. "tersoff-1989-SiC@9e2ac85132"
        card = _card(spec)
        if card is not None and (card.id == potential or card.id.split("@")[0] == base):
            return spec
    raise KeyError(f"ase_reference engine has no calculator for potential {potential!r}; "
                   f"available: {sorted(CALCULATORS)}")


# ---------------------------------------------------------------- matscipy
# registry name -> (matscipy parameter set, card-id prefix accepted as alias or None).  matscipy 1.3.0
# explicit_forms indexes the Erhart-Albe three-body parameters by the i-j pair, so it implements the LEGACY
# harness variant, not the published potential (which is registered with an ASE Tersoff calculator below).
_MATSCIPY_SETS = {
    "tersoff_1989_SiC": ("Tersoff_PRB_39_5566_Si_C", "tersoff-1989-SiC"),
    "erhart_albe_2005_SiC_matscipy_ij": ("Erhart_PRB_71_035211_SiC", None),
}


def _matscipy_params(set_name: str) -> dict:
    from matscipy.calculators.manybody.explicit_forms import tersoff_brenner

    return getattr(tersoff_brenner, set_name)


def _matscipy_switching(set_name: str) -> dict[tuple[str, str], tuple[float, float]]:
    """Cutoff-function intervals of a matscipy Tersoff/Brenner parameter set (pairs in matscipy order)."""
    p = _matscipy_params(set_name)
    el = list(p["el"])
    pairs = [(el[i], el[j]) for i in range(len(el)) for j in range(i, len(el))]
    return {tuple(sorted(pr)): (float(r1), float(r2)) for pr, r1, r2 in zip(pairs, p["r1"], p["r2"])}


def matscipy_tersoff_spec(name: str, set_name: str, card_potential: Optional[str] = None,
                          aliases: tuple[str, ...] = ()) -> CalculatorSpec:
    """matscipy ``Manybody(**TersoffBrenner(<set_name>))`` — analytic energy/forces/stress in float64 numpy."""

    def factory():
        from matscipy.calculators.manybody import Manybody
        from matscipy.calculators.manybody.explicit_forms import TersoffBrenner

        return Manybody(**TersoffBrenner(_matscipy_params(set_name)))

    sw = cut = None
    if importlib.util.find_spec("matscipy") is not None:
        sw = _matscipy_switching(set_name)
        cut = max(r2 for _, r2 in sw.values())
    return CalculatorSpec(name=name, backend="matscipy", factory=factory, card_potential=card_potential,
                          description=f"matscipy Manybody(TersoffBrenner({set_name}))", cutoff_A=cut,
                          switching_A=sw, aliases=aliases, requires=("matscipy",))


# ---------------------------------------------------------------- ASE Tersoff (LAMMPS semantics)
def _tersoff_switching(params, elements: Sequence[str]) -> dict[tuple[str, str], tuple[float, float]]:
    """Cutoff-function intervals (R1, R2) per sorted element pair from harness Tersoff tables."""
    r1, r2 = np.asarray(params["R1"]), np.asarray(params["R2"])
    return {tuple(sorted((a, b))): (float(r1[i, j]), float(r2[i, j]))
            for i, a in enumerate(elements) for j, b in enumerate(elements)}


def ase_tersoff_spec(name: str, potential: str, card_potential: Optional[str] = None,
                     aliases: tuple[str, ...] = ()) -> CalculatorSpec:
    """``ase.calculators.tersoff.Tersoff`` with the LAMMPS-format file written from the harness tables of
    ``potential`` by :func:`fracture_md.potentials.tersoff.to_lammps_tersoff`.

    ASE's Tersoff is an independent pure-python implementation of LAMMPS ``pair_style tersoff`` (entries
    i j k: three-body gamma, c, d, h of entry (i, j, k), two-body of (i, j, j)), so it evaluates the
    *published* i-k Erhart-Albe convention.  The parameters are shared with the JAX model, but the export is
    tested against the LAMMPS-distributed ``SiC_Erhart-Albe.tersoff`` to its printed precision
    (tests/test_ea_three_body_convention.py).  Cost: ~0.1-0.4 s per energy+force call for ~100 atoms.
    """
    from fracture_md.potentials import get_potential

    model = get_potential(potential)
    elements = list(model.elements)

    def factory():
        from ase.calculators.tersoff import Tersoff, TersoffParameters

        from fracture_md.potentials.tersoff import to_lammps_tersoff

        text = to_lammps_tersoff(get_potential(potential).params, elements)
        tok = " ".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#")).split()
        # same 17-token blocks as Tersoff.read_lammps_format (e1 e2 e3 + 14 numbers), without a temporary file
        table = {tuple(tok[i:i + 3]): TersoffParameters(*map(float, tok[i + 3:i + 17])) for i in range(0, len(tok), 17)}
        return Tersoff(parameters=table)

    sw = _tersoff_switching(model.params, elements)
    return CalculatorSpec(name=name, backend="ase-tersoff", factory=factory, card_potential=card_potential,
                          description=f"ase.calculators.tersoff.Tersoff (LAMMPS pair_style tersoff semantics) on "
                                      f"to_lammps_tersoff({potential})",
                          cutoff_A=max(r2 for _, r2 in sw.values()), switching_A=sw, aliases=aliases,
                          requires=("ase.calculators.tersoff",))


# ---------------------------------------------------------------- mace-torch (optional)
def mace_torch_spec(name: str = "mace_mp0_small_torch", model: str = "small",
                    card_potential: Optional[str] = "mace_mp0_small_jax",
                    aliases: tuple[str, ...] = ("mace_mp0_small", "mace_mp0_small_jax")) -> CalculatorSpec:
    """mace-torch ``mace_mp(model=...)`` MACECalculator in float64 on CPU (same MACE-MP-0 checkpoint as the JAX port)."""

    def factory():
        from mace.calculators import mace_mp

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return mace_mp(model=model, device="cpu", default_dtype="float64")

    return CalculatorSpec(name=name, backend="mace-torch", factory=factory, card_potential=card_potential,
                          description=f"mace-torch MACECalculator, MACE-MP-0 {model}, float64, CPU", cutoff_A=6.0,
                          switching_A=None, aliases=aliases, requires=("mace", "torch"))


# ---------------------------------------------------------------- LAMMPS hook (not required)
def lammps_spec(name: str, *, pair_style: str, pair_coeff: Sequence[str], elements: Sequence[str],
                files: Sequence[str] = (), card_potential: Optional[str] = None, cutoff_A: Optional[float] = None,
                switching_A: Optional[dict] = None, description: str = "") -> CalculatorSpec:
    """Hook for a LAMMPS-backed calculator (``ase.calculators.lammpslib`` if the ``lammps`` python module is
    importable, else ``ase.calculators.lammpsrun`` if ``ASE_LAMMPSRUN_COMMAND`` is set).

    Example (not registered by default because LAMMPS is not a dependency)::

        register_calculator(lammps_spec("lammps:tersoff_1989_SiC", pair_style="tersoff",
                                        pair_coeff=["* * SiC.tersoff Si C"], elements=["Si", "C"],
                                        files=["/path/to/lammps/potentials/SiC.tersoff"],
                                        card_potential="tersoff_1989_SiC", cutoff_A=3.0))

    Before trusting such a spec, run the cross-engine check (examples/08) against
    the matscipy spec of the same model: parameter files and mixing rules differ
    between codes.
    """
    elements = list(elements)

    def availability() -> tuple[bool, str]:
        if importlib.util.find_spec("lammps") is not None:
            return True, "lammps python module (lammpslib)"
        if os.environ.get("ASE_LAMMPSRUN_COMMAND"):
            return True, f"lammpsrun via ASE_LAMMPSRUN_COMMAND={os.environ['ASE_LAMMPSRUN_COMMAND']}"
        return False, "neither the `lammps` python module nor ASE_LAMMPSRUN_COMMAND is available"

    def factory():
        if importlib.util.find_spec("lammps") is not None:
            from ase.calculators.lammpslib import LAMMPSlib

            cmds = [f"pair_style {pair_style}"] + [f"pair_coeff {c}" for c in pair_coeff]
            return LAMMPSlib(lmpcmds=cmds, atom_types={el: i + 1 for i, el in enumerate(elements)}, keep_alive=True,
                             log_file=None)
        from ase.calculators.lammpsrun import LAMMPS

        return LAMMPS(pair_style=pair_style, pair_coeff=list(pair_coeff), files=list(files), specorder=elements,
                      keep_tmp_files=False)

    return CalculatorSpec(name=name, backend="lammps", factory=factory, card_potential=card_potential,
                          description=description or f"LAMMPS pair_style {pair_style}", cutoff_A=cutoff_A,
                          switching_A=switching_A, availability_fn=availability)


for _name, (_set, _card_id) in _MATSCIPY_SETS.items():
    register_calculator(matscipy_tersoff_spec(_name, _set, card_potential=_name,
                                              aliases=(_card_id,) if _card_id else ()))
# published Erhart-Albe (2005), three-body parameters indexed by the i-k pair (LAMMPS/NIST/OpenKIM convention)
register_calculator(ase_tersoff_spec("erhart_albe_2005_SiC", "erhart_albe_2005_SiC",
                                     card_potential="erhart_albe_2005_SiC", aliases=("erhart-albe-2005-SiC",)))
register_calculator(mace_torch_spec())


_CARD_CACHE: dict[str, Any] = {}


def _card(spec: CalculatorSpec):
    """PotentialCard of the model this calculator implements (None if unavailable)."""
    if spec.card_potential is None:
        return None
    if spec.card_potential not in _CARD_CACHE:
        try:
            from fracture_md.potentials import get_potential

            _CARD_CACHE[spec.card_potential] = get_potential(spec.card_potential).card
        except Exception:  # noqa: BLE001 - optional potentials (MACE weights) may be missing
            _CARD_CACHE[spec.card_potential] = None
    return _CARD_CACHE[spec.card_potential]


def _cutoff(spec: CalculatorSpec) -> float:
    if spec.cutoff_A is not None:
        return float(spec.cutoff_A)
    card = _card(spec)
    if card is not None and card.cutoff is not None:
        return float(card.cutoff.magnitude(Unit.ANGSTROM))
    raise ValueError(f"calculator {spec.name!r} has no interaction range (cutoff_A)")


def _versions() -> str:
    out = []
    for p in ("ase", "matscipy", "mace-torch", "torch"):
        try:
            out.append(f"{p} {md.version(p)}")
        except md.PackageNotFoundError:
            pass
    return " / ".join(out)


# ============================================================================
# cross-engine comparison
# ============================================================================
#: absolute tolerances (in the record's own unit) for cross-engine agreement
DEFAULT_TOLERANCES: dict[PropertyName, float] = {
    PropertyName.LATTICE_PARAMETER: 1e-5,  # Å
    PropertyName.DENSITY: 1e-5,  # g/cm^3
    PropertyName.VOLUME_PER_ATOM: 1e-4,  # Å^3/atom
    PropertyName.ENERGY_PER_ATOM: 1e-6,  # eV/atom
    PropertyName.COHESIVE_ENERGY: 1e-6,  # eV/atom
    PropertyName.RESIDUAL_STRESS: 1e-3,  # GPa (both ~0)
    PropertyName.WORK_OF_SEPARATION: 1e-5,  # J/m^2 (same slab: ~1e-9; own lattice: a0 differences enter)
    PropertyName.SURFACE_ENERGY: 1e-5,
    PropertyName.SURFACE_ENERGY_AVERAGE_PAIR: 1e-5,
    PropertyName.G_IC_ATOMISTIC: 1e-5,
    PropertyName.PEAK_NORMAL_TRACTION: 0.05,  # GPa: FD of E at a C1 cutoff-function kink, h/2·|ΔT'|
    PropertyName.OPENING_AT_PEAK_TRACTION: 2e-3,  # Å
    PropertyName.UBER_LENGTH_SCALE: 1e-3,  # Å
}
#: elastic quantities in GPa use the protocol's fd_vs_autodiff_tolerance_GPa; these are dimensionless
_DIMENSIONLESS_TOL = 5e-3
_RELAXED_W_TOL = 2e-3  # J/m^2: independent local minimisers (ASE vs scipy/Newton) to fmax 1e-4 eV/Å


def record_key(r: PropertyRecord) -> tuple:
    """Engine-independent identity of a record: what was computed, not how."""
    plane = tuple(r.orientation.plane_miller) if (r.orientation is not None and r.orientation.plane_miller) else None
    term = r.interface.termination.label if (r.interface is not None and r.interface.termination is not None) else None
    deriv = tuple(r.derivation.assumptions) if r.derivation is not None else ()
    return (r.name.value, r.shape_description, r.conditions.stress_state, plane, term, deriv)


def _reference_records(context: RunContext) -> list[PropertyRecord]:
    ref = context.extra.get("reference_output")
    if ref is None:
        ref = context.extra.get("reference_records")
    if ref is None:
        return []
    if isinstance(ref, ProtocolOutput):
        return list(ref.properties)
    return [r for r in ref if isinstance(r, PropertyRecord)]


class _Comparator:
    """Looks up the other engine's record for each value and builds evidence + a check."""

    def __init__(self, context: RunContext, gpa_tol: Optional[float] = None):
        self.refs: dict[tuple, PropertyRecord] = {}
        for r in _reference_records(context):
            self.refs.setdefault(record_key(r), r)
        self.over = {PropertyName(k) if not isinstance(k, PropertyName) else k: float(v)
                     for k, v in context.extra.get("reference_tolerances", {}).items()}
        self.gpa_tol = gpa_tol
        self.rows: list[dict[str, Any]] = []

    def tolerance(self, name: PropertyName, unit: Unit, key: tuple) -> Optional[float]:
        if name in self.over:
            return self.over[name]
        relaxed_state = key[2].startswith("fully relaxed") or any("uses the relaxed" in a for a in key[5])
        if relaxed_state and name in (PropertyName.WORK_OF_SEPARATION, PropertyName.SURFACE_ENERGY,
                                      PropertyName.SURFACE_ENERGY_AVERAGE_PAIR, PropertyName.G_IC_ATOMISTIC):
            return _RELAXED_W_TOL
        if name in DEFAULT_TOLERANCES:
            return DEFAULT_TOLERANCES[name]
        if unit == Unit.GPA and self.gpa_tol is not None:
            return self.gpa_tol
        if unit == Unit.PER_GPA:
            return 1e-5
        if unit == Unit.DIMENSIONLESS:
            return _DIMENSIONLESS_TOL
        return None

    def __call__(self, key: tuple, name: PropertyName, value: Any, unit: Unit):
        ref = self.refs.get(key)
        if ref is None:
            return [], []
        tol = self.tolerance(name, unit, key)
        a = np.asarray(ref.value.as_array(unit), float)
        b = np.asarray(value, float)
        if tol is None or a.shape != b.shape:
            return [], []
        diff = float(np.max(np.abs(a - b)))
        rel = diff / max(float(np.max(np.abs(a))), 1e-300)
        ok = diff <= tol
        eng = ref.method.engine or "?"
        self.rows.append(dict(quantity=name.value, key=key, reference_engine=eng, max_abs_diff=diff, tolerance=tol,
                              unit=unit.value, passed=ok))
        ev = ValidationEvidence(
            kind="reference_implementation", passed=ok, relative_difference=rel,
            reference_value=ref.value, computed_value=Quantity(value=b.tolist() if b.ndim else float(b), unit=unit),
            description=f"{ENGINE_NAME} vs engine {eng!r} ({ref.method.name}; record {ref.id}): max |Δ| = {diff:.3e} "
                        f"{unit.value} (tolerance {tol:g}); independent implementation of the same model")
        ck = CheckResult(name="cross_engine_agreement", passed=ok, metric=diff, tolerance=tol, unit=unit,
                         details=f"vs {eng} record {ref.id}")
        return [ev], [ck]


def compare_outputs(computed: ProtocolOutput, reference: ProtocolOutput, tolerances: Optional[dict] = None,
                    gpa_tol: float = 0.1) -> list[dict[str, Any]]:
    """Table of matched records (same :func:`record_key`) of two engines' outputs, with max |Δ| and pass/fail."""
    ctx = RunContext(extra={"reference_output": reference, "reference_tolerances": tolerances or {}})
    cmp = _Comparator(ctx, gpa_tol=gpa_tol)
    for r in computed.properties:
        cmp(record_key(r), r.name, r.value.as_array(r.value.unit), r.value.unit)
    return cmp.rows


# ============================================================================
# numerical kernels (ASE only)
# ============================================================================
#: systems with more atoms than this use LBFGS for "auto"/"engine_default" (ASE BFGS diagonalises a dense
#: 3N x 3N matrix every step, which is slow for slabs, especially with oversubscribed BLAS threads)
BFGS_MAX_ATOMS = 32


def _optimizer(name: str, n_atoms: int):
    from ase.optimize import BFGS, FIRE, LBFGS

    if name in ("auto", "engine_default", "cg"):
        return BFGS if n_atoms <= BFGS_MAX_ATOMS else LBFGS
    return {"fire": FIRE, "bfgs": BFGS, "lbfgs": LBFGS}.get(name, LBFGS)


def _fmax_of(atoms) -> float:
    f = atoms.get_forces()
    return float(np.max(np.linalg.norm(f, axis=1))) if len(f) else 0.0


def relax_atoms(atoms, calc, *, fmax: float, max_steps: int = 20000, optimizer: str = "auto",
                relax_cell: bool = False, cell_mode: str = "full", stress_tol_GPa: Optional[float] = None):
    """Relax an ASE Atoms (positions, optionally the cell via FrechetCellFilter) with an ASE optimiser.

    For cell relaxation the optimiser's force criterion is tightened so that the
    generalized cell forces (virial / N) imply |σ| well below ``stress_tol_GPa``.
    ``optimizer`` is a RelaxSettings name or "auto" (BFGS up to BFGS_MAX_ATOMS atoms, else LBFGS); an
    optimiser that does not converge is followed by a quasi-Newton polish (BFGS/LBFGS).
    Returns (relaxed copy, info dict).
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS, LBFGS

    at = atoms.copy()
    at.calc = calc
    f_eff = float(fmax)
    if relax_cell:
        target = FrechetCellFilter(at, hydrostatic_strain=(cell_mode == "isotropic"))
        if stress_tol_GPa is not None:
            f_eff = min(f_eff, 0.1 * stress_tol_GPa / EV_PER_A3_TO_GPA * at.get_volume() / len(at))
    else:
        target = at
    cls = _optimizer(optimizer, len(at))
    opt = cls(target, logfile=None)
    ok = bool(opt.run(fmax=f_eff, steps=max_steps))
    steps = int(opt.nsteps)
    used = cls.__name__
    if not ok:
        pol = _optimizer("auto", len(at))
        if pol is cls:
            pol = LBFGS if cls is BFGS else BFGS
        opt2 = pol(target, logfile=None)
        ok = bool(opt2.run(fmax=f_eff, steps=max_steps))
        steps += int(opt2.nsteps)
        used += "+" + pol.__name__
    fm = _fmax_of(at)
    sig = at.get_stress(voigt=True) * EV_PER_A3_TO_GPA
    at.calc = calc
    return at, dict(converged=ok and fm <= fmax * 1.0001, n_steps=steps, fmax=fm, optimizer=used,
                    optimizer_fmax=f_eff, energy=float(at.get_potential_energy()), stress_GPa=sig.tolist(),
                    filter="FrechetCellFilter" + ("(hydrostatic)" if cell_mode == "isotropic" else "") if relax_cell
                    else "none")


def _eps_from_voigt(e: np.ndarray) -> np.ndarray:
    return np.array([[e[0], e[5] / 2, e[4] / 2], [e[5] / 2, e[1], e[3] / 2], [e[4] / 2, e[3] / 2, e[2]]])


def strained_copy(atoms, e_voigt: np.ndarray):
    """Affinely strained copy: every lattice vector (and position) x -> (I + ε) x, engineering shear strains."""
    F = np.eye(3) + _eps_from_voigt(np.asarray(e_voigt, float))
    at = atoms.copy()
    at.set_cell(atoms.cell.array @ F.T, scale_atoms=True)
    return at


def fd_elastic_constants(atoms, calc, strains: Sequence[float], *, relax_internal: bool = True, fmax: float = 1e-8,
                         max_steps: int = 2000, optimizer: str = "auto") -> dict[str, Any]:
    """C_ij = [σ_i(+δ e_j) − σ_i(−δ e_j)] / (2δ) from the calculator stress, internal coordinates relaxed.

    Returns per-magnitude C (GPa), the internal-displacement response Γ
    (Å per unit strain, mean removed) and the worst residual force.
    """
    n = len(atoms)
    per: dict[float, np.ndarray] = {}
    gam: dict[float, np.ndarray] = {}
    worst, steps, notconv = 0.0, 0, 0
    for d in [float(s) for s in strains]:
        C = np.zeros((6, 6))
        G = np.zeros((n, 3, 6))
        for j in range(6):
            sig, disp = [], []
            for sgn in (+1.0, -1.0):
                e = np.zeros(6)
                e[j] = sgn * d
                at = strained_copy(atoms, e)
                aff = at.get_positions()
                at.calc = calc
                if relax_internal:
                    at, info = relax_atoms(at, calc, fmax=fmax, max_steps=max_steps, optimizer=optimizer)
                    worst = max(worst, info["fmax"])
                    steps += info["n_steps"]
                    notconv += 0 if info["converged"] else 1
                sig.append(at.get_stress(voigt=True))
                u = at.get_positions() - aff
                disp.append(u - u.mean(axis=0))
            C[:, j] = (sig[0] - sig[1]) / (2 * d)
            G[:, :, j] = (disp[0] - disp[1]) / (2 * d)
        per[d] = C * EV_PER_A3_TO_GPA
        gam[d] = G
    return dict(per_magnitude=per, gamma=gam, max_residual_force=worst, n_steps=steps, n_not_converged=notconv)


def _slab_normal_area(atoms) -> tuple[np.ndarray, float]:
    c = np.cross(atoms.cell.array[0], atoms.cell.array[1])
    a = float(np.linalg.norm(c))
    return c / a, a


def _block_positions(x0: np.ndarray, upper: np.ndarray, disp: np.ndarray) -> np.ndarray:
    x = x0.copy()
    x[upper] += disp
    return x


class _RigidOpening:
    """E(δ), FD traction and force traction of a rigidly displaced upper block (ASE calculator)."""

    def __init__(self, slab_atoms, calc, upper, fd_step: float):
        self.at = slab_atoms.copy()
        self.at.calc = calc
        self.x0 = self.at.get_positions().copy()
        self.upper = np.asarray(upper, bool)
        self.n, self.area = _slab_normal_area(self.at)
        self.h = float(fd_step)
        self.n_calls = 0

    def _set(self, delta: float):
        self.at.set_positions(_block_positions(self.x0, self.upper, delta * self.n), apply_constraint=False)

    def energy(self, delta: float) -> float:
        self._set(delta)
        self.n_calls += 1
        return float(self.at.get_potential_energy())

    def energy_force_traction(self, delta: float) -> tuple[float, float]:
        """(E, T_forces) with T_forces = −(1/A) Σ_upper F·n in GPa (calculator's analytic forces)."""
        self._set(delta)
        self.n_calls += 1
        e = float(self.at.get_potential_energy())
        f = self.at.get_forces()
        return e, float(-np.sum(f[self.upper] @ self.n)) / self.area * EV_PER_A3_TO_GPA

    def traction_fd(self, delta: float) -> float:
        """Central FD of the energy, (E(δ+h) − E(δ−h)) / (2hA), in GPa."""
        return (self.energy(delta + self.h) - self.energy(delta - self.h)) / (2 * self.h) / self.area * EV_PER_A3_TO_GPA

    def scan(self, grid: Sequence[float]) -> dict[str, np.ndarray]:
        E, TF, TD = [], [], []
        for d in grid:
            e, tf = self.energy_force_traction(float(d))
            E.append(e)
            TF.append(tf)
            TD.append(self.traction_fd(float(d)))
        E = np.asarray(E)
        return dict(openings=np.asarray(grid, float), energy_eV=E,
                    energy_per_area=(E - E[0]) / self.area * EV_PER_A2_TO_J_PER_M2,
                    traction=np.asarray(TD), traction_forces=np.asarray(TF))

    def atoms_at(self, delta: float):
        self._set(delta)
        return self.at.copy()


def cross_gap_pairs_ase(atoms, upper, rmax: float) -> list[tuple[int, int, float]]:
    """(i_lower, j_upper, r) for pairs across the cut with r < rmax (ASE neighbour list, all images)."""
    from ase.neighborlist import neighbor_list

    up = np.asarray(upper, bool)
    ii, jj, dd = neighbor_list("ijd", atoms, rmax)
    sel = (~up[ii]) & up[jj]
    return [(int(i), int(j), float(d)) for i, j, d in zip(ii[sel], jj[sel], dd[sel])]


def cutoff_artifact_flag(spec: CalculatorSpec, atoms_at_peak, upper, label: str) -> dict[str, Any]:
    """Are cross-gap pairs inside the calculator's switching interval (R1, R2) at the peak-traction opening?"""
    if not spec.switching_A:
        return dict(flagged=False, applicable=False, n_pairs_in_switching_region=0,
                    details="calculator has no tabulated switching interval; check not applicable")
    rmax = max(r2 for _, r2 in spec.switching_A.values()) + 1e-9
    sym = atoms_at_peak.get_chemical_symbols()
    in_sw, bonded = [], 0
    pairs = cross_gap_pairs_ase(atoms_at_peak, upper, rmax)
    for i, j, r in pairs:
        r1, r2 = spec.switching_A.get(tuple(sorted((sym[i], sym[j]))), (np.inf, np.inf))
        if r1 < r < r2:
            in_sw.append((sym[i], sym[j], r, r1, r2))
        elif r <= r1:
            bonded += 1
    if in_sw:
        rs = [x[2] for x in in_sw]
        details = (f"peak traction set by the cutoff function: at the {label} opening, {len(in_sw)} cross-gap pairs "
                   f"(of {len(pairs)} within max R2) lie inside the switching interval (r = {min(rs):.3f}-{max(rs):.3f} Å; "
                   f"e.g. {in_sw[0][0]}-{in_sw[0][1]} R1={in_sw[0][3]:.3f}, R2={in_sw[0][4]:.3f} Å) [ASE neighbour list]")
    else:
        details = f"no cross-gap pair inside a switching interval at the {label} opening ({bonded} fully bonded pairs)"
    return dict(flagged=bool(in_sw), applicable=True, n_pairs_in_switching_region=len(in_sw), n_bonded_pairs=bonded,
                n_pairs_considered=len(pairs), details=details)


# ============================================================================
# shared runner helpers
# ============================================================================
def _check(name, passed, metric=None, tol=None, unit=None, details="") -> CheckResult:
    return CheckResult(name=name, passed=bool(passed),
                       metric=None if metric is None or not np.isfinite(metric) else float(metric),
                       tolerance=None if tol is None else float(tol), unit=unit, details=details)


def _provenance(spec: CalculatorSpec, protocol, context: RunContext):
    import sys

    from fracture_md import provenance

    card = _card(spec)
    try:
        own = provenance.source_sha256(sys.modules[__name__])
    except (OSError, TypeError):  # pragma: no cover
        own = "unknown"
    return provenance.collect(
        engine=ENGINE_NAME, engine_version=_versions(), potential_id=card.id if card is not None else spec.name,
        protocol_id=protocol.protocol_id(), requested_by=context.requested_by, parent_ids=list(context.parent_ids),
        float_precision="float64", device="cpu (numpy/ASE)",
        implementation_ids={**context.implementation_ids, "fracture_md.engines.ase_reference": own,
                            "calculator": f"{spec.name} [{spec.backend}: {spec.description}]"},
    )


def _method(name: str, protocol, description: str, assumptions: list[str]):
    from fracture_md.schema.properties import MethodSpec

    return MethodSpec(name=name, protocol_id=protocol.protocol_id(), engine=ENGINE_NAME, description=description,
                      assumptions=assumptions)


def _starting_atoms(candidate, structure, supercell=None):
    from ase import Atoms

    from fracture_md.structures.system import AtomicSystem

    if isinstance(structure, AtomicSystem):
        atoms = structure.to_ase()
    elif isinstance(structure, Atoms):
        atoms = structure.copy()
    elif candidate.structure is not None:
        r = candidate.structure
        atoms = Atoms(symbols=r.symbols, positions=r.positions_angstrom, cell=r.cell_rows_angstrom, pbc=r.pbc)
    elif (candidate.phases and "3C-SiC" in candidate.phases[0].name) or candidate.composition.formula == "SiC":
        from fracture_md.structures.builders import sic_3c_conventional

        atoms = sic_3c_conventional()
    else:
        raise ValueError(f"no structure for candidate {candidate.id!r}: pass `structure` or set candidate.structure")
    if supercell is not None and tuple(supercell) != (1, 1, 1):
        atoms = atoms.repeat(tuple(supercell))
    atoms.pbc = True
    return atoms


def _isolated_atom_energy(calc, element: str, cutoff: float) -> float:
    from ase import Atoms

    L = 2.0 * (cutoff + 1.0) + 1.0
    a = Atoms(element, positions=[[L / 2] * 3], cell=[L, L, L], pbc=True)
    a.calc = calc
    return float(a.get_potential_energy())


# ============================================================================
# protocol runners: run_<kind>(protocol, candidate, spec, structure=None, context=None)
# ============================================================================
def run_bulk_relax(protocol, candidate, spec: CalculatorSpec, structure=None, context=None) -> ProtocolOutput:
    """BulkRelaxProtocol with ASE: FrechetCellFilter + ASE optimiser; same records as the JAX runner."""
    from fracture_md.records import make_record
    from fracture_md.structures.symmetry import conventional_cell, lattice_parameters, space_group
    from fracture_md.structures.system import AtomicSystem

    context = context or RunContext()
    calc = spec.make()
    card = _card(spec)
    atoms0 = _starting_atoms(candidate, structure, protocol.supercell)
    fmax = protocol.relax.fmax_eV_per_A
    at, info = relax_atoms(atoms0, calc, fmax=fmax, max_steps=protocol.relax.max_steps,
                           optimizer=protocol.relax.optimizer, relax_cell=protocol.relax_cell,
                           cell_mode=protocol.cell_mode, stress_tol_GPa=protocol.stress_tol_GPa)
    e_tot = float(at.get_potential_energy())
    fm = _fmax_of(at)
    sig = at.get_stress(voigt=True) * EV_PER_A3_TO_GPA
    n = len(at)
    V = float(at.get_volume())
    rho = float(np.sum(at.get_masses())) / V * AMU_PER_A3_TO_G_PER_CM3
    syms = at.get_chemical_symbols()
    cut = _cutoff(spec)
    e_iso = {el: _isolated_atom_energy(calc, el, cut) for el in sorted(set(syms))}
    e_coh = (sum(e_iso[x] for x in syms) - e_tot) / n

    s = AtomicSystem.from_ase(at)
    sg = space_group(s)
    conv = conventional_cell(s)
    lengths, angles = lattice_parameters(np.asarray(conv.box if conv is not None else s.box))
    cubic = sg.crystal_system == "cubic"

    checks = [
        _check("relaxation_converged", info["converged"], None, None, None,
               f"ASE {info['optimizer']} on {info['filter']}, {info['n_steps']} steps, optimiser fmax "
               f"{info['optimizer_fmax']:.2e}"),
        _check("max_force", fm <= max(fmax, 1e-12) * 1.0001, fm, fmax, Unit.EV_PER_ANGSTROM),
    ]
    if protocol.relax_cell:
        smax = float(np.max(np.abs(sig)))
        crit = smax if protocol.cell_mode == "full" else abs(float(np.mean(sig[:3])))
        checks.append(_check("residual_stress", crit < protocol.stress_tol_GPa, crit, protocol.stress_tol_GPa, Unit.GPA,
                             "max |σ_i|" if protocol.cell_mode == "full" else "|pressure|"))
    expected = candidate.phases[0].space_group_number if candidate.phases else None
    if expected is not None:
        checks.append(_check("space_group", sg.number == expected, float(sg.number), 0.0, None,
                             f"spglib: {sg.international} (#{sg.number}); expected #{expected}"))
    if cubic:
        dev = float(np.max(np.abs(lengths - lengths.mean())))
        checks.append(_check("cubic_cell_metric", dev < 1e-6 and float(np.max(np.abs(angles - 90))) < 1e-6, dev, 1e-6,
                             Unit.ANGSTROM, f"conventional a,b,c = {lengths.round(8).tolist()}"))

    prov = _provenance(spec, protocol, context)
    method = _method("zero-stress bulk relaxation", protocol,
                     f"ASE {info['optimizer']} + {info['filter']} with {spec.description}",
                     ["0 K lattice statics (no zero-point or thermal expansion)", "defect-free periodic crystal",
                      f"{n}-atom periodic cell (ASE/matscipy neighbour lists include all periodic images)"])
    cond = Conditions(stress_state="zero stress" if protocol.relax_cell else "fixed cell")
    cmp = _Comparator(context)

    def rec(name, value, unit, shape="", notes=""):
        key = (name.value, shape, cond.stress_state, None, None, ())
        ev, ck = cmp(key, name, value, unit)
        return make_record(name=name, value=value, unit=unit, candidate=candidate, method=method, card=card,
                           conditions=cond, checks=checks + ck, evidence=ev, frame="crystal" if shape else "not_applicable",
                           shape_description=shape, notes=notes, provenance=prov)

    if cubic:
        lat = rec(PropertyName.LATTICE_PARAMETER, float(lengths.mean()), Unit.ANGSTROM,
                  notes=f"conventional cubic a of {sg.international}")
    else:
        lat = rec(PropertyName.LATTICE_PARAMETER, lengths.tolist(), Unit.ANGSTROM, shape="conventional [a, b, c]",
                  notes=f"angles = {angles.round(6).tolist()} degree; {sg.international}")
    props = [
        lat,
        rec(PropertyName.DENSITY, rho, Unit.G_PER_CM3, notes="standard atomic masses (ASE) / relaxed volume"),
        rec(PropertyName.VOLUME_PER_ATOM, V / n, Unit.ANGSTROM3_PER_ATOM),
        rec(PropertyName.ENERGY_PER_ATOM, e_tot / n, Unit.EV_PER_ATOM, notes="relative to the potential's own energy zero"),
        rec(PropertyName.COHESIVE_ENERGY, e_coh, Unit.EV_PER_ATOM,
            notes=f"E_isolated_atoms - E_bulk per atom; isolated atoms with the same calculator ({e_iso} eV)"),
        rec(PropertyName.RESIDUAL_STRESS, sig.tolist(), Unit.GPA, shape="voigt 6 (xx,yy,zz,yz,xz,xy), positive = tensile"),
    ]
    out = ProtocolOutput(properties=props, checks=checks + [c for r in props for c in r.validation.checks
                                                            if c.name == "cross_engine_agreement"],
                         structure=s)
    out.artifacts.update(stress_GPa=sig, atoms=at, lattice_lengths_A=lengths, lattice_angles_deg=angles,
                         isolated_atom_energies_eV=e_iso)
    out.raw.update(relax_info=info, space_group={"number": sg.number, "symbol": sg.international}, n_atoms=n,
                   energy_eV=e_tot, cohesive_energy_eV_per_atom=e_coh, density_g_cm3=rho,
                   lattice_a_A=float(lengths.mean()) if cubic else lengths.tolist(), calculator=spec.description,
                   cross_engine=cmp.rows)
    return out


def run_elastic_tensor(protocol, candidate, spec: CalculatorSpec, structure=None, context=None) -> ProtocolOutput:
    """ElasticTensorProtocol with ASE: FD stress-strain (±δ per Voigt component), internal coordinates relaxed.

    The "autodiff_relaxed_ion" method is not available on this engine; the FD
    tensor is the result.  Post-processing (VRH, directional moduli, Born) uses
    ``physics.elastic.analyze_elastic`` so both engines derive moduli identically.
    """
    from fracture_md.physics.elastic import (HARDNESS_CHEN_NIU, HARDNESS_TIAN, analyze_elastic, chen_niu_score_uncertainty,
                                             kleinman_parameter)
    from fracture_md.records import make_record
    from fracture_md.schema.material import Orientation
    from fracture_md.schema.properties import Derivation, PropertyOrigin
    from fracture_md.schema.protocol import BulkRelaxProtocol
    from fracture_md.structures.symmetry import (
        VOIGT_LABELS,
        check_tensor_invariance,
        conventional_cell,
        cubic_pattern_deviation,
        lattice_parameters,
        space_group,
        voigt_to_tensor_stiffness,
    )
    from fracture_md.structures.system import AtomicSystem

    context = context or RunContext()
    calc = spec.make()
    card = _card(spec)
    out = ProtocolOutput()
    if structure is None:
        rel = run_bulk_relax(BulkRelaxProtocol(relax=protocol.relax), candidate, spec, context=context)
        out = out.merge(rel)
        atoms = rel.artifacts["atoms"]
    else:
        atoms = _starting_atoms(candidate, structure)
    atoms = atoms.copy()
    atoms.calc = calc
    ftol = protocol.fd_vs_autodiff_tolerance_GPa
    tol = protocol.symmetry_tolerance_GPa
    fmax_fd = min(protocol.relax.fmax_eV_per_A, 1e-7) * 0.1
    checks: list[CheckResult] = []

    f0 = _fmax_of(atoms)
    sig0 = atoms.get_stress(voigt=True) * EV_PER_A3_TO_GPA
    st = float(np.max(np.abs(sig0)))
    fmax_tol = max(10 * protocol.relax.fmax_eV_per_A, 1e-6)
    checks.append(_check("equilibrium_zero_force", f0 < fmax_tol, f0, fmax_tol, Unit.EV_PER_ANGSTROM))
    checks.append(_check("equilibrium_zero_stress", st < 1e-3, st, 1e-3, Unit.GPA,
                         "C = dσ/de at zero initial stress"))

    mags = [float(m) for m in protocol.strain_magnitudes]
    fd = fd_elastic_constants(atoms, calc, mags, relax_internal=protocol.relax_internal, fmax=fmax_fd)
    fdc = fd_elastic_constants(atoms, calc, mags, relax_internal=False)
    primary = sorted(mags)[len(mags) // 2]
    C = fd["per_magnitude"][primary]
    Cc = fdc["per_magnitude"][primary]
    spread = max(float(np.max(np.abs(Ci - C))) for Ci in fd["per_magnitude"].values())
    if protocol.relax_internal:
        checks.append(_check("fd_internal_relaxation_converged", fd["n_not_converged"] == 0 and
                             fd["max_residual_force"] <= fmax_fd * 1.0001, fd["max_residual_force"], fmax_fd,
                             Unit.EV_PER_ANGSTROM, f"ASE quasi-Newton at {12 * len(mags)} strained cells, {fd['n_steps']} steps"))
    checks.append(_check("fd_strain_magnitude_convergence", spread < ftol, spread, ftol, Unit.GPA,
                         f"max |C(δ)-C(δ*={primary:g})| over δ in {mags}"))
    s = AtomicSystem.from_ase(atoms)
    sg = space_group(s)
    rot = sg.rotations_cart
    for label, M in (("relaxed", C), ("clamped", Cc)):
        asym = float(np.max(np.abs(M - M.T)))
        checks.append(_check(f"fd_{label}_major_symmetry", asym < tol, asym, tol, Unit.GPA))
        inv = check_tensor_invariance(M, rot)
        checks.append(_check(f"fd_{label}_point_group_invariance", inv < tol, inv, tol, Unit.GPA,
                             f"{len(rot)} Cartesian operations of {sg.international} (#{sg.number})"))
        if sg.number >= 195:
            cp = cubic_pattern_deviation(M)
            checks.append(_check(f"fd_{label}_cubic_pattern", cp < tol, cp, tol, Unit.GPA))
    C_an = None
    if spec.backend == "matscipy" and hasattr(calc, "get_elastic_constants"):
        try:
            from matscipy.elasticity import full_3x3x3x3_to_Voigt_6x6

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                C4 = calc.get_elastic_constants(atoms, cg_parameters={"x0": None, "rtol": 1e-12, "maxiter": None,
                                                                      "M": None, "callback": None, "atol": 1e-14})
            C_an = np.asarray(full_3x3x3x3_to_Voigt_6x6(C4)) * EV_PER_A3_TO_GPA
            dan = float(np.max(np.abs(C_an - C)))
            checks.append(_check("fd_vs_calculator_analytic_relaxed_C", dan < ftol, dan, ftol, Unit.GPA,
                                 "same calculator's analytic Born + non-affine elastic constants (matscipy)"))
        except Exception as exc:  # noqa: BLE001 - optional internal check
            out.raw["calculator_analytic_error"] = repr(exc)

    Cs = 0.5 * (C + C.T)
    Ccs = 0.5 * (Cc + Cc.T)
    rho = float(np.sum(atoms.get_masses())) / atoms.get_volume() * AMU_PER_A3_TO_G_PER_CM3
    an = analyze_elastic(Cs, density_g_cm3=rho)
    conv = conventional_cell(s)
    conv_a = float(lattice_parameters(np.asarray(conv.box))[0][0]) if conv is not None else None
    zeta = None
    if sg.number == 216 and conv_a is not None and protocol.relax_internal:
        zeta = kleinman_parameter(fd["gamma"][primary], np.asarray(s.species), conv_a)

    prov = _provenance(spec, protocol, context)
    cond = Conditions(stress_state="zero stress, 0 K lattice statics")
    crystal = Orientation(description="crystal Cartesian frame (conventional cubic axes for cubic phases)")
    method = _method("finite-difference relaxed-ion elastic tensor", protocol,
                     f"C_ij = [σ_i(+δe_j) − σ_i(−δe_j)]/(2δ) from {spec.description} stresses, δ = {primary:g}, "
                     f"internal coordinates relaxed with ASE BFGS/LBFGS to {fmax_fd:.0e} eV/Å",
                     ["0 K, static", "zero initial stress and force", "linear elasticity (small strain)",
                      f"{len(atoms)}-atom periodic cell", "central differences (O(δ²) truncation)"])
    unc = [Uncertainty(kind=UncertaintyKind.NUMERICAL_CONVERGENCE, value=spread, unit=Unit.GPA, interval="max_abs_diff",
                       description=f"FD spread over strain magnitudes {mags}")]
    cmp = _Comparator(context, gpa_tol=ftol)
    recs: list[PropertyRecord] = []

    def rec(name, value, unit, *, shape="", frame="crystal", orientation=crystal, chk=None, notes="", unc_=None,
            origin=PropertyOrigin.ATOMISTIC_COMPUTED, derivation=None, meth=None):
        key = (name.value, shape, cond.stress_state, None, None, tuple(derivation.assumptions) if derivation else ())
        ev, ck = cmp(key, name, value, unit)
        r = make_record(name=name, value=value, unit=unit, candidate=candidate, method=meth or method, card=card,
                        conditions=cond, orientation=orientation, uncertainty=list(unc if unc_ is None else unc_),
                        checks=list(checks if chk is None else chk) + ck, evidence=ev, frame=frame,
                        shape_description=shape, notes=notes, provenance=prov, origin=origin, derivation=derivation)
        recs.append(r)
        return r

    vo = "voigt 6x6, order " + ",".join(VOIGT_LABELS) + "; engineering shear strains"
    rec(PropertyName.ELASTIC_STIFFNESS_VOIGT, Cs, Unit.GPA, shape=vo,
        notes=f"C11={an.get('C11', Cs[0, 0]):.3f}, C12={an.get('C12', Cs[0, 1]):.3f}, C44={an.get('C44', Cs[3, 3]):.3f} "
              "GPa (relaxed ion, finite differences, symmetrised)")
    rec(PropertyName.ELASTIC_COMPLIANCE_VOIGT, np.asarray(an["S_per_GPa"]), Unit.PER_GPA, shape=vo + " (compliance, S = C^-1)")
    rec(PropertyName.CLAMPED_ION_STIFFNESS_VOIGT, Ccs, Unit.GPA, shape=vo + " (clamped ion, no internal relaxation)",
        chk=[c for c in checks if "relaxed" not in c.name or c.name.startswith("equilibrium")])
    rec(PropertyName.ELASTIC_STIFFNESS_TENSOR, voigt_to_tensor_stiffness(Cs), Unit.GPA, shape="3x3x3x3 C_ijkl")
    polycrystal = "isotropic untextured dense polycrystal (Voigt-Reuss-Hill average of the single-crystal C)"
    mod = {}
    for pn, k in [(PropertyName.BULK_MODULUS_VOIGT, "B_voigt"), (PropertyName.BULK_MODULUS_REUSS, "B_reuss"),
                  (PropertyName.BULK_MODULUS_HILL, "B_hill"), (PropertyName.SHEAR_MODULUS_VOIGT, "G_voigt"),
                  (PropertyName.SHEAR_MODULUS_REUSS, "G_reuss"), (PropertyName.SHEAR_MODULUS_HILL, "G_hill"),
                  (PropertyName.YOUNGS_MODULUS_HILL, "E_hill")]:
        mod[k] = rec(pn, float(an[k]), Unit.GPA, frame="not_applicable", orientation=None, notes=polycrystal)
    rec(PropertyName.POISSON_RATIO_HILL, float(an["nu_hill"]), Unit.DIMENSIONLESS, frame="not_applicable",
        orientation=None, notes=polycrystal)
    rec(PropertyName.YOUNGS_MODULUS_DIRECTIONAL, [an["E_directional"][k] for k in ("[100]", "[110]", "[111]")], Unit.GPA,
        shape="E(n) for n = [100], [110], [111] (crystal frame)",
        orientation=Orientation(description="directions [100],[110],[111]"))
    rec(PropertyName.YOUNGS_MODULUS_DIRECTIONAL, [an["E_min"], an["E_max"]], Unit.GPA,
        shape="[E_min, E_max] over the unit sphere", orientation=Orientation(description="extrema of E(n) over the unit sphere"))
    rec(PropertyName.SHEAR_MODULUS_DIRECTIONAL, list(an["G_directional"].values()), Unit.GPA,
        shape="G(plane normal n, shear direction m) for " + ", ".join(an["G_directional"].keys()),
        orientation=Orientation(description="(n)[m] = " + ", ".join(an["G_directional"].keys())))
    rec(PropertyName.POISSON_RATIO_DIRECTIONAL, list(an["nu_directional"].values()), Unit.DIMENSIONLESS,
        shape="ν(load n, lateral m) for " + ", ".join(an["nu_directional"].keys()),
        orientation=Orientation(description="n/m = " + ", ".join(an["nu_directional"].keys())))
    if an["is_cubic"]:
        rec(PropertyName.ZENER_ANISOTROPY, float(an["zener_ratio"]), Unit.DIMENSIONLESS, frame="not_applicable",
            orientation=None, notes="A = 2 C44 / (C11 - C12); 1 = isotropic")
    rec(PropertyName.UNIVERSAL_ANISOTROPY_INDEX, float(an["universal_anisotropy"]), Unit.DIMENSIONLESS,
        frame="not_applicable", orientation=None, notes="A^U = 5 G_V/G_R + B_V/B_R - 6")
    rec(PropertyName.PUGH_RATIO, float(an["pugh_ratio_B_over_G"]), Unit.DIMENSIONLESS, frame="not_applicable",
        orientation=None, notes="B_H/G_H (heuristic ductility indicator, Pugh 1954)")
    born_ok = bool(an["born_stable"])
    rec(PropertyName.BORN_STABILITY, float(an["born_min_eigenvalue"]), Unit.GPA, frame="crystal",
        shape="minimum eigenvalue of the 6x6 relaxed-ion C (>0: mechanically stable at zero stress)",
        chk=checks + [_check("born_stability", born_ok, an["born_min_eigenvalue"], 0.0, Unit.GPA)])
    hv = Derivation(model=HARDNESS_CHEN_NIU[0], equation="Hv = 2 (k^2 G)^0.585 - 3,  k = G_H / B_H",
                    assumptions=["empirical correlation fitted to measured Vickers hardness of mostly covalent/ionic solids",
                                 "uses single-crystal VRH moduli of a defect-free lattice; ignores dislocation mobility, "
                                 "microstructure, load/indentation-size effects",
                                 "not a computed plastic property"],
                    input_record_ids=[mod["B_hill"].id, mod["G_hill"].id])
    rec(PropertyName.HARDNESS_ESTIMATE, float(an["hardness_chen_niu"]), Unit.GPA, frame="not_applicable", orientation=None,
        origin=PropertyOrigin.DERIVED_MODEL, derivation=hv, chk=[],
        unc_=[chen_niu_score_uncertainty(), Uncertainty(kind=UncertaintyKind.UNQUANTIFIED,
                          description=f"model-form: {HARDNESS_TIAN[0]} gives {an['hardness_tian']:.2f} GPa")],
        notes=HARDNESS_CHEN_NIU[1],
        meth=_method("Chen-Niu hardness model", protocol, "empirical hardness from Hill B and G", []))

    out.properties.extend(recs)
    out.checks.extend(checks + [_check("born_stability", born_ok, an["born_min_eigenvalue"], 0.0, Unit.GPA)])
    out.checks.extend(c for r in recs for c in r.validation.checks if c.name == "cross_engine_agreement")
    out.structure = s
    out.artifacts.update(C_fd_relaxed_GPa=Cs, C_fd_clamped_GPa=Ccs, C_fd_per_magnitude_GPa=fd["per_magnitude"],
                         C_fd_clamped_per_magnitude_GPa=fdc["per_magnitude"], analysis=an, atoms=atoms,
                         internal_displacement_Gamma_A=fd["gamma"][primary], rotations_cart=rot)
    if C_an is not None:
        out.artifacts["C_calculator_analytic_GPa"] = C_an
    out.raw.update(finite_difference=dict(C_GPa=Cs.tolist(), C_clamped_GPa=Ccs.tolist(), primary_magnitude=primary,
                                          magnitudes=mags, convergence_spread_GPa=spread,
                                          max_residual_force_eV_per_A=fd["max_residual_force"],
                                          per_magnitude_GPa={str(k): v.tolist() for k, v in fd["per_magnitude"].items()}),
                   space_group={"number": sg.number, "symbol": sg.international}, kleinman_zeta=zeta,
                   conventional_a_A=conv_a, calculator=spec.description, cross_engine=cmp.rows)
    return out


def _bulk_for_cleavage(candidate, spec, calc, structure, protocol):
    """(relaxed conventional ASE Atoms, miller_cell rows, info) — mirrors the JAX runner's bulk reference."""
    from fracture_md.structures.slabs import conventional_lattice

    if structure is not None:
        atoms = _starting_atoms(candidate, structure)
        return atoms, conventional_lattice(atoms), dict(method="structure supplied by caller (assumed relaxed)")
    atoms0 = _starting_atoms(candidate, None)
    mc0 = conventional_lattice(atoms0)
    atoms, info = relax_atoms(atoms0, calc, fmax=1e-6, optimizer="auto", relax_cell=True, cell_mode="full",
                              stress_tol_GPa=1e-5)
    F = atoms.cell.array.T @ np.linalg.inv(atoms0.cell.array.T)
    info = dict(method=f"ASE {info['optimizer']} + FrechetCellFilter (this engine)", **{k: info[k] for k in ("converged", "fmax",
                                                                                              "stress_GPa", "n_steps")})
    return atoms, (F @ mc0.T).T, info


def run_cleavage(protocol, candidate, spec: CalculatorSpec, structure=None, context=None) -> ProtocolOutput:
    """CleavageProtocol (mode="rigid") with ASE: same slab as the JAX engine, FD tractions of the calculator energy.

    ``context.extra["cleavage"]`` (shared with the JAX runner) may set ``skin``
    (0.5 Å, only for slab construction), ``fine_step`` (0.005 Å; None = protocol
    grid only), ``relaxed_surfaces`` (True), ``bond_cutoff``; engine-specific
    ``fd_step`` (1e-5 Å) and ``force_traction_tol_GPa`` (0.02) can be given in
    ``context.extra["ase_reference"]["cleavage"]``.
    """
    from fracture_md.physics.cleavage import fit_uber
    from fracture_md.records import make_record
    from fracture_md.schema.material import InterfaceKind, InterfaceSpec
    from fracture_md.schema.properties import (
        CohesiveLawFit,
        Derivation,
        PropertyOrigin,
        SampledCurve,
        TractionSeparationLaw,
    )
    from fracture_md.structures.slabs import build_cleavage_slab
    from fracture_md.structures.system import AtomicSystem, perpendicular_widths
    from fracture_md.units import q

    if protocol.mode != "rigid":
        raise NotImplementedError("ase_reference engine implements CleavageProtocol(mode='rigid') only; "
                                  "use the jax_md engine for relaxed (grip-controlled) opening curves")
    ctx = context or RunContext()
    opts = dict(skin=0.5, fine_step=0.005, relaxed_surfaces=True, bond_cutoff=None, fd_step=1e-5,
                force_traction_tol_GPa=0.02)
    opts.update({k: v for k, v in ctx.extra.get("cleavage", {}).items() if k in opts})
    opts.update(ctx.extra.get("ase_reference", {}).get("cleavage", {}))
    skin = float(opts["skin"])
    calc = spec.make()
    card = _card(spec)
    cut = _cutoff(spec)
    phase_name = candidate.phases[0].name if candidate.phases else candidate.name

    conv_atoms, miller_cell, bulk_info = _bulk_for_cleavage(candidate, spec, calc, structure, protocol)
    slab = build_cleavage_slab(conv_atoms, protocol.plane_miller, protocol.termination, protocol.n_layers,
                               protocol.in_plane_repeats, protocol.vacuum_A, cutoff=cut, skin=skin,
                               bond_cutoff=opts["bond_cutoff"], miller_cell=miller_cell)
    openings = np.asarray(protocol.openings_A, float)
    ro = _RigidOpening(slab.atoms, calc, slab.upper_mask, opts["fd_step"])
    area = ro.area
    need = cut + skin
    # vacuum along the normal between the top surface and the image of the bottom surface
    zn = slab.atoms.get_positions() @ ro.n
    vg = float(perpendicular_widths(np.asarray(slab.atoms.cell.array).T)[2] - (zn.max() - zn.min()))
    eff = vg - openings[-1]
    if eff <= need:
        raise ValueError(f"vacuum gap {vg:.3f} Å minus δ_max {openings[-1]:.3f} Å <= cutoff+skin {need:.3f} Å")
    checks = [_check("vacuum_gap_exceeds_interaction_range_at_max_opening", eff > need, eff, need, Unit.ANGSTROM,
                     f"vacuum gap {vg:.3f} Å - δ_max {openings[-1]:.3f} Å > cutoff+skin {need:.3f} Å")]

    coarse = ro.scan(openings)
    fine = None
    if opts["fine_step"]:
        nf = int(math.ceil(openings[-1] / float(opts["fine_step"]))) + 1
        fine = ro.scan(np.linspace(0.0, openings[-1], nf))
    cur = fine if fine is not None else coarse
    x, T, TF, phi = cur["openings"], cur["traction"], cur["traction_forces"], cur["energy_per_area"]
    dF = float(np.max(np.abs(np.concatenate([coarse["traction"] - coarse["traction_forces"], T - TF]))))
    ftol = float(opts["force_traction_tol_GPa"])
    checks.append(_check("traction_fd_equals_calculator_forces", dF <= ftol, dF, ftol, Unit.GPA,
                         f"central FD of E (h={ro.h:g} Å) vs −(1/A)ΣF·n from the calculator's analytic forces, "
                         f"{len(x) + len(openings)} openings; residual set by C1 cutoff-function kinks (≈ h/2·|ΔT'|)"))
    W = float(coarse["energy_per_area"][-1])
    integ = float(np.trapezoid(T, x) / EV_PER_A3_TO_GPA * EV_PER_A2_TO_J_PER_M2)
    if fine is not None:
        rel = abs(integ - phi[-1]) / max(abs(phi[-1]), 1e-12)
        checks.append(_check("integral_of_traction_equals_energy_change", rel <= 1e-3, rel, 1e-3, Unit.DIMENSIONLESS,
                             f"trapezoid ∫T_FD dδ on {len(x)}-point grid = {integ:.8f} J/m^2 vs Φ(δ_max) = {phi[-1]:.8f}"))
        near = np.argmin(np.abs(x[:, None] - openings[None, :]), axis=0)
        if np.allclose(x[near], openings, rtol=0.0, atol=1e-12):  # protocol openings lie on the fine grid
            dgrid = float(np.max(np.abs(fine["energy_per_area"][near] - coarse["energy_per_area"])))
            checks.append(_check("fine_and_protocol_grid_energies_agree", dgrid <= 1e-10, dgrid, 1e-10, Unit.J_PER_M2))

    # peak: grid maximum, refined with bounded Brent on the FD traction
    from scipy.optimize import minimize_scalar

    k = int(np.argmax(T))
    lo, hi = x[max(k - 1, 0)], x[min(k + 1, len(x) - 1)]
    if hi > lo:
        r = minimize_scalar(lambda d: -ro.traction_fd(d), bounds=(lo, hi), method="bounded", options=dict(xatol=1e-6))
        at_peak, peak = float(r.x), float(-r.fun)
    else:  # pragma: no cover
        at_peak, peak = float(lo), ro.traction_fd(lo)
    gmax = float(np.max(np.concatenate([T, coarse["traction"]])))
    if gmax > peak:
        allx = np.concatenate([x, openings])
        at_peak, peak = float(allx[int(np.argmax(np.concatenate([T, coarse["traction"]])))]), gmax
    uber = fit_uber(x, phi)
    art = cutoff_artifact_flag(spec, ro.atoms_at(at_peak), slab.upper_mask, "rigid peak traction")
    plaus = _check("plausibility:peak_traction_not_set_by_cutoff_function", not art["flagged"],
                   float(art["n_pairs_in_switching_region"]), 0.0, Unit.DIMENSIONLESS, art["details"])

    term = slab.termination
    polar = bool(term.polar)
    symmetric = (not polar) and sorted(term.species_below) == sorted(term.species_above)
    lateral = "laterally constrained: in-plane lattice fixed at the relaxed bulk value"
    interface = InterfaceSpec(kind=InterfaceKind.CLEAVAGE_PLANE, phase_a=phase_name, orientation_a=slab.orientation,
                              termination=slab.termination,
                              description=f"cleavage on {slab.orientation.plane_miller} ({slab.termination.label})")
    base_assumptions = [
        "0 K lattice statics; no thermal or zero-point effects", lateral,
        f"slab of {slab.n_layers} atomic layers ({slab.n_cells} oriented cells), {slab.in_plane_repeats} in-plane repeats, "
        f"vacuum gap {slab.vacuum_gap:.2f} Å, periodic in all directions",
        "bulk-terminated (unreconstructed) surfaces",
    ]
    if polar:
        base_assumptions.append("polar termination: only γ_top + γ_bottom = W_sep is defined without a "
                                "chemical-potential reference")
    rigid_assumptions = base_assumptions + [
        "rigid separation: upper block translated along the plane normal, no relaxation (unrelaxed UBER-type curve)",
        f"traction = central finite difference of the {spec.backend} energy, h = {ro.h:g} Å"]
    if art["flagged"]:
        rigid_assumptions.append("FLAG: " + art["details"])
    cmp = _Comparator(ctx)
    prov = _provenance(spec, protocol, ctx)
    common = dict(candidate=candidate, card=card, phase=phase_name, interface=interface, orientation=slab.orientation,
                  provenance=prov)
    plane = tuple(slab.orientation.plane_miller)
    records: list[PropertyRecord] = []

    def rec(name, value, unit, *, method, conditions, chk, notes="", origin=PropertyOrigin.ATOMISTIC_COMPUTED,
            derivation=None, unc=()):
        key = (name.value, "", conditions.stress_state, plane, term.label,
               tuple(derivation.assumptions) if derivation else ())
        ev, ck = cmp(key, name, value, unit)
        r = make_record(name=name, value=value, unit=unit, method=method, conditions=conditions, checks=list(chk) + ck,
                        evidence=ev, notes=notes, origin=origin, derivation=derivation, uncertainty=list(unc), **common)
        records.append(r)
        return r

    m_rigid = _method("rigid cleavage opening (finite-difference tractions)", protocol,
                      "upper block translated rigidly along the plane normal; T = (1/A) dE/dδ by central FD of the "
                      f"{spec.description} energy", rigid_assumptions)
    cond_rigid = Conditions(stress_state="rigid normal separation, " + lateral)
    gname = PropertyName.SURFACE_ENERGY if symmetric else PropertyName.SURFACE_ENERGY_AVERAGE_PAIR
    gnote = ("γ = W_sep/2 (both faces identical by symmetry)" if symmetric else
             "(γ_top + γ_bottom)/2 = W_sep/2; individual polar-face energies need a chemical-potential reference")
    rW = rec(PropertyName.WORK_OF_SEPARATION, W, Unit.J_PER_M2, method=m_rigid, conditions=cond_rigid, chk=checks,
             notes=f"unrelaxed W_sep = 2γ_unrelaxed = Φ(δ={openings[-1]} Å)")
    rec(gname, W / 2, Unit.J_PER_M2, method=m_rigid, conditions=cond_rigid, chk=checks, notes="unrelaxed; " + gnote)
    pk = checks + [plaus]
    rec(PropertyName.PEAK_NORMAL_TRACTION, peak, Unit.GPA, method=m_rigid, conditions=cond_rigid, chk=pk,
        notes=art["details"])
    rec(PropertyName.OPENING_AT_PEAK_TRACTION, at_peak, Unit.ANGSTROM, method=m_rigid, conditions=cond_rigid, chk=pk)
    rec(PropertyName.UBER_LENGTH_SCALE, uber["l_A"], Unit.ANGSTROM, method=m_rigid, conditions=cond_rigid, chk=pk,
        unc=[Uncertainty(kind=UncertaintyKind.FIT_STANDARD_ERROR, value=uber["l_stderr_A"], unit=Unit.ANGSTROM,
                         interval="1sigma", description=f"UBER fit rms/W = {uber['relative_rms']:.3g}")],
        notes=f"UBER fit W={uber['W_J_m2']:.4f} J/m^2, σ_max={uber['sigma_max_GPa']:.2f} GPa, R^2={uber['r_squared']:.5f}")

    raw: dict[str, Any] = dict(bulk=bulk_info, calculator=spec.description, slab=dict(
        n_atoms=len(slab.atoms), in_plane_repeats=slab.in_plane_repeats, n_layers=slab.n_layers, area_A2=area,
        vacuum_gap_A=vg, termination=dict(slab.termination_info)),
        rigid=dict(openings=openings.tolist(), energy_per_area=coarse["energy_per_area"].tolist(),
                   traction=coarse["traction"].tolist(), traction_forces=coarse["traction_forces"].tolist(),
                   peak_traction=peak, opening_at_peak=at_peak, work_of_separation=W, integrated_work=integ,
                   fd_step=ro.h, n_energy_calls=ro.n_calls),
        uber_rigid=uber, cutoff_artifact_rigid=art)

    W_g, W_rec, W_kind = W, rW, "unrelaxed (rigid)"
    all_checks = list(checks)
    if opts["relaxed_surfaces"]:
        sep = max(float(openings[-1]), cut + 2.0)
        if vg - sep <= need:
            raise ValueError("vacuum too small for the separation used for relaxed surfaces")
        fm = protocol.relax.fmax_eV_per_A
        a0, i0 = relax_atoms(slab.atoms, calc, fmax=fm, max_steps=protocol.relax.max_steps, optimizer="auto")
        a1, i1 = relax_atoms(ro.atoms_at(sep), calc, fmax=fm, max_steps=protocol.relax.max_steps, optimizer="auto")
        pairs = cross_gap_pairs_ase(a1, slab.upper_mask, cut)
        W_rel = (i1["energy"] - i0["energy"]) / area * EV_PER_A2_TO_J_PER_M2
        W_un = (ro.energy(sep) - ro.energy(0.0)) / area * EV_PER_A2_TO_J_PER_M2
        rchecks = [_check("relaxed_states_converged", i0["converged"] and i1["converged"], max(i0["fmax"], i1["fmax"]),
                          fm, Unit.EV_PER_ANGSTROM, f"ASE {i0['optimizer']}, {i0['n_steps']} + {i1['n_steps']} steps"),
                   _check("relaxed_halves_remain_separated", len(pairs) == 0, float(len(pairs)), 0.0, Unit.DIMENSIONLESS,
                          f"cross-gap pairs within cutoff after relaxation: {len(pairs)}")]
        all_checks += rchecks
        cond_rel = Conditions(stress_state="fully relaxed intact and separated slabs, " + lateral)
        m_rel = _method("relaxed work of separation (energy minimisation)", protocol,
                        "E(two relaxed separated half-slabs) − E(relaxed intact slab), per area; ASE LBFGS/BFGS",
                        base_assumptions + ["local relaxation from bulk-terminated positions (no reconstruction search)"])
        W_rec = rec(PropertyName.WORK_OF_SEPARATION, W_rel, Unit.J_PER_M2, method=m_rel, conditions=cond_rel,
                    chk=rchecks, notes=f"relaxation lowers W_sep by {W_un - W_rel:.4f} J/m^2")
        rec(gname, W_rel / 2, Unit.J_PER_M2, method=m_rel, conditions=cond_rel, chk=rchecks, notes="relaxed; " + gnote)
        W_g, W_kind = W_rel, "relaxed"
        raw["relaxed_surfaces"] = dict(W_relaxed_J_m2=W_rel, W_unrelaxed_J_m2=W_un, separation_A=sep,
                                       relaxation_energy_J_m2=W_un - W_rel, intact=i0, separated=i1)

    griffith = ["ideal brittle (Griffith) cleavage: G_Ic = W_sep = 2γ_s",
                "no lattice trapping (the atomistic crack-tip barrier is ignored)",
                "no plasticity, phase transformation or other dissipation", "straight crack on this plane and termination",
                f"uses the {W_kind} work of separation"]
    rec(PropertyName.G_IC_ATOMISTIC, W_g, Unit.J_PER_M2,
        method=_method("Griffith ideal brittle fracture energy", protocol, "G_Ic = W_sep", griffith),
        conditions=Conditions(stress_state=lateral), chk=[], origin=PropertyOrigin.DERIVED_MODEL,
        derivation=Derivation(model="Griffith: G_Ic = 2 γ_s = W_sep", equation="G_Ic = [E_separated − E_intact]/A",
                              assumptions=griffith, input_record_ids=[W_rec.id]),
        notes="atomistic ideal value; NOT a measured or macroscopic toughness")

    tsl = TractionSeparationLaw(
        mode="normal", interface=interface, protocol_mode="rigid",
        curve=SampledCurve(x_name="opening", x=q(x.tolist(), Unit.ANGSTROM), y_name="normal_traction",
                           y=q(T.tolist(), Unit.GPA),
                           extra={"energy_per_area": q(phi.tolist(), Unit.J_PER_M2),
                                  "traction_calculator_forces": q(TF.tolist(), Unit.GPA)}),
        energy_per_area=q(phi.tolist(), Unit.J_PER_M2), peak_traction=q(peak, Unit.GPA),
        opening_at_peak=q(at_peak, Unit.ANGSTROM), work_of_separation=q(W, Unit.J_PER_M2),
        integrated_work=q(integ, Unit.J_PER_M2),
        fits=[CohesiveLawFit(model="UBER", parameters={"W": q(uber["W_J_m2"], Unit.J_PER_M2), "l": q(uber["l_A"], Unit.ANGSTROM),
                                                       "sigma_max": q(uber["sigma_max_GPa"], Unit.GPA)},
                             rms_residual=q(uber["rms_residual_J_m2"], Unit.J_PER_M2),
                             domain=f"energy curve 0-{x[-1]:.3g} Å; R^2={uber['r_squared']:.5f}")],
        assumptions=rigid_assumptions + ([f"curve sampled on a uniform {len(x)}-point grid (h = {x[1]:.4g} Å)"]
                                         if fine is not None else []),
        checks=checks + [plaus])
    cross = [c for r in records for c in r.validation.checks if c.name == "cross_engine_agreement"]
    raw["cross_engine"] = cmp.rows
    return ProtocolOutput(properties=records, traction_separation=[tsl], checks=all_checks + cross,
                          structure=AtomicSystem.from_ase(slab.atoms),
                          artifacts=dict(rigid_openings_A=openings, rigid_energy_per_area_J_m2=coarse["energy_per_area"],
                                         rigid_traction_GPa=coarse["traction"],
                                         rigid_traction_forces_GPa=coarse["traction_forces"],
                                         rigid_fine_openings_A=None if fine is None else fine["openings"],
                                         rigid_fine_traction_GPa=None if fine is None else fine["traction"],
                                         rigid_fine_energy_per_area_J_m2=None if fine is None else fine["energy_per_area"],
                                         slab_atoms=slab.atoms, upper_mask=slab.upper_mask),
                          raw=raw)


RUNNERS: dict[str, Callable] = {
    "bulk_relax": run_bulk_relax,
    "elastic_tensor": run_elastic_tensor,
    "cleavage": run_cleavage,
}


# ============================================================================
# engine object
# ============================================================================
class AseReferenceEngine:
    """Engine contract (engines/base.py): name, version, capabilities(), run()."""

    name = ENGINE_NAME

    def __init__(self) -> None:
        self.version = _versions() or "unknown"

    def capabilities(self) -> set[str]:
        """Protocol kinds runnable with at least one available calculator."""
        if not any(spec.availability()[0] for spec in CALCULATORS.values()):
            return set()
        return set(RUNNERS)

    def potentials(self) -> dict[str, str]:
        """Registered calculators and their availability."""
        return {n: ("available" if ok else f"unavailable: {why}") for n, (ok, why) in
                ((n, s.availability()) for n, s in CALCULATORS.items())}

    def supports(self, kind: str, potential: str) -> bool:
        try:
            return kind in RUNNERS and resolve_calculator(potential).availability()[0]
        except KeyError:
            return False

    def run(self, protocol: Any, candidate, potential: str, structure: Any = None,
            context: Optional[RunContext] = None) -> ProtocolOutput:
        kind = protocol.kind
        if kind not in RUNNERS:
            raise NotImplementedError(f"ase_reference engine has no runner for protocol kind {kind!r}; "
                                      f"supported: {sorted(RUNNERS)}")
        spec = resolve_calculator(potential)
        with warnings.catch_warnings():
            # matscipy's Tersoff kernel uses np.power(..., where=) without out=; the masked entries are replaced
            warnings.filterwarnings("ignore", message=".*'where' used without 'out'.*")
            return RUNNERS[kind](protocol, candidate, spec, structure=structure, context=context or RunContext())


ENGINE = AseReferenceEngine()


def get_engine() -> AseReferenceEngine:
    return ENGINE


__all__ = [
    "ENGINE",
    "AseReferenceEngine",
    "CalculatorSpec",
    "CalculatorUnavailable",
    "CALCULATORS",
    "DEFAULT_TOLERANCES",
    "register_calculator",
    "resolve_calculator",
    "matscipy_tersoff_spec",
    "ase_tersoff_spec",
    "mace_torch_spec",
    "lammps_spec",
    "record_key",
    "compare_outputs",
    "relax_atoms",
    "fd_elastic_constants",
    "run_bulk_relax",
    "run_elastic_tensor",
    "run_cleavage",
]
