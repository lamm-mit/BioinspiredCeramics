"""ACE (pacemaker) run directories for the indentation decks on the GB10 (LAMMPS 30Sep2026 KOKKOS, ``pair pace/kk``).

A WRAPPER of :mod:`fracture_md.engines.lammps_indent`, which stays unchanged (the Vashishta decks generate exactly as
before; tests/test_ace_indent.py checks it).  The decks are the ACE variants in ``examples/indent/ace/``
(``make_variants.py``, ``lineage.json``): the parents plus declared substitutions (host ``fix external``, optional
host CSVR, the extrapolation grade γ, an ACE_META line).  This module

* describes an ACE potential (:class:`AcePotential`: the ``.yace`` for ``pace``; for γ the B-basis ``.yaml`` and the
  active set ``.asi`` for ``pace/extrapolation``; cutoff; a status that says whether the fit is a toy);
* reuses the variable builders of the parent generator (``atomistic_deck_variables``, ``deck_variables``) and swaps
  the potential: ``pstyle``/``pname``/``pfile`` and the ACE variables (``sfx``, ``bathhost``, ``gamma``, ``pyaml``,
  ``pasi``, ``ghi``, ``gdump``, ``ghalt``);
* writes ``lmp.args``, ``run.sh`` (GPU: ``scripts/lmp_kk.sh PREC``, which refuses to run outside
  ``scripts/gpu_exclusive.sh`` + ``scripts/run_guarded.sh``), ``run_cpu.sh`` (CPU LAMMPS of the fmd-lammps env with
  ``-var sfx none``, one pinned core) and ``manifest.json`` (protocol, potential with sha256, deck lineage, recipe).

The z pre-expansion ``zscale`` is a property of the potential (a(T) / c(0 K) at the lateral a(T)); the Vashishta value
is never applied to ACE: it must be passed (measure it with ``examples/indent/ace/in.lattice_ace.lmp``).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shlex
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from fracture_md.engines import lammps_indent as LI

PathLike = Union[str, Path]

REPO = Path(__file__).resolve().parents[3]
ACE_DIR = REPO / "examples" / "indent" / "ace"
ATOMISTIC_ACE_DECK = "in.berk_atomistic_ace.lmp"
SDF_ACE_DECK = "in.berk_sdf_ace.lmp"
POPIN_ACE_DECK = "in.popin_sphere_ace.lmp"
LMP_KK = REPO / "scripts" / "lmp_kk.sh"
CPU_LAMMPS_ENV = Path.home() / "miniconda3" / "envs" / "fmd-lammps"
GENERATOR = "fracture_md.engines.ace_indent"
GENERATOR_VERSION = 1
DEFAULT_HOST_CORE = 19  # the indentation-readiness track's core (lead's booking, 2026-10-06)
#: toy: a fit for speed/mechanics tests; not_admitted: a real potential that failed admission (e.g. the downloaded
#: HEGB_SiCCs.yace, outputs/mlip_design/mlip-lit/screen_2026-10-06.json), used only to exercise the decks;
#: candidate: a project fit under validation; production: admitted (docs/MLIP_PLAN.md gates)
STATUSES: tuple[str, ...] = ("toy", "not_admitted", "candidate", "production")
MECHANICS_ONLY: tuple[str, ...] = ("toy", "not_admitted")


@dataclasses.dataclass(frozen=True)
class AcePotential:
    """An ACE potential for LAMMPS: ``yace`` (pair pace), optionally ``yaml`` + ``asi`` (pair pace/extrapolation)."""

    yace: Path
    label: str
    cutoff_A: float
    yaml: Optional[Path] = None
    asi: Optional[Path] = None
    status: str = "toy"  # "toy" (mechanics tests only; physics meaningless) | "candidate" | "production"
    elements: tuple[str, ...] = ("C", "Si")
    source: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, not {self.status!r}")
        if not Path(self.yace).exists():
            raise FileNotFoundError(self.yace)
        for p in (self.yaml, self.asi):
            if p is not None and not Path(p).exists():
                raise FileNotFoundError(p)

    @property
    def has_active_set(self) -> bool:
        return self.yaml is not None and self.asi is not None

    def summary(self) -> dict[str, Any]:
        d = {"label": self.label, "status": self.status, "cutoff_A": self.cutoff_A, "elements": list(self.elements),
             "source": self.source, "yace": str(self.yace), "yace_sha256": LI.sha256_file(self.yace)}
        for k in ("yaml", "asi"):
            p = getattr(self, k)
            d[k] = str(p) if p is not None else None
            d[f"{k}_sha256"] = LI.sha256_file(p) if p is not None else None
        if self.status in MECHANICS_ONLY:
            d["warning"] = (f"{self.status.upper()} potential: tests the mechanics of the decks only; every physical "
                            "number from it is meaningless for SiC")
        return d


def potential_from_fit_dir(fit_dir: PathLike, *, label: Optional[str] = None, status: str = "toy",
                           asi: Optional[PathLike] = None) -> AcePotential:
    """An :class:`AcePotential` from a pacemaker fit directory: the single ``*.yace``, ``output_potential.yaml`` and
    (if present, or ``asi``) ``output_potential.asi``; the cutoff from ``input.yaml`` (``cutoff:``)."""
    d = Path(fit_dir)
    yaces = sorted(d.glob("*.yace"))
    if len(yaces) != 1:
        raise FileNotFoundError(f"{d}: expected exactly one .yace, found {[p.name for p in yaces]}")
    cut = None
    inp = d / "input.yaml"
    if inp.exists():
        m = re.search(r"^cutoff:\s*([0-9.eE+-]+)", inp.read_text(), re.M)
        cut = float(m.group(1)) if m else None
    if cut is None:
        raise ValueError(f"{d}: no 'cutoff:' in input.yaml")
    yml = d / "output_potential.yaml"
    a = Path(asi) if asi is not None else d / "output_potential.asi"
    return AcePotential(yace=yaces[0], label=label or d.name, cutoff_A=cut, yaml=yml if yml.exists() else None,
                        asi=a if a.exists() else None, status=status, source=str(d))


def ace_variables(pot: AcePotential, *, pstyle_options: str = "", gamma_every: int = 0, gamma_hi: float = 5.0,
                  gamma_dump: float = 0.0, gamma_halt: float = 0.0, target: str = "kokkos",
                  bath_host: bool = False) -> dict[str, str]:
    """The ACE ``-var`` values of the variant decks (file names are the run-directory copies)."""
    if target not in ("kokkos", "cpu"):
        raise ValueError("target must be kokkos or cpu")
    if bath_host and target != "kokkos":
        raise ValueError("bath_host (host temp/csvr under -sf kk) only makes sense for the kokkos target")
    if gamma_every < 0:
        raise ValueError("gamma_every must be >= 0")
    if gamma_every > 0 and not pot.has_active_set:
        raise ValueError(f"gamma needs the B-basis .yaml and the .asi of {pot.label}")
    # pair pace/kk refuses the recursive evaluator ("Must use 'product' algorithm with pair pace/kk")
    opts = pstyle_options.split()
    if "recursive" in opts:
        raise ValueError("pair pace/kk needs the product algorithm, not 'recursive'")
    pst = " ".join(["pace"] + ([] if "product" in opts else ["product"]) + opts)
    # sfx is written only for the cpu target: the decks default to kk, and the first -var of a name wins in LAMMPS,
    # so run_cpu.sh / run_dir.sh can append "-var sfx none" to a kokkos run directory's arguments
    v = {"pstyle": pst, "pname": "pace", "pfile": Path(pot.yace).name, **({"sfx": "none"} if target == "cpu" else {}),
         "bathhost": int(bool(bath_host)), "gamma": int(gamma_every),
         "pyaml": Path(pot.yaml).name if (gamma_every and pot.yaml) else "none",
         "pasi": Path(pot.asi).name if (gamma_every and pot.asi) else "none",
         "ghi": float(gamma_hi), "gdump": float(gamma_dump), "ghalt": float(gamma_halt)}
    return {k: LI._fmt(x) for k, x in v.items()}


def _copy_potential(pot: AcePotential, rd: Path, files: dict[str, Path], gamma_every: int) -> None:
    files["potential"] = Path(shutil.copyfile(pot.yace, rd / Path(pot.yace).name))
    if gamma_every:
        files["potential_yaml"] = Path(shutil.copyfile(pot.yaml, rd / Path(pot.yaml).name))
        files["active_set"] = Path(shutil.copyfile(pot.asi, rd / Path(pot.asi).name))


def _deck_lineage(deck_name: str) -> dict[str, Any]:
    lin = json.loads((ACE_DIR / "lineage.json").read_text())
    for v in lin["variants"]:
        if v["variant"].endswith("/" + deck_name):
            return {"variant": v["variant"], "variant_sha256": v["variant_sha256"], "parent": v["parent"],
                    "parent_sha256": v["parent_sha256"], "lineage_file": "examples/indent/ace/lineage.json"}
    raise KeyError(deck_name)


def run_scripts(tag: str, *, precision: str = "mixed", host_core: int = DEFAULT_HOST_CORE,
                cpu_env: PathLike = CPU_LAMMPS_ENV) -> dict[str, str]:
    """``run.sh`` (GPU, through scripts/lmp_kk.sh; launch it as
    ``scripts/gpu_exclusive.sh -- scripts/run_guarded.sh --min-avail-gb 24 -- bash run.sh [PREC]``) and ``run_cpu.sh``
    (CPU LAMMPS, ``-var sfx none``; launch it under ``scripts/run_guarded.sh --min-avail-gb 28``)."""
    if precision not in ("mixed", "double"):
        raise ValueError("precision must be mixed or double")
    gpu = f"""#!/bin/bash
# Generated by {GENERATOR} (v{GENERATOR_VERSION}): GPU (KOKKOS) run '{tag}' with an ACE potential.
# Launch:  {REPO}/scripts/gpu_exclusive.sh -- {REPO}/scripts/run_guarded.sh --min-avail-gb 24 -- bash run.sh [mixed|double]
# scripts/lmp_kk.sh refuses to start outside gpu_exclusive.sh + run_guarded.sh.  Host core: HOSTCORE (default {host_core}).
set -uo pipefail
cd "$(dirname "$0")"
PREC=${{1:-{precision}}}
mapfile -t ARGS < lmp.args
start=$(date +%s)
echo "{tag} START $(date -Iseconds) prec=$PREC hostcore=${{HOSTCORE:-{host_core}}} memavail_kB=$(awk '/MemAvailable/ {{print $2}}' /proc/meminfo)" | tee -a walltime.txt
HOSTCORE=${{HOSTCORE:-{host_core}}} "{LMP_KK}" "$PREC" "${{ARGS[@]}}" -screen none > run_{tag}.err 2>&1
rc=$?
echo "{tag} END $(date -Iseconds) rc=$rc wall_s=$(( $(date +%s) - start ))" | tee -a walltime.txt
exit $rc
"""
    cpu = f"""#!/bin/bash
# Generated by {GENERATOR} (v{GENERATOR_VERSION}): CPU run '{tag}' (fmd-lammps LAMMPS, pair pace, -var sfx none), one
# pinned core.  Launch:  {REPO}/scripts/run_guarded.sh --min-avail-gb 28 -- bash run_cpu.sh [CORE]
set -uo pipefail
cd "$(dirname "$0")"
CORE=${{1:-{host_core}}}
LMP_ENV=${{LMP_ENV:-{cpu_env}}}
mapfile -t ARGS < lmp.args
start=$(date +%s)
echo "{tag} CPU START $(date -Iseconds) core=$CORE" | tee -a walltime.txt
OMP_NUM_THREADS=1 taskset -c "$CORE" "$LMP_ENV/bin/lmp" "${{ARGS[@]}}" -var sfx none -screen none > run_{tag}.err 2>&1
rc=$?
echo "{tag} CPU END $(date -Iseconds) rc=$rc wall_s=$(( $(date +%s) - start ))" | tee -a walltime.txt
exit $rc
"""
    return {"run.sh": gpu, "run_cpu.sh": cpu}


def _finish(rd: Path, tag: str, deck_name: str, files: dict[str, Path], variables: dict[str, str],
            manifest: dict[str, Any], precision: str, host_core: int) -> LI.LammpsIndentDeck:
    deck = LI.LammpsIndentDeck(rd, tag, files, variables, manifest)
    files["args"] = rd / "lmp.args"
    files["args"].write_text("\n".join(deck.lmp_args()) + "\n")
    for name, text in run_scripts(tag, precision=precision, host_core=host_core).items():
        p = rd / name
        p.write_text(text)
        os.chmod(p, 0o755)
        files["run_script" if name == "run.sh" else "run_cpu_script"] = p
    manifest["sha256"] = {role: LI.sha256_file(p) for role, p in files.items()}
    manifest["command_example"] = (f"{REPO}/scripts/gpu_exclusive.sh -- {REPO}/scripts/run_guarded.sh --min-avail-gb 24 "
                                   f"-- bash {rd}/run.sh {precision}")
    files["manifest"] = rd / "manifest.json"
    files["manifest"].write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return deck


def _common_manifest(kind: str, tag: str, deck_name: str, pot: AcePotential, variables: Mapping[str, str],
                     precision: str, zscale: float) -> dict[str, Any]:
    return {"generator": GENERATOR, "generator_version": GENERATOR_VERSION, "tag": tag, "engine": "lammps-kokkos",
            "deck_kind": kind, "deck": deck_name, "deck_lineage": _deck_lineage(deck_name),
            "pair_style": variables["pstyle"], "potential": pot.summary(),
            "potential_card_id": f"ace:{pot.label}@{LI.sha256_file(pot.yace)[:12]}",
            "precision_default": precision, "zscale": zscale,
            "status_note": ("model-specific MD values of an ACE fit; never a measured hardness"
                            + (f"; {pot.status.upper()} potential: deck mechanics only" if pot.status in MECHANICS_ONLY else ""))}


def write_ace_atomistic_deck(protocol: Any, run_dir: PathLike, pot: AcePotential, *, zscale: float,
                             tag: str = "ace_atom", precision: str = "mixed", host_core: int = DEFAULT_HOST_CORE,
                             tipmotion: str = "move", amap: str = "none", gamma_every: int = 0,
                             gamma_hi: float = 5.0, gamma_dump: float = 0.0, gamma_halt: float = 0.0,
                             bath_host: bool = False, pstyle_options: str = "", tip_height_A: Optional[float] = None,
                             anr_A: Optional[float] = 0.0, restarts: bool = False, checkpoint_steps: int = 0,
                             nmin: int = 300, extra_vars: Optional[Mapping[str, Any]] = None,
                             recipe: Any = LI.DEFAULT_RECIPE) -> LI.LammpsIndentDeck:
    """Run directory of the rigid ATOMISTIC Berkovich (in.berk_atomistic_ace.lmp; the S1 tip) with an ACE potential.

    The variables are those of :func:`lammps_indent.atomistic_deck_variables` for ``protocol`` (built with the
    Vashishta pair style name, then swapped for ACE) plus :func:`ace_variables`.  ``zscale`` is required (the
    potential's own a(T)/c(0 K); 1.0 = no pre-expansion).  ``tipmotion`` "move" (the default since 2026-10-06, the
    lead's decision for production decks) moves the tip with the host ``fix move``: exact (0.5000011 against 0.5 A/ps
    nominal, review/gammaoff_move) at about 7 % of the speed (0.927x the nve rate, 120,849 atoms).  "nve" keeps
    the tip on the device (the S1 KOKKOS default of make_s1.py); in MIXED precision the positions are float32 on the
    device, so the tip runs 0.71 % fast at z = 113 A and about 2.4 % slow is expected at z = 256-512 A (the
    production cells; outputs/indent_ready/README.md, precision).
    """
    if tipmotion not in ("nve", "move"):
        raise ValueError("tipmotion must be nve or move")
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {"deck": Path(shutil.copyfile(ACE_DIR / ATOMISTIC_ACE_DECK, rd / ATOMISTIC_ACE_DECK)),
                              "sdf_module": Path(shutil.copyfile(LI.EXAMPLES_DIR / LI.SDF_MODULE, rd / LI.SDF_MODULE))}
    _copy_potential(pot, rd, files, gamma_every)
    base = LI.atomistic_deck_variables(protocol, pair_style="vashishta", tip_height_A=tip_height_A, anr_A=anr_A,
                                       restarts=restarts, checkpoint_steps=checkpoint_steps, nmin=nmin,
                                       zscale=float(zscale))
    acev = ace_variables(pot, pstyle_options=pstyle_options, gamma_every=gamma_every, gamma_hi=gamma_hi,
                         gamma_dump=gamma_dump, gamma_halt=gamma_halt, bath_host=bath_host)
    variables = {"tag": tag, "sdfdir": ".", **base, **acev, "tipmotion": tipmotion, "amap": amap}
    if extra_vars:
        variables.update({k: LI._fmt(x) for k, x in extra_vars.items()})
    nx, nz = LI.atomistic_cells(protocol)
    manifest = _common_manifest("atomistic", tag, ATOMISTIC_ACE_DECK, pot, variables, precision, float(zscale))
    manifest.update({
        "recipe": LI._copy_recipe(recipe, protocol, rd, files, variables, thickness_A=nz * protocol.lattice_parameter_A),
        "protocol": protocol.model_dump(mode="json"),
        "slab": {"nx": nx, "nz": nz, "lateral_size_A": [nx * protocol.lattice_parameter_A] * 2,
                 "thickness_A": nz * protocol.lattice_parameter_A, "lattice_constant_A": protocol.lattice_parameter_A,
                 "fixed_thickness_A": protocol.fixed_thickness_A, "bath_thickness_A": protocol.thermostat_thickness_A,
                 "atoms": LI.atomistic_atom_estimate(protocol, tip_height_A), "orientation": "(001) C-terminated",
                 "built_by": "LAMMPS lattice commands in in.berk_atomistic_ace.lmp"},
        "variables": variables})
    return _finish(rd, tag, ATOMISTIC_ACE_DECK, files, variables, manifest, precision, host_core)


def write_ace_sdf_deck(protocol: Any, slab: Any, run_dir: PathLike, pot: AcePotential, *, zscale: float,
                       tag: str = "ace_sdf", precision: str = "mixed", host_core: int = DEFAULT_HOST_CORE,
                       amap: str = "none", gamma_every: int = 0, gamma_hi: float = 5.0, gamma_dump: float = 0.0,
                       gamma_halt: float = 0.0, bath_host: bool = False, pstyle_options: str = "",
                       restarts: bool = False, checkpoint_steps: int = 0, anr_A: Optional[float] = None,
                       extra_vars: Optional[Mapping[str, Any]] = None,
                       recipe: Any = LI.DEFAULT_RECIPE) -> LI.LammpsIndentDeck:
    """Run directory of the harness SDF tip deck (in.berk_sdf_ace.lmp) on ``slab`` with an ACE potential."""
    LI._check_slab(protocol, slab)
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {"deck": Path(shutil.copyfile(ACE_DIR / SDF_ACE_DECK, rd / SDF_ACE_DECK)),
                              "sdf_module": Path(shutil.copyfile(LI.EXAMPLES_DIR / LI.SDF_MODULE, rd / LI.SDF_MODULE))}
    files["data"] = LI.write_lammps_data(slab.atoms, rd / "slab.data", title=f"{tag}: ACE {pot.label}")
    _copy_potential(pot, rd, files, gamma_every)
    base = LI.deck_variables(protocol, slab, pair_style="vashishta", restarts=restarts, anr_A=anr_A,
                             checkpoint_steps=checkpoint_steps, zscale=float(zscale))
    acev = ace_variables(pot, pstyle_options=pstyle_options, gamma_every=gamma_every, gamma_hi=gamma_hi,
                         gamma_dump=gamma_dump, gamma_halt=gamma_halt, bath_host=bath_host)
    acev.pop("pname")  # the SDF deck has no hybrid sub-style name
    variables = {"tag": tag, "data": files["data"].name, "sdfdir": ".", **base, **acev, "amap": amap}
    if extra_vars:
        variables.update({k: LI._fmt(x) for k, x in extra_vars.items()})
    manifest = _common_manifest("sdf", tag, SDF_ACE_DECK, pot, variables, precision, float(zscale))
    manifest.update({
        "recipe": LI._copy_recipe(recipe, protocol, rd, files, variables, thickness_A=slab.thickness_A),
        "protocol": protocol.model_dump(mode="json"),
        "slab": {"n_atoms": int(slab.n_atoms), "lateral_size_A": list(slab.lateral_size_A),
                 "thickness_A": slab.thickness_A, "lattice_constant_A": getattr(slab, "lattice_constant_A", None)},
        "variables": variables})
    return _finish(rd, tag, SDF_ACE_DECK, files, variables, manifest, precision, host_core)


def popin_variables(pot: AcePotential, base: Mapping[str, Any], **ace_kw: Any) -> dict[str, str]:
    """The popin deck's ``-var`` values (``base``: examples/indent/popin/make_popin_run.deck_vars output) with the ACE
    potential swapped in."""
    v = {k: LI._fmt(x) if not isinstance(x, str) else x for k, x in base.items()}
    acev = ace_variables(pot, **ace_kw)
    acev.pop("pname")
    v.update(acev)
    v["elems"] = "C Si"  # data-file types 1 = C, 2 = Si (engines.lammps_indent.TYPE_MAP)
    return v


__all__ = ["AcePotential", "ace_variables", "popin_variables", "potential_from_fit_dir", "run_scripts",
           "write_ace_atomistic_deck", "write_ace_sdf_deck"]
