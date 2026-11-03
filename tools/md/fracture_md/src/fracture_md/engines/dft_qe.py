"""Quantum ESPRESSO plane-wave DFT adapter (``engine = "quantum_espresso"``).

Runs ``pw.x`` through ASE's :class:`~ase.calculators.espresso.Espresso` calculator with an
:class:`~ase.calculators.espresso.EspressoProfile` whose command is an MPI launch pinned to physical
cores (``mpirun --map-by pe-list=...:ordered --bind-to core``; Open MPI ignores ``taskset``), and turns
the results into harness :class:`~fracture_md.schema.properties.PropertyRecord` objects whose provenance
names everything needed to reproduce the number: the pw.x version and conda build, the functional, the
pseudopotential family and the sha256 of every pseudopotential file, ``ecutwfc``/``ecutrho``, the k-point
mesh and offset, the occupation/smearing scheme, the SCF threshold, MPI layout and per-run timings.

Protocols (all 0 K, static, Born-Oppenheimer, collinear spin-unpolarised):

* :func:`single_point` - energy, forces and the stress at a GIVEN cell (``cell_stress`` record);
* :func:`relax_structure` - ``vc-relax`` (zero-stress cell, or a target pressure) or fixed-cell ``relax``;
* :func:`equation_of_state` - SCF energies on an isotropic strain grid + 3rd-order Birch-Murnaghan fit
  (``bulk_modulus_eos``, ``bulk_modulus_pressure_derivative``, V0, a0), with the stress-vs-dE/dV Pulay check;
* :func:`elastic_constants` - central finite-strain stress differences ``C_ij = [s_i(+d e_j) - s_i(-d e_j)]/2d``
  with internal coordinates relaxed (QE ``relax``) at every strained cell; ``mode="cubic"`` strains e1 and
  e4 only (C11, C12, C44), ``mode="full"`` all six components;
* :func:`convergence_scan` - ecutwfc or k-mesh convergence of the energy per atom and the stress.

DFT is treated as one more "potential": every record carries a :class:`PotentialCard` of kind ``dft``
(:func:`dft_card`) whose id hashes code version + functional + pseudopotential files.  Numerical
parameters (cutoffs, k-points, smearing) are NOT part of the card id: they are in the record's method
description and in ``provenance.extra["dft"]``.  A comparison with published values of the same functional
is attached as ``literature_model_value`` evidence (:class:`LiteratureValue`), which gives the status
LITERATURE_MODEL_AGREEMENT; nothing here is labelled measured or validated against experiment.

Environment: ``scripts/setup_dft_env.sh`` (conda env ``fmd-dft`` with conda-forge ``qe``; SSSP 1.3.0 PBE
under ``weights/pseudos``).  Overrides: ``FMD_PW_X`` (pw.x path), ``FMD_DFT_ENV`` (env prefix),
``FMD_PSEUDO_ROOT`` (pseudopotential root), ``FMD_DFT_CORES`` (e.g. ``10,11,12,13``).  See docs/DFT.md.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from fracture_md.engines.base import ProtocolOutput
from fracture_md.schema.common import (
    CheckResult,
    Citation,
    Conditions,
    Uncertainty,
    UncertaintyKind,
    ValidationEvidence,
)
from fracture_md.schema.properties import MethodSpec, PropertyName, PropertyRecord
from fracture_md.units import AMU_PER_A3_TO_G_PER_CM3, EV_PER_A3_TO_GPA, Quantity, Unit, convert

ENGINE_NAME = "quantum_espresso"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_PREFIX = Path.home() / "miniconda3" / "envs" / "fmd-dft"
DEFAULT_PSEUDO_FAMILY = "SSSP_1.3.0_PBE_efficiency"
RY_EV = 13.605693122994  # Rydberg energy in eV (CODATA 2018, as in QE 7.x constants.f90)
KBAR_PER_GPA = 10.0

CITATIONS = {
    "qe2009": Citation(key="Giannozzi2009", doi="10.1088/0953-8984/21/39/395502",
                       text="P. Giannozzi et al., QUANTUM ESPRESSO: a modular and open-source software project for "
                            "quantum simulations of materials, J. Phys.: Condens. Matter 21, 395502 (2009)"),
    "qe2017": Citation(key="Giannozzi2017", doi="10.1088/1361-648X/aa8f79",
                       text="P. Giannozzi et al., Advanced capabilities for materials modelling with Quantum "
                            "ESPRESSO, J. Phys.: Condens. Matter 29, 465901 (2017)"),
    "pbe": Citation(key="Perdew1996", doi="10.1103/PhysRevLett.77.3865",
                    text="J. P. Perdew, K. Burke, M. Ernzerhof, Generalized gradient approximation made simple, "
                         "Phys. Rev. Lett. 77, 3865 (1996)"),
    "sssp": Citation(key="Prandini2018", doi="10.1038/s41524-018-0127-2",
                     text="G. Prandini, A. Marrazzo, I. E. Castelli, N. Mounet, N. Marzari, Precision and efficiency "
                          "in solid-state pseudopotential calculations, npj Comput. Mater. 4, 72 (2018)"),
    "sssp_data": Citation(key="SSSP-1.3.0", doi="10.24435/materialscloud:f3-ym",
                          url="https://archive.materialscloud.org/records/rcyfm-68h65",
                          text="SSSP library (Materials Cloud Archive record rcyfm-68h65, files SSSP_1.3.0_PBE_*), "
                               "CC-BY-4.0 with per-family pseudopotential licences"),
    "birch": Citation(key="Birch1947", doi="10.1103/PhysRev.71.809",
                      text="F. Birch, Finite elastic strain of cubic crystals, Phys. Rev. 71, 809 (1947)"),
}


class QEUnavailable(FileNotFoundError):
    """pw.x (or the pseudopotential library) is not installed / configured."""


class QERunError(RuntimeError):
    """pw.x exited with an error or did not finish ("JOB DONE" missing)."""


# ============================================================================
# locating the installation
# ============================================================================
def env_prefix() -> Path:
    return Path(os.environ.get("FMD_DFT_ENV", str(DEFAULT_ENV_PREFIX)))


def find_pw_x() -> Optional[Path]:
    """pw.x from $FMD_PW_X, the fmd-dft conda env, or $PATH (None if absent)."""
    cand = [os.environ.get("FMD_PW_X"), str(env_prefix() / "bin" / "pw.x"), shutil.which("pw.x")]
    for c in cand:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return Path(c)
    return None


def pseudo_root() -> Path:
    return Path(os.environ.get("FMD_PSEUDO_ROOT", str(REPO_ROOT / "weights" / "pseudos")))


def qe_available() -> tuple[bool, str]:
    """(available, reason): pw.x found and the default pseudopotential family present."""
    pw = find_pw_x()
    if pw is None:
        return False, "pw.x not found (run scripts/setup_dft_env.sh or set FMD_PW_X)"
    fam = pseudo_root() / f"{DEFAULT_PSEUDO_FAMILY}.json"
    if not fam.is_file():
        return False, f"pseudopotential family {fam} missing (run scripts/setup_dft_env.sh)"
    return True, str(pw)


def conda_build(pw_x: Path) -> str:
    """Conda package string of the qe build that provides `pw_x` (e.g. 'qe-7.5-hc91ee90_1'), or ''."""
    meta = Path(pw_x).resolve().parent.parent / "conda-meta"
    if meta.is_dir():
        hits = sorted(p.stem for p in meta.glob("qe-*.json"))
        if hits:
            return hits[0]
    return ""


# ============================================================================
# pseudopotentials
# ============================================================================
@functools.lru_cache(maxsize=512)
def _file_digest(path: str, mtime_ns: int, size: int, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_digest(path: os.PathLike | str, algo: str = "sha256") -> str:
    st = os.stat(path)
    return _file_digest(str(path), st.st_mtime_ns, st.st_size, algo)


@dataclasses.dataclass(frozen=True)
class PseudoFamily:
    """An SSSP-style pseudopotential family: ``<root>/<name>.json`` + ``<root>/<name>/*.UPF``."""

    name: str
    root: Path
    entries: dict

    @property
    def directory(self) -> Path:
        return self.root / self.name

    @property
    def functional(self) -> str:
        m = re.search(r"_(PBEsol|PBE|LDA|PW91)_", self.name)
        return m.group(1) if m else "unknown"

    def entry(self, element: str) -> dict:
        if element not in self.entries:
            raise KeyError(f"{element} not in pseudopotential family {self.name}")
        return self.entries[element]

    def path(self, element: str) -> Path:
        return self.directory / self.entry(element)["filename"]

    def pseudopotentials(self, elements: Iterable[str]) -> dict[str, str]:
        return {el: self.entry(el)["filename"] for el in sorted(set(elements))}

    def checksums(self, elements: Iterable[str], verify: bool = True) -> dict[str, dict[str, str]]:
        """{filename: {element, md5, sha256, family_type}}; md5 verified against the library JSON."""
        out = {}
        for el in sorted(set(elements)):
            e = self.entry(el)
            p = self.path(el)
            if not p.is_file():
                raise QEUnavailable(f"pseudopotential file {p} missing (run scripts/setup_dft_env.sh)")
            md5 = file_digest(p, "md5")
            if verify and e.get("md5") and md5 != e["md5"]:
                raise ValueError(f"{p.name}: md5 {md5} differs from the {self.name} library value {e['md5']}")
            out[e["filename"]] = {"element": el, "md5": md5, "sha256": file_digest(p, "sha256"),
                                  "type": str(e.get("pseudopotential", ""))}
        return out

    def recommended_cutoffs_ry(self, elements: Iterable[str]) -> tuple[float, float]:
        """Largest SSSP-recommended (ecutwfc, ecutrho) over `elements`, Ry."""
        es = [self.entry(el) for el in set(elements)]
        return max(float(e["cutoff_wfc"]) for e in es), max(float(e["cutoff_rho"]) for e in es)


def load_pseudo_family(name: str = DEFAULT_PSEUDO_FAMILY, root: Optional[os.PathLike | str] = None) -> PseudoFamily:
    r = Path(root) if root is not None else pseudo_root()
    meta = r / f"{name}.json"
    if not meta.is_file():
        raise QEUnavailable(f"pseudopotential family {meta} missing (run scripts/setup_dft_env.sh)")
    return PseudoFamily(name=name, root=r, entries=json.loads(meta.read_text()))


# ============================================================================
# settings and runner
# ============================================================================
@dataclasses.dataclass(frozen=True)
class DFTSettings:
    """Numerical DFT parameters (energies in Ry, as in pw.x input).

    k-points: ``kpts`` (explicit Monkhorst-Pack mesh) wins; otherwise ``kspacing_per_A`` gives
    n_i = max(1, ceil(2 pi |b_i| / kspacing)) with b_i the reciprocal vectors without 2 pi (VASP KSPACING
    convention).  ``koffset`` 0 = Gamma-centred, 1 = shifted by half a step (pw.x K_POINTS automatic).
    ``conv_thr_ry_per_atom`` is multiplied by the number of atoms (pw.x conv_thr is per cell).

    Smearing: the default Marzari-Vanderbilt (cold) smearing is fine for gapped states, but in nearly metallic
    states (large shear, high pressure) its non-monotonic occupations can give a non-unique Fermi level and energies
    inconsistent with the forces (docs/DFT.md section 8.1); use ``smearing="gaussian"`` there.
    """

    ecutwfc_ry: float = 60.0
    dual: float = 8.0
    ecutrho_ry: Optional[float] = None
    kpts: Optional[tuple[int, int, int]] = None
    kspacing_per_A: Optional[float] = 0.15
    koffset: tuple[int, int, int] = (0, 0, 0)
    occupations: str = "smearing"
    smearing: str = "mv"
    degauss_ry: float = 0.01
    conv_thr_ry_per_atom: float = 1.0e-11
    mixing_beta: float = 0.4
    pseudo_family: str = DEFAULT_PSEUDO_FAMILY
    input_dft: Optional[str] = None  # None = the functional the pseudopotentials were generated with
    extra_input: tuple[tuple[str, str, Any], ...] = ()  # (namelist, key, value), e.g. ("system", "nbnd", 16)

    @property
    def ecutrho(self) -> float:
        return float(self.ecutrho_ry) if self.ecutrho_ry is not None else float(self.dual * self.ecutwfc_ry)

    def kmesh(self, atoms) -> tuple[int, int, int]:
        if self.kpts is not None:
            return tuple(int(k) for k in self.kpts)  # type: ignore[return-value]
        if self.kspacing_per_A is None:
            raise ValueError("set kpts or kspacing_per_A")
        b = np.linalg.norm(np.asarray(atoms.cell.reciprocal()), axis=1)
        return tuple(max(1, int(math.ceil(2 * math.pi * bi / self.kspacing_per_A - 1e-9))) for bi in b)  # type: ignore

    def replace(self, **kw) -> "DFTSettings":
        return dataclasses.replace(self, **kw)

    def describe(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["ecutrho_ry"] = self.ecutrho
        d["ecutwfc_eV"] = self.ecutwfc_ry * RY_EV
        d["ecutrho_eV"] = self.ecutrho * RY_EV
        d["extra_input"] = [list(x) for x in self.extra_input]
        return d


@dataclasses.dataclass(frozen=True)
class QERunner:
    """How pw.x is launched.  ``cores`` pins one MPI rank per listed physical core (Open MPI pe-list).

    ``npool`` = pw.x ``-nk`` (k-point pools; default = number of ranks, the efficient choice for small
    cells with many k-points).  OMP_NUM_THREADS is forced to ``omp_threads`` for every rank.  ``mca`` are
    Open MPI parameters: the default single-node shared-memory transport (pml ob1, btl self,sm) avoids a
    ~3 s UCX start-up per pw.x launch measured with the conda-forge Open MPI 5.0.11 build on this machine.
    pw.x is launched through mpirun whenever mpirun is available (also for one rank, so pinning and the
    transport choice apply); without mpirun it runs serially.
    """

    pw_x: str
    mpirun: Optional[str] = None
    cores: tuple[int, ...] = ()
    nprocs: int = 1
    npool: Optional[int] = None
    omp_threads: int = 1
    extra_args: tuple[str, ...] = ()
    mca: tuple[tuple[str, str], ...] = (("pml", "ob1"), ("btl", "self,sm"))

    @classmethod
    def from_env(cls, cores: Optional[Sequence[int]] = None, **kw) -> "QERunner":
        pw = find_pw_x()
        if pw is None:
            raise QEUnavailable("pw.x not found (run scripts/setup_dft_env.sh or set FMD_PW_X)")
        if cores is None and os.environ.get("FMD_DFT_CORES"):
            cores = [int(c) for c in os.environ["FMD_DFT_CORES"].replace(" ", "").split(",") if c]
        cores = tuple(int(c) for c in (cores or ()))
        mpirun = pw.parent / "mpirun"
        mp = str(mpirun) if mpirun.is_file() else shutil.which("mpirun")
        n = len(cores) if cores else int(kw.pop("nprocs", 1))
        return cls(pw_x=str(pw), mpirun=mp, cores=cores, nprocs=n, **kw)

    def argv(self) -> list[str]:
        n = len(self.cores) if self.cores else int(self.nprocs)
        nk = int(self.npool) if self.npool else n
        pw = [self.pw_x, *(["-nk", str(nk)] if nk > 1 else []), *self.extra_args]
        if not self.mpirun:
            if n > 1 or self.cores:
                raise QEUnavailable("mpirun not found next to pw.x or on $PATH")
            return pw
        cmd = [self.mpirun, "-np", str(n)]
        if self.cores:
            cmd += ["--map-by", "pe-list=" + ",".join(str(c) for c in self.cores) + ":ordered", "--bind-to", "core"]
        for k, v in self.mca:
            cmd += ["--mca", k, v]
        cmd += ["-x", f"OMP_NUM_THREADS={int(self.omp_threads)}"]
        return cmd + pw

    def command(self) -> str:
        return shlex.join(self.argv())

    @property
    def ranks(self) -> int:
        return len(self.cores) if self.cores else int(self.nprocs)

    def profile(self, pseudo_dir: os.PathLike | str):
        from ase.calculators.espresso import EspressoProfile

        return EspressoProfile(command=self.command(), pseudo_dir=str(pseudo_dir))

    def describe(self) -> dict[str, Any]:
        return {"command": self.command(), "ranks": self.ranks, "cores": list(self.cores),
                "npool": int(self.npool) if self.npool else self.ranks, "omp_threads": self.omp_threads,
                "pw_x": self.pw_x, "conda_build": conda_build(Path(self.pw_x))}


# ============================================================================
# one pw.x run
# ============================================================================
_TIME_RE = re.compile(r"PWSCF\s*:\s*(?P<cpu>[0-9hms. ]+?)\s*CPU\s+(?P<wall>[0-9hms. ]+?)\s*WALL")
#: the final SCF of a vc-relax in the basis of the relaxed cell: QE 7.5 prints 'Final scf calculation at the relaxed
#: structure', older releases 'A final scf calculation at the relaxed structure'
_FINAL_SCF_RE = re.compile(r"(?:A final scf|Final scf) calculation at the relaxed structure")


def parse_qe_time(s: str) -> Optional[float]:
    """'1h 2m', '3m45.67s', '12.34s' (pw.x clock format) -> seconds."""
    s = s.strip()
    if not s:
        return None
    tot, found = 0.0, False
    for val, unit in re.findall(r"([0-9.]+)\s*([hms])", s):
        tot += float(val) * {"h": 3600.0, "m": 60.0, "s": 1.0}[unit]
        found = True
    return tot if found else None


def parse_pw_output(text: str) -> dict[str, Any]:
    """Metadata of a pw.x output (version, timings, SCF/ionic counts, convergence flags, warnings)."""
    out: dict[str, Any] = {}
    m = re.search(r"Program PWSCF v\.(\S+)", text)
    out["qe_version"] = m.group(1) if m else "unknown"
    tm = list(_TIME_RE.finditer(text))
    out["qe_cpu_s"] = parse_qe_time(tm[-1].group("cpu")) if tm else None
    out["qe_wall_s"] = parse_qe_time(tm[-1].group("wall")) if tm else None
    its = [int(x) for x in re.findall(r"convergence has been achieved in\s+(\d+)\s+iterations", text)]
    out["scf_iterations"] = its
    out["n_scf_cycles"] = len(re.findall(r"^!\s", text, flags=re.M))
    out["scf_not_converged"] = "convergence NOT achieved" in text
    nk = re.findall(r"number of k points=\s*(\d+)", text)
    out["n_kpoints"] = int(nk[-1]) if nk else None
    npw = re.findall(r"Parallelization info.*?\n\s*sticks:.*?\n\s*Min.*?\n\s*Max.*?\n\s*Sum\s+\d+\s+\d+\s+\d+\s+(\d+)",
                     text, flags=re.S)
    out["n_g_vectors_rho"] = int(npw[-1]) if npw else None
    out["job_done"] = "JOB DONE." in text
    out["bfgs_converged"] = bool(re.search(r"bfgs converged", text))
    out["max_steps_reached"] = "The maximum number of steps has been reached" in text
    out["final_scf_after_vc_relax"] = bool(_FINAL_SCF_RE.search(text))
    errs = re.findall(r"%{5,}\n(.*?)\n\s*%{5,}", text, flags=re.S)
    out["errors"] = [" ".join(e.split()) for e in errs]
    warns = sorted({" ".join(w.split()) for w in re.findall(r"^\s*(?:Warning|WARNING|Message from routine)[^\n]*", text,
                                                               flags=re.M)})
    out["warnings"] = warns[:20]
    return out


@dataclasses.dataclass
class PWResult:
    """Outcome of one pw.x execution (final ionic configuration)."""

    label: str
    calculation: str
    directory: Path
    atoms: Any  # final ase.Atoms (positions/cell of the last SCF) with a SinglePointCalculator
    energy_eV: float
    forces_eV_A: np.ndarray
    stress_GPa: np.ndarray  # voigt 6 (xx,yy,zz,yz,xz,xy), positive = tensile (ASE convention)
    wall_s: float
    meta: dict[str, Any]
    kmesh: tuple[int, int, int]
    settings: dict[str, Any]
    runner: dict[str, Any]
    input_sha256: str

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def energy_per_atom_eV(self) -> float:
        return self.energy_eV / len(self.atoms)

    @property
    def fmax(self) -> float:
        f = np.asarray(self.forces_eV_A)
        return float(np.max(np.linalg.norm(f, axis=1))) if f.size else 0.0

    @property
    def converged(self) -> bool:
        m = self.meta
        ok = m["job_done"] and not m["scf_not_converged"] and not m["errors"]
        if self.calculation in ("relax", "vc-relax"):
            ok = ok and m["bfgs_converged"] and not m["max_steps_reached"]
        return bool(ok)

    def timing_row(self) -> dict[str, Any]:
        m = self.meta
        return {"label": self.label, "calculation": self.calculation, "n_atoms": self.n_atoms,
                "formula": self.atoms.get_chemical_formula(), "kmesh": list(self.kmesh),
                "n_kpoints_irreducible": m.get("n_kpoints"), "ecutwfc_ry": self.settings["ecutwfc_ry"],
                "ecutrho_ry": self.settings["ecutrho_ry"], "ranks": self.runner["ranks"],
                "cores": self.runner["cores"], "npool": self.runner["npool"], "wall_s": round(self.wall_s, 3),
                "qe_wall_s": m.get("qe_wall_s"), "qe_cpu_s": m.get("qe_cpu_s"),
                "scf_cycles": m.get("n_scf_cycles"), "scf_iterations_total": int(sum(m.get("scf_iterations", []))),
                "converged": self.converged, "directory": str(self.directory)}


def _input_data(settings: DFTSettings, nat: int, calculation: str, relax: Optional[dict]) -> dict:
    control = {"calculation": calculation, "tstress": True, "tprnfor": True, "outdir": "./tmp", "prefix": "pw",
               "disk_io": "low", "verbosity": "low"}
    system: dict[str, Any] = {"ecutwfc": float(settings.ecutwfc_ry), "ecutrho": settings.ecutrho,
                              "occupations": settings.occupations}
    if settings.occupations == "smearing":
        system.update(smearing=settings.smearing, degauss=float(settings.degauss_ry))
    if settings.input_dft:
        system["input_dft"] = settings.input_dft
    electrons = {"conv_thr": float(settings.conv_thr_ry_per_atom) * nat, "mixing_beta": float(settings.mixing_beta),
                 "electron_maxstep": 200}
    data: dict[str, dict] = {"control": control, "system": system, "electrons": electrons}
    if calculation in ("relax", "vc-relax"):
        r = dict(relax or {})
        control["etot_conv_thr"] = float(r.get("etot_conv_thr_ry", 1.0e-7 * nat))
        control["forc_conv_thr"] = float(r.get("forc_conv_thr_ry_bohr", 1.0e-5))
        control["nstep"] = int(r.get("nstep", 200))
        data["ions"] = {"ion_dynamics": "bfgs"}
        if calculation == "vc-relax":
            data["cell"] = {"cell_dynamics": "bfgs", "press": float(r.get("press_GPa", 0.0)) * KBAR_PER_GPA,
                            "press_conv_thr": float(r.get("press_conv_thr_GPa", 0.01)) * KBAR_PER_GPA,
                            "cell_dofree": str(r.get("cell_dofree", "all"))}
    for nl, key, val in settings.extra_input:
        data.setdefault(nl, {})[key] = val
    return data


def ase_calculator(atoms, settings: DFTSettings, runner: QERunner, directory: os.PathLike | str, *,
                   calculation: str = "scf", relax: Optional[dict] = None, kmesh: Optional[Sequence[int]] = None,
                   pseudo: Optional[PseudoFamily] = None):
    """ASE ``Espresso`` calculator (pinned-MPI ``EspressoProfile``) for `atoms` with these settings.

    For interactive ASE use (optimisers, NEB, phonopy-style displacements).  Harness protocols call
    :func:`run_pw`, which uses the same calculator and adds provenance, convergence flags and timing.
    """
    from ase.calculators.espresso import Espresso

    pseudo = pseudo or load_pseudo_family(settings.pseudo_family)
    km = tuple(int(k) for k in (kmesh if kmesh is not None else settings.kmesh(atoms)))
    return Espresso(profile=runner.profile(pseudo.directory), directory=str(directory),
                    pseudopotentials=pseudo.pseudopotentials(atoms.get_chemical_symbols()),
                    input_data=_input_data(settings, len(atoms), calculation, relax), kpts=km,
                    koffset=tuple(settings.koffset))


def run_pw(atoms, settings: DFTSettings, runner: QERunner, directory: os.PathLike | str, *, calculation: str = "scf",
           relax: Optional[dict] = None, kmesh: Optional[Sequence[int]] = None, label: str = "",
           pseudo: Optional[PseudoFamily] = None, keep_scratch: bool = False, reuse: bool = True,
           timing_log: Optional[os.PathLike | str] = None) -> PWResult:
    """Run pw.x once through the ASE Espresso calculator and read back the final configuration.

    ``reuse``: if ``directory`` already holds a finished output whose input file is byte-identical to the one
    this call would write, it is read instead of re-running pw.x (restartable scans).  The wavefunction
    scratch (``tmp/``) is deleted afterwards unless ``keep_scratch``.
    """
    from ase.io import read

    pseudo = pseudo or load_pseudo_family(settings.pseudo_family)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    nat = len(atoms)
    km = tuple(int(k) for k in (kmesh if kmesh is not None else settings.kmesh(atoms)))
    calc = ase_calculator(atoms, settings, runner, directory, calculation=calculation, relax=relax, kmesh=km,
                          pseudo=pseudo)
    pwi, pwo = directory / "espresso.pwi", directory / "espresso.pwo"
    at = atoms.copy()
    at.calc = calc
    # write the input first so an identical finished run can be reused
    calc.template.write_input(calc.profile, directory, at, dict(calc.parameters), ["energy", "forces", "stress"])
    in_sha = file_digest(pwi, "sha256")
    stamp = directory / "fmd_run.json"
    reused = False
    if reuse and pwo.is_file() and stamp.is_file():
        try:
            prev = json.loads(stamp.read_text())
            reused = prev.get("input_sha256") == in_sha and "JOB DONE." in pwo.read_text(errors="replace")
        except (OSError, ValueError):
            reused = False
    t0 = time.perf_counter()
    if reused:
        wall = float(json.loads(stamp.read_text()).get("wall_s", float("nan")))
    else:
        try:
            calc.template.execute(directory, calc.profile)
        except subprocess.CalledProcessError as exc:
            tail = pwo.read_text(errors="replace")[-3000:] if pwo.is_file() else ""
            err = (directory / "espresso.err")
            etail = err.read_text(errors="replace")[-2000:] if err.is_file() else ""
            raise QERunError(f"pw.x failed (exit {exc.returncode}) in {directory}:\n{etail}\n{tail}") from exc
        wall = time.perf_counter() - t0
    text = pwo.read_text(errors="replace")
    meta = parse_pw_output(text)
    if not meta["job_done"]:
        raise QERunError(f"pw.x did not finish in {directory} (no 'JOB DONE'); errors {meta['errors']}")
    final = read(pwo, format="espresso-out", index=-1)
    res = final.calc.results
    stress = np.asarray(res.get("stress", np.full(6, np.nan)), float).reshape(-1)
    if stress.size == 9:
        s = stress.reshape(3, 3)
        stress = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
    out = PWResult(label=label or directory.name, calculation=calculation, directory=directory, atoms=final,
                   energy_eV=float(res.get("free_energy", res["energy"])),
                   forces_eV_A=np.asarray(res.get("forces", np.zeros((nat, 3))), float),
                   stress_GPa=stress * EV_PER_A3_TO_GPA, wall_s=float(wall), meta=meta, kmesh=km,
                   settings=settings.describe(), runner=runner.describe(), input_sha256=in_sha)
    if not reused:
        stamp.write_text(json.dumps({"input_sha256": in_sha, "wall_s": wall, "label": out.label,
                                     "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    if not keep_scratch:
        shutil.rmtree(directory / "tmp", ignore_errors=True)
    log = Path(timing_log) if timing_log is not None else None
    if log is not None and not reused:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as fh:
            fh.write(json.dumps(out.timing_row()) + "\n")
    out.meta["reused_previous_output"] = reused
    return out


# ============================================================================
# card, provenance, records
# ============================================================================
def dft_card(settings: DFTSettings, elements: Iterable[str], *, qe_version: str = "unknown",
             pseudo: Optional[PseudoFamily] = None):
    """PotentialCard (kind ``dft``) for functional + pseudopotential files + code version."""
    from fracture_md.schema.potential import PotentialCard, PotentialKind, TrainingDomain

    pseudo = pseudo or load_pseudo_family(settings.pseudo_family)
    els = sorted(set(elements))
    files = pseudo.checksums(els)
    functional = settings.input_dft or pseudo.functional
    ident = {"code": f"Quantum ESPRESSO pw.x {qe_version}", "functional": functional, "pseudo_family": pseudo.name,
             "pseudopotentials_sha256": {k: v["sha256"] for k, v in files.items()}}
    psha = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    wfc, _rho = pseudo.recommended_cutoffs_ry(els)
    return PotentialCard(
        id=f"dft-qe{qe_version}-{functional.lower()}-{pseudo.name}@{psha[:8]}",
        name=f"DFT-{functional} (Quantum ESPRESSO {qe_version}, {pseudo.name})",
        kind=PotentialKind.DFT,
        functional_form=f"Kohn-Sham DFT, {functional} exchange-correlation, plane-wave basis with "
                        "ultrasoft/PAW/norm-conserving pseudopotentials (" +
                        ", ".join(f"{v['element']}: {k} [{v['type']}]" for k, v in files.items()) + ")",
        elements=els,
        cutoff=Quantity(value=wfc * RY_EV, unit=Unit.EV),
        parameters_sha256=psha,
        parameter_source=f"{pseudo.name} (Materials Cloud Archive, doi:10.24435/materialscloud:f3-ym); files "
                         + ", ".join(f"{k} sha256={v['sha256']}" for k, v in files.items()),
        license="SSSP library CC-BY-4.0; per-file licences of the pseudopotential families (pslibrary GPL-2+, "
                "GBRV GPL-3, SG15 CC-BY-SA-4.0, PseudoDojo CC-BY-4.0); see weights/pseudos/SSSP_LICENSE.txt",
        code_license="GPL-2.0-or-later (Quantum ESPRESSO)",
        implementation="fracture_md.engines.dft_qe (ASE Espresso calculator -> pw.x)",
        citations=[CITATIONS["qe2009"], CITATIONS["qe2017"], CITATIONS["pbe"], CITATIONS["sssp"], CITATIONS["sssp_data"]],
        training_domain=TrainingDomain(elements=els, reference_method=f"DFT-{functional}",
                                       description="first principles: no parameters fitted to the target "
                                                   "properties (functional and pseudopotential generation only)"),
        known_limitations=[
            "PBE (GGA) underbinds: equilibrium lattice parameters of sp solids typically ~0.5-1.5 % above "
            "experiment and elastic moduli correspondingly low (systematic functional error, not numerics)",
            "0 K static lattice: no zero-point or thermal expansion unless added explicitly",
            "no van der Waals dispersion (layered / molecular phases need a correction)",
            "Kohn-Sham gaps underestimated: gap closure under large strain may occur too early",
            "numbers depend on ecutwfc/ecutrho, the k-mesh and smearing: see each record's method and "
            "provenance.extra['dft'] and the convergence tables in docs/DFT.md",
        ],
        differentiable=False,
        precision="float64",
        notes="card.cutoff holds the largest SSSP-recommended plane-wave cutoff (ecutwfc) of these elements in eV; "
              "DFT has no real-space interaction cutoff.  The card id hashes code version, functional and "
              "pseudopotential files only; numerical settings are per record.",
    )


def _provenance(card, runs: Sequence[PWResult], settings: DFTSettings, runner: QERunner, label: str,
                pseudo: PseudoFamily, elements: Iterable[str], requested_by: str = "human"):
    import ase
    import sys

    from fracture_md import provenance

    try:
        own = provenance.source_sha256(sys.modules[__name__])
    except (OSError, TypeError):  # pragma: no cover
        own = "unknown"
    files = pseudo.checksums(elements)
    ver = runs[0].meta["qe_version"] if runs else "unknown"
    build = conda_build(Path(runner.pw_x))
    rows = [r.timing_row() for r in runs]
    return provenance.collect(
        engine=ENGINE_NAME,
        engine_version=f"Quantum ESPRESSO pw.x v.{ver} ({build or 'build unknown'}); ASE {ase.__version__}",
        potential_id=card.id,
        protocol_id=label,
        float_precision="float64",
        device=f"cpu: {runner.ranks} MPI rank(s)" + (f" pinned to cores {list(runner.cores)}" if runner.cores else ""),
        requested_by=requested_by,
        implementation_ids={"fracture_md.engines.dft_qe": own},
        input_hashes={**{f"pseudo:{k}": v["sha256"] for k, v in files.items()},
                      **{f"pw_input:{r.label}": r.input_sha256 for r in runs}},
        extra={"dft": {"code": "Quantum ESPRESSO pw.x", "qe_version": ver, "conda_build": build,
                       "functional": settings.input_dft or pseudo.functional, "pseudo_family": pseudo.name,
                       "pseudopotentials": files, "settings": settings.describe(),
                       "kmeshes": sorted({tuple(r.kmesh) for r in runs}), "runner": runner.describe(),
                       "spin": "unpolarised", "n_pw_runs": len(runs),
                       "wall_s_total": round(float(sum(r.wall_s for r in runs)), 3)},
               "qe_runs": rows},
    )


@dataclasses.dataclass(frozen=True)
class LiteratureValue:
    """A published value of the SAME model used as literature_model_value evidence.

    Evidence rule (docs/DFT.md sections 2 and 7): the same functional (PBE here, not merely another GGA such as PW91)
    and a comparable protocol (the same quantity, relaxation constraints and cell type).  The tolerance ``rel_tol``
    covers the spread between implementations of that one functional (pseudopotentials, basis, k-sampling): about
    1 % for lattice parameters and 2-10 % for elastic constants, strengths and transition pressures.  Values of
    other functionals or protocols, figure readings and preprints are comparisons: keep them in the data files, do
    not attach them as evidence."""

    name: PropertyName
    value: float
    unit: Unit
    citation: str
    rel_tol: float
    doi: Optional[str] = None
    notes: str = ""
    index: Optional[tuple[int, ...]] = None  # element of an array record, e.g. (0, 0) for C11

    def evidence(self, computed: float, computed_unit: Unit) -> ValidationEvidence:
        ref = convert(self.value, self.unit, computed_unit)
        rd = (computed - ref) / ref if ref != 0 else float("inf")
        cite = self.citation + (f" doi:{self.doi}" if self.doi else "")
        return ValidationEvidence(
            kind="literature_model_value",
            description=(f"published value of the same functional{(' ' + str(list(self.index))) if self.index else ''}"
                         f": {self.value:g} {self.unit.value}; tolerance {100 * self.rel_tol:.1f} % "
                         "(spread expected between PBE implementations, pseudopotentials and protocols). "
                         + self.notes).strip(),
            reference_value=Quantity(value=float(ref), unit=computed_unit),
            computed_value=Quantity(value=float(computed), unit=computed_unit),
            relative_difference=float(rd), citation=cite, passed=bool(abs(rd) <= self.rel_tol))


def _evidence_for(name: PropertyName, value: Any, unit: Unit, refs: Sequence[LiteratureValue]) -> list[ValidationEvidence]:
    ev = []
    for r in refs or ():
        if r.name != name:
            continue
        v = np.asarray(value, float)
        comp = float(v[r.index]) if r.index is not None else (float(v) if v.ndim == 0 else None)
        if comp is None:
            continue
        ev.append(r.evidence(comp, unit))
    return ev


def _check(name, passed, metric=None, tol=None, unit=None, details="") -> CheckResult:
    return CheckResult(name=name, passed=bool(passed),
                       metric=None if metric is None or not np.isfinite(metric) else float(metric),
                       tolerance=None if tol is None else float(tol), unit=unit, details=details)


def _run_checks(runs: Sequence[PWResult]) -> list[CheckResult]:
    bad = [r.label for r in runs if not r.converged]
    worst_it = max((max(r.meta["scf_iterations"] or [0]) for r in runs), default=0)
    return [_check("pw_runs_converged", not bad, float(len(bad)), 0.0, None,
                   f"{len(runs)} pw.x runs: JOB DONE, SCF converged, BFGS converged for relaxations"
                   + (f"; not converged: {bad}" if bad else "") + f"; max SCF iterations {worst_it}")]


def reference_candidate(phase_name: str):
    """Reference Candidate for Si (diamond), C (diamond) or 3C-SiC (zincblende)."""
    from fracture_md.schema.material import Candidate, CandidateKind, Composition, Phase

    table = {
        "Si": ("Si", {"Si": 1.0}, Phase(name="Si (diamond)", prototype="diamond (A4)", space_group_symbol="Fd-3m",
                                        space_group_number=227, crystal_system="cubic",
                                        external_ids={"materials_project": "mp-149"})),
        "C": ("C", {"C": 1.0}, Phase(name="C (diamond)", prototype="diamond (A4)", space_group_symbol="Fd-3m",
                                     space_group_number=227, crystal_system="cubic",
                                     external_ids={"materials_project": "mp-66"})),
        "3C-SiC": ("SiC", {"Si": 1.0, "C": 1.0}, Phase(name="3C-SiC", prototype="zincblende (B3)",
                                                       space_group_symbol="F-43m", space_group_number=216,
                                                       polytype="3C", crystal_system="cubic",
                                                       external_ids={"materials_project": "mp-8062"})),
    }
    key = {"si": "Si", "c": "C", "diamond": "C", "sic": "3C-SiC", "csi": "3C-SiC", "3c-sic": "3C-SiC"}.get(
        phase_name.lower(), phase_name)
    f, els, ph = table[key]
    return Candidate(id=f"ref-{key.lower()}", name=ph.name, kind=CandidateKind.REFERENCE,
                     composition=Composition(formula=f, elements=els), phases=[ph])


def _space_group_number(atoms, symprec: float = 0.1) -> Optional[int]:
    """spglib space-group number of an ASE cell (None when spglib is unavailable or finds none)."""
    try:
        import spglib
    except ImportError:  # pragma: no cover - spglib ships with the harness environment
        return None
    ds = spglib.get_symmetry_dataset((np.asarray(atoms.cell), atoms.get_scaled_positions(),
                                      atoms.get_atomic_numbers()), symprec=symprec)
    if ds is None:
        return None
    return int(getattr(ds, "number", None) or ds["number"])


def _auto_candidate(atoms):
    """Reference candidate for a diamond Si / C cell (Fd-3m, #227) or a zincblende 3C-SiC cell (F-43m, #216),
    recognised by its spglib space group (symprec 0.1 A: a 1 % elastic strain keeps it).  Any other
    cell (rocksalt, hexagonal polytypes, amorphous or strongly sheared cells) gets a composition-only candidate, so
    that a space-group check against the reference phase cannot fail with a misleading candidate."""
    from fracture_md.schema.material import Candidate, CandidateKind, Composition

    els = set(atoms.get_chemical_symbols())
    names = sorted(els)
    key = "Si" if names == ["Si"] else ("C" if names == ["C"] else ("3C-SiC" if names == ["C", "Si"] else None))
    if key is not None:
        cand = reference_candidate(key)
        sg = _space_group_number(atoms)
        if sg is None or sg == cand.phases[0].space_group_number:
            return cand
    f = atoms.get_chemical_formula(mode="hill", empirical=True)  # ASE has no mode "empirical"
    syms = atoms.get_chemical_symbols()
    return Candidate(id=f"dft-{f}", name=f, kind=CandidateKind.COMPOSITION,
                     composition=Composition(formula=f, elements={e: float(syms.count(e)) for e in sorted(els)}))


def primitive_cell(phase_name: str, a: float):
    """2-atom fcc primitive cell (cubic Cartesian axes) of Si / C diamond or 3C-SiC with conventional a (Å)."""
    from ase.build import bulk

    key = phase_name.lower()
    if key in ("si",):
        return bulk("Si", "diamond", a=a)
    if key in ("c", "diamond"):
        return bulk("C", "diamond", a=a)
    if key in ("sic", "3c-sic"):
        return bulk("SiC", "zincblende", a=a)
    raise ValueError(phase_name)


class _Builder:
    """Shared record construction for one protocol call."""

    def __init__(self, candidate, card, method: MethodSpec, prov, conditions: Conditions,
                 refs: Sequence[LiteratureValue] = ()):
        self.candidate, self.card, self.method, self.prov, self.cond = candidate, card, method, prov, conditions
        self.refs = list(refs or ())
        self.records: list[PropertyRecord] = []

    def __call__(self, name: PropertyName, value, unit: Unit, *, checks: Sequence[CheckResult] = (), shape: str = "",
                 frame: str = "not_applicable", notes: str = "", uncertainty: Sequence[Uncertainty] = (),
                 orientation=None) -> PropertyRecord:
        from fracture_md.records import make_record

        v = value.tolist() if isinstance(value, np.ndarray) else value
        r = make_record(name=name, value=v, unit=unit, candidate=self.candidate, method=self.method, card=self.card,
                        conditions=self.cond, checks=list(checks), evidence=_evidence_for(name, value, unit, self.refs),
                        shape_description=shape, frame=frame, notes=notes, provenance=self.prov,
                        uncertainty=list(uncertainty), orientation=orientation,
                        citations=[CITATIONS["qe2009"], CITATIONS["pbe"], CITATIONS["sssp"]])
        self.records.append(r)
        return r


def _method(name: str, description: str, assumptions: Sequence[str], label: str) -> MethodSpec:
    return MethodSpec(name=name, protocol_id=label, engine=ENGINE_NAME, description=description,
                      assumptions=["0 K static lattice (no zero-point or thermal effects)", "spin-unpolarised",
                                   "defect-free periodic crystal", *assumptions])


def _settings_text(settings: DFTSettings, km, pseudo: PseudoFamily) -> str:
    occ = (f"{settings.smearing} smearing {settings.degauss_ry:g} Ry" if settings.occupations == "smearing"
           else settings.occupations + " occupations")
    return (f"pw.x, {settings.input_dft or pseudo.functional}, {pseudo.name}, ecutwfc {settings.ecutwfc_ry:g} Ry / "
            f"ecutrho {settings.ecutrho:g} Ry, k-mesh {tuple(km)} offset {tuple(settings.koffset)}, {occ}, "
            f"conv_thr {settings.conv_thr_ry_per_atom:g} Ry/atom")


def _voigt(e) -> np.ndarray:
    e = np.asarray(e, float)
    return np.array([[e[0], e[5] / 2, e[4] / 2], [e[5] / 2, e[1], e[3] / 2], [e[4] / 2, e[3] / 2, e[2]]])


def strained(atoms, e_voigt) -> Any:
    """Affinely strained copy x -> (I + eps) x (symmetric eps, engineering shear strains in e_voigt)."""
    F = np.eye(3) + _voigt(e_voigt)
    at = atoms.copy()
    at.set_cell(np.asarray(atoms.cell) @ F.T, scale_atoms=True)
    return at


def _cubic_frame_ok(atoms, sg) -> bool:
    """True if every Cartesian point-group operation is a signed permutation (cubic axes = Cartesian axes)."""
    R = np.asarray(sg.rotations_cart)
    return bool(np.all(np.abs(np.abs(R) - np.round(np.abs(R))) < 1e-6))


# ============================================================================
# protocols
# ============================================================================
def single_point(atoms, *, settings: DFTSettings, runner: QERunner, workdir: os.PathLike | str, candidate=None,
                 label: str = "single_point", references: Sequence[LiteratureValue] = (), stress_state: str = "",
                 requested_by: str = "human", kmesh: Optional[Sequence[int]] = None) -> ProtocolOutput:
    """SCF energy, forces and Cauchy stress at the given cell (records ``energy_per_atom`` and ``cell_stress``)."""
    pseudo = load_pseudo_family(settings.pseudo_family)
    workdir = Path(workdir)
    r = run_pw(atoms, settings, runner, workdir / label, label=label, pseudo=pseudo, kmesh=kmesh,
               timing_log=workdir / "timings.jsonl")
    els = atoms.get_chemical_symbols()
    card = dft_card(settings, els, qe_version=r.meta["qe_version"], pseudo=pseudo)
    prov = _provenance(card, [r], settings, runner, label, pseudo, els, requested_by)
    cell = np.asarray(atoms.cell).round(10).tolist()
    cond = Conditions(stress_state=stress_state or f"fixed cell (rows, angstrom) {cell}; ions fixed")
    method = _method("DFT single point (energy, forces, stress)", _settings_text(settings, r.kmesh, pseudo),
                     ["ions held at the given positions"], label)
    checks = _run_checks([r])
    b = _Builder(candidate or _auto_candidate(atoms), card, method, prov,
                 cond, references)
    b(PropertyName.ENERGY_PER_ATOM, r.energy_per_atom_eV, Unit.EV_PER_ATOM, checks=checks,
      notes="pseudopotential total (free) energy per atom: meaningful only relative to energies with the same "
            "functional, pseudopotentials and cutoffs")
    b(PropertyName.CELL_STRESS, r.stress_GPa, Unit.GPA, checks=checks, frame="lab",
      shape="voigt 6 (xx,yy,zz,yz,xz,xy) in the Cartesian frame of the cell, positive = tensile",
      notes=f"max |F| = {r.fmax:.3e} eV/angstrom at the given positions")
    out = ProtocolOutput(properties=b.records, checks=checks)
    out.artifacts.update(atoms=r.atoms, forces_eV_A=r.forces_eV_A, stress_GPa=r.stress_GPa)
    out.raw.update(qe_runs=[r.timing_row()], energy_eV=r.energy_eV, stress_GPa=r.stress_GPa.tolist(),
                   fmax_eV_A=r.fmax)
    return out


def relax_structure(atoms, *, settings: DFTSettings, runner: QERunner, workdir: os.PathLike | str, candidate=None,
                    cell: str = "full", press_GPa: float = 0.0, forc_conv_thr_ry_bohr: float = 1.0e-5,
                    press_conv_thr_GPa: float = 0.01, stress_tol_GPa: float = 0.1, label: str = "relax",
                    references: Sequence[LiteratureValue] = (), requested_by: str = "human",
                    kmesh: Optional[Sequence[int]] = None) -> ProtocolOutput:
    """QE BFGS relaxation: ``cell="full"`` (vc-relax, all cell dof), ``"shape"``/``"volume"`` etc. (pw.x
    cell_dofree), or ``"fixed"`` (positions only).  Records lattice parameter, volume, density, energy and
    the residual stress of the final SCF (which, for vc-relax, pw.x repeats with the final G-vector set)."""
    from fracture_md.structures.symmetry import conventional_cell, lattice_parameters, space_group
    from fracture_md.structures.system import AtomicSystem

    pseudo = load_pseudo_family(settings.pseudo_family)
    workdir = Path(workdir)
    calc = "relax" if cell == "fixed" else "vc-relax"
    relax = {"forc_conv_thr_ry_bohr": forc_conv_thr_ry_bohr, "press_GPa": press_GPa,
             "press_conv_thr_GPa": press_conv_thr_GPa, "cell_dofree": "all" if cell == "full" else cell}
    r = run_pw(atoms, settings, runner, workdir / label, calculation=calc, relax=relax, label=label, pseudo=pseudo,
               kmesh=kmesh, timing_log=workdir / "timings.jsonl")
    at = r.atoms
    els = at.get_chemical_symbols()
    n, V = len(at), float(at.get_volume())
    rho = float(np.sum(at.get_masses())) / V * AMU_PER_A3_TO_G_PER_CM3
    sig = r.stress_GPa
    s = AtomicSystem.from_ase(at)
    sg = space_group(s)
    conv = conventional_cell(s)
    lengths, angles = lattice_parameters(np.asarray(conv.box if conv is not None else s.box))
    card = dft_card(settings, els, qe_version=r.meta["qe_version"], pseudo=pseudo)
    prov = _provenance(card, [r], settings, runner, label, pseudo, els, requested_by)
    checks = _run_checks([r])
    fmax_tol = forc_conv_thr_ry_bohr * RY_EV / 0.529177210903 * 2.0
    checks.append(_check("max_force", r.fmax <= fmax_tol, r.fmax, fmax_tol, Unit.EV_PER_ANGSTROM,
                         "final SCF forces (2x the pw.x forc_conv_thr)"))
    if calc == "vc-relax":
        dev = sig + press_GPa * np.array([1, 1, 1, 0, 0, 0])
        smax = float(np.max(np.abs(dev)))
        checks.append(_check("residual_stress", smax < stress_tol_GPa, smax, stress_tol_GPa, Unit.GPA,
                             f"max |sigma_i + P delta_i| of the final SCF (target pressure {press_GPa:g} GPa)"))
    cand = candidate or _auto_candidate(at)
    exp_sg = cand.phases[0].space_group_number if cand.phases else None
    if exp_sg is not None:
        checks.append(_check("space_group", sg.number == exp_sg, float(sg.number), 0.0, None,
                             f"spglib: {sg.international} (#{sg.number}); expected #{exp_sg}"))
    stress_state = ("zero stress" if press_GPa == 0 else f"hydrostatic pressure {press_GPa:g} GPa") if calc == "vc-relax" \
        else "fixed cell"
    cond = Conditions(stress_state=stress_state)
    method = _method(f"DFT {calc} (BFGS)", _settings_text(settings, r.kmesh, pseudo),
                     [f"pw.x BFGS, forc_conv_thr {forc_conv_thr_ry_bohr:g} Ry/bohr"
                      + (f", press_conv_thr {press_conv_thr_GPa:g} GPa, cell_dofree {relax['cell_dofree']}"
                         if calc == "vc-relax" else "")], label)
    b = _Builder(cand, card, method, prov, cond, references)
    cubic = sg.crystal_system == "cubic"
    if cubic:
        b(PropertyName.LATTICE_PARAMETER, float(lengths.mean()), Unit.ANGSTROM, checks=checks,
          notes=f"conventional cubic a of {sg.international} from the vc-relax cell")
    else:
        b(PropertyName.LATTICE_PARAMETER, lengths, Unit.ANGSTROM, checks=checks, shape="conventional [a, b, c]",
          notes=f"angles {np.round(angles, 6).tolist()} degree; {sg.international}")
    b(PropertyName.VOLUME_PER_ATOM, V / n, Unit.ANGSTROM3_PER_ATOM, checks=checks)
    b(PropertyName.DENSITY, rho, Unit.G_PER_CM3, checks=checks, notes="ASE standard atomic masses / relaxed volume")
    b(PropertyName.ENERGY_PER_ATOM, r.energy_per_atom_eV, Unit.EV_PER_ATOM, checks=checks,
      notes="pseudopotential total (free) energy per atom (relative use only)")
    b(PropertyName.RESIDUAL_STRESS, sig, Unit.GPA, checks=checks, frame="lab",
      shape="voigt 6 (xx,yy,zz,yz,xz,xy), positive = tensile")
    out = ProtocolOutput(properties=b.records, checks=checks, structure=s)
    out.artifacts.update(atoms=at, stress_GPa=sig)
    out.raw.update(qe_runs=[r.timing_row()], space_group={"number": sg.number, "symbol": sg.international},
                   lattice_a_A=float(lengths.mean()) if cubic else lengths.tolist(), energy_eV=r.energy_eV,
                   residual_stress_GPa=sig.tolist(), fmax_eV_A=r.fmax)
    return out


def birch_murnaghan_fit(volumes: Sequence[float], energies: Sequence[float]) -> dict[str, float]:
    """3rd-order Birch-Murnaghan E(V) fit (linear least squares in x = V^(-2/3), then exact parameters).

    E(V) = E0 + 9 V0 B0 / 16 * { [(V0/V)^(2/3) - 1]^3 B0' + [(V0/V)^(2/3) - 1]^2 [6 - 4 (V0/V)^(2/3)] }.
    Units follow the inputs (eV, angstrom^3 -> B0 in eV/angstrom^3).  Returns V0, E0, B0, B0', the fit RMS
    and the pressure function coefficients.
    """
    V = np.asarray(volumes, float)
    E = np.asarray(energies, float)
    x = V ** (-2.0 / 3.0)
    c = np.polyfit(x, E, 3)  # E = c0 x^3 + c1 x^2 + c2 x + c3
    p = np.poly1d(c)
    dp = p.deriv()
    roots = [r.real for r in dp.roots if abs(r.imag) < 1e-12 and r.real > 0]
    xmid = float(np.mean(x))
    if not roots:
        raise ValueError("Birch-Murnaghan fit has no minimum")
    x0 = min(roots, key=lambda r: abs(r - xmid))
    if p.deriv(2)(x0) <= 0:
        raise ValueError("Birch-Murnaghan fit stationary point is not a minimum")
    V0 = x0 ** (-1.5)
    # B = V d2E/dV2 ; with x = V^-2/3: dE/dV = E_x * (-2/3) V^-5/3 ; d2E/dV2 = E_xx (4/9) V^-10/3 + E_x (10/9) V^-8/3
    def pressure(v):
        xv = v ** (-2.0 / 3.0)
        return -dp(xv) * (-2.0 / 3.0) * v ** (-5.0 / 3.0)

    def bulk(v):
        xv = v ** (-2.0 / 3.0)
        return v * (p.deriv(2)(xv) * (4.0 / 9.0) * v ** (-10.0 / 3.0) + dp(xv) * (10.0 / 9.0) * v ** (-8.0 / 3.0))

    B0 = float(bulk(V0))
    h = 1e-4 * V0
    dBdV = (bulk(V0 + h) - bulk(V0 - h)) / (2 * h)
    dPdV = (pressure(V0 + h) - pressure(V0 - h)) / (2 * h)
    Bp = float(dBdV / dPdV)
    resid = E - p(x)
    return {"V0": float(V0), "E0": float(p(x0)), "B0": B0, "B0_prime": Bp,
            "rms_residual": float(np.sqrt(np.mean(resid ** 2))), "max_residual": float(np.max(np.abs(resid))),
            "pressure": pressure, "poly": p}


def equation_of_state(atoms, *, settings: DFTSettings, runner: QERunner, workdir: os.PathLike | str, candidate=None,
                      linear_strains: Sequence[float] = tuple(np.linspace(-0.02, 0.02, 9)), relax_internal: bool = False,
                      label: str = "eos", references: Sequence[LiteratureValue] = (), requested_by: str = "human",
                      kmesh: Optional[Sequence[int]] = None, pulay_tol_GPa: float = 0.5,
                      fit_rms_tol_meV_per_atom: float = 0.1) -> ProtocolOutput:
    """E(V) on an isotropic strain grid (cell x (1+s)), fixed k-mesh, 3rd-order Birch-Murnaghan fit.

    Checks: all runs converged; fit RMS; V0 inside the sampled range; |P_stress - P_fit| (the stress from pw.x
    vs -dE/dV of the fit; they differ by the Pulay stress at an unconverged basis).  ``relax_internal`` relaxes
    the ions at every volume (pw.x relax) for structures with free internal coordinates.
    """
    from fracture_md.structures.symmetry import conventional_cell, lattice_parameters, space_group
    from fracture_md.structures.system import AtomicSystem

    pseudo = load_pseudo_family(settings.pseudo_family)
    workdir = Path(workdir)
    km = tuple(kmesh) if kmesh is not None else settings.kmesh(atoms)
    runs = []
    for i, s in enumerate(linear_strains):
        at = strained(atoms, [s, s, s, 0, 0, 0])
        runs.append(run_pw(at, settings, runner, workdir / label / f"v{i:02d}_{s:+.4f}",
                           calculation="relax" if relax_internal else "scf", kmesh=km,
                           label=f"{label}:v{i:02d}", pseudo=pseudo, timing_log=workdir / "timings.jsonl"))
    n = len(atoms)
    Vs = np.array([r.atoms.get_volume() / n for r in runs])
    Es = np.array([r.energy_per_atom_eV for r in runs])
    Ps = np.array([-np.mean(r.stress_GPa[:3]) for r in runs])
    fit = birch_murnaghan_fit(Vs, Es)
    B0 = fit["B0"] * EV_PER_A3_TO_GPA
    P_fit = np.array([fit["pressure"](v) for v in Vs]) * EV_PER_A3_TO_GPA
    pulay = float(np.max(np.abs(Ps - P_fit)))
    rms = fit["rms_residual"] * 1e3
    inside = bool(Vs.min() < fit["V0"] < Vs.max())
    checks = _run_checks(runs) + [
        _check("eos_fit_rms", rms < fit_rms_tol_meV_per_atom, rms, fit_rms_tol_meV_per_atom, Unit.MEV_PER_ATOM,
               "RMS residual of the 3rd-order Birch-Murnaghan fit"),
        _check("eos_minimum_inside_range", inside, fit["V0"], None, Unit.ANGSTROM3_PER_ATOM,
               f"sampled {Vs.min():.4f}-{Vs.max():.4f} angstrom^3/atom"),
        _check("eos_stress_vs_dEdV", pulay < pulay_tol_GPa, pulay, pulay_tol_GPa, Unit.GPA,
               "max |P(pw.x stress) - P(-dE/dV of the fit)| over the grid (Pulay stress + fit error)"),
    ]
    els = atoms.get_chemical_symbols()
    card = dft_card(settings, els, qe_version=runs[0].meta["qe_version"], pseudo=pseudo)
    prov = _provenance(card, runs, settings, runner, label, pseudo, els, requested_by)
    s0 = AtomicSystem.from_ase(atoms)
    sg = space_group(s0)
    conv = conventional_cell(s0)
    a_ref = float(lattice_parameters(np.asarray(conv.box))[0][0]) if conv is not None else None
    v_ref = atoms.get_volume() / n
    cond = Conditions(stress_state="zero pressure (E(V) minimum)")
    method = _method("DFT equation of state, 3rd-order Birch-Murnaghan E(V) fit",
                     _settings_text(settings, km, pseudo) + f"; {len(runs)} volumes, linear strains "
                     f"{float(min(linear_strains)):+.3f}..{float(max(linear_strains)):+.3f}, same k-mesh at every volume",
                     ["isotropic scaling of the cell" + (" with ions relaxed" if relax_internal else
                                                        " (fractional coordinates fixed)")], label)
    b = _Builder(candidate or _auto_candidate(atoms), card, method, prov, cond,
                 references)
    unc = [Uncertainty(kind=UncertaintyKind.NUMERICAL_CONVERGENCE, value=pulay, unit=Unit.GPA, interval="max_abs_diff",
                       description="stress vs dE/dV disagreement over the grid")]
    b(PropertyName.VOLUME_PER_ATOM, fit["V0"], Unit.ANGSTROM3_PER_ATOM, checks=checks, notes="Birch-Murnaghan V0")
    if sg.crystal_system == "cubic" and a_ref is not None:
        b(PropertyName.LATTICE_PARAMETER, a_ref * (fit["V0"] / v_ref) ** (1 / 3), Unit.ANGSTROM, checks=checks,
          notes=f"conventional cubic a0 = a_ref (V0/V_ref)^(1/3) of {sg.international}")
    b(PropertyName.BULK_MODULUS_EOS, B0, Unit.GPA, checks=checks, uncertainty=unc,
      notes="B0 of the 3rd-order Birch-Murnaghan E(V) fit")
    b(PropertyName.BULK_MODULUS_PRESSURE_DERIVATIVE, fit["B0_prime"], Unit.DIMENSIONLESS, checks=checks,
      notes="B0' of the 3rd-order Birch-Murnaghan E(V) fit (sensitive to the strain range)")
    b(PropertyName.ENERGY_PER_ATOM, fit["E0"], Unit.EV_PER_ATOM, checks=checks,
      notes="E0 of the fit; pseudopotential total energy (relative use only)")
    out = ProtocolOutput(properties=b.records, checks=checks, structure=s0)
    out.raw.update(qe_runs=[r.timing_row() for r in runs], volumes_A3_per_atom=Vs.tolist(),
                   energies_eV_per_atom=Es.tolist(), pressures_stress_GPa=Ps.tolist(), pressures_fit_GPa=P_fit.tolist(),
                   V0=fit["V0"], E0=fit["E0"], B0_GPa=B0, B0_prime=fit["B0_prime"], fit_rms_meV_per_atom=rms,
                   a0_A=(a_ref * (fit["V0"] / v_ref) ** (1 / 3)) if a_ref else None, kmesh=list(km))
    return out


def elastic_constants(atoms, *, settings: DFTSettings, runner: QERunner, workdir: os.PathLike | str, candidate=None,
                      strains: Sequence[float] = (0.005, 0.01), mode: str = "cubic", relax_internal: bool = True,
                      clamped: bool = True, label: str = "elastic", references: Sequence[LiteratureValue] = (),
                      requested_by: str = "human", kmesh: Optional[Sequence[int]] = None,
                      fd_spread_tol_GPa: float = 2.0, reference_stress_tol_GPa: float = 0.2,
                      forc_conv_thr_ry_bohr: float = 1.0e-5) -> ProtocolOutput:
    """Relaxed-ion elastic constants from central finite-strain stress differences.

    The reference cell should be at zero stress with the SAME settings (e.g. from :func:`relax_structure` or the
    EOS a0); the k-mesh of the reference cell is kept for every strained cell.  ``mode="cubic"`` applies e1 and
    e4 (requires the cubic axes along x, y, z; checked) and assembles C11, C12, C44; ``mode="full"`` applies all
    six components.  The primary tensor uses the median magnitude of ``strains`` (the smaller one of two);
    the spread over magnitudes is the reported numerical uncertainty.
    """
    from fracture_md.physics.elastic import analyze_elastic
    from fracture_md.structures.symmetry import (
        VOIGT_LABELS,
        check_tensor_invariance,
        conventional_cell,
        cubic_pattern_deviation,
        cubic_voigt,
        lattice_parameters,
        space_group,
        voigt_to_tensor_stiffness,
    )
    from fracture_md.structures.system import AtomicSystem

    pseudo = load_pseudo_family(settings.pseudo_family)
    workdir = Path(workdir)
    km = tuple(kmesh) if kmesh is not None else settings.kmesh(atoms)
    s0 = AtomicSystem.from_ase(atoms)
    sg = space_group(s0)
    if mode == "cubic" and (sg.number < 195 or not _cubic_frame_ok(atoms, sg)):
        raise ValueError("mode='cubic' needs a cubic crystal with its cubic axes along Cartesian x, y, z")
    comps = (0, 3) if mode == "cubic" else tuple(range(6))
    runs: list[PWResult] = []
    ref = run_pw(atoms, settings, runner, workdir / label / "ref", kmesh=km, label=f"{label}:ref", pseudo=pseudo,
                 timing_log=workdir / "timings.jsonl")
    runs.append(ref)
    sig_ref = ref.stress_GPa
    relax = {"forc_conv_thr_ry_bohr": forc_conv_thr_ry_bohr, "etot_conv_thr_ry": 1.0e-8 * len(atoms)}
    per: dict[float, np.ndarray] = {}
    per_cl: dict[float, np.ndarray] = {}
    gamma: dict[float, np.ndarray] = {}
    fmax_worst = 0.0
    for d in [float(x) for x in strains]:
        C = np.full((6, 6), np.nan)
        Ccl = np.full((6, 6), np.nan)
        G = np.zeros((len(atoms), 3, 6))
        for j in comps:
            sig, sig_cl, disp = [], [], []
            for sgn in (+1, -1):
                e = np.zeros(6)
                e[j] = sgn * d
                at = strained(atoms, e)
                tag = f"e{j + 1}_{'p' if sgn > 0 else 'm'}{d:.4f}"
                needs_relax = relax_internal and _has_free_internal_coordinates(at)
                r = run_pw(at, settings, runner, workdir / label / tag, kmesh=km,
                           calculation="relax" if needs_relax else "scf", relax=relax, label=f"{label}:{tag}",
                           pseudo=pseudo, timing_log=workdir / "timings.jsonl")
                runs.append(r)
                fmax_worst = max(fmax_worst, r.fmax)
                sig.append(r.stress_GPa)
                u = r.atoms.get_positions() - at.get_positions()
                disp.append(u - u.mean(axis=0))
                moved = float(np.max(np.abs(disp[-1]))) > 1e-5  # angstrom; else clamped == relaxed
                if clamped and needs_relax and moved:
                    rc = run_pw(at, settings, runner, workdir / label / (tag + "_clamped"), kmesh=km,
                                label=f"{label}:{tag}_clamped", pseudo=pseudo, timing_log=workdir / "timings.jsonl")
                    runs.append(rc)
                    sig_cl.append(rc.stress_GPa)
                else:
                    sig_cl.append(r.stress_GPa)
            C[:, j] = (sig[0] - sig[1]) / (2 * d)
            Ccl[:, j] = (sig_cl[0] - sig_cl[1]) / (2 * d)
            G[:, :, j] = (disp[0] - disp[1]) / (2 * d)
        per[d], per_cl[d], gamma[d] = C, Ccl, G
    mags = sorted(per)
    primary = mags[len(mags) // 2] if len(mags) % 2 == 1 else mags[0]

    def assemble(M: np.ndarray) -> np.ndarray:
        if mode == "cubic":
            return cubic_voigt(float(M[0, 0]), float(0.5 * (M[1, 0] + M[2, 0])), float(M[3, 3]))
        return 0.5 * (M + M.T)

    Cs = assemble(per[primary])
    Ccs = assemble(per_cl[primary])
    spread = max(float(np.max(np.abs(assemble(per[m]) - Cs))) for m in mags)
    checks = _run_checks(runs)
    st = float(np.max(np.abs(sig_ref)))
    checks.append(_check("reference_zero_stress", st < reference_stress_tol_GPa, st, reference_stress_tol_GPa, Unit.GPA,
                         "max |sigma_i| of the unstrained reference cell (C = dsigma/de at zero stress)"))
    checks.append(_check("fd_strain_magnitude_convergence", spread < fd_spread_tol_GPa, spread, fd_spread_tol_GPa,
                         Unit.GPA, f"max |C(d) - C(d*={primary:g})| over d in {mags}"))
    fmax_tol = forc_conv_thr_ry_bohr * RY_EV / 0.529177210903 * 2.0
    checks.append(_check("fd_internal_relaxation_forces", fmax_worst <= max(fmax_tol, 1e-12), fmax_worst, fmax_tol,
                         Unit.EV_PER_ANGSTROM, "worst final force over the strained cells"))
    if mode == "cubic":
        raw_cols = per[primary][:, list(comps)]
        off = float(max(abs(raw_cols[1, 0] - raw_cols[2, 0]), np.max(np.abs(raw_cols[3:, 0])),
                        np.max(np.abs(np.delete(raw_cols[:, 1], 3)))))
        checks.append(_check("fd_cubic_pattern", off < fd_spread_tol_GPa, off, fd_spread_tol_GPa, Unit.GPA,
                             "|dsigma2/de1 - dsigma3/de1|, shear responses to e1 and non-44 responses to e4"))
    else:
        asym = float(np.max(np.abs(per[primary] - per[primary].T)))
        checks.append(_check("fd_major_symmetry", asym < fd_spread_tol_GPa, asym, fd_spread_tol_GPa, Unit.GPA))
        inv = check_tensor_invariance(Cs, sg.rotations_cart)
        checks.append(_check("fd_point_group_invariance", inv < fd_spread_tol_GPa, inv, fd_spread_tol_GPa, Unit.GPA,
                             f"{len(sg.rotations_cart)} operations of {sg.international}"))
        if sg.number >= 195:
            cp = cubic_pattern_deviation(Cs)
            checks.append(_check("fd_cubic_pattern", cp < fd_spread_tol_GPa, cp, fd_spread_tol_GPa, Unit.GPA))
    rho = float(np.sum(atoms.get_masses())) / atoms.get_volume() * AMU_PER_A3_TO_G_PER_CM3
    an = analyze_elastic(Cs, density_g_cm3=rho)
    born_ok = bool(an["born_stable"])
    conv = conventional_cell(s0)
    a_conv = float(lattice_parameters(np.asarray(conv.box))[0][0]) if conv is not None else None
    zeta = None
    if sg.number == 216 and a_conv and relax_internal:
        from fracture_md.physics.elastic import kleinman_parameter

        zeta = kleinman_parameter(gamma[primary], np.asarray(s0.species), a_conv)
    elif sg.number == 227 and a_conv and relax_internal:
        # diamond: the two atoms of the primitive cell are the two sublattices
        ux = gamma[primary][:, 0, 3]
        zeta = abs(float(ux[1] - ux[0])) / (a_conv / 4.0) if len(ux) == 2 else None

    els = atoms.get_chemical_symbols()
    card = dft_card(settings, els, qe_version=ref.meta["qe_version"], pseudo=pseudo)
    prov = _provenance(card, runs, settings, runner, label, pseudo, els, requested_by)
    cond = Conditions(stress_state="zero stress, 0 K lattice statics")
    method = _method("DFT finite-strain relaxed-ion elastic tensor",
                     _settings_text(settings, km, pseudo) + f"; C_ij = [s_i(+d e_j) - s_i(-d e_j)]/(2d), d = {primary:g} "
                     f"(also {[m for m in mags if m != primary]}), components {[c + 1 for c in comps]}, ions relaxed "
                     f"with pw.x BFGS to {forc_conv_thr_ry_bohr:g} Ry/bohr; same k-mesh for all strained cells",
                     ["linear elasticity (small strain)", "central differences (O(d^2) truncation)",
                      "cubic symmetry imposed when mode='cubic' (C11 from s1/e1, C12 from (s2+s3)/2e1, C44 from s4/e4)"
                      if mode == "cubic" else "tensor symmetrised (C + C^T)/2"], label)
    unc = [Uncertainty(kind=UncertaintyKind.NUMERICAL_CONVERGENCE, value=spread, unit=Unit.GPA, interval="max_abs_diff",
                       description=f"spread over strain magnitudes {mags}")]
    from fracture_md.schema.material import Orientation

    crystal = Orientation(description="crystal Cartesian frame (conventional cubic axes for cubic phases)")
    b = _Builder(candidate or _auto_candidate(atoms), card, method, prov, cond,
                 references)
    vo = "voigt 6x6, order " + ",".join(VOIGT_LABELS) + "; engineering shear strains"
    b(PropertyName.ELASTIC_STIFFNESS_VOIGT, Cs, Unit.GPA, checks=checks, shape=vo, frame="crystal",
      orientation=crystal, uncertainty=unc,
      notes=f"C11={Cs[0, 0]:.2f}, C12={Cs[0, 1]:.2f}, C44={Cs[3, 3]:.2f} GPa (relaxed ion)"
            + (f"; Kleinman zeta = {zeta:.3f}" if zeta is not None else ""))
    b(PropertyName.CLAMPED_ION_STIFFNESS_VOIGT, Ccs, Unit.GPA, checks=checks, shape=vo + " (clamped ion)",
      frame="crystal", orientation=crystal, notes=f"C44(clamped) = {Ccs[3, 3]:.2f} GPa")
    b(PropertyName.ELASTIC_COMPLIANCE_VOIGT, np.asarray(an["S_per_GPa"]), Unit.PER_GPA, checks=checks,
      shape=vo + " (compliance, S = C^-1)", frame="crystal", orientation=crystal)
    b(PropertyName.ELASTIC_STIFFNESS_TENSOR, voigt_to_tensor_stiffness(Cs), Unit.GPA, checks=checks,
      shape="3x3x3x3 C_ijkl", frame="crystal", orientation=crystal)
    poly = "isotropic untextured dense polycrystal (Voigt-Reuss-Hill average of the single-crystal C)"
    for pn, k in [(PropertyName.BULK_MODULUS_VOIGT, "B_voigt"), (PropertyName.BULK_MODULUS_REUSS, "B_reuss"),
                  (PropertyName.BULK_MODULUS_HILL, "B_hill"), (PropertyName.SHEAR_MODULUS_VOIGT, "G_voigt"),
                  (PropertyName.SHEAR_MODULUS_REUSS, "G_reuss"), (PropertyName.SHEAR_MODULUS_HILL, "G_hill"),
                  (PropertyName.YOUNGS_MODULUS_HILL, "E_hill")]:
        b(pn, float(an[k]), Unit.GPA, checks=checks, notes=poly)
    b(PropertyName.POISSON_RATIO_HILL, float(an["nu_hill"]), Unit.DIMENSIONLESS, checks=checks, notes=poly)
    if an["is_cubic"]:
        b(PropertyName.ZENER_ANISOTROPY, float(an["zener_ratio"]), Unit.DIMENSIONLESS, checks=checks,
          notes="A = 2 C44 / (C11 - C12)")
    b(PropertyName.BORN_STABILITY, float(an["born_min_eigenvalue"]), Unit.GPA, frame="crystal",
      checks=checks + [_check("born_stability", born_ok, an["born_min_eigenvalue"], 0.0, Unit.GPA)],
      shape="minimum eigenvalue of the 6x6 relaxed-ion C (>0: mechanically stable at zero stress)")
    out = ProtocolOutput(properties=b.records, checks=checks, structure=s0)
    out.artifacts.update(C_GPa=Cs, C_clamped_GPa=Ccs, analysis=an)
    out.raw.update(qe_runs=[r.timing_row() for r in runs], C_GPa=Cs.tolist(), C_clamped_GPa=Ccs.tolist(),
                   per_magnitude_GPa={str(k): np.nan_to_num(v).tolist() for k, v in per.items()},
                   primary_magnitude=primary, spread_GPa=spread, reference_stress_GPa=sig_ref.tolist(),
                   kleinman_zeta=zeta, conventional_a_A=a_conv, kmesh=list(km), mode=mode,
                   C11=float(Cs[0, 0]), C12=float(Cs[0, 1]), C44=float(Cs[3, 3]), B_voigt=float(an["B_voigt"]))
    return out


def _has_free_internal_coordinates(atoms) -> bool:
    """False for a 1-atom cell; otherwise True (pw.x stops after one SCF if the forces vanish by symmetry)."""
    return len(atoms) > 1


# ============================================================================
# convergence
# ============================================================================
@dataclasses.dataclass
class ConvergenceScan:
    parameter: str
    values: list
    rows: list[dict[str, Any]]
    reference: Any
    converged_value: Any
    energy_tol_meV_per_atom: float
    stress_tol_GPa: float
    check: CheckResult

    def table(self) -> str:
        head = f"{self.parameter:>12s} {'dE(meV/at)':>11s} {'max|ds|(GPa)':>13s} {'P(GPa)':>9s} {'wall(s)':>8s}"
        lines = [head]
        for r in self.rows:
            lines.append(f"{str(r['value']):>12s} {r['dE_meV_per_atom']:11.4f} {r['dstress_GPa']:13.4f} "
                         f"{r['pressure_GPa']:9.4f} {r['wall_s']:8.2f}")
        return "\n".join(lines)


def convergence_scan(atoms, *, settings: DFTSettings, runner: QERunner, workdir: os.PathLike | str, parameter: str,
                     values: Sequence[Any], energy_tol_meV_per_atom: float = 1.0, stress_tol_GPa: float = 0.5,
                     label: Optional[str] = None, strain: Optional[Sequence[float]] = None) -> ConvergenceScan:
    """Scan ``parameter`` ("ecutwfc": values in Ry, ecutrho = dual x ecutwfc; "kmesh": n or (n1,n2,n3)).

    The reference is the last (most accurate) value.  The converged value is the smallest value from which ALL
    larger values agree with the reference within both tolerances (energy per atom and max |sigma_i - sigma_i,ref|).
    ``strain`` (voigt 6) optionally strains the cell first, so stresses under load are tested too.
    """
    workdir = Path(workdir)
    label = label or f"conv_{parameter}"
    pseudo = load_pseudo_family(settings.pseudo_family)
    at0 = strained(atoms, strain) if strain is not None else atoms
    results = []
    for v in values:
        if parameter == "ecutwfc":
            st, km, tag = settings.replace(ecutwfc_ry=float(v), ecutrho_ry=None), None, f"ecut{float(v):06.1f}"
        elif parameter == "kmesh":
            kk = (int(v),) * 3 if np.isscalar(v) else tuple(int(x) for x in v)
            st, km, tag = settings, kk, "k" + "x".join(str(x) for x in kk)
        else:
            raise ValueError(parameter)
        r = run_pw(at0, st, runner, workdir / label / tag, kmesh=km, label=f"{label}:{tag}", pseudo=pseudo,
                   timing_log=workdir / "timings.jsonl")
        results.append((v, r))
    ref = results[-1][1]
    rows = []
    for v, r in results:
        dE = abs(r.energy_per_atom_eV - ref.energy_per_atom_eV) * 1e3
        ds = float(np.max(np.abs(r.stress_GPa - ref.stress_GPa)))
        rows.append({"value": v if np.isscalar(v) else list(v), "energy_eV_per_atom": r.energy_per_atom_eV,
                     "stress_GPa": r.stress_GPa.tolist(), "pressure_GPa": float(-np.mean(r.stress_GPa[:3])),
                     "dE_meV_per_atom": dE, "dstress_GPa": ds, "ok": dE <= energy_tol_meV_per_atom and ds <= stress_tol_GPa,
                     "wall_s": r.wall_s, "kmesh": list(r.kmesh), "n_kpoints": r.meta.get("n_kpoints"),
                     "ecutwfc_ry": r.settings["ecutwfc_ry"], "ecutrho_ry": r.settings["ecutrho_ry"],
                     "converged_scf": r.converged})
    conv = None
    for i in range(len(rows) - 1):  # the reference itself is excluded as a candidate
        if all(x["ok"] for x in rows[i:]):
            conv = rows[i]["value"]
            break
    chk = _check(f"{parameter}_converged", conv is not None, None, energy_tol_meV_per_atom, Unit.MEV_PER_ATOM,
                 f"smallest {parameter} with |dE| <= {energy_tol_meV_per_atom} meV/atom and max |dsigma| <= "
                 f"{stress_tol_GPa} GPa vs the reference {rows[-1]['value']} for all larger values: {conv}")
    return ConvergenceScan(parameter=parameter, values=list(values), rows=rows, reference=rows[-1]["value"],
                           converged_value=conv, energy_tol_meV_per_atom=energy_tol_meV_per_atom,
                           stress_tol_GPa=stress_tol_GPa, check=chk)
