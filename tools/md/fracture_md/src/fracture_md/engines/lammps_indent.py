"""LAMMPS deck generator for indentation runs (the production engine for large cells).

Division of labour (user decision G03, synthesis steps 9-10): the JAX harness is the verified reference
engine for small cells and the record builder; LAMMPS runs production-size cells.  This module turns an
:class:`~fracture_md.schema.protocol.IndentationProtocol` and a slab from
:func:`~fracture_md.structures.indentation.build_indentation_slab` into a self-contained LAMMPS run
directory:

* ``slab.data``: the harness slab, atoms written with full float64 precision (types 1 = C, 2 = Si), so
  both engines start from bitwise the same configuration;
* ``in.berk_sdf.lmp`` and ``berk_sdf.py``: copies of the production deck and of the numpy SDF indenter
  in ``examples/indent`` (the same tip functional as ``physics.indenter``; LAMMPS couples it through
  ``fix external pf/callback``);
* the potential file (Vashishta 2007: the parameter text embedded in ``potentials.vashishta``, i.e.
  the LAMMPS-distributed ``SiC.vashishta``; its card id is re-derived from the file by the importer);
* ``lmp.args`` (one command-line argument per line: ``-in``, every ``-var``, ``-log``), ``run.sh`` (MPI
  launch with explicit core binding) and ``manifest.json`` (protocol, slab summary, variables, sha256 of
  every input file);
* a copy of the frozen indentation recipe (``recipe=``, default indent_recipe.DEFAULT_RECIPE = recipe_v2 of
  data/indentation; role "recipe") with its id, sha256, protocol-section digest and the protocol's deviations from
  it in ``manifest.json["recipe"]``
  (physics.indent_recipe; ``recipe=None`` writes none).  The importer's records carry the sha256 of the recipe
  they are built under (physics.indentation.records_from_lammps_run) and check the deck's protocol section against it.

The deck variables follow the protocol field by field (:func:`deck_variables`): tip geometry and K,
gap, speed, hold, unload, relaxation, temperature, time step, CSVR/Langevin boundary layer and tau, the
60/40 split of ``equilibration_ps`` between all-mobile Langevin and the boundary thermostat alone (the
harness ``prepare_indentation_slab`` split), log/frame periods, slice width and full-dump spacing; a
``quasistatic`` protocol runs one minimisation per ``depth_step_A`` (the G19 cross-engine check).
Imported runs (``io.lammps_indent``) keep the engine status cap of ``physics.indentation``.

Restarts and branches (critic item A4): with ``restarts=True`` both decks write ``restart_<tag>_<step>.bin`` plus the
tip-work state ``restart_<tag>_<step>.state.json`` at loading start, at every ``full_dump_every_A`` of depth and at the
end of the hold, and the run's constants once (``run_<tag>.vars``).  :func:`restart_args` turns a run's ``lmp.args``
into the arguments of a restarted segment (``branch="continue"``: the same run from that step, bitwise on the same
ranks; ``branch="unload"``: an unload branch from that depth), :func:`restart_points` lists what a run directory
holds.  :func:`write_atomistic_deck` is the generator of the rigid atomistic-tip deck (``in.berk_atomistic.lmp``,
the S1 tip), and :func:`tier_protocol` sizes a protocol by a cell tier ((L/a, t_el/a), e.g. the (4, 4) scoping tier
of the S1 run, examples/indent/s1/).

Core binding (machine rule): Open MPI 5 ignores a ``taskset`` mask with ``--bind-to core``; ranks must be
placed with ``mpirun --map-by pe-list=<cores>:ordered --bind-to core`` (:func:`mpirun_prefix`).  The X925
cores are reserved for production and never used by default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from fracture_md.physics.indent_recipe import DEFAULT_RECIPE

PathLike = Union[str, Path]

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "indent"
SDF_DECK = "in.berk_sdf.lmp"
ATOMISTIC_DECK = "in.berk_atomistic.lmp"
SDF_MODULE = "berk_sdf.py"
DEFAULT_LAMMPS_ENV = Path.home() / "miniconda3" / "envs" / "fmd-lammps"
TYPE_MAP: dict[str, int] = {"C": 1, "Si": 2}  # the importer's default type map (io.lammps_indent.DEFAULT_TYPE_MAP)
MASSES_AMU: dict[str, float] = {"C": 12.011, "Si": 28.0855}
X925_CORES: tuple[int, ...] = (5, 6, 7, 8, 9, 15, 16, 17, 18, 19)  # fast cores: reserved for production runs
A725_CORES: tuple[int, ...] = (0, 1, 2, 3, 4, 10, 11, 12, 13, 14)
GENERATOR = "fracture_md.engines.lammps_indent"
GENERATOR_VERSION = 2  # 2: depth-aligned segments, restarts + tip-work state, unload to zero force, atomistic deck
DEFAULT_UNLOAD_CLEARANCE_A = 1.0  # retraction beyond the zero-force point before the relaxation
# recipe_v0 protocol/preparation/zscale (G18/B3; r4-kokkos): after the CG, z of the mobile atoms is scaled by
# a(300 K) / c(0 K) of Vashishta 2007 about the top of the fixed layer, against the slab breathing mode.  The MD decks
# apply it by default at T > 0 (deck_variables, atomistic_deck_variables); 1.0 at 0 K (no thermal expansion).
DEFAULT_ZSCALE = 1.004897
# rigid atomistic tip (in.berk_atomistic.lmp; the G06 tip): WCA wall eps, sigma and the tip height above the apex
ATOMISTIC_WCA_EPS_EV = 0.1
ATOMISTIC_WCA_SIGMA_A = 2.0
# LAMMPS pair styles per harness potential (the table variant only after the table P-h check, G07-R5)
PAIR_STYLES: dict[str, tuple[str, ...]] = {"vashishta_2007_SiC": ("vashishta", "vashishta/table")}


def sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================================ inputs
def write_lammps_data(atoms: Any, path: PathLike, *, type_map: Mapping[str, int] = TYPE_MAP,
                      masses: Mapping[str, float] = MASSES_AMU, title: str = "") -> Path:
    """Write an ASE ``Atoms`` with an orthorhombic cell as a LAMMPS ``atomic`` data file.

    Box [0, Lx) x [0, Ly) x [0, Lz); positions with repr precision (the float64 values round-trip exactly);
    atom ids 1..N in the ASE order.
    """
    cell = np.asarray(atoms.cell.array, dtype=float)
    off = np.abs(cell - np.diag(np.diag(cell)))
    if off.max() > 1e-9:
        raise ValueError("write_lammps_data needs an orthorhombic cell (use structures.indentation slabs)")
    sym = list(atoms.get_chemical_symbols())
    unknown = sorted(set(sym) - set(type_map))
    if unknown:
        raise ValueError(f"no LAMMPS type for element(s) {unknown}; type map {dict(type_map)}")
    X = np.asarray(atoms.get_positions(), dtype=float)
    ntypes = max(type_map.values())
    by_type = {t: e for e, t in type_map.items()}
    lines = [f"LAMMPS data file: {title or 'fracture-md indentation slab'} (written by {GENERATOR})", "",
             f"{len(X)} atoms", f"{ntypes} atom types", "",
             f"0.0 {float(cell[0, 0])!r} xlo xhi", f"0.0 {float(cell[1, 1])!r} ylo yhi",
             f"0.0 {float(cell[2, 2])!r} zlo zhi", "",
             "Masses", ""]
    lines += [f"{t} {float(masses[by_type[t]])!r}  # {by_type[t]}" for t in range(1, ntypes + 1)]
    lines += ["", "Atoms # atomic", ""]
    lines += [f"{i + 1} {type_map[s]} {x!r} {y!r} {z!r}" for i, (s, (x, y, z)) in enumerate(zip(sym, X.tolist()))]
    p = Path(path)
    p.write_text("\n".join(lines) + "\n")
    return p


def write_potential_file(potential: str, path: PathLike) -> Path:
    """Write the LAMMPS parameter file of a harness potential (only ``vashishta_2007_SiC`` so far)."""
    if potential != "vashishta_2007_SiC":
        raise NotImplementedError(f"no LAMMPS parameter file for {potential!r} (supported: vashishta_2007_SiC); "
                                  "pass potential_file= to use an existing file")
    from fracture_md.potentials.vashishta import SIC_VASHISHTA_2007_FILE_SHA256, SIC_VASHISHTA_2007_LAMMPS

    head = ("# Vashishta et al., J. Appl. Phys. 101, 103515 (2007): the parameters of the LAMMPS-distributed\n"
            f"# potentials/SiC.vashishta (sha256 {SIC_VASHISHTA_2007_FILE_SHA256}), written by {GENERATOR}\n"
            "# from fracture_md.potentials.vashishta.SIC_VASHISHTA_2007_LAMMPS.  Value of D here equals D/2 in paper.\n"
            "# e1 e2 e3 H eta Zi Zj lambda1 D lambda4 W rc B gamma r0 C costheta\n")
    p = Path(path)
    p.write_text(head + SIC_VASHISHTA_2007_LAMMPS.lstrip("\n"))
    return p


def slab_for_protocol(protocol: Any, conventional_atoms: Any, *, cutoff: float, vacuum_A: Optional[float] = None) -> Any:
    """The slab the JAX runner builds for ``protocol`` (``structures.indentation.slab_for_protocol``, the mapping
    ``physics.indentation.run_indentation`` uses too)."""
    from fracture_md.structures.indentation import slab_for_protocol as _slab_for_protocol

    return _slab_for_protocol(protocol, conventional_atoms, cutoff=cutoff, vacuum_A=vacuum_A)


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return str(v)


def deck_variables(protocol: Any, slab: Any, *, pair_style: str = "vashishta", table: tuple[int, float] = (100000, 0.2),
                   zref_A: Optional[float] = None, axis_xy: Optional[Sequence[float]] = None,
                   qs_fmax_eV_per_A: Optional[float] = None, qs_dump: bool = False, anr_A: Optional[float] = None,
                   restarts: bool = False, lattice_constant_A: Optional[float] = None,
                   unload_clearance_A: float = DEFAULT_UNLOAD_CLEARANCE_A, checkpoint_steps: int = 0,
                   zscale: Optional[float] = None, extra: Optional[Mapping[str, Any]] = None) -> dict[str, str]:
    """``-var`` values of in.berk_sdf.lmp for ``protocol`` on ``slab`` (strings, in deck order).

    ``pair_style``: "vashishta" or "vashishta/table" (with ``table`` = (N, inner cutoff Å)).  ``zref_A``:
    absolute depth zero (default: measured in LAMMPS as the mean z of the relaxed top layer, the harness
    rule).  ``axis_xy``: tip axis (default: the slab centre, as the harness runner).  ``qs_fmax_eV_per_A``:
    minimiser tolerance of a quasi-static protocol (max per-atom force; default the protocol's relax fmax).
    ``lattice_constant_A``: conventional a(T) of the sample, printed as ``a0`` in the SDF_META line (default
    ``slab.lattice_constant_A``; 0 = unknown, when the slab was built from a cell as is).  ``restarts``: restart
    files + tip-work state at loading start, every ``protocol.full_dump_every_A`` of depth and at the end of the
    hold.  ``unload_clearance_A``: retraction beyond the zero-force point of the unload (deck ``hunl``).
    ``checkpoint_steps`` > 0: periodic restarts (two alternating files) for crash recovery (deck ``ckpt``).
    ``zscale``: z pre-expansion of the mobile atoms after the CG (MD; default :data:`DEFAULT_ZSCALE` at T > 0, the
    recipe's protocol/preparation/zscale, and 1.0 at 0 K).
    """
    ind = protocol.indenter
    if ind.tip_model != "rigid_sdf":
        raise NotImplementedError("in.berk_sdf.lmp indents with the rigid SDF tip; atomistic tips use "
                                  "examples/indent/in.berk_atomistic.lmp (see its header for the -var list)")
    if pair_style == "vashishta":
        pstyle = "vashishta"
    elif pair_style == "vashishta/table":
        pstyle = f"vashishta/table {int(table[0])} {float(table[1])!r}"
    else:
        raise ValueError(f"unsupported pair_style {pair_style!r} (vashishta | vashishta/table)")
    dt_ps = protocol.timestep_fs * 1e-3
    ax = tuple(float(v) for v in (slab.center_xy if axis_xy is None else axis_xy))
    angle = {"berkovich": ind.face_angle_deg, "cone": ind.cone_half_angle_deg}.get(ind.shape, 0.0)
    md = protocol.mode == "md"
    v: dict[str, Any] = {
        "mode": "md" if md else "qs",
        "pstyle": pstyle,
        "shape": ind.shape, "angle": float(angle), "R": ind.tip_radius_A, "az": ind.azimuth_deg,
        "cap": ind.height_cap_A, "K": ind.stiffness_eV_per_A3,
        "xa": ax[0], "ya": ax[1], "gap": protocol.initial_gap_A, "zref": -1.0e30 if zref_A is None else float(zref_A),
        "fixt": protocol.fixed_thickness_A, "batht": protocol.thermostat_thickness_A,
        "ftolprep": protocol.relax.fmax_eV_per_A, "nmin": int(protocol.relax.max_steps),
        "a0": _sample_lattice_constant(slab, lattice_constant_A),
    }
    if md:
        eq_steps = int(round(protocol.equilibration_ps / dt_ps))
        nlang = int(round(0.6 * eq_steps)) if protocol.temperature_K > 0 else 0
        frame = 0 if protocol.frame_every_ps is None else max(1, int(round(protocol.frame_every_ps / dt_ps)))
        full = protocol.full_dump_every_A if protocol.full_dump_every_A is not None else protocol.max_depth_A + protocol.initial_gap_A
        v.update({
            "T": protocol.temperature_K, "seed": int(protocol.seed) + 1, "dtps": dt_ps, "bath": protocol.thermostat,
            "tau": protocol.thermostat_tau_ps, "nlang": nlang, "nbath": (eq_steps - nlang) if protocol.temperature_K > 0 else 0,
            "vel": protocol.speed_A_per_ps(), "hmax": protocol.max_depth_A, "hold": protocol.hold_ps,
            "unload": int(protocol.unload), "relax": protocol.relax_after_unload_ps, "logevery": protocol.log_every_steps,
            "framesteps": frame, "slicesteps": frame, "slicehw": protocol.slice_half_width_A, "fulldumpA": float(full),
            "anr": 0.0 if anr_A is None else float(anr_A), "restarts": int(restarts),
            "hunl": float(unload_clearance_A), "ckpt": int(checkpoint_steps),
            "zscale": _default_zscale(protocol) if zscale is None else float(zscale),
        })
    else:
        nqs = int(round(protocol.max_depth_A / protocol.depth_step_A))
        if abs(nqs * protocol.depth_step_A - protocol.max_depth_A) > 1e-9:
            raise ValueError("max_depth_A must be a whole number of depth_step_A for the quasi-static deck")
        v.update({"dh": protocol.depth_step_A, "nqs": nqs,
                  "ftolqs": protocol.relax.fmax_eV_per_A if qs_fmax_eV_per_A is None else float(qs_fmax_eV_per_A),
                  "qsdump": int(qs_dump)})
    if extra:
        v.update(extra)
    return {k: _fmt(x) for k, x in v.items()}


def _default_zscale(protocol: Any) -> float:
    """:data:`DEFAULT_ZSCALE` for an MD protocol at T > 0, else 1.0 (a 0 K run has no thermal expansion to anticipate)."""
    return DEFAULT_ZSCALE if protocol.temperature_K > 0 else 1.0


def _sample_lattice_constant(slab: Any, lattice_constant_A: Optional[float] = None) -> float:
    """a(T) of the sample: the argument, else the slab's (``structures.indentation`` scales the conventional cell
    to it), else 0.0 (unknown: the slab was built from a cell as is; the importer then needs it from the caller)."""
    a = lattice_constant_A if lattice_constant_A is not None else getattr(slab, "lattice_constant_A", None)
    return float(a) if a is not None else 0.0


def _check_slab(protocol: Any, slab: Any) -> None:
    """The slab must be at least the protocol's cell (the builder rounds up to whole cells)."""
    lx, ly = slab.lateral_size_A
    if lx < protocol.lateral_size_A[0] - 1e-6 or ly < protocol.lateral_size_A[1] - 1e-6:
        raise ValueError(f"slab lateral size {lx:.3f} x {ly:.3f} A is smaller than the protocol's "
                         f"{protocol.lateral_size_A}")
    if abs(slab.fixed_thickness_A - protocol.fixed_thickness_A) > 1e-9:
        raise ValueError("slab.fixed_thickness_A differs from protocol.fixed_thickness_A")
    if tuple(slab.orientation.plane_miller) != tuple(protocol.plane_miller):
        raise ValueError(f"slab plane {slab.orientation.plane_miller} != protocol plane {protocol.plane_miller}")
    geom_cap = protocol.indenter.height_cap_A
    if slab.vacuum_above_A < geom_cap:
        raise ValueError(f"vacuum above the slab {slab.vacuum_above_A:.2f} A < height cap {geom_cap:.2f} A")


# ============================================================================================ run directory
@dataclasses.dataclass
class LammpsIndentDeck:
    """A generated LAMMPS indentation run directory."""

    run_dir: Path
    tag: str
    files: dict[str, Path]  # role -> path (deck, sdf_module, data, potential, args, run_script, manifest)
    variables: dict[str, str]
    manifest: dict[str, Any]

    def lmp_args(self, *, log: bool = True) -> list[str]:
        """Arguments of ``lmp`` (relative to the run directory)."""
        args = ["-in", self.files["deck"].name]
        for k, val in self.variables.items():
            args += ["-var", k, val]
        if log:
            args += ["-log", f"log.{self.tag}"]
        return args + ["-nocite"]

    def command(self, *, cores: Sequence[int] = (), env: PathLike = DEFAULT_LAMMPS_ENV,
                allow_x925: bool = False, lmp_extra: Sequence[str] = ()) -> list[str]:
        """Full command line (run it in ``run_dir``); ``cores`` = explicit MPI ranks (pe-list binding)."""
        return mpirun_prefix(cores, env=env, allow_x925=allow_x925) + [str(Path(env) / "bin" / "lmp"), *self.lmp_args(),
                                                                        *lmp_extra]

    def table_path(self) -> Path:
        name = f"berk_log_{self.tag}.txt" if self.variables.get("mode") == "md" else f"qs_{self.tag}.txt"
        return self.run_dir / name


def check_cores(cores: Sequence[int], *, allow_x925: bool = False) -> list[int]:
    """Validate a core list (no duplicates; the X925 cores only with ``allow_x925``, i.e. a reservation)."""
    cs = [int(c) for c in cores]
    if len(set(cs)) != len(cs):
        raise ValueError(f"duplicate cores in {cs}")
    fast = sorted(set(cs) & set(X925_CORES))
    if fast and not allow_x925:
        raise ValueError(f"cores {fast} are X925 cores, reserved for production runs; pass allow_x925=True only "
                         "with a reservation (examples/indent/run_x925.sh)")
    if set(cs) & set(X925_CORES) and set(cs) & set(A725_CORES):
        warnings.warn("mixing X925 and A725 cores in one MPI run is never faster than X925 alone (G07)", stacklevel=2)
    return cs


def mpirun_prefix(cores: Sequence[int] = (), *, env: PathLike = DEFAULT_LAMMPS_ENV, allow_x925: bool = False,
                  np_unbound: int = 1) -> list[str]:
    """``mpirun`` prefix with explicit pe-list binding (never taskset + mpirun: Open MPI 5 ignores the mask).

    ``cores`` empty: ``np_unbound`` ranks with ``--bind-to none`` (they inherit the caller's CPU affinity,
    e.g. a ``taskset`` around pytest).
    """
    mpirun = str(Path(env) / "bin" / "mpirun")
    if cores:
        cs = check_cores(cores, allow_x925=allow_x925)
        return [mpirun, "-np", str(len(cs)), "--map-by", f"pe-list={','.join(map(str, cs))}:ordered",
                "--bind-to", "core"]
    return [mpirun, "-np", str(int(np_unbound)), "--bind-to", "none"]


def _run_script(deck: LammpsIndentDeck, default_cores: Sequence[int], env: PathLike) -> str:
    cores = ",".join(map(str, default_cores))
    cores_line = (f"CORES=${{1:-{cores}}}" if cores else
                  "CORES=${1:?usage: bash run.sh CORES (comma list of booked cores, e.g. 12,13,14)}")
    return f"""#!/bin/bash
# Generated by {GENERATOR} (v{GENERATOR_VERSION}): LAMMPS indentation run '{deck.tag}'.
# Usage: bash run.sh [CORES]   (comma list; ranks are bound with --map-by pe-list=CORES:ordered --bind-to core;
# never 'taskset ... mpirun').  The X925 cores (5-9, 15-19) are reserved: use examples/indent/run_x925.sh.
set -euo pipefail
cd "$(dirname "$0")"
LMP_ENV=${{LMP_ENV:-{env}}}
{cores_line}
NP=$(echo "$CORES" | tr ',' '\\n' | wc -l)
mapfile -t ARGS < lmp.args
start=$(date +%s)
echo "{deck.tag} START $(date -Iseconds) cores=$CORES np=$NP load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a walltime.txt
OMP_NUM_THREADS=1 "$LMP_ENV/bin/mpirun" -np "$NP" --map-by "pe-list=$CORES:ordered" --bind-to core \\
  "$LMP_ENV/bin/lmp" "${{ARGS[@]}}" -screen none > run_{deck.tag}.err 2>&1
rc=$?
echo "{deck.tag} END $(date -Iseconds) rc=$rc wall_s=$(( $(date +%s) - start ))" | tee -a walltime.txt
exit $rc
"""


def _copy_recipe(recipe: Any, protocol: Any, rd: Path, files: dict[str, Path], variables: Mapping[str, Any],
                 thickness_A: Optional[float] = None) -> Optional[dict[str, Any]]:
    """Copy the recipe into the run directory (file role "recipe") and return its manifest block
    (physics.indent_recipe.manifest_entry: id, sha256, protocol digest, the preparation the deck applies, read from
    the copied deck and its -var ``variables`` by physics.indent_recipe.lammps_deck_preparation, and the protocol
    deviations, preparation included); None for ``recipe=None``."""
    from fracture_md.physics import indent_recipe

    rec = indent_recipe.resolve_recipe(recipe)
    if rec is None:
        return None
    prep = indent_recipe.lammps_deck_preparation(files["deck"], variables, thickness_A=thickness_A)
    dst = rd / rec.path.name
    if dst.resolve() != rec.path.resolve():
        shutil.copyfile(rec.path, dst)
    if sha256_file(dst) != rec.sha256:
        raise RuntimeError(f"{dst}: the recipe copy does not hash to {rec.recipe_id} {rec.sha256} (changed while copying?)")
    files["recipe"] = dst
    return indent_recipe.manifest_entry(rec, protocol, preparation=prep)


def write_indentation_deck(
    protocol: Any,
    slab: Any,
    run_dir: PathLike,
    *,
    tag: str = "indent",
    potential: str = "vashishta_2007_SiC",
    pair_style: str = "vashishta",
    table: tuple[int, float] = (100000, 0.2),
    potential_file: Optional[PathLike] = None,
    zref_A: Optional[float] = None,
    axis_xy: Optional[Sequence[float]] = None,
    qs_fmax_eV_per_A: Optional[float] = None,
    qs_dump: bool = False,
    restarts: bool = False,
    anr_A: Optional[float] = None,
    lattice_constant_A: Optional[float] = None,
    unload_clearance_A: float = DEFAULT_UNLOAD_CLEARANCE_A,
    checkpoint_steps: int = 0,
    extra_vars: Optional[Mapping[str, Any]] = None,
    default_cores: Sequence[int] = (),
    env: PathLike = DEFAULT_LAMMPS_ENV,
    examples_dir: PathLike = EXAMPLES_DIR,
    recipe: Any = DEFAULT_RECIPE,
) -> LammpsIndentDeck:
    """Write a LAMMPS run directory for ``protocol`` on ``slab`` (see the module docstring).

    ``potential``: harness potential name (its LAMMPS file is written unless ``potential_file`` is given).
    ``pair_style``: "vashishta" (exact) or "vashishta/table" (1.43x faster; only after the table P-h check;
    never for tight 0 K minimisations: CG stalls at max |F| ~ 1e-4 eV/A with the table).  ``anr_A``: radius of
    the analysis dump around the tip axis (default 4.34 h_max + R + 20 A).  ``lattice_constant_A``: a(T) of the
    sample when the slab was built from a cell as is (default ``slab.lattice_constant_A``); it goes into the
    SDF_META line (``a0``) and manifest.json (``slab.lattice_constant_A``), so the run is self-describing for
    the importer.  The deck and the numpy SDF module are copied from ``examples_dir`` (examples/indent of the
    repository).  ``recipe``: the frozen recipe the run is made under (a name, a path or a loaded
    physics.indent_recipe.IndentationRecipe; default indent_recipe.DEFAULT_RECIPE), copied into the run directory
    and stated in ``manifest.json["recipe"]``; None writes no recipe.
    """
    if potential not in PAIR_STYLES:
        raise NotImplementedError(f"the SDF deck supports {sorted(PAIR_STYLES)}, not {potential!r}")
    if pair_style not in PAIR_STYLES[potential]:
        raise ValueError(f"pair_style {pair_style!r} does not implement {potential!r} ({PAIR_STYLES[potential]})")
    _check_slab(protocol, slab)
    ex = Path(examples_dir)
    src_deck, src_mod = ex / SDF_DECK, ex / SDF_MODULE
    for f in (src_deck, src_mod):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found: the deck generator copies the deck and the SDF module from the "
                                    "repository's examples/indent (pass examples_dir=)")
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    files["deck"] = Path(shutil.copyfile(src_deck, rd / SDF_DECK))
    files["sdf_module"] = Path(shutil.copyfile(src_mod, rd / SDF_MODULE))
    files["data"] = write_lammps_data(slab.atoms, rd / "slab.data", title=f"{tag}: {slab.orientation.description[:60]}")
    if potential_file is not None:
        files["potential"] = Path(shutil.copyfile(potential_file, rd / Path(potential_file).name))
    else:
        files["potential"] = write_potential_file(potential, rd / "SiC.vashishta")
    variables = {"tag": tag, "data": files["data"].name, "pfile": files["potential"].name, "sdfdir": ".",
                 **deck_variables(protocol, slab, pair_style=pair_style, table=table, zref_A=zref_A, axis_xy=axis_xy,
                                  qs_fmax_eV_per_A=qs_fmax_eV_per_A, qs_dump=qs_dump, restarts=restarts,
                                  anr_A=anr_A, lattice_constant_A=lattice_constant_A,
                                  unload_clearance_A=unload_clearance_A, checkpoint_steps=checkpoint_steps,
                                  extra=extra_vars)}
    from fracture_md.io.lammps_indent import card_for_lammps_potential

    card_id = card_for_lammps_potential(files["potential"], pair_style)
    masks = slab.masks()
    manifest: dict[str, Any] = {
        "generator": GENERATOR, "generator_version": GENERATOR_VERSION, "tag": tag,
        "engine": "lammps", "pair_style": variables["pstyle"], "potential": potential, "potential_card_id": card_id,
        "recipe": _copy_recipe(recipe, protocol, rd, files, variables, thickness_A=slab.thickness_A),
        "protocol": protocol.model_dump(mode="json"),
        "slab": {"n_atoms": int(slab.n_atoms), "lateral_size_A": list(slab.lateral_size_A),
                 "thickness_A": slab.thickness_A, "elastic_thickness_A": slab.elastic_thickness_A,
                 "fixed_thickness_A": slab.fixed_thickness_A, "bath_thickness_A": slab.bath_thickness_A,
                 "surface_z_unrelaxed_A": slab.surface_z, "top_z_A": slab.top_z, "bottom_z_A": slab.bottom_z,
                 "vacuum_above_A": slab.vacuum_above_A, "center_xy": [float(c) for c in slab.center_xy],
                 # a(T) of the sample; None if unknown (slab built from a cell as is, no lattice_constant_A given)
                 "lattice_constant_A": _sample_lattice_constant(slab, lattice_constant_A) or None,
                 "top_termination": slab.top_termination.model_dump(mode="json") if hasattr(slab.top_termination, "model_dump") else str(slab.top_termination),
                 "orientation": slab.orientation.model_dump(mode="json") if hasattr(slab.orientation, "model_dump") else str(slab.orientation),
                 "sizing": json.loads(json.dumps(slab.sizing, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))),
                 "mask_counts": {k: int(np.count_nonzero(m)) for k, m in masks.items()}},
        "variables": variables,
    }
    deck = LammpsIndentDeck(rd, tag, files, variables, manifest)
    files["args"] = rd / "lmp.args"
    files["args"].write_text("\n".join(deck.lmp_args()) + "\n")
    files["run_script"] = rd / "run.sh"
    files["run_script"].write_text(_run_script(deck, default_cores, env))
    os.chmod(files["run_script"], 0o755)
    manifest["sha256"] = {role: sha256_file(p) for role, p in files.items()}
    manifest["command_example"] = " ".join(shlex.quote(a) for a in deck.command(cores=default_cores, env=env))
    files["manifest"] = rd / "manifest.json"
    files["manifest"].write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return deck


# ============================================================================================ cell tiers (S1)
def tier_protocol(h_max_A: float, tier: Union[str, Sequence[float]] = "scoping", *, fixed_thickness_A: float = 10.0,
                  **fields: Any) -> Any:
    """An MD :class:`IndentationProtocol` whose cell is sized by a cell tier at ``h_max_A``.

    ``tier``: a name of ``schema.protocol.CELL_TIERS`` ("production" (8, 10), "pilot" (6, 8), "scoping" (4, 4)) or
    the ratios (L/a, t_el/a), a = the equivalent contact radius of the protocol's indenter at h_max (2.7923 h_max for
    the Berkovich): lateral size L x L and thickness t_el + ``fixed_thickness_A`` (the slab builders round up to whole
    unit cells).  ``fields``: any other protocol field (indenter, lattice_parameter_A, speed_m_per_s, hold_ps, ...).
    """
    from fracture_md.schema.protocol import CELL_TIERS, IndentationProtocol, IndenterSpec

    if isinstance(tier, str):
        ratios = {n: (lr, tr) for n, lr, tr in CELL_TIERS}
        if tier not in ratios:
            raise ValueError(f"unknown tier {tier!r}; known: {sorted(ratios)}")
        lr, tr = ratios[tier]
    else:
        lr, tr = (float(v) for v in tier)
    ind = fields.pop("indenter", None) or IndenterSpec()
    a = ind.equivalent_contact_radius_A(float(h_max_A))
    L = lr * a
    return IndentationProtocol(mode="md", lateral_size_A=(L, L), thickness_A=tr * a + float(fixed_thickness_A),
                               fixed_thickness_A=float(fixed_thickness_A), max_depth_A=float(h_max_A), indenter=ind,
                               **fields)


# ============================================================================================ atomistic-tip deck
def atomistic_cells(protocol: Any) -> tuple[int, int]:
    """(nx, nz): conventional cells of the LAMMPS-built (001) slab of in.berk_atomistic.lmp (square, rounded up)."""
    a0 = protocol.lattice_parameter_A
    if a0 is None:
        raise ValueError("the atomistic deck builds the slab at protocol.lattice_parameter_A: give a(T)")
    lx, ly = protocol.lateral_size_A
    if abs(lx - ly) > 1e-6:
        raise ValueError("in.berk_atomistic.lmp builds square cells (lateral_size_A must be (L, L))")
    return int(math.ceil(lx / a0 - 1e-9)), int(math.ceil(protocol.thickness_A / a0 - 1e-9))


def atomistic_deck_variables(protocol: Any, *, pair_style: str = "vashishta", table: tuple[int, float] = (100000, 0.2),
                             tip_height_A: Optional[float] = None, eps_eV: float = ATOMISTIC_WCA_EPS_EV,
                             sigma_A: float = ATOMISTIC_WCA_SIGMA_A, anr_A: Optional[float] = 0.0,
                             restarts: bool = False, unload_clearance_A: float = DEFAULT_UNLOAD_CLEARANCE_A,
                             checkpoint_steps: int = 0, nmin: int = 300, zscale: Optional[float] = None,
                             extra: Optional[Mapping[str, Any]] = None) -> dict[str, str]:
    """``-var`` values of in.berk_atomistic.lmp (rigid diamond Berkovich, WCA wall) for an MD ``protocol``.

    The deck builds a (001) C-terminated slab of ``atomistic_cells(protocol)`` conventional cells at
    ``protocol.lattice_parameter_A`` and carves the tip (faces 65.27 deg, outward normals at the protocol azimuth,
    height ``tip_height_A``, default max(h_max + 10, 20) A) from a diamond lattice.  Equilibration: Langevin on all
    mobile atoms for 2/3 of ``equilibration_ps`` (the deck's ``nequil``), then the boundary thermostat alone for 1/3.
    ``anr_A``: analysis-dump radius (0: 4.34 h_max + 20 A; < 0: the whole sample, the G06 dump).  The time step is
    fixed at 1 fs in the deck.  ``zscale``: z pre-expansion of the mobile atoms after the CG (default
    :data:`DEFAULT_ZSCALE` at T > 0, 1.0 at 0 K).
    """
    ind = protocol.indenter
    if ind.tip_model != "atomistic":
        raise ValueError("atomistic_deck_variables needs indenter.tip_model = 'atomistic'")
    if ind.shape != "berkovich" or abs(ind.face_angle_deg - 65.27) > 1e-9:
        raise NotImplementedError("in.berk_atomistic.lmp carves a 65.27 deg Berkovich only")
    if protocol.mode != "md" or not protocol.unload:
        raise NotImplementedError("in.berk_atomistic.lmp runs MD with an unload (mode='md', unload=True)")
    if abs(protocol.timestep_fs - 1.0) > 1e-12:
        raise NotImplementedError("in.berk_atomistic.lmp runs at dt = 1 fs")
    if protocol.thermostat not in ("csvr", "langevin"):
        raise NotImplementedError("in.berk_atomistic.lmp thermostats the boundary layer with csvr or langevin")
    if pair_style == "vashishta":
        pstyle, pname = "vashishta", "vashishta"
    elif pair_style == "vashishta/table":
        pstyle, pname = f"vashishta/table {int(table[0])} {float(table[1])!r}", "vashishta/table"
    else:
        raise ValueError(f"unsupported pair_style {pair_style!r} (vashishta | vashishta/table)")
    nx, nz = atomistic_cells(protocol)
    dt_ps = 1.0e-3
    eq = int(round(protocol.equilibration_ps / dt_ps))
    frame = 0 if protocol.frame_every_ps is None else max(1, int(round(protocol.frame_every_ps / dt_ps)))
    v: dict[str, Any] = {
        "mode": "md", "pstyle": pstyle, "pname": pname, "a0": float(protocol.lattice_parameter_A), "nx": nx, "nz": nz,
        "vel": protocol.speed_A_per_ps(), "hmax": protocol.max_depth_A, "gap": protocol.initial_gap_A,
        "hold": protocol.hold_ps, "hunl": float(unload_clearance_A), "relax": protocol.relax_after_unload_ps,
        "htip": float(tip_height_A if tip_height_A is not None else max(protocol.max_depth_A + 10.0, 20.0)),
        "eps": float(eps_eV), "sig": float(sigma_A), "seed": int(protocol.seed) + 1, "nequil": int(round(eq / 1.5)),
        "dumpevery": frame, "nmin": int(nmin), "az": float(ind.azimuth_deg), "fixt": protocol.fixed_thickness_A,
        "batht": protocol.thermostat_thickness_A, "bath": protocol.thermostat, "tau": protocol.thermostat_tau_ps,
        "T": protocol.temperature_K, "slicesteps": frame, "slicehw": protocol.slice_half_width_A,
        "fulldumpA": float(protocol.full_dump_every_A or 0.0), "restarts": int(restarts),
        "anr": -1.0 if anr_A is None else float(anr_A), "logevery": protocol.log_every_steps,
        "ckpt": int(checkpoint_steps), "zscale": _default_zscale(protocol) if zscale is None else float(zscale),
    }
    if extra:
        v.update(extra)
    return {k: _fmt(x) for k, x in v.items()}


def atomistic_atom_estimate(protocol: Any, tip_height_A: Optional[float] = None) -> dict[str, int]:
    """Sample atoms of the atomistic deck (8 nx^2 nz) and an estimate of the tip atoms (diamond, 8 / 3.567^3 per A^3,
    pyramid volume C0 H^3 / 3)."""
    nx, nz = atomistic_cells(protocol)
    H = tip_height_A if tip_height_A is not None else max(protocol.max_depth_A + 10.0, 20.0)
    c0 = protocol.indenter.projected_area_constant() or 24.494
    tip = int(round(c0 * H**3 / 3.0 * 8.0 / 3.567**3))
    return {"sample": 8 * nx * nx * nz, "tip_estimate": tip, "total_estimate": 8 * nx * nx * nz + tip}


def write_atomistic_deck(
    protocol: Any,
    run_dir: PathLike,
    *,
    tag: str = "atom",
    potential: str = "vashishta_2007_SiC",
    pair_style: str = "vashishta",
    table: tuple[int, float] = (100000, 0.2),
    potential_file: Optional[PathLike] = None,
    tip_height_A: Optional[float] = None,
    anr_A: Optional[float] = 0.0,
    restarts: bool = False,
    unload_clearance_A: float = DEFAULT_UNLOAD_CLEARANCE_A,
    checkpoint_steps: int = 0,
    extra_vars: Optional[Mapping[str, Any]] = None,
    default_cores: Sequence[int] = (),
    env: PathLike = DEFAULT_LAMMPS_ENV,
    examples_dir: PathLike = EXAMPLES_DIR,
    recipe: Any = DEFAULT_RECIPE,
) -> LammpsIndentDeck:
    """Write a run directory for the rigid ATOMISTIC tip deck (in.berk_atomistic.lmp; the slab is built in LAMMPS).

    Same layout as :func:`write_indentation_deck` (deck, ``berk_sdf.py`` for the tip-work tally, potential file,
    ``lmp.args``, ``run.sh``, ``manifest.json``, the recipe copy), without a data file.  Imported with
    ``io.lammps_indent`` (preset "g06-berkovich").
    """
    if potential not in PAIR_STYLES:
        raise NotImplementedError(f"the atomistic deck supports {sorted(PAIR_STYLES)}, not {potential!r}")
    if pair_style not in PAIR_STYLES[potential]:
        raise ValueError(f"pair_style {pair_style!r} does not implement {potential!r} ({PAIR_STYLES[potential]})")
    ex = Path(examples_dir)
    src_deck, src_mod = ex / ATOMISTIC_DECK, ex / SDF_MODULE
    for f in (src_deck, src_mod):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found (pass examples_dir=)")
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {"deck": Path(shutil.copyfile(src_deck, rd / ATOMISTIC_DECK)),
                              "sdf_module": Path(shutil.copyfile(src_mod, rd / SDF_MODULE))}
    if potential_file is not None:
        files["potential"] = Path(shutil.copyfile(potential_file, rd / Path(potential_file).name))
    else:
        files["potential"] = write_potential_file(potential, rd / "SiC.vashishta")
    variables = {"tag": tag, "pfile": files["potential"].name, "sdfdir": ".",
                 **atomistic_deck_variables(protocol, pair_style=pair_style, table=table, tip_height_A=tip_height_A,
                                            anr_A=anr_A, restarts=restarts, unload_clearance_A=unload_clearance_A,
                                            checkpoint_steps=checkpoint_steps, extra=extra_vars)}
    from fracture_md.io.lammps_indent import card_for_lammps_potential

    nx, nz = atomistic_cells(protocol)
    manifest: dict[str, Any] = {
        "generator": GENERATOR, "generator_version": GENERATOR_VERSION, "tag": tag, "engine": "lammps",
        "deck_kind": "atomistic", "pair_style": variables["pstyle"], "potential": potential,
        "potential_card_id": card_for_lammps_potential(files["potential"], pair_style),
        "recipe": _copy_recipe(recipe, protocol, rd, files, variables, thickness_A=nz * protocol.lattice_parameter_A),
        "protocol": protocol.model_dump(mode="json"),
        "slab": {"nx": nx, "nz": nz, "lateral_size_A": [nx * protocol.lattice_parameter_A] * 2,
                 "thickness_A": nz * protocol.lattice_parameter_A, "lattice_constant_A": protocol.lattice_parameter_A,
                 "fixed_thickness_A": protocol.fixed_thickness_A, "bath_thickness_A": protocol.thermostat_thickness_A,
                 "atoms": atomistic_atom_estimate(protocol, tip_height_A), "orientation": "(001) C-terminated",
                 "built_by": "LAMMPS lattice commands in in.berk_atomistic.lmp"},
        "variables": variables,
    }
    deck = LammpsIndentDeck(rd, tag, files, variables, manifest)
    files["args"] = rd / "lmp.args"
    files["args"].write_text("\n".join(deck.lmp_args()) + "\n")
    files["run_script"] = rd / "run.sh"
    files["run_script"].write_text(_run_script(deck, default_cores, env))
    os.chmod(files["run_script"], 0o755)
    manifest["sha256"] = {role: sha256_file(q) for role, q in files.items()}
    manifest["command_example"] = " ".join(shlex.quote(a) for a in deck.command(cores=default_cores, env=env))
    files["manifest"] = rd / "manifest.json"
    files["manifest"].write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return deck


# ============================================================================================ restarts and branches
def read_args_file(path: PathLike) -> list[str]:
    """The arguments of an ``lmp.args`` file (one per line)."""
    return [ln for ln in Path(path).read_text().split("\n") if ln != ""]


def args_variables(args: Sequence[str]) -> dict[str, str]:
    """The ``-var name value`` pairs of an argument list."""
    out: dict[str, str] = {}
    a = list(args)
    for i, tok in enumerate(a[:-2]):
        if tok == "-var":
            out[a[i + 1]] = a[i + 2]
    return out


def override_args(args: Sequence[str], overrides: Mapping[str, Any], *, log: Optional[str] = None) -> list[str]:
    """``args`` with the ``-var`` values in ``overrides`` replaced (new names appended before ``-nocite``) and the
    ``-log`` file set to ``log`` (default ``log.<tag>`` of the new tag)."""
    a = list(args)
    ov = {k: _fmt(v) for k, v in overrides.items()}
    out: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(a):
        if a[i] == "-var" and i + 2 < len(a):
            k = a[i + 1]
            out += ["-var", k, ov.get(k, a[i + 2])]
            seen.add(k)
            i += 3
        elif a[i] == "-log" and i + 1 < len(a):
            i += 2
        elif a[i] == "-nocite":
            i += 1
        else:
            out.append(a[i])
            i += 1
    for k, v in ov.items():
        if k not in seen:
            out += ["-var", k, v]
    tag = ov.get("tag", args_variables(a).get("tag", "indent"))
    return out + ["-log", log or f"log.{tag}", "-nocite"]


def restart_args(args: Sequence[str], restart_step: int, *, tag: str, branch: str = "continue",
                 parent_tag: Optional[str] = None, stop_after_segment: int = 0,
                 extra: Optional[Mapping[str, Any]] = None) -> list[str]:
    """Arguments of a restarted segment of a run (both decks, ``mode restart``).

    ``args``: the parent run's ``lmp.args``.  ``restart_step``: a step with ``restart_<parent>_<step>.bin`` and its
    ``.state.json`` (or ``"ckpt:<a|b>"``-style files passed through ``extra``).  ``branch``: "continue" (the same run
    from that step: crash recovery, continuity check) or "unload" (an unload branch from that depth).  ``tag``: the
    new run's tag (must differ from the parent's).  ``stop_after_segment`` > 0: stop after that loading segment
    (deck ``kstop``).
    """
    if branch not in ("continue", "unload"):
        raise ValueError(f"branch must be 'continue' or 'unload', not {branch!r}")
    ptag = parent_tag or args_variables(args).get("tag")
    if not ptag:
        raise ValueError("the parent tag is unknown (no -var tag in args)")
    if tag == ptag:
        raise ValueError("a restarted segment needs a new tag")
    ov: dict[str, Any] = {"tag": tag, "mode": "restart", "restart": f"restart_{ptag}_{int(restart_step)}.bin",
                          "rvars": f"run_{ptag}.vars", "rstate": f"restart_{ptag}_{int(restart_step)}.state.json",
                          "branch": branch, "kstop": int(stop_after_segment)}
    if extra:
        ov.update(extra)
    return override_args(args, ov)


_FULLDUMP_RE = re.compile(r"^(SDF|BERK)_FULLDUMP\s+(.*)$")


def restart_points(run_dir: PathLike, tag: str) -> list[dict[str, Any]]:
    """Full dumps and restarts of a finished (or running) deck run: one entry per ``*_FULLDUMP`` log line with step,
    depth h (A), role (reference / depth / end_hold / final) and the paths of the full dump, the restart and its
    tip-work state when they exist."""
    rd = Path(run_dir)
    log = rd / f"log.{tag}"
    out: list[dict[str, Any]] = []
    if not log.exists():
        return out
    for ln in log.read_text(errors="replace").splitlines():
        m = _FULLDUMP_RE.match(ln.strip())
        if not m:
            continue
        kv = dict(item.split("=", 1) for item in m.group(2).split() if "=" in item)
        if kv.get("tag") != tag:
            continue
        st = int(kv["step"])
        e: dict[str, Any] = {"step": st, "h_A": float(kv["h"]), "role": kv.get("role", ""),
                             "full_dump": rd / f"full_{tag}_{st}.lammpstrj.gz"}
        rs, ss = rd / f"restart_{tag}_{st}.bin", rd / f"restart_{tag}_{st}.state.json"
        e["restart"] = rs if rs.exists() else None
        e["state"] = ss if ss.exists() else None
        out.append(e)
    return out


# ============================================================================================ running (checks, tests)
def lammps_env(env: Optional[PathLike] = None) -> Optional[Path]:
    """The LAMMPS conda env (``FRACTURE_MD_LAMMPS_ENV`` or ~/miniconda3/envs/fmd-lammps) if it has ``bin/lmp``."""
    p = Path(env or os.environ.get("FRACTURE_MD_LAMMPS_ENV", DEFAULT_LAMMPS_ENV))
    return p if (p / "bin" / "lmp").exists() else None


def run_deck(deck: LammpsIndentDeck, *, cores: Sequence[int] = (), np_unbound: int = 1,
             env: Optional[PathLike] = None, timeout_s: Optional[float] = None,
             allow_x925: bool = False) -> subprocess.CompletedProcess:
    """Run a generated deck (blocking) with explicit core binding; raises on a non-zero exit code."""
    envp = lammps_env(env)
    if envp is None:
        raise FileNotFoundError("LAMMPS environment not found (set FRACTURE_MD_LAMMPS_ENV)")
    cmd = mpirun_prefix(cores, env=envp, allow_x925=allow_x925, np_unbound=np_unbound) + \
        [str(envp / "bin" / "lmp"), *deck.lmp_args(), "-screen", "none"]
    envv = {**os.environ, "OMP_NUM_THREADS": "1"}
    res = subprocess.run(cmd, cwd=deck.run_dir, capture_output=True, text=True, timeout=timeout_s, env=envv)
    if res.returncode != 0:
        log = deck.run_dir / f"log.{deck.tag}"
        tail = log.read_text()[-3000:] if log.exists() else ""
        raise RuntimeError(f"LAMMPS failed (rc {res.returncode}): {res.stderr[-2000:]}\n--- log tail ---\n{tail}")
    return res


def read_quasistatic_table(path: PathLike) -> dict[str, np.ndarray]:
    """The ``qs_<tag>.txt`` table of a quasi-static SDF run (columns named by its header)."""
    from fracture_md.io.indent_traj import read_lammps_table

    return read_lammps_table(path)


def quasistatic_rows(deck: LammpsIndentDeck) -> list[dict[str, float]]:
    """Per-depth rows of a finished quasi-static run: h, load (eV/Å and nN), energies, pen_fw, contact."""
    from fracture_md.io.indent_traj import EV_PER_A_TO_NN, read_lammps_meta_line

    tb = read_quasistatic_table(deck.table_path())
    meta = read_lammps_meta_line(deck.run_dir / f"log.{deck.tag}", "SDF_META")
    rows = []
    for i in range(len(tb["i"])):
        rows.append({"step": int(tb["i"][i]), "h_A": float(tb["h_A"][i]), "z_apex_A": float(tb["z_apex_A"][i]),
                     "load_eV_per_A": float(tb["Fz"][i]), "load_nN": float(tb["Fz"][i]) * EV_PER_A_TO_NN,
                     "lateral_eV_per_A": [float(tb["Fx"][i]), float(tb["Fy"][i])],
                     "energy_eV": float(tb["PE_eV"][i]), "E_ind_eV": float(tb["E_ind_eV"][i]),
                     "pen_fw_A": float(tb["pen_fw_A"][i]), "n_contact": int(tb["n_contact"][i]),
                     "max_pen_A": float(tb["max_pen_A"][i]), "fmax": float(tb["fmax_eV_per_A"][i]),
                     "reaction_z_eV_per_A": float(tb["Rz"][i])})
    for r in rows:
        r["zs0_A"] = float(meta["zs0"])
    return rows


# ============================================================================================ cross-engine check (G19)
# The G19 reference case (synthesis step 10): Vashishta 2007, (001) C-terminated slab of 2754 atoms (27.7 A
# lateral, 36 A thick, bottom 8 A fixed), rounded Berkovich R = 5 A (the protocol forbids R = 10 A in a cell this
# small: corners at 4.34 h + R must stay inside L/2), K = 100 eV/A^3, 0 K, 0.25 A steps to 2 A; both engines
# minimise to max |F_i| <= 1e-10 eV/A (LAMMPS CG, harness FIRE with newton_polish off).
G19_CASE: dict[str, Any] = dict(lateral_size_A=(27.6, 27.6), thickness_A=34.0, fixed_thickness_A=8.0,
                                thermostat_thickness_A=0.0, max_depth_A=2.0, depth_step_A=0.25,
                                lattice_parameter_A=4.358174, tip_radius_A=5.0, fmax_eV_per_A=1e-10)


def crossengine_protocol(**overrides: Any) -> Any:
    """The quasi-static :class:`IndentationProtocol` of the G19 cross-engine case (``overrides``: G19_CASE keys)."""
    from fracture_md.schema.protocol import IndentationProtocol, IndenterSpec, RelaxSettings

    c = {**G19_CASE, **overrides}
    return IndentationProtocol(
        mode="quasistatic", lateral_size_A=tuple(c["lateral_size_A"]), thickness_A=c["thickness_A"],
        fixed_thickness_A=c["fixed_thickness_A"], thermostat_thickness_A=c["thermostat_thickness_A"],
        max_depth_A=c["max_depth_A"], depth_step_A=c["depth_step_A"], lattice_parameter_A=c["lattice_parameter_A"],
        indenter=IndenterSpec(tip_radius_A=c["tip_radius_A"]),
        relax=RelaxSettings(fmax_eV_per_A=c["fmax_eV_per_A"], max_steps=2_000_000))


def harness_quasistatic(protocol: Any, slab: Any, model: Any, *, depths_A: Optional[Sequence[float]] = None,
                        zs0_A: Optional[float] = None, axis_xy: Optional[Sequence[float]] = None,
                        fmax_eV_per_A: Optional[float] = None, max_steps: int = 2_000_000) -> dict[str, Any]:
    """The JAX side of the cross-engine check: the runner's quasi-static path on the same slab.

    FIRE relaxation with the fixed bottom and ``newton_polish=False`` (as ``physics.indentation.run_indentation``),
    depth zero = mean z of the relaxed top layer unless ``zs0_A`` is given (pass the LAMMPS value to use the same
    absolute apex positions), then ``physics.indentation.quasistatic_indentation`` at ``depths_A``.
    """
    from fracture_md.physics.indentation import quasistatic_indentation
    from fracture_md.physics.indenter import IndenterGeometry, with_indenter
    from fracture_md.physics.relax import relax_positions
    from fracture_md.structures.system import AtomicSystem

    fmax = protocol.relax.fmax_eV_per_A if fmax_eV_per_A is None else float(fmax_eV_per_A)
    geom = IndenterGeometry(**protocol.indenter.geometry_kwargs())
    mc = with_indenter(model, geom, K_eV_per_A3=float(protocol.indenter.stiffness_eV_per_A3))
    masks = slab.masks()
    s, info = relax_positions(mc, AtomicSystem.from_ase(slab.atoms), fixed=masks["fixed"], method="fire", fmax=fmax,
                              newton_polish=False, skin=protocol.skin_A, max_steps=max_steps)
    X0 = np.asarray(s.frac, float) @ np.asarray(s.box, float).T
    zs_meas = float(np.mean(X0[masks["top_layer"], 2]))
    zs0 = zs_meas if zs0_A is None else float(zs0_A)
    xy = tuple(float(v) for v in (slab.center_xy if axis_xy is None else axis_xy))
    if depths_A is None:
        depths_A = list(np.arange(1, int(round(protocol.max_depth_A / protocol.depth_step_A)) + 1) * protocol.depth_step_A)
    qs = quasistatic_indentation(mc, s, fixed=masks["fixed"], apex_xy=xy, surface_z0=zs0, depths_A=list(depths_A),
                                 fmax=fmax, max_steps=max_steps, skin=protocol.skin_A)
    return {"rows": qs["rows"], "X_relaxed": X0, "X_at": [np.asarray(x) for x in qs["X_at"]], "zs0_A": zs0,
            "zs0_measured_A": zs_meas, "axis_xy": xy, "relax": dataclasses.asdict(info), "fmax_eV_per_A": fmax}


def lammps_quasistatic_result(deck: LammpsIndentDeck) -> dict[str, Any]:
    """Rows and positions (from the ``qsdump`` dumps, sorted by id = harness index + 1) of a finished run."""
    from fracture_md.io.lammps_indent import iter_dump_frames

    rows = quasistatic_rows(deck)
    X = {}
    for i in range(len(rows) + 1):
        f = deck.run_dir / f"qs_{deck.tag}_{i}.lammpstrj"
        if f.exists():
            fr = next(iter_dump_frames(f))
            X[i] = np.asarray(fr.positions, float)
    from fracture_md.io.indent_traj import read_lammps_meta_line

    meta = read_lammps_meta_line(deck.run_dir / f"log.{deck.tag}", "SDF_META")
    return {"rows": rows, "X_relaxed": X.get(0), "X_at": [X.get(i) for i in range(1, len(rows) + 1)],
            "zs0_A": float(meta["zs0"]), "relaxed_energy_eV": float(meta["relax_pe"]),
            "relaxed_fmax_eV_per_A": float(meta["relax_fmax"]), "meta": meta}


def _minimum_image_xy(d: np.ndarray, box_xy: Sequence[float]) -> np.ndarray:
    d = np.array(d, float)
    L = np.asarray(box_xy, float)
    d[..., :2] -= L * np.round(d[..., :2] / L)
    return d


def compare_quasistatic(harness: Mapping[str, Any], lammps: Mapping[str, Any], box_xy: Sequence[float]) -> dict[str, Any]:
    """Per-step relative differences of load, energy change, E_ind and displacements (G19 metrics).

    Displacements are u(h) = x(h) - x_relaxed of each engine; the position metric is max |u_J - u_L| / max |u_L|.
    """
    hr, lr = harness["rows"], lammps["rows"]
    if len(hr) != len(lr):
        raise ValueError("different numbers of depth steps")
    from fracture_md.io.indent_traj import EV_PER_A_TO_NN

    e0_j, e0_l = harness["relax"]["energy"], lammps.get("relaxed_energy_eV")
    steps = []
    for k, (a, b) in enumerate(zip(hr, lr)):
        Pj = a["load_nN"] / EV_PER_A_TO_NN
        Pl = b["load_eV_per_A"]
        row = {"h_A": b["h_A"], "load_eV_per_A_lammps": Pl, "load_eV_per_A_harness": Pj,
               "load_rel_diff": abs(Pj - Pl) / abs(Pl), "n_contact": [a["n_contact"], b["n_contact"]],
               "pen_fw_rel_diff": abs(a["pen_fw_A"] - b["pen_fw_A"]) / abs(b["pen_fw_A"]) if b["pen_fw_A"] else 0.0,
               "energy_abs_diff_eV": abs(a["energy_eV"] - b["energy_eV"])}
        if e0_l is not None:
            dj, dl = a["energy_eV"] - e0_j, b["energy_eV"] - e0_l
            row["energy_change_rel_diff"] = abs(dj - dl) / abs(dl)
        Xj, Xl = harness["X_at"][k], lammps["X_at"][k]
        if Xl is not None and harness.get("X_relaxed") is not None and lammps.get("X_relaxed") is not None:
            uj = _minimum_image_xy(Xj - harness["X_relaxed"], box_xy)
            ul = _minimum_image_xy(Xl - lammps["X_relaxed"], box_xy)
            row["displacement_rel_diff"] = float(np.max(np.linalg.norm(uj - ul, axis=1)) / np.max(np.linalg.norm(ul, axis=1)))
            row["position_max_abs_diff_A"] = float(np.max(np.linalg.norm(_minimum_image_xy(Xj - Xl, box_xy), axis=1)))
        steps.append(row)
    out: dict[str, Any] = {"steps": steps, "max_load_rel_diff": max(r["load_rel_diff"] for r in steps)}
    for key in ("energy_change_rel_diff", "displacement_rel_diff", "position_max_abs_diff_A"):
        vals = [r[key] for r in steps if key in r]
        if vals:
            out[f"max_{key}"] = max(vals)
    if harness.get("X_relaxed") is not None and lammps.get("X_relaxed") is not None:
        out["relaxed_position_max_abs_diff_A"] = float(np.max(np.linalg.norm(
            _minimum_image_xy(harness["X_relaxed"] - lammps["X_relaxed"], box_xy), axis=1)))
    return out


def work_identity(rows: Sequence[Mapping[str, Any]], *, load_key: str = "load_eV_per_A") -> dict[str, float]:
    """Quasi-static work identity on the loading steps: trapezoid of P dh against E(h_last) - E(h_first).

    At a relaxed state dE/dh = P exactly (the apex enters only through the tip energy), so the difference is
    the trapezoid quadrature error, O(dh^2 P'').
    """
    h = np.array([r["h_A"] for r in rows])
    P = np.array([r[load_key] for r in rows])
    E = np.array([r["energy_eV"] for r in rows])
    W = float(np.sum(0.5 * (P[1:] + P[:-1]) * np.diff(h)))
    dE = float(E[-1] - E[0])
    # second-order (Euler-Maclaurin) end correction from one-sided differences of P
    if len(h) >= 3:
        dP0 = (P[1] - P[0]) / (h[1] - h[0])
        dP1 = (P[-1] - P[-2]) / (h[-1] - h[-2])
        W_c = W - (h[1] - h[0]) ** 2 / 12.0 * (dP1 - dP0)
    else:
        W_c = W
    return {"W_trapezoid_eV": W, "W_corrected_eV": W_c, "dE_eV": dE, "rel_err_trapezoid": abs(W - dE) / abs(dE),
            "rel_err_corrected": abs(W_c - dE) / abs(dE)}


__all__ = [
    "A725_CORES", "ATOMISTIC_DECK", "DEFAULT_UNLOAD_CLEARANCE_A", "EXAMPLES_DIR", "G19_CASE", "LammpsIndentDeck",
    "MASSES_AMU", "SDF_DECK", "TYPE_MAP", "X925_CORES", "args_variables", "atomistic_atom_estimate", "atomistic_cells",
    "atomistic_deck_variables", "check_cores", "compare_quasistatic", "crossengine_protocol", "deck_variables",
    "harness_quasistatic", "lammps_env", "lammps_quasistatic_result", "mpirun_prefix", "override_args",
    "quasistatic_rows", "read_args_file", "read_quasistatic_table", "restart_args", "restart_points", "run_deck",
    "sha256_file", "slab_for_protocol", "tier_protocol", "work_identity", "write_atomistic_deck",
    "write_indentation_deck", "write_lammps_data", "write_potential_file",
]
