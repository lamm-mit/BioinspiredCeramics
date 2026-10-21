"""Import LAMMPS indentation runs: load-depth table, log metadata, streamed dumps and sha256 provenance.

LAMMPS is the production engine for large indentation cells (a deck generator writes the input; this
importer reads the output back); the JAX harness is the verified reference engine for small cells and
the record builder.  An imported run becomes a ``physics.indent_analysis.IndentationRunData`` (analysed by
exactly the same code as a harness run) and, through ``physics.indentation``, an ``IndentationCurve``
plus ``*_MD`` records.  Those records state ``engine_run.engine = "lammps"`` with the sha256 of the input
deck, of every potential file the deck reads (plus a hash of the inline ``pair_style``/``pair_coeff``
block, e.g. a WCA tip-sample wall) and of the log, table and dump that were imported; their validation
status is capped at ``numerically_verified`` (physics.indentation.ENGINE_STATUS_CAP): G19
(tests/test_indentation_crossengine.py) verifies the LAMMPS SDF deck against the harness only in the quasi-static
elastic regime, and LAMMPS MD indentation is checked only statistically.

Run layouts (``preset``)
------------------------
* ``g06-berkovich`` -- gapfill/G06/in.g06_berk.lmp and the rigid atomistic-tip deck
  examples/indent/in.berk_atomistic.lmp (engines.lammps_indent.write_atomistic_deck; its BERK_META states ``fixt``,
  G06's does not, which picks the deck a run is hashed with): rigid atomistic diamond Berkovich (faces 65.27 deg,
  outward face normals at 90/210/330 deg, WCA wall eps/sigma), files ``log.<tag>``,
  ``berk_log_<tag>.txt`` (fix print: step time_ps z_apex_A h_A Fx Fy Fz Rx Ry Rz T_mobile_K PE_eV T_core_K;
  F on the tip, R the pre-zero force sum of the fixed bottom layer, eV/A), ``dump_<tag>.lammpstrj.gz``
  (sample atoms: id type x y z c_cn[1] c_cn[2] c_ctip c_peat; c_ctip > 0 = in contact with a tip atom),
  metadata line ``BERK_META`` (zs0, xa0, ya0, nload, nhold, nunl, lx, ly, zmin0, ...).
* ``sphere`` -- lammps-reference/indent_sphere/in.indent_sphere.lmp: ``fix indent sphere`` (the SDF force
  law F = K (R - r)^2, i.e. a rigid_sdf sphere), files ``log.<tag>``, ``indent_log_<tag>.txt`` (step time_ps
  z_center_A h_A Fx Fy Fz Rx Ry Rz T_mobile_K PE_eV), ``dump_<tag>.lammpstrj`` (id type x y z c_cn),
  metadata line ``INDENT_META`` (R, K, vel, hmax, gap, zs0, xc, yc, nload, lx, ly, ...).
* ``sdf-berkovich`` -- examples/indent/in.berk_sdf.lmp (written by engines.lammps_indent.write_indentation_deck):
  the harness SDF tip through ``fix external pf/callback`` (examples/indent/berk_sdf.py), files ``log.<tag>``,
  ``berk_log_<tag>.txt`` (the G06 columns plus E_ind_eV pen_fw_A n_contact max_pen_A W_tip_eV Q_bath_eV KE_eV
  ledger_eV; Q_bath_eV is LAMMPS ``ecouple``, re-signed and re-zeroed on import, see
  :meth:`LammpsIndentRun.bath_heat_eV`), ``dump_<tag>.lammpstrj.gz`` (id type x y z d_pen c_cn[1] c_cn[2] c_peat;
  d_pen > 0 = in contact),
  metadata lines ``SDF_META`` (zs0, xa0, ya0, vel, schedule, shape, angle, R, az, cap, K, fixt, batht, ...) and
  ``SDF_SETUP``.  It shares the table name with ``g06-berkovich``; "auto" tells them apart by the SDF_META line.
Type map for all: 1 = C, 2 = Si (3 = tip atoms, never dumped).

Decks of generator version 2 (examples/indent/in.berk_sdf.lmp and in.berk_atomistic.lmp, critic item A4) add:
``*_UNLOAD`` lines (the unload ends at zero tip force, then a retraction and a relaxation at a fixed tip: segments
load / hold / unload / relax, :meth:`LammpsIndentRun.segments`), ``*_FULLDUMP`` lines (full dumps and restarts every
fulldumpA of depth, :meth:`LammpsIndentRun.full_dumps`), restarted segments and unload branches (``mode=restart``
in the META line, ``step_start`` and ``ec0`` for the heat zero), the atomistic tip's ledger columns (W_tip_eV,
Q_bath_eV, KE_eV of the sample, ledger_eV) and its azimuth (``az``; G06: 90).  For the atomistic tip the frames carry
the per-atom WCA tip force ``tip_force`` (from the tip snapshot ``tip_<tag>.lammpstrj`` and the logged apex), which
the estimated-contact rule of physics.indent_analysis needs.

Dumps are read frame by frame (:func:`iter_dump_frames`), never whole: a 3.7M-atom S1 dump does not fit in
memory as one array.  ``io.indent_traj`` keeps the in-memory reader used by the README renderer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

import numpy as np

from fracture_md.io.indent_traj import (
    EV_PER_A_TO_NN,
    file_sha256,
    open_text,
    read_lammps_meta_line,
    read_lammps_table,
)

PathLike = Union[str, Path]
DEFAULT_TYPE_MAP: dict[int, str] = {1: "C", 2: "Si"}
_POS_SETS = ((("x", "y", "z"), False), (("xu", "yu", "zu"), False), (("xs", "ys", "zs"), True),
             (("xsu", "ysu", "zsu"), True))


# ============================================================================================ dumps
@dataclasses.dataclass
class DumpFrame:
    """One frame of a LAMMPS text dump, atoms sorted by id."""

    step: int
    box_lo: np.ndarray  # (3,)
    box_hi: np.ndarray  # (3,)
    pbc: np.ndarray  # (3,) bool ('pp' bounds)
    ids: np.ndarray  # (N,)
    types: Optional[np.ndarray]  # (N,) int, or None (element column)
    elements: Optional[np.ndarray]  # (N,) str, or None
    positions: np.ndarray  # (N, 3) unscaled
    columns: dict[str, np.ndarray]  # every other numeric per-atom column (e.g. "c_ctip")


def iter_dump_frames(path: PathLike, *, every: int = 1, steps: Optional[Sequence[int]] = None,
                     columns: Optional[Sequence[str]] = None, dtype: Any = np.float64) -> Iterator[DumpFrame]:
    """Stream a text LAMMPS dump (``dump atom``/``custom``, optionally .gz) frame by frame.

    Only orthogonal boxes; atoms are sorted by id.  ``every`` keeps every k-th frame, ``steps`` only those
    timesteps (frames in between are skipped without parsing their atom block).  ``columns`` limits the
    extra per-atom columns that are parsed (default: all numeric ones).  A truncated last frame (a dump still
    being written) ends the iteration silently.
    """
    want = None if steps is None else set(int(s) for s in steps)
    k = 0
    with open_text(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            try:
                step = int(fh.readline().split()[0])
                if not fh.readline().startswith("ITEM: NUMBER OF ATOMS"):
                    raise ValueError("expected 'ITEM: NUMBER OF ATOMS'")
                n = int(fh.readline())
                hdr = fh.readline()
                if not hdr.startswith("ITEM: BOX BOUNDS"):
                    raise ValueError(f"unexpected dump line {hdr!r}")
                flags = hdr.split()[3:]
                if any(f in ("xy", "xz", "yz") for f in flags):
                    raise NotImplementedError("triclinic dumps are not supported (orthogonal boxes only)")
                bounds = np.array([[float(v) for v in fh.readline().split()[:2]] for _ in range(3)])
                cols = fh.readline().split()[2:]
                take = (k % every == 0) and (want is None or step in want)
                k += 1
                if not take:
                    for _ in range(n):
                        fh.readline()
                    continue
                if n == 0:  # e.g. a thresholded dump (dump_modify thresh) before contact: an empty frame
                    yield DumpFrame(step=step, box_lo=bounds[:, 0], box_hi=bounds[:, 1],
                                    pbc=np.array([f == "pp" for f in (flags[:3] if len(flags) >= 3 else ["pp"] * 3)]),
                                    ids=np.zeros(0, np.int64), types=np.zeros(0, int) if "type" in cols else None,
                                    elements=np.zeros(0, str) if "element" in cols else None,
                                    positions=np.zeros((0, 3), dtype), columns={})
                    continue
                txt = "".join(fh.readline() for _ in range(n))
                if not txt.endswith("\n"):
                    return
                arr = np.array(txt.split(), dtype=object if "element" in cols else float)
                if arr.size != n * len(cols):
                    return  # truncated frame
                arr = arr.reshape(n, len(cols))
            except (EOFError, IndexError, ValueError) as exc:
                if isinstance(exc, ValueError) and "expected" in str(exc):
                    raise
                return
            ci = {c: i for i, c in enumerate(cols)}
            if "id" not in ci:
                raise ValueError("dump needs an 'id' column")
            ids = arr[:, ci["id"]].astype(np.int64)
            order = np.argsort(ids, kind="stable")
            arr = arr[order]
            ids = ids[order]
            lo, hi = bounds[:, 0], bounds[:, 1]
            for trip, scaled in _POS_SETS:
                if all(c in ci for c in trip):
                    xyz = arr[:, [ci[c] for c in trip]].astype(float)
                    if scaled:
                        xyz = lo + xyz * (hi - lo)
                    break
            else:
                raise ValueError(f"no position columns in dump (columns: {cols})")
            skip = {"id", "type", "element", "mol", "ix", "iy", "iz"} | {c for tr, _ in _POS_SETS for c in tr}
            extra = {}
            for c in cols:
                if c in skip or (columns is not None and c not in columns):
                    continue
                try:
                    extra[c] = arr[:, ci[c]].astype(dtype)
                except (TypeError, ValueError):
                    pass
            bflags = flags[:3] if len(flags) >= 3 else ["pp", "pp", "pp"]
            yield DumpFrame(step=step, box_lo=lo, box_hi=hi, pbc=np.array([f == "pp" for f in bflags]), ids=ids,
                            types=arr[:, ci["type"]].astype(int) if "type" in ci else None,
                            elements=arr[:, ci["element"]].astype(str) if "element" in ci else None,
                            positions=xyz.astype(dtype), columns=extra)


def dump_steps(path: PathLike) -> list[int]:
    """Timesteps present in a text dump (header scan; the atom blocks are skipped)."""
    out = []
    with open_text(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return out
            if line.startswith("ITEM: TIMESTEP"):
                step = int(fh.readline().split()[0])
                fh.readline()
                n = int(fh.readline())
                for _ in range(5):
                    fh.readline()
                for _ in range(n - 1):
                    fh.readline()
                last = fh.readline()
                if last.endswith("\n"):
                    out.append(step)


# ============================================================================================ logs
@dataclasses.dataclass
class LammpsLogInfo:
    """What a log.lammps says about the run (the expanded command echo and the run summaries)."""

    version: str
    units: Optional[str]
    timestep_ps: Optional[float]
    mpi_tasks: Optional[int]
    omp_threads: Optional[int]
    cpu_use_percent: list[float]
    pair_style: list[str]  # expanded (no ${}) pair_style lines, in order
    pair_coeff: list[str]  # expanded pair_coeff lines
    lattice_constant_A: Optional[float]  # the first expanded 'lattice custom <a>' value (sample lattice)
    total_wall_s: Optional[float]
    dangerous_builds: list[int]
    # one entry per run/minimize: loop_s, procs, steps, atoms, cpu_percent (the "Loop time" + "% CPU use" pair)
    cpu_segments: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def cpu_use_check(self, min_percent: float = 97.0, *, min_steps: int = 100, min_loop_s: float = 1.0) -> dict[str, Any]:
        """Recipe item 2 / item 22: every run segment must report >= ``min_percent`` CPU use per MPI task (a shared
        core slows an MPI run 2-3x, G07).  Segments shorter than ``min_steps`` steps or ``min_loop_s`` seconds (e.g.
        ``run 0``, which reports a meaningless 400 % from a microsecond loop) are listed but not judged."""
        judged = [c for c in self.cpu_segments if c["steps"] >= min_steps and c["loop_s"] >= min_loop_s]
        bad = [c for c in judged if c["cpu_percent"] < min_percent]
        return {"passes": bool(judged) and not bad, "min_percent": float(min_percent), "n_segments": len(self.cpu_segments),
                "n_judged": len(judged), "min_judged_percent": min((c["cpu_percent"] for c in judged), default=None),
                "failed": bad, "mpi_tasks": self.mpi_tasks, "omp_threads": self.omp_threads,
                "reason": ("no run segment long enough to judge" if not judged else
                           f"{len(bad)} of {len(judged)} segments below {min_percent:g} % CPU" if bad else
                           f"all {len(judged)} segments >= {min_percent:g} % CPU")}

    def pair_block_sha256(self) -> str:
        """sha256 of the effective pair_style + pair_coeff block (inline coefficients, e.g. a WCA tip wall)."""
        txt = "\n".join(self.pair_style[-1:] + self.pair_coeff).encode()
        return hashlib.sha256(txt).hexdigest()

    def style_for_file(self, path: PathLike) -> str:
        """Pair style that reads ``path``: the sub-style named before it in a hybrid pair_coeff line, else the
        pair_style itself (e.g. "vashishta/table" in "pair_coeff * * vashishta/table SiC.vashishta C Si NULL")."""
        name = Path(path).name
        for ln in self.pair_coeff:
            tok = ln.split()
            for i, t in enumerate(tok):
                if Path(t).name == name and i >= 1:
                    prev = tok[i - 1]
                    if prev != "*" and not re.fullmatch(r"[0-9*]+", prev):
                        return prev
        return self.pair_style[-1].split()[1] if self.pair_style else ""

    def potential_files(self, base_dir: Optional[PathLike] = None) -> list[Path]:
        """Files named in the expanded pair_coeff lines that exist on disk.

        A relative name is resolved against ``base_dir`` first (the directory LAMMPS ran in, i.e. the log's
        directory, e.g. ``pair_coeff * * SiC.vashishta C Si`` in a generated run directory), then against the
        current directory."""
        out = []
        for ln in self.pair_coeff:
            for tok in ln.split()[1:]:
                if ("/" in tok or "." in tok) and not re.fullmatch(r"[-+0-9.eE]+", tok):
                    p = Path(tok)
                    if base_dir is not None and not p.is_absolute() and (Path(base_dir) / p).is_file():
                        p = Path(base_dir) / p
                    if p.is_file() and p not in out:
                        out.append(p)
        return out


def parse_lammps_log(path: PathLike) -> LammpsLogInfo:
    version, units, dt, tasks, threads, wall = "unknown", None, None, None, None, None
    cpu, ps, pc, lat, danger = [], [], [], None, []
    segs: list[dict[str, Any]] = []
    loop: Optional[dict[str, Any]] = None
    with open_text(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("LAMMPS (") and version == "unknown":
                version = s
            elif s.startswith("units "):
                units = s.split()[1]
            elif s.startswith("timestep ") and "$" not in s:
                dt = float(s.split()[1])
            elif s.startswith("Loop time of "):
                m = re.match(r"Loop time of ([-+0-9.eE]+) on (\d+) procs for (\d+) steps with (\d+) atoms", s)
                loop = None if m is None else {"loop_s": float(m.group(1)), "procs": int(m.group(2)),
                                               "steps": int(m.group(3)), "atoms": int(m.group(4))}
            elif "% CPU use with" in s:
                m = re.match(r"([0-9.]+)% CPU use with (\d+) MPI tasks x (\d+) OpenMP", s)
                if m:
                    cpu.append(float(m.group(1)))
                    tasks, threads = int(m.group(2)), int(m.group(3))
                    if loop is not None:
                        segs.append({**loop, "cpu_percent": float(m.group(1))})
                    loop = None
            elif s.startswith("pair_style ") and "$" not in s:
                ps.append(s)
            elif s.startswith("pair_coeff ") and "$" not in s:
                pc.append(s)
            elif s.startswith("lattice custom ") and "$" not in s and lat is None:
                lat = float(s.split()[2])
            elif s.startswith("Total wall time:"):
                h_, m_, s_ = (int(v) for v in s.split()[-1].split(":"))
                wall = 3600.0 * h_ + 60.0 * m_ + s_
            elif s.startswith("Dangerous builds"):
                try:
                    danger.append(int(s.split("=")[-1]))
                except ValueError:
                    pass
    if units == "metal" and dt is not None:
        dt_ps = dt  # metal: ps
    elif units == "real" and dt is not None:
        dt_ps = dt * 1e-3  # fs
    else:
        dt_ps = dt
    # later pair_coeff lines for the same type pair override earlier ones: keep the last occurrence
    last: dict[str, str] = {}
    for ln in pc:
        tok = ln.split()
        key = " ".join(tok[1:3]) + (" " + tok[3] if len(tok) > 3 and not re.fullmatch(r"[-+0-9.eE]+", tok[3]) else "")
        last[key] = ln
    return LammpsLogInfo(version, units, dt_ps, tasks, threads, cpu, ps, list(last.values()), lat, wall, danger, segs)


# ============================================================================================ runs
_PRESETS: dict[str, dict[str, Any]] = {
    "g06-berkovich": dict(table="berk_log_{tag}.txt", dump="dump_{tag}.lammpstrj.gz", log="log.{tag}",
                          meta_key="BERK_META", extra_meta=("TIP_META", "BERK_UNLOAD"), step="step", time="time_ps", depth="h_A",
                          tip_z="z_apex_A", fx="Fx", fy="Fy", fz="Fz", rz="Rz", contact_column="c_ctip",
                          axis=("xa0", "ya0"), decks=("in.berk_atomistic.lmp", "in.g06_berk.lmp"),
                          data="final_{tag}.data"),
    "sphere": dict(table="indent_log_{tag}.txt", dump="dump_{tag}.lammpstrj", log="log.{tag}",
                   meta_key="INDENT_META", extra_meta=(), step="step", time="time_ps", depth="h_A",
                   tip_z="z_center_A", fx="Fx_eV/A", fy="Fy_eV/A", fz="Fz_eV/A", rz="Rz", contact_column=None,
                   axis=("xc", "yc"), decks=("in.indent_sphere.lmp",), data="final_{tag}.data"),
    "sdf-berkovich": dict(table="berk_log_{tag}.txt", dump="dump_{tag}.lammpstrj.gz", log="log.{tag}",
                          meta_key="SDF_META", extra_meta=("SDF_SETUP", "SDF_UNLOAD"), step="step", time="time_ps", depth="h_A",
                          tip_z="z_apex_A", fx="Fx", fy="Fy", fz="Fz", rz="Rz", contact_column="d_pen",
                          axis=("xa0", "ya0"), decks=("in.berk_sdf.lmp",), data="final_{tag}.data"),
}


@dataclasses.dataclass
class LammpsIndentRun:
    """An imported LAMMPS indentation run: files and their sha256, metadata, the load table and the log."""

    run_dir: Path
    tag: str
    preset: str
    files: dict[str, Path]  # role -> path (log, table, dump, deck, data, potential:<name>)
    sha256: dict[str, str]  # role -> sha256 of the file bytes
    meta: dict[str, Any]  # META line(s) key=value
    table: dict[str, np.ndarray]  # load-depth table, duplicate boundary steps dropped
    log: LammpsLogInfo
    type_map: dict[int, str]
    n_duplicate_rows: int = 0

    # ------------------------------------------------------------------ derived quantities
    @property
    def timestep_ps(self) -> float:
        return float(self.log.timestep_ps or 0.001)

    @property
    def surface_z0_A(self) -> float:
        return float(self.meta["zs0"])

    @property
    def axis_xy(self) -> tuple[float, float]:
        ax, ay = _PRESETS[self.preset]["axis"]
        return float(self.meta[ax]), float(self.meta[ay])

    @property
    def lateral_L_A(self) -> tuple[float, float]:
        return float(self.meta["lx"]), float(self.meta["ly"])

    @property
    def speed_m_per_s(self) -> float:
        return float(self.meta["vel"]) * 100.0  # A/ps -> m/s

    @property
    def lattice_constant_A(self) -> Optional[float]:
        return self.log.lattice_constant_A

    def segments(self) -> np.ndarray:
        """Per-row segment labels from the metadata (load / hold / unload / relax).

        Generator-2 decks state the unload start and the relaxation start (``*_UNLOAD`` line: the unload ends at zero
        tip force plus a retraction, then the tip is held while the sample relaxes); an unload branch starts its
        unload at its first row.  Older decks: load nload steps, hold nhold, unload to the end (G06), or load and
        straight back up (sphere deck)."""
        st = np.asarray(self.table["step"], float)
        m = self.meta
        if "step_unload" in m:
            nl, nh = float(m["nload"]), float(m.get("nhold", 0.0))
            su = float(m["step_unload"])
            sr = float(m.get("step_relax", np.inf))
            base = np.where(st <= nl, "load", np.where(st <= nl + nh, "hold", "unload"))
            return np.where(st <= su, base, np.where(st <= sr, "unload", "relax"))
        if "nhold" in m:  # G06 deck: load nload steps, hold nhold, unload nunl
            nl, nh = float(m["nload"]), float(m["nhold"])
            return np.where(st <= nl, "load", np.where(st <= nl + nh, "hold", "unload"))
        nl = float(m["nload"])  # sphere deck: down for nload steps, then straight back up
        return np.where(st <= nl, "load", "unload")

    @property
    def is_restarted(self) -> bool:
        """A restarted segment or an unload branch (``mode=restart`` in the META line)."""
        return str(self.meta.get("mode", "")) == "restart"

    def full_dumps(self) -> list[dict[str, Any]]:
        """The ``*_FULLDUMP`` entries of the log: step, depth h (A), role (reference / depth / end_hold / final), the
        full dump path and, when present, the restart file and the tip-work state written with it."""
        out = []
        with open_text(self.files["log"]) as fh:
            for ln in fh:
                m = _FULLDUMP_RE.match(ln.strip())
                if not m:
                    continue
                kv = dict(item.split("=", 1) for item in m.group(2).split() if "=" in item)
                if kv.get("tag") != self.tag:
                    continue
                st = int(kv["step"])
                rs = self.run_dir / f"restart_{self.tag}_{st}.bin"
                ss = self.run_dir / f"restart_{self.tag}_{st}.state.json"
                out.append({"step": st, "h_A": float(kv["h"]), "role": kv.get("role", ""),
                            "path": self.run_dir / f"full_{self.tag}_{st}.lammpstrj.gz",
                            "restart": rs if rs.exists() else None, "state": ss if ss.exists() else None})
        return out

    def geometry(self) -> Any:
        """physics.indenter.IndenterGeometry of the tip (sharp atomistic Berkovich, or the fix-indent sphere)."""
        from fracture_md.physics.indenter import IndenterGeometry

        if self.preset == "sphere":
            return IndenterGeometry.sphere(float(self.meta["R"]))
        if self.preset == "sdf-berkovich":
            m = self.meta
            shape = str(m["shape"])
            ang = None if shape == "sphere" or float(m["angle"]) <= 0 else float(m["angle"])
            return IndenterGeometry(shape=shape, half_angle_deg=ang, tip_radius_A=float(m["R"]),
                                    azimuth_deg=float(m["az"]), height_cap_A=float(m["cap"]))
        # atomistic deck: faces at az, az + 120, az + 240 deg (the G06 tip: 90)
        return IndenterGeometry.berkovich(tip_radius_A=0.0, azimuth_deg=float(self.meta.get("az", 90.0)))

    def tip_model(self) -> str:
        return "atomistic" if self.preset == "g06-berkovich" else "rigid_sdf"

    def tip_description(self) -> str:
        if self.preset == "g06-berkovich":
            return (f"rigid diamond-lattice Berkovich ({self.meta.get('natoms_tip', self.meta.get('ntip', '?')):g} "
                    f"atoms, faces 65.27 deg, sharp), WCA tip-sample wall eps {self.meta.get('eps', '?')} eV, "
                    f"sigma {self.meta.get('sig', '?')} A (LAMMPS lj/cut at 2^(1/6) sigma)")
        if self.preset == "sdf-berkovich":
            return (f"harness SDF {self.geometry().label}, K = {self.meta['K']:g} eV/A^3 (U = K/3 sum delta^3; "
                    "LAMMPS fix external pf/callback + examples/indent/berk_sdf.py)")
        return f"LAMMPS fix indent sphere R = {self.meta['R']:g} A, K = {self.meta['K']:g} eV/A^3 (SDF force law)"

    # ------------------------------------------------------------------ provenance
    def potential_files(self) -> list[Path]:
        return [p for r, p in self.files.items() if r.startswith("potential:")]

    def engine_run(self, *, potential_card_id: Optional[str] = None, potential_convention: str = "",
                   deck_generator: str = "", recipe: Optional[PathLike] = None, accelerator: str = "cpu",
                   command: str = "") -> Any:
        """EngineRunProvenance (engine "lammps"): deck and potential-file sha256 (plus the inline pair block),
        the LAMMPS version and MPI ranks from the log."""
        from fracture_md.schema.common import EngineRunProvenance

        if "deck" not in self.files:
            raise ValueError("the LAMMPS input deck was not found; pass deck= to read_lammps_indent_run")
        pot = {p.name: self.sha256[f"potential:{p.name}"] for p in self.potential_files()}
        pot["pair_style+pair_coeff block (inline)"] = self.log.pair_block_sha256()
        from fracture_md.engines.base import sha256_file

        return EngineRunProvenance(
            engine="lammps", engine_version=self.log.version, potential_card_id=potential_card_id,
            potential_convention=potential_convention or "; ".join(self.log.pair_style[-1:]),
            potential_files_sha256=pot, deck_sha256=self.sha256["deck"], deck_generator=deck_generator,
            recipe_sha256=sha256_file(recipe) if recipe is not None else None, dtype="float64",
            accelerator=accelerator, mpi_ranks=self.log.mpi_tasks, command=command)

    def provenance_dict(self) -> dict[str, Any]:
        """Files, hashes and log facts, for Provenance.input_hashes / raw output."""
        return {"run_dir": str(self.run_dir), "tag": self.tag, "preset": self.preset,
                "files": {k: str(v) for k, v in self.files.items()}, "sha256": dict(self.sha256),
                "lammps_version": self.log.version, "mpi_tasks": self.log.mpi_tasks,
                "omp_threads": self.log.omp_threads, "cpu_use_percent": self.log.cpu_use_percent,
                "pair_style": self.log.pair_style, "pair_coeff": self.log.pair_coeff,
                "pair_block_sha256": self.log.pair_block_sha256(), "dangerous_builds": self.log.dangerous_builds,
                "cpu_use_check": self.log.cpu_use_check(),
                "total_wall_s": self.log.total_wall_s, "table_duplicate_rows_dropped": self.n_duplicate_rows}

    # ------------------------------------------------------------------ data
    def frames(self, *, every: int = 1, steps: Optional[Sequence[int]] = None,
               tip_force: bool = True) -> Iterator[Any]:
        """physics.indent_analysis.FrameData for each dump frame (streamed).

        Atomistic tip (preset g06-berkovich) with a tip snapshot (:meth:`tip_snapshot`) and ``tip_force``: each frame
        gets the per-atom field ``tip_force`` (eV/A), the magnitude of the WCA force of the rigid tip on each touching
        sample atom, with the tip translated to the logged apex height of that step."""
        from fracture_md.physics import indent_analysis as _ia
        from fracture_md.physics.indent_analysis import FrameData

        tip_field = getattr(_ia, "TIP_FORCE_FIELD", "tip_force")
        ccol = _PRESETS[self.preset]["contact_column"]
        tip = self.tip_snapshot() if (tip_force and self.preset == "g06-berkovich") else None
        apex = None
        if tip is not None:
            st_tab = np.asarray(self.table["step"], float)
            z_tab = np.asarray(self.table[_PRESETS[self.preset]["tip_z"]], float)
            apex = (st_tab, z_tab)
        for fr in iter_dump_frames(self.files["dump"], every=every, steps=steps):
            contact = None if ccol is None or ccol not in fr.columns else fr.columns[ccol] > 0
            fields = {k: v for k, v in fr.columns.items() if k != ccol}
            if tip is not None and contact is not None and apex is not None:
                k = np.flatnonzero(apex[0] == fr.step)
                if k.size:
                    dz = float(apex[1][k[0]]) - tip["apex_z"]
                    fields[tip_field] = wca_tip_force(fr.positions, contact, tip["positions"] + [0.0, 0.0, dz],
                                                      eps_eV=tip["eps"], sigma_A=tip["sig"], L_xy=self.lateral_L_A)
            yield FrameData(step=fr.step, time_ps=fr.step * self.timestep_ps, positions=fr.positions, contact=contact,
                            fields=fields)

    def tip_snapshot(self) -> Optional[dict[str, Any]]:
        """Positions of the rigid atomistic tip at loading start (``tip_<tag>.lammpstrj``, or the parent run's for a
        restarted segment), its apex height then (BERK_META ``apex_z``) and the WCA eps / sigma; None if absent."""
        if self.preset != "g06-berkovich":
            return None
        m = self.meta
        names = [f"tip_{self.tag}.lammpstrj"] + ([f"tip_{m['parent']}.lammpstrj"] if "parent" in m else [])
        path = next((self.run_dir / n for n in names if (self.run_dir / n).exists()), None)
        if path is None or "apex_z" not in m or "eps" not in m or "sig" not in m:
            return None
        fr = next(iter_dump_frames(path, columns=()))
        return {"positions": np.asarray(fr.positions, float), "apex_z": float(m["apex_z"]), "eps": float(m["eps"]),
                "sig": float(m["sig"]), "path": path}

    def species(self) -> np.ndarray:
        """Species of the dumped atoms (from the first frame)."""
        fr = next(iter_dump_frames(self.files["dump"], columns=()))
        if fr.elements is not None:
            return fr.elements
        return np.array([self.type_map[int(t)] for t in fr.types])

    def bath_heat_eV(self) -> Optional[np.ndarray]:
        """Cumulative heat ADDED to the sample by the thermostat since loading start (eV), per table row.

        This is the harness ``Q_eV`` convention (physics.indent_md: q = (alpha^2 - 1) KE_bath per step, ledger
        = E - E(0) - W - Q), the one the ``thermostat_heat`` channel of an IndentationCurve carries.  The SDF
        deck logs LAMMPS ``ecouple`` as ``Q_bath_eV`` instead: the cumulative energy REMOVED by the thermostat
        fixes (econserve = etotal + ecouple), counted from the start of the LAMMPS run, so it carries the
        equilibration total.  Hence Q = -(ecouple - ecouple at step 0), step 0 being the loading start (the deck's
        ``reset_timestep 0``; its ledger etotal + ecouple - W - E0 is then E - E(0) - W - Q).  None if the table
        has no ``Q_bath_eV`` column or no step-0 row (the zero point is then unknown).
        """
        tb = self.table
        if "Q_bath_eV" not in tb:
            return None
        ec = np.asarray(tb["Q_bath_eV"], dtype=np.float64)
        i0 = np.flatnonzero(np.asarray(tb["step"]) == 0)
        if i0.size:
            return -(ec - ec[i0[0]])
        if "ec0" in self.meta:  # a restarted segment / unload branch: ecouple at the parent's loading start
            return -(ec - float(self.meta["ec0"]))
        return None

    def run_data(self, *, every: int = 1, steps: Optional[Sequence[int]] = None, potential: str = "",
                 lattice_constant_A: Optional[float] = None) -> Any:
        """The engine-independent :class:`~fracture_md.physics.indent_analysis.IndentationRunData`."""
        from fracture_md.physics.indent_analysis import IndentationRunData

        p = _PRESETS[self.preset]
        tb = self.table
        geom = self.geometry()
        a = lattice_constant_A or self.lattice_constant_A
        if a is None:
            raise ValueError("lattice constant unknown (no 'lattice custom' line in the log): pass lattice_constant_A")
        ch: dict[str, np.ndarray] = {"step": tb["step"], "tip_z_A": tb[p["tip_z"]],
                                     "lateral_load_x_nN": tb[p["fx"]] * EV_PER_A_TO_NN,
                                     "lateral_load_y_nN": tb[p["fy"]] * EV_PER_A_TO_NN,
                                     "reaction_nN": tb[p["rz"]] * EV_PER_A_TO_NN}
        for col, key in (("T_mobile_K", "T_mobile_K"), ("T_core_K", "T_core_K"), ("PE_eV", "PE_eV"),
                         ("pen_fw_A", "pen_fw_A"), ("n_contact", "n_contact"), ("W_tip_eV", "W_eV"),
                         ("Q_bath_eV", "ecouple_eV"), ("ledger_eV", "ledger_eV"), ("E_ind_eV", "E_ind_eV")):
            if col in tb:
                ch[key] = tb[col]
        q_bath = self.bath_heat_eV()  # ecouple (removed, run total) -> harness Q_eV (added, from loading start)
        if q_bath is not None:
            ch["Q_eV"] = q_bath
        zmin0 = self.meta.get("zmin0")
        fixt = float(self.meta.get("fixt", 3.0))  # G06: 3 A fixed + 7 A bath; SDF / new atomistic deck: in the meta
        batht = float(self.meta.get("batht", 7.0))
        fixed_top = float(zmin0) + fixt if zmin0 is not None else None
        return IndentationRunData(
            t_ps=tb[p["time"]], h_A=tb[p["depth"]], load_nN=tb[p["fz"]] * EV_PER_A_TO_NN, segment=self.segments(),
            species=self.species(), surface_z0_A=self.surface_z0_A, axis_xy=self.axis_xy,
            lateral_L_A=self.lateral_L_A, lattice_constant_A=float(a), channels=ch,
            frames=lambda: self.frames(every=every, steps=steps), geometry=geom, tip_model=self.tip_model(),
            ideal_area_coefficient=geom.ideal_area_coefficient or 0.0,
            face_azimuths_deg=tuple(float(geom.azimuth_deg) + 120.0 * i for i in range(3)),
            fixed_top_z=fixed_top, bath_top_z=None if zmin0 is None else float(zmin0) + fixt + batht,
            potential=potential, meta={"engine": "lammps", **self.provenance_dict(), "lammps_meta": dict(self.meta)})


def _deck_names(preset: str, meta: dict[str, Any]) -> tuple[str, ...]:
    """Input-deck file names a run of ``preset`` may have been made with, looked up in the run directory, then its
    parent.  The g06-berkovich layout is shared by two decks: the rigid atomistic-tip deck (``in.berk_atomistic.lmp``,
    the name engines.lammps_indent.write_atomistic_deck copies it under) states its boundary layers (``fixt``) in
    BERK_META, the G06 deck (``in.g06_berk.lmp``) does not.  A run is matched only with the deck of its own kind, so
    that a neighbouring deck of the other kind is never hashed as the run's deck."""
    names = tuple(_PRESETS[preset]["decks"])
    if preset == "g06-berkovich":
        atomistic = "fixt" in meta
        names = tuple(n for n in names if (n == "in.berk_atomistic.lmp") == atomistic)
    return names


def read_lammps_indent_run(run_dir: PathLike, tag: str, *, preset: str = "auto", deck: Optional[PathLike] = None,
                           type_map: Optional[dict[int, str]] = None,
                           potential_files: Optional[Sequence[PathLike]] = None, hash_dump: bool = True) -> LammpsIndentRun:
    """Read the files of one LAMMPS indentation run and hash them.

    ``preset``: "g06-berkovich", "sphere", "sdf-berkovich" or "auto" (by the files present; the SDF and G06
    decks share ``berk_log_<tag>.txt`` and are told apart by the SDF_META line of ``log.<tag>``).  ``deck``: the input deck
    (default: the deck of the run's kind in ``run_dir``, else its parent: ``in.berk_sdf.lmp``, ``in.indent_sphere.lmp``;
    for g06-berkovich ``in.berk_atomistic.lmp`` when BERK_META states ``fixt`` (the atomistic-tip deck), else
    ``in.g06_berk.lmp``; when none is found, :meth:`LammpsIndentRun.engine_run` refuses).  ``potential_files``: default
    = the files named in the log's pair_coeff lines.  ``hash_dump``: also hash the dump (streamed; ~1 s per 100 MB).
    """
    rd = Path(run_dir)
    if preset == "auto":  # the SDF and G06 decks share berk_log_<tag>.txt: decide by the metadata line
        lg = rd / f"log.{tag}"
        if lg.exists():
            with open_text(lg) as fh:
                if any(ln.startswith("SDF_META ") for ln in fh):
                    preset = "sdf-berkovich"
    if preset == "auto":
        preset = next((k for k, v in _PRESETS.items() if (rd / v["table"].format(tag=tag)).exists()), None)
        if preset is None:
            raise FileNotFoundError(f"no known indentation table for tag {tag!r} in {rd}")
    if preset not in _PRESETS:
        raise ValueError(f"unknown preset {preset!r}; known: {sorted(_PRESETS)}")
    p = _PRESETS[preset]
    files: dict[str, Path] = {}
    for role in ("table", "dump", "log", "data"):
        f = rd / p[role].format(tag=tag)
        if f.exists():
            files[role] = f
        elif role != "data":
            raise FileNotFoundError(f"{role} file {f} not found")
    log = parse_lammps_log(files["log"])
    pots = ([Path(f) for f in potential_files] if potential_files is not None
            else log.potential_files(base_dir=files["log"].parent))
    for f in pots:
        files[f"potential:{f.name}"] = f
    meta = read_lammps_meta_line(files["log"], p["meta_key"])
    for key in p["extra_meta"]:
        try:
            meta.update({k: v for k, v in read_lammps_meta_line(files["log"], key).items() if k not in meta})
        except ValueError:
            pass
    dk = Path(deck) if deck is not None else next(
        (c / n for c in (rd, rd.parent) for n in _deck_names(preset, meta) if (c / n).is_file()), None)
    if dk is not None:
        files["deck"] = dk
    tb = read_lammps_table(files["table"])
    st = tb[p["step"]]
    _, first = np.unique(st, return_index=True)
    keep = np.sort(first)
    ndup = len(st) - len(keep)
    tb = {k: np.asarray(v)[keep] for k, v in tb.items()}
    tb["step"] = tb[p["step"]]
    sha = {role: file_sha256(f) for role, f in files.items() if role != "dump" or hash_dump}
    return LammpsIndentRun(rd, tag, preset, files, sha, meta, tb, log, dict(type_map or DEFAULT_TYPE_MAP), ndup)


_FULLDUMP_RE = re.compile(r"^(SDF|BERK)_FULLDUMP\s+(.*)$")


def wca_tip_force(positions: Any, touching: Any, tip_positions: Any, *, eps_eV: float, sigma_A: float,
                  L_xy: Sequence[float]) -> np.ndarray:
    """Magnitude (eV/A) of the WCA force (LJ eps, sigma cut and shifted at 2^(1/6) sigma) of the rigid tip atoms on
    each sample atom; zero for atoms not ``touching`` (no tip atom within the cutoff).  Lateral minimum image."""
    from scipy.spatial import cKDTree

    X = np.asarray(positions, float)
    out = np.zeros(len(X))
    idx = np.flatnonzero(np.asarray(touching, bool))
    if idx.size == 0:
        return out
    T = np.asarray(tip_positions, float)
    rc = 2.0 ** (1.0 / 6.0) * float(sigma_A)
    L = np.asarray(L_xy, float)
    zlo = min(X[idx, 2].min(), T[:, 2].min()) - 2.0 * rc
    zhi = max(X[idx, 2].max(), T[:, 2].max()) + 2.0 * rc
    box = np.array([L[0], L[1], zhi - zlo])

    def wrap(P):
        W = np.column_stack([np.mod(P[:, 0], L[0]), np.mod(P[:, 1], L[1]), P[:, 2] - zlo])
        return np.minimum(W, np.nextafter(box, 0))

    tree = cKDTree(wrap(T), boxsize=box)
    for k, nb in zip(idx, tree.query_ball_point(wrap(X[idx]), rc)):
        if not nb:
            continue
        d = X[k] - T[nb]
        d[:, :2] -= L * np.round(d[:, :2] / L)
        r2 = np.einsum("ij,ij->i", d, d)
        sr6 = (float(sigma_A) ** 2 / r2) ** 3
        f = (24.0 * float(eps_eV) * (2.0 * sr6 * sr6 - sr6) / r2)[:, None] * d
        out[k] = float(np.linalg.norm(f.sum(axis=0)))
    return out


def card_for_lammps_potential(path: PathLike, pair_style: str) -> Optional[str]:
    """Harness card id whose parameters equal a LAMMPS potential file (None if the style is not mapped).

    ``vashishta`` / ``vashishta/table``: the file is parsed by potentials.vashishta and the resulting card id
    is returned; it equals the registry ``vashishta_2007_SiC`` card id iff the parameters are identical.
    """
    style = pair_style.split("/")[0]
    if style == "vashishta":
        from fracture_md.potentials.vashishta import vashishta_2007_sic

        return vashishta_2007_sic(param_text=Path(path).read_text()).card.id
    return None


__all__ = [
    "DEFAULT_TYPE_MAP", "DumpFrame", "LammpsIndentRun", "LammpsLogInfo", "card_for_lammps_potential", "dump_steps",
    "iter_dump_frames", "parse_lammps_log", "read_lammps_indent_run", "wca_tip_force",
]
