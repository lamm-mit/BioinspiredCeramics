"""Indentation trajectories: the ``indentviz-traj/1`` file format, its reader and writer, and a LAMMPS adapter.

The format is the hand-off between the MD engines and the README animation (``fracture_md.viz.indentviz``). The
harness frame writer (physics/indent_md.py) writes it through :class:`IndentTrajWriter`; LAMMPS text dumps are
converted by :func:`trajectory_from_lammps`. The module is numpy-only (no JAX), so a trajectory can be read and
rendered on any machine.

FILE FORMAT ``indentviz-traj/1`` (a numpy ``.npz``; every length in Angstrom, time in ps, force in nN)
-------------------------------------------------------------------------------------------------------
required
  format          str               "indentviz-traj/1"
  positions       float (T, N, 3)   Cartesian atom positions per frame (A), stored as float32. May be wrapped into
                                    the periodic box; displacements are taken with the minimum image.
  species         str   (N,)        element symbols ("Si", "C", ...), constant over frames
  box_lo, box_hi  float (3,) or (T, 3)  orthogonal simulation-box bounds (A); per frame if the box changes
  pbc             bool  (3,)        periodic directions (the indentation axis z must be non-periodic)
  time_ps         float (T,)        simulation time of each frame (ps), strictly increasing
  tip_xyz         float (T, 3)      LOWEST point of the rigid indenter ("apex"; for a sphere the centre minus R
                                    z-hat) at each frame (A). The tip moves toward -z.
  depth_A         float (T,)        surface_z0_A - tip_xyz[:, 2]  (> 0 = tip below the initial surface)
  load_nN         float (T,)        force of the sample on the tip along +z (nN); > 0 = pushing back
  surface_z0_A    float ()          height of the initial free surface (A): mean z of the top atomic layer
  indenter_json   str               shape, see :class:`Indenter` (sphere | cone | berkovich)
  meta_json       str               free-form provenance (see "meta keys" below)
optional
  atom_ids        int   (N,)        engine atom ids (sorted ascending)
  curve_time_ps, curve_depth_A, curve_load_nN   (K,)  densely sampled load-depth record (e.g. every 50 MD
                                    steps); if absent the per-frame values are used
  atomfield__<name> float (T, N)    per-atom quantities carried over from the engine (e.g. the LAMMPS dump
                                    columns c_cn[1], c_ctip); render one with field "col:<name>"
  contact_xy      float (K, 2)      projected (x, y) positions (A) of EVERY atom touching the indenter, all frames
  contact_count   int   (T,)        concatenated: frame t owns rows sum(count[:t]) .. sum(count[:t+1]). Written
                                    when ``positions`` is a subset that cannot hold every contact atom (a LAMMPS
                                    slice dump plus a ``dump_modify thresh`` contact dump; :func:`attach_contact_dump`).
                                    The estimated-contact outline (E3) then uses these instead of the subset.
  contact_force   float (K,)        optional, with contact_xy: the tip force magnitude on each of those atoms (any
                                    common scale, e.g. eV/A); lets the outline apply the analysis's force rule.

meta keys read by the renderer (all optional)
  run_id, potential (plain name, e.g. "Vashishta 2007"), potential_id (harness registry name), temperature_K,
  tip_speed_m_s, n_atoms_total (the whole simulated system; ``positions`` may hold a subset), material_plain,
  engine, tip_model, lattice_constant_A (sets the estimated-contact (E3) dilation a/(2 sqrt 2)), contact_field (an atom
  field that is > 0 for atoms touching the indenter, e.g. LAMMPS "c_ctip"; without it contact is sdf < 0),
  subset (what ``positions`` holds when it is not every atom), status_note (one plain-language sentence printed
  on the frame), synthetic (True for made-up test data: stamped on every rendered frame).

A subset (for example a slice through the tip axis plus the top surface layer; :func:`animation_subset_mask`)
must be the SAME atoms in every frame, and a slice must carry a halo at least as wide as the strain cutoff
(3.35 A) on each side, because the local strain of an atom is fitted to its neighbours.

Units: LAMMPS "metal" forces are eV/A; 1 eV/A = 1.602176634 nN exactly (e and the A are SI-exact).
Files are written byte-deterministically (fixed zip timestamps, sorted members), so identical content gives an
identical sha256.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT = "indentviz-traj/1"
EV_PER_A_TO_NN = 1.602176634  # exact
A_PER_PS_TO_M_PER_S = 100.0  # 1 A/ps = 1e-10 m / 1e-12 s = 100 m/s
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)  # fixed member timestamp: identical content -> identical bytes
_BIG = 1e30


def file_sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    """sha256 of a file, read in chunks (dumps are 100 MB+)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ====================================================================================== indenter geometry
@dataclass(frozen=True)
class Indenter:
    """Rigid indenter shape as a numpy signed-distance function (SDF), used for drawing and contact detection.

    Convention (the same as physics/indenter.py and the risk-register JAX SDF): lab frame, the indenter comes
    down from +z; every shape is described relative to its LOWEST point (the "apex"); ``sdf(d)`` takes offsets
    ``d = p - apex`` (Angstrom, shape (..., 3)) and returns the signed distance (negative inside the solid).

    Shapes (``Indenter.from_dict`` keys):
      {"kind": "sphere", "radius_A": R}
      {"kind": "cone", "half_angle_deg": a, "tip_radius_A": R=0}
      {"kind": "berkovich", "face_angle_deg": 65.27, "tip_radius_A": R=0, "azimuth_deg": 0}
         azimuth = in-plane direction of the first face's outward horizontal normal, degrees from +x.
         The rounded pyramid is the Minkowski sum (sharp pyramid eroded by R) + ball(R): exact faces,
         cylindrical edges, spherical apex.
    Extra keys (e.g. a note on an atomistic tip) are kept in ``extra`` and written back unchanged.
    The dict of the harness indenter (``physics.indenter.IndenterGeometry.to_dict()``: shape, half_angle_deg,
    tip_radius_A, azimuth_deg, ...) is accepted as well; the conventions (apex, azimuth of face 0, outward
    normals) are the same.
    """

    kind: str
    radius_A: float = 0.0  # sphere radius, or tip radius of a rounded cone / pyramid
    angle_deg: float = 65.27  # cone half angle, or Berkovich face-to-axis angle
    azimuth_deg: float = 0.0
    extra: dict = field(default_factory=dict, compare=False)

    # ------------------------------------------------------------------ construction / serialisation
    @classmethod
    def from_dict(cls, d: dict) -> "Indenter":
        d = dict(d)
        kind = str(d.pop("kind") if "kind" in d else d.pop("shape")).lower()
        if kind == "sphere":
            r = d.pop("radius_A", None)
            r = float(d.pop("tip_radius_A") if r is None else r)  # IndenterGeometry keeps R in tip_radius_A
            d.pop("half_angle_deg", None)
            if r <= 0:
                raise ValueError("sphere radius_A must be > 0")
            return cls("sphere", radius_A=r, angle_deg=90.0, extra=d)
        if kind == "cone":
            a = float(d.pop("half_angle_deg"))
            r = float(d.pop("tip_radius_A", 0.0))
            if not 0 < a < 90 or r < 0:
                raise ValueError("cone needs 0 < half_angle_deg < 90 and tip_radius_A >= 0")
            return cls("cone", radius_A=r, angle_deg=a, extra=d)
        if kind == "berkovich":
            a = d.pop("face_angle_deg", None)
            h = d.pop("half_angle_deg", None)  # the harness name of the same angle
            a = float(a if a is not None else (h if h is not None else 65.27))
            r = float(d.pop("tip_radius_A", 0.0))
            az = float(d.pop("azimuth_deg", 0.0))
            if not 0 < a < 90 or r < 0:
                raise ValueError("berkovich needs 0 < face_angle_deg < 90 and tip_radius_A >= 0")
            return cls("berkovich", radius_A=r, angle_deg=a, azimuth_deg=az, extra=d)
        raise ValueError(f"unknown indenter kind {kind!r} (sphere | cone | berkovich)")

    @classmethod
    def from_json(cls, s: str) -> "Indenter":
        return cls.from_dict(json.loads(s))

    def to_dict(self) -> dict:
        if self.kind == "sphere":
            d = {"kind": "sphere", "radius_A": self.radius_A}
        elif self.kind == "cone":
            d = {"kind": "cone", "half_angle_deg": self.angle_deg, "tip_radius_A": self.radius_A}
        else:
            d = {"kind": "berkovich", "face_angle_deg": self.angle_deg, "tip_radius_A": self.radius_A,
                 "azimuth_deg": self.azimuth_deg}
        d.update(self.extra)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def describe(self) -> str:
        """Plain-language name for captions."""
        if self.kind == "sphere":
            return f"rigid spherical tip, radius {self.radius_A / 10:.1f} nm"
        if self.kind == "cone":
            return f"rigid conical tip ({self.angle_deg:.1f}° half angle)"
        r = f", tip radius {self.radius_A / 10:.1f} nm" if self.radius_A > 0 else ""
        return f"rigid three-sided (Berkovich) pyramid{r}"

    # ------------------------------------------------------------------ geometry
    def _face_dirs(self) -> np.ndarray:
        az = math.radians(self.azimuth_deg)
        return np.array([[math.cos(az + 2 * math.pi * i / 3), math.sin(az + 2 * math.pi * i / 3)] for i in range(3)])

    def sdf(self, d: np.ndarray) -> np.ndarray:
        """Signed distance (A) of offsets d = p - apex (..., 3); negative inside the indenter."""
        d = np.asarray(d, float)
        R = self.radius_A
        if self.kind == "sphere":
            return np.linalg.norm(d - np.array([0.0, 0.0, R]), axis=-1) - R
        w = d - np.array([0.0, 0.0, R])  # offsets from the apex of the eroded (sharp) solid
        th = math.radians(self.angle_deg)
        if self.kind == "cone":
            rho = np.hypot(w[..., 0], w[..., 1])
            z = w[..., 2]
            s = rho * math.cos(th) - z * math.sin(th)  # signed distance to the generator line (outward +)
            t = rho * math.sin(th) + z * math.cos(th)  # coordinate along the generator
            dout = np.where(t >= 0, s, np.hypot(rho, z))
            return np.where(s <= 0, s, dout) - R
        # Berkovich: exact distance to a convex polyhedral cone = min over the feasible candidates (apex, the
        # three faces, the three edges); inside = max of the face distances
        u = self._face_dirs()
        n = np.column_stack([math.cos(th) * u, -math.sin(th) * np.ones(3)])  # (3, 3) outward unit normals
        e = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            v = np.cross(n[i], n[j])
            v = v / np.linalg.norm(v)
            e.append(v if v[2] > 0 else -v)
        e = np.asarray(e)
        s = w @ n.T  # (..., 3)
        inside = np.all(s <= 0.0, axis=-1)
        cands = [np.linalg.norm(w, axis=-1)]
        for i in range(3):
            q = w - s[..., i:i + 1] * n[i]
            sq = q @ n.T
            others = [k for k in range(3) if k != i]
            feas = np.all(sq[..., others] <= 1e-12, axis=-1) & (s[..., i] > 0)
            cands.append(np.where(feas, np.abs(s[..., i]), _BIG))
        for k in range(3):
            t = w @ e[k]
            dist = np.linalg.norm(w - t[..., None] * e[k], axis=-1)
            cands.append(np.where(t > 0, dist, _BIG))
        dout = np.min(np.stack(cands, axis=0), axis=0)
        din = np.max(s, axis=-1)
        return np.where(inside, din, dout) - R

    def lower_surface_height(self, dxy: np.ndarray, zmax: float, tol: float = 1e-3) -> np.ndarray:
        """Height above the apex of the indenter's lower surface at lateral offsets dxy (..., 2).

        Found by bisection on the SDF along vertical lines (exact for the convex shapes here). Where the
        vertical line never enters the solid below ``zmax`` the result is NaN (e.g. outside a sphere).
        """
        dxy = np.asarray(dxy, float)
        if self.kind == "sphere":
            zmax = min(float(zmax), self.radius_A)  # the equator: lower hemisphere only
        shp = dxy.shape[:-1]
        flat = dxy.reshape(-1, 2)
        lo = np.zeros(len(flat))
        hi = np.full(len(flat), float(zmax))
        ok = self.sdf(np.column_stack([flat, hi])) <= 0
        # along a vertical line through a convex body the SDF is positive below the lower surface and <= 0
        # inside, so bisection between z = 0 and z = zmax brackets the crossing wherever ``ok``
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            inside = self.sdf(np.column_stack([flat, mid])) <= 0
            hi = np.where(inside, mid, hi)
            lo = np.where(inside, lo, mid)
            if np.max(hi - lo) < tol:
                break
        out = np.where(ok, 0.5 * (lo + hi), np.nan)
        return out.reshape(shp)

    def section_outline(self, direction_xy=(1.0, 0.0), half_width_A: float = 60.0, height_A: float = 80.0,
                        n: int = 401) -> np.ndarray:
        """Closed polygon (M, 2) of the indenter's section in the vertical plane through its axis.

        Coordinates are (s, z) relative to the apex, s along ``direction_xy``. For a sphere this is the full
        circle; for cones and pyramids the region between the lower surface and ``height_A``.
        """
        if self.kind == "sphere":
            t = np.linspace(0, 2 * np.pi, n)
            return np.column_stack([self.radius_A * np.sin(t), self.radius_A * (1 - np.cos(t))])
        u = np.asarray(direction_xy, float)
        u = u / np.linalg.norm(u)
        s = np.linspace(-half_width_A, half_width_A, n)
        h = self.lower_surface_height(s[:, None] * u[None, :], zmax=height_A)
        keep = np.isfinite(h)
        s, h = s[keep], h[keep]
        return np.vstack([np.column_stack([s, h]), [[s[-1], height_A], [s[0], height_A]], [[s[0], h[0]]]])

    def footprint_area_sharp(self, depth_A: float) -> float:
        """Analytic projected area (A^2) of a SHARP shape at depth h (tests only)."""
        th = math.radians(self.angle_deg)
        if self.kind == "berkovich":
            return 3 * math.sqrt(3) * (depth_A * math.tan(th)) ** 2
        if self.kind == "cone":
            return math.pi * (depth_A * math.tan(th)) ** 2
        return math.pi * (2 * self.radius_A * depth_A - depth_A ** 2)


# ====================================================================================== trajectory
@dataclass
class Trajectory:
    """One indentation run: positions of a fixed atom set per frame plus the tip path and the load (see module doc)."""

    positions: np.ndarray
    species: np.ndarray
    box_lo: np.ndarray
    box_hi: np.ndarray
    pbc: np.ndarray
    time_ps: np.ndarray
    tip_xyz: np.ndarray
    depth_A: np.ndarray
    load_nN: np.ndarray
    surface_z0_A: float
    indenter: Indenter
    meta: dict = field(default_factory=dict)
    atom_ids: np.ndarray | None = None
    curve_time_ps: np.ndarray | None = None
    curve_depth_A: np.ndarray | None = None
    curve_load_nN: np.ndarray | None = None
    atom_fields: dict = field(default_factory=dict)  # name -> (T, N) float, e.g. LAMMPS c_ctip
    contact_xy: np.ndarray | None = None  # (K, 2) projected positions of every contact atom, frames concatenated
    contact_count: np.ndarray | None = None  # (T,) contact atoms per frame (rows of contact_xy)
    contact_force: np.ndarray | None = None  # (K,) tip force on each contact_xy atom (optional)

    # ------------------------------------------------------------------ basic properties
    @property
    def n_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[1])

    def box(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        lo = self.box_lo if self.box_lo.ndim == 1 else self.box_lo[t]
        hi = self.box_hi if self.box_hi.ndim == 1 else self.box_hi[t]
        return np.asarray(lo, float), np.asarray(hi, float)

    def curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(time_ps, depth_A, load_nN) of the densest available load-depth record."""
        if self.curve_depth_A is not None and self.curve_load_nN is not None and self.curve_time_ps is not None:
            return self.curve_time_ps, self.curve_depth_A, self.curve_load_nN
        return self.time_ps, self.depth_A, self.load_nN

    def contact_xy_at(self, t: int) -> np.ndarray | None:
        """Projected (x, y) of every atom touching the tip at frame t ((n_t, 2)), or None without contact_xy."""
        if self.contact_xy is None or self.contact_count is None:
            return None
        off = np.r_[0, np.cumsum(np.asarray(self.contact_count, np.int64))]
        return np.asarray(self.contact_xy[off[t]:off[t + 1]], float)

    def contact_force_at(self, t: int) -> np.ndarray | None:
        """Tip force on each contact_xy atom of frame t ((n_t,)), or None without contact_force."""
        if self.contact_force is None or self.contact_count is None:
            return None
        off = np.r_[0, np.cumsum(np.asarray(self.contact_count, np.int64))]
        return np.asarray(self.contact_force[off[t]:off[t + 1]], float)

    def tip_speed_m_s(self) -> float:
        """Median |d depth / dt| over the samples where the tip moves (m/s)."""
        t, d, _ = self.curve()
        if len(t) < 2:
            return float("nan")
        dt = np.diff(t)
        v = np.abs(np.diff(d) / np.where(dt == 0, np.nan, dt))
        v = v[np.isfinite(v) & (v > 1e-6)]
        return float(np.median(v) * A_PER_PS_TO_M_PER_S) if len(v) else float("nan")

    # ------------------------------------------------------------------ validation
    def validate(self) -> "Trajectory":
        P = self.positions
        if P.ndim != 3 or P.shape[2] != 3:
            raise ValueError(f"positions must be (T, N, 3), got {P.shape}")
        T, N, _ = P.shape
        if T < 1 or N < 1:
            raise ValueError("empty trajectory")
        if not np.all(np.isfinite(P)):
            raise ValueError("positions contain NaN/inf")
        if self.species.shape != (N,):
            raise ValueError(f"species must be (N,) = ({N},), got {self.species.shape}")
        for name in ("box_lo", "box_hi"):
            b = getattr(self, name)
            if b.shape not in ((3,), (T, 3)):
                raise ValueError(f"{name} must be (3,) or (T, 3), got {b.shape}")
        if np.any(np.asarray(self.box_hi) <= np.asarray(self.box_lo)):
            raise ValueError("box_hi must exceed box_lo")
        if self.pbc.shape != (3,) or self.pbc.dtype != bool:
            raise ValueError("pbc must be a bool (3,) array")
        if self.pbc[2]:
            raise ValueError("indentation axis z must be non-periodic (pbc[2] False)")
        for name, shape in (("time_ps", (T,)), ("tip_xyz", (T, 3)), ("depth_A", (T,)), ("load_nN", (T,))):
            a = getattr(self, name)
            if a.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"{name} contains NaN/inf")
        if T > 1 and np.any(np.diff(self.time_ps) <= 0):
            raise ValueError("time_ps must be strictly increasing")
        if not np.allclose(self.surface_z0_A - self.tip_xyz[:, 2], self.depth_A, atol=1e-3):
            raise ValueError("depth_A must equal surface_z0_A - tip_xyz[:, 2] (to 1e-3 A)")
        if self.atom_ids is not None and self.atom_ids.shape != (N,):
            raise ValueError("atom_ids must be (N,)")
        for k, v in self.atom_fields.items():
            if np.shape(v) != (T, N):
                raise ValueError(f"atom field {k!r} must be (T, N) = ({T}, {N}), got {np.shape(v)}")
        if (self.contact_xy is None) != (self.contact_count is None):
            raise ValueError("contact_xy and contact_count must be given together")
        if self.contact_count is not None:
            cc = np.asarray(self.contact_count)
            if cc.shape != (T,) or np.any(cc < 0):
                raise ValueError(f"contact_count must be (T,) = ({T},) non-negative, got {cc.shape}")
            if np.shape(self.contact_xy) != (int(cc.sum()), 2):
                raise ValueError(f"contact_xy must be (sum(contact_count), 2) = ({int(cc.sum())}, 2), "
                                 f"got {np.shape(self.contact_xy)}")
        if self.contact_force is not None:
            if self.contact_count is None or np.shape(self.contact_force) != (int(np.sum(self.contact_count)),):
                raise ValueError("contact_force needs contact_xy/contact_count and one value per contact_xy row")
        cv = [self.curve_time_ps, self.curve_depth_A, self.curve_load_nN]
        if any(c is not None for c in cv):
            if any(c is None for c in cv):
                raise ValueError("curve_time_ps, curve_depth_A and curve_load_nN must be given together")
            if not (cv[0].shape == cv[1].shape == cv[2].shape) or cv[0].ndim != 1:
                raise ValueError("curve_* arrays must be 1-D and of equal length")
            if len(cv[0]) > 1 and np.any(np.diff(cv[0]) <= 0):
                raise ValueError("curve_time_ps must be strictly increasing (drop duplicated steps)")
        return self

    # ------------------------------------------------------------------ derived trajectories
    def subset(self, atoms: np.ndarray, note: str | None = None) -> "Trajectory":
        """The same run restricted to an atom subset (index array or bool mask), e.g. for a compact file."""
        idx = np.flatnonzero(atoms) if np.asarray(atoms).dtype == bool else np.asarray(atoms, int)
        meta = dict(self.meta)
        meta.setdefault("n_atoms_total", self.n_atoms)
        if note:
            meta["subset"] = note
        return Trajectory(
            positions=self.positions[:, idx], species=self.species[idx], box_lo=self.box_lo, box_hi=self.box_hi,
            pbc=self.pbc, time_ps=self.time_ps, tip_xyz=self.tip_xyz, depth_A=self.depth_A, load_nN=self.load_nN,
            surface_z0_A=self.surface_z0_A, indenter=self.indenter, meta=meta,
            atom_ids=None if self.atom_ids is None else self.atom_ids[idx], curve_time_ps=self.curve_time_ps,
            curve_depth_A=self.curve_depth_A, curve_load_nN=self.curve_load_nN,
            atom_fields={k: np.asarray(v)[:, idx] for k, v in self.atom_fields.items()},
            contact_xy=self.contact_xy, contact_count=self.contact_count, contact_force=self.contact_force,
        ).validate()

    # ------------------------------------------------------------------ npz I/O
    def _arrays(self) -> dict[str, np.ndarray]:
        arrs = dict(
            format=np.array(FORMAT), positions=np.asarray(self.positions, np.float32),
            species=np.asarray(self.species).astype("U3"), box_lo=np.asarray(self.box_lo, float),
            box_hi=np.asarray(self.box_hi, float), pbc=np.asarray(self.pbc, bool),
            time_ps=np.asarray(self.time_ps, float), tip_xyz=np.asarray(self.tip_xyz, float),
            depth_A=np.asarray(self.depth_A, float), load_nN=np.asarray(self.load_nN, float),
            surface_z0_A=np.array(float(self.surface_z0_A)), indenter_json=np.array(self.indenter.to_json()),
            meta_json=np.array(json.dumps(self.meta, sort_keys=True, default=_json_default)),
        )
        if self.atom_ids is not None:
            arrs["atom_ids"] = np.asarray(self.atom_ids, np.int64)
        if self.curve_time_ps is not None:
            arrs.update(curve_time_ps=np.asarray(self.curve_time_ps, float),
                        curve_depth_A=np.asarray(self.curve_depth_A, float),
                        curve_load_nN=np.asarray(self.curve_load_nN, float))
        for k, v in self.atom_fields.items():
            arrs[f"atomfield__{k}"] = np.asarray(v, np.float32)
        if self.contact_count is not None:  # optional: files without it keep their bytes and content hash
            arrs["contact_xy"] = np.asarray(self.contact_xy, np.float32).reshape(-1, 2)
            arrs["contact_count"] = np.asarray(self.contact_count, np.int64)
            if self.contact_force is not None:
                arrs["contact_force"] = np.asarray(self.contact_force, np.float32).reshape(-1)
        return arrs

    def content_sha256(self) -> str:
        """sha256 over the stored arrays (names, dtypes, shapes and bytes, sorted by name): the run's identity."""
        h = hashlib.sha256()
        for k, v in sorted(self._arrays().items()):
            a = np.ascontiguousarray(v)
            h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def save(self, path: str | Path) -> Path:
        """Write the ``indentviz-traj/1`` .npz (float32 positions and atom fields, deflate, deterministic bytes)."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_npz_deterministic(path, self._arrays())
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        with np.load(path, allow_pickle=False) as z:
            fmt = str(z["format"]) if "format" in z else "?"
            if fmt != FORMAT:
                raise ValueError(f"{path}: format {fmt!r}, expected {FORMAT!r}")

            def g(k):
                return np.asarray(z[k]) if k in z else None

            tr = cls(
                positions=g("positions").astype(float), species=g("species").astype(str), box_lo=g("box_lo"),
                box_hi=g("box_hi"), pbc=g("pbc").astype(bool), time_ps=g("time_ps"), tip_xyz=g("tip_xyz"),
                depth_A=g("depth_A"), load_nN=g("load_nN"), surface_z0_A=float(z["surface_z0_A"]),
                indenter=Indenter.from_json(str(z["indenter_json"])), meta=json.loads(str(z["meta_json"])),
                atom_ids=g("atom_ids"), curve_time_ps=g("curve_time_ps"), curve_depth_A=g("curve_depth_A"),
                curve_load_nN=g("curve_load_nN"),
                atom_fields={k[len("atomfield__"):]: np.asarray(z[k], float) for k in z.files
                             if k.startswith("atomfield__")},
                contact_xy=None if g("contact_xy") is None else g("contact_xy").astype(float),
                contact_count=g("contact_count"),
                contact_force=None if g("contact_force") is None else g("contact_force").astype(float),
            )
        return tr.validate()


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return str(x)


def _save_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """``np.savez_compressed`` with sorted members and a fixed timestamp (numpy stamps the wall clock)."""
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k in sorted(arrays):
            info = zipfile.ZipInfo(k + ".npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(arrays[k]), allow_pickle=False)
    tmp.replace(path)


def read_indent_traj(path: str | Path) -> Trajectory:
    """Read and validate an ``indentviz-traj/1`` file."""
    return Trajectory.load(path)


def write_indent_traj(path: str | Path, tr: Trajectory) -> Path:
    """Validate and write ``tr`` as an ``indentviz-traj/1`` file."""
    return tr.save(path)


class IndentTrajWriter:
    """Accumulate frames (float32 positions) and write one ``indentviz-traj/1`` file on :meth:`close`.

    For engines that produce frames one at a time (the harness MD driver calls :meth:`add_frame` at chunk
    boundaries). The atom set is fixed at construction: pass the positions of the SAME atoms every frame (for a
    subset, index with a mask chosen once, e.g. :func:`animation_subset_mask` on the first frame). Memory is
    T x N x 12 bytes (a 2e4-atom subset x 300 frames = 72 MB).

        w = IndentTrajWriter("traj.npz", species=..., box_lo=..., box_hi=..., pbc=(True, True, False),
                             surface_z0_A=z0, indenter={"kind": "berkovich", "tip_radius_A": 10.0}, meta={...})
        w.add_frame(t_ps, positions, tip_xyz, load_nN, atom_fields={"contact": sdf_mask})
        w.set_curve(t_dense, depth_dense, load_dense)       # optional dense load-depth record
        w.close()
    """

    def __init__(self, path: str | Path, *, species, box_lo, box_hi, pbc, surface_z0_A: float,
                 indenter: Indenter | dict, meta: dict | None = None, atom_ids=None, max_frames: int | None = None):
        self.path = Path(path)
        self.species = np.asarray(species).astype(str)
        self.box_lo0 = np.asarray(box_lo, float)
        self.box_hi0 = np.asarray(box_hi, float)
        self.pbc = np.asarray(pbc, bool)
        self.surface_z0_A = float(surface_z0_A)
        self.indenter = indenter if isinstance(indenter, Indenter) else Indenter.from_dict(indenter)
        self.meta = dict(meta or {})
        self.atom_ids = None if atom_ids is None else np.asarray(atom_ids, np.int64)
        self.max_frames = max_frames
        self._t, self._pos, self._tip, self._load, self._lo, self._hi = [], [], [], [], [], []
        self._fields: dict[str, list] = {}
        self._curve = None
        self.closed = False

    @property
    def n_frames(self) -> int:
        return len(self._t)

    def add_frame(self, time_ps: float, positions, tip_xyz, load_nN: float, *, box_lo=None, box_hi=None,
                  atom_fields: dict | None = None) -> bool:
        """Append one frame; returns False (and stores nothing) once ``max_frames`` is reached."""
        if self.closed:
            raise RuntimeError("writer is closed")
        if self.max_frames is not None and self.n_frames >= self.max_frames:
            return False
        P = np.asarray(positions, np.float32)
        if P.shape != (len(self.species), 3):
            raise ValueError(f"positions must be ({len(self.species)}, 3), got {P.shape}")
        if self._t and float(time_ps) <= self._t[-1]:
            raise ValueError("time_ps must increase from frame to frame")
        fields = dict(atom_fields or {})
        if self._t and set(fields) != set(self._fields):
            raise ValueError(f"atom fields must be the same every frame ({sorted(self._fields)})")
        self._t.append(float(time_ps))
        self._pos.append(P)
        self._tip.append(np.asarray(tip_xyz, float).reshape(3))
        self._load.append(float(load_nN))
        self._lo.append(self.box_lo0 if box_lo is None else np.asarray(box_lo, float))
        self._hi.append(self.box_hi0 if box_hi is None else np.asarray(box_hi, float))
        for k, v in fields.items():
            self._fields.setdefault(k, []).append(np.asarray(v, np.float32).reshape(len(self.species)))
        return True

    def set_curve(self, time_ps, depth_A, load_nN) -> None:
        """Dense load-depth record (e.g. every logged MD step); replaces any earlier one."""
        self._curve = tuple(np.asarray(a, float).ravel() for a in (time_ps, depth_A, load_nN))

    def to_trajectory(self) -> Trajectory:
        if not self._t:
            raise ValueError("no frames")
        lo, hi = np.stack(self._lo), np.stack(self._hi)
        if np.all(lo == lo[0]) and np.all(hi == hi[0]):
            lo, hi = lo[0], hi[0]
        tip = np.stack(self._tip)
        tr = Trajectory(
            positions=np.stack(self._pos), species=self.species, box_lo=lo, box_hi=hi, pbc=self.pbc,
            time_ps=np.asarray(self._t), tip_xyz=tip, depth_A=self.surface_z0_A - tip[:, 2],
            load_nN=np.asarray(self._load), surface_z0_A=self.surface_z0_A, indenter=self.indenter, meta=self.meta,
            atom_ids=self.atom_ids, atom_fields={k: np.stack(v) for k, v in self._fields.items()},
        )
        if self._curve is not None:
            tr.curve_time_ps, tr.curve_depth_A, tr.curve_load_nN = self._curve
        return tr.validate()

    def close(self) -> Path:
        path = self.to_trajectory().save(self.path)
        self.closed = True
        return path

    def __enter__(self) -> "IndentTrajWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None and not self.closed and self._t:
            self.close()


def estimate_surface_z(pos0: np.ndarray, bin_A: float = 4.0, window_A: float = 0.45, n_iter: int = 3) -> float:
    """Mean z of the top atomic layer of a flat slab (robust to thermal noise and a few adatoms).

    The highest atom of every lateral (x, y) column of width ``bin_A`` is (almost always) a top-layer atom, so
    the median of these column maxima is a robust first guess; it is then refined to the mean of the atoms
    within ``window_A`` of the current estimate. A gap-based layer split was tried first and failed on a
    300 K (001) SiC slab whose relaxed top C layer and the Si layer below were closer than the gap threshold
    (0.50 A error). Agrees with LAMMPS xcm(top, z) to ~1e-5 A on three real runs.
    """
    p = np.asarray(pos0, float)
    xy = p[:, :2] - p[:, :2].min(axis=0)
    n = np.maximum(1, np.ceil((xy.max(axis=0) + 1e-9) / bin_A).astype(int))
    ix = np.minimum((xy[:, 0] / bin_A).astype(int), n[0] - 1)
    iy = np.minimum((xy[:, 1] / bin_A).astype(int), n[1] - 1)
    colmax = np.full(n[0] * n[1], -np.inf)
    np.maximum.at(colmax, iy * n[0] + ix, p[:, 2])
    zref = float(np.median(colmax[np.isfinite(colmax)]))
    for _ in range(n_iter):
        sel = np.abs(p[:, 2] - zref) <= window_A
        zref = float(np.mean(p[sel, 2]))
    return zref


def animation_subset_mask(pos0: np.ndarray, box_lo, box_hi, pbc, axis_xy, surface_z0_A: float, *,
                          slice_half_width_A: float = 2.0, halo_A: float = 3.5, near_surface_A: float = 6.0,
                          direction_xy=(1.0, 0.0)) -> np.ndarray:
    """Atoms to keep in an animation trajectory (bool (N,)), chosen once from the first frame.

    * a slice through the tip axis (|offset normal to ``direction_xy``| <= half width + halo; the halo lets the
      local strain and bonds of the slice atoms see their neighbours: use halo >= the strain cutoff, 3.35 A);
    * every atom within ``near_surface_A`` below the initial surface (the tracked top layer of the height map
      and the atoms that can touch the tip, which define the estimated contact outline).
    """
    p = np.asarray(pos0, float)
    lo, hi = np.asarray(box_lo, float), np.asarray(box_hi, float)
    d = p[:, :2] - np.asarray(axis_xy, float)
    L = (hi - lo)[:2]
    for k in range(2):
        if pbc[k]:
            d[:, k] -= L[k] * np.round(d[:, k] / L[k])
    u = np.asarray(direction_xy, float)
    u = u / np.linalg.norm(u)
    normal = np.abs(d @ np.array([-u[1], u[0]]))
    in_slice = normal <= slice_half_width_A + halo_A
    near_top = p[:, 2] >= surface_z0_A - near_surface_A
    return in_slice | near_top


# ====================================================================================== LAMMPS adapter
def open_text(path: str | Path):
    """Open a plain or gzip-compressed (``.gz``) ASCII text file for reading."""
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="ascii")
    return open(path, encoding="ascii")


_open_text = open_text  # backward-compatible alias


def read_lammps_dump(path: str | Path, type_map: dict[int, str] | None = None, every: int = 1,
                     max_frames: int | None = None) -> dict:
    """Read a text LAMMPS dump (``dump atom`` / ``dump custom``; .gz allowed) with an orthogonal box.

    Needs an ``id`` column, ``type`` (with ``type_map``) or ``element``, and one position set among
    x y z | xu yu zu | xs ys zs | xsu ysu zsu (scaled ones are converted). Atoms are sorted by id and
    every frame must contain the same ids. Returns dict(timestep (T,), positions (T,N,3), ids (N,),
    species (N,), box_lo (T,3), box_hi (T,3), pbc (3,), columns [...], extra {column: (T,N)} for every other
    numeric per-atom column, e.g. c_cn[1]; truncated: True if an incomplete last frame was skipped).
    """
    steps, pos_all, lo_all, hi_all = [], [], [], []
    ids_ref = species = pbc = None
    cols_ref = None
    extra_all: dict[str, list] = {}
    truncated = False
    skip = {"id", "type", "element", "x", "y", "z", "xu", "yu", "zu", "xs", "ys", "zs", "xsu", "ysu", "zsu",
            "ix", "iy", "iz", "mol"}
    k = 0
    with open_text(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
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
                bflags = flags[:3] if len(flags) >= 3 else ["pp", "pp", "pp"]
                cols = fh.readline().split()[2:]
                block = [fh.readline() for _ in range(n)]
                if not block or not block[-1].endswith("\n") or len(block[-1].split()) != len(cols):
                    raise EOFError
            except (EOFError, IndexError):
                truncated = True  # a frame still being written (growing dump): ignore it
                break
            take = (k % every == 0)
            k += 1
            if not take:
                continue
            arr = np.loadtxt(io.StringIO("".join(block)), dtype=str, ndmin=2)
            ci = {c: i for i, c in enumerate(cols)}
            if "id" not in ci:
                raise ValueError("dump needs an 'id' column")
            ids = arr[:, ci["id"]].astype(np.int64)
            order = np.argsort(ids, kind="stable")
            ids = ids[order]
            arr = arr[order]
            lo, hi = bounds[:, 0], bounds[:, 1]
            for trip, scaled in ((("x", "y", "z"), False), (("xu", "yu", "zu"), False), (("xs", "ys", "zs"), True),
                                 (("xsu", "ysu", "zsu"), True)):
                if all(c in ci for c in trip):
                    xyz = arr[:, [ci[c] for c in trip]].astype(float)
                    if scaled:
                        xyz = lo + xyz * (hi - lo)
                    break
            else:
                raise ValueError(f"no position columns in dump (columns: {cols})")
            if ids_ref is None:
                ids_ref, cols_ref = ids, cols
                pbc = np.array([f == "pp" for f in bflags], bool)
                if "element" in ci:
                    species = arr[:, ci["element"]].astype(str)
                elif "type" in ci:
                    types = arr[:, ci["type"]].astype(int)
                    if type_map is None:
                        raise ValueError("dump has 'type' but no 'element' column: pass type_map {1: 'C', ...}")
                    missing = sorted(set(types.tolist()) - set(type_map))
                    if missing:
                        raise ValueError(f"type_map lacks types {missing}")
                    species = np.array([type_map[t] for t in types.tolist()])
                else:
                    raise ValueError("dump needs 'type' or 'element'")
            elif not np.array_equal(ids, ids_ref):
                raise ValueError(f"atom ids change at timestep {step} (lost atoms?): not supported")
            for c in cols:
                if c not in skip:
                    try:
                        extra_all.setdefault(c, []).append(arr[:, ci[c]].astype(float))
                    except ValueError:
                        pass
            steps.append(step)
            pos_all.append(xyz)
            lo_all.append(lo)
            hi_all.append(hi)
            if max_frames is not None and len(steps) >= max_frames:
                break
    if not steps:
        raise ValueError(f"{path}: no frames")
    extra = {k: np.stack(v) for k, v in extra_all.items() if len(v) == len(steps)}
    return dict(timestep=np.array(steps), positions=np.stack(pos_all), ids=ids_ref, species=species,
                box_lo=np.stack(lo_all), box_hi=np.stack(hi_all), pbc=pbc, columns=cols_ref, extra=extra,
                truncated=truncated)


def read_lammps_dumps(paths, type_map: dict[int, str] | None = None, every: int = 1,
                      max_frames: int | None = None) -> dict:
    """:func:`read_lammps_dump` over several files of one run (e.g. one dump per restart segment), merged by step.

    The files are read in the order given; when two files hold the same timestep the LATER file wins (a
    segment restarted from a checkpoint recomputes the steps after it, so frames a killed segment wrote past
    the restart point are stale). Every file must dump the same atoms. ``every`` and ``max_frames`` apply to
    the merged frame list. A single path behaves exactly like :func:`read_lammps_dump`.
    """
    if isinstance(paths, (str, Path)):
        return read_lammps_dump(paths, type_map, every=every, max_frames=max_frames)
    paths = [Path(q) for q in paths]
    if not paths:
        raise ValueError("no dump files")
    if len(paths) == 1:
        return read_lammps_dump(paths[0], type_map, every=every, max_frames=max_frames)
    parts = [read_lammps_dump(q, type_map) for q in paths]
    ref = parts[0]
    for q, d in zip(paths[1:], parts[1:]):
        if not np.array_equal(d["ids"], ref["ids"]):
            raise ValueError(f"{q}: dumps a different atom set than {paths[0]}")
    owner: dict[int, tuple[int, int]] = {}
    for k, d in enumerate(parts):
        for j, st in enumerate(d["timestep"].tolist()):
            owner[int(st)] = (k, j)  # later files override earlier ones
    steps = sorted(owner)[::every]
    if max_frames is not None:
        steps = steps[:max_frames]
    pick = [owner[s_] for s_ in steps]

    def gather(key):
        return np.stack([parts[k][key][j] for k, j in pick])

    extra_keys = set.intersection(*(set(d["extra"]) for d in parts))
    return dict(timestep=np.array(steps), positions=gather("positions"), ids=ref["ids"], species=ref["species"],
                box_lo=gather("box_lo"), box_hi=gather("box_hi"), pbc=ref["pbc"], columns=ref["columns"],
                extra={c: np.stack([parts[k]["extra"][c][j] for k, j in pick]) for c in sorted(extra_keys)},
                truncated=any(d["truncated"] for d in parts), files=[str(q) for q in paths])


def _strip_parens(s: str) -> str:
    """'Fz_eV/A(on indenter, +up) Rx' -> 'Fz_eV/A Rx' (annotations in fix-print titles contain spaces)."""
    out, depth = [], 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def read_lammps_table(path: str | Path) -> dict[str, np.ndarray]:
    """Read a whitespace table written by ``fix ave/time`` or ``fix print`` (header on a '#' line).

    The LAST comment line whose token count matches the data columns names them (``# TimeStep v_z f_ind[3]``);
    parenthesised annotations in a header are removed first (``Fz(sum over tip atoms, eV/A, +up)`` -> ``Fz``).
    Columns can always also be addressed as ``c0``, ``c1``, ... Multi-row (``mode vector``) ave/time files are
    not supported. Rows are returned as written (duplicated steps at run boundaries included).
    """
    names, rows = None, []
    with open_text(path) as fh:
        comments = []
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                comments.append(_strip_parens(s.lstrip("#")).split())
                continue
            rows.append([float(v) for v in s.split()])
    if not rows:
        raise ValueError(f"{path}: no data rows")
    ncol = len(rows[0])
    for c in reversed(comments):
        if len(c) == ncol:
            names = c
            break
    if names is None:
        names = [f"c{i}" for i in range(ncol)]
    data = np.array([r for r in rows if len(r) == ncol])
    out = {f"c{i}": data[:, i] for i in range(ncol)}
    out.update({nm: data[:, i] for i, nm in enumerate(names)})
    return out


def read_lammps_tables(paths, step_col: str = "step") -> dict[str, np.ndarray]:
    """:func:`read_lammps_table` over several files of one run (restart segments), merged by ``step_col``.

    Within one file a repeated step keeps its first occurrence (``fix print`` writes the boundary step of
    consecutive runs twice); across files the LATER file wins (a restarted segment recomputes the steps after
    its checkpoint). Rows are returned sorted by step. A single path is read unchanged (no de-duplication:
    :func:`trajectory_from_lammps` does that and records it).
    """
    if isinstance(paths, (str, Path)):
        return read_lammps_table(paths)
    paths = [Path(q) for q in paths]
    if len(paths) == 1:
        return read_lammps_table(paths[0])
    tabs = [read_lammps_table(q) for q in paths]
    cols = [c for c in tabs[0] if all(c in t for t in tabs[1:])]
    owner: dict[float, tuple[int, int]] = {}
    for k, t in enumerate(tabs):
        st = np.asarray(t[step_col], float)
        _, first = np.unique(st, return_index=True)
        for j in first.tolist():
            owner[float(st[j])] = (k, j)
    steps = sorted(owner)
    return {c: np.array([tabs[k][c][j] for k, j in (owner[s_] for s_ in steps)]) for c in cols}


def read_lammps_log_thermo(path: str | Path) -> list[dict[str, np.ndarray]]:
    """Thermo blocks of a log.lammps: one dict per ``run`` (keys = thermo column names).

    A block starts at a header line beginning with ``Step`` and ends at ``Loop time`` (or end of file);
    lines inside it that are not all-numeric with the header's column count (warnings) are skipped.
    """
    blocks, cur, names = [], None, None

    def _close():
        if cur:
            arr = np.array(cur)
            blocks.append({nm: arr[:, i] for i, nm in enumerate(names)})

    with open_text(path) as fh:
        for line in fh:
            tok = line.split()
            if tok[:1] == ["Step"]:
                _close()
                names, cur = tok, []
                continue
            if cur is None:
                continue
            if line.startswith("Loop time"):
                _close()
                cur = None
                continue
            if len(tok) == len(names):
                try:
                    cur.append([float(v) for v in tok])
                except ValueError:
                    pass
    _close()
    return blocks


def read_lammps_meta_line(path: str | Path, key: str) -> dict:
    """``key=value`` pairs of the LAST log line that starts with ``key`` (e.g. a ``print "BERK_META zs0=..."``).

    The echo of the print command itself starts with ``print`` and is ignored. Numeric values become floats.
    """
    found = None
    with open_text(path) as fh:
        for line in fh:
            if line.startswith(key + " "):
                found = line
    if found is None:
        raise ValueError(f"no line starting with {key!r} in {path}")
    out = {}
    for kv in found.split()[1:]:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def trajectory_from_lammps(dump: str | Path | list, *, type_map: dict[int, str] | None, timestep_ps: float,
                           indenter: Indenter | dict, table: dict[str, np.ndarray], step_col: str, tip_z_col: str,
                           force_z_col: str, tip_ref: str = "centre", tip_xy: tuple[float, float] | None = None,
                           force_sign: float = 1.0, force_units: str = "eV/A", surface_z0_A: float | None = None,
                           every: int = 1, max_frames: int | None = None, meta: dict | None = None) -> Trajectory:
    """Build a Trajectory from a LAMMPS dump plus a per-step indenter table (``fix print`` / ``fix ave/time``).

    table[step_col], table[tip_z_col], table[force_z_col]: the indenter's reference height and the force
    ON THE INDENTER along z (LAMMPS ``fix indent`` f_ID[3], or the summed force on rigid tip atoms; eV/A in
    metal units; > 0 when the surface pushes back). ``tip_ref`` = "centre" (LAMMPS sphere ``fix indent``
    coordinate: apex = centre - R) or "apex" (already the lowest point). Rows with a repeated step (``fix
    print`` writes the boundary step of consecutive runs twice) keep their first occurrence. The table is
    interpolated linearly in step onto the dump frames and also kept as the dense load-depth curve.
    ``surface_z0_A`` defaults to the mean height of the top atomic layer of frame 0 (estimate_surface_z); pass
    the engine's own value (LAMMPS xcm(top, z)) when it exists. ``dump`` may be a list of files of one run
    (restart segments; :func:`read_lammps_dumps`: a later file wins on a repeated step).
    """
    ind = indenter if isinstance(indenter, Indenter) else Indenter.from_dict(indenter)
    d = read_lammps_dumps(dump, type_map, every=every, max_frames=max_frames)
    P = d["positions"]
    steps = d["timestep"].astype(float)
    tstep = np.asarray(table[step_col], float)
    order = np.argsort(tstep, kind="stable")
    tstep = tstep[order]
    _, first = np.unique(tstep, return_index=True)
    n_dup = len(tstep) - len(first)
    keep = order[np.sort(first)] if n_dup else order
    tstep = np.asarray(table[step_col], float)[keep]
    ztab = np.asarray(table[tip_z_col], float)[keep]
    ftab = np.asarray(table[force_z_col], float)[keep] * force_sign
    if force_units in ("eV/A", "eV/Angstrom"):
        ftab = ftab * EV_PER_A_TO_NN
    elif force_units != "nN":
        raise ValueError("force_units must be 'eV/A' or 'nN'")
    if tip_ref == "centre":
        if ind.kind == "sphere":
            ztab = ztab - ind.radius_A
        # a LAMMPS cone indenter is given by its apex already; pyramids are not in LAMMPS
    elif tip_ref != "apex":
        raise ValueError("tip_ref must be 'centre' or 'apex'")
    # ``fix print N`` does not write step 0, so the first dump frame can precede the table by one interval:
    # allow up to one table interval outside the table (tip height extrapolated linearly from the two nearest
    # rows, force held at the nearest row) and record which frames were treated that way.
    dstep = float(np.median(np.diff(tstep))) if len(tstep) > 1 else 0.0
    if steps.min() < tstep.min() - 1.001 * dstep or steps.max() > tstep.max() + 1.001 * dstep:
        raise ValueError(f"indenter table covers steps {tstep.min():g}..{tstep.max():g}, "
                         f"dump {steps.min():g}..{steps.max():g}")
    zt = np.interp(steps, tstep, ztab)
    ft = np.interp(steps, tstep, ftab)
    extrap = []
    if len(tstep) > 1:
        lo_m, hi_m = steps < tstep[0], steps > tstep[-1]
        s0 = (ztab[1] - ztab[0]) / (tstep[1] - tstep[0])
        s1 = (ztab[-1] - ztab[-2]) / (tstep[-1] - tstep[-2])
        zt = np.where(lo_m, ztab[0] + s0 * (steps - tstep[0]), zt)
        zt = np.where(hi_m, ztab[-1] + s1 * (steps - tstep[-1]), zt)
        extrap = [int(v) for v in steps[lo_m | hi_m]]
    lo, hi = d["box_lo"], d["box_hi"]
    if tip_xy is None:
        tip_xy = tuple(0.5 * (lo[0, :2] + hi[0, :2]))
    if surface_z0_A is None:
        surface_z0_A = estimate_surface_z(P[0])
    tip = np.column_stack([np.full(len(zt), tip_xy[0]), np.full(len(zt), tip_xy[1]), zt])
    meta = dict(meta or {})
    meta.setdefault("source", {"dump": [str(q) for q in dump] if isinstance(dump, (list, tuple)) else str(dump)})
    meta.setdefault("engine", "LAMMPS")
    meta.setdefault("timestep_ps", timestep_ps)
    meta.setdefault("n_atoms_total", int(P.shape[1]))
    if extrap:
        meta["table_extrapolated_steps"] = extrap
    if n_dup:
        meta["table_duplicate_steps_dropped"] = int(n_dup)
    if d["truncated"]:
        meta["dump_truncated_last_frame_skipped"] = True
    tr = Trajectory(
        positions=P, species=np.asarray(d["species"]), box_lo=lo, box_hi=hi, pbc=d["pbc"],
        time_ps=steps * timestep_ps, tip_xyz=tip, depth_A=surface_z0_A - zt, load_nN=ft,
        surface_z0_A=float(surface_z0_A), indenter=ind, meta=meta, atom_ids=d["ids"], curve_time_ps=tstep * timestep_ps,
        curve_depth_A=surface_z0_A - ztab, curve_load_nN=ftab, atom_fields=d["extra"],
    )
    if tr.pbc[2]:
        raise ValueError("dump box is periodic in z; indentation needs a free surface along z")
    if np.all(tr.box_lo == tr.box_lo[0]) and np.all(tr.box_hi == tr.box_hi[0]):
        tr.box_lo, tr.box_hi = tr.box_lo[0], tr.box_hi[0]  # fixed box: store it once
    return tr.validate()


def read_lammps_contact_dump(path, flag_col: str | None = None) -> dict:
    """Read a LAMMPS dump whose atom set CHANGES from frame to frame (``dump_modify thresh c_ctip > 0``).

    Used for the atoms touching the indenter: a slice dump cannot hold them all once the tip is deeper than the
    top layer. Every frame needs id and x y (z optional; scaled xs ys are converted); a frame with zero atoms
    (no contact) is allowed. ``flag_col`` (e.g. "c_ctip" or "d_pen"): keep only rows with flag > 0 (for a dump
    written without ``thresh``). Returns dict(timestep (T,), xy [list of (n_t, 2)], xyz [list of (n_t, 3); z is
    NaN without a z column], ids [list of (n_t,)], truncated: True if an incomplete last frame was skipped).
    ``path`` may be a list of files of one run (restart segments, in run order): frames are merged by timestep, a
    LATER file winning on a repeated step.
    """
    if isinstance(path, (list, tuple)):
        if not path:
            raise ValueError("no contact dump files")
        parts = [read_lammps_contact_dump(q, flag_col) for q in path]
        owner: dict[int, tuple[int, int]] = {}
        for k, d in enumerate(parts):
            for j, st in enumerate(d["timestep"].tolist()):
                owner[int(st)] = (k, j)
        steps_m = sorted(owner)
        pick = [owner[s_] for s_ in steps_m]
        return dict(timestep=np.array(steps_m), xy=[parts[k]["xy"][j] for k, j in pick],
                    xyz=[parts[k]["xyz"][j] for k, j in pick], ids=[parts[k]["ids"][j] for k, j in pick],
                    truncated=any(d["truncated"] for d in parts))
    steps, xys, xyzs, idss = [], [], [], []
    truncated = False
    with open_text(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
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
                bounds = np.array([[float(v) for v in fh.readline().split()[:2]] for _ in range(3)])
                cols = fh.readline().split()[2:]
                block = [fh.readline() for _ in range(n)]
                if n and (not block[-1].endswith("\n") or len(block[-1].split()) != len(cols)):
                    raise EOFError
            except (EOFError, IndexError):
                truncated = True
                break
            ci = {c: i for i, c in enumerate(cols)}
            if "id" not in ci:
                raise ValueError("contact dump needs an 'id' column")
            arr = np.array("".join(block).split(), dtype=object).reshape(n, len(cols)) if n else \
                np.zeros((0, len(cols)), dtype=object)
            lo, hi = bounds[:, 0], bounds[:, 1]
            if all(c in ci for c in ("x", "y")):
                xyz = np.full((n, 3), np.nan)
                xyz[:, :2] = arr[:, [ci["x"], ci["y"]]].astype(float)
                if "z" in ci:
                    xyz[:, 2] = arr[:, ci["z"]].astype(float)
            elif all(c in ci for c in ("xs", "ys")):
                xyz = np.full((n, 3), np.nan)
                xyz[:, :2] = lo[:2] + arr[:, [ci["xs"], ci["ys"]]].astype(float) * (hi[:2] - lo[:2])
                if "zs" in ci:
                    xyz[:, 2] = lo[2] + arr[:, ci["zs"]].astype(float) * (hi[2] - lo[2])
            else:
                raise ValueError(f"no x y (or xs ys) columns in contact dump (columns: {cols})")
            ids = arr[:, ci["id"]].astype(np.int64)
            if flag_col is not None and n:
                if flag_col not in ci:
                    raise ValueError(f"contact dump has no column {flag_col!r} (columns: {cols})")
                keep = arr[:, ci[flag_col]].astype(float) > 0
                xyz, ids = xyz[keep], ids[keep]
            order = np.argsort(ids, kind="stable")
            steps.append(step)
            xyzs.append(xyz[order].reshape(-1, 3))
            xys.append(xyz[order, :2].reshape(-1, 2))
            idss.append(ids[order])
    if not steps:
        raise ValueError(f"{path}: no frames")
    return dict(timestep=np.array(steps), xy=xys, xyz=xyzs, ids=idss, truncated=truncated)


def attach_contact_dump(tr: Trajectory, path, *, timestep_ps: float, flag_col: str | None = None,
                        tol_ps: float = 1e-6, force_fn=None) -> Trajectory:
    """Set ``tr.contact_xy``/``contact_count`` from a contact dump (:func:`read_lammps_contact_dump`; a file or a
    list of restart-segment files).

    Every trajectory frame must have a contact-dump frame at the same time (step x timestep_ps, to ``tol_ps``);
    extra contact frames are ignored. Raises if a frame is missing (a silent zero would draw no contact).
    ``force_fn(t, xyz) -> (n,)``: the tip force on the contact atoms of trajectory frame t from their positions
    (e.g. the WCA wall of an atomistic tip); stored as ``tr.contact_force`` (the dump needs a z column).
    """
    d = read_lammps_contact_dump(path, flag_col=flag_col)
    t_c = d["timestep"].astype(float) * timestep_ps
    xy, cnt, frc, missing = [], [], [], []
    for t, tt in enumerate(tr.time_ps):
        k = int(np.argmin(np.abs(t_c - tt)))
        if abs(t_c[k] - tt) > tol_ps:
            missing.append(round(float(tt), 6))
            continue
        xy.append(d["xy"][k])
        cnt.append(len(d["xy"][k]))
        if force_fn is not None:
            xyz = d["xyz"][k]
            if len(xyz) and not np.all(np.isfinite(xyz)):
                raise ValueError(f"{path}: the contact dump has no z column (needed for the tip force)")
            frc.append(np.asarray(force_fn(t, xyz), float).reshape(-1))
    if missing:
        raise ValueError(f"{path}: no contact frame at t = {missing[:5]} ps ({len(missing)} of {tr.n_frames} frames)")
    tr.contact_xy = np.vstack(xy) if xy else np.zeros((0, 2))
    tr.contact_count = np.asarray(cnt, np.int64)
    tr.contact_force = np.concatenate(frc) if force_fn is not None and frc else None
    tr.meta["contact_source"] = {"contact_dump": [str(q) for q in path] if isinstance(path, (list, tuple))
                                 else str(path), "flag_col": flag_col,
                                 "truncated_last_frame_skipped": bool(d["truncated"])}
    return tr.validate()

