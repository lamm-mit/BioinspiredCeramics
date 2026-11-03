"""Periodic 3D microstructures of particle-filled resins (spheres) and their voxelisation.

Two-scale treatment of multimodal dental fillers
------------------------------------------------
Dental "nano-hybrid" fillers span two decades in size (e.g. ~0.7 um silanised
Ba-Al-borosilicate glass plus ~40 nm silica/zirconia nanoclusters).  Resolving
both in one voxel RVE is impossible at 64^3-96^3, so the nano-population is
homogenised first into an effective *nanofilled matrix* phase (Mori-Tanaka,
see :func:`fracture_md.composites.homogenization.nanofilled_matrix`), and only
the micron-scale particles (and, optionally, sub-micron particles >= ~8 voxels
across) are placed explicitly here.  The total filler fraction is then
``phi_total = phi_rve + phi_nano_in_matrix * (1 - phi_rve)``.

Packing algorithm (deterministic for a given seed)
--------------------------------------------------
1. Particle counts per family from the target volume fraction and an
   approximate total particle number; diameters drawn from a log-normal with
   the requested coefficient of variation and rescaled so each family hits its
   volume fraction exactly (analytic sphere volume, not voxels).
2. Random sequential addition (RSA) at a reduced radius scale (large spheres
   first), then a Lubachevsky-Stillinger-"lite" growth: the radius scale is
   raised in steps to 1 and after each step overlaps (including a minimum
   surface gap) are removed by Jacobi pairwise push-apart moves weighted by
   inverse particle volume, with periodic minimum-image distances.
3. The result reports the analytic volume fraction, the largest remaining
   relative overlap and whether the relaxation converged.  Voxelisation reports
   the *measured* voxel volume fraction of every phase, which is what the FFT
   solver actually sees.

The optional interphase shell (thickness ``t``) models a silane/interphase
layer of different modulus around every particle; shell voxels never overwrite
particle voxels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

MATRIX_PHASE = "matrix"
INTERPHASE_PHASE = "interphase"


@dataclass(frozen=True)
class SphereFamily:
    """One particle population placed explicitly in the RVE.

    ``diameter`` is the median diameter (any length unit, used consistently,
    conventionally um); ``volume_fraction`` is its fraction of the RVE volume;
    ``cv`` the log-normal coefficient of variation of the diameter.
    """

    name: str
    phase: str
    diameter: float
    volume_fraction: float
    cv: float = 0.0


@dataclass
class SpherePacking:
    box: float
    centers: np.ndarray  # (n, 3) in [0, box)
    radii: np.ndarray  # (n,)
    phase_index: np.ndarray  # (n,) index into phase_names (0 is the matrix)
    phase_names: tuple[str, ...]
    family_index: np.ndarray  # (n,) index into families
    families: tuple[SphereFamily, ...]
    target_volume_fraction: float
    volume_fraction: float  # analytic sum of sphere volumes / box^3
    min_gap: float
    max_relative_overlap: float  # max over pairs of overlap / min(r_i, r_j) (gap included)
    converged: bool
    seed: int
    iterations: int
    notes: list[str] = field(default_factory=list)

    @property
    def n_particles(self) -> int:
        return int(self.radii.shape[0])

    def summary(self) -> dict:
        return {
            "box": float(self.box),
            "n_particles": self.n_particles,
            "phase_names": list(self.phase_names),
            "families": [dict(f.__dict__) for f in self.families],
            "target_volume_fraction": float(self.target_volume_fraction),
            "volume_fraction_analytic": float(self.volume_fraction),
            "min_gap": float(self.min_gap),
            "max_relative_overlap": float(self.max_relative_overlap),
            "converged": bool(self.converged),
            "seed": int(self.seed),
            "iterations": int(self.iterations),
            "notes": list(self.notes),
        }


@dataclass
class VoxelMicrostructure:
    labels: np.ndarray  # (n, n, n) int32 phase index
    phase_names: tuple[str, ...]
    voxel_size: float
    volume_fractions: dict[str, float]
    description: str = ""
    packing: Optional[SpherePacking] = None

    @property
    def n(self) -> int:
        return int(self.labels.shape[0])


def _sample_diameters(rng: np.random.Generator, fam: SphereFamily, count: int) -> np.ndarray:
    if fam.cv <= 0.0:
        return np.full(count, fam.diameter, dtype=float)
    s2 = np.log1p(fam.cv**2)
    return fam.diameter * np.exp(rng.normal(0.0, np.sqrt(s2), size=count))


def _min_image(d: np.ndarray, box: float) -> np.ndarray:
    return d - box * np.round(d / box)


def _overlaps(centers, radii, box, gap):
    """Pairwise surface overlaps (need - distance), including the requested minimum gap."""
    d = _min_image(centers[:, None, :] - centers[None, :, :], box)
    dist = np.sqrt(np.sum(d * d, axis=-1))
    need = radii[:, None] + radii[None, :] + gap
    ov = need - dist
    np.fill_diagonal(ov, -np.inf)
    return d, dist, ov


def pack_spheres(
    families: Sequence[SphereFamily],
    *,
    n_particles: int = 60,
    seed: int = 0,
    min_gap: float = 0.0,
    box: Optional[float] = None,
    growth_steps: int = 24,
    relax_iterations: int = 400,
    overlap_tol: float = 1e-3,
) -> SpherePacking:
    """Periodic random packing of polydisperse spheres at the families' target volume fractions.

    ``min_gap`` is an absolute surface-to-surface gap (same length unit as the
    diameters); a gap of >= 1 voxel prevents voxelisation from merging
    neighbours into spurious load paths.  ``box`` (edge length) defaults to the
    value that yields about ``n_particles`` particles.
    """
    fams = tuple(f for f in families if f.volume_fraction > 0.0)
    if not fams:
        raise ValueError("at least one family with volume_fraction > 0 is required")
    phi_target = float(sum(f.volume_fraction for f in fams))
    if phi_target >= 0.74:
        raise ValueError(f"target volume fraction {phi_target:.3f} exceeds sphere packing limits")
    rng = np.random.default_rng(seed)
    mean_vol = [np.pi / 6.0 * f.diameter**3 * (1.0 + f.cv**2) ** 3 for f in fams]  # E[d^3], log-normal
    if box is None:
        box = (n_particles / sum(f.volume_fraction / v for f, v in zip(fams, mean_vol))) ** (1.0 / 3.0)
    box = float(box)
    radii_l, fam_l = [], []
    for k, (f, v) in enumerate(zip(fams, mean_vol)):
        count = max(1, int(round(f.volume_fraction * box**3 / v)))
        d = _sample_diameters(rng, f, count)
        vol = np.pi / 6.0 * np.sum(d**3)
        d *= (f.volume_fraction * box**3 / vol) ** (1.0 / 3.0)  # hit the family fraction exactly
        radii_l.append(0.5 * d)
        fam_l.append(np.full(count, k))
    radii = np.concatenate(radii_l)
    fam_idx = np.concatenate(fam_l)
    order = np.argsort(-radii, kind="stable")
    radii, fam_idx = radii[order], fam_idx[order]
    n = radii.shape[0]
    if np.max(2 * radii) + min_gap >= box:
        raise ValueError("box too small for the largest particle; increase n_particles")

    # --- RSA at reduced radius scale (large spheres first)
    scale0 = 0.5
    centers = np.zeros((n, 3))
    for i in range(n):
        c = rng.uniform(0.0, box, size=3)
        for _attempt in range(2000):
            if i == 0:
                break
            d = _min_image(centers[:i] - c, box)
            dist = np.sqrt(np.sum(d * d, axis=1))
            if np.all(dist >= scale0 * (radii[:i] + radii[i]) + min_gap):
                break
            c = rng.uniform(0.0, box, size=3)
        centers[i] = c

    # --- growth + overlap relaxation (Lubachevsky-Stillinger-lite)
    vol_w = radii**3
    wi = vol_w[None, :] / (vol_w[:, None] + vol_w[None, :])  # the lighter particle moves more
    iterations = 0
    max_rel = np.inf
    for step in range(1, growth_steps + 1):
        s = scale0 + (1.0 - scale0) * step / growth_steps
        r = s * radii
        tol = overlap_tol * np.min(r)
        its = relax_iterations if step < growth_steps else 20 * relax_iterations
        for _ in range(its):
            iterations += 1
            d, dist, ov = _overlaps(centers, r, box, min_gap)
            mask = ov > tol
            if not mask.any():
                break
            dist_safe = np.where(dist > 1e-12, dist, 1e-12)
            unit = d / dist_safe[..., None]  # direction from j to i
            push = np.where(mask, 1.02 * ov * wi, 0.0)
            disp = np.sum(push[..., None] * unit, axis=1)
            centers = np.mod(centers + disp, box)
        _, _, ov = _overlaps(centers, r, box, min_gap)
        rmin = np.minimum(r[:, None], r[None, :])
        max_rel = float(np.max(np.where(np.isfinite(ov), ov / rmin, -np.inf)))
    converged = max_rel <= 10 * overlap_tol
    phase_names = [MATRIX_PHASE]
    for f in fams:
        if f.phase not in phase_names:
            phase_names.append(f.phase)
    phase_index = np.array([phase_names.index(fams[k].phase) for k in fam_idx], dtype=np.int32)
    phi = float(np.sum(4.0 / 3.0 * np.pi * radii**3) / box**3)
    notes = []
    if not converged:
        notes.append(
            f"overlap relaxation did not fully converge (max relative overlap {max_rel:.3g}); "
            "voxel volume fraction will be below target where particles overlap"
        )
    return SpherePacking(
        box=box,
        centers=centers,
        radii=radii,
        phase_index=phase_index,
        phase_names=tuple(phase_names),
        family_index=fam_idx,
        families=fams,
        target_volume_fraction=phi_target,
        volume_fraction=phi,
        min_gap=float(min_gap),
        max_relative_overlap=max(max_rel, 0.0),
        converged=bool(converged),
        seed=int(seed),
        iterations=iterations,
        notes=notes,
    )


def voxelize(packing: SpherePacking, n: int, *, shell_thickness: float = 0.0) -> VoxelMicrostructure:
    """Voxelise a periodic sphere packing on an n^3 grid (voxel-centre sampling).

    Label 0 is the (possibly nanofilled) matrix, labels 1.. the particle phases
    in ``packing.phase_names`` order, and, if ``shell_thickness > 0``, the last
    label is the interphase shell.
    """
    h = packing.box / n
    labels = np.zeros((n, n, n), dtype=np.int32)
    names = list(packing.phase_names)
    shell_label = None
    if shell_thickness > 0.0:
        names.append(INTERPHASE_PHASE)
        shell_label = len(names) - 1
    grid1 = (np.arange(n) + 0.5) * h

    def _paint(center, radius, label, only_matrix):
        lo = np.floor((center - radius) / h - 0.5).astype(int)
        hi = np.ceil((center + radius) / h + 0.5).astype(int)
        idx = []
        for a in range(3):
            ix = np.arange(lo[a], hi[a] + 1)
            ix = np.unique(np.mod(ix, n))  # never paint a voxel twice in one call
            idx.append(ix)
        dx = [_min_image(grid1[idx[a]] - center[a], packing.box) for a in range(3)]
        r2 = dx[0][:, None, None] ** 2 + dx[1][None, :, None] ** 2 + dx[2][None, None, :] ** 2
        inside = r2 <= radius**2
        ii = np.ix_(idx[0], idx[1], idx[2])
        sub = labels[ii]
        if only_matrix:
            sub = np.where(inside & (sub == 0), label, sub)
        else:
            sub = np.where(inside, label, sub)
        labels[ii] = sub

    if shell_label is not None:
        for c, r in zip(packing.centers, packing.radii):
            _paint(c, r + shell_thickness, shell_label, only_matrix=True)
    for c, r, p in zip(packing.centers, packing.radii, packing.phase_index):
        _paint(c, r, int(p), only_matrix=False)
    vf = {name: float(np.mean(labels == k)) for k, name in enumerate(names)}
    return VoxelMicrostructure(
        labels=labels,
        phase_names=tuple(names),
        voxel_size=h,
        volume_fractions=vf,
        description=f"{packing.n_particles} spheres, target phi={packing.target_volume_fraction:.3f}, n={n}",
        packing=packing,
    )


# ---------------------------------------------------------------------------
# Simple verification geometries
# ---------------------------------------------------------------------------
def laminate(n: int, fractions: Sequence[float], axis: int = 0) -> VoxelMicrostructure:
    """Layered microstructure: phase k occupies a slab of round(f_k n) voxels normal to ``axis``."""
    counts = [int(round(f * n)) for f in fractions]
    counts[-1] = n - sum(counts[:-1])
    if min(counts) < 0:
        raise ValueError("fractions incompatible with grid size")
    prof = np.concatenate([np.full(c, k, dtype=np.int32) for k, c in enumerate(counts)])
    shape = [1, 1, 1]
    shape[axis] = n
    labels = np.broadcast_to(prof.reshape(shape), (n, n, n)).copy()
    names = tuple(f"layer{k}" for k in range(len(fractions)))
    vf = {name: float(np.mean(labels == k)) for k, name in enumerate(names)}
    return VoxelMicrostructure(labels=labels, phase_names=names, voxel_size=1.0 / n, volume_fractions=vf,
                               description=f"laminate normal to axis {axis}, voxel counts {counts}")


def single_sphere(n: int, volume_fraction: float) -> VoxelMicrostructure:
    """One centred sphere in a unit periodic cube (simple-cubic array of spheres)."""
    r = (3.0 * volume_fraction / (4.0 * np.pi)) ** (1.0 / 3.0)
    if r >= 0.5:
        raise ValueError("volume fraction too high for a non-touching simple-cubic array")
    x = (np.arange(n) + 0.5) / n - 0.5
    r2 = x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2
    labels = (r2 <= r * r).astype(np.int32)
    names = (MATRIX_PHASE, "inclusion")
    vf = {name: float(np.mean(labels == k)) for k, name in enumerate(names)}
    return VoxelMicrostructure(labels=labels, phase_names=names, voxel_size=1.0 / n, volume_fractions=vf,
                               description=f"single sphere (SC array), target phi={volume_fraction}, n={n}")
