"""Render an indentation trajectory as a README-friendly GIF (plus a static PNG of the last frame).

Input: a :class:`fracture_md.io.indent_traj.Trajectory` (format ``indentviz-traj/1``), written by the harness MD
driver or converted from a LAMMPS dump. Output: a GIF readable by materials scientists who do not run MD.

Panels (plain-language titles, fixed colour limits across frames, no silent exaggeration):
  (a) side view of a thin slice through the indenter axis, atoms coloured by a per-atom quantity (local shear
      strain | bonds gained or lost | displacement | an engine column), the indenter drawn as an outline;
  (b) force on the tip versus depth, drawn progressively. The force is a centred moving average over
      ``load_window_ps`` (0.5 ps, the protocol's block length): single MD samples scatter by ~14 nN at 300 K;
  (c) top view of the surface height (dent and pile-up) with two outlines: the ESTIMATED contact (convex hull of
      the estimated-contact atoms, dilated by a/(2 sqrt 2): the E3 estimator of the indentation analysis,
      physics/indent_analysis.py) and the NOMINAL one (where the rigid tip would cut the original surface).
      The estimated-contact atoms are those of the analysis (``estimated_contact_mask``): the atoms touching the
      tip whose tip force is at least CONTACT_FORCE_FRACTION (1/2) of the mean over the touching atoms, when a
      per-atom tip force is known (an SDF tip: delta^2 from the positions or the frame writer's penetration
      field; an atomistic LAMMPS tip: the WCA force rebuilt from its tip dump, a column named by
      ``meta["tip_force_field"]``, see scripts/make_indent_figures.py); otherwise every touching atom (contact
      flag c_ctip only), and the render record says which rule each frame used.  It is called "estimated", never
      "true": no atomistic definition of contact is unique (critic item B1; the touching-atom hull alone
      over-counts an elastic Hertz contact by ~15-21 % in radius).

Guarantees (each has a unit test in tests/test_viz_indentation.py):
  * byte-deterministic output for identical input (no timestamps; deterministic palette);
  * colour limits fixed for the whole GIF; every colour-map step kept exactly by the GIF palette;
  * displacement exaggeration (``amplify != 1``) stamped on every frame; synthetic data stamped as such;
  * frames are subsampled evenly and never interpolated; the frames of maximum depth and maximum (windowed)
    load are always kept;
  * bonds use a potential-aware rule (``analysis.local_strain.bond_rule_for``): heteronuclear-only Si-C < 2.4 A
    for SiC, so compressed homonuclear second neighbours are not drawn as new bonds;
  * thermal mottling of the height map is capped: before contact < 5 % of the map bins fall outside the two
    colour steps next to zero (the positive colour range is widened until this holds);
  * layout "phone" gives a 720 px wide portrait frame with every text >= 8 pt.

Style follows scripts/make_figures.py: DejaVu Sans, muted ink colours, faint full curve + bold progressive curve,
one shared palette, no dithering, optimize=True, first and last frames held.
"""

from __future__ import annotations

import hashlib
import math
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from fracture_md.analysis import local_strain as F
from fracture_md.io.indent_traj import Indenter, Trajectory

# Palette slots copied from scripts/make_figures.py COL (so the GIF matches the README figures)
COL = {"ink": "#1f1f1d", "ink2": "#52514e", "muted": "#8a8984", "grid": "#e4e3df", "fail": "#e34948",
       "tip_fill": "#e9e8e4"}
SPECIES_RADIUS_A = {"Si": 0.80, "C": 0.55, "O": 0.60, "Zr": 0.90, "Ba": 1.0, "Ca": 0.9, "P": 0.7, "H": 0.3}
# Depth rates of laboratory nanoindentation span roughly 0.5-10 nm/s; the footer states the speed ratio as a range
LAB_TIP_SPEED_RANGE_M_S = (0.5e-9, 1.0e-8)
DEFAULT_LATTICE_A = 4.3705  # Vashishta 3C-SiC at 300 K: dilation a/(2 sqrt 2) = 1.545 A (protocol E3)
BOND_RULES = {"vashishta": F.VASHISHTA_SIC_BONDS, "sic_hetero_2.4": F.SIC_HETERONUCLEAR_2P4,
              "all_pairs_2.35": F.ALL_PAIRS_2P35, "all_pairs_2.5": F.LAMMPS_COORD_2P5}
# Plain-language name of the solid outline of panel (c) (protocol estimator E3; the analysis's name,
# physics.indent_analysis analyze_indentation()["contact"]["name"]). It is an ESTIMATE, not the "true" contact (no
# atomistic definition of contact is unique; critic item B1), so no frame or caption may call it the true contact.
CONTACT_OUTLINE_NAME = "estimated contact"
CONTACT_RULES = ("force_fraction", "touching")  # as physics.indent_analysis.CONTACT_RULES
RULE_NO_FORCE = "touching (no per-atom tip force)"


@dataclass
class RenderOptions:
    field: str = "shear"  # shear | bonds | displacement | col:<atom field>
    layout: str = "wide"  # wide: 900 x 520 px (README) | phone: 720 px wide portrait, all text >= 8 pt
    fps: float = 20.0
    seconds: float = 7.0  # playback time of the moving part (first/last holds add ~2 s)
    hold_first_s: float = 0.6
    hold_last_s: float = 1.6
    slab_thickness_A: float = 4.0
    view_halfwidth_A: float | None = None
    show_map: bool = True
    map_bin_A: float = 4.0
    amplify: float = 1.0  # displacement exaggeration of panel (a); != 1 is stamped on every frame
    vmax: float | None = None  # colour limit of panel (a) (shear: default 0.5; displacement: auto)
    bond_rule: str = "auto"  # auto (from meta potential / species) or a key of BOND_RULES
    strain_cutoff_A: float = F.SIC_STRAIN_CUTOFF_A
    load_window_ps: float = 0.5  # centred moving average of the load in panel (b); 0 = raw samples
    contact_dilation_A: float | None = None  # None: meta lattice_constant_A / (2 sqrt 2), default 1.545 A
    show_estimated_contact: bool = True  # solid outline of panel (c): the E3 contact estimate
    contact_rule: str = "force_fraction"  # estimated-contact atoms (CONTACT_RULES), as the analysis
    contact_force_fraction: float | None = None  # None: physics.indent_analysis.CONTACT_FORCE_FRACTION (1/2)
    mottling_cap: float = 0.05  # max share of pre-contact map bins outside the two colour steps next to zero
    keep_extrema: bool = True  # always keep the frames of maximum depth and maximum load
    title: str | None = None
    dpi: int = 100
    n_colors: int = 160
    n_levels: int = 10  # discrete colour steps of panel (a) for continuous fields
    antialias_atoms: bool = False  # off: crisp discs, far fewer changed pixels per frame (smaller GIF)
    max_frames: int | None = None  # cap (after fps * seconds), for quick previews


# figure geometry per layout (figure fractions) and font sizes (pt); phone keeps every text >= 8 pt
LAYOUTS = {
    "wide": dict(
        size_in=(9.0, 5.2), title_y=0.975, title_fs=13.0, title_wrap=90, sub_y=0.915, sub_fs=9.0, sub_wrap=140,
        axa=[0.015, 0.20, 0.47, 0.64], aspect_a=1.25, cax=[0.06, 0.135, 0.38, 0.022],
        axb=[0.585, 0.585, 0.395, 0.25], axb_nomap=[0.585, 0.22, 0.395, 0.6],
        axc=[0.585, 0.115, 0.2, 0.35], cax2=[0.79, 0.135, 0.011, 0.29], note_xy=(0.874, 0.445), note_inline=False,
        footer_y=0.012, footer_wrap=175,
        fs=dict(panel=9.5, label=8.5, tick=7.5, cbar_label=7.3, cbar_tick=7.0, map_tick=6.5, note=6.7, legend=7.5,
                legend_title=7.0, counter=8.5, tip=8.5, surface=7.0, scale=8.0, footer=7.2, status=7.0,
                stamp=11.0, synthetic=14.0, slice_label=6.3)),
    "phone": dict(
        size_in=(7.2, 9.6), title_y=0.99, title_fs=12.5, title_wrap=52, sub_y=0.922, sub_fs=9.0, sub_wrap=86,
        axa=[0.02, 0.475, 0.96, 0.37], aspect_a=1.85, cax=[0.08, 0.415, 0.84, 0.016],
        axb=[0.12, 0.13, 0.34, 0.2], axb_nomap=[0.12, 0.13, 0.8, 0.2],
        axc=[0.585, 0.13, 0.27, 0.2], cax2=[0.875, 0.13, 0.018, 0.2], note_xy=(0.5, 0.085), note_inline=True,
        footer_y=0.004, footer_wrap=100,
        fs=dict(panel=9.5, label=8.5, tick=8.0, cbar_label=8.0, cbar_tick=8.0, map_tick=8.0, note=8.0, legend=8.0,
                legend_title=8.0, counter=8.5, tip=8.5, surface=8.0, scale=8.0, footer=8.0, status=8.0,
                stamp=11.0, synthetic=14.0, slice_label=8.0)),
}

FIELD_LABEL = {
    "shear": "local shear strain (how much each atom's surroundings are distorted; 0 = undisturbed)",
    "bonds": "bonds gained or lost since the start ({rule})",
    "displacement": "distance each atom has moved since the start (nm)",
}
FIELD_SUBTITLE = {
    "shear": "Colours show how strongly the crystal is sheared around each atom; dark = heavily distorted.",
    "bonds": "Colours show atoms whose bonding changed: blue lost bonds, red gained bonds, grey unchanged.",
    "displacement": "Colours show how far each atom has been pushed from where it started.",
}


def _mpl(fs: dict | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = fs or LAYOUTS["wide"]["fs"]
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 100, "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 9.0, "axes.titlesize": fs["panel"], "axes.titleweight": "bold", "axes.labelsize": fs["label"],
        "axes.edgecolor": COL["muted"], "axes.labelcolor": COL["ink"], "text.color": COL["ink"],
        "xtick.color": COL["ink2"], "ytick.color": COL["ink2"], "xtick.labelsize": fs["tick"],
        "ytick.labelsize": fs["tick"], "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": COL["grid"], "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": False,
        "legend.fontsize": fs["legend"], "lines.linewidth": 2.0, "font.family": "DejaVu Sans", "path.simplify": False,
    })
    return plt


# ============================================================================================ helpers
def select_frames(n_total: int, fps: float, seconds: float, max_frames: int | None = None, keep=()):
    """Evenly spaced frame indices (never interpolated) and the per-frame delay in ms (10 ms steps).

    Every index in ``keep`` (e.g. the frames of maximum depth and maximum load) is always selected: it replaces
    the nearest evenly spaced index, so the frame count does not change. The first and last frames are always
    kept.
    """
    n_target = max(2, int(round(fps * seconds)))
    if max_frames is not None:
        n_target = min(n_target, max_frames)
    if n_total <= n_target:
        idx = list(range(n_total))
    else:
        idx = sorted(set(np.round(np.linspace(0, n_total - 1, n_target)).astype(int).tolist()))
        protected = {0, n_total - 1} | {int(k) for k in keep if 0 <= int(k) < n_total}
        for k in sorted({int(k) for k in keep if 0 <= int(k) < n_total}):
            if k in idx:
                continue
            cands = [i for i in idx if i not in protected]
            if cands:
                idx.remove(min(cands, key=lambda i: (abs(i - k), i)))
            idx = sorted(set(idx) | {k})
    idx = np.asarray(idx, int)
    delay = seconds * 1000.0 / max(len(idx), 1)
    delay = int(min(250, max(40, round(delay / 10.0) * 10)))  # GIF delays are centiseconds; >= 40 ms is safe
    return idx, delay


def windowed_mean(t: np.ndarray, y: np.ndarray, window: float) -> tuple[np.ndarray, np.ndarray]:
    """Centred moving average of y over |t' - t| <= window/2 (irregular sampling allowed).

    Returns (mean, number of samples in each window). ``window <= 0`` returns y unchanged.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if window <= 0 or len(t) < 2:
        return y.copy(), np.ones(len(y), int)
    c = np.r_[0.0, np.cumsum(y)]
    lo = np.searchsorted(t, t - 0.5 * window - 1e-9, side="left")
    hi = np.searchsorted(t, t + 0.5 * window + 1e-9, side="right")
    return (c[hi] - c[lo]) / (hi - lo), hi - lo


def _nice_ceil(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(x))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * e >= x * (1 - 1e-9):  # 5.000000000004 -> 5, not 6
            return m * e
    return 10 * e


def _nice_next(x: float) -> float:
    """The next 'nice' value strictly above x (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10 x 10^k)."""
    return _nice_ceil(x * 1.0001 + 1e-12)


def _times_words(x: float) -> str:
    """Round a large ratio to one significant figure in words ("5 billion")."""
    if not np.isfinite(x) or x <= 0:
        return "?"
    for scale, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if x >= scale:
            v = float(f"{x / scale:.1g}")
            return f"{v:g} {word}"
    return f"{float(f'{x:.1g}'):g}"


def rate_ratio_words(v_m_s: float) -> str:
    """'5–100 billion' for 50 m/s against laboratory depth rates of 0.5-10 nm/s."""
    lo = _times_words(v_m_s / LAB_TIP_SPEED_RANGE_M_S[1])
    hi = _times_words(v_m_s / LAB_TIP_SPEED_RANGE_M_S[0])
    wl, wh = lo.split(" ", 1), hi.split(" ", 1)
    if len(wl) == 2 and len(wh) == 2 and wl[1] == wh[1]:
        return f"{wl[0]}–{wh[0]} {wl[1]}"
    return f"{lo} to {hi}"


def contact_outline(xy: np.ndarray, dilate_A: float, n_arc: int = 48) -> tuple[np.ndarray, float]:
    """Estimated-contact outline (E3): convex hull of the projected contact atoms, dilated by ``dilate_A``.

    Returns (closed polygon (M, 2), exact area in A^2). The area is the Minkowski sum hull + disc,
    A = A_hull + P_hull d + pi d^2 (the E3 estimator of the indentation protocol); the polygon samples the
    rounded corners with ``n_arc`` directions. Empty input gives an empty polygon and area 0.
    """
    from scipy.spatial import ConvexHull, QhullError

    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) == 0:
        return np.empty((0, 2)), 0.0
    d = float(dilate_A)
    A = P = 0.0
    if len(xy) >= 3:
        try:
            h = ConvexHull(xy)
            A, P = float(h.volume), float(h.area)
        except QhullError:  # collinear points: a segment
            A, P = 0.0, 0.0
    if A == 0.0 and len(xy) >= 2:
        ext = xy - xy.mean(axis=0)
        _, _, vt = np.linalg.svd(ext, full_matrices=False)
        s = ext @ vt[0]
        P = 2.0 * float(s.max() - s.min())
    area = A + P * d + math.pi * d * d
    if d > 0:
        ang = np.linspace(0.0, 2 * np.pi, n_arc, endpoint=False)
        pts = (xy[:, None, :] + d * np.stack([np.cos(ang), np.sin(ang)], -1)[None]).reshape(-1, 2)
    else:
        pts = xy
    try:
        hv = ConvexHull(pts).vertices
        poly = pts[hv]
    except QhullError:
        poly = pts[:1]
    return np.vstack([poly, poly[:1]]), area


def contact_atoms(tr: Trajectory, t: int) -> np.ndarray:
    """Atoms touching the indenter at frame t (bool (N,)): an engine flag if the run has one, else sdf < 0.

    ``meta["contact_field"]`` names an atom field that is > 0 for contact atoms (LAMMPS ``c_ctip``: sample atoms
    with a tip atom inside the repulsive wall range); otherwise the atoms inside the SDF indenter (sdf < 0),
    which are exactly the atoms that feel the wall of a fix-indent style SDF tip.
    """
    fld = tr.meta.get("contact_field")
    if fld and fld in tr.atom_fields:
        return np.asarray(tr.atom_fields[fld][t]) > 0
    lo, hi = tr.box(t)
    d = F.min_image(tr.positions[t] - tr.tip_xyz[t], lo, hi, tr.pbc)
    near = np.abs(d[:, 2]) < max(float(tr.depth_A.max()), 0.0) + 5.0 + tr.indenter.radius_A
    out = np.zeros(tr.n_atoms, bool)
    out[near] = tr.indenter.sdf(d[near]) < 0.0
    return out


def tip_force_on_atoms(tr: Trajectory, t: int) -> np.ndarray | None:
    """Per-atom tip force at frame t, up to a constant factor ((N,) float), or None when it is unknown.

    ``meta["tip_force_field"]`` names an engine column with the per-atom tip force; else the frame writer's
    ``penetration_A`` field of an SDF tip (force K delta^2); else, for a run without an engine contact flag (the
    contact comes from the SDF), delta^2 with delta = max(0, -sdf) from the positions.  An atomistic tip with
    only a contact flag (LAMMPS c_ctip) has no per-atom force: None.
    """
    fld = tr.meta.get("tip_force_field")
    if fld and fld in tr.atom_fields:
        return np.asarray(tr.atom_fields[fld][t], float)
    if "penetration_A" in tr.atom_fields:
        return np.asarray(tr.atom_fields["penetration_A"][t], float) ** 2
    cf = tr.meta.get("contact_field")
    if cf and cf in tr.atom_fields:
        return None
    lo, hi = tr.box(t)
    d = F.min_image(tr.positions[t] - tr.tip_xyz[t], lo, hi, tr.pbc)
    near = np.abs(d[:, 2]) < max(float(tr.depth_A.max()), 0.0) + 5.0 + tr.indenter.radius_A
    f = np.zeros(tr.n_atoms)
    f[near] = np.clip(-tr.indenter.sdf(d[near]), 0.0, None) ** 2
    return f


def estimated_contact_atoms(tr: Trajectory, t: int, rule: str = "force_fraction",
                            fraction: float | None = None) -> tuple[np.ndarray, str]:
    """(estimated-contact atoms (N,) bool, rule used) at frame t, with the analysis's rule.

    ``force_fraction``: :func:`fracture_md.physics.indent_analysis.estimated_contact_mask` of the touching atoms
    (:func:`contact_atoms`) and the per-atom tip force (:func:`tip_force_on_atoms`); without a per-atom force
    every touching atom is kept (rule "touching (no per-atom tip force)").  ``touching``: every touching atom.
    """
    from fracture_md.physics.indent_analysis import CONTACT_FORCE_FRACTION, estimated_contact_mask

    if rule not in CONTACT_RULES:
        raise ValueError(f"contact rule must be one of {CONTACT_RULES}, got {rule!r}")
    touching = contact_atoms(tr, t)
    if rule == "touching":
        return touching, "touching"
    force = tip_force_on_atoms(tr, t)
    if force is None:
        return touching, RULE_NO_FORCE
    frac = CONTACT_FORCE_FRACTION if fraction is None else float(fraction)
    return estimated_contact_mask(touching, force, fraction=frac), "force_fraction"


def n_contact_atoms(tr: Trajectory, t: int) -> int:
    """Number of atoms touching the tip at frame t (from ``tr.contact_xy`` when present, else contact_atoms)."""
    xy = tr.contact_xy_at(t)
    return int(len(xy)) if xy is not None else int(contact_atoms(tr, t).sum())


def contact_source_label(tr: Trajectory) -> str:
    """Where the touching atoms of the estimated-contact outline come from (stated in the render record)."""
    if tr.contact_count is not None:
        return "contact_xy (every atom touching the tip, from an engine contact dump)" + (
            "; tip force: contact_force" if tr.contact_force is not None else "")
    if tr.meta.get("contact_field") in tr.atom_fields:
        src = f"atom field {tr.meta['contact_field']} > 0"
    else:
        src = "indenter sdf < 0"
    if tr.meta.get("tip_force_field") in tr.atom_fields:
        src += f"; tip force: atom field {tr.meta['tip_force_field']}"
    elif "penetration_A" in tr.atom_fields:
        src += "; tip force: penetration_A^2 (SDF wall)"
    elif tr.meta.get("contact_field") not in tr.atom_fields:
        src += "; tip force: (-sdf)^2 (SDF wall)"
    return src


def default_title(tr: Trajectory) -> str:
    mat = tr.meta.get("material_plain") or tr.meta.get("material") or "a crystal"
    shape = {"sphere": "a rigid ball-shaped tip", "cone": "a rigid cone-shaped tip",
             "berkovich": "a rigid three-sided pyramid tip (Berkovich)"}[tr.indenter.kind]
    return f"Pressing {shape} into {mat}"


def footer_text(tr: Trajectory, opts: RenderOptions, n_shown_ps: float, shown_s: float) -> str:
    """Plain-language footer: system size, model, temperature, time compression, tip speed, scale statement."""
    m = tr.meta
    parts = []
    n_tot = m.get("n_atoms_total", tr.n_atoms)
    sim = f"Computer simulation of {int(n_tot):,} atoms"
    if m.get("potential"):
        sim += f", {m['potential']} model"
    if m.get("temperature_K") is not None:
        sim += f", {float(m['temperature_K']):.0f} K"
    shown = f"{shown_s:.0f}" if shown_s >= 10 else f"{shown_s:.1f}".rstrip("0").rstrip(".")
    parts.append(sim + f"; {n_shown_ps:.0f} ps of simulated time shown in {shown} s.")
    v = tr.meta.get("tip_speed_m_s") or tr.tip_speed_m_s()
    if v and np.isfinite(v):
        parts.append(f"The tip moves at {float(v):.0f} m/s, roughly {rate_ratio_words(float(v))} times faster than "
                     f"in a laboratory test.")
    if opts.amplify == 1.0:
        parts.append("Atom positions are to scale (not exaggerated).")
    return " ".join(parts)


# ============================================================================================ renderer
class IndentRenderer:
    def __init__(self, tr: Trajectory, opts: RenderOptions | None = None):
        self.tr = tr.validate()
        self.o = opts or RenderOptions()
        o = self.o
        if o.field not in FIELD_LABEL and not o.field.startswith("col:"):
            raise ValueError(f"field must be one of {sorted(FIELD_LABEL)} or 'col:<atom field name>'")
        if o.field.startswith("col:") and o.field[4:] not in tr.atom_fields:
            raise ValueError(f"trajectory has no atom field {o.field[4:]!r} (has {sorted(tr.atom_fields)})")
        if o.amplify <= 0:
            raise ValueError("amplify must be > 0")
        if o.contact_rule not in CONTACT_RULES:
            raise ValueError(f"contact_rule must be one of {CONTACT_RULES}")
        if o.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {sorted(LAYOUTS)}")
        self.L = LAYOUTS[o.layout]
        if o.bond_rule == "auto":
            self.bond_rule = F.bond_rule_for(tr.meta.get("potential_id") or tr.meta.get("potential"), tr.species)
        elif o.bond_rule in BOND_RULES:
            self.bond_rule = BOND_RULES[o.bond_rule]
        else:
            raise ValueError(f"bond_rule must be 'auto' or one of {sorted(BOND_RULES)}")
        a = float(tr.meta.get("lattice_constant_A") or DEFAULT_LATTICE_A)
        self.dilation = float(o.contact_dilation_A if o.contact_dilation_A is not None else a / (2 * math.sqrt(2)))
        lo0, hi0 = tr.box(0)
        self.lo0, self.hi0 = lo0, hi0
        self.P0 = tr.positions[0]
        self.axis_xy = tr.tip_xyz[0, :2]
        self.surf_idx = F.top_layer_atoms(self.P0, tr.surface_z0_A)
        self._prepare_load()
        keep = []
        if o.keep_extrema:
            keep = [int(np.argmax(tr.depth_A)), int(np.argmax(self.frame_load))]
        self.keep = keep
        self.idx, self.delay_ms = select_frames(tr.n_frames, o.fps, o.seconds, o.max_frames, keep=keep)
        self._select_slab()
        self._prepare_field()
        self._prepare_limits()

    # -------------------------------------------------------------- data preparation
    def _prepare_load(self):
        """Windowed load: the dense curve averaged over load_window_ps; frame values interpolated from it."""
        tr = self.tr
        ct, cd, cl = tr.curve()
        wl, nwin = windowed_mean(ct, cl, self.o.load_window_ps)
        self.curve_t, self.curve_d, self.curve_l = np.asarray(ct, float), np.asarray(cd, float), wl
        self.window_samples = int(np.median(nwin))
        self.frame_load = np.interp(tr.time_ps, self.curve_t, self.curve_l)
        # tip velocity (A/ps) along the curve, for the plain-language phase label
        if len(ct) > 1:
            v = np.gradient(np.asarray(cd, float), np.asarray(ct, float))
            vw, _ = windowed_mean(ct, v, max(self.o.load_window_ps, 1e-9))
            moving = np.abs(vw) > 1e-6
            self.v_tol = 0.2 * float(np.median(np.abs(vw[moving]))) if moving.any() else 1e-6
            self.frame_v = np.interp(tr.time_ps, self.curve_t, vw)
        else:
            self.v_tol, self.frame_v = 1e-6, np.zeros(tr.n_frames)

    def phase(self, t: int) -> str:
        v, d = self.frame_v[t], self.tr.depth_A[t]
        if v > self.v_tol:
            return "approaching" if d < 0 else "pressing in"
        if v < -self.v_tol:
            return "pulling out"
        return "holding" if d > 0 else "at rest"

    def _footprint_extent_along_slice(self, depth: float) -> float:
        """Largest |s| along the slice direction (x) where the tip's lower surface is below ``depth``."""
        if depth <= 0:
            return 0.0
        ind = self.tr.indenter
        s = np.linspace(-8 * depth - 3 * ind.radius_A - 5, 8 * depth + 3 * ind.radius_A + 5, 801)
        h = ind.lower_surface_height(np.column_stack([s, 0 * s]), zmax=depth * 1.001 + 1e-6)
        ok = np.isfinite(h) & (h <= depth)
        return float(np.abs(s[ok]).max()) if ok.any() else 0.0

    def _select_slab(self):
        tr, o = self.tr, self.o
        rel0 = F.min_image(self.P0 - np.r_[self.axis_xy, 0.0], self.lo0, self.hi0, tr.pbc)
        dmax = float(np.max(tr.depth_A))
        s_ext = self._footprint_extent_along_slice(dmax)
        Lx = self.hi0[0] - self.lo0[0]
        W = o.view_halfwidth_A or min(0.5 * Lx, max(30.0, 1.3 * s_ext + 8.0, 3.0 * max(dmax, 1.0)))
        self.W = float(W)
        self.span_z = 2 * self.W / self.L["aspect_a"]
        self.z_top = 0.32 * self.span_z
        zmin_slab = float(self.P0[:, 2].min() - tr.surface_z0_A)
        self.z_bot = max(self.z_top - self.span_z, zmin_slab - 1.5)
        sel = (np.abs(rel0[:, 1]) <= 0.5 * o.slab_thickness_A) & (np.abs(rel0[:, 0]) <= self.W + 3.0)
        self.slab = np.flatnonzero(sel)
        # draw back to front: we look along +y, so larger y is further away
        self.slab = self.slab[np.argsort(-rel0[self.slab, 1], kind="stable")]
        sp = tr.species[self.slab]
        self.radius = np.array([SPECIES_RADIUS_A.get(str(s), 0.7) for s in sp])
        self.species_present = sorted(set(map(str, sp)), key=lambda s: -SPECIES_RADIUS_A.get(s, 0.7))

    def _prepare_field(self):
        tr, o = self.tr, self.o
        if o.field == "shear":
            self._shear = F.ShearStrain(self.P0, self.lo0, self.hi0, tr.pbc, cutoff=o.strain_cutoff_A,
                                        centers=self.slab)
        elif o.field == "bonds":
            self._cn0 = F.coordination_by_rule(self.P0, tr.species, self.lo0, self.hi0, tr.pbc, self.bond_rule,
                                               centers=self.slab)

    def field_values(self, t: int) -> np.ndarray:
        """Per-atom colour values of the slice atoms at frame t (aligned with ``self.slab``)."""
        tr, o = self.tr, self.o
        lo, hi = tr.box(t)
        P = tr.positions[t]
        if o.field == "shear":
            return self._shear(P, lo, hi)
        if o.field == "bonds":
            cn = F.coordination_by_rule(P, tr.species, lo, hi, tr.pbc, self.bond_rule, centers=self.slab)
            return (cn - self._cn0).astype(float)
        if o.field.startswith("col:"):
            return np.asarray(tr.atom_fields[o.field[4:]][t], float)[self.slab]
        return np.linalg.norm(F.displacement(P[self.slab], self.P0[self.slab], lo, hi, tr.pbc), axis=1) / 10.0  # nm

    def slab_xz(self, t: int) -> np.ndarray:
        """Panel-(a) coordinates (s, z - z0) of the slab atoms at frame t, with optional amplification."""
        tr = self.tr
        lo, hi = tr.box(t)
        u = F.displacement(tr.positions[t][self.slab], self.P0[self.slab], lo, hi, tr.pbc)
        base = F.min_image(self.P0[self.slab] - np.r_[self.axis_xy, 0.0], self.lo0, self.hi0, tr.pbc)
        p = base + self.o.amplify * u
        return np.column_stack([p[:, 0], p[:, 2] - tr.surface_z0_A])

    def precontact_frames(self) -> list[int]:
        """Frames before the first contact (tip above the surface and no atom touching it)."""
        out = []
        for t in range(self.tr.n_frames):
            if self.tr.depth_A[t] >= 0 or n_contact_atoms(self.tr, t) > 0:
                break
            out.append(t)
        return out

    def _map_steps(self, h_min: float, h_max: float) -> tuple[float, float]:
        """Width (A) of the colour steps next to zero: negative side (8 steps), positive side (4 steps)."""
        return abs(h_min) / 8.0, h_max / 4.0

    def mottling_fraction(self, frames, h_min: float, h_max: float) -> float:
        """Share of height-map bins outside the two colour steps next to zero, averaged over ``frames``."""
        sn, sp = self._map_steps(h_min, h_max)
        fr = []
        for t in frames:
            H, _, _ = self.height_map(t)
            H = H[np.isfinite(H)]
            fr.append(float(np.mean((H < -sn) | (H >= sp))) if H.size else 0.0)
        return float(np.mean(fr)) if fr else float("nan")

    def _prepare_limits(self):
        tr, o = self.tr, self.o
        if o.field == "shear":
            self.vmin, self.vmax = 0.0, float(o.vmax or 0.5)
        elif o.field == "bonds":
            self.vmin, self.vmax = -2.0, 2.0
        else:
            vals = None
            if o.vmax:
                self.vmax = float(o.vmax)
            else:
                vals = np.concatenate([self.field_values(t) for t in self.idx[:: max(1, len(self.idx) // 12)]]
                                      + [self.field_values(self.idx[-1])])
                self.vmax = _nice_ceil(float(np.nanpercentile(vals, 99.5)))
            self.vmin = 0.0
            if o.field.startswith("col:"):
                if vals is None:
                    vals = self.field_values(self.idx[-1])
                self.vmin = float(np.floor(np.nanpercentile(vals, 0.5)))
                if self.vmax <= self.vmin:
                    self.vmax = self.vmin + 1.0
        # height-map limits (fixed over all frames)
        self.mottling = None
        self.pre_frames = []
        if o.show_map:
            depth_max = max(float(np.max(tr.depth_A)), 0.5)
            pile = 0.0
            for t in np.unique(np.r_[self.idx[:: max(1, len(self.idx) // 12)], self.idx[-1]]):
                H, _, _ = self.height_map(t)
                pile = max(pile, float(np.nanpercentile(H, 99.5)))
            self.h_min = -_nice_ceil(depth_max)
            # floor of 1 A: thermal roughness of the top layer (~0.1-0.3 A per bin at 300 K) stays near-white
            h_max = max(_nice_ceil(max(pile, 0.25)), 1.0)
            # mottling cap: widen the positive range until, before contact, < mottling_cap of the bins fall
            # outside the two colour steps next to zero (thermal texture must not read as topography)
            self.pre_frames = self.precontact_frames()
            if self.pre_frames:
                frac = self.mottling_fraction(self.pre_frames, self.h_min, h_max)
                while frac >= o.mottling_cap and h_max < abs(self.h_min):
                    h_max = _nice_next(h_max)
                    frac = self.mottling_fraction(self.pre_frames, self.h_min, h_max)
                self.mottling = frac
            self.h_max = h_max
            self.map_stretch = abs(self.h_min) / self.h_max

    def height_map(self, t: int):
        """Top view: mean height of the atoms that formed the top layer at frame 0 (tracked), per bin."""
        tr = self.tr
        lo, hi = tr.box(t)
        if len(self.surf_idx) >= 10:
            return F.surface_height_map(tr.positions[t][self.surf_idx], lo, hi, tr.pbc, tr.surface_z0_A,
                                        bin_A=self.o.map_bin_A, reduce="mean")
        return F.surface_height_map(tr.positions[t], lo, hi, tr.pbc, tr.surface_z0_A, bin_A=self.o.map_bin_A)

    def estimated_contact(self, t: int) -> tuple[np.ndarray, float, int]:
        """(outline polygon relative to the axis (A), E3 area (A^2), number of estimated-contact atoms) at frame t.

        The rule used is kept in ``self.last_contact_rule`` (module docstring, panel (c))."""
        tr = self.tr
        lo, hi = tr.box(t)
        xy = tr.contact_xy_at(t)
        if xy is not None:  # every touching atom, from an engine contact dump (the positions may be a slice)
            frc = tr.contact_force_at(t)
            if self.o.contact_rule == "touching":
                self.last_contact_rule = "touching"
            elif frc is None:
                self.last_contact_rule = RULE_NO_FORCE
            else:
                from fracture_md.physics.indent_analysis import estimated_contact_mask

                self.last_contact_rule = "force_fraction"
                xy = xy[estimated_contact_mask(np.ones(len(xy), bool), frc, fraction=self._force_fraction())]
            if len(xy) == 0:
                return np.empty((0, 2)), 0.0, 0
            d = np.column_stack([xy - self.axis_xy, np.zeros(len(xy))])
            d = F.min_image(d, lo, hi, tr.pbc)
            poly, area = contact_outline(d[:, :2], self.dilation)
            return poly, area, int(len(xy))
        m, self.last_contact_rule = estimated_contact_atoms(tr, t, self.o.contact_rule, self.o.contact_force_fraction)
        if not m.any():
            return np.empty((0, 2)), 0.0, 0
        d = F.min_image(tr.positions[t][m] - np.r_[self.axis_xy, 0.0], lo, hi, tr.pbc)
        poly, area = contact_outline(d[:, :2], self.dilation)
        return poly, area, int(m.sum())

    def _force_fraction(self) -> float:
        from fracture_md.physics.indent_analysis import CONTACT_FORCE_FRACTION

        f = self.o.contact_force_fraction
        return float(CONTACT_FORCE_FRACTION if f is None else f)

    # -------------------------------------------------------------- colour maps
    def _cmap(self):
        """Colour map + norm of panel (a). Continuous fields use n_levels discrete steps of a perceptually
        uniform map (magma, reversed so 0 = light): steps read better than a gradient for a lay audience and
        stop thermal jitter from re-colouring every atom in every frame (smaller GIF)."""
        import matplotlib as mpl
        from matplotlib.colors import BoundaryNorm, ListedColormap

        if self.o.field == "bonds":
            cols = ["#2065ab", "#90c4dd", "#cfcdc8", "#f3a481", "#b1182b"]  # RdBu_r slots, grey centre
            cm = ListedColormap(cols).with_extremes(under=cols[0], over=cols[-1], bad="#b9b8b2")
            norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], cm.N)
            return cm, norm
        n = self.o.n_levels
        base = mpl.colormaps["magma"]
        cm = ListedColormap(base(np.linspace(0.95, 0.12, n)), name="magma_r_steps").with_extremes(
            over=base(0.02), under=base(0.95), bad="#b9b8b2")  # light = 0, dark = high
        return cm, BoundaryNorm(np.linspace(self.vmin, self.vmax, n + 1), n)

    def _height_cmap(self):
        """Stepped diverging map for panel (c), in nm: 8 equal steps from the deepest dent to 0 and 4 equal steps
        from 0 to the pile-up limit (the positive side is stretched by map_stretch, which is printed on the
        frame). Colours are RdBu_r sampled at the step centres as a two-slope norm, except that the two steps
        next to zero are pulled towards white (a near-neutral band): 300 K roughness of the top layer lives in
        those two steps (the mottling cap guarantees it) and must read as flat, not as a blue/red texture."""
        import matplotlib as mpl
        from matplotlib.colors import BoundaryNorm, ListedColormap

        lo, hi = self.h_min / 10.0, self.h_max / 10.0
        b = np.r_[np.linspace(lo, 0.0, 9), np.linspace(0.0, hi, 5)[1:]]
        c = 0.5 * (b[:-1] + b[1:])
        pos = np.where(c < 0, 0.5 + 0.5 * c / abs(lo), 0.5 + 0.5 * c / hi)
        pos[7:9] = 0.5 + 0.35 * (pos[7:9] - 0.5)  # steps 7 (just below 0) and 8 (just above 0)
        base = mpl.colormaps["RdBu_r"]
        cm = ListedColormap(base(pos), name="height_steps").with_extremes(under=base(0.0), over=base(1.0),
                                                                          bad="#b9b8b2")
        return cm, BoundaryNorm(b, cm.N)

    def _key_colors(self) -> list[tuple[int, int, int]]:
        """Colours that must be exact in the GIF palette (colour-map steps, extremes, ink)."""
        from matplotlib.colors import to_rgb

        cols = [to_rgb("white")] + [to_rgb(v) for v in COL.values()]
        cms = [self.cm] + ([self._map_cmap] if self.o.show_map else [])
        for cm in cms:
            cols += [tuple(c[:3]) for c in cm(np.arange(cm.N))]
            cols += [tuple(cm.get_over()[:3]), tuple(cm.get_under()[:3]), tuple(cm.get_bad()[:3])]
        out = []
        for c in cols:
            t = tuple(int(round(255 * v)) for v in c)
            if t not in out:
                out.append(t)
        return out

    # -------------------------------------------------------------- figure
    def build(self):
        tr, o, L = self.tr, self.o, self.L
        fs = L["fs"]
        plt = _mpl(fs)
        import matplotlib.patches as mpatches

        fig = plt.figure(figsize=L["size_in"], dpi=o.dpi)
        self.fig = fig
        title = o.title or tr.meta.get("title") or default_title(tr)
        fig.text(0.5, L["title_y"], textwrap.fill(title, L["title_wrap"]), ha="center", va="top",
                 fontsize=L["title_fs"], weight="bold", linespacing=1.15)
        sub = FIELD_SUBTITLE.get(o.field, f"Colours show the per-atom quantity '{o.field[4:]}' from the simulation.")
        fig.text(0.5, L["sub_y"], textwrap.fill(sub, L["sub_wrap"]), ha="center", va="top", fontsize=L["sub_fs"],
                 color=COL["ink2"])

        # ---------------- (a) slice
        axa = fig.add_axes(L["axa"])
        self.axa = axa
        axa.set_xlim(-self.W, self.W)
        axa.set_ylim(self.z_bot, self.z_top)
        axa.set_aspect("equal", adjustable="box")
        axa.axis("off")
        axa.set_title("(a) Slice through the tip, seen from the side", loc="left", fontsize=fs["panel"], pad=2)
        fig.canvas.draw()
        bb = axa.get_window_extent()
        px_per_A = bb.width / (2 * self.W)
        s_pts = (2 * self.radius * px_per_A * 72.0 / o.dpi) ** 2
        cm, norm = self._cmap()
        self.cm, self.norm = cm, norm
        xz = self.slab_xz(0)
        self.sc = axa.scatter(xz[:, 0], xz[:, 1], s=s_pts, c=np.zeros(len(xz)), cmap=cm, norm=norm,
                              linewidths=0, zorder=2, antialiased=o.antialias_atoms)
        axa.axhline(0.0, color=COL["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
        axa.text(-self.W + 0.5, 0.6, "original surface", fontsize=fs["surface"], color=COL["ink2"], va="bottom",
                 zorder=5)
        # indenter outline (translated each frame)
        ind = tr.indenter
        if ind.kind == "sphere":
            self.tip_patch = mpatches.Circle((0, 0), ind.radius_A, fc=COL["tip_fill"], ec=COL["ink"], lw=1.6,
                                             alpha=0.85, zorder=4)
        else:
            # true section of the (azimuth-rotated) pyramid or cone by the x-z plane through its axis
            poly = ind.section_outline((1.0, 0.0), half_width_A=self.W + 5, height_A=self.span_z * 1.5)
            self.tip_patch = mpatches.Polygon(poly, closed=True, fc=COL["tip_fill"], ec=COL["ink"], lw=1.6,
                                              alpha=0.85, zorder=4)
            self._tip_poly0 = poly
        axa.add_patch(self.tip_patch)
        self.tip_label = axa.text(0, 0, "tip", ha="center", va="center", fontsize=fs["tip"], color=COL["ink"],
                                  zorder=5, weight="bold", clip_on=True)
        # scale bar 1 nm
        x0 = self.W - 14.0
        y0 = self.z_bot + 0.05 * self.span_z
        axa.plot([x0, x0 + 10.0], [y0, y0], color=COL["ink"], lw=3, solid_capstyle="butt", zorder=5)
        axa.text(x0 + 5.0, y0 + 0.02 * self.span_z, "1 nm", ha="center", va="bottom", fontsize=fs["scale"], zorder=5)
        # species legend (size encodes species)
        handles = [plt.scatter([], [], s=(2 * SPECIES_RADIUS_A.get(s, 0.7) * 3.2) ** 2, c=COL["muted"], label=s)
                   for s in self.species_present]
        axa.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.005), ncol=len(handles),
                   fontsize=fs["legend"], handletextpad=0.1, columnspacing=0.8, title="atom size:",
                   title_fontsize=fs["legend_title"], alignment="left")
        self.counter = axa.text(0.01, 0.985, "", transform=axa.transAxes, ha="left", va="top", fontsize=fs["counter"],
                                color=COL["ink"], zorder=6, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        if o.amplify != 1.0:
            axa.text(0.5, 0.5, f"DISPLACEMENTS EXAGGERATED ×{o.amplify:g}", transform=axa.transAxes, ha="center",
                     va="center", fontsize=fs["stamp"], color=COL["fail"], weight="bold", alpha=0.9, zorder=7)
        if tr.meta.get("synthetic"):
            fig.text(0.26 if o.layout == "wide" else 0.5, 0.5 if o.layout == "wide" else 0.66,
                     "SYNTHETIC TEST DATA\n(not a simulation)", ha="center", va="center", fontsize=fs["synthetic"],
                     color=COL["fail"], weight="bold", alpha=0.55, rotation=20, zorder=10)
        # colour bar
        cax = fig.add_axes(L["cax"])
        cb = fig.colorbar(self.sc, cax=cax, orientation="horizontal",
                          extend="neither" if o.field == "bonds" else "max")
        cb.outline.set_linewidth(0.5)
        if o.field == "bonds":
            cb.set_ticks([-2, -1, 0, 1, 2])
            cb.set_ticklabels(["lost 2+", "lost 1", "same", "gained 1", "gained 2+"])
        lab = FIELD_LABEL[o.field].format(rule=self.bond_rule.label()) if o.field in FIELD_LABEL else \
            f"per-atom value '{o.field[4:]}' written by the simulation"
        cb.set_label(textwrap.fill(lab, 95 if o.layout == "wide" else 90), fontsize=fs["cbar_label"], labelpad=1.5)
        cb.ax.tick_params(labelsize=fs["cbar_tick"])
        cax.grid(False)

        # ---------------- (b) load-depth (windowed load)
        axb = fig.add_axes(L["axb"] if o.show_map else L["axb_nomap"])
        self.axb = axb
        cd_nm = self.curve_d / 10.0
        use_uN = np.nanmax(np.abs(self.curve_l)) > 5000
        self._lscale = 1e-3 if use_uN else 1.0
        axb.plot(cd_nm, self.curve_l * self._lscale, color=COL["muted"], lw=1.0, alpha=0.35)
        self.lb, = axb.plot([], [], color=COL["ink"], lw=1.8)
        self.pb, = axb.plot([], [], "o", color=COL["ink"], ms=7.5, mec="white", mew=1.2, zorder=5)
        axb.set_xlabel("depth of the tip below the original surface (nm)" if o.layout == "wide" else
                       "tip depth below the surface (nm)", labelpad=1)
        unit = "µN" if use_uN else "nN"
        axb.set_ylabel(f"force on tip ({unit})")
        axb.set_title("(b) How strongly the crystal pushes back" if o.layout == "wide" else "(b) Force on the tip",
                      loc="left", fontsize=fs["panel"], pad=3)
        dpad = 0.04 * (np.nanmax(cd_nm) - np.nanmin(cd_nm) + 1e-9)
        axb.set_xlim(np.nanmin(cd_nm) - dpad, np.nanmax(cd_nm) + dpad)
        lmin, lmax = np.nanmin(self.curve_l * self._lscale), np.nanmax(self.curve_l * self._lscale)
        axb.set_ylim(min(0.0, lmin) - 0.05 * (lmax - lmin + 1e-9), lmax + 0.08 * (lmax - lmin + 1e-9))
        axb.axvline(0.0, color=COL["muted"], lw=0.8, ls=(0, (4, 3)))
        if o.load_window_ps > 0:
            wtxt = f"averaged over {o.load_window_ps:g} ps"
            axb.text(0.99, 0.02, wtxt, transform=axb.transAxes, ha="right", va="bottom", fontsize=fs["note"],
                     color=COL["ink2"])
        if tr.meta.get("status_note"):
            axb.text(0.02, 0.97, textwrap.fill(str(tr.meta["status_note"]), 34 if o.layout == "wide" else 26),
                     transform=axb.transAxes, ha="left", va="top", fontsize=fs["status"], color=COL["ink2"],
                     linespacing=1.15, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0), zorder=6)

        # ---------------- (c) height map with the nominal and estimated contact outlines
        if o.show_map:
            axc = fig.add_axes(L["axc"])
            self.axc = axc
            H, xe, ye = self.height_map(0)
            ext = [xe[0] - self.axis_xy[0], xe[-1] - self.axis_xy[0], ye[0] - self.axis_xy[1], ye[-1] - self.axis_xy[1]]
            ext = [v / 10.0 for v in ext]
            self._map_cmap, self._map_norm = self._height_cmap()
            self.im = axc.imshow(H / 10.0, origin="lower", extent=ext, cmap=self._map_cmap, norm=self._map_norm,
                                 interpolation="nearest")
            axc.set_aspect("equal")
            axc.grid(False)
            axc.set_xlabel("nm", labelpad=0)
            axc.tick_params(labelsize=fs["map_tick"], pad=1)
            axc.set_title("(c) Surface from above", loc="left", fontsize=fs["panel"], pad=3)
            axc.axhline(0.0, color=COL["ink"], lw=0.7, ls=(0, (2, 2)))
            axc.text(ext[0] + 0.03 * (ext[1] - ext[0]), 0.0, "slice (a)", fontsize=fs["slice_label"], color=COL["ink"],
                     va="bottom", ha="left", bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
            self.fp_line, = axc.plot([], [], color=COL["ink2"], lw=1.0, ls=(0, (3, 2)))
            self.tc_line, = axc.plot([], [], color=COL["ink"], lw=1.4)
            cax2 = fig.add_axes(L["cax2"])
            cb2 = fig.colorbar(self.im, cax=cax2, orientation="vertical")
            cb2.outline.set_linewidth(0.5)
            cb2.ax.tick_params(labelsize=fs["map_tick"])
            lo_nm, hi_nm = self.h_min / 10.0, self.h_max / 10.0
            cb2.set_ticks([lo_nm, lo_nm / 2, 0.0, hi_nm])
            cb2.set_ticklabels([f"{v:g}" for v in (round(lo_nm, 3), round(lo_nm / 2, 3), 0.0, round(hi_nm, 3))])
            cb2.set_label("height (nm)", fontsize=fs["cbar_label"], labelpad=2)
            cax2.grid(False)
            stretch = f"colours above 0 are stretched ×{self.map_stretch:.0f} so small bumps are visible" \
                if self.map_stretch > 1.5 else ""
            if L["note_inline"]:
                note = (f"blue: dented, red: pushed up. Solid line: {CONTACT_OUTLINE_NAME}, drawn around the atoms "
                        f"the tip presses on; dashed: where the tip would cut the original surface.")
                if stretch:
                    note += " Red " + stretch.replace("colours above 0 are", "is") + "."
                fig.text(L["note_xy"][0], L["note_xy"][1], textwrap.fill(note, 96), fontsize=fs["note"],
                         color=COL["ink2"], va="top", ha="center", linespacing=1.15)
            else:
                note = (f"blue: dented\nred: pushed up\n\nsolid line:\n{CONTACT_OUTLINE_NAME},\naround the atoms\n"
                        "the tip presses on\n\ndashed: where the\ntip would cut the\noriginal surface")
                if stretch:
                    note += f"\n\ncolours above 0\nare stretched ×{self.map_stretch:.0f}\nso small bumps\nare visible"
                fig.text(L["note_xy"][0], L["note_xy"][1], note, fontsize=fs["note"], color=COL["ink2"], va="top",
                         ha="left", linespacing=1.15)

        n_shown_ps = float(tr.time_ps[self.idx[-1]] - tr.time_ps[self.idx[0]])
        shown_s = len(self.idx) * self.delay_ms / 1000.0
        fig.text(0.5, L["footer_y"], textwrap.fill(footer_text(tr, o, n_shown_ps, shown_s), L["footer_wrap"]),
                 ha="center", va="bottom", fontsize=fs["footer"], color=COL["ink2"], linespacing=1.2)
        return fig

    def _footprint(self, depth: float):
        """Nominal contact outline at the original surface (A, relative to the axis); cached per 0.01 A."""
        key = round(float(depth), 2)
        cache = self.__dict__.setdefault("_fp_cache", {})
        if key not in cache:
            cache[key] = self._footprint_uncached(key)
        return cache[key]

    def _footprint_uncached(self, depth: float):
        ind = self.tr.indenter
        if depth <= 0:
            return np.empty((0, 2))
        if ind.kind == "berkovich" and ind.radius_A == 0:
            r_in = depth * math.tan(math.radians(ind.angle_deg))
            u = ind._face_dirs()
            # vertices: intersection of adjacent face traces p.u_i = r_in, p.u_j = r_in
            verts = []
            for i, j in ((0, 1), (1, 2), (2, 0)):
                verts.append(np.linalg.solve(np.vstack([u[i], u[j]]), [r_in, r_in]))
            verts.append(verts[0])
            return np.asarray(verts)
        ang = np.linspace(0, 2 * np.pi, 121)
        if ind.kind == "sphere":
            R = ind.radius_A
            a = math.sqrt(max(0.0, 2 * R * min(depth, R) - min(depth, R) ** 2))
            return np.column_stack([a * np.cos(ang), a * np.sin(ang)])
        # rounded pyramid or cone: largest radius per direction where the lower surface is below depth
        rmax = 12 * depth + 3 * ind.radius_A + 5
        rr = np.linspace(0, rmax, 300)
        dxy = np.stack([np.outer(np.cos(ang), rr), np.outer(np.sin(ang), rr)], axis=-1)  # (121, 300, 2)
        h = ind.lower_surface_height(dxy, zmax=depth * 1.001 + 1e-6)
        ok = np.isfinite(h) & (h <= depth)
        r = np.where(ok.any(axis=1), np.max(np.where(ok, rr[None, :], 0.0), axis=1), 0.0)
        return np.column_stack([r * np.cos(ang), r * np.sin(ang)])

    def update(self, t: int):
        tr, o = self.tr, self.o
        self._last_contact = (0.0, 0)
        xz = self.slab_xz(t)
        self.sc.set_offsets(xz)
        self.sc.set_array(self.field_values(t))
        tz = tr.tip_xyz[t][2] - tr.surface_z0_A
        if tr.indenter.kind == "sphere":
            self.tip_patch.center = (0.0, tz + tr.indenter.radius_A)
            self.tip_label.set_position((0.0, min(tz + tr.indenter.radius_A, 0.5 * (tz + self.z_top))))
        else:
            self.tip_patch.set_xy(self._tip_poly0 + np.array([0.0, tz]))
            self.tip_label.set_position((0.0, 0.5 * (tz + self.z_top)))
        dnm = tr.depth_A[t] / 10.0
        where = f"depth {dnm:4.2f} nm" if dnm >= 0 else f"{-dnm:4.2f} nm above surface"
        self.counter.set_text(f"time {tr.time_ps[t]:.1f} ps · {where} · {self.phase(t)}")
        # (b): windowed load up to this frame's time
        k = self.curve_t <= tr.time_ps[t] + 1e-9
        d_now, l_now = tr.depth_A[t] / 10.0, self.frame_load[t]
        self.lb.set_data(np.r_[self.curve_d[k] / 10.0, d_now], np.r_[self.curve_l[k], l_now] * self._lscale)
        self.pb.set_data([d_now], [l_now * self._lscale])
        # (c)
        if o.show_map:
            H, _, _ = self.height_map(t)
            self.im.set_data(H / 10.0)
            fp = self._footprint(float(tr.depth_A[t]))
            self.fp_line.set_data(fp[:, 0] / 10.0 if len(fp) else [], fp[:, 1] / 10.0 if len(fp) else [])
            if o.show_estimated_contact:
                poly, area, n = self.estimated_contact(t)
                self._last_contact = (area, n)
                self.tc_line.set_data(poly[:, 0] / 10.0 if len(poly) else [], poly[:, 1] / 10.0 if len(poly) else [])

    # -------------------------------------------------------------- output
    def frame_image(self, t: int):
        from PIL import Image

        self.update(t)
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        return Image.frombuffer("RGBA", (w, h), self.fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")

    def render(self, out_gif: str | Path, png_last: str | Path | None = "auto") -> dict:
        from PIL import Image

        out_gif = Path(out_gif)
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        self.build()
        idx = [int(i) for i in self.idx]
        rgb, contact, rules = [], [], []
        for t in idx:
            self.last_contact_rule = None
            rgb.append(self.frame_image(t))
            contact.append(getattr(self, "_last_contact", (0.0, 0)))
            rules.append(self.last_contact_rule)
        # one shared palette: the exact colour-map colours + a deterministic MEDIANCUT of a montage of 8 evenly
        # spaced frames and the frames of maximum depth and load (a palette from too few frames maps unseen
        # colours, e.g. the deep-blue dent at maximum depth, onto wrong hues)
        pos = {t: i for i, t in enumerate(idx)}
        sample = sorted(set([idx[int(k)] for k in np.linspace(0, len(idx) - 1, 8).round()]
                            + [t for t in self.keep if t in pos]))
        w, h = rgb[0].size
        mont = Image.new("RGB", (w, h * len(sample)), "white")
        for i, t in enumerate(sample):
            mont.paste(rgb[pos[t]], (0, i * h))
        key = self._key_colors()
        n_free = max(16, self.o.n_colors - len(key))
        mc = mont.quantize(colors=n_free, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        mcp = mc.getpalette()[: 3 * n_free]
        extra = [tuple(mcp[i:i + 3]) for i in range(0, len(mcp), 3)]
        colors = key + [c for c in extra if c not in key]
        colors = colors[:255]  # keep one index free for Pillow's transparency optimisation
        pal = Image.new("P", (1, 1))
        pal.putpalette([v for c in colors for v in c])
        self.palette_size = len(colors)
        frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in rgb]
        dur = [self.delay_ms] * len(frames)
        dur[0] = max(dur[0], int(round(self.o.hold_first_s * 1000 / 10)) * 10)
        dur[-1] = max(dur[-1], int(round(self.o.hold_last_s * 1000 / 10)) * 10)
        frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True,
                       disposal=1)
        png_path = None
        if png_last:
            png_path = out_gif.with_name(out_gif.stem + "_last_frame.png") if png_last == "auto" else Path(png_last)
            rgb[-1].save(png_path, optimize=True)
        import matplotlib.pyplot as plt

        plt.close(self.fig)
        data = out_gif.read_bytes()
        tr = self.tr
        i_dmax = int(np.argmax(tr.depth_A))
        return dict(
            gif=str(out_gif), png=str(png_path) if png_path else None, bytes=len(data), MB=len(data) / 1e6,
            sha256=hashlib.sha256(data).hexdigest(), png_sha256=hashlib.sha256(Path(png_path).read_bytes()).hexdigest()
            if png_path else None, n_frames=len(frames), frame_indices=idx, delay_ms=self.delay_ms,
            fps_effective=1000.0 / self.delay_ms, durations_ms=dur, total_s=sum(dur) / 1000.0,
            size_px=list(frames[0].size), layout=self.o.layout, field=self.o.field,
            field_limits=[self.vmin, self.vmax],
            map_limits_A=[getattr(self, "h_min", None), getattr(self, "h_max", None)],
            map_stretch=getattr(self, "map_stretch", None), amplify=self.o.amplify,
            bond_rule=self.bond_rule.to_dict(),
            kept_frames={"argmax_depth": self.keep[0] if self.keep else None,
                         "argmax_windowed_load": self.keep[1] if self.keep else None,
                         "argmax_depth_value_A": float(tr.depth_A[i_dmax]),
                         "max_windowed_load_nN": float(np.max(self.frame_load)),
                         "max_curve_windowed_load_nN": float(np.max(self.curve_l))},
            load_window_ps=self.o.load_window_ps, load_samples_per_window_median=self.window_samples,
            mottling_fraction_precontact=self.mottling, mottling_cap=self.o.mottling_cap,
            precontact_frames=self.pre_frames,
            contact_source=contact_source_label(tr),
            contact_dilation_A=self.dilation,
            contact_outline_name=CONTACT_OUTLINE_NAME,
            contact_rule=self.o.contact_rule,
            contact_force_fraction=self._force_fraction(),
            contact_rules_used=sorted({r for r in rules if r is not None}),
            estimated_contact_area_E3_A2=[round(float(a), 2) for a, _ in contact],
            n_contact_atoms=[int(n) for _, n in contact],
            n_slab_atoms=int(len(self.slab)), palette_size=self.palette_size, palette_sample_frames=sample,
            view_halfwidth_A=self.W, options=asdict(self.o),
        )


def render_gif(tr: Trajectory, out_gif, opts: RenderOptions | None = None, png_last="auto") -> dict:
    """Render ``tr`` to ``out_gif`` (+ ``<stem>_last_frame.png``); returns the render record (sha256, frames, ...)."""
    return IndentRenderer(tr, opts).render(out_gif, png_last=png_last)


# ============================================================================================ synthetic data
def zincblende_slab(a0=4.36, nx=6, ny=6, nz=4):
    """A (001) zincblende slab (Si + C), periodic box in x and y, 2 A of vacuum margin in z (tests only)."""
    basis_si = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    basis_c = basis_si + 0.25
    cells = np.array([[i, j, k] for i in range(nx) for j in range(ny) for k in range(nz)], float)
    pos = []
    spec = []
    for b, s in ((basis_si, "Si"), (basis_c, "C")):
        p = (cells[:, None, :] + b[None, :, :]).reshape(-1, 3) * a0
        pos.append(p)
        spec += [s] * len(p)
    pos = np.vstack(pos)
    spec = np.array(spec)
    order = np.lexsort((pos[:, 0], pos[:, 1], pos[:, 2]))
    lo = np.array([0.0, 0.0, -2.0])
    hi = np.array([nx * a0, ny * a0, nz * a0 + 2.0])
    return pos[order], spec[order], lo, hi


def make_synthetic(a0=4.36, nx=6, ny=6, nz=4, radius_A=8.0, depth_max_A=4.0, n_frames=30, speed_A_per_ps=0.5,
                   noise_A=0.03, seed=0, kind="sphere", n_curve_per_frame=0) -> Trajectory:
    """Synthetic (NOT physical) indentation trajectory for unit tests and layout previews.

    A zincblende (001) slab is dented by a rigid tip that moves down and back up; atoms follow a smooth made-up
    displacement field (pushed out of the sphere) plus Gaussian noise, and the load is a Hertz-like formula.
    ``meta["synthetic"] = True`` makes the renderer stamp "SYNTHETIC TEST DATA" on every frame. With
    ``n_curve_per_frame > 0`` a dense, noisy load record is added (for the windowed-load tests).
    """
    pos0, spec, lo, hi = zincblende_slab(a0, nx, ny, nz)
    z0 = float(pos0[:, 2].max())
    axis = 0.5 * (lo[:2] + hi[:2])
    start = -2.0  # tip starts 2 A above the surface
    n_down = int(0.7 * n_frames)
    d_down = np.linspace(start, depth_max_A, n_down)
    d_up = np.linspace(depth_max_A, 0.4 * depth_max_A, n_frames - n_down + 1)[1:]
    depth = np.r_[d_down, d_up]
    path = np.r_[0.0, np.cumsum(np.abs(np.diff(depth)))]
    time = path / speed_A_per_ps
    if kind == "sphere":
        ind = Indenter.from_dict({"kind": "sphere", "radius_A": radius_A})
    else:
        ind = Indenter.from_dict({"kind": "berkovich", "face_angle_deg": 65.27, "tip_radius_A": 0.0})
    rng = np.random.default_rng(seed)
    P = np.empty((n_frames, len(pos0), 3))
    rel = pos0.copy()
    rel[:, :2] -= axis
    for t, h in enumerate(depth):
        hp = max(h, 0.0)
        a = max(np.sqrt(max(2 * radius_A * hp, 1e-9)), 2.0)
        r2 = rel[:, 0] ** 2 + rel[:, 1] ** 2 + (z0 - rel[:, 2]) ** 2
        f = a * a / (a * a + r2)
        u = np.zeros_like(rel)
        u[:, 2] = -hp * f
        rho = np.hypot(rel[:, 0], rel[:, 1]) + 1e-9
        u[:, 0] = 0.15 * hp * f * rel[:, 0] / rho * np.minimum(rho / a, 1.0)
        u[:, 1] = 0.15 * hp * f * rel[:, 1] / rho * np.minimum(rho / a, 1.0)
        p = rel + u
        if kind == "sphere":  # push atoms out of the ball
            c = np.array([0.0, 0.0, z0 - h + radius_A])
            dv = p - c
            dn = np.linalg.norm(dv, axis=1)
            inside = dn < radius_A
            p[inside] = c + dv[inside] / dn[inside, None] * radius_A
        p = p + rng.normal(0.0, noise_A, p.shape)
        p[:, :2] += axis
        P[t] = p

    def _load(d, frac_unload):
        return np.where(d > 0, 4.0 * np.sqrt(radius_A) * np.clip(d, 0, None) ** 1.5, 0.0) * frac_unload

    unl = np.ones(n_frames)
    unl[n_down:] = np.linspace(1.0, 0.2, n_frames - n_down)
    load = _load(depth, unl)
    tip = np.column_stack([np.full(n_frames, axis[0]), np.full(n_frames, axis[1]), z0 - depth])
    meta = dict(synthetic=True, material_plain="a made-up test crystal", potential="none (synthetic)",
                temperature_K=None, n_atoms_total=int(len(pos0)), tip_speed_m_s=speed_A_per_ps * 100.0,
                notes="made-up displacement field for renderer tests; not physics")
    tr = Trajectory(positions=P, species=spec, box_lo=lo, box_hi=hi, pbc=np.array([True, True, False]),
                    time_ps=time, tip_xyz=tip, depth_A=z0 - tip[:, 2], load_nN=load, surface_z0_A=z0,
                    indenter=ind, meta=meta)
    if n_curve_per_frame > 0:
        ct = np.linspace(time[0], time[-1], (n_frames - 1) * n_curve_per_frame + 1)
        cd = np.interp(ct, time, depth)
        cu = np.interp(ct, time, unl)
        tr.curve_time_ps, tr.curve_depth_A = ct, cd
        tr.curve_load_nN = _load(cd, cu) + rng.normal(0.0, 5.0, len(ct))
    return tr.validate()
