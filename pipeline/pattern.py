"""A toolpath pattern, the structure of a design as a few primitives with dimensionless parameters.

In direct ink writing the toolpath is the structure. material() gives the phase, ply and modulus at any
point of the bar, and roads() gives the toolpaths of one ply from the same formulas, so the simulated
section, the STL files and the G-code describe one part. Lengths are in road widths w or in ceramic layer
thicknesses h, so a ratio measured in an organism carries over while its absolute size does not. The road
width is 1.2 ply pitches, the filament diameter for a z-spacing of one pitch [Zhou 2023].

Plies stack along z with a pitch of h plus the interlayer thickness. Ply k holds ceramic up to h above its
base and interlayer above that, and the top ply has no interlayer. Phase 0 is ceramic and 2 interlayer.
"""

import math

import numpy as np

PATTERN = {
    "road_path": {"options": ["lines", "waves", "rings"], "default": "lines",
                  "description": "shape of the roads in a ply: straight lines, sine waves, or concentric rings "
                                 "around the centers of a square grid"},
    "road_angle_deg": {"default": 0.0, "range": [-90, 90], "unit": "degrees",
                       "description": "direction of the roads of the bottom ply, from the bar length"},
    "wave_amplitude_w": {"default": 0.0, "range": [0, 5], "unit": "road widths", "description": "amplitude of wavy roads"},
    "wave_length_w": {"default": 0.0, "range": [0, 500], "unit": "road widths", "description": "wavelength of wavy roads"},
    "ring_spacing_w": {"default": 0.0, "range": [0, 500], "unit": "road widths",
                       "description": "distance between neighboring ring centers"},
    "break_length_h": {"default": 0.0, "range": [0, 1000], "unit": "layer thicknesses",
                       "description": "length of a road segment between two breaks, which is the tablet length over the "
                                      "tablet thickness. 0 keeps roads continuous"},
    "break_shift": {"default": 0.5, "range": [0, 1], "unit": "segment lengths",
                    "description": "shift of the breaks from one ply to the next"},
    "ply_rotation_deg": {"default": 0.0, "range": [-90, 90], "unit": "degrees",
                         "description": "rotation of the road direction from one ply to the next, the pitch angle of a "
                                        "Bouligand stack"},
    "ply_rotation_alternates": {"options": [False, True], "default": False,
                                "description": "true to turn back and forth between the bottom ply's angle and that angle "
                                               "plus the rotation, as in crossed lamellae"},
    "ply_wave_amplitude_h": {"default": 0.0, "range": [0, 1], "unit": "layer thicknesses",
                             "description": "amplitude of a wave of the plies along the bar length, printed by non-planar toolpaths"},
    "ply_wave_length_h": {"default": 0.0, "range": [0, 2000], "unit": "layer thicknesses",
                          "description": "wavelength of that wave"},
    "interlayer_placement": {"options": ["sheet", "shell"], "default": "sheet",
                             "description": "sheet puts the interlayer between plies only. shell also wraps every road in "
                                            "it, so neighboring roads and road ends meet at a weak interface"},
    "modulus_profile": {"options": ["uniform", "linear", "power"], "default": "uniform",
                        "description": "how Young's modulus changes from the middle of the thickness to both faces"},
    "outer_layer_modulus_GPa": {"default": None, "range": [1, 1500], "unit": "GPa",
                                "description": "Young's modulus of the outermost plies, for a profile other than uniform"},
    "profile_exponent": {"default": 1.0, "range": [0.1, 10], "unit": "",
                         "description": "exponent of the power profile"},
}
# A slope steeper than this would bring the nozzle into the road it just printed.
MAX_SLOPE = math.tan(math.radians(30))


def complete(pattern):
    return {k: pattern.get(k, v["default"]) for k, v in PATTERN.items()}


def layered(pattern):
    """True when the structure is a stack of uniform flat layers, which any process can print."""
    p = complete(pattern)
    return (p["interlayer_placement"] == "sheet" and not p["break_length_h"] and not p["ply_wave_amplitude_h"]
            and p["modulus_profile"] == "uniform")


def pattern_errors(pattern, design):
    """What is wrong with a pattern, as parameter -> reason."""
    errors = {k: "is not a parameter of the pattern." for k in pattern if k not in PATTERN}
    p = complete(pattern)
    for k, spec in PATTERN.items():
        v = p[k]
        if "options" in spec:
            if v not in spec["options"]:
                errors[k] = f"{v!r} is not one of {spec['options']}."
        elif v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                                or not spec["range"][0] <= v <= spec["range"][1]):
            errors[k] = f"{v!r} is not a number from {spec['range'][0]} to {spec['range'][1]} {spec['unit']}."
    if errors:
        return errors
    # A parameter that acts only with another one would let the pattern claim a feature the part does not have.
    acts = {"wave_amplitude_w": (p["road_path"] == "waves", "road_path is waves"),
            "wave_length_w": (p["road_path"] == "waves", "road_path is waves"),
            "ring_spacing_w": (p["road_path"] == "rings", "road_path is rings"),
            "break_shift": (bool(p["break_length_h"]), "break_length_h is set"),
            "ply_rotation_alternates": (bool(p["ply_rotation_deg"]), "ply_rotation_deg is set"),
            "ply_wave_length_h": (bool(p["ply_wave_amplitude_h"]), "ply_wave_amplitude_h is set"),
            "outer_layer_modulus_GPa": (p["modulus_profile"] != "uniform", "modulus_profile is linear or power"),
            "profile_exponent": (p["modulus_profile"] == "power", "modulus_profile is power")}
    errors |= {k: f"has no effect unless {when}." for k, (on, when) in acts.items()
               if pattern.get(k, PATTERN[k]["default"]) != PATTERN[k]["default"] and not on}
    w, h = road_width(design), design["layer_thickness_um"] / 1000
    if p["road_path"] == "waves" and (p["wave_amplitude_w"] <= 0 or p["wave_length_w"] < 4
                                      or 2 * math.pi * p["wave_amplitude_w"] / p["wave_length_w"] > 1):
        # Neighboring wavy roads keep their spacing across the mean direction, so at a slope of 1 the spacing
        # normal to a road falls to 0.71 road widths.
        errors["wave_length_w"] = ("waves need an amplitude above 0, a wavelength of at least 4 road widths so that a road "
                                   "bends no tighter than its width, and a slope 2 pi amplitude / wavelength of at most 1.")
    if p["road_path"] == "rings" and p["ring_spacing_w"] < 3:
        errors["ring_spacing_w"] = "rings need centers at least 3 road widths apart."
    if p["break_length_h"] and p["break_length_h"] * h < 2 * w:
        errors["break_length_h"] = f"a segment of {p['break_length_h'] * h * 1000:.0f} um is shorter than 2 road widths of {w * 1000:.0f} um."
    if p["ply_wave_amplitude_h"] and (not p["ply_wave_length_h"]
                                      or 2 * math.pi * p["ply_wave_amplitude_h"] / p["ply_wave_length_h"] > MAX_SLOPE):
        errors["ply_wave_length_h"] = "a ply wave needs a wavelength long enough that its slope stays below 30 degrees."
    if p["modulus_profile"] != "uniform" and p["outer_layer_modulus_GPa"] is None:
        errors["outer_layer_modulus_GPa"] = "a graded profile needs the modulus of the outer plies."
    if p["interlayer_placement"] == "sheet" and not p["break_length_h"] and (
            p["road_path"] != "lines" or p["ply_rotation_deg"]):
        errors["interlayer_placement"] = ("waves, rings or a ply rotation leave no interface inside a ply when the interlayer "
                                          "is a sheet, so the part is a flat laminate. Use shell, which wraps every road in "
                                          "the interlayer.")
    if not layered(p) and design.get("process") not in (None, "direct ink writing"):
        errors["interlayer_placement"] = errors.get("interlayer_placement", "") + (
            f" Only direct ink writing prints this structure, not {design['process']}.")
    return {k: v.strip() for k, v in errors.items()}


def road_width(design):
    """Road width in mm, 1.2 ply pitches."""
    return 1.2 * (design["layer_thickness_um"] + design["interlayer_thickness_um"]) / 1000


def ply_angle(p, k):
    """Road direction of ply k in degrees. k may be an array."""
    turns = np.mod(k, 2) if p["ply_rotation_alternates"] else k
    return p["road_angle_deg"] + p["ply_rotation_deg"] * turns


def plies(pattern, design, n):
    """The n plies, bottom first, with nominal z range, road angle, modulus and outer paste fraction.

    The modulus is that of the design at mid-thickness and that of the pattern at both faces, with the
    fraction d of the way out given by the profile. The outer paste fraction is d, assuming that the
    modulus mixes linearly with the volume fraction of the two pastes.
    """
    p, h, s = complete(pattern), design["layer_thickness_um"] / 1000, design["interlayer_thickness_um"] / 1000
    E = design["layer_modulus_GPa"]
    outer = E if p["modulus_profile"] == "uniform" else p["outer_layer_modulus_GPa"]
    rows = []
    for k in range(n):
        d = abs(2 * (k + 0.5) / n - 1)
        d = 0.0 if p["modulus_profile"] == "uniform" else d ** (p["profile_exponent"] if p["modulus_profile"] == "power" else 1)
        rows.append({"index": k, "z0_mm": round(k * (h + s), 6), "z1_mm": round(k * (h + s) + h, 6),
                     "angle_deg": float(ply_angle(p, k)), "modulus_GPa": round(E + (outer - E) * d, 3),
                     "outer_fraction": round(d, 4)})
    return rows


def ply_shift(p, design, height, x, z):
    """Height of the ply wave at x, faded to 0 over one pitch at the bottom and the top of the bar, which stay flat."""
    h, s = design["layer_thickness_um"] / 1000, design["interlayer_thickness_um"] / 1000
    if not p["ply_wave_amplitude_h"]:
        return np.zeros(np.broadcast(x, z).shape)
    fade = np.clip(z / (h + s), 0, 1) * np.clip((height - z) / (h + s), 0, 1)
    return p["ply_wave_amplitude_h"] * h * np.sin(2 * np.pi * x / (p["ply_wave_length_h"] * h)) * fade


def frame(p, design, k, x, y):
    """Across and along coordinates of the roads of ply k at (x, y), and the road width.

    For lines and waves, across runs perpendicular to the roads and along runs with them. For rings,
    across is the radius from the nearest center and along the arc length.
    """
    w, t = road_width(design), np.radians(ply_angle(p, k))
    u, v = -x * np.sin(t) + y * np.cos(t), x * np.cos(t) + y * np.sin(t)
    if p["road_path"] == "rings":
        S = p["ring_spacing_w"] * w
        du, dv = u - (np.floor(u / S) + 0.5) * S, v - (np.floor(v / S) + 0.5) * S
        r = np.hypot(du, dv)
        return r, r * np.arctan2(du, dv), w
    if p["road_path"] == "waves":
        u = u - p["wave_amplitude_w"] * w * np.sin(2 * np.pi * v / (p["wave_length_w"] * w))
    return u, v, w


def joint(p, design, k, along):
    """True where the along coordinate of ply k falls in a break."""
    if not p["break_length_h"]:
        return np.zeros(np.shape(along), bool)
    L = p["break_length_h"] * design["layer_thickness_um"] / 1000
    return np.mod(along + k * p["break_shift"] * L, L) < design["interlayer_thickness_um"] / 1000


def material(pattern, design, n, height, x, y, z):
    """Phase (0 ceramic, 2 interlayer), ply index and modulus in GPa at points (x, y, z) of a bar of n plies, in mm."""
    p = complete(pattern)
    h, s = design["layer_thickness_um"] / 1000, design["interlayer_thickness_um"] / 1000
    x, y, z = np.broadcast_arrays(*map(np.asarray, (x, y, z)))
    level = z - ply_shift(p, design, height, x, z)
    k = np.clip(np.floor(level / (h + s)).astype(int), 0, n - 1)
    weak = level - k * (h + s) > h
    across, along, w = frame(p, design, k, x, y)
    if p["interlayer_placement"] == "shell":
        weak |= np.abs(across - (np.floor(across / w) + 0.5) * w) > (w - s) / 2
    weak |= joint(p, design, k, along)
    moduli = np.array([row["modulus_GPa"] for row in plies(p, design, n)])
    return np.where(weak, 2, 0), k, moduli[k]


def runs(points, keep):
    """The stretches of consecutive kept points, each with at least two points."""
    out, start = [], None
    for i, ok in enumerate(list(keep) + [False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= 2:
                out.append(points[start:i])
            start = None
    return out


def roads(pattern, design, n, height, k, footprint):
    """Toolpaths of ply k as {"ceramic": [...], "interlayer": [...]}, each a list of (m, 3) arrays of x, y, z in mm.

    A road is extruded along its points and the nozzle travels between roads. Ceramic roads run at the top of
    the ply's ceramic and, under sheet placement, interlayer roads across them at the top of the interlayer.
    Road centers stay half a road width inside the footprint, and breaks leave gaps of the interlayer thickness.
    """
    p = complete(pattern)
    h, s = design["layer_thickness_um"] / 1000, design["interlayer_thickness_um"] / 1000
    length, width = footprint
    w, theta = road_width(design), math.radians(ply_angle(p, k))
    L = p["break_length_h"] * h
    # Samples along a road. Straight roads on flat plies need only their ends and the break edges.
    step = min([x for x in [p["wave_length_w"] * w / 24 * (p["road_path"] == "waves"),
                            p["ply_wave_length_h"] * h / 24 * bool(p["ply_wave_amplitude_h"]),
                            w / 2 * (p["road_path"] == "rings"), s / 2 * bool(L) * (p["road_path"] != "lines")] if x] or [width + length])
    span = math.hypot(length, width)

    def on_footprint(xy):
        return ((xy[:, 0] >= w / 2 - 1e-9) & (xy[:, 0] <= length - w / 2 + 1e-9)
                & (xy[:, 1] >= w / 2 - 1e-9) & (xy[:, 1] <= width - w / 2 + 1e-9))

    def lift(xy, top):
        """z of points on the surface of the plies at nominal height top, found by fixed-point iteration."""
        z = np.full(len(xy), top)
        for _ in range(8):
            z = top + ply_shift(p, design, height, xy[:, 0], z)
        return np.column_stack([xy, z])

    def ceramic():
        out = []
        if p["road_path"] == "rings":
            S = p["ring_spacing_w"] * w
            centers = (np.arange(math.floor(-span / S), math.ceil(span / S)) + 0.5) * S  # the grid of material()
            for cu in centers:
                for cv in centers:
                    for m in range(int(S / math.sqrt(2) / w) + 1):
                        r = (m + 0.5) * w
                        phi = np.linspace(-math.pi, math.pi, max(16, int(2 * math.pi * r / step)) + 1)
                        du, dv = r * np.sin(phi), r * np.cos(phi)
                        u, v = cu + du, cv + dv
                        xy = np.column_stack([v * math.cos(theta) - u * math.sin(theta), v * math.sin(theta) + u * math.cos(theta)])
                        keep = (np.abs(du) <= S / 2) & (np.abs(dv) <= S / 2) & on_footprint(xy) & ~joint(p, design, k, r * phi)
                        out += [lift(seg, k * (h + s) + h) for seg in runs(xy, keep)]
            return out
        for j, uj in enumerate((np.arange(math.floor(-span / w), math.ceil(span / w)) + 0.5) * w):
            v = np.arange(-span, span + step, step)
            if L:
                edges = np.arange(math.floor(-span / L) - 1, math.ceil(span / L) + 1) * L - k * p["break_shift"] * L
                v = np.concatenate([v, edges - 1e-9, edges + s / 2, edges + s + 1e-9])
            if p["road_path"] == "lines":  # the ends of a straight road where it enters and leaves the footprint
                c, si = math.cos(theta), math.sin(theta)
                for lo, hi, a, b in [(w / 2, length - w / 2, c, -uj * si), (w / 2, width - w / 2, si, uj * c)]:
                    if abs(a) > 1e-12:
                        v = np.concatenate([v, [(lo - b) / a, (hi - b) / a]])
            v = np.unique(v[(v >= -span) & (v <= span)])
            u = uj + (p["wave_amplitude_w"] * w * np.sin(2 * np.pi * v / (p["wave_length_w"] * w)) if p["road_path"] == "waves" else 0)
            xy = np.column_stack([v * math.cos(theta) - u * math.sin(theta), v * math.sin(theta) + u * math.cos(theta)])
            segs = runs(xy, on_footprint(xy) & ~joint(p, design, k, v))
            # Every other road runs backwards, so the nozzle moves in a serpentine.
            out += [lift(seg[::-1] if j % 2 else seg, k * (h + s) + h) for seg in (segs[::-1] if j % 2 else segs)]
        return out

    def interlayer():
        if p["interlayer_placement"] != "sheet" or k == n - 1:
            return []
        out, across = [], theta + math.pi / 2
        for j, uj in enumerate((np.arange(math.floor(-span / w), math.ceil(span / w)) + 0.5) * w):
            v = np.arange(-span, span + step, step)
            c, si = math.cos(across), math.sin(across)
            for lo, hi, a, b in [(w / 2, length - w / 2, c, -uj * si), (w / 2, width - w / 2, si, uj * c)]:
                if abs(a) > 1e-12:
                    v = np.concatenate([v, [(lo - b) / a, (hi - b) / a]])
            v = np.unique(v[(v >= -span) & (v <= span)])
            xy = np.column_stack([v * c - uj * si, v * si + uj * c])
            out += [lift(seg[::-1] if j % 2 else seg, (k + 1) * (h + s)) for seg in runs(xy, on_footprint(xy))]
        return out

    return {"ceramic": ceramic(), "interlayer": interlayer()}


def describe(pattern):
    """The pattern in words, for the novelty statement and the report."""
    p = complete(pattern)
    words = {"lines": "straight roads", "waves": "wavy roads", "rings": "concentric rings"}[p["road_path"]]
    parts = [words]
    if p["break_length_h"]:
        parts.append(f"tablets {p['break_length_h']:g} layer thicknesses long, staggered by {p['break_shift']:g}")
    if p["ply_rotation_deg"]:
        parts.append(f"plies turned {'back and forth ' if p['ply_rotation_alternates'] else ''}by {p['ply_rotation_deg']:g} degrees"
                     + (" in a Bouligand stack" if not p["ply_rotation_alternates"] else ", crossed"))
    if p["ply_wave_amplitude_h"]:
        parts.append("wavy plies")
    parts.append("interlayer shells around every road" if p["interlayer_placement"] == "shell" else "interlayer sheets between plies")
    if p["modulus_profile"] != "uniform":
        parts.append(f"modulus graded to {p['outer_layer_modulus_GPa']:g} GPa at the faces")
    return ", ".join(parts)
