"""Print files for the bar of a design, for the additive manufacturing process the design chose.

The bar of geometry.json is the part to print. Direct ink writing walks the toolpaths of its pattern ply by
ply, as pattern.roads() gives them. Vat photopolymerization and binder jetting print only layered patterns,
whose ceramic layers and interlayers alternate along z with a ceramic layer at the bottom and at the top.
A layer is printed in passes of equal height, as few as the process allows, so the passes add up to the bar
height. A print parameter the recipe does not give takes its value from DEFAULTS, which hold typical values
for oxide ceramics and are labeled as defaults.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from .pattern import describe, ply_shift, roads
from .render import COLOR

PROCESSES = ["direct ink writing", "vat photopolymerization", "binder jetting"]
# A travel longer than this lifts the nozzle by z_hop_mm, so it does not drag through printed roads. The step to
# the next road of a serpentine stays at print height, since a lift there would add two Z moves to every road.
HOP_MM = 1.0
# A parameter name ends in its unit, as the design fields do. The writers read only the value.
DEFAULTS = {
    "direct ink writing": {
        "nozzle_diameter_um": {"value": 250, "unit": "µm"},
        "max_layer_height_fraction": {"value": 0.8, "unit": "nozzle diameters"},
        "min_layer_height_fraction": {"value": 0.3, "unit": "nozzle diameters"},
        "print_speed_mm_s": {"value": 10, "unit": "mm/s"},
        "travel_speed_mm_s": {"value": 30, "unit": "mm/s"},
        "syringe_diameter_mm": {"value": 14.5, "unit": "mm"},
        "z_hop_mm": {"value": 1, "unit": "mm"},
        "tool_change_s": {"value": 30, "unit": "s"},
    },
    "vat photopolymerization": {
        "slice_thickness_um": {"value": 50, "unit": "µm"},
        "pixel_size_um": {"value": 50, "unit": "µm"},
        "exposure_s": {"value": 5, "unit": "s"},
        "base_exposure_s": {"value": 20, "unit": "s"},
        "base_layers": {"value": 3, "unit": "layers"},
        "lift_mm": {"value": 5, "unit": "mm"},
        "lift_speed_mm_s": {"value": 1, "unit": "mm/s"},
        "material_change_s": {"value": 120, "unit": "s"},
    },
    "binder jetting": {
        "powder_layer_um": {"value": 100, "unit": "µm"},
        "binder_saturation_percent": {"value": 70, "unit": "%"},
        "layer_time_s": {"value": 30, "unit": "s"},
    },
}


def passes(spec, most_um):
    """The passes of the bar, bottom first, as (role, height in µm, top in mm), and a summary per role.

    A layer of thickness t takes n = ceil(t / most_um) passes of t / n, so a layer thinner than most_um is one pass.
    """
    d, n = spec["design"], spec["bar"]["ceramic_layers"]
    plan = []
    for role, count in [("ceramic", n), ("interlayer", n - 1)]:
        t = d["layer_thickness_um" if role == "ceramic" else "interlayer_thickness_um"]
        k = math.ceil(t / most_um - 1e-9)
        plan.append({"role": role, "material": d[role], "layers": count, "thickness_um": t, "passes_each": k,
                     "pass_um": round(t / k, 3)})
    steps, z = [], 0.0
    for i in range(2 * n - 1):
        p = plan[i % 2]
        for _ in range(p["passes_each"]):
            z += p["thickness_um"] / p["passes_each"]
            steps.append((p["role"], p["thickness_um"] / p["passes_each"], z / 1000))
    return steps, plan


def direct_ink_writing(spec, v, p, out):
    """print.gcode that walks the roads of every ply.

    Under sheet placement T0 prints the ceramic roads with the ceramic paste and T1 the interlayer roads across
    them with the interlayer paste. Under shell placement T0 is one coaxial head that prints every road as a
    ceramic core in an interlayer shell [Zhou 2023].
    """
    d, pat, plan, w = spec["design"], spec["pattern"], spec["plies"], spec["road_width_mm"]
    (length, width, height), n = spec["bar"]["size_mm"], spec["bar"]["ceramic_layers"]
    h, s = d["layer_thickness_um"] / 1000, d["interlayer_thickness_um"] / 1000
    shell, graded = pat["interlayer_placement"] == "shell", pat["modulus_profile"] != "uniform"
    plunger = math.pi * (v["syringe_diameter_mm"] / 2) ** 2
    fp, ft = 60 * v["print_speed_mm_s"], 60 * v["travel_speed_mm_s"]
    smallest = v["min_layer_height_fraction"] * v["nozzle_diameter_um"]
    if shell:
        pitch = round((h + s) * 1000, 3)
        layers = [{"role": "ply", "material": f"{d['ceramic']} core in a {d['interlayer']} shell", "layers": n,
                   "thickness_um": pitch, "passes_each": 1, "pass_um": pitch}]
        notes = [f"T0 is one coaxial core-shell head [Zhou 2023] that prints every road with a core of the {d['ceramic']} "
                 f"paste and a shell of the {d['interlayer']} paste. It is a Marlin mixing extruder whose inputs A and B "
                 "feed the core and C the shell. M165 before each ply sets their volume shares, so C is the shell's share "
                 "of the road cross-section."
                 + " Input B is unused, since the modulus is uniform." * (not graded)]
        notes += [f"A {pitch:g} µm ply is thinner than the {smallest:g} µm smallest pass of a "
                  f"{v['nozzle_diameter_um']:g} µm nozzle."] * (pitch < smallest)
    else:
        layers = passes(spec, v["max_layer_height_fraction"] * v["nozzle_diameter_um"])[1]
        notes = [f"T0 holds the {d['ceramic']} paste and T1 the {d['interlayer']} paste. The tool offsets are set in the printer firmware."]
        notes += [f"A {l['pass_um']:g} µm {l['material']} pass is thinner than the {smallest:g} µm smallest pass of a "
                  f"{v['nozzle_diameter_um']:g} µm nozzle, so this film is better made by dip or spray coating."
                  for l in layers if l["pass_um"] < smallest]
    if graded:
        notes.append(f"{'Input A' if shell else 'T0 is a Marlin mixing extruder whose input A'} holds the {d['ceramic']} paste "
                     f"of {d['layer_modulus_GPa']:g} GPa and input B the outer paste, the composition with the outer modulus of "
                     f"{pat['outer_layer_modulus_GPa']:g} GPa. M165 before each ply sets the shares of A and B from the ply's "
                     f"outer fraction{'' if shell else ', with C at 0'}. The mix assumes that the modulus mixes linearly with "
                     "the volume fraction of the two pastes.")
    if abs(v["nozzle_diameter_um"] / 1000 / w - 1) > 0.2:
        notes.append(f"The {v['nozzle_diameter_um']:g} µm nozzle differs from the {w * 1000:g} µm road width by "
                     f"{abs(v['nozzle_diameter_um'] / 1000 / w - 1) * 100:.0f} %. Roads are spaced and fed for the road width, "
                     f"so a nozzle near {w * 1000:g} µm prints the designed part.")
    if pat["break_length_h"] and s < w / 2:
        notes.append(f"Breaks leave {s * 1000:g} µm gaps between road ends. A bead spreads past its end by up to about half "
                     f"its {w * 1000:g} µm width, so a gap this narrow may close before sintering.")

    body, first, here, tool, changes, printed, travel, count = [], [], np.zeros(3), 0, 0, 0.0, 0.0, 0
    for k, ply in enumerate(plan):
        r = roads(pat, d, n, height, k, (length, width))
        first += [r] * (k < 3)
        if shell:
            # Ply 0 rests on the plate and is h high. Every later ply adds a pitch, core and shell together.
            t = h if k == 0 else h + s
            jobs, c = [("ceramic", r["ceramic"], 0.0, t)], 1 - (w - s) * h / (w * t)
        else:
            # The roads of a layer thicker than the largest pass repeat in passes of equal height below its top.
            jobs = [(role, r[role], -(m - 1 - j) * t / m, t / m)
                    for role, t, m in [("ceramic", h, layers[0]["passes_each"]), ("interlayer", s, layers[1]["passes_each"])]
                    for j in range(m)]
            c = 0.0
        a, b = round((1 - c) * (1 - ply["outer_fraction"]), 4), round((1 - c) * ply["outer_fraction"], 4)
        mix = f"M165 A{a:.4f} B{b:.4f} C{1 - a - b:.4f}" if shell or graded else None
        for role, rs, dz, t in jobs:
            if not rs:
                continue
            count += 1
            want, e = int(role == "interlayer"), w * t / plunger  # plunger travel per mm of road
            body.append(f"; ply {k + 1} of {n}, {role} roads at {ply['angle_deg'] + 90 * want:g} degrees, "
                        f"{t * 1000:g} um high, top at {rs[0][0, 2] + dz:.4f} mm")
            for road in rs:
                road = road + (0, 0, dz)
                # A point on the straight line between its neighbors adds a line of G-code and no shape.
                bend = np.linalg.norm(np.cross(road[1:-1] - road[:-2], road[2:] - road[:-2]), axis=1)
                road = road[np.r_[True, bend > 1e-9, True]]
                gap = math.dist(here[:2], road[0, :2])
                if want != tool or gap > HOP_MM:
                    up = max(here[2], road[0, 2]) + v["z_hop_mm"]
                    body += [f"G0 Z{up:.3f} F{ft:g}"] + [f"T{want}"] * (want != tool) + [
                        f"G0 X{road[0, 0]:.3f} Y{road[0, 1]:.3f}", f"G0 Z{road[0, 2]:.3f}"]
                    changes, tool = changes + (want != tool), want
                    travel += 2 * up - here[2] - road[0, 2] + gap
                else:
                    body.append(f"G0 X{road[0, 0]:.3f} Y{road[0, 1]:.3f} Z{road[0, 2]:.3f} F{ft:g}")
                    travel += math.dist(here, road[0])
                if mix:
                    body.append(mix)
                    mix = None
                seg = np.linalg.norm(np.diff(road, axis=0), axis=1)
                z = [f"{x:.3f}" for x in road[:, 2]]
                body += [f"G1 X{x:.3f} Y{y:.3f}" + f" Z{z[i + 1]}" * (z[i + 1] != z[i]) + f" E{e * q:.8f}"
                         + f" F{fp:g}" * (i == 0) for i, (x, y, q) in enumerate(zip(road[1:, 0], road[1:, 1], seg))]
                printed += seg.sum()
                here = road[-1]
    minutes = (printed / v["print_speed_mm_s"] + travel / v["travel_speed_mm_s"] + changes * v["tool_change_s"]) / 60

    head = [f"; Direct ink writing of a bar of {length:g} x {width:g} x {height:g} mm, {n} plies stacked along z",
            f"; Design with {d['ceramic']} layers of {d['layer_thickness_um']:g} um and {d['interlayer']} interlayers of "
            f"{d['interlayer_thickness_um']:g} um, layer modulus {d['layer_modulus_GPa']:g} GPa",
            f"; Toolpath pattern of {describe(pat)}",
            "; Road angles of the plies in degrees from x, bottom first, " + " ".join(f"{x['angle_deg']:g}" for x in plan)]
    head += ["; Ply moduli in GPa, bottom first, " + " ".join(f"{x['modulus_GPa']:g}" for x in plan)] * graded
    head += ["; Parameters as name, value, unit and source"] + [f";   {k} {x['value']} {x['unit']} ({x['source']})" for k, x in p.items()]
    head += ["; Layer plan"] + ([f";   {n} plies of core-shell roads {w * 1000:g} um apart. Ply 1 is {h * 1000:g} um high and "
                                 f"each later ply adds {layers[0]['pass_um']:g} um."] if shell else [
        f";   {l['layers']} {l['material']} {'layers' if l['role'] == 'ceramic' else 'interlayers'} of {l['thickness_um']:g} um, "
        f"each in {l['passes_each']} pass{'es' * (l['passes_each'] > 1)} of {l['pass_um']:g} um, tool T{int(l['role'] == 'interlayer')}"
        for l in layers] + [f";   {count} passes. Ceramic roads run at the ply angle and interlayer roads across them, "
                            f"{w * 1000:g} um apart."])
    head += ["; Origin at a corner of the bar footprint on the build plate. XYZ absolute in mm, Z at the top of each road.",
             f"; E is syringe plunger travel in mm, road width x layer height x road length over the plunger area of {plunger:.2f} mm^2.",
             f"; A travel longer than {HOP_MM:g} mm lifts the nozzle by {v['z_hop_mm']:g} mm."]
    head += [f"; {x}" for x in notes] + [f"; Estimated print time {minutes:.1f} min", ";",
                                         "G21 ; mm", "G90 ; absolute XYZ", "M83 ; relative E", "G28 ; home", "T0"]
    tail = [f"G0 Z{height + v['z_hop_mm']:.3f} F{ft:g}", "M84 ; motors off"]
    (out / "print.gcode").write_text("\n".join(head + body + tail).replace("µ", "u") + "\n")
    legend = {"ceramic": f"{d['ceramic']} {'core' if shell else 'paste'}" + f", {d['layer_modulus_GPa']:g} GPa" * graded + ", T0",
              "interlayer": f"{d['interlayer']} " + ("shell, T0" if shell else "paste, T1")}
    return {"files": {"gcode": str(out / "print.gcode")}, "layers": layers, "printed_layers": count,
            "counts": {"tool_changes": changes}, "minutes": minutes, "notes": notes, "toolpaths": first, "legend": legend,
            "method": "G-code that walks the toolpaths of the pattern in geometry.json ply by ply, with "
                      + ("one coaxial core-shell head" if shell else "two tools")}


def vat_photopolymerization(spec, v, p, out):
    """print_job.json with one entry per printed layer, and mask.png, the cross-section that every layer exposes."""
    d, (length, width, height) = spec["design"], spec["bar"]["size_mm"]
    steps, layers = passes(spec, v["slice_thickness_um"])
    name = {"ceramic": f"{d['ceramic']} slurry", "interlayer": f"{d['interlayer']} interlayer resin"}
    job = [{"z_mm": round(z, 6), "thickness_um": round(h, 3), "material": name[r],
            "exposure_s": v["base_exposure_s"] if i < v["base_layers"] else v["exposure_s"]} for i, (r, h, z) in enumerate(steps)]
    changes = sum(a[0] != b[0] for a, b in zip(steps, steps[1:]))
    px = v["pixel_size_um"] / 1000
    nx, ny, m = round(length / px), round(width / px), round(1 / px)  # a dark margin of 1 mm around the footprint
    mask = np.zeros((ny + 2 * m, nx + 2 * m), bool)
    mask[m:m + ny, m:m + nx] = True
    Image.fromarray(mask).save(out / "mask.png")
    minutes = (sum(x["exposure_s"] for x in job) + len(job) * 2 * v["lift_mm"] / v["lift_speed_mm_s"]
               + changes * v["material_change_s"]) / 60
    notes = [f"Each of the {changes} changes between the {name['ceramic']} and the {name['interlayer']} needs a vat change "
             "with a rinse, or a multi-vat printer.",
             f"The first {v['base_layers']} layers bond the part to the build platform and take the base exposure.",
             "exposure_s applies to both materials. An interlayer resin with another cure depth needs its own exposure."]
    (out / "print_job.json").write_text(json.dumps({
        "process": "vat photopolymerization", "design": d, "bar_size_mm": [length, width, height],
        "layer_z": "z_mm is the top of each layer above the build platform",
        "mask": {"file": "mask.png", "pixel_size_um": v["pixel_size_um"], "image_px": [nx + 2 * m, ny + 2 * m],
                 "footprint_px": [nx, ny], "offset_px": [m, m],
                 "meaning": "white pixels are exposed, x along image columns. Every layer has this footprint, so one mask serves all layers."},
        "material_changes": changes, "parameters": p, "notes": notes, "estimated_print_time_min": round(minutes, 1),
        "layers": job}, indent=1, ensure_ascii=False))
    return {"files": {"print_job_json": str(out / "print_job.json"), "mask_png": str(out / "mask.png")}, "steps": steps,
            "layers": layers, "printed_layers": len(steps), "counts": {"material_changes": changes}, "minutes": minutes,
            "notes": notes, "hatched": [i for i in range(len(steps)) if i < v["base_layers"]], "hatch_label": "base layers",
            "legend": {"ceramic": name["ceramic"], "interlayer": name["interlayer"]},
            "method": "layer job from the bar of geometry.json, one mask for every layer"}


def binder_jetting(spec, v, p, out):
    """print_job.json with the powder layers of the ceramic and the interlayers made between print sessions."""
    d, (length, width, height) = spec["design"], spec["bar"]["size_mm"]
    steps, layers = passes(spec, v["powder_layer_um"])
    job, session = [], 1
    for i, (r, h, z) in enumerate(steps):
        session += i > 0 and r == "ceramic" and steps[i - 1][0] == "interlayer"
        job.append({"z_mm": round(z, 6), "thickness_um": round(h, 3)} | (
            {"material": f"{d['ceramic']} powder", "binder_saturation_percent": v["binder_saturation_percent"], "session": session}
            if r == "ceramic" else
            {"material": d["interlayer"], "printed": False, "step": f"infiltrate or coat between sessions {session} and {session + 1}"}))
    printed = sum(r == "ceramic" for r, _, _ in steps)
    ceramic, between = layers
    notes = [f"A binder jetting bed spreads one powder, so the {d['interlayer']} interlayer is not printable in the "
             f"{d['ceramic']} powder bed. Each of the {between['layers']} interlayers must be infiltrated or coated between "
             f"print sessions, which splits the print into {session} sessions.",
             f"Each {ceramic['thickness_um']:g} µm ceramic layer is spread as {ceramic['passes_each']} powder "
             f"layer{'s' * (ceramic['passes_each'] > 1)} of {ceramic['pass_um']:g} µm, at most the set {v['powder_layer_um']:g} µm.",
             "The estimated time counts the printed powder layers only. Coating, curing, depowdering and sintering are not included."]
    if between["thickness_um"] < v["powder_layer_um"]:
        notes.append(f"A {between['thickness_um']:g} µm interlayer is thinner than one {v['powder_layer_um']:g} µm powder "
                     "layer, so it cannot be left as an unbound gap in the bed for later infiltration and must be coated.")
    minutes = printed * v["layer_time_s"] / 60
    (out / "print_job.json").write_text(json.dumps({
        "process": "binder jetting", "design": d, "bar_size_mm": [length, width, height],
        "layer_z": "z_mm is the top of each layer above the build plate", "powder_layer_um": v["powder_layer_um"],
        "binder_saturation_percent": v["binder_saturation_percent"], "print_sessions": session,
        "parameters": p, "notes": notes, "estimated_print_time_min": round(minutes, 1), "layers": job},
        indent=1, ensure_ascii=False))
    return {"files": {"print_job_json": str(out / "print_job.json")}, "steps": steps, "layers": layers, "printed_layers": printed,
            "counts": {"print_sessions": session}, "minutes": minutes, "notes": notes,
            "hatched": [i for i, s in enumerate(steps) if s[0] == "interlayer"], "hatch_label": "not printed, coated between sessions",
            "legend": {"ceramic": f"{d['ceramic']} powder with binder", "interlayer": f"{d['interlayer']} interlayer"},
            "method": "powder layer job from the bar of geometry.json, one print session per ceramic layer"}


WRITERS = {"direct ink writing": direct_ink_writing, "vat photopolymerization": vat_photopolymerization,
           "binder jetting": binder_jetting}


def write_fabrication(geometry_json, process, params, folder):
    """Write the print files of the bar and preview.png, a figure of the print plan, into folder, and return what they hold.

    params maps a parameter name to {"value", "unit", "source"}. A parameter of DEFAULTS that params lacks takes
    its default value with source "default".
    """
    if process not in PROCESSES:
        raise ValueError(f"{process!r} is not one of {PROCESSES}")
    spec = json.loads(Path(geometry_json).read_text())
    p = {k: x | {"source": "default"} for k, x in DEFAULTS[process].items()} | params
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    r = WRITERS[process](spec, {k: x["value"] for k, x in p.items()}, p, out)

    d, (length, width, height) = spec["design"], spec["bar"]["size_mm"]
    handles = [Patch(color=COLOR[k], label=label) for k, label in r["legend"].items()]
    if "toolpaths" in r:
        # Top views over 8 mm of length at mid-bar, where single roads stay legible, and the whole bar from the side.
        top = [(f"ply {k}", k, "ceramic") for k in range(len(r["toolpaths"]))]
        top += [("interlayer", 0, "interlayer")] * bool(r["toolpaths"][0]["interlayer"])
        rows = [[t[0] for t in top[:2]], [t[0] for t in top[2:]]] if len(top) == 4 else [[t[0] for t in top]]
        rows.append(["whole"] * len(rows[0]))
        fig, ax = plt.subplot_mosaic(rows, figsize=(4.6 * len(rows[0]) + 0.8, 2.5 * len(rows) + 1.2), layout="constrained")
        for key, k, role in top:
            angle = spec["plies"][k]["angle_deg"] + 90 * (role == "interlayer")
            ax[key].add_collection(LineCollection([x[:, :2] for x in r["toolpaths"][k][role]], color=COLOR[role], linewidth=0.7))
            ax[key].add_patch(Rectangle((0, 0), length, width, fill=False, edgecolor="#8a8a8a", linewidth=0.8))
            ax[key].set(aspect="equal", xlim=(length / 2 - 4, length / 2 + 4), ylim=(-0.2, width + 0.2), xlabel="x (mm)",
                        ylabel="y (mm)", title=f"Ply {k + 1}{' Interlayer' * (role == 'interlayer')} at {angle:g}°, Top View")
        # Ceramic plies over the interlayer, lighter by the share of the outer paste.
        shade = lambda f: np.array(matplotlib.colors.to_rgb(COLOR["ceramic"])) * (1 - 0.6 * f) + 0.6 * f
        x = np.linspace(0, length, 1000)
        ax["whole"].fill_between([0, length], 0, height, color=COLOR["interlayer"], linewidth=0)
        for ply in spec["plies"]:
            lo, hi = (z + ply_shift(spec["pattern"], d, height, x, z) for z in (ply["z0_mm"], ply["z1_mm"]))
            ax["whole"].fill_between(x, lo, hi, color=shade(ply["outer_fraction"]), linewidth=0)
        ax["whole"].set(xlim=(0, length), ylim=(0, height), xlabel="x (mm)", ylabel="z (mm)", title="Whole Bar, Side View")
        if spec["pattern"]["modulus_profile"] != "uniform":
            handles.insert(1, Patch(color=shade(1), label=f"outer paste, {spec['pattern']['outer_layer_modulus_GPa']:g} GPa"))
    else:
        fig, ax = plt.subplot_mosaic([["whole", "bottom"]], figsize=(10, 3.4), layout="constrained")
        bottom = min(height, (2 * d["layer_thickness_um"] + d["interlayer_thickness_um"]) / 1000)
        for key, top, title in [("whole", height, "Whole Bar, Side View"), ("bottom", bottom, "Bottom Three Layers, Side View")]:
            shown = [(i, s) for i, s in enumerate(r["steps"]) if s[2] <= top + 1e-9]
            bars = ax[key].bar([length / 2] * len(shown), [h / 1000 for _, (_, h, _) in shown], length,
                               [z - h / 1000 for _, (_, h, z) in shown], color=[COLOR[s[0]] for _, s in shown],
                               edgecolor="white", linewidth=0.6 if key == "bottom" else 0)
            for patch, (i, _) in zip(bars, shown):
                patch.set_hatch("////" if i in r["hatched"] else None)
            ax[key].set(xlim=(0, length), ylim=(0, top), xlabel="x (mm)", ylabel="z (mm)", title=title)
        handles += [Patch(facecolor="#9a9a9a", edgecolor="white", hatch="////", label=r.get("hatch_label"))] * bool(r["hatched"])
    for a in ax.values():
        a.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    fig.suptitle(f"{process.title()} Print Plan", fontsize=14)
    fig.savefig(out / "preview.png", dpi=150)
    plt.close(fig)

    return {"process": process, "files": r["files"] | {"preview_png": str(out / "preview.png")}, "parameters": p,
            "layers": r["layers"], "printed_layers": r["printed_layers"]} | r["counts"] | {
            "estimated_print_time_min": round(r["minutes"], 1), "notes": r["notes"],
            "method": r["method"] + ". Parameters without a recipe value are defaults of the fabrication step"}
