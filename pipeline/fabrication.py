"""Print files for the bar of a layered design, for the additive manufacturing process the design chose.

The bar of geometry.json is the part to print. Its ceramic layers and interlayers alternate along z, with a
ceramic layer at the bottom and at the top. Each layer is printed in passes of equal height, as few as the
process allows, so the passes add up to the bar height. A print parameter the recipe does not give takes
its value from DEFAULTS, which hold typical values for oxide ceramics and are labeled as defaults.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from .render import COLOR

PROCESSES = ["direct ink writing", "vat photopolymerization", "binder jetting"]
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
    """print.gcode for a two-tool printer, T0 with the ceramic paste and T1 with the interlayer paste."""
    d, (length, width, height) = spec["design"], spec["bar"]["size_mm"]
    nozzle = v["nozzle_diameter_um"] / 1000
    steps, layers = passes(spec, v["max_layer_height_fraction"] * v["nozzle_diameter_um"])
    plunger = math.pi * (v["syringe_diameter_mm"] / 2) ** 2
    fp, ft = 60 * v["print_speed_mm_s"], 60 * v["travel_speed_mm_s"]
    smallest = v["min_layer_height_fraction"] * v["nozzle_diameter_um"]
    notes = [f"T0 holds the {d['ceramic']} paste and T1 the {d['interlayer']} paste. The tool offsets are set in the printer firmware."]
    notes += [f"A {l['pass_um']:g} µm {l['material']} pass is thinner than the {smallest:g} µm smallest pass of a "
              f"{v['nozzle_diameter_um']:g} µm nozzle, so this film is better made by dip or spray coating."
              for l in layers if l["pass_um"] < smallest]
    body, toolpaths, here, tool, changes, printed, travel = [], {}, (0.0, 0.0, 0.0), 0, 0, 0.0, 0.0
    for i, (role, h, z) in enumerate(steps):
        # A serpentine raster along x on even passes and along y on odd ones. Road centers keep half a nozzle
        # from the edge, so the bead stays inside the footprint.
        a, b = (length, width) if i % 2 == 0 else (width, length)
        n = int(b / nozzle + 1e-9)
        across = (b - n * nozzle) / 2 + nozzle / 2 + nozzle * np.arange(n)
        pts = [(s, c) if i % 2 == 0 else (c, s) for k, c in enumerate(across)
               for s in ((nozzle / 2, a - nozzle / 2) if k % 2 == 0 else (a - nozzle / 2, nozzle / 2))]
        toolpaths.setdefault(role, pts)
        hop = here[2] + v["z_hop_mm"]
        body += [f"; pass {i + 1} of {len(steps)}, {d[role]}, {h:g} um, top at {z:.4f} mm", f"G0 Z{hop:.4f} F{ft:g}"]
        if int(role == "interlayer") != tool:
            tool, changes = 1 - tool, changes + 1
            body.append(f"T{tool}")
        body += [f"G0 X{pts[0][0]:.4f} Y{pts[0][1]:.4f} F{ft:g}", f"G0 Z{z:.4f} F{ft:g}"]
        travel += v["z_hop_mm"] + math.dist(here[:2], pts[0]) + hop - z
        e = nozzle * h / 1000 / plunger  # plunger travel per mm of road
        for q, r in zip(pts, pts[1:]):
            body.append(f"G1 X{r[0]:.4f} Y{r[1]:.4f} E{e * math.dist(q, r):.8f} F{fp:g}")
            printed += math.dist(q, r)
        here = (*pts[-1], z)
    minutes = (printed / v["print_speed_mm_s"] + travel / v["travel_speed_mm_s"] + changes * v["tool_change_s"]) / 60

    head = [f"; Direct ink writing of a layered bar, {length:g} x {width:g} x {height:g} mm, layers stacked along z",
            f"; Design with {d['ceramic']} layers of {d['layer_thickness_um']:g} um and {d['interlayer']} interlayers of "
            f"{d['interlayer_thickness_um']:g} um", "; Parameters as name, value, unit and source"]
    head += [f";   {k} {x['value']} {x['unit']} ({x['source']})" for k, x in p.items()]
    head += ["; Layer plan"] + [
        f";   {l['layers']} {l['material']} {'layers' if l['role'] == 'ceramic' else 'interlayers'} of {l['thickness_um']:g} um, "
        f"each in {l['passes_each']} pass{'es' * (l['passes_each'] > 1)} of {l['pass_um']:g} um, tool T{int(l['role'] == 'interlayer')}"
        for l in layers]
    head += [f";   {len(steps)} passes, serpentine raster with road spacing {v['nozzle_diameter_um']:g} um, 0 and 90 degrees alternating",
             "; Origin at a corner of the bar footprint on the build plate. XYZ absolute in mm, Z at the top of each pass.",
             f"; E is syringe plunger travel in mm, road width x pass height x road length over the plunger area of {plunger:.2f} mm^2."]
    head += [f"; {x}" for x in notes] + [f"; Estimated print time {minutes:.1f} min", ";",
                                         "G21 ; mm", "G90 ; absolute XYZ", "M83 ; relative E", "G28 ; home", "T0"]
    tail = [f"G0 Z{height + v['z_hop_mm']:.4f} F{ft:g}", "M84 ; motors off"]
    (out / "print.gcode").write_text("\n".join(head + body + tail).replace("µ", "u") + "\n")
    return {"files": {"gcode": str(out / "print.gcode")}, "steps": steps, "layers": layers, "printed_layers": len(steps),
            "counts": {"tool_changes": changes}, "minutes": minutes, "notes": notes, "toolpaths": toolpaths, "hatched": [],
            "legend": {"ceramic": f"{d['ceramic']} paste, T0", "interlayer": f"{d['interlayer']} paste, T1"},
            "method": "G-code from the bar of geometry.json, a serpentine raster per pass with two tools"}


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
    rows = [["ceramic"] * 2, ["interlayer"] * 2] * ("toolpaths" in r) + [["whole", "bottom"]]
    fig, ax = plt.subplot_mosaic(rows, figsize=(10, 1.9 * len(rows) + 1.5), layout="constrained")
    for role, pts in r.get("toolpaths", {}).items():
        xy = np.array(pts)
        ax[role].plot(xy[:, 0], xy[:, 1], color=COLOR[role], linewidth=0.7)
        ax[role].plot(*xy[0], "o", color=COLOR["mark"], markersize=5)
        ax[role].add_patch(Rectangle((0, 0), length, width, fill=False, edgecolor="#8a8a8a", linewidth=0.8))
        ax[role].set(aspect="equal", xlim=(-0.5, length + 0.5), ylim=(-0.5, width + 0.5), xlabel="x (mm)", ylabel="y (mm)",
                     title=f"First {role.title()} Pass, Top View")
    bottom = min(height, (2 * d["layer_thickness_um"] + d["interlayer_thickness_um"]) / 1000)
    for key, top, title in [("whole", height, "Whole Bar, Side View"), ("bottom", bottom, "Bottom Three Layers, Side View")]:
        shown = [(i, s) for i, s in enumerate(r["steps"]) if s[2] <= top + 1e-9]
        bars = ax[key].bar([length / 2] * len(shown), [h / 1000 for _, (_, h, _) in shown], length,
                           [z - h / 1000 for _, (_, h, z) in shown], color=[COLOR[s[0]] for _, s in shown],
                           edgecolor="white", linewidth=0.6 if key == "bottom" else 0)
        for patch, (i, _) in zip(bars, shown):
            patch.set_hatch("////" if i in r["hatched"] else None)
        ax[key].set(xlim=(0, length), ylim=(0, top), xlabel="x (mm)", ylabel="z (mm)", title=title)
    for a in ax.values():
        a.spines[["top", "right"]].set_visible(False)
    handles = [Patch(color=COLOR[k], label=label) for k, label in r["legend"].items()]
    handles += [Patch(facecolor="#9a9a9a", edgecolor="white", hatch="////", label=r.get("hatch_label"))] * bool(r["hatched"])
    handles += [Line2D([], [], marker="o", color=COLOR["mark"], linestyle="none", label="pass start")] * ("toolpaths" in r)
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    fig.suptitle(f"{process.title()} Print Plan", fontsize=14)
    fig.savefig(out / "preview.png", dpi=150)
    plt.close(fig)

    return {"process": process, "files": r["files"] | {"preview_png": str(out / "preview.png")}, "parameters": p,
            "layers": r["layers"], "printed_layers": r["printed_layers"]} | r["counts"] | {
            "estimated_print_time_min": round(r["minutes"], 1), "notes": r["notes"],
            "method": r["method"] + ". Parameters without a recipe value are defaults of the fabrication step"}
