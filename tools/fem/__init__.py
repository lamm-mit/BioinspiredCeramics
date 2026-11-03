"""Phase-field crack propagation in the notched cell of a layered ceramic.

The solver is jax-pfczm, vendored in jax_pfczm/. Importing it turns on float64 in JAX for the whole process.
"""

import csv
import json
import math
import os
from pathlib import Path

# JAX reserves 75% of GPU memory by default. On a unified-memory machine that memory also
# holds the served models, so allocate on demand instead.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from ..schema import number, spec
from .cell import LENGTH_PIXELS, POISSON, draw, simulate

# He and Hutchinson 1989: between elastically similar layers a crack deflects into the interface
# when the interface fracture energy is below about a quarter of that of the next layer.
HE_HUTCHINSON = 0.25
CRITERION = ("deflects when the simulated crack runs through interlayer more than one layer thickness sideways from "
             "the notch before it reaches the middle of the ceramic of the ply above the notched one. It penetrated when it "
             "reached that middle by breaking ceramic, and ran through a joint when it reached it through interlayer only. "
             "Neither of those counts as a deflection")
OUTCOME_WORDS = {"deflected": "deflected", "through joint": "ran through a joint", "penetrated": "penetrated"}


def read_cell(geometry_json, layer_thickness_um):
    """The notched cell of geometry_json, its section file, and the layer and interlayer thicknesses it was built from."""
    path = Path(geometry_json)
    if not path.is_file():
        raise FileNotFoundError(f"geometry_json {geometry_json} is not a file")
    geometry = json.loads(path.read_text())
    try:
        cell = {k: geometry["cell"][k] for k in ["size_mm", "ceramic_layers", "notch", "section"]}
        cell |= {k: geometry["design"][k] for k in ["layer_thickness_um", "interlayer_thickness_um"]}
    except (KeyError, TypeError) as e:
        raise ValueError(f"geometry_json has no notched cell, {e} is missing") from None
    if geometry.get("units") != "mm" or geometry.get("stack_axis") != "z" or cell["ceramic_layers"] < 3:
        raise ValueError("geometry_json must describe a cell in mm, stacked along z, with at least three ceramic layers")
    built = cell["layer_thickness_um"]
    if not math.isclose(built, layer_thickness_um, rel_tol=1e-6):
        raise ValueError(f"layer_thickness_um is {layer_thickness_um}, but geometry_json was built with {built}")
    return cell


def write_csv(path, sim, k_unit, k_layer, h_um, energy_unit):
    """One row per load step: the applied K, where the crack is, and the energies per unit thickness."""
    res = sim["result"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "K_MPa_sqrt_m", "K_over_K_layer", "crack_top_um", "along_interface_left_um",
                    "along_interface_right_um", "into_next_layer_um", "elastic_J_m", "dissipated_J_m", "external_work_J_m",
                    "stagger_iterations", "converged"])
        for k, m in enumerate(sim["track"]):
            K = res["load"][k] * k_unit
            lengths = [m[k] * h_um for k in ["top", "left", "right", "above"]]
            energies = [res[e][k] * energy_unit for e in ["elastic", "dissipated", "external_work"]]
            w.writerow([k + 1] + [f"{x:.5g}" for x in [K, K / k_layer] + lengths + energies]
                       + [int(res["n_stagger"][k]), bool(res["converged"][k])])


def fem_fracture(layer_modulus_GPa, layer_thickness_um, layer_fracture_energy_J_m2,
                 interface_fracture_energy_J_m2, interface_strength_MPa, geometry_json):
    """Phase-field simulation of a crack that grows from the notch of the cell into the ply above it.

    The section file of geometry_json becomes a 2D plane-strain x-z section on square pixels. Each pixel
    has the modulus of its ply. Interlayer pixels, between plies and in the joints of the pattern, have
    the interface fracture energy and the rest the layer fracture energy. The boundary follows the mode-I
    field of a crack at the notch tip, raised step by step, and the AT2 model grows the crack until it
    deflects, runs through a joint or penetrates. The interface strength enters only the cohesive length
    E G / s^2. At these sizes that length is far below a pixel, so the fracture is controlled by energy
    and the strength does not change the result. When the length exceeds a tenth of the layer thickness,
    method says that the strength would matter.

    Writes fem_damage.png, the damage field, and fem_load_path.csv, the load steps, next to geometry_json.
    """
    values = {k: v for k, v in locals().items() if k != "geometry_json"}
    for k, v in values.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{k} must be a positive number, got {v!r}")
    cell = read_cell(geometry_json, layer_thickness_um)
    ratio = interface_fracture_energy_J_m2 / layer_fracture_energy_J_m2
    sim = simulate(cell, ratio, layer_modulus_GPa)
    res, sec, last = sim["result"], sim["section"], sim["track"][-1]

    # Scaled units back to physical ones: K in sqrt(E G), energy per thickness in G h, length in h.
    E, G, h_um = layer_modulus_GPa * 1e9, layer_fracture_energy_J_m2, sec["pixel_mm"] * 1000
    k_unit, energy_unit = math.sqrt(E * G) / 1e6, G * h_um * 1e-6
    k_layer = math.sqrt(E * G / (1 - POISSON ** 2)) / 1e6
    cohesive_um = E * interface_fracture_energy_J_m2 / (interface_strength_MPa * 1e6) ** 2 * 1e6
    outcome, joint = sim["outcome"], sec["nearest_joint_px"]
    reached = sim["reached"]
    reached_K = None if reached is None else round(float(res["load"][reached] * k_unit), 4)
    final_K = float(res["load"][-1] * k_unit)

    folder = Path(geometry_json).parent
    csv_path, png_path = folder / "fem_load_path.csv", folder / "fem_damage.png"
    write_csv(csv_path, sim, k_unit, k_layer, h_um, energy_unit)
    draw(sim, png_path,
         f"energy ratio {ratio:.3g}, the crack {OUTCOME_WORDS[outcome]}, K = {final_K:.3g} MPa m^0.5 at the last step")
    rows, cols = sec["grid"].ny, sec["grid"].nx
    # A cohesive length well below the layer thickness puts the cell in the Griffith limit.
    scale = f"The cohesive length E G / s^2 is {cohesive_um:.3g} um against a pixel of {h_um:.3g} um, " + (
        "so the fracture is controlled by energy and the interface strength does not enter"
        if cohesive_um < 0.1 * layer_thickness_um else
        "which is not small against the layers, so the interface strength would matter, and AT2 leaves it out")
    # A crack that runs straight up a joint has not deflected, whatever the joint is made of.
    return {"crack_deflects": outcome == "deflected",
            "energy_ratio": round(ratio, 3),
            "criterion": CRITERION,
            "cohesive_length_over_layer_thickness": cohesive_um / layer_thickness_um,
            "he_hutchinson_deflects": bool(ratio < HE_HUTCHINSON),
            "outcome": outcome,
            "along_interface_left_um": round(last["left"] * h_um, 2),
            "along_interface_right_um": round(last["right"] * h_um, 2),
            "into_next_layer_um": round(last["above"] * h_um, 2),
            "nearest_joint_um": None if joint is None else round(joint * h_um, 2),
            "interface_reached_K_MPa_sqrt_m": reached_K,
            "interface_broke_ahead_of_crack": sim["band_broke"] is not None and sim["band_broke"] < (reached or 0),
            "layer_K_MPa_sqrt_m": round(k_layer, 4),
            "band_energy_ratio": round(sim["band_energy_ratio"], 3),
            "pixel_um": round(h_um, 3),
            "interlayer_band_um": round(sec["band_pixels"] * h_um, 3),
            "energy_balance": float(abs(res["external_work"][-1] - res["total"][-1]) / res["external_work"][-1]),
            "converged": bool(res["converged"].all()),
            "seconds": round(sim["wall_s"], 1),
            "device": sim["device"],
            "damage_png": str(png_path), "load_path_csv": str(csv_path), "geometry_json": geometry_json,
            "method": (f"phase-field fracture (AT2, jax-pfczm, l = {LENGTH_PIXELS:g} pixels) on a {rows} x {cols} pixel "
                       "plane-strain x-z section of the cell, sampled from its section file, under a mode-I K-field. "
                       f"{scale}. "
                       "The section is 2D, so the crack cannot twist out of its plane as it would between the rotated plies "
                       "of a Bouligand stack. A modulus gradient changes only the modulus of each ply, and the fracture "
                       "energy of the ceramic stays uniform")}


SPEC = spec("fem_fracture",
            "Finite-element phase-field crack propagation from the notch of the cell, on a section of its toolpath pattern. "
            "Reports whether the crack deflects along the interlayer, runs through a joint, or penetrates the next layer.",
            {"layer_modulus_GPa": number("from the design"),
             "layer_thickness_um": number("from the design"),
             "layer_fracture_energy_J_m2": number("cleavage_energy_J_m2 of dft_cleavage"),
             "interface_fracture_energy_J_m2": number("separation_energy_J_m2 of md_interface"),
             "interface_strength_MPa": number("strength_MPa of md_interface"),
             "geometry_json": {"type": "string", "description": "geometry_json of the geometry"}})
