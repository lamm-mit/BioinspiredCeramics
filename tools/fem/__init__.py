"""Finite-element crack propagation in a layered ceramic. A stand-in until the FEM tool is attached."""

import os

# JAX reserves 75% of GPU memory by default. On a unified-memory machine that memory also
# holds the served models, so allocate on demand instead.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from ..schema import number, spec


def crack_fea(layer_modulus_GPa, layer_thickness_um, layer_cleavage_energy_J_m2,
              interface_energy_J_m2, interface_strength_MPa, geometry_json):
    """Stand-in for a JAX finite-element model of a crack in a layered ceramic.

    geometry_json is the description that layered_geometry wrote. The real solver meshes the
    notched cell it describes. The stand-in only passes the path through.

    A crack that meets an interface between elastically similar layers deflects into it
    when the interface separation energy is below about a quarter of the energy to cleave
    the next layer [He and Hutchinson 1989]. The cohesive length E G / s^2 sets the mesh
    size the real solver needs. The real tool returns the crack path, the load-displacement
    curve and the work of fracture.
    """
    ratio = jnp.asarray(interface_energy_J_m2) / layer_cleavage_energy_J_m2
    cohesive_length = layer_modulus_GPa * 1e9 * interface_energy_J_m2 / (interface_strength_MPa * 1e6) ** 2
    return {"crack_deflects": bool(ratio < 0.25),
            "energy_ratio": round(float(ratio), 3),
            "criterion": "deflects when energy_ratio < 0.25 (He-Hutchinson, elastically similar layers)",
            "cohesive_length_over_layer_thickness": float(cohesive_length / (layer_thickness_um * 1e-6)),
            "device": str(jax.devices()[0]), "geometry_json": geometry_json,
            "method": "stand-in (He-Hutchinson deflection criterion), not an FEA result"}


SPEC = spec("crack_fea",
            "Finite-element crack propagation through a layered ceramic with cohesive interfaces. "
            "Reports whether a crack deflects along the interfaces.",
            {"layer_modulus_GPa": number("from the design"),
             "layer_thickness_um": number("from the design"),
             "layer_cleavage_energy_J_m2": number("separation_energy_J_m2 of md_interface with the ceramic "
                                                  "as both materials"),
             "interface_energy_J_m2": number("separation_energy_J_m2 of md_interface with the ceramic and "
                                             "the interlayer"),
             "interface_strength_MPa": number("strength_MPa of md_interface with the ceramic and the interlayer"),
             "geometry_json": {"type": "string", "description": "geometry_json of layered_geometry"}})
