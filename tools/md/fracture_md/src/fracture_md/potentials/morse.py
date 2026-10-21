"""Smooth Morse pair potential: a small deterministic *test fixture*.

    φ(r) = D [ (1 - e^{-a (r - r0)})² - 1 ] · S(r)
    S(r) = 1 - 10x³ + 15x⁴ - 6x⁵,  x = clip((r - r_on)/(r_cut - r_on), 0, 1)

S is C² continuous, so energies, forces and elastic constants are smooth and
finite-difference checks are clean.  Used for numerical tests that must not
depend on large weights or on the bond-order kernel:

* energy/force/stress consistency,
* the Cauchy relation C12 = C44 (central pair forces, every atom at an
  inversion centre, zero stress) for fcc crystals,
* cleavage-curve integration and area normalisation.

It is *not* a model of any ceramic; its card says so (kind=TEST_FIXTURE).
"""

from __future__ import annotations

import jax.numpy as jnp

from fracture_md.potentials.base import EnergyModel, params_sha256
from fracture_md.potentials.tersoff import neighbor_displacements, _safe_norm
from fracture_md.schema.potential import DomainAssessment, PotentialCard, PotentialKind, PropertyClass, TrainingDomain
from fracture_md.units import Unit, q


def _switch(r, r_on, r_cut):
    x = jnp.clip((r - r_on) / (r_cut - r_on), 0.0, 1.0)
    return 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x * x)


def morse_energy(frac, box, species, idx, p):
    d, mask, _ = neighbor_displacements(frac, box, idx)
    r = _safe_norm(d, mask)
    e = p["D"] * ((1.0 - jnp.exp(-p["a"] * (r - p["r0"]))) ** 2 - 1.0) * _switch(r, p["r_on"], p["r_cut"])
    return 0.5 * jnp.sum(jnp.where(mask, e, 0.0))


def morse_test_fixture(
    element: str = "X",
    D: float = 0.5,
    a: float = 1.6,
    r0: float = 2.5,
    r_on: float = 3.6,
    r_cut: float = 4.2,
) -> EnergyModel:
    params = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in dict(D=D, a=a, r0=r0, r_on=r_on, r_cut=r_cut).items()}
    h = params_sha256(params)
    card = PotentialCard(
        id=f"morse-test-fixture@{h[:10]}",
        name="Smooth Morse test fixture",
        kind=PotentialKind.TEST_FIXTURE,
        functional_form="pair Morse with C2 quintic switching",
        elements=[element],
        cutoff=q(r_cut, Unit.ANGSTROM),
        parameters_sha256=h,
        parameter_source="fracture_md defaults (arbitrary, deterministic)",
        license="MIT (fracture_md)",
        implementation="fracture_md.potentials.morse.morse_energy",
        training_domain=TrainingDomain(
            elements=[element], description="None: numerical test fixture, not a physical model."
        ),
        known_limitations=["Not a physical model of any material; use only for numerical checks."],
        # explicit (not "known_deficient"): indentation-machinery tests run on this fixture without
        # allow_out_of_domain; its records are never physical predictions
        domain_assessment=[
            DomainAssessment(property_class=pc, status="unvalidated",
                             notes="numerical test fixture, not a physical model")
            for pc in (PropertyClass.HIGH_PRESSURE, PropertyClass.PLASTICITY)
        ],
    )
    return EnergyModel(
        name="morse_test_fixture",
        elements=(element,),
        cutoff=r_cut,
        params=params,
        energy_fn=morse_energy,
        card=card,
    )
