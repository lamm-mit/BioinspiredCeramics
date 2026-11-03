"""Multi-species Abell–Tersoff–Brenner bond-order potentials in pure JAX.

JAX-MD 0.2.29 ships a Tersoff potential but it raises NotImplementedError for
more than one species, so this module implements the multi-component form

    E = 1/2 Σ_i Σ_{j≠i} f_c(r_ij) [ A_ij e^{-λ_ij r_ij} - b_ij B_ij e^{-μ_ij r_ij} ]
    b_ij = χ_ij (1 + (β_ij ζ_ij)^{n_ij})^{-1/(2 n_ij)}
    ζ_ij = Σ_{k≠i,j} f_c(r_ik) γ_ijk g_ijk(cos θ_ijk) exp(λ3_ik (r_ij - r_ik))
    g    = 1 + c²/d² - c²/(d² + (h - cos θ)²)  =  1 + (c/d)² x²/(d² + x²),  x = h - cos θ
    f_c  = ½ (1 + cos(π (r - R1)/(R2 - R1)))  on R1 < r < R2

with parameter tables indexed like LAMMPS ``pair_style tersoff`` entries (i, j, k):

* "tersoff" style (Tersoff 1989): β, n, c, d, h depend on atom i only (γ_ijk = 1);
  identical to matscipy's ``TersoffBrenner``.
* "brenner" style (Erhart–Albe 2005): pair parameters from (D0, r0, S, β), β = n = 1 and
  b_ij = (1 + ζ_ij)^{-1/2}; the three-body parameters γ, c, d, h (and λ3 = 2μ) are indexed
  by the **i–k pair**, g uses (h + cos θ) [equivalently h -> -h in the form above].
  This is the convention of the code the parameters were fitted with (it reproduces
  Erhart & Albe, PRB 71, 035211 (2005), Table XIII: B2 SiC ΔE = 3.04 eV/atom, a0 = 2.668 Å,
  and the Table XIV "w/o" defect energies), of the authors' related papers (g_ik, 2μ_ik in
  PRB 66, 035205 (2002) eq. 4; JPCM 15, 5649 (2003) eq. 4; JPCM 18, 6585 (2006) eq. 5),
  and of LAMMPS/NIST/OpenKIM ``SiC_Erhart-Albe.tersoff``, Atomistica and atomicrex
  ("classic" mode).  ``three_body_index="ij"`` reproduces matscipy 1.3.0's
  ``explicit_forms.TersoffBrenner`` (γ, c, d, h indexed by the i–j pair), which differs
  whenever a homonuclear pair is inside the cutoff (antisites, compression beyond ~4 %,
  high temperature, amorphous or molten SiC); it is kept only as a named legacy variant.

Everything is expressed as lookup tables so one energy kernel serves both
styles, and all parameters are differentiable leaves of a pytree.

Numerics.  The kernel evaluates g in the cancellation-free form (right-hand side above):
for Erhart-Albe Si-C, c²/d² ≈ 2.3e6 while g - 1 ≈ 8.5 at the tetrahedral angle, so the
textbook form loses ~6 significant digits (harmless in float64; in float32 it left force
errors of 0.08 eV/Å on a 131k-atom slab, 2.3e-3 eV/Å with the stable form).  Kernels are
traced with ``jax_default_matmul_precision = "highest"`` (:func:`highest_matmul_precision`):
on GPUs the default lets float32 matmuls (the fractional -> real-space displacement
transform) run in TF32, which gave 0.4 eV/Å force errors on the GB10; float64 results are
unaffected.
"""

from __future__ import annotations

import functools
from math import sqrt

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from fracture_md.potentials.base import EnergyModel, params_sha256
from fracture_md.schema.common import Citation, ValidationEvidence
from fracture_md.schema.potential import (
    DomainAssessment,
    PotentialCard,
    PotentialKind,
    PropertyClass,
    TrainingDomain,
)
from fracture_md.units import Unit, q

# ----------------------------------------------------------------------------
# Parameter sets (copied from the primary references; matscipy layout:
# pair arrays are ordered [el0-el0, el0-el1, el1-el1]).
# ----------------------------------------------------------------------------
TERSOFF_PRB_39_5566_SI_C = {
    "__ref__": "J. Tersoff, Phys. Rev. B 39, 5566 (1989); erratum Phys. Rev. B 41, 3248 (1990)",
    "style": "tersoff",
    "el": ["C", "Si"],
    "A": [1.3936e3, sqrt(1.3936e3 * 1.8308e3), 1.8308e3],
    "B": [3.4674e2, sqrt(3.4674e2 * 4.7118e2), 4.7118e2],
    "chi": [1.0, 0.9776e0, 1.0],
    "lambda": [3.4879e0, (3.4879e0 + 2.4799e0) / 2, 2.4799e0],
    "mu": [2.2119e0, (2.2119e0 + 1.7322e0) / 2, 1.7322e0],
    "lambda3": [0.0, 0.0, 0.0],
    "beta": [1.5724e-7, 1.1000e-6],
    "n": [7.2751e-1, 7.8734e-1],
    "c": [3.8049e4, 1.0039e5],
    "d": [4.3484e0, 1.6217e1],
    "h": [-5.7058e-1, -5.9825e-1],
    "r1": [1.80, sqrt(1.80 * 2.70), 2.70],
    "r2": [2.10, sqrt(2.10 * 3.00), 3.00],
}

ERHART_PRB_71_035211_SIC = {
    "__ref__": "P. Erhart and K. Albe, Phys. Rev. B 71, 035211 (2005)",
    "style": "brenner",
    "el": ["C", "Si"],
    "D0": [6.00, 4.36, 3.24],
    "r0": [1.4276, 1.79, 2.232],
    "S": [2.167, 1.847, 1.842],
    "beta": [2.0099, 1.6991, 1.4761],
    "gamma": [0.11233, 0.011877, 0.114354],
    "c": [181.910, 273987.0, 2.00494],
    "d": [6.28433, 180.314, 0.81472],
    "h": [0.5556, 0.68, 0.259],
    "mu": [0.0, 0.0, 0.0],
    "n": [1.0, 1.0, 1.0],
    "r1": [1.85, 2.20, 2.68],
    "r2": [2.15, 2.60, 2.96],
}


def _pair_index(i: int, j: int, nel: int) -> int:
    return min(i + j * nel, j + i * nel) - min(i * (i + 1) // 2, j * (j + 1) // 2)


def tables_from_matscipy_style(p: dict, three_body_index: str = "ik") -> dict[str, jnp.ndarray]:
    """Convert a matscipy-layout parameter dict into dense lookup tables.

    Returns pair tables [ne, ne] (A, B, lam, mu, chi, R1, R2, lambda3, beta, n)
    and triplet tables [ne, ne, ne] (c, d, h and, for the "brenner" style with
    ``three_body_index="ik"``, gamma3) indexed (i, j, k).

    ``three_body_index`` only affects the "brenner" (Erhart-Albe) style: "ik" (default)
    is the published/fitting-code convention, "ij" the matscipy 1.3.0 explicit_forms one.
    """
    if three_body_index not in ("ik", "ij"):
        raise ValueError(f"three_body_index must be 'ik' or 'ij', got {three_body_index!r}")
    style = p["style"].lower()
    el = p["el"]
    ne = len(el)
    P = lambda key: np.array([[p[key][_pair_index(i, j, ne)] for j in range(ne)] for i in range(ne)], float)  # noqa: E731
    E = lambda key: np.array([p[key][i] for i in range(ne)], float)  # noqa: E731

    if style == "tersoff":
        A, B, lam, mu, chi = P("A"), P("B"), P("lambda"), P("mu"), P("chi")
        lam3 = P("lambda3")
        beta_e, n_e = E("beta"), E("n")
        beta = np.repeat(beta_e[:, None], ne, axis=1)  # depends on i only
        n = np.repeat(n_e[:, None], ne, axis=1)
        c3 = np.broadcast_to(E("c")[:, None, None], (ne, ne, ne)).copy()
        d3 = np.broadcast_to(E("d")[:, None, None], (ne, ne, ne)).copy()
        h3 = np.broadcast_to(E("h")[:, None, None], (ne, ne, ne)).copy()
    elif style == "brenner":
        D0, S, r0, beta_b, mu_b = P("D0"), P("S"), P("r0"), P("beta"), P("mu")
        lam = beta_b * np.sqrt(2.0 * S)
        mu = beta_b * np.sqrt(2.0 / S)
        A = D0 / (S - 1.0) * np.exp(lam * r0)
        B = S * D0 / (S - 1.0) * np.exp(mu * r0)
        chi = np.ones((ne, ne))
        lam3 = 2.0 * mu_b  # indexed (i, k) by the kernel
        n = np.ones((ne, ne))
        extra = {}
        if three_body_index == "ik":
            # γ, c, d, h of the i-k pair: table[i, j, k] = X[pair(i, k)]; β = 1 so b = (1 + ζ)^(-1/2)
            beta = np.ones((ne, ne))
            extra["gamma3"] = np.broadcast_to(P("gamma")[:, None, :], (ne, ne, ne)).copy()
            c3 = np.broadcast_to(P("c")[:, None, :], (ne, ne, ne)).copy()
            d3 = np.broadcast_to(P("d")[:, None, :], (ne, ne, ne)).copy()
            h3 = -np.broadcast_to(P("h")[:, None, :], (ne, ne, ne)).copy()
        else:
            # legacy (matscipy explicit_forms): γ, c, d, h of the i-j pair; γ folded into β (n = 1)
            beta = P("gamma")  # (1 + γ ζ)^(-1/2)  ==  (1 + (β ζ)^n)^(-1/2n) with β=γ, n=1
            c3 = np.broadcast_to(P("c")[:, :, None], (ne, ne, ne)).copy()
            d3 = np.broadcast_to(P("d")[:, :, None], (ne, ne, ne)).copy()
            h3 = -np.broadcast_to(P("h")[:, :, None], (ne, ne, ne)).copy()
    else:
        raise ValueError(f"unknown style {style}")

    t = dict(A=A, B=B, lam=lam, mu=mu, chi=chi, R1=P("r1"), R2=P("r2"), lambda3=lam3, beta=beta, n=n, c=c3, d=d3, h=h3)
    if style == "brenner":
        t.update(extra)
    return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in t.items()}


def to_lammps_tersoff(p: dict[str, jnp.ndarray], elements: list[str] | tuple[str, ...]) -> str:
    """Write harness tables as a LAMMPS ``pair_style tersoff`` file (m = 1, entries i j k).

    Exact for tables produced by :func:`tables_from_matscipy_style` whose three-body part
    can be written per (i, j, k) entry (both Erhart-Albe conventions and Tersoff 1989 with
    n, β of atom i).  Used for cross-checks against LAMMPS and ``ase.calculators.tersoff``.
    """
    q = {k: np.asarray(v) for k, v in p.items()}
    ne = len(elements)
    if not np.allclose(q["chi"], 1.0):
        # LAMMPS has no χ: fold it into B (b_ij χ_ij B_ij)
        q["B"] = q["B"] * q["chi"]
    g3 = q.get("gamma3", np.ones((ne, ne, ne)))
    lines = ["# generated by fracture_md.potentials.tersoff.to_lammps_tersoff",
             "# e1 e2 e3 m gamma lambda3 c d costheta0 n beta lambda2 B R D lambda1 A"]
    for i in range(ne):
        for j in range(ne):
            for k in range(ne):
                R = 0.5 * (q["R1"][i, k] + q["R2"][i, k])
                D = 0.5 * (q["R2"][i, k] - q["R1"][i, k])
                if j == k:
                    two = [q["n"][i, j], q["beta"][i, j], q["mu"][i, j], q["B"][i, j], R, D, q["lam"][i, j], q["A"][i, j]]
                else:
                    two = [0.0, 0.0, 0.0, 0.0, R, D, 0.0, 0.0]
                vals = [1.0, g3[i, j, k], q["lambda3"][i, k], q["c"][i, j, k], q["d"][i, j, k], q["h"][i, j, k]] + two
                lines.append(f"{elements[i]} {elements[j]} {elements[k]} " + " ".join(repr(float(v)) for v in vals))
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Energy kernel
# ----------------------------------------------------------------------------
def highest_matmul_precision(energy_fn):
    """Trace ``energy_fn`` with ``jax_default_matmul_precision = "highest"``.

    The setting is read when each dot is traced, so it applies to the forward pass and to
    every derivative (JVP/transpose) of the wrapped kernel, whatever the caller's global
    setting.  It changes nothing in float64; in float32 on GPUs it prevents TF32 matmuls.
    """

    @functools.wraps(energy_fn)
    def wrapped(*args, **kwargs):
        with jax.default_matmul_precision("highest"):
            return energy_fn(*args, **kwargs)

    return wrapped


def _cutoff_fn(r, r1, r2):
    x = jnp.clip((r - r1) / (r2 - r1), 0.0, 1.0)
    return 0.5 * (1.0 + jnp.cos(jnp.pi * x))


def neighbor_displacements(frac, box, idx):
    """Real-space displacement vectors R_j - R_i for a Dense neighbour list.

    Uses JAX-MD's periodic_general displacement so that gradients with respect
    to `frac` are real-space gradients and gradients with respect to `box` are
    exact affine-deformation derivatives.
    """
    n = frac.shape[0]
    disp, _ = space.periodic_general(box, fractional_coordinates=True)
    mask = idx < n
    j = jnp.where(mask, idx, 0)
    d = jax.vmap(lambda si, sjs: jax.vmap(lambda sj: disp(sj, si))(sjs))(frac, frac[j])
    return d, mask, j


def _safe_norm(d, mask):
    r2 = jnp.sum(d * d, axis=-1)
    r2 = jnp.where(mask, r2, 1.0)
    return jnp.sqrt(r2)


@highest_matmul_precision
def tersoff_energy(frac, box, species, idx, p):
    """Total energy (eV) of a periodic system; see module docstring."""
    d, mask, j = neighbor_displacements(frac, box, idx)
    r = _safe_norm(d, mask)
    ti = jnp.broadcast_to(species[:, None], idx.shape)
    tj = species[j]

    R1, R2 = p["R1"][ti, tj], p["R2"][ti, tj]
    fc = jnp.where(mask, _cutoff_fn(r, R1, R2), 0.0)

    # Three-body: axes (i, j-slot, k-slot)
    m = idx.shape[1]
    tk = jnp.broadcast_to(tj[:, None, :], (idx.shape[0], m, m))
    ti3 = jnp.broadcast_to(ti[:, :, None], tk.shape)
    tj3 = jnp.broadcast_to(tj[:, :, None], tk.shape)
    rr = r[:, :, None] * r[:, None, :]
    cos = jnp.sum(d[:, :, None, :] * d[:, None, :, :], axis=-1) / rr
    mask3 = mask[:, :, None] & mask[:, None, :] & (~jnp.eye(m, dtype=bool))[None]
    c, dd, h = p["c"][ti3, tj3, tk], p["d"][ti3, tj3, tk], p["h"][ti3, tj3, tk]
    x2 = (h - cos) ** 2
    g = 1.0 + (c / dd) ** 2 * (x2 / (dd**2 + x2))  # == 1 + c²/d² - c²/(d² + x²) without cancellation
    if "gamma3" in p:  # static (pytree structure): Erhart-Albe γ of the i-k pair inside the sum
        g = p["gamma3"][ti3, tj3, tk] * g
    lam3 = p["lambda3"][ti3, tk]
    expo = jnp.exp(lam3 * (r[:, :, None] - r[:, None, :]))
    fc_ik = jnp.broadcast_to(fc[:, None, :], tk.shape)
    zeta = jnp.sum(jnp.where(mask3, fc_ik * g * expo, 0.0), axis=-1)

    beta, n, chi = p["beta"][ti, tj], p["n"][ti, tj], p["chi"][ti, tj]
    bz = beta * zeta
    pos = bz > 0.0
    bz_safe = jnp.where(pos, bz, 1.0)
    bzn = jnp.where(pos, bz_safe**n, 0.0)  # double-where: finite gradient at ζ = 0
    b = chi * (1.0 + bzn) ** (-0.5 / n)

    A, B, lam, mu = p["A"][ti, tj], p["B"][ti, tj], p["lam"][ti, tj], p["mu"][ti, tj]
    v = fc * (A * jnp.exp(-lam * r) - b * B * jnp.exp(-mu * r))
    return 0.5 * jnp.sum(jnp.where(mask, v, 0.0))


# ----------------------------------------------------------------------------
# Potential cards
# ----------------------------------------------------------------------------
_TERSOFF_LIMITATIONS = [
    "Short, element-dependent cutoff (f_c between R1 and R2, 2.2-3.0 Å) placed between first and "
    "second neighbours.  In this harness's rigid cleavage curves every cross-cut bond is inside the "
    "switching interval at the traction peak, so peak tractions and UBER length scales are set by the "
    "cutoff function (flagged by the plausibility check).  Pastewka et al. report that unscreened "
    "Tersoff/Erhart-Albe crack tips in 3C-SiC do not break the crack-tip bond at all (spurious lattice "
    "trapping) and that surface energies are 27-60% below DFT-LDA (PRB 78, 161402(R) (2008); PRB 87, "
    "205410 (2013)).",
    "Only first neighbours interact in zincblende/wurtzite SiC, and there is no dihedral term, so "
    "hexagonal and cubic stacking are degenerate: stacking-fault, twin-boundary and polytype energy "
    "differences are ~0 and must not be used from this potential.",
    "No charge transfer or electrostatics; polar-surface energetics and reconstructions are unreliable.",
    "Surface energies of individual polar terminations need a chemical-potential reference; only the "
    "sum over the two cleavage faces is defined by a cleavage calculation.",
]


# Indentation-relevant classes (records.INDENTATION_DOMAIN_CLASSES): evidence from the MD-indentation de-risking
# study (outputs/indent_derisk/SYNTHESIS.md; potential-diagnostics eos_summary.json, reproduced by a verify agent).
_T89_INDENTATION_DOMAIN = [
    DomainAssessment(
        property_class=PropertyClass.HIGH_PRESSURE,
        status="known_deficient",
        notes="Compressive cutoff wall at 13.5-13.7 GPa hydrostatic (linear strain ~ -2 %): Si-Si second neighbours "
        "at a0/sqrt2 = 3.055 A enter the 2.7-3.0 A switching zone; Hertz p0 at the first artefact 10.9-13.8 GPa, "
        "far below indentation contact pressures (30-100 GPa).",
    ),
    DomainAssessment(
        property_class=PropertyClass.PLASTICITY,
        status="known_deficient",
        notes="Ideal shear strength rises with pressure (35.3 -> 39.2 GPa at 50 GPa; DFT: falls); intrinsic "
        "stacking-fault energy 0; an R = 30 A sphere pushed to 8 A leaves 98 defective atoms; secular heating 600x "
        "that of Vashishta at dt = 1 fs.  Excluded from indentation hardness (negative control only).",
    ),
]
_EA_INDENTATION_DOMAIN = [
    DomainAssessment(
        property_class=PropertyClass.HIGH_PRESSURE,
        status="known_deficient",
        notes="Compressive cutoff wall at 34.2-35.0 GPa hydrostatic in both three-body conventions (Si-Si second "
        "neighbours at 3.083 A enter R2 = 2.96 A), i.e. at the experimental hardness; B1 unreachable "
        "(B3->B1 at 349 GPa i-k).",
    ),
    DomainAssessment(
        property_class=PropertyClass.PLASTICITY,
        status="known_deficient",
        notes="Intrinsic stacking-fault energy 0; the twin shear strength (32.9 GPa, +19 % vs DFT-PBE 27.76; "
        "PW91 30.3 is a comparison) lies behind the compressive wall and cannot be reached under a tip; bond "
        "breaking is set by the cutoff function.  "
        "Allowed only as an OUT_OF_DOMAIN sensitivity bracket.",
    ),
]


def _tersoff_card(params, source: dict, name: str, ident: str, citations, limitations, domain_notes,
                  three_body_index: str = "ik", extra_domain=()):
    h = params_sha256(params)
    cutoff = float(np.max(np.asarray(params["R2"])))
    return PotentialCard(
        id=f"{ident}@{h[:10]}",
        name=name,
        kind=PotentialKind.BOND_ORDER,
        functional_form=("Abell-Tersoff-Brenner bond order (matscipy/Atomistica conventions); style=tersoff"
                         if source["style"] == "tersoff" else
                         "Abell-Tersoff-Brenner bond order (LAMMPS pair_style tersoff (i,j,k) entries); style=brenner; "
                         f"three-body gamma,c,d,h indexed by the {'i-k' if three_body_index == 'ik' else 'i-j'} pair"),
        elements=list(source["el"]),
        cutoff=q(cutoff, Unit.ANGSTROM),
        parameters_sha256=h,
        parameter_source=source["__ref__"] + " (values as tabulated in matscipy 1.3.0 "
        "calculators/manybody/explicit_forms/tersoff_brenner.py)",
        license="Parameters are published scientific data (no license restriction); "
        + ("reference implementation matscipy is GPL-2.0-or-later (test-time only, not redistributed)."
           if source["style"] == "tersoff" or three_body_index == "ij" else
           "reference implementations LAMMPS (GPL-2.0, stored test data only) and ase.calculators.tersoff (LGPL) are "
           "test-time only, not redistributed."),
        code_license="MIT (fracture_md)",
        implementation="fracture_md.potentials.tersoff.tersoff_energy",
        citations=citations,
        training_domain=TrainingDomain(
            elements=list(source["el"]),
            description=domain_notes,
            reference_method="empirical fit to experimental and ab initio data",
            includes_surfaces=False,
            includes_large_strain=False,
            includes_bond_breaking=False,
        ),
        domain_assessment=[
            DomainAssessment(
                property_class=PropertyClass.EQUILIBRIUM_BULK,
                status="partially_validated",
                notes="Fitted to lattice constants/cohesive energies; see tests and docs/POTENTIALS.md.",
            ),
            DomainAssessment(property_class=PropertyClass.ELASTIC, status="partially_validated"),
            DomainAssessment(
                property_class=PropertyClass.BOND_BREAKING,
                status="known_deficient",
                evidence=[
                    ValidationEvidence(
                        kind="literature_model_value",
                        description="Short-cutoff bond-order potentials overestimate bond-breaking forces.",
                        citation="Pastewka, Pou, Pérez, Gumbsch, Moseler, PRB 78, 161402(R) (2008)",
                    )
                ],
            ),
            DomainAssessment(property_class=PropertyClass.LARGE_STRAIN, status="known_deficient"),
            DomainAssessment(property_class=PropertyClass.STACKING_FAULTS, status="known_deficient"),
            DomainAssessment(property_class=PropertyClass.SURFACES, status="unvalidated"),
            *extra_domain,
        ],
        known_limitations=limitations,
        differentiable=True,
        precision="float64",
    )


def tersoff_1989_sic() -> EnergyModel:
    params = tables_from_matscipy_style(TERSOFF_PRB_39_5566_SI_C)
    card = _tersoff_card(
        params,
        TERSOFF_PRB_39_5566_SI_C,
        "Tersoff (1989) Si-C",
        "tersoff-1989-SiC",
        [
            Citation(key="Tersoff1989", text="J. Tersoff, Phys. Rev. B 39, 5566 (1989)", doi="10.1103/PhysRevB.39.5566"),
            Citation(key="Tersoff1990", text="J. Tersoff, Phys. Rev. B 41, 3248 (1990) (erratum)", doi="10.1103/PhysRevB.41.3248.2"),
        ],
        _TERSOFF_LIMITATIONS,
        "Si and C parameters from Tersoff's elemental fits (Si: PRB 38, 9902 (1988); C: PRL 61, 2879 (1988)); "
        "one Si-C mixing parameter chi=0.9776 fitted to the heat of formation of 3C-SiC.",
        extra_domain=_T89_INDENTATION_DOMAIN,
    )
    return EnergyModel(
        name="tersoff_1989_SiC",
        elements=tuple(TERSOFF_PRB_39_5566_SI_C["el"]),
        cutoff=card.cutoff.magnitude(),
        params=params,
        energy_fn=tersoff_energy,
        card=card,
    )


_EA_CITATIONS = [
    Citation(key="Erhart2005", text="P. Erhart and K. Albe, Phys. Rev. B 71, 035211 (2005)", doi="10.1103/PhysRevB.71.035211")
]
_EA_DOMAIN = ("Analytical bond-order potential fitted to DFT and experimental data for Si, C and SiC "
              "(structures, elastic constants, point defects).")


def erhart_albe_2005_sic() -> EnergyModel:
    """Erhart-Albe (2005) Si-C, parameter set Si-I, three-body parameters indexed by the i-k pair.

    Reproduces the paper's Table XIII (B2: ΔE = 3.04 eV/atom, a0 = 2.668 Å) and LAMMPS
    ``SiC_Erhart-Albe.tersoff`` (see tests/test_ea_three_body_convention.py).
    """
    params = tables_from_matscipy_style(ERHART_PRB_71_035211_SIC, three_body_index="ik")
    card = _tersoff_card(
        params,
        ERHART_PRB_71_035211_SIC,
        "Erhart-Albe (2005) Si-C",
        "erhart-albe-2005-SiC",
        _EA_CITATIONS,
        _TERSOFF_LIMITATIONS
        + [
            "Three-body parameters (gamma, c, d, h, 2mu) are indexed by the i-k pair, the convention of the "
            "authors' fitting code (reproduces PRB 71, 035211 Table XIII B2 and Table XIV 'w/o' values) and of "
            "LAMMPS/NIST/OpenKIM/Atomistica.  matscipy 1.3.0 explicit_forms indexes them by the i-j pair; the "
            "two agree only while no homonuclear pair is inside the cutoff (perfect or rattled 3C at |strain| < 4 %).",
        ],
        _EA_DOMAIN,
        three_body_index="ik",
        extra_domain=_EA_INDENTATION_DOMAIN,
    )
    return EnergyModel(
        name="erhart_albe_2005_SiC",
        elements=tuple(ERHART_PRB_71_035211_SIC["el"]),
        cutoff=card.cutoff.magnitude(),
        params=params,
        energy_fn=tersoff_energy,
        card=card,
    )


def erhart_albe_2005_sic_matscipy_ij() -> EnergyModel:
    """LEGACY: Erhart-Albe numbers with matscipy 1.3.0's i-j three-body indexing (pre-fix harness behaviour).

    Kept so that records made before the fix (card id erhart-albe-2005-SiC@504bf563e0) resolve to the model
    that produced them and for the matscipy cross-check.  Not the published potential when homonuclear
    pairs are inside the cutoff: B2 SiC ΔE = 2.93 eV/atom instead of the paper's 3.04 eV/atom; hydrostatic
    pressure at -4.5 % linear strain 392 GPa instead of 197 GPa; antisite formation energies +0.25/+0.23 eV.
    """
    params = tables_from_matscipy_style(ERHART_PRB_71_035211_SIC, three_body_index="ij")
    card = _tersoff_card(
        params,
        ERHART_PRB_71_035211_SIC,
        "Erhart-Albe (2005) Si-C [LEGACY matscipy i-j three-body indexing]",
        "erhart-albe-2005-SiC",  # same ident as before the fix: the hash (unchanged params) reproduces old card ids
        _EA_CITATIONS,
        _TERSOFF_LIMITATIONS
        + [
            "LEGACY variant: three-body parameters indexed by the i-j pair as in matscipy 1.3.0 explicit_forms. "
            "This is NOT the convention the parameters were fitted with; it deviates from the published potential "
            "whenever homonuclear (Si-Si within 2.96 A, C-C within 2.15 A) pairs are inside the cutoff: antisites, "
            "compression beyond ~4 % linear strain, hot/molten/amorphous SiC.  Use erhart_albe_2005_SiC.",
        ],
        _EA_DOMAIN,
        three_body_index="ij",
        extra_domain=[
            _EA_INDENTATION_DOMAIN[0].model_copy(update={"notes": _EA_INDENTATION_DOMAIN[0].notes + "  The legacy i-j "
                                                         "indexing doubles the artefact: P(-4.5 % linear strain) = "
                                                         "392 GPa vs 197 GPa for the published i-k form."}),
            _EA_INDENTATION_DOMAIN[1],
        ],
    )
    return EnergyModel(
        name="erhart_albe_2005_SiC_matscipy_ij",
        elements=tuple(ERHART_PRB_71_035211_SIC["el"]),
        cutoff=card.cutoff.magnitude(),
        params=params,
        energy_fn=tersoff_energy,
        card=card,
    )
