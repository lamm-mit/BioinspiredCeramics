"""Atomistic layer of the dental-composite workstream: MACE-MP-0 on filler phases and tooth mineral.

Phases (crystal structures from literature Wyckoff positions, see ``PHASES``):

* tetragonal ZrO2 (P4_2/nmc, #137)  - proxy for the 3Y-TZP / nano-zirconia filler
* monoclinic ZrO2 (P2_1/c, #14)
* alpha-quartz (P3_121, #152) and alpha-cristobalite (P4_12_12, #92) - *crystalline* proxies
  for the (amorphous) silica / glass fillers; fused silica is NOT computed here
* hydroxyapatite Ca10(PO4)6(OH)2 - enamel mineral.  The experimental P6_3/m structure has
  half-occupied OH sites; we use the ordered P6_3 (#173) model (all OH in a column pointing
  the same way), the standard 44-atom ordered hexagonal model of periodic calculations.

What is computed (all 0 K lattice statics, MACE-MP-0 small via MACE-JAX, float64):

1. zero-stress relaxation (symmetry-constrained ASE LBFGS on cell + positions, then joint Newton steps
   with the exact q = 0 Hessian);
2. clamped- and relaxed-ion elastic tensors from *exact* autodiff second derivatives.  The
   internal-relaxation Hessian is the Gamma-point (q = 0) Hessian of the supercell:
   periodic displacement patterns of the n_prim basis atoms only (3 n_prim Hessian-vector
   products instead of 3N).  For homogeneous strain the internal relaxation has the
   periodicity of the primitive cell, so this is exact (verified against the harness'
   full-Hessian ``physics.elastic.elastic_constants_autodiff`` in the tests) and ~N/n_prim
   times cheaper - the harness full-Hessian path does not fit in GPU memory for MACE at the
   >= 200-atom cells MACE's 6 Å cutoff requires;
3. finite-difference cross-checks of C (clamped: stress differences; relaxed: stress
   differences after Newton relaxation of the internal coordinates at each strained cell);
4. VRH polycrystal moduli (``physics.elastic.analyze_elastic``) and MACE/reference ratios;
5. relaxed surface energies from stoichiometric slabs built from the relaxed bulk
   (``structures.slabs.oriented_bulk_cell``) by a *molecular-unit window cut*: PO4 and OH
   groups are never broken, every unit appears exactly once, so each slab is exactly
   n_periods bulk periods (stoichiometric by construction); the cut height is chosen by an
   unrelaxed-energy scan over all distinct cut positions; gamma = (E_slab - N e_bulk)/(2A)
   is the average of the (complementary) top and bottom terminations.  Hydroxylation /
   water adsorption and surface reconstructions are neglected.
6. (not implemented yet) rigid-cleavage traction-separation curve for PF-CZM filler cracking.

Validation: MACE-MP-0's card marks elastic properties *unvalidated* and surfaces
*unvalidated*; comparisons with experiment/DFT are attached as ValidationEvidence with
``passed`` = |MACE/ref - 1| <= tolerance, so records only become VALIDATED_* when they
really agree.  :func:`filler_properties` returns, per filler, the MACE value, the literature
value and the value *recommended for micromechanics* (the literature value; MACE is the
cross-check), with an uncertainty.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from fracture_md.units import AMU_PER_A3_TO_G_PER_CM3, EV_PER_A2_TO_J_PER_M2, EV_PER_A3_TO_GPA, Unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "dental" / "atomistic_constituents.json"
SKIN = 0.5  # Å, neighbour-list skin (harness default)

# =============================================================================
# Literature reference values (only values with a stated source; see docs/dental/ATOMISTIC.md)
# =============================================================================
# Literature values were checked against the sources (or the named tabulation) on 2026-09-28;
# "verified" says how: "source" = number seen in the cited paper/table, "secondary" = seen in the
# named tabulation only.
PABST_2004 = ("W. Pabst, G. Ticha, E. Gregorova, Ceramics-Silikaty 48, 41 (2004), "
              "https://www.ceramics-silikaty.cz/2004/pdf/2004_02_041.pdf")
PABST_2013 = ("W. Pabst, E. Gregorova, Ceramics-Silikaty 57, 167 (2013), "
              "https://www.ceramics-silikaty.cz/2013/pdf/2013_03_167.pdf")
FADDA_2009 = "G. Fadda, L. Colombo, G. Zanzotto, Phys. Rev. B 79, 214102 (2009), doi:10.1103/PhysRevB.79.214102"
LIN_HEINZ_2016 = "T.-J. Lin, H. Heinz, J. Phys. Chem. C 120, 4975 (2016), doi:10.1021/acs.jpcc.5b12504 (arXiv:1512.00122)"
REF_T_ZRO2_KISI = ("E. H. Kisi, C. J. Howard, J. Am. Ceram. Soc. 81, 1682 (1998), doi:10.1111/j.1151-2916.1998.tb02533.x "
                   "(12 mol% CeO2-TZP, neutron diffraction under load), as tabulated in " + PABST_2004 + " Table 3 and "
                   + FADDA_2009 + " Table V")
REF_HOWARD_1988 = "C. J. Howard, R. J. Hill, B. E. Reichert, Acta Cryst. B44, 116 (1988), doi:10.1107/S0108768187010279"
REF_MZRO2_CHAN = ("S.-K. Chan et al., J. Am. Ceram. Soc. 74, 1742 (1991), doi:10.1111/j.1151-2916.1991.tb07177.x "
                  "(Brillouin), as tabulated in " + FADDA_2009 + " Table VI")
REF_QUARTZ_LEVIEN = "L. Levien, C. T. Prewitt, D. J. Weidner, Am. Mineral. 65, 920 (1980)"
REF_QUARTZ_HEYLIGER = ("P. Heyliger, H. Ledbetter, S. Kim, J. Acoust. Soc. Am. 114, 644 (2003), natural quartz, "
                       "resonant ultrasound (Table II)")
REF_CRIST_DOWNS = "R. T. Downs, D. C. Palmer, Am. Mineral. 79, 9 (1994)"
REF_CRIST_YEGANEH = ("A. Yeganeh-Haeri, D. J. Weidner, J. B. Parise, Science 257, 650 (1992), "
                     "doi:10.1126/science.257.5070.650 (Brillouin), as tabulated in " + PABST_2013 + " Tables 10, 12")
REF_HAP_HUGHES = "J. M. Hughes, M. Cameron, K. D. Crowley, Am. Mineral. 74, 870 (1989) (Holly Springs HAp)"
REF_HAP_KATZ = ("J. L. Katz, K. Ukraincik, J. Biomech. 4, 221 (1971), doi:10.1016/0021-9290(71)90007-8, as tabulated "
                "in " + LIN_HEINZ_2016 + " (derived from polycrystal ultrasonics + fluorapatite single-crystal data)")
REF_HAP_CHING = ("W. Y. Ching, P. Rulis, A. Misra, Acta Biomater. 5, 3067 (2009), doi:10.1016/j.actbio.2009.04.030 "
                 "(DFT-PBE/PAW), as tabulated in " + LIN_HEINZ_2016)
REF_HAP_ASTALA = ("R. Astala, M. J. Stott, Phys. Rev. B 78, 075427 (2008) (DFT-PBE), as quoted by Wang et al., "
                  "RSC Adv. 2018 (PMC9077697)")
REF_HAP_CORNO = ("M. Corno, A. Rimola, V. Bolis, P. Ugliengo, Phys. Chem. Chem. Phys. 12, 6309 (2010) (DFT-B3LYP), "
                 "as tabulated in " + LIN_HEINZ_2016)
REF_ZRO2_CC = ("A. Christensen, E. A. Carter, Phys. Rev. B 58, 8050 (1998) (DFT), t-ZrO2 fluorite-(111) = "
               "P4_2/nmc (101), as tabulated in arXiv:2111.05168 SI Table S2")
REF_QUARTZ_FEYA = ("O. D. Feya, Q. Wang, S. V. Lepeshkin, V. S. Baturin, Y. A. Uspenskii, A. R. Oganov, Sci. Rep. 8, "
                   "11947 (2018) (DFT-PBE), alpha-quartz (001) Table 1")

#: Reference data used for MACE/reference ratios (ValidationEvidence ``kind`` given per entry).
REFERENCES: dict[str, dict[str, Any]] = {
    "t-ZrO2": {
        "lattice_A": {"a": 3.6055, "c": 5.1797},
        "lattice_source": REF_HOWARD_1988 + " (Zr0.935Y0.065O1.968, 295 K)", "lattice_verified": "source (abstract)",
        "Cij_GPa": {"C11": 327.0, "C33": 264.0, "C44": 59.0, "C12": 100.0, "C13": 62.0, "C66": 64.0},
        "Cij_source": REF_T_ZRO2_KISI, "Cij_kind": "experimental_reference", "Cij_verified": "secondary (two tabulations)",
        "vrh": {"E": 201.1, "G": 78.7, "K": 150.6, "nu": 0.278},
        "vrh_source": PABST_2004 + " Tables 6-7 (VRH of the Kisi & Howard constants)", "vrh_kind": "experimental_reference",
        "polycrystal": {"E": 210.0, "G": 80.0, "K": 184.0, "nu": 0.31},
        "polycrystal_source": PABST_2004 + " Table 9 (recommended 3Y-TZP; NIST-compiled E 210 +- 6 GPa, nu 0.31 +- 0.01, Table 8)",
        "surface_energy_J_m2": {"(101)": 1.24}, "surface_energy_source": REF_ZRO2_CC,
        "surface_energy_kind": "dft_reference", "surface_energy_verified": "secondary",
    },
    "m-ZrO2": {
        "lattice_A": {"a": 5.1505, "b": 5.2116, "c": 5.3173, "beta_deg": 99.230},
        "lattice_source": REF_HOWARD_1988 + " (295 K, neutron)", "lattice_verified": "source (abstract)",
        "Cij_full_GPa": {"C11": 361.0, "C22": 408.0, "C33": 258.0, "C44": 99.9, "C55": 81.2, "C66": 126.0, "C12": 142.0,
                         "C13": 55.0, "C15": -21.3, "C23": 196.0, "C25": 31.2, "C35": -18.2, "C46": -22.7},
        "Cij_source": REF_MZRO2_CHAN, "Cij_kind": "experimental_reference", "Cij_verified": "secondary",
        "Cij_note": "frame convention of the monoclinic tensor not checked: only rotation-invariant VRH moduli are compared",
    },
    "alpha-quartz": {
        "lattice_A": {"a": 4.9160, "c": 5.4054}, "lattice_source": REF_QUARTZ_LEVIEN + " (1 atm, Table 5)",
        "lattice_verified": "source (OCR)",
        "Cij_GPa": {"C11": 87.26, "C33": 105.8, "C44": 57.15, "C66": 40.35, "C12": 6.57, "C13": 11.95, "|C14|": 17.18},
        "Cij_source": REF_QUARTZ_HEYLIGER, "Cij_kind": "experimental_reference", "Cij_verified": "source",
        "vrh": {"E": 95.6, "G": 44.1, "K": 38.3, "nu": 0.084},
        "vrh_source": PABST_2013 + " Table 16 (recommended low-quartz, experiment)", "vrh_kind": "experimental_reference",
        "surface_energy_J_m2": {"(001)": 2.183}, "surface_energy_kind": "dft_reference", "surface_energy_verified": "source",
        "surface_energy_source": REF_QUARTZ_FEYA + ": O-terminated cleaved (001) 2.183 J/m2 (the reconstructed 'dense' "
                                 "(001) is 0.413 J/m2 and is not reachable by local relaxation)",
    },
    "alpha-cristobalite": {
        "lattice_A": {"a": 4.9717, "c": 6.9223}, "lattice_source": REF_CRIST_DOWNS + " (room pressure)",
        "lattice_verified": "source",
        "Cij_GPa": {"C11": 59.4, "C33": 42.4, "C44": 67.2, "C66": 25.7, "C12": 3.8, "C13": -4.4},
        "Cij_source": REF_CRIST_YEGANEH, "Cij_kind": "experimental_reference", "Cij_verified": "secondary",
        "vrh": {"E": 65.2, "G": 39.1, "K": 16.4, "nu": -0.164},
        "vrh_source": PABST_2013 + " Table 12 (VRH of the Yeganeh-Haeri constants)", "vrh_kind": "experimental_reference",
    },
    "hydroxyapatite": {
        "lattice_A": {"a": 9.4166, "c": 6.8745}, "lattice_source": REF_HAP_HUGHES + " Table 1", "lattice_verified": "source",
        "Cij_GPa": {"C11": 137.0, "C33": 172.0, "C44": 39.6, "C12": 42.5, "C13": 54.9},
        "Cij_source": REF_HAP_KATZ, "Cij_kind": "experimental_reference", "Cij_verified": "secondary",
        "vrh": {"E": 114.0, "G": 44.5, "K": 89.0},
        "vrh_source": REF_HAP_KATZ + " (isotropic E, G, K)", "vrh_kind": "experimental_reference",
        "Cij_dft_GPa": {"C11": 140.0, "C33": 174.8, "C44": 47.5, "C12": 42.4, "C13": 58.3},
        "vrh_dft": {"E": 120.6, "G": 47.7, "K": 84.5}, "Cij_dft_source": REF_HAP_CHING,
        "surface_energy_J_m2": {"(001)": 1.29, "(010)": 1.76}, "surface_energy_source": REF_HAP_ASTALA,
        "surface_energy_kind": "dft_reference", "surface_energy_verified": "secondary",
        "surface_energy_other": {
            "B3LYP (Corno 2010)": {"(001)": 1.058, "(010)": 1.709, "source": REF_HAP_CORNO},
            "INTERFACE FF (Lin & Heinz 2016), per surface, PO4-terminated, neat": {
                "(001)": 0.870, "(010)": 1.080, "(101)": 0.980, "source": LIN_HEINZ_2016 + " Table 6, SI Eq. S3"},
            "INTERFACE FF, protonated pH ~5": {"(001)": 0.220, "(010)": 0.320, "source": LIN_HEINZ_2016 + " Fig. 5"},
        },
    },
    "amorphous-silica": {
        "vrh": {"E": 72.2, "G": 30.9, "K": 36.3, "nu": 0.168},
        "vrh_source": PABST_2013 + " Table 16 (recommended silica glass)", "vrh_kind": "experimental_reference",
        "verified": "source",
    },
}

# =============================================================================
# Structures
# =============================================================================


@dataclasses.dataclass(frozen=True)
class PhaseSpec:
    name: str
    formula: str
    space_group_number: int  # of the (ordered) model actually simulated
    space_group_symbol: str
    crystal_system: str
    builder: Callable[[], Any]
    source: str
    notes: str = ""
    formula_units_per_cell: int = 1


def build_t_zro2(a: float = 3.6055, c: float = 5.1797, z_O: float = 0.455):
    """t-ZrO2, P4_2/nmc (#137, origin choice 2): Zr 2a (3/4,1/4,3/4), O 4d (1/4,1/4,z); 6-atom cell.

    Starting geometry only (a, c ~ room-temperature 3Y-TZP-like values, O column offset
    d_z = |z - 1/2| ~ 0.045); every quantity is recomputed after MACE relaxation.
    """
    from ase.spacegroup import crystal

    return crystal(["Zr", "O"], basis=[(0.75, 0.25, 0.75), (0.25, 0.25, z_O)], spacegroup=137, setting=2,
                   cellpar=[a, a, c, 90, 90, 90])


def build_m_zro2():
    """m-ZrO2 (baddeleyite), P2_1/c (#14), 12 atoms; Howard, Hill & Reichert (1988) cell and positions
    (positions as tabulated by Fadda et al. 2009, Table VI)."""
    from ase.spacegroup import crystal

    return crystal(["Zr", "O", "O"], basis=[(0.2754, 0.0395, 0.2083), (0.0700, 0.3317, 0.3447), (0.4496, 0.7569, 0.4792)],
                   spacegroup=14, cellpar=[5.1505, 5.2116, 5.3173, 90, 99.230, 90])


def build_alpha_quartz():
    """alpha-quartz, P3_121 (#152), 9 atoms: Si 3a (x,0,1/3), O 6c (x,y,z).

    Levien et al. (1980) coordinates (Si x = 0.4697; O 0.4135, 0.2669, 0.1191 in their setting)
    expressed in the ITA P3_121 setting (O z = 1/3 - 0.1191); verified by the resulting
    Si-O bond lengths 1.605/1.614 Å and 4-fold Si coordination (tests).
    """
    from ase.spacegroup import crystal

    return crystal(["Si", "O"], basis=[(0.4697, 0.0, 1.0 / 3.0), (0.4135, 0.2669, 1.0 / 3.0 - 0.1191)], spacegroup=152,
                   cellpar=[4.9160, 4.9160, 5.4054, 90, 90, 120])


def build_alpha_cristobalite():
    """alpha-cristobalite, P4_12_12 (#92), 12 atoms: Si 4a (u,u,0), O 8b (Downs & Palmer 1994, room pressure)."""
    from ase.spacegroup import crystal

    return crystal(["Si", "O"], basis=[(0.30028, 0.30028, 0.0), (0.2392, 0.1044, 0.1787)], spacegroup=92,
                   cellpar=[4.9717, 4.9717, 6.9223, 90, 90, 90])


def build_hydroxyapatite():
    """Hydroxyapatite Ca10(PO4)6(OH)2, 44 atoms, ordered P6_3 (#173) model.

    Framework (Ca1 4f, Ca2 6h, P 6h, O1 6h, O2 6h, O3 12i) and cell from the P6_3/m refinement of
    Hughes, Cameron & Crowley (1989, Tables 1-2); the half-occupied 4e O(H) (z 0.1979) and H
    (z 0.04 +- 0.02) sites are ordered with both OH of the c-axis channel pointing along -c
    (O at z, z + 1/2; H at z, z + 1/2), giving the P6_3 ordered-column model.
    """
    from ase import Atom
    from ase.spacegroup import crystal

    at = crystal(["Ca", "Ca", "P", "O", "O", "O"],
                 basis=[(1 / 3, 2 / 3, 0.00144), (-0.00657, 0.24706, 0.25), (0.36860, 0.39866, 0.25), (0.4850, 0.3289, 0.25),
                        (0.4649, 0.5871, 0.25), (0.2580, 0.3435, 0.0703)],
                 spacegroup=176, cellpar=[9.4166, 9.4166, 6.8745, 90, 90, 120])
    for zO, zH in [(0.1979, 0.04), (0.6979, 0.54)]:
        at.append(Atom("O", at.cell.cartesian_positions([0.0, 0.0, zO])))
        at.append(Atom("H", at.cell.cartesian_positions([0.0, 0.0, zH])))
    return at


PHASES: dict[str, PhaseSpec] = {
    "t-ZrO2": PhaseSpec("t-ZrO2", "ZrO2", 137, "P4_2/nmc", "tetragonal", build_t_zro2,
                        "P4_2/nmc (Zr 2a, O 4d); cell of " + REF_HOWARD_1988,
                        "undoped tetragonal ZrO2 as a proxy for 3Y-TZP (no Y, no O vacancies)", 2),
    "m-ZrO2": PhaseSpec("m-ZrO2", "ZrO2", 14, "P2_1/c", "monoclinic", build_m_zro2, REF_HOWARD_1988, "", 4),
    "alpha-quartz": PhaseSpec("alpha-quartz", "SiO2", 152, "P3_121", "trigonal", build_alpha_quartz, REF_QUARTZ_LEVIEN,
                              "crystalline proxy; dental silica/glass fillers are amorphous", 3),
    "alpha-cristobalite": PhaseSpec("alpha-cristobalite", "SiO2", 92, "P4_12_12", "tetragonal", build_alpha_cristobalite,
                                    REF_CRIST_DOWNS, "crystalline proxy; dental silica/glass fillers are amorphous", 4),
    "hydroxyapatite": PhaseSpec("hydroxyapatite", "Ca5(PO4)3(OH)", 173, "P6_3", "hexagonal", build_hydroxyapatite,
                                REF_HAP_HUGHES + " (framework); OH ordered (P6_3)",
                                "stoichiometric, ordered-OH model; enamel mineral is carbonated, Ca-deficient HAp", 2),
}

#: bonded groups that must not be broken by a surface cut
MOLECULAR_BONDS = {"hydroxyapatite": {("P", "O"): 1.8, ("O", "H"): 1.2}}


def phase_candidate(name: str):
    """REFERENCE Candidate for one constituent phase (records need a candidate)."""
    from fracture_md.schema.material import Candidate, CandidateKind, Composition, Phase

    spec = PHASES[name]
    return Candidate(
        id=f"dental-constituent:{name}", name=name, kind=CandidateKind.REFERENCE,
        composition=Composition.from_formula(spec.formula),
        phases=[Phase(name=name, space_group_symbol=spec.space_group_symbol, space_group_number=spec.space_group_number,
                      crystal_system=spec.crystal_system)],
        hypothesis="constituent of a ceramic-filled dental composite / tooth mineral (atomistic cross-check)",
        proposed_by="dental-atomistic",
    )


# =============================================================================
# Compiled kernels (energy, forces, strain gradient) and the q = 0 second derivatives
# =============================================================================


@functools.lru_cache(maxsize=None)
def _kernels(energy_fn):
    import jax
    import jax.numpy as jnp

    from fracture_md.physics.elastic import strained_box

    z6 = jnp.zeros(6)

    def e_of(frac, box, sp, idx, p, e):
        return energy_fn(frac, strained_box(box, e), sp, idx, p)

    def efs(frac, box, sp, idx, p):
        E, (gf, ge) = jax.value_and_grad(e_of, argnums=(0, 5))(frac, box, sp, idx, p, z6)
        return E, gf, ge

    def grads(frac, box, sp, idx, p, e):
        return jax.grad(e_of, argnums=(0, 5))(frac, box, sp, idx, p, e)

    def jvp_strain(frac, box, sp, idx, p, t_e):
        # d/de of (grad_R, grad_e) = (Lambda e_v, H_ee e_v): same derivative order as the harness'
        # _mixed_ue / _hess_ee (strain derivative of the position gradient).
        _, (dgf, dge) = jax.jvp(lambda e: grads(frac, box, sp, idx, p, e), (z6,), (t_e,))
        return dgf, dge

    def hvp_pos(frac, box, sp, idx, p, t_frac):
        # K v at fixed box (as the harness' _hvp_batch); tangent of frac = real-space displacement.
        # NOTE: differentiating grad_e (stress) w.r.t. positions through JAX-MD's fractional custom JVP
        # is wrong (docs/AUTODIFF.md) and for MACE even yields NaN, so positions and strain tangents
        # are never mixed in one JVP.
        return jax.jvp(lambda f: jax.grad(energy_fn, argnums=0)(f, box, sp, idx, p), (frac,), (t_frac,))[1]

    return {"efs": jax.jit(efs), "jvp_strain": jax.jit(jvp_strain), "hvp_pos": jax.jit(hvp_pos), "grads": jax.jit(grads)}


def energy_forces_stress(model, system, idx):
    """(E eV, forces (N,3) eV/Å, Voigt Cauchy stress (6,) GPa, positive = tensile)."""
    k = _kernels(model.energy_fn)["efs"]
    E, gf, ge = k(system.frac, system.box, model.species_for(system), idx, model.params)
    return float(E), -np.asarray(gf), np.asarray(ge) / system.volume * EV_PER_A3_TO_GPA


@dataclasses.dataclass
class Q0Derivatives:
    K_cell: np.ndarray  # (3 n_p, 3 n_p) Gamma-point Hessian per primitive cell, eV/Å^2
    Lam_cell: np.ndarray  # (3 n_p, 6) mixed position-strain derivative per primitive cell, eV/Å
    H_ee: np.ndarray  # (6, 6) total strain Hessian of the supercell, eV
    n_cells: int
    n_prim: int
    volume: float  # supercell volume, Å^3
    energy: float
    forces: np.ndarray
    stress_GPa: np.ndarray
    seconds: float
    n_fd_fallback: int = 0  # JVP columns that were non-finite and were replaced by central differences


def q0_derivatives(model, system, idx, n_prim: int) -> Q0Derivatives:
    """Gamma-point second derivatives of a supercell made by ``prim.repeat(reps)`` (atom = t*n_prim + a).

    3 n_prim + 6 JVPs of the exact gradient; the per-cell K and Lambda are averaged over the
    n_cells translation images (they are identical by symmetry up to round-off).
    """
    import jax.numpy as jnp

    t0 = time.time()
    n = system.n_atoms
    if n % n_prim:
        raise ValueError("supercell atom count is not a multiple of n_prim")
    nc = n // n_prim
    kern = _kernels(model.energy_fn)
    sp = model.species_for(system)
    args = (system.frac, system.box, sp, idx, model.params)
    z6 = jnp.zeros(6)
    K = np.zeros((3 * n_prim, 3 * n_prim))
    L = np.zeros((3 * n_prim, 6))
    n_fb = 0
    h = 1e-4  # Å or strain, for the central-difference fallback (exact first derivatives)
    for b in range(n_prim):
        for beta in range(3):
            t = np.zeros((nc, n_prim, 3))
            t[:, b, beta] = 1.0
            dgf = np.asarray(kern["hvp_pos"](*args, jnp.asarray(t.reshape(n, 3))))
            if not np.all(np.isfinite(dgf)):
                # MACE-JAX occasionally returns non-finite forward-over-reverse derivatives (observed for a
                # few columns, dependent on the neighbour-list padding); fall back to central differences
                # of the exact gradient along the same periodic displacement.
                n_fb += 1
                dx = t.reshape(n, 3) * h
                binv = np.linalg.inv(np.asarray(system.box))
                gp = np.asarray(kern["grads"](jnp.asarray(np.asarray(system.frac) + dx @ binv.T), system.box, sp, idx,
                                              model.params, z6)[0])
                gm = np.asarray(kern["grads"](jnp.asarray(np.asarray(system.frac) - dx @ binv.T), system.box, sp, idx,
                                              model.params, z6)[0])
                dgf = (gp - gm) / (2 * h)
            K[:, 3 * b + beta] = dgf.reshape(nc, n_prim * 3).mean(axis=0)
    H = np.zeros((6, 6))
    for v in range(6):
        te = np.zeros(6)
        te[v] = 1.0
        dgf, dge = (np.asarray(x) for x in kern["jvp_strain"](*args, jnp.asarray(te)))
        if not (np.all(np.isfinite(dgf)) and np.all(np.isfinite(dge))):
            n_fb += 1
            gp = kern["grads"](*args, jnp.asarray(te * h))
            gm = kern["grads"](*args, jnp.asarray(-te * h))
            dgf = (np.asarray(gp[0]) - np.asarray(gm[0])) / (2 * h)
            dge = (np.asarray(gp[1]) - np.asarray(gm[1])) / (2 * h)
        L[:, v] = dgf.reshape(nc, n_prim * 3).mean(axis=0)
        H[:, v] = dge
    E, F, S = energy_forces_stress(model, system, idx)
    if not (np.all(np.isfinite(K)) and np.all(np.isfinite(L)) and np.all(np.isfinite(H))):
        raise FloatingPointError("non-finite second derivatives")
    return Q0Derivatives(0.5 * (K + K.T), L, 0.5 * (H + H.T), nc, n_prim, system.volume, E, F, S, time.time() - t0, n_fb)


def elastic_from_q0(d: Q0Derivatives) -> dict[str, Any]:
    """Clamped/relaxed-ion C (GPa), internal-strain response and the Gamma-point spectrum."""
    from fracture_md.physics.elastic import relaxed_ion_correction

    corr, Gamma, spec, Kinv = relaxed_ion_correction(d.n_cells * d.K_cell, d.n_cells * d.Lam_cell)
    C_cl = d.H_ee / d.volume * EV_PER_A3_TO_GPA
    C_rel = (d.H_ee - corr) / d.volume * EV_PER_A3_TO_GPA
    from fracture_md.physics.elastic import translation_projector

    P = translation_projector(d.n_prim)
    Kp = P @ (d.n_cells * d.K_cell) @ P
    w = np.linalg.eigvalsh(0.5 * (Kp + Kp.T))
    return {"C_clamped": 0.5 * (C_cl + C_cl.T), "C_relaxed": 0.5 * (C_rel + C_rel.T), "Gamma": Gamma, "spectrum": spec,
            "K_pinv_cell": Kinv * d.n_cells, "negative_eigenvalues_eV_A2": w[w < -spec.tol].tolist()}


# =============================================================================
# Relaxation
# =============================================================================


class _HarnessCalculator:
    """Minimal ASE calculator around the jitted MACE/harness kernels (energy, forces, stress)."""

    def __new__(cls, model, elements):
        from ase.calculators.calculator import Calculator, all_changes

        from fracture_md.potentials.neighbors import build_neighbors
        from fracture_md.structures.system import AtomicSystem

        class Calc(Calculator):
            implemented_properties = ["energy", "free_energy", "forces", "stress"]

            def __init__(self):
                super().__init__()
                self.nh = None
                self.n_calls = 0

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                s = AtomicSystem.from_ase(self.atoms, elements=elements)
                # allocate once; Dense updates recompute the list exactly and keep its shape (no recompilation)
                if self.nh is None:
                    self.nh = build_neighbors(s, model.cutoff, skin=SKIN, capacity_multiplier=1.6)
                else:
                    self.nh = self.nh.update(s.frac, box=s.box)
                E, F, S = energy_forces_stress(model, s, self.nh.idx)
                self.n_calls += 1
                # ASE stress convention: (1/V) dE/d(strain), eV/Å^3, Voigt xx,yy,zz,yz,xz,xy
                self.results = {"energy": E, "free_energy": E, "forces": F, "stress": S / EV_PER_A3_TO_GPA}

        return Calc()


def relax_bulk(model, atoms, *, fmax: float = 5e-4, steps: int = 400, symprec: float = 1e-3, logfile=None):
    """Symmetry-constrained zero-stress relaxation (ASE FixSymmetry + FrechetCellFilter + LBFGS).

    LBFGS (no dense 3N x 3N linear algebra) is used because BFGS' eigendecomposition dominates
    for >300 atoms on a loaded machine; the tight convergence comes from :func:`newton_polish`.
    """
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS

    a = atoms.copy()
    a.set_constraint(FixSymmetry(a, symprec=symprec))
    a.calc = _HarnessCalculator(model, model.elements)
    opt = LBFGS(FrechetCellFilter(a), logfile=logfile, maxstep=0.1)
    ok = bool(opt.run(fmax=fmax, steps=steps))
    a.set_constraint()
    return a, {"converged": ok, "steps": int(opt.nsteps), "n_calls": int(a.calc.n_calls), "fmax_target": fmax}


def newton_polish(model, system, idx, n_prim: int, d: Optional[Q0Derivatives] = None, *, max_iter: int = 3,
                  fmax: float = 2e-5, stress_tol_GPa: float = 2e-4):
    """Joint Newton steps on (periodic internal coordinates, strain) with the exact q=0 Hessian."""
    from fracture_md.physics.elastic import strained_box, translation_projector
    import jax.numpy as jnp

    s = system
    for _ in range(max_iter):
        if d is None:
            d = q0_derivatives(model, s, idx, n_prim)
        fm = float(np.max(np.linalg.norm(d.forces, axis=1)))
        smax = float(np.max(np.abs(d.stress_GPa)))
        if fm <= fmax and smax <= stress_tol_GPa:
            return s, d
        nc, npr = d.n_cells, n_prim
        g_u = -d.forces.reshape(nc, npr * 3).sum(axis=0)  # dE/du for a periodic displacement
        g_e = d.stress_GPa * d.volume / EV_PER_A3_TO_GPA
        P = translation_projector(npr)
        H = np.block([[nc * P @ d.K_cell @ P, nc * P @ d.Lam_cell], [nc * d.Lam_cell.T @ P, d.H_ee]])
        g = np.concatenate([P @ g_u, g_e])
        step = -np.linalg.lstsq(H, g, rcond=1e-10)[0]
        u = step[: 3 * npr].reshape(npr, 3)
        e = step[3 * npr:]
        x = np.asarray(s.positions) + np.tile(u, (nc, 1))
        box = np.asarray(strained_box(s.box, jnp.asarray(e)))
        x = x @ (np.eye(3) + _eps(e)).T
        s = s.replace(box=jnp.asarray(box), frac=jnp.asarray((x @ np.linalg.inv(box).T) % 1.0))
        d = None
    d = q0_derivatives(model, s, idx, n_prim)
    return s, d


def _eps(e):
    return np.array([[e[0], e[5] / 2, e[4] / 2], [e[5] / 2, e[1], e[3] / 2], [e[4] / 2, e[3] / 2, e[2]]])


# =============================================================================
# Finite-difference cross-checks
# =============================================================================


def fd_elastic(model, system, idx, n_prim: int, K_pinv_cell: np.ndarray, *, delta: float = 1e-3, relax: bool = True,
               fmax: float = 1e-6, max_iter: int = 25) -> tuple[np.ndarray, float]:
    """Central-difference C (GPa) from the autodiff stress at +-delta strains.

    relax=True: internal coordinates (periodic, q = 0) relaxed at each strained cell with
    quasi-Newton steps u = -K_cell^+ g_cell (true forces judge convergence).  Returns
    (C, max residual force eV/Å).
    """
    import jax.numpy as jnp

    from fracture_md.physics.elastic import strained_box

    C = np.zeros((6, 6))
    worst = 0.0
    nc = system.n_atoms // n_prim
    for v in range(6):
        sig = []
        for sgn in (+1, -1):
            e = np.zeros(6)
            e[v] = sgn * delta
            box = strained_box(system.box, jnp.asarray(e))
            s = system.replace(box=box)
            _, F, S = energy_forces_stress(model, s, idx)
            if relax:
                binv = np.linalg.inv(np.asarray(box))
                for _ in range(max_iter):
                    fm = float(np.max(np.linalg.norm(F, axis=1)))
                    if fm <= fmax:
                        break
                    g = -F.reshape(nc, n_prim * 3).mean(axis=0)
                    u = -(K_pinv_cell @ g).reshape(n_prim, 3)
                    x = np.asarray(s.positions) + np.tile(u, (nc, 1))
                    s = s.replace(frac=jnp.asarray(x @ binv.T))
                    _, F, S = energy_forces_stress(model, s, idx)
                worst = max(worst, float(np.max(np.linalg.norm(F, axis=1))))
            sig.append(S)
        C[:, v] = (sig[0] - sig[1]) / (2 * delta)
    return 0.5 * (C + C.T), worst


# =============================================================================
# Bulk runner
# =============================================================================


def _supercell(atoms, model):
    from fracture_md.structures.system import auto_supercell

    reps = auto_supercell(atoms, model.cutoff + SKIN)
    return atoms.repeat(reps), reps


def _vrh_of_full(cij: dict[str, float]) -> dict[str, float]:
    """VRH E, G, K, nu of a (possibly monoclinic) tensor given as named Voigt components."""
    from fracture_md.physics.elastic import analyze_elastic

    C = np.zeros((6, 6))
    for k, v in cij.items():
        i, j = int(k[1]) - 1, int(k[2]) - 1
        C[i, j] = C[j, i] = v
    an = analyze_elastic(C)
    return {"E": float(an["E_hill"]), "G": float(an["G_hill"]), "K": float(an["B_hill"]), "nu": float(an["nu_hill"])}


def _hex_or_tet_cij(C: np.ndarray) -> dict[str, float]:
    return {"C11": C[0, 0], "C22": C[1, 1], "C33": C[2, 2], "C44": C[3, 3], "C55": C[4, 4], "C66": C[5, 5],
            "C12": C[0, 1], "C13": C[0, 2], "C23": C[1, 2], "C14": C[0, 3], "C15": C[0, 4], "C16": C[0, 5],
            "C25": C[1, 4], "C35": C[2, 4], "C46": C[3, 5]}


def ratio_evidence(quantity: str, computed: float, reference: float, unit: Unit, citation: str, kind: str,
                   tol: float) -> Any:
    from fracture_md.schema.common import ValidationEvidence
    from fracture_md.units import Quantity

    r = computed / reference
    return ValidationEvidence(
        kind=kind,  # type: ignore[arg-type]
        description=f"{quantity}: MACE/reference = {r:.3f} (pass if within +-{tol:.0%})",
        reference_value=Quantity(value=reference, unit=unit), computed_value=Quantity(value=computed, unit=unit),
        relative_difference=r - 1.0, citation=citation, passed=bool(abs(r - 1.0) <= tol),
    )


def run_constituent_bulk(model, name: str, *, fd_check: bool = True, relax_fmax: float = 5e-4,
                         elastic_tol: float = 0.10, lattice_tol: float = 0.02, log=print) -> dict[str, Any]:
    """Relax one phase, compute C (autodiff q=0, FD-checked), VRH moduli; return records + summary."""
    from fracture_md import provenance
    from fracture_md.engines.base import ProtocolOutput
    from fracture_md.physics.elastic import analyze_elastic, tensor_checks
    from fracture_md.potentials.neighbors import build_neighbors
    from fracture_md.records import make_record
    from fracture_md.schema.common import CheckResult, Conditions, Uncertainty, UncertaintyKind
    from fracture_md.schema.material import Orientation
    from fracture_md.schema.properties import MethodSpec, PropertyName
    from fracture_md.structures.symmetry import lattice_parameters, space_group
    from fracture_md.structures.system import AtomicSystem

    spec = PHASES[name]
    cand = phase_candidate(name)
    t0 = time.time()
    prim = spec.builder()
    n_prim = len(prim)
    sup, reps = _supercell(prim, model)
    log(f"[{name}] relaxing {len(sup)} atoms (reps {reps})")
    relaxed, rinfo = relax_bulk(model, sup, fmax=relax_fmax)
    log(f"[{name}] relaxed: {rinfo} ({time.time() - t0:.0f} s)")
    s = AtomicSystem.from_ase(relaxed, elements=model.elements)
    nh = build_neighbors(s, model.cutoff, skin=SKIN)
    d = q0_derivatives(model, s, nh.idx, n_prim)
    # joint Newton polish; the Dense list (radius cutoff + skin) stays valid for these tiny steps
    s, d = newton_polish(model, s, nh.idx, n_prim, d)
    nh = nh.update(s.frac, box=s.box)
    el = elastic_from_q0(d)
    C = el["C_relaxed"]
    t_relax = time.time() - t0
    log(f"[{name}] {s.n_atoms} atoms (reps {reps}), q0 derivatives {d.seconds:.1f} s, max|sigma| "
        f"{np.max(np.abs(d.stress_GPa)):.2e} GPa, fmax {np.max(np.linalg.norm(d.forces, axis=1)):.2e} eV/A "
        f"({time.time() - t0:.0f} s)")

    sg = space_group(s, symprec=1e-3)
    prim_box = np.asarray(s.box) @ np.diag(1.0 / np.asarray(reps, float))
    lengths, angles = lattice_parameters(prim_box)
    fm = float(np.max(np.linalg.norm(d.forces, axis=1)))
    smax = float(np.max(np.abs(d.stress_GPa)))
    tc = tensor_checks(C, sg.rotations_cart, sg.number, 0.5)
    checks = [
        CheckResult(name="relaxation_converged", passed=rinfo["converged"], details=json.dumps(rinfo)),
        CheckResult(name="equilibrium_zero_force", passed=fm < 1e-4, metric=fm, tolerance=1e-4, unit=Unit.EV_PER_ANGSTROM),
        CheckResult(name="equilibrium_zero_stress", passed=smax < 1e-3, metric=smax, tolerance=1e-3, unit=Unit.GPA),
        CheckResult(name="space_group_preserved", passed=sg.number == spec.space_group_number, metric=float(sg.number),
                    tolerance=0.0, details=f"spglib (symprec 1e-3): {sg.international} #{sg.number}; "
                                           f"expected #{spec.space_group_number}"),
        CheckResult(name="gamma_point_stability", passed=el["spectrum"].n_negative_modes == 0,
                    metric=float(el["spectrum"].min_nonzero), unit=Unit.EV_PER_ANGSTROM2,
                    details=f"q=0 Hessian: {el['spectrum'].n_negative_modes} negative, "
                            f"{el['spectrum'].n_zero_modes} zero modes (3 translations expected); "
                            f"{d.n_fd_fallback} of {3 * n_prim + 6} derivative columns by central differences "
                            "(non-finite MACE-JAX JVP)"),
        CheckResult(name="C_major_symmetry", passed=tc["asymmetry"] < 1e-6, metric=tc["asymmetry"], tolerance=1e-6, unit=Unit.GPA),
        CheckResult(name="C_point_group_invariance", passed=tc["point_group_invariance"] < 0.05,
                    metric=tc["point_group_invariance"], tolerance=0.05, unit=Unit.GPA,
                    details=f"{len(sg.rotations_cart)} Cartesian operations of {sg.international}"),
    ]
    unc = []
    fd_res = {}
    if fd_check:
        log(f"[{name}] FD checks")
        C_fd_cl, _ = fd_elastic(model, s, nh.idx, n_prim, el["K_pinv_cell"], relax=False)
        C_fd_rel, fres = fd_elastic(model, s, nh.idx, n_prim, el["K_pinv_cell"], relax=True)
        dcl = float(np.max(np.abs(C_fd_cl - el["C_clamped"])))
        drel = float(np.max(np.abs(C_fd_rel - C)))
        checks += [
            CheckResult(name="autodiff_vs_fd_clamped", passed=dcl < 1.0, metric=dcl, tolerance=1.0, unit=Unit.GPA,
                        details="central differences of the autodiff stress, delta = 1e-3"),
            CheckResult(name="autodiff_vs_fd_relaxed", passed=drel < 1.0, metric=drel, tolerance=1.0, unit=Unit.GPA,
                        details=f"internal coordinates Newton-relaxed at each strained cell (max residual force {fres:.1e} eV/Å)"),
        ]
        unc.append(Uncertainty(kind=UncertaintyKind.AUTODIFF_VS_FINITE_DIFFERENCE, value=drel, unit=Unit.GPA,
                               interval="max_abs_diff", description="max |C_AD - C_FD| (relaxed ion)"))
        fd_res = {"C_fd_clamped_GPa": C_fd_cl.tolist(), "C_fd_relaxed_GPa": C_fd_rel.tolist(),
                  "max_abs_diff_clamped_GPa": dcl, "max_abs_diff_relaxed_GPa": drel, "fd_max_residual_force": fres}
    mass = float(np.sum(np.asarray(s.masses_amu)))
    rho = mass / s.volume * AMU_PER_A3_TO_G_PER_CM3
    an = analyze_elastic(C, density_g_cm3=rho)

    # --- reference comparisons (evidence)
    ref = REFERENCES.get(name, {})
    ev_C, ev_E, ev_lat, ratios = [], [], [], {}
    cij = _hex_or_tet_cij(C)
    cij["|C14|"] = abs(cij["C14"])
    for key_c, key_src, kind in [("Cij_GPa", "Cij_source", ref.get("Cij_kind")), ("Cij_dft_GPa", "Cij_dft_source", "dft_reference")]:
        for k, rv in ref.get(key_c, {}).items():
            tag = k if key_c == "Cij_GPa" else k + "_dft"
            ratios[tag] = cij[k] / rv
            ev_C.append(ratio_evidence(tag, float(cij[k]), rv, Unit.GPA, ref[key_src], kind, elastic_tol))
    vrh_refs = []
    if "vrh" in ref:
        vrh_refs.append(("", ref["vrh"], ref["vrh_source"], ref["vrh_kind"]))
    if "vrh_dft" in ref:
        vrh_refs.append(("_dft", ref["vrh_dft"], ref["Cij_dft_source"], "dft_reference"))
    if "polycrystal" in ref:
        vrh_refs.append(("_polycrystal", ref["polycrystal"], ref["polycrystal_source"], "experimental_reference"))
    if "Cij_full_GPa" in ref:
        vrh_refs.append(("", _vrh_of_full(ref["Cij_full_GPa"]), ref["Cij_source"] + " (VRH computed here)",
                         ref["Cij_kind"]))
    mine = {"E": an["E_hill"], "G": an["G_hill"], "K": an["B_hill"], "nu": an["nu_hill"]}
    for tag, vals, src, kind in vrh_refs:
        for k, rv in vals.items():
            ratios[f"{k}_hill{tag}"] = mine[k] / rv
            if k in ("E", "G", "K"):
                ev_E.append(ratio_evidence(f"{k} (VRH){tag}", float(mine[k]), rv, Unit.GPA, src, kind, elastic_tol))
    lat_named = {"a": lengths[0], "b": lengths[1], "c": lengths[2], "beta_deg": angles[1]}
    for k, rv in ref.get("lattice_A", {}).items():
        ratios["lattice_" + k] = lat_named[k] / rv
        ev_lat.append(ratio_evidence(f"lattice {k}", float(lat_named[k]), rv, Unit.DEGREE if k.endswith("deg") else Unit.ANGSTROM,
                                     ref["lattice_source"], "experimental_reference", lattice_tol))

    prov = provenance.collect(engine="jax_md", potential_id=model.card.id, protocol_id=f"dental_atomistic_bulk:{name}",
                              requested_by="dental-atomistic")
    method = MethodSpec(
        name="q0 autodiff relaxed-ion elastic tensor", protocol_id=f"dental_atomistic_bulk:{name}", engine="jax_md",
        description="symmetry-constrained zero-stress relaxation (ASE LBFGS on the harness MACE kernels) + joint Newton "
                    "polish; C = (H_ee - Lambda^T K_q0^+ Lambda)/V from exact JAX second derivatives restricted to "
                    "q = 0 (periodic) internal displacements; FD stress-strain cross-check",
        assumptions=["0 K static lattice (no zero-point, no thermal expansion; experiment is at ~295 K)",
                     "defect-free, stoichiometric, ordered crystal", f"{s.n_atoms}-atom supercell {reps} of the "
                     f"{n_prim}-atom cell", "internal relaxation with primitive-cell periodicity (Gamma point)",
                     spec.notes or "", "MACE-MP-0 small (PBE/PBE+U-trained): expect PBE-like overestimated lattice "
                     "constants and softened moduli"],
    )
    cond = Conditions(stress_state="zero stress, 0 K lattice statics")
    crystal = Orientation(description="crystal Cartesian frame (ASE/spglib standard setting of the input cell)")
    recs = []

    def rec(pn, value, unit, *, shape="", evid=(), notes="", frame="not_applicable", orientation=None, u=()):
        r = make_record(name=pn, value=value, unit=unit, candidate=cand, method=method, card=model.card, conditions=cond,
                        checks=checks, evidence=list(evid), uncertainty=list(u), frame=frame, orientation=orientation,
                        shape_description=shape, notes=notes, provenance=prov, phase=name)
        recs.append(r)
        return r

    rec(PropertyName.LATTICE_PARAMETER, [float(x) for x in lengths], Unit.ANGSTROM, shape="[a, b, c] of the input cell",
        evid=ev_lat, notes=f"angles {np.round(angles, 4).tolist()} deg; {sg.international}", frame="crystal")
    rec(PropertyName.DENSITY, rho, Unit.G_PER_CM3)
    rec(PropertyName.VOLUME_PER_ATOM, s.volume / s.n_atoms, Unit.ANGSTROM3_PER_ATOM)
    rec(PropertyName.RESIDUAL_STRESS, d.stress_GPa.tolist(), Unit.GPA, shape="voigt 6", frame="crystal")
    vo = "voigt 6x6 (xx,yy,zz,yz,xz,xy), engineering shear strains, relaxed ion"
    rec(PropertyName.ELASTIC_STIFFNESS_VOIGT, C, Unit.GPA, shape=vo, evid=ev_C, frame="crystal", orientation=crystal, u=unc)
    rec(PropertyName.CLAMPED_ION_STIFFNESS_VOIGT, el["C_clamped"], Unit.GPA, shape=vo.replace("relaxed", "clamped"),
        frame="crystal", orientation=crystal)
    poly = "isotropic dense polycrystal, Voigt-Reuss-Hill average of the single-crystal C"
    for pn, key in [(PropertyName.BULK_MODULUS_HILL, "B_hill"), (PropertyName.SHEAR_MODULUS_HILL, "G_hill")]:
        rec(pn, float(an[key]), Unit.GPA, notes=poly)
    rec(PropertyName.YOUNGS_MODULUS_HILL, float(an["E_hill"]), Unit.GPA, notes=poly, evid=ev_E, u=unc)
    rec(PropertyName.POISSON_RATIO_HILL, float(an["nu_hill"]), Unit.DIMENSIONLESS, notes=poly)
    rec(PropertyName.BORN_STABILITY, float(an["born_min_eigenvalue"]), Unit.GPA,
        shape="minimum eigenvalue of relaxed-ion C", frame="crystal")
    out = ProtocolOutput(properties=recs, checks=checks, structure=s)
    extra = {}
    if el["spectrum"].n_negative_modes or an["born_min_eigenvalue"] <= 0:
        an_cl = analyze_elastic(el["C_clamped"])
        extra = {"mace_lattice_unstable": True,
                 "unstable_lattice_note": "MACE-MP-0 finds this symmetry-constrained structure unstable at 0 K "
                                          f"({el['spectrum'].n_negative_modes} negative Gamma modes; relaxed-ion Born "
                                          f"minimum eigenvalue {an['born_min_eigenvalue']:.1f} GPa). Its relaxed-ion "
                                          "moduli are not physical (records FAILED); clamped-ion values are an upper "
                                          "bound only.",
                 "clamped_ion_E_hill_GPa": float(an_cl["E_hill"]), "clamped_ion_nu_hill": float(an_cl["nu_hill"]),
                 "clamped_ion_born_min_eigenvalue_GPa": float(an_cl["born_min_eigenvalue"])}
    summary = {
        "phase": name, "formula": spec.formula, "space_group": {"number": sg.number, "symbol": sg.international},
        "n_atoms": s.n_atoms, "supercell": list(reps), "n_prim": n_prim,
        "lattice_A": [float(x) for x in lengths], "angles_deg": [float(x) for x in angles],
        "density_g_cm3": rho, "energy_eV_per_atom": d.energy / s.n_atoms,
        "residual_stress_GPa": d.stress_GPa.tolist(), "max_force_eV_A": fm,
        "C_relaxed_GPa": C.tolist(), "C_clamped_GPa": el["C_clamped"].tolist(),
        "Cij_named_GPa": {k: float(v) for k, v in cij.items() if abs(v) > 1e-6},
        "B_hill_GPa": float(an["B_hill"]), "G_hill_GPa": float(an["G_hill"]), "E_hill_GPa": float(an["E_hill"]),
        "nu_hill": float(an["nu_hill"]), "B_voigt_GPa": float(an["B_voigt"]), "B_reuss_GPa": float(an["B_reuss"]),
        "G_voigt_GPa": float(an["G_voigt"]), "G_reuss_GPa": float(an["G_reuss"]),
        "born_min_eigenvalue_GPa": float(an["born_min_eigenvalue"]),
        "gamma_negative_modes": int(el["spectrum"].n_negative_modes),
        "gamma_negative_eigenvalues_eV_A2_supercell": el["negative_eigenvalues_eV_A2"],
        "ratios_mace_over_reference": {k: float(v) for k, v in ratios.items()},
        "relax_info": rinfo, "fd": fd_res, "q0_seconds": d.seconds, "q0_fd_fallback_columns": d.n_fd_fallback, "wall_seconds": time.time() - t0,
        "relax_and_derivative_seconds": t_relax,
        "checks": {c.name: {"passed": c.passed, "metric": c.metric} for c in checks},
        "validation_status": {r.name.value: r.validation.status.value for r in recs},
        **extra,
    }
    return {"output": out, "summary": summary, "system": s, "reps": reps, "n_prim": n_prim}


# =============================================================================
# Surfaces
# =============================================================================


def molecular_units(atoms, bonds: Optional[dict] = None) -> list[np.ndarray]:
    """Groups of atom indices that must stay together (bonded pairs within the given cutoffs)."""
    from ase.neighborlist import neighbor_list

    n = len(atoms)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if bonds:
        sy = np.array(atoms.get_chemical_symbols())
        rmax = max(bonds.values())
        ii, jj, dd = neighbor_list("ijd", atoms, rmax)
        for i, j, dist in zip(ii, jj, dd):
            key = (sy[i], sy[j]) if (sy[i], sy[j]) in bonds else (sy[j], sy[i])
            if key in bonds and dist <= bonds[key]:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.asarray(g) for g in groups.values()]


def in_plane_supercells(a1: np.ndarray, a2: np.ndarray, min_width: float, max_entry: int = 5) -> list[np.ndarray]:
    """2x2 integer matrices M (rows: new in-plane vectors) with both in-plane widths > min_width,
    sorted by (number of cells, non-orthogonality)."""
    found = []
    for m in itertools.product(range(-max_entry, max_entry + 1), repeat=4):
        M = np.array(m).reshape(2, 2)
        det = int(round(np.linalg.det(M)))
        if det <= 0:
            continue
        A1 = M[0, 0] * a1 + M[0, 1] * a2
        A2 = M[1, 0] * a1 + M[1, 1] * a2
        area = np.linalg.norm(np.cross(A1, A2))
        w1, w2 = area / np.linalg.norm(A2), area / np.linalg.norm(A1)
        if min(w1, w2) <= min_width:
            continue
        ortho = abs(np.dot(A1, A2)) / (np.linalg.norm(A1) * np.linalg.norm(A2))
        found.append(((det, round(ortho, 6), -min(w1, w2)), M))
    found.sort(key=lambda t: t[0])
    return [M for _, M in found]


def oriented_supercell(bulk_atoms, miller, model, n_periods: int):
    """Periodic oriented bulk: a1, a2 in the plane, a3 = n_periods stacking vectors reduced to be as
    perpendicular to the plane as possible; the smallest in-plane supercell whose 3-D perpendicular
    widths along a1, a2 exceed 2 (cutoff + skin) is used."""
    from ase.build import make_supercell

    from fracture_md.structures.slabs import oriented_bulk_cell
    from fracture_md.structures.system import perpendicular_widths

    need = 2.0 * (model.cutoff + SKIN)
    ob = oriented_bulk_cell(bulk_atoms, tuple(miller))
    c = np.asarray(ob.cell.array)
    for M2 in in_plane_supercells(c[0], c[1], need):
        P = np.zeros((3, 3), int)
        P[:2, :2] = M2
        P[2, 2] = n_periods
        big = make_supercell(ob, P)
        cell = np.asarray(big.cell.array).copy()
        coef = np.linalg.lstsq(cell[:2, :2].T, cell[2, :2], rcond=None)[0]
        cell[2] = cell[2] - np.round(coef) @ cell[:2]
        big.set_cell(cell, scale_atoms=False)
        big.wrap()
        w = perpendicular_widths(cell.T)
        if w[0] > need and w[1] > need:
            big.info.update(ob.info)
            big.info["in_plane_matrix"] = M2.tolist()
            big.info["min_image_safe_3d"] = bool(w[2] > need)
            return big, ob
    raise ValueError("no min-image-safe oriented supercell found")


def window_slab(oriented, units: list[np.ndarray], z0_frac: float, vacuum: float):
    """Stoichiometric slab: shift units by their centre's fractional a3 coordinate so centres lie in
    [z0, z0 + 1) (units never broken), then add ``vacuum`` Å along the normal (+z)."""
    from ase import Atoms

    cell = np.asarray(oriented.cell.array)
    pos = oriented.get_positions()
    inv = np.linalg.inv(cell)
    new = pos.copy()
    for g in units:
        ref = pos[g[0]]
        dfrac = (pos[g] - ref) @ inv
        dfrac -= np.round(dfrac)  # unwrap the unit around its first atom
        xg = ref + dfrac @ cell
        cen = xg.mean(axis=0) @ inv
        cen_shift = np.floor(cen - np.array([0.0, 0.0, z0_frac]))
        shift = -np.array([np.floor(cen[0]), np.floor(cen[1]), cen_shift[2]]) @ cell
        new[g] = xg + shift - z0_frac * cell[2]
    new[:, 2] -= new[:, 2].min()
    slab_cell = cell.copy()
    slab_cell[2] = np.array([0.0, 0.0, float(new[:, 2].max()) + vacuum])
    slab = Atoms(symbols=oriented.get_chemical_symbols(), positions=new, cell=slab_cell, pbc=True)
    slab.wrap()
    return slab


def _cut_candidates(oriented, units, n_periods: int, max_candidates: int = 12) -> list[float]:
    """Fractional cut heights (of the oriented supercell a3) midway between distinct unit-centre levels."""
    cell = np.asarray(oriented.cell.array)
    inv = np.linalg.inv(cell)
    pos = oriented.get_positions()
    zs = []
    for g in units:
        ref = pos[g[0]]
        dfrac = (pos[g] - ref) @ inv
        dfrac -= np.round(dfrac)
        zs.append(((ref @ inv + dfrac.mean(axis=0))[2]) % 1.0)
    per = np.sort(np.unique(np.round((np.asarray(zs) * n_periods) % 1.0, 4)))  # levels within one period
    levels = np.concatenate([per, [per[0] + 1.0]])
    gaps = np.diff(levels)
    mids = (levels[:-1] + gaps / 2.0) % 1.0
    order = np.argsort(-gaps)[:max_candidates]
    return [float(mids[i] / n_periods) for i in order], [float(gaps[i]) for i in order]


def _same_ratio(a, b) -> bool:
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    if set(ca) != set(cb):
        return False
    k = next(iter(cb))
    return all(abs(ca[e] * cb[k] - cb[e] * ca[k]) == 0 for e in cb)


def run_surface_energy(model, name: str, bulk_system, reps, miller, *, n_periods: int, vacuum: float = 15.0,
                       relax_fmax: float = 0.01, n_relax: int = 2, log=print) -> dict[str, Any]:
    """Relaxed surface energy (J/m^2, average of top/bottom) of a stoichiometric slab of ``name``."""
    import jax.numpy as jnp

    from fracture_md import provenance
    from fracture_md.physics.relax import relax_positions
    from fracture_md.potentials.neighbors import build_neighbors
    from fracture_md.records import make_record
    from fracture_md.schema.common import CheckResult, Conditions
    from fracture_md.schema.material import InterfaceKind, InterfaceSpec, Orientation, Termination
    from fracture_md.schema.properties import MethodSpec, PropertyName
    from fracture_md.structures.system import AtomicSystem

    t0 = time.time()
    spec = PHASES[name]
    cand = phase_candidate(name)
    bulk_atoms = bulk_system.to_ase()
    # single conventional cell of the relaxed bulk (undo the supercell)
    from ase.build import make_supercell

    prim_cell = np.asarray(bulk_atoms.cell.array) / np.asarray(reps, float)[:, None]
    n_prim = bulk_system.n_atoms // int(np.prod(reps))
    one = bulk_atoms[:n_prim]
    one.set_cell(prim_cell, scale_atoms=False)
    one.wrap()
    e_bulk_atom = energy_forces_stress(model, bulk_system, build_neighbors(bulk_system, model.cutoff, SKIN).idx)[0] / bulk_system.n_atoms
    ori, ob = oriented_supercell(one, miller, model, n_periods)
    units = molecular_units(ori, MOLECULAR_BONDS.get(name))
    # check 1: oriented periodic bulk has the bulk energy per atom (pure rotation of the relaxed lattice)
    ori_chk = ori if ori.info["min_image_safe_3d"] else ori.repeat((1, 1, 2))
    so = AtomicSystem.from_ase(ori_chk, elements=model.elements)
    e_or = energy_forces_stress(model, so, build_neighbors(so, model.cutoff, SKIN).idx)[0] / so.n_atoms
    area = float(np.linalg.norm(np.cross(ori.cell[0], ori.cell[1])))
    cuts, gaps = _cut_candidates(ori, units, n_periods)
    scan = []
    nh = None
    for zc, gap in zip(cuts, gaps):
        slab = window_slab(ori, units, zc, vacuum)
        ss = AtomicSystem.from_ase(slab, elements=model.elements)
        if nh is None:
            nh = build_neighbors(ss, model.cutoff, SKIN)
        else:
            nh = nh.update(ss.frac)
        E = energy_forces_stress(model, ss, nh.idx)[0]
        g = (E - ss.n_atoms * e_bulk_atom) / (2 * area) * EV_PER_A2_TO_J_PER_M2
        scan.append({"cut_frac": zc, "level_gap_frac_of_period": gap, "gamma_unrelaxed_J_m2": g, "slab": slab})
    scan.sort(key=lambda r: r["gamma_unrelaxed_J_m2"])
    # distinct terminations only (equal unrelaxed energies -> equivalent cuts)
    distinct = []
    for r in scan:
        if all(abs(r["gamma_unrelaxed_J_m2"] - q["gamma_unrelaxed_J_m2"]) > 1e-4 for q in distinct):
            distinct.append(r)
    log(f"[{name} {miller}] {len(ori)} atoms, area {area:.2f} Å2, e_bulk-e_oriented {e_bulk_atom - e_or:.2e} eV/atom, "
        f"unrelaxed scan {[round(r['gamma_unrelaxed_J_m2'], 3) for r in distinct]}")
    relaxed = []
    for r in distinct[:n_relax]:
        ss = AtomicSystem.from_ase(r["slab"], elements=model.elements)
        sr, info = relax_positions(model, ss, fmax=relax_fmax, newton_polish=False, skin=SKIN)
        nh2 = build_neighbors(sr, model.cutoff, SKIN)
        E, F, _ = energy_forces_stress(model, sr, nh2.idx)
        g = (E - sr.n_atoms * e_bulk_atom) / (2 * area) * EV_PER_A2_TO_J_PER_M2
        # vacuum check: no pair within cutoff across the gap
        zs = np.sort(np.asarray(sr.positions)[:, 2] % float(sr.box[2, 2]))
        gap_A = float(max(np.max(np.diff(zs)), zs[0] + float(sr.box[2, 2]) - zs[-1]))
        relaxed.append({"cut_frac": r["cut_frac"], "gamma_unrelaxed_J_m2": r["gamma_unrelaxed_J_m2"], "gamma_relaxed_J_m2": g,
                        "relax_converged": bool(info.converged), "fmax": float(np.max(np.linalg.norm(F, axis=1))),
                        "n_steps": info.n_steps, "vacuum_gap_A": gap_A, "system": sr})
        log(f"   cut {r['cut_frac']:.4f}: gamma unrelaxed {r['gamma_unrelaxed_J_m2']:.4f} -> relaxed {g:.4f} J/m2 "
            f"({info.n_steps} steps, fmax {relaxed[-1]['fmax']:.3g})")
    best = min(relaxed, key=lambda r: r["gamma_relaxed_J_m2"])
    counts_ok = sorted(ori.get_chemical_symbols()) == sorted(best["system"].symbols)
    checks = [
        CheckResult(name="oriented_bulk_energy_consistency", passed=abs(e_bulk_atom - e_or) < 1e-6,
                    metric=abs(e_bulk_atom - e_or), tolerance=1e-6, unit=Unit.EV_PER_ATOM,
                    details="periodic oriented cell (rotated relaxed bulk) vs bulk energy per atom"),
        CheckResult(name="stoichiometric_slab", passed=counts_ok and _same_ratio(best["system"].symbols, one.get_chemical_symbols()),
                    details=f"{len(ori)} atoms = {len(ori) // len(one)} x bulk cell; molecular units kept whole"),
        CheckResult(name="slab_relaxation_converged", passed=best["relax_converged"] and best["fmax"] <= relax_fmax * 1.01,
                    metric=best["fmax"], tolerance=relax_fmax, unit=Unit.EV_PER_ANGSTROM),
        CheckResult(name="vacuum_exceeds_cutoff", passed=best["vacuum_gap_A"] > model.cutoff + SKIN,
                    metric=best["vacuum_gap_A"], tolerance=model.cutoff + SKIN, unit=Unit.ANGSTROM,
                    details="no MACE edge crosses the vacuum gap"),
        CheckResult(name="relaxation_lowers_energy", passed=best["gamma_relaxed_J_m2"] <= best["gamma_unrelaxed_J_m2"] + 1e-9,
                    metric=best["gamma_relaxed_J_m2"] - best["gamma_unrelaxed_J_m2"], unit=Unit.J_PER_M2),
    ]
    ref = REFERENCES.get(name, {}).get("surface_energy_J_m2", {})
    mkey = "(" + "".join(str(int(x)) for x in miller) + ")"
    alias = {"(100)": "(010)"}  # hexagonal (100) and (010) are symmetry-equivalent prism planes
    evid = []
    if mkey in ref or alias.get(mkey) in ref:
        rv = ref.get(mkey, ref.get(alias.get(mkey)))
        evid.append(ratio_evidence(f"surface energy {mkey}", best["gamma_relaxed_J_m2"], rv, Unit.J_PER_M2,
                                   REFERENCES[name]["surface_energy_source"], REFERENCES[name]["surface_energy_kind"], 0.25))
    orient = Orientation(plane_miller=tuple(int(x) for x in miller), description=f"{mkey} plane of {spec.space_group_symbol} cell")
    iface = InterfaceSpec(kind=InterfaceKind.FREE_SURFACE if hasattr(InterfaceKind, "FREE_SURFACE") else list(InterfaceKind)[0],
                          phase_a=name, orientation_a=orient,
                          termination=Termination(label=f"window-cut@{best['cut_frac']:.4f}", cut_fraction=best["cut_frac"],
                                                  stoichiometric_halves=True,
                                                  description="lowest-energy stoichiometric molecular-unit cut; "
                                                              "top/bottom terminations complementary"))
    prov = provenance.collect(engine="jax_md", potential_id=model.card.id, protocol_id=f"dental_atomistic_surface:{name}{mkey}",
                              requested_by="dental-atomistic")
    method = MethodSpec(name="relaxed stoichiometric slab surface energy", protocol_id=f"dental_atomistic_surface:{name}{mkey}",
                        engine="jax_md",
                        description=f"{len(ori)}-atom slab ({n_periods} stacking periods, {vacuum} Å vacuum), in-plane "
                                    "lattice fixed at the MACE bulk value, positions L-BFGS relaxed; gamma = "
                                    "(E_slab - N e_bulk)/(2A); cut chosen by unrelaxed scan over all distinct cut heights",
                        assumptions=["0 K, vacuum, clean surface: no hydroxylation / water / adsorbates",
                                     "no surface reconstruction beyond local relaxation",
                                     "average of the complementary top and bottom terminations",
                                     "MACE-MP-0 surfaces are outside its training domain (unvalidated)"])
    records = []
    for pn, val, note in [(PropertyName.SURFACE_ENERGY_AVERAGE_PAIR, best["gamma_relaxed_J_m2"], "relaxed"),
                          (PropertyName.SURFACE_ENERGY_AVERAGE_PAIR, best["gamma_unrelaxed_J_m2"], "unrelaxed (same cut)")]:
        records.append(make_record(name=pn, value=float(val), unit=Unit.J_PER_M2, candidate=cand, method=method,
                                   card=model.card, conditions=Conditions(stress_state="bulk in-plane lattice"),
                                   orientation=orient, interface=iface, checks=checks,
                                   evidence=evid if note == "relaxed" else [], notes=note, provenance=prov, phase=name))
    summary = {"phase": name, "miller": list(miller), "n_atoms": len(ori), "n_periods": n_periods, "area_A2": area,
               "stacking_period_A": float(ob.info.get("stacking_period", np.nan)), "vacuum_A": vacuum,
               "in_plane_matrix": ori.info["in_plane_matrix"],
               "e_bulk_eV_per_atom": e_bulk_atom, "e_oriented_minus_bulk_eV_per_atom": e_or - e_bulk_atom,
               "unrelaxed_scan": [{k: v for k, v in r.items() if k != "slab"} for r in scan],
               "relaxed": [{k: v for k, v in r.items() if k != "system"} for r in relaxed],
               "gamma_relaxed_J_m2": best["gamma_relaxed_J_m2"], "gamma_unrelaxed_J_m2": best["gamma_unrelaxed_J_m2"],
               "reference": ({"value_J_m2": ref.get(mkey, ref.get(alias.get(mkey))),
                              "source": REFERENCES[name].get("surface_energy_source")} if evid else None),
               "ratio_mace_over_reference": (evid[0].relative_difference + 1.0) if evid else None,
               "checks": {c.name: {"passed": c.passed, "metric": c.metric} for c in checks},
               "validation_status": records[0].validation.status.value, "seconds": time.time() - t0}
    return {"records": records, "summary": summary, "best_system": best["system"]}


# =============================================================================
# Hand-off to micromechanics
# =============================================================================
_FILLER_MAP = {
    # filler key -> (atomistic cross-check phases, literature reference key for the filler, block, what it applies to)
    "zirconia_3Y-TZP": (("t-ZrO2", "m-ZrO2"), "t-ZrO2", "polycrystal",
                        "dense 3Y-TZP polycrystal (Pabst 2004 recommended values)"),
    "silica_amorphous": (("alpha-quartz", "alpha-cristobalite"), "amorphous-silica", "vrh",
                         "fused silica / silica glass (crystalline quartz and cristobalite are structural proxies only)"),
    "hydroxyapatite": (("hydroxyapatite",), "hydroxyapatite", "vrh", "isotropic HAp (Katz & Ukraincik)"),
}


def _phase_literature_E(phase: str) -> Optional[tuple[float, str]]:
    ref = REFERENCES.get(phase, {})
    if "vrh" in ref and "E" in ref["vrh"]:
        return ref["vrh"]["E"], ref["vrh_source"]
    if "Cij_full_GPa" in ref:
        return _vrh_of_full(ref["Cij_full_GPa"])["E"], ref["Cij_source"] + " (VRH computed here)"
    return None


def filler_properties(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """E and nu per filler: MACE value, literature value, value recommended for micromechanics, uncertainty.

    The recommended value is always the literature (experimental) value; the MACE value is the
    atomistic cross-check (its ratio to literature quantifies MACE-MP-0 softening for that
    chemistry).  MACE values are read from data/dental/atomistic_constituents.json (written by
    examples/10_dental_atomistic.py); if absent they are None.
    """
    p = Path(path) if path is not None else DATA_PATH
    data = json.loads(p.read_text()) if p.exists() else {}
    bulk = data.get("bulk", {})
    out: dict[str, dict[str, Any]] = {}
    for key, (phases, refkey, block, how) in _FILLER_MAP.items():
        ref = REFERENCES[refkey]
        vals = ref[block]
        E_lit = vals["E"]
        nu_lit = vals.get("nu", vals["E"] / (2.0 * vals["G"]) - 1.0)
        cross = {}
        for ph in phases:
            mb = bulk.get(ph)
            if not mb:
                continue
            lit = _phase_literature_E(ph)
            unstable = bool(mb.get("mace_lattice_unstable") or mb.get("gamma_negative_modes")
                            or mb.get("born_min_eigenvalue_GPa", 1.0) <= 0)
            E_m = None if unstable else mb.get("E_hill_GPa")
            cross[ph] = {"mace_E_GPa": E_m, "mace_nu": None if unstable else mb.get("nu_hill"),
                         "literature_E_GPa_same_phase": lit[0] if lit else None,
                         "literature_source_same_phase": lit[1] if lit else None,
                         "mace_over_literature_E": (E_m / lit[0]) if (E_m and lit) else None,
                         "mace_lattice_unstable": unstable,
                         "note": mb.get("unstable_lattice_note", "")}
        ratios = [c["mace_over_literature_E"] for c in cross.values() if c["mace_over_literature_E"]]
        out[key] = {
            "E_GPa": {"recommended": E_lit, "uncertainty": round(0.05 * E_lit, 3),
                      "uncertainty_kind": "1sigma, estimated 5 % (spread of reported polycrystal/VRH values)",
                      "literature": E_lit},
            "nu": {"recommended": nu_lit, "uncertainty": 0.02, "uncertainty_kind": "1sigma, estimated",
                   "literature": nu_lit},
            "literature_source": ref[f"{block}_source"], "applies_to": how,
            "atomistic_cross_check": cross,
            "mace_softening_ratio_E": (float(np.mean(ratios)) if ratios else None),
            "note": "use 'recommended' (literature). MACE-MP-0 values are cross-checks against the literature value of "
                    "the SAME crystalline phase (mace_over_literature_E); they quantify MACE softening for this "
                    "chemistry and are not filler properties.",
        }
    return out
