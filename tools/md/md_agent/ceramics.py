"""
Atomistic Ceramic Simulations
================================

The "ceramic" model family for `md_agent.py`: periodic, atomistic crystals
and glasses (oxides, carbides, perovskites, ...), as opposed to the
coarse-grained chain models in `md_engine.py`.

    material (library key or user CIF)
      -> build_ceramic()          periodic supercell (ASE Atoms) from an
                                  experimental reference structure
      -> potential                "classical": Pedone et al. 2006 rigid-ion
                                  potential (Ewald Coulomb + Morse + C/r^12),
                                  implemented here in numpy with analytic
                                  forces and stress
                                  "mace": MACE-MP-0 universal ML potential
                                  (optional; covers most of the periodic table)
      -> relax_cell()             0 K lattice parameters / density
      -> elastic_constants()      6x6 C_ij by finite strain with internal
                                  relaxation -> bulk/shear/Young's moduli
      -> run_ceramic_md()         NPT (Berendsen) or NVT (Langevin) MD,
                                  optional melt-quench to make glasses
      -> ceramic_properties()     density, lattice params at T, partial RDF
                                  peaks, coordination numbers, per-species
                                  diffusion; thermal expansion across runs

Grounding rules match the rest of the agent: structures come only from the
built-in library (experimental reference structures, sources listed) or a
CIF file the user named; potential parameters come only from the published
set below. The LLM never chooses parameters.

`python3 ceramics.py` runs the validation harness:
  - Ewald: NaCl Madelung constant, independence from the splitting
    parameter, finite-difference forces and stress
  - Pedone vs experiment: relaxed lattice constants, density, coordination,
    bulk modulus for MgO, CaO, alpha-Al2O3, alpha-quartz, Na2O
  - MACE-MP-0 vs experiment (if installed)
  - NPT MD density at 300 K
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EV_PER_A3_TO_GPA = 160.21766208
KE_EV_A = 14.399645478            # e^2 / (4 pi eps0), eV * Angstrom
AMU_PER_A3_TO_G_PER_CM3 = 1.66053907


class CeramicNotConfiguredError(RuntimeError):
    """Missing optional package (ase, mace-torch) — distinct from a bad spec,
    so the agent reports it instead of retrying."""


def _require_ase():
    try:
        import ase  # noqa: F401
    except Exception as e:
        raise CeramicNotConfiguredError(
            f"ceramic simulations need ASE ({type(e).__name__}: {e}); pip install ase") from e


def mace_available() -> bool:
    try:
        import mace.calculators  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reference structure library (experimental room-temperature structures)
# ---------------------------------------------------------------------------

# Each entry: how to build it with ASE, plus the experimental values the
# validation harness and the agent's report compare against. Lattice
# parameters are standard room-temperature crystallographic values;
# densities are X-ray (crystallographic) densities, the right comparison for
# a defect-free simulated crystal. Extend with structures you have
# provenance for (or pass a CIF).
CERAMIC_LIBRARY: Dict[str, Dict[str, Any]] = {
    "MgO": {
        "name": "periclase (rocksalt MgO)",
        "build": ("bulk", dict(name="MgO", crystalstructure="rocksalt", a=4.212, cubic=True)),
        "experimental": {"a": 4.212, "density_g_cm3": 3.58, "bulk_modulus_GPa": 160,
                         "coordination": {"Mg-O": 6}},
    },
    "CaO": {
        "name": "lime (rocksalt CaO)",
        "build": ("bulk", dict(name="CaO", crystalstructure="rocksalt", a=4.811, cubic=True)),
        "experimental": {"a": 4.811, "density_g_cm3": 3.34, "bulk_modulus_GPa": 112,
                         "coordination": {"Ca-O": 6}},
    },
    "Al2O3": {
        "name": "corundum (alpha-Al2O3)",
        "build": ("crystal", dict(symbols=["Al", "O"], basis=[(0, 0, 0.35216), (0.30624, 0, 0.25)],
                                  spacegroup=167, cellpar=[4.7589, 4.7589, 12.991, 90, 90, 120])),
        "experimental": {"a": 4.7589, "c": 12.991, "density_g_cm3": 3.99, "bulk_modulus_GPa": 252,
                         "coordination": {"Al-O": 6}},
    },
    "SiO2_quartz": {
        "name": "alpha-quartz (SiO2)",
        # P3_221 in ASE's origin setting: Si on 3a must be (x, 0, 2/3) — (x, 0, 0) is a
        # general position there and silently yields 6 Si with 2-fold Si (caught by the
        # structure checks in validate()).
        "build": ("crystal", dict(symbols=["Si", "O"], basis=[(0.4697, 0, 2 / 3), (0.4135, 0.2669, 0.7858)],
                                  spacegroup=154, cellpar=[4.9134, 4.9134, 5.4052, 90, 90, 120])),
        "experimental": {"a": 4.9134, "c": 5.4052, "density_g_cm3": 2.65, "bulk_modulus_GPa": 37.5,
                         "coordination": {"Si-O": 4}},
    },
    "Na2O": {
        "name": "sodium oxide (antifluorite)",
        "build": ("crystal", dict(symbols=["O", "Na"], basis=[(0, 0, 0), (0.25, 0.25, 0.25)],
                                  spacegroup=225, cellpar=[5.55, 5.55, 5.55, 90, 90, 90])),
        # X-ray density from a = 5.55 A (handbook bulk value 2.27 includes porosity)
        "experimental": {"a": 5.55, "density_g_cm3": 2.41, "coordination": {"Na-O": 4}},
    },
    "TiO2_rutile": {
        "name": "rutile (TiO2)",
        "build": ("crystal", dict(symbols=["Ti", "O"], basis=[(0, 0, 0), (0.3048, 0.3048, 0)],
                                  spacegroup=136, cellpar=[4.594, 4.594, 2.959, 90, 90, 90])),
        "experimental": {"a": 4.594, "c": 2.959, "density_g_cm3": 4.25, "bulk_modulus_GPa": 211,
                         "coordination": {"Ti-O": 6}},
    },
    "ZrO2_cubic": {
        "name": "cubic zirconia (fluorite ZrO2; the high-temperature / stabilized phase)",
        "build": ("crystal", dict(symbols=["Zr", "O"], basis=[(0, 0, 0), (0.25, 0.25, 0.25)],
                                  spacegroup=225, cellpar=[5.07, 5.07, 5.07, 90, 90, 90])),
        # X-ray density of undoped cubic ZrO2 at a = 5.07 A (Y-stabilized samples are ~6.0)
        "experimental": {"a": 5.07, "density_g_cm3": 6.28, "coordination": {"Zr-O": 8}},
    },
    "CeO2": {
        "name": "ceria (fluorite CeO2)",
        "build": ("crystal", dict(symbols=["Ce", "O"], basis=[(0, 0, 0), (0.25, 0.25, 0.25)],
                                  spacegroup=225, cellpar=[5.411, 5.411, 5.411, 90, 90, 90])),
        "experimental": {"a": 5.411, "density_g_cm3": 7.22, "bulk_modulus_GPa": 220,
                         "coordination": {"Ce-O": 8}},
    },
    "SiC_3C": {
        "name": "cubic silicon carbide (3C / beta-SiC, zincblende)",
        "build": ("bulk", dict(name="SiC", crystalstructure="zincblende", a=4.3596, cubic=True)),
        "experimental": {"a": 4.3596, "density_g_cm3": 3.21, "bulk_modulus_GPa": 225,
                         "coordination": {"Si-C": 4}},
    },
    "ZnO": {
        "name": "zincite (wurtzite ZnO)",
        "build": ("crystal", dict(symbols=["Zn", "O"], basis=[(1 / 3, 2 / 3, 0), (1 / 3, 2 / 3, 0.3819)],
                                  spacegroup=186, cellpar=[3.2495, 3.2495, 5.2069, 90, 90, 120])),
        # X-ray density (handbook bulk value 5.61)
        "experimental": {"a": 3.2495, "c": 5.2069, "density_g_cm3": 5.68, "bulk_modulus_GPa": 143,
                         "coordination": {"Zn-O": 4}},
    },
    "SrTiO3": {
        "name": "strontium titanate (cubic perovskite)",
        "build": ("crystal", dict(symbols=["Sr", "Ti", "O"], basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0, 0.5)],
                                  spacegroup=221, cellpar=[3.905, 3.905, 3.905, 90, 90, 90])),
        "experimental": {"a": 3.905, "density_g_cm3": 5.12, "bulk_modulus_GPa": 179,
                         "coordination": {"Ti-O": 6}},
    },
}


def list_materials() -> Dict[str, Any]:
    return {k: {"name": v["name"], "elements": sorted(set(_build_unit(k).get_chemical_symbols())),
                "classical_potential_available": _pedone_covers(_build_unit(k)),
                "experimental_reference": v["experimental"]}
            for k, v in CERAMIC_LIBRARY.items()}


def _build_unit(key: str):
    _require_ase()
    from ase.build import bulk
    from ase.spacegroup import crystal
    kind, kw = CERAMIC_LIBRARY[key]["build"]
    return bulk(**kw) if kind == "bulk" else crystal(**kw)


def _supercell_reps(cell: np.ndarray, min_width: float) -> Tuple[int, int, int]:
    """Smallest repetition so every perpendicular cell width is >= min_width."""
    vol = abs(np.linalg.det(cell))
    widths = [vol / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])) for i in range(3)]
    return tuple(max(1, math.ceil(min_width / w)) for w in widths)


# ---------------------------------------------------------------------------
# Pedone et al. 2006 rigid-ion potential (J. Phys. Chem. B 110, 11780)
#   U_ij = q_i q_j e^2/r + D_ij [ (1 - exp(-a_ij (r - r0_ij)))^2 - 1 ] + C_ij / r^12
# Partial charges; D in eV, a in 1/A, r0 in A, C in eV A^12. Transcribed from
# the published table — the validation harness checks them against
# experimental lattice constants, which is what would expose a typo.
# ---------------------------------------------------------------------------

# Na (q = 0.6; Na-O D=0.023363, a=1.763867, r0=3.006315, C=5.0) is deliberately
# NOT enabled: as transcribed it gives crystalline Na2O a = 5.32 A vs 5.55 A
# experimental (-4%), and without the paper at hand a transcription error
# can't be ruled out. Na-containing materials fall through to MACE.
PEDONE_CHARGES = {"O": -1.2, "Si": 2.4, "Al": 1.8, "Mg": 1.2, "Ca": 1.2}
PEDONE_PAIRS = {
    ("O", "O"): (0.042395, 1.379316, 3.618701, 22.0),
    ("Si", "O"): (0.340554, 2.006700, 2.100000, 1.0),
    ("Al", "O"): (0.361581, 1.900442, 2.164818, 0.9),
    ("Mg", "O"): (0.038908, 2.281000, 2.586153, 5.0),
    ("Ca", "O"): (0.030211, 2.241334, 2.923245, 5.0),
}


def _pedone_covers(atoms) -> bool:
    syms = set(atoms.get_chemical_symbols())
    if not syms <= set(PEDONE_CHARGES):
        return False
    q = sum(PEDONE_CHARGES[s] for s in atoms.get_chemical_symbols())
    return abs(q) < 1e-6


def pair_vectors(positions: np.ndarray, cell: np.ndarray, rcut: float):
    """All (i, j, |d|, d) with |d| < rcut over periodic images, both orderings
    (like ase.neighborlist 'ijdD', but vectorized per image shift; ~50x faster
    for the few-hundred-atom cells used here)."""
    cell = np.asarray(cell, float)
    inv = np.linalg.inv(cell)
    frac = positions @ inv
    pos = (frac - np.floor(frac)) @ cell                 # wrapped; only differences matter
    vol = abs(np.linalg.det(cell))
    widths = [vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)]
    m = [int(math.ceil(rcut / w)) for w in widths]
    n = len(pos)
    base = pos[None, :, :] - pos[:, None, :]             # d_ij = r_j - r_i
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    out_i, out_j, out_D = [], [], []
    r2 = rcut * rcut
    for a in range(-m[0], m[0] + 1):
        for b in range(-m[1], m[1] + 1):
            for c in range(-m[2], m[2] + 1):
                T = a * cell[0] + b * cell[1] + c * cell[2]
                D = base + T
                dd = np.einsum("ijk,ijk->ij", D, D)
                mask = dd < r2
                if a == 0 and b == 0 and c == 0:
                    mask &= I != J
                if mask.any():
                    out_i.append(I[mask])
                    out_j.append(J[mask])
                    out_D.append(D[mask])
    i = np.concatenate(out_i)
    j = np.concatenate(out_j)
    D = np.concatenate(out_D)
    return i, j, np.sqrt(np.einsum("ij,ij->i", D, D)), D


def _ewald_alpha(rcut: float, tol: float) -> float:
    return math.sqrt(-math.log(tol)) / rcut


def _kvectors(cell: np.ndarray, kmax: float) -> np.ndarray:
    """Half-space set of reciprocal vectors with 0 < |k| <= kmax (k and -k
    contribute equally, so callers double the sum)."""
    rec = 2.0 * np.pi * np.linalg.inv(cell).T          # rows are b1, b2, b3
    # bound each integer index by the perpendicular spacing of reciprocal-lattice
    # planes (|b_i| alone under-counts for skewed cells such as hexagonal ones)
    widths = [abs(np.linalg.det(rec)) / np.linalg.norm(np.cross(rec[(i + 1) % 3], rec[(i + 2) % 3]))
              for i in range(3)]
    nmax = [int(math.ceil(kmax / w)) for w in widths]
    grid = np.array(np.meshgrid(*[np.arange(-n, n + 1) for n in nmax], indexing="ij")).reshape(3, -1).T
    # half-space: first nonzero index positive
    keep = (grid[:, 0] > 0) | ((grid[:, 0] == 0) & (grid[:, 1] > 0)) | \
           ((grid[:, 0] == 0) & (grid[:, 1] == 0) & (grid[:, 2] > 0))
    k = grid[keep] @ rec
    k2 = np.einsum("ij,ij->i", k, k)
    return k[k2 <= kmax * kmax]


def ewald_and_pairs(positions: np.ndarray, cell: np.ndarray, charges: np.ndarray,
                    symbols: List[str], pair_params: Dict[Tuple[str, str], Tuple[float, ...]],
                    rcut: float = 10.0, tol: float = 1e-7,
                    alpha: Optional[float] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """Energy (eV), forces (eV/A), and stress (eV/A^3, ASE sign convention:
    stress = (1/V) dE/d(strain)) for Ewald Coulomb + Morse + C/r^12."""
    n = len(positions)
    vol = abs(np.linalg.det(cell))
    alpha = alpha if alpha is not None else _ewald_alpha(rcut, tol)
    kmax = 2.0 * alpha * math.sqrt(-math.log(tol))

    # --- real space: erfc Coulomb + short-range ---------------------------
    i, j, d, D = pair_vectors(positions, cell, rcut)
    qq = charges[i] * charges[j]
    erfc_v = _erfc(alpha * d)
    gauss = np.exp(-(alpha * d) ** 2)
    u = KE_EV_A * qq * erfc_v / d
    du = -KE_EV_A * qq * (erfc_v / d ** 2 + 2 * alpha / math.sqrt(math.pi) * gauss / d)

    # short-range parameters per pair (symmetric lookup)
    sym = np.array(symbols)
    Dm, am, r0m, Cm = (np.zeros(len(d)) for _ in range(4))
    si, sj = sym[i], sym[j]
    for (a_, b_), (D_, a__, r0_, C_) in pair_params.items():
        m = ((si == a_) & (sj == b_)) | ((si == b_) & (sj == a_))
        Dm[m], am[m], r0m[m], Cm[m] = D_, a__, r0_, C_
    has = Dm != 0
    e = np.exp(-am * (d - r0m))
    u += np.where(has, Dm * ((1 - e) ** 2 - 1) + Cm / d ** 12, 0.0)
    du += np.where(has, 2 * Dm * am * (1 - e) * e - 12 * Cm / d ** 13, 0.0)

    energy = 0.5 * float(np.sum(u))
    fpair = (du / d)[:, None] * D                     # force on i from pair (i -> j)
    forces = np.zeros((n, 3))
    for k_ in range(3):
        forces[:, k_] = np.bincount(i, fpair[:, k_], minlength=n)
    stress = 0.5 * np.einsum("p,pa,pb->ab", du / d, D, D)

    # --- reciprocal space -------------------------------------------------
    kv = _kvectors(cell, kmax)
    if len(kv):
        k2 = np.einsum("ij,ij->i", kv, kv)
        g = np.exp(-k2 / (4 * alpha ** 2)) / k2
        phase = positions @ kv.T                       # (N, K)
        cos, sin = np.cos(phase), np.sin(phase)
        sre, sim = charges @ cos, charges @ sin        # S(k)
        s2 = sre ** 2 + sim ** 2
        pref = 2.0 * 2.0 * math.pi * KE_EV_A / vol     # extra 2: half-space sum
        e_rec = pref * float(np.sum(g * s2))
        energy += e_rec
        # F_i = (4 pi ke q_i / V) sum_k g k Im[e^{ik r_i} S*]   (x2 for half-space)
        im = sin * sre[None, :] - cos * sim[None, :]   # Im[e^{ik r_i} conj(S)]
        forces += 2.0 * (4 * math.pi * KE_EV_A / vol) * charges[:, None] * ((im * g[None, :]) @ kv)
        w = pref * g * s2 * (2.0 / k2 + 1.0 / (2 * alpha ** 2))
        stress += -e_rec * np.eye(3) + np.einsum("k,ka,kb->ab", w, kv, kv)

    # --- self energy --------------------------------------------------------
    energy -= KE_EV_A * alpha / math.sqrt(math.pi) * float(np.sum(charges ** 2))
    return energy, forces, stress / vol


def _erfc(x: np.ndarray) -> np.ndarray:
    from scipy.special import erfc   # scipy is a hard dependency of ASE
    return erfc(x)


def _make_pedone_calculator(rcut: float = 10.0, tol: float = 1e-7):
    _require_ase()
    from ase.calculators.calculator import Calculator, all_changes

    class PedoneCalculator(Calculator):
        implemented_properties = ["energy", "free_energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            syms = self.atoms.get_chemical_symbols()
            q = np.array([PEDONE_CHARGES[s] for s in syms])
            e, f, s = ewald_and_pairs(self.atoms.get_positions(), np.array(self.atoms.get_cell()),
                                      q, syms, PEDONE_PAIRS, rcut=rcut, tol=tol)
            # ASE Voigt order: xx, yy, zz, yz, xz, xy
            self.results = {"energy": e, "free_energy": e, "forces": f,
                            "stress": np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])}

    return PedoneCalculator()


_MACE_CACHE: Dict[Tuple[str, str, str], Any] = {}


def torch_device() -> str:
    """'cuda[:N]' when PyTorch sees a GPU, else 'cpu'. MD_AGENT_DEVICE forces a
    device string; MD_AGENT_GPU_INDEX picks the GPU (shared with OpenMM)."""
    import os
    forced = os.environ.get("MD_AGENT_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
    except Exception:
        return "cpu"
    idx = os.environ.get("MD_AGENT_GPU_INDEX")
    return f"cuda:{idx}" if idx else "cuda"


def mace_dtype(device: str) -> str:
    """float64 on CPU (cheap there, and tightest for relaxations / finite-strain
    elastic constants); float32 on GPU, where float64 throughput is poor on most
    cards. Override with MD_AGENT_MACE_DTYPE."""
    import os
    return os.environ.get("MD_AGENT_MACE_DTYPE") or ("float64" if device == "cpu" else "float32")


def _make_mace_calculator(model: str = "small"):
    try:
        from mace.calculators import mace_mp
    except Exception as e:
        raise CeramicNotConfiguredError(
            f"MACE not importable ({type(e).__name__}: {e}); pip install mace-torch, "
            "or use potential='classical' for Pedone-covered oxides") from e
    device = torch_device()
    key = (model, device, mace_dtype(device))
    if key not in _MACE_CACHE:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            _MACE_CACHE[key] = mace_mp(model=model, device=device, default_dtype=key[2])
    return _MACE_CACHE[key]


# ---------------------------------------------------------------------------
# Spec / system
# ---------------------------------------------------------------------------

class CeramicSpecError(ValueError):
    """Invalid ceramic spec; returned to the LLM so it can correct itself."""


MAX_CERAMIC_ATOMS = {"classical": 800, "mace": 300}   # pair search is O(N^2 x images)
MAX_MACE_ATOMS_GPU = 3000


def max_atoms(potential: str) -> int:
    if potential == "mace" and torch_device() != "cpu":
        return MAX_MACE_ATOMS_GPU
    return MAX_CERAMIC_ATOMS[potential]


@dataclass
class CeramicSpec:
    material: Optional[str] = None          # CERAMIC_LIBRARY key
    cif_path: Optional[str] = None          # or a CIF file the user provided
    potential: str = "auto"                 # auto | classical | mace
    min_cell_width: float = 10.0            # Angstrom; sets supercell size
    supercell: Optional[List[int]] = None   # explicit repetition overrides min_cell_width
    temperature_K: float = 300.0
    pressure_GPa: float = 0.0
    mace_model: str = "small"              # "medium" is more accurate, ~2.5x slower
    seed: int = 0


@dataclass
class CeramicSystem:
    spec: CeramicSpec
    atoms: Any                               # ase.Atoms (current state)
    potential: str
    reference: Optional[Dict[str, Any]]      # experimental values for library materials
    unit_reps: Tuple[int, int, int]
    relaxed: bool = False
    relax_info: Optional[Dict[str, Any]] = None

    def calculator(self):
        if self.potential == "classical":
            return _make_pedone_calculator()
        return _make_mace_calculator(self.spec.mace_model)


def build_ceramic(spec: CeramicSpec) -> CeramicSystem:
    _require_ase()
    from ase.build import make_supercell  # noqa: F401
    if bool(spec.material) == bool(spec.cif_path):
        raise CeramicSpecError("give exactly one of material (library key) or cif_path")
    if spec.material:
        if spec.material not in CERAMIC_LIBRARY:
            raise CeramicSpecError(f"unknown material {spec.material!r}; "
                                   f"library: {sorted(CERAMIC_LIBRARY)} (or pass a CIF)")
        unit = _build_unit(spec.material)
        ref = CERAMIC_LIBRARY[spec.material]["experimental"]
    else:
        from ase.io import read
        try:
            unit = read(spec.cif_path)
        except Exception as e:
            raise CeramicSpecError(f"could not read CIF {spec.cif_path!r}: {e}") from e
        ref = None
    if not (0 < spec.temperature_K <= 5000):
        raise CeramicSpecError("temperature_K must be in (0, 5000]")
    if not (-10 <= spec.pressure_GPa <= 100):
        raise CeramicSpecError("pressure_GPa must be in [-10, 100]")

    pot = spec.potential
    if pot == "auto":
        pot = "classical" if _pedone_covers(unit) else "mace"
    if pot == "classical" and not _pedone_covers(unit):
        raise CeramicSpecError(
            f"classical (Pedone) potential covers only {sorted(PEDONE_CHARGES)} oxides with "
            f"neutral Pedone charges; {unit.get_chemical_formula()} needs potential='mace'")
    if pot == "mace" and not mace_available():
        raise CeramicNotConfiguredError(
            f"{unit.get_chemical_formula()} needs the MACE potential, which is not installed "
            "(pip install mace-torch)")
    if pot not in ("classical", "mace"):
        raise CeramicSpecError("potential must be auto, classical, or mace")

    reps = tuple(spec.supercell) if spec.supercell else _supercell_reps(np.array(unit.get_cell()),
                                                                       spec.min_cell_width)
    atoms = unit.repeat(reps)
    if len(atoms) > max_atoms(pot):
        raise CeramicSpecError(f"{len(atoms)} atoms exceeds the {pot} limit of "
                               f"{max_atoms(pot)}; lower min_cell_width or supercell")
    sysm = CeramicSystem(spec=spec, atoms=atoms, potential=pot, reference=ref, unit_reps=reps)
    atoms.calc = sysm.calculator()
    return sysm


# ---------------------------------------------------------------------------
# Relaxation and elastic constants
# ---------------------------------------------------------------------------

def _cell_filter(atoms, pressure_GPa: float):
    from ase import units
    try:
        from ase.filters import FrechetCellFilter as F
    except Exception:
        from ase.constraints import ExpCellFilter as F  # older ASE
    return F(atoms, scalar_pressure=pressure_GPa * units.GPa)


def lattice_summary(atoms, reps: Tuple[int, int, int]) -> Dict[str, Any]:
    """Conventional-cell lattice parameters (supercell divided by repetitions)."""
    cp = atoms.cell.cellpar()
    abc = [cp[k] / reps[k] for k in range(3)]
    mass = float(np.sum(atoms.get_masses()))
    vol = atoms.get_volume()
    return {"a": abc[0], "b": abc[1], "c": abc[2], "alpha": cp[3], "beta": cp[4], "gamma": cp[5],
            "volume_per_atom_A3": vol / len(atoms),
            "density_g_cm3": mass / vol * AMU_PER_A3_TO_G_PER_CM3}


def relax_cell(sysm: CeramicSystem, fmax: float = 0.01, max_steps: int = 400) -> Dict[str, Any]:
    from ase.optimize import FIRE
    atoms = sysm.atoms
    atoms.calc = sysm.calculator()
    t0 = time.time()
    opt = FIRE(_cell_filter(atoms, sysm.spec.pressure_GPa), logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=max_steps))
    out = lattice_summary(atoms, sysm.unit_reps)
    out.update(energy_per_atom_eV=atoms.get_potential_energy() / len(atoms),
               converged=converged, optimizer_steps=opt.nsteps,
               max_residual_force_eV_A=float(np.max(np.linalg.norm(atoms.get_forces(), axis=1))),
               wall_seconds=round(time.time() - t0, 1))
    sysm.relaxed, sysm.relax_info = True, out
    return out


def elastic_constants(sysm: CeramicSystem, delta: float = 0.005, fmax: float = 0.005) -> Dict[str, Any]:
    """C_ij (GPa) by central finite differences of stress under +/- Voigt strains,
    relaxing internal coordinates at each fixed strained cell (needed for
    non-centrosymmetric / multi-sublattice crystals like quartz or corundum)."""
    from ase.optimize import BFGS
    if not sysm.relaxed:
        relax_cell(sysm)
    base = sysm.atoms.copy()
    cell0 = np.array(base.get_cell())
    C = np.zeros((6, 6))
    voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    t0 = time.time()
    for col, (a, b) in enumerate(voigt):
        stresses = []
        for sgn in (+1, -1):
            eps = np.zeros((3, 3))
            if a == b:
                eps[a, a] = sgn * delta
            else:
                eps[a, b] = eps[b, a] = sgn * delta / 2.0   # engineering shear strain = delta
            at = base.copy()
            at.set_cell(cell0 @ (np.eye(3) + eps), scale_atoms=True)
            at.calc = sysm.calculator()
            BFGS(at, logfile=None).run(fmax=fmax, steps=200)
            stresses.append(at.get_stress(voigt=True))
        C[:, col] = (stresses[0] - stresses[1]) / (2 * delta) * EV_PER_A3_TO_GPA
    C = 0.5 * (C + C.T)
    S = np.linalg.inv(C)
    KV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
    GV = (C[0, 0] + C[1, 1] + C[2, 2] - (C[0, 1] + C[0, 2] + C[1, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15
    KR = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    GR = 15 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
               + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    K, G = (KV + KR) / 2, (GV + GR) / 2
    return {
        "C_GPa": np.round(C, 1).tolist(),
        "bulk_modulus_GPa": K, "shear_modulus_GPa": G,
        "youngs_modulus_GPa": 9 * K * G / (3 * K + G),
        "poisson_ratio": (3 * K - 2 * G) / (2 * (3 * K + G)),
        "pugh_ratio_G_over_K": G / K,
        "mechanically_stable": bool(np.all(np.linalg.eigvalsh(C) > 0)),
        "averaging": "Voigt-Reuss-Hill, 0 K, relaxed-ion",
        "wall_seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Molecular dynamics
# ---------------------------------------------------------------------------

@dataclass
class CeramicTrajectory:
    positions: List[np.ndarray] = field(default_factory=list)
    cells: List[np.ndarray] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    target_temperatures: List[float] = field(default_factory=list)
    energies: List[float] = field(default_factory=list)
    times_fs: List[float] = field(default_factory=list)
    n_steps: int = 0
    timestep_fs: float = 1.0
    ensemble: str = "npt"
    wall_seconds: float = 0.0
    protocol: str = "equilibrium"


def run_ceramic_md(sysm: CeramicSystem, n_steps: int, ensemble: str = "npt",
                   temperature_K: Optional[float] = None, timestep_fs: float = 1.0,
                   n_frames: int = 100, schedule: Optional[List[Tuple[int, float]]] = None,
                   traj: Optional[CeramicTrajectory] = None) -> CeramicTrajectory:
    """NPT (Berendsen, isotropic) or NVT (Langevin). `schedule` is a list of
    (n_steps, target_T) segments with linear ramps between them (melt-quench)."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    atoms = sysm.atoms
    atoms.calc = sysm.calculator()
    T0 = temperature_K or sysm.spec.temperature_K
    if schedule is None:
        schedule = [(n_steps, T0)]
    total = sum(s for s, _ in schedule)
    traj = traj or CeramicTrajectory(timestep_fs=timestep_fs, ensemble=ensemble)
    if traj.n_steps == 0:
        MaxwellBoltzmannDistribution(atoms, temperature_K=schedule[0][1],
                                     rng=np.random.default_rng(sysm.spec.seed))
        Stationary(atoms)
    dt = timestep_fs * units.fs
    if ensemble == "npt":
        dyn = NPTBerendsen(atoms, timestep=dt, temperature_K=schedule[0][1],
                           pressure_au=sysm.spec.pressure_GPa * units.GPa,
                           taut=100 * units.fs, taup=500 * units.fs,
                           compressibility_au=1.0 / (100 * units.GPa))
    elif ensemble == "nvt":
        dyn = Langevin(atoms, timestep=dt, temperature_K=schedule[0][1], friction=0.01 / units.fs,
                       rng=np.random.default_rng(sysm.spec.seed + traj.n_steps))
    else:
        raise CeramicSpecError("ensemble must be npt or nvt")
    every = max(1, total // n_frames)
    t0 = time.time()
    done, T_prev = 0, schedule[0][1]
    for seg_steps, T_target in schedule:
        for s in range(seg_steps):
            T_now = T_prev + (T_target - T_prev) * (s + 1) / seg_steps
            dyn.set_temperature(temperature_K=T_now)
            dyn.run(1)
            done += 1
            if done % every == 0:
                traj.positions.append(atoms.get_positions().copy())
                traj.cells.append(np.array(atoms.get_cell()))
                traj.temperatures.append(float(atoms.get_temperature()))
                traj.target_temperatures.append(T_now)
                traj.energies.append(float(atoms.get_potential_energy()) / len(atoms))
                traj.times_fs.append((traj.n_steps + done) * timestep_fs)
        T_prev = T_target
    traj.n_steps += total
    traj.wall_seconds += time.time() - t0
    if len(schedule) > 1:
        traj.protocol = "melt-quench: " + " -> ".join(f"{T:.0f} K x {n} steps" for n, T in schedule)
    return traj


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _block(series: np.ndarray, n_blocks: int = 5) -> Dict[str, float]:
    series = np.asarray(series, float)
    n = len(series) // n_blocks * n_blocks
    if n < n_blocks:
        return {"mean": float(np.mean(series)), "sem": float("nan")}
    b = series[:n].reshape(n_blocks, -1).mean(axis=1)
    return {"mean": float(np.mean(series)), "sem": float(np.std(b, ddof=1) / math.sqrt(n_blocks))}


def partial_rdf(frames: List[np.ndarray], cells: List[np.ndarray], symbols: List[str],
                pair: Tuple[str, str], rmax: float = 6.0, nbins: int = 240) -> Tuple[np.ndarray, np.ndarray]:
    sym = np.array(symbols)
    ia, ib = np.where(sym == pair[0])[0], np.where(sym == pair[1])[0]
    edges = np.linspace(0, rmax, nbins + 1)
    hist = np.zeros(nbins)
    rho_sum = 0.0
    for pos, cell in zip(frames, cells):
        i, j, d, _ = pair_vectors(pos, cell, rmax)
        m = (sym[i] == pair[0]) & (sym[j] == pair[1])
        hist += np.histogram(d[m], bins=edges)[0]
        rho_sum += len(ib) / abs(np.linalg.det(cell))
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 4 * np.pi * r ** 2 * (edges[1] - edges[0])
    g = hist / (len(frames) * len(ia) * (rho_sum / len(frames)) * shell)
    return r, g


def _first_peak_and_min(r: np.ndarray, g: np.ndarray, sigma: float = 0.1) -> Tuple[Optional[float], Optional[float]]:
    """First-shell peak and the first minimum after it, on a Gaussian-smoothed
    g(r). Smoothing (sigma = 0.1 A) merges a split first shell — e.g.
    corundum's 3 short + 3 long Al-O bonds (1.86 / 1.97 A), which in a 0 K
    crystal are two sharp peaks — so the cutoff falls after the whole shell."""
    if not np.any(g > 0):
        return None, None
    dr = r[1] - r[0]
    half = int(4 * sigma / dr)
    x = np.arange(-half, half + 1) * dr
    k = np.exp(-0.5 * (x / sigma) ** 2)
    gs = np.convolve(g, k / k.sum(), mode="same")
    nz = np.nonzero(gs > 0.05 * gs.max())[0][0]            # start of the first shell
    ip = nz + int(np.argmax(gs[nz:nz + int(1.0 / dr)]))      # peak within 1 A of shell onset
    # first local minimum after the peak, or the first point where g has
    # dropped to ~0 (glasses and covalent solids have an empty gap after the
    # first shell, where a strict local-minimum test only fires much later)
    floor = 0.01 * gs[ip]
    for m in range(ip + 1, len(gs) - 1):
        if gs[m] < floor or (gs[m] <= gs[m - 1] and gs[m] < gs[m + 1]):
            return float(r[ip]), float(r[m])
    return float(r[ip]), None


def coordination(frames, cells, symbols, center: str, ligand: str, rcut: float) -> Dict[str, Any]:
    sym = np.array(symbols)
    idx = np.where(sym == center)[0]
    counts = []
    for pos, cell in zip(frames, cells):
        i, j, _, _ = pair_vectors(pos, cell, rcut)
        m = (sym[i] == center) & (sym[j] == ligand)
        cn = np.bincount(i[m], minlength=len(sym))[idx]
        counts.append(cn)
    counts = np.concatenate(counts)
    vals, freq = np.unique(counts, return_counts=True)
    return {"mean": float(np.mean(counts)), "cutoff_A": rcut,
            "distribution": {int(v): round(float(f) / len(counts), 3) for v, f in zip(vals, freq)}}


def _msd_diffusion(traj: CeramicTrajectory, symbols: List[str], start: int = 0,
                   end: Optional[int] = None) -> Dict[str, Any]:
    """Per-species D from MSD slope (positions are unwrapped in ASE MD).
    Uses fractional-coordinate displacement mapped through the mean cell to
    remove barostat rescaling."""
    pos, cells = traj.positions[start:end], np.array(traj.cells[start:end])
    if len(pos) < 10:
        return {}
    frac = np.array([p @ np.linalg.inv(c) for p, c in zip(pos, cells)])
    cart = frac @ cells.mean(axis=0)
    t = np.array(traj.times_fs[start:end])
    sym = np.array(symbols)
    out = {}
    for s in sorted(set(symbols)):
        m = sym == s
        msd = np.mean(np.sum((cart[:, m] - cart[0, m]) ** 2, axis=2), axis=1)
        mobile = bool(msd[-1] > 1.0)       # > 1 A^2: beyond thermal vibration amplitude
        slope = np.polyfit(t - t[0], msd, 1)[0]                  # A^2 / fs
        out[s] = {"mobile": mobile, "final_msd_A2": float(msd[-1]),
                  "D_cm2_s": slope / 6.0 * 1e-1 if mobile else None}
    return out


def segment_mobility(traj: CeramicTrajectory, symbols: List[str], temperature_K: float) -> Dict[str, Any]:
    """MSD over the frames held at `temperature_K` (e.g. the melt segment of a
    melt-quench). A real melt needs every species to move several A — otherwise
    the 'glass' is just a heated-and-cooled crystal."""
    idx = [k for k, T in enumerate(traj.target_temperatures) if abs(T - temperature_K) < 1e-6]
    if len(idx) < 10:
        return {"melted": False, "note": "too few frames at the melt temperature"}
    res = _msd_diffusion(traj, symbols, idx[0], idx[-1] + 1)
    melted = all(v["final_msd_A2"] > 4.0 for v in res.values())   # every species moved > 2 A rms
    return {"melted": melted, "per_species_msd_A2": {k: v["final_msd_A2"] for k, v in res.items()},
            "criterion": "every species rms displacement > 2 A during the melt segment"}


CERAMIC_PROPERTIES = ["lattice", "density", "rdf", "coordination", "diffusion", "energy"]


def ceramic_properties(sysm: CeramicSystem, traj: CeramicTrajectory,
                       properties: Optional[List[str]] = None, burn_in_fraction: float = 0.3) -> Dict[str, Any]:
    props = properties or CERAMIC_PROPERTIES
    bad = sorted(set(props) - set(CERAMIC_PROPERTIES))
    if bad:
        raise CeramicSpecError(f"unknown properties {bad}; available {CERAMIC_PROPERTIES}")
    if len(traj.positions) < 10:
        raise CeramicSpecError("trajectory has < 10 frames; run longer")
    b = int(len(traj.positions) * burn_in_fraction)
    frames, cells = traj.positions[b:], traj.cells[b:]
    symbols = sysm.atoms.get_chemical_symbols()
    mass = float(np.sum(sysm.atoms.get_masses()))
    vols = np.array([abs(np.linalg.det(c)) for c in cells])
    out: Dict[str, Any] = {"n_atoms": len(symbols), "potential": sysm.potential,
                           "ensemble": traj.ensemble, "protocol": traj.protocol,
                           "simulated_time_ps": traj.n_steps * traj.timestep_fs / 1000,
                           "production_frames": len(frames),
                           "mean_temperature_K": _block(traj.temperatures[b:]),
                           "target_temperature_K": traj.target_temperatures[-1]}
    if "density" in props:
        out["density_g_cm3"] = _block(mass / vols * AMU_PER_A3_TO_G_PER_CM3)
        out["volume_per_atom_A3"] = _block(vols / len(symbols))
    if "lattice" in props:
        cps = np.array([_cellpar(c) for c in cells])
        reps = sysm.unit_reps
        out["lattice_A"] = {k: _block(cps[:, i] / reps[i]) for i, k in enumerate("abc")}
        out["lattice_angles_deg"] = {k: float(np.mean(cps[:, 3 + i])) for i, k in
                                     enumerate(["alpha", "beta", "gamma"])}
    if "energy" in props:
        out["potential_energy_per_atom_eV"] = _block(traj.energies[b:])
    if "rdf" in props or "coordination" in props:
        species = sorted(set(symbols))
        anion = "O" if "O" in species else ("C" if "C" in species else None)
        pairs = [(c, anion) for c in species if anion and c != anion] or [(species[0], species[-1])]
        rdf_out, cn_out = {}, {}
        for p in pairs:
            r, g = partial_rdf(frames[::max(1, len(frames) // 30)], cells[::max(1, len(cells) // 30)],
                               symbols, p)
            peak, rmin = _first_peak_and_min(r, g)
            rdf_out[f"{p[0]}-{p[1]}"] = {"first_peak_A": peak, "first_min_A": rmin,
                                        "g_at_peak": float(np.max(g))}
            if "coordination" in props and rmin:
                cn_out[f"{p[0]}-{p[1]}"] = coordination(frames[::max(1, len(frames) // 20)],
                                                        cells[::max(1, len(cells) // 20)],
                                                        symbols, p[0], p[1], rmin)
        if "rdf" in props:
            out["rdf"] = rdf_out
        if "coordination" in props:
            out["coordination"] = cn_out
    if "diffusion" in props:
        out["diffusion"] = _msd_diffusion(traj, symbols, start=b)
        out["diffusion_note"] = ("mobile=false (MSD < 1 A^2 over the production window) means "
                                 "vibrating in place, i.e. solid; D is reported only for mobile "
                                 "species and is order-of-magnitude for ps runs.")
    T = np.array(traj.temperatures[b:])
    half = len(vols) // 2
    a, c = _block(vols[:half], 3), _block(vols[half:], 3)
    pooled = math.sqrt(np.nan_to_num(a["sem"]) ** 2 + np.nan_to_num(c["sem"]) ** 2)
    out["diagnostics"] = {
        "volume_drift_over_sem": abs(a["mean"] - c["mean"]) / pooled if pooled > 0 else float("inf"),
        "temperature_ratio": float(np.mean(T) / traj.target_temperatures[-1]),
    }
    d = out["diagnostics"]
    d["looks_equilibrated"] = bool(d["volume_drift_over_sem"] < 3 and abs(d["temperature_ratio"] - 1) < 0.1)
    if sysm.reference:
        out["experimental_reference"] = sysm.reference
    return _round(out)


def _cellpar(cell: np.ndarray) -> np.ndarray:
    a, b, c = (np.linalg.norm(v) for v in cell)
    al = math.degrees(math.acos(np.dot(cell[1], cell[2]) / (b * c)))
    be = math.degrees(math.acos(np.dot(cell[0], cell[2]) / (a * c)))
    ga = math.degrees(math.acos(np.dot(cell[0], cell[1]) / (a * b)))
    return np.array([a, b, c, al, be, ga])


def thermal_expansion(points: List[Tuple[float, float]]) -> Dict[str, Any]:
    """points: (temperature_K, mean volume per atom). Linear fit of V(T)."""
    if len(points) < 2:
        raise CeramicSpecError("need runs at >= 2 temperatures")
    T, V = np.array(points).T
    slope, icpt = np.polyfit(T, V, 1)
    Vmean = float(np.mean(V))
    beta = slope / Vmean
    return {"volumetric_K^-1": beta, "linear_K^-1_isotropic_estimate": beta / 3,
            "temperatures_K": T.tolist(), "volume_per_atom_A3": V.tolist(),
            "note": "isotropic barostat: anisotropic expansion (c/a change) is not captured"}


def _round(d: Any, digits: int = 4) -> Any:
    if isinstance(d, dict):
        return {k: _round(v, digits) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_round(v, digits) for v in d]
    if isinstance(d, (float, np.floating)):
        x = float(d)
        return float(f"{x:.{digits}g}") if math.isfinite(x) and x != 0 else x
    if isinstance(d, np.integer):
        return int(d)
    return d


# ---------------------------------------------------------------------------
# Validation harness
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _validate_ewald() -> List[Check]:
    from ase.build import bulk
    out = []
    # 1. Madelung constant of NaCl with +/-1 charges: E/ion pair = -1.747565 ke / r_nn
    a0 = 5.64
    at = bulk("NaCl", "rocksalt", a=a0, cubic=True).repeat(2)
    q = np.array([1.0 if s == "Na" else -1.0 for s in at.get_chemical_symbols()])
    e, _, _ = ewald_and_pairs(at.get_positions(), np.array(at.get_cell()), q,
                              at.get_chemical_symbols(), {}, rcut=10.0)
    M = -e / (len(at) / 2) * (a0 / 2) / KE_EV_A
    out.append(Check("Ewald NaCl Madelung constant", abs(M - 1.747565) < 1e-5, f"{M:.6f} (exact 1.747565)"))

    # 2. alpha independence on a distorted, charged-ion, multi-species cell
    at = _build_unit("Al2O3")
    rng = np.random.default_rng(0)
    at.positions += rng.normal(scale=0.05, size=at.positions.shape)
    at.set_cell(np.array(at.get_cell()) @ (np.eye(3) + rng.normal(scale=0.01, size=(3, 3))), scale_atoms=True)
    syms = at.get_chemical_symbols()
    q = np.array([PEDONE_CHARGES[s] for s in syms])
    es = [ewald_and_pairs(at.get_positions(), np.array(at.get_cell()), q, syms, PEDONE_PAIRS,
                          rcut=10.0, alpha=al)[0] for al in (0.40, 0.50)]
    out.append(Check("Ewald splitting-parameter independence", abs(es[0] - es[1]) < 1e-4,
                     f"E(alpha=0.40) - E(alpha=0.50) = {es[0] - es[1]:.2e} eV"))

    # 3. finite-difference forces and stress
    pos, cell = at.get_positions(), np.array(at.get_cell())
    e0, f, s = ewald_and_pairs(pos, cell, q, syms, PEDONE_PAIRS)
    h = 1e-5
    ferr = 0.0
    for (ia, k) in [(0, 0), (3, 1), (7, 2), (12, 0), (20, 2)]:
        p1, p2 = pos.copy(), pos.copy()
        p1[ia, k] += h
        p2[ia, k] -= h
        fd = -(ewald_and_pairs(p1, cell, q, syms, PEDONE_PAIRS)[0]
               - ewald_and_pairs(p2, cell, q, syms, PEDONE_PAIRS)[0]) / (2 * h)
        ferr = max(ferr, abs(fd - f[ia, k]))
    out.append(Check("finite-difference forces (Pedone, distorted corundum)", ferr < 1e-4,
                     f"max |F_fd - F| = {ferr:.1e} eV/A"))
    vol = abs(np.linalg.det(cell))
    serr = 0.0
    for (a_, b_) in [(0, 0), (2, 2), (0, 1), (1, 2)]:
        eps = np.zeros((3, 3))
        eps[a_, b_] = eps[b_, a_] = h if a_ != b_ else h
        def E(sign):
            M_ = np.eye(3) + sign * eps
            return ewald_and_pairs(pos @ M_, cell @ M_, q, syms, PEDONE_PAIRS)[0]
        fd = (E(1) - E(-1)) / (2 * h) / vol
        if a_ != b_:
            fd /= 2.0     # symmetric shear strain perturbs both (a,b) and (b,a)
        serr = max(serr, abs(fd - s[a_, b_]) * EV_PER_A3_TO_GPA)
    out.append(Check("finite-difference stress (Pedone, distorted corundum)", serr < 1e-3,
                     f"max |sigma_fd - sigma| = {serr:.1e} GPa"))
    return out


def _validate_against_experiment(potential: str, materials: List[str],
                                 with_elastic: List[str]) -> List[Check]:
    out = []
    for m in materials:
        sysm = build_ceramic(CeramicSpec(material=m, potential=potential, min_cell_width=6.0))
        r = relax_cell(sysm)
        ref = sysm.reference
        for key in ("a", "c"):
            if key in ref:
                rel = (r[key] - ref[key]) / ref[key]
                out.append(Check(f"{potential} {m} lattice {key}", abs(rel) < 0.03,
                                 f"{r[key]:.3f} A vs exp {ref[key]} ({rel:+.1%})"))
        rel = (r["density_g_cm3"] - ref["density_g_cm3"]) / ref["density_g_cm3"]
        out.append(Check(f"{potential} {m} density", abs(rel) < 0.08,
                         f"{r['density_g_cm3']:.3f} vs exp {ref['density_g_cm3']} g/cm3 ({rel:+.1%})"))
        pair, cn_ref = next(iter(ref["coordination"].items()))
        c, l = pair.split("-")
        rr, g = partial_rdf([sysm.atoms.get_positions()], [np.array(sysm.atoms.get_cell())],
                            sysm.atoms.get_chemical_symbols(), (c, l))
        _, rmin = _first_peak_and_min(rr, g)
        cn = coordination([sysm.atoms.get_positions()], [np.array(sysm.atoms.get_cell())],
                          sysm.atoms.get_chemical_symbols(), c, l, rmin)
        out.append(Check(f"{potential} {m} {pair} coordination", abs(cn["mean"] - cn_ref) < 0.05,
                         f"{cn['mean']:.2f} vs {cn_ref}"))
        if m in with_elastic and "bulk_modulus_GPa" in ref:
            el = elastic_constants(sysm)
            rel = (el["bulk_modulus_GPa"] - ref["bulk_modulus_GPa"]) / ref["bulk_modulus_GPa"]
            out.append(Check(f"{potential} {m} bulk modulus", abs(rel) < 0.25 and el["mechanically_stable"],
                             f"{el['bulk_modulus_GPa']:.0f} vs exp {ref['bulk_modulus_GPa']} GPa ({rel:+.0%})"))
    return out


def _validate_library_structures() -> List[Check]:
    """Every hand-entered reference structure must reproduce its own
    experimental density and cation coordination before anything is simulated."""
    out = []
    for key, entry in CERAMIC_LIBRARY.items():
        at = _build_unit(key)
        rho = float(np.sum(at.get_masses())) / at.get_volume() * AMU_PER_A3_TO_G_PER_CM3
        ref = entry["experimental"]
        pair, cn_ref = next(iter(ref["coordination"].items()))
        c, l = pair.split("-")
        big = at.repeat(_supercell_reps(np.array(at.get_cell()), 8.0))
        r, g = partial_rdf([big.get_positions()], [np.array(big.get_cell())], big.get_chemical_symbols(), (c, l))
        _, rmin = _first_peak_and_min(r, g)
        cn = coordination([big.get_positions()], [np.array(big.get_cell())], big.get_chemical_symbols(),
                          c, l, rmin)["mean"] if rmin else float("nan")
        ok = abs(rho - ref["density_g_cm3"]) / ref["density_g_cm3"] < 0.03 and abs(cn - cn_ref) < 0.05
        out.append(Check(f"library structure {key}", ok,
                         f"{at.get_chemical_formula()}: density {rho:.3f} vs {ref['density_g_cm3']}, "
                         f"{pair} CN {cn:.2f} vs {cn_ref}"))
    return out


def validate(include_mace: bool = True) -> List[Check]:
    _require_ase()
    checks = _validate_library_structures() + _validate_ewald()
    checks += _validate_against_experiment("classical", ["MgO", "CaO", "Al2O3", "SiO2_quartz"],
                                           with_elastic=["MgO", "Al2O3"])
    # NPT MD density at 300 K (classical MgO)
    sysm = build_ceramic(CeramicSpec(material="MgO", potential="classical", min_cell_width=8.0))
    relax_cell(sysm)
    traj = run_ceramic_md(sysm, n_steps=2000, ensemble="npt")
    p = ceramic_properties(sysm, traj, ["density", "coordination"])
    rho = p["density_g_cm3"]["mean"]
    checks.append(Check("classical MgO NPT 300 K density", abs(rho - 3.58) / 3.58 < 0.05,
                        f"{rho} +/- {p['density_g_cm3']['sem']} g/cm3 vs exp 3.58; "
                        f"<T> = {p['mean_temperature_K']['mean']} K"))
    if include_mace and mace_available():
        checks += _validate_against_experiment("mace", ["MgO", "TiO2_rutile", "SiC_3C", "Na2O"],
                                               with_elastic=["MgO"])
    elif include_mace:
        checks.append(Check("MACE validation", True, "SKIPPED (mace-torch not installed)"))
    return checks


if __name__ == "__main__":
    t0 = time.time()
    results = validate()
    for c in results:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
    nf = sum(not c.passed for c in results)
    print(f"\n{len(results) - nf}/{len(results)} checks passed ({time.time() - t0:.0f} s)")
    raise SystemExit(1 if nf else 0)
