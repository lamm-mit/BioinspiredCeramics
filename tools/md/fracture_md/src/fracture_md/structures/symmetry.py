"""Crystal symmetry (spglib) and elastic-tensor symmetry utilities.

Conventions
-----------
* ``AtomicSystem.box`` holds lattice vectors as **columns** ``L``; spglib wants
  them as rows, so the spglib cell is ``(L.T, frac, species)``.
* A spglib rotation ``W`` (integer, fractional basis) maps ``f -> W f``.  The
  corresponding Cartesian rotation is ``R = L W L^{-1}``; it is orthogonal for
  a genuine symmetry operation (checked).
* Voigt order is xx, yy, zz, yz, xz, xy (indices 0..5).  Stiffness ``C`` and
  stress use Voigt components without factors; strain and compliance use
  *engineering* shear strains (``e4 = 2 eps_yz`` ...), therefore

      S_ijkl = S_mn * w_m * w_n,   w = (1, 1, 1, 1/2, 1/2, 1/2)

  when converting a compliance matrix to the 4th-rank tensor, while
  ``C_ijkl = C_mn`` with no factors.
* A 4th-rank tensor transforms as ``T'_ijkl = R_ia R_jb R_kc R_ld T_abcd``.
  A crystal symmetry ``R`` leaves the elastic tensor of the crystal invariant.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Optional

import numpy as np
import spglib

VOIGT_PAIRS: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")
# 3x3 index pair -> Voigt index
_VOIGT_INDEX = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
_ENGINEERING_WEIGHT = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# spglib
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SpaceGroupInfo:
    number: int
    international: str
    point_group: str
    crystal_system: str
    rotations_frac: np.ndarray  # (n_ops, 3, 3) integer, unique point-group part in the input basis
    rotations_cart: np.ndarray  # (n_ops, 3, 3) Cartesian (orthogonal) rotations, unique
    symprec: float


def crystal_system_from_number(n: int) -> str:
    if n <= 2:
        return "triclinic"
    if n <= 15:
        return "monoclinic"
    if n <= 74:
        return "orthorhombic"
    if n <= 142:
        return "tetragonal"
    if n <= 167:
        return "trigonal"
    if n <= 194:
        return "hexagonal"
    return "cubic"


def spglib_cell(system) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lattice rows, fractional positions, atomic numbers-like ints) for spglib."""
    box = np.asarray(system.box, dtype=float)
    frac = np.asarray(system.frac, dtype=float) % 1.0
    # Use the element index (+1) as the type label: only equality matters to spglib.
    numbers = np.asarray(system.species, dtype=int) + 1
    return box.T.copy(), frac, numbers


def space_group(system, symprec: float = 1e-4, angle_tolerance: float = -1.0) -> SpaceGroupInfo:
    """Space group of an :class:`AtomicSystem` and its unique Cartesian point-group rotations.

    ``symprec`` is a Cartesian distance tolerance in Å (spglib semantics).
    Raises ``RuntimeError`` if spglib cannot determine the symmetry.
    """
    ds = spglib.get_symmetry_dataset(spglib_cell(system), symprec=symprec, angle_tolerance=angle_tolerance)
    if ds is None:
        raise RuntimeError(f"spglib failed to find symmetry: {spglib.get_error_message()}")
    L = np.asarray(system.box, dtype=float)
    Linv = np.linalg.inv(L)
    rot_frac = np.asarray(ds.rotations)
    uniq_f: list[np.ndarray] = []
    uniq_c: list[np.ndarray] = []
    seen: set[bytes] = set()
    for W in rot_frac:
        key = np.asarray(W, dtype=np.int64).tobytes()
        if key in seen:  # supercells repeat each rotation with lattice translations
            continue
        seen.add(key)
        R = L @ W @ Linv
        uniq_f.append(np.asarray(W))
        uniq_c.append(R)
    rc = np.asarray(uniq_c)
    orth_err = float(np.max(np.abs(np.einsum("nij,nkj->nik", rc, rc) - np.eye(3)[None])))
    if orth_err > 1e-3:
        raise RuntimeError(f"Cartesian symmetry operations are not orthogonal (max err {orth_err:.2e})")
    return SpaceGroupInfo(
        number=int(ds.number),
        international=str(ds.international),
        point_group=str(ds.pointgroup),
        crystal_system=crystal_system_from_number(int(ds.number)),
        rotations_frac=np.asarray(uniq_f),
        rotations_cart=rc,
        symprec=symprec,
    )


def conventional_cell(system, symprec: float = 1e-4):
    """Standardised conventional cell of ``system`` *without* rotating the lattice.

    Returns an :class:`AtomicSystem` in the same Cartesian frame (spglib
    ``no_idealize=True``) or ``None`` if spglib fails.  Used to rebuild a
    different supercell of the same crystal (size-convergence checks) and to
    report conventional lattice parameters.
    """
    from fracture_md.structures.system import AtomicSystem  # local: avoid import cycle

    out = spglib.standardize_cell(spglib_cell(system), to_primitive=False, no_idealize=True, symprec=symprec)
    if out is None:
        return None
    lat, pos, nums = out
    import jax.numpy as jnp

    return AtomicSystem(
        frac=jnp.asarray(np.asarray(pos, float) % 1.0),
        box=jnp.asarray(np.asarray(lat, float).T),
        species=jnp.asarray(np.asarray(nums, int) - 1, dtype=jnp.int32),
        elements=tuple(system.elements),
        pbc=tuple(system.pbc),
    )


def lattice_parameters(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(a, b, c) in Å and (alpha, beta, gamma) in degrees for column-vector ``box``."""
    b = np.asarray(box, dtype=float)
    v = [b[:, 0], b[:, 1], b[:, 2]]
    lengths = np.array([np.linalg.norm(x) for x in v])

    def ang(x, y):
        return np.degrees(np.arccos(np.clip(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y), -1, 1)))

    angles = np.array([ang(v[1], v[2]), ang(v[0], v[2]), ang(v[0], v[1])])
    return lengths, angles


# ---------------------------------------------------------------------------
# Voigt <-> tensor
# ---------------------------------------------------------------------------
def voigt_index(i: int, j: int) -> int:
    return int(_VOIGT_INDEX[i, j])


def voigt_to_tensor_stiffness(C: np.ndarray) -> np.ndarray:
    """6x6 Voigt stiffness -> 3x3x3x3 tensor (no factors)."""
    C = np.asarray(C, dtype=float)
    T = np.empty((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    T[i, j, k, m] = C[_VOIGT_INDEX[i, j], _VOIGT_INDEX[k, m]]
    return T


def tensor_to_voigt_stiffness(T: np.ndarray) -> np.ndarray:
    """3x3x3x3 stiffness -> 6x6 Voigt (symmetrised over minor index pairs)."""
    T = np.asarray(T, dtype=float)
    Ts = 0.25 * (T + T.transpose(1, 0, 2, 3) + T.transpose(0, 1, 3, 2) + T.transpose(1, 0, 3, 2))
    C = np.empty((6, 6))
    for m, (i, j) in enumerate(VOIGT_PAIRS):
        for n, (k, l) in enumerate(VOIGT_PAIRS):
            C[m, n] = Ts[i, j, k, l]
    return C


def voigt_to_tensor_compliance(S: np.ndarray) -> np.ndarray:
    """6x6 Voigt compliance (engineering shear strains) -> 3x3x3x3 tensor."""
    S = np.asarray(S, dtype=float)
    w = _ENGINEERING_WEIGHT
    T = np.empty((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for m in range(3):
                    a, b = _VOIGT_INDEX[i, j], _VOIGT_INDEX[k, m]
                    T[i, j, k, m] = S[a, b] * w[a] * w[b]
    return T


def tensor_to_voigt_compliance(T: np.ndarray) -> np.ndarray:
    """3x3x3x3 compliance -> 6x6 Voigt compliance (engineering shear strains)."""
    T = np.asarray(T, dtype=float)
    w = _ENGINEERING_WEIGHT
    S = np.empty((6, 6))
    for m, (i, j) in enumerate(VOIGT_PAIRS):
        for n, (k, l) in enumerate(VOIGT_PAIRS):
            S[m, n] = T[i, j, k, l] / (w[m] * w[n])
    return S


def strain_voigt_to_tensor(e: np.ndarray) -> np.ndarray:
    """Voigt strain (engineering shears) -> symmetric 3x3 strain tensor."""
    e = np.asarray(e, dtype=float)
    return np.array(
        [[e[0], e[5] / 2, e[4] / 2], [e[5] / 2, e[1], e[3] / 2], [e[4] / 2, e[3] / 2, e[2]]]
    )


def strain_tensor_to_voigt(eps: np.ndarray) -> np.ndarray:
    eps = np.asarray(eps, dtype=float)
    return np.array([eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]])


def stress_voigt_to_tensor(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    return np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])


def stress_tensor_to_voigt(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=float)
    return np.array([sig[0, 0], sig[1, 1], sig[2, 2], sig[1, 2], sig[0, 2], sig[0, 1]])


# ---------------------------------------------------------------------------
# Rotations and invariance
# ---------------------------------------------------------------------------
def rotate_tensor4(T: np.ndarray, R: np.ndarray) -> np.ndarray:
    """T'_ijkl = R_ia R_jb R_kc R_ld T_abcd."""
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, np.asarray(T, dtype=float), optimize=True)


def rotate_voigt_stiffness(C: np.ndarray, R: np.ndarray) -> np.ndarray:
    return tensor_to_voigt_stiffness(rotate_tensor4(voigt_to_tensor_stiffness(C), R))


def rotation_matrix_from_directions(z_dir: Iterable[float], x_dir: Optional[Iterable[float]] = None) -> np.ndarray:
    """Rotation whose rows are the new (x, y, z) axes expressed in the crystal frame.

    ``R @ v_crystal = v_lab``; ``z_dir`` becomes the lab z axis.
    """
    z = np.asarray(z_dir, dtype=float)
    z = z / np.linalg.norm(z)
    if x_dir is None:
        trial = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = trial - np.dot(trial, z) * z
    else:
        x = np.asarray(x_dir, dtype=float)
        if abs(np.dot(x, z)) > 1e-8 * np.linalg.norm(x):
            raise ValueError("x_dir must be perpendicular to z_dir")
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def check_tensor_invariance(C_voigt: np.ndarray, rotations: Iterable[np.ndarray]) -> float:
    """Max |C(R) - C| over all given Cartesian rotations (same unit as C)."""
    C = np.asarray(C_voigt, dtype=float)
    T = voigt_to_tensor_stiffness(C)
    worst = 0.0
    for R in rotations:
        worst = max(worst, float(np.max(np.abs(tensor_to_voigt_stiffness(rotate_tensor4(T, R)) - C))))
    return worst


def cubic_pattern_deviation(C_voigt: np.ndarray) -> float:
    """Max deviation of a 6x6 matrix from the cubic pattern (C11=C22=C33, C12=C13=C23, C44=C55=C66, rest 0)."""
    C = np.asarray(C_voigt, dtype=float)
    c11 = np.mean([C[0, 0], C[1, 1], C[2, 2]])
    c12 = np.mean([C[0, 1], C[0, 2], C[1, 2], C[1, 0], C[2, 0], C[2, 1]])
    c44 = np.mean([C[3, 3], C[4, 4], C[5, 5]])
    ideal = cubic_voigt(c11, c12, c44)
    return float(np.max(np.abs(C - ideal)))


def cubic_voigt(c11: float, c12: float, c44: float) -> np.ndarray:
    C = np.zeros((6, 6))
    C[:3, :3] = c12
    C[np.arange(3), np.arange(3)] = c11
    C[np.arange(3, 6), np.arange(3, 6)] = c44
    return C
