"""Engine-facing atomic system representation and ASE interchange.

Conventions (read carefully, these are the classic sources of silent bugs):

* ASE: ``atoms.cell`` has lattice vectors as **rows**; positions are Cartesian.
* JAX-MD ``space.periodic_general(box)``: real = ``box @ frac`` with lattice
  vectors as **columns** of ``box``.  Hence ``box = atoms.cell.T``.
* We always store **fractional** coordinates for the JAX path, because
  JAX-MD's ``periodic_general(fractional_coordinates=False)`` returns incorrect
  derivatives with respect to ``box`` (verified numerically; see
  docs/AUTODIFF.md).  With fractional coordinates JAX-MD's custom JVP makes
  ``grad(E)(S)`` a *real-space* gradient (eV/Å), and ``grad(E)(box)`` exact.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

from fracture_md.schema.material import StructureRecord


@dataclasses.dataclass(frozen=True)
class AtomicSystem:
    """Periodic atomic system in JAX-MD conventions (fractional coordinates)."""

    frac: jnp.ndarray  # (N, 3) fractional coordinates in [0, 1)
    box: jnp.ndarray  # (3, 3) lattice vectors as COLUMNS (angstrom)
    species: jnp.ndarray  # (N,) int, index into `elements`
    elements: tuple[str, ...]  # species index -> chemical symbol
    pbc: tuple[bool, bool, bool] = (True, True, True)

    @property
    def n_atoms(self) -> int:
        return int(self.frac.shape[0])

    @property
    def positions(self) -> jnp.ndarray:
        """Cartesian positions (N, 3) in angstrom."""
        return self.frac @ self.box.T

    @property
    def volume(self) -> float:
        return float(jnp.abs(jnp.linalg.det(self.box)))

    @property
    def symbols(self) -> list[str]:
        return [self.elements[int(s)] for s in np.asarray(self.species)]

    @property
    def masses_amu(self) -> jnp.ndarray:
        z = [atomic_numbers[s] for s in self.symbols]
        return jnp.asarray(atomic_masses[z], dtype=jnp.float64)

    def replace(self, **kw) -> "AtomicSystem":
        return dataclasses.replace(self, **kw)

    # ------------------------------------------------------------------ ASE
    @classmethod
    def from_ase(cls, atoms: Atoms, elements: tuple[str, ...] | None = None) -> "AtomicSystem":
        syms = atoms.get_chemical_symbols()
        if elements is None:
            elements = tuple(sorted(set(syms)))
        idx = {e: i for i, e in enumerate(elements)}
        frac = np.asarray(atoms.get_scaled_positions(wrap=True), dtype=np.float64)
        box = np.asarray(atoms.cell.array, dtype=np.float64).T
        return cls(
            frac=jnp.asarray(frac),
            box=jnp.asarray(box),
            species=jnp.asarray([idx[s] for s in syms], dtype=jnp.int32),
            elements=tuple(elements),
            pbc=tuple(bool(p) for p in atoms.pbc),  # type: ignore[arg-type]
        )

    def to_ase(self) -> Atoms:
        return Atoms(
            symbols=self.symbols,
            positions=np.asarray(self.positions),
            cell=np.asarray(self.box).T,
            pbc=self.pbc,
        )

    def to_record(self, source: str = "") -> StructureRecord:
        rec = StructureRecord(
            symbols=self.symbols,
            cell_rows_angstrom=np.asarray(self.box).T.tolist(),
            positions_angstrom=np.asarray(self.positions).tolist(),
            pbc=self.pbc,
            source=source,
        )
        payload = json.dumps(rec.model_dump(mode="json", exclude={"sha256", "source"}), sort_keys=True)
        return rec.model_copy(update={"sha256": hashlib.sha256(payload.encode()).hexdigest()})


def perpendicular_widths(box: np.ndarray) -> np.ndarray:
    """Distances between opposite faces of the cell (angstrom), one per lattice vector.

    ``box`` has lattice vectors as columns.  The minimum-image convention used by
    JAX-MD is exact only if every width exceeds twice the interaction range.
    """
    b = np.asarray(box, dtype=float)
    a = [b[:, 0], b[:, 1], b[:, 2]]
    vol = abs(np.linalg.det(b))
    return np.array(
        [
            vol / np.linalg.norm(np.cross(a[1], a[2])),
            vol / np.linalg.norm(np.cross(a[2], a[0])),
            vol / np.linalg.norm(np.cross(a[0], a[1])),
        ]
    )


def minimum_image_ok(box: np.ndarray, interaction_range: float) -> bool:
    return bool(np.all(perpendicular_widths(box) > 2.0 * interaction_range))


def auto_supercell(atoms: Atoms, interaction_range: float, periodic_axes=(0, 1, 2)) -> tuple[int, int, int]:
    """Smallest repeats so that every periodic width exceeds 2*interaction_range."""
    widths = perpendicular_widths(np.asarray(atoms.cell.array).T)
    reps = [1, 1, 1]
    for ax in periodic_axes:
        reps[ax] = int(np.floor(2.0 * interaction_range / widths[ax])) + 1
    return tuple(reps)  # type: ignore[return-value]
