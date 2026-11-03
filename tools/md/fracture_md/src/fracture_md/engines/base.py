"""Engine contract.

An engine turns (protocol, candidate, potential) into a :class:`ProtocolOutput`
whose `properties`, `traction_separation` and `indentation_curves` entries use
the common schema.  Engines differ only in *how* they compute; they must not
invent new result fields.  Engine-specific diagnostics go in `raw` and are
never used for ranking or export.

Records from simulation runs state their engine in
``Provenance.engine_run`` (:func:`jax_engine_run` for the JAX harness,
:func:`lammps_engine_run` for LAMMPS decks whose output is imported); MD
indentation records require it.

Runner functions (one per protocol kind) have the signature

    run_<kind>(protocol, candidate, model, structure=None, context=None) -> ProtocolOutput

where `model` is the engine's potential object (an EnergyModel for the JAX
engine, an ASE calculator factory for the reference engine) and `structure`
optionally supplies a pre-relaxed AtomicSystem.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol as TypingProtocol, Union

from fracture_md.schema.common import CheckResult, EngineRunProvenance
from fracture_md.schema.material import Candidate
from fracture_md.schema.properties import IndentationCurve, PropertyRecord, TractionSeparationLaw


@dataclasses.dataclass
class RunContext:
    requested_by: str = "human"
    output_dir: Optional[str] = None  # where runners may write artifacts (plots, npz)
    implementation_ids: dict[str, str] = dataclasses.field(default_factory=dict)
    parent_ids: list[str] = dataclasses.field(default_factory=list)
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ProtocolOutput:
    properties: list[PropertyRecord] = dataclasses.field(default_factory=list)
    traction_separation: list[TractionSeparationLaw] = dataclasses.field(default_factory=list)
    checks: list[CheckResult] = dataclasses.field(default_factory=list)
    structure: Any = None  # e.g. relaxed AtomicSystem, reusable by later protocols
    artifacts: dict[str, Any] = dataclasses.field(default_factory=dict)  # numpy arrays etc.
    raw: dict[str, Any] = dataclasses.field(default_factory=dict)
    indentation_curves: list[IndentationCurve] = dataclasses.field(default_factory=list)  # P(h) with units

    @property
    def all_checks_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def merge(self, other: "ProtocolOutput") -> "ProtocolOutput":
        return ProtocolOutput(
            properties=self.properties + other.properties,
            traction_separation=self.traction_separation + other.traction_separation,
            checks=self.checks + other.checks,
            structure=other.structure if other.structure is not None else self.structure,
            artifacts={**self.artifacts, **other.artifacts},
            raw={**self.raw, **other.raw},
            indentation_curves=self.indentation_curves + other.indentation_curves,
        )


class Engine(TypingProtocol):
    name: str
    version: str

    def capabilities(self) -> set[str]: ...

    def run(
        self,
        protocol: Any,
        candidate: Candidate,
        potential: str,
        structure: Any = None,
        context: Optional[RunContext] = None,
    ) -> ProtocolOutput: ...


def resolve(spec: str):
    """Import 'package.module:function'."""
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn)


# --------------------------------------------------------------------------
# Engine-run provenance
# --------------------------------------------------------------------------
def sha256_file(path: Union[str, Path]) -> str:
    """sha256 hex digest of a file's bytes (decks, potential files, recipes)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jax_engine_run(*, potential_card_id: Optional[str] = None, potential_convention: str = "",
                   recipe_sha256: Optional[str] = None, command: str = "") -> EngineRunProvenance:
    """EngineRunProvenance of the current JAX process (version, dtype, matmul precision, XLA flags, device)."""
    import jax

    try:
        from importlib.metadata import version

        jmd = version("jax-md")
    except Exception:  # noqa: BLE001 - optional detail
        jmd = "unknown"
    x64 = bool(jax.config.read("jax_enable_x64"))
    mm = os.environ.get("JAX_DEFAULT_MATMUL_PRECISION") or getattr(jax.config, "jax_default_matmul_precision", None)
    return EngineRunProvenance(
        engine="jax",
        engine_version=f"jax {jax.__version__}, jax-md {jmd}",
        potential_card_id=potential_card_id,
        potential_convention=potential_convention,
        recipe_sha256=recipe_sha256,
        dtype="float64" if x64 else "float32",
        matmul_precision=str(mm) if mm else ("default" if x64 else None),
        xla_flags=os.environ.get("XLA_FLAGS"),
        accelerator=jax.default_backend(),
        command=command,
    )


def lammps_engine_run(*, deck: Union[str, Path], potential_files: Iterable[Union[str, Path]], lammps_version: str,
                      potential_card_id: Optional[str] = None, potential_convention: str = "",
                      deck_generator: str = "", recipe: Optional[Union[str, Path]] = None,
                      dtype: str = "float64", accelerator: str = "cpu", mpi_ranks: Optional[int] = None,
                      command: str = "") -> EngineRunProvenance:
    """EngineRunProvenance of a LAMMPS run: hashes the input deck, every potential file and the recipe."""
    return EngineRunProvenance(
        engine="lammps",
        engine_version=lammps_version,
        potential_card_id=potential_card_id,
        potential_convention=potential_convention,
        potential_files_sha256={Path(f).name: sha256_file(f) for f in potential_files},
        deck_sha256=sha256_file(deck),
        deck_generator=deck_generator,
        recipe_sha256=sha256_file(recipe) if recipe is not None else None,
        dtype=dtype,  # type: ignore[arg-type]
        accelerator=accelerator,
        mpi_ranks=mpi_ranks,
        command=command,
    )
