"""Engine selection for the agent interface.

``select_engine(requested, kind)`` returns an :class:`EngineSelection` that
knows how to run one protocol kind:

* ``"jax_md"``: the built-in trusted runner ``engines.jax_md_engine.RUNNERS[kind]``
  (a ``"module:function"`` spec, resolved lazily; a callable is accepted too,
  which is how tests inject fast fake runners).  The runner is called directly
  with an :class:`~fracture_md.potentials.base.EnergyModel`, so that agent-
  registered potential implementations (``impl:<id>``) can be passed as models.
* ``"ase_reference"``: ``fracture_md.engines.ase_reference`` if that module is
  importable and exposes an Engine object (``ENGINE``, ``get_engine()`` or an
  ``*Engine`` class with ``name == "ase_reference"``).  It is driven through
  the generic ``Engine.run(protocol, candidate, potential_id, ...)`` contract
  and therefore only accepts built-in potential ids.
* ``"auto"``: ``jax_md`` if it has a runner for the kind, otherwise
  ``ase_reference`` if available and capable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.metadata as md
import inspect
from typing import Any, Callable, Optional

from fracture_md.engines.base import resolve


class EngineUnavailable(RuntimeError):
    """No engine can run the requested protocol kind (a pre-run rejection)."""


@dataclasses.dataclass
class EngineSelection:
    name: str
    version: str
    runner_id: str  # e.g. "builtin:fracture_md.physics.elastic:run_elastic_tensor@<sha12>"
    runner: Optional[Callable] = None  # jax_md: runner(protocol, candidate, model, structure=, context=)
    engine: Any = None  # ase_reference: Engine object with .run(protocol, candidate, potential_id, ...)
    accepts_energy_model: bool = True

    def run(self, protocol, candidate, model, potential_id: str, structure=None, context=None):
        if self.runner is not None:
            return self.runner(protocol, candidate, model, structure=structure, context=context)
        return self.engine.run(protocol, candidate, potential_id, structure=structure, context=context)


def _callable_id(fn: Callable, spec: str) -> str:
    try:
        sha = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:12]
    except (OSError, TypeError):
        sha = "nosource"
    return f"builtin:{spec}@{sha}"


def _jax_md_version() -> str:
    try:
        return f"jax-md {md.version('jax-md')} / jax {md.version('jax')}"
    except md.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def jax_md_runner(kind: str) -> tuple[Callable, str]:
    """(runner, runner_id) for the built-in trusted JAX-MD runner of `kind`."""
    from fracture_md.engines import jax_md_engine

    if kind not in jax_md_engine.RUNNERS:
        raise EngineUnavailable(f"jax_md engine has no runner for protocol kind {kind!r}")
    spec = jax_md_engine.RUNNERS[kind]
    if callable(spec):
        name = f"{getattr(spec, '__module__', '?')}:{getattr(spec, '__qualname__', repr(spec))}"
        return spec, _callable_id(spec, name)
    try:
        fn = resolve(spec)
    except (ImportError, AttributeError) as exc:
        raise EngineUnavailable(f"jax_md runner {spec!r} for {kind!r} is not importable: {exc}") from exc
    return fn, _callable_id(fn, spec)


def ase_reference_engine() -> Any:
    """The ASE/matscipy reference Engine object, or None if not installed/implemented."""
    try:
        mod = importlib.import_module("fracture_md.engines.ase_reference")
    except ImportError:
        return None
    if hasattr(mod, "ENGINE"):
        return mod.ENGINE
    if callable(getattr(mod, "get_engine", None)):
        return mod.get_engine()
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if inspect.isclass(obj) and attr.endswith("Engine") and getattr(obj, "name", None) == "ase_reference":
            try:
                return obj()
            except TypeError:
                continue
    return None


def _ase_capable(engine: Any, kind: str) -> bool:
    try:
        return kind in set(engine.capabilities())
    except Exception:  # noqa: BLE001 - a broken optional engine must not break dispatch
        return False


def select_engine(requested: str, kind: str, *, need_energy_model: bool = False) -> EngineSelection:
    """Choose an engine for protocol `kind`; raise EngineUnavailable with the reasons otherwise.

    `need_energy_model` is True when the potential is an agent-registered
    implementation that exists only as a JAX energy function.
    """
    reasons = []
    if requested in ("jax_md", "auto"):
        try:
            fn, rid = jax_md_runner(kind)
            return EngineSelection("jax_md", _jax_md_version(), rid, runner=fn)
        except EngineUnavailable as exc:
            reasons.append(str(exc))
            if requested == "jax_md":
                raise
    if requested in ("ase_reference", "auto"):
        eng = ase_reference_engine()
        if eng is None:
            reasons.append("ase_reference engine (fracture_md.engines.ase_reference) is not available")
        elif not _ase_capable(eng, kind):
            reasons.append(f"ase_reference engine does not support protocol kind {kind!r}")
        elif need_energy_model:
            reasons.append("ase_reference engine cannot run agent-registered JAX energy functions (impl:<id>)")
        else:
            return EngineSelection("ase_reference", str(getattr(eng, "version", "unknown")),
                                   f"engine:ase_reference:{kind}", engine=eng, accepts_energy_model=False)
    if requested not in ("jax_md", "ase_reference", "auto"):
        reasons.append(f"unknown engine {requested!r}")
    raise EngineUnavailable("; ".join(reasons))
