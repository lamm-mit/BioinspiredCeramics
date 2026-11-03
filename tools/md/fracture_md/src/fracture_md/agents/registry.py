"""Content-addressed store of versioned, agent-proposed implementations.

An agent that wants to change *how* something is computed (a new energy
kernel, a faster cleavage runner, ...) never edits trusted code in place.  It
registers the new source as an immutable, content-hashed implementation:

    implementations/<kind>/<name>/<sha256[:12]>/impl.py
    implementations/<kind>/<name>/<sha256[:12]>/manifest.json

with id ``"<name>@<sha256[:12]>"``.  A freshly registered implementation is
``pending``; only :func:`fracture_md.agents.gates.gate` can move it to
``trusted`` or ``rejected`` (by running the numerical regression gate and
recording every :class:`~fracture_md.schema.common.CheckResult` in the
manifest).  :meth:`ImplementationRegistry.load` re-hashes ``impl.py`` before
importing it, so an edited file is refused (tamper detection).

Trust model / honest limitations
--------------------------------
* The id is bound to the code: ``<name>@<sha12>`` names the directory
  ``<kind>/<name>/<sha12>``, and every read of a manifest requires
  ``manifest["id"] == impl_id``, ``manifest["name"]``/``["kind"]`` equal to the
  path components and ``manifest["sha256"][:12] == sha12``; :meth:`load`
  additionally re-hashes ``impl.py``.  Editing the source *and* rewriting the
  manifest hash therefore no longer passes (the id's sha12 would change).
* Trust is not a single field: :meth:`trust_problems` requires
  ``status == "trusted"`` **and** a recorded gate with ``passed: true``,
  the current ``GATE_VERSION`` and a non-empty list of gate results that all
  passed.  Flipping ``status`` by hand is refused, and so is a manifest gated
  by an older gate version (re-gate it).
* The manifest is still a plain JSON file: someone who hand-writes a complete
  fake gate record can forge trust.  The store is meant to be committed to
  git, whose history is the audit trail; this is integrity, not security.
* Loading executes the registered Python module in-process.  Only register
  code you are prepared to run; there is no sandbox.
* The gate verifies numerical consistency and regression against a trusted
  base, not that the physics is right for a new property class.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Optional, Union

ImplKind = Literal["potential_energy_fn", "protocol_runner"]
IMPL_KINDS: tuple[str, ...] = ("potential_energy_fn", "protocol_runner")
STATUSES: tuple[str, ...] = ("pending", "trusted", "rejected")
MANIFEST_VERSION = "0.1.0"

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,79}$")
_ID_RE = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9_.-]{0,79})@(?P<sha12>[0-9a-f]{12})$")


class RegistryError(ValueError):
    """Unknown id, bad name/kind, or inconsistent store."""


class TamperError(RuntimeError):
    """The stored source no longer matches the hash it was registered with."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_id(impl_id: str) -> tuple[str, str]:
    m = _ID_RE.fullmatch(impl_id) if isinstance(impl_id, str) else None
    if not m:
        raise RegistryError(f"malformed implementation id {impl_id!r}; expected '<name>@<sha256[:12]>'")
    return m.group("name"), m.group("sha12")


def _read_source(source: Union[str, Path]) -> tuple[str, str]:
    """Return (text, origin).  `source` is a path to a .py file or the source text itself."""
    # Only the file name is recorded as origin: absolute paths are machine-specific.
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8"), source.name
    if "\n" not in source and len(source) < 4096 and Path(source).is_file():
        return Path(source).read_text(encoding="utf-8"), Path(source).name
    return source, "<text>"


class ImplementationRegistry:
    """Filesystem-backed, content-addressed implementation store."""

    def __init__(self, root: Union[str, Path] = "implementations") -> None:
        self.root = Path(root)

    # ------------------------------------------------------------ paths
    def _dir(self, kind: str, name: str, sha12: str) -> Path:
        return self.root / kind / name / sha12

    def locate(self, impl_id: str) -> Path:
        """Directory holding `impl_id` (searches every kind)."""
        name, sha12 = parse_id(impl_id)
        hits = [self._dir(k, name, sha12) for k in IMPL_KINDS if (self._dir(k, name, sha12) / "manifest.json").is_file()]
        if not hits:
            raise RegistryError(f"implementation {impl_id!r} not found under {self.root}")
        if len(hits) > 1:  # pragma: no cover - would need two kinds with identical name+hash
            raise RegistryError(f"implementation {impl_id!r} is ambiguous: {hits}")
        return hits[0]

    def exists(self, impl_id: str) -> bool:
        try:
            self.locate(impl_id)
            return True
        except RegistryError:
            return False

    # ------------------------------------------------------------ register
    def register(
        self,
        kind: ImplKind,
        name: str,
        source: Union[str, Path],
        author: str,
        parent_id: Optional[str] = None,
        description: str = "",
        protocol_kind: Optional[str] = None,
        base_potential: Optional[str] = None,
    ) -> str:
        """Store `source` immutably and return its id ``"<name>@<sha256[:12]>"``.

        Registration is idempotent: re-registering identical source under the
        same name returns the existing id and leaves its manifest (including a
        recorded gate status) untouched.  The status of a new entry is
        ``pending``; run :func:`fracture_md.agents.gates.gate` to change it.
        """
        if kind not in IMPL_KINDS:
            raise RegistryError(f"kind must be one of {IMPL_KINDS}, got {kind!r}")
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
            raise RegistryError(f"invalid implementation name {name!r} (letters, digits, _ . -; start with a letter)")
        if kind == "protocol_runner" and not protocol_kind:
            raise RegistryError("protocol_runner implementations must declare protocol_kind")
        if parent_id is not None:
            parse_id(parent_id)  # format check; the parent may live in another store
        text, origin = _read_source(source)
        sha = sha256_text(text)
        impl_id = f"{name}@{sha[:12]}"
        d = self._dir(kind, name, sha[:12])
        if (d / "manifest.json").is_file():
            man = self.get_manifest(impl_id)
            if man["sha256"] != sha:  # pragma: no cover - 48-bit prefix collision
                raise RegistryError(f"hash-prefix collision for {impl_id}")
            return impl_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "impl.py").write_text(text, encoding="utf-8")
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "id": impl_id,
            "kind": kind,
            "name": name,
            "sha256": sha,
            "parent_id": parent_id,
            "author": author,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "protocol_kind": protocol_kind,
            "base_potential": base_potential,
            "source_origin": origin,
            "status": "pending",
            "gate": None,  # {gate_version, run_utc, passed, environment}
            "gate_results": [],  # list of CheckResult dicts
        }
        self._write_manifest(d, manifest)
        return impl_id

    # ------------------------------------------------------------ queries
    def get_manifest(self, impl_id: str) -> dict[str, Any]:
        """The manifest of `impl_id`, after checking that it is bound to the id and its path.

        Raises RegistryError for an unreadable/corrupt manifest and TamperError
        when the manifest's id, name, kind or sha256 prefix disagree with the
        id and the directory it lives in.
        """
        d = self.locate(impl_id)
        try:
            man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryError(f"{impl_id}: manifest.json is unreadable or corrupt ({exc})") from exc
        if not isinstance(man, dict):
            raise RegistryError(f"{impl_id}: manifest.json is not a JSON object")
        missing = [k for k in ("id", "kind", "name", "sha256", "status") if k not in man]
        if missing:
            raise RegistryError(f"{impl_id}: manifest.json lacks required field(s) {missing}")
        name, sha12 = parse_id(impl_id)
        kind_dir = d.parent.parent.name
        problems = []
        if man["id"] != impl_id:
            problems.append(f"manifest id {man['id']!r} != {impl_id!r}")
        if man["name"] != name or d.parent.name != name:
            problems.append(f"manifest name {man['name']!r} != id/path name {name!r}")
        if man["kind"] != kind_dir:
            problems.append(f"manifest kind {man['kind']!r} != path kind {kind_dir!r}")
        if not isinstance(man["sha256"], str) or man["sha256"][:12] != sha12 or d.name != sha12:
            problems.append(f"manifest sha256 {str(man['sha256'])[:12]!r}... does not start with the id's {sha12!r}")
        if man["status"] not in STATUSES:
            problems.append(f"unknown status {man['status']!r}")
        if problems:
            raise TamperError(f"{impl_id}: manifest is not bound to its id ({'; '.join(problems)}); the store was "
                              "edited by hand -- register the code again instead")
        return man

    def trust_problems(self, impl_id: str, man: Optional[dict] = None) -> list[str]:
        """Reasons why `impl_id` is not trusted (empty list = trusted).

        Trusted means: status 'trusted' AND a recorded gate with passed=True, the
        current gate version and a non-empty list of gate results that all passed.
        """
        from fracture_md.agents.gates import GATE_VERSION

        man = self.get_manifest(impl_id) if man is None else man
        out = []
        if man.get("status") != "trusted":
            out.append(f"status {man.get('status')!r}; only implementations that passed the regression gate "
                       "('trusted') may run")
            return out
        g = man.get("gate")
        res = man.get("gate_results")
        if not isinstance(g, dict) or g.get("passed") is not True:
            out.append("status says 'trusted' but no passing gate record is present")
        elif g.get("gate_version") != GATE_VERSION:
            out.append(f"gated with gate version {g.get('gate_version')!r}, current is {GATE_VERSION!r}; re-run the "
                       "gate (fracture-md impl gate <id>)")
        if not isinstance(res, list) or not res:
            out.append("no gate results are recorded")
        elif not all(isinstance(c, dict) and c.get("passed") is True for c in res):
            out.append("not every recorded gate check passed")
        return out

    def is_trusted(self, impl_id: str) -> bool:
        try:
            return not self.trust_problems(impl_id)
        except (RegistryError, TamperError):
            return False

    def status(self, impl_id: str) -> str:
        return self.get_manifest(impl_id)["status"]

    def source(self, impl_id: str) -> str:
        return (self.locate(impl_id) / "impl.py").read_text(encoding="utf-8")

    def source_path(self, impl_id: str) -> Path:
        return self.locate(impl_id) / "impl.py"

    def list(self, kind: Optional[str] = None, status: Optional[str] = None) -> list[dict[str, Any]]:
        """All manifests (optionally filtered), sorted by creation time."""
        out = []
        kinds = [kind] if kind else list(IMPL_KINDS)
        for k in kinds:
            for mf in sorted((self.root / k).glob("*/*/manifest.json")) if (self.root / k).is_dir() else []:
                try:
                    man = json.loads(mf.read_text(encoding="utf-8"))
                    man["id"]
                except (OSError, ValueError, KeyError, TypeError):
                    continue  # corrupt entries are skipped; get_manifest() reports them
                if status is None or man.get("status") == status:
                    out.append(man)
        return sorted(out, key=lambda m: (str(m.get("created_utc", "")), str(m["id"])))

    def verify(self, impl_id: str) -> bool:
        """True iff the manifest is bound to the id and impl.py still hashes to the registered sha256."""
        try:
            man = self.get_manifest(impl_id)
            return sha256_text(self.source(impl_id)) == man["sha256"]
        except (RegistryError, TamperError, OSError, UnicodeDecodeError):
            return False

    def lineage(self, impl_id: str) -> list[str]:
        """[impl_id, parent, grandparent, ...] as far as the parents exist in this store."""
        chain = [impl_id]
        man = self.get_manifest(impl_id)
        while man.get("parent_id") and self.exists(man["parent_id"]) and man["parent_id"] not in chain:
            chain.append(man["parent_id"])
            man = self.get_manifest(man["parent_id"])
        return chain

    # ------------------------------------------------------------ loading
    def load(self, impl_id: str) -> ModuleType:
        """Import the stored module after verifying its content hash.

        Raises :class:`TamperError` if ``impl.py`` was modified after
        registration.  The module is imported under a unique name derived from
        the id, so different versions never shadow each other.
        """
        man = self.get_manifest(impl_id)
        path = self.source_path(impl_id)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RegistryError(f"{impl_id}: impl.py is missing or unreadable ({exc})") from exc
        actual = sha256_text(text)
        if actual != man["sha256"]:
            raise TamperError(
                f"{impl_id}: impl.py hash {actual[:12]}... != registered {man['sha256'][:12]}...; "
                "the stored source was modified after registration (register it as a new version instead)"
            )
        name, sha12 = parse_id(impl_id)
        mod_name = f"fracture_md_impl__{man['kind']}__{re.sub(r'[^A-Za-z0-9_]', '_', name)}_{sha12}"
        cached = sys.modules.get(mod_name)
        if cached is not None and getattr(cached, "__impl_sha256__", None) == actual:
            return cached
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise RegistryError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        # Execute the exact bytes that were hashed (not a re-read of the file).
        code = compile(text, str(path), "exec")
        module.__impl_id__ = impl_id  # type: ignore[attr-defined]
        module.__impl_sha256__ = actual  # type: ignore[attr-defined]
        sys.modules[mod_name] = module
        try:
            exec(code, module.__dict__)
        except BaseException:
            sys.modules.pop(mod_name, None)
            raise
        return module

    # ------------------------------------------------------------ status (gate only)
    def _record_gate(self, impl_id: str, *, passed: bool, checks: list[dict], gate_info: dict) -> dict:
        """Write gate outcome into the manifest.  Called only by agents.gates.gate()."""
        d = self.locate(impl_id)
        man = self.get_manifest(impl_id)
        man["status"] = "trusted" if passed else "rejected"
        man["gate_results"] = checks
        man["gate"] = {**gate_info, "passed": passed}
        self._write_manifest(d, man)
        return man

    @staticmethod
    def _write_manifest(d: Path, manifest: dict) -> None:
        tmp = d / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        tmp.replace(d / "manifest.json")

    # convenience
    def gate(self, impl_id: str, **kw: Any):
        """Run the regression gate for `impl_id` and record the result (see agents.gates.gate)."""
        from fracture_md.agents.gates import gate

        return gate(self, impl_id, **kw)

    def energy_model(self, impl_id: str, *, require_trusted: bool = True):
        """Wrap a registered ``potential_energy_fn`` implementation into an EnergyModel.

        The model reuses the *base potential's* parameters, element ordering,
        cutoff and domain assessment (the implementation claims to compute the
        same model, and the gate checks exactly that claim), but gets a new card
        id ``"<base card id>+impl:<impl_id>"`` and the implementation's hash, so
        every PropertyRecord it produces is traceable to the exact source.
        """
        import dataclasses

        from fracture_md.potentials import get_potential

        man = self.get_manifest(impl_id)
        if man["kind"] != "potential_energy_fn":
            raise RegistryError(f"{impl_id} is a {man['kind']}, not a potential_energy_fn")
        if require_trusted:
            why = self.trust_problems(impl_id, man)
            if why:
                raise RegistryError(f"{impl_id} is not trusted: {'; '.join(why)}")
        module = self.load(impl_id)
        base_name = getattr(module, "BASE_POTENTIAL", None) or man.get("base_potential")
        if not base_name:
            raise RegistryError(f"{impl_id} does not declare BASE_POTENTIAL")
        if man.get("base_potential") and man["base_potential"] != base_name:
            raise RegistryError(
                f"{impl_id}: module BASE_POTENTIAL={base_name!r} != manifest base_potential={man['base_potential']!r}"
            )
        if not callable(getattr(module, "energy", None)):
            raise RegistryError(f"{impl_id} does not define energy(frac, box, species, idx, params)")
        base = get_potential(base_name)
        card = base.card.model_copy(
            update=dict(
                id=f"{base.card.id}+impl:{impl_id}",
                name=f"{base.card.name} [implementation {impl_id}]",
                implementation=f"{self.source_path(impl_id)}:energy",
                implementation_sha256=man["sha256"],
                notes=(base.card.notes + " " if base.card.notes else "")
                + f"Agent-proposed implementation {impl_id} (status {man['status']}, author {man['author']}) "
                f"of the same model; parameters, cutoff and domain assessment inherited from {base.card.id}.",
            )
        )
        return dataclasses.replace(base, name=f"{base.name}+impl:{impl_id}", energy_fn=module.energy, card=card)
