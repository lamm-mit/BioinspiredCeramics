"""Provenance collection: code version, package versions, device, hashes."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import inspect
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import jax

from fracture_md.schema.common import Provenance

_TRACKED_PACKAGES = (
    "fracture-md",
    "jax",
    "jaxlib",
    "jax-md",
    "numpy",
    "scipy",
    "ase",
    "pymatgen",
    "spglib",
    "pydantic",
    "matscipy",
    "mace-jax",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def git_version() -> tuple[str, bool | None]:
    """(git describe, dirty).  Dirty includes *untracked* files under src/ and
    tests/ — `git describe --dirty` alone ignores them, which would let a record
    claim a clean commit that does not contain the code that produced it."""
    try:
        desc = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=12"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if desc.returncode != 0:
            return "unknown", None
        s = desc.stdout.strip()
        st = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", "src", "tests", "pyproject.toml"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        dirty = s.endswith("-dirty") or bool(st.stdout.strip())
        if dirty and not s.endswith("-dirty"):
            s += "-dirty"
        return s, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def source_tree_sha256() -> str:
    """Content hash of every .py file in the installed fracture_md package.

    Identifies the exact code even before it is committed."""
    pkg = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(pkg.rglob("*.py")):
        h.update(str(f.relative_to(pkg)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    out = {}
    for p in _TRACKED_PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


def device_string() -> str:
    try:
        d = jax.devices()[0]
        return f"{d.platform}:{getattr(d, 'device_kind', '')}".strip(":")
    except RuntimeError:
        return "unknown"


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def source_sha256(obj: Any) -> str:
    """Content hash of the source code of a function/class/module."""
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()


def collect(**fields: Any) -> Provenance:
    code, dirty = git_version()
    if code == "unknown" and os.environ.get("FRACTURE_MD_CODE_VERSION"):
        # Containers may not see the worktree's .git; the host passes these in.
        code = os.environ["FRACTURE_MD_CODE_VERSION"]
        dirty = code.endswith("-dirty")
    x64 = bool(jax.config.read("jax_enable_x64"))
    base = dict(
        code_version=code,
        code_dirty=dirty,
        implementation_ids={"fracture_md_source_tree": source_tree_sha256()},
        package_versions=package_versions(),
        platform=f"{platform.system()}-{platform.machine()}-py{platform.python_version()}",
        device=device_string(),
        float_precision="float64" if x64 else "float32",
    )
    extra_ids = fields.pop("implementation_ids", None) or {}
    base.update(fields)
    base["implementation_ids"] = {**base["implementation_ids"], **extra_ids}
    return Provenance(**base)
