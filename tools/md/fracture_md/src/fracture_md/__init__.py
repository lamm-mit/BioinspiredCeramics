"""fracture_md: JAX-first atomistic harness for ceramic strength and fracture.

Importing this package enables float64 in JAX.  Elastic constants, finite
difference checks and traction-separation integrals are not reliable in
float32 (see docs/AUTODIFF.md), so float64 is the default everywhere; code
that deliberately runs in float32 must record it in Provenance.float_precision.
"""

import os as _os

import jax as _jax

_jax.config.update("jax_enable_x64", True)


def _enable_compilation_cache() -> None:
    """Persist XLA compilations across processes (big win for tests and CLIs).

    Directory: $FRACTURE_MD_JAX_CACHE, default ~/.cache/fracture_md/jax.
    Set FRACTURE_MD_JAX_CACHE=off to disable.  Cache keys include the backend,
    JAX version and compile options, so CPU and GPU entries never collide.
    """
    path = _os.environ.get("FRACTURE_MD_JAX_CACHE", _os.path.join(_os.path.expanduser("~"), ".cache", "fracture_md", "jax"))
    if path.strip().lower() in ("", "0", "off", "none", "false"):
        return
    try:
        _os.makedirs(path, exist_ok=True)
        _jax.config.update("jax_compilation_cache_dir", path)
        _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.05)
        _jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    except (OSError, AttributeError, ValueError):  # read-only home, old JAX...
        pass


_enable_compilation_cache()

__version__ = "0.1.0"
