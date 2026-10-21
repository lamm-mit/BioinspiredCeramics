"""jax_pfczm: 2D plane-strain phase-field fracture (AT2 and Wu's PF-CZM) on a structured Q4 grid, in JAX.

Importing this package enables float64 in JAX (``jax_enable_x64``).  The solver's tolerances
(PCG 1e-9 / 1e-10, Newton 1e-9, energy 1e-7) are not reachable in float32.
See README.md for equations, conventions, the API table and verification results.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

from .bcs import bar_bc, crack_path, crack_tip_x, k_field, sent_bc, surfing_bc  # noqa: E402
from .grid import Grid, bmatrix, shape_gradients  # noqa: E402
from .microstructure import (  # noqa: E402
    PhaseProps,
    build_notched_disk_specimen,
    notch_void_mask,
    random_disks,
    rasterize_microstructure,
    run_notched_microstructure,
)
from .models import (  # noqa: E402
    Driving,
    Material,
    ModelKind,
    PhaseFieldModel,
    Softening,
    check_pfczm_length,
    discrete_toughness_factor,
    element_parameters,
    softening_parameters,
)
from .plotting import plot_damage, save_damage_gif  # noqa: E402
from .solver import PhaseFieldSolver  # noqa: E402
from .verification import run_verification_suite, verify_at2_bar, verify_pfczm_bar, verify_surfing  # noqa: E402

__version__ = "0.1.0"


def enable_compilation_cache(path: str) -> None:
    """Opt-in persistent XLA compilation cache (saves ~seconds of jit time per new grid/model in later processes)."""
    import os

    os.makedirs(path, exist_ok=True)
    _jax.config.update("jax_compilation_cache_dir", path)
    _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.05)
    _jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


__all__ = [
    "Grid", "shape_gradients", "bmatrix",
    "Material", "PhaseFieldModel", "ModelKind", "Softening", "Driving", "softening_parameters",
    "element_parameters", "check_pfczm_length", "discrete_toughness_factor",
    "PhaseFieldSolver",
    "bar_bc", "sent_bc", "k_field", "surfing_bc", "crack_tip_x", "crack_path",
    "verify_at2_bar", "verify_pfczm_bar", "verify_surfing", "run_verification_suite",
    "PhaseProps", "random_disks", "rasterize_microstructure", "notch_void_mask", "build_notched_disk_specimen",
    "run_notched_microstructure",
    "plot_damage", "save_damage_gif",
    "enable_compilation_cache",
]
