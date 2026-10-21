"""Energy models for the JAX path (all pure, differentiable JAX functions)."""

from fracture_md.potentials.base import EnergyModel, params_sha256
from fracture_md.potentials.morse import morse_test_fixture
from fracture_md.potentials.neighbors import (
    ALL_PAIRS_MAX_N,
    CellListError,
    MinimumImageError,
    NeighborHandle,
    all_pairs_idx,
    build_neighbors,
    check_minimum_image,
    missing_pairs,
    neighbor_update_fn,
    occupancy,
    overflowed,
    resolve_method,
)
from fracture_md.potentials.tersoff import erhart_albe_2005_sic, erhart_albe_2005_sic_matscipy_ij, tersoff_1989_sic
from fracture_md.potentials.vashishta import (
    ThreeBodyStatus,
    energy_and_status_fn,
    vashishta_2007_sic,
    with_three_body_slots,
)


def _mace_mp0_small():
    # Lazy: mace-jax and the converted weights are optional (docs/GPU_AND_MACE.md).
    from fracture_md.potentials.mace_jax import load_mace_mp0

    return load_mace_mp0("small")


REGISTRY = {
    "tersoff_1989_SiC": tersoff_1989_sic,
    "erhart_albe_2005_SiC": erhart_albe_2005_sic,
    # legacy (pre-fix) i-j three-body indexing; listed after the corrected model so that a bare card-id prefix
    # "erhart-albe-2005-SiC" resolves to the corrected one while the old full id ...@504bf563e0 resolves here
    "erhart_albe_2005_SiC_matscipy_ij": erhart_albe_2005_sic_matscipy_ij,
    "vashishta_2007_SiC": vashishta_2007_sic,
    "morse_test_fixture": morse_test_fixture,
    "mace_mp0_small_jax": _mace_mp0_small,
}


def get_potential(name: str) -> EnergyModel:
    """Look up a potential by registry name or by card id prefix."""
    if name in REGISTRY:
        return REGISTRY[name]()
    for factory in REGISTRY.values():
        try:
            m = factory()
        except ImportError:  # optional potential not installed
            continue
        if m.card.id == name or m.card.id.split("@")[0] == name:
            return m
    raise KeyError(f"unknown potential {name!r}; available: {sorted(REGISTRY)}")


__all__ = [
    "EnergyModel",
    "params_sha256",
    "morse_test_fixture",
    "tersoff_1989_sic",
    "erhart_albe_2005_sic",
    "erhart_albe_2005_sic_matscipy_ij",
    "vashishta_2007_sic",
    "ThreeBodyStatus",
    "energy_and_status_fn",
    "with_three_body_slots",
    "build_neighbors",
    "all_pairs_idx",
    "check_minimum_image",
    "MinimumImageError",
    "NeighborHandle",
    "ALL_PAIRS_MAX_N",
    "CellListError",
    "missing_pairs",
    "neighbor_update_fn",
    "occupancy",
    "overflowed",
    "resolve_method",
    "REGISTRY",
    "get_potential",
]
