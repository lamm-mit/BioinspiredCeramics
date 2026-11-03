"""Optional adapter: pretrained MACE-MP-0 foundation model evaluated with MACE-JAX.

This module turns a Torch->JAX converted MACE-MP-0 model (ACEsuit/mace-jax,
Flax NNX) into a fracture_md :class:`EnergyModel`, i.e. a *pure* JAX function

    energy(frac, box, species, idx, params) -> scalar eV

that obeys the harness conventions (see potentials/base.py):

* Edge vectors are built with :func:`fracture_md.potentials.tersoff.neighbor_displacements`
  (JAX-MD ``periodic_general(fractional_coordinates=True)``), so
  ``jax.grad(E, 0)(frac)`` is the real-space gradient dE/dR and
  ``jax.grad(E, 1)(box)`` is the exact affine (strain) derivative.  The MACE
  graph is fed with ``positions = 0`` and ``shifts = edge vector``: MACE forms
  ``vectors = positions[receiver] - positions[sender] + shifts``, so the edge
  vector it sees is exactly our JAX-MD displacement and every derivative flows
  through JAX-MD's custom JVP.  MACE-JAX's own force/stress code is not used.
* Graph: every non-padding entry ``idx[i, m] = j`` of the Dense neighbour list
  becomes a directed edge ``sender j -> receiver i`` with vector ``R_i - R_j``
  (MACE convention; Dense lists contain both directions).  Padding entries
  (``idx == N``) become self-edges with a constant vector of length
  ``3 r_max``; MACE's polynomial envelope is exactly zero (value and gradient)
  for ``r >= r_max``, so they contribute nothing.  Neighbour-list entries with
  ``r_max < r < r_max + skin`` are likewise exactly zero.  MACE normalises
  messages by the constant ``avg_num_neighbors`` (not by the per-atom edge
  count), so padding does not change the normalisation.
* Species: ``model.elements`` is the full MACE atomic-number table (89
  elements for MACE-MP-0), so ``species`` indices are z-table indices and the
  node attributes are one-hot vectors over that table.  The total energy is the
  sum of node energies *including* the isolated-atom reference energies E0
  (MPtrj-fitted, PBE/PBE+U); cohesive energies therefore need isolated-atom
  energies of the same model.
* ``params`` is the pytree of *floating-point* leaves of the Flax NNX state
  (nested dict).  Non-float state (integer layout flags, atomic-number table,
  normalisation config) is closed over, so ``jax.grad`` w.r.t. ``params``
  works.  Weights are converted from the float64 Torch checkpoint and kept in
  float64.  Caveat: mace-jax wraps ``scale_shift.scale/shift`` and the e3nn
  activation-normalisation constants in ``stop_gradient`` (Torch buffers), so
  their parameter derivatives are exactly zero; weight leaves are fully
  differentiable (checked against finite differences in the tests).

Minimum image: MACE-MP-0 has r_max = 6 Å and two message-passing layers, but
each layer only uses edges with r < r_max, so the minimum-image requirement is
the usual ``width > 2 (r_max + skin)`` (JAX-MD computes one image per pair).
3C-SiC needs a 3x3x3 conventional cell (216 atoms, width 13.1 Å) with skin
<= 0.5 Å.

Everything here imports ``mace_jax`` / ``flax`` lazily; the core harness never
requires them.  Use ``scripts/setup_mace_env.sh`` to create ``.venv-mace`` and
convert the weights, then::

    from fracture_md.potentials.mace_jax import load_mace_mp0
    model = load_mace_mp0("small")

CLI (inside .venv-mace)::

    python -m fracture_md.potentials.mace_jax convert --model small --out weights/
    python -m fracture_md.potentials.mace_jax info    --model small
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import importlib.metadata as _md
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from fracture_md.schema.common import Citation, ValidationEvidence
from fracture_md.schema.potential import (
    DomainAssessment,
    PotentialCard,
    PotentialKind,
    PropertyClass,
    TrainingDomain,
)
from fracture_md.units import Unit, q

# ----------------------------------------------------------------------------
# Constants (verified 2026-09-28; see docs/GPU_AND_MACE.md)
# ----------------------------------------------------------------------------
#: Torch checkpoints of the MACE-MP-0 ("0a") release, as used by mace-torch 0.3.16.
MACE_MP0_URLS = {
    "small": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model",
    "medium": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
    "large": "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/MACE_MPtrj_2022.9.model",
}
#: sha256 of Torch checkpoints downloaded and hashed by us (others are hashed at conversion time).
KNOWN_TORCH_SHA256 = {
    "small": "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736",
}
#: Weights license: the ACEsuit/mace-mp and ACEsuit/mace-foundations model tables list
#: MACE-MP-0a (small/medium/large) as MIT (checked 2026-09-28).  Later foundation models
#: (e.g. MPA-0, OMAT-0) carry different licenses - do not extend this blindly.
MACE_MP0_WEIGHTS_LICENSE = (
    "MIT (MACE-MP-0a weights; per the model table of github.com/ACEsuit/mace-mp and "
    "github.com/ACEsuit/mace-foundations, checked 2026-09-28)"
)
MACE_JAX_REPO = "https://github.com/ACEsuit/mace-jax"
PARAMS_FILE = "params.msgpack"
CONFIG_FILE = "params.json"
PROVENANCE_FILE = "provenance.json"


class MaceJaxUnavailable(ImportError):
    """Raised when mace-jax (or flax / converted weights) is not available."""


_INSTALL_HINT = (
    "The MACE-JAX path is optional.  Create its environment with "
    "`bash scripts/setup_mace_env.sh` (separate venv .venv-mace: mace-jax pinned to a git "
    "commit + torch/mace-torch for the Torch->JAX conversion), then run with "
    ".venv-mace/bin/python.  See docs/GPU_AND_MACE.md."
)


def _require_mace_jax():
    try:
        import flax  # noqa: F401
        from flax import nnx  # noqa: F401
        from mace_jax.tools import bundle as _bundle  # noqa: F401
    except Exception as exc:  # ImportError, or a broken transitive import
        raise MaceJaxUnavailable(f"mace-jax is not importable ({type(exc).__name__}: {exc}). {_INSTALL_HINT}") from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def weights_root() -> Path:
    """Directory holding converted weights ($FRACTURE_MD_WEIGHTS or <repo>/weights, git-ignored)."""
    env = os.environ.get("FRACTURE_MD_WEIGHTS")
    return Path(env).expanduser() if env else _repo_root() / "weights"


def weights_dir_for(size: str, root: Optional[Path] = None) -> Path:
    return (Path(root) if root is not None else weights_root()) / f"mace-mp-0-{size}-jax"


def availability(size: str = "small", weights_dir: Optional[Path] = None) -> tuple[bool, str]:
    """(available, reason).  Never raises; used by tests/examples to skip cleanly."""
    try:
        _require_mace_jax()
    except MaceJaxUnavailable as exc:
        return False, str(exc)
    wd = Path(weights_dir) if weights_dir is not None else weights_dir_for(size)
    missing = [f for f in (PARAMS_FILE, CONFIG_FILE) if not (wd / f).exists()]
    if missing:
        return False, (
            f"converted MACE-MP-0 '{size}' weights not found in {wd} (missing {missing}). "
            f"Run `.venv-mace/bin/python -m fracture_md.potentials.mace_jax convert --model {size}`."
        )
    return True, "ok"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions(pkgs=("jax", "jaxlib", "flax", "e3nn-jax", "cuequivariance-jax", "mace-jax", "torch", "mace-torch", "e3nn")):
    out = {}
    for p in pkgs:
        try:
            out[p] = _md.version(p)
        except _md.PackageNotFoundError:
            pass
    return out


def _mace_jax_commit() -> Optional[str]:
    """Git commit of the installed mace-jax (from PEP 610 direct_url.json), if recorded."""
    try:
        dist = _md.distribution("mace-jax")
        txt = dist.read_text("direct_url.json")
        if txt:
            return json.loads(txt).get("vcs_info", {}).get("commit_id")
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# Torch -> JAX conversion (needs torch + mace-torch + mace-jax)
# ----------------------------------------------------------------------------
def convert_foundation(size: str = "small", out_root: Optional[Path] = None, overwrite: bool = False) -> Path:
    """Download the MACE-MP-0 Torch checkpoint and convert it to a MACE-JAX bundle.

    The Torch model is loaded in **float64** before conversion (the upstream CLI
    ``mace-jax-from-torch`` loads it with mace-torch's default float32, which
    rounds weights and the E0 / scale-shift constants to float32 and costs
    ~5e-7 eV/atom agreement with the float64 Torch reference).

    Writes ``params.msgpack`` (Flax serialisation), ``params.json`` (model
    config) and ``provenance.json`` (source URL, Torch sha256, package
    versions) into ``<out_root>/mace-mp-0-<size>-jax/``.
    """
    if size not in MACE_MP0_URLS:
        raise ValueError(f"size must be one of {sorted(MACE_MP0_URLS)} (MACE-MP-0a only)")
    _require_mace_jax()
    try:
        import torch  # noqa: F401
        from mace.calculators.foundations_models import download_mace_mp_checkpoint
        from mace.tools.scripts_utils import extract_config_mace_model
    except Exception as exc:
        raise MaceJaxUnavailable(f"conversion needs torch + mace-torch ({exc}). {_INSTALL_HINT}") from exc
    from flax import serialization
    from mace_jax.cli.mace_jax_from_torch import _serialize_for_json, convert_model
    from mace_jax.nnx_utils import state_to_serializable_dict
    from mace_jax.tools.foundation_models import load_foundation_torch_model

    out = weights_dir_for(size, out_root)
    if (out / PARAMS_FILE).exists() and not overwrite:
        return out
    out.mkdir(parents=True, exist_ok=True)

    ckpt = Path(download_mace_mp_checkpoint(size))
    torch_sha = _sha256_file(ckpt)
    if size in KNOWN_TORCH_SHA256 and torch_sha != KNOWN_TORCH_SHA256[size]:
        raise RuntimeError(f"sha256 of {ckpt} = {torch_sha} != recorded {KNOWN_TORCH_SHA256[size]}")

    torch_model = load_foundation_torch_model(source="mp", model=size, device="cpu", default_dtype="float64")
    torch_model.eval()
    config = extract_config_mace_model(torch_model)
    if "error" in config:
        raise RuntimeError(config["error"])
    config["torch_model_class"] = torch_model.__class__.__name__
    # e3nn normalize2mom activation constants as held by the float64 Torch model.  The
    # upstream converter round-trips them through float32 (relative error ~1e-8, i.e.
    # ~1e-5 eV on a 216-atom cell); we restore the Torch values in config and state.
    csts = _torch_normalize2mom_consts(torch_model)
    if csts:
        config["normalize2mom_consts"] = dict(csts)
    _, state, _ = convert_model(torch_model, config)
    variables = state_to_serializable_dict(state)
    if csts:
        config["normalize2mom_consts"] = dict(csts)
        variables = _patch_act_consts(variables, csts)
    (out / PARAMS_FILE).write_bytes(serialization.to_bytes(variables))
    (out / CONFIG_FILE).write_text(json.dumps(_serialize_for_json(config), indent=2))
    prov = {
        "model": f"MACE-MP-0 {size}",
        "source_url": MACE_MP0_URLS[size],
        "torch_checkpoint": str(ckpt),
        "torch_checkpoint_sha256": torch_sha,
        "converted_params_sha256": _sha256_file(out / PARAMS_FILE),
        "conversion_dtype": "float64",
        "converted_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "mace_jax_repo": MACE_JAX_REPO,
        "mace_jax_commit": _mace_jax_commit(),
        "package_versions": _versions(),
        "weights_license": MACE_MP0_WEIGHTS_LICENSE,
        "torch_normalize2mom_consts": csts,
    }
    (out / PROVENANCE_FILE).write_text(json.dumps(prov, indent=2))
    return out


def _torch_normalize2mom_consts(torch_model) -> dict[str, float]:
    """{'silu': cst, 'swish': cst} from e3nn normalize2mom modules of a Torch MACE model."""
    vals = []
    for _, mod in torch_model.named_modules():
        if hasattr(mod, "cst") and hasattr(mod, "f"):
            vals.append(float(mod.cst))
    if not vals:
        return {}
    if max(vals) - min(vals) > 1e-12:
        # several distinct activations; leave upstream behaviour untouched
        return {}
    return {"silu": vals[0], "swish": vals[0]}


def _patch_act_consts(tree, csts: dict[str, float]):
    """Replace float32-rounded activation constants in a serialisable state dict."""
    v = np.float64(csts["silu"])
    if isinstance(tree, dict):
        out = {}
        for k, val in tree.items():
            if k in ("act_scale", "_act_scale_init") and np.ndim(val) == 0 and not isinstance(val, dict):
                out[k] = np.asarray(v, dtype=np.float64)
            elif k == "_normalize2mom_consts_var" and isinstance(val, dict):
                out[k] = {kk: np.asarray(csts.get(kk, csts["silu"]), dtype=np.float64) for kk in val}
            else:
                out[k] = _patch_act_consts(val, csts)
        return out
    return tree


# ----------------------------------------------------------------------------
# Energy function
# ----------------------------------------------------------------------------
def _split_state(pure: dict) -> tuple[dict, dict]:
    """Split a Flax pure-dict state into (float leaves, everything else), both nested dicts."""
    from flax import traverse_util

    flat = traverse_util.flatten_dict(pure)
    fl, st = {}, {}
    for k, v in flat.items():
        a = np.asarray(v) if not hasattr(v, "keys") else None
        if a is not None and a.dtype.kind == "f":
            fl[k] = v
        else:
            st[k] = v
    return traverse_util.unflatten_dict(fl), traverse_util.unflatten_dict(st)


def _merge_state(float_part: dict, static_part: dict) -> dict:
    from flax import traverse_util

    flat = dict(traverse_util.flatten_dict(static_part))
    flat.update(traverse_util.flatten_dict(float_part))
    return traverse_util.unflatten_dict(flat)


def make_mace_energy_fn(graphdef, static_state: dict, num_species: int, r_max: float):
    """Build ``energy(frac, box, species, idx, params)`` for a MACE-JAX NNX graphdef.

    ``species`` are indices into the model's atomic-number table; ``params`` the
    float-leaf pytree returned by :func:`load_mace_mp0` (``model.params``).
    """
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from fracture_md.potentials.tersoff import neighbor_displacements

    pad_len = 3.0 * float(r_max)

    def energy_fn(frac, box, species, idx, params):
        n, m = idx.shape
        d, mask, j = neighbor_displacements(frac, box, idx)  # d[i, m] = R_j - R_i
        dtype = d.dtype
        receivers = jnp.repeat(jnp.arange(n, dtype=jnp.int32), m)
        senders = jnp.where(mask, j, jnp.arange(n, dtype=jnp.int32)[:, None]).reshape(-1).astype(jnp.int32)
        pad_vec = jnp.asarray([pad_len, 0.0, 0.0], dtype=dtype)
        vectors = jnp.where(mask[..., None], -d, pad_vec).reshape(-1, 3)  # R_receiver - R_sender
        species = jnp.asarray(species, dtype=jnp.int32)
        data = {
            "positions": jnp.zeros((n, 3), dtype=dtype),
            "shifts": vectors,
            "node_attrs": jax.nn.one_hot(species, num_species, dtype=dtype),
            "node_attrs_index": species,
            "edge_index": jnp.stack([senders, receivers], axis=0),
            "batch": jnp.zeros((n,), dtype=jnp.int32),
            "ptr": jnp.asarray([0, n], dtype=jnp.int32),
            "cell": jnp.zeros((1, 3, 3), dtype=dtype),
        }
        module = nnx.merge(graphdef, _merge_state(params, static_state))
        out = module(data, compute_force=False, compute_stress=False)
        return out["energy"][0]

    return energy_fn


# ----------------------------------------------------------------------------
# Potential card
# ----------------------------------------------------------------------------
_MACE_LIMITATIONS = [
    "Training data (MPtrj: Materials Project GGA/GGA+U relaxation trajectories, PBE/PBE+U) samples "
    "mostly near-equilibrium bulk crystals; free surfaces, cleavage openings, crack tips and "
    "bond rupture are outside the training distribution.",
    "Systematic PES softening reported for universal MLIPs including MACE-MP-0: energies, forces and "
    "PES curvature (hence elastic constants, phonons, surface energies, migration barriers) tend to "
    "be underpredicted (B. Deng et al., npj Comput. Mater. 11, 9 (2025), doi:10.1038/s41524-024-01500-6).",
    "Surface energies of universal MLIPs (MACE-MP-0, CHGNet, M3GNet) show significant errors correlated "
    "with the distance from the bulk-dominated training set (B. Focassio, L. P. M. Freitas, "
    "G. R. Schleder, ACS Appl. Mater. Interfaces 17, 13111 (2025), doi:10.1021/acsami.4c03815).",
    "Reference level of theory is PBE(+U): lattice constants are typically overestimated and "
    "elastic moduli underestimated relative to experiment, independent of the ML error.",
    "Finite 6 Å receptive field per layer (12 Å for 2 layers); no long-range electrostatics.  The "
    "rigid-opening traction vanishes for openings beyond ~r_max by construction.",
    "MACE-MP-0a (2023) predates the 0b/0b2/0b3 releases that add core repulsion and fix high-pressure "
    "and phonon issues; short interatomic distances (< ~1 Å) are not reliable.",
]


def mace_mp0_card(size: str, params_sha256: str, prov: dict, config: dict, precision: str) -> PotentialCard:
    zs = [int(z) for z in _as_list(config.get("atomic_numbers", []))]
    from ase.data import chemical_symbols

    elements = [chemical_symbols[z] for z in zs]
    r_max = float(config["r_max"])
    lit_bulk = ValidationEvidence(
        kind="literature_model_value",
        description="Foundation-model paper benchmarks (bulk structures, phonons, MD) for MACE-MP-0; "
        "not specific to SiC.",
        citation="I. Batatia et al., 'A foundation model for atomistic materials chemistry', "
        "J. Chem. Phys. 163, 184110 (2025), doi:10.1063/5.0297006 (arXiv:2401.00096)",
    )
    soft = ValidationEvidence(
        kind="literature_model_value",
        description="Systematic softening (underpredicted PES curvature) in MACE-MP-0/CHGNet/M3GNet.",
        citation="B. Deng et al., npj Comput. Mater. 11, 9 (2025), doi:10.1038/s41524-024-01500-6",
        passed=False,
    )
    return PotentialCard(
        id=f"mace-mp-0-{size}-jax@{params_sha256[:10]}",
        name=f"MACE-MP-0 {size} (MACE-JAX port of the Torch checkpoint)",
        kind=PotentialKind.MACHINE_LEARNED,
        functional_form=(
            f"MACE equivariant message passing ({config.get('torch_model_class', 'ScaleShiftMACE')}, "
            f"{config.get('num_interactions')} interactions, hidden irreps {config.get('hidden_irreps')}, "
            f"max_ell {config.get('max_ell')}, correlation {config.get('correlation')}, r_max {r_max} Å); "
            "E = sum_i [E0(Z_i) + scale * e_i + shift]"
        ),
        elements=elements,
        cutoff=q(r_max, Unit.ANGSTROM),
        parameters_sha256=params_sha256,
        parameter_source=(
            f"Torch checkpoint {prov.get('source_url', MACE_MP0_URLS.get(size, '?'))} "
            f"(sha256 {prov.get('torch_checkpoint_sha256', 'unknown')}), converted to MACE-JAX "
            f"({MACE_JAX_REPO} @ {prov.get('mace_jax_commit') or 'unknown commit'}) in "
            f"{prov.get('conversion_dtype', 'unknown')} precision"
        ),
        license=MACE_MP0_WEIGHTS_LICENSE,
        code_license="mace-jax: MIT; mace-torch: MIT; fracture_md adapter: MIT",
        implementation="fracture_md.potentials.mace_jax.make_mace_energy_fn",
        citations=[
            Citation(
                key="Batatia2025MACEMP0",
                text="I. Batatia et al., A foundation model for atomistic materials chemistry, J. Chem. Phys. 163, 184110 (2025)",
                doi="10.1063/5.0297006",
                url="https://arxiv.org/abs/2401.00096",
            ),
            Citation(
                key="Batatia2022MACE",
                text="I. Batatia, D. P. Kovacs, G. N. C. Simm, C. Ortner, G. Csanyi, MACE: Higher order equivariant "
                "message passing neural networks for fast and accurate force fields, NeurIPS 35 (2022)",
                url="https://arxiv.org/abs/2206.07697",
            ),
            Citation(
                key="Deng2023CHGNet",
                text="B. Deng et al., CHGNet as a pretrained universal neural network potential for charge-informed "
                "atomistic modelling, Nat. Mach. Intell. 5, 1031 (2023) (MPtrj dataset)",
                doi="10.1038/s42256-023-00716-3",
            ),
            Citation(
                key="Deng2025Softening",
                text="B. Deng et al., Systematic softening in universal machine learning interatomic potentials, "
                "npj Comput. Mater. 11, 9 (2025)",
                doi="10.1038/s41524-024-01500-6",
            ),
            Citation(
                key="Focassio2025Surfaces",
                text="B. Focassio, L. P. M. Freitas, G. R. Schleder, Performance assessment of universal machine "
                "learning interatomic potentials: challenges and directions for materials' surfaces, "
                "ACS Appl. Mater. Interfaces 17, 13111 (2025)",
                doi="10.1021/acsami.4c03815",
            ),
        ],
        training_domain=TrainingDomain(
            elements=elements,
            description=(
                "MPtrj: ~1.58M structures from ~146k Materials Project GGA/GGA+U relaxation trajectories "
                "(energies, forces, stresses, magnetic moments).  Dominated by near-equilibrium bulk "
                "crystals; no explicit cleavage, surface-separation or fracture data."
            ),
            data_sources=["MPtrj (Materials Project relaxation trajectories, PBE/PBE+U)"],
            reference_method="DFT-PBE / PBE+U (Materials Project settings)",
            includes_surfaces=False,
            includes_large_strain=False,
            includes_bond_breaking=False,
        ),
        domain_assessment=[
            DomainAssessment(
                property_class=PropertyClass.EQUILIBRIUM_BULK,
                status="partially_validated",
                evidence=[lit_bulk],
                notes="Within the training distribution; accuracy relative to PBE, not experiment.",
            ),
            DomainAssessment(
                property_class=PropertyClass.ELASTIC,
                status="unvalidated",
                evidence=[soft],
                notes="PES-curvature underprediction (softening) reported; compare with PBE C_ij before use.",
            ),
            DomainAssessment(property_class=PropertyClass.SURFACES, status="unvalidated",
                             notes="Out of training distribution; literature reports significant surface-energy errors."),
            DomainAssessment(property_class=PropertyClass.BOND_BREAKING, status="unvalidated",
                             notes="No bond-rupture data in MPtrj; no evidence for cleavage tractions."),
            DomainAssessment(property_class=PropertyClass.LARGE_STRAIN, status="unvalidated"),
            DomainAssessment(property_class=PropertyClass.STACKING_FAULTS, status="unvalidated"),
            DomainAssessment(property_class=PropertyClass.DEFECTS, status="unvalidated"),
            DomainAssessment(property_class=PropertyClass.FINITE_TEMPERATURE, status="unvalidated"),
            # 3C-SiC evidence from the MD-indentation de-risking study (outputs/indent_derisk/SYNTHESIS.md,
            # potential-diagnostics; reproduced by a verify agent)
            DomainAssessment(
                property_class=PropertyClass.HIGH_PRESSURE,
                status="unvalidated",
                notes="3C-SiC: smooth P(V) to 211 GPa (no cutoff wall), but B' = 2.09, M(001) = 292.5 GPa (-31 % vs "
                "experiment), tetragonal Born instability of B3 at 87 GPa; unphysical collapse at extreme "
                "compression (B3 pressure maximum 240.7 GPa at d(Si-C) = 1.62 A; B1 energy diverges downwards "
                "below d(Si-C) = 1.80 A).",
            ),
            DomainAssessment(
                property_class=PropertyClass.PLASTICITY,
                status="unvalidated",
                notes="3C-SiC: semi-quantitative ideal shear ({111}<1-10> 28.0 vs DFT-PBE 30.6 GPa; "
                "twin 37.0 vs DFT-PBE 27.76 GPa, +33 %; "
                "-41 % at 50 GPa).  At most ~3,200 atoms fit on the GB10 (2.4-2.7e3 atom-steps/s): use only to "
                "re-evaluate <= 3k-atom snapshots cut from indentation runs, never for indentation MD.",
            ),
        ],
        known_limitations=list(_MACE_LIMITATIONS),
        differentiable=True,
        precision=precision,  # type: ignore[arg-type]
        notes=(
            "Energies include MPtrj-fitted isolated-atom E0s.  Weights license verified from the "
            "official model table; MPtrj data license not asserted here."
        ),
    )


def _as_list(v):
    if isinstance(v, str):
        return json.loads(v)
    return list(v)


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class MaceBundleInfo:
    weights_dir: Path
    config: dict
    provenance: dict
    atomic_numbers: tuple[int, ...]
    r_max: float


def load_mace_mp0(
    size: str = "small",
    weights_dir: Optional[Path] = None,
    dtype: str = "float64",
):
    """Load converted MACE-MP-0 weights as an :class:`EnergyModel`.

    Raises :class:`MaceJaxUnavailable` (an ImportError) if mace-jax/flax are not
    importable or the converted weights are missing.
    """
    _require_mace_jax()
    import jax
    import jax.numpy as jnp
    from ase.data import chemical_symbols
    from mace_jax.tools import bundle as bundle_tools

    from fracture_md.potentials.base import EnergyModel

    wd = Path(weights_dir) if weights_dir is not None else weights_dir_for(size)
    ok, why = availability(size, wd)
    if not ok:
        raise MaceJaxUnavailable(why)
    if dtype not in ("float64", "float32"):
        raise ValueError("dtype must be float64 or float32")
    if dtype == "float64" and not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)

    # load_model_bundle(dtype) toggles jax_enable_x64 globally; pass "" and keep the harness setting.
    b = bundle_tools.load_model_bundle(str(wd / PARAMS_FILE), "")
    config = b.config
    zs = tuple(int(z) for z in _as_list(config["atomic_numbers"]))
    r_max = float(config["r_max"])
    prov_path = wd / PROVENANCE_FILE
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    fdt = jnp.float64 if dtype == "float64" else jnp.float32

    float_part, static_part = _split_state(b.params)
    float_part = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=fdt), float_part)
    params_sha = prov.get("converted_params_sha256") or _sha256_file(wd / PARAMS_FILE)
    src_dtype = prov.get("conversion_dtype", "unknown")
    precision = dtype if src_dtype in ("float64", dtype) else "mixed"
    card = mace_mp0_card(size, params_sha, prov, config, precision)
    energy_fn = make_mace_energy_fn(b.graphdef, static_part, len(zs), r_max)
    model = EnergyModel(
        name=f"mace_mp0_{size}_jax",
        elements=tuple(chemical_symbols[z] for z in zs),
        cutoff=r_max,
        params=float_part,
        energy_fn=energy_fn,
        card=card,
    )
    info = MaceBundleInfo(weights_dir=wd, config=config, provenance=prov, atomic_numbers=zs, r_max=r_max)
    object.__setattr__(model, "_mace_info", info)  # frozen dataclass: attach metadata without a schema change
    return model


def bundle_info(model) -> MaceBundleInfo:
    return getattr(model, "_mace_info")


def torch_reference_calculator(size: str = "small", dtype: str = "float64"):
    """mace-torch ASE calculator for the same checkpoint (reference implementation)."""
    try:
        from mace.calculators import mace_mp
    except Exception as exc:
        raise MaceJaxUnavailable(f"mace-torch not importable ({exc}). {_INSTALL_HINT}") from exc
    return mace_mp(model=size, default_dtype=dtype, device="cpu")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fracture_md.potentials.mace_jax")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="download + convert MACE-MP-0 (float64) to a MACE-JAX bundle")
    c.add_argument("--model", default="small", choices=sorted(MACE_MP0_URLS))
    c.add_argument("--out", default=None, help="weights root (default: $FRACTURE_MD_WEIGHTS or <repo>/weights)")
    c.add_argument("--overwrite", action="store_true")
    i = sub.add_parser("info", help="print availability and the potential card")
    i.add_argument("--model", default="small", choices=sorted(MACE_MP0_URLS))
    args = ap.parse_args(argv)
    if args.cmd == "convert":
        out = convert_foundation(args.model, Path(args.out) if args.out else None, overwrite=args.overwrite)
        print(f"converted MACE-MP-0 {args.model} -> {out}")
        print((out / PROVENANCE_FILE).read_text())
        return 0
    ok, why = availability(args.model)
    print(f"available: {ok} ({why})")
    if ok:
        m = load_mace_mp0(args.model)
        print(m.card.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
