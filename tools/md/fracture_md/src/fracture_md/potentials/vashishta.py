"""Vashishta-Kalia-Nakano-Rino (2007) Si-C potential in pure JAX.

Reference: P. Vashishta, R. K. Kalia, A. Nakano, J. P. Rino, "Interaction potential for
silicon carbide: A molecular dynamics study of elastic constants and vibrational density
of states for crystalline and amorphous silicon carbide", J. Appl. Phys. 101, 103515 (2007),
doi:10.1063/1.2724570.

Functional form and conventions follow LAMMPS ``pair_style vashishta``
(src/MANYBODY/pair_vashishta.cpp, functions ``setup_params``/``twobody``/``threebody``),
the de-facto reference implementation this kernel is verified against (float64, 15
configurations incl. antisites, 2500/4500 K snapshots, -8 % compression, rock salt and a
(001) slab: |ΔE|/atom <= 4e-13 eV, max |ΔF| <= 1.3e-11 eV/Å; tests/test_vashishta.py):

    E  = Σ_{i<j} V2s_ij(r_ij)  +  Σ_i Σ_{j<k} V3_jik

    V2_ij(r)  = H_ij / r^η_ij + Z_i Z_j e² exp(-r/λ1)/r - D_ij exp(-r/λ4)/r⁴ - W_ij / r⁶
    V2s_ij(r) = V2(r) - V2(rc) - (r - rc) V2'(rc)       for r < rc_ij, else 0
                (shifted force: V and dV/dr vanish at rc; d²V/dr² does NOT)
    V3_jik    = B_jik f_ij f_ik (cosθ_jik - cosθ0)² / (1 + C_jik (cosθ_jik - cosθ0)²)
    f_ij      = exp(γ_ij / (r_ij - r0_ij))                for r_ij < r0_ij, else 0

Parameter-file semantics (LAMMPS "metal" units, one entry per ordered triplet i j k):
    two-body parameters of the pair (i, j) come from entry (i, j, j);
    γ, r0 of the pair (i, j) come from entry (i, j, j), of (i, k) from entry (i, k, k);
    B, C, cosθ0 of the angle j-i-k (vertex i) come from entry (i, j, k).
    Z_i Z_j is multiplied by e²/(4πε0) = qqr2e = 14.399645 eV·Å (LAMMPS metal value).
    D is used as written in the file, whose header states "Value of D here equals D/2 in
    paper" (the paper writes the charge-dipole term as D/(2 r⁴) with D in e² Å³, so
    D_file = D_paper·e²/2; the file's values imply e² = 14.3971 eV·Å, 1.8e-4 below qqr2e,
    which changes E by 0.17 meV/atom and no Cij by more than 0.02 GPa); no further 1/2 here.

Neighbour lists: ONE Dense list (N, M) with padding value N built for the two-body cutoff
rc (7.35 Å) + skin, i.e. the harness convention (``energy_fn(frac, box, species, idx,
params)``).  The three-body sub-list is extracted inside the kernel: for each atom the first
``k3`` list entries with r < r0(ti, tj) are selected (``lax.top_k`` on an integer key, no
gradient through the selection; exact because f_ij -> 0 with all derivatives at r0).  Cost:
two-body N·M, three-body N·k3².

Three-body slot overflow (more than ``k3`` partners inside r0 for some atom) is reported,
not hidden:

* :func:`vashishta_energy_and_status` returns ``(E, ThreeBodyStatus)``; E is finite (the
  first k3 partners only) and must be discarded when ``status.overflow`` is set: a driver
  uses ``jax.value_and_grad(..., has_aux=True)``, ORs the flag over its lax.scan chunk,
  rolls the chunk back and continues with :func:`with_three_body_slots` (a model with more
  slots; same card).  The same function is attached to every energy function made here as
  ``energy_fn.energy_and_status`` (see :func:`energy_and_status_fn`).
* the plain scalar ``energy_fn`` used by flag-unaware code (relaxation, elastic constants,
  ``physics.md.run_md``, the agent gates) returns NaN on overflow, so a truncated energy can
  never pass silently; ``make_vashishta_energy(on_overflow="truncate")`` opts out.

Measured occupancy: perfect 3C 4, rock salt 6, 4500 K liquid 9, liquid at 333 GPa 11, so the
default k3 = 12 is not expected to overflow in indentation.

Minimum image: like every harness kernel this assumes each periodic cell width exceeds
2 (rc + skin) = 15.7 Å for rc = 7.35 Å, skin = 0.5 Å; smaller cells (e.g. the 64-atom
2x2x2 3C cell, 8.7 Å) must be replicated (structures.system.auto_supercell).

Precision: kernels are traced with ``jax_default_matmul_precision = "highest"``
(:func:`fracture_md.potentials.tersoff.highest_matmul_precision`); with the GPU default
(TF32 matmuls) float32 energies were off by 1.7 meV/atom at 131k atoms (5.6e-6 with
"highest").  Production and records use float64.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax_md import space

from fracture_md.potentials.base import EnergyModel, params_sha256
from fracture_md.potentials.tersoff import _safe_norm, highest_matmul_precision
from fracture_md.schema.common import Citation, ValidationEvidence
from fracture_md.schema.potential import (
    DomainAssessment,
    PotentialCard,
    PotentialKind,
    PropertyClass,
    TrainingDomain,
)
from fracture_md.units import Quantity, Unit, q

QQR2E_METAL = 14.399645  # e^2/(4 pi eps0) in eV*Angstrom, LAMMPS force->qqr2e for units metal
K3_DEFAULT = 12  # three-body slots per atom (measured max: 3C 4; B1 6; 4500 K liquid 9; liquid at 333 GPa 11)

# ----------------------------------------------------------------------------
# Parameters: verbatim copy of the LAMMPS-distributed potentials/SiC.vashishta
# (DATE: 2015-10-14, CONTRIBUTOR: Aidan Thompson; sha256 of the distributed file
# d7892279709e8659e99a1696cace10d07c4f9a87804500e1b04af06dd5104d3f; also NIST IPR
# 2007--Vashishta-P-Kalia-R-K-Nakano-A-Rino-J-P--Si-C).  File header: "Value of D here
# equals D/2 in paper".
# G12 provenance check (outputs/impl/side-vashishta-file.md, single side-check agent, LAMMPS):
# every Table I parameter is in this file (D as above); the paper's C44 = 191 GPa and ISF =
# 80 mJ/m^2 are not reproducible from Table I by any definition, and the 0.94 meV/atom E0
# offset is explained by a Z_i Z_j e^2 1.28e-4 larger in the authors' code (no mechanical
# effect).  TODO: independent re-verification of that report before citing it as settled.
# Columns: e1 e2 e3  H eta Zi Zj lambda1 D lambda4  W rc B gamma r0 C costheta
# ----------------------------------------------------------------------------
SIC_VASHISHTA_2007_LAMMPS = """
C   C    C  471.74538  7  -1.201  -1.201  5.0  0.0  3.0
            0.0  7.35  0.0  0.0  0.0  0.0  0.0
Si  Si  Si  23.67291  7  1.201  1.201  5.0  15.575  3.0
            0.0  7.35  0.0  0.0  0.0  0.0  0.0
C   Si  Si  447.09026  9  -1.201  1.201  5.0  7.7874  3.0
            61.4694  7.35  9.003  1.0  2.90  5.0  -0.333333333333
Si  C   C   447.09026  9  1.201  -1.201  5.0  7.7874  3.0
            61.4694  7.35  9.003  1.0  2.90  5.0  -0.333333333333
C   C   Si  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0  0.0  0.0  0.0  0.0  0.0  0.0
C   Si  C   0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0  0.0  0.0  0.0  0.0  0.0  0.0
Si  C   Si  0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0  0.0  0.0  0.0  0.0  0.0  0.0
Si  Si  C   0.0  0.0  0.0  0.0  0.0  0.0  0.0
            0.0  0.0  0.0  0.0  0.0  0.0  0.0
"""
SIC_VASHISHTA_2007_FILE_SHA256 = "d7892279709e8659e99a1696cace10d07c4f9a87804500e1b04af06dd5104d3f"

_FIELDS = ("H", "eta", "Zi", "Zj", "lambda1", "D", "lambda4", "W", "rc", "B", "gamma", "r0", "C", "costheta")


def parse_lammps_vashishta(text: str) -> dict[tuple[str, str, str], dict[str, float]]:
    """Parse a LAMMPS pair_style vashishta file (comments '#', free-format whitespace)."""
    toks: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        toks.extend(line.split())
    entries = {}
    stride = 3 + len(_FIELDS)
    if len(toks) % stride:
        raise ValueError("malformed vashishta file: token count is not a multiple of 17")
    for s in range(0, len(toks), stride):
        el = tuple(toks[s : s + 3])
        entries[el] = {k: float(v) for k, v in zip(_FIELDS, toks[s + 3 : s + stride])}
    return entries


def tables_from_lammps(entries: dict, elements: Sequence[str], qqr2e: float = QQR2E_METAL) -> dict[str, jnp.ndarray]:
    """Dense lookup tables in model element order.

    Pair tables [ne, ne] (from entry (i, j, j)): H, eta, zz (= Zi Zj qqr2e), lam1inv, D,
    lam4inv, W, rc, gamma, r0.  Triplet tables [ne, ne, ne] (entry (i, j, k)): B, C, cos0.
    """
    ne = len(elements)
    for i in elements:
        for j in elements:
            for k in elements:
                if (i, j, k) not in entries:
                    raise ValueError(f"missing vashishta entry {(i, j, k)}")
    P = lambda key: np.array([[entries[(a, b, b)][key] for b in elements] for a in elements], float)  # noqa: E731
    T = lambda key: np.array(  # noqa: E731
        [[[entries[(a, b, c)][key] for c in elements] for b in elements] for a in elements], float
    )
    inv = lambda x: np.where(x == 0.0, 0.0, 1.0 / np.where(x == 0.0, 1.0, x))  # noqa: E731  (LAMMPS: 1/0 -> 0)
    t = dict(
        H=P("H"),
        eta=P("eta"),
        zz=P("Zi") * P("Zj") * qqr2e,
        lam1inv=inv(P("lambda1")),
        D=P("D"),
        lam4inv=inv(P("lambda4")),
        W=P("W"),
        rc=P("rc"),
        gamma=P("gamma"),
        r0=P("r0"),
        B=T("B"),
        C=T("C"),
        cos0=T("costheta"),
    )
    return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in t.items()}


# ----------------------------------------------------------------------------
# Kernel
# ----------------------------------------------------------------------------
class ThreeBodyStatus(NamedTuple):
    """Three-body slot diagnostics of one energy evaluation (a pytree; usable as ``has_aux`` output).

    ``max_partners``: largest number of three-body partners (r < r0) of any atom (int32);
    ``overflow``: max_partners > number of slots, i.e. the returned energy is NOT the model energy.
    """

    max_partners: jnp.ndarray
    overflow: jnp.ndarray


def _v2_raw(r, H, eta, zz, lam1inv, D, lam4inv, W):
    """Unshifted two-body V2(r) and dV2/dr (LAMMPS twobody/setup_params algebra)."""
    rinv = 1.0 / r
    reta = jnp.exp(-eta * jnp.log(r))  # r^-eta
    r4inv = rinv**4
    r6inv = r4inv * rinv * rinv
    vc2 = zz * jnp.exp(-r * lam1inv) * rinv
    vc3 = D * r4inv * jnp.exp(-r * lam4inv)
    v = H * reta + vc2 - vc3 - W * r6inv
    dv = vc3 * (4.0 * rinv + lam4inv) + 6.0 * W * r6inv * rinv - eta * H * reta * rinv - vc2 * (rinv + lam1inv)
    return v, dv


def _v2_shifted(r, rc, H, eta, zz, lam1inv, D, lam4inv, W):
    v, _ = _v2_raw(r, H, eta, zz, lam1inv, D, lam4inv, W)
    vrc, dvrc = _v2_raw(rc, H, eta, zz, lam1inv, D, lam4inv, W)
    return v - vrc - (r - rc) * dvrc


_PAIR_KEYS = ("H", "eta", "zz", "lam1inv", "D", "lam4inv", "W")


def _pair_energy(r, pt, p, vrc, dvrc):
    """Shifted-force V2 for distances r and flattened pair-type indices pt (0 beyond rc)."""
    g = lambda a: a.reshape(-1)[pt]  # noqa: E731
    v, _ = _v2_raw(r, *[g(p[k]) for k in _PAIR_KEYS])
    rc = g(p["rc"])
    return jnp.where(r < rc, v - g(vrc) - (r - rc) * g(dvrc), 0.0)


_pair_energy_ck = jax.checkpoint(_pair_energy)


def _select_first_k(valid: jnp.ndarray, k: int):
    """Column indices of the first k True entries per row (stable), their validity, row counts."""
    n, m = valid.shape
    key = jnp.where(valid, m - jnp.arange(m, dtype=jnp.int32)[None, :], 0)
    kk = min(k, m)
    vals, sel = lax.top_k(key, kk)
    return sel, vals > 0, jnp.sum(valid, axis=1)


def _three_body_candidates(r, mask, ti, tj, p):
    return mask & (r < p["r0"][ti, tj])


def _rows_energy(frac, box, species, idx_rows, rows, p, k3: int):
    """(energy of the listed rows, max three-body count) for centre atoms ``rows`` with list rows ``idx_rows``.

    ``rows=None`` means all atoms in order (idx_rows = idx).  Displacements use JAX-MD's
    periodic_general(fractional_coordinates=True), so grad w.r.t. ``frac`` is the real-space
    gradient (harness convention, see potentials/base.py).
    """
    n = frac.shape[0]
    disp, _ = space.periodic_general(box, fractional_coordinates=True)
    mask = idx_rows < n
    j = jnp.where(mask, idx_rows, 0)
    fi = frac if rows is None else frac[rows]
    si = species if rows is None else species[rows]
    d = jax.vmap(lambda a, bs: jax.vmap(lambda b: disp(b, a))(bs))(fi, frac[j])
    r = _safe_norm(d, mask)
    ti = jnp.broadcast_to(si[:, None], idx_rows.shape)
    tj = species[j]

    # ---------------- two-body (shifted force), full list -> factor 1/2
    # Per-pair-type shift constants V(rc), V'(rc) are computed once on the (ne, ne) tables; the per-entry
    # pair energy is rematerialised in the backward pass (jax.checkpoint): residuals are only r and the
    # pair-type index: compiled temp memory 33.4 -> 25.6 KB/atom on the GB10 at equal speed and identical results.
    ne = p["H"].shape[0]
    pt = ti * ne + tj
    vrc, dvrc = _v2_raw(p["rc"], *[p[k] for k in _PAIR_KEYS])
    e2 = 0.5 * jnp.sum(jnp.where(mask, _pair_energy_ck(r, pt, p, vrc, dvrc), 0.0))

    # ---------------- three-body on the in-kernel short sub-list
    in3 = _three_body_candidates(lax.stop_gradient(r), mask, ti, tj, p)
    sel, ok, n3 = _select_first_k(in3, k3)
    d3 = jnp.take_along_axis(d, sel[:, :, None], axis=1)
    r3 = jnp.take_along_axis(r, sel, axis=1)
    t3 = jnp.take_along_axis(tj, sel, axis=1)
    t0 = si[:, None]
    r0 = p["r0"][t0, t3]
    gam = p["gamma"][t0, t3]
    x = jnp.where(ok, r3 - r0, -1.0)  # < 0 inside r0
    f = jnp.where(ok, jnp.exp(gam / x), 0.0)
    cos = jnp.sum(d3[:, :, None, :] * d3[:, None, :, :], axis=-1) / (r3[:, :, None] * r3[:, None, :])
    ia = jnp.broadcast_to(t0[:, :, None], cos.shape)
    ja = jnp.broadcast_to(t3[:, :, None], cos.shape)
    ka = jnp.broadcast_to(t3[:, None, :], cos.shape)
    dc = cos - p["cos0"][ia, ja, ka]
    dc2 = dc * dc
    pang = dc2 / (1.0 + p["C"][ia, ja, ka] * dc2)
    kk = sel.shape[1]
    pair_ok = ok[:, :, None] & ok[:, None, :] & (~jnp.eye(kk, dtype=bool))[None]
    v3 = p["B"][ia, ja, ka] * f[:, :, None] * f[:, None, :] * pang
    e3 = 0.5 * jnp.sum(jnp.where(pair_ok, v3, 0.0))  # ordered (j, k) pairs -> j < k
    return e2 + e3, jnp.max(n3).astype(jnp.int32)


def _energy_status(frac, box, species, idx, p, k3: int, chunk: Optional[int]):
    """(energy with the first k3 three-body partners per atom, ThreeBodyStatus)."""
    m = idx.shape[1]
    if chunk is None:
        e, n3max = _rows_energy(frac, box, species, idx, None, p, k3)
    else:
        n = idx.shape[0]
        nb = -(-n // chunk)
        pad = nb * chunk - n
        rows = jnp.minimum(jnp.arange(nb * chunk), n - 1).reshape(nb, chunk)
        idx_p = jnp.concatenate([idx, jnp.full((pad, m), n, dtype=idx.dtype)], axis=0).reshape(nb, chunk, m)
        blk = jax.checkpoint(lambda ib, rb: _rows_energy(frac, box, species, ib, rb, p, k3))

        def body(acc, xs):
            ib, rb = xs
            eb, n3 = blk(ib, rb)
            return (acc[0] + eb, jnp.maximum(acc[1], n3)), None

        init = (jnp.zeros((), dtype=frac.dtype), jnp.zeros((), dtype=jnp.int32))
        (e, n3max), _ = lax.scan(body, init, (idx_p, rows))
    return e, ThreeBodyStatus(max_partners=n3max, overflow=n3max > min(k3, m))


@highest_matmul_precision
def vashishta_energy_and_status(frac, box, species, idx, p, k3: int = K3_DEFAULT, chunk: Optional[int] = None):
    """(energy in eV, :class:`ThreeBodyStatus`); ``k3`` and ``chunk`` must be static Python ints.

    The energy is always finite; it is the model energy only if ``status.overflow`` is False.
    ``chunk``: evaluate over blocks of this many centre atoms (lax.scan + jax.checkpoint), peak memory
    ~ chunk·M instead of N·M in the forward and backward pass; equal to the unchunked energy to
    summation round-off.  Use for >~1e5 atoms in float64 on the GPU.
    """
    return _energy_status(frac, box, species, idx, p, k3, chunk)


def _poison(e, status: ThreeBodyStatus):
    """NaN energy (and, through the product, NaN forces) on three-body slot overflow."""
    return e * jnp.where(status.overflow, jnp.nan, 1.0).astype(e.dtype)


@highest_matmul_precision
def vashishta_energy(frac, box, species, idx, p):
    """Harness energy_fn(frac, box, species, idx, params) with k3 = K3_DEFAULT (NaN on slot overflow)."""
    e, st = _energy_status(frac, box, species, idx, p, K3_DEFAULT, None)
    return _poison(e, st)


vashishta_energy.energy_and_status = vashishta_energy_and_status  # type: ignore[attr-defined]
vashishta_energy.k3, vashishta_energy.chunk, vashishta_energy.on_overflow = K3_DEFAULT, None, "nan"  # type: ignore[attr-defined]


def make_vashishta_energy(k3: int = K3_DEFAULT, chunk: Optional[int] = None, on_overflow: str = "nan") -> Callable:
    """Scalar energy_fn with ``k3`` three-body slots and optional atom chunking (both static).

    ``on_overflow``: "nan" (default) poisons energy and forces when an atom has more than k3
    partners; "truncate" returns the energy of the first k3 partners (only for callers that
    check ``energy_fn.energy_and_status`` for the same state).  The returned function carries
    ``.energy_and_status`` (same k3/chunk, returns ``(E, ThreeBodyStatus)``), ``.k3`` and ``.chunk``.
    Functions are cached per argument set, so repeated calls share compiled kernels.
    """
    if on_overflow not in ("nan", "truncate"):
        raise ValueError(f"on_overflow must be 'nan' or 'truncate', got {on_overflow!r}")
    if k3 < 1:
        raise ValueError("k3 must be >= 1")
    return _make_energy(int(k3), None if chunk is None else int(chunk), on_overflow)


_ENERGY_FNS: dict[tuple, Callable] = {(K3_DEFAULT, None, "nan"): vashishta_energy}


def _make_energy(k3: int, chunk: Optional[int], on_overflow: str) -> Callable:
    key = (k3, chunk, on_overflow)
    if key in _ENERGY_FNS:
        return _ENERGY_FNS[key]

    @highest_matmul_precision
    def energy_and_status(frac, box, species, idx, p):
        return _energy_status(frac, box, species, idx, p, k3, chunk)

    @highest_matmul_precision
    def efn(frac, box, species, idx, p):
        e, st = _energy_status(frac, box, species, idx, p, k3, chunk)
        return _poison(e, st) if on_overflow == "nan" else e

    tag = f"k{k3}" + ("" if chunk is None else f"_chunk{chunk}") + ("" if on_overflow == "nan" else "_truncate")
    efn.__name__ = efn.__qualname__ = f"vashishta_energy_{tag}"
    energy_and_status.__name__ = energy_and_status.__qualname__ = f"vashishta_energy_and_status_{tag}"
    efn.energy_and_status = energy_and_status  # type: ignore[attr-defined]
    efn.k3, efn.chunk, efn.on_overflow = k3, chunk, on_overflow  # type: ignore[attr-defined]
    _ENERGY_FNS[key] = efn
    return efn


def energy_and_status_fn(model: EnergyModel) -> Optional[Callable]:
    """``(frac, box, species, idx, params) -> (E, ThreeBodyStatus)`` of a model built here, else None."""
    return getattr(model.energy_fn, "energy_and_status", None)


def slots_needed(max_partners: int, headroom: int = 4) -> int:
    """Three-body slot count to reallocate to after an overflow with ``max_partners`` partners."""
    return int(max_partners) + int(headroom)


def with_three_body_slots(model: EnergyModel, k3: int) -> EnergyModel:
    """The same Vashishta model (same params and card) with ``k3`` three-body slots (rollback path).

    Keeps the model's chunking and overflow policy; raises for models not built by this module.
    """
    efn = model.energy_fn
    if not hasattr(efn, "energy_and_status"):
        raise ValueError(f"{model.name} has no three-body slot capacity to change")
    return dataclasses.replace(model, energy_fn=make_vashishta_energy(k3, efn.chunk, efn.on_overflow))


@jax.jit
@highest_matmul_precision
def three_body_occupancy(frac, box, species, idx, p):
    """(max, mean) number of three-body partners (r < r0) per atom; must stay <= k3.

    Evaluated in row blocks of 32768 atoms (lax.map) so that it never needs more memory than the
    energy kernel itself.
    """
    n, m = idx.shape
    blk = 32768 if n > 32768 else n
    nb = -(-n // blk)
    rows = jnp.minimum(jnp.arange(nb * blk), n - 1).reshape(nb, blk)
    valid = (jnp.arange(nb * blk) < n).reshape(nb, blk)
    disp, _ = space.periodic_general(box, fractional_coordinates=True)

    def one(args):
        rr, vv = args
        ir = idx[rr]
        mask = ir < n
        j = jnp.where(mask, ir, 0)
        d = jax.vmap(lambda a, bs: jax.vmap(lambda b: disp(b, a))(bs))(frac[rr], frac[j])
        r = _safe_norm(d, mask)
        ti = jnp.broadcast_to(species[rr][:, None], ir.shape)
        c = jnp.sum(_three_body_candidates(r, mask, ti, species[j], p), axis=1)
        return jnp.where(vv, c, 0)

    counts = lax.map(one, (rows, valid)).reshape(-1)[:n]
    return jnp.max(counts), jnp.mean(counts.astype(frac.dtype))


def pair_energy_curve(r, ti: int, tj: int, p, shifted: bool = True):
    """V2 (shifted or raw) for a species pair on an array of distances (for plots/tests)."""
    args = [p[k][ti, tj] for k in ("H", "eta", "zz", "lam1inv", "D", "lam4inv", "W")]
    if shifted:
        rc = p["rc"][ti, tj]
        return jnp.where(r < rc, _v2_shifted(r, rc, *args), 0.0)
    return _v2_raw(r, *args)[0]


# ----------------------------------------------------------------------------
# Potential card
# ----------------------------------------------------------------------------
_ELEMENTS = ("C", "Si")  # same order as the harness Tersoff models (species 0 = C, 1 = Si)

_CITATIONS = [
    Citation(
        key="Vashishta2007",
        text="P. Vashishta, R. K. Kalia, A. Nakano, J. P. Rino, J. Appl. Phys. 101, 103515 (2007)",
        doi="10.1063/1.2724570",
    ),
    Citation(
        key="LAMMPS_pair_vashishta",
        text="LAMMPS pair_style vashishta (src/MANYBODY/pair_vashishta.cpp; potentials/SiC.vashishta, "
        "2015-10-14, A. Thompson) - reference implementation and parameter file",
        url="https://docs.lammps.org/pair_vashishta.html",
    ),
]

_UMENO_2007 = ("Y. Umeno, Y. Kinoshita, T. Kitamura, Modelling Simul. Mater. Sci. Eng. 15, 27 (2007), p. 31: "
               "ideal shear strength of 3C-SiC 30.3 GPa (DFT-PW91; a literature comparison, not the reference)")
# DFT-PBE reference values computed in this project (Quantum ESPRESSO 7.5, SSSP 1.3.0 efficiency; docs/DFT.md
# section 8).  data/dft_reference/potential_comparison_sic.json is the single admission table (criteria and
# tolerances; tests E and S included since its version 2, accepted by the user on 2026-10-05).  These are model
# values (PBE), never measurements.
_PBE = "fracture-md DFT-PBE (QE 7.5, SSSP 1.3.0 efficiency; docs/DFT.md section 8), data/dft_reference/"
_PBE_SHEAR = _PBE + "sic_3c_ideal_shear_pbe.json: (111) ideal shear strength, same relaxed-ion protocol"
_PBE_GSFE = _PBE + "sic_3c_gsfe_pbe.json: (111) intrinsic stacking fault 7.6 mJ/m^2 (z-relaxed)"
_PBE_ELASTIC = _PBE + "sic_3c_elastic_pbe.json: relaxed-ion C44 239.4 GPa"
_PBE_B3B1 = _PBE + "sic_b3_b1_transition_pbe.json: static B3 -> B1 enthalpy crossing 65.3 GPa"
_DJEMIA_M = ("indentation modulus M(111) = 463.9 GPa (Vlassak-Nix) from the measured C_ij of P. Djemia et al. "
             "(2004); admission test E (data/dft_reference/potential_comparison_sic.json; proposed 2026-10-05 in "
             "docs/POPIN_VALIDATION.md section 4.3, accepted by the user 2026-10-05; tolerance 10 %)")
# Admission verdict (potential_comparison_sic.json "verdicts"; tests E and S, proposed 2026-10-05 in
# docs/POPIN_VALIDATION.md section 4.3 and accepted by the user 2026-10-05, are admission criteria of the table).
_ADMISSION = ("Fails the project's DFT-PBE admission table for contact loading (data/dft_reference/"
              "potential_comparison_sic.json; docs/DFT.md section 8.6): C_ij (C44 -43 %, C12 +12 %; tolerance 5 %), "
              "twinning-sense ideal shear strength (test S: 14.13 vs 27.76 GPa, -49 %; tolerance 15 %), intrinsic "
              "stacking fault (191 vs 7.6 mJ/m^2; window 0-50) and B3 -> B1 pressure (89.3 vs 65.3 GPa; +-10 GPa); "
              "only the glide-set gamma_us passes (-6 %).  Test E fails too (indentation modulus M(111) 336 vs 464 "
              "GPa from measured C_ij, -28 %; tolerance 10 %).  Tests E and S are admission criteria accepted by the "
              "user on 2026-10-05.")
_EA_2005_DEFECTS = "P. Erhart and K. Albe, PRB 71, 035211 (2005), Table XIV (DFT antisite pair 8.52 / 8.4 eV)"

_LIMITATIONS = [
    _ADMISSION + "  Plasticity and large strain are therefore known_deficient: indentation, pop-in and "
    "ideal-strength results are OUT_OF_DOMAIN, model-specific numbers for mechanisms and methods, never values "
    "for SiC (agents.submit needs allow_out_of_domain=True).",
    "Shear too soft: relaxed-ion C44 = 136.9 GPa (clamped-ion 203.8 GPa, 129.5 GPa at 300 K; LAMMPS gives the same) "
    "vs 256 GPa measured.  The 191 GPa printed in the paper (Tables II, IV) is not a property of this parameter set: "
    "no definition reproduces it (it corresponds to ~19 % internal relaxation; G12 side check).  C11 = 390.1 and "
    "C12 = 142.6 GPa are fit "
    "targets and correct.  The (001) indentation modulus is M = 350.5 GPa (0 K) vs 426 GPa from measured Cij, so "
    "shear-dominated responses (indentation load, pop-in, dislocation nucleation) are systematically softened.",
    "Ideal shear strength (111)[11-2] (twinning sense) = 14.1 GPa vs 27.76 GPa from DFT-PBE computed in this "
    "project (-49 %; (111)[1-10] 16.1 vs 30.6 GPa, anti-twinning 21.6 vs 47.6 GPa; the PW91 value 30.3 GPa of "
    "Umeno et al. 2007 is a comparison only), falling 54 % at 50 GPa pressure; intrinsic stacking-fault energy "
    "190-191 mJ/m^2 (by every route, incl. the paper's slab protocol) vs 7.6 mJ/m^2 from DFT-PBE (this project; "
    "published DFT 10-24 mJ/m^2): partial-dislocation and twinning mechanisms are biased.  The paper's MD "
    "value (Sec. V.C, 5 meV/A^2 = 80 mJ/m^2) is not reproducible and contradicts its own 2H energy.",
    "Fixed effective charges (Z = +-1.201 e) screened by lambda1 = 5 A: no charge transfer; a separated antisite "
    "pair costs 27.0 eV vs 8.4-8.5 eV from DFT, so chemical disorder and amorphisation are penalised.  "
    "Homonuclear Si-Si/C-C bonds are described by the two-body term only.",
    "Three-body term only for Si-C-Si and C-Si-C angles within r0 = 2.90 A; mixed triplets (e.g. C-Si-Si with one "
    "homonuclear leg) carry no angular energy.",
    "Shifted-force truncation at rc = 7.35 A: energy and force are continuous, the second derivative is not.  Along "
    "hydrostatic compression the bulk modulus jumps where a neighbour shell crosses rc: +10 % at 23.0 GPa (-2.63 % "
    "linear strain), -18 % at 68.7 GPa (-5.54 %), +24 % at 177.6 GPa (-9.85 %); Born instability of B3 at 94.8 GPa.  "
    "Forces stay continuous, so MD energy conservation is unaffected.",
    "Long two-body range (rc = 7.35 A, ~182 neighbours inside rc + 0.5 A at P = 0): ~10x the pair work of "
    "Tersoff; periodic cells must be wider than 2 (rc + skin) = 15.7 A (the 512-atom 4x4x4 cell is the smallest "
    "cubic 3C cell).",
    "Parameter provenance: LAMMPS SiC.vashishta (= Table I of the paper; D_file = D_paper/2 in the file's units).  "
    "The cohesive energy is 6.340145 eV/atom, 0.94 meV/atom less bound than the paper's 6.3410868 eV (Table II), "
    "consistent with a Z_i Z_j e^2 1.28e-4 larger in the authors' code (G12 side check; no mechanical effect).",
    "float32 is not adequate for records: at 131k atoms E/atom was off by 1.7 meV with TF32 matmuls and the 200-step "
    "NVE error was 1.8e-3 eV/atom (float64: 5e-5).  Kernels force 'highest' matmul precision; use float64.",
]

_NO_EVIDENCE_NOTE = "model-specific: MD quantities from this potential are not validated against experiment"


def _evidence(kind: str, description: str, citation: str, computed: Quantity, reference: Quantity) -> ValidationEvidence:
    """A comparison this model FAILS (evidence of a known bias, not of validity)."""
    c, r = computed.magnitude(), reference.magnitude()
    return ValidationEvidence(kind=kind, description=description, citation=citation, computed_value=computed,
                              reference_value=reference, relative_difference=(c - r) / r, passed=False)


def _card(params, k3: int = K3_DEFAULT) -> PotentialCard:
    h = params_sha256(params)
    cutoff = float(np.max(np.asarray(params["rc"])))
    kind = getattr(PotentialKind, "ANALYTIC_MANY_BODY", PotentialKind.ANALYTIC_PAIR)
    return PotentialCard(
        id=f"vashishta-2007-SiC@{h[:10]}",
        name="Vashishta-Kalia-Nakano-Rino (2007) Si-C",
        kind=kind,
        functional_form="Vashishta two-body (steric + screened Coulomb + screened charge-dipole + vdW, "
        "shifted-force at rc) + modified Stillinger-Weber three-body (LAMMPS pair_style vashishta)",
        elements=list(_ELEMENTS),
        cutoff=q(cutoff, Unit.ANGSTROM),
        parameters_sha256=h,
        parameter_source="LAMMPS potentials/SiC.vashishta (2015-10-14, A. Thompson; sha256 "
        f"{SIC_VASHISHTA_2007_FILE_SHA256[:16]}...; same parameters as NIST IPR "
        "2007--Vashishta-P-Kalia-R-K-Nakano-A-Rino-J-P--Si-C), from Table I of J. Appl. Phys. 101, 103515 (2007); "
        "file convention D_file = D_paper/2 in the file's units (header note; = D_paper[e^2 A^3] x e^2/2 with an "
        "implied e^2 = 14.3971 eV*A, 1.8e-4 below qqr2e: no measurable effect); Coulomb constant qqr2e = 14.399645 "
        "eV*A (LAMMPS metal).  Table I transcription check G12: file == paper (outputs/impl/side-vashishta-file.md).",
        license="Parameters are published scientific data (no license restriction); LAMMPS (GPL-2.0) used "
        "only as a test-time reference, not redistributed.",
        code_license="MIT (fracture_md)",
        implementation="fracture_md.potentials.vashishta.vashishta_energy",
        citations=_CITATIONS,
        training_domain=TrainingDomain(
            elements=list(_ELEMENTS),
            description="Fitted to the cohesive energy, bulk modulus and C11 of 3C-SiC at the experimental "
            "density (paper Sec. II); exponents eta and screening lengths fixed a priori.  Surfaces, defects, "
            "large strain and the rock-salt phase were not fit targets.",
            reference_method="empirical fit to experimental data",
            includes_surfaces=False,
            includes_large_strain=False,
            includes_bond_breaking=False,
        ),
        domain_assessment=[
            DomainAssessment(
                property_class=PropertyClass.EQUILIBRIUM_BULK,
                status="partially_validated",
                notes="a0 = 4.358174 A, E_coh = 6.340145 eV/atom, B = 225.1 GPa reproduce paper Table II (a0 4.3581 A, "
                "6.3410868 eV, 225.2 GPa) and experiment (a0 4.3596 A).",
            ),
            DomainAssessment(
                property_class=PropertyClass.ELASTIC,
                status="partially_validated",
                evidence=[_evidence("experimental_reference", "relaxed-ion C44 of this kernel (= LAMMPS) vs measured",
                                    "Vashishta et al., JAP 101, 103515 (2007), Table IV (experimental column)",
                                    q(136.93, Unit.GPA), q(256.0, Unit.GPA)),
                          _evidence("dft_reference", "relaxed-ion C44 of this kernel vs DFT-PBE (admission criterion "
                                    "C_ij: each within 5 %)", _PBE_ELASTIC, q(136.93, Unit.GPA), q(239.38, Unit.GPA)),
                          _evidence("experimental_reference", "test E: indentation modulus M(111) of this potential "
                                    "(Vlassak-Nix, 300 K relaxed-ion C_ij) vs measured C_ij", _DJEMIA_M,
                                    q(336.1, Unit.GPA), q(463.9, Unit.GPA))],
                notes="C11 = 390.11, C12 = 142.61 GPa (fit targets; C11 +2 %, C12 +12 % vs DFT-PBE); relaxed-ion C44 = "
                "136.93 GPa is 43 % below DFT-PBE (239.4) and 47 % below experiment: fails the admission C_ij "
                "criterion.  Shear moduli and the indentation modulus are deficient (test E fails: M(111) 336 vs "
                "464 GPa, -28 %; M(001) 350.5 at 0 K vs 426 GPa).  Kept partially_validated (B, C11 correct); the "
                "contact-loading consequence is carried by the known_deficient plasticity and large-strain classes.",
            ),
            DomainAssessment(
                property_class=PropertyClass.PHASE_STABILITY,
                status="partially_validated",
                evidence=[_evidence("dft_reference", "static B3 -> B1 enthalpy crossing vs DFT-PBE (admission "
                                    "criterion: within 10 GPa)", _PBE_B3B1, q(89.3, Unit.GPA), q(65.3, Unit.GPA))],
                notes="B3->B1 enthalpy crossing 89.3 GPa, volume collapse 18.8 %: fails the admission criterion "
                "against DFT-PBE (65.3 GPa, this project; tolerance +-10 GPa); experiment > 100 GPa (20.3 %), "
                "published DFT ~60-67 GPa.  E(2H) - E(3C) = 18.9 meV/atom (polytype ordering not validated).",
            ),
            DomainAssessment(
                property_class=PropertyClass.HIGH_PRESSURE,
                status="partially_validated",
                notes="No switching-function wall (unlike Tersoff/Erhart-Albe); smooth P(V) to > 200 GPa with the "
                "B3->B1 transition at 89.3 GPa.  Bulk-modulus jumps of +10 % at 23.0 GPa and -18 % at 68.7 GPa "
                "where neighbour shells cross rc; B3 Born instability at 94.8 GPa.  The B3->B1 pressure fails the "
                "DFT-PBE admission criterion (phase stability above).",
            ),
            DomainAssessment(
                property_class=PropertyClass.PLASTICITY,
                status="known_deficient",
                evidence=[
                    _evidence("dft_reference", "test S: (111)[11-2] twinning-sense ideal shear strength of this "
                              "potential vs DFT-PBE (admission tolerance 15 %)", _PBE_SHEAR,
                              q(14.13, Unit.GPA), q(27.76, Unit.GPA)),
                    _evidence("dft_reference", "(111) intrinsic stacking-fault energy (z-relaxed) vs DFT-PBE "
                              "(admission window 0-50 mJ/m^2)", _PBE_GSFE,
                              q(191.0, Unit.MJ_PER_M2), q(7.644, Unit.MJ_PER_M2)),
                    _evidence("dft_reference", "separated antisite pair C_Si + Si_C (512-atom cell, relaxed) vs DFT",
                              _EA_2005_DEFECTS,
                              q(27.0, Unit.EV), q(8.5, Unit.EV)),
                ],
                notes="Fails the DFT-PBE admission table, including tests E and S (admission criteria accepted by "
                "the user on 2026-10-05; docs/DFT.md section 8.6, docs/POPIN_VALIDATION.md section 4.3), so contact "
                "plasticity is known_deficient: indentation, "
                "pop-in and other plasticity records are OUT_OF_DOMAIN, numbers of this model for mechanisms and "
                "methods, never values for SiC (agents.submit runs them only with allow_out_of_domain=True).  It "
                "remains the only empirical Si-C potential here without a cutoff-function artefact at indentation "
                "pressures, hence the project's potential for mechanism studies; "
                f"{_NO_EVIDENCE_NOTE}.  Known biases: twinning-sense ideal shear strength 14.13 vs 27.76 GPa "
                "DFT-PBE (-49 %; PW91 30.3 GPa, Umeno 2007, as a comparison), ISF 191 vs 7.6 mJ/m^2 (DFT-PBE), "
                "antisites 3.2x too expensive, relaxed C44 43 % below DFT-PBE.",
            ),
            DomainAssessment(
                property_class=PropertyClass.LARGE_STRAIN,
                status="known_deficient",
                evidence=[
                    _evidence("dft_reference", "(111)[11-2] twinning-sense ideal shear strength vs DFT-PBE (test S, "
                              "tolerance 15 %)", _PBE_SHEAR, q(14.13, Unit.GPA), q(27.76, Unit.GPA)),
                    _evidence("dft_reference", "(111)[-1-12] anti-twinning ideal shear strength vs DFT-PBE",
                              _PBE_SHEAR, q(21.63, Unit.GPA), q(47.57, Unit.GPA)),
                    _evidence("dft_reference", "(111)[1-10] ideal shear strength vs DFT-PBE", _PBE_SHEAR,
                              q(16.11, Unit.GPA), q(30.59, Unit.GPA)),
                ],
                notes="Fails admission test S: twinning-sense ideal shear strength 14.13 vs 27.76 GPa DFT-PBE (-49 %; "
                "tolerance 15 %); (111)[1-10] 16.11 vs 30.59 GPa (-47 %), anti-twinning 21.63 vs 47.57 GPa (-55 %), "
                "same relaxed-ion protocol (data/dft_reference/potential_comparison_sic.json).  Literature "
                "comparisons: PW91 30.3 GPa (Umeno 2007); [111] ideal tension 39 GPa vs published DFT 45-51 GPa.",
            ),
            DomainAssessment(
                property_class=PropertyClass.STACKING_FAULTS,
                status="known_deficient",
                evidence=[_evidence("dft_reference", "(111) intrinsic stacking-fault energy (z-relaxed) vs DFT-PBE "
                                    "(admission window 0-50 mJ/m^2)", _PBE_GSFE,
                                    q(191.0, Unit.MJ_PER_M2), q(7.644, Unit.MJ_PER_M2))],
                notes="Intrinsic stacking-fault energy 190-191 mJ/m^2 (tilted (111) cells) vs 7.6 mJ/m^2 DFT-PBE (this "
                "project; admission window 0-50 mJ/m^2) and 80 mJ/m^2 in the paper (not reproducible); published "
                "DFT 10-24 mJ/m^2 as a comparison.  Glide-set gamma_us 2.67 vs 2.84 J/m^2 DFT-PBE (-6 %, passes the "
                "15 % criterion); shuffle-set gamma_us 1.83 vs 3.79 J/m^2 (-52 %).",
            ),
            DomainAssessment(
                property_class=PropertyClass.DEFECTS,
                status="known_deficient",
                evidence=[_evidence("dft_reference", "separated antisite pair C_Si + Si_C vs DFT", _EA_2005_DEFECTS,
                                    q(27.0, Unit.EV), q(8.5, Unit.EV))],
                notes="Chemical disorder (antisites) 3.2x too expensive; vacancies/interstitials not assessed.",
            ),
            DomainAssessment(
                property_class=PropertyClass.SURFACES,
                status="unvalidated",
                notes="(110) surface energy 1.81 / 1.69 J/m^2 (unrelaxed / relaxed) reproduces the paper's model value "
                "to -2.4 %; no comparison with DFT or experiment; polar (001) unassessed.",
            ),
            DomainAssessment(property_class=PropertyClass.BOND_BREAKING, status="unvalidated",
                             notes="Smooth (no switching function) but never validated against DFT tractions."),
            DomainAssessment(property_class=PropertyClass.FINITE_TEMPERATURE, status="unvalidated",
                             notes="Paper: melting at 3250 +- 50 K; 300 K C11/C12/C44 = 369/134/130 GPa."),
            DomainAssessment(property_class=PropertyClass.INTERFACES, status="unvalidated"),
        ],
        known_limitations=_LIMITATIONS,
        differentiable=True,
        precision="float64",
        notes=f"Single Dense neighbour list at rc + skin; three-body sub-list of k3 = {k3} slots selected in-kernel; "
        "slot overflow is flagged (energy_fn.energy_and_status) and poisons the scalar energy with NaN.",
    )


def vashishta_2007_sic(k3: int = K3_DEFAULT, param_text: Optional[str] = None, chunk: Optional[int] = None,
                       on_overflow: str = "nan") -> EnergyModel:
    """EnergyModel for the Vashishta (2007) SiC potential (elements C, Si).

    k3: three-body slots per atom (static); chunk: evaluate in blocks of this many atoms
    (memory-bounded; for large systems).  Both change only performance and the overflow
    threshold, never the energy.  ``param_text``: an alternative LAMMPS vashishta file
    (different parameters -> different card id).
    """
    entries = parse_lammps_vashishta(param_text or SIC_VASHISHTA_2007_LAMMPS)
    params = tables_from_lammps(entries, _ELEMENTS)
    card = _card(params, k3)
    return EnergyModel(
        name="vashishta_2007_SiC",
        elements=_ELEMENTS,
        cutoff=card.cutoff.magnitude(),
        params=params,
        energy_fn=make_vashishta_energy(k3, chunk, on_overflow),
        card=card,
    )
