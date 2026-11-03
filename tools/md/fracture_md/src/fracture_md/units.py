"""Explicit unit system for fracture_md.

Internal computational units are LAMMPS/ASE "metal" units:

    length       angstrom (Å)
    energy       electronvolt (eV)
    mass         atomic mass unit (amu, Da)
    time         picosecond (ps) for integrators; inputs may be given in fs
    temperature  kelvin (K)

Every number that leaves the physics core is wrapped in a :class:`Quantity`
carrying a :class:`Unit`.  Conversions go through SI using CODATA 2018 exact
or recommended values, and are dimension-checked so that e.g. ``J/m^2`` can be
converted to ``eV/angstrom^2`` but never to ``GPa``.

The dimension vector is (mass, length, time, temperature, atom-count).  The
"atom" count dimension keeps ``eV`` and ``eV/atom`` distinct.

Hardness trap: ``Unit.HV`` is the Vickers number (kgf/mm^2 on the *sloped*
pyramid area, ISO 6507-1), so the unit table converts 1 HV = 9.80665 MPa.  A
*projected-area* hardness (ISO 14577 H_IT, Oliver-Pharr, or an MD mean contact
pressure P/A_proj) is converted with HV = 0.0945 H_IT[MPa] instead
(:func:`h_it_to_hv`); converting it with the unit table overstates HV by 7.9 %
(39.1 GPa -> 3987 instead of 3695 HV).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

# --- CODATA 2018 (exact where the SI defines them) ---------------------------
ELEMENTARY_CHARGE_C = 1.602176634e-19  # exact
EV_J = ELEMENTARY_CHARGE_C  # 1 eV in J, exact
ANGSTROM_M = 1.0e-10  # exact
AMU_KG = 1.66053906660e-27  # CODATA 2018
AVOGADRO = 6.02214076e23  # exact
BOLTZMANN_EV_K = 8.617333262e-5  # exact (k_B / e)

# Frequently used derived factors (kept explicit for auditability).
EV_PER_A3_TO_GPA = EV_J / ANGSTROM_M**3 / 1.0e9  # 160.2176634 (exact)
EV_PER_A2_TO_J_PER_M2 = EV_J / ANGSTROM_M**2  # 16.02176634
AMU_PER_A3_TO_G_PER_CM3 = AMU_KG * 1.0e3 / (ANGSTROM_M * 1.0e2) ** 3  # 1.66053906660
EV_PER_A_TO_NN = EV_J / ANGSTROM_M * 1.0e9  # 1.602176634 (exact): 1 eV/angstrom in nN
# ISO 14577-1: HV = 0.0945 H_IT with H_IT in MPa (projected contact area).
ISO14577_HV_PER_MPA = 0.0945

_Dim = tuple[int, int, int, int, int]  # (M, L, T, Theta, atom)


class Unit(str, Enum):
    """Controlled vocabulary of units used in results and schemas."""

    DIMENSIONLESS = "1"
    PERCENT = "%"
    ANGSTROM = "angstrom"
    NANOMETER = "nm"
    METER = "m"
    ANGSTROM3 = "angstrom^3"
    ANGSTROM3_PER_ATOM = "angstrom^3/atom"
    ANGSTROM2 = "angstrom^2"
    PER_ANGSTROM2 = "1/angstrom^2"
    EV = "eV"
    EV_PER_ATOM = "eV/atom"
    MEV_PER_ATOM = "meV/atom"
    J = "J"
    EV_PER_ANGSTROM = "eV/angstrom"
    EV_PER_ANGSTROM2 = "eV/angstrom^2"
    EV_PER_ANGSTROM3 = "eV/angstrom^3"
    J_PER_M2 = "J/m^2"
    MJ_PER_M2 = "mJ/m^2"
    PA = "Pa"
    MPA = "MPa"
    GPA = "GPa"
    PER_GPA = "1/GPa"
    PER_PA = "1/Pa"
    MPA_SQRT_M = "MPa*m^0.5"
    AMU = "amu"
    KG = "kg"
    G_PER_CM3 = "g/cm^3"
    KG_PER_M3 = "kg/m^3"
    KELVIN = "K"
    PER_KELVIN = "1/K"
    SECOND = "s"
    PICOSECOND = "ps"
    FEMTOSECOND = "fs"
    PER_SECOND = "1/s"
    PER_PICOSECOND = "1/ps"
    ANGSTROM_PER_PS = "angstrom/ps"
    METER_PER_SECOND = "m/s"  # indenter / loading speed
    NEWTON = "N"
    MICRONEWTON = "uN"  # instrumented-indentation loads
    NANONEWTON = "nN"  # MD indentation loads (1 eV/angstrom = 1.602176634 nN)
    DEGREE = "degree"
    RADIAN = "rad"
    HV = "HV"  # Vickers hardness number, kgf/mm^2 (ISO 6507-1); 1 HV = 9.80665 MPa
    MICROGRAM_PER_MM3 = "ug/mm^3"  # water sorption / solubility (ISO 4049)
    PA_S = "Pa*s"  # dynamic viscosity
    MJ_PER_CM2 = "mJ/cm^2"  # photopolymerisation dose
    MICROMETER = "um"
    MILLIMETER = "mm"
    WT_PERCENT = "wt%"
    VOL_PERCENT = "vol%"


# (SI factor, dimension).  SI base: kg, m, s, K, atom-count.
_UNIT_TABLE: dict[Unit, tuple[float, _Dim]] = {
    Unit.DIMENSIONLESS: (1.0, (0, 0, 0, 0, 0)),
    Unit.PERCENT: (1.0e-2, (0, 0, 0, 0, 0)),
    Unit.DEGREE: (np.pi / 180.0, (0, 0, 0, 0, 0)),
    Unit.RADIAN: (1.0, (0, 0, 0, 0, 0)),
    Unit.HV: (9.80665e6, (1, -1, -2, 0, 0)),
    Unit.MICROGRAM_PER_MM3: (1.0e-9 / 1.0e-9, (1, -3, 0, 0, 0)),
    Unit.PA_S: (1.0, (1, -1, -1, 0, 0)),
    Unit.MJ_PER_CM2: (1.0e-3 / 1.0e-4, (1, 0, -2, 0, 0)),
    Unit.MICROMETER: (1.0e-6, (0, 1, 0, 0, 0)),
    Unit.MILLIMETER: (1.0e-3, (0, 1, 0, 0, 0)),
    Unit.WT_PERCENT: (1.0e-2, (0, 0, 0, 0, 0)),
    Unit.VOL_PERCENT: (1.0e-2, (0, 0, 0, 0, 0)),
    Unit.ANGSTROM: (ANGSTROM_M, (0, 1, 0, 0, 0)),
    Unit.NANOMETER: (1.0e-9, (0, 1, 0, 0, 0)),
    Unit.METER: (1.0, (0, 1, 0, 0, 0)),
    Unit.ANGSTROM2: (ANGSTROM_M**2, (0, 2, 0, 0, 0)),
    Unit.PER_ANGSTROM2: (ANGSTROM_M**-2, (0, -2, 0, 0, 0)),
    Unit.ANGSTROM3: (ANGSTROM_M**3, (0, 3, 0, 0, 0)),
    Unit.ANGSTROM3_PER_ATOM: (ANGSTROM_M**3, (0, 3, 0, 0, -1)),
    Unit.EV: (EV_J, (1, 2, -2, 0, 0)),
    Unit.J: (1.0, (1, 2, -2, 0, 0)),
    Unit.EV_PER_ATOM: (EV_J, (1, 2, -2, 0, -1)),
    Unit.MEV_PER_ATOM: (1.0e-3 * EV_J, (1, 2, -2, 0, -1)),
    Unit.EV_PER_ANGSTROM: (EV_J / ANGSTROM_M, (1, 1, -2, 0, 0)),
    Unit.EV_PER_ANGSTROM2: (EV_J / ANGSTROM_M**2, (1, 0, -2, 0, 0)),
    Unit.J_PER_M2: (1.0, (1, 0, -2, 0, 0)),
    Unit.MJ_PER_M2: (1.0e-3, (1, 0, -2, 0, 0)),
    Unit.EV_PER_ANGSTROM3: (EV_J / ANGSTROM_M**3, (1, -1, -2, 0, 0)),
    Unit.PA: (1.0, (1, -1, -2, 0, 0)),
    Unit.MPA: (1.0e6, (1, -1, -2, 0, 0)),
    Unit.GPA: (1.0e9, (1, -1, -2, 0, 0)),
    Unit.PER_GPA: (1.0e-9, (-1, 1, 2, 0, 0)),
    Unit.PER_PA: (1.0, (-1, 1, 2, 0, 0)),
    Unit.MPA_SQRT_M: (1.0e6, (1, -1, -2, 0, 0)),  # Pa*m^0.5 -> handled below
    Unit.AMU: (AMU_KG, (1, 0, 0, 0, 0)),
    Unit.KG: (1.0, (1, 0, 0, 0, 0)),
    Unit.G_PER_CM3: (1.0e3, (1, -3, 0, 0, 0)),
    Unit.KG_PER_M3: (1.0, (1, -3, 0, 0, 0)),
    Unit.KELVIN: (1.0, (0, 0, 0, 1, 0)),
    Unit.PER_KELVIN: (1.0, (0, 0, 0, -1, 0)),
    Unit.SECOND: (1.0, (0, 0, 1, 0, 0)),
    Unit.PICOSECOND: (1.0e-12, (0, 0, 1, 0, 0)),
    Unit.FEMTOSECOND: (1.0e-15, (0, 0, 1, 0, 0)),
    Unit.PER_SECOND: (1.0, (0, 0, -1, 0, 0)),
    Unit.PER_PICOSECOND: (1.0e12, (0, 0, -1, 0, 0)),
    Unit.ANGSTROM_PER_PS: (ANGSTROM_M / 1.0e-12, (0, 1, -1, 0, 0)),
    Unit.METER_PER_SECOND: (1.0, (0, 1, -1, 0, 0)),
    Unit.NEWTON: (1.0, (1, 1, -2, 0, 0)),
    Unit.MICRONEWTON: (1.0e-6, (1, 1, -2, 0, 0)),
    Unit.NANONEWTON: (1.0e-9, (1, 1, -2, 0, 0)),
}

# Stress-intensity units carry a half-integer length power; keep them in their
# own family so they can only convert among themselves.
_KIC_FAMILY = {Unit.MPA_SQRT_M}


class UnitError(ValueError):
    """Raised on a dimensionally invalid conversion."""


def dimension(unit: Unit) -> _Dim:
    return _UNIT_TABLE[Unit(unit)][1]


def conversion_factor(from_unit: Unit | str, to_unit: Unit | str) -> float:
    """Multiplicative factor f such that value[to_unit] = f * value[from_unit]."""
    fu, tu = Unit(from_unit), Unit(to_unit)
    if (fu in _KIC_FAMILY) != (tu in _KIC_FAMILY):
        raise UnitError(f"cannot convert {fu.value} to {tu.value}")
    f_si, f_dim = _UNIT_TABLE[fu]
    t_si, t_dim = _UNIT_TABLE[tu]
    if f_dim != t_dim:
        raise UnitError(
            f"incompatible dimensions: {fu.value} {f_dim} -> {tu.value} {t_dim}"
        )
    return f_si / t_si


def compatible(a: Unit | str, b: Unit | str) -> bool:
    """True if values in unit `a` can be converted to unit `b` (same dimension and family)."""
    try:
        conversion_factor(a, b)
    except UnitError:
        return False
    return True


def convert(value: Any, from_unit: Unit | str, to_unit: Unit | str) -> Any:
    """Convert a scalar or array-like value between compatible units."""
    f = conversion_factor(from_unit, to_unit)
    if isinstance(value, (list, tuple)):
        return (np.asarray(value, dtype=float) * f).tolist()
    return value * f


Number = Union[float, int]
ArrayLike = Union[Number, list]


class Quantity(BaseModel):
    """A value (scalar or nested list = tensor/array) with an explicit unit.

    Arrays are stored as nested lists so that the JSON form is self-describing;
    use :meth:`as_array` to recover a numpy array (or ``jnp.asarray`` for JAX).
    """

    model_config = ConfigDict(frozen=True)

    value: ArrayLike
    unit: Unit

    @field_validator("value", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> Any:
        if hasattr(v, "tolist"):
            v = v.tolist()
        if isinstance(v, float) and not np.isfinite(v):
            raise ValueError("non-finite quantity value")
        return v

    def to(self, unit: Unit | str) -> "Quantity":
        return Quantity(value=convert(self.value, self.unit, unit), unit=Unit(unit))

    def as_array(self, unit: Unit | str | None = None) -> np.ndarray:
        q = self if unit is None else self.to(unit)
        return np.asarray(q.value, dtype=float)

    def magnitude(self, unit: Unit | str | None = None) -> float:
        q = self if unit is None else self.to(unit)
        if isinstance(q.value, list):
            raise TypeError("magnitude() is only defined for scalar quantities")
        return float(q.value)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Quantity({self.value!r}, {self.unit.value!r})"


def q(value: Any, unit: Unit | str) -> Quantity:
    """Shorthand constructor."""
    return Quantity(value=value, unit=Unit(unit))


def require_compatible(quantity: Optional["Quantity"], like: Unit | str, what: str) -> None:
    """Raise UnitError unless `quantity` (if given) converts to unit `like` (e.g. a hardness in K)."""
    if quantity is not None and not compatible(quantity.unit, like):
        raise UnitError(f"{what} must have the dimension of {Unit(like).value}, got {quantity.unit.value!r}")


# --- Hardness conversions (see the module docstring for the HV trap) ---------
def h_it_to_hv(h_it: Quantity) -> Quantity:
    """ISO 14577-1 conversion of a projected-area indentation hardness to a Vickers number.

    HV = 0.0945 H_IT[MPa] (39.1 GPa -> 3695 HV).  Use this for H_IT, Oliver-Pharr and MD
    projected-area hardness, never ``Quantity.to(Unit.HV)`` (the unit table assumes the sloped
    Vickers area and overstates HV by 7.9 %).  The result is a *model conversion* (it assumes
    Vickers/Berkovich area equivalence and no indentation-size effect between the two tests), so
    records built from it are DERIVED_MODEL (see ``records.vickers_estimate_from_indentation``).
    """
    require_compatible(h_it, Unit.MPA, "projected-area hardness")
    if h_it.unit == Unit.HV:
        raise UnitError("h_it_to_hv expects a projected-area hardness in pressure units, not a Vickers number")
    return Quantity(value=_scale(convert(h_it.value, h_it.unit, Unit.MPA), ISO14577_HV_PER_MPA), unit=Unit.HV)


def hv_to_h_it(hv: Quantity, unit: Unit | str = Unit.GPA) -> Quantity:
    """Inverse of :func:`h_it_to_hv`: H_IT[MPa] = HV / 0.0945, returned in `unit` (pressure)."""
    if Unit(hv.unit) != Unit.HV:
        raise UnitError(f"hv_to_h_it expects a Vickers number in HV, got {hv.unit.value!r}")
    mpa = _scale(hv.value, 1.0 / ISO14577_HV_PER_MPA)
    return Quantity(value=convert(mpa, Unit.MPA, unit), unit=Unit(unit))


def _scale(value: Any, factor: float) -> Any:
    if isinstance(value, (list, tuple)):
        return (np.asarray(value, dtype=float) * factor).tolist()
    return value * factor
