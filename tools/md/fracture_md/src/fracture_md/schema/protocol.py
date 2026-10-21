"""Engine-independent simulation protocols.

A protocol says *what* to compute and under which numerical settings, never
*how* a particular engine does it.  The JAX-MD engine, the ASE/matscipy
reference engine and any future TorchSim or LAMMPS engine consume the same
protocol objects and must return the same result schema.

All lengths are in angstrom, energies in eV, stresses in GPa and forces in
eV/angstrom unless the field name says otherwise.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import Field, model_validator

from fracture_md.schema.common import AreaEstimator, Conditions, ContactConditions, StrictModel
from fracture_md.units import Unit, q


class _ProtocolBase(StrictModel):
    protocol_version: str = "0.1.0"
    notes: str = ""

    def protocol_id(self) -> str:
        """Content hash of the protocol (stable across runs)."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return f"{self.kind}@{hashlib.sha256(payload).hexdigest()[:12]}"  # type: ignore[attr-defined]


class RelaxSettings(StrictModel):
    fmax_eV_per_A: float = 1.0e-5
    max_steps: int = 20000
    optimizer: Literal["fire", "bfgs", "lbfgs", "cg", "engine_default"] = "fire"


class BulkRelaxProtocol(_ProtocolBase):
    kind: Literal["bulk_relax"] = "bulk_relax"
    relax: RelaxSettings = Field(default_factory=RelaxSettings)
    relax_cell: bool = True
    # Cell relaxation: "isotropic" keeps symmetry for cubic phases; "full" relaxes all 6 strains.
    cell_mode: Literal["isotropic", "full"] = "full"
    stress_tol_GPa: float = 1.0e-4
    supercell: Optional[tuple[int, int, int]] = None  # None = auto (minimum image safe)


class ElasticTensorProtocol(_ProtocolBase):
    kind: Literal["elastic_tensor"] = "elastic_tensor"
    # "autodiff_relaxed_ion": C = C_clamped - Λᵀ K⁺ Λ / V from exact Hessians.
    # "finite_difference": ± strain, relax internal coordinates, linear fit of stress.
    methods: list[Literal["autodiff_relaxed_ion", "finite_difference"]] = Field(
        default_factory=lambda: ["autodiff_relaxed_ion", "finite_difference"]
    )
    strain_magnitudes: list[float] = Field(default_factory=lambda: [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3])
    relax_internal: bool = True
    relax: RelaxSettings = Field(default_factory=lambda: RelaxSettings(fmax_eV_per_A=1e-7))
    symmetry_tolerance_GPa: float = 0.5
    fd_vs_autodiff_tolerance_GPa: float = 1.0


class CleavageProtocol(_ProtocolBase):
    """Opening a crystal on a plane with a given termination.

    mode="rigid": the upper block is translated rigidly by δ along the plane
      normal; no relaxation (the "unrelaxed" / universal-binding-energy curve).
    mode="relaxed": `grip_layers` atomic layers at the top and bottom of the
      slab are displaced by the prescribed opening; all other atoms relax.
      The in-plane lattice is held at the bulk value (laterally constrained,
      as for a crack plane embedded in a large body).  This curve is slab-size
      dependent after the peak; see docs/PHYSICS_NOTES.md.
    """

    kind: Literal["cleavage"] = "cleavage"
    plane_miller: tuple[int, int, int]
    termination: str = "auto_min_broken_bonds"  # or a label returned by the termination enumerator
    n_layers: int = 12  # atomic layers (or bilayers for polar zincblende planes) in the slab
    in_plane_repeats: Optional[tuple[int, int]] = None  # None = auto (minimum image safe)
    vacuum_A: float = 15.0
    mode: Literal["rigid", "relaxed"] = "rigid"
    openings_A: list[float] = Field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    )
    grip_layers: int = 2
    relax: RelaxSettings = Field(default_factory=lambda: RelaxSettings(fmax_eV_per_A=1e-4))
    neighbor_list_policy: Literal["rebuild_each_step", "static_with_skin"] = "rebuild_each_step"
    lateral_constraint: Literal["fixed_bulk_lattice"] = "fixed_bulk_lattice"

    @model_validator(mode="after")
    def _check_openings(self) -> "CleavageProtocol":
        o = self.openings_A
        if sorted(o) != list(o) or o[0] != 0.0:
            raise ValueError("openings_A must be sorted ascending and start at 0.0")
        return self


class TangentialSlipProtocol(_ProtocolBase):
    """Extension point: rigid in-plane sliding at fixed normal opening (γ-line)."""

    kind: Literal["tangential_slip"] = "tangential_slip"
    plane_miller: tuple[int, int, int]
    slip_direction: tuple[int, int, int]
    termination: str = "auto_min_broken_bonds"
    normal_opening_A: float = 0.0
    n_layers: int = 12
    vacuum_A: float = 15.0
    n_points: int = 41
    mode: Literal["rigid", "relaxed_normal"] = "rigid"


class IdealStrengthProtocol(_ProtocolBase):
    kind: Literal["ideal_strength"] = "ideal_strength"
    loading: Literal["tension", "shear"] = "tension"
    direction: tuple[int, int, int] = (0, 0, 1)
    shear_plane: Optional[tuple[int, int, int]] = None
    lateral: Literal["uniaxial_stress", "uniaxial_strain"] = "uniaxial_stress"
    max_strain: float = 0.4
    strain_step: float = 0.01
    relax: RelaxSettings = Field(default_factory=lambda: RelaxSettings(fmax_eV_per_A=1e-4))


class MDSettings(StrictModel):
    ensemble: Literal["nve", "nvt_nose_hoover", "nvt_langevin"] = "nve"
    timestep_fs: float = 1.0
    n_steps: int = 100
    temperature_K: float = 300.0
    seed: int = 0


class AutodiffAcceptanceProtocol(_ProtocolBase):
    """Differentiate a short trajectory's final observable and compare to FD."""

    kind: Literal["autodiff_acceptance"] = "autodiff_acceptance"
    md: MDSettings = Field(default_factory=MDSettings)
    observable: Literal["final_potential_energy", "final_virial_stress_zz", "mean_potential_energy"] = (
        "final_potential_energy"
    )
    parameters: list[str] = Field(default_factory=lambda: ["applied_strain_zz", "potential:A_scale"])
    fd_step: float = 1.0e-5
    rel_tolerance: float = 1.0e-5


# --------------------------------------------------------------------------
# Indentation
# --------------------------------------------------------------------------
BERKOVICH_FACE_ANGLE_DEG = 65.27  # modified Berkovich: projected area 24.494 h^2 (= Vickers at equal depth)
# Wall stiffness floor: K = 10 eV/A^3 biases the depth by 7 % at 8 A (G09); K = 100 with the in-run
# penetration correction h_c = h - pen_fw is the recipe default, K = 1000 a one-depth check.
MIN_WALL_STIFFNESS_EV_PER_A3 = 100.0
# Cell-size tiers in units of the equivalent contact radius a at h_max: (L/a, t_el/a), t_el = thickness above
# the fixed layer.  Continuum elastic bias about +3 % (production), +1-4 % (pilot), +11-25 % (scoping; an
# upper bound on H only).
CELL_TIERS: tuple[tuple[str, float, float], ...] = (("production", 8.0, 10.0), ("pilot", 6.0, 8.0),
                                                    ("scoping", 4.0, 4.0))


class IndenterSpec(StrictModel):
    """A rigid indenter (or an atomistic tip).  Angles in degrees, lengths in angstrom.

    ``rigid_sdf``: frictionless, purely repulsive wall on the exact signed-distance function of the
    (Minkowski-rounded) tip, U = (K/3) sum_i max(0, -sdf_i)^3 -- the same functional in JAX and LAMMPS.
    ``atomistic``: a tip made of atoms (e.g. rigid diamond with a WCA wall), described in `atomistic_tip`.
    """

    shape: Literal["berkovich", "sphere", "cone"] = "berkovich"
    face_angle_deg: float = BERKOVICH_FACE_ANGLE_DEG  # Berkovich centre-line-to-face angle
    cone_half_angle_deg: float = 70.296  # cone only; 70.296 deg = area-equivalent cone of the 65.27 deg Berkovich
    tip_radius_A: float = 10.0  # rounding radius (Berkovich/cone) or the sphere radius; 0 = sharp
    azimuth_deg: float = 0.0  # face-1 normal from lab +x (lab x = [110] on the harness (001) slab)
    height_cap_A: float = 60.0  # the tip acts up to this height above its lowest point
    stiffness_eV_per_A3: float = 100.0  # wall stiffness K (rigid_sdf only)
    tip_model: Literal["rigid_sdf", "atomistic"] = "rigid_sdf"
    atomistic_tip: str = ""  # atomistic tips: material, height, tip-sample interaction

    @model_validator(mode="after")
    def _check(self) -> "IndenterSpec":
        for name in ("face_angle_deg", "cone_half_angle_deg"):
            if not 0.0 < getattr(self, name) < 90.0:
                raise ValueError(f"indenter.{name} must lie in (0, 90)")
        if self.tip_radius_A < 0.0 or (self.shape == "sphere" and self.tip_radius_A <= 0.0):
            raise ValueError("indenter.tip_radius_A must be >= 0 (> 0 for a sphere)")
        if not self.height_cap_A > 0.0:
            raise ValueError("indenter.height_cap_A must be > 0")
        if self.tip_model == "rigid_sdf" and not self.stiffness_eV_per_A3 >= MIN_WALL_STIFFNESS_EV_PER_A3:
            raise ValueError(f"indenter.stiffness_eV_per_A3 must be >= {MIN_WALL_STIFFNESS_EV_PER_A3} (K = 10 biases the "
                             "depth by 7 %; use K = 100 with the penetration correction, K = 1000 as a check)")
        if self.tip_model == "atomistic" and not self.atomistic_tip.strip():
            raise ValueError("an atomistic tip must be described in indenter.atomistic_tip")
        return self

    def half_angle_deg(self) -> Optional[float]:
        """Face-to-axis angle (Berkovich) or cone half angle; None for a sphere."""
        return {"berkovich": self.face_angle_deg, "cone": self.cone_half_angle_deg}.get(self.shape)

    def geometry_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of ``physics.indenter.IndenterGeometry`` for this tip (rigid_sdf)."""
        return dict(shape=self.shape, half_angle_deg=self.half_angle_deg(), tip_radius_A=self.tip_radius_A,
                    azimuth_deg=self.azimuth_deg, height_cap_A=self.height_cap_A, form="cubic")

    def projected_area_constant(self) -> Optional[float]:
        """C0 of the ideal sharp tip, A = C0 h^2 (24.494 for 65.27 deg); None for a sphere."""
        if self.shape == "berkovich":
            return 3.0 * math.sqrt(3.0) * math.tan(math.radians(self.face_angle_deg)) ** 2
        if self.shape == "cone":
            return math.pi * math.tan(math.radians(self.cone_half_angle_deg)) ** 2
        return None

    def equivalent_contact_radius_A(self, depth_A: float) -> float:
        """Radius of the circle with the ideal projected contact area at `depth_A` (2.7923 h, Berkovich)."""
        c0 = self.projected_area_constant()
        if c0 is not None:
            return math.sqrt(c0 / math.pi) * depth_A
        r = self.tip_radius_A
        return math.sqrt(max(2.0 * r * depth_A - depth_A**2, 0.0)) if depth_A < r else r

    def lateral_extent_A(self, depth_A: float) -> float:
        """Upper bound of the indenter's lateral reach at `depth_A` below its lowest point.

        Berkovich: the contact corners sit at 2 tan(face) h = 4.34 h, plus the rounding radius; cone:
        tan(half angle) h + R; sphere: the chord radius (at most R).
        """
        r = self.tip_radius_A
        if self.shape == "berkovich":
            return 2.0 * math.tan(math.radians(self.face_angle_deg)) * depth_A + r
        if self.shape == "cone":
            return math.tan(math.radians(self.cone_half_angle_deg)) * depth_A + r
        return self.equivalent_contact_radius_A(depth_A)


class IndentationProtocol(_ProtocolBase):
    """Displacement-controlled indentation of a slab: MD at constant speed or quasi-static 0 K steps.

    The same protocol drives the JAX harness (verified reference for small cells, record building) and
    the LAMMPS deck generator (production cells); both report through the common schema
    (``IndentationCurve`` + ``*_MD`` records, capped status).  Defaults follow recipe_v0 (synthesis
    protocol items 3-12): K = 100 eV/A^3, dt = 1 fs, 50 m/s, 300 K, 10 ps hold, CSVR boundary layer.
    Geometry: lateral periodic cell, free top surface `plane_miller`/`top_termination`, fixed bottom
    layer, thermostatted layer above it, NVE core (MD).
    """

    kind: Literal["indentation"] = "indentation"
    # --- sample
    plane_miller: tuple[int, int, int] = (0, 0, 1)
    top_termination: str = "C"  # termination label of the indented surface
    lateral_size_A: tuple[float, float]
    thickness_A: float  # total slab thickness including the fixed and thermostat layers
    vacuum_A: float = 30.0  # requested vacuum above the top layer; both engines raise it to height_cap_A + 2 (cutoff
    # + skin) and (MD) to height_cap_A + max(5, initial_gap_A) + 1 A (structures.indentation.protocol_vacuum_A)
    lattice_parameter_A: Optional[float] = None  # None = the model's 0 K a0; give a(T) for finite-T runs
    fixed_thickness_A: float = 10.0  # bottom layer held fixed (>= the potential cutoff)
    thermostat_thickness_A: float = 10.0  # thermostatted layer above the fixed one (MD)
    thermostat: Literal["csvr", "langevin", "none"] = "csvr"
    thermostat_tau_ps: float = 0.1
    # --- indenter and loading
    indenter: IndenterSpec = Field(default_factory=IndenterSpec)
    mode: Literal["quasistatic", "md"] = "md"
    max_depth_A: float  # nominal tip displacement below the surface at the end of loading
    initial_gap_A: float = 3.0  # tip-surface gap before loading
    speed_m_per_s: float = 50.0  # MD loading and unloading speed (1 m/s = 0.01 angstrom/ps)
    depth_step_A: float = 0.25  # quasi-static depth increment
    hold_ps: float = 10.0  # MD hold at max depth
    unload: bool = True
    relax_after_unload_ps: float = 10.0
    # --- MD / relaxation numerics
    temperature_K: float = 300.0  # MD only (quasi-static runs are 0 K)
    timestep_fs: float = 1.0
    equilibration_ps: float = 5.0
    seed: int = 0
    relax: RelaxSettings = Field(default_factory=lambda: RelaxSettings(fmax_eV_per_A=1e-4))
    skin_A: float = 1.0
    neighbor_method: Literal["auto", "all_pairs", "cell_list"] = "auto"
    # --- analysis and output
    area_estimators: list[AreaEstimator] = Field(
        default_factory=lambda: ["dilated_hull", "closed_discs", "rounded_area_function", "nominal_ideal",
                                 "triangle_fit"])
    primary_area_estimator: AreaEstimator = "dilated_hull"
    block_average_ps: float = 0.5  # load block averages (>= 5 autocorrelation times)
    log_every_steps: int = 50
    frame_every_ps: Optional[float] = 1.0  # animation frames (<= 1 ps gives >= 140 frames); None = none
    frame_region: Literal["slice", "all"] = "slice"
    slice_half_width_A: float = 2.0  # |y - y_tip| <= this, plus the tracked top layer
    full_dump_every_A: Optional[float] = 2.5  # full snapshots / restarts every this much depth

    @model_validator(mode="after")
    def _check(self) -> "IndentationProtocol":
        lx, ly = self.lateral_size_A
        t, h = self.thickness_A, self.max_depth_A
        if not (lx > 0 and ly > 0 and t > 0 and self.vacuum_A > 0):
            raise ValueError("lateral_size_A, thickness_A and vacuum_A must be positive")
        if not 0.0 < h < t / 5.0:
            raise ValueError(f"max_depth_A must lie in (0, thickness_A / 5) = (0, {t / 5.0:g})")
        if self.fixed_thickness_A < 0 or self.thermostat_thickness_A < 0 or \
                self.fixed_thickness_A + self.thermostat_thickness_A >= t / 2.0:
            raise ValueError("fixed + thermostat layers must be non-negative and thinner than half the slab")
        ext = self.indenter.lateral_extent_A(h)
        if not ext < min(lx, ly) / 2.0:
            raise ValueError(f"the indenter reaches {ext:.1f} A from its axis at max_depth_A (corners at "
                             f"{2 * math.tan(math.radians(self.indenter.face_angle_deg)):.2f} h for a Berkovich); "
                             f"it must stay below min(lateral_size_A)/2 = {min(lx, ly) / 2.0:g} A")
        if self.initial_gap_A < 0:
            raise ValueError("initial_gap_A must be >= 0")
        if not self.indenter.height_cap_A >= 1.25 * h:
            raise ValueError(f"indenter.height_cap_A = {self.indenter.height_cap_A:g} A must be >= 1.25 max_depth_A "
                             "(room for pile-up above the original surface)")
        if self.mode == "md":
            if not self.speed_m_per_s > 0:
                raise ValueError("speed_m_per_s must be > 0")
            if not 0.0 < self.timestep_fs <= 2.0:
                raise ValueError("timestep_fs must lie in (0, 2]")
            if self.temperature_K < 0 or self.hold_ps < 0 or self.equilibration_ps < 0 or \
                    self.relax_after_unload_ps < 0 or self.thermostat_tau_ps <= 0:
                raise ValueError("temperature, hold, equilibration and relaxation times must be >= 0 "
                                 "(thermostat_tau_ps > 0)")
        elif not 0.0 < self.depth_step_A <= h:
            raise ValueError("depth_step_A must lie in (0, max_depth_A]")
        if self.primary_area_estimator not in self.area_estimators:
            raise ValueError("primary_area_estimator must be one of area_estimators")
        if self.frame_every_ps is not None and not self.frame_every_ps > 0:
            raise ValueError("frame_every_ps must be > 0 (or None)")
        if self.full_dump_every_A is not None and not self.full_dump_every_A > 0:
            raise ValueError("full_dump_every_A must be > 0 (or None)")
        if not (self.slice_half_width_A > 0 and self.block_average_ps > 0 and self.log_every_steps >= 1
                and self.skin_A >= 0):
            raise ValueError("slice_half_width_A, block_average_ps > 0, log_every_steps >= 1, skin_A >= 0")
        if self.lattice_parameter_A is not None and not self.lattice_parameter_A > 0:
            raise ValueError("lattice_parameter_A must be > 0")
        return self

    # ---- derived quantities used by runners, deck generators and pre-run checks
    def equivalent_contact_radius_A(self) -> float:
        """Equivalent contact radius a at max depth (the unit of the cell-size tiers)."""
        return self.indenter.equivalent_contact_radius_A(self.max_depth_A)

    def cell_ratios(self) -> tuple[float, float]:
        """(L/a, t_el/a) with L = min lateral size and t_el = thickness above the fixed layer."""
        a = self.equivalent_contact_radius_A()
        return min(self.lateral_size_A) / a, (self.thickness_A - self.fixed_thickness_A) / a

    def size_tier(self, rtol: float = 5.0e-3) -> str:
        """'production' | 'pilot' | 'scoping' | 'below_scoping' from :data:`CELL_TIERS` (ratios within
        `rtol` of a tier boundary count as reaching it: cells are built from whole unit cells)."""
        lr, tr = self.cell_ratios()
        for name, l_min, t_min in CELL_TIERS:
            if lr >= l_min * (1.0 - rtol) and tr >= t_min * (1.0 - rtol):
                return name
        return "below_scoping"

    def speed_A_per_ps(self) -> float:
        return self.speed_m_per_s * 1.0e-2

    def strain_rate_per_s(self) -> Optional[float]:
        """Nominal indentation strain rate v / h_max (MD), None for quasi-static runs."""
        return self.speed_m_per_s / (self.max_depth_A * 1.0e-10) if self.mode == "md" else None

    def estimated_atoms(self, number_density_per_A3: float) -> int:
        lx, ly = self.lateral_size_A
        return int(round(number_density_per_A3 * lx * ly * self.thickness_A))

    def contact_conditions(self, **overrides: Any) -> ContactConditions:
        """Record-level ContactConditions of this protocol (``overrides``: e.g. depth, area_estimator,
        depth_zero_rule, azimuth_reference, surface_termination, boundary)."""
        ind = self.indenter
        fields: dict[str, Any] = dict(
            indenter_shape=ind.shape,
            tip_model=ind.tip_model,
            face_angle=q(ind.face_angle_deg, Unit.DEGREE) if ind.shape == "berkovich" else None,
            cone_half_angle=q(ind.cone_half_angle_deg, Unit.DEGREE) if ind.shape == "cone" else None,
            tip_radius=q(ind.tip_radius_A, Unit.ANGSTROM),
            azimuth=q(ind.azimuth_deg, Unit.DEGREE),
            wall_stiffness=q(ind.stiffness_eV_per_A3, Unit.EV_PER_ANGSTROM3) if ind.tip_model == "rigid_sdf" else None,
            tip_description=ind.atomistic_tip,
            indentation_speed=q(self.speed_m_per_s, Unit.METER_PER_SECOND) if self.mode == "md" else None,
            max_depth=q(self.max_depth_A, Unit.ANGSTROM),
            surface_termination=f"{self.plane_miller} {self.top_termination}-terminated",
            lateral_size=q(list(self.lateral_size_A), Unit.ANGSTROM),
            thickness=q(self.thickness_A, Unit.ANGSTROM),
            boundary=(f"bottom {self.fixed_thickness_A:g} A fixed; "
                      + (f"next {self.thermostat_thickness_A:g} A {self.thermostat} {self.temperature_K:g} K "
                         f"(tau {self.thermostat_tau_ps:g} ps); NVE above; " if self.mode == "md" else "")
                      + "lateral periodic; free top surface"),
        )
        fields.update(overrides)
        return ContactConditions(**fields)

    def conditions(self, **contact_overrides: Any) -> Conditions:
        """Record-level Conditions (temperature, loading, strain rate v/h_max, contact)."""
        md = self.mode == "md"
        return Conditions(
            temperature=q(self.temperature_K if md else 0.0, Unit.KELVIN),
            loading="dynamic" if md else "quasistatic",
            strain_rate=q(self.strain_rate_per_s(), Unit.PER_SECOND) if md else None,
            stress_state="contact loading by a rigid indenter" if self.indenter.tip_model == "rigid_sdf"
            else "contact loading by an atomistic tip",
            contact=self.contact_conditions(**contact_overrides),
        )


Protocol = Annotated[
    Union[
        BulkRelaxProtocol,
        ElasticTensorProtocol,
        CleavageProtocol,
        TangentialSlipProtocol,
        IdealStrengthProtocol,
        AutodiffAcceptanceProtocol,
        IndentationProtocol,
    ],
    Field(discriminator="kind"),
]
