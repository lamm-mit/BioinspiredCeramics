"""Bonded direct (Class I) restoration mechanics: polymerisation shrinkage + occlusal load.

A small, self-contained 2D plane-strain linear-elastic finite-element solver (bilinear Q4
elements on a graded tensor-product grid) with an isotropic eigenstrain field, used to ask:
does shrinkage plus biting load debond a hand-placed, light-cured composite from
enamel/dentin, or crack it?

Units: lengths in mm, moduli/stresses in MPa (E given in GPa at the API and converted),
forces in N; per-unit-thickness quantities in N/mm.

Physics / modelling statements (all deliberate, all reported):
- 2D PLANE STRAIN (eps_zz = 0): the cavity is treated as a long groove; the out-of-plane
  constraint adds sigma_zz and raises the in-plane effective eigenstrain to (1+nu) eps*.
- LINEAR ELASTIC, small strain; all phases isotropic.
- Shrinkage = isotropic eigenstrain eps* = -f_post_gel * (vol shrinkage)/3 applied
  INSTANTANEOUSLY to a fully-stiff composite (upper bound: no pre-gel flow except through
  the post-gel fraction f_post_gel, which is an ASSUMED parameter; stresses scale linearly).
- Adhesive = continuum band of finite thickness (default 15 um) meshed with >= 3 elements.
- Interface tractions are read from the adhesive element row adjacent to the tooth
  (traction continuity), at element centres.
- Re-entrant bimaterial corners (wall-floor line angle, occlusal margin) are stress
  SINGULAR in linear elasticity: their peak values grow without bound under refinement and
  are reported as such; strength comparisons use (a) values away from corners (mesh
  converged) and (b) a point-stress value at a stated characteristic distance.
- Debond: a sequential-linear (event-by-event) strength-based knock-down of adhesive
  elements (quadratic normal/shear criterion); brittle, no fracture energy -> mesh
  dependent near singular corners; indicative only.

The dense linear algebra is done with scipy.sparse (direct LU): the problem is small and a
direct solve is far more robust for the 30x stiffness contrast than an iterative JAX solve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# --------------------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IsoMaterial:
    name: str
    E_gpa: float
    nu: float
    source: str = ""

    @property
    def E(self) -> float:  # MPa
        return self.E_gpa * 1e3

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self) -> float:
        return self.E / (2 * (1 + self.nu))


# Tooth-tissue moduli: RESEARCH_BRIEF.md sec 2.5 ([S51] Zhang 2014; [S52] Kinney 2003) and the
# moduli used in the published CZM study [B54] Tseng 2025 (enamel 84.1, dentin 18.6 GPa).
# Poisson ratios are ASSUMED (0.30/0.31), not verified in the brief.
ENAMEL = IsoMaterial("enamel", 84.1, 0.30, "E: [B54] Tseng 2025 (within 70-90 GPa of [S51]); nu assumed")
DENTIN = IsoMaterial("dentin", 18.6, 0.31, "E: [B54] Tseng 2025 (19-25 GPa [S51,S52]); nu assumed")
ADHESIVE = IsoMaterial("adhesive", 3.0, 0.30, "E: assumed within 2-5 GPa (brief); nu assumed")


@dataclass(frozen=True)
class DirectComposite:
    name: str
    E_gpa: float
    nu: float
    vol_shrinkage_pct: float
    filler_vol_pct: float
    note: str = ""

    def material(self) -> IsoMaterial:
        return IsoMaterial(self.name, self.E_gpa, self.nu, self.note)

    def eigenstrain(self, f_post_gel: float = 1.0) -> float:
        """Isotropic linear eigenstrain (negative = contraction)."""
        return -f_post_gel * self.vol_shrinkage_pct / 100.0 / 3.0


# Two bracketing cases (no DIRECT_COMPOSITE.md available at build time): the classic
# modulus / shrinkage trade-off -- more filler raises E and lowers shrinkage.
CONVENTIONAL_60 = DirectComposite("conventional_60vol", 12.0, 0.30, 2.5, 60.0,
                                  "generic 60 vol% hybrid; E and shrinkage are brief-supplied bracketing values")
HIGHFILL_70 = DirectComposite("highfill_70vol", 18.0, 0.30, 1.6, 70.0,
                              "generic 70 vol% high-fill; E and shrinkage are brief-supplied bracketing values")

# Aged tooth-side bond strengths (MPa): docs/dental/BONDING.md (SBU-Plus-type protocol,
# aged): enamel 29.8 [18.4-40.0], superficial dentin 24.0 [16.1-30.7], deep 20.5 [14.3-25.7].
AGED_BOND_MPA = {"enamel": (29.8, 18.4, 40.0), "dentin": (24.0, 16.1, 30.7), "deep_dentin": (20.5, 14.3, 25.7)}
# Published CZM strengths [B54] Tseng 2025: enamel-composite 34.5 MPa, dentin-composite 17.0 MPa.
CZM_SIGMA_C_MPA = {"enamel": 34.5, "dentin": 17.0}
ISO4049_FLEXURAL_MIN_MPA = 80.0  # ISO 4049:2019 Table 1, Type 1 Class 1/Class 2 Group 1 (data/dental/targets.json)

# --------------------------------------------------------------------------------------
# mesh
# --------------------------------------------------------------------------------------


def graded_axis(breaks: Sequence[float], h_at: Sequence[float], h_max: float, growth: float = 0.15) -> np.ndarray:
    """1D node coordinates through all `breaks`; local size h(s) = min(h_b + growth*|s-b|, h_max)."""
    breaks = np.asarray(breaks, float)
    h_at = np.asarray(h_at, float)

    def h(s):
        return float(min(h_max, np.min(h_at + growth * np.abs(s - breaks))))

    pts = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        seg = [a]
        while seg[-1] + h(seg[-1]) < b - 1e-12:
            seg.append(seg[-1] + h(seg[-1]))
        if len(seg) > 1 and (b - seg[-1]) < 0.5 * h(seg[-1]):
            seg[-1] = b
        else:
            seg.append(b)
        pts.extend(seg[1:])
    return np.asarray(pts)


@dataclass
class Mesh:
    x: np.ndarray  # (nx+1,)
    y: np.ndarray  # (ny+1,)
    mat: np.ndarray  # (ny, nx) int material id
    materials: List[IsoMaterial]
    eig: np.ndarray  # (ny, nx) isotropic eigenstrain
    tags: Dict[str, np.ndarray] = field(default_factory=dict)  # (ny,nx) bool masks

    @property
    def nx(self):
        return len(self.x) - 1

    @property
    def ny(self):
        return len(self.y) - 1

    @property
    def xc(self):
        return 0.5 * (self.x[1:] + self.x[:-1])

    @property
    def yc(self):
        return 0.5 * (self.y[1:] + self.y[:-1])

    def node(self, i, j):  # i: x index, j: y index
        return j * (self.nx + 1) + i

    @property
    def n_nodes(self):
        return (self.nx + 1) * (self.ny + 1)


def rect_mesh(x: np.ndarray, y: np.ndarray, material: IsoMaterial) -> Mesh:
    ny, nx = len(y) - 1, len(x) - 1
    return Mesh(np.asarray(x, float), np.asarray(y, float), np.zeros((ny, nx), int), [material], np.zeros((ny, nx)))


# --------------------------------------------------------------------------------------
# Q4 plane-strain FE with eigenstrain
# --------------------------------------------------------------------------------------
_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)


def _B_rect(dx, dy, xi, eta):
    """Strain-displacement matrices (ne,3,8) of axis-aligned rectangles at (xi,eta)."""
    # node order: (-,-), (+,-), (+,+), (-,+)
    sx = np.array([-1, 1, 1, -1.0])
    sy = np.array([-1, -1, 1, 1.0])
    dNdxi = 0.25 * sx * (1 + sy * eta)
    dNdeta = 0.25 * sy * (1 + sx * xi)
    dNdx = dNdxi[None, :] * (2.0 / dx)[:, None]
    dNdy = dNdeta[None, :] * (2.0 / dy)[:, None]
    ne = len(dx)
    B = np.zeros((ne, 3, 8))
    B[:, 0, 0::2] = dNdx
    B[:, 1, 1::2] = dNdy
    B[:, 2, 0::2] = dNdy
    B[:, 2, 1::2] = dNdx
    return B


def _D(lam, mu):
    ne = len(lam)
    D = np.zeros((ne, 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = lam + 2 * mu
    D[:, 0, 1] = D[:, 1, 0] = lam
    D[:, 2, 2] = mu
    return D


def _elem_arrays(mesh: Mesh, stiff_scale: Optional[np.ndarray] = None):
    dx = np.diff(mesh.x)
    dy = np.diff(mesh.y)
    DX, DY = np.meshgrid(dx, dy)  # (ny,nx)
    lam = np.array([m.lam for m in mesh.materials])[mesh.mat]
    mu = np.array([m.mu for m in mesh.materials])[mesh.mat]
    if stiff_scale is not None:
        sc = np.reshape(stiff_scale, lam.shape)
        lam, mu = lam * sc, mu * sc
    return DX.ravel(), DY.ravel(), lam.ravel(), mu.ravel()


def _connectivity(mesh: Mesh) -> np.ndarray:
    I, J = np.meshgrid(np.arange(mesh.nx), np.arange(mesh.ny))
    I, J = I.ravel(), J.ravel()
    n = np.stack([mesh.node(I, J), mesh.node(I + 1, J), mesh.node(I + 1, J + 1), mesh.node(I, J + 1)], axis=1)
    dofs = np.empty((len(I), 8), int)
    dofs[:, 0::2] = 2 * n
    dofs[:, 1::2] = 2 * n + 1
    return dofs


@dataclass
class FESolution:
    mesh: Mesh
    u: np.ndarray  # (2*n_nodes,)
    sxx: np.ndarray  # (ny,nx) element-centre stresses, MPa
    syy: np.ndarray
    sxy: np.ndarray
    szz: np.ndarray

    @property
    def s1(self):  # max in-plane principal stress
        c = 0.5 * (self.sxx + self.syy)
        r = np.sqrt((0.5 * (self.sxx - self.syy)) ** 2 + self.sxy ** 2)
        return c + r

    def ux(self):
        return self.u[0::2].reshape(self.mesh.ny + 1, self.mesh.nx + 1)

    def uy(self):
        return self.u[1::2].reshape(self.mesh.ny + 1, self.mesh.nx + 1)


class PlaneStrainFE:
    """Assemble once per stiffness field; solve many (eigenstrain, traction) load cases."""

    def __init__(self, mesh: Mesh, fixed: Dict[int, float], stiff_scale: Optional[np.ndarray] = None,
                 plane: str = "strain"):
        """plane='strain' (eps_zz = 0, default) or 'stress' (sigma_zz = 0: lambda -> 2 lam mu/(lam+2mu))."""
        if plane not in ("strain", "stress"):
            raise ValueError(plane)
        self.plane = plane
        self.mesh = mesh
        self.dofs = _connectivity(mesh)
        self.dx, self.dy, self.lam, self.mu = _elem_arrays(mesh, stiff_scale)
        if plane == "stress":
            self.lam = 2 * self.lam * self.mu / (self.lam + 2 * self.mu)
        # isotropic eigenstress coefficient: sigma0 = c0 * eps* (in-plane components)
        # plane strain: 3 lam + 2 mu (eps_zz^mech = -eps*); plane stress: 2 lam' + 2 mu = E/(1-nu)
        self._c0 = (3 * self.lam + 2 * self.mu) if plane == "strain" else (2 * self.lam + 2 * self.mu)
        D = _D(self.lam, self.mu)
        ne = len(self.dx)
        Ke = np.zeros((ne, 8, 8))
        self._Bg = []
        for xi in _GP:
            for eta in _GP:
                B = _B_rect(self.dx, self.dy, xi, eta)
                self._Bg.append(B)
                w = (self.dx * self.dy / 4.0)[:, None, None]
                Ke += w * np.einsum("eij,ejk,ekl->eil", B.transpose(0, 2, 1), D, B)
        self.D = D
        ndof = 2 * mesh.n_nodes
        rows = np.repeat(self.dofs, 8, axis=1).ravel()
        cols = np.tile(self.dofs, (1, 8)).ravel()
        K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(ndof, ndof)).tocsr()
        self.fixed_dofs = np.array(sorted(fixed), int)
        self.fixed_vals = np.array([fixed[d] for d in self.fixed_dofs])
        free = np.ones(ndof, bool)
        free[self.fixed_dofs] = False
        self.free = np.nonzero(free)[0]
        self.K = K
        self._Kfc = K[self.free][:, self.fixed_dofs]
        self._lu = spla.splu(K[self.free][:, self.free].tocsc())

    def eigen_force(self, eig: np.ndarray) -> np.ndarray:
        """Equivalent nodal force of isotropic eigenstrain: f = sum_e int B^T sigma0, sigma0 = (3lam+2mu) e* [1,1,0]."""
        e = eig.ravel()
        s0 = self._c0 * e
        sig0 = np.stack([s0, s0, np.zeros_like(s0)], axis=1)
        fe = np.zeros((len(e), 8))
        w = (self.dx * self.dy / 4.0)[:, None]
        for B in self._Bg:
            fe += w * np.einsum("eji,ej->ei", B, sig0)
        f = np.zeros(2 * self.mesh.n_nodes)
        np.add.at(f, self.dofs.ravel(), fe.ravel())
        return f

    def solve(self, eig: Optional[np.ndarray] = None, f_ext: Optional[np.ndarray] = None) -> FESolution:
        m = self.mesh
        eig = np.zeros((m.ny, m.nx)) if eig is None else eig
        f = self.eigen_force(eig)
        if f_ext is not None:
            f = f + f_ext
        u = np.zeros(2 * m.n_nodes)
        u[self.fixed_dofs] = self.fixed_vals
        rhs = f[self.free] - self._Kfc @ self.fixed_vals
        u[self.free] = self._lu.solve(rhs)
        # element-centre stresses
        B0 = _B_rect(self.dx, self.dy, 0.0, 0.0)
        eps = np.einsum("eij,ej->ei", B0, u[self.dofs])
        e = eig.ravel()
        s0 = self._c0 * e
        sig = np.einsum("eij,ej->ei", self.D, eps)
        sxx = sig[:, 0] - s0
        syy = sig[:, 1] - s0
        sxy = sig[:, 2]
        if self.plane == "strain":
            szz = self.lam * (eps[:, 0] + eps[:, 1]) - s0
        else:
            szz = np.zeros_like(sxx)
        shp = (m.ny, m.nx)
        return FESolution(m, u, sxx.reshape(shp), syy.reshape(shp), sxy.reshape(shp), szz.reshape(shp))


def top_pressure(mesh: Mesh, x0: float, x1: float, p: float) -> np.ndarray:
    """Consistent nodal forces of a uniform downward pressure p (MPa) on the top edge over [x0,x1]."""
    f = np.zeros(2 * mesh.n_nodes)
    j = mesh.ny
    for i in range(mesh.nx):
        a, b = mesh.x[i], mesh.x[i + 1]
        lo, hi = max(a, x0), min(b, x1)
        if hi <= lo:
            continue
        # linear shape functions integrated over [lo,hi]
        L = b - a
        for node_i, xn in ((i, a), (i + 1, b)):
            # int N dx with N = 1 - |x - xn|/L over [lo,hi]
            g = lambda s: s - np.sign(s - xn) * (s - xn) ** 2 / (2 * L)
            f[2 * mesh.node(node_i, j) + 1] -= p * (g(hi) - g(lo))
    return f


# --------------------------------------------------------------------------------------
# analytic references
# --------------------------------------------------------------------------------------


def timoshenko_bimaterial_curvature(E1, E2, t1, t2, d_eps):
    """Timoshenko (1925) bimetal curvature for a free strip; layer 1 strain minus layer 2 = d_eps.
    Returns kappa (1/length), positive = concave towards layer 2 side as in the formula's convention
    (the side of the layer with the LARGER expansion is convex)."""
    m, n, h = t1 / t2, E1 / E2, t1 + t2
    return 6 * d_eps * (1 + m) ** 2 / (h * (3 * (1 + m) ** 2 + (1 + m * n) * (m ** 2 + 1 / (m * n))))


def plane_strain_constrained_stress(E, nu, eps_star, mode: str) -> float:
    """Exact uniform stresses for isotropic eigenstrain eps* in plane strain (eps_zz = 0).
    mode 'full': ux=uy=0 -> sigma_xx = sigma_yy = -E eps*/(1-2nu).
    mode 'uniaxial': ux=0, sigma_yy=0 -> sigma_xx = -E eps*/(1-nu)."""
    if mode == "full":
        return -E * eps_star / (1 - 2 * nu)
    if mode == "uniaxial":
        return -E * eps_star / (1 - nu)
    raise ValueError(mode)


# --------------------------------------------------------------------------------------
# verification cases
# --------------------------------------------------------------------------------------


def _edge_nodes(mesh: Mesh, side: str) -> np.ndarray:
    I, J = np.meshgrid(np.arange(mesh.nx + 1), np.arange(mesh.ny + 1))
    sel = {"left": I == 0, "right": I == mesh.nx, "bottom": J == 0, "top": J == mesh.ny}[side]
    return mesh.node(I[sel], J[sel])


def verify_free_eigenstrain(eps_star=-0.005, n=6) -> dict:
    mat = IsoMaterial("m", 12.0, 0.3)
    mesh = rect_mesh(np.linspace(0, 1, n + 1), np.linspace(0, 0.7, n + 1), mat)
    mesh.eig[:] = eps_star
    fixed = {2 * mesh.node(0, 0): 0.0, 2 * mesh.node(0, 0) + 1: 0.0, 2 * mesh.node(mesh.nx, 0) + 1: 0.0}
    sol = PlaneStrainFE(mesh, fixed).solve(mesh.eig)
    inplane = float(max(np.abs(sol.sxx).max(), np.abs(sol.syy).max(), np.abs(sol.sxy).max()))
    szz_exact = -mat.E * eps_star
    ux_exact = (1 + mat.nu) * eps_star * mesh.x[-1]
    return {"max_inplane_stress_MPa": inplane, "szz_MPa": float(sol.szz.mean()), "szz_exact_MPa": szz_exact,
            "ux_right_mm": float(sol.ux()[0, -1]), "ux_right_exact_mm": ux_exact,
            "passed": bool(inplane < 1e-8 * mat.E and abs(sol.szz.mean() - szz_exact) < 1e-9 * mat.E
                           and abs(sol.ux()[0, -1] - ux_exact) < 1e-12)}


def verify_constrained(eps_star=-0.005, n=6) -> dict:
    mat = IsoMaterial("m", 12.0, 0.3)
    x, y = np.linspace(0, 1, n + 1), np.linspace(0, 0.5, n + 1)
    # full clamp
    mesh = rect_mesh(x, y, mat)
    mesh.eig[:] = eps_star
    fixed = {}
    for s in ("left", "right", "bottom", "top"):
        for nd in _edge_nodes(mesh, s):
            fixed[2 * nd] = 0.0
            fixed[2 * nd + 1] = 0.0
    sf = PlaneStrainFE(mesh, fixed).solve(mesh.eig)
    ex_full = plane_strain_constrained_stress(mat.E, mat.nu, eps_star, "full")
    # uniaxial: rollers left/right (ux=0), bottom roller (uy=0), top free
    mesh2 = rect_mesh(x, y, mat)
    mesh2.eig[:] = eps_star
    fixed2 = {}
    for nd in _edge_nodes(mesh2, "left"):
        fixed2[2 * nd] = 0.0
    for nd in _edge_nodes(mesh2, "right"):
        fixed2[2 * nd] = 0.0
    for nd in _edge_nodes(mesh2, "bottom"):
        fixed2[2 * nd + 1] = 0.0
    su = PlaneStrainFE(mesh2, fixed2).solve(mesh2.eig)
    ex_uni = plane_strain_constrained_stress(mat.E, mat.nu, eps_star, "uniaxial")
    err_full = float(max(np.abs(sf.sxx - ex_full).max(), np.abs(sf.syy - ex_full).max()) / abs(ex_full))
    err_uni = float(max(np.abs(su.sxx - ex_uni).max() / abs(ex_uni), np.abs(su.syy).max() / abs(ex_uni)))
    return {"full_sxx_MPa": float(sf.sxx.mean()), "full_exact_MPa": ex_full, "full_rel_err": err_full,
            "uniaxial_sxx_MPa": float(su.sxx.mean()), "uniaxial_exact_MPa": ex_uni, "uniaxial_rel_err": err_uni,
            "passed": bool(err_full < 1e-10 and err_uni < 1e-10)}


def verify_bimaterial_strip(L=40.0, t1=0.5, t2=0.5, E1=10.0, E2=20.0, nu=0.3, eps1=-0.004, nx=320, ny_layer=8) -> dict:
    """Free bilayer strip, eigenstrain in layer 1 only -> Timoshenko curvature (plane-strain form:
    E -> E/(1-nu^2), d_eps -> (1+nu) d_eps*, exact for equal nu)."""
    x = np.linspace(-L / 2, L / 2, nx + 1)
    y = np.concatenate([np.linspace(0, t1, ny_layer + 1), np.linspace(t1, t1 + t2, ny_layer + 1)[1:]])
    m1, m2 = IsoMaterial("l1", E1, nu), IsoMaterial("l2", E2, nu)
    mesh = Mesh(x, y, np.zeros((len(y) - 1, nx), int), [m1, m2], np.zeros((len(y) - 1, nx)))
    yc = mesh.yc
    mesh.mat[yc > t1, :] = 1
    mesh.eig[yc < t1, :] = eps1
    ic = nx // 2
    fixed = {2 * mesh.node(ic, 0): 0.0, 2 * mesh.node(ic, 0) + 1: 0.0, 2 * mesh.node(ic, mesh.ny): 0.0}
    sol = PlaneStrainFE(mesh, fixed).solve(mesh.eig)
    # fit uy(x) on the mid-thickness line over the central half: uy = c + kappa/2 x^2
    jm = np.argmin(np.abs(mesh.y - (t1 + t2) / 2))
    uy = sol.uy()[jm]
    sel = np.abs(x) < L / 4
    coef = np.polyfit(x[sel], uy[sel], 2)
    kappa_fe = 2 * coef[0]
    kappa_ex = timoshenko_bimaterial_curvature(E1 / (1 - nu ** 2), E2 / (1 - nu ** 2), t1, t2, (1 + nu) * eps1)
    # layer 1 (bottom) contracts -> bottom concave -> strip bends with ends UP?  contraction at
    # bottom makes bottom shorter -> centre of curvature below -> uy(x) ~ -kappa x^2/2 (ends down).
    rel = abs(abs(kappa_fe) - abs(kappa_ex)) / abs(kappa_ex)
    sign_ok = kappa_fe < 0  # shrinking bottom layer -> ends curl down (convex up)
    return {"kappa_fe_per_mm": float(kappa_fe), "kappa_timoshenko_per_mm": float(kappa_ex), "rel_err": float(rel),
            "sign_ok": bool(sign_ok), "passed": bool(rel < 0.02 and sign_ok)}


# --------------------------------------------------------------------------------------
# Class I cavity model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CavityGeometry:
    tooth_half_width: float = 4.0  # mm
    tooth_height: float = 5.0  # mm (bottom clamped = supporting tooth/root)
    enamel_thickness: float = 1.0  # mm occlusal enamel rim
    cavity_half_width: float = 2.0  # mm (4 mm wide)
    cavity_depth: float = 2.0  # mm
    adhesive_thickness: float = 0.015  # mm (15 um)
    out_of_plane_width: float = 4.0  # mm, used only to convert N -> MPa for the occlusal load (ASSUMED)

    @property
    def y_floor(self):
        return self.tooth_height - self.cavity_depth

    @property
    def y_dej(self):
        return self.tooth_height - self.enamel_thickness


MAT_DENTIN, MAT_ENAMEL, MAT_ADH, MAT_COMP = 0, 1, 2, 3


def cavity_mesh(geo: CavityGeometry, composite: DirectComposite, h_fine: Optional[float] = None,
                h_max: float = 0.25, growth: float = 0.15, extra_x_breaks: Iterable[float] = (-1.0, 1.0, 3.0)) -> Mesh:
    t = geo.adhesive_thickness
    hf = t / 3 if h_fine is None else h_fine
    W, a = geo.tooth_half_width, geo.cavity_half_width
    xb = sorted({-W, W, -a, -a + t, a - t, a, *extra_x_breaks})
    xh = [hf if min(abs(v - s) for s in (-a, -a + t, a - t, a)) < 1e-12 else h_max for v in xb]
    yb = [0.0, geo.y_floor, geo.y_floor + t, geo.y_dej, geo.tooth_height]
    yh = [h_max, hf, hf, 4 * hf, hf]
    x = graded_axis(xb, xh, h_max, growth)
    y = graded_axis(yb, yh, h_max, growth)
    mats = [DENTIN, ENAMEL, ADHESIVE, composite.material()]
    ny, nx = len(y) - 1, len(x) - 1
    mesh = Mesh(x, y, np.zeros((ny, nx), int), mats, np.zeros((ny, nx)))
    XC, YC = np.meshgrid(mesh.xc, mesh.yc)
    cav = (np.abs(XC) < a) & (YC > geo.y_floor)
    mesh.mat[YC > geo.y_dej] = MAT_ENAMEL
    adh = cav & ((np.abs(XC) > a - t) | (YC < geo.y_floor + t))
    comp = cav & ~adh
    mesh.mat[adh] = MAT_ADH
    mesh.mat[comp] = MAT_COMP
    mesh.tags = {"cavity": cav, "adhesive": adh, "composite": comp}
    return mesh


def clamp_bottom(mesh: Mesh) -> Dict[int, float]:
    fixed = {}
    for nd in _edge_nodes(mesh, "bottom"):
        fixed[2 * nd] = 0.0
        fixed[2 * nd + 1] = 0.0
    return fixed


def interface_path(mesh: Mesh, geo: CavityGeometry) -> dict:
    """Tooth-adjacent adhesive elements along: left wall (margin -> corner) then floor (corner -> centre).
    Returns s (arc length from the occlusal margin, mm), (j,i) indices, normal direction, substrate."""
    a = geo.cavity_half_width
    i_wall = int(np.searchsorted(mesh.x, -a + 1e-12))  # first element column inside x=-a
    j_floor = int(np.searchsorted(mesh.y, geo.y_floor + 1e-12))  # first element row above floor
    js_wall = [j for j in range(mesh.ny) if mesh.yc[j] > geo.y_floor][::-1]  # top -> bottom
    is_floor = [i for i in range(mesh.nx) if -a < mesh.xc[i] <= 0.0]
    s, idx, normal, subst = [], [], [], []
    for j in js_wall:
        s.append(geo.tooth_height - mesh.yc[j])
        idx.append((j, i_wall))
        normal.append("x")
        subst.append("enamel" if mesh.yc[j] > geo.y_dej else "dentin")
    for i in is_floor:
        if i == i_wall:
            continue
        s.append(geo.cavity_depth + (mesh.xc[i] + a))
        idx.append((j_floor, i))
        normal.append("y")
        subst.append("dentin")
    return {"s": np.array(s), "idx": idx, "normal": normal, "substrate": subst,
            "s_corner": geo.cavity_depth, "s_dej": geo.enamel_thickness}


def interface_tractions(sol: FESolution, path: dict) -> Tuple[np.ndarray, np.ndarray]:
    """(sigma_n, tau) along the path; sigma_n > 0 = tension pulling the restoration off the tooth.
    Left wall: tooth on the -x side, outward normal of the restoration is -x -> sigma_n = sigma_xx.
    Floor: tooth below -> sigma_n = sigma_yy."""
    sn, tau = [], []
    for (j, i), nrm in zip(path["idx"], path["normal"]):
        sn.append(sol.sxx[j, i] if nrm == "x" else sol.syy[j, i])
        tau.append(sol.sxy[j, i])
    return np.array(sn), np.array(tau)


def failure_index(sn, tau, strength, tau_ratio=1.0):
    """Quadratic interface criterion sqrt(<sn>^2/S^2 + tau^2/(tau_ratio*S)^2) (tau strength ASSUMED = S)."""
    S = np.asarray(strength, float)
    return np.sqrt(np.maximum(sn, 0) ** 2 / S ** 2 + tau ** 2 / (tau_ratio * S) ** 2)


def _value_at(s, v, s0):
    return float(np.interp(s0, s, v))


@dataclass
class CavityResult:
    composite: str
    load_N: float
    f_post_gel: float
    path_s: np.ndarray
    sn: np.ndarray
    tau: np.ndarray
    substrate: List[str]
    summary: dict


def run_cavity_case(composite: DirectComposite, geo: CavityGeometry = CavityGeometry(), *, f_post_gel=1.0,
                    loads_N=(0.0, 150.0), load_span=(-1.0, 1.0), h_fine=None, h_max=0.25,
                    char_distance=0.05, fe: Optional[PlaneStrainFE] = None, mesh: Optional[Mesh] = None) -> dict:
    """Shrinkage (+ occlusal pressure over load_span) for one composite. Linear -> fields superpose."""
    if mesh is None:
        mesh = cavity_mesh(geo, composite, h_fine=h_fine, h_max=h_max)
    if fe is None:
        fe = PlaneStrainFE(mesh, clamp_bottom(mesh))
    eig = np.where(mesh.tags["composite"], composite.eigenstrain(f_post_gel), 0.0)
    s_shr = fe.solve(eig)
    s_unit = fe.solve(None, top_pressure(mesh, load_span[0], load_span[1], 1.0))
    path = interface_path(mesh, geo)
    sn_s, tau_s = interface_tractions(s_shr, path)
    sn_u, tau_u = interface_tractions(s_unit, path)
    strength = np.array([AGED_BOND_MPA[k][0] for k in path["substrate"]])
    out = {"composite": composite.name, "E_gpa": composite.E_gpa, "vol_shrinkage_pct": composite.vol_shrinkage_pct,
           "eigenstrain": composite.eigenstrain(f_post_gel), "f_post_gel": f_post_gel, "n_elements": mesh.nx * mesh.ny,
           "h_fine_um": (geo.adhesive_thickness / 3 if h_fine is None else h_fine) * 1e3, "cases": []}
    s = path["s"]
    sc = path["s_corner"]
    wall = np.array(path["normal"]) == "x"
    comp = mesh.tags["composite"]
    enam = mesh.mat == MAT_ENAMEL
    for F in loads_N:
        p = F / ((load_span[1] - load_span[0]) * geo.out_of_plane_width)
        sn, tau = sn_s + p * sn_u, tau_s + p * tau_u
        fi = failure_index(sn, tau, strength)
        s1 = _combine_s1(s_shr, s_unit, p)
        away = (np.abs(s - sc) > 0.3) & (s > 0.3)  # >= 0.3 mm from both singular points
        k = int(np.argmax(fi))
        case = {
            "load_N": F, "pressure_MPa": p,
            "sn_mid_floor_MPa": float(sn[-1]), "tau_mid_floor_MPa": float(tau[-1]),
            "sn_mid_dentin_wall_MPa": _value_at(s[wall], sn[wall], 1.5),
            "sn_mid_enamel_wall_MPa": _value_at(s[wall], sn[wall], 0.5),
            "sn_peak_away_from_corners_MPa": float(sn[away].max()),
            "sn_at_corner_minus_d0_wall_MPa": _value_at(s, sn, sc - char_distance),
            "sn_at_corner_plus_d0_floor_MPa": _value_at(s, sn, sc + char_distance),
            "sn_at_margin_d0_MPa": _value_at(s, sn, char_distance),
            "sn_max_any_MPa": float(sn.max()), "tau_max_abs_MPa": float(np.abs(tau).max()),
            "failure_index_max": float(fi[k]), "failure_index_max_s_mm": float(s[k]),
            "failure_index_max_location": _loc_name(s[k], sc, path["s_dej"]),
            "failure_index_away_from_corners_max": float(fi[away].max()),
            "fraction_of_path_FI_gt_1": float(np.mean(fi > 1.0)),
            "composite_s1_max_MPa": float(s1[comp].max()),
            "composite_s1_max_away_from_interface_MPa": float(_interior_max(mesh, s1, comp, geo)),
            "enamel_s1_max_MPa": float(s1[enam].max()),
            "sn_profile": sn, "tau_profile": tau, "fi_profile": fi,
        }
        out["cases"].append(case)
    out["path_s"] = s
    out["substrate"] = path["substrate"]
    out["char_distance_mm"] = char_distance
    return out


def _combine_s1(a: FESolution, b: FESolution, p: float) -> np.ndarray:
    sxx, syy, sxy = a.sxx + p * b.sxx, a.syy + p * b.syy, a.sxy + p * b.sxy
    return 0.5 * (sxx + syy) + np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)


def _interior_max(mesh, s1, comp, geo, margin=0.1):
    XC, YC = np.meshgrid(mesh.xc, mesh.yc)
    inner = comp & (np.abs(XC) < geo.cavity_half_width - margin) & (YC > geo.y_floor + margin)
    return s1[inner].max()


def _loc_name(s, sc, sdej, tol=0.1):
    if s < tol:
        return "occlusal margin (enamel, free-edge corner)"
    if abs(s - sc) < tol:
        return "wall-floor line angle (re-entrant corner)"
    if abs(s - sdej) < tol:
        return "wall at DEJ"
    return "floor" if s > sc else ("enamel wall" if s < sdej else "dentin wall")


# --------------------------------------------------------------------------------------
# incremental vs bulk placement (linear superposition)
# --------------------------------------------------------------------------------------


def incremental_vs_bulk(composite: DirectComposite, geo: CavityGeometry = CavityGeometry(), n_increments: int = 2,
                        h_fine=None, h_max=0.25, f_post_gel=1.0, void_scale=1e-6) -> dict:
    """Horizontal increments cured bottom-up. Increment k shrinks while increments > k are absent
    (stiffness x void_scale, no eigenstrain); stresses from successive steps superpose (linear).
    Returns interface tractions for bulk (one step) and incremental placement."""
    mesh = cavity_mesh(geo, composite, h_fine=h_fine, h_max=h_max)
    comp = mesh.tags["composite"]
    _, YC = np.meshgrid(mesh.xc, mesh.yc)
    y0 = geo.y_floor + geo.adhesive_thickness
    edges = np.linspace(y0, geo.tooth_height, n_increments + 1)
    fixed = clamp_bottom(mesh)
    path = interface_path(mesh, geo)
    e = composite.eigenstrain(f_post_gel)
    bulk = PlaneStrainFE(mesh, fixed).solve(np.where(comp, e, 0.0))
    sn_b, tau_b = interface_tractions(bulk, path)
    sn_i, tau_i = np.zeros_like(sn_b), np.zeros_like(tau_b)
    for k in range(n_increments):
        layer = comp & (YC >= edges[k]) & (YC < edges[k + 1] + (1e-9 if k == n_increments - 1 else 0))
        absent = comp & (YC >= edges[k + 1])
        scale = np.where(absent, void_scale, 1.0).ravel()
        sol = PlaneStrainFE(mesh, fixed, stiff_scale=scale).solve(np.where(layer, e, 0.0))
        sn_k, tau_k = interface_tractions(sol, path)
        # tractions in absent-increment regions are ~0 automatically (void stiffness)
        sn_i += sn_k
        tau_i += tau_k
    strength = np.array([AGED_BOND_MPA[k][0] for k in path["substrate"]])
    s = path["s"]
    away = (np.abs(s - path["s_corner"]) > 0.3) & (s > 0.3)
    res = {}
    for name, sn, tau in (("bulk", sn_b, tau_b), ("incremental", sn_i, tau_i)):
        fi = failure_index(sn, tau, strength)
        res[name] = {"sn_mid_floor_MPa": float(sn[-1]), "sn_peak_away_MPa": float(sn[away].max()),
                     "fi_away_max": float(fi[away].max()), "fi_max": float(fi.max()),
                     "sn_mean_wall_MPa": float(np.mean(sn[np.array(path['normal']) == 'x'])),
                     "sn_profile": sn, "tau_profile": tau}
    res["path_s"] = s
    res["n_increments"] = n_increments
    res["c_factor_2d_bulk"] = (2 * geo.cavity_depth + 2 * geo.cavity_half_width) / (2 * geo.cavity_half_width)
    return res


# --------------------------------------------------------------------------------------
# sequential-linear debond (strength-based, brittle)
# --------------------------------------------------------------------------------------


def sequential_debond(composite: DirectComposite, geo: CavityGeometry = CavityGeometry(), load_N=150.0,
                      load_span=(-1.0, 1.0), strengths: Optional[Dict[str, float]] = None, h_fine=None, h_max=0.25,
                      f_post_gel=1.0, knock=1e-4, max_iter=60) -> dict:
    """Iteratively knock down (stiffness x knock) tooth-adjacent adhesive elements whose quadratic failure
    index > 1 under shrinkage + load, re-solving each time, until no new element fails.
    Default strengths: published CZM sigma_c [B54] (enamel 34.5, dentin 17.0 MPa)."""
    strengths = strengths or CZM_SIGMA_C_MPA
    mesh = cavity_mesh(geo, composite, h_fine=h_fine, h_max=h_max)
    fixed = clamp_bottom(mesh)
    path = interface_path(mesh, geo)
    S = np.array([strengths[k] for k in path["substrate"]])
    eig = np.where(mesh.tags["composite"], composite.eigenstrain(f_post_gel), 0.0)
    p = load_N / ((load_span[1] - load_span[0]) * geo.out_of_plane_width)
    fext = top_pressure(mesh, load_span[0], load_span[1], p) if load_N else None
    scale = np.ones((mesh.ny, mesh.nx))
    broken = np.zeros(len(path["idx"]), bool)
    history = []
    for it in range(max_iter):
        sol = PlaneStrainFE(mesh, fixed, stiff_scale=scale.ravel()).solve(eig, fext)
        sn, tau = interface_tractions(sol, path)
        fi = failure_index(sn, tau, S)
        fi[broken] = 0.0
        new = fi > 1.0
        history.append({"iter": it, "n_new": int(new.sum()), "fi_max_intact": float(fi.max())})
        if not new.any():
            break
        # break only the most critical few (event-by-event, avoids over-shooting)
        order = np.argsort(-fi)
        nb = max(1, int(np.ceil(0.25 * new.sum())))
        for k in order[:nb]:
            if fi[k] <= 1.0:
                break
            broken[k] = True
            j, i = path["idx"][k]
            scale[j, i] = knock
            # knock the whole adhesive thickness at that location (through-thickness debond)
            if path["normal"][k] == "x":
                cols = [ii for ii in range(mesh.nx) if -geo.cavity_half_width < mesh.xc[ii] < -geo.cavity_half_width + geo.adhesive_thickness]
                scale[j, cols] = knock
            else:
                rows = [jj for jj in range(mesh.ny) if geo.y_floor < mesh.yc[jj] < geo.y_floor + geo.adhesive_thickness]
                scale[rows, i] = knock
    s = path["s"]
    ds = np.gradient(s)
    wall = np.array(path["normal"]) == "x"
    comp = mesh.tags["composite"]
    s1 = sol.s1
    return {"composite": composite.name, "load_N": load_N, "strengths_MPa": strengths, "iterations": len(history),
            "converged": bool(not new.any()), "debonded_wall_length_mm": float(np.sum(np.abs(ds)[broken & wall])),
            "debonded_floor_length_mm_half": float(np.sum(np.abs(ds)[broken & ~wall])),
            "debonded_fraction_of_half_path": float(np.sum(np.abs(ds)[broken]) / (geo.cavity_depth + geo.cavity_half_width)),
            "debonded_s_mm": s[broken].tolist(), "composite_s1_max_MPa_after": float(s1[comp].max()),
            "history": history}


# --------------------------------------------------------------------------------------
# mesh convergence
# --------------------------------------------------------------------------------------


def mesh_convergence(composite: DirectComposite = CONVENTIONAL_60, geo: CavityGeometry = CavityGeometry(),
                     levels=((0.0075, 0.4), (0.005, 0.25), (0.0025, 0.125), (0.00125, 0.0625))) -> List[dict]:
    rows = []
    for hf, hm in levels:
        r = run_cavity_case(composite, geo, loads_N=(0.0,), h_fine=hf, h_max=hm)
        c = r["cases"][0]
        rows.append({"h_fine_um": hf * 1e3, "h_max_mm": hm, "n_elements": r["n_elements"],
                     "sn_mid_floor_MPa": c["sn_mid_floor_MPa"], "sn_mid_dentin_wall_MPa": c["sn_mid_dentin_wall_MPa"],
                     "sn_mid_enamel_wall_MPa": c["sn_mid_enamel_wall_MPa"],
                     "sn_corner_d0_floor_MPa": c["sn_at_corner_plus_d0_floor_MPa"],
                     "sn_max_any_MPa": c["sn_max_any_MPa"], "tau_max_abs_MPa": c["tau_max_abs_MPa"]})
    return rows


# --------------------------------------------------------------------------------------
# production driver: python -m fracture_md.composites.restoration
# --------------------------------------------------------------------------------------


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def verification_checks() -> dict:
    return {"free_eigenstrain": verify_free_eigenstrain(), "constrained": verify_constrained(),
            "bimaterial_timoshenko": verify_bimaterial_strip()}


def critical_post_gel_fraction(case_shrink_only: dict) -> dict:
    """Linear scaling: the post-gel shrinkage fraction at which a shrinkage-only criterion reaches 1."""
    c = case_shrink_only
    return {"bond_away_from_corners": 1.0 / c["failure_index_away_from_corners_max"],
            "bond_at_corner_d0": 1.0 / max(c["sn_at_corner_minus_d0_wall_MPa"] / AGED_BOND_MPA["dentin"][0],
                                           c["sn_at_corner_plus_d0_floor_MPa"] / AGED_BOND_MPA["dentin"][0], 1e-12),
            "composite_cracking_interior_vs_ISO4049_80MPa": ISO4049_FLEXURAL_MIN_MPA / c["composite_s1_max_away_from_interface_MPa"]}


def main(outdir: str = "data/dental/restoration", figdir: str = "docs/dental/figures") -> dict:
    import json
    import os

    os.makedirs(outdir, exist_ok=True)
    geo = CavityGeometry()
    out = {"geometry_mm": geo.__dict__, "materials": {m.name: m.__dict__ for m in (ENAMEL, DENTIN, ADHESIVE)},
           "composites": {c.name: c.__dict__ for c in (CONVENTIONAL_60, HIGHFILL_70)},
           "aged_bond_MPa": AGED_BOND_MPA, "czm_sigma_c_MPa": CZM_SIGMA_C_MPA,
           "verification": verification_checks()}
    out["mesh_convergence"] = mesh_convergence()
    cases, crit = {}, {}
    for comp in (CONVENTIONAL_60, HIGHFILL_70):
        central = run_cavity_case(comp, geo, loads_N=(0.0, 100.0, 150.0, 200.0), load_span=(-1.0, 1.0))
        margin = run_cavity_case(comp, geo, loads_N=(150.0,), load_span=(1.0, 3.0))
        cases[comp.name] = {"central_load": central, "load_over_margin": margin}
        crit[comp.name] = critical_post_gel_fraction(central["cases"][0])
    out["cases"] = cases
    out["critical_post_gel_fraction"] = crit
    out["incremental_vs_bulk"] = {c.name: incremental_vs_bulk(c, geo) for c in (CONVENTIONAL_60, HIGHFILL_70)}
    out["sequential_debond"] = {}
    for c in (CONVENTIONAL_60, HIGHFILL_70):
        for f in (1.0, 0.5, 0.25):
            out["sequential_debond"][f"{c.name}_fpg{f}"] = sequential_debond(c, geo, load_N=150.0, f_post_gel=f)
    with open(os.path.join(outdir, "restoration_results.json"), "w") as fh:
        json.dump(_jsonable(out), fh, indent=1)
    try:
        _figures(out, figdir, geo)
    except Exception as exc:  # figures are non-essential
        print("figure generation failed:", exc)
    return out


def _figures(out, figdir, geo):
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    # 1. interface tractions
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axs, ("conventional_60vol", "highfill_70vol")):
        r = out["cases"][name]["central_load"]
        s = np.array(r["path_s"])
        for c, ls in zip(r["cases"], ("-", ":", "--", "-.")):
            if c["load_N"] in (0.0, 150.0):
                ax.plot(s, c["sn_profile"], ls, label=f"sigma_n, {c['load_N']:.0f} N")
                ax.plot(s, np.abs(c["tau_profile"]), ls, alpha=0.6, label=f"|tau|, {c['load_N']:.0f} N")
        ax.axvline(geo.enamel_thickness, color="0.6", lw=0.8)
        ax.axvline(geo.cavity_depth, color="k", lw=0.8)
        ax.axhspan(AGED_BOND_MPA["dentin"][1], AGED_BOND_MPA["dentin"][2], color="C3", alpha=0.12, label="aged dentin uTBS range")
        ax.axhline(AGED_BOND_MPA["enamel"][0], color="C2", lw=0.8, ls="--", label="aged enamel uTBS")
        ax.set_ylim(-40, 120)
        ax.set_xlabel("path from occlusal margin (mm): wall 0-2 (DEJ at 1), floor 2-4 (centre)")
        ax.set_title(f"{name}: instantaneous shrinkage (f_pg=1) +/- 150 N")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("interface traction (MPa)")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "restoration_interface_stress.png"), dpi=130)
    plt.close(fig)
    # 2. mesh convergence
    mc = out["mesh_convergence"]
    h = [m["h_fine_um"] for m in mc]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, lab in (("sn_mid_floor_MPa", "mid-floor"), ("sn_mid_dentin_wall_MPa", "mid dentin wall"),
                   ("sn_corner_d0_floor_MPa", "50 um from line angle"), ("sn_max_any_MPa", "max (singular corner)")):
        ax.plot(h, [m[k] for m in mc], "o-", label=lab)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("finest element size in adhesive (um)")
    ax.set_ylabel("interface normal stress, shrinkage only (MPa)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("mesh convergence (conventional 60 vol%)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "restoration_mesh_convergence.png"), dpi=130)
    plt.close(fig)
    # 3. stress map
    comp = CONVENTIONAL_60
    mesh = cavity_mesh(geo, comp)
    fe = PlaneStrainFE(mesh, clamp_bottom(mesh))
    sol = fe.solve(np.where(mesh.tags["composite"], comp.eigenstrain(1.0), 0.0))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pc = ax.pcolormesh(mesh.x, mesh.y, sol.s1, cmap="magma", vmin=-20, vmax=150, shading="flat")
    ax.contour(mesh.xc, mesh.yc, mesh.mat, levels=[0.5, 1.5, 2.5], colors="w", linewidths=0.5)
    fig.colorbar(pc, ax=ax, label="max in-plane principal stress (MPa)")
    ax.set_aspect("equal")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(1.5, 5.05)
    ax.set_title("Class I, conventional 60 vol%, instantaneous shrinkage (upper bound)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "restoration_stress_map.png"), dpi=130)
    plt.close(fig)


# ======================================================================================
# CUSP-DEFLECTION TEST (3M aluminium-block method) -- flowable composites
# ======================================================================================
# PRIMARY (used by default): 3M Filtek Supreme Flowable Restorative TDS (2021) p.11 -- 4 x 4 mm
#   open-ended (MOD-like) cavity in an aluminium block, sandblasted + silane + adhesive, TWO separately
#   cured 2 mm increments, LVDT on the cavity wall (ref. Thalacker et al., CED-IADR 2007 #0289).
#   Block size and wall width are not in the TDS -> taken from the patent apparatus below and varied.
# Earlier-recorded apparatus (3M patents, e.g. US 11,103,425 and US 9,056,043 Joly et al.; not
# re-verified in this run), kept as the PATENT_SLOT_GEOMETRY variant:
#   rectangular 15 x 8 x 8 mm aluminium block; slot 8 mm long (through, open-ended),
#   4 mm deep, 2 mm across, located 2 mm from an edge -> a 2 mm wide aluminium "cusp";
#   slot sandblasted (Rocatec), RelyX Ceramic Primer, Scotchbond Universal (bonded walls);
#   filled (bulk), light-cured 60 s; LVDT on the cusp tip; value read 9 min after light off.
# 2D model: cross-section normal to the slot (x across the 15 mm, y up the 8 mm height),
#   PLANE STRAIN (slot long vs cross-section) with PLANE STRESS as the bracket (the 8 mm
#   slot is open-ended and equal to the block depth, so reality lies in between).
#   Bottom face fully clamped (ASSUMED fixture). Deflection = inward (+x) displacement of
#   the cusp's outer top corner (x=0, y=8 mm), where an LVDT would sit.
# Eigenstrain: eps* = -f_pg * S_v / 3 (post-gel fraction f_pg: the ONE calibrated factor;
#   it also absorbs 2D idealisation, compliance of the adhesive and viscoelastic relief).
AL_6061 = IsoMaterial("aluminium", 70.0, 0.33, "E, nu: handbook values for Al alloys (brief-specified)")

# 3M internal data (FLOWABLE_TARGET.md): cusp deflection anchors, um
CUSP_ANCHORS_UM = {"flowable_average_proxy": 13.4, "filtek_supreme_flowable_proxy": 13.3}
TYPICAL_DENTIN_BOND_MPA = (20.0, 25.0)  # aged uTBS range, docs/dental/BONDING.md
POST_GEL_FRACTION_LIT = (0.3, 0.6)  # range stated in the task brief (literature range, not independently verified)


@dataclass(frozen=True)
class AlBlockGeometry:
    """Default = 3M Filtek Supreme Flowable TDS (2021) test: 4 x 4 mm open-ended cavity in an Al block,
    filled in 2 separately cured 2 mm increments. Block outer size and cusp (wall) width are NOT given in
    the TDS; 15 x 8 mm and a 2 mm wall are taken from the 3M patent apparatus (see PATENT_SLOT_GEOMETRY)."""
    block_width: float = 15.0  # mm (x)
    block_height: float = 8.0  # mm (y)
    cusp_width: float = 2.0  # mm, wall between block edge and slot (ASSUMED, patent value)
    slot_width: float = 4.0  # mm (TDS: 4 x 4 mm cavity)
    slot_depth: float = 4.0  # mm (TDS)
    n_increments: int = 2  # TDS: 2 separately cured 2 mm increments

    @property
    def x_slot0(self):
        return self.cusp_width

    @property
    def x_slot1(self):
        return self.cusp_width + self.slot_width

    @property
    def y_floor(self):
        return self.block_height - self.slot_depth


# Patent apparatus (US 11,103,425 / US 9,056,043, as recorded by the previous agent; not re-verified here):
# slot 2 mm across x 4 mm deep, bulk fill. Used only as a geometry-sensitivity variant.
PATENT_SLOT_GEOMETRY = AlBlockGeometry(slot_width=2.0, n_increments=1)
TDS_GEOMETRY = AlBlockGeometry()


def al_block_mesh(geo: AlBlockGeometry, composite: DirectComposite, h: float = 0.05, h_max: float = 0.5,
                  growth: float = 0.25) -> Mesh:
    g = geo
    # extra breaks at 0.5 mm spacing across the cusp + slot so the element size there stays ~h
    # (interface stresses along the walls are sampled at element centres)
    xs = np.arange(0.0, g.x_slot1 + 1e-9, 0.5)
    ys = np.arange(g.y_floor, g.block_height + 1e-9, 0.5)
    xb = sorted({*np.round(xs, 9), g.x_slot0, g.x_slot1, g.block_width})
    yb = sorted({0.0, *np.round(ys, 9)})
    x = graded_axis(xb, [h if v <= g.x_slot1 + 1e-9 else h_max for v in xb], h_max, growth)
    y = graded_axis(yb, [h if v >= g.y_floor - 1e-9 else h_max for v in yb], h_max, growth)
    mesh = Mesh(x, y, np.zeros((len(y) - 1, len(x) - 1), int), [AL_6061, composite.material()],
                np.zeros((len(y) - 1, len(x) - 1)))
    XC, YC = np.meshgrid(mesh.xc, mesh.yc)
    comp = (XC > g.x_slot0) & (XC < g.x_slot1) & (YC > g.y_floor)
    mesh.mat[comp] = 1
    mesh.tags = {"composite": comp}
    return mesh


VOID_STIFFNESS_SCALE = 1e-6  # not-yet-placed composite increments: stiffness x 1e-6 (numerical void)


def increment_masks(mesh: Mesh, geo: AlBlockGeometry) -> List[np.ndarray]:
    """Composite element masks of the n_increments horizontal layers, bottom (floor) first."""
    comp = mesh.tags["composite"]
    edges = geo.y_floor + np.linspace(0.0, geo.slot_depth, geo.n_increments + 1)
    YC = np.broadcast_to(mesh.yc[:, None], comp.shape)
    return [comp & (YC > a) & (YC < b) for a, b in zip(edges[:-1], edges[1:])]


def incremental_fill_solution(mesh: Mesh, geo: AlBlockGeometry, plane: str, eps_star: float) -> FESolution:
    """Sequential incremental filling, linear superposition.  Step k: increments 1..k present (k-1 already
    cured and stressed, uncured ones = void), eigenstrain eps* applied to increment k only; the total field
    is the sum of the step solutions (exact for linear elasticity with stiffness frozen per step).
    Assumes each increment shrinks only after placement and bonds stress-free to the cured layer below."""
    masks = increment_masks(mesh, geo)
    comp = mesh.tags["composite"]
    total = None
    for k, mk in enumerate(masks):
        present = ~comp.copy()
        for m in masks[: k + 1]:
            present |= m
        scale = np.where(present, 1.0, VOID_STIFFNESS_SCALE)
        fe = PlaneStrainFE(mesh, clamp_bottom(mesh), stiff_scale=scale, plane=plane)
        sol = fe.solve(np.where(mk, eps_star, 0.0))
        if total is None:
            total = sol
        else:
            total = FESolution(mesh, total.u + sol.u, total.sxx + sol.sxx, total.syy + sol.syy,
                               total.sxy + sol.sxy, total.szz + sol.szz)
    return total


def cusp_unit_response(E_gpa: float, nu: float, geo: AlBlockGeometry = AlBlockGeometry(), plane: str = "strain",
                       h: float = 0.05, corner_exclusion: float = 0.25) -> dict:
    """Cusp deflection and interface normal stresses per unit (negative) linear eigenstrain eps* = -1.
    Linear elasticity -> results scale exactly with f_pg * S_v / 3."""
    comp = DirectComposite("unit", E_gpa, nu, 3.0, 0.0)
    mesh = al_block_mesh(geo, comp, h=h)
    sol = incremental_fill_solution(mesh, geo, plane, -1.0)
    ux = sol.ux()
    j_top = mesh.ny
    defl = float(ux[j_top, 0])  # +x = towards the slot
    far_wall = float(-ux[j_top, int(np.argmin(np.abs(mesh.x - geo.x_slot1)))])
    # interface normal stress (composite side, element centres): cusp wall x=2, far wall x=4, floor y=4
    ic0 = int(np.searchsorted(mesh.x, geo.x_slot0 + 1e-12))  # first composite column
    ic1 = int(np.searchsorted(mesh.x, geo.x_slot1 - 1e-12)) - 1  # last composite column
    jf = int(np.searchsorted(mesh.y, geo.y_floor + 1e-12))  # first composite row
    yc, xc = mesh.yc, mesh.xc
    rows = np.arange(jf, mesh.ny)
    cols = np.arange(ic0, ic1 + 1)
    wall_ok = (yc[rows] > geo.y_floor + corner_exclusion) & (yc[rows] < geo.block_height - corner_exclusion)
    # the top of every non-final increment was a free surface meeting the wall when it cured -> another
    # singular bimaterial corner; exclude a band of +/- corner_exclusion around each increment boundary too
    for yb in geo.y_floor + geo.slot_depth * np.arange(1, geo.n_increments) / geo.n_increments:
        wall_ok &= np.abs(yc[rows] - yb) > corner_exclusion
    wall_sel = rows[wall_ok]
    floor_sel = cols[(xc[cols] > geo.x_slot0 + corner_exclusion) & (xc[cols] < geo.x_slot1 - corner_exclusion)]
    sn_wall = sol.sxx[wall_sel, ic0]  # tension (>0) pulls composite off the cusp wall
    sn_far = sol.sxx[wall_sel, ic1]
    sn_floor = sol.syy[jf, floor_sel]
    # "wall mid" = mid-height of the TOP increment (away from every singular corner)
    y_mid = geo.block_height - 0.5 * geo.slot_depth / geo.n_increments
    jm = wall_sel[np.argmin(np.abs(yc[wall_sel] - y_mid))]
    im = floor_sel[np.argmin(np.abs(xc[floor_sel] - (geo.x_slot0 + geo.slot_width / 2)))]
    # peak value INCLUDING corners (singular in linear elasticity: grows with refinement; reported only)
    sn_all = max(sol.sxx[rows, ic0].max(), sol.sxx[rows, ic1].max(), sol.syy[jf, cols].max())
    return {"E_gpa": E_gpa, "nu": nu, "plane": plane, "h_mm": h, "n_elements": mesh.nx * mesh.ny,
            "n_increments": geo.n_increments, "slot_width_mm": geo.slot_width,
            "sn_peak_incl_singular_corners_MPa_per_unit": float(sn_all),
            "defl_mm_per_unit": defl, "far_wall_defl_mm_per_unit": far_wall,
            "sn_wall_mid_MPa_per_unit": float(sol.sxx[jm, ic0]),
            "sn_floor_mid_MPa_per_unit": float(sol.syy[jf, im]),
            "sn_max_away_from_corners_MPa_per_unit": float(max(sn_wall.max(), sn_far.max(), sn_floor.max())),
            "corner_exclusion_mm": corner_exclusion}


SURROGATE_E_GRID_GPA = np.geomspace(2.0, 30.0, 16)
SURROGATE_NU_GRID = (0.26, 0.30, 0.34)
_UNIT_KEYS = ("defl_mm_per_unit", "sn_max_away_from_corners_MPa_per_unit", "sn_wall_mid_MPa_per_unit",
              "sn_floor_mid_MPa_per_unit")


class UnitResponseSurrogate:
    """Per-unit-eigenstrain responses tabulated on a (log E, nu) grid of FE solves (16 x 3) and interpolated:
    cubic spline in log E, quadratic Lagrange in nu. Linear elasticity makes every response exactly linear in
    the eigenstrain, so only the (E, nu) dependence needs interpolating. Accuracy vs direct FE is checked in
    verify_surrogate()."""

    def __init__(self, geo: AlBlockGeometry, plane: str, h: float):
        from scipy.interpolate import CubicSpline

        self.geo, self.plane, self.h = geo, plane, h
        tab = {k: np.zeros((len(SURROGATE_NU_GRID), len(SURROGATE_E_GRID_GPA))) for k in _UNIT_KEYS}
        for i, nu in enumerate(SURROGATE_NU_GRID):
            for j, E in enumerate(SURROGATE_E_GRID_GPA):
                u = cusp_unit_response(float(E), float(nu), geo, plane, h)
                for k in _UNIT_KEYS:
                    tab[k][i, j] = u[k]
        lx = np.log(SURROGATE_E_GRID_GPA)
        self._spl = {k: [CubicSpline(lx, tab[k][i]) for i in range(len(SURROGATE_NU_GRID))] for k in _UNIT_KEYS}

    def __call__(self, E_gpa: float, nu: float) -> dict:
        if not (SURROGATE_E_GRID_GPA[0] <= E_gpa <= SURROGATE_E_GRID_GPA[-1]):
            return cusp_unit_response(float(E_gpa), float(nu), self.geo, self.plane, self.h)
        n = np.asarray(SURROGATE_NU_GRID)
        w = np.array([np.prod([(nu - n[m]) / (n[l] - n[m]) for m in range(3) if m != l]) for l in range(3)])
        lE = np.log(E_gpa)
        return {k: float(sum(w[i] * self._spl[k][i](lE) for i in range(3))) for k in _UNIT_KEYS}


_SURROGATES: Dict[tuple, UnitResponseSurrogate] = {}


def unit_response(E_gpa, nu, geo=TDS_GEOMETRY, plane="strain", h=0.05) -> dict:
    key = (geo, plane, h)
    if key not in _SURROGATES:
        _SURROGATES[key] = UnitResponseSurrogate(geo, plane, h)
    return _SURROGATES[key](float(E_gpa), float(nu))


def verify_surrogate(points=((5.43, 0.309), (8.08, 0.299), (13.5, 0.283), (19.3, 0.289)), plane="strain",
                     h=0.05) -> dict:
    rows = []
    for E, nu in points:
        a = unit_response(E, nu, TDS_GEOMETRY, plane, h)
        b = cusp_unit_response(E, nu, TDS_GEOMETRY, plane, h)
        rows.append({"E_GPa": E, "nu": nu, "rel_err_defl": abs(a["defl_mm_per_unit"] / b["defl_mm_per_unit"] - 1),
                     "rel_err_sn_max": abs(a["sn_max_away_from_corners_MPa_per_unit"]
                                           / b["sn_max_away_from_corners_MPa_per_unit"] - 1)})
    ed = max(r["rel_err_defl"] for r in rows)
    es = max(r["rel_err_sn_max"] for r in rows)
    # deflection is smooth in (E, nu): 0.1 % tolerance; the max-over-elements stress is only piecewise smooth
    # (its location can jump), so 3 % tolerance
    return {"rows": rows, "max_rel_err_defl": ed, "max_rel_err_sn_max": es, "passed": bool(ed < 1e-3 and es < 0.03)}


def cusp_deflection_um(E_gpa, nu, vol_shrinkage_pct, f_post_gel, plane="strain", geo=TDS_GEOMETRY, h=0.05) -> dict:
    u = unit_response(E_gpa, nu, geo, plane, h)
    e = f_post_gel * vol_shrinkage_pct / 100.0 / 3.0  # magnitude of linear post-gel shrinkage
    return {"cusp_deflection_um": u["defl_mm_per_unit"] * e * 1e3,
            "max_interface_normal_stress_MPa": u["sn_max_away_from_corners_MPa_per_unit"] * e,
            "wall_mid_normal_stress_MPa": u["sn_wall_mid_MPa_per_unit"] * e,
            "floor_mid_normal_stress_MPa": u["sn_floor_mid_MPa_per_unit"] * e, "eps_lin": e}


def calibrate_post_gel(anchor: dict, target_um: float, plane="strain", h=0.05) -> float:
    r = cusp_deflection_um(anchor["E_GPa"], anchor["nu"], anchor["shrinkage_vol_pct"], 1.0, plane, h=h)
    return target_um / r["cusp_deflection_um"]


# ------------------------------ verification specific to this model -----------------------


def verify_plane_stress_uniaxial(eps_star=-0.005, n=6) -> dict:
    """Plane stress, ux=0 on left/right, uy=0 bottom, top free -> sigma_xx = -E eps*, sigma_yy = 0."""
    mat = IsoMaterial("m", 7.0, 0.3)
    mesh = rect_mesh(np.linspace(0, 1, n + 1), np.linspace(0, 0.5, n + 1), mat)
    mesh.eig[:] = eps_star
    fixed = {}
    for nd in _edge_nodes(mesh, "left"):
        fixed[2 * nd] = 0.0
    for nd in _edge_nodes(mesh, "right"):
        fixed[2 * nd] = 0.0
    for nd in _edge_nodes(mesh, "bottom"):
        fixed[2 * nd + 1] = 0.0
    s = PlaneStrainFE(mesh, fixed, plane="stress").solve(mesh.eig)
    ex = -mat.E * eps_star
    err = float(max(np.abs(s.sxx - ex).max(), np.abs(s.syy).max()) / ex)
    return {"sxx_MPa": float(s.sxx.mean()), "exact_MPa": ex, "rel_err": err, "passed": bool(err < 1e-10)}


def verify_free_plane_stress(eps_star=-0.005, n=6) -> dict:
    mat = IsoMaterial("m", 7.0, 0.3)
    mesh = rect_mesh(np.linspace(0, 1, n + 1), np.linspace(0, 0.7, n + 1), mat)
    mesh.eig[:] = eps_star
    fixed = {2 * mesh.node(0, 0): 0.0, 2 * mesh.node(0, 0) + 1: 0.0, 2 * mesh.node(mesh.nx, 0) + 1: 0.0}
    s = PlaneStrainFE(mesh, fixed, plane="stress").solve(mesh.eig)
    st = float(max(np.abs(s.sxx).max(), np.abs(s.syy).max(), np.abs(s.sxy).max()))
    ux_ex = eps_star * mesh.x[-1]
    return {"max_stress_MPa": st, "ux_right_mm": float(s.ux()[0, -1]), "ux_exact_mm": ux_ex,
            "passed": bool(st < 1e-8 * mat.E and abs(s.ux()[0, -1] - ux_ex) < 1e-12)}


def verify_bimaterial_plane_stress(L=40.0, t1=0.5, t2=0.5, E1=7.0, E2=70.0, nu=0.3, eps1=-0.004, nx=320,
                                   ny_layer=8) -> dict:
    """Plane-stress bilayer (composite-like on aluminium-like, 10:1 modulus) vs Timoshenko (E, d_eps unmodified)."""
    x = np.linspace(-L / 2, L / 2, nx + 1)
    y = np.concatenate([np.linspace(0, t1, ny_layer + 1), np.linspace(t1, t1 + t2, ny_layer + 1)[1:]])
    mesh = Mesh(x, y, np.zeros((len(y) - 1, nx), int), [IsoMaterial("l1", E1, nu), IsoMaterial("l2", E2, nu)],
                np.zeros((len(y) - 1, nx)))
    mesh.mat[mesh.yc > t1, :] = 1
    mesh.eig[mesh.yc < t1, :] = eps1
    ic = nx // 2
    fixed = {2 * mesh.node(ic, 0): 0.0, 2 * mesh.node(ic, 0) + 1: 0.0, 2 * mesh.node(ic, mesh.ny): 0.0}
    sol = PlaneStrainFE(mesh, fixed, plane="stress").solve(mesh.eig)
    jm = np.argmin(np.abs(mesh.y - (t1 + t2) / 2))
    sel = np.abs(x) < L / 4
    kfe = 2 * np.polyfit(x[sel], sol.uy()[jm][sel], 2)[0]
    kex = timoshenko_bimaterial_curvature(E1, E2, t1, t2, eps1)
    rel = abs(abs(kfe) - abs(kex)) / abs(kex)
    return {"kappa_fe_per_mm": float(kfe), "kappa_timoshenko_per_mm": float(kex), "rel_err": float(rel),
            "passed": bool(rel < 0.02 and kfe < 0)}


def lame_inclusion_pressure(E_i, nu_i, E_m, nu_m, eps_star):
    """Plane-strain circular inclusion (radius a) with isotropic eigenstrain eps* in an infinite matrix:
    interface pressure p = (1+nu_i) eps* / ((1+nu_m)/E_m + (1+nu_i)(1-2nu_i)/E_i); p>0 = compression
    (expanding inclusion). Shrinking inclusion (eps*<0) -> p<0: radial TENSION sigma_rr = -p at r=a,
    which is also the uniform in-plane stress inside the inclusion (Eshelby/Lame)."""
    return (1 + nu_i) * eps_star / ((1 + nu_m) / E_m + (1 + nu_i) * (1 - 2 * nu_i) / E_i)


def verify_lame_inclusion(E_i=7.0, nu_i=0.3, E_m=70.0, nu_m=0.33, eps_star=-0.004, a=1.0, R=12.0, h=0.05) -> dict:
    """Staircase-discretised circular shrinking inclusion in a large square; sigma inside at centre vs Lame.
    Uses the same Q4 grid machinery as the block model (the curved interface is pixelated)."""
    x = graded_axis([-R, -a, a, R], [0.6, h, h, 0.6], 0.6, 0.2)
    mesh = Mesh(x, x.copy(), np.zeros((len(x) - 1, len(x) - 1), int),
                [IsoMaterial("m", E_m, nu_m), IsoMaterial("i", E_i, nu_i)], np.zeros((len(x) - 1, len(x) - 1)))
    XC, YC = np.meshgrid(mesh.xc, mesh.yc)
    inc = XC ** 2 + YC ** 2 < a ** 2
    mesh.mat[inc] = 1
    mesh.eig[inc] = eps_star
    fixed = {}
    for s in ("left", "right", "bottom", "top"):
        for nd in _edge_nodes(mesh, s):
            fixed[2 * nd] = 0.0
            fixed[2 * nd + 1] = 0.0
    sol = PlaneStrainFE(mesh, fixed).solve(mesh.eig)
    core = XC ** 2 + YC ** 2 < (0.5 * a) ** 2
    s_fe = float(0.5 * (sol.sxx[core] + sol.syy[core]).mean())
    p = lame_inclusion_pressure(E_i * 1e3, nu_i, E_m * 1e3, nu_m, eps_star)
    s_ex = -p
    rel = abs(s_fe - s_ex) / abs(s_ex)
    return {"sigma_inclusion_fe_MPa": s_fe, "sigma_inclusion_lame_MPa": float(s_ex), "rel_err": float(rel),
            "note": "finite outer box R/a=12 (clamped) and pixelated interface; tolerance 3 %",
            "passed": bool(rel < 0.03)}


def cusp_mesh_convergence(E_gpa=7.36, nu=0.31, hs=(0.2, 0.1, 0.05, 0.025), plane="strain") -> List[dict]:
    rows = []
    for h in hs:
        u = cusp_unit_response(E_gpa, nu, plane=plane, h=h)
        rows.append({"h_mm": h, "n_elements": u["n_elements"], "defl_mm_per_unit": u["defl_mm_per_unit"],
                     "sn_wall_mid_MPa_per_unit": u["sn_wall_mid_MPa_per_unit"],
                     "sn_max_away_from_corners_MPa_per_unit": u["sn_max_away_from_corners_MPa_per_unit"]})
    ref = rows[-1]["defl_mm_per_unit"]
    for r in rows:
        r["defl_rel_diff_vs_finest"] = abs(r["defl_mm_per_unit"] - ref) / abs(ref)
    return rows


def cusp_verification() -> dict:
    v = {"free_eigenstrain_plane_strain": verify_free_eigenstrain(), "constrained_plane_strain": verify_constrained(),
         "bimaterial_timoshenko_plane_strain": verify_bimaterial_strip(),
         "free_eigenstrain_plane_stress": verify_free_plane_stress(),
         "constrained_uniaxial_plane_stress": verify_plane_stress_uniaxial(),
         "bimaterial_timoshenko_plane_stress_10to1": verify_bimaterial_plane_stress(),
         "lame_circular_inclusion": verify_lame_inclusion()}
    conv = cusp_mesh_convergence()
    v["cusp_mesh_convergence"] = {"rows": conv, "passed": bool(conv[-2]["defl_rel_diff_vs_finest"] < 0.01),
                                  "note": "deflection (global) converges to <0.1 %; interface stresses away from the "
                                          "corners still change ~5-10 % per mesh halving (corner fields reach far "
                                          "into a 2 mm increment) -> +/-15 % mesh uncertainty on stresses at h=0.05"}
    v["all_passed"] = bool(all(x["passed"] for x in v.values() if isinstance(x, dict)))
    return v


# ------------------------------ production: calibration + predictions ---------------------


def _load_candidates(path="data/dental/direct/materials_for_restoration.json") -> dict:
    import json

    with open(path) as fh:
        return json.load(fh)["candidates"]


GEOMETRY_VARIANTS = {
    "tds_4x4_two_increments": TDS_GEOMETRY,
    "tds_4x4_bulk": AlBlockGeometry(n_increments=1),
    "tds_4x4_two_increments_3mm_wall": AlBlockGeometry(cusp_width=3.0),
    "patent_2mm_slot_bulk": PATENT_SLOT_GEOMETRY,
}


def geometry_variant_study(cands: dict, h=0.05, plane="strain") -> dict:
    """Model-form check: recalibrate f_pg on the flowable-average anchor for each plausible geometry and
    compare the Filtek check and the candidate predictions (spread = geometry model-form uncertainty)."""
    avg, fil = cands["flowable_average_proxy"], cands["filtek_supreme_flowable_proxy"]
    res = {}
    for name, g in GEOMETRY_VARIANTS.items():
        f = CUSP_ANCHORS_UM["flowable_average_proxy"] / cusp_deflection_um(
            avg["E_GPa"], avg["nu"], avg["shrinkage_vol_pct"], 1.0, plane, geo=g, h=h)["cusp_deflection_um"]
        pr = {cid: cusp_deflection_um(c["E_GPa"], c["nu"], c["shrinkage_vol_pct"], f, plane, geo=g, h=h)
              for cid, c in cands.items()}
        res[name] = {"geometry": g.__dict__, "f_post_gel_eff": f,
                     "filtek_check_um": pr["filtek_supreme_flowable_proxy"]["cusp_deflection_um"],
                     "predictions_um": {cid: v["cusp_deflection_um"] for cid, v in pr.items()},
                     "max_interface_normal_stress_MPa": {cid: v["max_interface_normal_stress_MPa"]
                                                         for cid, v in pr.items()}}
    return res


def two_anchor_lsq_f_pg(cands: dict, plane="strain", h=0.05, geo=TDS_GEOMETRY) -> dict:
    """Alternative calibration: least-squares f_pg over BOTH anchors (no held-out check left)."""
    d = np.array([cusp_deflection_um(cands[k]["E_GPa"], cands[k]["nu"], cands[k]["shrinkage_vol_pct"], 1.0, plane,
                                     geo=geo, h=h)["cusp_deflection_um"] for k in CUSP_ANCHORS_UM])
    t = np.array(list(CUSP_ANCHORS_UM.values()))
    f = float(d @ t / (d @ d))
    return {"f_post_gel_eff": f, "anchors": list(CUSP_ANCHORS_UM), "fitted_um": (f * d).tolist(), "target_um": t.tolist(),
            "rel_err": (np.abs(f * d - t) / t).tolist()}


def predict_candidates(cands: dict, f_pg: Dict[str, float], h=0.05) -> dict:
    """Central value: plane strain calibrated. Band: min/max over {plane strain, plane stress (each with its
    own calibrated f_pg)} x {E_low, E, E_high} x {S_low, S, S_high} from the candidate file."""
    out = {}
    for cid, c in cands.items():
        cen = cusp_deflection_um(c["E_GPa"], c["nu"], c["shrinkage_vol_pct"], f_pg["strain"], "strain", h=h)
        vals, sns = [], []
        for plane in ("strain", "stress"):
            for E in (c.get("E_GPa_low", c["E_GPa"]), c["E_GPa"], c.get("E_GPa_high", c["E_GPa"])):
                for S in (c.get("shrinkage_vol_pct_low", c["shrinkage_vol_pct"]), c["shrinkage_vol_pct"],
                          c.get("shrinkage_vol_pct_high", c["shrinkage_vol_pct"])):
                    r = cusp_deflection_um(E, c["nu"], S, f_pg[plane], plane, h=h)
                    vals.append(r["cusp_deflection_um"])
                    sns.append(r["max_interface_normal_stress_MPa"])
        stress_only = cusp_deflection_um(c["E_GPa"], c["nu"], c["shrinkage_vol_pct"], f_pg["stress"], "stress", h=h)
        out[cid] = {"cusp_deflection_um": cen["cusp_deflection_um"], "lo": float(min(vals)), "hi": float(max(vals)),
                    "cusp_deflection_plane_stress_um": stress_only["cusp_deflection_um"],
                    "max_interface_normal_stress_MPa": cen["max_interface_normal_stress_MPa"],
                    "max_interface_normal_stress_MPa_lo": float(min(sns)),
                    "max_interface_normal_stress_MPa_hi": float(max(sns)),
                    "wall_mid_normal_stress_MPa": cen["wall_mid_normal_stress_MPa"],
                    "floor_mid_normal_stress_MPa": cen["floor_mid_normal_stress_MPa"],
                    "interface_stress_over_dentin_bond_20_25MPa": [
                        cen["max_interface_normal_stress_MPa"] / TYPICAL_DENTIN_BOND_MPA[1],
                        cen["max_interface_normal_stress_MPa"] / TYPICAL_DENTIN_BOND_MPA[0]],
                    "beats_filtek_13p3_absolute": bool(cen["cusp_deflection_um"] < 13.3),
                    "beats_filtek_13p3_at_hi": bool(max(vals) < 13.3),
                    "inputs": {"E_GPa": c["E_GPa"], "nu": c["nu"], "shrinkage_vol_pct": c["shrinkage_vol_pct"]}}
    ref = out.get("filtek_supreme_flowable_proxy")
    if ref is not None:
        for cid, p in out.items():
            r = p["cusp_deflection_um"] / ref["cusp_deflection_um"]
            p["ratio_to_filtek_proxy"] = r
            p["cusp_deflection_um_filtek_referenced"] = r * CUSP_ANCHORS_UM["filtek_supreme_flowable_proxy"]
            p["beats_filtek_relative"] = bool(r < 1.0)
    return out


def cusp_main(outdir="data/dental/restoration", figdir="docs/dental/figures",
              cand_path="data/dental/direct/materials_for_restoration.json", h=0.05, figures=True) -> dict:
    import json
    import os

    os.makedirs(outdir, exist_ok=True)
    ver = cusp_verification()
    ver["surrogate_vs_direct_fe"] = verify_surrogate(h=h)
    ver["all_passed"] = bool(ver["all_passed"] and ver["surrogate_vs_direct_fe"]["passed"])
    if not ver["all_passed"]:
        raise RuntimeError("cusp verification failed; refusing to predict: " + json.dumps(_jsonable(ver))[:2000])
    cands = _load_candidates(cand_path)
    cal_id, chk_id = "flowable_average_proxy", "filtek_supreme_flowable_proxy"
    f_pg = {p: calibrate_post_gel(cands[cal_id], CUSP_ANCHORS_UM[cal_id], p, h) for p in ("strain", "stress")}
    preds = predict_candidates(cands, f_pg, h)
    chk = preds[chk_id]
    calib = {"method": "one-parameter fit of the post-gel shrinkage fraction f_pg on ONE anchor "
                       f"({cal_id} -> {CUSP_ANCHORS_UM[cal_id]} um), checked on the second ({chk_id} -> "
                       f"{CUSP_ANCHORS_UM[chk_id]} um)",
             "f_post_gel_plane_strain": f_pg["strain"], "f_post_gel_plane_stress": f_pg["stress"],
             "literature_range_f_post_gel": POST_GEL_FRACTION_LIT,
             "f_pg_within_literature_range": bool(POST_GEL_FRACTION_LIT[0] <= f_pg["strain"] <= POST_GEL_FRACTION_LIT[1]),
             "check_anchor_predicted_um": chk["cusp_deflection_um"], "check_anchor_target_um": CUSP_ANCHORS_UM[chk_id],
             "check_anchor_rel_err": abs(chk["cusp_deflection_um"] - CUSP_ANCHORS_UM[chk_id]) / CUSP_ANCHORS_UM[chk_id],
             "check_anchor_band_um": [chk["lo"], chk["hi"]],
             "check_anchor_within_band": bool(chk["lo"] <= CUSP_ANCHORS_UM[chk_id] <= chk["hi"]),
             "note": "the two anchors differ by only 0.1 um (0.75 %), far below the input uncertainty of the proxies' "
                     "E and shrinkage; the held-out check therefore tests the consistency of the proxy inputs more "
                     "than the mechanics. f_pg is an EFFECTIVE factor lumping post-gel fraction, 3D vs 2D constraint, "
                     "adhesive compliance, viscoelastic relief during the 9-min read-out and the unknown wall width.",
             "two_anchor_lsq_alternative": two_anchor_lsq_f_pg(cands, "strain", h)}
    E_grid = np.linspace(3.0, 16.0, 14)
    S_grid = np.linspace(2.0, 6.0, 17)
    unit = np.array([cusp_deflection_um(E, 0.30, 1.0, f_pg["strain"], "strain", h=h)["cusp_deflection_um"]
                     for E in E_grid])
    unit_sn = np.array([cusp_deflection_um(E, 0.30, 1.0, f_pg["strain"], "strain", h=h)["max_interface_normal_stress_MPa"]
                        for E in E_grid])
    sens = {"E_GPa": E_grid, "shrinkage_vol_pct": S_grid, "nu": 0.30,
            "cusp_deflection_um": np.outer(S_grid, unit), "max_interface_normal_stress_MPa": np.outer(S_grid, unit_sn)}
    geo = TDS_GEOMETRY
    geo_study = geometry_variant_study(cands, h=h)
    for cid, p in preds.items():
        vals = [v["predictions_um"][cid] for v in geo_study.values()]
        p["geometry_variant_range_um"] = [float(min(vals)), float(max(vals))]
        p["lo"] = float(min(p["lo"], min(vals)))
        p["hi"] = float(max(p["hi"], max(vals)))
    bulk = cusp_unit_response(7.36, 0.31, AlBlockGeometry(n_increments=1), "strain", h)["defl_mm_per_unit"]
    inc = cusp_unit_response(7.36, 0.31, TDS_GEOMETRY, "strain", h)["defl_mm_per_unit"]
    ver["incremental_vs_bulk_trend_versluis1996"] = {
        "defl_bulk_per_unit": bulk, "defl_two_increments_per_unit": inc, "ratio_incremental_over_bulk": inc / bulk,
        "expected": "incremental > bulk for a well-bonded elastic restoration (Versluis, Douglas, Cross, Sakaguchi, "
                    "J Dent Res 1996;75(3):871-878, doi:10.1177/00220345960750030301)",
        "passed": bool(inc > bulk)}
    out = {"description": "Cusp deflection (3M aluminium-block MOD-slot method) by 2D FE with post-gel shrinkage "
                          "eigenstrain; calibrated DERIVED_MODEL estimates, not measurements.",
           "units": {"cusp_deflection_um": "um", "stress": "MPa", "E": "GPa", "shrinkage": "% volumetric"},
           "protocol_assumed": {"block_mm": [geo.block_width, geo.block_height, 8.0], "slot_mm": {
               "width": geo.slot_width, "depth": geo.slot_depth, "length": 8.0, "open_ended": True},
               "cusp_width_mm": geo.cusp_width, "substrate": AL_6061.__dict__, "bonded_walls": True,
               "fill": f"{geo.n_increments} separately cured increments of {geo.slot_depth / geo.n_increments} mm "
                       "(3M Filtek Supreme Flowable TDS 2021)",
               "assumed_not_in_tds": "block 15 x 8 mm outer size, 2 mm wall (cusp) width, clamped base; taken from "
                                     "the 3M patent apparatus; varied in geometry_variants",
               "support": "bottom face clamped (assumed)",
               "readout": "inward x-displacement of cusp outer top corner (LVDT position)",
               "sources": ["3M Filtek Supreme Flowable Restorative Technical Data Sheet (2021), p.11: 4 x 4 mm "
                           "open-ended cavity in Al block (modulus similar to enamel), sandblasted + silane + adhesive, "
                           "2 increments of 2 mm, LVDT on cavity wall; ref. C. Thalacker et al., CED-IADR 2007 #0289",
                           "US 11,103,425 (3M) cusp deflection method: 15x8x8 mm Al block, slot 8x4x2 mm, 2 mm "
                           "cusp, Rocatec + RelyX Ceramic Primer + Scotchbond Universal, LVDT, 60 s cure, 9 min",
                           "US 9,056,043 (Joly et al., 3M) same apparatus",
                           "3M Filtek Bulk Fill Flowable Technical Product Profile: 4x4 mm open-ended cavity, "
                           "Al chosen for enamel-like modulus"]},
           "model": {"plane": "plane strain (central) with plane stress bracket", "eigenstrain":
                     "eps* = -f_pg * S_v / 3, isotropic, instantaneous, linear elastic", "mesh_h_mm": h},
           "verification": ver, "calibration": calib, "predictions": preds, "sensitivity": sens,
           "geometry_variants": geo_study,
           "interface_vs_bond": {"typical_aged_dentin_uTBS_MPa": TYPICAL_DENTIN_BOND_MPA,
                                 "note": "max interface normal stress excludes +/-0.25 mm around singular corners "
                                         "(slot floor, margin, increment boundaries); corner peaks are unbounded in "
                                         "linear elasticity and not compared with strength."}}
    with open(os.path.join(outdir, "cusp_deflection_results.json"), "w") as fh:
        json.dump(_jsonable(out), fh, indent=1)
    slim = {cid: {k: p[k] for k in ("cusp_deflection_um", "lo", "hi", "max_interface_normal_stress_MPa",
                                    "ratio_to_filtek_proxy", "cusp_deflection_um_filtek_referenced",
                                    "beats_filtek_relative", "beats_filtek_13p3_absolute")}
            for cid, p in preds.items()}
    with open(os.path.join(outdir, "cusp_deflection_predictions.json"), "w") as fh:
        json.dump(_jsonable(slim), fh, indent=1)
    if figures:
        _cusp_figures(out, figdir, f_pg["strain"], h)
    return out


def _cusp_figures(out, figdir, f_pg, h):
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    sens = out["sensitivity"]
    E, S, D = np.asarray(sens["E_GPa"]), np.asarray(sens["shrinkage_vol_pct"]), np.asarray(sens["cusp_deflection_um"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    cs = ax.contourf(E, S, D, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="cusp deflection (um)")
    cl = ax.contour(E, S, D, levels=[10.0, 13.3, 13.4, 16.0], colors=["w", "r", "orange", "w"], linewidths=1.2)
    ax.clabel(cl, fmt="%.1f")
    for cid, p in out["predictions"].items():
        inp = p["inputs"]
        mk = "s" if "proxy" in cid else "o"
        ax.plot(inp["E_GPa"], inp["shrinkage_vol_pct"], mk, color="r" if "proxy" in cid else "k", ms=5)
        if "proxy" in cid:
            ax.annotate(cid.replace("_proxy", ""), (inp["E_GPa"], inp["shrinkage_vol_pct"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("composite E (GPa)")
    ax.set_ylabel("volumetric shrinkage (%)")
    ax.set_title(f"Al-block cusp deflection, plane strain, f_pg={f_pg:.3f}\nred line: Filtek 13.3 um (lower-left is better)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "restoration_cusp_sensitivity.png"), dpi=130)
    plt.close(fig)
    # deformed shape + sigma_xx map for the average proxy
    ai = out["predictions"].get("flowable_average_proxy", {}).get("inputs", {"E_GPa": 7.36, "nu": 0.31,
                                                                            "shrinkage_vol_pct": 5.37})
    comp = DirectComposite("avg", ai["E_GPa"], ai["nu"], ai["shrinkage_vol_pct"], 42.0)
    geo = TDS_GEOMETRY
    mesh = al_block_mesh(geo, comp, h=h)
    sol = incremental_fill_solution(mesh, geo, "strain", comp.eigenstrain(f_pg))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    pc = ax.pcolormesh(mesh.x, mesh.y, sol.sxx, cmap="RdBu_r", vmin=-15, vmax=15, shading="flat")
    fig.colorbar(pc, ax=ax, label="sigma_xx (MPa)")
    X, Y = np.meshgrid(mesh.x, mesh.y)
    sc = 40.0
    ax.plot(X[-1] + sc * sol.ux()[-1], Y[-1] + sc * sol.uy()[-1], "k-", lw=0.8, label=f"top edge, displ. x{sc:.0f}")
    ax.set_xlim(-0.5, 8)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("flowable-average proxy, 3M TDS 4x4 mm Al-block cavity, 2 x 2 mm increments (plane strain)",
                 fontsize=9)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "restoration_cusp_block.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    res = main()
    import json
    print(json.dumps(_jsonable({"verification": res["verification"], "crit": res["critical_post_gel_fraction"]}), indent=1))
