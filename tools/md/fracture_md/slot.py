"""fracture-md cleavage of one material, returned in the units of the md_interface slot.

cleavage_for_slot splits a perfect crystal on one plane with fracture-md and returns the work of separation and
the peak traction of that plane. It splits ONE material. It builds no interface between a ceramic and an
interlayer, so it can serve md_interface only when the interface fails inside a material it models, for example
when both sides are silicon carbide. README.md in this folder gives the details and the validation status.

The calculation runs in a child Python process on the CPU. Importing fracture_md turns on float64 JAX for the
whole process, and the child keeps that away from the pipeline. The child uses the Python of the calling process,
or FRACTURE_MD_PYTHON when that is set.

    from tools.md.fracture_md.slot import cleavage_for_slot
    cleavage_for_slot("silicon carbide", plane="111", potential="erhart_albe_2005_SiC")

From the shell: python tools/md/fracture_md/slot.py "silicon carbide" 111 erhart_albe_2005_SiC
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Materials of tools/schema.py that fracture-md can cleave, and the crystal it builds for each.
MATERIALS = {"silicon carbide": "3C-SiC"}
# Plane label -> (Miller indices in the cubic cell, termination). The cut leaves no atom on the plane.
PLANES = {"111": ((1, 1, 1), "shuffle"), "110": ((1, 1, 0), "nonpolar")}
# Interatomic models whose SiC cleavage fracture-md has run and checked. Vashishta 2007 and MACE-MP-0 are
# fracture-md models too, but their cleavage needs other slab settings (a longer cutoff, and for MACE an extra
# install) that have not been tested, so they are not offered here.
POTENTIALS = {"erhart_albe_2005_SiC": "Erhart-Albe 2005", "tersoff_1989_SiC": "Tersoff 1989"}


def cleavage_for_slot(material, plane="111", potential="erhart_albe_2005_SiC", out_dir=None, timeout_s=900):
    """Rigid cleavage of a perfect crystal of `material` on `plane` with fracture-md, at 0 K.

    Returns separation_energy_J_m2 (work of separation with relaxed surfaces, the Griffith energy of the plane),
    strength_MPa (peak traction of the rigid opening), method, further scalars, and curve_json, a file with the
    traction-separation curve and every fracture-md record. Raises ValueError for a material, plane or potential
    it does not support, and RuntimeError when the calculation fails or a numerical check does not pass.
    """
    if material not in MATERIALS:
        raise ValueError(f"fracture-md cleaves {sorted(MATERIALS)} only, not {material!r}. "
                         "It has no tested model for the other materials of tools/schema.py.")
    if str(plane) not in PLANES:
        raise ValueError(f"plane must be one of {sorted(PLANES)}, not {plane!r}")
    if potential not in POTENTIALS:
        raise ValueError(f"potential must be one of {sorted(POTENTIALS)}, not {potential!r}")
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="fracture_md_cleavage_"))
    out.mkdir(parents=True, exist_ok=True)
    args = {"material": material, "plane": str(plane), "potential": potential, "out_dir": str(out.resolve())}
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    python = os.environ.get("FRACTURE_MD_PYTHON", sys.executable)
    proc = subprocess.run([python, str(Path(__file__).resolve()), "--child"], input=json.dumps(args),
                          capture_output=True, text=True, timeout=timeout_s, env=env)
    if proc.returncode != 0:
        raise RuntimeError("fracture-md cleavage failed: " + proc.stderr.strip()[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _plain(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    return str(x)


def _compute(args):
    """Runs in the child process."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "4")
    t0 = time.time()
    import fracture_md  # noqa: F401  (turns on float64)
    from fracture_md.physics.cleavage import relax_bulk_lattice, run_cleavage
    from fracture_md.potentials import get_potential
    from fracture_md.schema.material import Candidate, CandidateKind
    from fracture_md.schema.protocol import CleavageProtocol
    from fracture_md.structures.builders import sic_3c_conventional, sic_3c_phase, sic_composition

    miller, termination = PLANES[args["plane"]]
    model = get_potential(args["potential"])
    cand = Candidate(id="pipeline-3C-SiC", name="3C-SiC", kind=CandidateKind.REFERENCE,
                     composition=sic_composition(), phases=[sic_3c_phase()],
                     hypothesis="cleavage for the md_interface slot of the pipeline")
    conv, _ = relax_bulk_lattice(model, sic_3c_conventional())
    out = run_cleavage(CleavageProtocol(plane_miller=miller, termination=termination, mode="rigid"), cand, model,
                       structure=conv)
    failed = [c.name for c in out.checks if not c.passed]
    if failed:
        raise RuntimeError("fracture-md numerical checks failed: " + ", ".join(failed))
    law = out.traction_separation[0]
    raw = out.raw
    status = {r.name.value: r.validation.status.value for r in out.properties}
    w_relaxed = float(raw["relaxed_surfaces"]["W_relaxed_J_m2"])
    peak_GPa = law.peak_traction.magnitude()
    cutoff = bool(raw["cutoff_artifact_rigid"]["flagged"])
    n_atoms = int(raw["slab"]["n_atoms"])
    plane, pot = args["plane"], args["potential"]
    name = POTENTIALS[pot]
    strength_note = ("set by the cutoff function of the potential, out of domain" if cutoff
                     else "status " + status.get("peak_normal_traction", "unknown"))
    method = (f"fracture-md rigid cleavage of perfect 3C-SiC on ({plane}) {termination}, {name} "
              f"({model.card.id}), {n_atoms}-atom slab, 0 K. Energy: work of separation with relaxed surfaces. "
              f"Strength: peak traction of the rigid opening, {strength_note}. One material split, "
              "no interface built, not validated against DFT or experiment.")
    curve_json = Path(args["out_dir"]) / f"cleavage_3C-SiC_{plane}_{pot}.json"
    with open(curve_json, "w") as fh:
        json.dump({"inputs": args, "method": method, "potential_card": model.card.id,
                   "a0_A": float(conv.cell.array[0, 0]), "slab": raw["slab"],
                   "curve": {"opening_A": raw["rigid"]["openings"], "energy_J_m2": raw["rigid"]["energy_per_area"],
                             "traction_GPa": raw["rigid"]["traction"]},
                   "uber_fit": raw["uber_rigid"], "cutoff_artifact": raw["cutoff_artifact_rigid"],
                   "relaxed_surfaces": raw["relaxed_surfaces"],
                   "records": [r.model_dump(mode="json") for r in out.properties],
                   "checks": [c.model_dump(mode="json") for c in out.checks]}, fh, indent=1, default=_plain)
    return {
        "separation_energy_J_m2": round(w_relaxed, 4),
        "strength_MPa": round(peak_GPa * 1000),
        "method": method,
        "rigid_separation_energy_J_m2": round(law.work_of_separation.magnitude(), 4),
        "opening_at_peak_nm": round(law.opening_at_peak.magnitude() / 10, 5),
        "uber_strength_MPa": round(float(raw["uber_rigid"]["sigma_max_GPa"]) * 1000),
        "uber_length_nm": round(float(raw["uber_rigid"]["l_A"]) / 10, 5),
        "strength_set_by_cutoff_function": cutoff,
        "separation_energy_status": status.get("work_of_separation", "unknown"),
        "strength_status": status.get("peak_normal_traction", "unknown"),
        "n_atoms": n_atoms,
        "wall_time_s": round(time.time() - t0, 1),
        "curve_json": str(curve_json),
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["--child"]:
        print(json.dumps(_compute(json.loads(sys.stdin.read()))))
    else:
        print(json.dumps(cleavage_for_slot(*sys.argv[1:]), indent=1))
