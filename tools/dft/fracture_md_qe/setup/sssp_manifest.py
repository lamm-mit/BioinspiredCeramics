"""Verify an extracted SSSP library and write its provenance manifest (called by setup_dft_env.sh in this folder).

For each family SSSP_<version>_PBE_{efficiency,precision} under --root:
  * every element entry of <family>.json must point at a file whose md5 equals the md5 in the JSON;
  * the sha256 of every UPF (used by engines.dft_qe in record provenance) is computed.
Writes <root>/MANIFEST_SSSP_<version>_PBE.json and <root>/SHA256SUMS (paths relative to <root>).
Standard library only, so it runs with any python 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--version", default="1.3.0")
    ap.add_argument("--record", default="rcyfm-68h65")
    ap.add_argument("--doi", default="10.24435/materialscloud:f3-ym")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    manifest = {
        "library": f"SSSP {a.version} PBE (efficiency and precision)",
        "source": f"https://archive.materialscloud.org/records/{a.record}",
        "doi": a.doi,
        "citation": "G. Prandini, A. Marrazzo, I. E. Castelli, N. Mounet, N. Marzari, npj Comput. Mater. 4, 72 "
                    "(2018), doi:10.1038/s41524-018-0127-2; library DOI " + a.doi,
        "licence": "CC-BY-4.0 for the library; each pseudopotential keeps the licence of its family "
                   "(GBRV GPL-3; SG15 CC-BY-SA-4.0; PseudoDojo CC-BY-4.0; pslibrary GPL-2+; Goedecker CC-BY-3.0; "
                   "Wentzcovitch GPL-3); see SSSP_LICENSE.txt",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "families": {},
        "archives": {},
    }
    dl = root / "_downloads"
    if dl.is_dir():
        for f in sorted(dl.iterdir()):
            if f.is_file() and not f.name.endswith(".part"):
                manifest["archives"][f.name] = {"md5": _digest(f, "md5"), "sha256": _digest(f, "sha256"),
                                                "bytes": f.stat().st_size}
    bad = []
    sums = []
    for proto in ("efficiency", "precision"):
        name = f"SSSP_{a.version}_PBE_{proto}"
        meta_path = root / f"{name}.json"
        fam_dir = root / name
        if not meta_path.is_file() or not fam_dir.is_dir():
            print(f"missing {meta_path} or {fam_dir}", file=sys.stderr)
            return 1
        meta = json.loads(meta_path.read_text())
        fam = {"directory": name, "json": meta_path.name, "json_sha256": _digest(meta_path, "sha256"),
               "elements": {}}
        for el, e in sorted(meta.items()):
            p = fam_dir / e["filename"]
            if not p.is_file():
                bad.append(f"{name}: {el} -> {e['filename']} missing")
                continue
            md5 = _digest(p, "md5")
            if md5 != e["md5"]:
                bad.append(f"{name}: {e['filename']} md5 {md5} != {e['md5']}")
            sha = _digest(p, "sha256")
            fam["elements"][el] = {"filename": e["filename"], "md5": md5, "sha256": sha,
                                   "pseudopotential": e.get("pseudopotential"),
                                   "cutoff_wfc_Ry": e.get("cutoff_wfc"), "cutoff_rho_Ry": e.get("cutoff_rho")}
            sums.append(f"{sha}  {name}/{e['filename']}")
        manifest["families"][name] = fam
    if bad:
        print("SSSP verification FAILED:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 1
    out = root / f"MANIFEST_SSSP_{a.version}_PBE.json"
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    (root / "SHA256SUMS").write_text("\n".join(sorted(set(sums))) + "\n")
    n = sum(len(f["elements"]) for f in manifest["families"].values())
    print(f"[sssp_manifest] verified {n} element entries (md5 vs library JSON); wrote {out.name} and SHA256SUMS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
