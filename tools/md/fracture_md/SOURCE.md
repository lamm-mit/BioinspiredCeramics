# Source of this copy

This folder holds a copy of the `fracture_md` Python package of fracture-md, so that the pipeline can install it with `pip install -e tools/md/fracture_md`.

| | |
|---|---|
| Repository | https://github.com/jaime-berkovich/fracture-md (private) |
| Author | Jaime Berkovich |
| Licence | MIT, in [LICENSE](LICENSE) |
| Commit | `bffe3e1c7aa1ad4f2f21bf2c08aa7b09ff9ac795` (2026-10-13), plus one uncommitted fix |
| The fix | `engines/dft_qe.py` accepts both "A final scf calculation at the relaxed structure" (QE before 7.5) and "Final scf calculation at the relaxed structure" (QE 7.5) when it checks a vc-relax. Without it, every QE 7.5 vc-relax is reported as not converged. sha256 of the file: `84f03f285fa31bb9e20c339f2c17da0f3cb2c328f22104117cfa8ad15d291fd5` |
| Copied | 2026-10-21 |
| Code hash | `fracture_md.provenance.source_tree_sha256()` = `d43b8b6127e304fac20340ad82e56afaeda713116b94d9c9064906b9e63070f0` |

## What was copied

- `src/fracture_md/`, all 84 files, without `__pycache__`.
- `pyproject.toml`, with the same pinned dependencies. Three changes: the README is the one in this folder, the test extra and the pytest settings are dropped because the tests stay in fracture-md, and the package-data rule is dropped because the package holds no data files.
- `LICENSE`, unchanged.

## What was not copied

Tests, examples, scripts, docs, data, weights and outputs. The cleavage, elastic, ideal-strength, potential and DFT modules need no data files. The modules that read the `data/` folder of fracture-md (the dental composites, the hardness models, the indentation recipes) do not work from this copy.

The MACE-MP-0 weights and the SSSP pseudopotentials are downloaded, never committed. [README.md](README.md) and [../../dft/fracture_md_qe/README.md](../../dft/fracture_md_qe/README.md) say how to get them.

## Provenance of records

A record written from this copy names the commit of this repository, because `fracture_md.provenance` asks the version control of the folder it sits in. The code hash above identifies the fracture-md code itself.

## Updating

Copy `src/fracture_md/` again from a newer fracture-md commit, keep this `pyproject.toml` unless the pins changed, and update the commit, the date and the code hash above.
