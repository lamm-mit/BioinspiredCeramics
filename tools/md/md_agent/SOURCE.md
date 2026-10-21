# Source

This folder is a copy of the Python files, the README and the requirements of [azhwang/md_agent](https://github.com/azhwang/md_agent) at commit 3582c28 (3582c28211c65de60512174f13c8b47b385f4d92), the MD code of Alex Wang. The files are unchanged.

The source repository has no licence file. The code is included here with the agreement of the team, and Alex Wang owns it. Ask him before using it outside this project.

The pipeline does not import it yet. `tools/md/__init__.py` keeps its stand-in, so the optional dependencies of md_agent (OpenMM, ASE, MACE and PyTorch) are not needed to run the pipeline. Its modules import each other by name, so run them from this folder, as its README describes.
