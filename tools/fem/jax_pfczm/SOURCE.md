# Source

This folder is a copy of `src/jax_pfczm` from [jaime-berkovich/jax-pfczm](https://github.com/jaime-berkovich/jax-pfczm) at commit bb76d8b (bb76d8b05fdf09568918195777213557ba064c00). The files are unchanged. The package is under the MIT licence in [LICENSE](LICENSE), copyright Jaime Berkovich.

The README of that repository gives the model, the API, the solver tolerances and the verification results. `tools/fem/cell.py` uses the package. To update it, copy `src/jax_pfczm` of a newer commit over this folder and change the commit above.

Importing the package turns on float64 in JAX (`jax_enable_x64`) for the whole process.
