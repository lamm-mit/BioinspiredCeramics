"""``python -m jax_pfczm [--level quick|production] [--out file.json]``: run the verification suite."""

from .verification import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
