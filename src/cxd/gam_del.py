"""Moved to ``scripts/gam_del.py``; run ``python -m scripts.gam_del`` from a CXD checkout."""

import sys
import warnings

warnings.warn(
    "cxd.gam_del moved to scripts/gam_del.py (python -m scripts.gam_del); "
    "this module will be removed in the release after 0.1",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    sys.exit("cxd.gam_del moved: run `python -m scripts.gam_del` from a CXD checkout")
