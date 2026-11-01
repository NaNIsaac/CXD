"""Moved to ``splotter gamdel`` in https://github.com/NaNIsaac/splotter."""

import sys
import warnings

warnings.warn(
    "cxd.gam_del moved to splotter gamdel in https://github.com/NaNIsaac/splotter; "
    "this module will be removed in the release after 0.1",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    sys.exit("cxd.gam_del moved: install splotter and run `splotter gamdel`")
