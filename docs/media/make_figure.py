"""Draw docs/media/rocking_q.png: the synthetic rocking scan, its rocking curve and its q-space map.

uv run --with matplotlib docs/media/make_figure.py
"""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from cxd.core.rocking import map_rocking_scan

ROOT = Path(__file__).resolve().parents[2]
ENERGY_KEV = 9.0  # not in the synthetic SPEC file; the value tests/test_rocking.py uses
PIXEL_MM = 0.055
DELTA, GAMMA = 60.0, 0.0  # the synthetic SPEC file's Delta = 2 THETA0

spec = importlib.util.spec_from_file_location("synthetic", ROOT / "examples/synthetic_scan.py")
synthetic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synthetic)

frames = synthetic.make_volume().astype(float)
n = synthetic.N_FRAMES
thetas = synthetic.THETA0 + synthetic.THETA_STEP * (np.arange(n) - n // 2)
center = (n // 2, synthetic.ROI // 2, synthetic.ROI // 2)
m = map_rocking_scan(frames, thetas, DELTA, GAMMA, center, synthetic.SDD, PIXEL_MM, ENERGY_KEV)

fig, (det, rock, qmap) = plt.subplots(1, 3, figsize=(7.2, 2.6), dpi=150, layout="constrained")
det.imshow(
    frames[n // 2], norm=LogNorm(vmin=1), cmap=plt.get_cmap("magma").with_extremes(bad="black")
)
det.set_title(f"detector ROI, frame {n // 2}", fontsize=8)
det.set_xlabel("x (px)", fontsize=8)
det.set_ylabel("y (px)", fontsize=8)

rock.semilogy(thetas, frames.sum(axis=(1, 2)), ".-", ms=3, lw=0.8)
rock.set_title("rocking curve", fontsize=8)
rock.set_xlabel("θ (°)", fontsize=8)
rock.set_ylabel("ROI counts", fontsize=8)
rock.ticklabel_format(axis="x", useOffset=False)

# mean of the measured bins along q_y; bins never measured are left out, not averaged as zero
hits = m.hits.sum(axis=1)
projection = np.divide(
    (m.intensity * (m.hits > 0)).sum(axis=1),
    (m.hits > 0).sum(axis=1),
    out=np.full(hits.shape, np.nan),
    where=hits > 0,
)
qx, _, qz = m.edges
qmap.pcolormesh(qx, qz, projection.T, norm=LogNorm(vmin=0.1), cmap="magma")
qmap.set_title("map_rocking_scan, mean over q$_y$", fontsize=8)
qmap.set_xlabel("q$_x$ (Å$^{-1}$)", fontsize=8)
qmap.set_ylabel("q$_z$ (Å$^{-1}$)", fontsize=8)
qmap.set_aspect("equal")

for ax in (det, rock, qmap):
    ax.tick_params(labelsize=7)
fig.savefig(ROOT / "docs/media/rocking_q.png")
