"""Make the third-party repositories in `thirdp/` importable.

DUSt3R (with CroCo) and VGGT are imported by their own package names, e.g.
`dust3r.losses`, `models.croco` (CroCo), `vggt.layers` and `training.loss`
(VGGT's training losses).
"""

import sys

from pathlib import Path

THIRDP_ROOT = Path(__file__).resolve().parents[1] / "thirdp"

for _path in ("dust3r", "dust3r/croco", "vggt", "vggt/training"):
    if str(THIRDP_ROOT / _path) not in sys.path:
        sys.path.append(str(THIRDP_ROOT / _path))
