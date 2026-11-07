"""Multi-view transformers (MVTs) trained from scratch on top of a frozen encoder."""

from mvt import _thirdp  # noqa: F401
from mvt.model import MVT

__all__ = ["MVT"]
