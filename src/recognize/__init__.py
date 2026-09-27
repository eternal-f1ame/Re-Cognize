"""Re:Cognize evaluation support package.

Single source of truth for protocol-level constants (`protocol_constants`)
and the provenance stamp embedded in every result JSON (`provenance`).
"""

from .protocol_constants import (
    B_MAX,
    CROP_PADDING,
    EVAL_SEEDS,
    GALLERY_RATIO,
    K_RANGE,
    TAU_NOV,
)
from .provenance import assert_hashseed_pinned, stamp

__all__ = [
    "TAU_NOV",
    "B_MAX",
    "GALLERY_RATIO",
    "EVAL_SEEDS",
    "CROP_PADDING",
    "K_RANGE",
    "stamp",
    "assert_hashseed_pinned",
]
