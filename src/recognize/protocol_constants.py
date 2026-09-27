"""Protocol constants for the Re:Cognize evaluation protocols (P1-P4).

This module is the single source of truth for protocol-level constants
shared across training, evaluation, and provenance stamping. Values here
must not be redefined or hard-coded elsewhere in the codebase.
"""

from __future__ import annotations

#: Novelty threshold for P3 sequential clustering (cosine similarity).
TAU_NOV = 0.55

#: Per-identity FIFO gallery buffer capacity for P4 gallery growth.
B_MAX = 50

#: Fraction of each identity's crops placed in the P1 gallery (the rest are queries).
GALLERY_RATIO = 0.2

#: Random seeds used for all reported protocol averages.
EVAL_SEEDS = (0, 1, 2, 3, 4)

#: Fractional padding applied to bounding-box crops.
CROP_PADDING = 0.10

#: Per-identity seed counts (K) swept in P2 seeded-gallery evaluation.
K_RANGE = (1, 2, 3, 4, 5)
