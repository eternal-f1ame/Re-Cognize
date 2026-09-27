"""Protocol drivers (P1-P4). Each takes pre-extracted unit-norm features and stream labels/order."""
from ._seeds import split_seeds
from .p1_closedset import p1_split, run_p1
from .p2_seeded import run_p2
from .p3_online import run_p3
from .p4_growth import UPDATE_POLICIES, run_p4
from .rules import RULE_GRIDS, RULES

__all__ = ["split_seeds", "run_p1", "p1_split", "run_p2", "run_p3", "run_p4", "RULES", "RULE_GRIDS", "UPDATE_POLICIES"]
