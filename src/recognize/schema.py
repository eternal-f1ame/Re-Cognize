"""Result JSON schema (`scripts/evaluate.py` output).

Layout of one `<out>/<series>.json`:
    provenance: recognize.provenance.stamp(...)
    series:     {name, n_crops, n_identities, reading_order_source, box_noise, pixel_noise, box_iou_mean}
    p1:         {seed: metrics}
    p2:         {strategy: {k: {seed: metrics}}}
    p4:         {strategy: {k: {seed: {update_policy: {b_max: metrics}}}}}
    p3:         {rule: metrics}
Only the protocol blocks named in provenance.args.protocols are required.
"""
from __future__ import annotations

from typing import Any, Dict

# stamp() also records the git commit and tree state of a run; a result is complete without them,
# and the published results carry neither.
REQUIRED_PROVENANCE = ("args", "seeds", "pythonhashseed", "checkpoint_sha256", "n_crops", "n_identities")
REQUIRED_SERIES = ("name", "n_crops", "n_identities")
PROTOCOLS = ("p1", "p2", "p3", "p4")


def _leaves(block: Any):
    """Yield the metric dicts at the bottom of a nested result block."""
    if isinstance(block, dict) and ("mAP" in block or "clusters" in block):
        yield block
        return
    if isinstance(block, dict):
        for v in block.values():
            yield from _leaves(v)


def validate(result: Dict[str, Any]) -> None:
    """Raise ValueError if `result` is not a complete result."""
    if not isinstance(result, dict):
        raise ValueError("result must be a dict")
    prov = result.get("provenance")
    if not isinstance(prov, dict):
        raise ValueError("missing provenance block")
    missing = [k for k in REQUIRED_PROVENANCE if k not in prov]
    if missing:
        raise ValueError(f"provenance missing keys: {missing}")
    if not prov.get("pythonhashseed"):
        raise ValueError("provenance.pythonhashseed is empty; PYTHONHASHSEED must be pinned")
    series = result.get("series")
    if not isinstance(series, dict) or any(k not in series for k in REQUIRED_SERIES):
        raise ValueError(f"series block must contain {REQUIRED_SERIES}")
    requested = prov["args"].get("protocols") if isinstance(prov.get("args"), dict) else None
    if not requested:
        raise ValueError("provenance.args.protocols is empty")
    for p in requested:
        if p not in PROTOCOLS:
            raise ValueError(f"unknown protocol {p!r} in provenance.args.protocols")
        block = result.get(p)
        if not isinstance(block, dict) or not block:
            raise ValueError(f"missing protocol block {p!r}")
        leaves = list(_leaves(block))
        if not leaves:
            raise ValueError(f"protocol block {p!r} has no metrics")
        key = "clusters" if p == "p3" else "mAP"
        if any(key not in leaf for leaf in leaves):
            raise ValueError(f"protocol block {p!r} has a leaf without {key!r}")
