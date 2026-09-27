"""Checked state-dict loading.

`torch.nn.Module.load_state_dict(..., strict=False)` silently ignores missing
and unexpected keys, which is how an un-loaded BNNeck or adapter can go
unnoticed. This loader is the only place `strict=False` may appear: it loads
non-strictly, then raises on anything that is not explicitly allowed.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import torch
from torch import nn


def load_state_dict_checked(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    allow_missing_prefixes: Iterable[str] = (),
    allow_unexpected_prefixes: Iterable[str] = (),
) -> None:
    """Load `state_dict` into `model`, raising on any unexplained mismatch.

    Args:
        model: destination module.
        state_dict: source parameters and buffers.
        allow_missing_prefixes: model keys starting with any of these prefixes
            may be absent from `state_dict` (e.g. a classification head the
            checkpoint never had). Document every prefix at the call site.
        allow_unexpected_prefixes: checkpoint keys starting with any of these
            prefixes may be absent from the model.

    Raises:
        RuntimeError: listing shape mismatches, disallowed missing keys, and
            disallowed unexpected keys.
    """
    allow_missing = tuple(allow_missing_prefixes)
    allow_unexpected = tuple(allow_unexpected_prefixes)
    model_state = model.state_dict()

    shape_mismatch = [
        k for k, v in state_dict.items()
        if k in model_state and tuple(v.shape) != tuple(model_state[k].shape)
    ]
    if shape_mismatch:
        raise RuntimeError(f"shape mismatch for keys: {shape_mismatch}")

    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(allow_missing)]
    unexpected = [k for k in result.unexpected_keys if not k.startswith(allow_unexpected)]
    problems = []
    if missing:
        problems.append(f"missing keys not covered by allow_missing_prefixes: {missing}")
    if unexpected:
        problems.append(f"unexpected keys not covered by allow_unexpected_prefixes: {unexpected}")
    if problems:
        raise RuntimeError("; ".join(problems))
