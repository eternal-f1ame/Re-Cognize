"""Feature extraction for the evaluation protocols, with or without the memory block.

`extract()` turns a series stream into one unit-norm row per crop, in stream
order. In memory mode the memory block is initialised from the gallery crops
only, the gallery is re-extracted through memory with its own labels, and every
other crop is processed in stream order by the model's two-pass routine
(`identify_character`) without ever seeing its label. The initialisation's one
random draw is seeded by the gallery, so a memory run depends only on the model,
the stream and the gallery.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


@dataclass
class Features:
    bn: np.ndarray                          # (N, D) unit vectors, in stream order
    pred_identity: Optional[np.ndarray]     # (N,) memory mode: pass-1 identity per query crop, -1 on gallery crops


class _IdentityBackbone(nn.Module):
    """Stands in for the backbone wrapper when class tokens are already computed.

    `MemoryEnhancedReID.forward` uses only the wrapper's second output, so returning the input
    reproduces the same arithmetic from the BNNeck onwards.
    """

    def forward(self, x: torch.Tensor):
        return None, x


class _TokenDataset(torch.utils.data.Dataset):
    """Serves cached class tokens in the place of images, keeping the stream's indices."""

    def __init__(self, tokens: torch.Tensor):
        self.tokens = tokens

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {"image": self.tokens[i], "index": i}


class _L2Neck(nn.Module):
    """BNNeck replacement for pretrained rows: identity map followed by L2 normalisation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1)


def identity_bnneck(model: nn.Module) -> nn.Module:
    """Replace `model.bnneck` by the identity map (+L2). Returns the same model."""
    model.bnneck = _L2Neck()
    return model


def _loader(stream, transform, indices, batch_size, num_workers, dataset=None) -> DataLoader:
    if dataset is None and isinstance(transform, torch.utils.data.Dataset):
        dataset = transform                      # cached-token path passes a dataset in place of a transform
    ds = dataset if dataset is not None else stream.to_dataset(transform)
    if indices is not None:
        ds = Subset(ds, list(indices))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def backbone_tokens(model, stream, transform, *, batch_size: int = 64, device: str = "cpu",
                    num_workers: int = 0) -> torch.Tensor:
    """Class token of every crop, in stream order.

    Memory-mode evaluation re-extracts features once per gallery configuration (55 of them for the
    full grid), but the backbone output does not depend on the gallery: only the BNNeck and memory
    do. Computing the tokens once turns 55 backbone passes into one.
    """
    out = []
    for batch in _loader(stream, transform, None, batch_size, num_workers):
        _, cls = model.backbone_wrapper(batch["image"].to(device))
        out.append(cls.detach().float().cpu())
    return torch.cat(out)


def _unit(x: torch.Tensor) -> np.ndarray:
    return F.normalize(x.detach().float().cpu(), p=2, dim=1).numpy()


@torch.no_grad()
def _extract_plain(model, loader, device, char_ids_for=None, use_memory=False) -> torch.Tensor:
    """Run the model without memory (or with memory routed by `char_ids_for(index)`), in loader order."""
    out = []
    for batch in loader:
        imgs = batch["image"].to(device)
        if use_memory:
            cids = char_ids_for(batch["index"]).to(device)
            o = model(imgs, char_ids=cids, use_memory=True, update_working=False, update_episodic=False)
        else:
            o = model(imgs, use_memory=False)
        out.append(o["bn_feat"].detach().float().cpu())
    return torch.cat(out) if out else torch.zeros(0)


def extract(
    model: nn.Module,
    stream,
    transform,
    *,
    mode: Literal["none", "memory"] = "none",
    gallery_idx: Optional[Sequence[int]] = None,
    batch_size: int = 64,
    device: str = "cpu",
    num_workers: int = 0,
    tokens: Optional[torch.Tensor] = None,
) -> Features:
    """Unit-norm features for every crop of `stream`, in stream order.

    mode="none": backbone + BNNeck only (memory bypassed).
    mode="memory": memory initialised from `gallery_idx` only; see module docstring.

    `tokens` are class tokens for the whole stream from `backbone_tokens()`. They do not depend on
    the gallery, so a caller that evaluates many gallery configurations of one series should compute
    them once: on TransReID/Bakuman the backbone costs 89 s and a memory pass over cached tokens
    0.6 s, so the 55 configurations of the full grid take 2 minutes instead of 82.
    """
    model.eval()
    if tokens is not None:
        original_wrapper = model.backbone_wrapper
        model.backbone_wrapper = _IdentityBackbone()
        try:
            return extract(model, stream, _TokenDataset(tokens), mode=mode, gallery_idx=gallery_idx,
                           batch_size=batch_size, device=device, num_workers=0)
        finally:
            model.backbone_wrapper = original_wrapper
    n = stream.n_crops
    if mode == "none":
        bn = _unit(_extract_plain(model, _loader(stream, transform, None, batch_size, num_workers), device))
        return Features(bn=bn, pred_identity=None)
    if mode != "memory":
        raise ValueError(f"mode must be 'none' or 'memory', got {mode!r}")
    if gallery_idx is None:
        raise ValueError("memory mode requires gallery_idx (the crops whose labels may initialise memory)")
    if getattr(model, "memory_block", None) is None:
        raise ValueError("memory mode requires a model with a memory block")

    gallery_idx = np.asarray(sorted(int(i) for i in gallery_idx), dtype=np.int64)
    labels = np.asarray(stream.labels)
    gal_labels = labels[gallery_idx]
    # Local identity space: only identities present in the gallery get memory slots.
    uniq, gal_local = np.unique(gal_labels, return_inverse=True)
    n_ident = len(uniq)
    local_of_index = {int(i): int(l) for i, l in zip(gallery_idx, gal_local)}

    def char_ids_for(batch_index: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor([local_of_index[int(i)] for i in batch_index], dtype=torch.long)

    gal_loader = _loader(stream, transform, gallery_idx, batch_size, num_workers)
    gal_local_t = torch.as_tensor(gal_local, dtype=torch.long)

    # The prototypes are chosen by farthest-point sampling from a random first crop. That draw comes
    # from the CPU generator, seeded here by the gallery; the generator's state is restored after.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(zlib.crc32(",".join(map(str, gallery_idx.tolist())).encode()))
        # 1) bootstrap: clean gallery features -> prototypes + working-memory prime
        gal_bn = _extract_plain(model, gal_loader, device)
        model.reinit_for_open_set(n_ident)
        model.memory_block.initialize_from_support(gal_bn.to(device), char_ids=gal_local_t.to(device), method="diverse")
        model.memory_block.working_memory.update(gal_bn.to(device), gal_local_t.to(device))
        # 2) refine: gallery through memory routed by its own (gallery) labels; prototypes overwritten
        gal_mem = _extract_plain(model, gal_loader, device, char_ids_for=char_ids_for, use_memory=True)
        model.memory_block.initialize_from_support(gal_mem.to(device), char_ids=gal_local_t.to(device), method="diverse")

    bn = np.zeros((n, gal_mem.shape[1]), dtype=np.float32)
    pred = np.full(n, -1, dtype=np.int64)
    bn[gallery_idx] = _unit(gal_mem)

    # 3) every other crop, in stream order, two-pass routing, no labels
    query_idx = np.setdiff1d(np.arange(n), gallery_idx)
    if len(query_idx):
        q_loader = _loader(stream, transform, query_idx, batch_size, num_workers)
        for batch in q_loader:
            imgs = batch["image"].to(device)
            idx = batch["index"].numpy()
            r = model.identify_character(imgs, update_memory=True, return_features=True)
            bn[idx] = _unit(r["features"])
            pred[idx] = uniq[r["predictions"].detach().cpu().numpy()]
    return Features(bn=bn, pred_identity=pred)
