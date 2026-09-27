"""What the memory block costs at inference, per backbone.

Measured on the reported checkpoints (`memory/seed0/epoch_0200.pth`, whose BNNeck-only path is the
same forward a finetuned model runs). MagiV2 runs with its ViT-MAE patch masking off, as everywhere
in the evaluation: masking would drop three quarters of its patches and understate its forward time.
Three timings:

  * backbone: `model(x, use_memory=False)`, the backbone and BNNeck, the path every P1-P3 number uses;
  * memory: `model.identify_character(x, update_memory=True)`, the two-pass inference of the
    appendix algorithm, with memory initialised from one crop per identity exactly as
    `recognize.features.extract(mode="memory")` initialises it. As implemented it runs the backbone
    once per pass;
  * block only: the same two passes over cached class tokens, i.e. what the block itself adds once
    the backbone feature is computed once.

Batch size 1, FP32 (the evaluation's precision), CUDA-synchronised wall time per crop after a warm-up.
Parameters: the memory block's trainable parameters (its banks are buffers and are excluded), and
the backbone parameters its image forward actually touches (found with forward hooks, since some
released checkpoints ship modules the crop embedding never calls).

    python analysis/latency.py --device cuda --out results/latency.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from memory_block.training.dataset import get_transforms          # noqa: E402
from recognize import features as FT                              # noqa: E402
from recognize.backbones import BACKBONE_REGISTRY                 # noqa: E402
from recognize.data import SeriesStream                           # noqa: E402
from recognize.checkpoints import load_checkpoint          # noqa: E402

SERIES = "Bakuman"


def used_parameters(module: torch.nn.Module, run) -> int:
    """Parameters of the submodules that execute during `run()`, each counted once."""
    ran = set()
    hooks = [m.register_forward_hook(lambda m, i, o: ran.add(id(m))) for m in module.modules()]
    try:
        run()
    finally:
        for h in hooks:
            h.remove()
    seen, total = set(), 0
    for m in module.modules():
        if id(m) in ran:
            for p in m.parameters(recurse=False):
                if id(p) not in seen:
                    seen.add(id(p))
                    total += p.numel()
    return total


def timed(fn, crops, warmup: int, device: str) -> list:
    out = []
    for i, x in enumerate(crops):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        if i >= warmup:
            out.append(1000.0 * (time.perf_counter() - t0))
    return out


@torch.no_grad()
def measure(bb: str, device: str, n: int, warmup: int) -> dict:
    spec = BACKBONE_REGISTRY[bb]
    lm = load_checkpoint(f"checkpoints/{bb}/memory/seed0/epoch_0200.pth", device=device)
    model = lm.model.eval()
    tf = get_transforms(spec.height, spec.width, is_train=False, normalize_type=spec.normalize)
    stream = SeriesStream(Path("Datasets/popcharacters") / SERIES)
    labels = np.asarray(stream.labels)
    order = np.asarray(stream.reading_order)

    # one gallery crop per identity (first appearance), memory initialised as extract() does
    first = {}
    for i in order:
        first.setdefault(int(labels[i]), int(i))
    gallery_idx = np.asarray(sorted(first.values()))
    ds = FT._loader(stream, tf, None, 1, 0).dataset
    x_of = lambda i: ds[int(i)]["image"].unsqueeze(0).to(device)

    gal = torch.cat([x_of(i) for i in gallery_idx])
    gal_local = torch.as_tensor(np.unique(labels[gallery_idx], return_inverse=True)[1], device=device)

    def init_memory(inputs):
        gal_bn = model(inputs, use_memory=False)["bn_feat"]
        model.reinit_for_open_set(len(gallery_idx))
        model.memory_block.initialize_from_support(gal_bn, char_ids=gal_local, method="diverse")
        model.memory_block.working_memory.update(gal_bn, gal_local)
        gal_mem = model(inputs, char_ids=gal_local, use_memory=True, update_working=False,
                        update_episodic=False)["bn_feat"]
        model.memory_block.initialize_from_support(gal_mem, char_ids=gal_local, method="diverse")

    queries = [i for i in order if int(i) not in set(gallery_idx.tolist())][: n + warmup]
    crops = [x_of(i) for i in queries]
    two_pass = lambda x: model.identify_character(x, update_memory=True, return_features=True)

    t_backbone = timed(lambda x: model(x, use_memory=False), crops, warmup, device)
    init_memory(gal)
    t_memory = timed(two_pass, crops, warmup, device)

    # the block alone: the same two passes over cached class tokens, so the backbone runs zero times
    # instead of the implementation's twice (identify_character calls forward once per pass)
    wrapper = model.backbone_wrapper
    gal_tok = wrapper(gal)[1]
    tokens = [wrapper(x)[1] for x in crops]
    model.backbone_wrapper = FT._IdentityBackbone()
    try:
        init_memory(gal_tok)
        t_block = timed(two_pass, tokens, warmup, device)
    finally:
        model.backbone_wrapper = wrapper

    backbone_used = used_parameters(model.backbone_wrapper,
                                    lambda: model.backbone_wrapper(crops[0]))
    memory_trainable = sum(p.numel() for p in model.memory_block.parameters())
    memory_buffers = sum(b.numel() for b in model.memory_block.buffers())
    return {
        "backbone": bb,
        "feature_dim": spec.native_dim,
        "input": [spec.height, spec.width],
        "n_timed": len(t_backbone),
        "ms_backbone_median": statistics.median(t_backbone),
        "ms_backbone_mean": statistics.fmean(t_backbone),
        "ms_memory_median": statistics.median(t_memory),
        "ms_memory_mean": statistics.fmean(t_memory),
        "ms_block_only_median": statistics.median(t_block),
        "ms_block_only_mean": statistics.fmean(t_block),
        "params_backbone_used": backbone_used,
        "params_backbone_total": sum(p.numel() for p in model.backbone_wrapper.parameters()),
        "params_memory_trainable": memory_trainable,
        "memory_buffer_elements_training_size": memory_buffers,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["transreid", "magiv2", "magiv3", "instructreid", "reid5o"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("results/latency.json"))
    args = ap.parse_args(argv)

    torch.backends.cudnn.benchmark = False
    rows = [measure(bb, args.device, args.n, args.warmup) for bb in args.backbones]
    gpu = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"
    out = {"device": gpu, "series": SERIES, "batch_size": 1, "precision": "fp32", "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[latency] {gpu}")
    print(f"{'backbone':<13}{'ms bb':>8}{'ms mem':>8}{'ms blk':>8}{'bb used':>10}{'bb total':>10}{'mem':>8}")
    for r in rows:
        print(f"{r['backbone']:<13}{r['ms_backbone_median']:>8.2f}{r['ms_memory_median']:>8.2f}"
              f"{r['ms_block_only_median']:>8.2f}"
              f"{r['params_backbone_used']/1e6:>9.1f}M{r['params_backbone_total']/1e6:>9.1f}M"
              f"{r['params_memory_trainable']/1e6:>7.2f}M")
    print(f"[latency] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
