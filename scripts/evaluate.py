#!/usr/bin/env python3
"""Evaluation entry point for the four protocols, P1 to P4 (defined in docs/protocols.md).

    PYTHONHASHSEED=0 python scripts/evaluate.py --checkpoint checkpoints/magiv2/memory/seed0/final.pth \
        --series Bakuman --protocols p1 p2 p3 p4 --seeds 0 1 2 3 4 \
        --out results/popcharacters/magiv2_memory_seed0
    PYTHONHASHSEED=0 python scripts/evaluate.py --pretrained magiv2 --split test

Writes one schema-validated JSON per series, with a provenance stamp, under --out. The default is
results/<tag>/, where the tag names the pretrained backbone or the checkpoint's run
(checkpoints/magiv2/memory/seed0/final.pth gives magiv2__memory__seed0). Memory-aware runs
re-extract features per gallery (memory is initialised from the gallery only).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _config import BACKBONE_REGISTRY, DATASET_REGISTRY, safe_name  # noqa: E402
from recognize.data import SeriesStream, load_split  # noqa: E402
from recognize.features import backbone_tokens, extract, identity_bnneck  # noqa: E402
from recognize.checkpoints import LoadedModel, current_mask_ratio, disable_mae_masking, load_checkpoint  # noqa: E402
from recognize.perturb import BOX_KINDS, PIXEL_KINDS  # noqa: E402
from recognize.protocol_constants import B_MAX, EVAL_SEEDS, K_RANGE, TAU_NOV  # noqa: E402
from recognize.protocols import RULES, UPDATE_POLICIES, p1_split, run_p1, run_p2, run_p3, run_p4, split_seeds  # noqa: E402
from recognize.provenance import assert_hashseed_pinned, stamp  # noqa: E402
from recognize.schema import validate  # noqa: E402

B_MAX_SWEEP = (0, 5, 10, 25, 50, 100, None)          # None = unbounded


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="trained checkpoint (best.pth)")
    src.add_argument("--pretrained", choices=sorted(BACKBONE_REGISTRY), help="frozen backbone, identity BNNeck")
    ap.add_argument("--dataset", default="popcharacters", choices=sorted(DATASET_REGISTRY))
    ap.add_argument("--data-root", type=Path, default=None, help="override the dataset directory (series subdirs)")
    ap.add_argument("--split", default="test", choices=("dev", "test"), help="series list when --series is omitted")
    ap.add_argument("--series", nargs="+", default=None)
    ap.add_argument("--protocols", nargs="+", default=["p1", "p2", "p3", "p4"], choices=("p1", "p2", "p3", "p4"))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(EVAL_SEEDS))
    ap.add_argument("--k", nargs="+", type=int, default=list(K_RANGE))
    ap.add_argument("--strategies", nargs="+", default=["random", "temporal"], choices=("random", "temporal"))
    ap.add_argument("--b-max", type=int, default=B_MAX)
    ap.add_argument("--b-max-sweep", action="store_true", help=f"also run B_max in {B_MAX_SWEEP}")
    ap.add_argument("--update-policies", nargs="+", default=["predicted"], choices=UPDATE_POLICIES)
    ap.add_argument("--p3-rules", nargs="+", default=["fixed"], choices=sorted(RULES))
    ap.add_argument("--tau", type=float, default=TAU_NOV, help="P3 fixed-rule threshold")
    ap.add_argument("--box-noise", choices=BOX_KINDS, default=None)
    ap.add_argument("--pixel-noise", choices=PIXEL_KINDS, default=None)
    ap.add_argument("--noise-seed", type=int, default=0)
    mem = ap.add_mutually_exclusive_group()
    mem.add_argument("--memory", dest="memory", action="store_true", default=None)
    mem.add_argument("--no-memory", dest="memory", action="store_false")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default results/<tag>/)")
    ap.add_argument("--reuse-cached", action="store_true", help="skip series whose valid JSON already exists")
    return ap


# --------------------------------------------------------------------------- model
def construct_model(config):
    """Build a MemoryEnhancedReID from a MemoryConfig (separate so tests can stub it)."""
    from memory_block.models.model import MemoryEnhancedReID
    return MemoryEnhancedReID(config)


def build_model(args, device) -> LoadedModel:
    """A trained checkpoint, or a frozen pretrained backbone (identity BNNeck)."""
    from memory_block.models.model import MemoryConfig
    if args.checkpoint is not None:
        return load_checkpoint(args.checkpoint, device=device, construct=construct_model)
    backbone = args.pretrained
    spec = BACKBONE_REGISTRY[backbone]
    use_mem = bool(args.memory)
    # Weights, revision, stride and config paths come from the registry inside the builders.
    config = MemoryConfig(
        backbone_type=backbone, num_classes=1, feat_dim=spec.native_dim, freeze_backbone=True,
        use_working_memory=use_mem, use_episodic_memory=use_mem,
        image_height=spec.height, image_width=spec.width,
    )
    model = identity_bnneck(construct_model(config))
    mask_found = disable_mae_masking(model)
    model = model.to(device).eval()
    flags = {"training_mode": "pretrained", "lora": False, "memory": use_mem, "mask_ratio_at_training": mask_found, "mask_ratio": current_mask_ratio(model),
             "config_label": "Pretrained+Memory" if use_mem else "Pretrained", "tag": f"pretrained__{backbone}"}
    return LoadedModel(model=model, backbone=backbone, checkpoint=None, normalize=spec.normalize, flags=flags)


def build_transform(backbone: str, normalize: str):
    from memory_block.training.dataset import get_transforms
    spec = BACKBONE_REGISTRY[backbone]
    return get_transforms(spec.height, spec.width, is_train=False, normalize_type=normalize)


# --------------------------------------------------------------------------- one series
def evaluate_series(model, transform, stream: SeriesStream, args, device: str, use_memory: bool) -> Dict:
    labels, order = stream.labels, stream.reading_order
    # The backbone output does not depend on the gallery, so it runs once per series and every
    # gallery configuration reuses its class tokens (82 minutes -> 2 minutes for a memory row).
    tokens = backbone_tokens(model, stream, transform, batch_size=args.batch_size, device=device,
                             num_workers=args.num_workers)
    kw = dict(batch_size=args.batch_size, device=device, num_workers=args.num_workers, tokens=tokens)
    _plain = {}

    def plain():
        if "f" not in _plain:
            _plain["f"] = extract(model, stream, transform, mode="none", **kw).bn
        return _plain["f"]

    def feats_for(gallery_idx):
        if not use_memory:
            return plain()
        return extract(model, stream, transform, mode="memory", gallery_idx=gallery_idx, **kw).bn

    out: Dict = {}
    if "p1" in args.protocols:
        out["p1"] = {}
        for seed in args.seeds:
            g_idx, _, _ = p1_split(labels, seed)
            out["p1"][str(seed)] = run_p1(feats_for(g_idx), labels, seed)
    if "p2" in args.protocols or "p4" in args.protocols:
        b_maxes = list(dict.fromkeys([args.b_max] + (list(B_MAX_SWEEP) if args.b_max_sweep else [])))
        if "p2" in args.protocols:
            out["p2"] = {s: {str(k): {} for k in args.k} for s in args.strategies}
        if "p4" in args.protocols:
            out["p4"] = {s: {str(k): {} for k in args.k} for s in args.strategies}
        for strategy in args.strategies:
            for k in args.k:
                for seed in args.seeds:
                    seed_map, _, gallery_only = split_seeds(labels, order, k, strategy, seed)
                    g_idx = [i for v in seed_map.values() for i in v] + [i for v in gallery_only.values() for i in v]
                    f = feats_for(g_idx)
                    if "p2" in args.protocols:
                        out["p2"][strategy][str(k)][str(seed)] = run_p2(f, labels, order, k, strategy, seed)
                    if "p4" in args.protocols:
                        cell = out["p4"][strategy][str(k)][str(seed)] = {}
                        for policy in args.update_policies:
                            cell[policy] = {}
                            for b in b_maxes:
                                cell[policy]["inf" if b is None else str(b)] = run_p4(
                                    f, labels, order, k, strategy, seed, b_max=b, update_policy=policy)
    if "p3" in args.protocols:
        # P3 has no seed images, so there is no gallery to initialise the memory from: P3 runs on
        # no-memory features, and the result records that under "_features".
        out["p3"] = {}
        for rule in args.p3_rules:
            params = {"tau": args.tau} if rule == "fixed" else {}
            out["p3"][rule] = run_p3(plain(), labels, order, rule=rule, **params)
        out["p3"]["_features"] = "no-memory"
    return out


def split_for(dataset: str, split: str) -> list:
    """Series of a dataset's split. POPCharacters has a train/dev/test file; other corpora are
    evaluated on their held-out list from the registry (they are never trained on here)."""
    if dataset == "popcharacters":
        return load_split()[split]
    spec = DATASET_REGISTRY[dataset]
    if split == "dev":
        raise SystemExit(f"{dataset} has no dev split; pass --series or --split test")
    return list(spec.test_manga)


def output_tag(args, backbone: str) -> str:
    if args.checkpoint is None:
        return f"pretrained__{backbone}" + ("__memory" if args.memory else "")
    parts = Path(args.checkpoint).resolve().parts
    try:
        i = parts.index("checkpoints")
        return "__".join(parts[i + 1:-1])          # e.g. magiv2__memory__seed0
    except ValueError:
        return Path(args.checkpoint).stem


def write_atomic(path: Path, text: str) -> None:
    """Write to a temporary file in the same directory, then rename onto the target.

    Two array tasks can end up pointed at one tag (a resubmission that overlaps a running array,
    say), and a half-written JSON is worse than a repeated one: `--reuse-cached` and every
    reporting script read whatever is on disk. os.replace is atomic within a filesystem.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main(argv=None) -> int:
    assert_hashseed_pinned()
    args = build_parser().parse_args(argv)
    device = args.device
    loaded = build_model(args, device)
    model, backbone, ckpt_path = loaded.model, loaded.backbone, loaded.checkpoint
    use_memory = (getattr(model, "memory_block", None) is not None) if args.memory is None else bool(args.memory)
    if use_memory and getattr(model, "memory_block", None) is None:
        raise SystemExit("--memory requested but the model has no memory block")
    transform = build_transform(backbone, loaded.normalize)
    data_root = args.data_root or DATASET_REGISTRY[args.dataset].data_dir
    series_list = args.series or split_for(args.dataset, args.split)
    out_dir = args.out or (ROOT / "results" / output_tag(args, backbone))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[evaluate] backbone={backbone} memory={use_memory} normalize={loaded.normalize} mask_ratio={loaded.flags.get('mask_ratio')} protocols={args.protocols} seeds={args.seeds} -> {out_dir}")

    for name in series_list:
        path = out_dir / f"{safe_name(name)}.json"
        if args.reuse_cached and path.exists():
            try:
                validate(json.loads(path.read_text()))
                print(f"[evaluate] {name}: reusing {path}")
                continue
            except (ValueError, json.JSONDecodeError):
                pass
        t0 = time.time()
        stream = SeriesStream(data_root / name, box_noise=args.box_noise, pixel_noise=args.pixel_noise,
                              noise_seed=args.noise_seed)
        result = evaluate_series(model, transform, stream, args, device, use_memory)
        prov_args = dict(vars(args)); prov_args.update({
            "checkpoint": None if ckpt_path is None else str(ckpt_path), "out": str(out_dir), "data_root": str(data_root),
            "n_crops": stream.n_crops, "n_identities": stream.n_identities, "split": args.split,
            "mask_ratio": loaded.flags.get("mask_ratio"), "use_memory": use_memory, "backbone": backbone,
            "normalize": loaded.normalize, "model_flags": loaded.flags,
        })
        result["provenance"] = stamp(prov_args, ckpt_path)
        result["series"] = {
            "name": name, "n_crops": stream.n_crops, "n_identities": stream.n_identities,
            "reading_order_source": "page_order_key", "box_noise": args.box_noise, "pixel_noise": args.pixel_noise,
            "box_iou_mean": float(np.mean(stream.box_ious())), "seconds": round(time.time() - t0, 1),
        }
        validate(result)
        write_atomic(path, json.dumps(result, indent=1, default=str))
        summary = []
        if "p1" in result:
            summary.append("P1 mAP %.3f" % np.mean([m["mAP"] for m in result["p1"].values()]))
        if "p3" in result:
            r = result["p3"].get("fixed") or next(v for k, v in result["p3"].items() if not k.startswith("_"))
            summary.append("P3 clusters %d purity %.3f" % (r["clusters"], r["purity"]))
        print(f"[evaluate] {name}: {stream.n_crops} crops, {stream.n_identities} ids, {' | '.join(summary)} ({result['series']['seconds']}s) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
