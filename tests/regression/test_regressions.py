"""Regression tests for properties the reported numbers depend on and that can break silently.

They guard: MagiV2's ViT-MAE masking is off at inference and every backbone forward is
deterministic; reading order is natural page order with no per-process hash; a held-out series
is evaluated on every crop; the P3 stream is in reading order, not grouped by identity; episodic
memory grows past its initial capacity; memory in the P2/P4 path is initialised from the gallery
only; ID-drop fires at its configured rate in the training call pattern; ReID5o has no dimension
adapter and uses CLIP normalisation; P4 mAP is exemplar-level AP; the launcher and every recorded
training config follow the recipe; and the Manga109 split is series-disjoint. Tests that need the
dataset, model weights or checkpoints skip when those are absent, so the file runs anywhere.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
DATA = REPO / "Datasets" / "popcharacters"
BAKUMAN = DATA / "Bakuman"
CKPT = {bb: REPO / "checkpoints" / "baseline" / bb / f"baseline_{bb}" / "best.pth"
        for bb in ("transreid", "magiv2", "magiv3", "instructreid", "reid5o")}
SIZES = {"transreid": (256, 128), "magiv2": (224, 224), "magiv3": (384, 384),
         "instructreid": (256, 128), "reid5o": (384, 128)}

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")  # timm/transformers import-time notices

needs_bakuman = pytest.mark.skipif(not BAKUMAN.exists(), reason="POPCharacters/Bakuman not present")
slow = pytest.mark.slow


def _load(bb):
    from types import SimpleNamespace
    import evaluate  # scripts/evaluate.py
    if not CKPT[bb].exists():
        pytest.skip(f"checkpoint for {bb} not present")
    if bb == "magiv3" and not os.environ.get("RUN_SLOW_MAGIV3"):
        pytest.skip("MagiV3 is fp16 Florence-2; set RUN_SLOW_MAGIV3=1 to build it on CPU")
    torch.set_num_threads(8)
    args = SimpleNamespace(checkpoint=CKPT[bb], pretrained=None, memory=None)
    return evaluate.build_model(args, "cpu").model


def _batch(bb, n=4):
    from torch.utils.data import DataLoader
    from memory_block.training.dataset import get_transforms
    from recognize.data import SeriesStream
    h, w = SIZES[bb]
    ds = SeriesStream(BAKUMAN).to_dataset(get_transforms(h, w, is_train=False))
    return next(iter(DataLoader(ds, batch_size=n, shuffle=False, num_workers=0)))["image"]


# --- MagiV2 ViT-MAE masking -------------------------------------------------------
# ViT-MAE masks patches at random on every forward, in eval mode too. The evaluation entry point
# (load_checkpoint / build_model) disables masking and records the value found; _build_magiv2
# asserts it.
@slow
def test_magiv2_is_built_with_masking_disabled():
    model = _load("magiv2")
    assert model.backbone_wrapper.backbone.config.mask_ratio == 0.0


@slow
@needs_bakuman
@pytest.mark.parametrize("bb", ["transreid", "instructreid", "reid5o", "magiv3", "magiv2"])
def test_backbone_forward_is_deterministic(bb):
    model = _load(bb)
    x = _batch(bb)
    outs = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        with torch.no_grad():
            outs.append(model(x, use_memory=False)["bn_feat"])
    assert torch.equal(outs[0], outs[1])


# --- Reading order ----------------------------------------------------------------
@needs_bakuman
def test_reading_order_is_natural_page_order_and_hash_free():
    from recognize.data import SeriesStream, page_order_key
    s = SeriesStream(BAKUMAN)
    crops = [s.crops[i] for i in s.reading_order]
    # page numbers restart in every chapter, so reading order is (chapter, page, annotation)
    keys_cpa = [tuple(int(x) for x in re.search(r"c(\d+).*?p(\d+)", c.page_name).groups()) + (c.annotation_idx,)
                for c in crops]
    assert keys_cpa == sorted(keys_cpa)
    assert [page_order_key(c.page_name, c.annotation_idx) for c in crops] == sorted(
        page_order_key(c.page_name, c.annotation_idx) for c in crops)
    text = (SRC / "recognize" / "data.py").read_text()
    assert not re.search(r"(?<![\w.])hash\(", text)


# --- Every crop of a held-out series is evaluated ---------------------------------
# SeriesStream serves every crop of a held-out series, not a train/val slice of it.
@needs_bakuman
def test_held_out_series_evaluates_every_crop():
    from recognize.data import SeriesStream
    assert SeriesStream(BAKUMAN).n_crops == 608


# --- P3 stream order --------------------------------------------------------------
# run_p3 streams SeriesStream.reading_order, never a stream grouped by identity.
@needs_bakuman
def test_no_memory_p3_stream_is_not_identity_sorted():
    from recognize.data import SeriesStream
    s = SeriesStream(BAKUMAN)
    labels = s.labels[s.reading_order]
    runs = 1 + int(np.sum(labels[1:] != labels[:-1]))
    assert runs > len(set(labels.tolist())), "stream visits each identity in one contiguous run"


# --- Episodic memory capacity -----------------------------------------------------
# The prototype bank grows on demand (ensure_capacity) instead of silently dropping identities
# beyond its initial capacity.
def test_episodic_memory_grows_beyond_initial_capacity():
    from memory_block.models.memory_modules import EpisodicMemory
    em = EpisodicMemory(num_characters=4, slots_per_char=5, feat_dim=16)
    feats = torch.nn.functional.normalize(torch.randn(1, 16), dim=1)
    em.update(feats, torch.tensor([5]))
    assert em.num_characters == 6 and bool(em.char_initialized[5])


# --- Memory initialisation in the P2/P4 path --------------------------------------
# recognize.features.extract initialises memory from gallery_idx only, never from every
# labelled crop.
def test_p2_p4_memory_path_does_not_initialise_from_all_features():
    from recognize.features import extract
    from test_features import FakeStream, StubModel   # tests/ is on sys.path via conftest
    labels = [10, 20, 10, 99, 20, 10, 99, 20]
    m = StubModel()
    f = extract(m, FakeStream(n=8, labels=labels), transform=None, mode="memory", gallery_idx=[0, 1], batch_size=3)
    init_labels = {int(v) for c in m.memory_block.episodic_memory.init_calls for v in c.tolist()}
    assert init_labels == {0, 1}                       # local ids of the two gallery identities only
    assert all(sorted(c.tolist()) == [0, 1] for c in m.memory_block.episodic_memory.init_calls)
    assert (f.pred_identity[:2] == -1).all() and set(f.pred_identity[2:].tolist()) <= {10, 20}


# --- ID-drop in the training step -------------------------------------------------
# ID-drop is per sample, and the training step's call pattern does not force search-all, so the
# drop fires at the configured rate.
def test_id_drop_fires_at_the_configured_rate_in_the_training_call_pattern():
    from memory_block.models.memory_modules import MemoryBlock
    torch.manual_seed(0)
    block = MemoryBlock(num_characters=8, feat_dim=16, working_capacity=4, slots_per_char=3,
                        num_heads=2, dropout=0.0, episodic_id_drop_rate=0.5)
    block.train()
    seen = []
    original = block.episodic_memory.forward

    def spy(*args, **kwargs):
        seen.append(kwargs.get("char_ids", args[1] if len(args) > 1 else None) is not None)
        return original(*args, **kwargs)

    block.episodic_memory.forward = spy
    feats = torch.nn.functional.normalize(torch.randn(6, 16), dim=1)
    ids = torch.arange(6)
    block.initialize_from_support(feats, ids)
    drop_masks = []
    original_fwd = block.forward

    def record(*a, **kw):
        out = original_fwd(*a, **kw)
        if getattr(block, "_last_id_drop_mask", None) is not None:
            drop_masks.append(block._last_id_drop_mask)
        return out

    block.forward = record
    for _ in range(200):
        # this is the argument pattern train.py's second pass uses
        block(feats, char_ids=ids, update_working=False, update_episodic=False)
    # per-sample drop: each call queries EM guided and/or search-all; the row-level rate is what matters
    rate = float(np.mean([m.float().mean().item() for m in drop_masks]))
    assert 0.45 <= rate <= 0.55, f"ID-drop row rate: {rate:.3f}"
    assert any(seen) and not all(seen), "both identity-guided and search-all EM queries must occur"


# --- Native feature widths --------------------------------------------------------
# The builders size the BNNeck and memory to the backbone's native dim, with no dimension adapter.
@slow
def test_reid5o_is_built_at_its_native_dim():
    from types import SimpleNamespace
    import evaluate
    from recognize.backbones import BACKBONE_REGISTRY
    if not BACKBONE_REGISTRY["reid5o"].weights_path().exists():
        pytest.skip("ReID5o released weights not present")
    torch.set_num_threads(8)
    lm = evaluate.build_model(SimpleNamespace(checkpoint=None, pretrained="reid5o", memory=None), "cpu")
    assert lm.model.bnneck is not None and lm.model.backbone_wrapper.feat_dim == 512 and lm.normalize == "clip"


# --- ReID5o normalisation ---------------------------------------------------------
# ReID5o uses CLIP statistics, not ImageNet ones; BackboneConfig.normalize is read by evaluate.py
# and train.py.
def test_reid5o_transform_uses_clip_statistics():
    from _config import BACKBONE_REGISTRY
    from memory_block.training.dataset import get_transforms
    from torchvision import transforms as T
    cfg = BACKBONE_REGISTRY["reid5o"]
    tf = get_transforms(cfg.height, cfg.width, is_train=False, normalize_type=cfg.normalize)
    norm = [t for t in tf.transforms if isinstance(t, T.Normalize)][0]
    assert abs(norm.mean[0] - 0.48145466) < 1e-6


# --- P4 mAP is exemplar-level AP --------------------------------------------------
# recognize.protocols.run_p4 scores exemplar-level AP against the per-query gallery snapshot, not
# identity-level reciprocal rank.
def test_p4_uses_exemplar_level_average_precision():
    from recognize.protocols import run_p4
    # identity 0: seeds s0 (close to q), s1 (far); identity 1: seed t (between them)
    q = np.array([1.0, 0.0]); s0 = np.array([0.99, 0.14]); s1 = np.array([0.0, 1.0]); t = np.array([0.7, 0.71])
    F = np.stack([s0, s1, t, q]); F = F / np.linalg.norm(F, axis=1, keepdims=True)
    labels = [0, 0, 1, 0]
    r = run_p4(F, labels, order=[0, 1, 2, 3], k=2, strategy="temporal", seed=0)
    # exemplar-level AP for q: ranked s0 (rel), t, s1 (rel) -> (1/1 + 2/3)/2 = 0.833; identity-level RR = 1.0
    assert abs(r["mAP"] - 0.8333) < 1e-3
    assert r["MRR"] == 1.0 and r["R1_identity"] == 1.0


# --- Launcher recipe --------------------------------------------------------------
# Every launched command carries the recipe (recognize.recipe.RECIPE) and the registry's PK
# values; the launcher has no hyperparameter defaults of its own.
def test_launcher_defaults_match_the_paper_recipe():
    import importlib
    train = importlib.import_module("train")               # scripts/train.py
    from _config import BACKBONE_REGISTRY
    assert all(cfg.k == 4 for cfg in BACKBONE_REGISTRY.values())
    for name in ("transreid_memory_seed0", "magiv2_finetuned_seed0"):
        cmd = train.build_command(train.get_run(name))
        flag = lambda f: cmd[cmd.index(f) + 1]
        assert flag("--epochs") == "200" and abs(float(flag("--lr")) - 1e-4) < 1e-12
        assert flag("--k") == "4" and flag("--k-support") == "2" and flag("--memory-lr-scale") == "1.0"


# Every recorded training config under checkpoints/ matches the recipe.
def test_recorded_training_config_matches_recipe():
    configs = sorted((REPO / "checkpoints").glob("**/config.json"))
    if not configs:
        pytest.skip("no checkpoints under checkpoints/ yet")
    for path in configs:
        if "smoke" in path.parts or "timing" in path.parts:
            continue
        cfg = json.loads(path.read_text())
        assert cfg["epochs"] == 200 and abs(cfg["lr"] - 1e-4) < 1e-12 and cfg["k"] == 4 and cfg["k_support"] == 2, path


# --- Manga109 split is series-disjoint --------------------------------------------
# No series has volumes on both sides of the split: both MoeruOnisan volumes are in train.
def test_manga109_split_is_series_disjoint():
    import yaml
    from _config import DATASET_REGISTRY
    base = lambda s: re.sub(r"_vol\d+$", "", s)
    ds = DATASET_REGISTRY["manga109"]
    shared = {base(v) for v in ds.test_manga} & {base(t) for t in ds.train_manga}
    assert shared == set()
    assert (len(ds.train_manga), len(ds.test_manga)) == (82, 27)
    split = yaml.safe_load((REPO / "configs" / "training" / "data_split_manga109.yaml").read_text())
    assert list(ds.train_manga) == split["train"] and list(ds.test_manga) == split["val"]
