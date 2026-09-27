"""Tests for recognize.features using a stub model and stream."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Spy:
    def __init__(self):
        self.calls = []


class _EpisodicStub:
    def __init__(self):
        self.init_calls, self.protos, self.n = [], None, 0

    def initialize_from_support(self, feats, char_ids=None, method="diverse"):
        self.init_calls.append(char_ids.clone())
        self.protos = torch.zeros(self.n, feats.shape[1])
        for c in range(self.n):
            m = char_ids == c
            if m.any():
                self.protos[c] = F.normalize(feats[m].mean(0), dim=0)

    def compute_similarity(self, q):
        return q @ self.protos.T


class _WorkingStub:
    def __init__(self):
        self.update_calls = []

    def update(self, feats, char_ids):
        self.update_calls.append(char_ids.clone())


class _MemoryStub:
    def __init__(self):
        self.episodic_memory, self.working_memory = _EpisodicStub(), _WorkingStub()

    def initialize_from_support(self, feats, char_ids=None, method="diverse"):
        self.episodic_memory.initialize_from_support(feats, char_ids, method)


class _StubBackbone(nn.Module):
    """Stands in for BackboneWrapper: returns (patch tokens, class token) like the real one."""

    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        return None, self.lin(x)


class StubModel(nn.Module):
    """Linear + L2 normalise; mirrors the MemoryEnhancedReID call surface used by extract()."""

    def __init__(self, d_in=6, d=4, with_memory=True):
        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(d_in, d)
        self.backbone_wrapper = _StubBackbone(self.lin)
        self.bnneck = nn.BatchNorm1d(d)
        self.memory_block = _MemoryStub() if with_memory else None
        self.forward_char_ids = []

    def forward(self, x, char_ids=None, use_memory=True, update_working=False,
                update_episodic=False, episodic_search_all=False, return_all=False):
        self.forward_char_ids.append(None if char_ids is None else char_ids.clone())
        _, cls = self.backbone_wrapper(x)
        f = F.normalize(cls, dim=1)
        if use_memory and char_ids is not None:
            f = F.normalize(f + 0.01 * char_ids.float().unsqueeze(1), dim=1)
        return {"bn_feat": f, "pre_memory_feat": f, "logits": f}

    def reinit_for_open_set(self, n):
        self.memory_block = _MemoryStub()
        self.memory_block.episodic_memory.n = n

    def identify_character(self, imgs, update_memory=True, return_features=False):
        out1 = self.forward(imgs, char_ids=None, use_memory=True)
        sims = self.memory_block.episodic_memory.compute_similarity(out1["bn_feat"])
        pred = sims.argmax(1)
        out2 = self.forward(imgs, char_ids=pred, use_memory=True, episodic_search_all=True)
        if update_memory:
            self.memory_block.working_memory.update(out2["pre_memory_feat"], pred)
        return {"predictions": pred, "similarities": sims, "confidences": sims.max(1).values,
                "features": out2["bn_feat"]}


class FakeStream:
    def __init__(self, n=10, d_in=6, labels=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, d_in, generator=g)
        self.labels = np.asarray(labels if labels is not None else [i % 3 for i in range(n)])
        self.n_crops = n

    def to_dataset(self, transform):
        x = self.x
        class DS(torch.utils.data.Dataset):
            def __len__(s): return x.shape[0]
            def __getitem__(s, i): return {"image": x[i], "index": i}
        return DS()


class TestNoMemory:
    def test_rows_unit_norm_and_deterministic(self):
        from recognize.features import extract
        s = FakeStream()
        a = extract(StubModel(), s, transform=None, batch_size=4)
        b = extract(StubModel(), s, transform=None, batch_size=3)
        assert a.bn.shape == (10, 4) and a.pred_identity is None
        np.testing.assert_allclose(np.linalg.norm(a.bn, axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(a.bn, b.bn, atol=1e-6)

    def test_no_memory_never_passes_char_ids(self):
        from recognize.features import extract
        m = StubModel()
        extract(m, FakeStream(), transform=None)
        assert all(c is None for c in m.forward_char_ids)


class TestMemory:
    def test_gallery_idx_required(self):
        from recognize.features import extract
        with pytest.raises(ValueError):
            extract(StubModel(), FakeStream(), transform=None, mode="memory")

    def test_memory_sees_only_gallery_labels(self):
        from recognize.features import extract
        # labels 10 and 20 in the gallery; 99 appears only among queries
        labels = [10, 20, 10, 99, 20, 10, 99, 20, 10, 20]
        m = StubModel()
        f = extract(m, FakeStream(labels=labels), transform=None, mode="memory", gallery_idx=[0, 1], batch_size=3)
        gallery_local = {0, 1}
        for call in m.memory_block.episodic_memory.init_calls:
            assert sorted(call.tolist()) == [0, 1]
        seen = {int(v) for c in m.forward_char_ids if c is not None for v in c.tolist()}
        seen |= {int(v) for c in m.memory_block.working_memory.update_calls for v in c.tolist()}
        assert seen <= gallery_local
        assert 99 not in seen and 10 not in seen and 20 not in seen
        # predictions are reported in the stream's label space; gallery rows are -1
        assert f.pred_identity.shape == (10,)
        assert f.pred_identity[0] == -1 and f.pred_identity[1] == -1
        assert set(f.pred_identity[2:].tolist()) <= {10, 20}
        np.testing.assert_allclose(np.linalg.norm(f.bn, axis=1), 1.0, atol=1e-5)

    def test_queries_processed_in_stream_order(self):
        from recognize.features import extract
        m = StubModel()
        s = FakeStream(labels=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        extract(m, s, transform=None, mode="memory", gallery_idx=[0, 1], batch_size=4)
        # WM primed once with the gallery (2 rows), then query batches of 4, 4
        sizes = [len(c) for c in m.memory_block.working_memory.update_calls]
        assert sizes == [2, 4, 4]

    def test_initialisation_draw_is_seeded_by_the_gallery(self, monkeypatch):
        # the prototypes' farthest-point sampling starts from a random crop; that draw must depend on
        # the gallery, not on the state the global generator happens to be in
        from recognize.features import extract
        draws = []
        original = _EpisodicStub.initialize_from_support

        def drawing(self, feats, char_ids=None, method="diverse"):
            draws.append(torch.randint(10**9, (1,)).item())
            original(self, feats, char_ids, method)

        monkeypatch.setattr(_EpisodicStub, "initialize_from_support", drawing)
        s = FakeStream()
        runs = []
        for global_seed in (1, 2):
            m = StubModel()                      # the stub seeds the generator itself, so seed after it
            torch.manual_seed(global_seed)
            draws.clear()
            f = extract(m, s, transform=None, mode="memory", gallery_idx=[0, 1, 2], batch_size=4)
            runs.append((list(draws), f.bn.copy(), f.pred_identity.copy()))
        assert len(runs[0][0]) == 2 and runs[0][0] == runs[1][0]      # bootstrap and refine, same draws
        np.testing.assert_array_equal(runs[0][1], runs[1][1])
        np.testing.assert_array_equal(runs[0][2], runs[1][2])
        m = StubModel(); torch.manual_seed(1); draws.clear()
        extract(m, s, transform=None, mode="memory", gallery_idx=[0, 1, 3], batch_size=4)
        assert draws != runs[0][0]                                     # another gallery, another draw


class TestIdentityBNNeck:
    def test_bnneck_becomes_l2_only(self):
        from recognize.features import identity_bnneck
        m = StubModel()
        with torch.no_grad():
            m.bnneck.weight.fill_(3.0); m.bnneck.bias.fill_(0.5)
        identity_bnneck(m)
        x = torch.randn(5, 4)
        torch.testing.assert_close(m.bnneck(x), F.normalize(x, dim=1))


class TestSharedTokens:
    """Class tokens do not depend on the gallery, so one series computes them once."""

    def test_tokens_reproduce_the_full_path(self):
        from recognize.features import backbone_tokens, extract
        s = FakeStream(n=12, labels=[0, 1, 2] * 4)
        a = extract(StubModel(), s, transform=None, mode="memory", gallery_idx=[0, 1, 2])
        tok = backbone_tokens(StubModel(), s, transform=None)
        b = extract(StubModel(), s, transform=None, mode="memory", gallery_idx=[0, 1, 2], tokens=tok)
        np.testing.assert_allclose(a.bn, b.bn, atol=1e-6)
        np.testing.assert_array_equal(a.pred_identity, b.pred_identity)

    def test_backbone_is_not_rerun_when_tokens_are_supplied(self):
        from recognize.features import backbone_tokens, extract
        m = StubModel()
        s = FakeStream(n=12, labels=[0, 1, 2] * 4)
        tok = backbone_tokens(m, s, transform=None)
        calls = []
        original = m.lin.forward
        m.lin.forward = lambda x, _o=original: (calls.append(x.shape[0]), _o(x))[1]  # the stub's "backbone"
        for gallery in ([0, 1, 2], [3, 4, 5], [6, 7, 8]):
            extract(m, s, transform=None, mode="memory", gallery_idx=gallery, tokens=tok, batch_size=6)
        assert calls == [], "the backbone ran again although tokens were supplied"

    def test_plain_mode_accepts_tokens_too(self):
        from recognize.features import backbone_tokens, extract
        s = FakeStream(n=8)
        tok = backbone_tokens(StubModel(), s, transform=None)
        a = extract(StubModel(), s, transform=None, mode="none")
        b = extract(StubModel(), s, transform=None, mode="none", tokens=tok)
        np.testing.assert_allclose(a.bn, b.bn, atol=1e-7)

    def test_the_wrapper_is_restored(self):
        from recognize.features import backbone_tokens, extract
        m = StubModel()
        before = m.backbone_wrapper
        s = FakeStream(n=8)
        extract(m, s, transform=None, mode="none", tokens=backbone_tokens(m, s, transform=None))
        assert m.backbone_wrapper is before
