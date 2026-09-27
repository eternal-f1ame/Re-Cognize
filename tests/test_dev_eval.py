"""Dev-set selection metric through the evaluation harness."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from test_features import StubModel


@pytest.fixture
def series(tmp_path):
    """Series with two identities: Bob x4, Ann x2 over three pages."""
    d = tmp_path / "Dev A"; (d / "images").mkdir(parents=True); (d / "annotations").mkdir()
    (d / "category_mapping.json").write_text(json.dumps({"0": "Ann", "3": "Bob"}))
    rng = np.random.default_rng(0)
    pages = {"p001": ["3 0.3 0.5 0.2 0.4", "0 0.8 0.5 0.1 0.3"], "p002": ["3 0.5 0.5 0.3 0.4", "3 0.2 0.5 0.1 0.2"],
             "p003": ["0 0.7 0.5 0.2 0.3", "3 0.4 0.4 0.2 0.3"]}
    for stem, rows in pages.items():
        Image.fromarray((rng.random((100, 200, 3)) * 255).astype(np.uint8)).save(d / "images" / f"X - c001 - {stem}.jpg")
        (d / "annotations" / f"X - c001 - {stem}.txt").write_text("\n".join(rows) + "\n")
    return d


def _six(img):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.tensor(np.concatenate([a.mean((0, 1)), a.std((0, 1))]), dtype=torch.float32)


class TestDevScore:
    def test_no_memory_uses_p1(self, series):
        from recognize.dev_eval import dev_score
        m = StubModel(with_memory=False); m.train()
        out = dev_score(m, _six, [series], memory=False)
        assert out["metric"] == "p1_map" and set(out["per_series"]) == {"Dev A"} and 0.0 <= out["value"] <= 1.0
        assert m.training is True                                  # mode restored

    def test_memory_uses_p2_k1_and_restores_the_training_block(self, series, monkeypatch):
        import recognize.dev_eval as de
        from recognize.protocols import split_seeds
        from recognize.data import SeriesStream
        m = StubModel(); block_before = m.memory_block; m.train()
        seen = {}
        real_extract = de.extract

        def spy(model, stream, transform, **kw):
            if kw.get("mode") == "memory":              # the extra no-memory pass records comparable metrics
                seen["gallery_idx"] = kw.get("gallery_idx"); seen["mode"] = kw.get("mode")
            return real_extract(model, stream, transform, **kw)

        monkeypatch.setattr(de, "extract", spy)
        out = de.dev_score(m, _six, [series], memory=True)
        s = SeriesStream(series)
        seed_map, _, gallery_only = split_seeds(s.labels, s.reading_order, 1, "random", 0)
        expected = [i for v in seed_map.values() for i in v] + [i for v in gallery_only.values() for i in v]
        assert seen["mode"] == "memory" and sorted(seen["gallery_idx"]) == sorted(expected)
        assert out["metric"] == "p2r1_map" and m.memory_block is block_before and m.training is True


class TestComparableMetrics:
    """The selection metric differs by mode, so dev_score also records metrics that are comparable."""

    def test_both_metrics_are_recorded(self, series):
        from recognize.dev_eval import dev_score
        m = StubModel()
        out = dev_score(m, _six, [series], memory=True)
        assert out["metric"] == "p2r1_map"
        assert "nomem_p1_map" in out and "nomem_p2r1_map" in out
        assert 0.0 <= out["nomem_p1_map"] <= 1.0 and 0.0 <= out["nomem_p2r1_map"] <= 1.0
        # the no-memory P2@1 is the like-for-like partner of the memory run's selection metric
        off = dev_score(StubModel(with_memory=False), _six, [series], memory=False)
        assert off["metric"] == "p1_map" and off["nomem_p1_map"] == pytest.approx(off["value"])

    def test_can_be_switched_off(self, series):
        from recognize.dev_eval import dev_score
        out = dev_score(StubModel(), _six, [series], memory=True, both_metrics=False)
        assert "nomem_p1_map" not in out
