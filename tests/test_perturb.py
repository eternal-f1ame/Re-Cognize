"""Tests for recognize.perturb and SeriesStream noise options."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


class TestBoxPerturb:
    @pytest.mark.parametrize("kind,expected", [("shift10", 0.822), ("shift20", 0.681), ("shift30", 0.566)])
    def test_mean_iou_of_shifts(self, kind, expected):
        from recognize.perturb import box_perturb
        rng = np.random.default_rng(0)
        ious = []
        for _ in range(4000):
            box = (0.5, 0.5, 0.2, 0.3)
            nb, iou = box_perturb(box, kind, rng)
            assert iou == pytest.approx(_iou(box, nb), abs=1e-9)
            ious.append(iou)
        assert float(np.mean(ious)) == pytest.approx(expected, abs=0.03)

    def test_scale_kinds_have_exact_iou(self):
        from recognize.perturb import box_perturb
        rng = np.random.default_rng(0)
        _, iou_t = box_perturb((0.5, 0.5, 0.2, 0.3), "tight07", rng)
        _, iou_l = box_perturb((0.5, 0.5, 0.2, 0.3), "loose13", rng)
        assert iou_t == pytest.approx(0.49) and iou_l == pytest.approx(1 / 1.69)

    def test_deterministic_for_a_seed(self):
        from recognize.perturb import box_perturb
        a = box_perturb((0.5, 0.5, 0.2, 0.3), "shift20", np.random.default_rng([3, 7]))
        b = box_perturb((0.5, 0.5, 0.2, 0.3), "shift20", np.random.default_rng([3, 7]))
        assert a == b

    def test_unknown_kind_raises(self):
        from recognize.perturb import box_perturb
        with pytest.raises(ValueError):
            box_perturb((0.5, 0.5, 0.2, 0.3), "shift99", np.random.default_rng(0))


class TestPixelPerturb:
    def test_occlusion_covers_the_requested_fraction(self):
        from recognize.perturb import pixel_perturb
        img = Image.new("RGB", (200, 100), (255, 255, 255))
        out = np.asarray(pixel_perturb(img, "occ30", np.random.default_rng(1)))
        grey = np.all(out == 128, axis=2).mean()
        assert grey == pytest.approx(0.30, abs=0.01)

    def test_blur_and_jitter_change_pixels_but_not_size(self):
        from recognize.perturb import pixel_perturb
        rng = np.random.default_rng(2)
        img = Image.fromarray((rng.random((60, 40, 3)) * 255).astype(np.uint8))
        for kind in ("jitter10", "jitter20", "blur2", "blur4"):
            out = pixel_perturb(img, kind, np.random.default_rng(5))
            assert out.size == img.size and not np.array_equal(np.asarray(out), np.asarray(img))


class TestStreamNoise:
    @pytest.fixture
    def series_dir(self, tmp_path):
        d = tmp_path / "X"; (d / "images").mkdir(parents=True); (d / "annotations").mkdir()
        (d / "category_mapping.json").write_text(json.dumps({"0": "Ann"}))
        rng = np.random.default_rng(0)
        Image.fromarray((rng.random((100, 200, 3)) * 255).astype(np.uint8)).save(d / "images" / "X - c001 - p001.jpg")
        (d / "annotations" / "X - c001 - p001.txt").write_text("0 0.5 0.5 0.2 0.4\n")
        return d

    def test_box_noise_changes_crop_not_labels_and_reports_iou(self, series_dir):
        from recognize.data import SeriesStream
        clean = SeriesStream(series_dir)
        noisy = SeriesStream(series_dir, box_noise="shift30", noise_seed=0)
        assert noisy.labels.tolist() == clean.labels.tolist()
        assert noisy.image(0).size != clean.image(0).size or not np.array_equal(np.asarray(noisy.image(0)), np.asarray(clean.image(0)))
        ious = noisy.box_ious()
        assert len(ious) == 1 and 0.0 < ious[0] < 1.0
        assert clean.box_ious() == [1.0]

    def test_pixel_noise_is_deterministic_per_crop(self, series_dir):
        from recognize.data import SeriesStream
        a = SeriesStream(series_dir, pixel_noise="occ15", noise_seed=4)
        b = SeriesStream(series_dir, pixel_noise="occ15", noise_seed=4)
        assert np.array_equal(np.asarray(a.image(0)), np.asarray(b.image(0)))
