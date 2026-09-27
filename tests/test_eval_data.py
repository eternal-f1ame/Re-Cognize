"""Tests for recognize.data.SeriesStream."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
BAKUMAN = REPO / "Datasets" / "popcharacters" / "Bakuman"


@pytest.fixture
def series_dir(tmp_path):
    """Two pages named like POPCharacters, three boxes, two characters; p010 sorts after p002."""
    d = tmp_path / "X"
    (d / "images").mkdir(parents=True)
    (d / "annotations").mkdir()
    (d / "category_mapping.json").write_text(json.dumps({"0": "Ann", "3": "Bob"}))
    for stem in ("X - c001 (web) - p010 [Unknown]", "X - c001 (web) - p002 [Unknown]"):
        Image.new("RGB", (200, 100), (255, 255, 255)).save(d / "images" / f"{stem}.jpg")
    # p010: one box of Bob; p002: Bob then Ann (annotation order)
    (d / "annotations" / "X - c001 (web) - p010 [Unknown].txt").write_text("3 0.5 0.5 0.2 0.4\n")
    (d / "annotations" / "X - c001 (web) - p002 [Unknown].txt").write_text("3 0.25 0.5 0.5 0.5\n0 0.9 0.5 0.1 0.2\n")
    return d


class TestSeriesStream:
    def test_reading_order_is_natural_pages_then_annotation_index(self, series_dir):
        from recognize.data import SeriesStream
        s = SeriesStream(series_dir)
        assert [c.page_name[-14:-10] for c in s.crops] == ["p002", "p002", "p010"]
        assert [c.annotation_idx for c in s.crops] == [0, 1, 0]
        assert s.n_crops == 3 and s.name == "X"
        assert np.array_equal(s.reading_order, np.arange(3))

    def test_labels_are_contiguous_in_first_appearance_order(self, series_dir):
        from recognize.data import SeriesStream
        s = SeriesStream(series_dir)
        assert s.labels.tolist() == [0, 1, 0]           # Bob first, then Ann
        assert s.identities == {0: "Bob", 1: "Ann"}
        assert s.n_identities == 2

    def test_crop_is_padded_and_clipped(self, series_dir):
        from recognize.data import SeriesStream
        s = SeriesStream(series_dir, padding=0.10)
        # p002 box 1: cx=0.25*200=50, w=0.5*200=100 -> x in [0,100] padded by 10 -> [-10,110] clipped to [0,110]
        #             cy=0.5*100=50,  h=0.5*100=50  -> y in [25,75]  padded by 5  -> [20,80]
        assert s.image(0).size == (110, 60)
        # p002 box 2: cx=180, w=20 -> [170,190] padded by 2 -> [168,192]; cy=50, h=20 -> [40,60] padded by 2 -> [38,62]
        assert s.image(1).size == (24, 24)

    def test_to_dataset_applies_transform_and_returns_index(self, series_dir):
        from recognize.data import SeriesStream
        from torchvision import transforms as T
        s = SeriesStream(series_dir)
        ds = s.to_dataset(T.Compose([T.Resize((8, 4)), T.ToTensor()]))
        item = ds[2]
        assert item["index"] == 2 and tuple(item["image"].shape) == (3, 8, 4)
        assert len(ds) == 3

    def test_exclude_categories_by_name(self, series_dir):
        from recognize.data import SeriesStream
        s = SeriesStream(series_dir, exclude_categories="ann")
        assert s.n_crops == 2 and s.identities == {0: "Bob"}

    @pytest.mark.slow
    @pytest.mark.skipif(not BAKUMAN.exists(), reason="POPCharacters/Bakuman not present")
    def test_bakuman_has_all_crops(self):
        from recognize.data import SeriesStream
        s = SeriesStream(BAKUMAN)
        assert s.n_crops == 608 and s.n_identities == 7


class TestSplit:
    def test_split_is_13_2_8_and_disjoint(self):
        from recognize.data import load_split
        split = load_split()
        assert (len(split["train"]), len(split["dev"]), len(split["test"])) == (13, 2, 8)
        assert set(split["dev"]) == {"Dragon Ball", "Kuroko S Basketball"}
        assert not (set(split["train"]) & set(split["dev"])) and not (set(split["train"]) & set(split["test"])) and not (set(split["dev"]) & set(split["test"]))

    def test_split_series_match_the_registry(self):
        import _config as C
        from recognize.data import load_split
        split = load_split(); ds = C.DATASET_REGISTRY["popcharacters"]
        assert set(split["train"]) == set(ds.train_manga) and set(split["dev"]) == set(ds.dev_manga)
        assert set(split["test"]) == set(ds.test_manga) == set(ds.val_manga)
        assert ds.split_config.name == "data_split.yaml"


class TestReverseCorpus:
    """Re:Verse is one series evaluated whole, resolved through the dataset registry."""

    def test_registered_with_no_split_file(self):
        from _config import DATASET_REGISTRY
        ds = DATASET_REGISTRY["reverse"]
        assert ds.test_manga == ["Re-Zero"] and ds.split_config is None
        assert not ds.train_manga and not ds.dev_manga

    def test_the_series_directory_is_in_the_canonical_layout(self):
        from _config import DATASET_REGISTRY
        d = DATASET_REGISTRY["reverse"].data_dir / "Re-Zero"
        if not d.exists():
            pytest.skip("Re:Verse is git-ignored and not present in every checkout")
        assert (d / "images").is_dir() and (d / "annotations").is_dir()
        assert (d / "category_mapping.json").is_file()

    def test_the_evaluator_resolves_its_test_split(self):
        import evaluate
        assert evaluate.split_for("reverse", "test") == ["Re-Zero"]

    def test_asking_for_a_dev_split_it_does_not_have_is_an_error(self):
        import evaluate
        with pytest.raises(SystemExit, match="no dev split"):
            evaluate.split_for("reverse", "dev")
