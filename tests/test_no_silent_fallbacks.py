"""Regression guards against silent fallbacks.

A silent fallback is code that keeps running when its assumption fails and
quietly computes something else. These tests keep out three kinds: a
per-process hash used as a reading-order key, `strict=False` checkpoint
loads that hide missing weights, and a placeholder value standing in for a
result.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
PY_FILES = sorted((REPO / "src").rglob("*.py")) + sorted((REPO / "scripts").rglob("*.py"))

# strict=False may appear only inside the checked loader itself.
STRICT_FALSE_ALLOWED = {
    REPO / "src" / "recognize" / "loading.py",
}


def _sources():
    for path in PY_FILES:
        yield path, path.read_text()


class TestSourceGuards:
    def test_no_hash_builtin_used_as_a_key(self):
        offenders = [
            f"{p.relative_to(REPO)}:{i}"
            for p, text in _sources()
            for i, line in enumerate(text.splitlines(), 1)
            if re.search(r"(?<![\w.])hash\(", line)
        ]
        assert offenders == [], f"hash() fallback found: {offenders}"

    def test_no_unchecked_strict_false(self):
        offenders = [
            f"{p.relative_to(REPO)}:{i}"
            for p, text in _sources()
            if p not in STRICT_FALSE_ALLOWED
            for i, line in enumerate(text.splitlines(), 1)
            if "strict=False" in line
        ]
        assert offenders == [], f"unchecked strict=False found: {offenders}"

    def test_no_placeholder_values(self):
        offenders = [
            f"{p.relative_to(REPO)}:{i}"
            for p, text in _sources()
            for i, line in enumerate(text.splitlines(), 1)
            if "Placeholder" in line
        ]
        assert offenders == [], f"placeholder found: {offenders}"


class TestReadingOrderKey:
    def test_page_order_key_is_natural(self):
        from recognize.data import page_order_key

        k31 = page_order_key("Bakuman - c001 (web) - p031 [Unknown]", 3)
        k34 = page_order_key("Bakuman - c001 (web) - p034 [Unknown]", 0)
        assert k31 < k34
        assert page_order_key("page_2", 0) < page_order_key("page_10", 0)
        assert page_order_key("002", 1) < page_order_key("002", 2) < page_order_key("003", 0)
        # chapters dominate pages: c002 p001 comes after c001 p120
        assert page_order_key("X - c001 (web) - p120 [Unknown]", 0) < page_order_key("X - c002 (web) - p001 [Unknown]", 0)

    def test_sorting_samples_by_page_name_gives_reading_order(self):
        from recognize.data import page_order_key

        names = ["p034", "p031", "p031", "p120"]
        idx = [0, 1, 0, 0]
        keys = [page_order_key(f"Bakuman - c001 (web) - {n} [Unknown]", a) for n, a in zip(names, idx)]
        assert sorted(range(4), key=keys.__getitem__) == [2, 1, 0, 3]

    def test_no_filename_regex_order_key_exists(self):
        offenders = [str(p.relative_to(REPO)) for p, text in _sources() if "extract_page_order_key" in text]
        assert offenders == [], f"filename-parsing order key reintroduced: {offenders}"


class TestCheckedLoader:
    @staticmethod
    def _model():
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

    def test_exact_match_loads(self):
        from recognize.loading import load_state_dict_checked

        src, dst = self._model(), self._model()
        load_state_dict_checked(dst, src.state_dict())
        assert torch.equal(dst[0].weight, src[0].weight)

    def test_unexpected_key_raises(self):
        from recognize.loading import load_state_dict_checked

        sd = self._model().state_dict()
        sd["ghost.weight"] = torch.zeros(1)
        with pytest.raises(RuntimeError, match="ghost.weight"):
            load_state_dict_checked(self._model(), sd)

    def test_missing_key_raises_unless_allowed(self):
        from recognize.loading import load_state_dict_checked

        sd = self._model().state_dict()
        del sd["1.weight"], sd["1.bias"]
        with pytest.raises(RuntimeError, match="1.weight"):
            load_state_dict_checked(self._model(), sd)
        load_state_dict_checked(self._model(), sd, allow_missing_prefixes=("1.",))

    def test_shape_mismatch_raises(self):
        from recognize.loading import load_state_dict_checked

        sd = self._model().state_dict()
        sd["0.weight"] = torch.zeros(3, 5)
        with pytest.raises(RuntimeError, match="0.weight"):
            load_state_dict_checked(self._model(), sd)


class TestEvaluateCacheIsOptIn:
    """Cached results are reused only on request (`--reuse-cached`, off by default).

    The end-to-end behaviour is covered in tests/test_evaluate_cli.py; this guards the flag.
    """

    def test_reuse_cached_defaults_off_and_no_skip_existing_alias(self):
        import evaluate

        args = evaluate.build_parser().parse_args(["--pretrained", "transreid"])
        assert args.reuse_cached is False
        with pytest.raises(SystemExit):
            evaluate.build_parser().parse_args(["--pretrained", "transreid", "--skip-existing"])
