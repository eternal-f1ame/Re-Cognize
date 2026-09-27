"""The results archive is verified before it is unpacked, unpacks only into results/, and every
evaluation result in it passes the schema."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

import fetch_results

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _archive(tmp_path, members):
    path = tmp_path / "results.tar.xz"
    with tarfile.open(path, "w:xz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_unpacks_a_verified_archive(tmp_path):
    path, digest = _archive(tmp_path, {"results/p3_tau.json": b"{}", "results/series/dev.lst": b"a\n"})
    dest = tmp_path / "repo"
    assert fetch_results.unpack(path, dest, expected=digest) == 2
    assert (dest / "results" / "p3_tau.json").read_bytes() == b"{}"
    assert (dest / "results" / "series" / "dev.lst").read_bytes() == b"a\n"


def test_rejects_a_checksum_mismatch(tmp_path):
    path, _ = _archive(tmp_path, {"results/p3_tau.json": b"{}"})
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        fetch_results.unpack(path, tmp_path / "repo", expected="0" * 64)
    assert not (tmp_path / "repo" / "results").exists()


def test_rejects_files_outside_results(tmp_path):
    path, digest = _archive(tmp_path, {"results/ok.json": b"{}", "src/recognize/data.py": b"x"})
    with pytest.raises(RuntimeError, match="outside results"):
        fetch_results.unpack(path, tmp_path / "repo", expected=digest)


def test_every_unpacked_evaluation_result_validates():
    from recognize.schema import validate
    checked, bad = 0, []
    for p in sorted(RESULTS.rglob("*.json")):
        d = json.loads(p.read_text())
        if not (isinstance(d, dict) and "provenance" in d and "series" in d):
            continue  # an analysis output, not an evaluation result
        checked += 1
        try:
            validate(d)
        except ValueError as e:
            bad.append(f"{p.relative_to(RESULTS)}: {e}")
    if not checked:
        pytest.skip("no results in this checkout")
    assert not bad, f"{len(bad)} of {checked} results fail the schema: {bad[:3]}"
