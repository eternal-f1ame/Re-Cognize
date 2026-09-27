"""Shared fixtures for Re-ID test suite."""

import sys
from pathlib import Path

import pytest
import torch

# Ensure src and scripts are on PYTHONPATH. scripts/ holds _config, train and evaluate, which
# several test modules import; without it here, whether they import depends on which other module
# ran first and inserted the path.
REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
SRC_DIR = REPO / "src"


@pytest.fixture
def device():
    """Return CPU device (tests run on CPU for reproducibility)."""
    return torch.device("cpu")


@pytest.fixture
def feat_dim():
    return 768


@pytest.fixture
def num_characters():
    return 10


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture
def random_features(batch_size, feat_dim):
    """Random L2-normalized feature batch."""
    feats = torch.randn(batch_size, feat_dim)
    return torch.nn.functional.normalize(feats, dim=-1)


@pytest.fixture
def random_labels(batch_size, num_characters):
    """Random labels from [0, num_characters)."""
    return torch.randint(0, num_characters, (batch_size,))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: builds a real backbone or reads the dataset (minutes on CPU)")
