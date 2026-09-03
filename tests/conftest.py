import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

import locate_event as le  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_logs(monkeypatch):
    """Keep locator logging out of test output."""
    monkeypatch.setattr(le, "log", lambda msg, level="INFO": None)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def fs():
    return 48000
