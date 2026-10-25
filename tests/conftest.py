import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_configure(config):
    config.addinivalue_line("markers", "fast: quick per-commit tests (seconds)")
    config.addinivalue_line("markers", "slow: recovery / long-chain tests (minutes in total)")


def pytest_collection_modifyitems(config, items):
    """Every test not marked slow is fast, so `pytest -m fast` needs no list."""
    for item in items:
        if item.get_closest_marker("slow") is None:
            item.add_marker(pytest.mark.fast)
