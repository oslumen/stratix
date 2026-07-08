"""Test configuration and shared fixtures for stratix."""

from __future__ import annotations

import pytest

import stratix


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (network or compute)")


@pytest.fixture
def sample_data() -> dict[str, float]:
    """Provide sample numeric data for tests."""
    return {"x": 1.0, "y": 2.0, "z": 3.0}






