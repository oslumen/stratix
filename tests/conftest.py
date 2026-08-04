"""Test configuration and shared fixtures for stratix."""

from __future__ import annotations

import numdiff as nd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (network or compute)")


@pytest.fixture
def sample_data() -> dict[str, float]:
    """Provide sample numeric data for tests."""
    return {"x": 1.0, "y": 2.0, "z": 3.0}


@pytest.fixture(params=nd.available_backends)
def set_backend(request):
    backend = request.param
    if backend not in nd.available_backends:
        pytest.skip(f"Backend {backend!r} not available")
    old = nd.get_backend()
    nd.set_backend(backend)
    if backend == "torch":
        import torch

        torch.set_default_dtype(torch.float64)
    if backend == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
    yield backend
    nd.set_backend(old)






