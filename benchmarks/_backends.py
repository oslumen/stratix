"""Backend management utilities for benchmarks."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import numdiff as nd


def available_backends() -> list[str]:
    """Return list of installed numdiff backend names.

    Delegates to :data:`numdiff.available_backends` — always reflects
    the current environment.

    Returns
    -------
    list[str]
        Backend names that are importable, e.g. ``['numpy', 'jax']``.
    """
    return nd.available_backends


def has_cuda() -> bool:
    """Return ``True`` if CUDA is available.

    Delegates to :data:`numdiff.HAS_CUDA`.
    """
    return nd.HAS_CUDA


@contextmanager
def backend_scope(
    name: str,
    gpu: bool = False,
) -> Generator[None, None, None]:
    """Temporarily switch numdiff backend (and optionally device).

    Parameters
    ----------
    name : str
        Backend name (``"numpy"``, ``"autograd"``, ``"jax"``, ``"torch"``).
    gpu : bool
        If ``True``, also enable GPU execution (falls back to CPU if the
        backend does not support CUDA). Default ``False``.

    Yields
    ------
    None
        Execute inside ``with backend_scope(...):``.

    Raises
    ------
    ValueError
        If *name* is not a recognised backend or not installed.
    """
    if name not in nd.available_backends:
        raise ValueError(
            f"Backend {name!r} is not installed. "
            f"Available: {nd.available_backends}"
        )

    old_backend = nd.get_backend()
    old_device = nd.get_device()

    try:
        nd.set_backend(name)
        device = "cuda" if gpu else "cpu"
        nd.use_gpu(device == "cuda")
        yield
    finally:
        nd.set_backend(old_backend)
        nd.use_gpu(old_device == "cuda")
