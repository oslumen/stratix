"""Stratified Media Electromagnetic Solver"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("stratix")
except PackageNotFoundError:
    __version__ = "unknown"

from ._convenience import solve_angles
from ._convenience import solve_from_source
from ._field import compute_field_profile
from ._result import Result
from ._solve import solve
from ._types import Method


def _register_jax_pytrees() -> None:
    """Register ``Polarization`` and ``Method`` as JAX pytree leaves.

    Called at import time when JAX is available.  Required so that
    :func:`solve` return values (which include enum fields in
    :class:`Result` and the ``intermediates`` dict) can be traced by
    :func:`jax.jit`.  Without this registration,
    ``nd.jit(solve, static_argnames=(...))`` would fail.
    """
    try:
        from jax.tree_util import register_pytree_node
    except ImportError:
        return

    from contextlib import suppress

    from phokaia import Polarization

    for enum_type in (Polarization, Method):
        # suppress: already registered (re-import or upstream registration)
        with suppress(ValueError):
            register_pytree_node(enum_type, lambda x: ([], x), lambda x, _: x)


_register_jax_pytrees()

__all__ = [
    "Method",
    "Result",
    "compute_field_profile",
    "solve",
    "solve_angles",
    "solve_from_source",
]
