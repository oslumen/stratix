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


def register_jax_pytrees() -> None:
    """Register ``Polarization`` and ``Method`` as static JAX pytree nodes.

    JIT-compiling :func:`solve` itself on the JAX backend —
    ``nd.jit(solve, static_argnames=(...))`` — returns a :class:`Result`
    whose enum fields JAX only accepts once their types are registered
    as leafless pytree nodes.  Call this once before doing that.

    It is an explicit opt-in rather than an import side effect: the
    registration mutates JAX's process-wide pytree registry, and one of
    the types (``phokaia.Polarization``) is not even stratix's own —
    global state a library must not rewrite just for being imported
    (issue #55).  A no-op when JAX is not installed, and safe to call
    more than once.  Compiling a wrapper that returns arrays instead of
    a ``Result`` needs no registration at all.
    """
    try:
        from jax.tree_util import register_pytree_node
    except ImportError:
        return

    from contextlib import suppress

    from phokaia import Polarization

    for enum_type in (Polarization, Method):
        # suppress: already registered (repeat call or upstream registration)
        with suppress(ValueError):
            register_pytree_node(enum_type, lambda x: ([], x), lambda x, _: x)


__all__ = [
    "Method",
    "Result",
    "compute_field_profile",
    "register_jax_pytrees",
    "solve",
    "solve_angles",
    "solve_from_source",
]
