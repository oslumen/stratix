"""Core solve() entry point for stratified-media computations."""

from __future__ import annotations

import numdiff as nd
from phokaia import Stack

from ._result import Result
from ._types import Method
from ._types import Polarization
from .methods._abeles import _abeles_solve
from .methods._smatrix import _smatrix_solve


def solve(
    stack: Stack,
    wavelength: float,
    kx: float,
    polarization: Polarization,
    method: Method = Method.AUTO,
    absorption: bool = False,
) -> Result:
    """Compute reflectance and transmittance for a planar multilayer stack.

    Parameters
    ----------
    stack : Planar multilayer stack (superstrate + substrate ± layers).
    wavelength : Vacuum wavelength in meters.
    kx : In-plane wavevector component in rad/m.
    polarization : ``TE`` or ``TM``.
    method : Solver method.  ``AUTO`` resolves to ``SMATRIX``.
    absorption : If ``True``, compute per-layer absorption (not yet implemented).

    Returns
    -------
    Result with ``R``, ``T``, and metadata fields.
    """
    resolved = Method.SMATRIX if method == Method.AUTO else method

    if resolved == Method.SMATRIX:
        R, T = _smatrix_solve(stack, wavelength, kx, polarization)
    elif resolved == Method.ABELES:
        R, T = _abeles_solve(stack, wavelength, kx, polarization)
    else:
        raise NotImplementedError(f"Method {resolved.value!r} not yet implemented")

    return Result(
        R=nd.array([float(R)]),
        T=nd.array([float(T)]),
        wavelengths=nd.array([wavelength]),
        kx=nd.array([kx]),
        polarization=polarization,
        method_used=resolved,
    )
