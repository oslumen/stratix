"""Autodiff-friendly wrappers for differentiable stratified-media computations.

These functions return raw scalar ndarrays suitable for ``nd.grad()``,
bypassing the ``Result`` model and ``float()`` casts that break the
autodiff trace.
"""

from __future__ import annotations

from phokaia import Polarization
from phokaia import Stack

from .methods._smatrix import _smatrix_solve


def _solve_raw(
    stack: Stack,
    wavelength: float,
    kx: float,
    polarization: Polarization,
) -> tuple:
    """Return (R, T) as scalar ndarrays, preserving the autodiff trace.

    Unlike :func:`stratix.solve`, this skips ``Result`` construction and
    ``float()`` conversions, making it suitable for ``nd.grad()``.

    Parameters
    ----------
    stack : Multilayer stack (superstrate, layers, substrate).
    wavelength : Vacuum wavelength in meters.
    kx : In-plane wavevector component in rad/m.
    polarization : ``TE`` or ``TM``.

    Returns
    -------
    R : 0-D ndarray — power reflectance.
    T : 0-D ndarray — power transmittance.
    """
    R, T, _ = _smatrix_solve(stack, wavelength, kx, polarization)
    return R, T


def _solve_raw_with_thicknesses(
    stack: Stack,
    thicknesses,
    wavelength: float,
    kx: float,
    polarization: Polarization,
) -> tuple:
    """Return (R, T) using traced thicknesses for gradient w.r.t. thickness.

    Routes through :func:`_smatrix_solve` with a ``thicknesses=`` override,
    bypassing the frozen ``Layer`` thicknesses in the Pydantic ``Stack``.

    Parameters
    ----------
    stack : Multilayer stack (superstrate, layers, substrate).
        The layers' materials are used, but their thicknesses are **ignored**
        in favour of the ``thicknesses`` array.
    thicknesses : 1-D ndarray, length ``len(stack.layers)``.
        Layer thicknesses in meters.  May contain traced values.
    wavelength : Vacuum wavelength in meters.
    kx : In-plane wavevector component in rad/m.
    polarization : ``TE`` or ``TM``.

    Returns
    -------
    R : 0-D ndarray — power reflectance.
    T : 0-D ndarray — power transmittance.
    """
    R, T, _ = _smatrix_solve(
        stack, wavelength, kx, polarization, thicknesses=thicknesses
    )
    return R, T
