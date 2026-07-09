"""Autodiff-friendly wrappers for differentiable stratified-media computations.

These functions return raw scalar ndarrays suitable for ``nd.grad()``,
bypassing the ``Result`` model and ``float()`` casts that break the
autodiff trace.
"""

from __future__ import annotations

import numdiff as nd
from phokaia import Stack

from ._types import Polarization
from .methods._smatrix import _interface_smatrix
from .methods._smatrix import _kz_single
from .methods._smatrix import _propagation_smatrix
from .methods._smatrix import _redheffer_star
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
    thicknesses: nd.ndarray,
    wavelength: float,
    kx: float,
    polarization: Polarization,
) -> tuple:
    """Return (R, T) using traced thicknesses for gradient w.r.t. thickness.

    The phokaia ``Layer``/``Stack`` models are frozen Pydantic models that
    reject traced (autodiff) values, so this function duplicates the
    layer-loop portion of the solve with a raw ndarray of thicknesses,
    bypassing ``Layer`` construction entirely.

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
    c = 299792458.0
    omega = 2 * nd.pi * c / wavelength
    k0 = 2 * nd.pi / wavelength

    media = (
        [stack.superstrate]
        + [layer.material for layer in stack.layers]
        + [stack.substrate]
    )
    epsilons = [m.epsilon(omega=omega) for m in media]
    mus = [m.mu(omega=omega) for m in media]
    kzs = [_kz_single(eps, mu, k0, kx) for eps, mu in zip(epsilons, mus, strict=True)]

    if polarization == Polarization.TE:
        denom_vals = mus
    elif polarization == Polarization.TM:
        denom_vals = epsilons
    else:
        raise NotImplementedError(f"Polarization {polarization!r} not supported")

    Zs = [kz / denom for kz, denom in zip(kzs, denom_vals, strict=True)]

    n_interfaces = len(media) - 1
    S_total = _interface_smatrix(Zs[0], Zs[1])

    for i in range(1, n_interfaces):
        d = thicknesses[i - 1]
        P = _propagation_smatrix(kzs[i], d)
        S_total = _redheffer_star(S_total, P)
        S_int = _interface_smatrix(Zs[i], Zs[i + 1])
        S_total = _redheffer_star(S_total, S_int)

    r_total = S_total[0, 0]
    t_total = S_total[1, 0]

    R = nd.abs(r_total) ** 2
    T = (
        nd.real(kzs[-1] / denom_vals[-1])
        / nd.real(kzs[0] / denom_vals[0])
        * nd.abs(t_total) ** 2
    )

    return R, T
