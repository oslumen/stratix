"""S-matrix method for stratified-medium transfer-matrix computations."""

from __future__ import annotations

import numdiff as nd
from phokaia import Stack

from .._types import Polarization


def _kz_single(epsilon: nd.ndarray, mu: nd.ndarray, k0: float, kx: float) -> nd.ndarray:
    """Compute the out-of-plane wavevector component kz.

    Chooses the physical branch: Im(kz) >= 0, or Re(kz) >= 0 when Im(kz) = 0.
    """
    kz_sq = epsilon * mu * k0**2 - kx**2
    kz_sq = kz_sq + 0j
    kz = nd.sqrt(kz_sq)
    neg = (nd.imag(kz) < 0) | ((nd.imag(kz) == 0) & (nd.real(kz) < 0))
    return nd.where(neg, -kz, kz)


def _single_interface_smatrix(
    stack: Stack, wavelength: float, kx: float, polarization: Polarization
) -> tuple[nd.ndarray, nd.ndarray]:
    """Compute R and T for a single interface between two media.

    Parameters
    ----------
    stack : Stack with no layers (superstrate + substrate only).
    wavelength : Vacuum wavelength in meters.
    kx : In-plane wavevector component in rad/m.
    polarization : Currently only TE is supported.

    Returns
    -------
    R : Power reflectance (0-D ndarray).
    T : Power transmittance (0-D ndarray).
    """
    c = 299792458.0
    omega = 2 * nd.pi * c / wavelength
    k0 = 2 * nd.pi / wavelength

    eps0 = stack.superstrate.epsilon(omega=omega)
    mu0 = stack.superstrate.mu(omega=omega)
    eps1 = stack.substrate.epsilon(omega=omega)
    mu1 = stack.substrate.mu(omega=omega)

    kz0 = _kz_single(eps0, mu0, k0, kx)
    kz1 = _kz_single(eps1, mu1, k0, kx)

    if polarization == Polarization.TE:
        Z0 = kz0 / mu0
        Z1 = kz1 / mu1
        r = (Z0 - Z1) / (Z0 + Z1)
        t = 2 * Z0 / (Z0 + Z1)
    else:
        raise NotImplementedError("Only TE polarization is currently supported")

    R = nd.abs(r) ** 2
    T = nd.real(kz1 / mu1) / nd.real(kz0 / mu0) * nd.abs(t) ** 2

    return R, T
