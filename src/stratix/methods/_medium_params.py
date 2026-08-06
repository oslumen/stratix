"""Pre-computed medium parameters (omega, k0, wavevectors, impedances).

Shared by Abélès, admittance, DTN, S-matrix, and autodiff solvers.
"""

from __future__ import annotations

import numdiff as nd
from phokaia import Polarization
from phokaia import Stack


def _kz_single(epsilon: nd.ndarray, mu: nd.ndarray, k0: float, kx: float) -> nd.ndarray:
    """Compute the out-of-plane wavevector component kz.

    Chooses the physical branch: Im(kz) >= 0, or Re(kz) >= 0 when Im(kz) = 0.
    """
    kz_sq = epsilon * mu * k0**2 - kx**2
    kz_sq = kz_sq + 0j
    kz = nd.sqrt(kz_sq)
    neg = (nd.imag(kz) < 0) | ((nd.imag(kz) == 0) & (nd.real(kz) < 0))
    return nd.where(neg, -kz, kz)


def _medium_params(
    stack: Stack, wavelength: float, kx: float, polarization: Polarization
) -> tuple:
    """Pre-compute omega, k0, kzs, epsilons, mus, and denom_vals for a stack.

    Parameters
    ----------
    stack : Multilayer stack (superstrate, layers, substrate).
    wavelength : Vacuum wavelength in meters.
    kx : In-plane wavevector component in rad/m.
    polarization : ``TE`` or ``TM``.

    Returns
    -------
    omega : 0-D ndarray — angular frequency (rad/s).
    k0 : 0-D ndarray — vacuum wavevector (rad/m).
    kzs : List of 0-D ndarrays — out-of-plane wavevector per medium.
    epsilons : List of 0-D ndarrays — epsilon(omega) per medium.
    mus : List of 0-D ndarrays — mu(omega) per medium.
    denom_vals : List of 0-D ndarrays — mus for TE, epsilons for TM.
    """
    c = 299792458.0
    omega = nd.array(2 * nd.pi * c / wavelength)
    k0 = nd.array(2 * nd.pi / wavelength)
    kx = nd.array(kx)

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

    return omega, k0, kzs, epsilons, mus, denom_vals
