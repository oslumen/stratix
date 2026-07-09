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


def _redheffer_star(
    S_A: nd.ndarray, S_B: nd.ndarray
) -> nd.ndarray:
    """Combine two 2x2 S-matrices via the Redheffer star product.

    Parameters
    ----------
    S_A : 2x2 ndarray (left-hand S-matrix).
    S_B : 2x2 ndarray (right-hand S-matrix).

    Returns
    -------
    2x2 ndarray: S_A (x) S_B.
    """
    A11, A12 = S_A[0, 0], S_A[0, 1]
    A21, A22 = S_A[1, 0], S_A[1, 1]
    B11, B12 = S_B[0, 0], S_B[0, 1]
    B21, B22 = S_B[1, 0], S_B[1, 1]

    denom = 1.0 - A22 * B11
    S11 = A11 + A12 * B11 * A21 / denom
    S12 = A12 * B12 / denom
    S21 = B21 * A21 / denom
    S22 = B22 + B21 * A22 * B12 / denom

    return nd.array([[S11, S12], [S21, S22]])


def _interface_smatrix(Z_left: nd.ndarray, Z_right: nd.ndarray) -> nd.ndarray:
    """Build the 2x2 interface S-matrix from wave impedances.

    Parameters
    ----------
    Z_left : Wave impedance of the incident medium.
    Z_right : Wave impedance of the transmitted medium.

    Returns
    -------
    2x2 ndarray [[r, t_rev], [t_fwd, -r]].
    """
    r = (Z_left - Z_right) / (Z_left + Z_right)
    t_fwd = 2 * Z_left / (Z_left + Z_right)
    t_rev = 2 * Z_right / (Z_left + Z_right)
    return nd.array([[r, t_rev], [t_fwd, -r]])


def _propagation_smatrix(kz: nd.ndarray, thickness: float) -> nd.ndarray:
    """Build the 2x2 propagation S-matrix for a homogeneous layer.

    Parameters
    ----------
    kz : Out-of-plane wavevector in the layer.
    thickness : Layer thickness in meters.

    Returns
    -------
    2x2 ndarray [[0, exp(i*phi)], [exp(i*phi), 0]].
    """
    phi = kz * thickness
    p = nd.exp(1j * phi)
    return nd.array([[0, p], [p, 0]])


def _smatrix_solve(
    stack: Stack, wavelength: float, kx: float, polarization: Polarization
) -> tuple[nd.ndarray, nd.ndarray]:
    """Compute R and T for a multilayer stack via S-matrix assembly.

    Assembles interface and propagation S-matrices and combines them
    left-to-right (superstrate → substrate) via the Redheffer star product.

    Parameters
    ----------
    stack : Multilayer stack (superstrate, layers, substrate).
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

    if polarization != Polarization.TE:
        raise NotImplementedError("Only TE polarization is currently supported")

    media = [stack.superstrate] + [layer.material for layer in stack.layers] + [stack.substrate]
    epsilons = [m.epsilon(omega=omega) for m in media]
    mus = [m.mu(omega=omega) for m in media]
    kzs = [_kz_single(eps, mu, k0, kx) for eps, mu in zip(epsilons, mus, strict=True)]
    Zs = [kz / mu for kz, mu in zip(kzs, mus, strict=True)]

    n_interfaces = len(media) - 1

    S_total = _interface_smatrix(Zs[0], Zs[1])

    for i in range(1, n_interfaces):
        d = stack.layers[i - 1].thickness
        P = _propagation_smatrix(kzs[i], d)
        S_total = _redheffer_star(S_total, P)
        S_int = _interface_smatrix(Zs[i], Zs[i + 1])
        S_total = _redheffer_star(S_total, S_int)

    r_total = S_total[0, 0]
    t_total = S_total[1, 0]

    R = nd.abs(r_total) ** 2
    T = nd.real(kzs[-1] / mus[-1]) / nd.real(kzs[0] / mus[0]) * nd.abs(t_total) ** 2

    return R, T
