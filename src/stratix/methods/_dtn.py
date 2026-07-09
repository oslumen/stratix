"""Dirichlet-to-Neumann (DTN) map method.

Propagates the effective impedance bottom-to-top via a Möbius composition,
then computes R from the total reflection coefficient and T via
characteristic-matrix forward propagation through the stack.
"""

from __future__ import annotations

import numdiff as nd
from phokaia import Stack

from .._types import Polarization
from ._smatrix import _kz_single


def _dtn_solve(
    stack: Stack, wavelength: float, kx: float, polarization: Polarization
) -> tuple[nd.ndarray, nd.ndarray, dict]:
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

    Z_super = kzs[0] / denom_vals[0]
    Z_sub = kzs[-1] / denom_vals[-1]

    Z_in = Z_sub
    for i in range(len(stack.layers) - 1, -1, -1):
        Z_layer = kzs[i + 1] / denom_vals[i + 1]
        phi = kzs[i + 1] * stack.layers[i].thickness
        t = nd.tan(phi)
        Z_in = Z_layer * (Z_in - 1j * Z_layer * t) / (Z_layer - 1j * Z_in * t)

    r = (Z_super - Z_in) / (Z_super + Z_in)
    R = nd.abs(r) ** 2

    E = nd.array(1.0) + r
    Z_current = Z_in
    for i in range(len(stack.layers)):
        Z_layer = kzs[i + 1] / denom_vals[i + 1]
        phi = kzs[i + 1] * stack.layers[i].thickness
        cos_phi = nd.cos(phi)
        sin_phi = nd.sin(phi)

        E = E * (cos_phi + 1j * Z_current / Z_layer * sin_phi)
        t = nd.tan(phi)
        Z_current = Z_layer * (Z_current + 1j * Z_layer * t) / (Z_layer + 1j * Z_current * t)

    t_total = E
    T = (
        nd.real(kzs[-1] / denom_vals[-1])
        / nd.real(kzs[0] / denom_vals[0])
        * nd.abs(t_total) ** 2
    )

    return R, T, {}
