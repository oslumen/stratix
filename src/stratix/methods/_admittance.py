"""Admittance recursion method for stratified-medium computations.

The optical admittance Y = H_tangential / E_tangential at an interface
is propagated through the stack from substrate to superstrate.
Reflection and transmission are extracted from the effective input admittance
at the first interface.
"""

from __future__ import annotations

import numdiff as nd
from phokaia import Stack

from .._types import Polarization
from ._smatrix import _kz_single


def _admittance_solve(
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

    Y_super = kzs[0] / denom_vals[0]
    Y_sub = kzs[-1] / denom_vals[-1]

    Y_in = Y_sub
    for i in range(len(stack.layers) - 1, -1, -1):
        Y_layer = kzs[i + 1] / denom_vals[i + 1]
        phi = kzs[i + 1] * stack.layers[i].thickness
        t = nd.tan(phi)
        Y_in = Y_layer * (Y_in - 1j * Y_layer * t) / (Y_layer - 1j * Y_in * t)

    r = (Y_super - Y_in) / (Y_super + Y_in)
    R = nd.abs(r) ** 2

    E = nd.array(1.0) + r
    Y_current = Y_in
    for i in range(len(stack.layers)):
        Y_layer = kzs[i + 1] / denom_vals[i + 1]
        phi = kzs[i + 1] * stack.layers[i].thickness
        cos_phi = nd.cos(phi)
        sin_phi = nd.sin(phi)

        E = E * (cos_phi + 1j * Y_current / Y_layer * sin_phi)
        t = nd.tan(phi)
        Y_current = Y_layer * (Y_current + 1j * Y_layer * t) / (Y_layer + 1j * Y_current * t)

    t_total = E
    T = (
        nd.real(kzs[-1] / denom_vals[-1])
        / nd.real(kzs[0] / denom_vals[0])
        * nd.abs(t_total) ** 2
    )

    return R, T, {}
