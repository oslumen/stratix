"""Abélès 2x2 characteristic matrix method.

The Abélès formalism assembles a 2x2 characteristic matrix per layer,
multiplies them, and extracts R/T from the total matrix. Equivalent
to S-matrix for all-dielectric stacks, but can become numerically
unstable for thick, metallic, or evanescent layers.
"""

from __future__ import annotations

import numdiff as nd
from phokaia import Stack

from .._types import Polarization
from ._smatrix import _kz_single


def _abeles_solve(
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

    Zs = [kz / denom for kz, denom in zip(kzs, denom_vals, strict=True)]
    Z_0 = Zs[0]
    Z_s = Zs[-1]

    m11 = nd.array(1.0 + 0j)
    m12 = nd.array(0j)
    m21 = nd.array(0j)
    m22 = nd.array(1.0 + 0j)

    for i in range(len(stack.layers)):
        Z = Zs[i + 1]
        phi = kzs[i + 1] * stack.layers[i].thickness

        cos_phi = nd.cos(phi)
        sin_phi = nd.sin(phi)

        L11 = cos_phi
        L12 = 1j * sin_phi / Z
        L21 = 1j * Z * sin_phi
        L22 = cos_phi

        n11 = m11 * L11 + m12 * L21
        n12 = m11 * L12 + m12 * L22
        n21 = m21 * L11 + m22 * L21
        n22 = m21 * L12 + m22 * L22

        m11, m12, m21, m22 = n11, n12, n21, n22

    A = m11 + m12 * Z_s
    B = m21 + m22 * Z_s

    denom = Z_0 * A + B
    r_total = (Z_0 * A - B) / denom
    t_total = nd.array(2) * Z_0 / denom

    R = nd.abs(r_total) ** 2
    T = (
        nd.real(kzs[-1] / denom_vals[-1])
        / nd.real(kzs[0] / denom_vals[0])
        * nd.abs(t_total) ** 2
    )

    return R, T, {}
