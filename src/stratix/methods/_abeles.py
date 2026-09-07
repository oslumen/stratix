"""Abélès 2x2 characteristic matrix method.

The Abélès formalism assembles a 2x2 characteristic matrix per layer,
multiplies them, and extracts R/T from the total matrix. Equivalent
to S-matrix for all-dielectric stacks, but can become numerically
unstable for thick, metallic, or evanescent layers.
"""

from __future__ import annotations

import numdiff as nd
from phokaia import Polarization
from phokaia import Stack

from ._medium_params import _medium_params
from ._util import _resolve_thicknesses
from ._util import _safe_R
from ._util import _safe_T


def _abeles_solve(
    stack: Stack,
    wavelength: float,
    kx: float,
    polarization: Polarization,
    thicknesses: nd.ndarray | None = None,
) -> tuple[nd.ndarray, nd.ndarray, dict]:
    _omega, k0, kzs, _, _, denom_vals = _medium_params(
        stack, wavelength, kx, polarization
    )
    Zs = [kz / denom for kz, denom in zip(kzs, denom_vals, strict=True)]
    Z_0 = Zs[0]
    Z_s = Zs[-1]

    _d = _resolve_thicknesses(stack, thicknesses)

    m11 = nd.array(1.0 + 0j)
    m12 = nd.array(0j)
    m21 = nd.array(0j)
    m22 = nd.array(1.0 + 0j)

    for i in range(len(stack.layers)):
        Z = Zs[i + 1]
        phi = kzs[i + 1] * _d[i]

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

    kz0, denom0 = kzs[0], denom_vals[0]
    R = _safe_R(r_total, kz0, denom0, k0)
    T = _safe_T(t_total, kz0, denom0, kzs[-1], denom_vals[-1], k0)

    return R, T, {"thicknesses": _d}
