"""Dirichlet-to-Neumann (DTN) map method.

Propagates the effective impedance bottom-to-top via a Möbius composition,
then computes R from the total reflection coefficient and T via
characteristic-matrix forward propagation through the stack.
"""

from __future__ import annotations

import numdiff as nd
from phokaia import Polarization
from phokaia import Stack

from ._medium_params import _medium_params
from ._util import _resolve_thicknesses
from ._util import _safe_R
from ._util import _safe_T


def _dtn_solve(
    stack: Stack,
    wavelength: float,
    kx: float,
    polarization: Polarization,
    thicknesses: nd.ndarray | None = None,
) -> tuple[nd.ndarray, nd.ndarray, dict]:
    _omega, k0, kzs, _, _, denom_vals = _medium_params(
        stack, wavelength, kx, polarization
    )

    _d = _resolve_thicknesses(stack, thicknesses)

    Z_super = kzs[0] / denom_vals[0]
    Z_sub = kzs[-1] / denom_vals[-1]

    Z_in = Z_sub
    for i in range(len(stack.layers) - 1, -1, -1):
        Z_layer = kzs[i + 1] / denom_vals[i + 1]
        phi = kzs[i + 1] * _d[i]
        t = nd.tan(phi)
        Z_in = Z_layer * (Z_in - 1j * Z_layer * t) / (Z_layer - 1j * Z_in * t)

    r = (Z_super - Z_in) / (Z_super + Z_in)
    kz0, denom0 = kzs[0], denom_vals[0]
    R = _safe_R(r, kz0, denom0, k0)

    E = nd.array(1.0) + r
    Z_current = Z_in
    for i in range(len(stack.layers)):
        Z_layer = kzs[i + 1] / denom_vals[i + 1]
        phi = kzs[i + 1] * _d[i]
        cos_phi = nd.cos(phi)
        sin_phi = nd.sin(phi)

        E = E * (cos_phi + 1j * Z_current / Z_layer * sin_phi)
        t = nd.tan(phi)
        Z_current = Z_layer * (Z_current + 1j * Z_layer * t) / (Z_layer + 1j * Z_current * t)

    t_total = E
    T = _safe_T(t_total, kz0, denom0, kzs[-1], denom_vals[-1], k0)

    return R, T, {"thicknesses": _d}
