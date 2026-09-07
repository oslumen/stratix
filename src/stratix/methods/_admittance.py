"""Admittance recursion method for stratified-medium computations.

The optical admittance Y = H_tangential / E_tangential at an interface
is propagated through the stack from substrate to superstrate.
Reflection and transmission are extracted from the effective input admittance
at the first interface.
"""

from __future__ import annotations

import numdiff as nd
from phokaia import Polarization
from phokaia import Stack

from ._medium_params import _medium_params
from ._util import _resolve_thicknesses
from ._util import _safe_R
from ._util import _safe_T


def _admittance_solve(
    stack: Stack,
    wavelength: float,
    kx: float,
    polarization: Polarization,
    thicknesses: nd.ndarray | None = None,
) -> tuple[nd.ndarray, nd.ndarray, dict]:
    _omega, _k0, kzs, _, _, denom_vals = _medium_params(
        stack, wavelength, kx, polarization
    )

    _d = _resolve_thicknesses(stack, thicknesses)

    Y_super = kzs[0] / denom_vals[0]
    Y_sub = kzs[-1] / denom_vals[-1]

    Y_in = Y_sub
    for i in range(len(stack.layers) - 1, -1, -1):
        Y_layer = kzs[i + 1] / denom_vals[i + 1]
        phi = kzs[i + 1] * _d[i]
        t = nd.tan(phi)
        Y_in = Y_layer * (Y_in - 1j * Y_layer * t) / (Y_layer - 1j * Y_in * t)

    r = (Y_super - Y_in) / (Y_super + Y_in)
    kz0, denom0 = kzs[0], denom_vals[0]
    R = _safe_R(r, kz0, denom0)

    E = nd.array(1.0) + r
    Y_current = Y_in
    for i in range(len(stack.layers)):
        Y_layer = kzs[i + 1] / denom_vals[i + 1]
        phi = kzs[i + 1] * _d[i]
        cos_phi = nd.cos(phi)
        sin_phi = nd.sin(phi)

        E = E * (cos_phi + 1j * Y_current / Y_layer * sin_phi)
        t = nd.tan(phi)
        Y_current = Y_layer * (Y_current + 1j * Y_layer * t) / (Y_layer + 1j * Y_current * t)

    t_total = E
    T = _safe_T(t_total, kz0, denom0, kzs[-1], denom_vals[-1])

    return R, T, {"thicknesses": _d}
