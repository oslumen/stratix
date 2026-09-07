"""S-matrix method for stratified-medium transfer-matrix computations."""

from __future__ import annotations

import numdiff as nd
from phokaia import Polarization
from phokaia import Stack

from ._medium_params import _medium_params
from ._util import _interface_coeffs
from ._util import _resolve_thicknesses
from ._util import _safe_R
from ._util import _safe_T

#: A 2x2 S-matrix carried as its four components ``(S11, S12, S21, S22)``.
#: The layer loop never assembles them into an array: stacking and
#: re-indexing a 2x2 of full sweep grids costs two allocations per
#: combination and buys nothing the scalar form does not give.
_Components = tuple[nd.ndarray, nd.ndarray, nd.ndarray, nd.ndarray]


def _redheffer_star(S_A: _Components, S_B: _Components) -> _Components:
    """Combine two S-matrices, given component-wise, via the star product.

    Parameters
    ----------
    S_A : Components ``(A11, A12, A21, A22)`` of the left-hand S-matrix.
    S_B : Components ``(B11, B12, B21, B22)`` of the right-hand S-matrix.

    Returns
    -------
    Components of ``S_A (x) S_B``.
    """
    A11, A12, A21, A22 = S_A
    B11, B12, B21, B22 = S_B

    denom = 1.0 - A22 * B11
    safe_denom = nd.where(denom == 0, nd.ones_like(denom), denom)
    S11 = A11 + A12 * B11 * A21 / safe_denom
    S12 = A12 * B12 / safe_denom
    S21 = B21 * A21 / safe_denom
    S22 = B22 + B21 * A22 * B12 / safe_denom

    return S11, S12, S21, S22


def _propagation_star(S_A: _Components, p: nd.ndarray) -> _Components:
    """Combine an S-matrix with a layer's propagation S-matrix.

    A propagation matrix is ``[[0, p], [p, 0]]``, so the star product's
    denominator ``1 - A22 * 0`` collapses to 1: the combination is four
    multiplications with no division and no guard.  Specialising it keeps
    the inner loop's arithmetic proportional to what the physics needs.

    Parameters
    ----------
    S_A : Components of the S-matrix accumulated so far.
    p : Phase factor ``exp(1j * kz * thickness)`` of the layer.

    Returns
    -------
    Components of ``S_A (x) [[0, p], [p, 0]]``.
    """
    A11, A12, A21, A22 = S_A
    return A11, A12 * p, p * A21, A22 * p * p


def _smatrix_solve(
    stack: Stack,
    wavelength: float,
    kx: float,
    polarization: Polarization,
    thicknesses: nd.ndarray | None = None,
) -> tuple[nd.ndarray, nd.ndarray, dict]:
    """Compute R and T for a multilayer stack via S-matrix assembly.

    Walks the stack left-to-right (superstrate to substrate), folding each
    interface and each layer's propagation into a running S-matrix via the
    Redheffer star product.  The running matrix is carried as four scalar
    components, so a solve allocates nothing per layer beyond the
    components themselves.

    Parameters
    ----------
    stack : Multilayer stack (superstrate, layers, substrate).
    wavelength : Vacuum wavelength in meters.
    kx : In-plane wavevector component in rad/m.
    polarization : ``TE`` or ``TM``.
    thicknesses : Optional 1-D ndarray overriding the stack's layer
        thicknesses.  Must have length ``len(stack.layers)``.  When
        provided, ``stack.layers[i].thickness`` is ignored in favour
        of ``thicknesses[i]``, enabling autodiff w.r.t. thickness.

    Returns
    -------
    R : Power reflectance (0-D ndarray).
    T : Power transmittance (0-D ndarray).
    intermediates : Dict of per-medium quantities the solve computed
        anyway.  Field profiles and per-layer absorption rebuild the
        interface coefficients from these on demand, so no per-layer
        matrix is retained by a solve that was not asked for one.
    """
    _omega, _k0, kzs, _, _, denom_vals = _medium_params(
        stack, wavelength, kx, polarization
    )
    Zs = [kz / denom for kz, denom in zip(kzs, denom_vals, strict=True)]
    n_interfaces = len(kzs) - 1

    kz0, denom0 = kzs[0], denom_vals[0]

    _thicknesses = _resolve_thicknesses(stack, thicknesses)

    S_total = _interface_coeffs(Zs[0], Zs[1])

    for i in range(1, n_interfaces):
        phi = kzs[i] * _thicknesses[i - 1]
        S_total = _propagation_star(S_total, nd.exp(1j * phi))
        S_total = _redheffer_star(S_total, _interface_coeffs(Zs[i], Zs[i + 1]))

    r_total = S_total[0]
    t_total = S_total[2]

    R = _safe_R(r_total, kz0, denom0)
    T = _safe_T(t_total, kz0, denom0, kzs[-1], denom_vals[-1])

    intermediates = {
        "kzs": kzs,
        "denom_vals": denom_vals,
        "r_total": r_total,
        "t_total": t_total,
        "thicknesses": _thicknesses,
        "polarization": polarization,
    }

    return R, T, intermediates
