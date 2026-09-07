"""Shared utilities for stratified-medium solver methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numdiff as nd

if TYPE_CHECKING:
    from phokaia import Stack


def _resolve_thicknesses(
    stack: Stack,
    thicknesses: nd.ndarray | None = None,
) -> nd.ndarray:
    """Resolve layer thicknesses from stack or override.

    When ``thicknesses`` is not ``None`` it overrides the frozen
    ``Layer.thickness`` values in ``stack.layers``, enabling autodiff
    w.r.t. thickness via ``nd.grad``.

    Parameters
    ----------
    stack : Multilayer stack.
    thicknesses : Optional 1-D array, length ``len(stack.layers)``.

    Returns
    -------
    1-D ndarray of resolved thicknesses.
    """
    if thicknesses is None:
        return nd.array([layer.thickness for layer in stack.layers])
    return thicknesses


def _safe_R(r_coeff: nd.ndarray, kz0: nd.ndarray, denom0: nd.ndarray) -> nd.ndarray:
    """Compute power reflectance; returns 1 for evanescent incidence.

    The standard formula ``R = |r|^2`` assumes a propagating incident wave
    (``Re(kz0 / denom0) > 0``).  When the incident medium is evanescent
    (``Re(kz0 / denom0) == 0``) no power flows in the +z direction, so the
    physical reflectance is always 1.
    """
    z0_real = nd.real(kz0 / denom0)
    raw = nd.abs(r_coeff) ** 2
    return nd.where(z0_real == 0, nd.ones_like(raw), raw)


def _safe_T(
    t_coeff: nd.ndarray,
    kz0: nd.ndarray,
    denom0: nd.ndarray,
    kzN: nd.ndarray,
    denomN: nd.ndarray,
) -> nd.ndarray:
    """Compute power transmittance; returns 0 for evanescent incidence.

    The standard formula ``T = Re(kzN/denomN) / Re(kz0/denom0) * |t|^2``
    assumes a propagating incident wave.  When the incident medium is
    evanescent (``Re(kz0 / denom0) == 0``) no power is carried toward the
    stack, so the physical transmittance is always 0.
    """
    z0_real = nd.real(kz0 / denom0)
    safe_z0 = nd.where(z0_real == 0, nd.ones_like(z0_real), z0_real)
    raw = nd.real(kzN / denomN) / safe_z0 * nd.abs(t_coeff) ** 2
    return nd.where(z0_real == 0, nd.zeros_like(raw), raw)


def _interface_coeffs(
    Z_left: nd.ndarray, Z_right: nd.ndarray
) -> tuple[nd.ndarray, nd.ndarray, nd.ndarray, nd.ndarray]:
    """Return the four components of an interface S-matrix.

    The matrix is ``[[r, t_rev], [t_fwd, -r]]``; handing back its
    components rather than the assembled 2x2 array keeps the S-matrix
    layer loop free of stacked allocations, and lets the amplitude
    recursion rebuild an interface on demand instead of retaining one
    per layer.

    Parameters
    ----------
    Z_left : Wave impedance of the incident medium.
    Z_right : Wave impedance of the transmitted medium.

    Returns
    -------
    Tuple ``(S11, S12, S21, S22) = (r, t_rev, t_fwd, -r)``.
    """
    total = Z_left + Z_right
    r = (Z_left - Z_right) / total
    t_fwd = 2 * Z_left / total
    t_rev = 2 * Z_right / total
    return r, t_rev, t_fwd, -r
