"""Shared utilities for stratified-medium solver methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numdiff as nd

if TYPE_CHECKING:
    from phokaia import Stack


def _rel_tol() -> nd.ndarray:
    """Relative tolerance for the active backend's real precision.

    ``sqrt(eps)`` rather than ``eps`` because every guarded quantity in
    this module reaches the guard through a subtraction that can cancel:
    ``kz^2 = eps*mu*k0^2 - kx^2`` near a medium's light line, and
    ``1 - A22*B11`` in the Redheffer product.  A cancelling subtraction
    keeps only half the significant digits, so ``sqrt(eps)`` is the floor
    below which the result carries no reliable information.

    The precision is read off a freshly built real scalar rather than off
    the guarded value itself.  That keeps the tolerance in step with
    ``nd.set_precision`` on every backend, and — because the guarded
    value may be an autodiff tracer — avoids putting a constant that
    nothing differentiates onto the tape.
    """
    return nd.sqrt(nd.array(nd.finfo(nd.array(0.0).dtype).eps))


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


def _no_incident_flux(
    kz0: nd.ndarray, denom0: nd.ndarray, k0: nd.ndarray
) -> nd.ndarray:
    """Boolean mask: the incident wave carries no usable z-directed flux.

    The flux normalisation is ``Re(kz0/denom0)``, and both R and T divide
    by it.  It has to be compared against a scale rather than against
    zero, because two distinct regimes drive it arbitrarily close to zero
    without landing exactly on it:

    * **Evanescent incidence.**  Past the superstrate's light line
      ``kz0`` is imaginary, so a lossless medium gives exactly zero.  A
      trace of loss (any complex ``epsilon`` or ``mu``) tips the real
      part to a tiny non-zero value instead, and the flux ratio then
      diverges as ``1/loss``.  The natural scale here is the wave's own
      magnitude ``|kz0|``: the wave is evanescent when its real part is
      negligible against it.
    * **Grazing incidence.**  As ``theta -> 90 deg`` the whole of ``kz0``
      collapses, so ``Re(kz0)`` is not small against ``|kz0|`` — it is
      small against the vacuum wavevector ``k0``.

    Taking ``max(|kz0|, k0)`` as the scale covers both: the guard fires
    when the incident flux is negligible against the larger of the wave's
    own size and the vacuum scale.  ``denom0`` divides through unchanged,
    so it cancels from the comparison up to its magnitude.

    The comparison is on the *magnitude* ``|Re(kz0/denom0)|``, so the
    guard fires only where the flux is negligible.  A normalisation that
    is large and negative — the branch convention picks the wrong sign in
    a gain or negative-index superstrate — is a separate, pre-existing
    problem in ``_kz_single``, not a collapsed flux.  Folding it in here
    would report ``R = 1, T = 0`` for an ordinary propagating wave and
    hide that problem behind a physically plausible answer.

    .. warning::

       The threshold is set by float precision, not by physics, and that
       is a known defect — see issue #57.  Past the superstrate's light
       line with genuine loss, ``T`` diverges as ``1/Im(epsilon)`` while
       ``|r|^2`` stays finite, so the guard's ``sqrt(eps)`` cut-off puts a
       seven-decade jump in ``T`` at a dtype-dependent location: at
       ``Im(epsilon) = 1e-7`` single precision answers ``T = 0`` and
       double answers ``T = 1.5e7``.  The correct criterion is the
       loss-independent light line ``kx > Re(n_super) * k0``.

       More broadly, ``R`` and ``T`` are not an energy partition for *any*
       lossy superstrate, evanescent or not: the incident and reflected
       waves share that medium, and their cross term carries real
       z-directed flux that ``|r|^2`` and ``T`` do not account for.
       ``R + T + sum(A)`` measures 1.00082 at ``Im(epsilon) = 0.01`` and
       1.124 at ``Im(epsilon) = 1``.  A lossy *substrate* is unaffected —
       it holds a single outgoing wave, so its flux is unambiguous and
       energy balance closes exactly.

    Parameters
    ----------
    kz0 : Out-of-plane wavevector in the incident medium.
    denom0 : ``mu`` for TE, ``epsilon`` for TM, in the incident medium.
    k0 : Vacuum wavevector, the natural scale for a wavevector ratio.

    Returns
    -------
    Boolean ndarray broadcasting with the sweep shape.
    """
    z0_real = nd.real(kz0 / denom0)
    scale = nd.maximum(nd.abs(kz0), nd.abs(k0)) / nd.abs(denom0)
    return nd.abs(z0_real) <= _rel_tol() * scale


def _safe_R(
    r_coeff: nd.ndarray, kz0: nd.ndarray, denom0: nd.ndarray, k0: nd.ndarray
) -> nd.ndarray:
    """Compute power reflectance; returns 1 when no flux enters the stack.

    The standard formula ``R = |r|^2`` assumes a propagating incident
    wave.  When the incident medium carries no usable z-directed flux —
    evanescent or grazing incidence, see :func:`_no_incident_flux` — no
    power flows in the +z direction, so the physical reflectance is 1.
    """
    raw = nd.abs(r_coeff) ** 2
    return nd.where(_no_incident_flux(kz0, denom0, k0), nd.ones_like(raw), raw)


def _safe_T(
    t_coeff: nd.ndarray,
    kz0: nd.ndarray,
    denom0: nd.ndarray,
    kzN: nd.ndarray,
    denomN: nd.ndarray,
    k0: nd.ndarray,
) -> nd.ndarray:
    """Compute power transmittance; returns 0 when no flux enters the stack.

    The standard formula ``T = Re(kzN/denomN) / Re(kz0/denom0) * |t|^2``
    assumes a propagating incident wave.  When the incident medium
    carries no usable z-directed flux — evanescent or grazing incidence,
    see :func:`_no_incident_flux` — no power is carried toward the stack,
    so the physical transmittance is 0.
    """
    z0_real = nd.real(kz0 / denom0)
    no_flux = _no_incident_flux(kz0, denom0, k0)
    safe_z0 = nd.where(no_flux, nd.ones_like(z0_real), z0_real)
    raw = nd.real(kzN / denomN) / safe_z0 * nd.abs(t_coeff) ** 2
    return nd.where(no_flux, nd.zeros_like(raw), raw)


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
