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


def _wave_admittances(
    kzs: list[nd.ndarray], denom_vals: list[nd.ndarray]
) -> list[nd.ndarray]:
    """Wave admittance ``Y = kz / denom`` for each medium.

    ``denom`` is ``mu`` for TE and ``epsilon`` for TM, so one expression
    covers both polarizations.  ``Y`` is the ratio of the tangential
    magnetic to the tangential electric field carried by a single
    forward-going plane wave, which is why every formulation in this
    package -- Fresnel coefficients, the Abeles characteristic matrix, the
    admittance recursion, the Dirichlet-to-Neumann kernel -- is written in
    terms of it rather than in terms of ``kz`` and the material constant
    separately.

    Parameters
    ----------
    kzs : Out-of-plane wavevector per medium, superstrate first.
    denom_vals : ``mu`` (TE) or ``epsilon`` (TM) per medium, same order.

    Returns
    -------
    List of admittances, one per medium.
    """
    return [kz / denom for kz, denom in zip(kzs, denom_vals, strict=True)]


def _layer_phases(kzs: list[nd.ndarray], thicknesses: nd.ndarray) -> list[nd.ndarray]:
    """Phase thickness ``phi = kz * d`` for each layer.

    ``kzs`` is indexed by *medium* and ``thicknesses`` by *layer*, and the
    two are offset by one because ``kzs[0]`` is the superstrate.  Getting
    that offset wrong shifts every layer's optical thickness onto its
    neighbour, so it is resolved in one place rather than in each solver.

    Parameters
    ----------
    kzs : Out-of-plane wavevector per medium, superstrate first.
    thicknesses : Layer thicknesses, as resolved by
        :func:`_resolve_thicknesses`.

    Returns
    -------
    List of phase thicknesses, one per layer.
    """
    return [kzs[i + 1] * thicknesses[i] for i in range(len(kzs) - 2)]


def _admittance_step(
    Y_outer: nd.ndarray, Y_layer: nd.ndarray, phi: nd.ndarray
) -> nd.ndarray:
    """Transform an admittance across one layer.

    The Moebius map that carries the admittance seen on one face of a layer
    to the admittance seen on the other::

        Y' = Y_layer * (Y + i*Y_layer*tan(phi)) / (Y_layer + i*Y*tan(phi))

    The admittance recursion applies it twice per layer and in opposite
    directions -- once walking up from the substrate to find the input
    admittance, once walking back down to accumulate the transmitted
    amplitude.  The two differ only in the sign of the phase, so passing
    ``-phi`` covers the upward pass and no second body is needed.

    Parameters
    ----------
    Y_outer : Admittance seen at the face being propagated from.
    Y_layer : Wave admittance of the layer itself.
    phi : Phase thickness ``kz * d``, negated to propagate the other way.

    Returns
    -------
    Admittance seen at the opposite face.
    """
    t = nd.tan(phi)
    return Y_layer * (Y_outer + 1j * Y_layer * t) / (Y_layer + 1j * Y_outer * t)


def _no_incident_flux(
    kx: nd.ndarray, epsilon0: nd.ndarray, mu0: nd.ndarray, k0: nd.ndarray
) -> nd.ndarray:
    """Boolean mask: the incident wave carries no z-directed flux.

    Both R and T divide by the incident flux ``Re(kz0/denom0)``, so the
    regime where that flux vanishes has to be identified before the
    division rather than repaired after it.  The criterion is the
    superstrate's own light line::

        |kx| >= Re(n_super) * k0,   n_super = sqrt(epsilon0 * mu0)

    which is where ``kz0**2 = epsilon0*mu0*k0**2 - kx**2`` stops being
    positive.  Inside the light cone ``kz0`` is real and the wave
    propagates; on the line it is exactly zero (grazing incidence); past
    it ``kz0`` is imaginary and the wave is evanescent.  Taking ``>=``
    rather than ``>`` folds exact grazing in with the evanescent side,
    which is where it belongs: no power crosses the first interface
    either way.

    The comparison is between two computed wavevectors, so it says the
    same thing at every working precision — unlike a tolerance on the
    residual flux, which decides the same physical input differently in
    single and double precision (issue #57).  Only the ulp of representing
    ``kx`` itself is left, against the multi-decade band the tolerance had.

    The formula assumes ``epsilon0 * mu0`` is real: ``Re(sqrt(eps*mu))``
    is not the light line otherwise, since a complex ``eps*mu`` has no
    sharp propagating/evanescent boundary to find.
    :func:`~stratix.methods._medium_params._reject_lossy_superstrate`
    normally rules that out — R and T do not partition energy there — but
    it cannot read a tracer, so a lossy superstrate can still reach this
    function inside a dispersive ``nd.jit`` trace.  The mask is then
    approximate rather than wrong-by-construction, which is the best that
    is available without a concrete value to branch on.

    A metallic superstrate (``epsilon0 * mu0 < 0``) has ``Re(n_super) =
    0``, so the mask covers every kx including normal incidence — correct,
    since nothing propagates in it.  A negative-index superstrate
    (``epsilon0 < 0`` *and* ``mu0 < 0``) has a real index and is not
    masked, even though ``Re(kz0/denom0)`` comes out large and negative
    there: that sign is a branch-choice problem in ``_kz_single``, and
    reporting it as ``R = 1, T = 0`` — a plausible-looking answer for an
    ordinary propagating wave — would bury it.

    ``|kx|`` because ``kz0`` depends on ``kx**2``: the guard has to be
    symmetric in the in-plane direction, as the physics is.

    Parameters
    ----------
    kx : In-plane wavevector component.
    epsilon0, mu0 : Permittivity and permeability of the incident medium.
    k0 : Vacuum wavevector.

    Returns
    -------
    Boolean ndarray broadcasting with the sweep shape.
    """
    n_super = nd.sqrt(epsilon0 * mu0 + 0j)
    return nd.abs(kx) >= nd.real(n_super) * nd.abs(k0)


def _safe_R(r_coeff: nd.ndarray, no_flux: nd.ndarray) -> nd.ndarray:
    """Compute power reflectance; returns 1 when no flux enters the stack.

    The standard formula ``R = |r|^2`` assumes a propagating incident
    wave.  When the incident medium carries none — evanescent or grazing
    incidence, see :func:`_no_incident_flux` — no power flows in the +z
    direction, so the physical reflectance is 1.
    """
    raw = nd.abs(r_coeff) ** 2
    return nd.where(no_flux, nd.ones_like(raw), raw)


def _safe_T(
    t_coeff: nd.ndarray,
    kz0: nd.ndarray,
    denom0: nd.ndarray,
    kzN: nd.ndarray,
    denomN: nd.ndarray,
    no_flux: nd.ndarray,
) -> nd.ndarray:
    """Compute power transmittance; returns 0 when no flux enters the stack.

    The standard formula ``T = Re(kzN/denomN) / Re(kz0/denom0) * |t|^2``
    assumes a propagating incident wave.  When the incident medium
    carries none — evanescent or grazing incidence, see
    :func:`_no_incident_flux` — no power is carried toward the stack, so
    the physical transmittance is 0.  The denominator is substituted
    before the division rather than after, so the masked entries never
    evaluate ``0/0`` and never put a NaN on an autodiff tape.
    """
    z0_real = nd.real(kz0 / denom0)
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
