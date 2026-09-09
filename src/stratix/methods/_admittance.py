"""Admittance recursion method for stratified-medium computations.

The optical admittance ``Y = H_tangential / E_tangential`` at an interface
is propagated through the stack from substrate to superstrate by the
Moebius map in :func:`~stratix.methods._util._admittance_step`.  Reflection
comes from the effective input admittance at the first interface, and the
transmitted amplitude from a second pass back down the stack, which reuses
the admittances the upward pass saw at each layer's bottom face and
multiplies the field by each layer's face-to-face ratio, written on
``p = exp(i phi)`` so an absorbing layer attenuates instead of overflowing
(issue #60).
"""

from __future__ import annotations

import numdiff as nd
from phokaia import Polarization
from phokaia import Stack

from ._medium_params import _medium_params
from ._util import _admittance_step
from ._util import _layer_phases
from ._util import _resolve_thicknesses
from ._util import _safe_R
from ._util import _safe_T
from ._util import _wave_admittances


def _admittance_solve(
    stack: Stack,
    wavelength: float,
    kx: float,
    polarization: Polarization,
    thicknesses: nd.ndarray | None = None,
) -> tuple[nd.ndarray, nd.ndarray, dict]:
    """Compute R and T for a multilayer stack by admittance recursion.

    Parameters
    ----------
    stack : Multilayer stack (superstrate, layers, substrate).
    wavelength : Vacuum wavelength in meters.
    kx : In-plane wavevector component in rad/m.
    polarization : ``TE`` or ``TM``.
    thicknesses : Optional 1-D ndarray overriding the stack's layer
        thicknesses.  Must have length ``len(stack.layers)``.  When
        provided, ``stack.layers[i].thickness`` is ignored in favour of
        ``thicknesses[i]``, enabling autodiff w.r.t. thickness.

    Returns
    -------
    R : Power reflectance.
    T : Power transmittance.
    intermediates : Dict carrying the resolved thicknesses.
    """
    _omega, _k0, kzs, _, _, denom_vals, no_flux = _medium_params(
        stack, wavelength, kx, polarization
    )

    resolved = _resolve_thicknesses(stack, thicknesses)
    admittances = _wave_admittances(kzs, denom_vals)
    phis = _layer_phases(kzs, resolved)

    Y_super = admittances[0]

    # Upward pass: start from the substrate, which holds a single outgoing
    # wave and so presents its own wave admittance, and carry it up to the
    # first interface.  ``-phi`` runs the map against the propagation
    # direction.  The admittance seen at each layer's *bottom* face is
    # kept: the downward pass needs it to know how much backward wave the
    # layer holds.
    Y_below = [nd.array(0j)] * len(phis)
    Y_in = admittances[-1]
    for i in reversed(range(len(phis))):
        Y_below[i] = Y_in
        Y_in = _admittance_step(Y_in, admittances[i + 1], -phis[i])

    r_total = (Y_super - Y_in) / (Y_super + Y_in)

    # Downward pass: the input admittance fixes the field at the first
    # interface, and each layer multiplies it by the ratio of its two
    # face fields.  What is left at the substrate is the transmitted
    # amplitude.  The ratio is carried on ``p = exp(i phi)`` rather than
    # on ``cos``/``sin``: an absorbing layer has ``Im(phi) > 0``, where
    # both trig functions grow like ``exp(Im phi)/2`` and the growing
    # exponential survives the sum instead of cancelling, so T diverged
    # with thickness (issue #60).  The physical branch of ``kz`` puts
    # ``Im(phi) >= 0``, so ``|p| <= 1`` and nothing here can grow — the
    # same substitution the S-matrix layer loop and the DtN assembly
    # make.  ``rho`` is the reflection the layer sees looking down from
    # its bottom face, read off the retained admittance.
    field = nd.array(1.0) + r_total
    for i, phi in enumerate(phis):
        Y_layer = admittances[i + 1]
        p = nd.exp(1j * phi)
        rho = (Y_layer - Y_below[i]) / (Y_layer + Y_below[i])
        field = field * p * (1.0 + rho) / (1.0 + rho * p * p)

    R = _safe_R(r_total, no_flux)
    T = _safe_T(field, kzs[0], denom_vals[0], kzs[-1], denom_vals[-1], no_flux)

    return R, T, {"thicknesses": resolved}
