"""Per-layer absorption from the z-directed Poynting flux.

The power absorbed inside a layer is the flux entering its top face minus
the flux leaving its bottom face, normalised to the incident flux.  Both
faces are evaluated from the reconstructed medium amplitudes, so a layer
is credited only with the power it actually dissipates — unlike a
truncated-stack ``R + T`` difference, which also picks up the change in
reflectance caused by whatever sits behind it.
"""

from __future__ import annotations

import numdiff as nd

from ._amplitudes import _medium_amplitudes
from .methods._util import _no_incident_flux


def _flux(admittance: nd.ndarray, A: nd.ndarray, B: nd.ndarray) -> nd.ndarray:
    """Time-averaged z-directed power flux of a forward/backward wave pair.

    With ``u = A + B`` the tangential field that is continuous across an
    interface and ``v = admittance * (A - B)`` its partner, the flux is
    ``Re(u * conj(v))`` up to a constant that is identical for every medium
    and every polarization, and therefore cancels once the flux is divided
    by the incident flux.

    Parameters
    ----------
    admittance : ``kz / denom`` for the medium (``denom`` is mu for TE,
        epsilon for TM), matching the convention used for R and T.
    A, B : Forward and backward amplitudes at the evaluation plane.

    Returns
    -------
    ndarray of the same broadcast shape as the inputs.
    """
    return nd.real(nd.conj(admittance) * (A + B) * nd.conj(A - B))


def _layer_absorption(intermediates: dict) -> list:
    """Absorbed power fraction of every layer in the stack.

    Parameters
    ----------
    intermediates : Dict returned by :func:`_smatrix_solve`.

    Returns
    -------
    List of length ``n_layers``.  Each entry has the sweep shape of the
    solve (0-D for scalar wavelength/kx inputs).  Evanescent and grazing
    incidence carry no power into the stack, so every layer reports 0
    there — the same tolerance-based guard R and T use.
    """
    k0 = intermediates["k0"]
    kzs: list = intermediates["kzs"]
    denom_vals: list = intermediates["denom_vals"]

    n_media = len(kzs)
    n_layers = n_media - 2
    if n_layers <= 0:
        return []

    A_left, B_left, A_right, B_right = _medium_amplitudes(intermediates)

    incident = nd.real(kzs[0] / denom_vals[0])
    no_flux = _no_incident_flux(kzs[0], denom_vals[0], k0)
    safe_incident = nd.where(no_flux, nd.ones_like(incident), incident)

    absorption = []
    for m in range(1, n_media - 1):
        admittance = kzs[m] / denom_vals[m]
        entering = _flux(admittance, A_left[m], B_left[m])
        leaving = _flux(admittance, A_right[m], B_right[m])
        absorbed = (entering - leaving) / safe_incident
        absorption.append(nd.where(no_flux, nd.zeros_like(absorbed), absorbed))

    return absorption
