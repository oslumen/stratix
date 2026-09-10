"""Field profile reconstruction from S-matrix results."""

from __future__ import annotations

import numdiff as nd
from phokaia import Polarization

from ._amplitudes import _medium_amplitudes
from ._amplitudes import _medium_boundaries
from ._amplitudes import _medium_index
from ._amplitudes import _medium_offsets
from ._result import Result


def _profile_from_intermediates(intr: dict, z: nd.ndarray) -> tuple:
    """Reconstruct (E, H) on ``z`` from one polarization's intermediates."""
    if not intr or "kzs" not in intr:
        raise ValueError(
            "No field intermediates in Result; smatrix solver was not used."
        )
    kzs: list = intr["kzs"]
    denom_vals: list = intr["denom_vals"]
    thicknesses = intr["thicknesses"]
    polarization: Polarization = intr["polarization"]

    n_media = len(kzs)
    A, B, _, _ = _medium_amplitudes(intr)

    boundaries = _medium_boundaries(thicknesses, n_media)
    offsets = _medium_offsets(boundaries)
    m_idx = _medium_index(z, boundaries)

    E_total = None
    H_total = None

    for m in range(n_media):
        in_medium = m_idx == m
        # Clamp the local coordinate outside the medium so that a lossy
        # layer's growing exponential never overflows on the masked-out
        # positions.
        z_rel = nd.where(in_medium, z - offsets[m], nd.zeros_like(z))

        # Open a trailing z axis so the (Nλ, Nk) sweep grid broadcasts
        # against the (Nz,) sample positions.
        kz_m = nd.asarray(kzs[m])[..., None]
        denom_m = nd.asarray(denom_vals[m])[..., None]
        A_m = nd.asarray(A[m])[..., None]
        B_m = nd.asarray(B[m])[..., None]

        forward = A_m * nd.exp(1j * kz_m * z_rel)
        backward = B_m * nd.exp(-1j * kz_m * z_rel)

        if polarization == Polarization.TE:
            E_m = forward + backward
            H_m = (kz_m / denom_m) * (forward - backward)
        else:
            H_m = forward + backward
            E_m = -(kz_m / denom_m) * (forward - backward)

        E_m = nd.where(in_medium, E_m, nd.zeros_like(E_m))
        H_m = nd.where(in_medium, H_m, nd.zeros_like(H_m))

        E_total = E_m if E_total is None else E_total + E_m
        H_total = H_m if H_total is None else H_total + H_m

    return E_total, H_total


def compute_field_profile(result: Result, z_positions: nd.ndarray) -> dict:
    """Compute E and H field profiles through the stack at given z positions.

    Parameters
    ----------
    result : Result from :func:`stratix.solve`.
    z_positions : 1-D array of z coordinates (meters).  z=0 at the
        superstrate/first-layer interface; positive z goes into the stack.

    Returns
    -------
    dict with keys ``E``, ``H``, ``z`` (all ndarrays).  ``E`` and ``H`` are
    always shaped ``(Nλ, Nk, Nz)``, following the sweep shape contract of
    :func:`stratix.solve`: a scalar wavelength or kx counts as a length-1
    axis, so a scalar solve gives ``(1, 1, Nz)``.

    A ``Polarization.BOTH`` Result carries both polarizations'
    intermediates, so its profile prepends a TE/TM axis of size 2 —
    ``(2, Nλ, Nk, Nz)`` — matching every other BOTH field.

    TE polarization (default):
        ``E`` = Ey (tangential electric field).
        ``H`` = (kz/denom) * (A·exp(i·kz·z) - B·exp(-i·kz·z)).

    TM polarization:
        ``E`` = Ex (tangential electric field).
        ``H`` = Hy (tangential magnetic field).
    """
    intr = result.intermediates
    if not intr:
        raise ValueError(
            "No field intermediates in Result; smatrix solver was not used."
        )

    z = nd.asarray(z_positions)

    if result.polarization == Polarization.BOTH:
        E_te, H_te = _profile_from_intermediates(intr["te"], z)
        E_tm, H_tm = _profile_from_intermediates(intr["tm"], z)
        return {
            "E": nd.stack([E_te, E_tm]),
            "H": nd.stack([H_te, H_tm]),
            "z": z,
        }

    E, H = _profile_from_intermediates(intr, z)
    return {"E": E, "H": H, "z": z}
