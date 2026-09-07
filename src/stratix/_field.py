"""Field profile reconstruction from S-matrix results."""

from __future__ import annotations

import numdiff as nd
from phokaia import Polarization

from ._amplitudes import _medium_amplitudes
from ._amplitudes import _medium_boundaries
from ._amplitudes import _medium_index
from ._amplitudes import _medium_offsets


def compute_field_profile(result, z_positions) -> dict:
    """Compute E and H field profiles through the stack at given z positions.

    Parameters
    ----------
    result : Result from :func:`stratix.solve`.
    z_positions : 1-D array of z coordinates (meters).  z=0 at the
        superstrate/first-layer interface; positive z goes into the stack.

    Returns
    -------
    dict with keys ``E``, ``H``, ``z`` (all ndarrays).  ``E`` and ``H`` are
    shaped ``sweep_shape + (Nz,)``: ``(Nz,)`` for scalar wavelength/kx,
    ``(Nλ, Nz)`` or ``(Nk, Nz)`` for a single swept axis and
    ``(Nλ, Nk, Nz)`` when both are swept.

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

    kzs: list = intr["kzs"]
    denom_vals: list = intr["denom_vals"]
    thicknesses = intr["thicknesses"]
    polarization: Polarization = intr["polarization"]

    n_media = len(kzs)
    A, B, _, _ = _medium_amplitudes(intr)

    z = nd.asarray(z_positions)
    boundaries = _medium_boundaries(thicknesses, n_media)
    offsets = _medium_offsets(thicknesses, n_media)
    m_idx = _medium_index(z, boundaries)

    E_total = None
    H_total = None

    for m in range(n_media):
        in_medium = m_idx == m
        # Clamp the local coordinate outside the medium so that a lossy
        # layer's growing exponential never overflows on the masked-out
        # positions.
        z_rel = nd.where(in_medium, z - offsets[m], nd.zeros_like(z))

        kz_m = nd.expand_dims(nd.asarray(kzs[m]), -1)
        denom_m = nd.expand_dims(nd.asarray(denom_vals[m]), -1)
        A_m = nd.expand_dims(nd.asarray(A[m]), -1)
        B_m = nd.expand_dims(nd.asarray(B[m]), -1)

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

    return {"E": E_total, "H": H_total, "z": z}
