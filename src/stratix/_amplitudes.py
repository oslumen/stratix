"""Full-stack forward/backward wave amplitude reconstruction.

The S-matrix solver returns the overall reflection and transmission
coefficients plus the per-interface S-matrices.  Field profiles and
per-layer absorption both need the amplitude of the forward and backward
travelling wave *inside* every medium, so the reconstruction lives here
once and is shared by both consumers.

Amplitudes are normalised to a unit-amplitude incident wave in the
superstrate, which makes the associated power fluxes directly comparable
with the ``R``/``T`` returned by the solver.
"""

from __future__ import annotations

import numdiff as nd


def _medium_amplitudes(intermediates: dict) -> tuple[list, list, list, list]:
    """Reconstruct wave amplitudes in every medium of the stack.

    Runs the interface recursion from the substrate back to the
    superstrate.  Every operation is elementwise, so the recursion is
    vectorised over whatever sweep grid the solver was called with: the
    returned amplitudes carry the same shape as ``intermediates['kzs']``
    entries (0-D for scalar inputs, (Nλ,), (Nk,) or (Nλ, Nk) for sweeps).

    Parameters
    ----------
    intermediates : Dict returned by :func:`_smatrix_solve`.

    Returns
    -------
    A_left, B_left : Lists of length ``n_media``.  Forward and backward
        amplitudes referenced at the entrance (small-z) face of each
        medium.  For the semi-infinite superstrate the reference is the
        first interface; for the substrate it is the last interface.
    A_right, B_right : Same amplitudes referenced at the exit (large-z)
        face of each medium.  For the semi-infinite superstrate and
        substrate the two references coincide.
    """
    kzs: list = intermediates["kzs"]
    t_total = intermediates["t_total"]
    interface_smatrices: list = intermediates["interface_smatrices"]
    thicknesses = intermediates["thicknesses"]

    n_media = len(kzs)
    zero = t_total * 0

    A_left: list = [None] * n_media
    B_left: list = [None] * n_media
    A_right: list = [None] * n_media
    B_right: list = [None] * n_media

    A_left[-1] = t_total
    B_left[-1] = zero
    A_right[-1] = t_total
    B_right[-1] = zero

    for k in range(n_media - 2, -1, -1):
        S_int = interface_smatrices[k]
        r = S_int[0, 0]
        t_fwd = S_int[1, 0]
        t_rev = S_int[0, 1]

        a_exit = (A_left[k + 1] + r * B_left[k + 1]) / t_fwd
        b_exit = r * a_exit + t_rev * B_left[k + 1]

        A_right[k] = a_exit
        B_right[k] = b_exit

        if k == 0:
            A_left[k] = a_exit
            B_left[k] = b_exit
        else:
            d_k = thicknesses[k - 1]
            kz_k = kzs[k]
            A_left[k] = a_exit * nd.exp(-1j * kz_k * d_k)
            B_left[k] = b_exit * nd.exp(1j * kz_k * d_k)

    return A_left, B_left, A_right, B_right


def _medium_boundaries(thicknesses: nd.ndarray, n_media: int) -> nd.ndarray:
    """Return the z coordinates of the interface planes, ascending.

    There are ``n_media - 1`` interfaces: the first at ``z = 0`` and the
    last at the total stack thickness.

    Parameters
    ----------
    thicknesses : 1-D ndarray of layer thicknesses (length ``n_media - 2``).
    n_media : Total number of media (layers plus superstrate and substrate).

    Returns
    -------
    1-D ndarray of length ``n_media - 1``.
    """
    if n_media <= 2:
        return nd.zeros(1)
    return nd.concatenate([nd.zeros(1), nd.cumsum(nd.asarray(thicknesses))])


def _medium_offsets(boundaries: nd.ndarray) -> nd.ndarray:
    """Return the z coordinate of each medium's entrance face.

    Every medium starts at the interface before it, so the offsets are the
    boundaries shifted by one.  The superstrate is unbounded below and has
    no interface before it; giving it offset 0 makes ``z - offset`` the
    correct local coordinate there too, since the superstrate's own fields
    are already referenced to ``z = 0``.

    Parameters
    ----------
    boundaries : 1-D ndarray from :func:`_medium_boundaries`.

    Returns
    -------
    1-D ndarray of length ``n_media``.
    """
    return nd.concatenate([nd.zeros(1), boundaries])


def _medium_index(z: nd.ndarray, boundaries: nd.ndarray) -> nd.ndarray:
    """Locate which medium each z coordinate falls in, vectorised.

    Counting how many interface planes lie at or below ``z`` gives the
    medium index directly: 0 for the superstrate, 1..n_layers for the
    layers and ``n_media - 1`` for the substrate.

    Parameters
    ----------
    z : 1-D ndarray of z coordinates in meters.
    boundaries : 1-D ndarray from :func:`_medium_boundaries`.

    Returns
    -------
    ndarray of the same shape as ``z`` holding the medium index.
    """
    counts = nd.zeros(nd.shape(z))
    for i in range(len(boundaries)):
        counts = counts + nd.where(
            z >= boundaries[i], nd.ones_like(counts), nd.zeros_like(counts)
        )
    return counts
