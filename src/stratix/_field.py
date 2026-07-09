"""Field profile reconstruction from S-matrix results."""

from __future__ import annotations

import numdiff as nd

from ._types import Polarization


def compute_field_profile(result, z_positions) -> dict:
    """Compute E and H field profiles through the stack at given z positions.

    Parameters
    ----------
    result : Result from :func:`stratix.solve`.
    z_positions : 1-D array of z coordinates (meters).  z=0 at the
        superstrate/first-layer interface; positive z goes into the stack.

    Returns
    -------
    dict with keys ``E``, ``H``, ``z`` (all ndarrays).

    TE polarization (default):
        ``E`` = Ey (tangential electric field).
        ``H`` = (kz/denom) * (A·exp(i·kz·z) - B·exp(-i·kz·z)).

    TM polarization:
        ``E`` = Ex (tangential electric field).
        ``H`` = Hy (tangential magnetic field).
    """
    intr = result._intermediates
    if not intr:
        raise ValueError(
            "No field intermediates in Result; smatrix solver was not used."
        )

    kzs: list = intr["kzs"]
    denom_vals: list = intr["denom_vals"]
    t_total = intr["t_total"]
    interface_smatrices: list = intr["interface_smatrices"]
    thicknesses: list = intr["thicknesses"]
    polarization: Polarization = intr["polarization"]

    n_media = len(kzs)
    total_thickness = sum(thicknesses) if thicknesses else 0.0

    A = [nd.array(0j) for _ in range(n_media)]
    B = [nd.array(0j) for _ in range(n_media)]

    A[-1] = t_total
    B[-1] = nd.array(0j)

    for k in range(n_media - 2, -1, -1):
        S_int = interface_smatrices[k]
        r = S_int[0, 0]
        t_fwd = S_int[1, 0]
        t_rev = S_int[0, 1]

        a1 = (A[k + 1] + r * B[k + 1]) / t_fwd
        b1_out = r * a1 + t_rev * B[k + 1]

        if k == 0:
            A[k] = a1
            B[k] = b1_out
        else:
            d_k = thicknesses[k - 1]
            kz_k = kzs[k]
            A[k] = a1 * nd.exp(-1j * kz_k * d_k)
            B[k] = b1_out * nd.exp(1j * kz_k * d_k)

    z_list = [float(z) for z in z_positions]

    E_vals = []
    H_vals = []

    for z in z_list:
        if z < 0:
            m = 0
            z_rel = z
        elif z >= total_thickness:
            m = n_media - 1
            z_rel = z - total_thickness
        else:
            cum = 0.0
            m = 1
            for d in thicknesses:
                if z < cum + d:
                    z_rel = z - cum
                    break
                cum += d
                m += 1

        kz_m = kzs[m]
        denom_m = denom_vals[m]

        exp_fwd = nd.exp(1j * kz_m * z_rel)
        exp_bwd = nd.exp(-1j * kz_m * z_rel)

        forward = A[m] * exp_fwd
        backward = B[m] * exp_bwd

        if polarization == Polarization.TE:
            E_val = forward + backward
            H_val = (kz_m / denom_m) * (forward - backward)
        else:
            H_val = forward + backward
            E_val = -(kz_m / denom_m) * (forward - backward)

        E_vals.append(E_val)
        H_vals.append(H_val)

    return {
        "E": nd.array(E_vals),
        "H": nd.array(H_vals),
        "z": nd.array(z_list),
    }
