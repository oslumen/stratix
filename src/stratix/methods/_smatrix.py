"""S-matrix method for stratified-medium transfer-matrix computations."""

from __future__ import annotations

import numdiff as nd
from phokaia import Stack

from .._types import Polarization


def _kz_single(epsilon: nd.ndarray, mu: nd.ndarray, k0: float, kx: float) -> nd.ndarray:
    """Compute the out-of-plane wavevector component kz.

    Chooses the physical branch: Im(kz) >= 0, or Re(kz) >= 0 when Im(kz) = 0.
    """
    kz_sq = epsilon * mu * k0**2 - kx**2
    kz_sq = kz_sq + 0j
    kz = nd.sqrt(kz_sq)
    neg = (nd.imag(kz) < 0) | ((nd.imag(kz) == 0) & (nd.real(kz) < 0))
    return nd.where(neg, -kz, kz)


def _redheffer_star(S_A: nd.ndarray, S_B: nd.ndarray) -> nd.ndarray:
    """Combine two 2x2 S-matrices via the Redheffer star product.

    Parameters
    ----------
    S_A : 2x2 ndarray (left-hand S-matrix).
    S_B : 2x2 ndarray (right-hand S-matrix).

    Returns
    -------
    2x2 ndarray: S_A (x) S_B.
    """
    A11, A12 = S_A[0, 0], S_A[0, 1]
    A21, A22 = S_A[1, 0], S_A[1, 1]
    B11, B12 = S_B[0, 0], S_B[0, 1]
    B21, B22 = S_B[1, 0], S_B[1, 1]

    denom = 1.0 - A22 * B11
    S11 = A11 + A12 * B11 * A21 / denom
    S12 = A12 * B12 / denom
    S21 = B21 * A21 / denom
    S22 = B22 + B21 * A22 * B12 / denom

    return nd.stack([nd.stack([S11, S12]), nd.stack([S21, S22])])


def _interface_smatrix(Z_left: nd.ndarray, Z_right: nd.ndarray) -> nd.ndarray:
    """Build the 2x2 interface S-matrix from wave impedances.

    Parameters
    ----------
    Z_left : Wave impedance of the incident medium.
    Z_right : Wave impedance of the transmitted medium.

    Returns
    -------
    2x2 ndarray [[r, t_rev], [t_fwd, -r]].
    """
    r = (Z_left - Z_right) / (Z_left + Z_right)
    t_fwd = 2 * Z_left / (Z_left + Z_right)
    t_rev = 2 * Z_right / (Z_left + Z_right)
    return nd.stack([nd.stack([r, t_rev]), nd.stack([t_fwd, -r])])


def _propagation_smatrix(kz: nd.ndarray, thickness: float) -> nd.ndarray:
    """Build the 2x2 propagation S-matrix for a homogeneous layer.

    Parameters
    ----------
    kz : Out-of-plane wavevector in the layer.
    thickness : Layer thickness in meters.

    Returns
    -------
    2x2 ndarray [[0, exp(i*phi)], [exp(i*phi), 0]].
    """
    phi = kz * thickness
    p = nd.exp(1j * phi)
    return nd.stack([nd.stack([0 * p, p]), nd.stack([p, 0 * p])])


def _smatrix_solve(
    stack: Stack,
    wavelength: float,
    kx: float,
    polarization: Polarization,
    thicknesses: nd.ndarray | None = None,
) -> tuple[nd.ndarray, nd.ndarray, dict]:
    """Compute R and T for a multilayer stack via S-matrix assembly.

    Assembles interface and propagation S-matrices and combines them
    left-to-right (superstrate → substrate) via the Redheffer star product.

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
    intermediates : Dict of S-matrix data needed for field reconstruction.
    """
    c = 299792458.0
    omega = 2 * nd.pi * c / wavelength
    k0 = 2 * nd.pi / wavelength

    media = (
        [stack.superstrate]
        + [layer.material for layer in stack.layers]
        + [stack.substrate]
    )
    epsilons = [m.epsilon(omega=omega) for m in media]
    mus = [m.mu(omega=omega) for m in media]
    kzs = [_kz_single(eps, mu, k0, kx) for eps, mu in zip(epsilons, mus, strict=True)]

    if polarization == Polarization.TE:
        denom_vals = mus
    elif polarization == Polarization.TM:
        denom_vals = epsilons
    else:
        raise NotImplementedError(f"Polarization {polarization!r} not supported")

    Zs = [kz / denom for kz, denom in zip(kzs, denom_vals, strict=True)]

    n_interfaces = len(media) - 1

    interface_smatrices = [_interface_smatrix(Zs[0], Zs[1])]
    propagation_smatrices: list[nd.ndarray] = []

    S_total = interface_smatrices[0]

    r_before = S_total[0, 0]
    t_before = S_total[1, 0]
    R_before = nd.abs(r_before) ** 2
    T_fac_before = (
        nd.real(kzs[1] / denom_vals[1]) / nd.real(kzs[0] / denom_vals[0])
    )
    T_before = T_fac_before * nd.abs(t_before) ** 2

    layer_abs_list: list = []

    if thicknesses is not None:
        _thicknesses = thicknesses
        _thicknesses_intr = [float(layer.thickness) for layer in stack.layers]
    else:
        _thicknesses = nd.array([layer.thickness for layer in stack.layers])
        _thicknesses_intr = [float(t) for t in _thicknesses]

    for i in range(1, n_interfaces):
        d = _thicknesses[i - 1]
        P = _propagation_smatrix(kzs[i], d)
        propagation_smatrices.append(P)
        S_total = _redheffer_star(S_total, P)
        S_int = _interface_smatrix(Zs[i], Zs[i + 1])
        interface_smatrices.append(S_int)
        S_total = _redheffer_star(S_total, S_int)

        r_after = S_total[0, 0]
        t_after = S_total[1, 0]
        R_after = nd.abs(r_after) ** 2
        T_fac_after = (
            nd.real(kzs[i + 1] / denom_vals[i + 1])
            / nd.real(kzs[0] / denom_vals[0])
        )
        T_after = T_fac_after * nd.abs(t_after) ** 2

        A_layer = (R_before + T_before) - (R_after + T_after)
        layer_abs_list.append(A_layer)

        R_before = R_after
        T_before = T_after

    r_total = S_total[0, 0]
    t_total = S_total[1, 0]

    R = nd.abs(r_total) ** 2
    T = (
        nd.real(kzs[-1] / denom_vals[-1])
        / nd.real(kzs[0] / denom_vals[0])
        * nd.abs(t_total) ** 2
    )

    energy_bal = R + T
    if layer_abs_list:
        for a in layer_abs_list:
            energy_bal = energy_bal + a

    intermediates = {
        "kzs": kzs,
        "denom_vals": denom_vals,
        "r_total": r_total,
        "t_total": t_total,
        "interface_smatrices": interface_smatrices,
        "propagation_smatrices": propagation_smatrices,
        "thicknesses": _thicknesses_intr,
        "polarization": polarization,
        "layer_absorption": layer_abs_list,
        "energy_balance": energy_bal,
    }

    return R, T, intermediates
