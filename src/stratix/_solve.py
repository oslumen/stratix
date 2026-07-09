"""Core solve() entry point for stratified-media computations."""

from __future__ import annotations

import numdiff as nd
from phokaia import Stack

from ._result import Result
from ._types import Method
from ._types import Polarization
from .methods._abeles import _abeles_solve
from .methods._admittance import _admittance_solve
from .methods._dtn import _dtn_solve
from .methods._smatrix import _smatrix_solve


def _scalar_solve(
    stack: Stack,
    wavelength: float,
    kx: float,
    polarization: Polarization,
    resolved: Method,
    absorption: bool,
) -> tuple[nd.ndarray, nd.ndarray, dict]:
    """Call the appropriate solver for scalar inputs."""
    if resolved == Method.SMATRIX:
        return _smatrix_solve(stack, wavelength, kx, polarization)
    elif resolved == Method.ABELES:
        return _abeles_solve(stack, wavelength, kx, polarization)
    elif resolved == Method.ADMITTANCE:
        return _admittance_solve(stack, wavelength, kx, polarization)
    elif resolved == Method.DTN:
        return _dtn_solve(stack, wavelength, kx, polarization)
    else:
        raise NotImplementedError(f"Method {resolved.value!r} not yet implemented")


def solve(
    stack: Stack,
    wavelength: float,
    kx: float,
    polarization: Polarization,
    method: Method = Method.AUTO,
    absorption: bool = False,
) -> Result:
    """Compute reflectance and transmittance for a planar multilayer stack.

    Parameters
    ----------
    stack : Planar multilayer stack (superstrate + substrate ± layers).
    wavelength : Vacuum wavelength in meters.  Scalar or 1-D array (Nλ,).
    kx : In-plane wavevector component in rad/m.  Scalar or 1-D array (Nk,).
    polarization : ``TE``, ``TM``, or ``BOTH``.
    method : Solver method.  ``AUTO`` resolves to ``SMATRIX``.
    absorption : If ``True``, compute per-layer absorption (not yet implemented).

    Returns
    -------
    Result with ``R``, ``T``, and metadata fields.
    """
    resolved = Method.SMATRIX if method == Method.AUTO else method

    if polarization == Polarization.BOTH:
        res_te = solve(stack, wavelength, kx, Polarization.TE, method, absorption)
        res_tm = solve(stack, wavelength, kx, Polarization.TM, method, absorption)
        layer_abs = None
        energy_bal = None
        if absorption:
            layer_abs = res_te.layer_absorption
            energy_bal = nd.stack([res_te.energy_balance, res_tm.energy_balance])
        result = Result(
            R=nd.stack([res_te.R, res_tm.R]),
            T=nd.stack([res_te.T, res_tm.T]),
            wavelengths=res_te.wavelengths,
            kx=res_te.kx,
            polarization=Polarization.BOTH,
            method_used=res_te.method_used,
            layer_absorption=layer_abs,
            energy_balance=energy_bal,
        )
        result._set_intermediates(res_te._intermediates)
        return result

    wl_arr = nd.asarray(wavelength)
    kx_arr = nd.asarray(kx)

    if wl_arr.ndim == 0 and kx_arr.ndim == 0:
        wl_val = float(wl_arr)
        kx_val = float(kx_arr)
        R, T, intermediates = _scalar_solve(
            stack, wl_val, kx_val, polarization, resolved, absorption
        )
        layer_abs = None
        energy_bal = None
        if absorption:
            la = intermediates.get("layer_absorption")
            eb_val = intermediates.get("energy_balance")
            if la is not None and eb_val is not None:
                layer_abs = nd.array(la)
                energy_bal = nd.array(eb_val)
            else:
                total_abs = 1.0 - R - T
                layer_abs = nd.array([total_abs])
                energy_bal = nd.array(1.0)
        result = Result(
            R=nd.array([float(R)]),
            T=nd.array([float(T)]),
            wavelengths=nd.array([wl_val]),
            kx=nd.array([kx_val]),
            polarization=polarization,
            method_used=resolved,
            layer_absorption=layer_abs,
            energy_balance=energy_bal,
        )
        result._set_intermediates(intermediates)
        return result

    wl_is_arr = wl_arr.ndim > 0
    kx_is_arr = kx_arr.ndim > 0

    if wl_is_arr:
        wl_vals = [float(wl_arr[i]) for i in range(wl_arr.shape[0])]
    else:
        wl_vals = [float(wl_arr)]

    if kx_is_arr:
        kx_vals = [float(kx_arr[j]) for j in range(kx_arr.shape[0])]
    else:
        kx_vals = [float(kx_arr)]

    N_wl = len(wl_vals)
    N_kx = len(kx_vals)

    if N_wl > 1 and N_kx > 1:
        R_list: list[list[float]] = []
        T_list: list[list[float]] = []
        for i in range(N_wl):
            R_row: list[float] = []
            T_row: list[float] = []
            for j in range(N_kx):
                R, T, _int = _scalar_solve(
                    stack, wl_vals[i], kx_vals[j], polarization, resolved, absorption
                )
                R_row.append(float(R))
                T_row.append(float(T))
            R_list.append(R_row)
            T_list.append(T_row)
        result = Result(
            R=nd.array(R_list),
            T=nd.array(T_list),
            wavelengths=nd.array(wl_vals),
            kx=nd.array(kx_vals),
            polarization=polarization,
            method_used=resolved,
        )
        result._set_intermediates({})
        return result

    N = max(N_wl, N_kx)
    R_list: list[float] = []
    T_list: list[float] = []
    for i in range(N):
        wl = wl_vals[i] if N_wl > 1 else wl_vals[0]
        kxv = kx_vals[i] if N_kx > 1 else kx_vals[0]
        R, T, _int = _scalar_solve(
            stack, wl, kxv, polarization, resolved, absorption
        )
        R_list.append(float(R))
        T_list.append(float(T))

    result = Result(
        R=nd.array(R_list),
        T=nd.array(T_list),
        wavelengths=nd.array(wl_vals),
        kx=nd.array(kx_vals),
        polarization=polarization,
        method_used=resolved,
    )
    result._set_intermediates({})
    return result
