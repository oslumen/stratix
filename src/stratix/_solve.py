"""Core solve() entry point for stratified-media computations."""

from __future__ import annotations

from typing import Any

import numdiff as nd
from phokaia import Polarization
from phokaia import Stack

from ._absorption import _layer_absorption
from ._result import Result
from ._types import Method
from .methods._abeles import _abeles_solve
from .methods._admittance import _admittance_solve
from .methods._dtn import _dtn_solve
from .methods._smatrix import _smatrix_solve


def _validate_thicknesses(thicknesses: Any, stack: Stack) -> None:
    """Validate the thicknesses override parameter.

    Raises ``ValueError`` if ``thicknesses`` is not ``None`` and does
    not match ``len(stack.layers)``.
    """
    if thicknesses is None:
        return
    try:
        n_thick = len(thicknesses)
    except TypeError as exc:
        raise TypeError(
            f"thicknesses must be a sequence, got {type(thicknesses).__name__}"
        ) from exc
    n_layers = len(stack.layers)
    if n_thick != n_layers:
        raise ValueError(
            f"thicknesses length ({n_thick}) must match "
            f"number of layers ({n_layers})"
        )


def _scalar_solve(
    stack: Stack,
    wavelength: float | nd.ndarray,
    kx: float | nd.ndarray,
    polarization: Polarization,
    resolved: Method,
    absorption: bool,
    thicknesses: Any = None,
) -> tuple[nd.ndarray, nd.ndarray, dict]:
    """Call the appropriate solver for scalar or array inputs."""
    if resolved == Method.SMATRIX:
        return _smatrix_solve(stack, wavelength, kx, polarization, thicknesses=thicknesses)
    elif resolved == Method.ABELES:
        return _abeles_solve(stack, wavelength, kx, polarization, thicknesses=thicknesses)
    elif resolved == Method.ADMITTANCE:
        return _admittance_solve(stack, wavelength, kx, polarization, thicknesses=thicknesses)
    elif resolved == Method.DTN:
        return _dtn_solve(stack, wavelength, kx, polarization, thicknesses=thicknesses)
    else:
        raise NotImplementedError(f"Method {resolved.value!r} not yet implemented")


def _absorption_terms(
    R: nd.ndarray,
    T: nd.ndarray,
    intermediates: dict,
    resolved: Method,
) -> list:
    """Per-layer absorbed power fractions for the requested method.

    The S-matrix path reconstructs the medium amplitudes and integrates the
    Poynting flux across each layer.  The other methods keep no amplitude
    information, so they report a single lumped term ``1 - R - T``.
    """
    if resolved == Method.SMATRIX:
        return _layer_absorption(intermediates)
    return [1.0 - R - T]


def _absorption_fields(
    R: nd.ndarray,
    T: nd.ndarray,
    intermediates: dict,
    resolved: Method,
) -> tuple[nd.ndarray, nd.ndarray]:
    """Assemble the ``layer_absorption`` and ``energy_balance`` arrays.

    ``layer_absorption`` gets the layer index as its leading axis and the
    sweep shape of ``R`` behind it; ``energy_balance`` carries the sweep
    shape alone.
    """
    terms = _absorption_terms(R, T, intermediates, resolved)

    layer_abs = nd.stack(terms) if terms else nd.zeros((0, *nd.shape(R)))

    energy_bal = R + T
    for term in terms:
        energy_bal = energy_bal + term

    return layer_abs, energy_bal


def solve(
    stack: Stack,
    wavelength: float,
    kx: float,
    polarization: Polarization,
    method: Method = Method.AUTO,
    absorption: bool = False,
    thicknesses: Any = None,
) -> Result:
    """Compute reflectance and transmittance for a planar multilayer stack.

    Parameters
    ----------
    stack : Planar multilayer stack (superstrate + substrate ± layers).
    wavelength : Vacuum wavelength in meters.  Scalar or 1-D array (Nλ,).
    kx : In-plane wavevector component in rad/m.  Scalar or 1-D array (Nk,).
    polarization : ``TE``, ``TM``, or ``BOTH``.
    method : Solver method.  ``AUTO`` resolves to ``SMATRIX``.
    absorption : If ``True``, also return per-layer absorption and the
        energy balance ``R + T + Σ(layer absorption)``.  With the S-matrix
        method each layer's share comes from the drop in z-directed
        Poynting flux across it; the other methods report a single lumped
        ``1 - R - T`` term.
    thicknesses : Optional 1-D array overriding the stack's layer thicknesses.
        When given, ``len(thicknesses)`` must equal ``len(stack.layers)``.
        Enables autodiff w.r.t. thickness via ``nd.grad``.

    Returns
    -------
    Result with ``R``, ``T``, and metadata fields.
    """
    resolved = Method.SMATRIX if method == Method.AUTO else method

    _validate_thicknesses(thicknesses, stack)

    if polarization == Polarization.BOTH:
        res_te = solve(stack, wavelength, kx, Polarization.TE, method, absorption, thicknesses)
        res_tm = solve(stack, wavelength, kx, Polarization.TM, method, absorption, thicknesses)
        layer_abs = None
        energy_bal = None
        if absorption:
            layer_abs = nd.stack(
                [res_te.layer_absorption, res_tm.layer_absorption]
            )
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
            intermediates=res_te.intermediates,
        )
        return result

    wl_arr = wavelength if hasattr(wavelength, 'ndim') else nd.asarray(wavelength)
    kx_arr = kx if hasattr(kx, 'ndim') else nd.asarray(kx)

    if wl_arr.ndim == 0 and kx_arr.ndim == 0:
        R, T, intermediates = _scalar_solve(
            stack, wl_arr, kx_arr, polarization, resolved, absorption, thicknesses
        )
        layer_abs = None
        energy_bal = None
        if absorption:
            layer_abs, energy_bal = _absorption_fields(
                R, T, intermediates, resolved
            )
        result = Result(
            R=nd.array([R]),
            T=nd.array([T]),
            wavelengths=nd.array([wl_arr]),
            kx=nd.array([kx_arr]),
            polarization=polarization,
            method_used=resolved,
            layer_absorption=layer_abs,
            energy_balance=energy_bal,
            intermediates=intermediates,
        )
        return result

    wl_is_arr = wl_arr.ndim > 0
    kx_is_arr = kx_arr.ndim > 0

    if wl_is_arr and kx_is_arr:
        wl_bcast = wl_arr.reshape(-1, 1)
        kx_bcast = kx_arr.reshape(1, -1)
    else:
        wl_bcast = wl_arr
        kx_bcast = kx_arr

    R, T, intermediates = _scalar_solve(
        stack, wl_bcast, kx_bcast, polarization, resolved, absorption, thicknesses
    )

    wl_out = wl_arr if wl_is_arr else nd.array([wl_arr])
    kx_out = kx_arr if kx_is_arr else nd.array([kx_arr])

    layer_abs = None
    energy_bal = None
    if absorption:
        layer_abs, energy_bal = _absorption_fields(R, T, intermediates, resolved)

    result = Result(
        R=R,
        T=T,
        wavelengths=wl_out,
        kx=kx_out,
        polarization=polarization,
        method_used=resolved,
        layer_absorption=layer_abs,
        energy_balance=energy_bal,
        intermediates=intermediates,
    )
    return result
