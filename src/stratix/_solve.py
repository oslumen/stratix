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


def _sweep_axis(value: Any) -> nd.ndarray:
    """Promote a sweep input to a 1-D ndarray axis.

    A scalar wavelength or kx becomes a length-1 axis, so that every solve
    carries both sweep axes and the output shape contract has no special
    cases.  Values that already expose ``ndim`` are passed through
    untouched, which keeps traced arrays traced.
    """
    arr = value if hasattr(value, "ndim") else nd.asarray(value)
    # Reshape through the ndarray's own method rather than ``nd.reshape``:
    # the dispatched call rejects an ndarray produced by a different backend
    # than the active one, which callers do pass (the benchmarks build their
    # sweep grids once and run them through every backend).
    return arr if arr.ndim > 0 else arr.reshape(1)


def _dispatch_solve(
    stack: Stack,
    wavelength: float | nd.ndarray,
    kx: float | nd.ndarray,
    polarization: Polarization,
    resolved: Method,
    thicknesses: Any = None,
) -> tuple[nd.ndarray, nd.ndarray, dict]:
    """Call the solver selected by ``resolved``."""
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


def _absorption_fields(
    R: nd.ndarray,
    T: nd.ndarray,
    intermediates: dict,
    resolved: Method,
) -> tuple[nd.ndarray, nd.ndarray]:
    """Assemble the ``layer_absorption`` and ``energy_balance`` arrays.

    The S-matrix path reconstructs the medium amplitudes and takes the drop
    in Poynting flux across each layer.  The other methods keep no
    amplitude information, so they report a single lumped ``1 - R - T``.

    ``layer_absorption`` gets the layer index as its leading axis and the
    ``(Nλ, Nk)`` sweep shape of ``R`` behind it; ``energy_balance`` carries
    the sweep shape alone.
    """
    if resolved == Method.SMATRIX:
        terms = _layer_absorption(intermediates)
    else:
        terms = [1.0 - R - T]

    layer_abs = nd.stack(terms) if terms else nd.zeros((0, *R.shape))

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

    Shapes
    ------
    ``R`` and ``T`` are always ``(Nλ, Nk)``; a scalar ``wavelength`` or
    ``kx`` counts as a length-1 axis, so there are no special cases to
    remember.  ``Polarization.BOTH`` prepends a TE/TM axis of size 2.
    ``energy_balance`` follows the same rule and ``layer_absorption``
    inserts the layer axis directly in front of the sweep axes.

    Nothing between the inputs and the returned fields casts to ``float``
    or wraps values in a Python list, so ``nd.grad`` of a function calling
    ``solve()`` works on the autodiff backends.
    """
    resolved = Method.SMATRIX if method == Method.AUTO else method

    _validate_thicknesses(thicknesses, stack)

    if polarization == Polarization.BOTH:
        res_te = solve(stack, wavelength, kx, Polarization.TE, method, absorption, thicknesses)
        res_tm = solve(stack, wavelength, kx, Polarization.TM, method, absorption, thicknesses)
        layer_abs = None
        energy_bal = None
        if absorption:
            layer_abs = nd.stack([res_te.layer_absorption, res_tm.layer_absorption])
            energy_bal = nd.stack([res_te.energy_balance, res_tm.energy_balance])
        # Keep both polarizations' intermediates so field profiles from a
        # BOTH Result can carry the TE/TM axis like every other field.
        intermediates = {"te": res_te.intermediates, "tm": res_tm.intermediates}
        return Result(
            R=nd.stack([res_te.R, res_tm.R]),
            T=nd.stack([res_te.T, res_tm.T]),
            wavelengths=res_te.wavelengths,
            kx=res_te.kx,
            polarization=Polarization.BOTH,
            method_used=res_te.method_used,
            layer_absorption=layer_abs,
            energy_balance=energy_bal,
            intermediates=intermediates,
        )

    wl_axis = _sweep_axis(wavelength)
    kx_axis = _sweep_axis(kx)

    R, T, intermediates = _dispatch_solve(
        stack,
        wl_axis.reshape(-1, 1),
        kx_axis.reshape(1, -1),
        polarization,
        resolved,
        thicknesses,
    )

    layer_abs = None
    energy_bal = None
    if absorption:
        layer_abs, energy_bal = _absorption_fields(R, T, intermediates, resolved)

    return Result(
        R=R,
        T=T,
        wavelengths=wl_axis,
        kx=kx_axis,
        polarization=polarization,
        method_used=resolved,
        layer_absorption=layer_abs,
        energy_balance=energy_bal,
        intermediates=intermediates,
    )
