"""Result NamedTuple for solve() output."""

from __future__ import annotations

from typing import Any
from typing import NamedTuple

from phokaia import Polarization

from ._types import Method


class Result(NamedTuple):
    """Reflectance and transmittance from a stratified-medium solve.

    Parameters
    ----------
    R : Power reflectance on the (wavelength, kx) grid.
    T : Power transmittance on the (wavelength, kx) grid.
    wavelengths : Vacuum wavelengths in meters.
    kx : In-plane wavevector components in rad/m.
    polarization : Polarization used for the computation.
    method_used : Solver method that was applied.
    layer_absorption : Per-layer absorbed power fraction (absorption=True only).
    energy_balance : R + T + sum(layer_absorption); ≈1 when absorption=True.
    intermediates : S-matrix solver intermediates for field profile.  For
        ``Polarization.BOTH`` this is a dict with ``"te"`` and ``"tm"``
        keys holding each polarization's intermediates.

    Shapes
    ------
    ``R`` and ``T`` are always ``(Nλ, Nk)``.  A scalar wavelength or kx
    counts as a length-1 axis, so a scalar solve returns ``(1, 1)`` — one
    rule, no special cases.  ``energy_balance`` carries the same
    ``(Nλ, Nk)`` shape and ``layer_absorption`` prepends the layer axis,
    giving ``(n_layers, Nλ, Nk)``.

    ``wavelengths`` and ``kx`` stay 1-D, ``(Nλ,)`` and ``(Nk,)``: they name
    the sweep coordinates rather than following the grid.

    ``Polarization.BOTH`` prepends a TE/TM axis of size 2 to ``R``, ``T``,
    ``energy_balance`` and ``layer_absorption`` alike.

    Non-S-matrix methods keep no per-medium amplitudes, so their
    ``layer_absorption`` holds a single lumped ``1 - R - T`` term instead
    of one entry per layer.
    """

    R: Any
    T: Any
    wavelengths: Any
    kx: Any
    polarization: Polarization
    method_used: Method
    layer_absorption: Any | None = None
    energy_balance: Any | None = None
    intermediates: Any = None
