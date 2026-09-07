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
    R : Power reflectance per (wavelength, kx) pair.
    T : Power transmittance per (wavelength, kx) pair.
    wavelengths : Vacuum wavelengths in meters.
    kx : In-plane wavevector components in rad/m.
    polarization : Polarization used for the computation.
    method_used : Solver method that was applied.
    layer_absorption : Per-layer absorbed power fraction (absorption=True only).
    energy_balance : R + T + sum(layer_absorption); ≈1 when absorption=True.
    intermediates : S-matrix solver intermediates for field profile.

    Shapes
    ------
    Write ``sweep`` for the shape ``R`` and ``T`` take: ``(1,)`` for scalar
    wavelength and kx, ``(Nλ,)`` or ``(Nk,)`` when one of the two is an
    array, and ``(Nλ, Nk)`` when both are.  ``energy_balance`` follows
    ``sweep`` — except for scalar inputs, where it is 0-D — and
    ``layer_absorption`` prepends the layer axis, giving
    ``(n_layers,) + sweep`` (``(n_layers,)`` for scalar inputs).

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
