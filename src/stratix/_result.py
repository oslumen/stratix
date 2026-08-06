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
