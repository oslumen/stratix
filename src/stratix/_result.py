"""Result model for solve() output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ._types import Method
from ._types import Polarization


class Result(BaseModel):
    """Reflectance and transmittance from a stratified-medium solve.

    Attributes
    ----------
    R : Power reflectance per (wavelength, kx) pair.
    T : Power transmittance per (wavelength, kx) pair.
    wavelengths : Vacuum wavelengths in meters.
    kx : In-plane wavevector components in rad/m.
    polarization : Polarization used for the computation.
    method_used : Solver method that was applied.
    layer_absorption : Per-layer absorbed power fraction (absorption=True only).
    energy_balance : R + T + sum(layer_absorption); ≈1 when absorption=True.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    R: Any
    T: Any
    wavelengths: Any
    kx: Any
    polarization: Polarization
    method_used: Method
    layer_absorption: Any | None = Field(default=None)
    energy_balance: Any | None = Field(default=None)

    def _set_intermediates(self, value: Any) -> None:
        object.__setattr__(self, "_intermediates", value)
