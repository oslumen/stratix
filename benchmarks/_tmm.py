"""tmm comparison utilities for benchmark scripts."""

from __future__ import annotations

import cmath
import sys
from typing import TYPE_CHECKING

import numdiff as nd
from phokaia import Polarization
from phokaia import Stack

from stratix import Method
from stratix import solve

if TYPE_CHECKING:
    import tmm as _tmm_mod
else:
    try:
        import tmm as _tmm_mod
    except ImportError:
        _tmm_mod = None  # type: ignore[assignment]


def _tmm_available() -> bool:
    """Return True if the ``tmm`` package is installed."""
    return _tmm_mod is not None


def _stack_to_tmm(
    stack: Stack,
) -> tuple[list[complex], list[float]]:
    """Extract n_list and d_list from a phokaia Stack for tmm consumption.

    tmm conventions: ``n_list = [n_super, n_layer1, ..., n_sub]``,
    ``d_list = [layer1_thickness, layer2_thickness, ...]`` (no inf entries).
    """
    n_list = [cmath.sqrt(complex(stack.superstrate.epsilon._raw))]
    d_list: list[float] = []
    for layer in stack.layers:
        n_list.append(cmath.sqrt(complex(layer.material.epsilon._raw)))
        d_list.append(float(layer.thickness))
    n_list.append(cmath.sqrt(complex(stack.substrate.epsilon._raw)))
    return n_list, d_list


def tmm_reference(
    stack: Stack,
    wavelength: float,
    kx: float = 0.0,
    polarization: Polarization = Polarization.TE,
) -> tuple[float, float]:
    """Compute (R, T) using tmm for comparison against stratix.

    Parameters
    ----------
    stack : Stack
        Planar multilayer stack.
    wavelength : float
        Vacuum wavelength in meters.
    kx : float
        In-plane wavevector in rad/m. Must be 0.0 (normal incidence) —
        tmm does not support a generic kx parameter; off-normal is
        specified via incidence angle.
    polarization : Polarization
        ``TE`` or ``TM``.
    """
    if not _tmm_available():
        raise RuntimeError("tmm is not installed; cannot compute reference.")
    if kx != 0.0:
        raise ValueError("tmm comparison supports normal incidence only (kx=0).")

    n_list, d_list = _stack_to_tmm(stack)
    pol_tmm = "s" if polarization == Polarization.TE else "p"
    infinity = float("inf")
    n_arr = nd.array(n_list)
    d_arr = nd.array([infinity] + d_list + [infinity])

    result = _tmm_mod.coh_tmm(pol_tmm, n_arr, d_arr, 0.0, wavelength)
    return float(result["R"]), float(result["T"])


def tmm_compare(
    stack: Stack,
    wavelength: float,
    kx: float = 0.0,
    polarization: Polarization = Polarization.TE,
    method: Method = Method.AUTO,
    tolerance: float = 1e-10,
) -> dict[str, float]:
    """Compare stratix result against tmm reference.

    Returns a dict with keys ``R_stratix``, ``T_stratix``,
    ``R_tmm``, ``T_tmm``, ``R_err``, ``T_err``.
    """
    if not _tmm_available():
        raise RuntimeError("tmm is not installed; cannot run comparison.")

    result = solve(
        stack,
        wavelength,
        kx=kx,
        polarization=polarization,
        method=method,
    )
    R_stratix = float(nd.array(result.R).flatten()[0])
    T_stratix = float(nd.array(result.T).flatten()[0])

    R_tmm, T_tmm = tmm_reference(stack, wavelength, kx=kx, polarization=polarization)

    eps = sys.float_info.epsilon
    R_err = abs(R_stratix - R_tmm) / (abs(R_tmm) + eps)
    T_err = abs(T_stratix - T_tmm) / (abs(T_tmm) + eps)

    return {
        "R_stratix": R_stratix,
        "T_stratix": T_stratix,
        "R_tmm": R_tmm,
        "T_tmm": T_tmm,
        "R_err": float(R_err),
        "T_err": float(T_err),
    }
