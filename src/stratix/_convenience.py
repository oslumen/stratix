"""Convenience APIs: solve_angles (Issue #19)."""

from __future__ import annotations

import numdiff as nd
from phokaia import Stack

from ._result import Result
from ._solve import solve
from ._types import Method
from ._types import Polarization


def solve_angles(
    stack: Stack,
    wavelengths: float,
    angles: float | list,
    polarization: Polarization,
    method: Method = Method.AUTO,
    absorption: bool = False,
) -> Result:
    """Compute reflectance/transmittance for given incidence angles.

    Converts incidence angle θ (degrees) to in-plane wavevector
    kx = (2π/λ) · n_super · sin(θ) and delegates to :func:`solve`.

    Parameters
    ----------
    stack : Planar multilayer stack.
    wavelengths : Vacuum wavelength in meters (scalar).
    angles : Incidence angle(s) in degrees from normal (scalar or array).
    polarization : ``TE`` or ``TM``.
    method : Solver method.
    absorption : If ``True``, compute per-layer absorption (not yet implemented).

    Returns
    -------
    Result with ``R``, ``T``, ``wavelengths``, ``kx`` arrays.
    """
    wl = float(nd.array(wavelengths))

    if isinstance(angles, (int, float)):
        angles_list = [float(angles)]
    else:
        angles_list = [float(a) for a in angles]

    if len(angles_list) == 0:
        raise ValueError("angles must be non-empty")

    for a in angles_list:
        if a < 0 or a > 90:
            raise ValueError(
                f"Incidence angle must be in [0, 90] degrees, got {a}"
            )

    eps_super = float(stack.superstrate.epsilon(wl))
    mu_super = float(stack.superstrate.mu(wl))
    n_super = float(nd.sqrt(nd.array(eps_super * mu_super)))

    k0 = 2 * nd.pi / wl

    R_list: list[float] = []
    T_list: list[float] = []
    kx_list: list[float] = []

    for theta_deg in angles_list:
        theta_rad = theta_deg * nd.pi / 180
        kx = float(n_super * k0 * nd.sin(nd.array(theta_rad)))

        result = solve(
            stack,
            wavelength=wl,
            kx=kx,
            polarization=polarization,
            method=method,
            absorption=absorption,
        )

        R_list.append(float(result.R[0]))
        T_list.append(float(result.T[0]))
        kx_list.append(kx)

    return Result(
        R=nd.array(R_list),
        T=nd.array(T_list),
        wavelengths=nd.array([wl] * len(R_list)),
        kx=nd.array(kx_list),
        polarization=polarization,
        method_used=result.method_used,
    )
