"""Convenience APIs: solve_angles, solve_from_source (Issue #19)."""

from __future__ import annotations

import numdiff as nd
from phokaia import PlaneWave
from phokaia import Polarization
from phokaia import Stack

from ._result import Result
from ._solve import solve
from ._types import Method

_C0: float = 299792458.0  # speed of light in vacuum (m/s)


def solve_angles(
    stack: Stack,
    wavelengths: float,
    angles: float | list,
    polarization: Polarization,
    method: Method = Method.AUTO,
    absorption: bool = False,
    thicknesses: nd.ndarray | None = None,
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
    absorption : If ``True``, also return per-layer absorption and the
        energy balance, as in :func:`solve`.
    thicknesses : Optional 1-D array overriding the stack's layer thicknesses.

    Returns
    -------
    Result following the ``(Nλ, Nk)`` shape contract of :func:`solve`.  One
    angle is one kx, so ``R`` and ``T`` are ``(1, n_angles)``.

    Notes
    -----
    Unlike :func:`solve`, this is **not** yet differentiable end to end: the
    wavelength and each angle are cast to ``float`` before the solve, which
    breaks the autodiff trace at the inputs.  Removing those casts, together
    with evaluating the superstrate index at ω rather than λ, is issue #51.
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

    # Each entry is one angle's sweep corner; stacking them builds the Nk axis.
    R_terms: list[nd.ndarray] = []
    T_terms: list[nd.ndarray] = []
    absorption_terms: list[nd.ndarray] = []
    balance_terms: list[nd.ndarray] = []
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
            thicknesses=thicknesses,
        )

        R_terms.append(result.R[..., 0, 0])
        T_terms.append(result.T[..., 0, 0])
        if absorption:
            absorption_terms.append(result.layer_absorption[..., 0, 0])
            balance_terms.append(result.energy_balance[..., 0, 0])
        kx_list.append(kx)

    # One angle is one kx, so the angles fill the Nk axis of the (Nλ, Nk)
    # shape contract and the scalar wavelength is the length-1 Nλ axis.  The
    # per-angle entries stack onto the end, then a length-1 Nλ axis is opened
    # in front of them; BOTH's leading TE/TM axis rides along untouched.
    def _grid(terms: list) -> nd.ndarray:
        return nd.stack(terms, axis=-1)[..., None, :]

    return Result(
        R=_grid(R_terms),
        T=_grid(T_terms),
        wavelengths=nd.asarray(wl).reshape(1),
        kx=nd.array(kx_list),
        polarization=polarization,
        method_used=result.method_used,
        layer_absorption=_grid(absorption_terms) if absorption else None,
        energy_balance=_grid(balance_terms) if absorption else None,
    )


def solve_from_source(
    stack: Stack,
    source: PlaneWave,
    polarization: Polarization,
    method: Method = Method.AUTO,
    absorption: bool = False,
    thicknesses: nd.ndarray | None = None,
) -> Result:
    """Compute reflectance/transmittance from a PlaneWave source.

    Extracts wavelength (from ω) and in-plane wavevector kx from a
    ``phokaia.PlaneWave`` and delegates to :func:`solve`.

    Parameters
    ----------
    stack : Planar multilayer stack.
    source : PlaneWave object defining frequency, angle, and medium.
    polarization : ``TE`` or ``TM``.
    method : Solver method.
    absorption : If ``True``, compute per-layer absorption.

    Returns
    -------
    Result following the ``(Nλ, Nk)`` shape contract of :func:`solve`.  A
    plane wave is a single (wavelength, kx) point, so ``R`` and ``T`` are
    ``(1, 1)``.
    """
    omega = complex(source.omega)
    if omega == 0:
        raise ValueError("source omega must be non-zero")

    wavelength = 2 * nd.pi * _C0 / omega.real

    kx: float | nd.ndarray
    if source.dim == 1:
        kx = 0.0
    else:
        wv = source.wavevector
        kx = float(wv[0].real)

    return solve(
        stack,
        wavelength=wavelength,
        kx=kx,
        polarization=polarization,
        method=method,
        absorption=absorption,
        thicknesses=thicknesses,
    )
