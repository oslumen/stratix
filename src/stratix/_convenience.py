"""Convenience APIs: solve_angles, solve_from_source (Issue #19)."""

from __future__ import annotations

import numdiff as nd
from phokaia import PlaneWave
from phokaia import Polarization
from phokaia import Stack

from ._result import Result
from ._solve import _resolve_method
from ._solve import _solve_grid
from ._solve import _sweep_axis
from ._solve import _validate_thicknesses
from ._solve import solve
from ._types import Method
from .methods._medium_params import _imaginary_part_is_negligible

_C0: float = 299792458.0  # speed of light in vacuum (m/s)


def solve_angles(
    stack: Stack,
    wavelengths: float | list | nd.ndarray,
    angles: float | list | nd.ndarray,
    polarization: Polarization,
    method: Method = Method.AUTO,
    absorption: bool = False,
    thicknesses: nd.ndarray | None = None,
) -> Result:
    """Compute reflectance/transmittance for given incidence angles.

    Converts every incidence angle θ (degrees) to an in-plane wavevector
    kx = n_super(ω) · (2π/λ) · sin(θ) and runs one vectorized solve over the
    whole grid.  The superstrate index is evaluated at the angular frequency
    ω = 2πc/λ, so a dispersive superstrate gets the index belonging to each
    wavelength.

    Parameters
    ----------
    stack : Planar multilayer stack.
    wavelengths : Vacuum wavelength(s) in meters (scalar or array).
    angles : Incidence angle(s) in degrees from normal (scalar or array).
    polarization : ``TE``, ``TM``, or ``BOTH``.
    method : Solver method.
    absorption : If ``True``, also return per-layer absorption and the
        energy balance, as in :func:`solve`.
    thicknesses : Optional 1-D array overriding the stack's layer thicknesses.

    Returns
    -------
    Result following the ``(Nλ, Nk)`` shape contract of :func:`solve`.  One
    angle is one kx, so ``R`` and ``T`` are ``(Nλ, n_angles)``.

    Nothing between the inputs and the returned fields casts to ``float``,
    so ``nd.grad`` w.r.t. wavelength, angle or thickness works on the
    autodiff backends.  The angle range check does read its input's value,
    though, so unlike :func:`solve` this is not traceable under ``nd.jit``.

    Raises
    ------
    ValueError
        If the superstrate is absorbing or metallic.  A complex index makes
        kx = n(ω)·k0·sin θ complex, and R and T stop being ``|r|²`` and the
        z-flux ratio, so the angle no longer names a meaningful sweep
        point.  For a *metallic* superstrate — ``eps·mu`` real but negative
        — :func:`solve` still accepts an explicit kx and reports the
        evanescent ``R = 1, T = 0``.  An *absorbing* one it does not: a
        lossy incident medium makes R and T stop partitioning energy at
        all, so :func:`solve` refuses it too.

    Shapes
    ------
    ``kx`` is the one field that cannot always stay a 1-D sweep coordinate:
    it depends on the wavelength as well as the angle.  For a scalar
    wavelength it is ``(n_angles,)`` as usual; for an array of wavelengths
    it follows the grid as ``(Nλ, n_angles)``, a length-1 array included.
    """
    resolved = _resolve_method(method)

    _validate_thicknesses(thicknesses, stack)

    # A scalar wavelength and a length-1 array both give an Nλ of 1, but
    # only the scalar leaves kx a 1-D sweep coordinate, so the distinction
    # has to be taken before the axis is promoted.
    wl_input = wavelengths if hasattr(wavelengths, "ndim") else nd.asarray(wavelengths)
    scalar_wavelength = wl_input.ndim == 0

    wl_axis = _sweep_axis(wl_input)
    angle_axis = _sweep_axis(angles)

    if angle_axis.shape[0] == 0:
        raise ValueError("angles must be non-empty")
    if bool(nd.any(angle_axis < 0)) or bool(nd.any(angle_axis > 90)):
        raise ValueError(
            f"Incidence angles must be in [0, 90] degrees, got {angle_axis}"
        )

    # The wavelength axis is a column and the angle axis a row, so the two
    # broadcast into the (Nλ, n_angles) grid the shape contract asks for.
    wl_col = wl_axis.reshape(-1, 1)
    omega = 2 * nd.pi * _C0 / wl_col
    eps_mu = nd.asarray(
        stack.superstrate.epsilon(omega=omega) * stack.superstrate.mu(omega=omega)
    )

    # n = sqrt(eps·mu) is real only where eps·mu is real and non-negative,
    # and kx is real only where n is.  The imaginary part is judged by the
    # same predicate ``solve()`` rejects a lossy superstrate with, so a
    # medium that is lossless to within rounding noise — ``2.25*(1+1e-16j)``
    # from complex arithmetic, or a zero-damping oscillator model — is
    # accepted by both entry points or by neither.
    imaginary = _imaginary_part_is_negligible(eps_mu) is False
    if imaginary or bool(nd.any(nd.real(eps_mu) < 0)):
        raise ValueError(
            "solve_angles needs a transparent superstrate: an angle maps to "
            "kx = n(omega)*k0*sin(theta), which is complex unless eps*mu is "
            f"real and non-negative (got {eps_mu}).  R and T are then no "
            "longer |r|^2 and the z-flux ratio.  A metallic superstrate "
            "(eps*mu real and negative) can still be driven by calling "
            "solve() with an explicit kx; an absorbing one cannot, because "
            "R and T stop partitioning energy there -- solve() rejects it "
            "for the same reason."
        )

    n_super = nd.sqrt(eps_mu)
    k0 = 2 * nd.pi / wl_col
    kx_grid = n_super * k0 * nd.sin(angle_axis * nd.pi / 180)

    grid = _solve_grid(
        stack, wl_col, kx_grid, polarization, resolved, absorption, thicknesses
    )

    return grid.to_result(
        wavelengths=wl_axis,
        kx=kx_grid.reshape(-1) if scalar_wavelength else kx_grid,
        polarization=polarization,
        method_used=resolved,
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
