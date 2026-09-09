"""Pre-computed medium parameters (omega, k0, wavevectors, impedances).

Shared by Abélès, admittance, DTN, S-matrix, and autodiff solvers.
"""

from __future__ import annotations

import sys

import numdiff as nd
from phokaia import Material
from phokaia import Polarization
from phokaia import Stack

from ._util import _no_incident_flux


def _kz_single(epsilon: nd.ndarray, mu: nd.ndarray, k0: float, kx: float) -> nd.ndarray:
    """Compute the out-of-plane wavevector component kz.

    Chooses the physical branch: Im(kz) >= 0, or Re(kz) >= 0 when Im(kz) = 0.
    """
    kz_sq = epsilon * mu * k0**2 - kx**2
    kz_sq = kz_sq + 0j
    kz = nd.sqrt(kz_sq)
    neg = (nd.imag(kz) < 0) | ((nd.imag(kz) == 0) & (nd.real(kz) < 0))
    return nd.where(neg, -kz, kz)


#: Relative size below which an imaginary part is rounding noise rather
#: than loss.  Fixed at double-precision epsilon rather than read off the
#: active backend: whether a stack is accepted must not move with the
#: working dtype, which is the whole complaint issue #57 opens with.  A
#: tolerance tied to ``nd.get_precision()`` would accept
#: ``epsilon = 2.25 + 1e-7j`` in single precision and refuse it in double
#: — a new version of the seven-decade dtype split this replaces.  Double
#: is also the precision material constants are written in: a Python
#: complex literal is a pair of doubles whatever backend consumes it.
_LOSSLESS_RTOL = sys.float_info.epsilon


def _imaginary_part_is_negligible(value: nd.ndarray) -> bool | None:
    """Whether ``value`` is real to within the working precision.

    The comparison is relative — ``|Im| <= _LOSSLESS_RTOL * |Re|`` —
    rather than against exact zero.  A material built by arithmetic
    rather than literally, ``2.25 * (1 + 1e-16j)`` say, carries an
    imaginary part that is pure rounding noise, and refusing it would be
    refusing a lossless medium.  Anything above the last bit of the real
    part is a loss the user meant.

    Returns
    -------
    ``True`` or ``False`` when the value can be read, and ``None`` when it
    cannot — inside ``nd.jit`` the value is a tracer and converting the
    comparison to a bool raises rather than answering.  ``None`` is "do
    not know", which callers must not read as "fine".
    """
    # ``x - real(x)`` rather than ``nd.imag``: torch refuses ``imag`` on a
    # real tensor, and a real-valued material is the common case this has
    # to pass through untouched.
    imaginary = nd.abs(value - nd.real(value))
    negligible = imaginary <= _LOSSLESS_RTOL * nd.abs(nd.real(value))
    try:
        return bool(nd.all(negligible))
    except TypeError:
        # Concretising a tracer raises TracerBoolConversionError, a
        # TypeError.  Nothing else in the expression above can raise one:
        # it is arithmetic on an ndarray throughout.
        return None


def _reject_lossy_superstrate(superstrate: Material, omega: nd.ndarray) -> None:
    """Raise unless the incident medium is lossless.

    ``R = |r|^2`` and ``T = Re(kzN/denomN)/Re(kz0/denom0) * |t|^2`` are an
    energy partition only when the superstrate is transparent.  It carries
    both the incident and the reflected wave; in a lossless medium the
    cross term between them contributes no real z-directed flux, but in a
    lossy or gain medium it does, and neither R nor T accounts for it.
    ``R + T + sum(A)`` then measures 1.0008 at ``Im(epsilon) = 0.01`` and
    1.124 at ``Im(epsilon) = 1``, at normal incidence, with no evanescence
    involved.  ``|r|^2`` is evaluated at the first interface on top of
    that, so R also becomes reference-plane dependent: a detector a
    distance ``d`` up in the superstrate sees it attenuated by
    ``exp(-2*Im(kz0)*d)``.

    Returning numbers that quietly fail to partition energy is worse than
    refusing, so the medium is refused.  The ``tmm`` reference library
    takes the same line for the mirror-image case, asserting on gain media
    because "it's ambiguous which beam is incoming vs outgoing".

    A lossy *substrate* is unaffected and deliberately still allowed: it
    is semi-infinite and holds a single outgoing wave, so there is no
    backward wave to interfere with, ``Re(kzN/denomN) * |t|^2`` is
    unambiguous, and energy balance closes exactly.

    .. warning::

       The check reads its input's value, which a tracer will not give up,
       so it does not run under ``nd.jit`` — a compiled solve on a lossy
       superstrate returns the non-partitioning numbers this function
       exists to refuse.  ``solve()`` is trace-safe (#50), and breaking
       compilation to close the hole would cost more than the hole does;
       every eager call, which is every call a user makes directly, is
       checked.  Asking the material for an omega-free value does not
       help: inside a trace numdiff hands back a tracer for any array it
       builds, constants included.  Each property is still decided on its
       own, so a readable one is never skipped because another was not.

    Parameters
    ----------
    superstrate : The incident medium.
    omega : Angular frequency the solve evaluates its properties at.

    Raises
    ------
    ValueError
        If either quantity has an imaginary part above the last bit of its
        real part.
    """
    for name, prop in (("epsilon", superstrate.epsilon), ("mu", superstrate.mu)):
        value = prop(omega=omega)
        # ``continue``, not ``return``: a property that cannot be read
        # says nothing about the other one, which may well be readable.
        if _imaginary_part_is_negligible(value) is False:
            raise ValueError(
                "stratix needs a lossless superstrate: got a superstrate "
                f"{name} of {value}, whose imaginary part is not negligible "
                "against its real part.  R and T are an energy partition "
                "only when the incident medium is transparent -- it carries "
                "both the incident and the reflected wave, and in a lossy or "
                "gain medium their cross term carries real z-directed flux "
                "that neither |r|^2 nor the z-flux ratio accounts for, so "
                "R + T + sum(absorption) does not close.  |r|^2 is also "
                "taken at the first interface, which makes R depend on where "
                "the detector sits.  A lossy substrate is fine and needs no "
                "change: it is semi-infinite with a single outgoing wave, so "
                "its flux is unambiguous.  To model this stack, set the "
                "superstrate's imaginary part to zero -- an oscillator model "
                "with small but non-zero damping is refused too, since where "
                "to draw the line between negligible and real loss is the "
                "caller's judgement, not the library's."
            )


def _medium_params(
    stack: Stack, wavelength: float, kx: float, polarization: Polarization
) -> tuple:
    """Pre-compute omega, k0, kzs, epsilons, mus, and denom_vals for a stack.

    Parameters
    ----------
    stack : Multilayer stack (superstrate, layers, substrate).
    wavelength : Vacuum wavelength in meters.
    kx : In-plane wavevector component in rad/m.
    polarization : ``TE`` or ``TM``.

    Returns
    -------
    omega : 0-D ndarray — angular frequency (rad/s).
    k0 : 0-D ndarray — vacuum wavevector (rad/m).
    kzs : List of 0-D ndarrays — out-of-plane wavevector per medium.
    epsilons : List of 0-D ndarrays — epsilon(omega) per medium.
    mus : List of 0-D ndarrays — mu(omega) per medium.
    denom_vals : List of 0-D ndarrays — mus for TE, epsilons for TM.
    no_flux : Boolean ndarray — where the incident wave carries no
        z-directed flux.  Computed here, from the same ``kzs[0]`` the
        flux denominator uses, because every method needs the same mask
        and deriving it from any other expression re-opens the ulp-wide
        disagreement band of issue #58.

    Raises
    ------
    ValueError
        If the superstrate is not lossless.  See
        :func:`_reject_lossy_superstrate`.
    """
    c = 299792458.0
    omega = nd.array(2 * nd.pi * c / wavelength)
    k0 = nd.array(2 * nd.pi / wavelength)
    kx = nd.array(kx)

    media = (
        [stack.superstrate]
        + [layer.material for layer in stack.layers]
        + [stack.substrate]
    )
    _reject_lossy_superstrate(stack.superstrate, omega)

    epsilons = [m.epsilon(omega=omega) for m in media]
    mus = [m.mu(omega=omega) for m in media]
    kzs = [_kz_single(eps, mu, k0, kx) for eps, mu in zip(epsilons, mus, strict=True)]

    if polarization == Polarization.TE:
        denom_vals = mus
    elif polarization == Polarization.TM:
        denom_vals = epsilons
    else:
        raise NotImplementedError(f"Polarization {polarization!r} not supported")

    no_flux = _no_incident_flux(kzs[0])

    return omega, k0, kzs, epsilons, mus, denom_vals, no_flux
