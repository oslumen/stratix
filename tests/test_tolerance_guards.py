"""Tolerance-based guards for near-degenerate incidence — Issue #53.

The power-flux formulas divide by the incident z-flux ``Re(kz0/denom0)``
and the Redheffer star product divides by ``1 - A22*B11``.  Both are
guarded, but the guards used to fire only on exact floating-point zero.
Just past the critical angle or at grazing incidence those quantities are
tiny rather than zero, so the guard missed and the division blew up.
These tests pin the tolerant behaviour.

The Redheffer guard is still the tolerance-based one #53 introduced.  The
incident-flux guard is not: issue #57 showed that a tolerance on the
residual flux decides the same physical input differently in single and
double precision, and replaced it with the superstrate's light line,
``|kx| >= Re(n_super)*k0``.  A lossy superstrate — the third regime this
module originally covered — is refused outright now, because R and T do
not partition energy there at all; see ``tests/test_lossy_superstrate.py``.
"""

from __future__ import annotations

import math

import numdiff as nd
import pytest
from phokaia import Layer
from phokaia import Material
from phokaia import Polarization
from phokaia import Stack

import stratix
from stratix._types import Method

_WL = 5e-7


def _stack(superstrate_eps: complex = 2.25, substrate_eps: complex = 9.0) -> Stack:
    """Superstrate over a single layer over a denser substrate.

    The substrate is denser than the superstrate so that it stays
    propagating past the superstrate's own cutoff: that is what leaves the
    transmitted flux finite while the incident flux collapses, which is
    the ratio the guard has to protect.
    """
    return Stack(
        superstrate=Material(epsilon=superstrate_eps),
        substrate=Material(epsilon=substrate_eps),
        layers=[Layer(thickness=100e-9, material=Material(epsilon=4.0))],
    )


def _assert_bounded(R, T, where: str) -> None:
    R_val, T_val = float(R), float(T)
    assert math.isfinite(R_val), f"R is not finite at {where}: {R_val}"
    assert math.isfinite(T_val), f"T is not finite at {where}: {T_val}"
    assert -1e-12 <= R_val <= 1.0 + 1e-9, f"R out of [0, 1+tol] at {where}: {R_val}"
    assert T_val >= -1e-12, f"T negative at {where}: {T_val}"
    assert R_val + T_val <= 1.0 + 1e-9, f"R+T exceeds 1 at {where}: {R_val + T_val}"


class TestEvanescentIncidenceWithFaintLoss:
    """A trace of superstrate loss is now refused rather than tolerated.

    This class used to pin the opposite: that ``epsilon = 2.25 + 1e-18j``
    past the light line came back agreeing with the lossless limit,
    because the flux guard's ``sqrt(eps)`` tolerance swallowed the tiny
    residual flux the loss produced.  Issue #57 showed the tolerance was
    the whole mechanism — raise the loss a few decades and the guard
    misses, ``T`` diverges as ``1/Im(epsilon)``, and which side of the
    jump a given input lands on depends on the working precision.

    Underneath that, R and T are not an energy partition for any lossy
    superstrate, so there was no correct number to return at either end
    of the range.  A lossy incident medium is refused instead, over the
    whole range, down to the last imaginary part that is bigger than
    double precision's rounding noise.
    Boundedness past the light line is now pinned by
    :class:`~tests.test_lossy_superstrate.TestRejectionIsDtypeIndependent`
    on a lossless superstrate, where it is well defined.
    """

    # ``1e-18``, which this class used to sample, is gone: against a real
    # part of 2.25 it sits below double precision's noise floor, so it is
    # not a loss anyone can have meant and is accepted as lossless.  That
    # end of the range is pinned by
    # :class:`~tests.test_lossy_superstrate.TestRoundingNoiseIsNotLoss`.
    @pytest.mark.parametrize("loss", [1e-15, 1e-12, 1e-9])
    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_faint_loss_is_refused_like_any_other_loss(self, set_backend, loss, pol):
        k0 = 2 * nd.pi / _WL
        kx = 1.5 * k0 * 1.2  # beyond the superstrate light line

        with pytest.raises(ValueError, match="lossless superstrate"):
            stratix.solve(
                _stack(superstrate_eps=2.25 + 1j * loss), _WL, kx=kx, polarization=pol
            )

    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_the_lossless_limit_itself_stays_bounded(self, set_backend, pol):
        """What the faint-loss stacks were standing in for is still checked."""
        k0 = 2 * nd.pi / _WL
        kx = 1.5 * k0 * 1.2

        result = stratix.solve(_stack(), _WL, kx=kx, polarization=pol)

        _assert_bounded(result.R[0, 0], result.T[0, 0], f"lossless, {pol.name}")
        assert float(result.R[0, 0]) == pytest.approx(1.0, abs=1e-12)
        assert float(result.T[0, 0]) == pytest.approx(0.0, abs=1e-12)


class TestCriticalAngleSweep:
    """Total internal reflection, sampled within float-eps of the transition."""

    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_through_the_critical_angle(self, set_backend, pol):
        n_sup, n_sub = 1.5, 1.0
        k0 = 2 * math.pi / _WL
        kx_c = n_sub * k0  # kz0 = 0 exactly at the critical angle

        # Stepping kx rather than the angle: consecutive angles a few ulps
        # apart round to the same float, so an angle sweep cannot actually
        # sample either side of the transition.  These fractions do, and
        # ``kx_c`` itself lands on kz0 == 0.
        fractions = [1 - 1e-3, 1 - 1e-9, 1 - 1e-15, 1.0, 1 + 1e-15, 1 + 1e-9, 1 + 1e-3]
        kx = nd.array([kx_c * f for f in fractions])

        result = stratix.solve(
            _stack(superstrate_eps=n_sup**2, substrate_eps=n_sub**2),
            _WL,
            kx=kx,
            polarization=pol,
        )

        for j, fraction in enumerate(fractions):
            _assert_bounded(result.R[0, j], result.T[0, j], f"kx/kx_c={fraction:.17g}")

        # Past the substrate light line the wave is totally internally
        # reflected: all the incident flux comes back.
        assert float(result.R[0, -1]) == pytest.approx(1.0, abs=1e-9)
        assert float(result.T[0, -1]) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_grazing_incidence(self, set_backend, pol):
        angles = nd.array([89.0, 89.9, 89.999, 89.999999, 90.0])
        result = stratix.solve_angles(
            _stack(superstrate_eps=1.0), _WL, angles, polarization=pol
        )

        for j in range(len(angles)):
            _assert_bounded(result.R[0, j], result.T[0, j], f"angle index {j}")

        # At exactly 90 degrees no power enters the stack.
        assert float(result.R[0, -1]) == pytest.approx(1.0, abs=1e-9)
        assert float(result.T[0, -1]) == pytest.approx(0.0, abs=1e-9)


#: The functions that divide by a quantity which can approach zero.  Only
#: these are held to the no-exact-equality rule: a module-wide ban would
#: also reject an integer early-out such as ``if n_layers == 0``, which is
#: exact by construction and has nothing to do with float tolerance.
GUARD_FUNCTIONS = {
    ("stratix.methods._medium_params", "_imaginary_part_is_negligible"),
    ("stratix.methods._util", "_no_incident_flux"),
    ("stratix.methods._util", "_safe_R"),
    ("stratix.methods._util", "_safe_T"),
    ("stratix.methods._smatrix", "_redheffer_star"),
    ("stratix._absorption", "_layer_absorption"),
}


class TestNoExactZeroComparisons:
    """Criterion: no ``== 0`` float comparisons left in the guarded code."""

    def test_flux_and_redheffer_guards_are_tolerance_based(self):
        import ast
        from pathlib import Path

        import stratix.methods._medium_params as medium_mod
        import stratix.methods._smatrix as smatrix_mod
        import stratix.methods._util as util_mod
        from stratix import _absorption as absorption_mod

        modules = {
            m.__name__: m for m in (util_mod, smatrix_mod, absorption_mod, medium_mod)
        }
        checked = set()

        for module_name, module in modules.items():
            tree = ast.parse(Path(module.__file__).read_text())
            for func in ast.walk(tree):
                if not isinstance(func, ast.FunctionDef):
                    continue
                if (module_name, func.name) not in GUARD_FUNCTIONS:
                    continue
                checked.add((module_name, func.name))
                offenders = [
                    ast.unparse(node)
                    for node in ast.walk(func)
                    if isinstance(node, ast.Compare)
                    and any(isinstance(op, ast.Eq | ast.NotEq) for op in node.ops)
                    and any(
                        isinstance(cmp, ast.Constant) and cmp.value == 0
                        for cmp in node.comparators
                    )
                ]
                assert not offenders, (
                    f"{module_name}.{func.name} still compares to zero "
                    f"exactly: {offenders}"
                )

        # A renamed or moved guard must fail loudly rather than silently
        # reducing this test to a no-op.
        assert checked == GUARD_FUNCTIONS, (
            f"guards not found: {GUARD_FUNCTIONS - checked}"
        )


class TestRedhefferDegeneracy:
    """The star-product denominator is guarded on magnitude, not equality."""

    def test_near_degenerate_denominator_is_guarded(self, set_backend):
        from stratix.methods._smatrix import _redheffer_star

        one = nd.array(1.0 + 0j)
        # A22 * B11 = 1 - 1e-12, so the denominator is 1e-12: far below
        # any scale the star product can meaningfully divide by, yet not
        # zero, so the old equality guard let it through.
        S_A = (one * 0.5, one * 0.5, one * 0.5, nd.array(1.0 - 1e-12 + 0j))
        S_B = (one, one * 0.5, one * 0.5, one * 0.5)

        guarded = _redheffer_star(S_A, S_B)

        # Compared against the star-product formulas evaluated by hand with
        # the substituted denominator, not against another call to the
        # function under test: routing the reference through the same
        # guarded branch would pass for any substituted value.
        A11, A12, A21, A22 = (complex(c) for c in S_A)
        B11, B12, B21, B22 = (complex(c) for c in S_B)
        safe_denom = 1.0  # what the guard substitutes for a negligible one
        expected = (
            A11 + A12 * B11 * A21 / safe_denom,
            A12 * B12 / safe_denom,
            B21 * A21 / safe_denom,
            B22 + B21 * A22 * B12 / safe_denom,
        )

        for component, reference in zip(guarded, expected, strict=True):
            assert math.isfinite(abs(complex(component))), f"non-finite: {component}"
            assert abs(complex(component) - reference) < 1e-12

        # Unguarded, the same inputs divide by 1e-12 and blow up by ~1e12.
        unguarded_S12 = A12 * B12 / (1.0 - A22 * B11)
        assert abs(unguarded_S12) > 1e11
        assert abs(complex(guarded[1])) < 1.0


class TestAbsorptionGuard:
    """Per-layer absorption divides by the same incident flux."""

    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_absorption_bounded_past_the_light_line(self, set_backend, pol):
        """A lossy layer under evanescent illumination absorbs nothing.

        The loss sits in the layer, where it is well posed.  The
        superstrate is lossless: a lossy one is refused outright now, so
        the guard can no longer be probed through it (issue #57).
        """
        k0 = 2 * nd.pi / _WL
        kx = 1.5 * k0 * 1.2
        stack = Stack(
            superstrate=Material(epsilon=2.25),
            substrate=Material(epsilon=9.0),
            layers=[Layer(thickness=100e-9, material=Material(epsilon=4.0 + 0.1j))],
        )

        result = stratix.solve(
            stack, _WL, kx=kx, polarization=pol, absorption=True, method=Method.SMATRIX
        )

        absorbed = result.layer_absorption
        assert absorbed is not None
        for value in nd.reshape(absorbed, (-1,)):
            assert math.isfinite(float(value))
            assert abs(float(value)) <= 1.0 + 1e-9


class TestAllMethodsShareTheGuard:
    """Every method routes R and T through the same tolerance guard.

    ``_safe_R`` and ``_safe_T`` no longer take the wavevectors the guard
    is computed from: the mask is built once in ``_medium_params`` and
    handed to them, and each of the four solvers had to be rewired to
    pass it.  The default-method tests above only exercise the S-matrix
    path, so the wiring in the other three would otherwise go unchecked.
    """

    @pytest.mark.parametrize(
        "method",
        [Method.SMATRIX, Method.ABELES, Method.ADMITTANCE, Method.DTN],
    )
    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_grazing_and_critical_angles_stay_bounded(self, set_backend, method, pol):
        n_sup, n_sub = 1.5, 1.0
        theta_c = math.degrees(math.asin(n_sub / n_sup))
        angles = nd.array([theta_c - 1e-13, theta_c, theta_c + 1e-13, 89.999999, 90.0])

        result = stratix.solve_angles(
            _stack(superstrate_eps=n_sup**2, substrate_eps=n_sub**2),
            _WL,
            angles,
            polarization=pol,
            method=method,
        )

        for j in range(len(angles)):
            _assert_bounded(result.R[0, j], result.T[0, j], f"{method.name} idx {j}")

        assert float(result.R[0, -1]) == pytest.approx(1.0, abs=1e-9)
        assert float(result.T[0, -1]) == pytest.approx(0.0, abs=1e-9)


class TestGuardMatchesFluxDenominator:
    """The no-flux mask tests the quantity the division uses — Issue #58.

    The superstrate light line ``|kx| >= Re(n_super)*k0`` and the flux
    denominator ``Re(kz0/denom0)`` are two floating-point expressions for
    the same physical boundary, and they disagree in the last couple of
    ulp.  A kx inside that band passed the wavevector-comparison guard and
    then divided a non-zero numerator by exactly zero, returning ``inf``
    for T and every absorption term.  The mask must therefore be derived
    from ``kz0`` itself, so mask and denominator cannot disagree — while
    staying a comparison of computed quantities, not a tolerance, so the
    cut-off still does not move with the working precision (issue #57).
    """

    #: The superstrate epsilon from the issue #58 report: its light line
    #: and the corresponding ``kz0`` disagree about a two-ulp band of kx.
    _EPS0 = 1.4030269036655723

    def _stack(self):
        return Stack(
            superstrate=Material(epsilon=self._EPS0),
            substrate=Material(epsilon=2.25),
            layers=[Layer(thickness=100e-9, material=Material(epsilon=4.0 + 0.1j))],
        )

    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_ulp_sweep_across_the_light_line(self, set_backend, pol):
        k0 = 2 * math.pi / _WL
        light = math.sqrt(self._EPS0) * k0

        kxs = [light]
        lo = hi = light
        for _ in range(3):
            lo = math.nextafter(lo, 0.0)
            hi = math.nextafter(hi, math.inf)
            kxs.extend([lo, hi])
        kxs.sort()

        result = stratix.solve(
            self._stack(), _WL, kx=nd.array(kxs), polarization=pol, absorption=True
        )

        for j, kx in enumerate(kxs):
            where = f"kx={kx!r} ({pol.name})"
            _assert_bounded(result.R[0, j], result.T[0, j], where)
            balance = float(result.energy_balance[0, j])
            assert math.isfinite(balance), f"energy balance not finite at {where}"
            assert balance == pytest.approx(1.0, abs=1e-9), where
            absorbed = float(result.layer_absorption[0, 0, j])
            assert math.isfinite(absorbed), f"absorption not finite at {where}"

    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_the_reported_kx_carries_no_flux(self, set_backend, pol):
        """The kx from the report: two ulp inside the light line, kz0 imaginary."""
        stack = Stack(
            superstrate=Material(epsilon=self._EPS0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve(
            stack, _WL, 14884795.196852127, polarization=pol, absorption=True
        )
        assert float(result.R[0, 0]) == pytest.approx(1.0, abs=1e-12)
        assert float(result.T[0, 0]) == pytest.approx(0.0, abs=1e-12)
        assert float(result.energy_balance[0, 0]) == pytest.approx(1.0, abs=1e-12)


class TestGuardFiresOnlyOnCollapsedFlux:
    """The guard keys on the light line, never on the sign of the flux.

    ``Re(kz0/denom0)`` comes out large and *negative* for a negative-index
    superstrate, because ``_kz_single`` picks the wrong branch there.
    That is a separate pre-existing problem.  Folding it into this guard
    would answer ``R = 1, T = 0`` — a physically plausible result for an
    ordinary propagating wave — and bury it.
    """

    def test_negative_flux_is_not_reported_as_no_flux(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=-2.25, mu=-1.0),
            substrate=Material(epsilon=9.0),
            layers=[Layer(thickness=100e-9, material=Material(epsilon=4.0))],
        )

        result = stratix.solve(stack, _WL, kx=0.0, polarization=Polarization.TE)

        R, T = float(result.R[0, 0]), float(result.T[0, 0])
        assert (R, T) != (1.0, 0.0), (
            "negative index superstrate took the no-flux branch: a "
            "propagating wave was reported as carrying no power"
        )

    def test_a_gain_superstrate_is_refused_rather_than_guarded(self, set_backend):
        """Gain used to reach the guard; it no longer reaches the solver.

        ``epsilon = 2.25 - 1e-6j`` is the mirror image of the loss case:
        the incident and reflected waves share the medium and their cross
        term carries flux that R and T do not account for.  Refusing it
        is what keeps this class's question — does the guard misfire on a
        negative flux? — about the branch choice alone.
        """
        stack = Stack(
            superstrate=Material(epsilon=2.25 - 1e-6j),
            substrate=Material(epsilon=9.0),
            layers=[Layer(thickness=100e-9, material=Material(epsilon=4.0))],
        )

        with pytest.raises(ValueError, match="lossless superstrate"):
            stratix.solve(stack, _WL, kx=0.0, polarization=Polarization.TE)

    def test_guard_fires_past_the_light_line_and_not_inside_it(self, set_backend):
        from stratix.methods._medium_params import _kz_single
        from stratix.methods._util import _no_incident_flux

        k0 = nd.array(2 * nd.pi / _WL)
        eps, mu = nd.array(2.25), nd.array(1.0)  # n = 1.5

        def fires(eps, mu, kx):
            return bool(_no_incident_flux(_kz_single(eps, mu, k0, kx)))

        # Inside the light cone: an ordinary propagating wave.
        assert not fires(eps, mu, nd.array(0.0))
        assert not fires(eps, mu, nd.array(1.4) * k0)
        # On it (kx = n*k0 with n = 1: the subtraction in kz0**2 is exact
        # there): kz0 = 0, grazing incidence, nothing enters.
        assert fires(nd.array(1.0), mu, nd.array(1.0) * k0)
        # Past it: evanescent.
        assert fires(eps, mu, nd.array(1.6) * k0)
        # A negative-index superstrate has a real kz0, so the light line
        # is where it always was and normal incidence propagates.
        assert not fires(nd.array(-2.25), nd.array(-1.0), nd.array(0.0))
