"""A lossy superstrate is rejected, and evanescence is decided by the light line — Issue #57.

Two defects, both about what happens in the incident medium.

The flux guard used to fire on a ``sqrt(eps)`` tolerance applied to the
residual incident flux.  Past the superstrate's light line that residual
is set by the superstrate's loss, not by the physics of the sweep, so the
guard's decision — and with it a seven-decade jump in ``T`` — moved with
the working precision.  The criterion is now the loss-independent,
dtype-independent light line ``kx >= Re(n_super) * k0``.

Underneath that sat the deeper problem: ``R = |r|^2`` and the z-flux ratio
``T`` are not an energy partition for *any* lossy superstrate, evanescent
or not.  The incident and reflected waves share that medium and their
cross term carries real z-directed flux that neither term accounts for,
so ``R + T + sum(A)`` does not close; ``|r|^2`` is also evaluated at the
first interface, which makes ``R`` reference-plane dependent.  Rather
than return numbers that quietly fail to partition energy, a superstrate
that is not lossless is now rejected.  A lossy *substrate* is untouched:
it is semi-infinite with a single outgoing wave, so its flux is
unambiguous.
"""

from __future__ import annotations

import contextlib
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

_ALL_METHODS = [Method.SMATRIX, Method.ABELES, Method.ADMITTANCE, Method.DTN]

#: The superstrate-loss range the old tolerance guard could not cover.
#: Below ~1e-8 it fired and answered R = 1, T = 0; above it, it missed and
#: let ``T`` diverge as ``1/Im(epsilon)``, with the crossover sitting
#: wherever the working precision put it.  Every one of them is now the
#: same clear error.
_LOSS_RANGE = [1e-9, 1e-8, 3e-8, 1e-7, 1e-6, 1e-4, 1e-2, 1.0]


@contextlib.contextmanager
def _precision(name: str):
    """Run the block with the backend's float precision set to ``name``."""
    previous = nd.get_precision()
    nd.set_precision(name)
    try:
        yield
    finally:
        nd.set_precision(previous)


def _stack(superstrate_eps=2.25, substrate_eps=9.0, layer_eps=4.0) -> Stack:
    return Stack(
        superstrate=Material(epsilon=superstrate_eps),
        substrate=Material(epsilon=substrate_eps),
        layers=[Layer(thickness=100e-9, material=Material(epsilon=layer_eps))],
    )


class TestLossySuperstrateIsRejected:
    """``solve()`` refuses an incident medium that is not lossless."""

    @pytest.mark.parametrize("loss", _LOSS_RANGE)
    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_every_loss_magnitude_raises(self, set_backend, loss, pol):
        k0 = 2 * math.pi / _WL
        with pytest.raises(ValueError, match="lossless superstrate"):
            stratix.solve(
                _stack(superstrate_eps=2.25 + 1j * loss),
                _WL,
                kx=1.8 * k0,
                polarization=pol,
            )

    def test_normal_incidence_raises_too(self, set_backend):
        """The energy-partition defect needs no evanescence to show up."""
        with pytest.raises(ValueError, match="lossless superstrate"):
            stratix.solve(
                _stack(superstrate_eps=2.25 + 0.01j),
                _WL,
                kx=0.0,
                polarization=Polarization.TE,
            )

    def test_gain_superstrate_raises(self, set_backend):
        """Gain is the same ambiguity with the opposite sign."""
        with pytest.raises(ValueError, match="lossless superstrate"):
            stratix.solve(
                _stack(superstrate_eps=2.25 - 1e-6j),
                _WL,
                kx=0.0,
                polarization=Polarization.TE,
            )

    def test_lossy_permeability_raises(self, set_backend):
        """The incident medium's ``mu`` carries the same ambiguity as its ``eps``."""
        stack = Stack(
            superstrate=Material(epsilon=2.25, mu=1.0 + 1e-3j),
            substrate=Material(epsilon=9.0),
        )
        with pytest.raises(ValueError, match="lossless superstrate"):
            stratix.solve(stack, _WL, kx=0.0, polarization=Polarization.TE)

    @pytest.mark.parametrize("method", _ALL_METHODS)
    def test_every_method_rejects(self, set_backend, method):
        with pytest.raises(ValueError, match="lossless superstrate"):
            stratix.solve(
                _stack(superstrate_eps=2.25 + 0.01j),
                _WL,
                kx=0.0,
                polarization=Polarization.TE,
                method=method,
            )

    def test_both_polarization_rejects(self, set_backend):
        with pytest.raises(ValueError, match="lossless superstrate"):
            stratix.solve(
                _stack(superstrate_eps=2.25 + 0.01j),
                _WL,
                kx=0.0,
                polarization=Polarization.BOTH,
            )

    def test_error_names_the_offending_quantity(self, set_backend):
        """The message has to say which of eps/mu is at fault, and its value."""
        with pytest.raises(ValueError) as excinfo:
            stratix.solve(
                _stack(superstrate_eps=2.25 + 0.01j),
                _WL,
                kx=0.0,
                polarization=Polarization.TE,
            )
        message = str(excinfo.value)
        assert "epsilon" in message
        assert "substrate" in message, "the message should say a lossy substrate is fine"


class TestRejectionIsDtypeIndependent:
    """The same physical input gets the same answer in single and double.

    This is the acceptance criterion the old guard failed outright: at
    ``Im(epsilon) = 1e-7`` past the light line it answered ``T = 0`` in
    single precision and ``T = 1.5e7`` in double, because the threshold
    was ``sqrt(eps)`` of the working precision rather than a property of
    the stack.  A rejection cannot depend on the dtype.
    """

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_the_documented_dtype_split_now_raises_in_both(
        self, set_backend, precision
    ):
        k0 = 2 * math.pi / _WL
        with (
            _precision(precision),
            pytest.raises(ValueError, match="lossless superstrate"),
        ):
            stratix.solve(
                _stack(superstrate_eps=2.25 + 1e-7j),
                _WL,
                kx=1.8 * k0,
                polarization=Polarization.TE,
            )

    def test_lossless_evanescence_agrees_across_precisions(self, set_backend):
        """Past the light line both precisions give exactly R = 1, T = 0."""
        k0 = 2 * math.pi / _WL
        answers = []
        for precision in ("single", "double"):
            with _precision(precision):
                result = stratix.solve(
                    _stack(), _WL, kx=1.8 * k0, polarization=Polarization.TE
                )
                answers.append((float(result.R[0, 0]), float(result.T[0, 0])))

        assert answers[0] == answers[1] == (1.0, 0.0)

    def test_the_transition_sits_at_the_light_line_in_both_precisions(
        self, set_backend
    ):
        """Which sweep points are cut off must not move with the dtype."""
        k0 = 2 * math.pi / _WL
        n_super = 1.5
        fractions = [0.5, 0.9, 0.99, 1.01, 1.1, 1.5]
        kx = [n_super * k0 * f for f in fractions]

        fired = []
        for precision in ("single", "double"):
            with _precision(precision):
                result = stratix.solve(
                    _stack(), _WL, kx=nd.array(kx), polarization=Polarization.TE
                )
                fired.append(
                    tuple(float(result.T[0, j]) == 0.0 for j in range(len(fractions)))
                )

        assert fired[0] == fired[1]
        # Below the light line power gets through; above it, none does.
        assert fired[0] == (False, False, False, True, True, True)


def _guard_fires(eps, mu, kx_factor):
    """Evaluate the no-flux guard the way ``_medium_params`` builds it.

    The mask keys on the incident ``kz0`` — the same wavevector the flux
    denominator divides by (issue #58) — so the tests route through
    ``_kz_single`` exactly as the solver does.
    """
    from stratix.methods._medium_params import _kz_single
    from stratix.methods._util import _no_incident_flux

    k0 = nd.array(2 * nd.pi / _WL)
    kx = nd.array(kx_factor) * k0
    return bool(_no_incident_flux(_kz_single(nd.array(eps), nd.array(mu), k0, kx)))


class TestLightLineCriterion:
    """The guard keys on the sign of ``Re(kz0)`` and nothing else."""

    def test_propagating_evanescent_and_exact_grazing(self, set_backend):
        # Inside the light cone: flux enters.
        assert not _guard_fires(2.25, 1.0, 1.4)
        # Exactly on the light line (n = 1, kx = k0: the cancellation in
        # kz0**2 is exact there): kz0 = 0, so nothing enters.
        assert _guard_fires(1.0, 1.0, 1.0)
        # Past it: evanescent.
        assert _guard_fires(2.25, 1.0, 1.8)

    def test_a_tiny_but_real_kz_still_carries_flux(self, set_backend):
        """The criterion is the light line, not a float-precision tolerance.

        At ``kx = 0.9999 * n * k0`` the incident ``kz0`` is a thousandth
        of ``k0`` — small, but the wave is genuinely propagating and the
        old ``sqrt(eps)`` tolerance is the only thing that could have
        called it evanescent.  Whether it does depends on the working
        precision, which is exactly what issue #57 is about.
        """
        for precision in ("single", "double"):
            with _precision(precision):
                assert not _guard_fires(2.25, 1.0, 1.5 * 0.9999)

    def test_a_metallic_superstrate_never_carries_flux(self, set_backend):
        """``eps*mu < 0`` makes ``kz0`` imaginary: evanescent at every kx."""
        assert _guard_fires(-2.25, 1.0, 0.0)
        assert _guard_fires(-2.25, 1.0, 1.0)

    def test_a_negative_index_superstrate_still_carries_flux(self, set_backend):
        """``eps < 0`` and ``mu < 0`` give a real ``kz0``: the wave propagates.

        ``Re(kz0/denom0)`` is large and *negative* here, which is a
        separate branch-choice problem in ``_kz_single``.  Reporting it as
        no flux would answer ``R = 1, T = 0`` and bury it.
        """
        assert not _guard_fires(-2.25, -1.0, 0.0)

    def test_the_sign_of_kx_does_not_matter(self, set_backend):
        """kz0 depends on kx**2, so the guard has to be symmetric in kx."""
        for magnitude in (1.4, 1.8):
            assert _guard_fires(2.25, 1.0, magnitude) == _guard_fires(
                2.25, 1.0, -magnitude
            )


class TestLossySubstrateIsUntouched:
    """A lossy substrate is well posed and must keep closing energy exactly.

    It is semi-infinite and holds a single outgoing wave, so there is no
    backward wave to interfere with and ``Re(kzN/denomN) * |t|^2`` is
    unambiguous.  This is the regression guard on the other half of the
    fix: rejecting the superstrate must not spill over onto the substrate.
    """

    @pytest.mark.parametrize("loss", [1e-9, 1e-6, 1e-3, 0.1, 1.0])
    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_energy_balance_closes_at_normal_incidence(self, set_backend, loss, pol):
        result = stratix.solve(
            _stack(substrate_eps=9.0 + 1j * loss),
            _WL,
            kx=0.0,
            polarization=pol,
            absorption=True,
            method=Method.SMATRIX,
        )
        assert float(result.energy_balance[0, 0]) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("loss", [1e-6, 1e-3, 1.0])
    def test_energy_balance_closes_off_normal(self, set_backend, loss):
        k0 = 2 * math.pi / _WL
        result = stratix.solve(
            _stack(substrate_eps=9.0 + 1j * loss),
            _WL,
            kx=0.7 * k0,
            polarization=Polarization.TM,
            absorption=True,
            method=Method.SMATRIX,
        )
        assert float(result.energy_balance[0, 0]) == pytest.approx(1.0, abs=1e-12)

    def test_a_lossy_layer_and_a_lossy_substrate_together_close(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=9.0 + 0.5j),
            layers=[Layer(thickness=120e-9, material=Material(epsilon=4.0 + 0.3j))],
        )
        result = stratix.solve(
            stack,
            _WL,
            kx=0.0,
            polarization=Polarization.TE,
            absorption=True,
            method=Method.SMATRIX,
        )
        assert float(result.energy_balance[0, 0]) == pytest.approx(1.0, abs=1e-12)


class TestRoundingNoiseIsNotLoss:
    """The imaginary part is judged against the real part, not against zero.

    A material built by arithmetic rather than written as a literal
    carries an imaginary part of pure rounding noise.  ``2.25*(1-1e-16j)``
    is a lossless medium expressed clumsily, and an exact ``!= 0`` test
    would refuse it with an error that says there is no way to proceed.
    """

    # The noise is on the loss side (``+``) throughout.  A *negative*
    # imaginary part of the same size flips the branch ``_kz_single``
    # picks, which is a separate pre-existing problem that issue #57
    # explicitly leaves alone; pinning it here would be pinning that bug,
    # not this tolerance.
    @pytest.mark.parametrize(
        "epsilon",
        [
            2.25 * (1 + 1e-16j),
            complex(2.25, 5e-324),  # denormal: as close to zero as floats go
            complex(2.25, 0.0),
        ],
    )
    def test_negligible_imaginary_parts_are_accepted(self, set_backend, epsilon):
        result = stratix.solve(
            _stack(superstrate_eps=epsilon), _WL, kx=0.0, polarization=Polarization.TE
        )
        assert 0.0 <= float(result.R[0, 0]) <= 1.0

    def test_a_loss_above_the_noise_floor_is_still_refused(self, set_backend):
        """The tolerance must not be wide enough to let real loss through."""
        with pytest.raises(ValueError, match="lossless superstrate"):
            stratix.solve(
                _stack(superstrate_eps=2.25 * (1 + 1e-9j)),
                _WL,
                kx=0.0,
                polarization=Polarization.TE,
            )

    @pytest.mark.parametrize("precision", ["single", "double"])
    def test_the_noise_floor_does_not_move_with_the_working_dtype(
        self, set_backend, precision
    ):
        """A tolerance read off the active dtype would reopen issue #57.

        ``Im/Re = 4.4e-8`` sits below single precision's machine epsilon
        and far above double's, so a tolerance tied to
        ``nd.get_precision()`` would accept this stack in float32 and
        refuse it in float64 — the same dtype-dependent split, one
        rewrite later.
        """
        with (
            _precision(precision),
            pytest.raises(ValueError, match="lossless superstrate"),
        ):
            stratix.solve(
                _stack(superstrate_eps=2.25 + 1e-7j),
                _WL,
                kx=0.0,
                polarization=Polarization.TE,
            )

    def test_solve_angles_draws_the_line_in_the_same_place(self, set_backend):
        """Both entry points must accept or refuse the same media.

        ``solve_angles`` has its own reason to want a real index — it maps
        an angle to ``kx = n(omega)*k0*sin(theta)`` — and used to test the
        imaginary part against exact zero.  A superstrate that is lossless
        to within rounding noise then solved through ``solve()`` and
        raised from ``solve_angles()``.
        """
        stack = _stack(superstrate_eps=2.25 * (1 + 1e-16j))

        by_angle = stratix.solve_angles(
            stack, _WL, [0.0, 30.0], polarization=Polarization.TE
        )
        by_kx = stratix.solve(stack, _WL, kx=0.0, polarization=Polarization.TE)

        assert float(by_angle.R[0, 0]) == pytest.approx(float(by_kx.R[0, 0]))

    def test_solve_angles_still_refuses_a_real_loss(self, set_backend):
        with pytest.raises(ValueError, match="transparent superstrate"):
            stratix.solve_angles(
                _stack(superstrate_eps=2.25 + 0.01j),
                _WL,
                [0.0, 30.0],
                polarization=Polarization.TE,
            )

    def test_a_purely_imaginary_permittivity_is_refused(self, set_backend):
        """Zero real part must not make the relative tolerance vacuous."""
        with pytest.raises(ValueError, match="lossless superstrate"):
            stratix.solve(
                _stack(superstrate_eps=1j), _WL, kx=0.0, polarization=Polarization.TE
            )


class TestEachPropertyIsDecidedOnItsOwn:
    """An unreadable ``epsilon`` must not excuse a lossy ``mu``.

    The check cannot read a tracer, and the loop originally answered that
    by returning — which meant the *second* property went unchecked
    whenever the first was unreadable, regardless of whether it could
    have been read.  Whether that combination arises depends on the
    backend, so it is pinned at the seam rather than through a solve:
    ``_imaginary_part_is_negligible`` is the one place that turns a value
    into a verdict, and ``None`` from it must mean "skip this one", never
    "stop looking".
    """

    def test_an_unreadable_epsilon_does_not_stop_the_scan(self, monkeypatch):
        from stratix.methods import _medium_params

        verdicts = iter([None, False])  # epsilon unreadable, mu lossy
        monkeypatch.setattr(
            _medium_params,
            "_imaginary_part_is_negligible",
            lambda value: next(verdicts),
        )

        with pytest.raises(ValueError, match="lossless superstrate"):
            _medium_params._reject_lossy_superstrate(
                Material(epsilon=2.25, mu=1.0), nd.array(1.0)
            )

    def test_an_unreadable_epsilon_alone_raises_nothing(self, monkeypatch):
        """Skipping is not the same as refusing: unknown must stay unknown."""
        from stratix.methods import _medium_params

        monkeypatch.setattr(
            _medium_params, "_imaginary_part_is_negligible", lambda value: None
        )

        _medium_params._reject_lossy_superstrate(
            Material(epsilon=2.25 + 1j), nd.array(1.0)
        )


class TestTracedSolvesStillRun:
    """The check reads a value, so it has to stand aside under ``nd.jit``.

    ``solve()`` is trace-safe (#50) and must stay that way.  Inside a
    trace numdiff hands back a tracer for every array it builds — a
    non-dispersive material's constant ``epsilon`` included — so no
    imaginary part is available to branch on and the check is skipped
    rather than breaking compilation.  A lossy superstrate can therefore
    reach a compiled solve and come back with the non-partitioning
    numbers; that gap is documented on
    ``_reject_lossy_superstrate``.  Backends differ in how far they
    actually trace, so what a compiled *lossy* solve does is deliberately
    not pinned here; what is pinned is that a lossless one keeps working.
    """

    def test_jit_still_compiles_and_matches_eager(self, set_backend):
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support jit")

        # Explicit opt-in: compiling solve() itself returns a Result whose
        # enum fields JAX accepts only once registered (issue #55).
        stratix.register_jax_pytrees()

        stack = _stack(superstrate_eps=1.0)
        wavelengths = nd.linspace(400e-9, 800e-9, 4)
        kx = nd.linspace(0.0, 1e7, 3)

        jit_solve = nd.jit(
            stratix.solve,
            static_argnames=(
                "stack",
                "polarization",
                "method",
                "absorption",
                "thicknesses",
            ),
        )
        eager = stratix.solve(
            stack, wavelengths, kx=kx,
            polarization=Polarization.TE, method=Method.SMATRIX,
        )
        compiled = jit_solve(
            stack, wavelengths, kx=kx,
            polarization=Polarization.TE, method=Method.SMATRIX,
        )
        assert float(nd.max(nd.abs(eager.R - compiled.R))) < 1e-12
        assert float(nd.max(nd.abs(eager.T - compiled.T))) < 1e-12

    def test_grad_still_flows(self, set_backend):
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")

        stack = _stack(superstrate_eps=1.0)

        def f(wl):
            return stratix.solve(
                stack, wl, kx=0.0, polarization=Polarization.TE
            ).R[0, 0]

        grad_ad = float(nd.grad(f)(_WL))
        h = 1e-12
        grad_fd = (float(f(_WL + h)) - float(f(_WL - h))) / (2 * h)
        assert abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-12) < 1e-4
