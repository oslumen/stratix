"""Output shape contract for :func:`stratix.solve` (Issue #50).

``R`` and ``T`` are always ``(Nlambda, Nk)``; scalar wavelength or kx counts
as a length-1 axis.  ``Polarization.BOTH`` prepends a TE/TM axis of size 2.
``energy_balance`` follows the same rule and ``layer_absorption`` inserts the
layer axis directly in front of the sweep axes.
"""

from __future__ import annotations

import numdiff as nd
import pytest
from phokaia import Layer
from phokaia import Material
from phokaia import Polarization
from phokaia import Stack

import stratix
from stratix import Method

ALL_METHODS = [Method.SMATRIX, Method.ABELES, Method.ADMITTANCE, Method.DTN]


@pytest.fixture
def stack() -> Stack:
    """Two-layer lossy stack: enough layers to exercise the layer axis."""
    return Stack(
        superstrate=Material(epsilon=1.0),
        substrate=Material(epsilon=2.25),
        layers=[
            Layer(thickness=100e-9, material=Material(epsilon=complex(1.9, 0.05))),
            Layer(thickness=180e-9, material=Material(epsilon=4.0)),
        ],
    )


WL_SCALAR = 5e-7
KX_SCALAR = 1e6


def _wl(n: int):
    return WL_SCALAR if n == 1 else nd.linspace(4e-7, 8e-7, n)


def _kx(n: int):
    return KX_SCALAR if n == 1 else nd.linspace(0.0, 2e6, n)


class TestRTShape:
    """R and T are (Nlambda, Nk) for every scalar/array input combination."""

    @pytest.mark.parametrize("n_wl", [1, 3])
    @pytest.mark.parametrize("n_kx", [1, 4])
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_rt_shape(self, set_backend, stack, n_wl, n_kx, method):
        """R/T carry both sweep axes even when an input is scalar."""
        result = stratix.solve(
            stack,
            _wl(n_wl),
            kx=_kx(n_kx),
            polarization=Polarization.TE,
            method=method,
        )
        assert result.R.shape == (n_wl, n_kx)
        assert result.T.shape == (n_wl, n_kx)

    @pytest.mark.parametrize("n_wl", [1, 3])
    @pytest.mark.parametrize("n_kx", [1, 4])
    def test_rt_shape_both_polarization(self, set_backend, stack, n_wl, n_kx):
        """BOTH prepends a TE/TM axis of size 2 in front of the sweep axes."""
        result = stratix.solve(
            stack, _wl(n_wl), kx=_kx(n_kx), polarization=Polarization.BOTH
        )
        assert result.R.shape == (2, n_wl, n_kx)
        assert result.T.shape == (2, n_wl, n_kx)

    @pytest.mark.parametrize("n_wl", [1, 3])
    @pytest.mark.parametrize("n_kx", [1, 4])
    def test_coordinate_shapes(self, set_backend, stack, n_wl, n_kx):
        """``wavelengths`` and ``kx`` stay 1-D, one entry per sweep point."""
        result = stratix.solve(
            stack, _wl(n_wl), kx=_kx(n_kx), polarization=Polarization.TE
        )
        assert result.wavelengths.shape == (n_wl,)
        assert result.kx.shape == (n_kx,)


class TestAbsorptionShape:
    """layer_absorption and energy_balance follow the same contract."""

    @pytest.mark.parametrize("n_wl", [1, 3])
    @pytest.mark.parametrize("n_kx", [1, 4])
    def test_absorption_shapes(self, set_backend, stack, n_wl, n_kx):
        """Layer axis sits directly in front of the (Nlambda, Nk) sweep axes."""
        result = stratix.solve(
            stack,
            _wl(n_wl),
            kx=_kx(n_kx),
            polarization=Polarization.TE,
            absorption=True,
        )
        assert result.layer_absorption.shape == (len(stack.layers), n_wl, n_kx)
        assert result.energy_balance.shape == (n_wl, n_kx)

    @pytest.mark.parametrize("n_wl", [1, 3])
    @pytest.mark.parametrize("n_kx", [1, 4])
    def test_absorption_shapes_both(self, set_backend, stack, n_wl, n_kx):
        """BOTH prepends the polarization axis to both absorption fields."""
        result = stratix.solve(
            stack,
            _wl(n_wl),
            kx=_kx(n_kx),
            polarization=Polarization.BOTH,
            absorption=True,
        )
        assert result.layer_absorption.shape == (2, len(stack.layers), n_wl, n_kx)
        assert result.energy_balance.shape == (2, n_wl, n_kx)

    def test_bare_interface_has_empty_layer_axis(self, set_backend):
        """A stack with no layers keeps a zero-length layer axis, not a crash."""
        bare = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[],
        )
        result = stratix.solve(
            bare,
            nd.linspace(4e-7, 8e-7, 3),
            kx=0.0,
            polarization=Polarization.TE,
            absorption=True,
        )
        assert result.layer_absorption.shape == (0, 3, 1)
        assert result.energy_balance.shape == (3, 1)


class TestValuesUnchanged:
    """Reshaping must not change the numbers."""

    def test_scalar_matches_sweep_entry(self, set_backend, stack):
        """A scalar solve equals the matching entry of a swept solve."""
        wavelengths = nd.linspace(4e-7, 8e-7, 3)
        swept = stratix.solve(
            stack, wavelengths, kx=KX_SCALAR, polarization=Polarization.TE
        )
        for i in range(3):
            single = stratix.solve(
                stack,
                float(wavelengths[i]),
                kx=KX_SCALAR,
                polarization=Polarization.TE,
            )
            assert abs(float(swept.R[i, 0]) - float(single.R[0, 0])) < 1e-12
            assert abs(float(swept.T[i, 0]) - float(single.T[0, 0])) < 1e-12

    def test_both_matches_separate_polarizations(self, set_backend, stack):
        """The TE/TM axis holds exactly the two single-polarization solves."""
        wavelengths = nd.linspace(4e-7, 8e-7, 3)
        both = stratix.solve(
            stack, wavelengths, kx=KX_SCALAR, polarization=Polarization.BOTH
        )
        te = stratix.solve(
            stack, wavelengths, kx=KX_SCALAR, polarization=Polarization.TE
        )
        tm = stratix.solve(
            stack, wavelengths, kx=KX_SCALAR, polarization=Polarization.TM
        )
        assert float(nd.max(nd.abs(both.R[0] - te.R))) < 1e-14
        assert float(nd.max(nd.abs(both.R[1] - tm.R))) < 1e-14


class TestTraceSafe:
    """solve() is differentiable end to end — no float casts, no list wrapping."""

    def test_grad_through_public_solve(self, set_backend, stack):
        """nd.grad of R w.r.t. a thickness override works through solve()."""
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")

        def f(t):
            result = stratix.solve(
                stack,
                WL_SCALAR,
                kx=0.0,
                polarization=Polarization.TE,
                thicknesses=nd.array([t, 180e-9]),
            )
            return result.R[0, 0]

        d0 = 100e-9
        grad_ad = float(nd.grad(f)(d0))
        h = 1e-12
        grad_fd = float((f(d0 + h) - f(d0 - h)) / (2 * h))
        rel_err = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-12)
        assert rel_err < 1e-4, f"AD={grad_ad}, FD={grad_fd}, rel_err={rel_err}"

    def test_grad_through_swept_solve(self, set_backend, stack):
        """Gradient of a sweep-reduced scalar flows through the (Nl, Nk) grid."""
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")

        wavelengths = nd.linspace(4e-7, 8e-7, 4)

        def f(t):
            result = stratix.solve(
                stack,
                wavelengths,
                kx=0.0,
                polarization=Polarization.TE,
                thicknesses=nd.array([t, 180e-9]),
            )
            return nd.sum(result.R)

        d0 = 100e-9
        grad_ad = float(nd.grad(f)(d0))
        h = 1e-12
        grad_fd = float((f(d0 + h) - f(d0 - h)) / (2 * h))
        rel_err = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-12)
        assert rel_err < 1e-4, f"AD={grad_ad}, FD={grad_fd}, rel_err={rel_err}"
