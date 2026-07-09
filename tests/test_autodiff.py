"""Tests for autodiff gradient verification (Issue #25)."""

from __future__ import annotations

import numdiff as nd
import pytest
from phokaia import Layer
from phokaia import Material
from phokaia import Stack

from stratix._types import Polarization


def _central_fd(f, x, h=1e-8):
    """Central finite difference derivative."""
    return (f(x + h) - f(x - h)) / (2 * h)


_GRAD_TORCH_REASON = (
    "numdiff torch backend does not preserve gradient trace "
    "through ``nd.array`` (Issue #25)"
)


@pytest.fixture(params=["numpy", "jax", "torch", "autograd"])
def set_backend(request):
    backend = request.param
    if not getattr(nd, f"HAS_{backend.upper()}", False):
        pytest.skip(f"Backend {backend!r} not available")
    old = nd.get_backend()
    nd.set_backend(backend)
    if backend == "torch":
        import torch

        torch.set_default_dtype(torch.float64)
    if backend == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
    yield backend
    nd.set_backend(old)


class TestGradWavelength:
    """Gradient of R with respect to wavelength."""

    def test_dR_dwavelength_single_layer(self, set_backend):
        """dR/dλ via nd.grad matches central finite difference."""
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")
        if nd.get_backend() == "torch":
            pytest.skip(_GRAD_TORCH_REASON)

        from stratix._autodiff import _solve_raw

        n_air, n_film, n_sub = 1.0, 1.38, 1.5
        wavelength_0 = 5e-7
        d_film = 100e-9  # off-resonance → non-zero gradient

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[Layer(thickness=d_film, material=Material(epsilon=n_film**2))],
        )

        def f(wl):
            R, _ = _solve_raw(stack, wl, kx=0.0, polarization=Polarization.TE)
            return R

        grad_ad = nd.grad(f)(wavelength_0)
        grad_fd = _central_fd(f, wavelength_0, h=1e-10)

        rel_err = abs(float(grad_ad - grad_fd)) / max(abs(float(grad_fd)), 1e-12)
        assert rel_err < 1e-4, f"dR/dλ: AD={grad_ad}, FD={grad_fd}, rel_err={rel_err}"


    def test_dR_dwavelength_conservation(self, set_backend):
        """d(R+T)/dλ = 0 for lossless dielectric at off-resonance λ."""
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")
        if nd.get_backend() == "torch":
            pytest.skip(_GRAD_TORCH_REASON)

        from stratix._autodiff import _solve_raw

        n_air, n_film, n_sub = 1.0, 1.38, 1.5
        wavelength_0 = 5e-7
        d_film = 100e-9

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[Layer(thickness=d_film, material=Material(epsilon=n_film**2))],
        )

        def f_R(wl):
            R, _ = _solve_raw(stack, wl, kx=0.0, polarization=Polarization.TE)
            return R

        def f_T(wl):
            _, T = _solve_raw(stack, wl, kx=0.0, polarization=Polarization.TE)
            return T

        grad_R = float(nd.grad(f_R)(wavelength_0))
        grad_T = float(nd.grad(f_T)(wavelength_0))
        assert abs(grad_R + grad_T) < 1e-8, (
            f"dR/dλ + dT/dλ = {grad_R + grad_T}"
        )


class TestGradThickness:
    """Gradient of R and T with respect to layer thickness."""

    def test_dR_dthickness_single_layer(self, set_backend):
        """dR/dt via nd.grad matches central finite difference."""
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")
        if nd.get_backend() == "torch":
            pytest.skip(_GRAD_TORCH_REASON)

        from stratix._autodiff import _solve_raw_with_thicknesses

        n_air, n_film, n_sub = 1.0, 1.38, 1.5
        wavelength = 5e-7
        d0 = 100e-9

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[Layer(thickness=d0, material=Material(epsilon=n_film**2))],
        )

        def f(t):
            thicknesses = nd.array([t])
            R, _ = _solve_raw_with_thicknesses(
                stack, thicknesses, wavelength, kx=0.0, polarization=Polarization.TE
            )
            return R

        grad_ad = nd.grad(f)(d0)
        grad_fd = _central_fd(f, d0, h=1e-12)

        rel_err = abs(float(grad_ad - grad_fd)) / max(abs(float(grad_fd)), 1e-12)
        assert rel_err < 1e-4, f"dR/dt: AD={grad_ad}, FD={grad_fd}, rel_err={rel_err}"

    def test_dT_dthickness_single_layer(self, set_backend):
        """dT/dt via nd.grad matches central finite difference."""
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")
        if nd.get_backend() == "torch":
            pytest.skip(_GRAD_TORCH_REASON)

        from stratix._autodiff import _solve_raw_with_thicknesses

        n_air, n_film, n_sub = 1.0, 1.38, 1.5
        wavelength = 5e-7
        d0 = 100e-9

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[Layer(thickness=d0, material=Material(epsilon=n_film**2))],
        )

        def f(t):
            thicknesses = nd.array([t])
            _, T = _solve_raw_with_thicknesses(
                stack, thicknesses, wavelength, kx=0.0, polarization=Polarization.TE
            )
            return T

        grad_ad = nd.grad(f)(d0)
        grad_fd = _central_fd(f, d0, h=1e-12)

        rel_err = abs(float(grad_ad - grad_fd)) / max(abs(float(grad_fd)), 1e-12)
        assert rel_err < 1e-4, f"dT/dt: AD={grad_ad}, FD={grad_fd}, rel_err={rel_err}"

    def test_dRdt_plus_dTdt_zero_lossless(self, set_backend):
        """dR/dt + dT/dt = 0 for lossless dielectrics (energy conservation)."""
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")
        if nd.get_backend() == "torch":
            pytest.skip(_GRAD_TORCH_REASON)

        from stratix._autodiff import _solve_raw_with_thicknesses

        n_air, n_film, n_sub = 1.0, 1.38, 1.5
        wavelength = 5e-7
        d0 = 100e-9

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[Layer(thickness=d0, material=Material(epsilon=n_film**2))],
        )

        def f_R(t):
            thicknesses = nd.array([t])
            R, _ = _solve_raw_with_thicknesses(
                stack, thicknesses, wavelength, kx=0.0, polarization=Polarization.TE
            )
            return R

        def f_T(t):
            thicknesses = nd.array([t])
            _, T = _solve_raw_with_thicknesses(
                stack, thicknesses, wavelength, kx=0.0, polarization=Polarization.TE
            )
            return T

        grad_R = float(nd.grad(f_R)(d0))
        grad_T = float(nd.grad(f_T)(d0))
        assert abs(grad_R + grad_T) < 1e-8, f"dR/dt + dT/dt = {grad_R + grad_T}"


class TestMultiLayerGrad:
    """Gradients through multi-layer stacks."""

    def test_two_layer_gradient_dR_dthickness(self, set_backend):
        """Layer-by-layer dR/dt matches FD for 2-layer stack."""
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")
        if nd.get_backend() == "torch":
            pytest.skip(_GRAD_TORCH_REASON)

        from stratix._autodiff import _solve_raw_with_thicknesses

        n_air, n_a, n_b, n_sub = 1.0, 1.38, 2.0, 1.5
        wavelength = 5e-7
        d_a, d_b = 100e-9, 200e-9

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[
                Layer(thickness=d_a, material=Material(epsilon=n_a**2)),
                Layer(thickness=d_b, material=Material(epsilon=n_b**2)),
            ],
        )

        for layer_idx, (orig_d, label) in enumerate(
            [(d_a, "layer_a"), (d_b, "layer_b")]
        ):
            def f(t, idx=layer_idx, od_a=d_a, od_b=d_b):
                ts = nd.array([t if idx == 0 else od_a, t if idx == 1 else od_b])
                R, _ = _solve_raw_with_thicknesses(
                    stack, ts, wavelength, kx=0.0, polarization=Polarization.TE
                )
                return R

            grad_ad = nd.grad(f)(orig_d)
            grad_fd = _central_fd(f, orig_d, h=1e-12)

            rel_err = abs(float(grad_ad - grad_fd)) / max(abs(float(grad_fd)), 1e-12)
            assert rel_err < 1e-4, (
                f"dR/d({label}): AD={grad_ad}, FD={grad_fd}, rel_err={rel_err}"
            )


class TestJit:
    """JIT compilation equivalency."""

    def test_jit_equivalency(self, set_backend):
        """nd.jit(f) and f produce same R."""
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")

        from stratix._autodiff import _solve_raw

        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[Layer(thickness=100e-9, material=Material(epsilon=1.38**2))],
        )

        def f(wl):
            R, _ = _solve_raw(stack, wl, kx=0.0, polarization=Polarization.TE)
            return R

        f_jit = nd.jit(f)

        wavelength = 5e-7
        result_normal = float(f(wavelength))
        result_jit = float(f_jit(wavelength))

        assert abs(result_normal - result_jit) < 1e-12, (
            f"JIT mismatch: {result_normal} vs {result_jit}"
        )
