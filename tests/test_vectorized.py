"""Tests for vectorized sweeps: Nlambda x Nk array inputs — Issue #20."""

from __future__ import annotations

import numdiff as nd
import pytest
from phokaia import Layer
from phokaia import Material
from phokaia import Stack

import stratix
from stratix._types import Method
from stratix._types import Polarization


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


class TestScalarRegression:
    """Scalar inputs still produce (1,) shaped results."""

    def test_scalar_returns_1d(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve(stack, 5e-7, kx=0.0, polarization=Polarization.TE)
        assert result.R.shape == (1,)
        assert result.T.shape == (1,)
        assert result.wavelengths.shape == (1,)
        assert result.kx.shape == (1,)

    def test_scalar_R_T_correct(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        expected_R = ((n_air - n_glass) / (n_air + n_glass)) ** 2
        result = stratix.solve(stack, 5e-7, kx=0.0, polarization=Polarization.TE)
        assert abs(float(result.R[0]) - expected_R) < 1e-12
        assert abs(float(result.R[0]) + float(result.T[0]) - 1.0) < 1e-12


class TestWavelengthArray:
    """1D wavelength array, scalar kx."""

    def test_shape(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        result = stratix.solve(
            stack, wavelengths, kx=0.0, polarization=Polarization.TE
        )
        assert result.R.shape == (3,)
        assert result.T.shape == (3,)
        assert result.wavelengths.shape == (3,)
        assert result.kx.shape == (1,)

    def test_energy_conservation(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelengths = nd.array([4e-7, 5e-7, 6e-7, 7e-7])
        result = stratix.solve(
            stack, wavelengths, kx=0.0, polarization=Polarization.TE
        )
        for i in range(len(wavelengths)):
            R = float(result.R[i])
            T = float(result.T[i])
            assert abs(R + T - 1.0) < 1e-12

    def test_equivalent_to_scalar_loop(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        result = stratix.solve(
            stack, wavelengths, kx=0.0, polarization=Polarization.TE
        )
        for i, wl in enumerate(float(w) for w in wavelengths):
            single = stratix.solve(
                stack, wl, kx=0.0, polarization=Polarization.TE
            )
            assert abs(float(result.R[i]) - float(single.R[0])) < 1e-12
            assert abs(float(result.T[i]) - float(single.T[0])) < 1e-12


class TestKxArray:
    """Scalar wavelength, 1D kx array."""

    def test_shape(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        kx_vals = nd.array([0.0, 5e6, 1e7])
        result = stratix.solve(
            stack, 5e-7, kx=kx_vals, polarization=Polarization.TE
        )
        assert result.R.shape == (3,)
        assert result.T.shape == (3,)
        assert result.wavelengths.shape == (1,)
        assert result.kx.shape == (3,)

    def test_energy_conservation(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        kx_vals = nd.array([0.0, 5e6, 1e7])
        result = stratix.solve(
            stack, 5e-7, kx=kx_vals, polarization=Polarization.TE
        )
        for i in range(len(kx_vals)):
            assert abs(float(result.R[i]) + float(result.T[i]) - 1.0) < 1e-12

    def test_equivalent_to_scalar_loop(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        kx_vals = nd.array([0.0, 5e6, 1e7])
        result = stratix.solve(
            stack, 5e-7, kx=kx_vals, polarization=Polarization.TE
        )
        for i, kv in enumerate(float(k) for k in kx_vals):
            single = stratix.solve(
                stack, 5e-7, kx=kv, polarization=Polarization.TE
            )
            assert abs(float(result.R[i]) - float(single.R[0])) < 1e-12


class TestWavelengthKxGrid:
    """Nlambda wavelength array x Nk kx array -> (Nlambda, Nk) grid."""

    def test_shape(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        kx_vals = nd.array([0.0, 5e6, 1e7, 1.5e7])
        result = stratix.solve(
            stack, wavelengths, kx=kx_vals, polarization=Polarization.TE
        )
        assert result.R.shape == (3, 4)
        assert result.T.shape == (3, 4)
        assert result.wavelengths.shape == (3,)
        assert result.kx.shape == (4,)

    def test_energy_conservation_grid(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        kx_vals = nd.array([0.0, 5e6, 1e7])
        result = stratix.solve(
            stack, wavelengths, kx=kx_vals, polarization=Polarization.TE
        )
        for i in range(len(wavelengths)):
            for j in range(len(kx_vals)):
                assert abs(float(result.R[i, j]) + float(result.T[i, j]) - 1.0) < 1e-12


class TestMultiLayerVectorized:
    """Vectorized solves with multi-layer stacks."""

    def test_ar_coating_wavelength_sweep(self, set_backend):
        n_sub = 1.5
        n_coat = float(nd.sqrt(nd.array(n_sub)))
        wavelength_design = 5e-7
        d = wavelength_design / (4 * n_coat)

        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=n_sub**2),
            layers=[Layer(thickness=d, material=Material(epsilon=n_coat**2))],
        )
        wavelengths = nd.array([4.5e-7, 5e-7, 5.5e-7])
        result = stratix.solve(
            stack, wavelengths, kx=0.0, polarization=Polarization.TE
        )
        assert result.R.shape == (3,)
        assert float(result.R[1]) < 1e-12  # zero reflection at design wavelength


class TestMethodWithArrays:
    """Vectorized solves with non-default methods."""

    def test_abeles_array(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        result = stratix.solve(
            stack, wavelengths, kx=0.0, polarization=Polarization.TE,
            method=Method.ABELES,
        )
        assert result.R.shape == (3,)
        assert result.method_used == Method.ABELES

    def test_admittance_array(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        result = stratix.solve(
            stack, wavelengths, kx=0.0, polarization=Polarization.TE,
            method=Method.ADMITTANCE,
        )
        assert result.R.shape == (3,)

    def test_dtn_array(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        result = stratix.solve(
            stack, wavelengths, kx=0.0, polarization=Polarization.TE,
            method=Method.DTN,
        )
        assert result.R.shape == (3,)
