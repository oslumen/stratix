"""Regression tests for evanescent incidence (kx > k0) — Issue #XX."""

from __future__ import annotations

import numdiff as nd
import pytest
from phokaia import Layer
from phokaia import Material
from phokaia import Stack

import stratix
from stratix._types import Method
from phokaia import Polarization


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


class TestEvanescentIncidence:
    """Evanescent incidence (kx > k0*n_superstrate): R=1, T=0, no warnings."""

    def _stack(self):
        return Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )

    def _multilayer_stack(self):
        return Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[Layer(thickness=100e-9, material=Material(epsilon=4.0))],
        )

    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    @pytest.mark.parametrize("wl", [4e-7, 5e-7, 6e-7])
    def test_evanescent_T_zero_scalar(self, set_backend, pol, wl):
        k0 = 2 * nd.pi / wl
        kx = k0 * 2.0  # well beyond free-space k0
        stack = self._stack()
        result = stratix.solve(stack, wl, kx=kx, polarization=pol)
        assert float(result.R[0]) == pytest.approx(1.0, abs=1e-12)
        assert float(result.T[0]) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_evanescent_T_zero_vectorized(self, set_backend, pol):
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        kx_vals = nd.array([0.0, 5e6, 1e7, 1.5e7])
        stack = self._stack()
        result = stratix.solve(stack, wavelengths, kx=kx_vals, polarization=pol)
        assert not nd.any(nd.isnan(result.T))
        assert not nd.any(nd.isinf(result.T))
        for i in range(len(wavelengths)):
            for j in range(len(kx_vals)):
                assert float(result.R[i, j]) + float(result.T[i, j]) == pytest.approx(
                    1.0, abs=1e-12
                )

    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_evanescent_multilayer(self, set_backend, pol):
        wl = 5e-7
        k0 = 2 * nd.pi / wl
        kx = k0 * 2.0
        stack = self._multilayer_stack()
        result = stratix.solve(stack, wl, kx=kx, polarization=pol)
        assert float(result.R[0]) == pytest.approx(1.0, abs=1e-12)
        assert float(result.T[0]) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("method", [Method.ABELES, Method.ADMITTANCE, Method.DTN])
    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_evanescent_other_methods(self, set_backend, method, pol):
        wl = 5e-7
        k0 = 2 * nd.pi / wl
        kx = k0 * 2.0
        stack = self._stack()
        result = stratix.solve(stack, wl, kx=kx, polarization=pol, method=method)
        assert float(result.R[0]) + float(result.T[0]) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_no_divide_by_zero_warning(self, set_backend, pol):
        wl = 5e-7
        k0 = 2 * nd.pi / wl
        kx = k0 * 2.0
        stack = self._stack()
        import warnings
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            stratix.solve(stack, wl, kx=kx, polarization=pol)
        divide_warnings = [
            w for w in record
            if issubclass(w.category, RuntimeWarning)
            and "divide by zero" in str(w.message).lower()
        ]
        assert len(divide_warnings) == 0, f"Unexpected divide-by-zero: {divide_warnings}"
