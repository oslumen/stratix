"""Tests for absorption computation and energy balance — Issue #27."""

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


class TestAbsorptionLossless:
    """Absorption = 0 for all-lossless dielectric stacks."""

    def test_single_interface_zero_absorption(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE, absorption=True,
        )
        assert result.layer_absorption is not None
        assert len(result.layer_absorption) == 0
        assert result.energy_balance is not None
        assert abs(float(result.energy_balance) - 1.0) < 1e-12

    def test_multi_layer_zero_absorption(self, set_backend):
        n_sub = 1.5
        n_coat = float(nd.sqrt(nd.array(n_sub)))
        d = 5e-7 / (4 * n_coat)
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=n_sub**2),
            layers=[Layer(thickness=d, material=Material(epsilon=n_coat**2))],
        )
        result = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE, absorption=True,
        )
        assert result.layer_absorption is not None
        assert len(result.layer_absorption) == 1
        assert abs(float(result.layer_absorption[0])) < 1e-12
        assert abs(float(result.energy_balance) - 1.0) < 1e-12

    def test_bragg_mirror_zero_absorption(self, set_backend):
        n_low, n_high = 1.38, 2.3
        d_low = 5e-7 / (4 * n_low)
        d_high = 5e-7 / (4 * n_high)
        layers = []
        for _ in range(5):
            layers.append(Layer(thickness=d_high, material=Material(epsilon=n_high**2)))
            layers.append(Layer(thickness=d_low, material=Material(epsilon=n_low**2)))
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=1.5**2),
            layers=layers,
        )
        result = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE, absorption=True,
        )
        assert result.layer_absorption is not None
        assert len(result.layer_absorption) == 10
        for a in result.layer_absorption:
            assert abs(float(a)) < 1e-12
        assert abs(float(result.energy_balance) - 1.0) < 1e-12


class TestAbsorptionLossy:
    """Absorption in stacks with complex-permittivity layers."""

    def test_lossy_layer_has_absorption(self, set_backend):
        n_air = 1.0
        n_lossy = 0.5 + 1j
        eps_lossy = n_lossy**2
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=2.25),
            layers=[
                Layer(thickness=100e-9, material=Material(epsilon=eps_lossy)),
            ],
        )
        result = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE, absorption=True,
        )
        assert result.layer_absorption is not None
        assert len(result.layer_absorption) == 1
        assert float(result.layer_absorption[0]) > 0
        assert float(result.layer_absorption[0]) < 1.0

    def test_energy_balance_lossy(self, set_backend):
        n_lossy = 0.5 + 1j
        eps_lossy = n_lossy**2
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[
                Layer(thickness=50e-9, material=Material(epsilon=eps_lossy)),
            ],
        )
        result = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE, absorption=True,
        )
        balance = float(result.R[0]) + float(result.T[0]) + float(sum(
            float(a) for a in result.layer_absorption
        ))
        assert abs(balance - 1.0) < 1e-12

    def test_energy_balance_equals_field(self, set_backend):
        n_lossy = 0.5 + 1j
        eps_lossy = n_lossy**2
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[
                Layer(thickness=50e-9, material=Material(epsilon=eps_lossy)),
            ],
        )
        result = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE, absorption=True,
        )
        assert abs(float(result.energy_balance) - 1.0) < 1e-12

    def test_mixed_lossy_lossless(self, set_backend):
        n_lossy = 0.5 + 1j
        eps_lossy = n_lossy**2
        n_diel = 1.5
        d_diel = 5e-7 / (4 * n_diel)
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[
                Layer(thickness=d_diel, material=Material(epsilon=n_diel**2)),
                Layer(thickness=50e-9, material=Material(epsilon=eps_lossy)),
            ],
        )
        result = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE, absorption=True,
        )
        assert result.layer_absorption is not None
        assert len(result.layer_absorption) == 2
        assert abs(float(result.layer_absorption[0])) < 1e-12
        assert float(result.layer_absorption[1]) > 0
        assert abs(float(result.energy_balance) - 1.0) < 1e-12


class TestAbsorptionOff:
    """absorption=False skips per-layer computation."""

    def test_absorption_false_returns_none(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE, absorption=False,
        )
        assert result.layer_absorption is None
        assert result.energy_balance is None

    def test_absorption_default_false(self, set_backend):
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE,
        )
        assert result.layer_absorption is None


class TestAbsorptionTM:
    """Absorption works for TM polarization."""

    def test_tm_lossy_layer(self, set_backend):
        n_lossy = 0.5 + 1.5j
        eps_lossy = n_lossy**2
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[
                Layer(thickness=30e-9, material=Material(epsilon=eps_lossy)),
            ],
        )
        result = stratix.solve(
            stack, 5e-7, kx=5e6, polarization=Polarization.TM, absorption=True,
        )
        assert result.layer_absorption is not None
        assert len(result.layer_absorption) == 1
        assert float(result.layer_absorption[0]) > 0
        balance = float(result.R[0]) + float(result.T[0]) + float(
            result.layer_absorption[0]
        )
        assert abs(balance - 1.0) < 1e-12


class TestAbsorptionWithMethods:
    """Absorption via non-default solver methods (total only, no per-layer)."""

    def test_admittance_absorption(self, set_backend):
        n_lossy = 0.5 + 1j
        eps_lossy = n_lossy**2
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[
                Layer(thickness=50e-9, material=Material(epsilon=eps_lossy)),
            ],
        )
        result = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE,
            method=Method.ADMITTANCE, absorption=True,
        )
        assert result.energy_balance is not None
        assert abs(float(result.energy_balance) - 1.0) < 1e-12

    def test_dtn_absorption(self, set_backend):
        n_lossy = 0.5 + 1j
        eps_lossy = n_lossy**2
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[
                Layer(thickness=50e-9, material=Material(epsilon=eps_lossy)),
            ],
        )
        result = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE,
            method=Method.DTN, absorption=True,
        )
        assert result.energy_balance is not None
        assert abs(float(result.energy_balance) - 1.0) < 1e-12
