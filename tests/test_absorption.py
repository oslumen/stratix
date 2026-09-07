"""Tests for absorption computation and energy balance — Issue #27."""

from __future__ import annotations

import numdiff as nd
from phokaia import Layer
from phokaia import Material
from phokaia import Polarization
from phokaia import Stack

import stratix
from stratix._types import Method


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


def _absorber_then_bragg(n_pairs: int = 6):
    """Lossy top layer in front of a lossless quarter-wave Bragg mirror."""
    wavelength = 5e-7
    n_lossy = 1.5 + 0.3j
    n_low, n_high = 1.38, 2.3
    layers = [
        Layer(thickness=40e-9, material=Material(epsilon=n_lossy**2)),
    ]
    for _ in range(n_pairs):
        layers.append(
            Layer(
                thickness=wavelength / (4 * n_high),
                material=Material(epsilon=n_high**2),
            )
        )
        layers.append(
            Layer(
                thickness=wavelength / (4 * n_low),
                material=Material(epsilon=n_low**2),
            )
        )
    stack = Stack(
        superstrate=Material(epsilon=1.0),
        substrate=Material(epsilon=1.5**2),
        layers=layers,
    )
    return stack, wavelength


class TestAbsorptionAttribution:
    """Per-layer absorption comes from Poynting flux — Issue #48."""

    def test_absorber_before_lossless_mirror(self, set_backend):
        """All absorption sits in the lossy layer, none in the Bragg mirror."""
        stack, wavelength = _absorber_then_bragg()
        result = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE,
            absorption=True,
        )
        A = [float(a) for a in result.layer_absorption]
        total = 1.0 - float(result.R[0]) - float(result.T[0])

        assert len(A) == len(stack.layers)
        assert abs(A[0] - total) < 1e-10, (
            f"absorber should carry all absorption: {A[0]} vs {total}"
        )
        for i, a in enumerate(A[1:], start=1):
            assert abs(a) < 1e-12, f"lossless layer {i} absorbs {a}"

    def test_absorber_before_lossless_mirror_tm(self, set_backend):
        stack, wavelength = _absorber_then_bragg()
        k0 = 2 * nd.pi / wavelength
        kx = float(k0 * nd.sin(nd.array(35.0 * nd.pi / 180)))
        result = stratix.solve(
            stack, wavelength, kx=kx, polarization=Polarization.TM,
            absorption=True,
        )
        A = [float(a) for a in result.layer_absorption]
        total = 1.0 - float(result.R[0]) - float(result.T[0])

        assert abs(A[0] - total) < 1e-10
        for a in A[1:]:
            assert abs(a) < 1e-12

    def test_sum_equals_one_minus_r_minus_t(self, set_backend):
        """Two separated absorbers: per-layer sum recovers total absorption."""
        n_lossy_a = 1.5 + 0.2j
        n_lossy_b = 0.5 + 2.0j
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[
                Layer(thickness=60e-9, material=Material(epsilon=n_lossy_a**2)),
                Layer(thickness=90e-9, material=Material(epsilon=1.38**2)),
                Layer(thickness=20e-9, material=Material(epsilon=n_lossy_b**2)),
            ],
        )
        result = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE, absorption=True,
        )
        A = [float(a) for a in result.layer_absorption]
        total = 1.0 - float(result.R[0]) - float(result.T[0])

        assert abs(sum(A) - total) < 1e-12
        assert A[0] > 0
        assert abs(A[1]) < 1e-12
        assert A[2] > 0

    def test_energy_balance_reported(self, set_backend):
        """R + T + Σ(layer absorption) is 1 for a deep lossy stack.

        Flux is continuous across every interface, so the per-layer terms
        telescope and this identity holds analytically; it guards the
        arithmetic, not the physics.  The physical content of the per-layer
        split is tested by :meth:`test_matches_ohmic_dissipation` below and
        against the ``tmm`` reference in ``test_tmm_integration.py``.
        """
        stack, wavelength = _absorber_then_bragg(n_pairs=8)
        for pol in (Polarization.TE, Polarization.TM):
            result = stratix.solve(
                stack, wavelength, kx=3e6, polarization=pol, absorption=True,
            )
            assert abs(float(result.energy_balance) - 1.0) < 1e-10

    def test_matches_ohmic_dissipation(self, set_backend):
        """Flux drop across a layer equals the ohmic loss integrated in it.

        For TE the power dissipated in a layer is
        ``½ ω ε0 Im(ε) ∫|E|² dz``; dividing by the incident flux
        ``Re(kz0/μ0_r) / (2 ω μ0)`` turns the constants into ``k0²``:

            A = k0² Im(ε) ∫|E|² dz / Re(kz0 / denom0)

        Integrating the field profile is an independent route to the same
        number as the Poynting-flux difference, so agreement tests the
        flux formula itself rather than a telescoping identity.
        """
        wavelength = 5e-7
        n_a = 1.5 + 0.25j
        n_b = 0.8 + 1.2j
        d_a, d_b = 70e-9, 45e-9
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[
                Layer(thickness=d_a, material=Material(epsilon=n_a**2)),
                Layer(thickness=90e-9, material=Material(epsilon=1.38**2)),
                Layer(thickness=d_b, material=Material(epsilon=n_b**2)),
            ],
        )
        result = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE,
            absorption=True,
        )

        k0 = float(2 * nd.pi / wavelength)
        intr = result.intermediates
        incident = float(nd.real(intr["kzs"][0] / intr["denom_vals"][0]))

        n_samples = 20001
        starts = [0.0, d_a, d_a + 90e-9]
        widths = [d_a, 90e-9, d_b]
        for layer, (start, width) in enumerate(zip(starts, widths, strict=True)):
            z = nd.array(
                [start + width * i / (n_samples - 1) for i in range(n_samples)]
            )
            field = stratix.compute_field_profile(result, z)
            e2 = [float(abs(value)) ** 2 for value in field["E"]]
            step = width / (n_samples - 1)
            integral = step * (sum(e2) - 0.5 * (e2[0] + e2[-1]))

            eps_layer = complex(stack.layers[layer].material.epsilon())
            expected = k0**2 * eps_layer.imag * integral / incident

            assert abs(float(result.layer_absorption[layer]) - expected) < 1e-8, (
                f"layer {layer}: flux drop "
                f"{float(result.layer_absorption[layer])} vs "
                f"dissipation integral {expected}"
            )

    def test_energy_balance_lossless_bragg(self, set_backend):
        n_low, n_high = 1.38, 2.3
        wavelength = 5e-7
        layers = []
        for _ in range(6):
            layers.append(
                Layer(
                    thickness=wavelength / (4 * n_high),
                    material=Material(epsilon=n_high**2),
                )
            )
            layers.append(
                Layer(
                    thickness=wavelength / (4 * n_low),
                    material=Material(epsilon=n_low**2),
                )
            )
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=layers,
        )
        result = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE,
            absorption=True,
        )
        assert abs(float(result.energy_balance) - 1.0) < 1e-10
        for a in result.layer_absorption:
            assert abs(float(a)) < 1e-12


def _lossy_bilayer():
    n_lossy = 0.5 + 1j
    return Stack(
        superstrate=Material(epsilon=1.0),
        substrate=Material(epsilon=2.25),
        layers=[
            Layer(thickness=80e-9, material=Material(epsilon=1.38**2)),
            Layer(thickness=40e-9, material=Material(epsilon=n_lossy**2)),
        ],
    )


class TestAbsorptionSweeps:
    """absorption=True across array inputs and BOTH — Issue #49."""

    def test_wavelength_array(self, set_backend):
        stack = _lossy_bilayer()
        wavelengths = nd.array([4e-7, 5e-7, 6e-7, 7e-7])
        result = stratix.solve(
            stack, wavelengths, kx=0.0, polarization=Polarization.TE,
            absorption=True,
        )

        assert result.layer_absorption is not None
        assert result.layer_absorption.shape == (2, 4)
        assert result.energy_balance.shape == result.R.shape == (4,)
        for i in range(4):
            assert abs(float(result.energy_balance[i]) - 1.0) < 1e-10

    def test_wavelength_array_matches_scalar(self, set_backend):
        stack = _lossy_bilayer()
        wl_list = [4e-7, 5e-7, 6e-7]
        swept = stratix.solve(
            stack, nd.array(wl_list), kx=0.0, polarization=Polarization.TE,
            absorption=True,
        )
        for i, wl in enumerate(wl_list):
            single = stratix.solve(
                stack, wl, kx=0.0, polarization=Polarization.TE, absorption=True,
            )
            for layer in range(2):
                assert abs(
                    float(swept.layer_absorption[layer][i])
                    - float(single.layer_absorption[layer])
                ) < 1e-12

    def test_kx_array(self, set_backend):
        stack = _lossy_bilayer()
        kx = nd.array([0.0, 2e6, 5e6])
        result = stratix.solve(
            stack, 5e-7, kx=kx, polarization=Polarization.TE, absorption=True,
        )
        assert result.layer_absorption.shape == (2, 3)
        assert result.energy_balance.shape == (3,)

    def test_two_dimensional_sweep(self, set_backend):
        stack = _lossy_bilayer()
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        kx = nd.array([0.0, 2e6])
        result = stratix.solve(
            stack, wavelengths, kx=kx, polarization=Polarization.TE,
            absorption=True,
        )
        assert result.R.shape == (3, 2)
        assert result.layer_absorption.shape == (2, 3, 2)
        assert result.energy_balance.shape == (3, 2)
        for i in range(3):
            for j in range(2):
                assert abs(float(result.energy_balance[i][j]) - 1.0) < 1e-10

    def test_sum_matches_one_minus_r_minus_t_on_grid(self, set_backend):
        stack = _lossy_bilayer()
        wavelengths = nd.array([4.5e-7, 6.5e-7])
        kx = nd.array([0.0, 3e6])
        result = stratix.solve(
            stack, wavelengths, kx=kx, polarization=Polarization.TE,
            absorption=True,
        )
        for i in range(2):
            for j in range(2):
                total = 1.0 - float(result.R[i][j]) - float(result.T[i][j])
                summed = sum(
                    float(result.layer_absorption[layer][i][j])
                    for layer in range(2)
                )
                assert abs(summed - total) < 1e-12


class TestAbsorptionBoth:
    """BOTH polarization carries a leading TE/TM axis — Issue #49."""

    def test_scalar_shapes(self, set_backend):
        stack = _lossy_bilayer()
        result = stratix.solve(
            stack, 5e-7, kx=3e6, polarization=Polarization.BOTH,
            absorption=True,
        )
        assert result.layer_absorption.shape == (2, 2)
        assert result.energy_balance.shape == (2,)

    def test_sweep_shapes(self, set_backend):
        stack = _lossy_bilayer()
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        result = stratix.solve(
            stack, wavelengths, kx=3e6, polarization=Polarization.BOTH,
            absorption=True,
        )
        assert result.R.shape == (2, 3)
        assert result.layer_absorption.shape == (2, 2, 3)
        assert result.energy_balance.shape == (2, 3)

    def test_te_tm_slices_match_single_polarization(self, set_backend):
        stack = _lossy_bilayer()
        wavelengths = nd.array([4e-7, 6e-7])
        both = stratix.solve(
            stack, wavelengths, kx=3e6, polarization=Polarization.BOTH,
            absorption=True,
        )
        for axis, pol in enumerate((Polarization.TE, Polarization.TM)):
            single = stratix.solve(
                stack, wavelengths, kx=3e6, polarization=pol, absorption=True,
            )
            for layer in range(2):
                for i in range(2):
                    assert abs(
                        float(both.layer_absorption[axis][layer][i])
                        - float(single.layer_absorption[layer][i])
                    ) < 1e-12

    def test_tm_absorption_against_one_minus_r_minus_t(self, set_backend):
        stack = _lossy_bilayer()
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        result = stratix.solve(
            stack, wavelengths, kx=4e6, polarization=Polarization.BOTH,
            absorption=True,
        )
        tm = 1
        for i in range(3):
            total = 1.0 - float(result.R[tm][i]) - float(result.T[tm][i])
            summed = sum(
                float(result.layer_absorption[tm][layer][i])
                for layer in range(2)
            )
            assert abs(summed - total) < 1e-12


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

        # The lumped term is 1 - R - T by construction, so the balance alone
        # proves nothing; check it against the S-matrix per-layer total.
        reference = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE,
            method=Method.SMATRIX, absorption=True,
        )
        expected = sum(float(a) for a in reference.layer_absorption)
        assert len(result.layer_absorption) == 1
        assert abs(float(result.layer_absorption[0]) - expected) < 1e-12
        assert expected > 0

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

        reference = stratix.solve(
            stack, 5e-7, kx=0.0, polarization=Polarization.TE,
            method=Method.SMATRIX, absorption=True,
        )
        expected = sum(float(a) for a in reference.layer_absorption)
        assert len(result.layer_absorption) == 1
        assert abs(float(result.layer_absorption[0]) - expected) < 1e-12
        assert expected > 0
