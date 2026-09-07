"""Tests for compute_field_profile (Issue #24)."""

from __future__ import annotations

import numdiff as nd
from phokaia import Layer
from phokaia import Material
from phokaia import Polarization
from phokaia import Stack

import stratix


class TestFieldSingleInterfaceTE:
    def test_e_continuous_at_interface(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        result = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        eps = 1e-20
        z = nd.array([-eps, eps])
        field = stratix.compute_field_profile(result, z)

        E_left = field["E"][0, 0][0]
        E_right = field["E"][0, 0][1]
        assert abs(E_left - E_right) < 1e-8

    def test_h_continuous_at_interface(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        result = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        eps = 1e-20
        z = nd.array([-eps, eps])
        field = stratix.compute_field_profile(result, z)

        H_left = field["H"][0, 0][0]
        H_right = field["H"][0, 0][1]
        assert abs(H_left - H_right) < 1e-4

    def test_standing_wave_superstrate(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        result = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        z = nd.array([-wavelength / 4, 0.0])
        field = stratix.compute_field_profile(result, z)

        E0_2 = float(abs(field["E"][0, 0][1]) ** 2)
        assert abs(E0_2 - 0.64) < 1e-12

        Em4_2 = float(abs(field["E"][0, 0][0]) ** 2)
        assert abs(Em4_2 - 1.44) < 1e-12

    def test_constant_power_substrate(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        result = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        z = nd.array([1e-9, 100e-9, 200e-9])
        field = stratix.compute_field_profile(result, z)

        E2_vals = [float(abs(ee) ** 2) for ee in field["E"][0, 0]]
        for v in E2_vals:
            assert abs(v - E2_vals[0]) < 1e-12

    def test_single_interface_off_normal(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        k0 = 2 * nd.pi / wavelength
        theta_deg = 30.0
        kx = float(n_air * k0 * nd.sin(nd.array(theta_deg * nd.pi / 180)))

        result = stratix.solve(
            stack, wavelength, kx=kx, polarization=Polarization.TE
        )

        eps = 1e-20
        z = nd.array([-eps, eps])
        field = stratix.compute_field_profile(result, z)

        E_left = field["E"][0, 0][0]
        E_right = field["E"][0, 0][1]
        H_left = field["H"][0, 0][0]
        H_right = field["H"][0, 0][1]

        assert abs(E_left - E_right) < 1e-8
        assert abs(H_left - H_right) < 1e-4


class TestFieldMultiLayerTE:
    def test_continuity_two_layer(self, set_backend):
        n_air, n_a, n_b, n_sub = 1.0, 1.38, 2.0, 1.5
        wavelength = 5e-7
        d_a = wavelength / (4 * n_a)
        d_b = wavelength / (4 * n_b)

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[
                Layer(thickness=d_a, material=Material(epsilon=n_a**2)),
                Layer(thickness=d_b, material=Material(epsilon=n_b**2)),
            ],
        )

        result = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        z0 = d_a
        z1 = d_a + d_b
        eps = 1e-20
        z = nd.array(
            [-eps, eps, z0 - eps, z0 + eps, z1 - eps, z1 + eps]
        )
        field = stratix.compute_field_profile(result, z)

        assert abs(field["E"][0, 0][0] - field["E"][0, 0][1]) < 1e-8
        assert abs(field["H"][0, 0][0] - field["H"][0, 0][1]) < 1e-4

        assert abs(field["E"][0, 0][2] - field["E"][0, 0][3]) < 1e-8
        assert abs(field["H"][0, 0][2] - field["H"][0, 0][3]) < 1e-4

        assert abs(field["E"][0, 0][4] - field["E"][0, 0][5]) < 1e-8
        assert abs(field["H"][0, 0][4] - field["H"][0, 0][5]) < 1e-4

    def test_continuity_bragg_mirror(self, set_backend):
        n_low, n_high = 1.38, 2.3
        wavelength = 5e-7
        d_low = wavelength / (4 * n_low)
        d_high = wavelength / (4 * n_high)
        n_air, n_sub = 1.0, 1.5

        N = 3
        layers = []
        for _ in range(N):
            layers.append(
                Layer(thickness=d_high, material=Material(epsilon=n_high**2))
            )
            layers.append(
                Layer(thickness=d_low, material=Material(epsilon=n_low**2))
            )

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=layers,
        )

        result = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        thicknesses = [d_high, d_low] * N
        z_interfaces = [sum(thicknesses[:i]) for i in range(len(thicknesses) + 1)]

        eps = 1e-20
        z_points = []
        for zi in z_interfaces:
            z_points.append(zi - eps)
            z_points.append(zi + eps)

        z = nd.array(z_points)
        field = stratix.compute_field_profile(result, z)

        n_interfaces = len(z_interfaces)
        for i in range(n_interfaces):
            E_left = field["E"][0, 0][2 * i]
            E_right = field["E"][0, 0][2 * i + 1]
            H_left = field["H"][0, 0][2 * i]
            H_right = field["H"][0, 0][2 * i + 1]
            assert abs(E_left - E_right) < 1e-8, (
                f"E discontinuity at interface {i}"
            )
            assert abs(H_left - H_right) < 1e-4, (
                f"H discontinuity at interface {i}"
            )

    def test_substrate_traveling_wave_bragg(self, set_backend):
        """|E|^2 constant in substrate for Bragg mirror (pure traveling wave)."""
        n_low, n_high = 1.38, 2.3
        wavelength = 5e-7
        d_low = wavelength / (4 * n_low)
        d_high = wavelength / (4 * n_high)
        n_air, n_sub = 1.0, 1.5

        N = 4
        layers = []
        for _ in range(N):
            layers.append(
                Layer(thickness=d_high, material=Material(epsilon=n_high**2))
            )
            layers.append(
                Layer(thickness=d_low, material=Material(epsilon=n_low**2))
            )

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=layers,
        )

        result = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        thicknesses = [d_high, d_low] * N
        total = sum(thicknesses)
        z = nd.array([total + 1e-9, total + 100e-9, total + 200e-9])
        field = stratix.compute_field_profile(result, z)

        E2_vals = [float(abs(ee) ** 2) for ee in field["E"][0, 0]]
        for v in E2_vals:
            assert abs(v - E2_vals[0]) < 1e-12, f"|E|^2 not constant in substrate: {E2_vals}"

    def test_returns_dict_with_arrays(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        result = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        z = nd.array([-200e-9, -100e-9, 0.0, 100e-9, 200e-9])
        field = stratix.compute_field_profile(result, z)

        assert "E" in field
        assert "H" in field
        assert "z" in field
        assert field["E"].shape == (1, 1, 5)
        assert field["H"].shape == (1, 1, 5)
        assert field["z"].shape == (5,)


class TestFieldTM:
    def test_continuity_at_interface(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        result = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TM
        )

        eps = 1e-20
        z = nd.array([-eps, eps])
        field = stratix.compute_field_profile(result, z)

        assert abs(field["E"][0, 0][0] - field["E"][0, 0][1]) < 1e-5
        assert abs(field["H"][0, 0][0] - field["H"][0, 0][1]) < 1e-4


def _sweep_stack():
    n_air, n_a, n_b, n_sub = 1.0, 1.38, 2.0, 1.5
    wavelength = 5e-7
    return Stack(
        superstrate=Material(epsilon=n_air**2),
        substrate=Material(epsilon=n_sub**2),
        layers=[
            Layer(
                thickness=wavelength / (4 * n_a),
                material=Material(epsilon=n_a**2),
            ),
            Layer(
                thickness=wavelength / (4 * n_b),
                material=Material(epsilon=n_b**2),
            ),
        ],
    )


class TestFieldSweep:
    """Field profiles for array wavelength/kx sweeps — Issue #47."""

    def test_wavelength_sweep_shape(self, set_backend):
        stack = _sweep_stack()
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        result = stratix.solve(
            stack, wavelengths, kx=0.0, polarization=Polarization.TE
        )

        z = nd.array([-100e-9, 0.0, 50e-9, 300e-9])
        field = stratix.compute_field_profile(result, z)

        assert field["E"].shape == (3, 1, 4)
        assert field["H"].shape == (3, 1, 4)
        assert field["z"].shape == (4,)

    def test_wavelength_sweep_matches_scalar(self, set_backend):
        stack = _sweep_stack()
        wl_list = [4e-7, 5e-7, 6e-7]
        z = nd.array([-100e-9, 0.0, 50e-9, 120e-9, 300e-9])

        swept = stratix.compute_field_profile(
            stratix.solve(
                stack, nd.array(wl_list), kx=0.0, polarization=Polarization.TE
            ),
            z,
        )

        for i, wl in enumerate(wl_list):
            single = stratix.compute_field_profile(
                stratix.solve(stack, wl, kx=0.0, polarization=Polarization.TE), z
            )
            for j in range(len(z)):
                assert abs(swept["E"][i, 0, j] - single["E"][0, 0, j]) < 1e-10
                scale = abs(single["H"][0, 0, j])
                assert abs(swept["H"][i, 0, j] - single["H"][0, 0, j]) < 1e-10 * max(
                    1.0, float(scale)
                )

    def test_two_dimensional_sweep_shape(self, set_backend):
        stack = _sweep_stack()
        wavelengths = nd.array([4e-7, 5e-7, 6e-7])
        kx = nd.array([0.0, 1e6])
        result = stratix.solve(
            stack, wavelengths, kx=kx, polarization=Polarization.TE
        )

        z = nd.array([-100e-9, 50e-9, 300e-9])
        field = stratix.compute_field_profile(result, z)

        assert field["E"].shape == (3, 2, 3)
        assert field["H"].shape == (3, 2, 3)

    def test_sweep_continuity_at_interfaces(self, set_backend):
        stack = _sweep_stack()
        wavelengths = nd.array([4e-7, 6e-7])
        result = stratix.solve(
            stack, wavelengths, kx=0.0, polarization=Polarization.TE
        )

        d_a = stack.layers[0].thickness
        d_b = stack.layers[1].thickness
        eps = 1e-20
        z = nd.array(
            [-eps, eps, d_a - eps, d_a + eps, d_a + d_b - eps, d_a + d_b + eps]
        )
        field = stratix.compute_field_profile(result, z)

        for i in range(2):
            for j in (0, 2, 4):
                assert abs(field["E"][i, 0, j] - field["E"][i, 0, j + 1]) < 1e-8
                assert abs(field["H"][i, 0, j] - field["H"][i, 0, j + 1]) < 1e-4
