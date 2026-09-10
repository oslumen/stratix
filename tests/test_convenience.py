"""Tests for convenience APIs: solve_angles, solve_from_source (Issue #19)."""

from __future__ import annotations

import numdiff as nd
import pytest
from phokaia import Layer
from phokaia import Material
from phokaia import PlaneWave
from phokaia import Polarization
from phokaia import Stack

import stratix

_C0: float = 299792458.0  # speed of light in vacuum (m/s)


class TestSolveAngles:
    def test_scalar_angle_matches_kx_solve(self, set_backend):
        """solve_angles(stack, λ, θ, TE) matches solve() with computed kx."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 633e-9
        theta_deg = 45.0

        # Expected kx = (2π/λ) * n_super * sin(θ)
        k0 = 2 * nd.pi / wavelength
        theta_rad = nd.array(theta_deg * nd.pi / 180)
        expected_kx = float(n_air * k0 * nd.sin(theta_rad))

        result_conv = stratix.solve_angles(
            stack, wavelength, theta_deg, polarization=Polarization.TE
        )
        result_direct = stratix.solve(
            stack, wavelength, kx=expected_kx, polarization=Polarization.TE
        )

        assert abs(float(result_conv.R[0, 0]) - float(result_direct.R[0, 0])) < 1e-12
        assert abs(float(result_conv.T[0, 0]) - float(result_direct.T[0, 0])) < 1e-12
        assert abs(float(result_conv.kx[0]) - float(result_direct.kx[0])) < (
            1e-12 * max(abs(expected_kx), 1.0)
        )

    def test_normal_incidence(self, set_backend):
        """θ=0 → kx=0, matches normal incidence solve."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 633e-9

        result_conv = stratix.solve_angles(
            stack, wavelength, 0.0, polarization=Polarization.TE
        )
        result_direct = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        assert abs(float(result_conv.R[0, 0]) - float(result_direct.R[0, 0])) < 1e-12
        assert abs(float(result_conv.T[0, 0]) - float(result_direct.T[0, 0])) < 1e-12

    def test_multiple_angles(self, set_backend):
        """Array of angles produces Result with matching kx values."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 633e-9
        angles = [0.0, 30.0, 60.0]

        result = stratix.solve_angles(
            stack, wavelength, angles, polarization=Polarization.TE
        )

        # Angles fill the Nk axis of the (Nλ, Nk) contract; the single
        # wavelength is the length-1 Nλ axis.
        assert result.R.shape == (1, 3)
        assert result.T.shape == (1, 3)
        assert result.kx.shape == (3,)
        assert result.wavelengths.shape == (1,)

        for i, theta_deg in enumerate(angles):
            theta_rad = nd.array(theta_deg * nd.pi / 180)
            k0 = 2 * nd.pi / wavelength
            expected_kx = float(n_air * k0 * nd.sin(theta_rad))
            assert abs(float(result.kx[i]) - expected_kx) < 1e-12 * max(
                abs(expected_kx), 1.0
            )

    def test_tm_polarization(self, set_backend):
        """solve_angles works with TM polarization."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 633e-9
        theta_deg = 30.0

        result_conv = stratix.solve_angles(
            stack, wavelength, theta_deg, polarization=Polarization.TM
        )

        k0 = 2 * nd.pi / wavelength
        theta_rad = nd.array(theta_deg * nd.pi / 180)
        expected_kx = float(n_air * k0 * nd.sin(theta_rad))

        result_direct = stratix.solve(
            stack, wavelength, kx=expected_kx, polarization=Polarization.TM
        )

        assert abs(float(result_conv.R[0, 0]) - float(result_direct.R[0, 0])) < 1e-12
        assert abs(float(result_conv.T[0, 0]) - float(result_direct.T[0, 0])) < 1e-12

    def test_angle_above_90_raises(self, set_backend):
        """θ > 90° should raise ValueError."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        with pytest.raises(ValueError, match="90"):
            stratix.solve_angles(stack, 633e-9, 91.0, polarization=Polarization.TE)

    def test_angle_below_0_raises(self, set_backend):
        """θ < 0° should raise ValueError."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        with pytest.raises(ValueError, match="0"):
            stratix.solve_angles(stack, 633e-9, -1.0, polarization=Polarization.TE)

    def test_R_plus_T_equals_one_lossless(self, set_backend):
        """Energy conservation holds for solve_angles."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 633e-9

        for theta in [0, 20, 45, 60]:
            result = stratix.solve_angles(
                stack, wavelength, theta, polarization=Polarization.TE
            )
            R = float(result.R[0, 0])
            T = float(result.T[0, 0])
            assert abs(R + T - 1.0) < 1e-12, f"θ={theta}: R+T={R + T}"

    def test_method_auto_resolves_to_smatrix(self, set_backend):
        """solve_angles defaults to AUTO → SMATRIX."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve_angles(stack, 633e-9, 30.0, polarization=Polarization.TE)
        assert result.method_used == stratix.Method.SMATRIX

    def test_absorption_flag_accepted(self, set_backend):
        """absorption=False is accepted (no-op)."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve_angles(
            stack, 633e-9, 30.0, polarization=Polarization.TE, absorption=False
        )
        assert result is not None
        assert abs(float(result.R[0, 0]) + float(result.T[0, 0]) - 1.0) < 1e-12


class TestSolveAnglesShapeContract:
    """solve_angles obeys the (Nλ, Nk) contract, BOTH and absorption too."""

    @staticmethod
    def _absorbing_stack() -> Stack:
        from phokaia import Layer

        return Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[
                Layer(
                    thickness=40e-9,
                    material=Material(epsilon=complex(-3.68, 2.90)),
                ),
            ],
        )

    def test_both_prepends_polarization_axis(self, set_backend):
        """BOTH gives (2, 1, n_angles), not a TE-only (1, n_angles)."""
        stack = self._absorbing_stack()
        angles = [0.0, 30.0, 60.0]
        both = stratix.solve_angles(
            stack, 633e-9, angles, polarization=Polarization.BOTH
        )
        assert both.R.shape == (2, 1, 3)
        assert both.T.shape == (2, 1, 3)

        te = stratix.solve_angles(stack, 633e-9, angles, polarization=Polarization.TE)
        tm = stratix.solve_angles(stack, 633e-9, angles, polarization=Polarization.TM)
        assert float(nd.max(nd.abs(both.R[0] - te.R))) < 1e-14
        assert float(nd.max(nd.abs(both.R[1] - tm.R))) < 1e-14

    def test_absorption_fields_are_populated(self, set_backend):
        """absorption=True fills layer_absorption and energy_balance."""
        stack = self._absorbing_stack()
        result = stratix.solve_angles(
            stack,
            633e-9,
            [0.0, 30.0, 60.0],
            polarization=Polarization.TE,
            absorption=True,
        )
        assert result.layer_absorption is not None
        assert result.layer_absorption.shape == (1, 1, 3)
        assert result.energy_balance.shape == (1, 3)
        for i in range(3):
            assert abs(float(result.energy_balance[0, i]) - 1.0) < 1e-10
            assert float(result.layer_absorption[0, 0, i]) > 0

    def test_absorption_fields_under_both(self, set_backend):
        """The TE/TM axis leads the absorption fields as well."""
        stack = self._absorbing_stack()
        result = stratix.solve_angles(
            stack,
            633e-9,
            [0.0, 45.0],
            polarization=Polarization.BOTH,
            absorption=True,
        )
        assert result.layer_absorption.shape == (2, 1, 1, 2)
        assert result.energy_balance.shape == (2, 1, 2)
        for pol_axis in range(2):
            for i in range(2):
                balance = float(result.energy_balance[pol_axis, 0, i])
                assert abs(balance - 1.0) < 1e-10


class TestSolveFromSource:
    _C0: float = _C0

    @staticmethod
    def _make_plane_wave(
        wavelength: float,
        theta_deg: float,
        n_mat: float = 1.0,
    ) -> PlaneWave:
        omega = 2 * nd.pi * _C0 / wavelength
        theta_rad = nd.array(theta_deg * nd.pi / 180)
        return PlaneWave(
            omega=omega,
            dim=3,
            theta=float(theta_rad),
            phi=0.0,
            amplitude=nd.array([0.0 + 0j, 1.0 + 0j, 0.0 + 0j]),
            material=Material(epsilon=n_mat**2),
        )

    def test_te_normal_incidence(self, set_backend):
        """solve_from_source with θ=0 TE matches solve() at normal."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        wavelength = 633e-9
        pw = self._make_plane_wave(wavelength, 0.0)

        result_src = stratix.solve_from_source(stack, pw, polarization=Polarization.TE)
        result_direct = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        assert abs(float(result_src.R[0, 0]) - float(result_direct.R[0, 0])) < 1e-12
        assert abs(float(result_src.T[0, 0]) - float(result_direct.T[0, 0])) < 1e-12

    def test_te_oblique_incidence(self, set_backend):
        """solve_from_source with θ=45 TE matches solve()."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 633e-9
        theta_deg = 45.0
        pw = self._make_plane_wave(wavelength, theta_deg, n_mat=n_air)

        k0 = 2 * nd.pi / wavelength
        theta_rad = nd.array(theta_deg * nd.pi / 180)
        expected_kx = float(n_air * k0 * nd.sin(theta_rad))

        result_src = stratix.solve_from_source(stack, pw, polarization=Polarization.TE)
        result_direct = stratix.solve(
            stack, wavelength, kx=expected_kx, polarization=Polarization.TE
        )

        assert abs(float(result_src.R[0, 0]) - float(result_direct.R[0, 0])) < 1e-12
        assert abs(float(result_src.T[0, 0]) - float(result_direct.T[0, 0])) < 1e-12
        assert abs(float(result_src.kx[0]) - expected_kx) < 1e-12

    def test_tm_oblique_incidence(self, set_backend):
        """solve_from_source with TM polarization matches solve()."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 633e-9
        theta_deg = 30.0
        pw = self._make_plane_wave(wavelength, theta_deg, n_mat=n_air)

        k0 = 2 * nd.pi / wavelength
        theta_rad = nd.array(theta_deg * nd.pi / 180)
        expected_kx = float(n_air * k0 * nd.sin(theta_rad))

        result_src = stratix.solve_from_source(stack, pw, polarization=Polarization.TM)
        result_direct = stratix.solve(
            stack, wavelength, kx=expected_kx, polarization=Polarization.TM
        )

        assert abs(float(result_src.R[0, 0]) - float(result_direct.R[0, 0])) < 1e-12
        assert abs(float(result_src.T[0, 0]) - float(result_direct.T[0, 0])) < 1e-12

    def test_dim_1_plane_wave(self, set_backend):
        """solve_from_source with dim=1 PlaneWave (kx=0)."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        wavelength = 633e-9
        omega = 2 * nd.pi * _C0 / wavelength
        pw = PlaneWave(
            omega=omega,
            dim=1,
            direction=1,
            amplitude=nd.array([1.0 + 0j]),
        )

        result_src = stratix.solve_from_source(stack, pw, polarization=Polarization.TE)
        result_direct = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        assert abs(float(result_src.R[0, 0]) - float(result_direct.R[0, 0])) < 1e-12
        assert abs(float(result_src.T[0, 0]) - float(result_direct.T[0, 0])) < 1e-12

    def test_dim_2_plane_wave(self, set_backend):
        """solve_from_source with dim=2 PlaneWave (phi-based)."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        wavelength = 633e-9
        omega = 2 * nd.pi * _C0 / wavelength
        theta_deg = 30.0
        theta_rad = float(nd.array(theta_deg * nd.pi / 180))
        pw = PlaneWave(
            omega=omega,
            dim=2,
            phi=theta_rad,
            amplitude=nd.array([1.0 + 0j, 0.0 + 0j]),
        )

        k0 = 2 * nd.pi / wavelength
        expected_kx = float(k0 * nd.cos(nd.array(theta_rad)))

        result_src = stratix.solve_from_source(stack, pw, polarization=Polarization.TE)
        result_direct = stratix.solve(
            stack, wavelength, kx=expected_kx, polarization=Polarization.TE
        )

        assert abs(float(result_src.R[0, 0]) - float(result_direct.R[0, 0])) < 1e-12
        assert abs(float(result_src.T[0, 0]) - float(result_direct.T[0, 0])) < 1e-12

    def test_zero_omega_raises(self, set_backend):
        """omega=0 should raise ValueError."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        pw = PlaneWave(
            omega=0.0,
            dim=3,
            theta=0.0,
            phi=0.0,
            amplitude=nd.array([0.0 + 0j, 1.0 + 0j, 0.0 + 0j]),
        )
        with pytest.raises(ValueError, match="omega"):
            stratix.solve_from_source(stack, pw, polarization=Polarization.TE)

    def test_method_auto_resolves_to_smatrix(self, set_backend):
        """solve_from_source defaults to AUTO → SMATRIX."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        pw = self._make_plane_wave(633e-9, 30.0)
        result = stratix.solve_from_source(stack, pw, polarization=Polarization.TE)
        assert result.method_used == stratix.Method.SMATRIX

    def test_R_plus_T_equals_one_lossless(self, set_backend):
        """Energy conservation holds for solve_from_source."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 633e-9

        for theta in [0, 20, 45, 60]:
            pw = self._make_plane_wave(wavelength, theta, n_mat=n_air)
            result = stratix.solve_from_source(stack, pw, polarization=Polarization.TE)
            R = float(result.R[0, 0])
            T = float(result.T[0, 0])
            assert abs(R + T - 1.0) < 1e-12, f"θ={theta}: R+T={R + T}"


class TestSolveAnglesVectorized:
    """solve_angles runs one vectorized solve and supports array wavelengths.

    Issue #51: the angle sweep is a single vectorized solve() over a kx grid,
    the superstrate index is evaluated at ω (not λ), and a complex-epsilon
    superstrate no longer hits a ``float()`` cast.
    """

    @staticmethod
    def _plasma_superstrate(amplitude: float = 4.0e30) -> Material:
        """A superstrate whose epsilon depends on ω, not λ."""
        return Material(epsilon=lambda omega: 1.0 + amplitude / omega**2)

    @staticmethod
    def _n_super_at(material: Material, wavelength: float) -> complex:
        omega = 2 * nd.pi * _C0 / wavelength
        eps = material.epsilon(omega=omega)
        mu = material.mu(omega=omega)
        return complex(nd.sqrt(nd.array(eps * mu + 0j)))

    def test_dispersive_superstrate_kx_uses_omega(self, set_backend):
        """kx = n(ω)·k0·sin θ, with the index taken at ω = 2πc/λ."""
        superstrate = self._plasma_superstrate()
        stack = Stack(superstrate=superstrate, substrate=Material(epsilon=4.0))
        wavelength = 633e-9
        angles = [0.0, 30.0, 60.0]

        n_super = self._n_super_at(superstrate, wavelength).real
        k0 = 2 * nd.pi / wavelength

        result = stratix.solve_angles(
            stack, wavelength, angles, polarization=Polarization.TE
        )

        for i, theta in enumerate(angles):
            expected = n_super * k0 * float(nd.sin(nd.array(theta * nd.pi / 180)))
            assert abs(float(result.kx[i]) - expected) < 1e-6 * max(expected, 1.0)

    def test_dispersive_superstrate_matches_direct_solve(self, set_backend):
        """R/T agree with solve() fed the ω-evaluated kx by hand."""
        superstrate = self._plasma_superstrate()
        stack = Stack(superstrate=superstrate, substrate=Material(epsilon=4.0))
        wavelength = 633e-9
        theta = 40.0

        n_super = self._n_super_at(superstrate, wavelength).real
        kx = (
            n_super
            * (2 * nd.pi / wavelength)
            * float(nd.sin(nd.array(theta * nd.pi / 180)))
        )

        conv = stratix.solve_angles(
            stack, wavelength, theta, polarization=Polarization.TE
        )
        direct = stratix.solve(stack, wavelength, kx=kx, polarization=Polarization.TE)

        assert abs(float(conv.R[0, 0]) - float(direct.R[0, 0])) < 1e-12
        assert abs(float(conv.T[0, 0]) - float(direct.T[0, 0])) < 1e-12

    def test_single_vectorized_solve_call(self, monkeypatch, set_backend):
        """All angles go through one dispatch, not one dispatch per angle."""
        from stratix import _solve as solve_module

        calls: list[int] = []
        original = solve_module._dispatch_solve

        def counting_dispatch(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(solve_module, "_dispatch_solve", counting_dispatch)

        stack = Stack(
            superstrate=Material(epsilon=1.0), substrate=Material(epsilon=2.25)
        )
        stratix.solve_angles(
            stack,
            633e-9,
            [0.0, 15.0, 30.0, 45.0, 60.0],
            polarization=Polarization.TE,
        )
        assert len(calls) == 1

    def test_array_wavelengths_grid_shape(self, set_backend):
        """Array wavelengths give the full (Nλ, Nk) grid."""
        stack = Stack(
            superstrate=Material(epsilon=1.0), substrate=Material(epsilon=2.25)
        )
        wavelengths = [500e-9, 600e-9, 700e-9]
        angles = [0.0, 30.0, 60.0, 80.0]

        result = stratix.solve_angles(
            stack, wavelengths, angles, polarization=Polarization.TE
        )

        assert result.R.shape == (3, 4)
        assert result.T.shape == (3, 4)
        assert result.wavelengths.shape == (3,)
        # kx varies with wavelength, so it follows the grid here.
        assert result.kx.shape == (3, 4)

    def test_array_wavelengths_match_scalar_calls(self, set_backend):
        """Each row of the grid equals the scalar-wavelength solve."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[Layer(thickness=120e-9, material=Material(epsilon=4.0))],
        )
        wavelengths = [500e-9, 650e-9]
        angles = [0.0, 35.0, 70.0]

        grid = stratix.solve_angles(
            stack, wavelengths, angles, polarization=Polarization.TE
        )
        for i, wl in enumerate(wavelengths):
            row = stratix.solve_angles(stack, wl, angles, polarization=Polarization.TE)
            assert float(nd.max(nd.abs(grid.R[i] - row.R[0]))) < 1e-12
            assert float(nd.max(nd.abs(grid.T[i] - row.T[0]))) < 1e-12

    def test_kx_stays_1d_for_scalar_wavelength(self, set_backend):
        """A scalar wavelength keeps kx a 1-D sweep coordinate."""
        stack = Stack(
            superstrate=Material(epsilon=1.0), substrate=Material(epsilon=2.25)
        )
        result = stratix.solve_angles(
            stack, 633e-9, [0.0, 30.0, 60.0], polarization=Polarization.TE
        )
        assert result.kx.shape == (3,)

    def test_array_wavelengths_under_both(self, set_backend):
        """BOTH prepends the TE/TM axis to the (Nλ, Nk) grid."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[Layer(thickness=80e-9, material=Material(epsilon=4.0))],
        )
        result = stratix.solve_angles(
            stack,
            [500e-9, 600e-9],
            [0.0, 45.0],
            polarization=Polarization.BOTH,
            absorption=True,
        )
        assert result.R.shape == (2, 2, 2)
        assert result.layer_absorption.shape == (2, 1, 2, 2)
        assert result.energy_balance.shape == (2, 2, 2)

    def test_complex_epsilon_superstrate_raises_clear_error(self, set_backend):
        """A lossy superstrate gets a clear error, not a TypeError from float()."""
        stack = Stack(
            superstrate=Material(epsilon=complex(2.25, 0.05)),
            substrate=Material(epsilon=4.0),
        )
        with pytest.raises(ValueError, match="transparent superstrate"):
            stratix.solve_angles(
                stack, 633e-9, [0.0, 30.0], polarization=Polarization.TE
            )

    def test_negative_epsilon_superstrate_raises_clear_error(self, set_backend):
        """A metallic superstrate hits the same guard, not a silent nan."""
        stack = Stack(
            superstrate=Material(epsilon=-2.0),
            substrate=Material(epsilon=4.0),
        )
        with pytest.raises(ValueError, match="transparent superstrate"):
            stratix.solve_angles(
                stack, 633e-9, [0.0, 30.0], polarization=Polarization.TE
            )

    def test_length_one_array_wavelength_keeps_the_grid(self, set_backend):
        """A length-1 array is an array, not a scalar: kx follows the grid."""
        stack = Stack(
            superstrate=Material(epsilon=1.0), substrate=Material(epsilon=2.25)
        )
        angles = [0.0, 30.0, 60.0]

        arrayed = stratix.solve_angles(
            stack, [633e-9], angles, polarization=Polarization.TE
        )
        scalar = stratix.solve_angles(
            stack, 633e-9, angles, polarization=Polarization.TE
        )

        assert arrayed.R.shape == (1, 3)
        assert arrayed.kx.shape == (1, 3)
        assert scalar.kx.shape == (3,)
        assert float(nd.max(nd.abs(arrayed.R - scalar.R))) < 1e-14

    def test_field_profile_from_an_angle_sweep(self, set_backend):
        """The Result carries intermediates, so field profiles work on it."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[Layer(thickness=100e-9, material=Material(epsilon=4.0))],
        )
        result = stratix.solve_angles(
            stack, 633e-9, [0.0, 30.0], polarization=Polarization.TE
        )
        z = nd.linspace(-50e-9, 150e-9, 7)
        profile = stratix.compute_field_profile(result, z)
        assert profile["E"].shape == (1, 2, 7)

    def test_metallic_superstrate_still_works_through_solve(self, set_backend):
        """The escape hatch named in the error message does work.

        ``solve_angles`` refuses ``eps*mu < 0`` only because it cannot map
        an angle to a real kx there.  ``solve()`` takes kx directly, and a
        metallic superstrate is perfectly well posed for it: nothing
        propagates in the incident medium, so every kx is evanescent and
        the answer is R = 1, T = 0.
        """
        stack = Stack(
            superstrate=Material(epsilon=-2.0),
            substrate=Material(epsilon=4.0),
        )
        result = stratix.solve(stack, 633e-9, kx=0.0, polarization=Polarization.TE)
        assert float(result.R[0, 0]) == pytest.approx(1.0)
        assert float(result.T[0, 0]) == pytest.approx(0.0)

    def test_absorbing_superstrate_is_refused_by_solve_too(self, set_backend):
        """The other half of the message: for loss there is no escape hatch.

        ``solve()`` used to accept a lossy superstrate and hand back R and
        T that did not partition energy (issue #57).  It now refuses, so
        the error text must not promise a way around it.
        """
        stack = Stack(
            superstrate=Material(epsilon=complex(2.25, 0.05)),
            substrate=Material(epsilon=4.0),
        )
        with pytest.raises(ValueError, match="lossless superstrate"):
            stratix.solve(stack, 633e-9, kx=0.0, polarization=Polarization.TE)

    def test_grad_through_wavelength(self, set_backend):
        """nd.grad w.r.t. wavelength flows through solve_angles."""
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")

        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[Layer(thickness=100e-9, material=Material(epsilon=1.9))],
        )

        def f(wl):
            return stratix.solve_angles(
                stack, wl, 30.0, polarization=Polarization.TE
            ).R[0, 0]

        wl0 = 5e-7
        grad_ad = float(nd.grad(f)(wl0))
        h = 1e-11
        grad_fd = (float(f(wl0 + h)) - float(f(wl0 - h))) / (2 * h)
        rel = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-12)
        assert rel < 1e-4, f"AD={grad_ad}, FD={grad_fd}"

    def test_grad_through_angle(self, set_backend):
        """nd.grad w.r.t. the incidence angle flows through solve_angles."""
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")

        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )

        def f(theta):
            return stratix.solve_angles(
                stack, 633e-9, theta, polarization=Polarization.TE
            ).R[0, 0]

        theta0 = 35.0
        grad_ad = float(nd.grad(f)(theta0))
        h = 1e-5
        grad_fd = (float(f(theta0 + h)) - float(f(theta0 - h))) / (2 * h)
        rel = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-12)
        assert rel < 1e-5, f"AD={grad_ad}, FD={grad_fd}"

    def test_grad_through_thickness(self, set_backend):
        """nd.grad w.r.t. layer thickness flows through solve_angles."""
        if nd.get_backend() == "numpy":
            pytest.skip("numpy backend does not support grad")

        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[Layer(thickness=100e-9, material=Material(epsilon=4.0))],
        )

        def f(d):
            return stratix.solve_angles(
                stack,
                633e-9,
                30.0,
                polarization=Polarization.TE,
                thicknesses=[d],
            ).R[0, 0]

        d0 = 100e-9
        grad_ad = float(nd.grad(f)(d0))
        h = 1e-13
        grad_fd = (float(f(d0 + h)) - float(f(d0 - h))) / (2 * h)
        rel = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-12)
        assert rel < 1e-4, f"AD={grad_ad}, FD={grad_fd}"
