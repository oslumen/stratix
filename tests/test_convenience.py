"""Tests for convenience APIs: solve_angles, solve_from_source (Issue #19)."""

from __future__ import annotations

import numdiff as nd
import pytest
from phokaia import Material
from phokaia import PlaneWave
from phokaia import Stack

import stratix
from phokaia import Polarization

_C0: float = 299792458.0  # speed of light in vacuum (m/s)


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

        assert abs(float(result_conv.R[0]) - float(result_direct.R[0])) < 1e-12
        assert abs(float(result_conv.T[0]) - float(result_direct.T[0])) < 1e-12
        assert abs(float(result_conv.kx[0]) - float(result_direct.kx[0])) < 1e-12

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

        assert abs(float(result_conv.R[0]) - float(result_direct.R[0])) < 1e-12
        assert abs(float(result_conv.T[0]) - float(result_direct.T[0])) < 1e-12

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

        assert len(result.R) == 3
        assert len(result.T) == 3
        assert len(result.kx) == 3
        assert len(result.wavelengths) == 3

        for i, theta_deg in enumerate(angles):
            theta_rad = nd.array(theta_deg * nd.pi / 180)
            k0 = 2 * nd.pi / wavelength
            expected_kx = float(n_air * k0 * nd.sin(theta_rad))
            assert abs(float(result.kx[i]) - expected_kx) < 1e-12

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

        assert abs(float(result_conv.R[0]) - float(result_direct.R[0])) < 1e-12
        assert abs(float(result_conv.T[0]) - float(result_direct.T[0])) < 1e-12

    def test_angle_above_90_raises(self, set_backend):
        """θ > 90° should raise ValueError."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        with pytest.raises(ValueError, match="90"):
            stratix.solve_angles(
                stack, 633e-9, 91.0, polarization=Polarization.TE
            )

    def test_angle_below_0_raises(self, set_backend):
        """θ < 0° should raise ValueError."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        with pytest.raises(ValueError, match="0"):
            stratix.solve_angles(
                stack, 633e-9, -1.0, polarization=Polarization.TE
            )

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
            R = float(result.R[0])
            T = float(result.T[0])
            assert abs(R + T - 1.0) < 1e-12, f"θ={theta}: R+T={R + T}"

    def test_method_auto_resolves_to_smatrix(self, set_backend):
        """solve_angles defaults to AUTO → SMATRIX."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve_angles(
            stack, 633e-9, 30.0, polarization=Polarization.TE
        )
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
        assert abs(float(result.R[0]) + float(result.T[0]) - 1.0) < 1e-12


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

        result_src = stratix.solve_from_source(
            stack, pw, polarization=Polarization.TE
        )
        result_direct = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        assert abs(float(result_src.R[0]) - float(result_direct.R[0])) < 1e-12
        assert abs(float(result_src.T[0]) - float(result_direct.T[0])) < 1e-12

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

        result_src = stratix.solve_from_source(
            stack, pw, polarization=Polarization.TE
        )
        result_direct = stratix.solve(
            stack, wavelength, kx=expected_kx, polarization=Polarization.TE
        )

        assert abs(float(result_src.R[0]) - float(result_direct.R[0])) < 1e-12
        assert abs(float(result_src.T[0]) - float(result_direct.T[0])) < 1e-12
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

        result_src = stratix.solve_from_source(
            stack, pw, polarization=Polarization.TM
        )
        result_direct = stratix.solve(
            stack, wavelength, kx=expected_kx, polarization=Polarization.TM
        )

        assert abs(float(result_src.R[0]) - float(result_direct.R[0])) < 1e-12
        assert abs(float(result_src.T[0]) - float(result_direct.T[0])) < 1e-12

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

        result_src = stratix.solve_from_source(
            stack, pw, polarization=Polarization.TE
        )
        result_direct = stratix.solve(
            stack, wavelength, kx=0.0, polarization=Polarization.TE
        )

        assert abs(float(result_src.R[0]) - float(result_direct.R[0])) < 1e-12
        assert abs(float(result_src.T[0]) - float(result_direct.T[0])) < 1e-12

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

        result_src = stratix.solve_from_source(
            stack, pw, polarization=Polarization.TE
        )
        result_direct = stratix.solve(
            stack, wavelength, kx=expected_kx, polarization=Polarization.TE
        )

        assert abs(float(result_src.R[0]) - float(result_direct.R[0])) < 1e-12
        assert abs(float(result_src.T[0]) - float(result_direct.T[0])) < 1e-12

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
            stratix.solve_from_source(
                stack, pw, polarization=Polarization.TE
            )

    def test_method_auto_resolves_to_smatrix(self, set_backend):
        """solve_from_source defaults to AUTO → SMATRIX."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        pw = self._make_plane_wave(633e-9, 30.0)
        result = stratix.solve_from_source(
            stack, pw, polarization=Polarization.TE
        )
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
            result = stratix.solve_from_source(
                stack, pw, polarization=Polarization.TE
            )
            R = float(result.R[0])
            T = float(result.T[0])
            assert abs(R + T - 1.0) < 1e-12, f"θ={theta}: R+T={R + T}"
