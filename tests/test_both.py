"""Tests for BOTH polarization support — Issue #26."""

from __future__ import annotations

import numdiff as nd
import pytest
from phokaia import Layer
from phokaia import Material
from phokaia import Stack

import stratix
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


class TestBothPolarization:
    """BOTH polarization must stack TE (index 0) and TM (index 1)."""

    def test_polarization_enum_has_BOTH(self):
        assert Polarization.BOTH == "BOTH"

    def test_result_R_shape_is_2_by_1(self, set_backend):
        """BOTH R has shape (2, 1) for scalar inputs."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve(stack, 5e-7, kx=0.0, polarization=Polarization.BOTH)
        assert result.R.shape == (2, 1)

    def test_index_0_matches_TE_normal_incidence(self, set_backend):
        """BOTH[0] equals independent TE solve at normal incidence."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        te = stratix.solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        both = stratix.solve(stack, wavelength, kx=0.0, polarization=Polarization.BOTH)
        assert abs(float(both.R[0, 0]) - float(te.R[0])) < 1e-12
        assert abs(float(both.T[0, 0]) - float(te.T[0])) < 1e-12

    def test_index_1_matches_TM_normal_incidence(self, set_backend):
        """BOTH[1] equals independent TM solve at normal incidence."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        tm = stratix.solve(stack, wavelength, kx=0.0, polarization=Polarization.TM)
        both = stratix.solve(stack, wavelength, kx=0.0, polarization=Polarization.BOTH)
        assert abs(float(both.R[1, 0]) - float(tm.R[0])) < 1e-12
        assert abs(float(both.T[1, 0]) - float(tm.T[0])) < 1e-12

    def test_off_normal_TE_and_TM_differ(self, set_backend):
        """BOTH TE[0] and TM[1] differ at off-normal incidence."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        k0 = 2 * nd.pi / wavelength
        theta_rad = nd.array(45.0 * nd.pi / 180)
        kx = float(n_air * k0 * nd.sin(theta_rad))
        result = stratix.solve(stack, wavelength, kx=kx, polarization=Polarization.BOTH)
        assert abs(float(result.R[0, 0]) - float(result.R[1, 0])) > 1e-6

    def test_brewster_TM_near_zero(self, set_backend):
        """BOTH TM channel near zero at Brewster angle."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        k0 = 2 * nd.pi / wavelength
        theta_B = float(nd.arctan(nd.array(n_glass / n_air)))
        kx_B = n_air * k0 * float(nd.sin(nd.array(theta_B)))
        result = stratix.solve(stack, wavelength, kx=kx_B, polarization=Polarization.BOTH)
        assert abs(float(result.R[1, 0])) < 1e-12

    def test_energy_conservation_both_polarizations(self, set_backend):
        """R+T=1 for both TE and TM with lossless dielectrics."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        result = stratix.solve(stack, 5e-7, kx=0.0, polarization=Polarization.BOTH)
        assert abs(float(result.R[0, 0]) + float(result.T[0, 0]) - 1.0) < 1e-12
        assert abs(float(result.R[1, 0]) + float(result.T[1, 0]) - 1.0) < 1e-12

    def test_polarization_field_reports_BOTH(self, set_backend):
        """result.polarization is Polarization.BOTH."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve(stack, 5e-7, kx=0.0, polarization=Polarization.BOTH)
        assert result.polarization == Polarization.BOTH

    def test_method_used_propagated(self, set_backend):
        """method_used matches the resolved method."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve(stack, 5e-7, kx=0.0, polarization=Polarization.BOTH)
        assert result.method_used is not None

    def test_multilayer_both_conserves_energy(self, set_backend):
        """Multi-layer BOTH: both polarizations satisfy R+T=1."""
        n_air, n_a, n_b, n_sub = 1.0, 1.38, 2.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[
                Layer(thickness=100e-9, material=Material(epsilon=n_a**2)),
                Layer(thickness=200e-9, material=Material(epsilon=n_b**2)),
                Layer(thickness=50e-9, material=Material(epsilon=n_a**2)),
            ],
        )
        result = stratix.solve(stack, 5e-7, kx=0.0, polarization=Polarization.BOTH)
        assert abs(float(result.R[0, 0]) + float(result.T[0, 0]) - 1.0) < 1e-12
        assert abs(float(result.R[1, 0]) + float(result.T[1, 0]) - 1.0) < 1e-12

    def test_total_internal_reflection_both(self, set_backend):
        """BOTH: both polarizations R→1 beyond critical angle."""
        n_glass, n_air = 1.5, 1.0
        stack = Stack(
            superstrate=Material(epsilon=n_glass**2),
            substrate=Material(epsilon=n_air**2),
        )
        wavelength = 5e-7
        k0 = 2 * nd.pi / wavelength
        kx = 1.1 * n_air * k0
        result = stratix.solve(
            stack, wavelength, kx=float(kx), polarization=Polarization.BOTH
        )
        assert abs(float(result.R[0, 0]) - 1.0) < 1e-12
        assert abs(float(result.R[1, 0]) - 1.0) < 1e-12
        assert abs(float(result.T[0, 0])) < 1e-12
        assert abs(float(result.T[1, 0])) < 1e-12

    def test_both_TE_matches_analytic_Fresnel(self, set_backend):
        """BOTH TE channel matches analytic TE Fresnel off-normal."""
        n_air, n_sub = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
        )
        wavelength = 5e-7
        k0 = 2 * nd.pi / wavelength
        theta_deg = 30.0
        theta_rad = nd.array(nd.pi * theta_deg / 180)
        kx = float(n_air * k0 * nd.sin(theta_rad))
        result = stratix.solve(stack, wavelength, kx=kx, polarization=Polarization.BOTH)
        kz0 = float(n_air * k0 * nd.cos(theta_rad))
        kz1 = float(nd.sqrt(nd.array(float(n_sub**2) * k0**2 - kx**2)).real)
        r_te = (kz0 - kz1) / (kz0 + kz1)
        expected_R = float(abs(r_te) ** 2)
        assert abs(float(result.R[0, 0]) - expected_R) < 1e-10

    def test_both_TM_matches_analytic_Fresnel(self, set_backend):
        """BOTH TM channel matches analytic TM Fresnel off-normal."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        k0 = 2 * nd.pi / wavelength
        theta_deg = 30.0
        theta_rad = nd.array(nd.pi * theta_deg / 180)
        kx = float(n_air * k0 * nd.sin(theta_rad))
        result = stratix.solve(stack, wavelength, kx=kx, polarization=Polarization.BOTH)
        eps1 = n_air**2
        eps2 = n_glass**2
        kz1 = float(nd.real(nd.sqrt(nd.array(eps1 * k0**2 - kx**2) + 0j)))
        kz2 = float(nd.real(nd.sqrt(nd.array(eps2 * k0**2 - kx**2) + 0j)))
        r_tm = (eps2 * kz1 - eps1 * kz2) / (eps2 * kz1 + eps1 * kz2)
        expected_R = float(abs(r_tm) ** 2)
        assert abs(float(result.R[1, 0]) - expected_R) < 1e-10
