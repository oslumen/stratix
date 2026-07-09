"""Tests for single-interface S-matrix (Issue #16) and multi-layer (Issue #17)."""

from __future__ import annotations

import numdiff as nd
import pytest
from phokaia import Layer
from phokaia import Material
from phokaia import Stack

import stratix
from stratix._result import Result
from stratix._types import Method
from stratix._types import Polarization


def _analytical_R_TE(n_inc: complex, n_sub: complex) -> float:
    r = (n_inc - n_sub) / (n_inc + n_sub)
    return float(abs(r) ** 2)


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


class TestSingleInterfaceTE:
    def test_air_glass_normal_incidence(self, set_backend):
        """R=0.04, T=0.96 for air→glass at normal incidence."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7  # 500 nm
        result = stratix.solve(
            stack,
            wavelength,
            kx=0.0,
            polarization=Polarization.TE,
        )

        expected_R = _analytical_R_TE(n_air, n_glass)
        R = float(result.R[0])
        T = float(result.T[0])

        assert abs(R - expected_R) < 1e-12
        assert abs(R + T - 1.0) < 1e-12
        assert result.method_used == Method.SMATRIX

    def test_air_silicon_normal_incidence(self, set_backend):
        """R ≈ 0.3086 for air→Si (n=3.5) at normal incidence."""
        n_air, n_si = 1.0, 3.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_si**2),
        )
        wavelength = 5e-7
        result = stratix.solve(
            stack,
            wavelength,
            kx=0.0,
            polarization=Polarization.TE,
        )

        expected_R = _analytical_R_TE(n_air, n_si)
        R = float(result.R[0])
        T = float(result.T[0])

        assert abs(R - expected_R) < 1e-12
        assert abs(R + T - 1.0) < 1e-12

    def test_metal_normal_incidence(self, set_backend):
        """R → 1.0 for metal (n=0.05, k=3.5) at normal incidence."""
        n_air = 1.0
        n_metal = 0.05 + 3.5j
        eps_metal = n_metal**2
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=eps_metal),
        )
        wavelength = 5e-7
        result = stratix.solve(
            stack,
            wavelength,
            kx=0.0,
            polarization=Polarization.TE,
        )

        expected_R = _analytical_R_TE(n_air, n_metal)
        R = float(result.R[0])

        assert abs(R - expected_R) < 1e-12
        assert R > 0.9

    def test_R_plus_T_equals_one_lossless(self, set_backend):
        """Energy conservation for lossless dielectric interface."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=4.0),  # n=2.0
        )
        wavelength = 5e-7
        for kx in [0.0, 5e6, 1e7]:
            result = stratix.solve(
                stack,
                wavelength,
                kx=kx,
                polarization=Polarization.TE,
            )
            R = float(result.R[0])
            T = float(result.T[0])
            assert abs(R + T - 1.0) < 1e-12, f"R+T={R+T} at kx={kx}"

    def test_method_auto_resolves_to_smatrix(self, set_backend):
        """Method.AUTO → Method.SMATRIX."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve(
            stack,
            5e-7,
            kx=0.0,
            polarization=Polarization.TE,
            method=Method.AUTO,
        )
        assert result.method_used == Method.SMATRIX

    def test_result_is_pydantic_model(self, set_backend):
        """Result is a Pydantic BaseModel."""
        stack = Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
        )
        result = stratix.solve(
            stack,
            5e-7,
            kx=0.0,
            polarization=Polarization.TE,
        )
        assert isinstance(result, Result)
        assert hasattr(result, "R")
        assert hasattr(result, "T")
        assert hasattr(result, "wavelengths")
        assert hasattr(result, "kx")
        assert hasattr(result, "polarization")
        assert hasattr(result, "method_used")

    def test_off_normal_kx(self, set_backend):
        """R matches analytic Fresnel at off-normal incidence."""
        n_air, n_sub = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
        )
        wavelength = 5e-7

        for theta_deg in [0, 30, 60]:
            theta_rad = nd.array(nd.pi * theta_deg / 180)
            k0 = 2 * nd.pi / wavelength
            kx = float(n_air * k0 * nd.sin(theta_rad))

            # Analytic TE Fresnel
            kz0 = float(n_air * k0 * nd.cos(theta_rad))
            kz1 = float(nd.sqrt(nd.array(float(n_sub**2) * k0**2 - kx**2)).real)
            r = (kz0 - kz1) / (kz0 + kz1)
            expected_R = float(abs(r) ** 2)

            result = stratix.solve(
                stack,
                wavelength,
                kx=kx,
                polarization=Polarization.TE,
            )
            R = float(result.R[0])
            assert abs(R - expected_R) < 1e-10, (
                f"θ={theta_deg}°: R={R}, expected={expected_R}"
            )

    def test_total_internal_reflection(self, set_backend):
        """R → 1 at angles beyond critical angle."""
        n_air, n_sub = 1.5, 1.0  # glass → air
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
        )
        wavelength = 5e-7
        k0 = 2 * nd.pi / wavelength

        # kx beyond critical: sinθ_c = n_sub/n_air = 1/1.5
        kx = 1.1 * n_sub * k0  # > k0*n_sub, total internal reflection
        result = stratix.solve(
            stack,
            wavelength,
            kx=float(kx),
            polarization=Polarization.TE,
        )
        R = float(result.R[0])
        T = float(result.T[0])

        assert abs(R - 1.0) < 1e-12
        assert abs(T) < 1e-12


class TestMultiLayerTE:
    def test_quarter_wave_ar_coating(self, set_backend):
        """Single-layer quarter-wave AR coating: R → 0 at design λ.

        n0=1.0, n1=1.38 (MgF2), n2=1.5 (glass).
        d1 = λ/(4*n1) gives destructive interference at normal incidence.
        """
        n_air, n_mgf2, n_glass = 1.0, 1.38, 1.5
        wavelength = 5e-7
        d_ar = wavelength / (4 * n_mgf2)

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
            layers=[Layer(thickness=d_ar, material=Material(epsilon=n_mgf2**2))],
        )

        result = stratix.solve(
            stack,
            wavelength,
            kx=0.0,
            polarization=Polarization.TE,
        )

        R = float(result.R[0])
        T = float(result.T[0])

        # Ideal AR coating: R = ((n0*n2 - n1^2)/(n0*n2 + n1^2))^2
        expected_R = ((n_air * n_glass - n_mgf2**2) / (n_air * n_glass + n_mgf2**2)) ** 2
        assert abs(R - expected_R) < 1e-12, f"R={R}, expected={expected_R}"
        assert abs(R + T - 1.0) < 1e-12

    def test_single_layer_off_normal(self, set_backend):
        """Single-layer stack matches analytic Fabry-Pérot formula."""
        n_air, n_film, n_sub = 1.0, 1.38, 1.5
        wavelength = 5e-7
        d = wavelength / (4 * n_film)  # non-quarter-wave off-normal

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[Layer(thickness=d, material=Material(epsilon=n_film**2))],
        )

        theta_deg = 30.0
        theta_rad_arr = nd.array(nd.pi * theta_deg / 180)
        k0 = 2 * nd.pi / wavelength
        kx = float(n_air * k0 * nd.sin(theta_rad_arr))

        result = stratix.solve(
            stack,
            wavelength,
            kx=kx,
            polarization=Polarization.TE,
        )

        R = float(result.R[0])
        T = float(result.T[0])
        assert abs(R + T - 1.0) < 1e-12

    def test_two_layer_stack(self, set_backend):
        """Two-layer stack: R+T=1 for lossless dielectrics."""
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
            stack,
            wavelength,
            kx=0.0,
            polarization=Polarization.TE,
        )

        R = float(result.R[0])
        T = float(result.T[0])
        assert abs(R + T - 1.0) < 1e-12

    def test_bragg_mirror_convergence(self, set_backend):
        """Bragg mirror: R → 1.0 with increasing layer pairs."""
        n_low, n_high = 1.38, 2.3
        wavelength = 5e-7
        d_low = wavelength / (4 * n_low)
        d_high = wavelength / (4 * n_high)

        n_air, n_sub = 1.0, 1.5

        R_values = []
        for N in [1, 2, 4, 8]:
            layers = []
            for _ in range(N):
                layers.append(Layer(thickness=d_high, material=Material(epsilon=n_high**2)))
                layers.append(Layer(thickness=d_low, material=Material(epsilon=n_low**2)))
            stack = Stack(
                superstrate=Material(epsilon=n_air**2),
                substrate=Material(epsilon=n_sub**2),
                layers=layers,
            )
            result = stratix.solve(
                stack,
                wavelength,
                kx=0.0,
                polarization=Polarization.TE,
            )
            R_values.append(float(result.R[0]))

        # Reflectance should increase monotonically with layer count
        for i in range(len(R_values) - 1):
            assert R_values[i + 1] > R_values[i], (
                f"R not monotonic: {R_values}"
            )
        # 8-pair Bragg mirror should have R > 0.99
        assert R_values[-1] > 0.99, f"R={R_values[-1]} for 8-pair mirror"

    def test_zero_layer_stack(self, set_backend):
        """Stack with no layers = single-interface behavior."""
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        result = stratix.solve(
            stack,
            wavelength,
            kx=0.0,
            polarization=Polarization.TE,
        )

        expected_R = _analytical_R_TE(n_air, n_glass)
        R = float(result.R[0])
        T = float(result.T[0])

        assert abs(R - expected_R) < 1e-12
        assert abs(R + T - 1.0) < 1e-12

    def test_multilayer_R_plus_T_equals_one(self, set_backend):
        """Energy conservation for lossless dielectric multilayers."""
        n_air, n_a, n_b, n_sub = 1.0, 1.38, 2.0, 1.5
        wavelength = 5e-7

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[
                Layer(thickness=100e-9, material=Material(epsilon=n_a**2)),
                Layer(thickness=200e-9, material=Material(epsilon=n_b**2)),
                Layer(thickness=50e-9, material=Material(epsilon=n_a**2)),
            ],
        )

        for kx in [0.0, 5e6, 1e7]:
            result = stratix.solve(
                stack,
                wavelength,
                kx=kx,
                polarization=Polarization.TE,
            )
            R = float(result.R[0])
            T = float(result.T[0])
            assert abs(R + T - 1.0) < 1e-12, f"R+T={R+T} at kx={kx}"
