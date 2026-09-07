"""Integration tests against sbyrnes321/tmm reference implementation."""

from __future__ import annotations

import numdiff as nd
import numpy as np
import pytest
from phokaia import Layer
from phokaia import Material
from phokaia import Polarization
from phokaia import Stack

import stratix

try:
    import tmm

    HAS_TMM = True
except ImportError:
    HAS_TMM = False

pytestmark = pytest.mark.skipif(not HAS_TMM, reason="tmm not installed")


@pytest.fixture
def set_numpy_backend():
    nd.set_backend("numpy")
    yield
    nd.set_backend("numpy")


def _tmm_reference(n_list, d_list, theta_deg, wavelength, pol):
    """Run tmm.coh_tmm and return (R, T).

    tmm convention: d_list = [inf, d1, d2, ..., inf], same length as n_list.
    """
    inf = np.inf
    d_tmm = np.array([inf, *d_list, inf], dtype=float)
    n_tmm = np.array(n_list, dtype=complex)
    pol_tmm = "s" if pol == Polarization.TE else "p"
    theta_rad = np.pi * theta_deg / 180.0
    tmm_result = tmm.coh_tmm(pol_tmm, n_tmm, d_tmm, theta_rad, wavelength)
    return float(tmm_result["R"]), float(tmm_result["T"])


def _stratix_solve(stack, wavelength, theta_deg, pol):
    """Compute R, T with stratix from incidence angle."""
    k0 = 2 * np.pi / wavelength
    eps_inc = complex(stack.superstrate.epsilon())
    n_inc = float(np.sqrt(eps_inc).real)
    kx = float(n_inc * k0 * np.sin(np.pi * theta_deg / 180))
    result = stratix.solve(stack, wavelength, kx, pol)
    return float(result.R[0]), float(result.T[0])


class TestTmmSingleInterface:
    def test_air_glass_normal_TE(self, set_numpy_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        R_s, T_s = _stratix_solve(stack, wavelength, 0.0, Polarization.TE)
        R_t, T_t = _tmm_reference([n_air, n_glass], [], 0.0, wavelength, Polarization.TE)
        assert abs(R_s - R_t) < 1e-12
        assert abs(T_s - T_t) < 1e-12

    def test_air_glass_normal_TM(self, set_numpy_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        R_s, T_s = _stratix_solve(stack, wavelength, 0.0, Polarization.TM)
        R_t, T_t = _tmm_reference([n_air, n_glass], [], 0.0, wavelength, Polarization.TM)
        assert abs(R_s - R_t) < 1e-12
        assert abs(T_s - T_t) < 1e-12

    def test_air_silicon_normal_TE(self, set_numpy_backend):
        n_air, n_si = 1.0, 3.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_si**2),
        )
        wavelength = 5e-7
        R_s, T_s = _stratix_solve(stack, wavelength, 0.0, Polarization.TE)
        R_t, T_t = _tmm_reference([n_air, n_si], [], 0.0, wavelength, Polarization.TE)
        assert abs(R_s - R_t) < 1e-12
        assert abs(T_s - T_t) < 1e-12

    def test_air_silicon_normal_TM(self, set_numpy_backend):
        n_air, n_si = 1.0, 3.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_si**2),
        )
        wavelength = 5e-7
        R_s, T_s = _stratix_solve(stack, wavelength, 0.0, Polarization.TM)
        R_t, T_t = _tmm_reference([n_air, n_si], [], 0.0, wavelength, Polarization.TM)
        assert abs(R_s - R_t) < 1e-12
        assert abs(T_s - T_t) < 1e-12

    def test_off_normal_TE(self, set_numpy_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        for theta in [15, 30, 60]:
            R_s, T_s = _stratix_solve(stack, wavelength, theta, Polarization.TE)
            R_t, T_t = _tmm_reference(
                [n_air, n_glass], [], theta, wavelength, Polarization.TE
            )
            assert abs(R_s - R_t) < 1e-12, f"theta={theta}: R mismatch"
            assert abs(T_s - T_t) < 1e-12, f"theta={theta}: T mismatch"

    def test_off_normal_TM(self, set_numpy_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        for theta in [15, 30, 60]:
            R_s, T_s = _stratix_solve(stack, wavelength, theta, Polarization.TM)
            R_t, T_t = _tmm_reference(
                [n_air, n_glass], [], theta, wavelength, Polarization.TM
            )
            assert abs(R_s - R_t) < 1e-12, f"theta={theta}: R mismatch"
            assert abs(T_s - T_t) < 1e-12, f"theta={theta}: T mismatch"

    def test_metal_TE(self, set_numpy_backend):
        n_air, n_metal = 1.0, 0.05 + 3.5j
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_metal**2),
        )
        wavelength = 5e-7
        R_s, T_s = _stratix_solve(stack, wavelength, 0.0, Polarization.TE)
        R_t, T_t = _tmm_reference(
            [n_air, n_metal], [], 0.0, wavelength, Polarization.TE
        )
        assert abs(R_s - R_t) < 1e-12
        assert abs(T_s - T_t) < 1e-12

    def test_metal_TM(self, set_numpy_backend):
        n_air, n_metal = 1.0, 0.05 + 3.5j
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_metal**2),
        )
        wavelength = 5e-7
        R_s, T_s = _stratix_solve(stack, wavelength, 0.0, Polarization.TM)
        R_t, T_t = _tmm_reference(
            [n_air, n_metal], [], 0.0, wavelength, Polarization.TM
        )
        assert abs(R_s - R_t) < 1e-12
        assert abs(T_s - T_t) < 1e-12


class TestTmmMultiLayer:
    def test_ar_coating_TE(self, set_numpy_backend):
        n_air, n_mgf2, n_glass = 1.0, 1.38, 1.5
        wavelength = 5e-7
        d_ar = wavelength / (4 * n_mgf2)
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
            layers=[Layer(thickness=d_ar, material=Material(epsilon=n_mgf2**2))],
        )
        R_s, T_s = _stratix_solve(stack, wavelength, 0.0, Polarization.TE)
        R_t, T_t = _tmm_reference(
            [n_air, n_mgf2, n_glass], [d_ar], 0.0, wavelength, Polarization.TE
        )
        assert abs(R_s - R_t) < 1e-12
        assert abs(T_s - T_t) < 1e-12

    def test_ar_coating_TM(self, set_numpy_backend):
        n_air, n_mgf2, n_glass = 1.0, 1.38, 1.5
        wavelength = 5e-7
        d_ar = wavelength / (4 * n_mgf2)
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
            layers=[Layer(thickness=d_ar, material=Material(epsilon=n_mgf2**2))],
        )
        R_s, T_s = _stratix_solve(stack, wavelength, 0.0, Polarization.TM)
        R_t, T_t = _tmm_reference(
            [n_air, n_mgf2, n_glass], [d_ar], 0.0, wavelength, Polarization.TM
        )
        assert abs(R_s - R_t) < 1e-12
        assert abs(T_s - T_t) < 1e-12

    def test_two_layer_TE(self, set_numpy_backend):
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
        R_s, T_s = _stratix_solve(stack, wavelength, 0.0, Polarization.TE)
        R_t, T_t = _tmm_reference(
            [n_air, n_a, n_b, n_sub], [d_a, d_b], 0.0, wavelength, Polarization.TE
        )
        assert abs(R_s - R_t) < 1e-12
        assert abs(T_s - T_t) < 1e-12

    def test_two_layer_TM(self, set_numpy_backend):
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
        R_s, T_s = _stratix_solve(stack, wavelength, 0.0, Polarization.TM)
        R_t, T_t = _tmm_reference(
            [n_air, n_a, n_b, n_sub], [d_a, d_b], 0.0, wavelength, Polarization.TM
        )
        assert abs(R_s - R_t) < 1e-12
        assert abs(T_s - T_t) < 1e-12

    def test_bragg_mirror_TE(self, set_numpy_backend):
        n_low, n_high = 1.38, 2.3
        wavelength = 5e-7
        d_low = wavelength / (4 * n_low)
        d_high = wavelength / (4 * n_high)
        n_air, n_sub = 1.0, 1.5

        layers = []
        n_list = [n_air]
        d_list = []
        for _ in range(4):
            layers.append(Layer(thickness=d_high, material=Material(epsilon=n_high**2)))
            layers.append(Layer(thickness=d_low, material=Material(epsilon=n_low**2)))
            n_list.append(n_high)
            n_list.append(n_low)
            d_list.append(d_high)
            d_list.append(d_low)
        n_list.append(n_sub)

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=layers,
        )
        R_s, T_s = _stratix_solve(stack, wavelength, 0.0, Polarization.TE)
        R_t, T_t = _tmm_reference(n_list, d_list, 0.0, wavelength, Polarization.TE)
        assert abs(R_s - R_t) < 1e-12
        assert abs(T_s - T_t) < 1e-12

    def test_bragg_mirror_TM(self, set_numpy_backend):
        n_low, n_high = 1.38, 2.3
        wavelength = 5e-7
        d_low = wavelength / (4 * n_low)
        d_high = wavelength / (4 * n_high)
        n_air, n_sub = 1.0, 1.5

        layers = []
        n_list = [n_air]
        d_list = []
        for _ in range(4):
            layers.append(Layer(thickness=d_high, material=Material(epsilon=n_high**2)))
            layers.append(Layer(thickness=d_low, material=Material(epsilon=n_low**2)))
            n_list.append(n_high)
            n_list.append(n_low)
            d_list.append(d_high)
            d_list.append(d_low)
        n_list.append(n_sub)

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=layers,
        )
        R_s, T_s = _stratix_solve(stack, wavelength, 0.0, Polarization.TM)
        R_t, T_t = _tmm_reference(n_list, d_list, 0.0, wavelength, Polarization.TM)
        assert abs(R_s - R_t) < 1e-12
        assert abs(T_s - T_t) < 1e-12

    def test_off_normal_multilayer_TE(self, set_numpy_backend):
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
        for theta in [15, 30, 45]:
            R_s, T_s = _stratix_solve(stack, wavelength, theta, Polarization.TE)
            R_t, T_t = _tmm_reference(
                [n_air, n_a, n_b, n_sub],
                [d_a, d_b],
                theta,
                wavelength,
                Polarization.TE,
            )
            assert abs(R_s - R_t) < 1e-12, f"theta={theta}: R mismatch"
            assert abs(T_s - T_t) < 1e-12, f"theta={theta}: T mismatch"

    def test_off_normal_multilayer_TM(self, set_numpy_backend):
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
        for theta in [15, 30, 45]:
            R_s, T_s = _stratix_solve(stack, wavelength, theta, Polarization.TM)
            R_t, T_t = _tmm_reference(
                [n_air, n_a, n_b, n_sub],
                [d_a, d_b],
                theta,
                wavelength,
                Polarization.TM,
            )
            assert abs(R_s - R_t) < 1e-12, f"theta={theta}: R mismatch"
            assert abs(T_s - T_t) < 1e-12, f"theta={theta}: T mismatch"


def _tmm_layer_absorption(n_list, d_list, theta_deg, wavelength, pol):
    """Run tmm.absorp_in_each_layer and return the per-layer fractions."""
    inf = np.inf
    d_tmm = np.array([inf, *d_list, inf], dtype=float)
    n_tmm = np.array(n_list, dtype=complex)
    pol_tmm = "s" if pol == Polarization.TE else "p"
    theta_rad = np.pi * theta_deg / 180.0
    tmm_result = tmm.coh_tmm(pol_tmm, n_tmm, d_tmm, theta_rad, wavelength)
    return [float(a) for a in tmm.absorp_in_each_layer(tmm_result)[1:-1]]


class TestTmmLayerAbsorption:
    """Per-layer absorption against tmm's Poynting-flux reference."""

    def test_absorber_before_bragg_pair(self, set_numpy_backend):
        n_air = 1.0
        n_lossy = 1.5 + 0.3j
        n_high, n_low, n_sub = 2.3, 1.38, 1.5
        wavelength = 5e-7
        d_list = [40e-9, wavelength / (4 * n_high), wavelength / (4 * n_low)]
        n_list = [n_air, n_lossy, n_high, n_low, n_sub]
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[
                Layer(thickness=d_list[0], material=Material(epsilon=n_lossy**2)),
                Layer(thickness=d_list[1], material=Material(epsilon=n_high**2)),
                Layer(thickness=d_list[2], material=Material(epsilon=n_low**2)),
            ],
        )
        k0 = 2 * np.pi / wavelength
        for theta in [0, 20, 40, 60]:
            for pol in (Polarization.TE, Polarization.TM):
                kx = float(n_air * k0 * np.sin(np.pi * theta / 180))
                result = stratix.solve(
                    stack, wavelength, kx, pol, absorption=True
                )
                A_s = [float(a) for a in result.layer_absorption]
                A_t = _tmm_layer_absorption(
                    n_list, d_list, theta, wavelength, pol
                )
                for i, (a_s, a_t) in enumerate(zip(A_s, A_t, strict=True)):
                    assert abs(a_s - a_t) < 1e-12, (
                        f"theta={theta} {pol} layer {i}: {a_s} vs {a_t}"
                    )

    def test_two_absorbers(self, set_numpy_backend):
        n_air = 1.0
        n_a = 1.5 + 0.2j
        n_mid = 1.38
        n_b = 0.5 + 2.0j
        n_sub = 1.5
        wavelength = 5e-7
        d_list = [60e-9, 90e-9, 20e-9]
        n_list = [n_air, n_a, n_mid, n_b, n_sub]
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[
                Layer(thickness=d_list[0], material=Material(epsilon=n_a**2)),
                Layer(thickness=d_list[1], material=Material(epsilon=n_mid**2)),
                Layer(thickness=d_list[2], material=Material(epsilon=n_b**2)),
            ],
        )
        k0 = 2 * np.pi / wavelength
        for theta in [0, 35]:
            for pol in (Polarization.TE, Polarization.TM):
                kx = float(n_air * k0 * np.sin(np.pi * theta / 180))
                result = stratix.solve(
                    stack, wavelength, kx, pol, absorption=True
                )
                A_s = [float(a) for a in result.layer_absorption]
                A_t = _tmm_layer_absorption(
                    n_list, d_list, theta, wavelength, pol
                )
                for i, (a_s, a_t) in enumerate(zip(A_s, A_t, strict=True)):
                    assert abs(a_s - a_t) < 1e-12, (
                        f"theta={theta} {pol} layer {i}: {a_s} vs {a_t}"
                    )
