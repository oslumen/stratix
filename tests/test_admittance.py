"""Tests for Admittance recursion method — Issue #22."""

from __future__ import annotations

import numdiff as nd
import pytest
from phokaia import Layer
from phokaia import Material
from phokaia import Polarization
from phokaia import Stack

import stratix
from stratix._types import Method


def _smatrix_ref(stack, wavelength, kx, polarization):
    return stratix.solve(stack, wavelength, kx, polarization, method=Method.SMATRIX)


def _admittance_solve(stack, wavelength, kx, polarization):
    return stratix.solve(stack, wavelength, kx, polarization, method=Method.ADMITTANCE)


class TestAdmittanceSingleInterface:
    def test_matches_smatrix_air_glass(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _admittance_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12
        assert res.method_used == Method.ADMITTANCE

    def test_matches_smatrix_air_silicon(self, set_backend):
        n_air, n_si = 1.0, 3.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_si**2),
        )
        wavelength = 5e-7
        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _admittance_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_off_normal(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        k0 = 2 * nd.pi / wavelength

        for theta_deg in [15, 30, 60]:
            theta_rad = nd.array(nd.pi * theta_deg / 180)
            kx = float(n_air * k0 * nd.sin(theta_rad))
            ref = _smatrix_ref(stack, wavelength, kx=kx, polarization=Polarization.TE)
            res = _admittance_solve(
                stack, wavelength, kx=kx, polarization=Polarization.TE
            )

            assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12, (
                f"theta={theta_deg}: R mismatch"
            )
            assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12, (
                f"theta={theta_deg}: T mismatch"
            )

    def test_matches_smatrix_total_internal_reflection(self, set_backend):
        n_glass, n_air = 1.5, 1.0
        stack = Stack(
            superstrate=Material(epsilon=n_glass**2),
            substrate=Material(epsilon=n_air**2),
        )
        wavelength = 5e-7
        k0 = 2 * nd.pi / wavelength
        kx = 1.1 * n_air * k0
        ref = _smatrix_ref(
            stack, wavelength, kx=float(kx), polarization=Polarization.TE
        )
        res = _admittance_solve(
            stack, wavelength, kx=float(kx), polarization=Polarization.TE
        )

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12


class TestAdmittanceMultiLayer:
    def test_matches_smatrix_ar_coating_TE(self, set_backend):
        n_air, n_mgf2, n_glass = 1.0, 1.38, 1.5
        wavelength = 5e-7
        d_ar = wavelength / (4 * n_mgf2)

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
            layers=[Layer(thickness=d_ar, material=Material(epsilon=n_mgf2**2))],
        )

        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _admittance_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_ar_coating_TM(self, set_backend):
        n_air, n_mgf2, n_glass = 1.0, 1.38, 1.5
        wavelength = 5e-7
        d_ar = wavelength / (4 * n_mgf2)

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
            layers=[Layer(thickness=d_ar, material=Material(epsilon=n_mgf2**2))],
        )

        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TM)
        res = _admittance_solve(stack, wavelength, kx=0.0, polarization=Polarization.TM)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_two_layer(self, set_backend):
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

        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _admittance_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_bragg_mirror_TE(self, set_backend):
        n_low, n_high = 1.38, 2.3
        wavelength = 5e-7
        d_low = wavelength / (4 * n_low)
        d_high = wavelength / (4 * n_high)
        n_air, n_sub = 1.0, 1.5

        layers = []
        for _ in range(5):
            layers.append(Layer(thickness=d_high, material=Material(epsilon=n_high**2)))
            layers.append(Layer(thickness=d_low, material=Material(epsilon=n_low**2)))

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=layers,
        )

        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _admittance_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_bragg_mirror_TM(self, set_backend):
        n_low, n_high = 1.38, 2.3
        wavelength = 5e-7
        d_low = wavelength / (4 * n_low)
        d_high = wavelength / (4 * n_high)
        n_air, n_sub = 1.0, 1.5

        layers = []
        for _ in range(5):
            layers.append(Layer(thickness=d_high, material=Material(epsilon=n_high**2)))
            layers.append(Layer(thickness=d_low, material=Material(epsilon=n_low**2)))

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=layers,
        )

        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TM)
        res = _admittance_solve(stack, wavelength, kx=0.0, polarization=Polarization.TM)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12

    def test_matches_smatrix_off_normal_multilayer(self, set_backend):
        n_air, n_a, n_b, n_sub = 1.0, 1.38, 2.0, 1.5
        wavelength = 5e-7
        d_a = 100e-9
        d_b = 200e-9

        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_sub**2),
            layers=[
                Layer(thickness=d_a, material=Material(epsilon=n_a**2)),
                Layer(thickness=d_b, material=Material(epsilon=n_b**2)),
                Layer(thickness=50e-9, material=Material(epsilon=n_a**2)),
            ],
        )

        for kx in [0.0, 5e6, 1e7]:
            ref = _smatrix_ref(stack, wavelength, kx=kx, polarization=Polarization.TE)
            res = _admittance_solve(
                stack, wavelength, kx=kx, polarization=Polarization.TE
            )

            assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12, (
                f"kx={kx}: R mismatch"
            )
            assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12, (
                f"kx={kx}: T mismatch"
            )

    def test_matches_smatrix_zero_layers(self, set_backend):
        n_air, n_glass = 1.0, 1.5
        stack = Stack(
            superstrate=Material(epsilon=n_air**2),
            substrate=Material(epsilon=n_glass**2),
        )
        wavelength = 5e-7
        ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=Polarization.TE)
        res = _admittance_solve(stack, wavelength, kx=0.0, polarization=Polarization.TE)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12


class TestAdmittanceThickAbsorber:
    """The downward pass must not amplify through an absorber — Issue #60.

    The transmitted amplitude used to accumulate ``cos(phi)`` and
    ``sin(phi)`` per layer, both of which grow like ``exp(Im phi)`` for an
    absorbing layer.  The growing exponential was retained instead of
    cancelling, so T diverged with thickness — 4e158 at 5 um of metal —
    where the physics decays to zero.
    """

    _METAL = (0.05 + 3.5j) ** 2

    def _stack(self, thickness):
        return Stack(
            superstrate=Material(epsilon=1.0),
            substrate=Material(epsilon=2.25),
            layers=[Layer(thickness=thickness, material=Material(epsilon=self._METAL))],
        )

    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_T_matches_smatrix_and_decays(self, set_backend, pol):
        wavelength = 5e-7
        previous = None
        for thickness in (1e-6, 2e-6, 5e-6):
            stack = self._stack(thickness)
            ref = _smatrix_ref(stack, wavelength, kx=0.0, polarization=pol)
            res = _admittance_solve(stack, wavelength, kx=0.0, polarization=pol)

            T_ref, T_adm = float(ref.T[0, 0]), float(res.T[0, 0])
            assert T_adm == pytest.approx(T_ref, rel=1e-10), (
                f"d={thickness}: T={T_adm} vs smatrix {T_ref}"
            )
            assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12

            if previous is not None:
                assert T_adm < previous, f"T not decreasing at d={thickness}"
            previous = T_adm

    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_far_past_the_naive_overflow_threshold(self, set_backend, pol):
        """20 um of metal: cos/sin overflow outright, exp(i*phi) cannot."""
        import math

        wavelength = 5e-7
        res = _admittance_solve(
            self._stack(20e-6), wavelength, kx=0.0, polarization=pol
        )
        T = float(res.T[0, 0])
        assert math.isfinite(T)
        assert 0.0 <= T <= 1e-300

        res_abs = stratix.solve(
            self._stack(20e-6),
            wavelength,
            0.0,
            pol,
            method=Method.ADMITTANCE,
            absorption=True,
        )
        balance = float(res_abs.energy_balance[0, 0])
        assert math.isfinite(balance)
        assert balance == pytest.approx(1.0, abs=1e-9)
