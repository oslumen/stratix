"""Tests for the component-wise S-matrix hot path (Issue #52).

The stable default method must not pay for bookkeeping the caller did not
ask for: no per-layer matrices are retained, and the Redheffer product
runs on the four scalar S-matrix components rather than on stacked 2x2
arrays.
"""

from __future__ import annotations

import numdiff as nd
import pytest
from phokaia import Layer
from phokaia import Material
from phokaia import Polarization
from phokaia import Stack

import stratix
from stratix.methods._smatrix import _propagation_star
from stratix.methods._smatrix import _redheffer_star
from stratix.methods._util import _interface_coeffs

#: Keys the S-matrix solver is allowed to hand back.  Every entry is a
#: per-medium or whole-stack quantity the solve computed anyway; nothing
#: is accumulated per layer.  ``no_flux`` is one boolean per sweep point,
#: not per layer: absorption has to cut off at exactly the sweep points R
#: and T do, and carrying the mask is cheaper than recomputing it.
LEAN_INTERMEDIATE_KEYS = {
    "k0",
    "kzs",
    "denom_vals",
    "no_flux",
    "r_total",
    "t_total",
    "thicknesses",
    "polarization",
}


def _bragg_stack(n_layers: int = 6) -> Stack:
    layers = [
        Layer(
            thickness=1e-6,
            material=Material(epsilon=2.25 if i % 2 == 0 else 12.25),
        )
        for i in range(n_layers)
    ]
    return Stack(
        superstrate=Material(epsilon=1.0),
        substrate=Material(epsilon=1.0),
        layers=layers,
    )


def _dense_star(
    S_A: list[list[complex]], S_B: list[list[complex]]
) -> list[list[complex]]:
    """Reference Redheffer star product written out on 2x2 nested lists."""
    (A11, A12), (A21, A22) = S_A
    (B11, B12), (B21, B22) = S_B
    denom = 1.0 - A22 * B11
    return [
        [A11 + A12 * B11 * A21 / denom, A12 * B12 / denom],
        [B21 * A21 / denom, B22 + B21 * A22 * B12 / denom],
    ]


class TestRedhefferComponents:
    def test_star_matches_dense_reference(self, set_backend):
        """The component-wise star reproduces the 2x2 matrix formula."""
        S_A = [[0.3 + 0.1j, 0.7 - 0.2j], [0.6 + 0.4j, -0.3 - 0.1j]]
        S_B = [[-0.2 + 0.5j, 0.8 + 0.1j], [0.4 - 0.3j, 0.2 + 0.5j]]

        got = _redheffer_star(
            (
                nd.array(S_A[0][0]),
                nd.array(S_A[0][1]),
                nd.array(S_A[1][0]),
                nd.array(S_A[1][1]),
            ),
            (
                nd.array(S_B[0][0]),
                nd.array(S_B[0][1]),
                nd.array(S_B[1][0]),
                nd.array(S_B[1][1]),
            ),
        )
        expected = _dense_star(S_A, S_B)

        assert len(got) == 4
        flat = [expected[0][0], expected[0][1], expected[1][0], expected[1][1]]
        for value, ref in zip(got, flat, strict=True):
            assert abs(complex(value) - ref) < 1e-12

    def test_propagation_star_matches_generic_star(self, set_backend):
        """The propagation fast path equals the generic star product."""
        S_A = [[0.3 + 0.1j, 0.7 - 0.2j], [0.6 + 0.4j, -0.3 - 0.1j]]
        p = 0.5 - 0.6j
        S_P = [[0.0 + 0.0j, p], [p, 0.0 + 0.0j]]

        components = (
            nd.array(S_A[0][0]),
            nd.array(S_A[0][1]),
            nd.array(S_A[1][0]),
            nd.array(S_A[1][1]),
        )
        got = _propagation_star(components, nd.array(p))
        expected = _dense_star(S_A, S_P)

        flat = [expected[0][0], expected[0][1], expected[1][0], expected[1][1]]
        for value, ref in zip(got, flat, strict=True):
            assert abs(complex(value) - ref) < 1e-12

    def test_interface_coeffs_are_scalars(self, set_backend):
        """Interface data comes back as four components, not a stacked array."""
        Z_left, Z_right = nd.array(1.0 + 0j), nd.array(2.5 + 0j)
        r, t_rev, t_fwd, minus_r = _interface_coeffs(Z_left, Z_right)

        assert abs(complex(r) - (1.0 - 2.5) / 3.5) < 1e-12
        assert abs(complex(t_fwd) - 2.0 / 3.5) < 1e-12
        assert abs(complex(t_rev) - 5.0 / 3.5) < 1e-12
        assert abs(complex(minus_r) + complex(r)) < 1e-12
        assert nd.asarray(r).ndim == 0


class TestNoPerLayerRetention:
    @pytest.mark.parametrize("absorption", [False, True])
    def test_intermediates_are_lean(self, set_backend, absorption):
        """No per-layer S-matrices are retained, with or without absorption."""
        stack = _bragg_stack()
        result = stratix.solve(
            stack,
            nd.linspace(400e-9, 800e-9, 5),
            kx=0.0,
            polarization=Polarization.TE,
            absorption=absorption,
        )

        assert set(result.intermediates) == LEAN_INTERMEDIATE_KEYS

    def test_both_polarization_intermediates_are_lean(self, set_backend):
        """The BOTH path keeps each polarization's lean intermediates."""
        stack = _bragg_stack()
        result = stratix.solve(stack, 5e-7, kx=0.0, polarization=Polarization.BOTH)

        assert set(result.intermediates) == {"te", "tm"}
        for pol in ("te", "tm"):
            assert set(result.intermediates[pol]) == LEAN_INTERMEDIATE_KEYS


class TestResultsUnchanged:
    def test_r_t_match_abeles_on_a_deep_stack(self, set_backend):
        """The rewritten product still agrees with the Abélès solver."""
        stack = _bragg_stack(n_layers=20)
        wavelengths = nd.linspace(400e-9, 800e-9, 32)
        kx = nd.linspace(0.0, 5e6, 4)

        smat = stratix.solve(stack, wavelengths, kx=kx, polarization=Polarization.TM)
        abeles = stratix.solve(
            stack,
            wavelengths,
            kx=kx,
            polarization=Polarization.TM,
            method=stratix.Method.ABELES,
        )

        assert float(nd.max(nd.abs(smat.R - abeles.R))) < 1e-10
        assert float(nd.max(nd.abs(smat.T - abeles.T))) < 1e-10

    def test_field_profile_still_available(self, set_backend):
        """Field profiles are reconstructed from the lean intermediates."""
        stack = _bragg_stack()
        result = stratix.solve(stack, 5e-7, kx=0.0, polarization=Polarization.TE)

        z = nd.linspace(-1e-6, 7e-6, 41)
        field = stratix.compute_field_profile(result, z)

        assert nd.asarray(field["E"]).shape == (1, 1, 41)
        assert float(nd.max(nd.abs(field["E"]))) > 0.0
