"""Cross-method agreement on absorbing stacks — Issue #59.

For a lossless stack the Abélès phase-sign error conjugates ``r`` and
``t`` and leaves R and T unchanged, so a lossless-only cross-method suite
cannot see this class of bug.  A single absorbing layer makes the layer
phase complex and turns a conjugated phase into gain: R + T comes out
above 1.  These tests pin every non-default method to the S-matrix
reference on an absorbing stack, at normal and off-normal incidence, for
both polarizations.
"""

from __future__ import annotations

import pytest
from phokaia import Layer
from phokaia import Material
from phokaia import Polarization
from phokaia import Stack

import stratix
from stratix._types import Method

_WL = 5e-7


def _absorbing_stack() -> Stack:
    """Thin metallic film between air and glass — the issue #59 reproducer."""
    return Stack(
        superstrate=Material(epsilon=1.0),
        substrate=Material(epsilon=1.5**2),
        layers=[Layer(thickness=50e-9, material=Material(epsilon=(0.05 + 3.5j) ** 2))],
    )


class TestAbsorbingLayerAgreement:
    @pytest.mark.parametrize("method", [Method.ABELES, Method.ADMITTANCE, Method.DTN])
    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    @pytest.mark.parametrize("kx", [0.0, 5e6])
    def test_matches_smatrix(self, set_backend, method, pol, kx):
        stack = _absorbing_stack()
        ref = stratix.solve(stack, _WL, kx, pol, method=Method.SMATRIX)
        res = stratix.solve(stack, _WL, kx, pol, method=method)

        assert abs(float(res.R[0, 0]) - float(ref.R[0, 0])) < 1e-12, (
            f"{method.name} R mismatch at kx={kx}"
        )
        assert abs(float(res.T[0, 0]) - float(ref.T[0, 0])) < 1e-12, (
            f"{method.name} T mismatch at kx={kx}"
        )

    @pytest.mark.parametrize("method", [Method.ABELES, Method.ADMITTANCE, Method.DTN])
    @pytest.mark.parametrize("pol", [Polarization.TE, Polarization.TM])
    def test_no_gain_from_an_absorbing_stack(self, set_backend, method, pol):
        """R + T + Σ(absorption) never exceeds 1: absorption is not gain."""
        res = stratix.solve(
            _absorbing_stack(), _WL, 0.0, pol, method=method, absorption=True
        )
        R, T = float(res.R[0, 0]), float(res.T[0, 0])
        assert R + T <= 1.0 + 1e-12, f"{method.name}: R + T = {R + T}"
        # The lumped absorption term is 1 - R - T for the non-default
        # methods, so it doubles as the sum of the absorption fields.
        total = R + T + float(sum(res.layer_absorption[:, 0, 0]))
        assert total <= 1.0 + 1e-12, f"{method.name}: R + T + A = {total}"
        assert float(res.layer_absorption[0, 0, 0]) >= -1e-12, (
            f"{method.name}: negative absorption"
        )
