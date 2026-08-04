"""Stack definitions matching vtmm benchmark configurations.

Each function returns a :class:`phokaia.Stack` with the exact refractive
indices and thicknesses used in vtmm's ``test_benchmark.py``.
"""

from __future__ import annotations

from phokaia import Layer
from phokaia import Material
from phokaia import Stack


def _m(n: float) -> Material:
    return Material(epsilon=n**2)


def single_stack() -> Stack:
    """Bare interface between air and a high-index medium.

    Matches vtmm single-omega/kx benchmark: ``n = [1.0, 3.5, 1.0]``,
    ``d = []`` (no layers).
    """
    return Stack(superstrate=_m(1.0), substrate=_m(3.5), layers=[])


def small_stack() -> Stack:
    """One high-index layer between air media.

    Matches vtmm single/small benchmark: ``n = [1.0, 3.5, 1.0]``,
    ``d = [1e-6]``.
    """
    return Stack(
        superstrate=_m(1.0),
        substrate=_m(1.0),
        layers=[Layer(thickness=1e-6, material=_m(3.5))],
    )


def medium_stack() -> Stack:
    """Three-layer stack with alternating indices.

    Matches vtmm medium benchmark: ``n = [1.0, 1.5, 3.5, 1.5, 1.0]``,
    ``d = [1e-6, 1e-6, 1e-6]``.
    """
    return Stack(
        superstrate=_m(1.0),
        substrate=_m(1.0),
        layers=[
            Layer(thickness=1e-6, material=_m(1.5)),
            Layer(thickness=1e-6, material=_m(3.5)),
            Layer(thickness=1e-6, material=_m(1.5)),
        ],
    )


def large_stack() -> Stack:
    """Eight-layer stack with varied indices and thicknesses.

    Matches vtmm large benchmark:
    ``n = [1.0, 1.5, 3.5, 1.5, 2.5, 3.0, 1.5, 2.0, 3.0, 1.0]``,
    ``d = [1e-6, 1.33e-6, 1e-6, 1e-6, 2e-6, 1e-5, 1.25e-6, 1e-6]``.
    """
    return Stack(
        superstrate=_m(1.0),
        substrate=_m(1.0),
        layers=[
            Layer(thickness=1e-6, material=_m(1.5)),
            Layer(thickness=1.33e-6, material=_m(3.5)),
            Layer(thickness=1e-6, material=_m(1.5)),
            Layer(thickness=1e-6, material=_m(2.5)),
            Layer(thickness=2e-6, material=_m(3.0)),
            Layer(thickness=1e-5, material=_m(1.5)),
            Layer(thickness=1.25e-6, material=_m(2.0)),
            Layer(thickness=1e-6, material=_m(3.0)),
        ],
    )


def lossy_stack() -> Stack:
    """Five-layer stack with absorptive (complex-epsilon) layers.

    Uses a 5-layer Bragg-like stack where alternating layers have a
    small imaginary index component to model weak absorption.
    """
    return Stack(
        superstrate=_m(1.0),
        substrate=_m(1.0),
        layers=[
            Layer(thickness=1e-6, material=Material(epsilon=2.25 + 0.001j)),
            Layer(thickness=1e-6, material=Material(epsilon=12.25 + 0.01j)),
            Layer(thickness=1e-6, material=Material(epsilon=2.25 + 0.001j)),
            Layer(thickness=1e-6, material=Material(epsilon=12.25 + 0.01j)),
            Layer(thickness=1e-6, material=Material(epsilon=2.25 + 0.001j)),
        ],
    )
