# stratix

Stratified Media Electromagnetic Solver

[![PyPI version](https://img.shields.io/pypi/v/stratix)](https://pypi.org/project/stratix/)
[![Python version](https://img.shields.io/pypi/pyversions/stratix)](https://pypi.org/project/stratix/)
[![License](https://img.shields.io/pypi/l/stratix)](https://pypi.org/project/stratix/)
[![CI](https://img.shields.io/github/actions/workflow/status/oslumen/stratix/test.yml)](https://github.com/oslumen/stratix/actions)
[![Codecov](https://img.shields.io/codecov/c/github/oslumen/stratix)](https://codecov.io/gh/oslumen/stratix)

## Install

```bash
uv pip install stratix
```

## Features

- **Backend-agnostic solver** — Compute reflectance, transmittance, and per-layer absorption for planar multilayer stacks. All linear algebra dispatched through `numdiff`, supporting numpy, JAX, PyTorch, and autograd backends.
- **Four solution methods** — Numerically stable S-matrix (default), Abélès 2×2 characteristic matrix, admittance formalism, and Dirichlet-to-Neumann (DTN). Method.AUTO resolves to S-matrix.
- **Autodiff-ready** — Gradients of R/T with respect to layer thicknesses via `nd.grad` for inverse design workflows.
- **Vectorized sweeps** — Nλ × Nk wavelength and wavevector arrays in a single call.
- **Dual polarization** — TE, TM, or BOTH in one call.
- **Per-layer absorption** — Optical absorption per layer with energy balance R+T+ΣA ≈ 1.
- **Field profiles** — Compute E(z) and H(z) through the stack with `compute_field_profile()`.
- **Convenience APIs** — `solve_angles()` for angle input and `solve_from_source()` for `phokaia.PlaneWave` sources.
- **Structured logging** — loguru-based logger scoped to the package: silent by default, opt in via `set_log_level()`; importing stratix never touches the host application's logging configuration.
- **Validated** — Tests against analytic Fresnel formulas, cross-method agreement, and the `tmm` reference library.

## Quick start

```python
from phokaia import Layer, Material, Stack
import stratix

stack = Stack(
    superstrate=Material(epsilon=1.0),       # air
    substrate=Material(epsilon=2.25),         # glass (n=1.5)
    layers=[
        Layer(thickness=90e-9, material=Material(epsilon=1.9044)),  # MgF₂ AR
    ],
)

result = stratix.solve(stack, wavelength=500e-9, kx=0.0, polarization=stratix.Polarization.TE)
# R and T are always (Nlambda, Nk); scalar inputs are length-1 axes.
print(f"R = {result.R[0, 0]:.6f}, T = {result.T[0, 0]:.6f}")
```

## Docs

See the [full documentation](https://stratix.readthedocs.io) for installation and API reference.

## License

This software is published under the GNU General Public License v3 (GPL-3.0).
