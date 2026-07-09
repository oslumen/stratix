# PRD — stratix Phase 1: Isotropic Stratified Media Solver

## Problem Statement

A computational physicist or optical engineer wants to compute reflectance, transmittance, and per-layer absorption of a planar multilayer stack — a stack of isotropic materials with known permittivity/thickness — under plane-wave illumination at arbitrary wavelength and incidence angle. Existing Python implementations (`tmm`, `PyMoosh`, `TMMax`) each force a choice: NumPy-only (no autodiff), JAX-only (no GPU portability), or GUI-bound. There is no library that combines backend-agnostic numerical dispatch (`numpy`, `autograd`, `jax`, `torch`), autodiff-ready gradients, and a numerically stable S-matrix default in a clean, composable API built on a shared electromagnetic toolkit (`phokaia`).

`stratix` fills this gap as the first `oslumen` numerical-method consumer package, proving the `phokaia` + `numdiff` stack on a well-understood, high-value algorithm family.

## Solution

Users construct a `phokaia.Stack` (superstrate material, ordered layers with thickness and material, substrate material), call `stratix.solve()` with wavelength and in-plane wavevector arrays, and receive a `stratix.Result` containing reflectance, transmittance, per-layer absorption, and energy balance — computed with the numerically stable S-matrix method by default, with optional faster methods available. All computation is backend-agnostic through `numdiff`, enabling JIT compilation and automatic differentiation for inverse design workflows.

## User Stories

1. As a researcher designing an anti-reflection coating, I want to compute R and T vs. wavelength for a 2-layer stack, so that I can verify my design meets the target spec.
2. As a solar cell engineer, I want per-layer absorption spectra, so that I can optimize the active layer thickness for maximum photocurrent.
3. As a physicist studying surface plasmons, I want to compute reflectance vs. angle for a metal-dielectric stack in Kretschmann configuration, so that I can identify the plasmon resonance angle.
4. As a photonics student, I want a simple API that accepts an incidence angle and returns results, so that I can focus on the physics without converting to wavevectors.
5. As an optimization researcher, I want gradients of R/T with respect to layer thicknesses via autodiff, so that I can run gradient-based inverse design.
6. As a power user running large parameter sweeps, I want to compute R/T over arrays of wavelengths and kx values in one call, so that I get maximum throughput from vectorized computation.
7. As a developer extending stratix, I want to swap between S-matrix, Abélès, and admittance methods, so that I can trade stability for speed when the problem permits.
8. As a user validating my simulation, I want the energy balance (R + T + A) reported automatically, so that I can verify numerical convergence at a glance.
9. As a user debugging a stack, I want to compute the full electric field profile E(z) through the structure, so that I can visualize standing-wave patterns and field enhancement.
10. As a user who only needs R and T, I want to disable per-layer absorption computation, so that my sweeps run faster.
11. As a researcher working in both TE and TM polarization, I want to compute both in one call, so that I don't need to re-assemble the stack twice.
12. As a user with a complex multi-layer filter, I want the solver to default to the numerically stable S-matrix method, so that I get correct results even for thick or metallic stacks without worrying about method selection.

## Implementation Decisions

### Architecture

- **Package boundary.** `stratix` is a numerical-method consumer. Domain types (`Layer`, `Stack`) live in `phokaia`. Numerical dispatch (`nd.linalg`, `nd.jit`, `nd.grad`) lives in `numdiff`. `stratix` adds only solver logic and the `Result` type.
- **Public API surface.** Three entry points:
  - `solve(stack, wavelengths, kx, polarization, method=Method.AUTO, absorption=False)` — core kx-engine. Accepts scalar or array kx. Vectorized over Nλ × Nk.
  - `solve_angles(stack, wavelengths, angles, polarization, method=Method.AUTO, absorption=False)` — convenience. Converts θ → kx using superstrate material index.
  - `solve_from_source(stack, source: PlaneWave, method=Method.AUTO, absorption=False)` — single-shot convenience. Extracts ω, θ, polarization from a `phokaia.PlaneWave`.
- **Method enum.** `Method.SMATRIX`, `Method.ABELES`, `Method.ADMITTANCE`, `Method.DTN`. `Method.AUTO` resolves to `Method.SMATRIX`.
- **Polarization parameter.** `TE`, `TM`, or `BOTH`. For `BOTH`, result arrays have leading dimension 2 = [TE, TM].
- **Result type.** Pydantic model with fields: `R`, `T`, `energy_balance`, `layer_absorption` (if `absorption=True`), `wavelengths`, `kx`, `polarization`, `method_used`. All ndarray fields.
- **Field profiles.** Separate function `compute_field_profile(result, z_positions)` returning E(z), H(z) arrays. Not part of `solve()` hot path.
- **Stack traversal.** Layers ordered superstrate → substrate (top-to-bottom). S-matrix assembled left-to-right.
- **Time convention.** `exp(-iωt)`. Documented convention, not enforced by code.

### Stack data structure (in `phokaia`)

- `Layer`: Pydantic model. Fields: `thickness: float` (>0), `material: Material`. Immutable.
- `Stack`: Pydantic model. Fields: `superstrate: Material`, `substrate: Material`, `layers: list[Layer]`. Immutable. Layers ordered incidence-to-exit.

### Math constraints

- All array operations use `numdiff` (`nd.linalg.solve`, `nd.linalg.inv`, matrix multiplication). No direct `numpy`, `scipy`, `jax`, or `torch` imports in stratix source.
- `nd.jit` wrapper available for JIT-compiling the core solve loop.
- `nd.grad` available for autodiff through the S-matrix assembly.
- Tests run across all available numdiff backends. Autodiff tests skip numpy backend (no grad support).

### Module structure

```
stratix/
├── methods/
│   ├── _smatrix.py      # S-matrix assembly (default)
│   ├── _abeles.py       # Abélès formalism
│   ├── _admittance.py   # Admittance formalism
│   └── _dtn.py          # Dirichlet-to-Neumann
├── _solve.py            # Core solve() orchestration
├── _result.py           # Result Pydantic model
├── _field.py            # compute_field_profile()
├── _types.py            # Method enum, Polarization enum
```

## Testing Decisions

### Test principles

Tests validate external behavior only — never internal method implementation. Reference values come from analytic formulas (single-interface Fresnel, thin-film formulas) and from the established `tmm` library (sbyrnes321/tmm). Tests run across all available numdiff backends. Reference values validated against numpy backend; other backends validated for consistency. Autodiff tests skip numpy backend (no grad support).

### Seams

| Seam | Location | What to test | Reference |
|------|----------|-------------|-----------|
| Layer/Stack data model | phokaia tests | Pydantic validation, construction, ordering | N/A (data model) |
| S-matrix method | stratix unit | Single-interface R/T vs analytic Fresnel | Fresnel equations |
| S-matrix method | stratix unit | 2-layer AR coating R/T vs analytic formula | Thin-film formula |
| S-matrix method | stratix unit | Bragg mirror R/T asymptote vs known result | Literature / tmm |
| Abélès method | stratix unit | Agreement with S-matrix in stable regime (no metal, <150 layers) | S-matrix result |
| Admittance method | stratix unit | Agreement with S-matrix for dielectric stacks | S-matrix result |
| solve() integration | stratix integ | R, T agreement with `tmm` library for representative stacks | sbyrnes321/tmm |
| solve() integration | stratix integ | Energy balance R+T+ΣA ≈ 1.0 (within 1e-12) | Conservation law |
| solve() integration | stratix integ | TE+TM polarization both match | Fresnel / tmm |
| solve_angles() | stratix integ | θ input produces same result as manual kx | Equivalent kx call |
| solve_from_source() | stratix integ | PlaneWave extraction yields same result | Equivalent solve() call |
| compute_field_profile() | stratix integ | E(z) continuity at interfaces | Analytic field solution for 1-layer |
| Absorption | stratix integ | Per-layer A sums to 1 - R - T | Conservation law |

### Prior art

Existing stratix test structure: `tests/test_package.py` (pytest, smoke tests), `tests/conftest.py` (fixtures). Extend with `tests/test_stack.py` (phokaia data model), `tests/test_methods.py` (method unit tests), `tests/test_solver.py` (integration tests).

## Out of Scope

- Anisotropic / Berreman 4×4 matrices
- RCWA (rigorous coupled-wave analysis)
- Guided and leaky mode computation
- Optimization / inverse design tooling (user calls `nd.grad` directly)
- Photovoltaic-specific pre/postprocessing (AM1.5 spectra, IQE/EQE, JV curves)
- Incoherent thick-substrate summation
- GPU-specific code paths (provided transparently by `numdiff` backend switching)
- GUI or interactive interface

## Further Notes

- `Layer` and `Stack` must be implemented in `phokaia` before `stratix` can begin. This is a blocking dependency.
- The `numdiff` dependency is already declared transitively via `phokaia` but should be explicit in `stratix`'s `pyproject.toml`.
- A future Phase 2 may add: anisotropic 4×4, guided mode solver, incoherent summation.
- The `biblio/survey.md` already exists as the implementation survey. Reference it for algorithm details and edge cases.
