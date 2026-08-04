# ADR 0004: Test strategy

## Status

Accepted (2024-08-01)

## Context

stratix's solver methods compute reflectance and transmittance. These must be validated against known-correct results. Multiple validation strategies exist: analytic formulas, cross-method comparison, and external reference implementations.

## Decision

Four-tier test strategy:

1. **Analytic Fresnel** — Single-interface R/T validated against closed-form Fresnel equations for TE and TM polarization across multiple angles.
2. **Cross-method agreement** — Abélès, Admittance, and DTN results validated against S-matrix in their stable regimes (dielectric stacks, no metal, moderate thicknesses).
3. **External reference** — Integration tests against the established `tmm` library (sbyrnes321/tmm) for representative stacks covering single-interface, multi-layer, Bragg mirrors, metals, and off-normal incidence.
4. **Backend consistency** — All tests run across numpy, jax, torch, and autograd backends. Numpy backend provides reference values; other backends validated for consistent results.

Autodiff tests (gradients of R/T w.r.t. layer thicknesses) skip numpy backend since numpy has no autodiff support.

## Consequences

- Full test suite requires `tmm` as an optional dev dependency.
- Cross-method tests catch regressions in individual method implementations.
- Backend matrix tests detect numdiff dispatch bugs.
- No single test tier is sufficient alone — they complement each other.
