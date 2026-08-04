# ADR 0002: numdiff-only math

## Status

Accepted (2024-08-01)

## Context

stratix needs linear algebra (matrix multiply, solve, inverse) and special functions (sqrt, sin, exp) for its solver methods. Python has multiple array backends (numpy, jax, torch, autograd), each with different APIs and capabilities. Direct imports would lock the library to one backend, preventing autodiff (numpy), GPU execution (autograd), or JIT compilation.

## Decision

All array operations go through `numdiff` (`nd.linalg`, `nd.special`, `nd.grad`, `nd.jit`). stratix never imports `numpy`, `scipy`, `jax`, or `torch` directly for array operations. Tests run across all available numdiff backends; autodiff tests skip numpy backend (no grad support).

## Consequences

- Single codebase supports numpy, jax, torch, and autograd without conditional branches.
- Users get autodiff, JIT, and GPU acceleration transparently via numdiff backend switching.
- Complex numbers must be constructed with `nd.array(complex_value)` — direct Python complex literals in array expressions break under certain backends.
- Reference values for tests validated against numpy backend; other backends validated for consistency (same results, different backends).
