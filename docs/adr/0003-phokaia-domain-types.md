# ADR 0003: phokaia domain types

## Status

Accepted (2024-08-01)

## Context

stratix needs `Layer` (thickness + material), `Stack` (superstrate, layers, substrate), and `PlaneWave` (frequency, angle, medium) types. These are electromagnetic domain types shared across the oslumen ecosystem. Defining them in stratix would create duplication and prevent reuse by other solver packages (e.g., RCWA, FDTD, BEM).

## Decision

`Layer`, `Stack`, `Material`, and `PlaneWave` live in `phokaia`. stratix imports them as dependencies. stratix does not define its own layer or stack types. `phokaia` is a required dependency declared in `pyproject.toml`.

## Consequences

- stratix source contains only solver logic and the `Result` type — clean separation of concerns.
- Changes to domain types (e.g., adding anisotropy fields) happen once in phokaia and propagate to all consumers.
- stratix cannot be used without phokaia installed. This is acceptable since phokaia is the intentional electromagnetic foundation layer.
- `Layer`, `Stack`, and `Material` are immutable Pydantic models. Testing their validation and construction happens in phokaia tests, not stratix.
