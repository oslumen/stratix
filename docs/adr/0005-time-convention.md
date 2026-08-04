# ADR 0005: Time convention

## Status

Accepted (2024-08-01)

## Context

Electromagnetic wave propagation involves complex exponentials. The sign convention determines how lossy materials are represented in the complex permittivity. Two conventions are common: `exp(-iωt)` (physics/optics) and `exp(+iωt)` (engineering/EE). Mixing conventions leads to sign errors in phase and incorrect loss/gain interpretation.

## Decision

stratix uses the `exp(-iωt)` time convention. This is the physics/optics standard — lossy materials have positive imaginary permittivity (`Im(ε) > 0`). The convention is documented but not enforced in code — the user is responsible for providing materials with the correct permittivity sign.

## Consequences

- Consistent with the wider oslumen ecosystem and numdiff conventions.
- Users coming from EE (where `exp(+iωt)` is common) must negate their imaginary permittivity sign.
- Future anisotropic or dispersive material models will use this convention.
- phokaia `Material` model does not enforce sign — it accepts any complex permittivity.
