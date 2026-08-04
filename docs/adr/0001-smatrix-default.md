# ADR 0001: S-matrix as default method

## Status

Accepted (2024-08-01)

## Context

The solver needs a default method when the user does not specify one (`Method.AUTO`). The choice affects numerical stability, speed, and correctness across the full range of problems (thin dielectrics, thick stacks, metallic layers, evanescent incidence).

- Abélès (2×2 characteristic matrix) is fast but suffers exponential blow-up for thick or metallic layers due to the mixing of decaying and growing exponentials in the transfer matrix.
- S-matrix (scattering matrix) separates forward and backward waves, maintaining numerical stability regardless of layer thickness or material loss. The cost is slightly higher memory (storing 2×2 blocks per layer) but this is negligible.
- Admittance and DTN methods are alternative stable formulations but less widely used.

## Decision

`Method.AUTO` resolves to `Method.SMATRIX`. The S-matrix is always numerically stable. Users who need speed may opt into `Method.ABELES`, `Method.ADMITTANCE`, or `Method.DTN` explicitly.

## Consequences

- Default path is correct-by-default — no silent failures on thick stacks.
- S-matrix overhead is acceptable: each layer adds one 2×2 matrix multiply.
- Users must read docs or source to discover alternative methods.
- Testing matrix: cross-method tests validate that alternative methods agree with S-matrix in their stable regimes.
