# CONTEXT.md — stratix

## Domain Glossary

**S-matrix (scattering matrix)** — Numerically stable transfer-matrix variant that separates forward/backward waves. Avoids exponential blow-up in thick or metallic layers. The default method when `Method.AUTO` is used.

**Abélès matrix** — 2×2 characteristic matrix formalism. Faster than S-matrix for thin dielectric stacks but numerically unstable for thick/metallic layers.

**Admittance method** — Optical-admittance recursion. Equivalent to Abélès for dielectric stacks but uses a different algebraic path.

**DTN (Dirichlet-to-Neumann)** — Maps tangential E-fields to H-fields layer by layer. An alternative stable method that avoids intermediate transfer matrices.

**Layer** (from phokaia) — A thickness + Material pair. Top-to-bottom ordering: superstrate → layers[0] → … → substrate.

**Stack** (from phokaia) — superstrate Material, ordered Layer list, substrate Material. Immutable Pydantic model.

**Polarization** — `TE` (s-polarized, E-field out of incidence plane) or `TM` (p-polarized, H-field out of incidence plane). `BOTH` computes both in one call.

**Energy balance** — `R + T + Σ(layer_absorption) ≈ 1`. Reflectance + transmittance + per-layer absorption must sum to unity for lossless stacks.

**kx** — In-plane wavevector component. Determines incidence angle: `kx = n_inc * k0 * sin(θ)`.

**Result** — Pydantic model returned by `solve()`. Fields: `R`, `T`, `energy_balance`, `layer_absorption` (if `absorption=True`), `wavelengths`, `kx`, `polarization`, `method_used`.

## Avoid

- _reflectivity, transmissivity_ → use _reflectance_, _transmittance_
- _TMX, TMY_ → use _TE_, _TM_
- _numpy array, jax array, torch tensor_ → use _ndarray_ (numdiff-dispatched)
