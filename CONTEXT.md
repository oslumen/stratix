# CONTEXT.md — stratix

## Domain Glossary

**S-matrix (scattering matrix)** — Numerically stable transfer-matrix variant that separates forward/backward waves. Avoids exponential blow-up in thick or metallic layers. The default method when `Method.AUTO` is used.

**Abélès matrix** — 2×2 characteristic matrix formalism. Faster than S-matrix for thin dielectric stacks but numerically unstable for thick/metallic layers.

**Admittance method** — Optical-admittance recursion. Equivalent to Abélès for dielectric stacks but uses a different algebraic path.

**DTN (Dirichlet-to-Neumann)** — Maps tangential E-fields to H-fields layer by layer. An alternative stable method that avoids intermediate transfer matrices.

**Layer** (from phokaia) — A thickness + Material pair. Top-to-bottom ordering: superstrate → layers[0] → … → substrate.

**Stack** (from phokaia) — superstrate Material, ordered Layer list, substrate Material. Immutable Pydantic model.

**Polarization** (from phokaia) — `TE` (s-polarized, E-field out of incidence plane), `TM` (p-polarized, H-field out of incidence plane), or `BOTH` (compute both in one call). Defined in phokaia as `phokaia.Polarization`.

**Energy balance** — `R + T + Σ(layer_absorption) ≈ 1`. Reflectance + transmittance + per-layer absorption must sum to unity for lossless stacks.

**kx** — In-plane wavevector component. Determines incidence angle: `kx = n_inc * k0 * sin(θ)`.

**Result** — NamedTuple returned by `solve()`. Fields: `R`, `T`, `energy_balance`, `layer_absorption` (if `absorption=True`), `wavelengths`, `kx`, `polarization`, `method_used`.

**Shape contract** — `R` and `T` are always `(Nλ, Nk)`. A scalar wavelength or kx counts as a length-1 axis, so a scalar solve returns `(1, 1)`; there are no special cases. `energy_balance` carries the same `(Nλ, Nk)` shape and `layer_absorption` prepends the layer axis as `(n_layers, Nλ, Nk)`. `Polarization.BOTH` prepends a TE/TM axis of size 2 to all four. `wavelengths` and `kx` stay 1-D, naming the sweep coordinates rather than following the grid. Field profiles from `compute_field_profile()` are `(Nλ, Nk, Nz)`; a `BOTH` Result carries only its TE intermediates, so field profiles are TE-only there and gain no polarization axis.

## Avoid

- _reflectivity, transmissivity_ → use _reflectance_, _transmittance_
- _TMX, TMY_ → use _TE_, _TM_
- _numpy array, jax array, torch tensor_ → use _ndarray_ (numdiff-dispatched)
