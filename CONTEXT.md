# CONTEXT.md — stratix

## Domain Glossary

**S-matrix (scattering matrix)** — Numerically stable transfer-matrix variant that separates forward/backward waves. Avoids exponential blow-up in thick or metallic layers. The default method when `Method.AUTO` is used.

**Abélès matrix** — 2×2 characteristic matrix formalism. Faster than S-matrix for thin dielectric stacks but numerically unstable for thick/metallic layers.

**Admittance method** — Optical-admittance recursion. Equivalent to Abélès for dielectric stacks but uses a different algebraic path.

**DTN (Dirichlet-to-Neumann)** — The one method that does not walk the stack. Each layer's DtN map sends the tangential field at its two faces to the derivative data there, `λ = iY·[[cotφ, −cscφ], [cscφ, −cotφ]]` with `φ = kz·d`. Requiring the tangential `H` to be continuous at every interface assembles those maps into one tridiagonal Poincaré–Steklov system whose unknowns are the interface field amplitudes themselves; the superstrate and substrate radiation conditions close it, and `r = u₀ − 1`, `t = u_N` are read off the ends after a single solve — no second pass to recover the transmitted amplitude. See _Dirichlet resonance_ for the one place it cannot be trusted.

**Dirichlet resonance** — A layer whose phase thickness `kz·d` is a multiple of π has no DtN map: `cot` and `csc` are both infinite, because the interior problem with prescribed face values is singular there. The physics is ordinary — a half-wave layer is an absentee layer — so DTN scales each assembled row by the phase factors `p = exp(i·phi)` of the layers it touches, which clears every denominator exactly and leaves the resonance as `p²−1 → 0` rather than as an entry going to infinity. Carrying the scaling on `p` rather than on `sinφ` also keeps a thick absorbing layer finite, since `sin` and `cos` both grow like `exp(Im φ)/2` and overflow. Rows are then equilibrated by their absolute row sum and solved with a pivoted `nd.linalg.solve` rather than an elimination straight down the diagonal, which would divide by a pivot the scaling drove to nothing. *On* the resonance the scaled operator is genuinely rank deficient — the interior resonance of the formulation, not an artefact of the assembly — and the solve raises rather than returning anything, so such a layer is rotated off the resonance by `sqrt(eps)` in phase, about a femtometre of thickness. That offset balances the perturbation it introduces (order `delta`) against the conditioning it avoids (`eps/delta`), capping the error through a resonant neighbourhood at about `1e-8` while everything else holds to `1e-16`. The `phi = 0` case is a zero-thickness layer, which is where an optimiser goes whenever it wants a layer gone. The S-matrix has no such thickness, which is one more reason `Method.AUTO` resolves to it. Resolved in issue #54.

**Layer** (from phokaia) — A thickness + Material pair. Top-to-bottom ordering: superstrate → layers[0] → … → substrate.

**Stack** (from phokaia) — superstrate Material, ordered Layer list, substrate Material. Immutable Pydantic model.

**Polarization** (from phokaia) — `TE` (s-polarized, E-field out of incidence plane), `TM` (p-polarized, H-field out of incidence plane), or `BOTH` (compute both in one call). Defined in phokaia as `phokaia.Polarization`.

**Reflectance (R), Transmittance (T)** — Ratios of time-averaged z-directed power flux, both normalised by the incident flux `Re(kz0/denom0)` in the superstrate. `R = |r|²` is evaluated at the first interface. `T` is the fraction of incident power delivered *across the last interface* into the substrate — not the power reaching infinity, which is zero whenever the substrate absorbs. Both assume a propagating incident wave; see _Lossless-superstrate precondition_.

**Energy balance** — `R + T + Σ(layer_absorption) ≈ 1`. Reflectance + transmittance + per-layer absorption must sum to unity. Holds for a lossy *substrate* and for lossy *layers* (whose absorption the sum counts), but not for a lossy superstrate.

**Lossless-superstrate precondition** — R and T partition energy only when the incident medium is lossless. The superstrate carries both the incident and the reflected wave; in a lossless medium their cross term contributes no real z-directed flux, but in a lossy one it does, and neither `|r|²` nor `T` accounts for it. `R + T + ΣA` then measures 1.00082 at `Im(ε_super) = 0.01` and 1.124 at `Im(ε_super) = 1`, at normal incidence. `R` also becomes reference-plane dependent, since a detector at distance `d` sees `|r|²` attenuated by `exp(-2·Im(kz0)·d)`. Past the superstrate light line it degrades further: `T` diverges as `1/Im(ε_super)` while `|r|²` stays finite. A lossy *substrate* is unaffected — it is semi-infinite with a single outgoing wave, so its flux is unambiguous and energy balance closes exactly. The precondition is enforced: `solve()` raises `ValueError` when the superstrate's `epsilon` or `mu` has a non-zero imaginary part, gain included, rather than returning numbers that quietly fail to partition energy. A superstrate with real but negative `epsilon`/`mu` is still accepted. The check reads a value, so it cannot run on a tracer: every eager call is checked, but a solve compiled with `nd.jit` may not be — inside a trace numdiff returns a tracer for every array it builds, constants included. A lossy superstrate can therefore reach a compiled solve and come back with the numbers that do not partition energy. Breaking `solve()`'s trace-safety (#50) to close that costs more than the gap does. Resolved in issue #57.

**Light line** — `|kx| = Re(n_super)·k0`, the boundary between propagating and evanescent incidence, where `kz0² = ε_super·μ_super·k0² − kx²` changes sign. At or past it no power crosses the first interface, so `R = 1`, `T = 0`, and every layer absorbs nothing. This is the criterion the incident-flux guard uses, in place of a floating-point tolerance on the residual flux: comparing two wavevectors puts the cut-off at the same physical place at every working precision, whereas a `sqrt(eps)` tolerance moved it with the dtype. A metallic superstrate (`ε·μ < 0`) has `Re(n_super) = 0`, so every `kx` including normal incidence is past its light line.

**kx** — In-plane wavevector component. Determines incidence angle: `kx = n_inc * k0 * sin(θ)`.

**Result** — NamedTuple returned by `solve()`. Fields: `R`, `T`, `energy_balance`, `layer_absorption` (if `absorption=True`), `wavelengths`, `kx`, `polarization`, `method_used`.

**Shape contract** — `R` and `T` are always `(Nλ, Nk)`. A scalar wavelength or kx counts as a length-1 axis, so a scalar solve returns `(1, 1)`; there are no special cases. `energy_balance` carries the same `(Nλ, Nk)` shape and `layer_absorption` prepends the layer axis as `(n_layers, Nλ, Nk)`. `Polarization.BOTH` prepends a TE/TM axis of size 2 to all four. `wavelengths` and `kx` stay 1-D, naming the sweep coordinates rather than following the grid. The one exception is `solve_angles()` over an array of wavelengths: `kx = n(ω)·k0·sin θ` then depends on the wavelength as well as the angle, so its `kx` follows the grid as `(Nλ, n_angles)`. Field profiles from `compute_field_profile()` are `(Nλ, Nk, Nz)`; a `BOTH` Result carries both polarizations' intermediates, so its profiles prepend the same TE/TM axis, giving `(2, Nλ, Nk, Nz)`.

## Avoid

- _reflectivity, transmissivity_ → use _reflectance_, _transmittance_
- _TMX, TMY_ → use _TE_, _TM_
- _numpy array, jax array, torch tensor_ → use _ndarray_ (numdiff-dispatched)
