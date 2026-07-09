# AGENTS.md — stratix

## Hard rules

- **numdiff for all math.** `stratix` never imports `numpy`, `scipy`, `jax`, or `torch` directly for array operations. All linear algebra, special functions, autodiff, and JIT go through `numdiff` (`nd.linalg`, `nd.special`, `nd.grad`, `nd.jit`).
- **Layer and Stack live in phokaia.** `stratix` consumes `phokaia.Layer` (thickness + `Material`) and `phokaia.Stack` (superstrate, substrate, ordered layers). `stratix` does not define its own layer or stack types.
- **S-matrix is AUTO default.** `Method.AUTO` resolves to `SMATRIX` — always numerically stable. `ABELES`, `ADMITTANCE`, `DTN` require explicit user opt-in.

## Agent skills

### Issue tracker

GitHub Issues at `oslumen/stratix`. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at repo root. See `docs/agents/domain.md`.

## Quick reminders

- Time convention: `exp(-iωt)` (documented, not enforced).
- Layer ordering: superstrate → substrate (top-to-bottom).
- All public types are Pydantic models.
- Tests run across all available numdiff backends. Autodiff tests skip numpy backend (no grad support).
