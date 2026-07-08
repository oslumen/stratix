# Domain docs

## Layout

**Single-context.** One `CONTEXT.md` at the repo root, and ADRs in `docs/adr/`.

```
/
├── CONTEXT.md          ← domain glossary
├── docs/
│   └── adr/            ← architectural decision records
│       ├── 0001-slug.md
│       └── 0002-slug.md
└── src/
```

## Consumer rules

Skills that read domain docs (`improve-codebase-architecture`, `diagnose`, `tdd`):

1. Read `CONTEXT.md` to learn the project's ubiquitous language — use its canonical terms throughout.
2. Scan `docs/adr/` for past architectural decisions before proposing changes — never contradict an existing ADR without explicit discussion.
3. If no `CONTEXT.md` exists, create one lazily when the first domain term is resolved.
4. `CONTEXT.md` is a glossary only — no implementation details, no specs, no scratch pad. One or two sentences per term + `_Avoid_` aliases.

## Dependencies

`stratix` depends on external packages that maintain their own domain docs:

- **phokaia** — `Material`, `Source`, `PlaneWave`, `Green's Function`, `Layer`, `Stack`. See phokaia's `CONTEXT.md` for their canonical definitions.
- **numdiff** — `nd.linalg`, `nd.special`, `nd.fft`, `nd.jit`, `nd.grad`. See numdiff's `CONTEXT.md` for backend terminology.

When a term is defined in a dependency's `CONTEXT.md`, reference it rather than redefining it in stratix's `CONTEXT.md`.
