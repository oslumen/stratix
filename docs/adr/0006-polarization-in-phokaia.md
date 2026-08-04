# Polarization type moves to phokaia

`Polarization` (`TE`, `TM`, `BOTH`) moves from `stratix._types` to `phokaia`. It belongs with phokaia's domain types (Material, Layer, Stack, PlaneWave) — stratix consumes it.

`BOTH` moves too. It is not a solver hack — it is a legitimate polarization request ("compute both polarizations"). Any EM solver in the ecosystem can consume it.

No re-export from stratix. Consumers import `from phokaia import Polarization` directly. This is a deliberate clean break to keep ownership unambiguous.

**Considered options:** Moving to numdiff was rejected — numdiff is a pure numerical dispatch layer with zero domain concepts. Keeping it in stratix was rejected — phokaia already owns PlaneWave (which carries a polarization), and oslumen will eventually migrate to the same type.
