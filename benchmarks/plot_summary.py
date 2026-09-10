"""
Cross-stack scaling overview
=============================

Aggregated summary: runs all four lossless stack sizes on the default
backend and produces a scaling plot plus a 4-stack × 4-backend timing
matrix.  Demonstrates how solve time grows with layer count and
compares vectorised (50×50 sweep) against scalar (single-point) cost.
"""

# %%
# Setup
# -----
# We compare four stacks — bare interface (single), 1-film (small),
# 3-film (medium), 8-film (large) — each with a 50×50 wavelength–kx
# sweep at TE polarisation, plus a scalar baseline for each.

import matplotlib.pyplot as plt
import numdiff as nd
from phokaia import Polarization

from benchmarks._backends import available_backends
from benchmarks._backends import backend_scope
from benchmarks._plotting import bar_chart_compare
from benchmarks._plotting import summary_table
from benchmarks._stacks import large_stack
from benchmarks._stacks import medium_stack
from benchmarks._stacks import single_stack
from benchmarks._stacks import small_stack
from benchmarks._timing import time_solve
from stratix import Method

stacks = {
    "bare": single_stack(),
    "small (1 film)": small_stack(),
    "medium (3 films)": medium_stack(),
    "large (8 films)": large_stack(),
}

wavelengths = nd.linspace(400e-9, 800e-9, 50)
kx_vals = nd.linspace(0, 1e7, 50)

backends = available_backends()
canonical = ("numpy", "autograd", "jax", "torch")
skipped = [b for b in canonical if b not in backends]
print(f"Available backends: {backends}")
if skipped:
    print(f"Skipped (not installed): {skipped}")

# %%
# 1. Default-backend scaling (vectorised vs scalar)
# -------------------------------------------------
# Each stack size is solved on the default backend (SMATRIX).  Two
# lines show: the 50×50 vectorised sweep (solid) and a scalar
# single-point reference (dashed).  The gap between them reveals the
# vectorisation advantage — smaller gaps mean the backend is already
# efficient at processing scalar calls, while large gaps show the
# benefit of batched inputs.

default_backend = backends[0]
scalar_wl = 500e-9
scalar_kx = 0.0

scaling_sweep: dict[str, float] = {}
scaling_scalar: dict[str, float] = {}
with backend_scope(default_backend):
    for label, s in stacks.items():
        t_sweep = time_solve(
            s,
            wavelengths,
            kx=kx_vals,
            polarization=Polarization.TE,
            method=Method.SMATRIX,
        )
        t_scalar = time_solve(
            s,
            scalar_wl,
            kx=scalar_kx,
            polarization=Polarization.TE,
            method=Method.SMATRIX,
        )
        scaling_sweep[label] = t_sweep["mean"]
        scaling_scalar[label] = t_scalar["mean"]

stack_labels = list(stacks.keys())

fig, ax = plt.subplots(figsize=(7, 4))
x = range(len(stack_labels))
ax.plot(x, [scaling_sweep[label] for label in stack_labels], "o-", label="50×50 sweep")
ax.plot(
    x,
    [scaling_scalar[label] * 2500 for label in stack_labels],
    "s--",
    label="scalar × 2500 (equiv. evals)",
)
ax.set_xticks(x)
ax.set_xticklabels(stack_labels, rotation=15, ha="right")
ax.set_ylabel("Solve time (s)")
ax.set_title(f"Scaling: vectorised vs scalar — {default_backend} backend, SMATRIX")
ax.legend()
ax.set_yscale("log")
plt.show()

# %%
# 2. Method scaling (default backend)
# -----------------------------------
# All four methods on the default backend sweep.  S-matrix and Abélès
# should track closely; DTN and Admittance may diverge at larger stack
# sizes due to different recurrence patterns.

METHODS = [Method.SMATRIX, Method.ABELES, Method.ADMITTANCE, Method.DTN]
method_scaling: dict[str, dict[str, float]] = {}
with backend_scope(default_backend):
    for m in METHODS:
        method_scaling[m.value] = {}
        for label, s in stacks.items():
            t = time_solve(
                s,
                wavelengths,
                kx=kx_vals,
                polarization=Polarization.TE,
                method=m,
            )
            method_scaling[m.value][label] = t["min"]

fig, ax = plt.subplots(figsize=(7, 4))
for mn in [m.value for m in METHODS]:
    ax.plot(x, [method_scaling[mn][label] for label in stack_labels], "o-", label=mn)
ax.set_xticks(x)
ax.set_xticklabels(stack_labels, rotation=15, ha="right")
ax.set_ylabel("Solve time (s)")
ax.set_title(f"Method scaling — {default_backend} backend, 50×50 sweep")
ax.legend()
ax.set_yscale("log")
plt.show()

# %%
# 3. 4 stacks × all backends timing matrix
# -----------------------------------------
# Full cross-product table showing which backends were available at
# build time.  Unavailable backends are marked with ``—``.

scaling: dict[str, dict[str, float]] = {}
for b in backends:
    scaling[b] = {}
    with backend_scope(b):
        for label, s in stacks.items():
            t = time_solve(
                s,
                wavelengths,
                kx=kx_vals,
                polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
            scaling[b][label] = t["min"]

table_cols = ["stack", *canonical]
table_data: dict[str, list[str]] = {"stack": list(stacks.keys())}
for b in canonical:
    if b in backends:
        table_data[b] = [f"{scaling[b][label]:.3g}" for label in stack_labels]
    else:
        table_data[b] = ["—"] * len(stack_labels)

summary_table(
    table_data,
    rows=[f"  {label}" for label in stack_labels],
    cols=table_cols,
)

# %%
# 4. Cross-backend scaling
# -------------------------
# Each stack size is solved on every available backend (SMATRIX).  The
# grouped bar chart shows how execution time grows with layer count
# across backends.

bar_chart_compare(
    {b: [scaling[b][label] for label in stack_labels] for b in backends},
    stack_labels,
    title="Scaling: solve time vs stack size across backends (SMATRIX)",
    ylabel="Solve time (s)",
)

# %%
# 5. Scaling behaviour
# --------------------
# As the number of layers increases the solve time grows roughly
# linearly with stack depth for S-matrix and Abélès methods, while
# DTN and Admittance may show different growth characteristics due to
# their recurrence patterns.
#
# The vectorised-vs-scalar comparison shows that batched sweeps
# amortise overhead — a 50×50 sweep (2500 evaluation points) costs far
# less than 2500 individual scalar calls.  JIT compilation (opt-in via
# ``nd.jit(solve)``) amplifies this advantage further because compile
# cost is paid once for the entire sweep.
#
# The default NumPy backend shows the most predictable scaling since
# there is no JIT compilation overhead.  For JAX and PyTorch, apply
# ``nd.jit(solve)`` explicitly when sweeping; the warm-up call alone
# (without JIT) only removes first-call overhead from eager mode.

print(f"Default backend: {default_backend}")
print(f"Available: {backends}")
if skipped:
    print(f"Skipped: {skipped}")
print("All times measured at TE polarisation.")
