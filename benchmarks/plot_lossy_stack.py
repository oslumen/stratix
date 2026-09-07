"""
Lossy-stack sweep benchmark
============================

Five-layer Bragg-like stack with weak absorption (complex epsilon).
The 50×50 wavelength–kx sweep exercises solvers with complex refractive
indices.  Correctness is assessed via cross-backend agreement since
R + T < 1 in the presence of absorption.
"""

# %%
# Setup
# -----
# Five alternating layers with epsilon = 2.25 + 0.001j and
# 12.25 + 0.01j (weak absorption).  50 wavelengths (400–800 nm) ×
# 50 kx values (0–1e7 rad/m).
#
# .. note::
#    stratix does not yet expose per-layer absorption via the public
#    ``Result``.  The ``absorption=True`` path is under development.
#    Cross-backend agreement on R and T confirms correctness here.

import timeit

import numdiff as nd
from phokaia import Polarization

from benchmarks._backends import available_backends
from benchmarks._backends import backend_scope
from benchmarks._plotting import bar_chart_compare
from benchmarks._plotting import heatmap
from benchmarks._plotting import summary_table
from benchmarks._stacks import lossy_stack
from benchmarks._timing import check_correctness
from benchmarks._timing import measure_memory
from benchmarks._timing import time_solve
from benchmarks._tmm import _tmm_available
from benchmarks._tmm import tmm_compare
from stratix import Method
from stratix import solve

stack = lossy_stack()
wavelengths = nd.linspace(400e-9, 800e-9, 50)
kx_vals = nd.linspace(0, 1e7, 50)

METHODS = [Method.SMATRIX, Method.ABELES, Method.ADMITTANCE, Method.DTN]

# %%
# 1. Backend comparison
# ---------------------

backends = available_backends()
backend_times: dict[str, float] = {}
backend_mems: dict[str, float] = {}

for b in backends:
    with backend_scope(b):
        t = time_solve(
            stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
            method=Method.SMATRIX,
        )
        m = measure_memory(
            stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
            method=Method.SMATRIX,
        )
    backend_times[b] = t["min"]
    backend_mems[b] = m["peak_delta_mb"]

bar_chart_compare(
    {"min time (s)": [backend_times[b] for b in backends]},
    backends,
    title="Backend comparison — SMATRIX, lossy 5-film 50×50 sweep",
    ylabel="Solve time (s)",
)

summary_table(
    {
        "backend": backends,
        "time (s)": [f"{backend_times[b]:.3g}" for b in backends],
        "mem Δ (MB)": [f"{backend_mems[b]:.2f}" for b in backends],
    },
    rows=[f"  {b}" for b in backends],
    cols=["backend", "time (s)", "mem Δ (MB)"],
)

# %%
# 2. Method comparison
# --------------------

method_times: dict[str, float] = {}
with backend_scope("numpy"):
    for m in METHODS:
        t = time_solve(
            stack, wavelengths, kx=kx_vals, polarization=Polarization.TE, method=m
        )
        method_times[m.value] = t["min"]

method_names = [m.value for m in METHODS]
bar_chart_compare(
    {"min time (s)": [method_times[n] for n in method_names]},
    method_names,
    title="Method comparison — NumPy backend, lossy 5-film 50×50 sweep",
    ylabel="Solve time (s)",
)

summary_table(
    {
        "method": method_names,
        "time (s)": [f"{method_times[n]:.3g}" for n in method_names],
    },
    rows=[f"  {n}" for n in method_names],
    cols=["method", "time (s)"],
)

# %%
# 3. Method × backend heatmap
# ---------------------------

heatmap_data: list[list[float]] = []
for m in METHODS:
    row: list[float] = []
    for b in backends:
        with backend_scope(b):
            t = time_solve(
                stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
                method=m,
            )
        row.append(t["min"])
    heatmap_data.append(row)

heatmap(
    heatmap_data,
    row_labels=[m.value for m in METHODS],
    col_labels=backends,
    title="Method × backend solve time (s) — lossy 5-film 50×50 sweep",
)

# %%
# 4. User-side JIT
# ----------------
# :func:`solve` can be JIT-compiled directly via
# ``nd.jit(solve, static_argnames=(...))``.  Non-array arguments
# (``stack``, ``polarization``, ``method``, ``absorption``,
# ``thicknesses``) are declared static; only ``wavelength`` and ``kx``
# are traced.  The ``Result`` return value (including enum fields in
# the ``intermediates`` dict) is now JAX-traceable thanks to automatic
# pytree registration of :class:`Polarization` and :class:`Method`.
#
# This section compares NumPy (eager baseline), JAX/Torch eager, and
# JIT-compiled timing side by side.  The warm-up call triggers tracing
# and compilation so the measured repeats reflect steady-state
# compiled performance.

STATIC_ARGS: tuple[str, ...] = (
    "stack",
    "polarization",
    "method",
    "absorption",
    "thicknesses",
)

jit_backends = [b for b in ("jax", "torch") if b in backends]
if jit_backends:
    with backend_scope("numpy"):
        _ = solve(
            stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
            method=Method.SMATRIX,
        )
        t_numpy = time_solve(
            stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
            method=Method.SMATRIX,
            jit=False,
        )

    jit_eager: dict[str, float] = {}
    jit_compiled: dict[str, float] = {}
    for b in jit_backends:
        with backend_scope(b):
            _ = solve(
                stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
            t_eager = time_solve(
                stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
                method=Method.SMATRIX,
                jit=False,
            )

            jit_solve_fn = nd.jit(  # type: ignore[assignment]
                solve, static_argnames=STATIC_ARGS,
            )
            _ = jit_solve_fn(
                stack, wavelengths, kx=kx_vals,
                polarization=Polarization.TE, method=Method.SMATRIX,
            )
            timer = timeit.Timer(
                lambda fn=jit_solve_fn: fn(
                    stack, wavelengths, kx=kx_vals,
                    polarization=Polarization.TE, method=Method.SMATRIX,
                )
            )
            times = timer.repeat(repeat=10, number=1)
            jit_compiled[b] = float(min(times))

        jit_eager[b] = t_eager["min"]

    numpy_eager = t_numpy["min"]
    chart_backends = ["numpy"] + jit_backends
    bar_chart_compare(
        {
            "numpy (eager)": [numpy_eager] + [0.0] * len(jit_backends),
            "eager (warm-up)": [0.0] + [jit_eager[b] for b in jit_backends],
            "jit-compiled": [0.0] + [jit_compiled[b] for b in jit_backends],
        },
        chart_backends,
        title="User-side JIT — SMATRIX, lossy 5-film 50×50 sweep",
        ylabel="Solve time (s)",
    )
else:
    print("JIT backends (jax, torch) not available — skipping JIT section.")

# %%
# 5. GPU acceleration
# -------------------

if nd.HAS_CUDA:
    gpu_backends = [b for b in ("jax", "torch") if b in backends]
    gpu_cpu: dict[str, float] = {}
    gpu_gpu: dict[str, float] = {}
    for b in gpu_backends:
        with backend_scope(b, gpu=False):
            _ = solve(
                stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
            t_cpu = time_solve(
                stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
        gpu_cpu[b] = t_cpu["mean"]
        with backend_scope(b, gpu=True):
            _ = solve(
                stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
            t_gpu = time_solve(
                stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
        gpu_gpu[b] = t_gpu["mean"]

    bar_chart_compare(
        {
            "CPU": [gpu_cpu[b] for b in gpu_backends],
            "GPU": [gpu_gpu[b] for b in gpu_backends],
        },
        gpu_backends,
        title="GPU vs CPU — SMATRIX, lossy 5-film 50×50 sweep",
        ylabel="Solve time (s)",
    )
else:
    print(
        "CUDA not detected on this machine. GPU comparison omitted. "
        "JAX and PyTorch run on CPU only."
    )

# %%
# 6. Cross-backend correctness + tmm
# -----------------------------------
# With absorption (complex epsilon) we cannot rely on R + T = 1.
# We verify cross-backend agreement (all backends produce the same
# R and T as the NumPy reference), then cross-check against tmm at a
# representative single point.
#
# Mean R + T ≈ 0.92 confirms absorption — about 8% of incident power is
# absorbed by the lossy layers.

with backend_scope("numpy"):
    ref = solve(
        stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
        method=Method.SMATRIX,
    )

max_errors: dict[str, float] = {}
for b in backends:
    if b == "numpy":
        continue
    with backend_scope(b):
        test = solve(
            stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
            method=Method.SMATRIX,
        )
    err = check_correctness(ref, test, tolerance=1e-8)
    max_errors[b] = err

R_sum = float(nd.array(ref.R).mean())
T_sum = float(nd.array(ref.T).mean())
print(f"Mean R + T = {R_sum + T_sum:.6f} (< 1 expected due to absorption)")
print("Cross-backend agreement (max rel err vs numpy):")
for b, err in max_errors.items():
    print(f"  {b}: {err:.2e}")
print()

if _tmm_available():
    cmp = tmm_compare(
        stack, 500e-9, kx=0.0, polarization=Polarization.TE, method=Method.SMATRIX
    )
    print(f"R — stratix: {cmp['R_stratix']:.8f}  tmm: {cmp['R_tmm']:.8f}  err: {cmp['R_err']:.2e}")
    print(f"T — stratix: {cmp['T_stratix']:.8f}  tmm: {cmp['T_tmm']:.8f}  err: {cmp['T_err']:.2e}")
    if cmp["R_err"] < 1e-10 and cmp["T_err"] < 1e-10:
        print("✓ stratix agrees with tmm within tolerance.")
    else:
        print("✗ stratix–tmm discrepancy exceeds tolerance!")
else:
    print("tmm not installed — skipping tmm cross-check.")
