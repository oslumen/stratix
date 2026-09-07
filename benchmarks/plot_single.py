"""
Scalar single-point benchmark
==============================

Single wavelength and single kx value with a 1-film lossless stack.
Establishes the six-subsection pattern used by all benchmark files:
backend comparison, method comparison, heatmap, steady-state timing,
GPU (when available), and tmm cross-check.
"""

# %%
# Setup
# -----
# We use a 1 µm air / high-index (n=3.5) / air stack at normal incidence
# (kx=0) with a single vacuum wavelength of 500 nm in the visible.

import numdiff as nd
from phokaia import Polarization

from benchmarks._backends import available_backends
from benchmarks._backends import backend_scope
from benchmarks._plotting import bar_chart_compare
from benchmarks._plotting import heatmap
from benchmarks._plotting import summary_table
from benchmarks._stacks import small_stack
from benchmarks._timing import measure_memory
from benchmarks._timing import time_solve
from benchmarks._tmm import _tmm_available
from benchmarks._tmm import tmm_compare
from stratix import Method
from stratix import solve

stack = small_stack()
wavelength = 500e-9
kx = 0.0

METHODS = [Method.SMATRIX, Method.ABELES, Method.ADMITTANCE, Method.DTN]

# %%
# 1. Backend comparison
# ---------------------
# All available backends, fixed method=SMATRIX.  Timing is measured with
# 10 repeats; the bar chart shows min solve time and the summary table
# adds memory peak delta.

backends = available_backends()
backend_times: dict[str, float] = {}
backend_mems: dict[str, float] = {}

for b in backends:
    with backend_scope(b):
        t = time_solve(
            stack, wavelength, kx=kx, polarization=Polarization.TE,
            method=Method.SMATRIX,
        )
        m = measure_memory(
            stack, wavelength, kx=kx, polarization=Polarization.TE,
            method=Method.SMATRIX,
        )
    backend_times[b] = t["min"]
    backend_mems[b] = m["peak_delta_mb"]

bar_chart_compare(
    {"min time (s)": [backend_times[b] for b in backends]},
    backends,
    title="Backend comparison — SMATRIX, 1-film stack",
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
# All four solver methods, fixed backend=numpy.  S-matrix should be
# identical to Abélès for this lossless normal-incidence case; DTN and
# Admittance provide alternative numerical formulations.

method_times: dict[str, float] = {}
with backend_scope("numpy"):
    for m in METHODS:
        t = time_solve(
            stack, wavelength, kx=kx, polarization=Polarization.TE, method=m
        )
        method_times[m.value] = t["min"]

method_names = [m.value for m in METHODS]
bar_chart_compare(
    {"min time (s)": [method_times[n] for n in method_names]},
    method_names,
    title="Method comparison — NumPy backend, 1-film stack",
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
# Timing matrix across all four methods and all available backends.

heatmap_data: list[list[float]] = []
for m in METHODS:
    row: list[float] = []
    for b in backends:
        with backend_scope(b):
            t = time_solve(
                stack, wavelength, kx=kx, polarization=Polarization.TE, method=m
            )
        row.append(t["min"])
    heatmap_data.append(row)

heatmap(
    heatmap_data,
    row_labels=[m.value for m in METHODS],
    col_labels=backends,
    title="Method × backend solve time (s) — 1-film stack",
)

# %%
# 4. Steady-state (warm-up excluded)
# ---------------------------------
# Eager-mode solve timing with a warm-up call to exclude first-call
# overhead (tracing, memory allocation).  JIT is opt-in — apply
# ``nd.jit(solve)`` for compiled performance on sweep workloads.

steady_backends = [b for b in ("jax", "torch") if b in backends]
if steady_backends:
    steady_times: dict[str, float] = {}
    for b in steady_backends:
        with backend_scope(b):
            _ = solve(
                stack, wavelength, kx=kx, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
            t_steady = time_solve(
                stack, wavelength, kx=kx, polarization=Polarization.TE,
                method=Method.SMATRIX,
                jit=False,
            )
        steady_times[b] = t_steady["min"]

    bar_chart_compare(
        {"steady-state (warm-up excluded)": [steady_times[b] for b in steady_backends]},
        steady_backends,
        title="Steady-state (warm-up excluded) — SMATRIX, 1-film stack",
        ylabel="Solve time (s)",
    )
else:
    print("JIT backends (jax, torch) not available — skipping steady-state section.")

# %%
# 5. GPU acceleration
# -------------------
# Compare CPU vs GPU timings for JAX and PyTorch backends when CUDA is
# available.  The GPU bar is omitted when CUDA is not detected.

if nd.HAS_CUDA:
    gpu_backends = [b for b in ("jax", "torch") if b in backends]
    gpu_cpu: dict[str, float] = {}
    gpu_gpu: dict[str, float] = {}
    for b in gpu_backends:
        with backend_scope(b, gpu=False):
            _ = solve(
                stack, wavelength, kx=kx, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
            t_cpu = time_solve(
                stack, wavelength, kx=kx, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
        gpu_cpu[b] = t_cpu["mean"]
        with backend_scope(b, gpu=True):
            _ = solve(
                stack, wavelength, kx=kx, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
            t_gpu = time_solve(
                stack, wavelength, kx=kx, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
        gpu_gpu[b] = t_gpu["mean"]

    bar_chart_compare(
        {
            "CPU": [gpu_cpu[b] for b in gpu_backends],
            "GPU": [gpu_gpu[b] for b in gpu_backends],
        },
        gpu_backends,
        title="GPU vs CPU — SMATRIX, 1-film stack",
        ylabel="Solve time (s)",
    )
else:
    print(
        "CUDA not detected on this machine. GPU comparison omitted. "
        "JAX and PyTorch run on CPU only."
    )

# %%
# 6. Comparison against tmm
# --------------------------
# Cross-check stratix (SMATRIX, NumPy) against the `tmm` reference
# package at normal incidence.  Both reflectance and transmittance must
# agree to within 1e-10 relative tolerance.

if _tmm_available():
    cmp = tmm_compare(
        stack, wavelength, kx=kx, polarization=Polarization.TE, method=Method.SMATRIX
    )

    print(f"R — stratix: {cmp['R_stratix']:.8f}  tmm: {cmp['R_tmm']:.8f}  err: {cmp['R_err']:.2e}")
    print(f"T — stratix: {cmp['T_stratix']:.8f}  tmm: {cmp['T_tmm']:.8f}  err: {cmp['T_err']:.2e}")

    if cmp["R_err"] < 1e-10 and cmp["T_err"] < 1e-10:
        print("✓ stratix agrees with tmm within tolerance.")
    else:
        print("✗ stratix–tmm discrepancy exceeds tolerance!")
else:
    print("tmm not installed — skipping tmm cross-check.")
