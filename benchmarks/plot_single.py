"""
Scalar single-point benchmark
==============================

Single wavelength and single kx value with a 1-film lossless stack.
Establishes the six-subsection pattern used by all benchmark files:
backend comparison, method comparison, heatmap, JIT acceleration,
GPU (when available), and tmm cross-check.
"""

# %%
# Setup
# -----
# We use a 1 µm air / high-index (n=3.5) / air stack at normal incidence
# (kx=0) with a single vacuum wavelength of 500 nm in the visible.

import matplotlib.pyplot as plt
import numdiff as nd
from phokaia import Polarization

from stratix import Method
from stratix import solve

from benchmarks._backends import available_backends
from benchmarks._backends import backend_scope
from benchmarks._plotting import bar_chart_compare
from benchmarks._plotting import heatmap
from benchmarks._plotting import summary_table
from benchmarks._stacks import small_stack
from benchmarks._timing import check_correctness
from benchmarks._timing import measure_memory
from benchmarks._timing import time_solve
from benchmarks._tmm import _tmm_available
from benchmarks._tmm import tmm_compare

stack = small_stack()
wavelength = 500e-9
kx = 0.0

METHODS = [Method.SMATRIX, Method.ABELES, Method.ADMITTANCE, Method.DTN]

# %%
# 1. Backend comparison
# ---------------------
# All available backends, fixed method=SMATRIX.  Timing is measured with
# 10 repeats; the bar chart shows mean solve time and the summary table
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
    backend_times[b] = t["mean"]
    backend_mems[b] = m["peak_delta_mb"]

bar_chart_compare(
    {"mean time (s)": [backend_times[b] for b in backends]},
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
        method_times[m.value] = t["mean"]

method_names = [m.value for m in METHODS]
bar_chart_compare(
    {"mean time (s)": [method_times[n] for n in method_names]},
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
        row.append(t["mean"])
    heatmap_data.append(row)

heatmap(
    heatmap_data,
    row_labels=[m.value for m in METHODS],
    col_labels=backends,
    title="Method × backend solve time (s) — 1-film stack",
)

# %%
# 4. JIT acceleration
# -------------------
# JAX and PyTorch both JIT-compile on first call.  The warm-up call is
# excluded from timing so the bars reflect steady-state solve
# performance after compilation.

jit_backends = [b for b in ("jax", "torch") if b in backends]
if jit_backends:
    jit_hot: dict[str, float] = {}
    for b in jit_backends:
        with backend_scope(b):
            _ = solve(
                stack, wavelength, kx=kx, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
            t_hot = time_solve(
                stack, wavelength, kx=kx, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
        jit_hot[b] = t_hot["mean"]

    bar_chart_compare(
        {"steady-state (warm-up excluded)": [jit_hot[b] for b in jit_backends]},
        jit_backends,
        title="JIT steady-state — SMATRIX, 1-film stack",
        ylabel="Solve time (s)",
    )
else:
    print("JIT backends (jax, torch) not available — skipping JIT section.")

# %%
# 5. GPU acceleration
# -------------------
# Compare CPU vs GPU timings for JAX and PyTorch backends when CUDA is
# available.  The GPU bar is omitted when CUDA is not detected.

_HAS_CUDA = False
try:
    import jax  # noqa: F811

    _HAS_CUDA = len(jax.devices("gpu")) > 0
except Exception:
    pass
if not _HAS_CUDA:
    try:
        import torch  # noqa: F811

        _HAS_CUDA = torch.cuda.is_available()
    except Exception:
        pass

if _HAS_CUDA:
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
