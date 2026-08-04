"""
Large-stack sweep benchmark
============================

Eight films with varied thicknesses and indices, 50×50 wavelength–kx
sweep.  The deepest stack tests scaling behaviour across all solver
methods and backends.
"""

# %%
# Setup
# -----
# Eight-layer stack with indices 1.5–3.5 and thicknesses 1–10 µm.
# 50 wavelengths (400–800 nm) × 50 kx values (0–1e7 rad/m).

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
from benchmarks._stacks import large_stack
from benchmarks._timing import check_correctness
from benchmarks._timing import measure_memory
from benchmarks._timing import time_solve
from benchmarks._tmm import _tmm_available
from benchmarks._tmm import tmm_compare

stack = large_stack()
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
    backend_times[b] = t["mean"]
    backend_mems[b] = m["peak_delta_mb"]

bar_chart_compare(
    {"mean time (s)": [backend_times[b] for b in backends]},
    backends,
    title="Backend comparison — SMATRIX, 8-film 50×50 sweep",
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
        method_times[m.value] = t["mean"]

method_names = [m.value for m in METHODS]
bar_chart_compare(
    {"mean time (s)": [method_times[n] for n in method_names]},
    method_names,
    title="Method comparison — NumPy backend, 8-film 50×50 sweep",
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
        row.append(t["mean"])
    heatmap_data.append(row)

heatmap(
    heatmap_data,
    row_labels=[m.value for m in METHODS],
    col_labels=backends,
    title="Method × backend solve time (s) — 8-film 50×50 sweep",
)

# %%
# 4. JIT acceleration
# -------------------

jit_backends = [b for b in ("jax", "torch") if b in backends]
if jit_backends:
    jit_hot: dict[str, float] = {}
    for b in jit_backends:
        with backend_scope(b):
            _ = solve(
                stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
            t_hot = time_solve(
                stack, wavelengths, kx=kx_vals, polarization=Polarization.TE,
                method=Method.SMATRIX,
            )
        jit_hot[b] = t_hot["mean"]

    bar_chart_compare(
        {"steady-state (warm-up excluded)": [jit_hot[b] for b in jit_backends]},
        jit_backends,
        title="JIT steady-state — SMATRIX, 8-film 50×50 sweep",
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
        title="GPU vs CPU — SMATRIX, 8-film 50×50 sweep",
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
