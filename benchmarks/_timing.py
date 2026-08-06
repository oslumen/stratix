"""Timing, memory, and correctness utilities for benchmarks."""

from __future__ import annotations

import timeit
import tracemalloc
from collections.abc import Callable
from typing import Any

import numdiff as nd
from phokaia import Polarization
from phokaia import Stack

from stratix import Method
from stratix import solve


def time_solve(
    stack: Stack,
    wavelength: float | Any,
    kx: float = 0.0,
    polarization: Polarization = Polarization.TE,
    method: Method = Method.AUTO,
    n_repeats: int = 10,
    solver: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Measure execution time of a solve function.

    Parameters
    ----------
    stack : Stack
        Planar multilayer stack.
    wavelength : float or array
        Vacuum wavelength(s) in meters.
    kx : float or array
        In-plane wavevector component(s) in rad/m. Default ``0.0``.
    polarization : Polarization
        Polarization to evaluate.
    method : Method
        Solver method. Default ``AUTO`` (resolves to ``SMATRIX``).
    n_repeats : int
        Number of timing repetitions. Default ``10``.
    solver : callable, optional
        Solve function to benchmark. Default ``stratix.solve``.
        Pass ``nd.jit(solve)`` to measure JIT-compiled performance.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``min``, ``max``, ``times`` (list of
        per-repeat times in seconds).
    """
    _solver = solve if solver is None else solver
    params: dict[str, Any] = {
        "stack": stack,
        "wavelength": wavelength,
        "kx": kx,
        "polarization": polarization,
        "method": method,
    }
    _solver(**params)
    timer = timeit.Timer(lambda: _solver(**params))
    times = timer.repeat(repeat=n_repeats, number=1)
    return {
        "mean": float(nd.mean(nd.array(times))),
        "std": float(nd.std(nd.array(times))),
        "min": float(min(times)),
        "max": float(max(times)),
        "times": times,
    }


def measure_memory(
    stack: Stack,
    wavelength: float | Any,
    kx: float = 0.0,
    polarization: Polarization = Polarization.TE,
    method: Method = Method.AUTO,
) -> dict[str, Any]:
    """Measure peak memory delta of :func:`stratix.solve`.

    Uses :mod:`tracemalloc`. Returns the difference between peak memory
    during the solve and the baseline before the solve.

    Parameters
    ----------
    stack : Stack
        Planar multilayer stack.
    wavelength : float or array
        Vacuum wavelength(s) in meters.
    kx : float or array
        In-plane wavevector component(s) in rad/m. Default ``0.0``.
    polarization : Polarization
        Polarization to evaluate.
    method : Method
        Solver method. Default ``AUTO`` (resolves to ``SMATRIX``).

    Returns
    -------
    dict
        Keys: ``peak_delta_mb`` (MB), ``current_mb``, ``peak_mb``.
    """
    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()

    _ = solve(
        stack,
        wavelength,
        kx=kx,
        polarization=polarization,
        method=method,
    )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "peak_delta_mb": (peak - baseline) / (1024 * 1024),
        "current_mb": (current - baseline) / (1024 * 1024),
        "peak_mb": peak / (1024 * 1024),
    }


def check_correctness(
    reference_result: Any,
    test_result: Any,
    tolerance: float = 1e-10,
) -> float:
    """Compare two :class:`stratix.Result` objects element-wise.

    Parameters
    ----------
    reference_result : Result
        Reference (trusted) result.
    test_result : Result
        Result to check against reference.
    tolerance : float
        Maximum acceptable relative error. Default ``1e-10``.

    Returns
    -------
    float
        Maximum relative error across ``R`` and ``T``.

    Raises
    ------
    ValueError
        If ``R`` or ``T`` shapes differ, or relative error exceeds
        *tolerance*.
    """
    ref_R = nd.array(reference_result.R)
    ref_T = nd.array(reference_result.T)
    tst_R = nd.array(test_result.R)
    tst_T = nd.array(test_result.T)

    if ref_R.shape != tst_R.shape:
        raise ValueError(f"R shape mismatch: ref {ref_R.shape} vs test {tst_R.shape}")
    if ref_T.shape != tst_T.shape:
        raise ValueError(f"T shape mismatch: ref {ref_T.shape} vs test {tst_T.shape}")

    eps = float(nd.finfo(ref_R.dtype).eps)  # type: ignore[attr-defined]

    def _rel_err(ref: Any, tst: Any) -> Any:
        denom = nd.abs(ref) + eps
        return nd.max(nd.abs(tst - ref) / denom)

    R_err = _rel_err(ref_R, tst_R)
    T_err = _rel_err(ref_T, tst_T)

    max_err: float = max(float(R_err), float(T_err))

    if max_err > tolerance:
        raise ValueError(
            f"Correctness check failed: R_err={float(R_err):.3e}, "
            f"T_err={float(T_err):.3e} (tolerance={tolerance})"
        )

    return max_err
