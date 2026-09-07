"""
Automatic differentiation
=========================

stratix is built on numdiff, which provides autodiff across backends
(JAX, PyTorch, Autograd). Gradients of reflectance w.r.t. wavelength
and layer thickness enable gradient-based inverse design.
"""

# %%
# Setup — materials, stack, and autodiff backend.

import matplotlib.pyplot as plt
import numdiff as nd
import numpy as np

nd.set_backend("jax")

from phokaia import Layer  # noqa: E402
from phokaia import Material  # noqa: E402
from phokaia import Polarization  # noqa: E402
from phokaia import Stack  # noqa: E402

from stratix import solve  # noqa: E402

air = Material(epsilon=1.0, name="Air")
mgf2 = Material(epsilon=1.38**2, name="MgF₂")
glass = Material(epsilon=1.52**2, name="Glass")

design_wl = 550e-9
thickness = design_wl / (4 * 1.38)

stack = Stack(
    superstrate=air,
    substrate=glass,
    layers=[Layer(thickness=thickness, material=mgf2)],
)

# %%
# Gradient w.r.t. wavelength
# --------------------------
# dR/dλ across 400–800 nm with finite-difference verification over the
# full range. Derivatives are scaled to µm\ :sup:`−1`\  .

SCALE = 1e-6


def R_vs_wl(wl):
    return solve(stack, wl, kx=0.0, polarization=Polarization.TE).R[0, 0]


_grad_wl = nd.grad(R_vs_wl)
dR_dwl_fn = nd.vmap(_grad_wl)

wavelengths_nm = np.linspace(400, 800, 100)
wavelengths = wavelengths_nm * 1e-9
grads_wl_ad = dR_dwl_fn(wavelengths)

# Finite-difference across the full range
eps_wl = 1e-12


def _R_vs_wl_fd(wl):
    return (R_vs_wl(wl + eps_wl) - R_vs_wl(wl - eps_wl)) / (2 * eps_wl)


grads_wl_fd = nd.vmap(_R_vs_wl_fd)(wavelengths)

# Scale to µm⁻¹
grads_wl_ad_nm = np.asarray(grads_wl_ad * SCALE)
grads_wl_fd_nm = np.asarray(grads_wl_fd * SCALE)
relerr_wl = np.abs(grads_wl_ad_nm - grads_wl_fd_nm) / np.maximum(
    np.abs(grads_wl_ad_nm), np.abs(grads_wl_fd_nm)
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax1.plot(wavelengths_nm, grads_wl_ad_nm, label="AD")
ax1.plot(wavelengths_nm, grads_wl_fd_nm, "--", label="FD")
ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax1.set_ylabel("dR/dλ (µm⁻¹)")
ax1.set_title("dR/dλ — MgF₂ AR coating")
ax1.legend()

ax2.plot(wavelengths_nm, relerr_wl)
ax2.set_yscale("log")
ax2.set_ylabel("Rel. error")
ax2.set_xlabel("Wavelength (nm)")

plt.tight_layout()
plt.show()

# %%
# Gradient w.r.t. layer thickness
# -------------------------------
# dR/dt at the design wavelength as a function of thickness. Pass
# traced thickness values via the ``thicknesses`` kwarg of ``solve()``.
# Finite differences verify the full thickness range. Derivatives are
# scaled to µm\ :sup:`−1`\ .


def _R_vs_thickness(t):
    return solve(
        stack, design_wl, kx=0.0, polarization=Polarization.TE,
        thicknesses=nd.array([t], dtype=nd.float64),
    ).R[0, 0]


_grad_t = nd.grad(_R_vs_thickness)
dR_dt_fn = nd.vmap(_grad_t)

t_vals = np.linspace(thickness * 0.5, thickness * 1.5, 100)
grads_t_ad = dR_dt_fn(t_vals)

# Finite-difference across the full range
eps_t = 1e-12


def _R_vs_t_fd(t):
    return (_R_vs_thickness(t + eps_t) - _R_vs_thickness(t - eps_t)) / (2 * eps_t)


grads_t_fd = nd.vmap(_R_vs_t_fd)(t_vals)

# Scale to µm⁻¹
grads_t_ad_nm = np.asarray(grads_t_ad * SCALE)
grads_t_fd_nm = np.asarray(grads_t_fd * SCALE)
relerr_t = np.abs(grads_t_ad_nm - grads_t_fd_nm) / np.maximum(
    np.abs(grads_t_ad_nm), np.abs(grads_t_fd_nm)
)

t_vals_nm = t_vals * 1e9

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
ax1.plot(t_vals_nm, grads_t_ad_nm, label="AD")
ax1.plot(t_vals_nm, grads_t_fd_nm, "--", label="FD")
ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax1.axvline(thickness * 1e9, color="gray", linestyle=":", alpha=0.5)
ax1.set_ylabel("dR/dt (µm⁻¹)")
ax1.set_title("dR/dt at λ₀ = 550 nm — MgF₂ AR coating")
ax1.legend()

ax2.plot(t_vals_nm, relerr_t)
ax2.set_yscale("log")
ax2.set_ylabel("Rel. error")
ax2.set_xlabel("Layer thickness (nm)")

plt.tight_layout()
plt.show()
