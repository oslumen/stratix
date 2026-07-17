"""
Automatic differentiation
=========================

stratix is built on numdiff, which provides autodiff across backends
(JAX, PyTorch, Autograd). Gradients of reflectance w.r.t. wavelength
and layer thickness enable gradient-based inverse design.
"""

# %%
# Setup — materials, stack, and autodiff backend.

import jax
import numpy as np
import matplotlib.pyplot as plt

import numdiff as nd
nd.set_backend("jax")

from phokaia import Material, Layer, Stack
from stratix import Polarization, solve

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
# dR/dλ across 400–800 nm. The derivative vanishes at the quarter-wave
# design point (λ₀ = 550 nm, R minimum), so we verify with finite
# differences at λ = 500 nm where the gradient is well-resolved.


def R_vs_wl(wl):
    return solve(stack, wl, kx=0.0, polarization=Polarization.TE).R[0]


_grad_wl = nd.grad(R_vs_wl)
dR_dwl_fn = jax.vmap(_grad_wl)

wavelengths = np.linspace(400e-9, 800e-9, 100)
grads_wl_ad = dR_dwl_fn(wavelengths)

# Finite-difference verification at 500 nm (away from the zero-gradient minimum)
wl_fd = 500e-9
eps_wl = 1e-9
R_plus = float(solve(stack, wl_fd + eps_wl, kx=0.0, polarization=Polarization.TE).R[0])
R_minus = float(solve(stack, wl_fd - eps_wl, kx=0.0, polarization=Polarization.TE).R[0])
grad_wl_fd = (R_plus - R_minus) / (2 * eps_wl)
grad_wl_ad = float(_grad_wl(nd.array(wl_fd)))
relerr_wl = abs(grad_wl_ad - grad_wl_fd) / max(abs(grad_wl_ad), abs(grad_wl_fd))

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(wavelengths * 1e9, grads_wl_ad)
ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax.axvline(550, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("dR/dλ (m⁻¹)")
ax.set_title("dR/dλ — MgF₂ AR coating")
ax.text(
    0.98, 0.95,
    f"λ = 500 nm verification:\nAD = {grad_wl_ad:.4e}\nFD = {grad_wl_fd:.4e}\n"
    f"rel err = {relerr_wl:.2e}",
    transform=ax.transAxes, va="top", ha="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
plt.tight_layout()
plt.show()

# %%
# Gradient w.r.t. layer thickness
# -------------------------------
# dR/dt at the design wavelength as a function of thickness. Pass
# traced thickness values via the ``thicknesses`` kwarg of ``solve()``.
# The gradient vanishes at the design thickness (quarter-wave minimum),
# so FD verification uses a 20 % offset where the gradient is
# well-resolved.


def _R_vs_thickness(t):
    return solve(
        stack, design_wl, kx=0.0, polarization=Polarization.TE,
        thicknesses=nd.array([t], dtype=nd.float64),
    ).R[0]

_grad_t = nd.grad(_R_vs_thickness)
dR_dt_fn = jax.vmap(_grad_t)

t_vals = np.linspace(thickness * 0.5, thickness * 1.5, 100)
grads_t_ad = dR_dt_fn(t_vals)

# Finite-difference verification at +20 % thickness
t_fd = thickness * 1.2
eps_t = 1e-12
t_plus = nd.array([t_fd + eps_t], dtype=nd.float64)
t_minus = nd.array([t_fd - eps_t], dtype=nd.float64)
R_plus = float(solve(stack, design_wl, kx=0.0, polarization=Polarization.TE,
                     thicknesses=t_plus).R[0])
R_minus = float(solve(stack, design_wl, kx=0.0, polarization=Polarization.TE,
                      thicknesses=t_minus).R[0])
grad_t_fd = (R_plus - R_minus) / (2 * eps_t)
grad_t_ad = float(_grad_t(nd.array(t_fd, dtype=nd.float64)))
relerr_t = abs(grad_t_ad - grad_t_fd) / max(abs(grad_t_ad), abs(grad_t_fd))

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(t_vals * 1e9, grads_t_ad)
ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
ax.axvline(thickness * 1e9, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Layer thickness (nm)")
ax.set_ylabel("dR/dt (m⁻¹)")
ax.set_title("dR/dt at λ₀ = 550 nm — MgF₂ AR coating")
ax.text(
    0.98, 0.95,
    f"t = {t_fd * 1e9:.1f} nm verification:\nAD = {grad_t_ad:.4e}\n"
    f"FD = {grad_t_fd:.4e}\nrel err = {relerr_t:.2e}",
    transform=ax.transAxes, va="top", ha="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
plt.tight_layout()
plt.show()
