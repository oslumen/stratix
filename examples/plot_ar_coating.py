"""
Single-layer anti-reflection coating
=====================================

A quarter-wave layer of MgF₂ on glass reduces reflectance near the
design wavelength. We sweep wavelength to see the AR band.
"""

# %%
# A quarter-wave AR coating has thickness d = λ₀/(4n) at the design
# wavelength λ₀. Here λ₀ = 550 nm, MgF₂ index n = 1.38, so
# d = 99.6 nm. We compare bare glass against the coated stack.

import numdiff as nd
import matplotlib.pyplot as plt

from phokaia import Material, Layer, Stack
from stratix import solve, Polarization

air = Material(epsilon=1.0, name="Air")
mgf2 = Material(epsilon=1.38**2, name="MgF₂")
glass = Material(epsilon=1.52**2, name="Glass")

design_wl = 550e-9
thickness = design_wl / (4 * 1.38)  # ~99.6 nm

bare = Stack(superstrate=air, substrate=glass, layers=[])
coated = Stack(
    superstrate=air,
    substrate=glass,
    layers=[Layer(thickness=thickness, material=mgf2)],
)

wavelengths = nd.linspace(400e-9, 800e-9, 300)

R_bare = solve(bare, wavelengths, kx=0.0, polarization=Polarization.TE).R
R_coated = solve(coated, wavelengths, kx=0.0, polarization=Polarization.TE).R

# %%
# The coated stack has near-zero reflectance at 550 nm. Away from the
# design wavelength the condition d = λ/(4n) no longer holds and R rises.

wl_nm = wavelengths * 1e9

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(wl_nm, R_bare, label="Bare glass", linestyle="--")
ax.plot(wl_nm, R_coated, label="MgF₂ AR coating")
ax.axvline(550, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.set_ylim(0, 0.08)
ax.legend()
ax.set_title("Normal incidence, TE")
plt.show()
