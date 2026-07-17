"""
Silicon anti-reflection coating
================================

Bare silicon reflects ~31% of visible light, limiting efficiency in
solar cells and photodetectors. A quarter-wave layer of Si₃N₄
(n ≈ 2.0) reduces reflectance at the design wavelength by 70×.
"""

# %%
# Silicon has a high refractive index (n ≈ 3.5), which makes it
# highly reflective at an air interface. The ideal single-layer AR
# coating index is √(n_si · n_air) ≈ 1.87. Si₃N₄ (n ≈ 2.0) is a
# close match and is commonly used in photovoltaics.
#
# The quarter-wave thickness is d = λ₀/(4n) = 550/(4·2.0) = 68.8 nm.

import numdiff as nd
import matplotlib.pyplot as plt

from phokaia import Material, Layer, Stack
from stratix import solve, Polarization

air = Material(epsilon=1.0, name="Air")
si3n4 = Material(epsilon=2.0**2, name="Si₃N₄")
silicon = Material(epsilon=3.5**2, name="Silicon")

design_wl = 550e-9
thickness = design_wl / (4 * 2.0)  # 68.8 nm

bare = Stack(superstrate=air, substrate=silicon, layers=[])
coated = Stack(
    superstrate=air,
    substrate=silicon,
    layers=[Layer(thickness=thickness, material=si3n4)],
)

wavelengths = nd.linspace(400e-9, 800e-9, 300)

R_bare = solve(bare, wavelengths, kx=0.0, polarization=Polarization.TE).R
R_coated = solve(coated, wavelengths, kx=0.0, polarization=Polarization.TE).R

# %%
# The coated stack drops from 30.9% to 0.44% reflectance at 550 nm.
# Away from the design wavelength the condition d = λ/(4n) no longer
# holds and R rises, forming the characteristic AR band.

wl_nm = wavelengths * 1e9

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(wl_nm, R_bare, label="Bare silicon", linestyle="--")
ax.plot(wl_nm, R_coated, label="Si₃N₄ AR coating")
ax.axvline(550, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.set_ylim(0, 0.35)
ax.legend()
ax.set_title("Normal incidence, TE")
plt.show()
