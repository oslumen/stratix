"""
Bragg mirror (dielectric reflector)
====================================

High-reflectance mirror built from alternating quarter-wave layers of
TiO₂ (n=2.6) and SiO₂ (n=1.46). The periodic index contrast opens a
photonic stop band.
"""

# %%
# A Bragg mirror alternates high/low index layers each λ₀/(4n) thick.
# We use 8 pairs (16 layers total) at λ₀ = 550 nm and sweep wavelength
# to show the stop band.

import numpy as np
import matplotlib.pyplot as plt

from phokaia import Material, Layer, Stack
from stratix import solve, Polarization, compute_field_profile, Method

air = Material(epsilon=1.0, name="Air")
tio2 = Material(epsilon=2.6**2, name="TiO₂")
sio2 = Material(epsilon=1.46**2, name="SiO₂")
substrate = Material(epsilon=1.52**2, name="Glass")

design_wl = 550e-9
d_tio2 = design_wl / (4 * 2.6)   # ~52.9 nm
d_sio2 = design_wl / (4 * 1.46)  # ~94.2 nm

num_pairs = 8
layers = []
for _ in range(num_pairs):
    layers.append(Layer(thickness=d_tio2, material=tio2))
    layers.append(Layer(thickness=d_sio2, material=sio2))

mirror = Stack(superstrate=air, substrate=substrate, layers=layers)

wavelengths = np.linspace(400e-9, 800e-9, 400)
result = solve(mirror, wavelengths, kx=0.0, polarization=Polarization.TE)

# %%
# The stop band appears as a region of high reflectance around 550 nm.
# Adding more pairs increases peak reflectance and sharpens edges.

wl_nm = wavelengths * 1e9

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(wl_nm, np.asarray(result.R))
ax.fill_between(wl_nm, 0.99, np.maximum(np.asarray(result.R), 0.99), color="C0", alpha=0.15)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Reflectance")
ax.set_ylim(0, 1.02)
ax.set_title(f"Bragg mirror ({num_pairs} pairs TiO₂/SiO₂)")
plt.show()



# %%
# The E-field magnitude is continuous across interfaces, while the
# H-field magnitude jumps according to the wave impedance contrast.


thickness = sum(layer.thickness for layer in layers)
z = np.linspace(-200e-9, thickness + 200e-9, 500)

for wl in [design_wl,452e-9]:
    result = solve(mirror, wl, kx=0.0, polarization=Polarization.TE, method=Method.SMATRIX)

    fields = compute_field_profile(result, z)
    E = np.asarray(fields["E"])
    H = np.asarray(fields["H"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    ax1.plot(z * 1e9, np.abs(E))
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(thickness * 1e9, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("|E| (a.u.)")
    ax1.set_title(f"Field profiles — Bragg mirror @ {wl*1e9} nm")

    ax2.plot(z * 1e9, np.abs(H))
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(thickness * 1e9, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("z (nm)")
    ax2.set_ylabel("|H| (a.u.)")

    plt.show()
# %%
