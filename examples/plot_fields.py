"""
Field profiles inside a multilayer
===================================

Electric and magnetic field distributions inside a multilayer stack at
the design wavelength of the AR coating. The field is reconstructed
from the stored intermediate S-matrices.
"""

# %%
# We compute the field profile through an MgF₂ AR coating on glass.
# At the design wavelength, the AR condition creates a standing-wave
# pattern with matched amplitudes — visible in the field profile.

import numpy as np
import matplotlib.pyplot as plt

from phokaia import Material, Layer, Stack
from stratix import solve, compute_field_profile, Polarization

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

result = solve(stack, design_wl, kx=0.0, polarization=Polarization.TE)

# z-axis: superstrate (negative) → coating → substrate (positive)
z = np.linspace(-200e-9, thickness + 200e-9, 500)
fields = compute_field_profile(result, z)
E = np.asarray(fields["E"])
H = np.asarray(fields["H"])

# %%
# The E-field magnitude is continuous across interfaces, while the
# H-field magnitude jumps according to the wave impedance contrast.

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

ax1.plot(z * 1e9, np.abs(E))
ax1.axvline(0, color="gray", linestyle="--", alpha=0.5)
ax1.axvline(thickness * 1e9, color="gray", linestyle="--", alpha=0.5)
ax1.set_ylabel("|E| (a.u.)")
ax1.set_title("Field profiles — MgF₂ AR coating @ 550 nm")

ax2.plot(z * 1e9, np.abs(H))
ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)
ax2.axvline(thickness * 1e9, color="gray", linestyle="--", alpha=0.5)
ax2.set_xlabel("z (nm)")
ax2.set_ylabel("|H| (a.u.)")

plt.show()
