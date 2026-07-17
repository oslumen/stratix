"""
Fresnel reflection at an interface
===================================

Angular reflectance at an air/glass interface for TE and TM polarization,
showing the Brewster angle where TM reflectance vanishes.
"""

# %%
# A bare interface between two semi-infinite media — air above, glass below.
# We sweep the incidence angle from 0° to 89.9° and compute reflectance
# for both TE (s) and TM (p) polarization.

import numpy as np
import matplotlib.pyplot as plt

from phokaia import Material, Stack
from stratix import solve_angles, Polarization

air = Material(epsilon=1.0, name="Air")
glass = Material(epsilon=1.52**2, name="Glass")

interface = Stack(superstrate=air, substrate=glass, layers=[])

wavelength = 550e-9  # green light
angles = np.linspace(0, 89.9, 200)

res_te = solve_angles(interface, wavelength, angles, Polarization.TE)
res_tm = solve_angles(interface, wavelength, angles, Polarization.TM)

# %%
# The TM reflectance drops to zero at the Brewster angle
# θ_B = arctan(n₂/n₁).

n_air, n_glass = 1.0, 1.52
brewster_deg = np.degrees(np.arctan(n_glass / n_air))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(angles, np.asarray(res_te.R), label="TE (s)")
ax.plot(angles, np.asarray(res_tm.R), label="TM (p)")
ax.axvline(brewster_deg, color="gray", linestyle="--", alpha=0.6)
ax.annotate(
    f"Brewster\n{brewster_deg:.1f}°",
    xy=(brewster_deg, 0),
    xytext=(brewster_deg + 8, 0.15),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=9,
)
ax.set_xlabel("Incidence angle (degrees)")
ax.set_ylabel("Reflectance")
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("Air → Glass")
plt.show()
