"""
Total internal reflection
=========================

When light travels from a higher-index medium to a lower-index medium
beyond the critical angle, the transmitted wave becomes evanescent —
its out-of-plane wavevector :math:`k_z` is purely imaginary and carries
no power away from the interface.  Reflectance goes to 1, transmittance
drops to 0.
"""

# %%
# We sweep incidence angles across the critical angle of a glass-to-air
# interface (:math:`n_\\text{glass} = 1.52`), computing reflectance and
# transmittance for TE polarization.

import numpy as np
import matplotlib.pyplot as plt

from phokaia import Material, Stack
from stratix import solve_angles, Polarization

glass = Material(epsilon=1.52**2, name="Glass")
air = Material(epsilon=1.0, name="Air")

interface = Stack(superstrate=glass, substrate=air, layers=[])

wavelength = 550e-9
angles = np.linspace(0, 89.9, 300)

te = solve_angles(interface, wavelength, angles, Polarization.TE)
tm = solve_angles(interface, wavelength, angles, Polarization.TM)

# %%
# The critical angle is :math:`\\theta_c = \\arcsin(n_2 / n_1)`.
# Beyond that, :math:`k_z` in air is purely imaginary —
# the transmitted wave is evanescent, so no energy leaves the interface.

n_glass, n_air = 1.52, 1.0
theta_c = np.degrees(np.arcsin(n_air / n_glass))

R_te = np.asarray(te.R)
T_te = np.asarray(te.T)
R_tm = np.asarray(tm.R)
T_tm = np.asarray(tm.T)

fig, (ax_r, ax_t) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

ax_r.plot(angles, R_te, label="TE")
ax_r.plot(angles, R_tm, label="TM")
ax_r.axvline(theta_c, color="gray", linestyle="--", alpha=0.6)
ax_r.annotate(
    f"θc = {theta_c:.1f}°",
    xy=(theta_c, 1.0),
    xytext=(theta_c + 8, 0.85),
    arrowprops=dict(arrowstyle="->", color="gray"),
    fontsize=9,
)
ax_r.set_ylabel("Reflectance")
ax_r.set_ylim(0, 1.02)
ax_r.legend()
ax_r.set_title("Glass → Air")

ax_t.plot(angles, T_te, label="TE")
ax_t.plot(angles, T_tm, label="TM")
ax_t.axvline(theta_c, color="gray", linestyle="--", alpha=0.6)
ax_t.set_xlabel("Incidence angle (degrees)")
ax_t.set_ylabel("Transmittance")
ax_t.set_ylim(-0.02, 1.02)
ax_t.legend()
ax_t.set_title("Transmittance drops to zero")

plt.tight_layout()
plt.show()

# %%
# For :math:`\\theta > \\theta_c` the transmitted power vanishes
# (:math:`T = 0`) and all energy is reflected (:math:`R = 1`).
# The transition at :math:`\\theta_c` is sharp in the reflectance
# derivative — a signature of the evanescent cut-off.
