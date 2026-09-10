"""
Absorption and energy balance
==============================

When a layer contains a lossy material, part of the incident power is
absorbed. stratix tracks per-layer absorption and verifies the energy
balance R + T + ΣA = 1.
"""

# %%
# A 40 nm gold film on glass. Gold has a complex permittivity at visible
# frequencies — the imaginary part models ohmic loss.
# Pass ``absorption=True`` to get per-layer absorption alongside R and T;
# it works over a whole wavelength sweep in a single vectorized call.

import matplotlib.pyplot as plt
import numpy as np
from phokaia import Layer
from phokaia import Material
from phokaia import Polarization
from phokaia import Stack

from stratix import solve

air = Material(epsilon=1.0, name="Air")
au = Material(epsilon=complex(-3.68, 2.90), name="Au")
glass = Material(epsilon=1.52**2, name="Glass")

thickness = 40e-9
stack = Stack(
    superstrate=air,
    substrate=glass,
    layers=[Layer(thickness=thickness, material=au)],
)

wavelengths = np.linspace(400e-9, 800e-9, 200)

res = solve(stack, wavelengths, kx=0.0, polarization=Polarization.TE, absorption=True)

# ``R`` and ``T`` are always (Nλ, Nk) — here (200, 1), since kx is scalar —
# and ``layer_absorption`` puts the layer index first, giving (n_layers, 200, 1).
# Take column 0 of each to get the spectrum.
R, T = res.R[:, 0], res.T[:, 0]
A = res.layer_absorption[0][:, 0]
balance = res.energy_balance[:, 0]

wl_nm = wavelengths * 1e9

# %%
# The gold layer absorbs ~33% of the incident power around 500 nm.
# The energy balance R + T + A stays at 1.0 to machine precision,
# confirming conservation.

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(wl_nm, R, label="R (reflectance)")
ax.plot(wl_nm, T, label="T (transmittance)")
ax.plot(wl_nm, A, label="A (absorption)")
ax.plot(wl_nm, balance, "k--", alpha=0.3, label="R + T + A")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Fraction")
ax.set_ylim(0, 1.05)
ax.legend(ncol=2)
ax.set_title("40 nm Au on glass")
plt.show()
