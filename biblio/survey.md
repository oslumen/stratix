# Transfer Matrix Method in Electromagnetics — Implementation Survey

## Method Overview

The TMM computes reflection/transmission/absorption of EM waves in planar stratified media by multiplying layer matrices (2×2 or 4×4). Three canonical variants:

| Method | Matrix Size | Media | Origin | Numerical Stability |
|---|---|---|---|---|
| Abélès / scalar TMM | 2×2 | Isotropic, non-magnetic | Abélès (1950) | Unstable for thick+evanescent layers |
| Berreman 4×4 TMM | 4×4 | Anisotropic, bianisotropic, magnetic | Berreman, JOSA 62, 502 (1972) | Same as above |
| Scattering matrix (S-matrix) | 2×2 or 4×4 | Any | Ko & Inkson (1988); Whittaker & Culshaw (1999) | Stable in all regimes |

The S-matrix reformulates layer concatenation to avoid exponentially growing evanescent terms — critical for thick layers, high absorption, and grazing incidence.

---

## Implementation Comparison

| Implementation | Year | Method | Language / Backend | Key Features | Maintained? | License | Notes |
|---|---|---|---|---|---|---|---|
| **tmm** | 2016 | 2×2 scalar TMM + incoherent | Python / NumPy | Thin+thick films, absorption profiles, ellipsometry, color, position-dependent absorption | Yes (v0.2.0, Nov 2024) | MIT | Most widely used Python TMM; accompanied by arXiv:1603.02720 paper with detailed derivations. `pip install tmm`. Repo: sbyrnes321/tmm |
| **TMMax** | 2025 | 2×2 scalar TMM (vectorized) | Python / JAX | GPU/TPU, autodiff for inverse design, built-in material DB, ~100× faster than NumPy TMM | Yes (v1.1.4, Oct 2025) | MIT | arXiv:2507.11341 (JOSS 2025). Full vectorization over λ and angle. Ideal for large parameter sweeps and gradient-based optimization. Repo: bahremsd/tmmax |
| **vtmm** | ~2023 | 2×2 scalar TMM (vectorized over λ+kx) | Python / JAX | Autograd for inverse design, ~100× faster than `tmm` for multi-λ/angle sweeps (0.78s vs 79.9s for 50λ×50kx) | Beta (8 commits, low activity) | MIT | From Fan lab (Stanford). Builds on Byrnes physics but vectorized simultaneously over wavelength and in-plane wavevector. Repo: fancompute/vtmm |
| **PyLlama** | 2020 | 4×4 TMM + S-matrix | Python / NumPy+SciPy | Anisotropic layers, cholesteric liquid crystals, stable S-matrix fallback, layer assembly API | Unknown (last release unclear) | Not verified | arXiv:2012.05945 (CPC 2022). Only Python lib offering both TMM and S-matrix in 4×4. Repo location not confirmed via fetch |
| **PyMoosh** | 2023 | TMM + S-matrix + RCWA + guided mode solver | Python / NumPy+SciPy | Reflectance, transmittance, absorption, guided modes, photovoltaic efficiency, global optimization, deep-learning-friendly, educational notebooks | Yes (315 commits, 64 stars, updated Jun 2026) | Open source (LICENSE.txt) | arXiv:2309.00654 (JOSA B 2024). Most feature-complete open-source option. Trilogy of tutorial papers. Repo: AnMoreau/PyMoosh |
| **TMM-Sim** | 2024 | 2×2 TMM | Python + GUI | Solar cell simulation: EQE, IQE, exciton generation, parasitic loss, JV from optical profiles, web (nanocalc.org) + desktop | Yes (Apr 2024) | Free (repo available) | arXiv:2404.12191 (CPC 2024). GUI-based, no programming required. Windows/macOS/Linux |
| **PyEOC** | 2022 | 2×2 TMM + fitting | Python / NumPy+SciPy+Matplotlib | Electro-optic coefficient extraction from reflectivity data, robust fitting | Unknown | MIT | arXiv:2205.05157. Repo: sidihamady/PyEOC. Niche but well-validated |
| **scikit-rf** (skrf) | 2013+ | TMM for RF circuits | Python / NumPy+SciPy | Microwave S-parameters, network analysis, transmission line TMM, VNA data, GUI apps | Yes (active, BSD) | BSD | RF/microwave domain, not optical. `pip install scikit-rf` |
| **PTMM** | 2022 | Perturbative TMM | Theory (arXiv only) | Time-dependent TMM for optical-pump-THz-probe, sub-picosecond dynamics, spintronics THz emitters | N/A (no code found) | N/A | arXiv:2206.00895. Novel extension of TMM to time-varying media. No public implementation verified |
| **Modified recursive TMM** | 2024 | Recursive TMM for spheres | MATLAB (supplement) | Multilayer sphere scattering, avoids Bessel overflow via logarithmic derivatives and hybrid recursion | N/A | N/A | arXiv:2409.10877. Includes MATLAB code supplement |

---

## Best Fit by Use Case

| Use case | Best choice |
|---|---|
| Quick single-wavelength calculation | `tmm` (sbyrnes321) — lowest overhead |
| Multi-λ/angle parameter sweeps + inverse design | `TMMax` (GPU autodiff, mature) or `vtmm` (simpler, fancompute) |
| Anisotropic / liquid crystal layers | `PyLlama` (4×4 TMM + S-matrix) |
| Guided modes + optimization + education | `PyMoosh` (most feature-complete) |
| Solar cell simulation | `TMM-Sim` (GUI, purpose-built) or `tmm` (absorption profiles) |
| RF/microwave circuits | `scikit-rf` |
| Multilayer sphere Mie scattering | Modified recursive TMM (arXiv:2409.10877) |
| Ultrafast time-domain THz | PTMM (theory only, no public code) |

---

## Method Variants by Application Domain

| Domain | Method | Typical matrix size | Key considerations |
|---|---|---|---|
| Optical coatings (AR, HR, filters) | Scalar 2×2 TMM | 2×2 | Coherent + incoherent summation for thick substrates |
| Liquid crystals, anisotropic films | Berreman 4×4 TMM | 4×4 | Needs S-matrix for thick cholesteric stacks |
| Solar cells | 2×2 TMM + absorption profiles | 2×2 | Position-dependent absorption, exciton generation rate, IQE/EQE |
| Ellipsometry | 2×2 (isotropic) or 4×4 (anisotropic) | 2×2 / 4×4 | Psi/Delta parameter extraction |
| 1D photonic crystals / Bragg stacks | 2×2 TMM + Bloch theorem | 2×2 | Band structure from T-matrix eigenvalues |
| Multilayer spheres (Mie) | Recursive TMM | N/A | Modified recursive algorithm to avoid Bessel overflow |
| Ultrafast time-domain THz | Perturbative TMM (PTMM) | 2×2 | Time-dependent dielectric function within one optical cycle |
| RF/microwave circuits | TMM for ABCD / S-parameters | 2×2 | Uses impedance/admittance rather than refractive index |

---

## Commercial / Proprietary Tools

| Tool | Vendor | Domain |
|---|---|---|
| Essential Macleod | Thin Film Center | Optical coating design |
| TFCalc | Software Spectra | Thin film design |
| OptiLayer | OptiLayer GmbH | Optical coating optimization |
| FilmStar | FTG Software | Thin film design & manufacturing |
| Lumerical STACK | Ansys | Optical thin film solver (built into FDTD/EME stack) |
| COMSOL | COMSOL Inc. | General FEM, multilayer via Wave Optics module |

---

## Gaps & Uncertainties

- **PyLlama repository location** — paper references GitHub but HTTP 404 on several candidate URLs; exact repo not confirmed.
- **No unified standalone S-matrix library** — PyLlama and PyMoosh implement S-matrix internally but no standalone, general-purpose S-matrix EM library exists in Python. Most implementations use TMM directly.
- **Berreman 1972 paper** — not on arXiv (pre-arXiv era); JOSA 62, 502. Full text not verified.
- **MATLAB TMM codes** — many File Exchange entries exist (Rasskazov et al., Rumpf emlab), not individually verified here.
- **GPU implementations** — only TMMax provides GPU/TPU support. vtmm uses JAX but GPU support not explicitly documented.
- **Bianisotropic / general constitutive relations** — PyLlama's 4×4 supports anisotropy but extent of bianisotropic (magnetoelectric) support not verified.
- **PTMM** — no public implementation found. Theory only (arXiv:2206.00895).

---

## Key References

| Paper | Year | Venue | ID |
|---|---|---|---|
| Byrnes, "Multilayer optical calculations" | 2016 | arXiv | [1603.02720](https://arxiv.org/abs/1603.02720) |
| Bay, Vignolini, Vynck, "PyLlama" | 2020 | Comput. Phys. Commun. 273, 108256 (2022) | [2012.05945](https://arxiv.org/abs/2012.05945) |
| Danis, Zayim, "TMMax" | 2025 | JOSS 10(94), 9088 (2025) | [2507.11341](https://arxiv.org/abs/2507.11341) |
| Langevin et al., "PyMoosh" | 2023 | JOSA B 41(2), A67-A78 (2024) | [2309.00654](https://arxiv.org/abs/2309.00654) |
| Benatto et al., "TMM-Sim" | 2024 | Comput. Phys. Commun. (2024) | [2404.12191](https://arxiv.org/abs/2404.12191) |
| Hamady, "PyEOC" | 2022 | arXiv | [2205.05157](https://arxiv.org/abs/2205.05157) |
| Yang, Dal Forno, Battiato, "PTMM" | 2022 | arXiv | [2206.00895](https://arxiv.org/abs/2206.00895) |
| Zhang, "Modified recursive TMM for multilayer sphere" | 2024 | arXiv | [2409.10877](https://arxiv.org/abs/2409.10877) |
