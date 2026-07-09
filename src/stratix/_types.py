"""Enumerations for solver methods and polarizations."""

from __future__ import annotations

from enum import StrEnum


class Method(StrEnum):
    """Solver method selection.

    Attributes
    ----------
    SMATRIX : S-matrix (scattering matrix) — numerically stable default.
    ABELES : Abélès 2x2 characteristic matrix formalism.
    ADMITTANCE : Admittance recursion.
    DTN : Dirichlet-to-Neumann map.
    AUTO : Resolves to ``SMATRIX``.
    """

    SMATRIX = "smatrix"
    ABELES = "abeles"
    ADMITTANCE = "admittance"
    DTN = "dtn"
    AUTO = "auto"


class Polarization(StrEnum):
    """Plane-wave polarization.

    Attributes
    ----------
    TE : Transverse electric (s-polarized).
    TM : Transverse magnetic (p-polarized).
    """

    TE = "TE"
    TM = "TM"
