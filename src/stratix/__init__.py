"""Stratified Media Electromagnetic Solver"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("stratix")
except PackageNotFoundError:
    __version__ = "unknown"

from ._result import Result
from ._solve import solve
from ._types import Method
from ._types import Polarization

__all__ = [
    "Method",
    "Polarization",
    "Result",
    "solve",
]
