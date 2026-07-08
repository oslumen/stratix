"""Package metadata."""

from __future__ import annotations

from importlib.metadata import metadata as _metadata_fn
from importlib.metadata import version as _version_fn


def get_meta(pkg: str) -> tuple[str, str, str, str]:
    try:
        data = _metadata_fn(pkg)
        __version__ = _version_fn(pkg)
        __author__ = data.get("author", "unknown")
        __description__ = data.get("summary", "unknown")
        __license__ = data.get("license", "unknown")
    except Exception:
        __version__ = "unknown"
        __author__ = "unknown"
        __description__ = "unknown"
        __license__ = "unknown"
    return __version__, __author__, __description__, __license__


__version__, __author__, __description__, __license__ = get_meta("stratix")
