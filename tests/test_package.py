"""Smoke tests for stratix."""

from __future__ import annotations

import stratix
from stratix.__about__ import __version__


def test_import() -> None:
    """Package imports successfully."""
    assert stratix is not None


def test_version_is_string() -> None:
    """__version__ is a non-empty string."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_about_metadata() -> None:
    """Package metadata tuple is populated."""
    from stratix.__about__ import __author__, __description__, __license__

    assert isinstance(__author__, str) and len(__author__) > 0
    assert isinstance(__description__, str) and len(__description__) > 0
    assert isinstance(__license__, str) and len(__license__) > 0


def test_logger_import() -> None:
    """Logging module is accessible."""
    from stratix.log import logger, set_log_level  # noqa: F811

    assert logger is not None






