"""Smoke tests for stratix packaging: import, version, logging hygiene."""

from __future__ import annotations

import sys

import stratix


def test_import() -> None:
    """Package imports successfully."""
    assert stratix is not None


def test_version_is_string() -> None:
    """__version__ is a non-empty string."""
    assert isinstance(stratix.__version__, str)
    assert len(stratix.__version__) > 0


def test_version_reads_the_installed_metadata() -> None:
    """One version mechanism: importlib.metadata on the installed package."""
    from importlib.metadata import version

    assert stratix.__version__ == version("stratix")


def test_logger_import() -> None:
    """Logging module is accessible."""
    from stratix.log import logger

    assert logger is not None


def test_import_leaves_host_loguru_handlers_untouched() -> None:
    """A library must not reconfigure the global logger on import.

    A handler the application added before importing stratix has to keep
    receiving records afterwards.  The stratix modules are re-imported
    fresh to exercise their import-time code, then the original modules
    are restored so the rest of the suite keeps its enum identities.
    """
    from loguru import logger

    records: list[str] = []
    handler_id = logger.add(records.append, level="DEBUG", format="{message}")
    saved = {
        name: mod for name, mod in sys.modules.items() if name.startswith("stratix")
    }
    for name in saved:
        del sys.modules[name]
    try:
        import stratix  # fresh import runs the module-level code under test
        import stratix.log  # noqa: F401

        logger.debug("host handler still alive")
        assert any("host handler still alive" in r for r in records), (
            "importing stratix removed a handler the application configured"
        )
    finally:
        for name in [n for n in sys.modules if n.startswith("stratix")]:
            del sys.modules[name]
        sys.modules.update(saved)
        logger.remove(handler_id)


def test_set_log_level_keeps_application_handlers() -> None:
    """Opting in to stratix records must not remove anyone else's handler."""
    from loguru import logger

    import stratix.log as slog

    records: list[str] = []
    handler_id = logger.add(records.append, level="DEBUG", format="{message}")
    try:
        slog.set_log_level("INFO")
        slog.set_log_level("DEBUG")  # replaces only its own handler
        logger.debug("app handler survives set_log_level")
        assert any("app handler survives set_log_level" in r for r in records)
    finally:
        logger.remove(handler_id)
        if slog._handler_id is not None:
            logger.remove(slog._handler_id)
            slog._handler_id = None
