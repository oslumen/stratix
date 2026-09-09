"""Package-scoped logging.

stratix logs through loguru but, being a library, must not reconfigure
the global logger it shares with the host application: importing this
module removes no handlers and adds none.  Following loguru's library
convention, stratix's own records are disabled by default; an
application opts in with :func:`set_log_level`, which enables them and
adds a single stderr handler that sees only stratix's records --
replacing that one handler, and nothing else, on later calls.
"""

from __future__ import annotations

__all__ = ["logger", "set_log_level"]

import sys
from typing import Any

from loguru import logger

LEVELS = dict(
    DEBUG="DEBUG",
    INFO="INFO",
    WARNING="WARNING",
    ERROR="ERROR",
    CRITICAL="CRITICAL",
)

logger.disable("stratix")

#: Handler id of the stderr sink :func:`set_log_level` added, so a later
#: call can replace stratix's own handler without touching anyone else's.
_handler_id: int | None = None


def _is_stratix_record(record: Any) -> bool:
    """Whether a loguru record was emitted from inside this package."""
    name = record["name"] or ""
    return name == "stratix" or name.startswith("stratix.")


def set_log_level(level: str) -> None:
    """Opt in to stratix's log records at the given verbosity.

    Enables the records this package emits (they are disabled by
    default) and adds one stderr handler filtered to them.  Handlers
    configured by the application are left untouched; calling again
    replaces only the handler a previous call added.

    Parameters
    ----------
    level : str
        The verbosity level.
        Valid values are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` or ``CRITICAL``.

    """
    global _handler_id

    LOG_LEVEL = LEVELS[level]
    logger.enable("stratix")
    if _handler_id is not None:
        logger.remove(_handler_id)
    _handler_id = logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
        level=LOG_LEVEL,
        filter=_is_stratix_record,
    )
