__all__ = ["logger", "set_log_level"]

import sys

from loguru import logger

LEVELS = dict(
    DEBUG="DEBUG",
    INFO="INFO",
    WARNING="WARNING",
    ERROR="ERROR",
    CRITICAL="CRITICAL",
)

logger.remove()
logger.add(
    sys.stderr,
    format="<level>{level} | </level><level>{message}</level>",
    colorize=True,
    level="WARNING",
)


def set_log_level(level: str) -> None:
    """Sets the log level.

    Parameters
    ----------
    level : str
        The verbosity level.
        Valid values are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` or ``CRITICAL``.

    """

    LOG_LEVEL = LEVELS[level]
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
        level=LOG_LEVEL,
    )
