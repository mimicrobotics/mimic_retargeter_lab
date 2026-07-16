"""Logger utilities for the project."""

import logging
from typing import Optional

# ----- CONSTANTS -----
_DEFAULT_FMT = "[%(levelname)s] [%(asctime)s] [%(name)s] [%(funcName)s]: %(message)s"
_CONFIGURED = False


def _resolve_level(level: str | int | None) -> int:
    """Resolve the logging level from the input.

    Sets logging level to INFO if not specified.
    """
    if isinstance(level, int):
        return level
    name = (level or "INFO").upper()
    return getattr(logging, name, logging.INFO)


def configure_logging(
    level: str | int | None = None,
    fmt: str = _DEFAULT_FMT,
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    force: bool = False,
) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    logging.basicConfig(
        level=_resolve_level(level),
        format=fmt,
        datefmt=datefmt,
        force=force,  # Python 3.8+, resets root handlers when needed
    )
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    return logger
