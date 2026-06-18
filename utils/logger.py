"""
utils/logger.py — Structured Experiment Logger
===============================================
Provides a consistent logging interface used across all training scripts,
evaluation runs, and ablation studies.

Each experiment gets its own timestamped log file under logs/.
Console output mirrors file output for real-time monitoring.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_dir: Optional[Path | str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create or retrieve a named logger with file + console handlers.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).
    log_dir : Path or str, optional
        Directory for log files. Defaults to <project_root>/logs/.
        Created automatically if it does not exist.
    level : int
        Logging level. Default: logging.INFO.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Example
    -------
    >>> log = get_logger(__name__, log_dir="logs/")
    >>> log.info("Training started | epoch=1 | lr=0.001")
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name.replace('.', '_')}_{timestamp}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


if __name__ == "__main__":
    log = get_logger("smoke_test", log_dir=Path(__file__).parent.parent / "logs")
    log.info("Logger smoke test: INFO message")
    log.warning("Logger smoke test: WARNING message")
    print("Logger smoke test passed.")
