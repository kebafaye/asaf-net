"""Structured logging for experiments."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logger(name: str = "depth_fusion", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger
