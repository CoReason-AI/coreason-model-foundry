# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

import sys
from pathlib import Path

from loguru import logger

# Ensure logs directory exists
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

# Configure logger
logger.remove()

# Console Sink (Stderr, Human-readable)
logger.add(sys.stderr, level="INFO")

# File Sink (JSON, Rotated, Retained)
logger.add(
    LOG_FILE,
    level="INFO",
    rotation="500 MB",
    retention="10 days",
    serialize=True,
)

__all__ = ["logger"]
