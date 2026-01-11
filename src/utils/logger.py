import sys

from loguru import logger

# Configure logger to output to stderr
logger.remove()
logger.add(sys.stderr, level="INFO")

__all__ = ["logger"]
