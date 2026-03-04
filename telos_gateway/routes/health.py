"""
Health Check Route
==================

GET /health endpoint with status, version, and uptime.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter

from telos_gateway import __version__

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Set at startup
_start_time: Optional[float] = None


def set_start_time() -> None:
    """Record gateway start time for uptime calculation."""
    global _start_time
    _start_time = time.time()


@router.get("/health")
async def health():
    """
    Health check endpoint.

    Returns status, version, and uptime in seconds.
    """
    uptime = int(time.time() - _start_time) if _start_time else 0
    return {
        "status": "healthy",
        "version": __version__,
        "uptime_seconds": uptime,
        "gateway": "telos",
    }
