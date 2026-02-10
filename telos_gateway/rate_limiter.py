"""
Sliding Window Rate Limiter
============================

Per-API-key rate limiting with configurable requests-per-minute
and burst capacity. Returns 429 with Retry-After header when exceeded.
"""

import logging
import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """
    In-memory sliding-window rate limiter keyed by API key.

    Each key tracks timestamps of recent requests within a 60-second window.
    When the count exceeds the configured limit, subsequent requests are rejected
    with HTTP 429 until old entries fall outside the window.
    """

    def __init__(self, requests_per_minute: int = 60, burst: int = 10):
        self.rpm = requests_per_minute
        self.burst = burst
        # key -> list of request timestamps
        self._windows: dict[str, list[float]] = defaultdict(list)

    def _extract_key(self, request: Request) -> str:
        """Extract rate-limit key from request (API key or IP)."""
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer ") and len(auth) > 7:
            return f"key:{auth[7:][:16]}"  # first 16 chars of key as bucket id
        # Fall back to client IP
        client = request.client
        return f"ip:{client.host}" if client else "ip:unknown"

    def _prune(self, key: str, now: float) -> None:
        """Remove timestamps older than 60 seconds from the window."""
        cutoff = now - 60.0
        timestamps = self._windows[key]
        # Find first index within window
        idx = 0
        for idx, ts in enumerate(timestamps):
            if ts >= cutoff:
                break
        else:
            idx = len(timestamps)
        if idx > 0:
            self._windows[key] = timestamps[idx:]

    def check(self, request: Request) -> Optional[int]:
        """
        Check if request is within rate limits.

        Returns None if allowed, or seconds to wait (Retry-After) if rejected.
        """
        key = self._extract_key(request)
        now = time.monotonic()
        self._prune(key, now)

        window = self._windows[key]
        if len(window) >= self.rpm:
            # Calculate when the oldest request in window will expire
            retry_after = int(60.0 - (now - window[0])) + 1
            return max(retry_after, 1)

        window.append(now)
        return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that applies rate limiting to all requests."""

    def __init__(self, app, limiter: SlidingWindowRateLimiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/"):
            return await call_next(request)

        retry_after = self.limiter.check(request)
        if retry_after is not None:
            key = self.limiter._extract_key(request)
            logger.warning(f"Rate limit exceeded for {key}, retry after {retry_after}s")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "message": f"Rate limit exceeded. Retry after {retry_after} seconds.",
                        "type": "rate_limit_error",
                        "code": 429,
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
