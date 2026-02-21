"""
Sliding Window Rate Limiter
============================

Per-API-key rate limiting with configurable requests-per-minute
and burst capacity. Returns 429 with Retry-After header when exceeded.
"""

import hashlib
import logging
import time
from collections import defaultdict
from typing import Optional

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Maximum number of tracked rate limit keys before forced eviction
_MAX_KEYS = 10_000


class SlidingWindowRateLimiter:
    """
    In-memory sliding-window rate limiter keyed by client IP.

    Each key tracks timestamps of recent requests within a 60-second window.
    When the count exceeds the configured limit, subsequent requests are rejected
    with HTTP 429 until old entries fall outside the window.

    Rate limiting is keyed by client IP address (not bearer token) to prevent
    bypass via token rotation.
    """

    def __init__(self, requests_per_minute: int = 60, burst: int = 10):
        self.rpm = requests_per_minute
        self.burst = burst
        # key -> list of request timestamps
        self._windows: dict[str, list[float]] = defaultdict(list)

    def _extract_key(self, request: Request) -> str:
        """Extract rate-limit key from request using client IP.

        Uses client IP address as the primary key. Bearer tokens are NOT used
        because auth accepts any non-empty token, allowing trivial key rotation.
        """
        client = request.client
        ip = client.host if client else "unknown"
        # Also incorporate X-Forwarded-For if behind a reverse proxy
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        return f"ip:{ip}"

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
        # Evict empty keys to prevent unbounded memory growth
        if not self._windows[key]:
            del self._windows[key]

    def _evict_stale_keys(self, now: float) -> None:
        """Remove stale keys if dictionary grows too large."""
        if len(self._windows) <= _MAX_KEYS:
            return
        cutoff = now - 60.0
        stale = [k for k, v in self._windows.items() if not v or v[-1] < cutoff]
        for k in stale:
            del self._windows[k]
        if len(self._windows) > _MAX_KEYS:
            logger.warning(f"Rate limiter has {len(self._windows)} active keys")

    def check(self, request: Request) -> Optional[int]:
        """
        Check if request is within rate limits.

        Returns None if allowed, or seconds to wait (Retry-After) if rejected.
        Allows burst capacity above rpm up to (rpm + burst) per minute.
        """
        key = self._extract_key(request)
        now = time.monotonic()
        self._prune(key, now)
        self._evict_stale_keys(now)

        window = self._windows[key]
        max_requests = self.rpm + self.burst
        if len(window) >= max_requests:
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
