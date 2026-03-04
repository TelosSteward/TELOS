"""
API Key Authentication
======================

Bearer token validation middleware for the TELOS Gateway.
Validates API keys against the agent registry using SHA-256 hashing,
or against a gateway-level API key allowlist.

Security: All tokens must either match a registered TELOS agent key
or match a gateway API key from the TELOS_GATEWAY_API_KEYS allowlist.
Unrecognized tokens are rejected with 401.
"""

import hashlib
import logging
import os
from typing import Optional, Set

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .registry import get_registry
from .registry.agent_profile import AgentProfile

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


def _hash_key(key: str) -> str:
    """Hash an API key with SHA-256 for comparison against stored hashes."""
    return hashlib.sha256(key.encode()).hexdigest()


def _load_gateway_api_keys() -> Set[str]:
    """Load gateway API key hashes from TELOS_GATEWAY_API_KEYS env var.

    The env var should contain comma-separated SHA-256 hashes of valid API keys.
    Example: TELOS_GATEWAY_API_KEYS="abc123hash,def456hash"

    If not set, returns empty set (no gateway keys accepted).
    """
    raw = os.environ.get("TELOS_GATEWAY_API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


# Load once at module init; reload on each call for runtime flexibility
_gateway_keys_cache: Optional[Set[str]] = None


def _get_gateway_keys() -> Set[str]:
    """Get gateway API key hashes, with lazy loading."""
    global _gateway_keys_cache
    if _gateway_keys_cache is None:
        _gateway_keys_cache = _load_gateway_api_keys()
    return _gateway_keys_cache


def _is_valid_gateway_key(api_key: str) -> bool:
    """Check if an API key matches the gateway allowlist (SHA-256 hash comparison)."""
    keys = _get_gateway_keys()
    if not keys:
        return False
    key_hash = _hash_key(api_key)
    return key_hash in keys


async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> str:
    """
    Validate Bearer token from the Authorization header.

    Tokens must be either:
    1. A registered TELOS agent key (prefixed "telos-agent-"), or
    2. A gateway API key matching the TELOS_GATEWAY_API_KEYS allowlist.

    Returns the raw API key string on success.
    Raises 401 if missing, empty, or not recognized.
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Missing Authorization header. Use: Bearer <api_key>",
                    "type": "authentication_error",
                    "code": 401,
                }
            },
        )

    api_key = credentials.credentials
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Empty API key",
                    "type": "authentication_error",
                    "code": 401,
                }
            },
        )

    # TELOS agent keys are validated via registry lookup downstream
    if api_key.startswith("telos-agent-"):
        return api_key

    # Non-agent keys must match the gateway allowlist
    if not _is_valid_gateway_key(api_key):
        logger.warning("Rejected unrecognized API key (not in gateway allowlist)")
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key. Register as a TELOS agent or use a valid gateway key.",
                    "type": "authentication_error",
                    "code": 401,
                }
            },
        )

    return api_key


def lookup_agent(api_key: str) -> Optional[AgentProfile]:
    """
    Look up an agent profile by API key.

    For TELOS agent keys (prefixed "telos-agent-"), performs registry lookup
    using SHA-256 hash comparison. Returns None for non-TELOS keys (passthrough).
    """
    if not api_key.startswith("telos-agent-"):
        return None

    registry = get_registry()
    profile = registry.get_agent_by_api_key(api_key)

    if profile is None:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid TELOS agent API key",
                    "type": "authentication_error",
                    "code": 401,
                }
            },
        )

    if not profile.is_active:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": "Agent has been deactivated",
                    "type": "permission_error",
                    "code": 403,
                }
            },
        )

    logger.info(f"Authenticated agent: {profile.name} ({profile.agent_id})")
    return profile
