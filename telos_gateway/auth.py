"""
API Key Authentication
======================

Bearer token validation middleware for the TELOS Gateway.
Validates API keys against the agent registry using SHA-256 hashing.
"""

import hashlib
import logging
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .registry import get_registry
from .registry.agent_profile import AgentProfile

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


def _hash_key(key: str) -> str:
    """Hash an API key with SHA-256 for comparison against stored hashes."""
    return hashlib.sha256(key.encode()).hexdigest()


async def require_auth(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> str:
    """
    Validate Bearer token from the Authorization header.

    Returns the raw API key string on success.
    Raises 401 if missing or invalid.

    This dependency can be injected into any route that needs authentication.
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
