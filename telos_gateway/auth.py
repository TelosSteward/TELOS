"""
API Key Authentication
======================

Bearer token validation middleware for the TELOS Gateway.
Validates API keys from the Authorization header.
"""

import logging
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


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
