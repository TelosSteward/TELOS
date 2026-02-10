"""
TELOS Gateway -- Main Entry Point
===================================

Production-hardened FastAPI gateway with:
- Configurable CORS (NEVER allow_origins=["*"])
- API key authentication
- Per-key sliding-window rate limiting
- Health checks with uptime
- Structured JSON error responses

Usage:
    uvicorn telos_gateway.main:app --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from telos_gateway import __version__
from telos_gateway.config import config
from telos_gateway.rate_limiter import RateLimitMiddleware, SlidingWindowRateLimiter
from telos_gateway.routes.chat_completions import router as chat_router
from telos_gateway.routes.health import router as health_router, set_start_time

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan -- initialize on startup, cleanup on shutdown."""
    logger.info("=" * 60)
    logger.info(f"TELOS Gateway v{__version__} Starting")
    logger.info("=" * 60)
    logger.info(f"Allowed origins: {config.allowed_origins}")
    logger.info(f"Rate limit: {config.rate_limit_rpm} req/min")
    logger.info("=" * 60)

    set_start_time()

    yield

    logger.info("TELOS Gateway shutting down")


# ============================================================================
# App
# ============================================================================


app = FastAPI(
    title="TELOS Gateway",
    description=(
        "OpenAI-compatible API gateway with TELOS governance. "
        "The Constitutional Filter for agentic AI."
    ),
    version=__version__,
    lifespan=lifespan,
)


# -- CORS: configurable origins, NEVER ["*"] --
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-TELOS-High-Risk"],
)


# -- Rate limiting middleware --
_limiter = SlidingWindowRateLimiter(
    requests_per_minute=config.rate_limit_rpm,
    burst=config.rate_limit_burst,
)
app.add_middleware(RateLimitMiddleware, limiter=_limiter)


# ============================================================================
# Structured error handler
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler that returns consistent JSON error format."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": 500,
            }
        },
    )


# ============================================================================
# Routes
# ============================================================================


app.include_router(health_router)
app.include_router(chat_router)


@app.get("/")
async def root():
    """Root endpoint -- gateway info."""
    return {
        "name": "TELOS Gateway",
        "version": __version__,
        "description": "OpenAI-compatible API gateway with TELOS governance",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "health": "/health",
        },
    }


@app.get("/v1/models")
async def list_models():
    """List available models (placeholder for proxy)."""
    return {
        "object": "list",
        "data": [
            {"id": "gpt-4", "object": "model", "owned_by": "openai", "permission": []},
            {"id": "gpt-4-turbo", "object": "model", "owned_by": "openai", "permission": []},
            {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai", "permission": []},
            {
                "id": "mistral-small-latest",
                "object": "model",
                "owned_by": "mistral",
                "permission": [],
            },
        ],
    }


# ============================================================================
# Entrypoint
# ============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "telos_gateway.main:app",
        host=config.host,
        port=config.port,
        reload=True,
    )
