"""
TELOS Gateway - Main Entry Point
=================================

OpenAI-compatible API gateway with TELOS governance.

Usage:
    uvicorn telos_gateway.main:app --port 8000

Then point your agent at:
    OPENAI_BASE_URL=http://localhost:8000/v1
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telos_gateway.config import config
from telos_gateway.routes.chat_completions import router as chat_router, initialize_components
from telos_gateway.routes.agent_registration import router as agent_router
from telos_gateway.registry import get_registry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_embedding_function():
    """
    Get the embedding function for fidelity calculation.

    Uses local SentenceTransformer by default (free, fast).
    Can also use Mistral API for higher quality.
    """
    if config.embedding_provider == "sentence_transformer":
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            logger.info(f"Loading SentenceTransformer model: {config.embedding_model}")
            model = SentenceTransformer(config.embedding_model)

            def embed(text: str) -> np.ndarray:
                return model.encode(text, convert_to_numpy=True)

            # Test the model
            test_embedding = embed("test")
            logger.info(f"Embedding model loaded. Dimension: {len(test_embedding)}")

            return embed

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise

    elif config.embedding_provider == "mistral":
        # Use Mistral API for embeddings
        import httpx
        import numpy as np

        api_key = config.mistral_api_key
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set for Mistral embedding provider")

        def embed(text: str) -> np.ndarray:
            response = httpx.post(
                "https://api.mistral.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "mistral-embed", "input": [text]},
            )
            response.raise_for_status()
            data = response.json()
            return np.array(data["data"][0]["embedding"])

        return embed

    else:
        raise ValueError(f"Unknown embedding provider: {config.embedding_provider}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize on startup, cleanup on shutdown."""
    # Startup
    logger.info("=" * 60)
    logger.info("TELOS Gateway Starting")
    logger.info("=" * 60)
    logger.info(f"Embedding provider: {config.embedding_provider}")
    logger.info(f"Execute threshold: {config.agentic_execute_threshold}")
    logger.info(f"Clarify threshold: {config.agentic_clarify_threshold}")
    logger.info(f"Suggest threshold: {config.agentic_suggest_threshold}")
    logger.info(f"Baseline threshold: {config.similarity_baseline}")
    logger.info("=" * 60)

    # Initialize embedding function and components
    try:
        embed_fn = get_embedding_function()
        initialize_components(embed_fn)

        # Initialize registry with embedding function
        registry = get_registry()
        registry.set_embed_fn(embed_fn)
        agent_count = len(registry.list_agents())
        logger.info(f"Agent Registry initialized with {agent_count} registered agents")

        logger.info("TELOS Gateway initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise

    yield

    # Shutdown
    logger.info("TELOS Gateway shutting down")


# Create FastAPI app
app = FastAPI(
    title="TELOS Gateway",
    description=(
        "OpenAI-compatible API gateway with TELOS governance. "
        "The Constitutional Filter for agentic AI."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(chat_router)
app.include_router(agent_router)


@app.get("/")
async def root():
    """Root endpoint - gateway info."""
    return {
        "name": "TELOS Gateway",
        "version": "0.1.0",
        "description": "OpenAI-compatible API gateway with TELOS governance",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "health": "/health",
        },
        "governance": {
            "execute_threshold": config.agentic_execute_threshold,
            "clarify_threshold": config.agentic_clarify_threshold,
            "suggest_threshold": config.agentic_suggest_threshold,
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "gateway": "telos"}


@app.get("/v1/models")
async def list_models():
    """
    List available models (proxied from OpenAI).

    For now, just return a placeholder. In production,
    this would proxy to the real OpenAI models endpoint.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-4",
                "object": "model",
                "owned_by": "openai",
                "permission": [],
            },
            {
                "id": "gpt-4-turbo",
                "object": "model",
                "owned_by": "openai",
                "permission": [],
            },
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "owned_by": "openai",
                "permission": [],
            },
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "telos_gateway.main:app",
        host=config.host,
        port=config.port,
        reload=True,
    )
