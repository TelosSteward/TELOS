"""Gateway configuration from environment variables."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class GatewayConfig(BaseModel):
    """Gateway configuration settings loaded from environment."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS - NEVER use ["*"] in production
    allowed_origins: list[str] = ["http://localhost:8501"]

    # Rate limiting
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10

    # Logging
    log_level: str = "INFO"

    # LLM Provider settings
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    mistral_api_key: Optional[str] = None

    # Embedding settings
    embedding_provider: str = "mistral"
    embedding_model: str = "mistral-embed"

    # Governance thresholds (calibrated for NORMALIZED fidelity 0-1)
    agentic_execute_threshold: float = 0.45
    agentic_clarify_threshold: float = 0.35
    agentic_suggest_threshold: float = 0.30
    similarity_baseline: float = 0.15

    # Governance behavior
    block_on_low_fidelity: bool = True
    log_governance_traces: bool = True
    trace_storage_dir: Path = Path("./telos_gateway_traces")

    # Streaming behavior
    allow_streaming: bool = False


    @classmethod
    def from_env(cls) -> "GatewayConfig":
        """Load configuration from environment variables."""
        origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501")
        return cls(
            host=os.environ.get("TELOS_GATEWAY_HOST", "0.0.0.0"),
            port=int(os.environ.get("TELOS_GATEWAY_PORT", "8000")),
            allowed_origins=[o.strip() for o in origins.split(",")],
            rate_limit_rpm=int(os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", "60")),
            rate_limit_burst=int(os.environ.get("RATE_LIMIT_BURST", "10")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            mistral_api_key=os.environ.get("MISTRAL_API_KEY"),
            embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "mistral"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "mistral-embed"),
            agentic_execute_threshold=float(
                os.environ.get("AGENTIC_EXECUTE_THRESHOLD", "0.45")
            ),
            agentic_clarify_threshold=float(
                os.environ.get("AGENTIC_CLARIFY_THRESHOLD", "0.35")
            ),
            agentic_suggest_threshold=float(
                os.environ.get("AGENTIC_SUGGEST_THRESHOLD", "0.30")
            ),
            similarity_baseline=float(os.environ.get("SIMILARITY_BASELINE", "0.15")),
            block_on_low_fidelity=os.environ.get("BLOCK_ON_LOW_FIDELITY", "true").lower()
            == "true",
            log_governance_traces=os.environ.get("LOG_GOVERNANCE_TRACES", "true").lower()
            == "true",
            trace_storage_dir=Path(
                os.environ.get("TRACE_STORAGE_DIR", "./telos_gateway_traces")
            ),
            allow_streaming=os.environ.get("ALLOW_STREAMING", "false").lower() == "true",
        )


# Global config instance -- loaded from environment
config = GatewayConfig.from_env()
