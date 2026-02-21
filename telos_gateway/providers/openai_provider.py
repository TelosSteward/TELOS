"""
OpenAI Provider
===============

Passthrough provider for OpenAI API.
The agent's requests go to OpenAI -- we just govern them first.
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Dict

import httpx

from .base import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    Passthrough to OpenAI API.

    The agent keeps using OpenAI -- we don't replace it.
    We just measure fidelity and pass through (or block).
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_provider_name(self) -> str:
        return "openai"

    async def chat_completion(
        self,
        request: Any,
        api_key: str,
    ) -> Dict[str, Any]:
        """Forward chat completion to OpenAI and return raw dict."""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        request_data = request.model_dump(exclude_none=True)
        request_data["stream"] = False

        logger.debug(f"Forwarding to OpenAI: model={request.model}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=request_data)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API error: HTTP {e.response.status_code}")
            raise ValueError("LLM provider returned an error") from None
        except httpx.RequestError as e:
            logger.error(f"OpenAI connection error: {type(e).__name__}")
            raise ValueError("Failed to connect to LLM provider") from None

    async def chat_completion_stream(
        self,
        request: Any,
        api_key: str,
    ) -> AsyncIterator[str]:
        """Stream chat completion from OpenAI as SSE chunks."""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        request_data = request.model_dump(exclude_none=True)
        request_data["stream"] = True

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", url, headers=headers, json=request_data
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            yield line + "\n\n"
                        elif line.strip() == "":
                            continue
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI streaming error: HTTP {e.response.status_code}")
            yield f'data: {{"error": "LLM provider returned an error"}}\n\n'
        except httpx.RequestError as e:
            logger.error(f"OpenAI streaming connection error: {type(e).__name__}")
            yield f'data: {{"error": "Failed to connect to LLM provider"}}\n\n'


class MockProvider(LLMProvider):
    """Mock provider for testing without real API calls."""

    def get_provider_name(self) -> str:
        return "mock"

    async def chat_completion(
        self,
        request: Any,
        api_key: str,
    ) -> Dict[str, Any]:
        """Return a mock response dict."""
        now = int(time.time())
        return {
            "id": f"mock-{now}",
            "object": "chat.completion",
            "created": now,
            "model": getattr(request, "model", "mock"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a mock response from TELOS Gateway testing.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            },
        }

    async def chat_completion_stream(
        self,
        request: Any,
        api_key: str,
    ) -> AsyncIterator[str]:
        """Stream a mock response."""
        now = int(time.time())
        chunk = {
            "id": f"mock-{now}",
            "object": "chat.completion.chunk",
            "created": now,
            "model": getattr(request, "model", "mock"),
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Mock streaming response."},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
