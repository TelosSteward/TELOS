"""
Mistral Provider
================

Provider for Mistral AI API -- compatible with OpenAI format.
"""

import json
import logging
from typing import Any, AsyncIterator, Dict

import httpx

from .base import LLMProvider

logger = logging.getLogger(__name__)


class MistralProvider(LLMProvider):
    """
    Provider for Mistral AI API.

    Mistral uses an OpenAI-compatible format, so this is straightforward.
    """

    def __init__(
        self,
        base_url: str = "https://api.mistral.ai/v1",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_provider_name(self) -> str:
        return "mistral"

    def _map_model(self, model: str) -> str:
        """Map OpenAI model names to Mistral equivalents."""
        if model.startswith("gpt"):
            return "mistral-small-latest"
        return model

    async def chat_completion(
        self,
        request: Any,
        api_key: str,
    ) -> Dict[str, Any]:
        """Forward chat completion to Mistral and return raw dict."""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        request_data = request.model_dump(exclude_none=True)
        request_data["model"] = self._map_model(request_data.get("model", "mistral-small-latest"))
        request_data["stream"] = False

        logger.debug(f"Forwarding to Mistral: model={request_data['model']}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=request_data)
            response.raise_for_status()
            return response.json()

    async def chat_completion_stream(
        self,
        request: Any,
        api_key: str,
    ) -> AsyncIterator[str]:
        """Stream chat completion from Mistral as SSE chunks."""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        request_data = request.model_dump(exclude_none=True)
        request_data["model"] = self._map_model(request_data.get("model", "mistral-small-latest"))
        request_data["stream"] = True

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
