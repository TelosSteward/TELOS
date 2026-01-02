"""
Base LLM Provider
=================

Abstract interface for LLM providers.
TELOS Gateway passes requests through to the real LLM provider
after governance checks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..models.openai_types import ChatCompletionRequest, ChatCompletionResponse


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    TELOS Gateway is AGNOSTIC to which LLM the agent uses.
    We just pass through to wherever they're calling.
    """

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        api_key: str,
    ) -> ChatCompletionResponse:
        """
        Send chat completion request to the real LLM.

        Args:
            request: The (possibly modified) chat completion request
            api_key: The customer's API key for the LLM provider

        Returns:
            The LLM's response
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')."""
        pass
