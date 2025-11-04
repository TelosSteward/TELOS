#!/usr/bin/env python3
"""
Mistral Client Adapter for Dual Attractor
==========================================

Adapts MistralClient to work with dual_attractor.py which expects
Anthropic-style message interface.

This allows dual_attractor.py to work with Mistral without modifying
the core dual_attractor code.
"""

from typing import List, Dict, Any
from telos_purpose.llm_clients.mistral_client import MistralClient


class MessageResponse:
    """Mock Anthropic message response structure"""
    def __init__(self, text: str):
        self.content = [type('obj', (object,), {'text': text})]


class Messages:
    """Mock Anthropic messages interface using Mistral backend"""

    def __init__(self, mistral_client: MistralClient):
        self.mistral_client = mistral_client

    def create(
        self,
        model: str,
        max_tokens: int,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> MessageResponse:
        """
        Create message using Mistral client but return Anthropic-style response

        Args:
            model: Ignored (uses MistralClient's configured model)
            max_tokens: Max tokens to generate
            messages: Message list in Anthropic format

        Returns:
            MessageResponse with .content[0].text structure
        """
        # Call Mistral generate
        response_text = self.mistral_client.generate(
            messages=messages,
            max_tokens=max_tokens
        )

        # Return in Anthropic format
        return MessageResponse(response_text)


class MistralClientAdapter:
    """
    Adapter that makes MistralClient compatible with dual_attractor.py

    Usage:
        mistral = MistralClient(api_key=os.getenv('MISTRAL_API_KEY'))
        adapter = MistralClientAdapter(mistral)

        # Now dual_attractor.py can call:
        await asyncio.to_thread(
            adapter.messages.create,
            model="ignored",
            max_tokens=100,
            messages=[{"role": "user", "content": "..."}]
        )
    """

    def __init__(self, mistral_client: MistralClient):
        self.mistral_client = mistral_client
        self.messages = Messages(mistral_client)

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Pass-through to Mistral generate for direct calls"""
        return self.mistral_client.generate(messages, **kwargs)

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Pass-through to Mistral chat_completion"""
        return self.mistral_client.chat_completion(messages, **kwargs)


def create_adapter(api_key: str = None) -> MistralClientAdapter:
    """
    Factory function to create adapted Mistral client

    Args:
        api_key: Mistral API key (or uses MISTRAL_API_KEY env var)

    Returns:
        MistralClientAdapter ready for use with dual_attractor.py
    """
    mistral = MistralClient(api_key=api_key)
    return MistralClientAdapter(mistral)
