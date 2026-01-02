"""
LangChain Chatbot with TELOS Governance
========================================

This demo shows a LangChain chatbot running through TELOS Gateway.
The gateway applies governance - blocking off-topic requests and
forwarding aligned requests to the real LLM.

Usage:
    1. Start TELOS Gateway: uvicorn telos_gateway.main:app --port 8000
    2. Run this script: python telos_gateway/demos/langchain_chatbot.py

The chatbot is configured as a "Python coding assistant" - it will:
    - EXECUTE: Python coding questions (high fidelity)
    - SUGGEST: Tangentially related tech questions (medium fidelity)
    - BLOCK: Completely off-topic requests like recipes (low fidelity)
"""

import os
import sys

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def create_governed_chatbot(
    gateway_url: str = "http://localhost:8000/v1",
    api_key: str = None,
) -> ChatOpenAI:
    """
    Create a LangChain chatbot that routes through TELOS Gateway.

    The only change needed: point base_url at the gateway.
    Everything else (API key, model, etc.) stays the same.
    """
    # Use provided key or fall back to environment
    key = api_key or os.getenv("OPENAI_API_KEY", "sk-test-key")

    return ChatOpenAI(
        model="gpt-4",
        base_url=gateway_url,  # <-- This is the magic: route through TELOS
        api_key=key,
        temperature=0.7,
    )


# The Primacy Attractor - defines what this agent is FOR
SYSTEM_PROMPT = """You are a helpful assistant that helps users write Python code.
Focus on Python programming, debugging, and best practices.
You can also help with related topics like software architecture and testing."""


def demo_governance():
    """
    Demonstrate TELOS governance with different types of queries.
    """
    print("=" * 60)
    print("LangChain Chatbot with TELOS Governance")
    print("=" * 60)
    print()

    # Create the governed chatbot
    chatbot = create_governed_chatbot()

    # Test queries with different fidelity levels
    test_queries = [
        ("HIGH FIDELITY - Should EXECUTE", "How do I write a Python function that calculates factorial?"),
        ("MEDIUM FIDELITY - Should SUGGEST", "What is machine learning?"),
        ("LOW FIDELITY - Should BLOCK", "What's the best recipe for chocolate cake?"),
    ]

    for label, query in test_queries:
        print(f"\n{'='*60}")
        print(f"Test: {label}")
        print(f"Query: {query}")
        print("-" * 60)

        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]

            response = chatbot.invoke(messages)
            print(f"Response: {response.content[:200]}...")

            # Check for governance metadata (if available)
            if hasattr(response, 'response_metadata'):
                meta = response.response_metadata
                if 'telos_governance' in meta:
                    gov = meta['telos_governance']
                    print(f"\nGovernance:")
                    print(f"  Decision: {gov.get('decision')}")
                    print(f"  Fidelity: {gov.get('input_fidelity', 'N/A')}")
                    print(f"  Blocked: {gov.get('blocked', False)}")

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                print("Response: [Governance PASSED - forwarded to LLM]")
                print("         (401 = no valid API key, but governance worked!)")
            elif "blocked" in error_msg.lower():
                print(f"Response: [BLOCKED by governance]")
            else:
                print(f"Error: {error_msg[:200]}")

        print()


def interactive_mode():
    """
    Run an interactive chat session with TELOS governance.
    """
    print("=" * 60)
    print("Interactive LangChain Chat with TELOS Governance")
    print("=" * 60)
    print(f"\nSystem Prompt: {SYSTEM_PROMPT[:100]}...")
    print("\nType 'quit' to exit. Try on-topic and off-topic queries!")
    print()

    chatbot = create_governed_chatbot()

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_input),
            ]

            response = chatbot.invoke(messages)
            print(f"\nAssistant: {response.content}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                print("\n[Governance PASSED - would forward to LLM with valid API key]\n")
            else:
                print(f"\nError: {error_msg[:200]}\n")

    print("\nGoodbye!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LangChain chatbot with TELOS governance")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--gateway", default="http://localhost:8000/v1", help="TELOS Gateway URL")
    args = parser.parse_args()

    print("\nMake sure TELOS Gateway is running:")
    print("  uvicorn telos_gateway.main:app --port 8000")
    print()

    if args.interactive:
        interactive_mode()
    else:
        demo_governance()
