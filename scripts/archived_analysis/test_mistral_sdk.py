#!/usr/bin/env python3
"""
Test Mistral SDK authentication issue.
"""

import os
import sys
sys.path.append('/Users/brunnerjf/Desktop/telos_privacy')

def test_mistral_sdk():
    """Test the exact SDK initialization used in the app."""

    api_key = os.getenv("MISTRAL_API_KEY")
    print(f"Using API key: {api_key[:10]}...")

    print("\n1. Testing with mistralai SDK (same as app):")
    try:
        from mistralai import Mistral

        # Initialize exactly as in MistralClient
        client = Mistral(api_key=api_key)

        # Test the chat.complete method (same as in MistralClient.generate)
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": "Say hello"}],
            temperature=0.7,
            max_tokens=10
        )

        if response and response.choices:
            print(f"✅ SDK call successful! Response: {response.choices[0].message.content}")
            return True
        else:
            print("❌ SDK call returned empty response")
            return False

    except Exception as e:
        print(f"❌ SDK call failed: {e}")
        print(f"   Error type: {type(e).__name__}")

        # Check if it's an authentication error
        if "401" in str(e) or "Unauthorized" in str(e):
            print("\n⚠️  Authentication issue detected!")
            print("The SDK is not accepting the API key properly")

        return False

def test_with_telos_client():
    """Test using the actual TELOSMistralClient."""

    print("\n2. Testing with TELOS MistralClient:")
    try:
        from telos_purpose.llm_clients.mistral_client import MistralClient

        # Initialize with explicit API key
        api_key = os.getenv("MISTRAL_API_KEY")
        client = MistralClient(api_key=api_key, model="mistral-large-latest")

        # Test generate method
        response = client.generate(
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )

        print(f"✅ TELOS client successful! Response: {response}")
        return True

    except Exception as e:
        print(f"❌ TELOS client failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MISTRAL SDK AUTHENTICATION TEST")
    print("=" * 60)

    sdk_works = test_mistral_sdk()
    client_works = test_with_telos_client()

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    if sdk_works and client_works:
        print("✅ Both SDK and client are working!")
    elif not sdk_works:
        print("❌ SDK authentication is failing")
        print("\nPossible fix: Update mistralai package")
        print("Run: pip3 install --upgrade mistralai")
    else:
        print("❌ Issue with TELOS client wrapper")