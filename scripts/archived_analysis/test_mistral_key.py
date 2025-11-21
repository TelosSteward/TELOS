#!/usr/bin/env python3
"""
Test if the MISTRAL_API_KEY is valid.
"""

import os
import sys

# Add project to path
sys.path.insert(0, '/Users/brunnerjf/Desktop/telos_privacy')

def test_mistral_api_key():
    """Test if Mistral API key is valid by making a test call."""

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        print("❌ No MISTRAL_API_KEY found in environment")
        return False

    print(f"✅ Found API key: {api_key[:10]}... (length: {len(api_key)})")

    try:
        # Test with the MistralClient from the project
        from telos_purpose.llm_clients.mistral_client import MistralClient

        print("\nTesting API key with MistralClient...")
        client = MistralClient(
            api_key=api_key,
            model="mistral-large-latest"
        )

        # Try a simple test generation
        response = client.generate(
            prompt="Hello, this is a test. Please respond with 'Test successful'.",
            max_tokens=20
        )

        print(f"✅ API call successful! Response: {response[:50]}...")
        return True

    except Exception as e:
        print(f"❌ API call failed: {e}")

        # Try direct API call to check if it's a client issue
        print("\nTrying direct API call...")
        import requests

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "mistral-large-latest",
            "messages": [
                {"role": "user", "content": "Test"}
            ],
            "max_tokens": 10
        }

        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                print("✅ Direct API call successful!")
                return True
            else:
                print(f"❌ API returned status {response.status_code}: {response.text}")
                return False

        except Exception as req_error:
            print(f"❌ Direct API call failed: {req_error}")
            return False

if __name__ == "__main__":
    print("=" * 60)
    print("MISTRAL API KEY VALIDATION")
    print("=" * 60)

    if test_mistral_api_key():
        print("\n✅ API key is valid and working!")
    else:
        print("\n❌ API key is invalid or there's a connection issue")
        print("\nTroubleshooting steps:")
        print("1. Check if your MISTRAL_API_KEY is correct")
        print("2. Ensure you have an active Mistral AI account")
        print("3. Check if the key has the correct permissions")
        print("4. Try regenerating the key from Mistral dashboard")