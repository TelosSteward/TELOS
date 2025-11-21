#!/usr/bin/env python3
"""
Verify and test your Mistral API key.
"""

import os
import requests

def verify_mistral_key():
    """Comprehensive Mistral API key verification."""

    print("=" * 60)
    print("MISTRAL API KEY VERIFICATION")
    print("=" * 60)

    # Get the API key
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        print("❌ No MISTRAL_API_KEY found in environment")
        print("\nTo set it, run:")
        print('export MISTRAL_API_KEY="your-api-key-here"')
        return False

    print(f"✅ Found API key: {api_key[:10]}...")
    print(f"   Length: {len(api_key)} characters")

    # Test the API key with a direct HTTP request
    print("\n🔍 Testing API key with Mistral API...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Simple test message
    data = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": "Say 'Hello' in one word"}
        ],
        "max_tokens": 10,
        "temperature": 0
    }

    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )

        if response.status_code == 200:
            print("✅ API key is VALID and working!")
            result = response.json()
            if 'choices' in result and result['choices']:
                content = result['choices'][0]['message']['content']
                print(f"✅ Test response: {content}")
            return True

        elif response.status_code == 401:
            print("❌ API key is INVALID (401 Unauthorized)")
            print("\nPossible issues:")
            print("1. The API key is incorrect")
            print("2. The API key has been revoked")
            print("3. The API key has expired")
            print("\nSolution:")
            print("1. Go to https://console.mistral.ai/")
            print("2. Navigate to API Keys section")
            print("3. Create a new API key")
            print("4. Update your environment variable:")
            print('   export MISTRAL_API_KEY="your-new-key"')
            return False

        elif response.status_code == 429:
            print("⚠️  Rate limit exceeded (429)")
            print("The API key is valid but you've hit the rate limit")
            return True

        else:
            print(f"❌ Unexpected status: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Request timed out - check your internet connection")
        return False

    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def check_environment():
    """Check environment and dependencies."""

    print("\n📋 Environment Check:")

    # Check if running in Streamlit
    try:
        import streamlit as st
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not installed")

    # Check mistralai package
    try:
        from mistralai import Mistral
        print("✅ mistralai package is installed")
    except ImportError:
        print("❌ mistralai package not installed")
        print("   Run: pip install mistralai")

    # Check for .env file
    if os.path.exists(".env"):
        print("✅ .env file found")
    else:
        print("ℹ️  No .env file (using environment variables)")

    # Check for Streamlit secrets
    streamlit_secrets = os.path.expanduser("~/.streamlit/secrets.toml")
    if os.path.exists(streamlit_secrets):
        print(f"✅ Streamlit secrets file found: {streamlit_secrets}")
    else:
        print("ℹ️  No Streamlit secrets file")

if __name__ == "__main__":
    # Run verification
    key_valid = verify_mistral_key()
    check_environment()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if key_valid:
        print("✅ Your Mistral API key is working correctly!")
        print("✅ BETA mode should now generate responses")
        print("\nIf you still see errors, try:")
        print("1. Restart the Streamlit app")
        print("2. Clear browser cache")
        print("3. Check the console for other errors")
    else:
        print("❌ Your Mistral API key needs to be fixed")
        print("\nNext steps:")
        print("1. Get a valid API key from https://console.mistral.ai/")
        print("2. Set it: export MISTRAL_API_KEY=\"your-key\"")
        print("3. Restart the Streamlit app")