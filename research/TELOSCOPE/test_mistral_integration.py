#!/usr/bin/env python3
"""
Test script to verify Mistral API integration in TELOSCOPE_BETA.
Tests the Steward LLM service directly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from services.steward_llm import StewardLLM

def test_mistral_api():
    """Test Mistral API integration."""
    print("=" * 60)
    print("TELOSCOPE_BETA - Mistral API Integration Test")
    print("=" * 60)

    # Initialize Steward LLM service
    print("\n1. Initializing Steward LLM service...")
    steward = StewardLLM()

    # Check if API key is loaded
    print("\n2. Checking API key configuration...")
    if hasattr(st, 'secrets') and 'MISTRAL_API_KEY' in st.secrets:
        print("✅ API key found in secrets")
    else:
        # Try to load from .streamlit/secrets.toml directly
        import toml
        secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
        if os.path.exists(secrets_path):
            secrets = toml.load(secrets_path)
            if 'MISTRAL_API_KEY' in secrets:
                print("✅ API key found in secrets.toml")
                os.environ['MISTRAL_API_KEY'] = secrets['MISTRAL_API_KEY']
            else:
                print("❌ API key not found in secrets.toml")
                return False
        else:
            print("❌ secrets.toml file not found")
            return False

    # Test API connection with a simple query
    print("\n3. Testing API connection...")
    test_prompt = "Hello, this is a test. Please respond with 'API connection successful'."

    try:
        response = steward.get_response(
            user_message=test_prompt,
            conversation_history=[]
        )

        if response:
            print(f"✅ API Response received: {response[:100]}...")
            return True
        else:
            print("❌ Empty response from API")
            return False

    except Exception as e:
        print(f"❌ Error calling Mistral API: {e}")
        return False

    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Mock streamlit secrets for testing
    if not hasattr(st, 'secrets'):
        st.secrets = {}

    success = test_mistral_api()

    if success:
        print("\n✅ MISTRAL API INTEGRATION TEST PASSED")
    else:
        print("\n❌ MISTRAL API INTEGRATION TEST FAILED")

    sys.exit(0 if success else 1)