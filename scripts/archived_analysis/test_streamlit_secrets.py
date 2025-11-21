#!/usr/bin/env python3
"""
Test if Streamlit can access the secrets file.
"""

import streamlit as st
import os

print("Testing Streamlit Secrets Access...")
print("=" * 60)

# Check environment variable
env_key = os.getenv("MISTRAL_API_KEY")
print(f"1. Environment variable: {env_key[:10] if env_key else 'NOT FOUND'}...")

# Check Streamlit secrets
try:
    secrets_key = st.secrets.get("MISTRAL_API_KEY")
    print(f"2. Streamlit secrets: {secrets_key[:10] if secrets_key else 'NOT FOUND'}...")
except Exception as e:
    print(f"2. Streamlit secrets: ERROR - {e}")

# Check secrets file exists
import os.path
secrets_path = os.path.expanduser("~/.streamlit/secrets.toml")
print(f"3. Secrets file exists: {os.path.exists(secrets_path)}")

if os.path.exists(secrets_path):
    with open(secrets_path) as f:
        content = f.read()
        print(f"4. Secrets file content length: {len(content)} chars")
        if "MISTRAL_API_KEY" in content:
            print("5. MISTRAL_API_KEY found in file: YES")
        else:
            print("5. MISTRAL_API_KEY found in file: NO")

print("=" * 60)

# Try to initialize the MistralClient
try:
    import sys
    sys.path.append('/Users/brunnerjf/Desktop/telos_privacy')
    from telos_purpose.llm_clients.mistral_client import MistralClient

    # Try with env var
    if env_key:
        print("\nTrying MistralClient with env var...")
        client = MistralClient(api_key=env_key, model="mistral-large-latest")
        print("✅ Client initialized with env var")

    # Try with secrets
    if secrets_key:
        print("\nTrying MistralClient with Streamlit secrets...")
        client = MistralClient(api_key=secrets_key, model="mistral-large-latest")
        print("✅ Client initialized with Streamlit secrets")

except Exception as e:
    print(f"\n❌ Error initializing client: {e}")