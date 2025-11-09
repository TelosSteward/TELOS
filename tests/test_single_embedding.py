#!/usr/bin/env python3
"""
Test single Mistral embedding API call to verify it works
"""
import os
from mistralai import Mistral

# Load API key
api_key = os.getenv('MISTRAL_API_KEY')
if not api_key:
    # Try loading from streamlit secrets
    import streamlit as st
    api_key = st.secrets.get('MISTRAL_API_KEY')

print(f"API Key: {api_key[:10]}...")

# Create client
client = Mistral(api_key=api_key)

# Make ONE embedding call
print("\n🧪 Testing single embedding call...")
try:
    response = client.embeddings.create(
        model='mistral-embed',
        inputs=["This is a test"]
    )
    print(f"✅ SUCCESS!")
    print(f"   Embedding dimension: {len(response.data[0].embedding)}")
    print(f"   First 5 values: {response.data[0].embedding[:5]}")
except Exception as e:
    print(f"❌ FAILED: {e}")
