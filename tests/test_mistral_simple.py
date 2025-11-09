#!/usr/bin/env python3
"""
Simplest possible Mistral API test - just one chat call
"""
from mistralai import Mistral
import os

# Load API key
api_key = "EAixOOQsXO3c7AeD3q1QIrU8miw4GjwI"
print(f"Using API key: {api_key[:10]}...")

client = Mistral(api_key=api_key)

print("\n🧪 Making ONE simple chat call to Mistral...")
print("=" * 80)

try:
    response = client.chat.complete(
        model='mistral-small-2501',
        messages=[
            {'role': 'user', 'content': 'Say exactly: API TEST'}
        ],
        max_tokens=10
    )

    print(f"✅ Response received: {response.choices[0].message.content}")
    print(f"\n📊 Metadata:")
    print(f"   Model: {response.model}")
    print(f"   Usage: {response.usage}")
    print(f"\n⚠️  CHECK MISTRAL DASHBOARD - this call should appear!")

except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
