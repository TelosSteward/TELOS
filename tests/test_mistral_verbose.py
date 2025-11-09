#!/usr/bin/env python3
"""
Test Mistral API with verbose logging to see actual HTTP requests
"""
import logging
import sys
from mistralai import Mistral

# Enable verbose logging to see actual API calls
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

api_key = "EAixOOQsXO3c7AeD3q1QIrU8miw4GjwI"
print(f"API Key: {api_key[:10]}...")

client = Mistral(api_key=api_key)

print("\n🧪 Making chat completion call with verbose logging...")
print("="*80)

try:
    response = client.chat.complete(
        model='mistral-small-2501',
        messages=[{'role': 'user', 'content': 'Say: TEST'}],
        max_tokens=10
    )
    print("="*80)
    print(f"\n✅ Response: {response.choices[0].message.content}")
    print(f"\n📊 Response metadata:")
    print(f"   Model: {response.model}")
    print(f"   Usage: {response.usage}")

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n🧪 Now testing embeddings...")
print("="*80)

try:
    emb_response = client.embeddings.create(
        model='mistral-embed',
        inputs=["Test embedding"]
    )
    print("="*80)
    print(f"\n✅ Embedding created")
    print(f"   Dimension: {len(emb_response.data[0].embedding)}")
    print(f"   First 5 values: {emb_response.data[0].embedding[:5]}")

except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
