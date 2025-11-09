#!/usr/bin/env python3
"""
Test Mistral rate limits - make slow sequential calls with delays
"""
import os
import time
from mistralai import Mistral

api_key = "EAixOOQsXO3c7AeD3q1QIrU8miw4GjwI"
print(f"API Key: {api_key[:10]}...")

client = Mistral(api_key=api_key)

print("\n🧪 Testing embedding calls with delays...")
print("Making 5 calls with 5-second delays between each\n")

for i in range(5):
    print(f"Call {i+1}/5...")
    try:
        response = client.embeddings.create(
            model='mistral-embed',
            inputs=[f"Test embedding call number {i+1}"]
        )
        print(f"  ✅ SUCCESS - Embedding dimension: {len(response.data[0].embedding)}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

    if i < 4:  # Don't wait after last call
        print(f"  ⏳ Waiting 5 seconds...\n")
        time.sleep(5)

print("\n✅ Test complete")
