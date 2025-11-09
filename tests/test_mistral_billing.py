#!/usr/bin/env python3
"""
Test to understand Mistral billing - make enough calls to see if it shows up
"""
from mistralai import Mistral
import time

api_key = "EAixOOQsXO3c7AeD3q1QIrU8miw4GjwI"
client = Mistral(api_key=api_key)

print("🧪 Making 5 chat calls to test billing visibility...")
print("=" * 80)

total_tokens = 0

for i in range(5):
    try:
        print(f"\n📞 Call {i+1}/5...")
        response = client.chat.complete(
            model='mistral-small-2501',
            messages=[
                {'role': 'user', 'content': f'Count to {i+1}'}
            ],
            max_tokens=50
        )

        tokens = response.usage.total_tokens
        total_tokens += tokens
        print(f"   ✅ Tokens: {tokens} (total so far: {total_tokens})")

        # Small delay to avoid rate limits
        time.sleep(1)

    except Exception as e:
        print(f"   ❌ FAILED: {e}")

print("\n" + "=" * 80)
print(f"📊 Total tokens used: {total_tokens}")
print(f"💰 Estimated cost (if $0.001 per 1K tokens): ${total_tokens / 1000 * 0.001:.6f}")
print(f"\n⚠️  CHECK MISTRAL DASHBOARD NOW")
print(f"   These {total_tokens} tokens SHOULD appear in your usage")
