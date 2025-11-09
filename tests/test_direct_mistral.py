#!/usr/bin/env python3
"""
Absolute minimal test - just make ONE chat completion call
No embeddings, no complexity - just verify chat API usage appears
"""
from mistralai import Mistral

api_key = "EAixOOQsXO3c7AeD3q1QIrU8miw4GjwI"
print(f"Testing with API key: {api_key[:10]}...")

client = Mistral(api_key=api_key)

print("\n🧪 Making ONE chat completion call...")
print("This should definitely show in your usage dashboard.\n")

try:
    response = client.chat.complete(
        model='mistral-small-2501',
        messages=[
            {
                'role': 'user',
                'content': 'Say exactly: MISTRAL API TEST SUCCESSFUL'
            }
        ],
        max_tokens=20
    )

    result = response.choices[0].message.content
    print(f"✅ SUCCESS!")
    print(f"Response: {result}")
    print(f"\n⚠️  CHECK YOUR MISTRAL DASHBOARD NOW")
    print(f"This call should appear in your usage within seconds/minutes")

except Exception as e:
    print(f"❌ FAILED: {e}")
