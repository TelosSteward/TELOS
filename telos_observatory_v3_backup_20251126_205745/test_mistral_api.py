"""
Test script to verify Mistral API is working and track usage.
This will make a real API call and show if it succeeds.
"""
import os
from mistralai import Mistral

def test_mistral_connection():
    """Test Mistral API connection and make a simple call."""

    # Get API key
    api_key = os.environ.get("MISTRAL_API_KEY")

    if not api_key:
        print("❌ MISTRAL_API_KEY not found in environment!")
        return False

    print(f"✓ Found API key: {api_key[:10]}...")

    try:
        # Initialize client
        client = Mistral(api_key=api_key)
        print("✓ Mistral client initialized")

        # Make a simple API call
        print("\n🔄 Making API call...")
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Hello, TELOS test!' in exactly those words."
                }
            ]
        )

        print("✓ API call successful!")
        print(f"\n📝 Response: {response.choices[0].message.content}")
        print(f"\n📊 Usage Stats:")
        print(f"  - Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  - Completion tokens: {response.usage.completion_tokens}")
        print(f"  - Total tokens: {response.usage.total_tokens}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("MISTRAL API CONNECTION TEST")
    print("="*60)

    success = test_mistral_connection()

    print("\n" + "="*60)
    if success:
        print("✅ TEST PASSED - API is working")
        print("\nIf you don't see usage in your Mistral dashboard:")
        print("1. Check you're looking at the correct API key")
        print("2. Usage may take time to appear")
        print("3. Free tier might not show detailed usage")
    else:
        print("❌ TEST FAILED - API is NOT working")
    print("="*60)
