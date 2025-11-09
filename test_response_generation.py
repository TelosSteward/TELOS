#!/usr/bin/env python3
"""
Quick test to verify TELOS response generation works end-to-end.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_CLEAN')

os.environ['MISTRAL_API_KEY'] = 'EAixOOQsXO3c7AeD3q1QIrU8miw4GjwI'

def test_mistral_client():
    """Test Mistral client works"""
    print("Testing Mistral client...")
    try:
        from telos.llm.mistral_client import MistralClient

        client = MistralClient(
            api_key=os.environ['MISTRAL_API_KEY'],
            model='mistral-small-latest'
        )

        response = client.generate_response(
            user_message="Hello, this is a test!",
            conversation_history=[]
        )

        print(f"✅ Mistral client works!")
        print(f"   Response: {response[:100]}...")
        return True

    except Exception as e:
        print(f"❌ Mistral client error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_provider():
    """Test embedding provider works"""
    print("\nTesting embedding provider...")
    try:
        from telos.core.embedding_provider import SentenceTransformerProvider

        provider = SentenceTransformerProvider()
        embedding = provider.encode("Test message")

        print(f"✅ Embedding provider works!")
        print(f"   Embedding shape: {embedding.shape}")
        return True

    except Exception as e:
        print(f"❌ Embedding provider error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_steward():
    """Test full TELOS unified steward"""
    print("\nTesting unified steward...")
    try:
        from telos.core.unified_steward import UnifiedGovernanceSteward
        from telos.core.embedding_provider import SentenceTransformerProvider
        from telos.llm.mistral_client import MistralClient

        embedding_provider = SentenceTransformerProvider()
        mistral_client = MistralClient(
            api_key=os.environ['MISTRAL_API_KEY'],
            model='mistral-small-latest'
        )

        steward = UnifiedGovernanceSteward(
            embedding_provider=embedding_provider,
            llm_client=mistral_client
        )

        print(f"✅ Unified steward initialized!")
        print(f"   Has LLM client: {hasattr(steward, 'llm_client')}")
        print(f"   Has embedding provider: {hasattr(steward, 'embedding_provider')}")
        return True

    except Exception as e:
        print(f"❌ Unified steward error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("="*60)
    print("TELOS Response Generation Test")
    print("="*60)

    results = []
    results.append(("Mistral Client", test_mistral_client()))
    results.append(("Embedding Provider", test_embedding_provider()))
    results.append(("Unified Steward", test_unified_steward()))

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests passed! Response generation is working!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Response generation NOT working.")
        sys.exit(1)
