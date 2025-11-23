"""
Direct test of TELOS response generation to diagnose response content issue.
"""
import os
os.environ['MISTRAL_API_KEY'] = 'iYsJab8PibuqxWgOFFLQ3WcMrTguE3X8'

from telos_purpose.core.unified_steward import UnifiedGovernanceSteward, PrimacyAttractor
from telos_purpose.core.embedding_provider import MistralEmbeddingProvider
from telos_purpose.llm_clients.mistral_client import MistralClient

print("=" * 80)
print("DIRECT TELOS TEST - Response Generation")
print("=" * 80)

# Create PA matching what BETA uses
attractor = PrimacyAttractor(
    purpose=["AI governance at runtime project called TELOS"],
    scope=["Discussion about AI alignment, governance, and TELOS framework"],
    boundaries=[
        "Maintain respectful dialogue",
        "Provide accurate information",
        "Stay within ethical guidelines"
    ],
    constraint_tolerance=0.02
)

print("\n1. Initializing TELOS engine...")
llm_client = MistralClient()
embedding_provider = MistralEmbeddingProvider()

steward = UnifiedGovernanceSteward(
    attractor=attractor,
    llm_client=llm_client,
    embedding_provider=embedding_provider,
    enable_interventions=True
)

print("2. Starting session...")
steward.start_session()
print("   ✓ Session started")

print("\n3. Generating response for off-topic PB&J request...")
user_input = "I would like to know the best methods for making a peanut butter and jelly sandwich."

result = steward.generate_governed_response(
    user_input=user_input,
    conversation_context=[]
)

print("\n" + "=" * 80)
print("RESULTS:")
print("=" * 80)
print(f"Fidelity: {result.get('telic_fidelity', 'N/A')}")
print(f"Intervention: {result.get('intervention_applied', False)}")
print(f"Intervention Type: {result.get('intervention_type', 'None')}")
print(f"Intervention Reason: {result.get('intervention_reason', 'None')}")
print(f"\nResponse keys: {list(result.keys())}")
print(f"\nResponse type: {type(result.get('response'))}")
print(f"Response length: {len(result.get('response', ''))}")
print(f"\nResponse content:")
print("-" * 80)
print(result.get('response', 'NO RESPONSE'))
print("-" * 80)
