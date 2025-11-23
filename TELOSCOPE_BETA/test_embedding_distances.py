"""
Test embedding distances between PA topics and off-topic requests.
This will show us if all-MiniLM-L6-v2 is discriminative enough.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model
print("Loading all-MiniLM-L6-v2...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device='cpu')

# PA embeddings (what we established)
pa_purpose = "AI governance at runtime project called TELOS"
pa_scope = "TELOS architecture, purpose alignment, drift detection"

# Test queries
on_topic = "How does TELOS detect purpose drift?"
off_topic = "I would like to know the best methods for making a peanut butter and jelly sandwich"

print("\n" + "="*80)
print("EMBEDDING DISTANCE TEST")
print("="*80)

# Encode
print("\nEncoding texts...")
pa_purpose_emb = model.encode(pa_purpose, convert_to_numpy=True)
pa_scope_emb = model.encode(pa_scope, convert_to_numpy=True)
on_topic_emb = model.encode(on_topic, convert_to_numpy=True)
off_topic_emb = model.encode(off_topic, convert_to_numpy=True)

# Calculate PA center (weighted average with tolerance=0.02)
tolerance = 0.02
rigidity = 1.0 - tolerance
center_unnorm = tolerance * pa_purpose_emb + rigidity * pa_scope_emb
center_norm = np.linalg.norm(center_unnorm)
pa_center = center_unnorm / center_norm if center_norm > 0 else center_unnorm

# Calculate distances (L2 norm)
dist_on_topic = np.linalg.norm(on_topic_emb - pa_center)
dist_off_topic = np.linalg.norm(off_topic_emb - pa_center)

# Basin radius calculation
rigidity_floored = max(rigidity, 0.25)
basin_radius = 1.0 / rigidity_floored

# Fidelity calculation
fidelity_on_topic = 1.0 - (dist_on_topic / basin_radius)
fidelity_off_topic = 1.0 - (dist_off_topic / basin_radius)

print(f"\nPA Purpose: {pa_purpose}")
print(f"PA Scope: {pa_scope}")
print(f"\nTolerance: {tolerance}")
print(f"Rigidity: {rigidity}")
print(f"Basin Radius: {basin_radius:.3f}")

print("\n" + "-"*80)
print("ON-TOPIC QUERY:")
print(f"  Query: {on_topic}")
print(f"  Distance from PA center: {dist_on_topic:.4f}")
print(f"  Fidelity: {fidelity_on_topic:.4f}")
print(f"  In basin: {dist_on_topic <= basin_radius}")

print("\n" + "-"*80)
print("OFF-TOPIC QUERY:")
print(f"  Query: {off_topic}")
print(f"  Distance from PA center: {dist_off_topic:.4f}")
print(f"  Fidelity: {fidelity_off_topic:.4f}")
print(f"  In basin: {dist_off_topic <= basin_radius}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

if fidelity_off_topic > 0.3:
    print(f"❌ PROBLEM: Off-topic fidelity ({fidelity_off_topic:.3f}) is too HIGH!")
    print(f"   Expected: < 0.3, Got: {fidelity_off_topic:.3f}")
    print(f"\n   Root cause: all-MiniLM-L6-v2 embeddings don't separate topics well.")
    print(f"   Distance between on-topic and off-topic: {abs(dist_on_topic - dist_off_topic):.4f}")
    print(f"   (Should be much larger for good discrimination)")

    print(f"\n   SOLUTIONS:")
    print(f"   1. Use better embedding model (e.g., all-mpnet-base-v2, 768 dims)")
    print(f"   2. Add distance amplification/scaling")
    print(f"   3. Use topic classification as pre-filter")
else:
    print(f"✅ GOOD: Off-topic fidelity ({fidelity_off_topic:.3f}) is appropriately LOW")

print("="*80)
