"""
Check actual distances to attractor center for one session.
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from telos_purpose.core.primacy_math import PrimacyAttractorMath

# Load one session file
session_file = "telos_observatory_v3/saved_sessions/sharegpt_filtered_1.json"
with open(session_file, 'r') as f:
    session = json.load(f)

print(f"Session: {session['session_id']}")
print("=" * 80)

# Get PA config
user_pa = session['primacy_attractor']
print("\nUser PA Configuration:")
print(f"  Purpose: {user_pa['purpose'][0][:70]}...")
print(f"  Constraint Tolerance: {user_pa.get('constraint_tolerance', 0.2)}")

# Initialize embedding provider
embedding_provider = SentenceTransformer('all-MiniLM-L6-v2')

# Build attractor
user_purpose_text = ' '.join(user_pa['purpose'])
user_scope_text = ' '.join(user_pa['scope'])

user_p_vec = embedding_provider.encode(user_purpose_text)
user_s_vec = embedding_provider.encode(user_scope_text)

attractor = PrimacyAttractorMath(
    purpose_vector=user_p_vec,
    scope_vector=user_s_vec,
    privacy_level=user_pa.get('privacy_level', 0.8),
    constraint_tolerance=user_pa.get('constraint_tolerance', 0.2),
    task_priority=user_pa.get('task_priority', 0.7)
)

print(f"\nAttractor Properties:")
print(f"  Basin Radius: {attractor.basin_radius:.4f}")
print(f"  Constraint Rigidity: {attractor.constraint_rigidity:.4f}")

# Analyze responses
print(f"\nAnalyzing {len(session['turns'])} turns:")
print("-" * 80)

distances = []
for turn in session['turns']:
    response_text = turn['assistant_response_telos']
    response_embedding = embedding_provider.encode(response_text)
    
    distance = np.linalg.norm(response_embedding - attractor.attractor_center)
    distances.append(distance)
    in_basin = distance <= attractor.basin_radius
    
    print(f"Turn {turn['turn']:2d}: distance={distance:.4f}, "
          f"in_basin={in_basin}, ratio={distance/attractor.basin_radius:.2%}")

print("\n" + "=" * 80)
print("DISTANCE STATISTICS")
print("=" * 80)
print(f"Mean distance:   {np.mean(distances):.4f}")
print(f"Median distance: {np.median(distances):.4f}")
print(f"Min distance:    {np.min(distances):.4f}")
print(f"Max distance:    {np.max(distances):.4f}")
print(f"Std dev:         {np.std(distances):.4f}")
print(f"\nBasin radius:    {attractor.basin_radius:.4f}")
print(f"Max distance is {np.max(distances)/attractor.basin_radius:.1%} of basin radius")
print(f"\nAll in basin:    {all(d <= attractor.basin_radius for d in distances)}")

# Interpretation
print("\n" + "=" * 80)
print("CALIBRATION ASSESSMENT")
print("=" * 80)
if all(d <= attractor.basin_radius for d in distances):
    if np.max(distances) < attractor.basin_radius * 0.5:
        print("⚠️  All responses well within basin (< 50% of radius)")
        print("   Basin may be too permissive for meaningful governance")
    elif np.max(distances) < attractor.basin_radius * 0.8:
        print("✓  All responses within basin with moderate utilization")
        print("  Basin calibration appears reasonable")
    else:
        print("✓  Some responses near basin boundary")
        print("  Basin calibration is effective")
else:
    print("✓  Some responses outside basin")
    print("  Governance is actively differentiating")
