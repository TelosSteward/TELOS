"""
Analyze basin calibration by checking actual distances from attractor centers.
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from telos_purpose.core.primacy_math import PrimacyAttractorMath

# Load results
with open('dual_pa_counterfactual_results.json', 'r') as f:
    data = json.load(f)

# Initialize embedding provider
embedding_provider = SentenceTransformer('all-MiniLM-L6-v2')

# Analyze first session in detail
session = data['session_results'][0]
print(f"Analyzing session: {session['session_id']}")
print(f"Governance mode: {session['governance_mode_actual']}")
print(f"Dual PA used: {session['dual_pa_used']}")
print()

# Get user PA configuration
user_pa = session['user_pa']
user_purpose_text = ' '.join(user_pa['purpose'])
user_scope_text = ' '.join(user_pa['scope'])

# Build user PA attractor
user_p_vec = embedding_provider.encode(user_purpose_text)
user_s_vec = embedding_provider.encode(user_scope_text)
user_attractor = PrimacyAttractorMath(
    purpose_vector=user_p_vec,
    scope_vector=user_s_vec,
    privacy_level=0.8,
    constraint_tolerance=0.2,
    task_priority=0.7
)

print(f"User PA Basin Radius: {user_attractor.basin_radius:.4f}")
print(f"Constraint Rigidity: {user_attractor.constraint_rigidity:.4f}")
print()

# Get AI PA configuration if dual mode
if session['dual_pa_used'] and session['ai_pa']:
    ai_pa = session['ai_pa']
    ai_purpose_text = ' '.join(ai_pa['purpose'])
    ai_scope_text = ' '.join(ai_pa['scope'])
    
    ai_p_vec = embedding_provider.encode(ai_purpose_text)
    ai_s_vec = embedding_provider.encode(ai_scope_text)
    ai_attractor = PrimacyAttractorMath(
        purpose_vector=ai_p_vec,
        scope_vector=ai_s_vec,
        privacy_level=0.8,
        constraint_tolerance=0.2,
        task_priority=0.7
    )
    
    print(f"AI PA Basin Radius: {ai_attractor.basin_radius:.4f}")
    print(f"Constraint Rigidity: {ai_attractor.constraint_rigidity:.4f}")
    print()

# Load actual session responses and calculate distances
session_file = Path(f"telos_observatory_v3/saved_sessions/{session['session_id']}.json")
with open(session_file, 'r') as f:
    session_data = json.load(f)

print("Turn-by-Turn Analysis:")
print("-" * 80)

for i, turn in enumerate(session_data['turns'][:5]):  # First 5 turns
    response_text = turn['assistant_response_telos']
    response_embedding = embedding_provider.encode(response_text)
    
    # Calculate distance to user PA
    user_distance = np.linalg.norm(response_embedding - user_attractor.attractor_center)
    user_in_basin = user_distance <= user_attractor.basin_radius
    
    print(f"\nTurn {turn['turn']}:")
    print(f"  User PA Distance: {user_distance:.4f}")
    print(f"  In Basin: {user_in_basin}")
    print(f"  Distance/Radius Ratio: {user_distance / user_attractor.basin_radius:.2%}")
    
    if session['dual_pa_used']:
        ai_distance = np.linalg.norm(response_embedding - ai_attractor.attractor_center)
        ai_in_basin = ai_distance <= ai_attractor.basin_radius
        print(f"  AI PA Distance: {ai_distance:.4f}")
        print(f"  In Basin: {ai_in_basin}")
        print(f"  Distance/Radius Ratio: {ai_distance / ai_attractor.basin_radius:.2%}")

print("\n" + "=" * 80)
print("CALIBRATION ASSESSMENT")
print("=" * 80)
print(f"Basin Radius: {user_attractor.basin_radius:.4f}")
print(f"If all distances < {user_attractor.basin_radius:.4f}, basin may be too permissive")
print(f"Recommended basin range for governance: 0.5 - 1.5")
print(f"Current setting allows distances up to {user_attractor.basin_radius:.1f}x attractor center")
