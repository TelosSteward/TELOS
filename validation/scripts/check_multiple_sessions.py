"""
Check distances across multiple sessions to validate calibration.
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from telos_purpose.core.primacy_math import PrimacyAttractorMath

# Initialize embedding provider
embedding_provider = SentenceTransformer('all-MiniLM-L6-v2')

# Analyze first 5 sessions
session_files = sorted(Path("telos_observatory_v3/saved_sessions").glob("sharegpt_filtered_*.json"))[:5]

all_distances = []
all_ratios = []

print("=" * 80)
print("MULTI-SESSION DISTANCE ANALYSIS")
print("=" * 80)

for session_file in session_files:
    with open(session_file, 'r') as f:
        session = json.load(f)
    
    user_pa = session['primacy_attractor']
    
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
    
    # Calculate distances for all turns
    session_distances = []
    for turn in session['turns']:
        response_text = turn['assistant_response_telos']
        response_embedding = embedding_provider.encode(response_text)
        distance = np.linalg.norm(response_embedding - attractor.attractor_center)
        session_distances.append(distance)
        all_distances.append(distance)
        all_ratios.append(distance / attractor.basin_radius)
    
    mean_dist = np.mean(session_distances)
    max_dist = np.max(session_distances)
    mean_ratio = mean_dist / attractor.basin_radius
    max_ratio = max_dist / attractor.basin_radius
    
    print(f"\n{session['session_id']} ({len(session['turns'])} turns)")
    print(f"  Basin radius: {attractor.basin_radius:.4f}")
    print(f"  Mean distance: {mean_dist:.4f} ({mean_ratio:.1%} of basin)")
    print(f"  Max distance:  {max_dist:.4f} ({max_ratio:.1%} of basin)")
    print(f"  All in basin:  {all(d <= attractor.basin_radius for d in session_distances)}")

print("\n" + "=" * 80)
print("AGGREGATE STATISTICS")
print("=" * 80)
print(f"Total turns analyzed: {len(all_distances)}")
print(f"\nDistance Statistics:")
print(f"  Mean:   {np.mean(all_distances):.4f}")
print(f"  Median: {np.median(all_distances):.4f}")
print(f"  Std:    {np.std(all_distances):.4f}")
print(f"  Min:    {np.min(all_distances):.4f}")
print(f"  Max:    {np.max(all_distances):.4f}")

print(f"\nBasin Utilization (distance/radius):")
print(f"  Mean ratio:   {np.mean(all_ratios):.1%}")
print(f"  Median ratio: {np.median(all_ratios):.1%}")
print(f"  Max ratio:    {np.max(all_ratios):.1%}")

print(f"\nAll responses in basin: {all(d <= 2.5 for d in all_distances)}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
if np.max(all_ratios) < 0.5:
    print("⚠️  All responses < 50% of basin radius")
    print("   Consider tightening constraint_tolerance to increase sensitivity")
elif np.max(all_ratios) < 0.8:
    print("✓  Good calibration: responses use 50-80% of basin")
    print("  Perfect 1.0 fidelity scores are legitimate")
    print("  System has headroom to detect drift")
else:
    print("✓  Excellent calibration: some responses near boundary")
    print("  System is actively governing at basin limits")
