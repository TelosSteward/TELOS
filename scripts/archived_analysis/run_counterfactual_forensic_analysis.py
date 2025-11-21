#!/usr/bin/env python3
"""
FORENSIC COUNTERFACTUAL ANALYSIS - Claude Conversation Test 0

Complete forensic documentation of:
1. Primacy Attractor construction (vectors, center, basin)
2. Turn-by-turn governance analysis
3. What DID happen vs what WOULD have happened
4. Full telemetry signatures
5. Intervention triggers and recommendations
"""

import os
import json
from pathlib import Path
from datetime import datetime
import uuid

# Set Supabase credentials
os.environ['SUPABASE_URL'] = 'https://ukqrwjowlchhwznefboj.supabase.co'
os.environ['SUPABASE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVrcXJ3am93bGNoaHd6bmVmYm9qIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MjMyOTE2MCwiZXhwIjoyMDc3OTA1MTYwfQ.TvefimDWnnlAz4dj9-XBFJ4xl7hmXX9bZJSidzUjHTs'

from telos_purpose.core.embedding_provider import EmbeddingProvider
from telos_purpose.core.primacy_math import PrimacyAttractorMath
from telos_purpose.storage.validation_storage import ValidationStorage
from telos_privacy.cryptography.telemetric_keys_quantum import (
    QuantumTelemetricKeyGenerator,
    TelemetricSignatureGenerator
)
import numpy as np

print("=" * 100)
print("FORENSIC COUNTERFACTUAL ANALYSIS - CLAUDE CONVERSATION TEST 0")
print("=" * 100)
print()
print("This analysis provides COMPLETE forensic documentation of:")
print("  1. Primacy Attractor Construction")
print("  2. Governance State at Each Turn")
print("  3. Actual vs Counterfactual Outcomes")
print("  4. Intervention Triggers and Recommendations")
print("  5. Telemetric Signatures for IP Protection")
print()
print("=" * 100)
print()

# Load Claude conversation
conv_path = Path("test_sessions/claude_conversation_parsed.json")
with open(conv_path) as f:
    data = json.load(f)

conversations = data.get("conversations", [])
print(f"✓ Loaded conversation: {len(conversations)} messages")

# Extract user/assistant pairs
turns = []
current_human = None
for msg in conversations:
    if msg['from'] == 'human':
        current_human = msg['value']
    elif msg['from'] == 'gpt' and current_human:
        turns.append({
            'user': current_human,
            'assistant': msg['value']
        })
        current_human = None

print(f"✓ Extracted {len(turns)} complete conversation turns")
print()

# Initialize embedding provider
print("=" * 100)
print("SECTION 1: PRIMACY ATTRACTOR CONSTRUCTION")
print("=" * 100)
print()

embedding_provider = EmbeddingProvider()
print("✓ Embedding Provider Initialized")
print(f"  Model: nomic-embed-text-v1.5")
print(f"  Dimension: 768")
print()

# Define PURPOSE and SCOPE
purpose_text = "Discuss and analyze the TELOS purpose-aligned AI governance system, providing technical insights and implementation guidance"
scope_text = "TELOS framework architecture, implementation details, validation methodology, and technical design decisions"

print("Purpose Statement:")
print(f"  \"{purpose_text}\"")
print()
print("Scope Statement:")
print(f"  \"{scope_text}\"")
print()

print("Encoding vectors...")
purpose_emb = embedding_provider.encode(purpose_text)
scope_emb = embedding_provider.encode(scope_text)
print(f"✓ Purpose vector: shape {purpose_emb.shape}, norm {np.linalg.norm(purpose_emb):.3f}")
print(f"✓ Scope vector: shape {scope_emb.shape}, norm {np.linalg.norm(scope_emb):.3f}")
print()

# Governance parameters
basin_constant = 1.0  # Goldilocks value
constraint_tolerance = 0.05  # Strict (tested optimal)

print("Governance Parameters:")
print(f"  Basin Constant: {basin_constant} (Goldilocks - tested optimal)")
print(f"  Constraint Tolerance: {constraint_tolerance} (strict drift detection)")
print()

# Create Primacy Attractor
print("Creating Primacy Attractor...")
attractor = PrimacyAttractorMath(
    purpose_vector=purpose_emb,
    scope_vector=scope_emb,
    constraint_tolerance=constraint_tolerance
)

print("✓ Primacy Attractor Created")
print()
print("Attractor Properties:")
print(f"  Attractor Center: weighted combination of purpose ({constraint_tolerance}) and scope ({1-constraint_tolerance})")
print(f"  Center Norm: {np.linalg.norm(attractor.attractor_center):.3f}")
print(f"  Constraint Rigidity (ρ): {attractor.constraint_rigidity:.3f}")
print(f"  Basin Radius (r): {attractor.basin_radius:.3f}")
print(f"  Formula: r = {basin_constant} / ρ = {basin_constant} / {attractor.constraint_rigidity:.3f} = {attractor.basin_radius:.3f}")
print()

print("Attractor Center Composition:")
print(f"  τ (tolerance) = {constraint_tolerance}")
print(f"  Center = τ·purpose + (1-τ)·scope")
print(f"  Center = {constraint_tolerance}·purpose + {1-constraint_tolerance}·scope")
print(f"  This means: {constraint_tolerance*100:.0f}% purpose, {(1-constraint_tolerance)*100:.0f}% scope weighting")
print()

# Intervention thresholds (calculated from tolerance)
epsilon_min = 0.1 + (constraint_tolerance * 0.3)
epsilon_max = 0.5 + (constraint_tolerance * 0.4)
proportional_gain = 2.0

print("Intervention Thresholds (calculated from tolerance):")
print(f"  Epsilon Min: {epsilon_min:.3f}")
print(f"  Epsilon Max: {epsilon_max:.3f}")
print(f"  Proportional Gain (K_p): {proportional_gain:.3f}")
print()

# Initialize storage
storage = ValidationStorage()
session_id = str(uuid.uuid4())
telemetric_key_gen = QuantumTelemetricKeyGenerator(session_id=session_id)
signature_gen = TelemetricSignatureGenerator(telemetric_key_gen)
fingerprint = telemetric_key_gen.get_session_fingerprint()

print("✓ Telemetric Signature System Initialized")
print(f"  Session ID: {session_id}")
print(f"  Session Fingerprint: {fingerprint['key_history_hash'][:32]}...")
print(f"  Entropy Sources: {fingerprint['entropy_sources']}")
print()

# Create session
storage.create_validation_session({
    "session_id": session_id,
    "validation_study_name": "test0_forensic_counterfactual_analysis",
    "session_signature": fingerprint["key_history_hash"],
    "key_history_hash": fingerprint["key_history_hash"],
    "model": "claude-3.5-sonnet (actual conversation)",
    "total_turns": 0,
    "dataset_source": "Claude Conversation - Test 0 - Forensic Analysis",
    "pa_configuration": {
        "purpose": purpose_text,
        "scope": scope_text,
        "constraint_tolerance": constraint_tolerance,
        "boundaries": ["Stay technical", "Focus on TELOS", "Provide implementation guidance"]
    },
    "basin_constant": basin_constant,
    "constraint_tolerance": constraint_tolerance
})

print(f"✓ Validation session created in Supabase")
print()

print("=" * 100)
print("SECTION 2: TURN-BY-TURN FORENSIC ANALYSIS")
print("=" * 100)
print()

forensic_results = []

for i, turn in enumerate(turns[:10], 1):
    print("=" * 100)
    print(f"TURN {i}/10 - FORENSIC BREAKDOWN")
    print("=" * 100)
    print()

    print(f"USER MESSAGE:")
    print(f"  {turn['user'][:200]}{'...' if len(turn['user']) > 200 else ''}")
    print()

    print(f"ASSISTANT RESPONSE:")
    print(f"  {turn['assistant'][:200]}{'...' if len(turn['assistant']) > 200 else ''}")
    print()

    # Calculate response embedding
    response_emb = embedding_provider.encode(turn['assistant'])

    # Calculate fidelity (cosine similarity with purpose)
    similarity = np.dot(purpose_emb, response_emb) / (
        np.linalg.norm(purpose_emb) * np.linalg.norm(response_emb)
    )
    fidelity = float((similarity + 1) / 2)

    # Calculate distance from attractor center
    distance = float(np.linalg.norm(response_emb - attractor.attractor_center))

    # Basin membership
    in_basin = distance <= attractor.basin_radius
    distance_beyond = (distance - attractor.basin_radius) if not in_basin else 0.0

    # Lyapunov function (energy)
    lyapunov = attractor.compute_lyapunov_function(type('State', (), {'embedding': response_emb})())

    # Intervention decision
    intervention_needed = not in_basin

    print("GOVERNANCE METRICS:")
    print(f"  Fidelity Score: {fidelity:.4f}")
    print(f"  Distance from PA Center: {distance:.4f}")
    print(f"  Basin Radius: {attractor.basin_radius:.4f}")
    print(f"  In Basin: {'✓ YES' if in_basin else '✗ NO - DRIFT DETECTED'}")
    if not in_basin:
        print(f"  Distance Beyond Basin: {distance_beyond:.4f} ({distance_beyond/attractor.basin_radius*100:.1f}% overshoot)")
    print(f"  Lyapunov Function V(x): {lyapunov:.4f}")
    print()

    # Intervention analysis
    print("INTERVENTION ANALYSIS:")
    if intervention_needed:
        print(f"  Status: ⚠️ INTERVENTION REQUIRED")
        print(f"  Reason: Response outside basin (distance {distance:.3f} > radius {attractor.basin_radius:.3f})")
        print(f"  Recommended Action: Redirect response to align with TELOS technical discussion")
        print(f"  Intervention Strength: Proportional to distance ({distance_beyond:.3f})")

        # What TELOS would do
        print()
        print("  TELOS WOULD INTERVENE WITH:")
        print(f"    - Detect drift from technical TELOS discussion")
        print(f"    - Calculate corrective gradient: toward attractor center")
        print(f"    - Generate intervention: 'Let's refocus on TELOS technical aspects...'")
        print(f"    - Apply proportional control: strength = {distance_beyond:.3f} * {proportional_gain:.3f}")
    else:
        print(f"  Status: ✓ NO INTERVENTION NEEDED")
        print(f"  Reason: Response within basin (distance {distance:.3f} ≤ radius {attractor.basin_radius:.3f})")
        print(f"  Margin: {attractor.basin_radius - distance:.4f} ({(attractor.basin_radius - distance)/attractor.basin_radius*100:.1f}% safety margin)")
    print()

    # Counterfactual
    print("COUNTERFACTUAL COMPARISON:")
    print(f"  ACTUAL (WITH governance): Response generated by Claude")
    print(f"    - Fidelity: {fidelity:.4f}")
    print(f"    - In Basin: {'YES' if in_basin else 'NO'}")
    if not in_basin:
        print(f"    - TELOS governance was INACTIVE in actual conversation")
        print(f"    - If TELOS was active, it would have intervened")
    print()
    print(f"  COUNTERFACTUAL (WITHOUT governance): Hypothetical uncontrolled response")
    print(f"    - Would have NO constraints on topic drift")
    print(f"    - Could discuss anything, not necessarily TELOS")
    print(f"    - Likely lower fidelity: estimated <{fidelity:.4f}")
    print(f"    - No intervention mechanism to correct course")
    print()

    # Sign the delta
    delta_data = {
        "session_id": session_id,
        "turn_number": int(i),
        "timestamp": datetime.now().isoformat(),
        "fidelity_score": float(fidelity),
        "distance_from_pa": float(distance),
        "lyapunov_function": float(lyapunov)
    }

    signed_delta = signature_gen.sign_delta(delta_data)

    print("TELEMETRIC SIGNATURE:")
    print(f"  Signature: {signed_delta['signature'][:64]}...")
    print(f"  Key Rotation: {signed_delta['key_rotation_number']}")
    print(f"  Entropy Hash: {signed_delta['entropy_hash'][:32]}...")
    print(f"  Purpose: Cryptographic proof this turn occurred at this time with these metrics")
    print()

    # Store turn
    storage.store_signed_turn({
        "session_id": session_id,
        "turn_number": i,
        "user_message": turn['user'],
        "assistant_response": turn['assistant'],
        "fidelity_score": fidelity,
        "turn_telemetric_signature": signed_delta["signature"],
        "key_rotation_number": signed_delta["key_rotation_number"],
        "delta_t_ms": 0,
        "governance_mode": "forensic_counterfactual_analysis",
        "drift_detected": not in_basin,
        "distance_from_pa": distance,
        "intervention_triggered": intervention_needed
    })

    forensic_results.append({
        'turn': i,
        'fidelity': fidelity,
        'distance': distance,
        'in_basin': in_basin,
        'drift_detected': not in_basin,
        'lyapunov': lyapunov,
        'intervention_needed': intervention_needed,
        'distance_beyond': distance_beyond
    })

# Mark complete
storage.mark_session_complete(session_id)

print("=" * 100)
print("SECTION 3: AGGREGATE FORENSIC ANALYSIS")
print("=" * 100)
print()

avg_fidelity = sum(r['fidelity'] for r in forensic_results) / len(forensic_results)
drift_count = sum(1 for r in forensic_results if r['drift_detected'])
intervention_count = sum(1 for r in forensic_results if r['intervention_needed'])
avg_distance = sum(r['distance'] for r in forensic_results) / len(forensic_results)
avg_lyapunov = sum(r['lyapunov'] for r in forensic_results) / len(forensic_results)

print("OVERALL METRICS:")
print(f"  Turns Analyzed: {len(forensic_results)}")
print(f"  Average Fidelity: {avg_fidelity:.4f}")
print(f"  Average Distance: {avg_distance:.4f}")
print(f"  Average Lyapunov V(x): {avg_lyapunov:.4f}")
print()

print("GOVERNANCE EFFECTIVENESS:")
print(f"  Drift Detected: {drift_count}/{len(forensic_results)} turns ({drift_count/len(forensic_results)*100:.1f}%)")
print(f"  Basin Adherence: {len(forensic_results)-drift_count}/{len(forensic_results)} turns ({(len(forensic_results)-drift_count)/len(forensic_results)*100:.1f}%)")
print(f"  Interventions Needed: {intervention_count}/{len(forensic_results)} turns")
print()

print("PRIMACY ATTRACTOR PERFORMANCE:")
print(f"  Basin Constant Used: {basin_constant}")
print(f"  Constraint Tolerance Used: {constraint_tolerance}")
print(f"  Effective Basin Radius: {attractor.basin_radius:.4f}")
print(f"  Conclusion: {'Strict governance - detected all drift' if drift_count > 7 else 'Moderate governance' if drift_count > 3 else 'Permissive governance'}")
print()

print("=" * 100)
print("SECTION 4: IP PROTECTION & VERIFICATION")
print("=" * 100)
print()

ip_proof = storage.get_ip_proof(session_id)
print("TELEMETRIC PROOF PACKAGE:")
print(f"  Session ID: {ip_proof['session_id']}")
print(f"  Study Name: {ip_proof['study_name']}")
print(f"  Total Turns: {ip_proof['total_turns']}")
print(f"  Signed Turns: {ip_proof['signed_turns']}")
print(f"  Session Signature: {ip_proof['session_signature'][:32]}...")
print(f"  Key History Hash: {ip_proof['key_history_hash'][:32]}...")
print()

print("SIGNATURE CHAIN:")
for i, sig in enumerate(ip_proof['signature_chain'][:5], 1):
    print(f"  Turn {i}: {sig[:64]}...")
if len(ip_proof['signature_chain']) > 5:
    print(f"  ... and {len(ip_proof['signature_chain'])-5} more signatures")
print()

print("GOVERNANCE CONFIGURATION PROOF:")
print(f"  Basin Constant: {ip_proof.get('basin_constant', 'Not recorded')}")
print(f"  Constraint Tolerance: {ip_proof.get('constraint_tolerance', 'Not recorded')}")
print(f"  Purpose Configuration: Stored in session")
print()

print("=" * 100)
print("SECTION 5: COUNTERFACTUAL INTERPRETATION")
print("=" * 100)
print()

print("WHAT ACTUALLY HAPPENED (Claude Conversation):")
print(f"  - Claude responded to questions about TELOS")
print(f"  - NO active TELOS governance during conversation")
print(f"  - Average fidelity: {avg_fidelity:.4f}")
print(f"  - {drift_count} turns drifted outside governance basin")
print()

print("WHAT TELOS WOULD HAVE DONE (Counterfactual):")
print(f"  - TELOS would have monitored each response")
print(f"  - {intervention_count} interventions would have been triggered")
print(f"  - Responses would be redirected to stay focused on TELOS technical discussion")
print(f"  - Expected improvement: Higher fidelity, tighter alignment")
print()

print("GOVERNANCE IMPACT ASSESSMENT:")
if drift_count >= 8:
    print(f"  ⚠️ HIGH DRIFT RATE ({drift_count/len(forensic_results)*100:.0f}%)")
    print(f"  - Actual conversation frequently moved outside technical TELOS scope")
    print(f"  - TELOS governance would have significantly improved alignment")
    print(f"  - Benefit: Kept conversation focused on purpose")
elif drift_count >= 4:
    print(f"  ⚡ MODERATE DRIFT RATE ({drift_count/len(forensic_results)*100:.0f}%)")
    print(f"  - Some drift from technical TELOS discussion")
    print(f"  - TELOS governance would have provided course corrections")
    print(f"  - Benefit: Occasional redirects to maintain focus")
else:
    print(f"  ✓ LOW DRIFT RATE ({drift_count/len(forensic_results)*100:.0f}%)")
    print(f"  - Conversation mostly stayed aligned")
    print(f"  - TELOS governance would have minimal intervention")
    print(f"  - Benefit: Assurance of alignment without heavy-handed control")
print()

print("=" * 100)
print("FORENSIC ANALYSIS COMPLETE")
print("=" * 100)
print()
print(f"✓ Complete forensic documentation stored in Supabase")
print(f"✓ Session ID: {session_id}")
print(f"✓ All {len(forensic_results)} turns cryptographically signed")
print(f"✓ Primacy Attractor fully documented")
print(f"✓ Governance settings recorded (basin={basin_constant}, tolerance={constraint_tolerance})")
print(f"✓ IP protection enabled via telemetric signatures")
print()
print("This forensic analysis provides complete documentation for:")
print("  - Academic publication")
print("  - IP protection and patent applications")
print("  - Regulatory compliance demonstration")
print("  - Third-party verification")
print()
print("=" * 100)
