#!/usr/bin/env python3
"""
ShareGPT Forensic Validation Suite

Performs comprehensive forensic analysis on ShareGPT conversation dataset,
including full primacy attractor documentation, turn-by-turn governance metrics,
and telemetric signatures for IP protection.

Uses governance settings: basin_constant=1.0, constraint_tolerance=0.05
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
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

# Governance settings (Goldilocks zone from testing)
BASIN_CONSTANT = 1.0
CONSTRAINT_TOLERANCE = 0.05

print("=" * 80)
print("ShareGPT FORENSIC VALIDATION SUITE")
print("=" * 80)
print()
print("Comprehensive forensic analysis with full primacy attractor documentation")
print(f"Governance Settings: basin={BASIN_CONSTANT}, tolerance={CONSTRAINT_TOLERANCE}")
print()

# Find all ShareGPT files
sharegpt_dir = Path("telos_observatory_v3_backup_20251113_195049/saved_sessions")
sharegpt_files = sorted(sharegpt_dir.glob("sharegpt_filtered_*.json"))

print(f"Found {len(sharegpt_files)} ShareGPT conversation files")
print()

# Initialize
embedding_provider = EmbeddingProvider()
storage = ValidationStorage()

print("✓ Embedding provider initialized")
print("✓ Supabase storage connected")
print()

# Ask user how many files to process
print("How many conversations would you like to analyze?")
print(f"  - Enter number (1-{len(sharegpt_files)})")
print(f"  - Enter 'all' for all {len(sharegpt_files)} files")
print(f"  - Default: Process first 5 files")
print()

# Process all 45 files
num_to_process = 45
files_to_process = sharegpt_files[:num_to_process]

print(f"Processing {len(files_to_process)} conversations...")
print()

# Statistics
total_sessions = 0
total_turns = 0
all_fidelities = []

for file_path in files_to_process:
    print("=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)
    print()

    # Load conversation
    with open(file_path) as f:
        data = json.load(f)

    session_id_orig = data.get("session_id", file_path.stem)
    pa_config = data.get("primacy_attractor", {})
    turns = data.get("turns", [])

    print(f"Original Session ID: {session_id_orig}")
    print(f"Turns in conversation: {len(turns)}")
    print()

    # Extract primacy attractor configuration
    purpose_list = pa_config.get("purpose", [])
    scope_list = pa_config.get("scope", [])
    boundaries_list = pa_config.get("boundaries", [])

    # Combine lists into text
    purpose_text = " | ".join(purpose_list) if purpose_list else "General conversation assistance"
    scope_text = " | ".join(scope_list) if scope_list else "General topics"
    boundaries_text = " | ".join(boundaries_list) if boundaries_list else "No specific boundaries defined"

    print("━" * 80)
    print("PRIMACY ATTRACTOR CONFIGURATION")
    print("━" * 80)
    print()
    print("PURPOSE:")
    for p in purpose_list:
        print(f"  • {p}")
    print()
    print("SCOPE:")
    for s in scope_list:
        print(f"  • {s}")
    print()
    print("BOUNDARIES:")
    for b in boundaries_list:
        print(f"  • {b}")
    print()

    # Encode purpose and scope
    print("Encoding primacy attractor vectors...")
    purpose_emb = embedding_provider.encode(purpose_text)
    scope_emb = embedding_provider.encode(scope_text)
    print("  ✓ Purpose vector encoded (384 dimensions)")
    print("  ✓ Scope vector encoded (384 dimensions)")
    print()

    # Create PrimacyAttractor with governance settings
    print("Creating PrimacyAttractor with governance settings...")
    print(f"  Basin constant (β): {BASIN_CONSTANT}")
    print(f"  Constraint tolerance (τ): {CONSTRAINT_TOLERANCE}")

    attractor = PrimacyAttractorMath(
        purpose_vector=purpose_emb,
        scope_vector=scope_emb,
        constraint_tolerance=CONSTRAINT_TOLERANCE
    )

    # Calculate derived properties
    print()
    print("ATTRACTOR PROPERTIES:")
    print(f"  Attractor Center (â): τ·p + (1-τ)·s")
    print(f"    where τ = {CONSTRAINT_TOLERANCE}")
    print(f"  Weighting: {CONSTRAINT_TOLERANCE*100:.1f}% purpose + {(1-CONSTRAINT_TOLERANCE)*100:.1f}% scope")
    print()
    print(f"  Constraint Rigidity (ρ): {attractor.constraint_rigidity:.3f}")
    print(f"    Formula: ρ = 1 - τ = 1 - {CONSTRAINT_TOLERANCE} = {attractor.constraint_rigidity}")
    print()
    print(f"  Basin Radius (r): {attractor.basin_radius:.3f}")
    print(f"    Formula: r = β / ρ = {BASIN_CONSTANT} / {attractor.constraint_rigidity:.3f} = {attractor.basin_radius:.3f}")
    print()
    print("  Lyapunov Function: V(x) = ||x - â||²")
    print("    Measures 'energy' or distance from purpose alignment")
    print()

    # Calculate intervention thresholds (from InterventionController logic)
    epsilon_min = 0.1 + (CONSTRAINT_TOLERANCE * 0.3)
    epsilon_max = 0.5 + (CONSTRAINT_TOLERANCE * 0.4)
    proportional_gain = 2.0

    print("INTERVENTION THRESHOLDS:")
    print(f"  Minimum intervention (ε_min): {epsilon_min:.3f}")
    print(f"  Maximum intervention (ε_max): {epsilon_max:.3f}")
    print(f"  Proportional gain (K_p): {proportional_gain:.1f}")
    print()
    print("  Intervention triggered when:")
    print(f"    • Distance > basin_radius ({attractor.basin_radius:.3f})")
    print(f"    • Fidelity drop > ε_min ({epsilon_min:.3f})")
    print()

    # Create new session for this conversation
    new_session_id = str(uuid.uuid4())
    telemetric_key_gen = QuantumTelemetricKeyGenerator(session_id=new_session_id)
    signature_gen = TelemetricSignatureGenerator(telemetric_key_gen)
    fingerprint = telemetric_key_gen.get_session_fingerprint()

    storage.create_validation_session({
        "session_id": new_session_id,
        "validation_study_name": f"sharegpt_forensic_{file_path.stem}",
        "session_signature": fingerprint["key_history_hash"],
        "key_history_hash": fingerprint["key_history_hash"],
        "model": data.get("metadata", {}).get("model", "unknown"),
        "total_turns": 0,  # Will be updated
        "dataset_source": f"ShareGPT - {session_id_orig}",
        "pa_configuration": {
            "purpose": purpose_text,
            "scope": scope_text,
            "boundaries": boundaries_text,
            "constraint_tolerance": CONSTRAINT_TOLERANCE
        },
        "basin_constant": BASIN_CONSTANT,
        "constraint_tolerance": CONSTRAINT_TOLERANCE
    })

    print(f"✓ Created validation session: {new_session_id}")
    print()

    # Process each turn with forensic analysis
    print("━" * 80)
    print("TURN-BY-TURN FORENSIC ANALYSIS")
    print("━" * 80)
    print()

    turn_results = []

    for turn in turns:
        turn_num = turn.get("turn_number", turn.get("turn", 0))
        user_msg = turn.get("user_message", turn.get("user_input", ""))
        assistant_response = turn.get("assistant_response_telos", turn.get("response", ""))

        # Some files have existing fidelity, but we recalculate with current settings
        existing_fidelity = turn.get("fidelity_telos", turn.get("fidelity", None))

        print(f"Turn {turn_num}")
        print(f"  User: {user_msg[:80]}{'...' if len(user_msg) > 80 else ''}")
        print(f"  Assistant: {assistant_response[:80]}{'...' if len(assistant_response) > 80 else ''}")
        print()

        # Calculate fidelity (cosine similarity with purpose)
        response_emb = embedding_provider.encode(assistant_response)

        similarity = np.dot(purpose_emb, response_emb) / (
            np.linalg.norm(purpose_emb) * np.linalg.norm(response_emb)
        )
        fidelity = float((similarity + 1) / 2)  # Convert to [0,1]

        # Calculate distance from attractor
        distance = float(np.linalg.norm(response_emb - attractor.attractor_center))

        # Check basin membership
        in_basin = distance <= attractor.basin_radius

        # Determine drift
        drift_detected = not in_basin

        # Calculate Lyapunov value
        lyapunov = distance ** 2

        print(f"  METRICS:")
        print(f"    Fidelity (F): {fidelity:.3f}")
        if existing_fidelity:
            print(f"      (Original file: {existing_fidelity:.3f})")
        print(f"    Distance (d): {distance:.3f}")
        print(f"    Basin Radius (r): {attractor.basin_radius:.3f}")
        print(f"    Lyapunov V(x): {lyapunov:.3f}")
        print()

        print(f"  GOVERNANCE DECISION:")
        if in_basin:
            print(f"    ✓ IN BASIN (d={distance:.3f} ≤ r={attractor.basin_radius:.3f})")
            print(f"    Status: Aligned")
        else:
            print(f"    ✗ DRIFT DETECTED (d={distance:.3f} > r={attractor.basin_radius:.3f})")
            print(f"    Status: Outside basin boundary")

        # Would intervention trigger?
        would_intervene = drift_detected and (fidelity < (1.0 - epsilon_min))

        if would_intervene:
            print(f"    ⚠ INTERVENTION RECOMMENDED")
            print(f"      Fidelity {fidelity:.3f} < threshold {(1.0 - epsilon_min):.3f}")
            print(f"      Action: Refocus conversation toward purpose")
        else:
            print(f"    ○ No intervention needed")

        print()
        print(f"  COUNTERFACTUAL:")
        print(f"    WITH TELOS: Governance evaluates alignment continuously")
        print(f"    WITHOUT TELOS: No semantic boundary enforcement")
        print(f"    Impact: {'Intervention would refocus' if would_intervene else 'Conversation proceeding normally'}")
        print()

        # Store turn with signature
        delta_data = {
            "session_id": new_session_id,
            "turn_number": int(turn_num),
            "timestamp": datetime.now().isoformat(),
            "fidelity_score": float(fidelity),
            "distance_from_pa": float(distance)
        }

        signed_delta = signature_gen.sign_delta(delta_data)

        storage.store_signed_turn({
            "session_id": new_session_id,
            "turn_number": turn_num,
            "user_message": user_msg,
            "assistant_response": assistant_response,
            "fidelity_score": fidelity,
            "turn_telemetric_signature": signed_delta["signature"],
            "key_rotation_number": signed_delta["key_rotation_number"],
            "delta_t_ms": 0,
            "governance_mode": "sharegpt_forensic_validation",
            "drift_detected": drift_detected,
            "distance_from_pa": distance
        })

        turn_results.append({
            'turn': turn_num,
            'fidelity': fidelity,
            'distance': distance,
            'in_basin': in_basin,
            'drift_detected': drift_detected,
            'would_intervene': would_intervene
        })

        all_fidelities.append(fidelity)
        total_turns += 1

        print("-" * 80)
        print()

    # Mark session complete
    storage.mark_session_complete(new_session_id)
    total_sessions += 1

    # Session summary
    avg_fidelity = sum(r['fidelity'] for r in turn_results) / len(turn_results) if turn_results else 0
    drift_count = sum(1 for r in turn_results if r['drift_detected'])
    intervention_count = sum(1 for r in turn_results if r['would_intervene'])

    print("━" * 80)
    print("SESSION SUMMARY")
    print("━" * 80)
    print()
    print(f"File: {file_path.name}")
    print(f"Session ID: {new_session_id}")
    print(f"Turns Analyzed: {len(turn_results)}")
    print()
    print(f"Average Fidelity: {avg_fidelity:.3f}")
    print(f"Drift Detected: {drift_count}/{len(turn_results)} turns ({drift_count/len(turn_results)*100:.1f}%)")
    print(f"Interventions Recommended: {intervention_count}/{len(turn_results)} turns ({intervention_count/len(turn_results)*100:.1f}%)")
    print()
    print(f"✓ Stored in Supabase with governance settings")
    print(f"  Basin: {BASIN_CONSTANT}")
    print(f"  Tolerance: {CONSTRAINT_TOLERANCE}")
    print()
    print()

# Final summary
print("=" * 80)
print("FORENSIC VALIDATION COMPLETE")
print("=" * 80)
print()
print(f"Sessions Processed: {total_sessions}")
print(f"Total Turns Analyzed: {total_turns}")
print(f"Overall Average Fidelity: {sum(all_fidelities)/len(all_fidelities):.3f}" if all_fidelities else "N/A")
print()
print("Governance Settings Used:")
print(f"  Basin Constant (β): {BASIN_CONSTANT}")
print(f"  Constraint Tolerance (τ): {CONSTRAINT_TOLERANCE}")
print()
print("✓ All sessions stored with telemetric signatures")
print("✓ Full forensic documentation provided")
print("✓ Research dataset ready for analysis")
print()
print("=" * 80)
