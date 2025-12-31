#!/usr/bin/env python3
"""
Regenerate PA Template Embeddings with Centroid Computation
============================================================

This script regenerates pa_template_embeddings.json with enriched centroid
embeddings that include example_queries. The centroid is computed as the
mean of normalized embeddings from:
  1. Purpose + Scope text
  2. Each example_query (typically 10 per template)

This creates a PA embedding that covers the semantic space of aligned queries,
resulting in higher fidelity scores for on-topic Turn 1 queries.

Usage:
    cd /Users/brunnerjf/Desktop/TELOS_Master
    python3 scripts/regenerate_pa_embeddings.py
"""

import sys
import os
import json
import numpy as np

# Add both paths for proper imports
sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master')
sys.path.insert(0, '/Users/brunnerjf/Desktop/TELOS_Master/telos_observatory_v3')

from sentence_transformers import SentenceTransformer

# Import templates
from config.pa_templates import PA_TEMPLATES

# Define INTENT_TO_ROLE_MAP locally to avoid complex import chain
INTENT_TO_ROLE_MAP = {
    'learn': 'teach', 'understand': 'explain', 'study': 'teach', 'grasp': 'explain',
    'master': 'guide mastery of', 'solve': 'help solve', 'fix': 'help fix',
    'debug': 'help debug', 'troubleshoot': 'help troubleshoot', 'diagnose': 'help diagnose',
    'create': 'help create', 'write': 'help write', 'build': 'help build',
    'develop': 'help develop', 'design': 'help design', 'produce': 'help produce',
    'craft': 'help craft', 'compose': 'help compose', 'generate': 'help generate',
    'brainstorm': 'help brainstorm', 'analyze': 'help analyze', 'review': 'help review',
    'evaluate': 'help evaluate', 'assess': 'help assess', 'examine': 'help examine',
    'investigate': 'help investigate', 'research': 'help research', 'synthesize': 'help synthesize',
    'plan': 'help plan', 'organize': 'help organize', 'strategize': 'help strategize',
    'structure': 'help structure', 'decide': 'help decide', 'explore': 'guide exploration of',
    'discover': 'help discover', 'optimize': 'help optimize', 'improve': 'help improve',
    'enhance': 'help enhance', 'refactor': 'help refactor', 'document': 'help document',
    'test': 'help test', 'implement': 'help implement',
}

# Output path
OUTPUT_PATH = '/Users/brunnerjf/Desktop/TELOS_Master/telos_observatory_v3/config/pa_template_embeddings.json'


def detect_user_intent(purpose: str) -> str:
    """
    Detect primary user intent from purpose statement.
    Purpose statements should START with intent verbs for proper detection.
    """
    purpose_lower = purpose.lower()
    first_part = purpose_lower[:80]

    # Priority 1: Check if purpose starts with intent verb
    for intent in INTENT_TO_ROLE_MAP.keys():
        if first_part.startswith(intent):
            return intent
        if f"{intent} and" in first_part[:30]:
            return intent

    # Priority 2: Check first part
    for intent in INTENT_TO_ROLE_MAP.keys():
        if intent in first_part:
            return intent

    # Priority 3: Check full purpose
    for intent in INTENT_TO_ROLE_MAP.keys():
        if intent in purpose_lower:
            return intent

    return 'understand'


def compute_centroid_embedding(
    user_pa_text: str,
    example_queries: list,
    model: SentenceTransformer
) -> np.ndarray:
    """
    Compute centroid embedding from purpose/scope + example queries.

    This creates a PA that covers the semantic space of aligned queries,
    resulting in higher fidelity scores for on-topic queries.
    """
    if example_queries:
        all_texts = [user_pa_text] + list(example_queries)
    else:
        all_texts = [user_pa_text]

    # Embed all texts
    all_embeddings = []
    for text in all_texts:
        emb = np.array(model.encode(text))
        # Normalize each embedding before averaging
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        all_embeddings.append(emb)

    # Compute centroid as mean of all embeddings
    centroid = np.mean(all_embeddings, axis=0)
    # Re-normalize the centroid
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

    print(f"  Centroid computed from {len(all_texts)} texts")
    return centroid


def derive_ai_pa_text(purpose: str, scope: list, intent: str) -> str:
    """
    Derive AI PA text from user purpose using intent-to-role mapping.
    """
    role_action = INTENT_TO_ROLE_MAP.get(intent, 'help with')
    scope_text = ', '.join(scope)

    if role_action == 'teach':
        return f"AI Role: Teach the user so they can: {purpose}. Supporting scope: {scope_text}."
    elif role_action == 'explain':
        return f"AI Role: Explain concepts clearly so the user can: {purpose}. Supporting scope: {scope_text}."
    elif role_action.startswith('help '):
        action_verb = role_action.replace('help ', '')
        return f"AI Role: Help the user {action_verb}: {purpose}. Supporting scope: {scope_text}."
    elif role_action.startswith('guide'):
        return f"AI Role: {role_action.capitalize()} the user as they work to: {purpose}. Supporting scope: {scope_text}."
    else:
        return f"AI Role: {role_action.capitalize()} to support: {purpose}. Supporting scope: {scope_text}."


def main():
    print("=" * 60)
    print("PA Template Embedding Regeneration with Centroid")
    print("=" * 60)
    print()

    # Load model
    print("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print()

    # Process each template
    all_embeddings = {}

    for template in PA_TEMPLATES:
        template_id = template['id']
        purpose = template['purpose']
        scope = template['scope']
        example_queries = template.get('example_queries', [])

        print(f"Processing template: {template_id}")
        print(f"  Purpose: {purpose[:60]}...")
        print(f"  Scope items: {len(scope)}")
        print(f"  Example queries: {len(example_queries)}")

        # Build user PA text
        scope_text = ', '.join(scope)
        user_pa_text = f"Purpose: {purpose} Scope: {scope_text}"

        # Detect intent (pass purpose string, not dict)
        intent = detect_user_intent(purpose)
        role_action = INTENT_TO_ROLE_MAP.get(intent, 'help with')
        print(f"  Detected intent: '{intent}' → role: '{role_action}'")

        # Compute user PA centroid embedding (NEW: includes example_queries)
        user_embedding = compute_centroid_embedding(user_pa_text, example_queries, model)

        # Derive AI PA text
        ai_pa_text = derive_ai_pa_text(purpose, scope, intent)

        # Get example AI responses for centroid computation
        example_ai_responses = template.get('example_ai_responses', [])
        print(f"  Example AI responses: {len(example_ai_responses)}")

        # Compute AI PA centroid embedding (NEW: includes example_ai_responses)
        # This creates an AI PA that covers the behavioral space of aligned responses,
        # resulting in higher fidelity scores for behaviorally-aligned AI responses.
        ai_embedding = compute_centroid_embedding(ai_pa_text, example_ai_responses, model)

        # Compute rho_PA (correlation between user and AI attractors)
        rho_pa = float(np.dot(user_embedding, ai_embedding))
        print(f"  rho_PA (attractor correlation): {rho_pa:.3f}")

        # Store
        all_embeddings[template_id] = {
            'user_pa_text': user_pa_text,
            'ai_pa_text': ai_pa_text,
            'detected_intent': intent,
            'role_action': role_action,
            'example_queries': example_queries,
            'example_ai_responses': example_ai_responses,  # NEW: store for reference
            'user_centroid_sources': 1 + len(example_queries),  # User PA sources
            'ai_centroid_sources': 1 + len(example_ai_responses),  # AI PA sources
            'rho_pa': rho_pa,  # Correlation between attractors
            'user_pa_embedding': user_embedding.tolist(),
            'ai_pa_embedding': ai_embedding.tolist()
        }
        print()

    # Save to JSON
    print("=" * 60)
    print(f"Saving to {OUTPUT_PATH}")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_embeddings, f, indent=2)

    print(f"Saved embeddings for {len(all_embeddings)} templates")
    print()

    # Verify by testing "What is recursion?" against learn_concept
    print("=" * 60)
    print("Verification Test: 'What is recursion?' vs learn_concept")
    print("=" * 60)

    test_query = "What is recursion?"
    test_embedding = np.array(model.encode(test_query))
    test_embedding = test_embedding / (np.linalg.norm(test_embedding) + 1e-10)

    learn_concept_embedding = np.array(all_embeddings['learn_concept']['user_pa_embedding'])

    raw_similarity = float(np.dot(test_embedding, learn_concept_embedding))

    # Apply normalization (from fidelity_display.py - 2025-12-28 CENTROID calibration)
    LINEAR_SLOPE = 1.167
    LINEAR_INTERCEPT = 0.117
    display_fidelity = LINEAR_SLOPE * raw_similarity + LINEAR_INTERCEPT
    display_fidelity = max(0.0, min(1.0, display_fidelity))

    print(f"Query: '{test_query}'")
    print(f"Raw similarity: {raw_similarity:.4f}")
    print(f"Display fidelity: {display_fidelity:.2%}")

    if display_fidelity >= 0.70:
        print("Zone: GREEN (Aligned) ✓")
    elif display_fidelity >= 0.60:
        print("Zone: YELLOW (Minor Drift)")
    elif display_fidelity >= 0.50:
        print("Zone: ORANGE (Drift Detected)")
    else:
        print("Zone: RED (Significant Drift)")

    print()
    print("Done!")


if __name__ == '__main__':
    main()
