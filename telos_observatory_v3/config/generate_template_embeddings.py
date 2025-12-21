"""
Generate Template Embeddings
============================

Pre-computes User PA and AI PA embeddings for all templates.
This replicates EXACTLY the same process as when a user establishes their own PA.

Run this script once to generate embeddings, then they're stored for instant loading.

Usage:
    cd /Users/brunnerjf/Desktop/telos_privacy
    python3 telos_observatory_v3/config/generate_template_embeddings.py
"""

import sys
import os
import json
import numpy as np

# Add project root to path
sys.path.insert(0, '/Users/brunnerjf/Desktop/telos_privacy')
os.chdir('/Users/brunnerjf/Desktop/telos_privacy')

from telos_observatory_v3.config.pa_templates import PA_TEMPLATES

# Intent detection keywords (same as beta_response_manager.py)
INTENT_KEYWORDS = {
    'learn': ['learn', 'understand', 'study', 'explore', 'discover', 'grasp'],
    'understand': ['understand', 'comprehend', 'clarify', 'explain'],
    'solve': ['solve', 'fix', 'resolve', 'address'],
    'create': ['create', 'build', 'develop', 'implement', 'write', 'design', 'make'],
    'decide': ['decide', 'choose', 'select', 'determine'],
    'explore': ['explore', 'investigate', 'research', 'examine'],
    'analyze': ['analyze', 'review', 'assess', 'evaluate'],
    'fix': ['fix', 'repair', 'correct', 'debug'],
    'debug': ['debug', 'troubleshoot', 'diagnose'],
    'optimize': ['optimize', 'improve', 'enhance', 'refine'],
    'research': ['research', 'investigate', 'study', 'survey'],
    'plan': ['plan', 'organize', 'structure', 'schedule']
}

# Intent to Role mapping (same as beta_response_manager.py)
INTENT_TO_ROLE_MAP = {
    'learn': 'teach',
    'understand': 'explain',
    'solve': 'help solve',
    'create': 'help create',
    'decide': 'help decide',
    'explore': 'guide exploration',
    'analyze': 'help analyze',
    'fix': 'help fix',
    'debug': 'help debug',
    'optimize': 'help optimize',
    'research': 'help research',
    'plan': 'help plan'
}


def detect_intent_from_purpose(purpose: str) -> str:
    """
    Detect primary intent from purpose statement.
    EXACT copy of logic from beta_response_manager.py
    """
    purpose_lower = purpose.lower()

    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in purpose_lower:
                return intent

    return 'explore'  # Default intent


def create_pa_text(purpose: str, scope: list) -> str:
    """
    Create User PA text string for embedding.
    EXACT format used in beta_response_manager.py
    """
    scope_str = ", ".join(scope) if isinstance(scope, list) else str(scope)
    return f"Purpose: {purpose}. Scope: {scope_str}."


def derive_ai_pa_text(purpose: str, scope: list) -> tuple:
    """
    Derive AI PA text from User PA.
    EXACT process used in beta_response_manager.py

    Returns:
        tuple: (ai_pa_text, detected_intent, role_action)
    """
    detected_intent = detect_intent_from_purpose(purpose)
    role_action = INTENT_TO_ROLE_MAP.get(detected_intent, 'help')

    ai_purpose = f"{role_action.capitalize()} the user as they work to: {purpose}"
    scope_str = ", ".join(scope) if isinstance(scope, list) else str(scope)
    ai_pa_text = f"AI Role: {ai_purpose}. Supporting scope: {scope_str}."

    return ai_pa_text, detected_intent, role_action


def generate_all_template_embeddings():
    """
    Generate embeddings for all templates using the EXACT same process
    as when a user establishes their own PA.
    """
    print("=" * 60)
    print("TEMPLATE EMBEDDING GENERATOR")
    print("Replicating exact PA establishment process for templates")
    print("=" * 60)

    # Import Mistral client directly (avoid complex imports)
    from mistralai import Mistral
    import streamlit as st

    # Get API key
    api_key = os.environ.get('MISTRAL_API_KEY') or st.secrets.get('MISTRAL_API_KEY', 'ZlU1mXEIwk963YxmDzjX1Uq99Hu05QpO')

    print("\nInitializing Mistral embedding client (1024-dim)...")
    client = Mistral(api_key=api_key)
    embedding_dim = 1024

    def get_embedding(text: str) -> np.ndarray:
        """Get embedding for a text using Mistral API."""
        response = client.embeddings.create(
            model="mistral-embed",
            inputs=[text]
        )
        return np.array(response.data[0].embedding)

    print(f"Embedding dimension: {embedding_dim}")

    # Storage for all template embeddings
    template_embeddings = {}

    for template in PA_TEMPLATES:
        template_id = template['id']
        purpose = template['purpose']
        scope = template['scope']

        print(f"\n--- Processing: {template_id} ---")
        print(f"Purpose: {purpose[:60]}...")

        # STEP 1: Create User PA text (EXACT same as user PA establishment)
        user_pa_text = create_pa_text(purpose, scope)
        print(f"User PA text: {user_pa_text[:80]}...")

        # STEP 2: Derive AI PA text (EXACT same as user PA establishment)
        ai_pa_text, detected_intent, role_action = derive_ai_pa_text(purpose, scope)
        print(f"Detected intent: {detected_intent} -> role: {role_action}")
        print(f"AI PA text: {ai_pa_text[:80]}...")

        # STEP 3: Compute embeddings (same as beta_response_manager.py)
        print("Computing embeddings...")
        user_pa_embedding = get_embedding(user_pa_text)
        ai_pa_embedding = get_embedding(ai_pa_text)

        # STEP 4: Compute PA correlation (rho_PA) - useful for diagnostics
        rho_pa = float(np.dot(user_pa_embedding, ai_pa_embedding) /
                      (np.linalg.norm(user_pa_embedding) * np.linalg.norm(ai_pa_embedding)))

        print(f"User PA embedding: {len(user_pa_embedding)} dims")
        print(f"AI PA embedding: {len(ai_pa_embedding)} dims")
        print(f"PA Correlation (rho_PA): {rho_pa:.4f}")

        # Store as lists (JSON serializable)
        template_embeddings[template_id] = {
            'user_pa_text': user_pa_text,
            'ai_pa_text': ai_pa_text,
            'detected_intent': detected_intent,
            'role_action': role_action,
            'user_pa_embedding': user_pa_embedding.tolist() if isinstance(user_pa_embedding, np.ndarray) else list(user_pa_embedding),
            'ai_pa_embedding': ai_pa_embedding.tolist() if isinstance(ai_pa_embedding, np.ndarray) else list(ai_pa_embedding),
            'rho_pa': rho_pa,
            'embedding_dim': embedding_dim,
            'embedding_model': 'mistral-embed'
        }

    # Save to JSON file
    output_path = '/Users/brunnerjf/Desktop/telos_privacy/telos_observatory_v3/config/pa_template_embeddings.json'
    print(f"\n{'=' * 60}")
    print(f"Saving embeddings to: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(template_embeddings, f, indent=2)

    # Also create a compact numpy version for faster loading
    embeddings_np = {}
    for tid, data in template_embeddings.items():
        embeddings_np[tid] = {
            'user_pa': np.array(data['user_pa_embedding']),
            'ai_pa': np.array(data['ai_pa_embedding']),
            'rho_pa': data['rho_pa']
        }

    np_path = '/Users/brunnerjf/Desktop/telos_privacy/telos_observatory_v3/config/pa_template_embeddings.npz'
    np.savez_compressed(np_path, **{f"{k}_{sub}": v for k, inner in embeddings_np.items() for sub, v in inner.items()})
    print(f"Saved numpy format to: {np_path}")

    print(f"\n{'=' * 60}")
    print("DONE! Template embeddings generated successfully.")
    print(f"Total templates processed: {len(template_embeddings)}")
    print("=" * 60)

    return template_embeddings


if __name__ == '__main__':
    generate_all_template_embeddings()
