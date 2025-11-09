#!/usr/bin/env python3
"""
Run REAL validation with actual API calls to Mistral.
This will generate actual fidelity measurements, not placeholder data.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mistralai import Mistral
from sentence_transformers import SentenceTransformer
from tests.validation.comparative_test import ComparativeValidator
from telos.core.unified_steward import PrimacyAttractor

class MistralLLMClient:
    """Wrapper for Mistral API to match expected interface."""

    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-small-2501"

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response from Mistral."""
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

class EmbeddingProvider:
    """Wrapper for SentenceTransformer embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str):
        """Generate embedding for text."""
        return self.model.encode(text)

def run_real_validation():
    """Run validation with actual API calls."""

    print("\n" + "="*70)
    print("REAL TELOS VALIDATION WITH API CALLS")
    print("="*70)

    # Load API key from secrets
    try:
        with open(".streamlit/secrets.toml", 'r') as f:
            content = f.read()
            # Extract API key (crude parsing)
            for line in content.split('\n'):
                if 'MISTRAL_API_KEY' in line:
                    api_key = line.split('=')[1].strip().strip('"')
                    break
    except:
        print("ERROR: Could not load Mistral API key from .streamlit/secrets.toml")
        return

    print("\n✓ Mistral API key loaded")

    # Initialize clients
    print("Initializing LLM client...")
    llm_client = MistralLLMClient(api_key)

    print("Initializing embedding provider...")
    embedding_provider = EmbeddingProvider()

    # Load the Claude conversation data
    data_file = Path("tests/validation_data/baseline_conversations/sharegpt_claude_conversation.json")
    if not data_file.exists():
        print(f"ERROR: Cannot find {data_file}")
        return

    with open(data_file, 'r') as f:
        session_data = json.load(f)

    # Extract conversation turns (just first 5 for testing to save API costs)
    print("\nExtracting conversation turns...")
    conversation = []
    for turn in session_data['turns'][:5]:  # Limit to 5 turns for testing
        user_input = turn.get('user_input', '')
        # We need a baseline response - for now use the garbled one as reference
        assistant_response = turn.get('assistant_response_telos', '')
        conversation.append((user_input, assistant_response))

    print(f"Processing {len(conversation)} turns")

    # Extract PA
    pa_data = session_data.get('primacy_attractor', {})
    attractor_config = PrimacyAttractor(
        purpose=pa_data.get('purpose', ['General conversation']),
        scope=pa_data.get('scope', ['Open dialogue']),
        boundaries=pa_data.get('boundaries', ['Maintain respectful discourse'])
    )

    # Run comparative validation
    print("\nStarting comparative validation...")
    validator = ComparativeValidator(
        llm_client=llm_client,
        embedding_provider=embedding_provider,
        output_dir="tests/validation_results/real_runs"
    )

    try:
        results = validator.run_comparative_study(
            conversation=conversation,
            attractor_config=attractor_config,
            study_id="claude_conversation_real"
        )

        print("\n" + "="*70)
        print("VALIDATION COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: tests/validation_results/real_runs/claude_conversation_real.json")

    except Exception as e:
        print(f"\nERROR during validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_real_validation()