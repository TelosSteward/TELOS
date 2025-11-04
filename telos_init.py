#!/usr/bin/env python3
"""
TELOS Session Initialization
=============================

Establishes mathematical Primacy Attractor from .claude_project.md
and activates real-time self-monitoring.

Usage:
    python3 telos_init.py

Creates:
    .telos_session_pa.json - PA embedding and metadata
    .telos_session_log.json - Turn-by-turn fidelity log
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from telos_purpose.core.embedding_provider import EmbeddingProvider


def extract_pa_from_claude_project():
    """Extract Session Purpose and Core Values from .claude_project.md"""

    project_file = Path('.claude_project.md')

    if not project_file.exists():
        raise FileNotFoundError('.claude_project.md not found')

    content = project_file.read_text()

    # Extract Session Purpose
    if '### Session Purpose (User PA)' not in content:
        raise ValueError('Session Purpose not found in .claude_project.md')

    lines = content.split('\n')

    # Find Session Purpose
    purpose_start = None
    purpose_end = None
    for i, line in enumerate(lines):
        if '### Session Purpose (User PA)' in line:
            purpose_start = i + 2  # Skip header and blank line
        elif purpose_start and line.startswith('###'):
            purpose_end = i
            break

    if purpose_start is None:
        raise ValueError('Could not parse Session Purpose')

    purpose_text = '\n'.join(lines[purpose_start:purpose_end]).strip()

    # Extract Core Values (for richer embedding)
    values_start = None
    values_end = None
    for i, line in enumerate(lines):
        if '### Core Values (What TELOS IS)' in line:
            values_start = i + 2
        elif values_start and line.startswith('###'):
            values_end = i
            break

    values_text = ""
    if values_start:
        values_text = '\n'.join(lines[values_start:values_end]).strip()

    # Combine for full PA
    full_pa_text = purpose_text
    if values_text:
        full_pa_text += "\n\n" + values_text

    return full_pa_text


def establish_pa():
    """Establish mathematical PA from .claude_project.md"""

    print("\n🔭 TELOS Session Initialization")
    print("="*60)

    # Extract PA text
    print("Reading PA from .claude_project.md...")
    pa_text = extract_pa_from_claude_project()

    print(f"✅ PA extracted ({len(pa_text)} chars)")
    print(f"   Purpose: {pa_text[:80]}...")

    # Generate embedding
    print("\nGenerating embedding (ℝ³⁸⁴ space)...")
    embeddings = EmbeddingProvider(deterministic=False)
    pa_embedding = embeddings.encode(pa_text)

    print(f"✅ Embedding generated")
    print(f"   Dimensions: {len(pa_embedding)}")
    print(f"   Norm: {(pa_embedding ** 2).sum() ** 0.5:.3f}")

    # Save session PA
    session_pa = {
        'pa_text': pa_text,
        'pa_embedding': pa_embedding.tolist(),
        'threshold': 0.65,
        'established_at': datetime.now().isoformat(),
        'session_start': datetime.now().isoformat()
    }

    with open('.telos_session_pa.json', 'w') as f:
        json.dump(session_pa, f, indent=2)

    print(f"\n💾 Session PA saved to .telos_session_pa.json")

    # Initialize session log
    session_log = {
        'session_start': datetime.now().isoformat(),
        'turns': []
    }

    with open('.telos_session_log.json', 'w') as f:
        json.dump(session_log, f, indent=2)

    print(f"💾 Session log initialized")

    # Summary
    print("\n" + "="*60)
    print("✅ TELOS Session Governance ACTIVE")
    print("="*60)
    print(f"\n📊 Primacy Attractor Established:")
    print(f"   Purpose: {pa_text.split('.')[0]}...")
    print(f"   Embedding: ℝ³⁸⁴ (384-dimensional space)")
    print(f"   Threshold: F ≥ 0.65")
    print(f"   Established: {session_pa['established_at']}")

    print(f"\n🔭 Active Monitoring:")
    print(f"   ✅ Real-time fidelity checking enabled")
    print(f"   ✅ Turn-by-turn self-evaluation active")
    print(f"   ✅ Automatic intervention on drift")
    print(f"   ✅ Session metrics logged")

    print(f"\n🚀 Ready for governed session.")
    print(f"\nUse: python3 telos_check.py \"[response text]\" to check fidelity")


if __name__ == '__main__':
    try:
        establish_pa()
    except Exception as e:
        print(f"\n❌ Error establishing PA: {e}")
        sys.exit(1)
