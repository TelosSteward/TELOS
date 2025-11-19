"""
Steward PM - Project Memory for TELOS
======================================

This module serves as the project memory and context repository for TELOS development.
It contains critical documentation that should be referenced throughout development.

Last Updated: November 18, 2025
"""

import os
from pathlib import Path

class StewardPM:
    """
    Steward Project Memory - Maintains critical TELOS documentation and context.
    """

    def __init__(self):
        self.base_path = Path(__file__).parent
        self.critical_docs = {
            'beta_status': 'BETA_STATUS_SUMMARY.md',
            'beta_experience': 'BETA_EXPERIENCE_MASTER_FLOW.md',
            'market_position': 'business/TELOS_Market_Position_Reality_Check.md',
            'regulatory': 'business/Regulatory_Forcing_Function.md'
        }

    def get_document(self, doc_key: str) -> str:
        """
        Retrieve a critical document by key.

        Args:
            doc_key: One of 'beta_status', 'beta_experience', 'market_position', 'regulatory'

        Returns:
            Content of the document as string
        """
        if doc_key not in self.critical_docs:
            raise ValueError(f"Unknown document key: {doc_key}")

        doc_path = self.base_path / self.critical_docs[doc_key]

        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        with open(doc_path, 'r') as f:
            return f.read()

    def get_all_documents(self) -> dict:
        """
        Retrieve all critical documents.

        Returns:
            Dictionary mapping doc_key to document content
        """
        return {key: self.get_document(key) for key in self.critical_docs.keys()}

    def get_beta_experience_flow(self) -> str:
        """
        Get the BETA Experience Master Flow document.
        This is the comprehensive guide to the BETA user experience.

        Returns:
            BETA Experience Master Flow document content
        """
        return self.get_document('beta_experience')

    def get_beta_status(self) -> str:
        """
        Get the current BETA integration status.

        Returns:
            BETA Status Summary document content
        """
        return self.get_document('beta_status')

    def get_market_context(self) -> str:
        """
        Get the TELOS market position and strategy context.

        Returns:
            Market Position Reality Check document content
        """
        return self.get_document('market_position')

    def get_regulatory_context(self) -> str:
        """
        Get the regulatory forcing function context.

        Returns:
            Regulatory Forcing Function document content
        """
        return self.get_document('regulatory')

    def print_context_summary(self):
        """
        Print a summary of all available context documents.
        """
        print("=" * 60)
        print("STEWARD PM - TELOS Project Memory")
        print("=" * 60)
        print("\nAvailable Critical Documents:")
        print("-" * 60)

        for key, filename in self.critical_docs.items():
            doc_path = self.base_path / filename
            exists = "✓" if doc_path.exists() else "✗"
            print(f"{exists} {key.upper():20s} -> {filename}")

        print("-" * 60)
        print("\nUsage:")
        print("  pm = StewardPM()")
        print("  beta_flow = pm.get_beta_experience_flow()")
        print("  all_docs = pm.get_all_documents()")
        print("=" * 60)


# ============================================================================
# EMBEDDED BETA EXPERIENCE MASTER FLOW
# ============================================================================
# This is embedded here for easy programmatic access

BETA_EXPERIENCE_MASTER_FLOW = """
# TELOS BETA Experience - Complete User Journey

**Date:** November 18, 2025
**Status:** Production-Ready Documentation
**Purpose:** Comprehensive guide to the BETA user experience flow

---

## Overview

TELOS BETA is a controlled A/B testing environment that allows users to experience both native LLM responses and TELOS-governed responses while TELOS maintains alignment monitoring in the background. This document details every step of the BETA journey.

---

## Phase 0: Entry & Consent

### Tab Structure on Entry
- **DEMO Tab** (left): Available, previously completed
- **BETA Tab** (center): Active (user has completed DEMO)
- **TELOS Tab** (right): Locked/greyed out (unlocks after BETA completion)

### Consent Flow
1. User clicks BETA tab
2. `BetaOnboarding` component shows privacy/consent information
3. User accepts consent to proceed
4. System transitions to PA establishment

**Component:** `components/beta_onboarding.py`

---

## Phase 1: Primacy Attractor (PA) Establishment

### The Four Questions

Users must answer 4 questions to establish their Primacy Attractor before conversation begins:

1. **Primary Goal**
   - Question: "What are you trying to accomplish in this conversation?"
   - Purpose: Establishes core purpose
   - **Required:** Yes

2. **Scope & Boundaries**
   - Question: "What topics should we focus on? What should we avoid?"
   - Purpose: Defines conversation scope
   - **Required:** Yes

3. **Success Criteria**
   - Question: "How will you know if this conversation is successful?"
   - Purpose: Sets measurable outcomes
   - **Required:** No (skippable after Q2)

4. **Style Preference**
   - Question: "Any communication style preferences?"
   - Purpose: Tailors response delivery
   - **Required:** No (skippable)

### Visual Design
- Yellow border (#FFD700, 3px) around all sections
- 32px header font
- 22px question font
- 20px user input font
- 16px help text font
- No emoji (professional presentation)

**Component:** `components/pa_onboarding.py`

---

## Phase 2: A/B Testing Sequence (Turns 1-15)

### Overview of Testing Strategy

TELOS uses a **pre-determined sequence** with controlled randomness to ensure balanced exposure:

- **Total Turns:** 15
- **Single-Blind Turns:** 10 (user sees one response, doesn't know source)
- **Head-to-Head Turns:** 5 (user sees both responses side-by-side)
- **TELOS Distribution:** 60% in single-blind turns (6 out of 10)
- **Native Distribution:** 40% in single-blind turns (4 out of 10)

### Phase 2A: Turns 1-5 (Single-Blind Only)

**Response Pool:** 3 TELOS + 2 Native (randomized order)

**What User Sees:**
- Turn number displayed above message
- Single response (source hidden)
- "Observation Deck" button at bottom
- Scrollable conversation history (read-only)
- Previous/Next navigation buttons

### Phase 2B: Turns 6-15 (Mixed Testing)

**Pattern:**
- **Even Turns (6, 8, 10, 12, 14):** Head-to-head (both responses shown)
- **Odd Turns (7, 9, 11, 13, 15):** Single-blind (pool of 3 TELOS + 2 Native, randomized)

**Component:** `services/beta_sequence_generator.py`

---

## Phase 3: Observatory Access

### When Observatory Unlocks

**Turn 10 (Midpoint Check):** Temporary access to review Turns 1-9 metrics
**Turn 15 (Completion):** Full Observatory access unlocked

### What Observatory Shows

#### 1. Fidelity Visualization (Bar Graphs)
- PS Score Over Time: Line graph showing alignment across all 15 turns
- Drift Detection: Highlights turns where drift occurred
- Intervention Points: Marks turns where TELOS intervened
- Comparison View: TELOS vs Native response fidelity scores

**Component:** `components/fidelity_visualization.py`

#### 2. Steward Explanations
For each turn with TELOS intervention, Steward provides:
- What happened: Description of drift event
- Why it matters: Relation to user's PA
- How TELOS handled it: Intervention details
- Impact: Effect on conversation alignment

**Component:** `components/observatory_review.py`

---

## Phase 4: BETA Completion & TELOS Unlock

Once user completes Turn 15:
1. Feedback Collection via `components/beta_feedback.py`
2. Final Observatory Review (full access to all metrics)
3. TELOS Tab Unlocks (user can now access full TELOS mode)

### Tab Structure After BETA
- **DEMO Tab** (left): Still accessible for reference
- **BETA Tab** (center): Completed, reviewable
- **TELOS Tab** (right): **UNLOCKED** - full governance mode available

---

## File Reference Map

| Component | File Path | Purpose |
|-----------|-----------|---------|
| PA Onboarding | `components/pa_onboarding.py` | 4-question PA establishment |
| Beta Consent | `components/beta_onboarding.py` | Privacy/consent flow |
| Sequence Generator | `services/beta_sequence_generator.py` | 15-turn A/B sequence logic |
| Response Manager | `services/beta_response_manager.py` | Generates/manages responses |
| PA Extractor | `services/pa_extractor.py` | Extracts PA from answers |
| Turn Markers | `components/turn_markers.py` | Turn number display |
| Observation Deck | `components/observation_deck.py` | Quick PS metrics view |
| Observatory Review | `components/observatory_review.py` | Full metrics analysis |
| Fidelity Viz | `components/fidelity_visualization.py` | Bar graphs, drift charts |
| Beta Feedback | `components/beta_feedback.py` | Post-completion feedback |
| Beta Completion | `components/beta_completion.py` | TELOS unlock logic |
| A/B Testing | `components/beta_ab_testing.py` | Metadata-only variant |
| Main Integration | `main.py` (lines 1034-1100+) | BETA mode rendering |

---

**END OF BETA EXPERIENCE MASTER FLOW**
"""


if __name__ == "__main__":
    # Demo usage
    pm = StewardPM()
    pm.print_context_summary()

    print("\n\nBETA EXPERIENCE FLOW (First 500 chars):")
    print("-" * 60)
    flow = pm.get_beta_experience_flow()
    print(flow[:500] + "...")
