"""
Phase 2 Standalone Counterfactual Branch Manager
=================================================

Simplified counterfactual branching specifically for Phase 2 batch studies.

Differences from CounterfactualBranchManager:
- No UnifiedSteward required (applies interventions directly)
- Saves evidence to files (not returns strings)
- Simpler interface for batch processing
- Optimized for ShareGPT historical data
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import json

# Type hint for external dependencies
try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from telos_purpose.llm_clients.mistral_client import MistralClient
        from telos_purpose.core.embedding_provider import SentenceTransformerProvider
except ImportError:
    pass


@dataclass
class Phase2Turn:
    """Single turn in a Phase 2 branch."""
    turn_number: int
    user_input: str
    assistant_response: str
    fidelity: float
    intervention_applied: bool = False


class Phase2BranchManager:
    """
    Standalone counterfactual branch manager for Phase 2 studies.

    Generates two independent branches:
    1. Original: Historical user inputs + historical responses
    2. TELOS: Historical user inputs + NEW API-generated responses

    Simple intervention: On first turn, prepend governance correction to response.
    """

    def __init__(
        self,
        llm_client: Any,
        embedding_provider: Any,
        distance_scale: float = 2.0
    ):
        """
        Initialize Phase 2 branch manager.

        Args:
            llm_client: LLM client for generating responses
            embedding_provider: Embedding provider for fidelity calculation
            distance_scale: Distance-to-fidelity scaling
        """
        self.llm_client = llm_client
        self.embedding_provider = embedding_provider
        self.distance_scale = distance_scale

        # Storage
        self.branches: Dict[str, Dict[str, Any]] = {}

    def trigger_counterfactual(
        self,
        trigger_turn: int,
        trigger_fidelity: float,
        remaining_turns: List[Tuple[str, str]],  # [(user, assistant), ...]
        attractor_center: np.ndarray,
        conversation_history: List[Tuple[str, str]],  # History up to trigger
        attractor_purpose: List[str],
        attractor_scope: List[str],
        attractor_boundaries: List[str],
        branch_length: int = 5
    ) -> str:
        """
        Generate counterfactual branches from drift point.

        Args:
            trigger_turn: Turn number where drift detected
            trigger_fidelity: Fidelity at drift point
            remaining_turns: Remaining conversation turns
            attractor_center: PA centroid for fidelity calculation
            conversation_history: Conversation up to drift point
            attractor_purpose: PA purpose for intervention
            attractor_scope: PA scope for intervention
            attractor_boundaries: PA boundaries for intervention
            branch_length: How many turns to generate

        Returns:
            Branch ID
        """
        # Limit to available or requested length
        num_turns = min(len(remaining_turns), branch_length)
        branch_turns = remaining_turns[:num_turns]

        # Generate branch ID
        timestamp = datetime.now().strftime("%H%M%S")
        branch_id = f"intervention_{trigger_turn}_{timestamp}"

        print(f"\n🌿 Generating branches ({num_turns} turns)...")

        # ============================================================
        # ORIGINAL BRANCH (Historical)
        # ============================================================

        original_turns = []
        for i, (user_msg, assistant_msg) in enumerate(branch_turns):
            turn_num = trigger_turn + i + 1

            # Calculate fidelity
            response_emb = self.embedding_provider.encode(assistant_msg)
            distance = np.linalg.norm(response_emb - attractor_center)
            fidelity = max(0.0, 1.0 - (distance / self.distance_scale))

            original_turns.append(Phase2Turn(
                turn_number=turn_num,
                user_input=user_msg,
                assistant_response=assistant_msg,
                fidelity=fidelity,
                intervention_applied=False
            ))

        # ============================================================
        # TELOS BRANCH (Counterfactual)
        # ============================================================

        telos_turns = []

        # Build conversation history in LLM format
        llm_history = []
        for user_msg, assistant_msg in conversation_history:
            llm_history.append({"role": "user", "content": user_msg})
            llm_history.append({"role": "assistant", "content": assistant_msg})

        for i, (user_msg, _historical_response) in enumerate(branch_turns):
            turn_num = trigger_turn + i + 1

            # Add user input to history
            llm_history.append({"role": "user", "content": user_msg})

            # Generate response
            raw_response = self.llm_client.generate(
                messages=llm_history,
                temperature=0.7,
                max_tokens=500
            )

            # Apply intervention on first turn
            if i == 0:
                intervention_response = self._apply_intervention(
                    raw_response=raw_response,
                    user_input=user_msg,
                    purpose=attractor_purpose,
                    scope=attractor_scope,
                    boundaries=attractor_boundaries
                )
                response = intervention_response
                intervention_applied = True
            else:
                response = raw_response
                intervention_applied = False

            # Add to history for next turn
            llm_history.append({"role": "assistant", "content": response})

            # Calculate fidelity
            response_emb = self.embedding_provider.encode(response)
            distance = np.linalg.norm(response_emb - attractor_center)
            fidelity = max(0.0, 1.0 - (distance / self.distance_scale))

            telos_turns.append(Phase2Turn(
                turn_number=turn_num,
                user_input=user_msg,
                assistant_response=response,
                fidelity=fidelity,
                intervention_applied=intervention_applied
            ))

            print(f"  Turn {turn_num}: Original F={original_turns[i].fidelity:.3f}, TELOS F={fidelity:.3f}")

        # ============================================================
        # STORE BRANCH DATA
        # ============================================================

        original_fidelities = [t.fidelity for t in original_turns]
        telos_fidelities = [t.fidelity for t in telos_turns]

        self.branches[branch_id] = {
            'branch_id': branch_id,
            'trigger_turn': trigger_turn,
            'trigger_fidelity': trigger_fidelity,
            'num_turns': num_turns,
            'original': {
                'turns': original_turns,
                'final_fidelity': original_fidelities[-1],
                'avg_fidelity': np.mean(original_fidelities),
                'fidelity_trajectory': original_fidelities
            },
            'telos': {
                'turns': telos_turns,
                'final_fidelity': telos_fidelities[-1],
                'avg_fidelity': np.mean(telos_fidelities),
                'fidelity_trajectory': telos_fidelities
            },
            'comparison': {
                'delta_f': telos_fidelities[-1] - original_fidelities[-1],
                'governance_effective': telos_fidelities[-1] > original_fidelities[-1]
            },
            'attractor': {
                'purpose': attractor_purpose,
                'scope': attractor_scope,
                'boundaries': attractor_boundaries
            },
            'timestamp': datetime.now().isoformat()
        }

        return branch_id

    def _apply_intervention(
        self,
        raw_response: str,
        user_input: str,
        purpose: List[str],
        scope: List[str],
        boundaries: List[str]
    ) -> str:
        """
        Apply TELOS governance intervention to response.

        Simple approach: Ask LLM to revise response to align with PA.

        Args:
            raw_response: Original LLM response
            user_input: User's question
            purpose: PA purpose
            scope: PA scope
            boundaries: PA boundaries

        Returns:
            Governed response
        """
        intervention_prompt = f"""You are a governance system correcting an AI response to align with conversation purpose.

CONVERSATION PURPOSE: {', '.join(purpose)}
ALLOWED SCOPE: {', '.join(scope)}
BOUNDARIES: {', '.join(boundaries)}

USER ASKED: {user_input}

ORIGINAL RESPONSE (may be off-topic):
{raw_response}

Provide a CORRECTED response that:
1. Directly addresses the user's question
2. Stays within the conversation's established purpose and scope
3. Respects the boundaries
4. Is helpful and clear

CORRECTED RESPONSE:"""

        corrected = self.llm_client.generate(
            messages=[{"role": "user", "content": intervention_prompt}],
            temperature=0.3,
            max_tokens=500
        )

        return corrected

    def get_branch(self, branch_id: str) -> Optional[Dict[str, Any]]:
        """Get branch data."""
        return self.branches.get(branch_id)

    def export_evidence(
        self,
        branch_id: str,
        output_dir: str
    ) -> Tuple[str, str]:
        """
        Export branch evidence to files.

        Args:
            branch_id: Branch to export
            output_dir: Directory to save files

        Returns:
            Tuple of (json_path, markdown_path)
        """
        branch_data = self.branches.get(branch_id)
        if not branch_data:
            raise ValueError(f"Branch {branch_id} not found")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # ============================================================
        # JSON EXPORT
        # ============================================================

        # Convert dataclasses to dicts (convert numpy types to Python native types)
        json_data = {
            'branch_id': branch_data['branch_id'],
            'trigger_turn': int(branch_data['trigger_turn']),
            'trigger_fidelity': float(branch_data['trigger_fidelity']),
            'num_turns': int(branch_data['num_turns']),
            'original': {
                'turns': [
                    {
                        'turn_number': int(t.turn_number),
                        'user_input': t.user_input,
                        'assistant_response': t.assistant_response,
                        'fidelity': float(t.fidelity)
                    }
                    for t in branch_data['original']['turns']
                ],
                'final_fidelity': float(branch_data['original']['final_fidelity']),
                'avg_fidelity': float(branch_data['original']['avg_fidelity']),
                'fidelity_trajectory': [float(f) for f in branch_data['original']['fidelity_trajectory']]
            },
            'telos': {
                'turns': [
                    {
                        'turn_number': int(t.turn_number),
                        'user_input': t.user_input,
                        'assistant_response': t.assistant_response,
                        'fidelity': float(t.fidelity),
                        'intervention_applied': bool(t.intervention_applied)
                    }
                    for t in branch_data['telos']['turns']
                ],
                'final_fidelity': float(branch_data['telos']['final_fidelity']),
                'avg_fidelity': float(branch_data['telos']['avg_fidelity']),
                'fidelity_trajectory': [float(f) for f in branch_data['telos']['fidelity_trajectory']]
            },
            'comparison': {
                'delta_f': float(branch_data['comparison']['delta_f']),
                'governance_effective': bool(branch_data['comparison']['governance_effective'])
            },
            'attractor': branch_data['attractor'],
            'timestamp': branch_data['timestamp']
        }

        json_path = output_path / f"{branch_id}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        # ============================================================
        # MARKDOWN EXPORT
        # ============================================================

        md_lines = [
            "# TELOS Intervention Evidence - Phase 2",
            "",
            "## Summary",
            f"- **Branch ID**: {branch_id}",
            f"- **Trigger Turn**: {branch_data['trigger_turn']}",
            f"- **Trigger Fidelity**: {branch_data['trigger_fidelity']:.3f}",
            f"- **Turns Analyzed**: {branch_data['num_turns']}",
            "",
            "### Results",
            f"- **Original Final F**: {branch_data['original']['final_fidelity']:.3f}",
            f"- **TELOS Final F**: {branch_data['telos']['final_fidelity']:.3f}",
            f"- **ΔF (Improvement)**: {branch_data['comparison']['delta_f']:+.3f}",
            f"- **Governance Effective**: {'✅ Yes' if branch_data['comparison']['governance_effective'] else '❌ No'}",
            "",
            "---",
            "",
            "## Primacy Attractor",
            f"**Purpose**: {', '.join(branch_data['attractor']['purpose'])}",
            f"**Scope**: {', '.join(branch_data['attractor']['scope'][:3])}...",
            f"**Boundaries**: {', '.join(branch_data['attractor']['boundaries'][:2])}...",
            "",
            "---",
            "",
            "## Turn-by-Turn Comparison",
            ""
        ]

        for i in range(branch_data['num_turns']):
            orig_turn = branch_data['original']['turns'][i]
            telos_turn = branch_data['telos']['turns'][i]

            md_lines.extend([
                f"### Turn {orig_turn.turn_number}",
                "",
                f"**User**: {orig_turn.user_input}",
                "",
                "#### Original Response (Historical)",
                f"```\n{orig_turn.assistant_response}\n```",
                f"**Fidelity**: {orig_turn.fidelity:.3f}",
                "",
                "#### TELOS Response (Counterfactual)",
                f"```\n{telos_turn.assistant_response}\n```",
                f"**Fidelity**: {telos_turn.fidelity:.3f}",
            ])

            if telos_turn.intervention_applied:
                md_lines.append("🛡️ **Intervention Applied**")

            md_lines.extend(["", "---", ""])

        md_lines.extend([
            "",
            "## Fidelity Trajectories",
            "",
            "| Turn | Original F | TELOS F | Δ |",
            "|------|-----------|---------|---|"
        ])

        for i in range(branch_data['num_turns']):
            turn_num = branch_data['original']['turns'][i].turn_number
            orig_f = branch_data['original']['fidelity_trajectory'][i]
            telos_f = branch_data['telos']['fidelity_trajectory'][i]
            delta = telos_f - orig_f
            md_lines.append(f"| {turn_num} | {orig_f:.3f} | {telos_f:.3f} | {delta:+.3f} |")

        md_lines.extend([
            "",
            "---",
            "",
            f"**Generated**: {branch_data['timestamp']}",
            "**Source**: TELOS Phase 2 Study",
            "**Framework**: ProgressivePrimacyExtractor + Phase2BranchManager"
        ])

        md_path = output_path / f"{branch_id}.md"
        with open(md_path, 'w') as f:
            f.write('\n'.join(md_lines))

        return (str(json_path), str(md_path))
