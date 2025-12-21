"""
Counterfactual Branch Manager for TELOS - API-Based Interventions
====================================================================

CRITICAL TELOS DEMONSTRATION FEATURE:
When drift is detected in session replay, this creates TWO independent branches:
  1. ORIGINAL (Historical): What actually happened
  2. COUNTERFACTUAL (TELOS): What WOULD have happened with intervention

Both branches use REAL:
  - User inputs from the session
  - LLM API calls for responses
  - Fidelity calculations
  - Governance interventions

This provides CONCRETE EVIDENCE of TELOS governance efficacy.

Example Grant Evidence:
-----------------------
"At turn 12, fidelity dropped to 0.73. TELOS detected drift and triggered
intervention. Over the next 5 turns:
  - Original branch: Fidelity continued degrading (0.73 → 0.65 → 0.58)
  - TELOS branch: Intervention corrected drift (0.73 → 0.89 → 0.92)
  - Evidence: Side-by-side comparison with full conversation text + metrics"
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import copy
import json


@dataclass
class BranchTurn:
    """Single turn in a counterfactual branch."""
    turn_number: int
    user_input: str
    assistant_response: str
    metrics: Dict[str, float]
    intervention_applied: bool = False
    intervention_type: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CounterfactualBranch:
    """
    Complete counterfactual branch from intervention point.

    Tracks full conversation + metrics for N turns after trigger.
    """
    branch_id: str
    branch_type: str  # "original" (historical) or "telos" (counterfactual)
    trigger_turn: int
    turns: List[BranchTurn]
    final_fidelity: float
    avg_fidelity: float
    drift_trajectory: List[float]
    fidelity_trajectory: List[float]
    metadata: Dict[str, Any]


class CounterfactualBranchManager:
    """
    Manages API-based counterfactual branch generation for TELOS.

    REAL DEMONSTRATION FLOW:
    ------------------------
    1. Session replay detects drift at turn N (F < 0.76)  # Goldilocks: Aligned threshold
    2. Manager triggered with:
       - Current conversation state
       - Remaining turns from session (N+1, N+2, ...)
       - Attractor/steward for intervention
    3. Generates TWO branches:
       a) Original: Replay historical responses, calc metrics
       b) TELOS: Generate NEW responses via API with intervention
    4. Both branches use SAME user inputs for fair comparison
    5. Track all metrics independently
    6. Export side-by-side evidence

    KEY: This is NOT simulation - it's REAL API calls creating
    counterfactual responses that would have been generated if
    TELOS had been governing in real-time.
    """

    def __init__(
        self,
        llm_client: Any,
        embedding_provider: Any,
        steward: Any,
        branch_length: int = 5
    ):
        """
        Initialize counterfactual branch manager.

        Args:
            llm_client: LLM client for generating counterfactual responses
            embedding_provider: Embedding provider for distance/fidelity calculations
            steward: UnifiedGovernanceSteward for interventions
            branch_length: Number of turns to continue after trigger
        """
        self.llm = llm_client
        self.embeddings = embedding_provider
        self.steward = steward
        self.branch_length = branch_length

        self._branches: Dict[str, Dict[str, Any]] = {}  # branch_id -> {original, telos, metadata}

    def trigger_counterfactual(
        self,
        trigger_turn: int,
        trigger_fidelity: float,
        trigger_reason: str,
        conversation_history: List[Dict[str, str]],
        remaining_turns: List[Tuple[str, str]],
        attractor_center: np.ndarray,
        distance_scale: float = 2.0
    ) -> str:
        """
        Trigger counterfactual branch generation with REAL API calls.

        Args:
            trigger_turn: Turn number where drift detected
            trigger_fidelity: Fidelity at trigger point
            trigger_reason: Human-readable trigger reason
            conversation_history: Full conversation up to trigger point
            remaining_turns: List of (user_input, historical_response) for continuation
            attractor_center: Attractor centroid for fidelity calculation
            distance_scale: Distance-to-fidelity scaling

        Returns:
            branch_id: Identifier for this intervention
        """
        branch_id = f"intervention_{trigger_turn}_{datetime.now().strftime('%H%M%S')}"

        print(f"\n🌿 BRANCHING at turn {trigger_turn} (F={trigger_fidelity:.3f})")
        print(f"   Reason: {trigger_reason}")
        print(f"   Generating {min(len(remaining_turns), self.branch_length)} turns for each branch...")

        # Limit to branch_length
        turns_to_generate = remaining_turns[:self.branch_length]

        # Generate both branches
        original_branch = self._generate_original_branch(
            trigger_turn=trigger_turn,
            trigger_fidelity=trigger_fidelity,
            conversation_history=conversation_history,
            remaining_turns=turns_to_generate,
            attractor_center=attractor_center,
            distance_scale=distance_scale,
            branch_id=branch_id
        )

        telos_branch = self._generate_telos_branch(
            trigger_turn=trigger_turn,
            trigger_fidelity=trigger_fidelity,
            conversation_history=conversation_history,
            remaining_turns=turns_to_generate,
            attractor_center=attractor_center,
            distance_scale=distance_scale,
            branch_id=branch_id
        )

        # Store branches
        self._branches[branch_id] = {
            'original': original_branch,
            'telos': telos_branch,
            'trigger_turn': trigger_turn,
            'trigger_fidelity': trigger_fidelity,
            'trigger_reason': trigger_reason,
            'generated_at': datetime.now().isoformat(),
            'comparison': self._compute_comparison(original_branch, telos_branch)
        }

        # Print summary
        comparison = self._branches[branch_id]['comparison']
        print(f"\n✅ BRANCHING COMPLETE:")
        print(f"   Original final F: {original_branch.final_fidelity:.3f}")
        print(f"   TELOS final F: {telos_branch.final_fidelity:.3f}")
        print(f"   ΔF = {comparison['delta_f']:+.3f}")
        print(f"   Avg improvement: {comparison['avg_improvement']:+.3f}")

        return branch_id

    def _generate_original_branch(
        self,
        trigger_turn: int,
        trigger_fidelity: float,
        conversation_history: List[Dict[str, str]],
        remaining_turns: List[Tuple[str, str]],
        attractor_center: np.ndarray,
        distance_scale: float,
        branch_id: str
    ) -> CounterfactualBranch:
        """
        Generate original (historical) branch.

        Uses the ACTUAL historical responses but recalculates metrics.
        This shows what DID happen without intervention.
        """
        turns = []

        for i, (user_input, historical_response) in enumerate(remaining_turns):
            turn_num = trigger_turn + i + 1

            # Use historical response (what actually happened)
            response = historical_response

            # Calculate REAL metrics
            metrics = self._calculate_metrics(
                response=response,
                attractor_center=attractor_center,
                distance_scale=distance_scale
            )

            turn = BranchTurn(
                turn_number=turn_num,
                user_input=user_input,
                assistant_response=response,
                metrics=metrics,
                intervention_applied=False,
                intervention_type=None
            )
            turns.append(turn)

        return self._create_branch_summary(
            branch_id=f"{branch_id}_original",
            branch_type="original",
            trigger_turn=trigger_turn,
            turns=turns
        )

    def _generate_telos_branch(
        self,
        trigger_turn: int,
        trigger_fidelity: float,
        conversation_history: List[Dict[str, str]],
        remaining_turns: List[Tuple[str, str]],
        attractor_center: np.ndarray,
        distance_scale: float,
        branch_id: str
    ) -> CounterfactualBranch:
        """
        Generate TELOS (counterfactual) branch using REAL API calls.

        For each turn:
        1. Use SAME user input as original (fair comparison)
        2. Generate NEW response via Mistral API
        3. Apply TELOS intervention if needed
        4. Calculate REAL metrics
        5. Continue with corrected conversation

        This shows what WOULD have happened with TELOS governance.
        """
        turns = []
        current_history = copy.deepcopy(conversation_history)

        for i, (user_input, _historical_response) in enumerate(remaining_turns):
            turn_num = trigger_turn + i + 1

            # Add user input to history
            current_history.append({"role": "user", "content": user_input})

            # Generate NEW response via API (counterfactual)
            try:
                raw_response = self.llm.generate(
                    messages=current_history,
                    max_tokens=500,
                    temperature=0.7
                )
            except Exception as e:
                print(f"⚠️  API error on turn {turn_num}: {e}")
                raw_response = f"[API Error: {str(e)}]"

            # Apply TELOS intervention on first turn after trigger
            if i == 0:
                corrected_response = self._apply_telos_intervention(
                    response=raw_response,
                    user_input=user_input,
                    conversation_history=current_history[:-1],  # Exclude current user input
                    attractor_center=attractor_center
                )
                response = corrected_response
                intervention_applied = True
                intervention_type = "boundary_correction"
            else:
                # Check if subsequent intervention needed
                metrics_check = self._calculate_metrics(raw_response, attractor_center, distance_scale)
                if metrics_check['telic_fidelity'] < 0.75:
                    corrected_response = self._apply_telos_intervention(
                        response=raw_response,
                        user_input=user_input,
                        conversation_history=current_history[:-1],
                        attractor_center=attractor_center
                    )
                    response = corrected_response
                    intervention_applied = True
                    intervention_type = "drift_correction"
                else:
                    response = raw_response
                    intervention_applied = False
                    intervention_type = None

            # Add response to history for next turn
            current_history.append({"role": "assistant", "content": response})

            # Calculate REAL metrics
            metrics = self._calculate_metrics(
                response=response,
                attractor_center=attractor_center,
                distance_scale=distance_scale
            )

            turn = BranchTurn(
                turn_number=turn_num,
                user_input=user_input,
                assistant_response=response,
                metrics=metrics,
                intervention_applied=intervention_applied,
                intervention_type=intervention_type
            )
            turns.append(turn)

        return self._create_branch_summary(
            branch_id=f"{branch_id}_telos",
            branch_type="telos",
            trigger_turn=trigger_turn,
            turns=turns
        )

    def _apply_telos_intervention(
        self,
        response: str,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        attractor_center: np.ndarray
    ) -> str:
        """
        Apply TELOS governance intervention via API.

        Uses the steward's correction logic to generate aligned response.
        """
        # Get governance boundaries from steward
        attractor = self.steward.attractor
        boundaries_text = "\n".join([f"- {b}" for b in attractor.boundaries])

        # Create intervention prompt
        intervention_prompt = f"""The following response has drifted from the governance scope.

GOVERNANCE BOUNDARIES:
{boundaries_text}

USER INPUT:
{user_input}

DRIFTED RESPONSE:
{response}

Please provide a corrected response that:
1. Stays within the governance boundaries
2. Properly addresses the user's input
3. Maintains the conversation flow
4. Is helpful and aligned

CORRECTED RESPONSE:"""

        try:
            corrected = self.llm.generate(
                messages=[{"role": "user", "content": intervention_prompt}],
                max_tokens=500,
                temperature=0.5
            )
            return corrected
        except Exception as e:
            print(f"⚠️  Intervention API error: {e}")
            return response  # Fallback to original if intervention fails

    def _calculate_metrics(
        self,
        response: str,
        attractor_center: np.ndarray,
        distance_scale: float
    ) -> Dict[str, float]:
        """
        Calculate REAL metrics for a response.

        Uses actual embedding distance and fidelity calculation.
        """
        # Get response embedding
        response_emb = self.embeddings.encode([response])[0]

        # Calculate distance to attractor
        distance = float(np.linalg.norm(response_emb - attractor_center))

        # Calculate fidelity (distance-based)
        fidelity = max(0.0, min(1.0, 1.0 - (distance / distance_scale)))

        return {
            'telic_fidelity': fidelity,
            'drift_distance': distance,
            'error_signal': 1.0 - fidelity,
            'primacy_basin_membership': distance < (distance_scale / 2)
        }

    def _create_branch_summary(
        self,
        branch_id: str,
        branch_type: str,
        trigger_turn: int,
        turns: List[BranchTurn]
    ) -> CounterfactualBranch:
        """Create summary statistics for a branch."""
        fidelities = [t.metrics['telic_fidelity'] for t in turns]
        distances = [t.metrics['drift_distance'] for t in turns]

        return CounterfactualBranch(
            branch_id=branch_id,
            branch_type=branch_type,
            trigger_turn=trigger_turn,
            turns=turns,
            final_fidelity=fidelities[-1] if fidelities else 0.0,
            avg_fidelity=sum(fidelities) / len(fidelities) if fidelities else 0.0,
            drift_trajectory=distances,
            fidelity_trajectory=fidelities,
            metadata={
                'branch_length': len(turns),
                'interventions_applied': sum(1 for t in turns if t.intervention_applied)
            }
        )

    def _compute_comparison(
        self,
        original: CounterfactualBranch,
        telos: CounterfactualBranch
    ) -> Dict[str, Any]:
        """Compute comparison metrics between branches."""
        return {
            'delta_f': telos.final_fidelity - original.final_fidelity,
            'avg_improvement': telos.avg_fidelity - original.avg_fidelity,
            'original_final_f': original.final_fidelity,
            'telos_final_f': telos.final_fidelity,
            'original_trajectory': original.fidelity_trajectory,
            'telos_trajectory': telos.fidelity_trajectory,
            'governance_effective': telos.final_fidelity > original.final_fidelity
        }

    def get_branch_comparison(self, branch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full comparison data for a branch.

        Returns complete side-by-side data for visualization.
        """
        if branch_id not in self._branches:
            return None

        data = self._branches[branch_id]
        return {
            'branch_id': branch_id,
            'trigger_turn': data['trigger_turn'],
            'trigger_fidelity': data['trigger_fidelity'],
            'trigger_reason': data['trigger_reason'],
            'original': self._branch_to_dict(data['original']),
            'telos': self._branch_to_dict(data['telos']),
            'comparison': data['comparison'],
            'generated_at': data['generated_at']
        }

    def export_evidence(self, branch_id: str, format: str = 'json') -> Optional[str]:
        """
        Export branch evidence for grants/papers/auditing.

        Args:
            branch_id: Branch to export
            format: 'json' or 'markdown'

        Returns:
            Formatted evidence string
        """
        comparison = self.get_branch_comparison(branch_id)
        if not comparison:
            return None

        if format == 'json':
            return json.dumps(comparison, indent=2)

        elif format == 'markdown':
            return self._format_markdown_evidence(comparison)

        return None

    def _format_markdown_evidence(self, comparison: Dict[str, Any]) -> str:
        """Format evidence as markdown report."""
        orig = comparison['original']
        telos = comparison['telos']
        comp = comparison['comparison']

        md = f"""# TELOS Intervention Evidence

## Intervention Summary
- **Trigger Turn**: {comparison['trigger_turn']}
- **Trigger Fidelity**: {comparison['trigger_fidelity']:.3f}
- **Trigger Reason**: {comparison['trigger_reason']}
- **Generated**: {comparison['generated_at']}

## Results
- **Original Final Fidelity**: {comp['original_final_f']:.3f}
- **TELOS Final Fidelity**: {comp['telos_final_f']:.3f}
- **ΔF (Improvement)**: {comp['delta_f']:+.3f}
- **Average Improvement**: {comp['avg_improvement']:+.3f}
- **Governance Effective**: {'✅ YES' if comp['governance_effective'] else '❌ NO'}

## Turn-by-Turn Comparison

"""
        # Add each turn
        for i in range(len(orig['turns'])):
            orig_turn = orig['turns'][i]
            telos_turn = telos['turns'][i]

            md += f"""### Turn {orig_turn['turn_number']}

**User**: {orig_turn['user_input']}

#### Original Response
```
{orig_turn['assistant_response']}
```
Fidelity: {orig_turn['metrics']['telic_fidelity']:.3f}

#### TELOS Response
```
{telos_turn['assistant_response']}
```
Fidelity: {telos_turn['metrics']['telic_fidelity']:.3f}
{f"🛡️ **Intervention Applied**: {telos_turn['intervention_type']}" if telos_turn['intervention_applied'] else ""}

---

"""
        return md

    def _branch_to_dict(self, branch: CounterfactualBranch) -> Dict[str, Any]:
        """Convert branch to dictionary for serialization."""
        return {
            'branch_id': branch.branch_id,
            'branch_type': branch.branch_type,
            'trigger_turn': branch.trigger_turn,
            'turns': [
                {
                    'turn_number': t.turn_number,
                    'user_input': t.user_input,
                    'assistant_response': t.assistant_response,
                    'metrics': t.metrics,
                    'intervention_applied': t.intervention_applied,
                    'intervention_type': t.intervention_type,
                    'timestamp': t.timestamp
                }
                for t in branch.turns
            ],
            'final_fidelity': branch.final_fidelity,
            'avg_fidelity': branch.avg_fidelity,
            'drift_trajectory': branch.drift_trajectory,
            'fidelity_trajectory': branch.fidelity_trajectory,
            'metadata': branch.metadata
        }

    def get_all_branches(self) -> List[str]:
        """Get all branch IDs."""
        return list(self._branches.keys())

    def clear_branches(self) -> None:
        """Clear all stored branches (for new session)."""
        self._branches = {}
