"""
Counterfactual Branch Simulator - AI-to-AI Conversation Extension
=================================================================

Simulates counterfactual conversations by generating both user messages
and assistant responses. Useful when historical data is limited or you
want to project longer-term drift patterns.

Key difference from CounterfactualBranchManager:
- BranchManager: Uses REAL historical user messages
- Simulator: GENERATES user messages via AI

Use cases:
- Extend conversations beyond available data
- Project long-term drift patterns
- Generate evidence for hypothetical scenarios
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import time
from datetime import datetime


@dataclass
class SimulatedTurn:
    """Single turn in simulated conversation."""
    turn_number: int
    user_message: str
    assistant_response: str
    fidelity: float
    salience: float
    is_simulated_user: bool  # True if user message was AI-generated
    timestamp: float


@dataclass
class SimulatedBranch:
    """Complete simulated conversation branch."""
    branch_id: str
    branch_type: str  # "original" or "telos"
    trigger_turn: int
    trigger_fidelity: float
    turns: List[SimulatedTurn]
    final_fidelity: float
    avg_fidelity: float
    total_simulated_turns: int
    governance_applied: bool


@dataclass
class SimulationEvidence:
    """Complete evidence package from simulation."""
    simulation_id: str
    trigger_turn: int
    trigger_fidelity: float
    trigger_reason: str
    original_branch: SimulatedBranch
    telos_branch: SimulatedBranch
    delta_f: float
    improvement_demonstrated: bool
    timestamp: float


class CounterfactualBranchSimulator:
    """
    Simulates counterfactual conversations with AI-generated user messages.

    Workflow:
    1. Start from drift point with context history
    2. Generate plausible user messages (AI-to-AI)
    3. Generate responses for each branch (original vs TELOS)
    4. Calculate fidelity for each turn
    5. Compare branches side-by-side
    """

    def __init__(
        self,
        llm_client: Any,
        embedding_provider: Any,
        steward: Any,
        simulation_turns: int = 5
    ):
        """
        Initialize simulator.

        Args:
            llm_client: LLM for generating messages and responses
            embedding_provider: For fidelity calculations
            steward: For TELOS governance (optional)
            simulation_turns: Number of turns to simulate (default: 5)
        """
        self.llm = llm_client
        self.embeddings = embedding_provider
        self.steward = steward
        self.simulation_turns = simulation_turns

        # Track simulations
        self.simulations: Dict[str, SimulationEvidence] = {}

    def simulate_counterfactual(
        self,
        trigger_turn: int,
        trigger_fidelity: float,
        trigger_reason: str,
        conversation_history: List[Dict[str, str]],
        attractor_center: np.ndarray,
        distance_scale: float = 2.0,
        topic_hint: Optional[str] = None
    ) -> str:
        """
        Simulate counterfactual branches with AI-generated user messages.

        Args:
            trigger_turn: Turn number where drift detected
            trigger_fidelity: Fidelity at trigger point
            trigger_reason: Why simulation was triggered
            conversation_history: Context up to trigger point
            attractor_center: For fidelity calculation
            distance_scale: Distance-to-fidelity scaling
            topic_hint: Optional hint for user message generation

        Returns:
            Simulation ID for retrieval
        """
        simulation_id = f"sim_{trigger_turn}_{datetime.now().strftime('%H%M%S')}"

        # Generate both branches
        original_branch = self._simulate_branch(
            branch_type="original",
            trigger_turn=trigger_turn,
            conversation_history=conversation_history.copy(),
            attractor_center=attractor_center,
            distance_scale=distance_scale,
            apply_governance=False,
            topic_hint=topic_hint
        )

        telos_branch = self._simulate_branch(
            branch_type="telos",
            trigger_turn=trigger_turn,
            conversation_history=conversation_history.copy(),
            attractor_center=attractor_center,
            distance_scale=distance_scale,
            apply_governance=True,
            topic_hint=topic_hint
        )

        # Calculate comparison
        delta_f = telos_branch.final_fidelity - original_branch.final_fidelity
        improvement_demonstrated = delta_f > 0.0

        # Create evidence package
        evidence = SimulationEvidence(
            simulation_id=simulation_id,
            trigger_turn=trigger_turn,
            trigger_fidelity=trigger_fidelity,
            trigger_reason=trigger_reason,
            original_branch=original_branch,
            telos_branch=telos_branch,
            delta_f=delta_f,
            improvement_demonstrated=improvement_demonstrated,
            timestamp=time.time()
        )

        self.simulations[simulation_id] = evidence

        return simulation_id

    def _simulate_branch(
        self,
        branch_type: str,
        trigger_turn: int,
        conversation_history: List[Dict[str, str]],
        attractor_center: np.ndarray,
        distance_scale: float,
        apply_governance: bool,
        topic_hint: Optional[str]
    ) -> SimulatedBranch:
        """
        Simulate single conversation branch.

        Args:
            branch_type: "original" or "telos"
            trigger_turn: Starting turn number
            conversation_history: Context messages
            attractor_center: For fidelity
            distance_scale: For fidelity calculation
            apply_governance: Whether to apply TELOS governance
            topic_hint: Optional topic for user message generation

        Returns:
            SimulatedBranch with all turns
        """
        branch_id = f"{branch_type}_{trigger_turn}"
        turns = []
        fidelities = []

        current_history = conversation_history.copy()

        for i in range(self.simulation_turns):
            turn_num = trigger_turn + i + 1

            # Generate user message (AI-to-AI simulation)
            user_message = self._generate_user_message(
                current_history,
                topic_hint if i == 0 else None
            )

            # Add user message to history
            current_history.append({"role": "user", "content": user_message})

            # Generate assistant response
            if apply_governance and self.steward:
                # TELOS branch: Use governed generation
                response = self._generate_governed_response(
                    user_message,
                    current_history[:-1]  # Exclude just-added user message
                )
            else:
                # Original branch: Direct LLM generation
                response = self._generate_direct_response(current_history)

            # Add response to history
            current_history.append({"role": "assistant", "content": response})

            # Calculate fidelity
            response_emb = self.embeddings.encode([response])[0]
            distance = float(np.linalg.norm(response_emb - attractor_center))
            fidelity = max(0.0, min(1.0, 1.0 - (distance / distance_scale)))
            fidelities.append(fidelity)

            # Calculate salience (simplified)
            salience = self._calculate_salience(current_history, attractor_center)

            # Record turn
            turns.append(SimulatedTurn(
                turn_number=turn_num,
                user_message=user_message,
                assistant_response=response,
                fidelity=fidelity,
                salience=salience,
                is_simulated_user=True,
                timestamp=time.time()
            ))

        # Create branch
        return SimulatedBranch(
            branch_id=branch_id,
            branch_type=branch_type,
            trigger_turn=trigger_turn,
            trigger_fidelity=fidelities[0] if fidelities else 0.0,
            turns=turns,
            final_fidelity=fidelities[-1] if fidelities else 0.0,
            avg_fidelity=sum(fidelities) / len(fidelities) if fidelities else 0.0,
            total_simulated_turns=len(turns),
            governance_applied=apply_governance
        )

    def _generate_user_message(
        self,
        conversation_history: List[Dict[str, str]],
        topic_hint: Optional[str] = None
    ) -> str:
        """
        Generate plausible user message via AI.

        Args:
            conversation_history: Recent conversation context
            topic_hint: Optional topic to guide generation

        Returns:
            Generated user message
        """
        # Build prompt for user message generation
        recent_context = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history

        context_summary = "\n".join([
            f"{msg['role'].title()}: {msg['content'][:100]}..."
            for msg in recent_context
        ])

        if topic_hint:
            prompt = f"""Based on this conversation, generate a plausible next user question or statement.

The user might drift toward: {topic_hint}

Recent conversation:
{context_summary}

Generate a natural user message (just the message, no labels):"""
        else:
            prompt = f"""Based on this conversation, generate a plausible next user question or statement.

Recent conversation:
{context_summary}

Generate a natural user message that continues the discussion (just the message, no labels):"""

        try:
            user_message = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            return user_message.strip()
        except Exception as e:
            print(f"⚠️ User message generation failed: {e}")
            return "Can you tell me more about that?"

    def _generate_direct_response(
        self,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate response directly from LLM (no governance).

        Args:
            conversation_history: Full conversation including latest user message

        Returns:
            Direct LLM response
        """
        try:
            response = self.llm.generate(
                messages=conversation_history,
                temperature=0.7,
                max_tokens=300
            )
            return response
        except Exception as e:
            print(f"⚠️ Direct response generation failed: {e}")
            return "[Error generating response]"

    def _generate_governed_response(
        self,
        user_input: str,
        conversation_context: List[Dict[str, str]]
    ) -> str:
        """
        Generate response through TELOS governance.

        Args:
            user_input: User's message
            conversation_context: Context before user message

        Returns:
            Governed response
        """
        if not self.steward:
            return self._generate_direct_response(
                conversation_context + [{"role": "user", "content": user_input}]
            )

        try:
            result = self.steward.generate_governed_response(
                user_input,
                conversation_context
            )
            return result['governed_response']
        except Exception as e:
            print(f"⚠️ Governed response generation failed: {e}")
            # Fallback to direct generation
            return self._generate_direct_response(
                conversation_context + [{"role": "user", "content": user_input}]
            )

    def _calculate_salience(
        self,
        conversation_history: List[Dict[str, str]],
        attractor_center: np.ndarray
    ) -> float:
        """
        Calculate attractor salience in conversation.

        Args:
            conversation_history: Recent messages
            attractor_center: Attractor embedding

        Returns:
            Salience score 0-1
        """
        # Get recent context
        recent = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        context_text = " ".join([msg['content'] for msg in recent])

        if not context_text.strip():
            return 0.5

        # Embed and compare
        context_emb = self.embeddings.encode([context_text])[0]
        similarity = float(np.dot(context_emb, attractor_center) /
                          (np.linalg.norm(context_emb) * np.linalg.norm(attractor_center)))

        # Convert to 0-1
        return (similarity + 1.0) / 2.0

    def get_simulation(self, simulation_id: str) -> Optional[SimulationEvidence]:
        """Get simulation evidence by ID."""
        return self.simulations.get(simulation_id)

    def get_comparison(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get side-by-side comparison for simulation.

        Returns:
            Dict with original, telos, and comparison data
        """
        evidence = self.simulations.get(simulation_id)
        if not evidence:
            return None

        return {
            'simulation_id': simulation_id,
            'trigger_turn': evidence.trigger_turn,
            'trigger_fidelity': evidence.trigger_fidelity,
            'trigger_reason': evidence.trigger_reason,
            'original': {
                'branch_id': evidence.original_branch.branch_id,
                'turns': [asdict(t) for t in evidence.original_branch.turns],
                'final_fidelity': evidence.original_branch.final_fidelity,
                'avg_fidelity': evidence.original_branch.avg_fidelity
            },
            'telos': {
                'branch_id': evidence.telos_branch.branch_id,
                'turns': [asdict(t) for t in evidence.telos_branch.turns],
                'final_fidelity': evidence.telos_branch.final_fidelity,
                'avg_fidelity': evidence.telos_branch.avg_fidelity
            },
            'comparison': {
                'delta_f': evidence.delta_f,
                'improvement': evidence.improvement_demonstrated,
                'original_trajectory': [t.fidelity for t in evidence.original_branch.turns],
                'telos_trajectory': [t.fidelity for t in evidence.telos_branch.turns]
            },
            'timestamp': evidence.timestamp
        }

    def export_evidence(
        self,
        simulation_id: str,
        format: str = 'json'
    ) -> Optional[str]:
        """
        Export simulation evidence.

        Args:
            simulation_id: Simulation to export
            format: 'json' or 'markdown'

        Returns:
            Formatted evidence string
        """
        comparison = self.get_comparison(simulation_id)
        if not comparison:
            return None

        if format == 'json':
            import json
            return json.dumps(comparison, indent=2)

        elif format == 'markdown':
            return self._format_markdown_evidence(comparison)

        return None

    def _format_markdown_evidence(self, comparison: Dict[str, Any]) -> str:
        """Format evidence as markdown report."""
        md = f"""# TELOS Counterfactual Simulation Evidence

## Simulation Summary
- **Simulation ID**: {comparison['simulation_id']}
- **Trigger Turn**: {comparison['trigger_turn']}
- **Trigger Fidelity**: {comparison['trigger_fidelity']:.3f}
- **Trigger Reason**: {comparison['trigger_reason']}
- **ΔF (Improvement)**: {comparison['comparison']['delta_f']:+.3f}
- **Governance Effective**: {"✅ Yes" if comparison['comparison']['improvement'] else "❌ No"}

## Fidelity Trajectories

**Original Branch**: {' → '.join([f"{f:.3f}" for f in comparison['comparison']['original_trajectory']])}

**TELOS Branch**: {' → '.join([f"{f:.3f}" for f in comparison['comparison']['telos_trajectory']])}

## Turn-by-Turn Comparison

"""

        original_turns = comparison['original']['turns']
        telos_turns = comparison['telos']['turns']

        for i in range(len(original_turns)):
            orig = original_turns[i]
            telos = telos_turns[i]

            md += f"""### Turn {orig['turn_number']}

**User**: {orig['user_message']}

**Original Response** (F={orig['fidelity']:.3f}):
> {orig['assistant_response']}

**TELOS Response** (F={telos['fidelity']:.3f}):
> {telos['assistant_response']}

---

"""

        return md
