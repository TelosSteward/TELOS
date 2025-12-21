"""
Live Interceptor for TELOSCOPE
==============================

Wraps LLM client to intercept all API calls and trigger counterfactual
branching when drift is detected. Integrates with UnifiedSteward for
governance and updates WebSessionManager for real-time UI.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import numpy as np
import threading


class LiveInterceptor:
    """
    Intercepts LLM calls to detect drift and trigger counterfactuals.

    This class wraps the LLM client and monitors every response for drift.
    When drift is detected, it triggers counterfactual branch generation
    and updates the web UI in real-time.
    """

    def __init__(
        self,
        llm_client: Any,
        embedding_provider: Any,
        steward: Any,
        session_manager: Any,
        branch_manager: Any,
        web_session_manager: Optional[Any] = None,
        drift_threshold: float = 0.76,  # Goldilocks: Aligned threshold
        enable_counterfactuals: bool = True
    ):
        """
        Initialize live interceptor.

        Args:
            llm_client: Base LLM client (e.g., TelosMistralClient)
            embedding_provider: Embedding provider for drift calculation
            steward: UnifiedGovernanceSteward for drift detection
            session_manager: SessionStateManager for state snapshots
            branch_manager: CounterfactualBranchManager for branch generation
            web_session_manager: WebSessionManager for UI updates (optional)
            drift_threshold: Fidelity threshold for drift detection (default: 0.76 Goldilocks Aligned)
            enable_counterfactuals: Enable counterfactual generation (default: True)
        """
        self.llm = llm_client
        self.embeddings = embedding_provider
        self.steward = steward
        self.session_manager = session_manager
        self.branch_manager = branch_manager
        self.web_manager = web_session_manager

        self.drift_threshold = drift_threshold
        self.enable_counterfactuals = enable_counterfactuals

        # State tracking
        self._turn_number = 0
        self._conversation_history: List[Dict[str, str]] = []
        self._ui_callbacks: List[Callable] = []

        # Metrics
        self._live_metrics = {
            'current_fidelity': 1.0,
            'current_distance': 0.0,
            'basin_status': True,
            'error_signal': 0.0,
            'last_update': None
        }

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Intercept LLM generation call with ACTIVE MITIGATION.

        This method uses the steward's active mitigation layer to:
        1. Check salience (is attractor prominent in context?)
        2. Inject reinforcement if needed
        3. Generate response through governed context
        4. Check coupling (did response drift?)
        5. Regenerate if decoupled
        6. Calculate drift metrics
        7. Trigger counterfactuals if drift detected
        8. Update session state and UI

        Args:
            messages: Conversation messages for LLM
            **kwargs: Additional arguments for LLM (ignored - steward controls generation)

        Returns:
            Final governed response
        """
        # Extract user input from messages
        if messages:
            last_user_msg = messages[-1]
            if last_user_msg.get('role') == 'user':
                user_input = last_user_msg['content']
            else:
                user_input = "[No user input]"
        else:
            user_input = "[Empty conversation]"

        # Build conversation context (exclude the last user message since generate_governed_response expects it separately)
        conversation_context = messages[:-1] if messages else []

        # Phase 4: Initialize both response variables
        native_response = None
        telos_response = None

        # ACTIVE GOVERNANCE: Generate through steward's mitigation layer
        if self.steward:
            # Use steward's generate_governed_response for ACTIVE mitigation
            result = self.steward.generate_governed_response(user_input, conversation_context)
            final_response = result['governed_response']
            metrics = {
                'telic_fidelity': result['fidelity'],
                'salience': result.get('salience', 1.0),
                'error_signal': 1.0 - result['fidelity'],
                'primacy_basin_membership': result['fidelity'] >= 0.76,  # Goldilocks: Aligned threshold
                'lyapunov_value': (1.0 - result['fidelity']) ** 2
            }
            intervention_applied = result['intervention_applied']

            # Store detailed intervention info for dashboard display
            intervention_details = {
                'type': result.get('intervention_type', 'none'),
                'fidelity': result['fidelity'],
                'salience_after': result.get('salience', 1.0),
                'intervention_applied': intervention_applied
            }

            # Get latest intervention record for additional details
            if hasattr(self.steward, 'llm_wrapper') and self.steward.llm_wrapper.interventions:
                latest = self.steward.llm_wrapper.interventions[-1]
                intervention_details['fidelity_original'] = latest.fidelity_original
                intervention_details['fidelity_governed'] = latest.fidelity_governed
                intervention_details['salience_before'] = latest.salience_before

                # Phase 4: Extract both responses for governance toggle
                native_response = latest.original_response
                telos_response = latest.governed_response

                # GovernanceLens data: Store both responses for side-by-side comparison
                intervention_details['original_response'] = native_response
                intervention_details['governed_response'] = telos_response

                # Calculate improvement if regeneration occurred
                if latest.fidelity_original is not None and latest.fidelity_governed is not None:
                    intervention_details['fidelity_improvement'] = (
                        latest.fidelity_governed - latest.fidelity_original
                    )
            else:
                # No intervention record: use final response for both
                native_response = final_response
                telos_response = final_response
        else:
            # Fallback: direct LLM call without governance
            final_response = self.llm.generate(messages=messages, **kwargs)
            native_response = final_response
            telos_response = final_response
            metrics = self._calculate_metrics(user_input, final_response)
            intervention_applied = False
            intervention_details = None

        # Update conversation history
        self._conversation_history.append({"role": "user", "content": user_input})
        self._conversation_history.append({"role": "assistant", "content": final_response})

        # Calculate embeddings
        user_emb = self.embeddings.encode([user_input])
        response_emb = self.embeddings.encode([final_response])

        # Get attractor center
        if self.steward and hasattr(self.steward, 'spc_engine'):
            attractor_center = self.steward.spc_engine.attractor_center
        else:
            # Fallback: use zero vector
            attractor_center = np.zeros_like(response_emb)

        # Phase 4: Save turn snapshot with both Native and TELOS responses
        turn_snapshot = self.session_manager.save_turn_snapshot(
            turn_number=self._turn_number,
            user_input=user_input,
            native_response=native_response,
            telos_response=telos_response,
            user_embedding=user_emb,
            response_embedding=response_emb,
            attractor_center=attractor_center,
            metrics=metrics,
            conversation_history=self._conversation_history.copy(),
            attractor_config=self._get_attractor_config(),
            metadata={
                'intervention_applied': intervention_applied,
                'intervention_details': intervention_details  # Intervention details for dashboard
            }
        )

        # Update live metrics
        self._live_metrics = {
            'current_fidelity': metrics.get('telic_fidelity', 1.0),
            'current_distance': metrics.get('drift_distance', 0.0),
            'basin_status': metrics.get('primacy_basin_membership', True),
            'error_signal': metrics.get('error_signal', 0.0),
            'last_update': datetime.now().isoformat()
        }

        # Check for drift and trigger counterfactual
        if self.enable_counterfactuals and self._check_drift(metrics):
            self._trigger_counterfactual_async(turn_snapshot, metrics)

        # Increment turn counter
        self._turn_number += 1

        # Notify UI callbacks
        self._trigger_callbacks('turn_completed', {
            'turn_number': self._turn_number - 1,
            'metrics': metrics,
            'intervention_applied': intervention_applied
        })

        return final_response

    def _calculate_metrics(
        self,
        user_input: str,
        response: str
    ) -> Dict[str, float]:
        """
        Calculate governance metrics without steward.

        Args:
            user_input: User's input
            response: Assistant's response

        Returns:
            Metrics dict
        """
        # Simplified metrics calculation
        user_emb = self.embeddings.encode([user_input])
        response_emb = self.embeddings.encode([response])

        # Calculate drift distance (simplified)
        distance = np.linalg.norm(response_emb - user_emb)

        # Estimate fidelity (placeholder - would use actual attractor comparison)
        fidelity = max(0.0, min(1.0, 1.0 - (distance / 5.0)))

        return {
            'telic_fidelity': fidelity,
            'drift_distance': distance,
            'error_signal': 1.0 - fidelity,
            'primacy_basin_membership': distance < 2.0,
            'lyapunov_value': distance ** 2
        }

    def _check_drift(self, metrics: Dict[str, float]) -> bool:
        """
        Check if drift is detected based on metrics.

        Args:
            metrics: Governance metrics

        Returns:
            True if drift detected, False otherwise
        """
        fidelity = metrics.get('telic_fidelity', 1.0)
        basin_status = metrics.get('primacy_basin_membership', True)

        # Drift detected if fidelity below threshold or outside basin
        return fidelity < self.drift_threshold or not basin_status

    def _trigger_counterfactual_async(
        self,
        turn_snapshot: Any,
        metrics: Dict[str, float]
    ) -> None:
        """
        Trigger counterfactual generation in background thread.

        Args:
            turn_snapshot: TurnSnapshot object
            metrics: Current metrics
        """
        # Determine trigger reason
        reasons = []
        if metrics['telic_fidelity'] < 0.5:
            reasons.append(f"Critical fidelity (F={metrics['telic_fidelity']:.3f})")
        elif metrics['telic_fidelity'] < self.drift_threshold:
            reasons.append(f"Low fidelity (F={metrics['telic_fidelity']:.3f})")

        if not metrics.get('primacy_basin_membership', True):
            reasons.append(f"Outside basin (d={metrics.get('drift_distance', 0):.3f})")

        trigger_reason = "; ".join(reasons) if reasons else "Potential drift"

        # Reconstruct state for branching
        turn_state = self.session_manager.reconstruct_state_at_turn(turn_snapshot.turn_number)

        if not turn_state:
            return

        # Launch branch generation in background thread
        def generate_branches():
            try:
                branch_id = self.branch_manager.trigger_counterfactual(
                    turn_state=turn_state,
                    trigger_reason=trigger_reason
                )

                # Notify UI
                self._trigger_callbacks('counterfactual_triggered', {
                    'branch_id': branch_id,
                    'turn_number': turn_snapshot.turn_number,
                    'reason': trigger_reason
                })
            except Exception as e:
                print(f"Error generating counterfactual branches: {e}")

        # Start thread (non-blocking)
        thread = threading.Thread(target=generate_branches, daemon=True)
        thread.start()

    def _get_attractor_config(self) -> Dict[str, Any]:
        """
        Get current attractor configuration.

        Returns:
            Attractor config dict
        """
        if self.steward and hasattr(self.steward, 'attractor'):
            attractor = self.steward.attractor
            return {
                'purpose': attractor.purpose if hasattr(attractor, 'purpose') else [],
                'scope': attractor.scope if hasattr(attractor, 'scope') else [],
                'boundaries': attractor.boundaries if hasattr(attractor, 'boundaries') else []
            }
        return {}

    def get_live_metrics(self) -> Dict[str, Any]:
        """
        Get current live metrics for UI display.

        Returns:
            Live metrics dict
        """
        return self._live_metrics.copy()

    def register_ui_callback(self, callback: Callable[[str, Any], None]) -> None:
        """
        Register callback for UI updates.

        Callback signature: callback(event_name: str, data: Any) -> None

        Events: 'turn_completed', 'counterfactual_triggered'

        Args:
            callback: Callback function
        """
        self._ui_callbacks.append(callback)

    def _trigger_callbacks(self, event_name: str, data: Any) -> None:
        """
        Trigger all registered callbacks.

        Args:
            event_name: Event name
            data: Event data
        """
        for callback in self._ui_callbacks:
            try:
                callback(event_name, data)
            except Exception as e:
                print(f"Callback error for {event_name}: {e}")

    def reset_session(self) -> None:
        """Reset interceptor state for new session."""
        self._turn_number = 0
        self._conversation_history = []
        self._live_metrics = {
            'current_fidelity': 1.0,
            'current_distance': 0.0,
            'basin_status': True,
            'error_signal': 0.0,
            'last_update': None
        }

    def get_turn_count(self) -> int:
        """Get total number of turns processed."""
        return self._turn_number

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get complete conversation history."""
        return self._conversation_history.copy()
