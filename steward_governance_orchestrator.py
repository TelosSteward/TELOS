#!/usr/bin/env python3
"""
Steward PM - Governance Orchestrator
=====================================

Bridges external TELOS measurement with .claude_project.md intervention.

Role:
- Interprets fidelity measurements
- Decides intervention strategy
- Orchestrates when/how to update governance
- Provides intelligent recommendations vs crude thresholds

This is the "intelligence layer" that makes governance decisions smart.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime


class StewardGovernanceOrchestrator:
    """
    Steward PM as intelligent governance orchestrator

    Sits between measurement (external TELOS) and intervention (.claude_project.md)
    """

    def __init__(self):
        self.intervention_history = []
        self.drift_patterns = []

    def recommend_intervention(
        self,
        user_fidelity: float,
        ai_fidelity: float,
        overall_pass: bool,
        dominant_failure: Optional[str],
        fidelity_history: List[float],
        turn_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Intelligent intervention recommendation

        This is where Steward PM adds value beyond simple thresholds:
        - Pattern detection (trending vs isolated drift)
        - Context awareness (what user is trying to do)
        - Proportional response (gentle nudge vs hard stop)
        - Learning from history (similar drifts before?)
        """

        # Analyze drift pattern
        drift_pattern = self._analyze_drift_pattern(fidelity_history)

        # Context-aware decision
        intervention_needed = self._should_intervene(
            user_fidelity=user_fidelity,
            ai_fidelity=ai_fidelity,
            overall_pass=overall_pass,
            drift_pattern=drift_pattern,
            turn_context=turn_context
        )

        if not intervention_needed:
            return None

        # Generate smart intervention
        return self._generate_intervention(
            user_fidelity=user_fidelity,
            ai_fidelity=ai_fidelity,
            dominant_failure=dominant_failure,
            drift_pattern=drift_pattern,
            turn_context=turn_context
        )

    def _analyze_drift_pattern(self, fidelity_history: List[float]) -> Dict[str, Any]:
        """Detect patterns: trending, oscillating, isolated, critical"""

        if len(fidelity_history) < 3:
            return {'type': 'insufficient_data'}

        recent = fidelity_history[-3:]

        # Trending down?
        if recent[0] > recent[1] > recent[2]:
            delta = recent[0] - recent[2]
            return {
                'type': 'trending_down',
                'severity': 'high' if delta > 0.2 else 'medium',
                'delta': delta
            }

        # Oscillating?
        if len(set([f > 0.7 for f in recent])) > 1:
            return {'type': 'oscillating', 'severity': 'medium'}

        # Isolated dip?
        if recent[-1] < 0.7 and recent[-2] >= 0.7:
            return {'type': 'isolated_dip', 'severity': 'low'}

        # Critical sustained drift?
        if all(f < 0.6 for f in recent):
            return {'type': 'critical_sustained', 'severity': 'critical'}

        return {'type': 'stable', 'severity': 'none'}

    def _should_intervene(
        self,
        user_fidelity: float,
        ai_fidelity: float,
        overall_pass: bool,
        drift_pattern: Dict[str, Any],
        turn_context: Optional[Dict[str, Any]]
    ) -> bool:
        """Smart intervention decision based on context"""

        # Critical drift - always intervene
        if drift_pattern.get('type') == 'critical_sustained':
            return True

        # Trending down - intervene early
        if drift_pattern.get('type') == 'trending_down':
            return True

        # Both PAs failing - intervene
        if user_fidelity < 0.65 and ai_fidelity < 0.70:
            return True

        # User PA failing significantly
        if user_fidelity < 0.60:
            return True

        # Isolated dip - maybe let it recover
        if drift_pattern.get('type') == 'isolated_dip':
            # Check context - is user exploring? Let them
            if turn_context and turn_context.get('exploratory', False):
                return False
            return user_fidelity < 0.55  # Only intervene if really low

        return False

    def _generate_intervention(
        self,
        user_fidelity: float,
        ai_fidelity: float,
        dominant_failure: Optional[str],
        drift_pattern: Dict[str, Any],
        turn_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate intelligent intervention message"""

        severity = drift_pattern.get('severity', 'medium')
        pattern_type = drift_pattern.get('type', 'unknown')

        # Determine intervention level
        if severity == 'critical':
            level = 'critical'
            urgency = '🚨 CRITICAL'
        elif severity == 'high':
            level = 'strong'
            urgency = '🚨 DRIFT DETECTED'
        elif severity == 'medium':
            level = 'moderate'
            urgency = '⚠️  DRIFT WARNING'
        else:
            level = 'gentle'
            urgency = '💡 SUGGESTION'

        # Tailor message based on which PA failed
        if dominant_failure == 'user':
            focus = "session purpose"
            suggestion = "Refocus on critical path (grants, validation, Observatory)"
        elif dominant_failure == 'ai':
            focus = "AI role constraints"
            suggestion = "Adjust tone/approach to match intended helping style"
        elif dominant_failure == 'both':
            focus = "both User and AI governance"
            suggestion = "STOP and explicitly refocus conversation"
        else:
            focus = "general alignment"
            suggestion = "Stay on critical path"

        # Pattern-specific guidance
        if pattern_type == 'trending_down':
            pattern_note = f"\n📉 **Pattern:** Fidelity trending down over last 3 turns (Δ={drift_pattern.get('delta', 0):.2f})"
        elif pattern_type == 'oscillating':
            pattern_note = "\n📊 **Pattern:** Fidelity oscillating - conversation lacks focus"
        elif pattern_type == 'critical_sustained':
            pattern_note = "\n🚨 **Pattern:** SUSTAINED DRIFT - Last 3 turns all below threshold"
        else:
            pattern_note = ""

        return {
            'level': level,
            'urgency': urgency,
            'focus': focus,
            'user_fidelity': user_fidelity,
            'ai_fidelity': ai_fidelity,
            'suggestion': suggestion,
            'pattern_note': pattern_note,
            'timestamp': datetime.now().isoformat()
        }

    def format_intervention_for_claude_project(
        self,
        intervention: Dict[str, Any],
        session_pa: Dict[str, Any]
    ) -> str:
        """Format intervention message for .claude_project.md"""

        return f"""
{intervention['urgency']} - Governance Intervention

**Turn Metrics:**
- User PA Fidelity: {intervention['user_fidelity']:.3f}
- AI PA Fidelity: {intervention['ai_fidelity']:.3f}
- Focus Area: {intervention['focus']}

{intervention.get('pattern_note', '')}

**Steward PM Recommendation:**
{intervention['suggestion']}

**Session Purpose (User PA):**
{session_pa['purpose'][0][:150]}...

**Action Required:**
{'IMMEDIATE refocus needed' if intervention['level'] == 'critical' else 'Adjust next response to increase fidelity'}

---
*Intervention applied: {intervention['timestamp']}*
*Orchestrated by: Steward PM Governance Orchestrator*
"""


# Convenience function for integration
def create_steward_orchestrator() -> StewardGovernanceOrchestrator:
    """Factory function for creating orchestrator"""
    return StewardGovernanceOrchestrator()


if __name__ == "__main__":
    # Demo
    steward = StewardGovernanceOrchestrator()

    # Simulate drift scenario
    fidelity_history = [0.92, 0.87, 0.73, 0.61]  # Trending down

    recommendation = steward.recommend_intervention(
        user_fidelity=0.61,
        ai_fidelity=0.78,
        overall_pass=False,
        dominant_failure='user',
        fidelity_history=fidelity_history,
        turn_context={'topic': 'UI animations'}
    )

    if recommendation:
        print("\n🤖 Steward PM Intervention Recommendation:")
        print("="*60)
        for key, value in recommendation.items():
            print(f"{key}: {value}")
        print("="*60)

        # Format for .claude_project.md
        session_pa = {
            'purpose': ["Guide TELOS toward Feb 2026 institutional deployment"]
        }

        message = steward.format_intervention_for_claude_project(
            recommendation, session_pa
        )

        print("\n📝 Formatted for .claude_project.md:")
        print(message)
