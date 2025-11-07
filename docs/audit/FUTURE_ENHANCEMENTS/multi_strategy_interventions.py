#!/usr/bin/env python3
"""
Multi-Strategy Intervention System - FUTURE ENHANCEMENT

**Status:** FUTURE implementation (post-validation studies)
**Timeline:** After baseline intervention system validated
**Purpose:** Expand beyond reminder/regeneration to context-aware intervention strategies

Current TELOS uses primarily two intervention strategies:
1. Reminder interventions (inject PA reminder into context)
2. Regeneration (regenerate response with stronger PA emphasis)

This system adds 5+ additional strategies with intelligent selection based on:
- Deviation type (semantic vs behavioral)
- Deviation magnitude (minor vs severe)
- Session history (intervention effectiveness)
- User context (conversation flow)

IMPLEMENTATION PHASES:
- Phase 1 (NOW): Document signatures, basic structure
- Phase 2 (FUTURE): Full multi-strategy system with learning
- Phase 3 (RESEARCH): Institutional validation studies

Dependencies:
    numpy>=1.24.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InterventionStrategy(Enum):
    """Available intervention strategies."""
    REMINDER = "reminder"  # Current: Inject PA reminder
    REGENERATION = "regeneration"  # Current: Regenerate with emphasis
    REFRAMING = "reframing"  # NEW: Reframe response toward PA
    SCAFFOLDING = "scaffolding"  # NEW: Add structure to response
    CONSTRAINT = "constraint"  # NEW: Apply hard constraints
    FALLBACK = "fallback"  # NEW: Use curated safe response
    ESCALATION = "escalation"  # NEW: Multi-turn intervention sequence


class DeviationType(Enum):
    """Types of PA deviation."""
    SEMANTIC = "semantic"  # Meaning/content deviation
    BEHAVIORAL = "behavioral"  # Tone/style deviation
    STRUCTURAL = "structural"  # Format/organization deviation
    FACTUAL = "factual"  # Incorrect information
    SAFETY = "safety"  # Safety boundary violation


@dataclass
class InterventionContext:
    """Context for intervention decision-making."""
    fidelity_user: float
    fidelity_ai: float
    deviation_magnitude: float
    deviation_type: DeviationType
    turn_count: int
    recent_intervention_count: int
    conversation_topic: str
    user_intent: str
    response_text: str


@dataclass
class InterventionPlan:
    """Plan for executing intervention."""
    strategy: InterventionStrategy
    parameters: Dict
    expected_effectiveness: float
    reasoning: str


class ReminderInterventionHandler:
    """
    Strategy: Reminder Intervention (Current TELOS implementation)

    Injects PA reminder into conversation context to nudge response alignment.
    Effective for minor deviations where LLM just needs refocusing.
    """

    @staticmethod
    def create_plan(context: InterventionContext) -> InterventionPlan:
        """Create reminder intervention plan."""
        # Extract PA reminder text (in real system, from PA vector)
        reminder_text = "Remember to maintain focus on the user's stated purpose."

        return InterventionPlan(
            strategy=InterventionStrategy.REMINDER,
            parameters={
                'reminder_text': reminder_text,
                'injection_position': 'context',  # Where to inject reminder
                'emphasis_level': 'medium'
            },
            expected_effectiveness=0.7 if context.deviation_magnitude < 0.15 else 0.4,
            reasoning="Minor deviation - reminder should suffice"
        )

    @staticmethod
    def execute(plan: InterventionPlan, context: InterventionContext) -> str:
        """Execute reminder intervention."""
        reminder = plan.parameters['reminder_text']
        # In real system, this would modify conversation context
        return f"[INTERVENTION: Reminder] {reminder}"


class RegenerationInterventionHandler:
    """
    Strategy: Regeneration Intervention (Current TELOS implementation)

    Regenerates response with stronger PA emphasis. Used for moderate deviations
    where reminder insufficient.
    """

    @staticmethod
    def create_plan(context: InterventionContext) -> InterventionPlan:
        """Create regeneration intervention plan."""
        return InterventionPlan(
            strategy=InterventionStrategy.REGENERATION,
            parameters={
                'temperature': 0.7,  # Lower temperature for more focused response
                'pa_weight': 2.0,  # Double PA emphasis
                'max_attempts': 2
            },
            expected_effectiveness=0.85 if context.deviation_magnitude < 0.25 else 0.6,
            reasoning="Moderate deviation - regeneration with PA emphasis"
        )

    @staticmethod
    def execute(plan: InterventionPlan, context: InterventionContext) -> str:
        """Execute regeneration intervention."""
        # In real system, this would trigger LLM regeneration
        return "[INTERVENTION: Regenerate with PA emphasis]"


class ReframingInterventionHandler:
    """
    Strategy: Reframing Intervention (NEW)

    Takes existing response and reframes it toward PA without full regeneration.
    More efficient than regeneration, preserves useful content while fixing alignment.

    Use cases:
    - Response has good content but wrong framing
    - Need to preserve specific facts/details
    - Want faster intervention than full regeneration
    """

    @staticmethod
    def create_plan(context: InterventionContext) -> InterventionPlan:
        """Create reframing intervention plan."""
        return InterventionPlan(
            strategy=InterventionStrategy.REFRAMING,
            parameters={
                'preserve_content': True,
                'reframe_directive': "Reframe this response to better align with user's purpose",
                'edit_scope': 'partial'  # Only modify misaligned sections
            },
            expected_effectiveness=0.75,
            reasoning="Good content with wrong framing - reframe preserving details"
        )

    @staticmethod
    def execute(plan: InterventionPlan, context: InterventionContext) -> str:
        """Execute reframing intervention."""
        original = context.response_text
        directive = plan.parameters['reframe_directive']

        # In real system, this would use LLM to reframe response
        return f"[INTERVENTION: Reframe] {directive}: {original[:100]}..."


class ScaffoldingInterventionHandler:
    """
    Strategy: Scaffolding Intervention (NEW)

    Adds structure/organization to response to guide it toward PA.
    Useful when response lacks organization or clear progression.

    Use cases:
    - Rambling responses that need structure
    - Missing key components expected by PA
    - Need to organize information toward purpose
    """

    @staticmethod
    def create_plan(context: InterventionContext) -> InterventionPlan:
        """Create scaffolding intervention plan."""
        return InterventionPlan(
            strategy=InterventionStrategy.SCAFFOLDING,
            parameters={
                'add_structure': True,
                'structure_type': 'outline',  # outline, steps, sections
                'guide_toward_pa': True
            },
            expected_effectiveness=0.70,
            reasoning="Response lacks structure - add scaffolding to guide alignment"
        )

    @staticmethod
    def execute(plan: InterventionPlan, context: InterventionContext) -> str:
        """Execute scaffolding intervention."""
        structure_type = plan.parameters['structure_type']
        # In real system, this would add structural elements to response
        return f"[INTERVENTION: Add {structure_type} structure]"


class ConstraintInterventionHandler:
    """
    Strategy: Constraint Intervention (NEW)

    Applies hard constraints to response generation to enforce PA alignment.
    Most aggressive intervention - used when soft interventions fail.

    Use cases:
    - Repeated deviations after multiple interventions
    - Critical alignment requirements
    - Safety-critical scenarios
    """

    @staticmethod
    def create_plan(context: InterventionContext) -> InterventionPlan:
        """Create constraint intervention plan."""
        return InterventionPlan(
            strategy=InterventionStrategy.CONSTRAINT,
            parameters={
                'constraint_type': 'hard',
                'allowed_topics': ['topic1', 'topic2'],  # Derived from PA
                'forbidden_patterns': [],
                'enforce_format': True
            },
            expected_effectiveness=0.90,
            reasoning="Severe/repeated deviation - apply hard constraints"
        )

    @staticmethod
    def execute(plan: InterventionPlan, context: InterventionContext) -> str:
        """Execute constraint intervention."""
        constraint_type = plan.parameters['constraint_type']
        return f"[INTERVENTION: Apply {constraint_type} constraints]"


class FallbackInterventionHandler:
    """
    Strategy: Fallback Intervention (NEW)

    Uses pre-curated safe response when all other interventions fail.
    Ensures graceful degradation rather than catastrophic failure.

    Use cases:
    - All other interventions failed
    - Safety-critical context
    - Need guaranteed alignment
    """

    @staticmethod
    def create_plan(context: InterventionContext) -> InterventionPlan:
        """Create fallback intervention plan."""
        return InterventionPlan(
            strategy=InterventionStrategy.FALLBACK,
            parameters={
                'fallback_template': 'safe_acknowledgment',
                'preserve_context': True,
                'explain_limitation': True
            },
            expected_effectiveness=1.0,  # Always produces aligned response (but minimal)
            reasoning="All other strategies failed - use safe fallback"
        )

    @staticmethod
    def execute(plan: InterventionPlan, context: InterventionContext) -> str:
        """Execute fallback intervention."""
        # In real system, this would use curated response template
        return "[INTERVENTION: Safe fallback response] I need to ensure my response aligns with your purpose. Could you help clarify what you're looking for?"


class EscalationInterventionHandler:
    """
    Strategy: Escalation Intervention (NEW)

    Multi-turn intervention sequence that escalates if needed.
    Starts gentle, escalates to stronger measures if deviation persists.

    Escalation sequence:
    1. Reminder
    2. Reframing
    3. Regeneration
    4. Constraint
    5. Fallback

    Use cases:
    - Persistent drift over multiple turns
    - Need adaptive response to deviation
    - Want to minimize intervention strength while ensuring alignment
    """

    def __init__(self):
        """Initialize escalation handler."""
        self.escalation_level = 0
        self.max_level = 4

        self.escalation_sequence = [
            InterventionStrategy.REMINDER,
            InterventionStrategy.REFRAMING,
            InterventionStrategy.REGENERATION,
            InterventionStrategy.CONSTRAINT,
            InterventionStrategy.FALLBACK
        ]

    def create_plan(self, context: InterventionContext) -> InterventionPlan:
        """Create escalation intervention plan."""
        # Select strategy based on current escalation level
        current_strategy = self.escalation_sequence[
            min(self.escalation_level, self.max_level)
        ]

        return InterventionPlan(
            strategy=InterventionStrategy.ESCALATION,
            parameters={
                'current_level': self.escalation_level,
                'current_strategy': current_strategy,
                'auto_escalate': True
            },
            expected_effectiveness=0.95,  # High overall effectiveness through escalation
            reasoning=f"Escalation level {self.escalation_level}: {current_strategy.value}"
        )

    def execute(self, plan: InterventionPlan, context: InterventionContext) -> str:
        """Execute escalation intervention."""
        current_strategy = plan.parameters['current_strategy']
        level = plan.parameters['current_level']

        return f"[INTERVENTION: Escalation L{level}] {current_strategy.value}"

    def escalate(self) -> None:
        """Escalate to next intervention level."""
        if self.escalation_level < self.max_level:
            self.escalation_level += 1
            logger.info(f"Escalating intervention to level {self.escalation_level}")

    def reset(self) -> None:
        """Reset escalation level."""
        self.escalation_level = 0


class MultiStrategyInterventionSystem:
    """
    Comprehensive intervention system with intelligent strategy selection.

    Selects optimal intervention strategy based on:
    - Deviation characteristics (type, magnitude)
    - Session context (history, topic)
    - Strategy effectiveness history
    - Resource constraints (time, cost)
    """

    def __init__(self):
        """Initialize multi-strategy intervention system."""
        # Register strategy handlers
        self.handlers = {
            InterventionStrategy.REMINDER: ReminderInterventionHandler(),
            InterventionStrategy.REGENERATION: RegenerationInterventionHandler(),
            InterventionStrategy.REFRAMING: ReframingInterventionHandler(),
            InterventionStrategy.SCAFFOLDING: ScaffoldingInterventionHandler(),
            InterventionStrategy.CONSTRAINT: ConstraintInterventionHandler(),
            InterventionStrategy.FALLBACK: FallbackInterventionHandler(),
            InterventionStrategy.ESCALATION: EscalationInterventionHandler()
        }

        # Strategy effectiveness tracking
        self.strategy_performance: Dict[InterventionStrategy, List[float]] = {
            strategy: [] for strategy in InterventionStrategy
        }

    def select_strategy(self, context: InterventionContext) -> InterventionPlan:
        """
        Select optimal intervention strategy based on context.

        Decision tree:
        1. If safety violation -> CONSTRAINT or FALLBACK
        2. If minor deviation (< 0.15) -> REMINDER
        3. If moderate deviation (< 0.25) -> REFRAMING or REGENERATION
        4. If severe deviation (< 0.35) -> CONSTRAINT or REGENERATION
        5. If catastrophic (>= 0.35) -> FALLBACK
        6. If repeated interventions -> ESCALATION

        Args:
            context: Intervention context

        Returns:
            InterventionPlan: Selected intervention plan
        """
        # Safety-critical deviation
        if context.deviation_type == DeviationType.SAFETY:
            logger.warning("Safety deviation detected - using constraint intervention")
            return self.handlers[InterventionStrategy.CONSTRAINT].create_plan(context)

        # Repeated interventions - use escalation
        if context.recent_intervention_count > 3:
            logger.info("Repeated interventions - using escalation strategy")
            return self.handlers[InterventionStrategy.ESCALATION].create_plan(context)

        # Deviation magnitude-based selection
        if context.deviation_magnitude < 0.15:
            # Minor deviation - reminder
            return self.handlers[InterventionStrategy.REMINDER].create_plan(context)

        elif context.deviation_magnitude < 0.25:
            # Moderate deviation - reframing or regeneration
            # Choose based on deviation type
            if context.deviation_type == DeviationType.STRUCTURAL:
                return self.handlers[InterventionStrategy.SCAFFOLDING].create_plan(context)
            else:
                return self.handlers[InterventionStrategy.REFRAMING].create_plan(context)

        elif context.deviation_magnitude < 0.35:
            # Severe deviation - regeneration or constraint
            if context.recent_intervention_count > 1:
                return self.handlers[InterventionStrategy.CONSTRAINT].create_plan(context)
            else:
                return self.handlers[InterventionStrategy.REGENERATION].create_plan(context)

        else:
            # Catastrophic deviation - fallback
            logger.error("Catastrophic deviation - using fallback intervention")
            return self.handlers[InterventionStrategy.FALLBACK].create_plan(context)

    def execute_intervention(
        self,
        plan: InterventionPlan,
        context: InterventionContext
    ) -> str:
        """
        Execute intervention according to plan.

        Args:
            plan: Intervention plan
            context: Intervention context

        Returns:
            str: Intervention result
        """
        handler = self.handlers[plan.strategy]
        result = handler.execute(plan, context)

        logger.info(
            f"Executed {plan.strategy.value} intervention: {plan.reasoning}"
        )

        return result

    def record_effectiveness(
        self,
        strategy: InterventionStrategy,
        effectiveness: float
    ) -> None:
        """
        Record intervention effectiveness for learning.

        Args:
            strategy: Strategy that was used
            effectiveness: Measured effectiveness (0-1)
        """
        self.strategy_performance[strategy].append(effectiveness)

        # Log if strategy consistently underperforming
        if len(self.strategy_performance[strategy]) > 10:
            avg_effectiveness = np.mean(self.strategy_performance[strategy][-10:])
            if avg_effectiveness < 0.5:
                logger.warning(
                    f"Strategy {strategy.value} underperforming: "
                    f"avg_effectiveness={avg_effectiveness:.2f}"
                )

    def get_strategy_statistics(self) -> Dict:
        """
        Get statistics on strategy usage and effectiveness.

        Returns:
            dict: Strategy statistics
        """
        stats = {}

        for strategy, performance in self.strategy_performance.items():
            if performance:
                stats[strategy.value] = {
                    'usage_count': len(performance),
                    'avg_effectiveness': np.mean(performance),
                    'std_effectiveness': np.std(performance),
                    'min_effectiveness': np.min(performance),
                    'max_effectiveness': np.max(performance)
                }
            else:
                stats[strategy.value] = {
                    'usage_count': 0,
                    'avg_effectiveness': None
                }

        return stats


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of Multi-Strategy Intervention System.

    Demonstrates strategy selection for different deviation scenarios.
    """

    print("="*80)
    print("MULTI-STRATEGY INTERVENTION SYSTEM - DEMONSTRATION")
    print("="*80 + "\n")

    # Initialize intervention system
    intervention_system = MultiStrategyInterventionSystem()

    # Test scenarios
    scenarios = [
        {
            'name': "Minor Semantic Deviation",
            'context': InterventionContext(
                fidelity_user=0.87,
                fidelity_ai=0.90,
                deviation_magnitude=0.13,
                deviation_type=DeviationType.SEMANTIC,
                turn_count=5,
                recent_intervention_count=0,
                conversation_topic="technical explanation",
                user_intent="learn concept",
                response_text="Response slightly off-topic but mostly aligned"
            )
        },
        {
            'name': "Moderate Structural Deviation",
            'context': InterventionContext(
                fidelity_user=0.78,
                fidelity_ai=0.82,
                deviation_magnitude=0.22,
                deviation_type=DeviationType.STRUCTURAL,
                turn_count=8,
                recent_intervention_count=1,
                conversation_topic="tutorial",
                user_intent="step-by-step guide",
                response_text="Content good but lacks structure"
            )
        },
        {
            'name': "Severe Repeated Deviation",
            'context': InterventionContext(
                fidelity_user=0.65,
                fidelity_ai=0.70,
                deviation_magnitude=0.30,
                deviation_type=DeviationType.BEHAVIORAL,
                turn_count=12,
                recent_intervention_count=4,
                conversation_topic="assistance",
                user_intent="specific help",
                response_text="Persistent drift despite interventions"
            )
        },
        {
            'name': "Safety Violation",
            'context': InterventionContext(
                fidelity_user=0.45,
                fidelity_ai=0.50,
                deviation_magnitude=0.40,
                deviation_type=DeviationType.SAFETY,
                turn_count=3,
                recent_intervention_count=0,
                conversation_topic="sensitive topic",
                user_intent="information request",
                response_text="Response crosses safety boundary"
            )
        }
    ]

    # Process each scenario
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Deviation: {scenario['context'].deviation_magnitude:.2f}")
        print(f"   Type: {scenario['context'].deviation_type.value}")
        print(f"   Recent interventions: {scenario['context'].recent_intervention_count}")

        # Select strategy
        plan = intervention_system.select_strategy(scenario['context'])

        print(f"\n   ✅ Selected Strategy: {plan.strategy.value}")
        print(f"   📊 Expected Effectiveness: {plan.expected_effectiveness:.2f}")
        print(f"   💡 Reasoning: {plan.reasoning}")

        # Execute intervention
        result = intervention_system.execute_intervention(plan, scenario['context'])
        print(f"   🔧 Execution: {result}")

        # Simulate effectiveness feedback
        simulated_effectiveness = plan.expected_effectiveness + np.random.normal(0, 0.1)
        simulated_effectiveness = np.clip(simulated_effectiveness, 0, 1)
        intervention_system.record_effectiveness(plan.strategy, simulated_effectiveness)

    # Print strategy statistics
    print("\n" + "="*80)
    print("STRATEGY PERFORMANCE STATISTICS")
    print("="*80)

    stats = intervention_system.get_strategy_statistics()
    for strategy_name, strategy_stats in stats.items():
        if strategy_stats['usage_count'] > 0:
            print(f"\n{strategy_name.upper()}:")
            print(f"  Usage count: {strategy_stats['usage_count']}")
            print(f"  Avg effectiveness: {strategy_stats['avg_effectiveness']:.2f}")
            print(f"  Range: [{strategy_stats['min_effectiveness']:.2f}, {strategy_stats['max_effectiveness']:.2f}]")
