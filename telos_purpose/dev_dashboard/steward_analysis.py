#!/usr/bin/env python3
"""
Steward Analysis Module - Backend for TELOSCOPE Research Panel

Provides AI-powered analysis of TELOS conversation sessions for research purposes.
Each session is treated as a micro research environment with extractable insights.

Uses the same Mistral API that powers TELOS governance for consistency and efficiency.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Use existing TELOS Mistral client
try:
    from telos_purpose.llm_clients.mistral_client import MistralClient
    HAS_MISTRAL = True
except ImportError:
    HAS_MISTRAL = False


class StewardAnalyzer:
    """
    AI-powered session analysis for TELOSCOPE research using Mistral.

    Steward is a specialized research instrument with clear boundaries:

    CAN DO:
    - Analyze TELOS conversation sessions (governance patterns, canonical inputs, etc.)
    - Answer research questions about session data
    - Provide statistical summaries and metrics
    - Interpret conversation patterns with AI
    - Compare sessions for research insights

    CANNOT DO:
    - Modify code or files
    - Run system commands
    - Access external resources
    - Perform actions outside research analysis scope
    - Answer general questions unrelated to TELOS research
    """

    # Define scope boundaries
    SCOPE_KEYWORDS = {
        'in_scope': [
            'analyze', 'session', 'conversation', 'governance', 'canonical',
            'primacy', 'attractor', 'pattern', 'telos', 'turn', 'message',
            'research', 'data', 'metric', 'summary', 'compare', 'interpretation'
        ],
        'out_of_scope': [
            'modify', 'change', 'update', 'delete', 'create', 'write', 'file',
            'code', 'run', 'execute', 'command', 'install', 'debug', 'fix'
        ]
    }

    def __init__(self, mistral_client: Optional[MistralClient] = None):
        """
        Initialize Steward analyzer with Mistral API.

        Args:
            mistral_client: Optional existing MistralClient instance to reuse.
                           If None, will create new client using MISTRAL_API_KEY.
        """
        self.has_ai = False

        if mistral_client:
            # Reuse existing client (efficient!)
            self.client = mistral_client
            self.has_ai = True
        elif HAS_MISTRAL and os.getenv('MISTRAL_API_KEY'):
            # Create new client
            try:
                self.client = MistralClient(model="mistral-small-latest")
                self.has_ai = True
            except Exception as e:
                print(f"Failed to initialize Mistral client: {e}")
                self.has_ai = False
        else:
            self.has_ai = False

    def load_session_data(self, session_id: str) -> Optional[Dict]:
        """Load session data from saved file or return current session."""
        if session_id == "Current Session":
            return None  # Will be handled by caller to use live session state

        # Load from file
        sessions_dir = Path("saved_sessions")
        session_file = sessions_dir / f"session_{session_id}.json"

        if not session_file.exists():
            return None

        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session: {e}")
            return None

    def analyze_governance_impact(self, session_data: Dict) -> Dict[str, Any]:
        """Analyze how governance (TELOS on/off) impacted conversation."""
        if not session_data:
            return {"error": "No session data available"}

        # Extract governance-related metrics
        messages = session_data.get('messages', [])
        governance_states = []
        governance_switches = 0
        last_state = None

        for msg in messages:
            current_state = msg.get('governance_enabled', True)
            if last_state is not None and current_state != last_state:
                governance_switches += 1
            governance_states.append(current_state)
            last_state = current_state

        telos_on_count = sum(1 for s in governance_states if s)
        telos_off_count = len(governance_states) - telos_on_count

        analysis = {
            'total_turns': len(messages),
            'telos_enabled_turns': telos_on_count,
            'telos_disabled_turns': telos_off_count,
            'governance_switches': governance_switches,
            'governance_ratio': telos_on_count / len(governance_states) if governance_states else 0
        }

        # AI-powered interpretation if available
        if self.has_ai:
            analysis['ai_interpretation'] = self._get_ai_governance_analysis(session_data, analysis)
        else:
            analysis['ai_interpretation'] = "AI analysis unavailable (set ANTHROPIC_API_KEY)"

        return analysis

    def extract_canonical_inputs(self, session_data: Dict) -> Dict[str, Any]:
        """Extract and categorize canonical inputs from session."""
        if not session_data:
            return {"error": "No session data available"}

        messages = session_data.get('messages', [])
        canonical_inputs = []

        # Extract messages marked as canonical or with primacy attractors
        for i, msg in enumerate(messages):
            if msg.get('role') == 'user':
                # Check for canonical markers
                content = msg.get('content', '')
                metadata = msg.get('metadata', {})

                if metadata.get('is_canonical') or 'primacy_attractor' in metadata:
                    canonical_inputs.append({
                        'turn': i + 1,
                        'content': content[:200] + '...' if len(content) > 200 else content,
                        'timestamp': msg.get('timestamp'),
                        'primacy_attractor': metadata.get('primacy_attractor', 'N/A')
                    })

        analysis = {
            'total_canonical_inputs': len(canonical_inputs),
            'inputs': canonical_inputs
        }

        # AI-powered categorization if available
        if self.has_ai and canonical_inputs:
            analysis['ai_categorization'] = self._get_ai_canonical_analysis(session_data, canonical_inputs)
        else:
            analysis['ai_categorization'] = "AI analysis unavailable or no canonical inputs found"

        return analysis

    def analyze_conversation_patterns(self, session_data: Dict) -> Dict[str, Any]:
        """Analyze turn-taking, topic shifts, and conversation flow patterns."""
        if not session_data:
            return {"error": "No session data available"}

        messages = session_data.get('messages', [])

        # Calculate basic patterns
        user_turns = sum(1 for m in messages if m.get('role') == 'user')
        assistant_turns = sum(1 for m in messages if m.get('role') == 'assistant')
        total_turns = len(messages)

        # Calculate average message lengths
        user_lengths = [len(m.get('content', '')) for m in messages if m.get('role') == 'user']
        assistant_lengths = [len(m.get('content', '')) for m in messages if m.get('role') == 'assistant']

        avg_user_length = sum(user_lengths) / len(user_lengths) if user_lengths else 0
        avg_assistant_length = sum(assistant_lengths) / len(assistant_lengths) if assistant_lengths else 0

        analysis = {
            'total_turns': total_turns,
            'user_turns': user_turns,
            'assistant_turns': assistant_turns,
            'avg_user_message_length': int(avg_user_length),
            'avg_assistant_message_length': int(avg_assistant_length),
            'turn_ratio': user_turns / assistant_turns if assistant_turns > 0 else 0
        }

        # AI-powered pattern recognition if available
        if self.has_ai:
            analysis['ai_pattern_analysis'] = self._get_ai_pattern_analysis(session_data, analysis)
        else:
            analysis['ai_pattern_analysis'] = "AI analysis unavailable"

        return analysis

    def analyze_primacy_attractor_evolution(self, session_data: Dict) -> Dict[str, Any]:
        """Track how primacy attractors evolved throughout conversation."""
        if not session_data:
            return {"error": "No session data available"}

        messages = session_data.get('messages', [])
        primacy_sequence = []

        for i, msg in enumerate(messages):
            metadata = msg.get('metadata', {})
            if 'primacy_attractor' in metadata:
                primacy_sequence.append({
                    'turn': i + 1,
                    'attractor': metadata['primacy_attractor'],
                    'timestamp': msg.get('timestamp')
                })

        analysis = {
            'total_primacy_attractors': len(primacy_sequence),
            'sequence': primacy_sequence
        }

        # AI-powered evolution analysis
        if self.has_ai and primacy_sequence:
            analysis['ai_evolution_analysis'] = self._get_ai_primacy_evolution(session_data, primacy_sequence)
        else:
            analysis['ai_evolution_analysis'] = "AI analysis unavailable or no primacy attractors tracked"

        return analysis

    def generate_session_summary(self, session_data: Dict) -> Dict[str, Any]:
        """Generate comprehensive session summary report."""
        if not session_data:
            return {"error": "No session data available"}

        # Gather all metrics
        governance = self.analyze_governance_impact(session_data)
        canonical = self.extract_canonical_inputs(session_data)
        patterns = self.analyze_conversation_patterns(session_data)
        primacy = self.analyze_primacy_attractor_evolution(session_data)

        summary = {
            'session_id': session_data.get('session_id', 'unknown'),
            'timestamp': session_data.get('timestamp', 'unknown'),
            'governance_summary': governance,
            'canonical_inputs_summary': canonical,
            'pattern_summary': patterns,
            'primacy_summary': primacy
        }

        # AI-powered executive summary
        if self.has_ai:
            summary['executive_summary'] = self._get_ai_executive_summary(session_data, summary)
        else:
            summary['executive_summary'] = "AI summary unavailable"

        return summary

    def _check_scope(self, query: str) -> Tuple[bool, str]:
        """
        Check if query is within Steward's research analysis scope.

        Returns:
            (is_in_scope, reason) - True if in scope, False with explanation if not
        """
        query_lower = query.lower()

        # Check for obvious out-of-scope keywords
        for keyword in self.SCOPE_KEYWORDS['out_of_scope']:
            if keyword in query_lower:
                return False, f"Steward cannot {keyword} - I'm a research analysis tool, not a code/system modification tool."

        # Check if query seems related to research analysis
        has_research_context = any(kw in query_lower for kw in self.SCOPE_KEYWORDS['in_scope'])

        if not has_research_context and len(query.split()) > 3:
            # Longer query with no research keywords might be out of scope
            return False, "This question seems outside my research analysis scope. I analyze TELOS session data - ask me about governance patterns, canonical inputs, conversation dynamics, or session metrics."

        return True, ""

    def chat_with_steward(self, query: str, session_data: Dict, context: str = "") -> str:
        """Answer research questions about session using AI."""
        if not self.has_ai:
            return "❌ AI features unavailable. Please set MISTRAL_API_KEY environment variable."

        if not session_data:
            return "❌ No session data available to analyze."

        # Check if query is in scope
        in_scope, reason = self._check_scope(query)
        if not in_scope:
            return f"""🚫 **Out of Scope Request**

{reason}

**What I CAN help with:**
- Analyze governance patterns (TELOS on/off effects)
- Extract canonical inputs and primacy attractors
- Identify conversation patterns
- Compare session metrics
- Interpret research data

**Example questions:**
- "What governance patterns emerged in this session?"
- "How did primacy attractors evolve?"
- "What was the TELOS impact ratio?"
- "What canonical inputs were identified?"
"""

        # Prepare context for Mistral
        session_context = self._prepare_session_context(session_data)

        # System prompt with clear boundaries
        prompt = f"""You are Steward, a specialized AI research assistant for analyzing TELOS conversation sessions.

IMPORTANT BOUNDARIES:
- You ONLY analyze existing session data - you do NOT modify code, run commands, or change files
- You provide research insights, metrics, and pattern analysis
- If asked to do something outside research analysis, politely decline and redirect to your research capabilities

TELOS Context:
TELOS is a governance framework that steers AI responses toward more reflective, thoughtful outputs.

Session Data:
{session_context}

{context}

Researcher Question: {query}

Provide clear, insightful research analysis. If the question is outside your analysis scope, politely explain your boundaries and suggest research-focused alternatives."""

        try:
            response = self.client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.7
            )
            return response

        except Exception as e:
            return f"❌ Analysis failed: {str(e)}"

    # ========================================================================
    # Private helper methods for AI analysis
    # ========================================================================

    def _prepare_session_context(self, session_data: Dict) -> str:
        """Prepare session data as context for AI analysis."""
        messages = session_data.get('messages', [])
        context_parts = []

        context_parts.append(f"Session ID: {session_data.get('session_id', 'unknown')}")
        context_parts.append(f"Total Turns: {len(messages)}")

        # Sample recent messages
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        context_parts.append("\nRecent Conversation:")

        for i, msg in enumerate(recent_messages):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:150]
            gov = "TELOS ON" if msg.get('governance_enabled', True) else "TELOS OFF"
            context_parts.append(f"{i+1}. [{role.upper()}|{gov}]: {content}...")

        return "\n".join(context_parts)

    def _get_ai_governance_analysis(self, session_data: Dict, metrics: Dict) -> str:
        """Get AI interpretation of governance impact."""
        context = self._prepare_session_context(session_data)

        prompt = f"""Analyze the governance impact in this TELOS session:

{context}

Metrics:
- TELOS enabled turns: {metrics['telos_enabled_turns']}
- TELOS disabled turns: {metrics['telos_disabled_turns']}
- Governance switches: {metrics['governance_switches']}
- TELOS usage ratio: {metrics['governance_ratio']:.2%}

Provide 2-3 sentence insight on how governance affected this conversation."""

        try:
            response = self.client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response
        except:
            return "AI interpretation unavailable"

    def _get_ai_canonical_analysis(self, session_data: Dict, canonical_inputs: List) -> str:
        """Get AI categorization of canonical inputs."""
        prompt = f"""Categorize these canonical inputs from a TELOS conversation:

{json.dumps(canonical_inputs, indent=2)}

Identify themes and research significance. Keep response to 2-3 sentences."""

        try:
            response = self.client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response
        except:
            return "AI categorization unavailable"

    def _get_ai_pattern_analysis(self, session_data: Dict, metrics: Dict) -> str:
        """Get AI analysis of conversation patterns."""
        context = self._prepare_session_context(session_data)

        prompt = f"""Analyze conversation patterns:

{context}

Metrics:
- Total turns: {metrics['total_turns']}
- User/Assistant ratio: {metrics['turn_ratio']:.2f}
- Avg user message: {metrics['avg_user_message_length']} chars
- Avg assistant message: {metrics['avg_assistant_message_length']} chars

Identify notable patterns. 2-3 sentences."""

        try:
            response = self.client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response
        except:
            return "AI pattern analysis unavailable"

    def _get_ai_primacy_evolution(self, session_data: Dict, sequence: List) -> str:
        """Get AI analysis of primacy attractor evolution."""
        prompt = f"""Analyze how primacy attractors evolved in this conversation:

{json.dumps(sequence, indent=2)}

How did the conversation's focus evolve? 2-3 sentences."""

        try:
            response = self.client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response
        except:
            return "AI evolution analysis unavailable"

    def _get_ai_executive_summary(self, session_data: Dict, full_summary: Dict) -> str:
        """Generate executive summary of entire session."""
        context = self._prepare_session_context(session_data)

        prompt = f"""Generate executive summary for this TELOS research session:

{context}

Summary Data:
{json.dumps(full_summary, indent=2, default=str)}

Provide 3-4 sentence executive summary highlighting key research findings."""

        try:
            response = self.client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response
        except:
            return "AI executive summary unavailable"
