"""
Turn Navigator - Timeline Playback Controls

Session playback system with timeline scrubber and playback controls.
Enables turn-by-turn navigation through conversation history.

Components:
1. Timeline Scrubber: Visual timeline with turn markers
2. Playback Controls: Play/Pause, Previous/Next, Jump to Turn
3. Speed Controls: Playback speed adjustment
4. Turn Marker Sync: Synchronizes all tools to current turn

Data Source: WebSessionManager (turn data)

Purpose:
- Navigate through session history
- Replay conversations for analysis
- Sync all tools to same turn marker
- "Movie playback" of governance sessions
"""

import streamlit as st
from typing import Dict, Any, Optional


class TurnNavigator:
    """
    Renders timeline navigator with playback controls.

    Enables turn-by-turn navigation and synchronizes all tools.
    """

    def __init__(self, session_manager, deck_manager):
        """
        Initialize Turn Navigator.

        Args:
            session_manager: WebSessionManager instance
            deck_manager: DeckManager instance for turn synchronization
        """
        self.session_manager = session_manager
        self.deck_manager = deck_manager

    def render(self):
        """
        Render turn navigator with timeline and controls.
        """
        st.markdown("### ⏯️ Turn Navigator")
        st.markdown("*Session Playback Controls*")

        # Get session data
        session_data = self._get_session_data()

        if not session_data or session_data['total_turns'] == 0:
            st.info("No session data available for navigation.")
            return

        # Render timeline scrubber
        self._render_timeline_scrubber(session_data)

        # Render playback controls
        self._render_playback_controls(session_data)

        # Render turn info
        self._render_turn_info(session_data)

    def _get_session_data(self) -> Dict[str, Any]:
        """
        Get session data for navigation.

        Returns:
            Dictionary with current_turn, total_turns, turn_data
        """
        # Get turns from WebSessionManager via session state
        if self.session_manager and self.session_manager.st_state:
            current_session = self.session_manager.st_state.get('current_session', {})
            turns = current_session.get('turns', [])

            # Get current turn from deck manager state
            current_turn = self.deck_manager.session_state.get('observation_deck', {}).get('current_turn', len(turns))

            # Default to latest turn if current_turn is 0 or invalid
            if current_turn == 0 or current_turn > len(turns):
                current_turn = len(turns)

            return {
                'current_turn': current_turn,
                'total_turns': len(turns),
                'turn_data': turns
            }

        return {
            'current_turn': 0,
            'total_turns': 0,
            'turn_data': []
        }

    def _render_timeline_scrubber(self, session_data: Dict[str, Any]):
        """
        Render timeline scrubber slider.

        Args:
            session_data: Session data dictionary
        """
        current_turn = session_data['current_turn']
        total_turns = session_data['total_turns']

        # Slider for turn selection
        selected_turn = st.slider(
            "Turn",
            min_value=1,
            max_value=max(total_turns, 1),
            value=current_turn if current_turn > 0 else 1,
            key="turn_navigator_slider"
        )

        # Update turn marker if changed
        if selected_turn != current_turn:
            self.deck_manager.set_current_turn(selected_turn)
            st.rerun()

    def _render_playback_controls(self, session_data: Dict[str, Any]):
        """
        Render playback control buttons.

        Args:
            session_data: Session data dictionary
        """
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("⏮️ First", key="nav_first"):
                self.deck_manager.set_current_turn(1)
                st.rerun()

        with col2:
            if st.button("⏪ Previous", key="nav_prev"):
                current = session_data['current_turn']
                if current > 1:
                    self.deck_manager.set_current_turn(current - 1)
                    st.rerun()

        with col3:
            if st.button("⏩ Next", key="nav_next"):
                current = session_data['current_turn']
                total = session_data['total_turns']
                if current < total:
                    self.deck_manager.set_current_turn(current + 1)
                    st.rerun()

        with col4:
            if st.button("⏭️ Last", key="nav_last"):
                self.deck_manager.set_current_turn(session_data['total_turns'])
                st.rerun()

    def _render_turn_info(self, session_data: Dict[str, Any]):
        """
        Render information about current turn.

        Args:
            session_data: Session data dictionary
        """
        st.markdown("---")
        st.markdown(f"**Turn {session_data['current_turn']} of {session_data['total_turns']}**")

        # Get current turn data
        current_turn_idx = session_data['current_turn'] - 1  # Convert to 0-indexed
        turns = session_data.get('turn_data', [])

        if current_turn_idx < 0 or current_turn_idx >= len(turns):
            st.info("No turn data available")
            return

        turn = turns[current_turn_idx]

        # Display turn metadata in compact format
        col1, col2 = st.columns(2)

        with col1:
            # Timestamp
            timestamp = turn.get('timestamp', 'N/A')
            if timestamp != 'N/A':
                from datetime import datetime
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp = dt.strftime('%H:%M:%S')
                except:
                    pass
            st.markdown(f"🕐 **Time:** {timestamp}")

            # Fidelity score
            fidelity = turn.get('fidelity', turn.get('metrics', {}).get('telic_fidelity', 1.0))
            fidelity_color = "🟢" if fidelity >= 0.76 else "🟡" if fidelity >= 0.67 else "🔴"  # Goldilocks zones
            st.markdown(f"{fidelity_color} **Fidelity:** {fidelity:.2f}")

        with col2:
            # Governance status
            metadata = turn.get('governance_metadata', turn.get('metadata', {}))
            intervention = metadata.get('intervention_applied', False)
            status = "✓ Governed" if intervention else "○ Native"
            st.markdown(f"⚡ **Status:** {status}")

            # Basin membership
            in_basin = metadata.get('primacy_basin_membership', True)
            basin_status = "✓ In Basin" if in_basin else "✗ Out of Basin"
            st.markdown(f"🎯 **Basin:** {basin_status}")

        # User input preview
        user_input = turn.get('user_message', turn.get('user_input', ''))
        if user_input:
            preview = user_input[:80] + "..." if len(user_input) > 80 else user_input
            st.markdown(f"**User:** {preview}")

        # Response preview (TELOS version)
        response = turn.get('telos_response', turn.get('assistant_response', ''))
        if response:
            preview = response[:80] + "..." if len(response) > 80 else response
            st.markdown(f"**Assistant:** {preview}")
